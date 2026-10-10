"""데이터 수집과 정렬.

- 가격: 야후 파이낸스(yfinance). QQQ는 배당·분할 반영 수정주가(총수익) 사용.
- 경제지표: FRED CSV (API 키 불필요).

경제지표는 실제 발표일보다 앞서 값을 쓰면 백테스트가 실제보다 좋아 보이므로,
`to_daily()`에서 발표 지연일만큼 날짜를 뒤로 민 뒤 영업일 단위로 앞값 채우기를 한다.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import pandas as pd
import requests

PRICE_TICKER = "QQQ"
TREND_TICKER = "^NDX"  # 장기추세 괴리 계산용 (QQQ보다 긴 1985년부터의 이력)

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"

# 앱에서 사용하는 FRED 시리즈
FRED_SERIES = [
    "DFEDTAR",       # 연방기금 목표금리 (~2008-12-15)
    "DFEDTARU",      # 연방기금 목표범위 상단 (2008-12-16~)
    "DGS10",         # 10년물 국채금리
    "T10Y2Y",        # 10년-2년 금리차
    "DFII10",        # 10년물 TIPS 실질금리
    "VIXCLS",        # VIX
    "BAA10Y",        # Baa 회사채 - 10년물 국채 스프레드
    "CPIAUCSL",      # 소비자물가지수
    "UNRATE",        # 실업률
    "M2SL",          # M2 통화량
    "WALCL",         # 연준 총자산
    "DTWEXBGS",      # 광의 달러지수
    "NFCI",          # 시카고 연은 금융여건지수
]


def load_prices(ticker: str = PRICE_TICKER) -> pd.DataFrame:
    """일봉 OHLCV 전체 이력 (수정주가)."""
    import yfinance as yf

    df = yf.Ticker(ticker).history(period="max", auto_adjust=True)
    if df is None or df.empty:
        raise RuntimeError(f"{ticker} 가격 데이터를 받지 못했습니다.")
    df.index = _naive_dates(df.index)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df[["Open", "High", "Low", "Close", "Volume"]].astype(float)


def load_latest_quote(ticker: str = PRICE_TICKER) -> dict:
    """장중 최신 시세(약 15분 지연)와 전일 종가."""
    import yfinance as yf

    t = yf.Ticker(ticker)
    intraday = t.history(period="5d", interval="1m", auto_adjust=False)
    daily = t.history(period="1mo", interval="1d", auto_adjust=False)
    if intraday.empty or daily.empty:
        raise RuntimeError(f"{ticker} 실시간 시세를 받지 못했습니다.")
    last_time = intraday.index[-1]
    last_price = float(intraday["Close"].iloc[-1])
    daily_dates = _naive_dates(daily.index)
    prev = daily["Close"][daily_dates < last_time.tz_localize(None).normalize()]
    prev_close = float(prev.iloc[-1]) if len(prev) else float("nan")
    return {
        "price": last_price,
        "time": last_time,
        "prev_close": prev_close,
        "change_pct": (last_price / prev_close - 1) * 100,
    }


def load_fred(series_id: str) -> pd.Series:
    resp = requests.get(FRED_CSV_URL.format(series_id), timeout=30)
    resp.raise_for_status()
    return parse_fred_csv(resp.text, series_id)


def parse_fred_csv(text: str, series_id: str) -> pd.Series:
    """FRED CSV(첫 열 날짜, 둘째 열 값, 결측은 '.')를 Series로."""
    df = pd.read_csv(io.StringIO(text), na_values=".")
    values = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    s = pd.Series(values.to_numpy(), index=pd.to_datetime(df.iloc[:, 0]), name=series_id)
    return s.dropna().sort_index()


def fed_target_rate(raw: dict[str, pd.Series]) -> pd.Series:
    """2008년 12월 목표범위 제도 전후를 이어 붙인 기준금리(목표 상단) 시계열."""
    old, new = raw.get("DFEDTAR"), raw.get("DFEDTARU")
    if new is None or new.empty:
        return old if old is not None else pd.Series(dtype=float)
    if old is None or old.empty:
        return new
    return pd.concat([old[old.index < new.index[0]], new]).rename("FEDTARGET")


def parse_custom_csv(data: bytes) -> pd.Series:
    """사용자 업로드 CSV: 첫 열 날짜, 둘째 열 값 (헤더 1줄)."""
    df = pd.read_csv(io.BytesIO(data))
    if df.shape[1] < 2:
        raise ValueError("CSV에 날짜와 값, 최소 두 열이 필요합니다.")
    values = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    s = pd.Series(values.to_numpy(), index=pd.to_datetime(df.iloc[:, 0]), name="CUSTOM")
    s = s.dropna().sort_index()
    if s.empty:
        raise ValueError("CSV에서 숫자 값을 찾지 못했습니다.")
    return s[~s.index.duplicated(keep="last")]


def to_daily(series: pd.Series, lag_days: int = 0, end: pd.Timestamp | None = None) -> pd.Series:
    """관측일을 발표 지연만큼 뒤로 민 뒤 영업일 단위로 앞값 채우기.

    주말·휴일에 공개된 값은 다음 영업일부터 쓰이므로 미래 정보가 섞이지 않는다.
    """
    s = series.dropna().sort_index()
    if s.empty:
        return s
    if lag_days:
        s = s.copy()
        s.index = s.index + pd.Timedelta(days=lag_days)
    s = s[~s.index.duplicated(keep="last")]
    end = max(end or pd.Timestamp.today().normalize(), s.index[-1])
    idx = weekdays(s.index[0], end)
    return s.reindex(s.index.union(idx)).ffill().reindex(idx)


def weekdays(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """월~금 날짜 (pd.bdate_range와 같은 결과, 훨씬 빠름)."""
    days = pd.date_range(pd.Timestamp(start).normalize(), end, freq="D")
    return days[days.dayofweek < 5]


def align(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """임의 날짜의 시계열을 대상 날짜(거래일)에 앞값 채우기로 맞춘다."""
    s = series.dropna().sort_index()
    if s.empty:
        return pd.Series(float("nan"), index=index)
    return s.reindex(s.index.union(index)).ffill().reindex(index)


@dataclass
class MarketData:
    prices: pd.DataFrame                       # QQQ 일봉
    trend_prices: pd.DataFrame | None = None   # 나스닥100 지수 일봉
    fred: dict[str, pd.Series] = field(default_factory=dict)
    custom: pd.Series | None = None            # 사용자 업로드 지표
    errors: list[str] = field(default_factory=list)

    @property
    def close(self) -> pd.Series:
        return self.prices["Close"]

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.prices.index

    def fred_series(self, series_id: str) -> pd.Series:
        if series_id == "FEDTARGET":
            return fed_target_rate(self.fred)
        s = self.fred.get(series_id)
        if s is None:
            return pd.Series(dtype=float)
        return s


def _naive_dates(index: pd.Index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()
