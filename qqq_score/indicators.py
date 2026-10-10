"""지표 레지스트리.

지표 하나 = `Indicator` 하나. 새 지표를 추가하려면 계산 함수를 만들고
`INDICATORS` 목록에 `Indicator(...)`를 하나 더하면 화면·점수·조건·통계에 자동으로 붙는다.

계산 함수는 원값 Series를 돌려준다.
- 기술적 지표: QQQ 거래일 인덱스
- 경제지표: 발표 지연을 반영한 영업일 인덱스 (`data.to_daily`)
점수는 원값의 고유 인덱스에서 계산한 뒤 QQQ 거래일에 맞춘다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .data import MarketData, to_daily

TECH, RATES, MACRO, VALUE = "기술적", "금리", "경기·심리", "가치평가"
CATEGORIES = [TECH, RATES, MACRO, VALUE]

# 점수 방향
LOW_GOOD = "low"    # 값이 낮을수록 고득점
HIGH_GOOD = "high"  # 값이 높을수록 고득점

# 점수 방식
PERCENTILE = "percentile"  # 과거 N년 분포 내 위치
RANGE = "range"            # 하한~상한 구간 선형 변환


@dataclass(frozen=True)
class Param:
    key: str
    label: str
    default: float
    min: float
    max: float
    step: float = 1
    integer: bool = True


@dataclass(frozen=True)
class ScoreDefaults:
    method: str
    direction: str
    window_years: float = 10.0
    low: float = 0.0
    high: float = 100.0


@dataclass(frozen=True)
class Indicator:
    key: str
    name: str
    category: str
    unit: str
    description: str
    compute: Callable[[MarketData, dict], pd.Series]
    score: ScoreDefaults
    params: tuple[Param, ...] = ()
    enabled: bool = False
    in_score: bool = True
    weight: float = 1.0
    overlay: Callable[[MarketData, dict], dict[str, pd.Series]] | None = None

    def default_params(self) -> dict:
        return {p.key: p.default for p in self.params}

    def label(self, params: dict | None = None) -> str:
        """차트·조건에 쓰는 이름. 파라미터가 있으면 괄호로 붙인다."""
        if not self.params:
            return self.name
        params = params or self.default_params()
        vals = ", ".join(_fmt_num(params.get(p.key, p.default)) for p in self.params)
        return f"{self.name}({vals})"


def _fmt_num(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:g}"


# ---------------------------------------------------------------- 기술적 계산


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(int(n), min_periods=int(n)).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=int(n), adjust=False, min_periods=int(n)).mean()


def rsi(close: pd.Series, n: int) -> pd.Series:
    """Wilder RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=int(n)).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=int(n)).mean()
    rs = avg_gain / avg_loss
    out = 100 - 100 / (1 + rs)
    return out.where(avg_loss != 0, 100.0).where(avg_gain.notna())


def bollinger(close: pd.Series, n: int, k: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(close, n)
    sd = close.rolling(int(n), min_periods=int(n)).std(ddof=0)
    return mid - k * sd, mid, mid + k * sd


def _ma_gap(d: MarketData, p: dict) -> pd.Series:
    return (d.close / sma(d.close, p["n"]) - 1) * 100


def _ma_gap_overlay(d: MarketData, p: dict) -> dict[str, pd.Series]:
    return {f"이동평균 {_fmt_num(p['n'])}일": sma(d.close, p["n"])}


def _ma_cross(d: MarketData, p: dict) -> pd.Series:
    return (sma(d.close, p["short"]) / sma(d.close, p["long"]) - 1) * 100


def _ma_cross_overlay(d: MarketData, p: dict) -> dict[str, pd.Series]:
    return {
        f"이동평균 {_fmt_num(p['short'])}일": sma(d.close, p["short"]),
        f"이동평균 {_fmt_num(p['long'])}일": sma(d.close, p["long"]),
    }


def _bb_pctb(d: MarketData, p: dict) -> pd.Series:
    lower, _, upper = bollinger(d.close, p["n"], p["k"])
    return (d.close - lower) / (upper - lower) * 100


def _bb_overlay(d: MarketData, p: dict) -> dict[str, pd.Series]:
    lower, mid, upper = bollinger(d.close, p["n"], p["k"])
    return {"볼린저 상단": upper, "볼린저 중심": mid, "볼린저 하단": lower}


def _rsi(d: MarketData, p: dict) -> pd.Series:
    return rsi(d.close, p["n"])


def _macd_hist(d: MarketData, p: dict) -> pd.Series:
    line = ema(d.close, p["fast"]) - ema(d.close, p["slow"])
    signal = ema(line, p["signal"])
    return (line - signal) / d.close * 100


def _stoch(d: MarketData, p: dict) -> pd.Series:
    n, m = int(p["n"]), int(p["smooth"])
    low = d.prices["Low"].rolling(n, min_periods=n).min()
    high = d.prices["High"].rolling(n, min_periods=n).max()
    k = (d.close - low) / (high - low) * 100
    return k.rolling(m, min_periods=m).mean()


def _drawdown(d: MarketData, p: dict) -> pd.Series:
    n = int(p["lookback"])
    peak = d.close.cummax() if n == 0 else d.close.rolling(n, min_periods=1).max()
    return (d.close / peak - 1) * 100


def _momentum(d: MarketData, p: dict) -> pd.Series:
    return (d.close / d.close.shift(int(p["n"])) - 1) * 100


def _volatility(d: MarketData, p: dict) -> pd.Series:
    n = int(p["n"])
    ret = np.log(d.close).diff()
    return ret.rolling(n, min_periods=n).std() * np.sqrt(252) * 100


# ---------------------------------------------------------------- 경제지표 계산


def change_over_months(s: pd.Series, months: int) -> pd.Series:
    """일별 시계열의 n개월 전 대비 차이 (그 시점에 알 수 있던 값 기준)."""
    if s.empty:
        return s
    prev = s.asof(s.index - pd.DateOffset(months=int(months)))
    return s - pd.Series(prev.to_numpy(), index=s.index)


def _fred_daily(series_id: str, lag: int) -> Callable[[MarketData, dict], pd.Series]:
    def compute(d: MarketData, p: dict) -> pd.Series:
        return to_daily(d.fred_series(series_id), lag)

    return compute


def _fed_rate_chg(d: MarketData, p: dict) -> pd.Series:
    return change_over_months(to_daily(d.fred_series("FEDTARGET"), 1), p["months"])


def _yoy(series_id: str, periods: int, lag: int) -> Callable[[MarketData, dict], pd.Series]:
    def compute(d: MarketData, p: dict) -> pd.Series:
        s = d.fred_series(series_id)
        return to_daily(s.pct_change(periods) * 100, lag)

    return compute


def _unrate_rise(d: MarketData, p: dict) -> pd.Series:
    u = d.fred_series("UNRATE")
    avg3 = u.rolling(3, min_periods=3).mean()
    return to_daily(avg3 - avg3.rolling(12, min_periods=12).min(), 35)


def _dollar_chg(d: MarketData, p: dict) -> pd.Series:
    s = to_daily(d.fred_series("DTWEXBGS"), 7)
    prev = s.asof(s.index - pd.DateOffset(months=int(p["months"])))
    return (s / pd.Series(prev.to_numpy(), index=s.index) - 1) * 100


# ---------------------------------------------------------------- 가치평가 계산


def log_trend_gap(price: pd.Series, window_years: float, min_years: float = 10) -> pd.Series:
    """로그가격의 선형추세(과거 데이터로만 적합) 대비 괴리율(%).

    각 시점의 추세선은 그 시점까지의 과거 N년(0이면 전체 누적)으로만 적합한다.
    """
    price = price.dropna()
    price = price[price > 0]
    if price.empty:
        return price
    y = np.log(price)
    t = pd.Series((price.index - price.index[0]).days / 365.25, index=price.index)
    w = int(window_years * 252)
    minp = int(min(min_years, window_years or min_years) * 252)
    frame = pd.DataFrame({"t": t, "y": y, "tt": t * t, "ty": t * y, "one": 1.0})
    sums = frame.expanding(min_periods=minp).sum() if w == 0 else frame.rolling(w, min_periods=minp).sum()
    n = sums["one"]
    denom = n * sums["tt"] - sums["t"] ** 2
    slope = (n * sums["ty"] - sums["t"] * sums["y"]) / denom
    intercept = (sums["y"] - slope * sums["t"]) / n
    resid = y - (intercept + slope * t)
    return (np.exp(resid) - 1) * 100


def _trend_gap(d: MarketData, p: dict) -> pd.Series:
    src = d.trend_prices["Close"] if d.trend_prices is not None else d.close
    return log_trend_gap(src, p["window"])


def _custom(d: MarketData, p: dict) -> pd.Series:
    if d.custom is None:
        return pd.Series(dtype=float)
    return to_daily(d.custom, int(p["lag"]))


# ---------------------------------------------------------------- 레지스트리

INDICATORS: list[Indicator] = [
    # 기술적
    Indicator(
        "rsi", "RSI", TECH, "",
        "상대강도지수. 30 이하 과매도, 70 이상 과매수로 흔히 본다.",
        _rsi, ScoreDefaults(RANGE, LOW_GOOD, low=30, high=70),
        params=(Param("n", "기간(일)", 14, 2, 100),),
        enabled=True,
    ),
    Indicator(
        "bb", "볼린저 %B", TECH, "%",
        "밴드 안 위치. 0%=하단 밴드, 100%=상단 밴드. 기간과 표준편차 배수를 조절할 수 있다.",
        _bb_pctb, ScoreDefaults(RANGE, LOW_GOOD, low=0, high=100),
        params=(Param("n", "기간(일)", 20, 5, 200), Param("k", "표준편차 배수", 2.0, 0.5, 4.0, 0.1, False)),
        enabled=True, overlay=_bb_overlay,
    ),
    Indicator(
        "ma_gap", "이동평균 이격도", TECH, "%",
        "종가가 이동평균보다 몇 % 위/아래인지.",
        _ma_gap, ScoreDefaults(PERCENTILE, LOW_GOOD, window_years=10),
        params=(Param("n", "기간(일)", 200, 5, 400),),
        enabled=True, overlay=_ma_gap_overlay,
    ),
    Indicator(
        "drawdown", "고점 대비 낙폭", TECH, "%",
        "최근 N일 최고가 대비 하락률. 기간 0이면 사상 최고가 기준.",
        _drawdown, ScoreDefaults(RANGE, LOW_GOOD, low=-20, high=0),
        params=(Param("lookback", "기간(일, 0=전체)", 0, 0, 2520),),
        enabled=True,
    ),
    Indicator(
        "ma_cross", "이동평균 교차", TECH, "%",
        "단기 이동평균이 장기 이동평균보다 몇 % 위인지. 0 위로 올라가면 골든크로스.",
        _ma_cross, ScoreDefaults(RANGE, HIGH_GOOD, low=-5, high=5),
        params=(Param("short", "단기(일)", 50, 2, 200), Param("long", "장기(일)", 200, 10, 400)),
        overlay=_ma_cross_overlay,
    ),
    Indicator(
        "macd", "MACD 히스토그램", TECH, "%",
        "MACD선 - 시그널선 (종가 대비 %). 양수면 상승 모멘텀.",
        _macd_hist, ScoreDefaults(PERCENTILE, HIGH_GOOD, window_years=10),
        params=(Param("fast", "단기 EMA", 12, 2, 100), Param("slow", "장기 EMA", 26, 5, 200),
                Param("signal", "시그널", 9, 2, 50)),
    ),
    Indicator(
        "stoch", "스토캐스틱 %K", TECH, "",
        "최근 N일 고저 범위 안에서 종가 위치(완만화). 20 이하 과매도, 80 이상 과매수.",
        _stoch, ScoreDefaults(RANGE, LOW_GOOD, low=20, high=80),
        params=(Param("n", "기간(일)", 14, 2, 100), Param("smooth", "완만화(일)", 3, 1, 20)),
    ),
    Indicator(
        "momentum", "모멘텀", TECH, "%",
        "N일 전 대비 수익률. 252일 ≈ 1년.",
        _momentum, ScoreDefaults(PERCENTILE, HIGH_GOOD, window_years=10),
        params=(Param("n", "기간(일)", 252, 5, 756),),
    ),
    Indicator(
        "volatility", "변동성", TECH, "%",
        "최근 N일 일간 수익률의 연율화 표준편차.",
        _volatility, ScoreDefaults(PERCENTILE, HIGH_GOOD, window_years=10),
        params=(Param("n", "기간(일)", 20, 5, 252),),
    ),
    # 금리
    Indicator(
        "fed_rate", "기준금리", RATES, "%",
        "연방기금 목표금리(2008년 12월 이후는 목표범위 상단). 출처 FRED DFEDTAR·DFEDTARU.",
        _fred_daily("FEDTARGET", 1), ScoreDefaults(RANGE, LOW_GOOD, low=1, high=5),
        enabled=True,
    ),
    Indicator(
        "fed_rate_chg", "기준금리 변화", RATES, "%p",
        "N개월 전 대비 기준금리 변화. 양수=인상기, 음수=인하기, 0=동결.",
        _fed_rate_chg, ScoreDefaults(RANGE, LOW_GOOD, low=-1, high=1),
        params=(Param("months", "기간(개월)", 6, 1, 36),),
        enabled=True,
    ),
    Indicator(
        "dgs10", "10년물 국채금리", RATES, "%",
        "미국 10년 만기 국채 수익률. 출처 FRED DGS10.",
        _fred_daily("DGS10", 1), ScoreDefaults(PERCENTILE, LOW_GOOD, window_years=10),
    ),
    Indicator(
        "curve", "장단기 금리차", RATES, "%p",
        "10년물 - 2년물. 음수(역전)는 경기침체 선행신호로 알려져 있다. 출처 FRED T10Y2Y.",
        _fred_daily("T10Y2Y", 1), ScoreDefaults(RANGE, HIGH_GOOD, low=-0.5, high=1.5),
        enabled=True,
    ),
    Indicator(
        "real10", "실질금리(10년)", RATES, "%",
        "10년물 물가연동국채(TIPS) 수익률. 2003년부터. 출처 FRED DFII10.",
        _fred_daily("DFII10", 1), ScoreDefaults(PERCENTILE, LOW_GOOD, window_years=10),
    ),
    # 경기·심리
    Indicator(
        "vix", "VIX", MACRO, "",
        "S&P500 옵션 내재변동성(공포지수). 높을수록 시장 불안. 출처 FRED VIXCLS.",
        _fred_daily("VIXCLS", 0), ScoreDefaults(RANGE, HIGH_GOOD, low=15, high=35),
        enabled=True,
    ),
    Indicator(
        "credit", "신용스프레드", MACRO, "%p",
        "Baa 회사채 금리 - 10년물 국채금리. 높을수록 신용 위험 회피. 출처 FRED BAA10Y.",
        _fred_daily("BAA10Y", 1), ScoreDefaults(PERCENTILE, HIGH_GOOD, window_years=10),
    ),
    Indicator(
        "cpi_yoy", "물가상승률(CPI)", MACRO, "%",
        "소비자물가 전년 대비 상승률. 발표 지연 45일 반영. 출처 FRED CPIAUCSL.",
        _yoy("CPIAUCSL", 12, 45), ScoreDefaults(RANGE, LOW_GOOD, low=2, high=5),
    ),
    Indicator(
        "unrate_rise", "실업률 상승폭", MACRO, "%p",
        "실업률 3개월 평균이 최근 12개월 최저치보다 얼마나 올랐는지(삼의 법칙 방식). 0.5 이상이면 침체 신호로 알려져 있다. 발표 지연 35일 반영.",
        _unrate_rise, ScoreDefaults(RANGE, LOW_GOOD, low=0, high=0.5),
    ),
    Indicator(
        "m2_yoy", "M2 증가율", MACRO, "%",
        "M2 통화량 전년 대비 증가율. 발표 지연 60일 반영. 출처 FRED M2SL.",
        _yoy("M2SL", 12, 60), ScoreDefaults(PERCENTILE, HIGH_GOOD, window_years=10),
    ),
    Indicator(
        "fed_bs_yoy", "연준 총자산 증가율", MACRO, "%",
        "연준 대차대조표 전년 대비 증가율(양적완화/긴축). 2002년 말부터. 출처 FRED WALCL.",
        _yoy("WALCL", 52, 2), ScoreDefaults(PERCENTILE, HIGH_GOOD, window_years=10),
    ),
    Indicator(
        "dollar", "달러지수 변화율", MACRO, "%",
        "광의 달러지수의 N개월 전 대비 변화율. 2006년부터. 출처 FRED DTWEXBGS.",
        _dollar_chg, ScoreDefaults(PERCENTILE, LOW_GOOD, window_years=10),
        params=(Param("months", "기간(개월)", 12, 1, 36),),
    ),
    Indicator(
        "nfci", "금융여건지수", MACRO, "",
        "시카고 연은 NFCI. 0보다 크면 평균보다 긴축적, 작으면 완화적. 출처 FRED NFCI.",
        _fred_daily("NFCI", 7), ScoreDefaults(PERCENTILE, LOW_GOOD, window_years=10),
    ),
    # 가치평가
    Indicator(
        "trend_gap", "장기추세 괴리", VALUE, "%",
        "나스닥100 지수(1985~)가 과거 N년 로그 추세선보다 몇 % 위/아래인지. 무료로 구할 수 있는 가격 기반 고평가/저평가 대용지표.",
        _trend_gap, ScoreDefaults(PERCENTILE, LOW_GOOD, window_years=10),
        params=(Param("window", "추세 기간(년, 0=전체)", 20, 0, 40),),
        enabled=True,
    ),
    Indicator(
        "custom", "사용자 지표", VALUE, "",
        "사이드바에서 올린 CSV(날짜, 값). 예: 나스닥100 PER, 이익수익률 등. 발표 지연일을 꼭 넣어야 한다.",
        _custom, ScoreDefaults(PERCENTILE, LOW_GOOD, window_years=10),
        params=(Param("lag", "발표 지연(일)", 0, 0, 120),),
    ),
]

BY_KEY: dict[str, Indicator] = {ind.key: ind for ind in INDICATORS}
assert len(BY_KEY) == len(INDICATORS), "지표 key 중복"
