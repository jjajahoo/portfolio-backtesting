"""가격 데이터 불러오기 (Yahoo Finance).

- 한국 종목/ETF는 6자리 코드(예: 069500)로 입력하면 코스피(.KS) -> 코스닥(.KQ) 순으로 찾는다.
- 미국 종목/ETF는 티커(예: SPY) 그대로 쓴다.
- 배당을 재투자했다고 가정한 수정 종가(auto_adjust)를 쓴다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

FX_TICKER = "KRW=X"  # 1달러 = ? 원

_KR_CODE = re.compile(r"^\d[0-9A-Z]{5}$")  # 069500, 0080G0 같은 한국 종목코드
_KR_SUFFIXES = (".KS", ".KQ")


def clean_ticker(raw: str) -> str:
    return str(raw).strip().upper()


def candidates(ticker: str) -> list[str]:
    """Yahoo Finance에서 찾아볼 심볼 후보."""
    ticker = clean_ticker(ticker)
    if _KR_CODE.match(ticker):
        return [ticker + s for s in _KR_SUFFIXES]
    return [ticker]


def currency_of(symbol: str) -> str:
    return "KRW" if symbol.upper().endswith(_KR_SUFFIXES) else "USD"


def fetch_close(symbol: str, start: date, end: date) -> pd.Series:
    """Yahoo Finance에서 수정 종가를 받아온다. 데이터가 없으면 빈 Series."""
    import yfinance as yf

    df = yf.download(
        symbol,
        start=start,
        end=end + timedelta(days=1),  # yfinance의 end는 그날을 포함하지 않는다
        auto_adjust=True,
        progress=False,
        threads=False,
        multi_level_index=False,
    )
    if df is None or df.empty or "Close" not in df:
        return pd.Series(dtype=float)
    close = df["Close"].dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    return close.astype(float)


@dataclass
class PriceData:
    prices: pd.DataFrame  # 기준 통화로 바꾼 가격, 모든 종목이 데이터가 있는 날부터
    symbols: dict[str, str]  # 입력한 코드 -> Yahoo 심볼
    first_dates: dict[str, pd.Timestamp]  # 종목별 데이터 시작일
    missing: list[str] = field(default_factory=list)  # 데이터를 찾지 못한 코드
    fx_first_date: pd.Timestamp | None = None  # 환율을 썼다면 환율 데이터 시작일


def load_prices(
    tickers: list[str],
    start: date,
    end: date,
    base_currency: str = "KRW",
    fetch=None,
) -> PriceData:
    """여러 종목의 가격을 받아 하나의 표로 맞춘다.

    한국과 미국은 휴장일이 달라 빈 날은 직전 가격으로 채운다.
    통화가 다른 종목은 환율(KRW=X)로 기준 통화에 맞춘다.
    fetch: (심볼, 시작일, 종료일) -> 종가 Series. 기본값은 Yahoo Finance.
    """
    fetch = fetch or fetch_close
    series: dict[str, pd.Series] = {}
    symbols: dict[str, str] = {}
    missing: list[str] = []
    for ticker in dict.fromkeys(clean_ticker(t) for t in tickers):
        for symbol in candidates(ticker):
            close = fetch(symbol, start, end)
            if not close.empty:
                series[ticker] = close
                symbols[ticker] = symbol
                break
        else:
            missing.append(ticker)

    if not series:
        return PriceData(pd.DataFrame(), symbols, {}, missing)

    first_dates = {t: s.index[0] for t, s in series.items()}
    prices = pd.DataFrame(series).sort_index().ffill()

    foreign = [t for t in series if currency_of(symbols[t]) != base_currency]
    fx_first_date = None
    if foreign:
        fx = fetch(FX_TICKER, start, end)
        if fx.empty:
            raise RuntimeError("환율 데이터를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요.")
        fx_first_date = fx.index[0]
        fx = fx.reindex(prices.index.union(fx.index)).sort_index().ffill().reindex(prices.index)
        for t in foreign:
            prices[t] = prices[t] * fx if base_currency == "KRW" else prices[t] / fx

    prices = prices.dropna()
    return PriceData(prices, symbols, first_dates, missing, fx_first_date)
