"""테스트용 가짜 시장 데이터 (네트워크 없이 앱 전체를 돌려보기 위함)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from qqq_score.data import MarketData


def _walk(rng, n, start, scale, lo=None, hi=None):
    x = start + np.cumsum(rng.normal(0, scale, n))
    if lo is not None or hi is not None:
        x = np.clip(x, lo, hi)
    return x


def make_prices(rng, start="1999-03-10", end="2026-10-09", drift=0.0004, vol=0.017, p0=100.0) -> pd.DataFrame:
    idx = pd.bdate_range(start, end)
    ret = rng.normal(drift, vol, len(idx))
    close = p0 * np.exp(np.cumsum(ret))
    spread = np.abs(rng.normal(0, vol / 2, len(idx)))
    return pd.DataFrame({
        "Open": close * (1 + rng.normal(0, vol / 4, len(idx))),
        "High": close * (1 + spread),
        "Low": close * (1 - spread),
        "Close": close,
        "Volume": rng.integers(1e7, 1e8, len(idx)).astype(float),
    }, index=idx)


def make_fred(rng) -> dict[str, pd.Series]:
    daily = pd.bdate_range("1982-01-01", "2026-10-08")
    monthly = pd.date_range("1960-01-01", "2026-08-01", freq="MS")
    weekly = pd.date_range("2002-12-18", "2026-10-07", freq="W-WED")

    # 기준금리: 평균 회귀하는 0.25%p 단위 계단
    x = np.empty(len(daily))
    x[0] = 8.0
    shocks = rng.normal(0, 0.04, len(daily))
    for i in range(1, len(daily)):
        x[i] = min(max(x[i - 1] + 0.002 * (3.5 - x[i - 1]) + shocks[i], 0.0), 10.0)
    steps = np.round(x * 4) / 4
    target = pd.Series(steps, index=daily)
    split = pd.Timestamp("2008-12-16")
    out = {
        "DFEDTAR": target[target.index < split],
        "DFEDTARU": target[target.index >= split],
        "DGS10": pd.Series(_walk(rng, len(daily), 6.0, 0.05, 0.3, 15), index=daily),
        "T10Y2Y": pd.Series(_walk(rng, len(daily), 1.0, 0.03, -2, 3), index=daily),
        "DFII10": pd.Series(_walk(rng, len(daily), 1.5, 0.03, -2, 4), index=daily)["2003-01-02":],
        "VIXCLS": pd.Series(_walk(rng, len(daily), 20, 0.8, 9, 80), index=daily)["1990-01-02":],
        "BAA10Y": pd.Series(_walk(rng, len(daily), 2.0, 0.02, 0.5, 6), index=daily),
        "CPIAUCSL": pd.Series(30 * np.exp(np.cumsum(rng.normal(0.003, 0.003, len(monthly)))), index=monthly),
        "UNRATE": pd.Series(_walk(rng, len(monthly), 5.0, 0.15, 3, 15), index=monthly),
        "M2SL": pd.Series(300 * np.exp(np.cumsum(rng.normal(0.005, 0.004, len(monthly)))), index=monthly),
        "WALCL": pd.Series(7e5 * np.exp(np.cumsum(rng.normal(0.001, 0.01, len(weekly)))), index=weekly),
        "DTWEXBGS": pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.003, len(daily)))), index=daily)["2006-01-02":],
        "NFCI": pd.Series(_walk(rng, len(weekly), 0.0, 0.05, -1.5, 3), index=weekly),
    }
    # FRED처럼 가끔 비는 날
    for key in ("DGS10", "T10Y2Y", "VIXCLS"):
        s = out[key].copy()
        s.iloc[rng.choice(len(s), 50, replace=False)] = np.nan
        out[key] = s.dropna()
    return out


def make_market_data(seed: int = 7) -> MarketData:
    rng = np.random.default_rng(seed)
    prices = make_prices(rng)
    trend = make_prices(rng, start="1985-10-01", p0=100.0)
    return MarketData(prices=prices, trend_prices=trend, fred=make_fred(rng))


def truncate(md: MarketData, end: pd.Timestamp) -> MarketData:
    """end 이후 데이터를 모두 지운 사본 (그 시점에 알 수 있던 정보만)."""
    return MarketData(
        prices=md.prices[md.prices.index <= end],
        trend_prices=None if md.trend_prices is None else md.trend_prices[md.trend_prices.index <= end],
        fred={k: v[v.index <= end] for k, v in md.fred.items()},
        custom=None if md.custom is None else md.custom[md.custom.index <= end],
    )
