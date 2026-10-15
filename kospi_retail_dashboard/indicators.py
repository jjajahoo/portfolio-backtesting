"""원자료로부터 파생 지표와 '개인 투자심리 지수'를 계산한다."""

from __future__ import annotations

import numpy as np
import pandas as pd

# 심리지수 구성 요소: (컬럼, 한글 이름)
SENTIMENT_COMPONENTS = {
    "retail_net_roll": "개인 순매수(이동합)",
    "retail_share": "개인 거래대금 비중",
    "deposit_chg": "예탁금 증가율",
    "credit_chg": "신용융자 증가율",
}

DERIVED_LABELS = {
    "retail_net_cum": "개인 누적 순매수(원)",
    "retail_net_roll": "개인 순매수 이동합(원)",
    "retail_volume": "개인 거래량(주, 매수+매도)",
    "retail_value": "개인 거래대금(원, 매수+매도)",
    "retail_share": "개인 거래대금 비중(%)",
    "deposit_chg": "예탁금 증가율(%)",
    "credit_chg": "신용융자 증가율(%)",
    "credit_to_deposit": "신용융자/예탁금(%)",
    "sentiment": "개인 투자심리 지수(z)",
    "kospi_ret": "KOSPI 수익률(%)",
    "kospi_fwd_ret": "KOSPI 향후 수익률(%)",
}


def rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    min_periods = max(window // 2, 10)
    mean = s.rolling(window, min_periods=min_periods).mean()
    std = s.rolling(window, min_periods=min_periods).std()
    return (s - mean) / std.replace(0, np.nan)


def add_indicators(df: pd.DataFrame, flow_window: int = 20, z_window: int = 120) -> pd.DataFrame:
    """
    flow_window : 순매수 이동합·증가율 계산 기간(거래일)
    z_window    : 심리지수 표준화(z-score) 기간(거래일)
    """
    out = df.copy()
    has = out.columns.__contains__

    if has("retail_net"):
        net = out["retail_net"]
        out["retail_net_cum"] = net.fillna(0).cumsum().where(net.notna().cummax())
        out["retail_net_roll"] = net.rolling(flow_window, min_periods=flow_window // 2).sum()
    if has("retail_buy_volume") and has("retail_sell_volume"):
        out["retail_volume"] = out["retail_buy_volume"] + out["retail_sell_volume"]
    if has("retail_buy_value") and has("retail_sell_value"):
        out["retail_value"] = out["retail_buy_value"] + out["retail_sell_value"]
        if has("market_value"):
            # 시장 거래대금 = 매수 총액 = 매도 총액, 개인 비중은 (매수+매도)/(2×시장)
            out["retail_share"] = out["retail_value"] / (2 * out["market_value"]) * 100
    if has("deposit"):
        out["deposit_chg"] = out["deposit"].pct_change(flow_window, fill_method=None) * 100
    credit_col = "credit" if has("credit") else ("credit_kospi" if has("credit_kospi") else None)
    if credit_col:
        out["credit_chg"] = out[credit_col].pct_change(flow_window, fill_method=None) * 100
        if has("deposit"):
            out["credit_to_deposit"] = out[credit_col] / out["deposit"] * 100
    if has("kospi_close"):
        close = out["kospi_close"]
        out["kospi_ret"] = close.pct_change(flow_window, fill_method=None) * 100
        out["kospi_fwd_ret"] = (close.shift(-flow_window) / close - 1) * 100

    comps = [c for c in SENTIMENT_COMPONENTS if c in out]
    if len(comps) >= 2:
        z = pd.concat({c: rolling_zscore(out[c], z_window) for c in comps}, axis=1)
        out["sentiment"] = z.mean(axis=1, skipna=True).where(z.notna().sum(axis=1) >= 2)
        for c in comps:
            out[f"z_{c}"] = z[c]
    return out


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    """각 지표와 KOSPI 동행·향후 수익률의 상관계수."""
    targets = [c for c in ("kospi_ret", "kospi_fwd_ret") if c in df]
    rows = [
        c
        for c in ("retail_net_roll", "retail_share", "deposit_chg", "credit_chg",
                  "credit_to_deposit", "sentiment")
        if c in df and df[c].notna().sum() > 30
    ]
    if not targets or not rows:
        return pd.DataFrame()
    table = pd.DataFrame(
        {DERIVED_LABELS[t]: [df[r].corr(df[t]) for r in rows] for t in targets},
        index=[DERIVED_LABELS[r] for r in rows],
    )
    return table.round(2)


def make_demo_data(start="2021-01-01", end=None, seed: int = 7) -> pd.DataFrame:
    """네트워크 없이 화면을 확인하기 위한 가상 데이터 (실제 시장 데이터 아님)."""
    rng = np.random.default_rng(seed)
    end = end or pd.Timestamp.today().normalize()
    idx = pd.bdate_range(start, end, name="date")
    n = len(idx)

    regime = np.sin(np.linspace(0, 5 * np.pi, n)) * 0.0006
    ret = regime + rng.normal(0, 0.011, n)
    close = 2900 * np.exp(np.cumsum(ret))
    open_ = close * np.exp(rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n)))

    market_value = np.abs(rng.normal(11e12, 2.5e12, n)) + 4e12
    retail_share = np.clip(0.55 + 6 * pd.Series(ret).rolling(20, min_periods=1).mean().values
                           + rng.normal(0, 0.04, n), 0.3, 0.8)
    retail_value = 2 * market_value * retail_share
    # 개인은 하락 시 순매수(역추세) 경향
    retail_net = -ret * 3e13 + rng.normal(0, 3e11, n)
    buy = (retail_value + retail_net) / 2
    sell = (retail_value - retail_net) / 2
    price_per_share = 25000 * close / 2900

    deposit = 55e12 * np.exp(np.cumsum(regime * 1.5 + rng.normal(0, 0.006, n)))
    credit = 18e12 * np.exp(np.cumsum(ret * 0.35 + regime + rng.normal(0, 0.002, n)))

    return pd.DataFrame(
        {
            "kospi_open": open_,
            "kospi_high": high,
            "kospi_low": low,
            "kospi_close": close,
            "kospi_value": market_value,
            "retail_net": retail_net,
            "retail_buy_value": buy,
            "retail_sell_value": sell,
            "market_value": market_value,
            "retail_buy_volume": buy / price_per_share,
            "retail_sell_volume": sell / price_per_share,
            "deposit": deposit,
            "credit": credit,
            "credit_kospi": credit * 0.55,
        },
        index=idx,
    )
