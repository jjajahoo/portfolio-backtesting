"""백테스트 계산 엔진.

가격표(날짜 x 종목)와 목표 비중을 받아 포트폴리오 가치의 변화를 계산하고,
수익률·위험 지표를 구한다. 화면(Streamlit)과 분리해 두어 테스트하기 쉽게 했다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# 리밸런싱 주기 -> pandas Period 빈도
REBALANCE_FREQS = {"none": None, "monthly": "M", "quarterly": "Q", "yearly": "Y"}


def rebalance_positions(index: pd.DatetimeIndex, rebalance: str) -> list[int]:
    """리밸런싱할 날의 위치(정수 인덱스)를 돌려준다.

    새 달/분기/해가 시작된 뒤 첫 거래일에 리밸런싱한다. 첫날은 처음 매수하는 날이라 제외.
    """
    if rebalance not in REBALANCE_FREQS:
        raise ValueError(f"알 수 없는 리밸런싱 주기: {rebalance}")
    freq = REBALANCE_FREQS[rebalance]
    if freq is None or len(index) < 2:
        return []
    periods = index.to_period(freq)
    changed = periods[1:] != periods[:-1]
    return [i + 1 for i in np.flatnonzero(changed)]


def run_backtest(
    prices: pd.DataFrame,
    weights: dict[str, float],
    rebalance: str = "yearly",
    initial: float = 10_000_000,
    fee_rate: float = 0.0,
) -> pd.Series:
    """포트폴리오 가치를 날짜별로 계산한다.

    prices: 종목별 (배당 반영) 가격. 빈칸이 없어야 한다.
    weights: {종목: 비중}. 합이 1이어야 한다.
    fee_rate: 사고판 금액에 붙는 비용 비율 (0.001 = 0.1%). 처음 매수할 때도 붙는다.
    """
    tickers = list(weights)
    w = np.array([weights[t] for t in tickers], dtype=float)
    if not np.isclose(w.sum(), 1.0):
        raise ValueError("비중의 합은 100%여야 합니다.")
    if (w < 0).any():
        raise ValueError("비중은 0 이상이어야 합니다.")

    px = prices[tickers].to_numpy(dtype=float)
    if np.isnan(px).any() or (px <= 0).any():
        raise ValueError("가격 데이터에 빈칸이나 0 이하 값이 있습니다.")

    n = len(px)
    values = np.empty(n)
    units = initial * (1 - fee_rate) * w / px[0]
    start = 0
    for pos in rebalance_positions(prices.index, rebalance) + [n]:
        values[start:pos] = px[start:pos] @ units
        if pos == n:
            break
        holdings = units * px[pos]
        total = holdings.sum()
        traded = np.abs(total * w - holdings).sum()
        units = (total - traded * fee_rate) * w / px[pos]
        start = pos

    return pd.Series(values, index=prices.index, name="value")


def drawdown(values: pd.Series) -> pd.Series:
    """각 날짜에 직전 최고점 대비 몇 % 빠져 있는지 (0 이하)."""
    return values / values.cummax() - 1


@dataclass
class DrawdownPeriod:
    peak: pd.Timestamp  # 떨어지기 시작한 고점
    trough: pd.Timestamp  # 바닥
    recovery: pd.Timestamp | None  # 고점을 되찾은 날 (아직이면 None)
    depth: float  # 고점 대비 하락률 (음수)

    @property
    def end(self) -> pd.Timestamp:
        return self.recovery if self.recovery is not None else self.trough

    def days(self, last_date: pd.Timestamp) -> int:
        """고점부터 회복까지 걸린 날 수. 회복 전이면 마지막 날까지."""
        end = self.recovery if self.recovery is not None else last_date
        return (end - self.peak).days


def drawdown_periods(values: pd.Series) -> list[DrawdownPeriod]:
    """고점 -> 바닥 -> 회복 구간들을 모두 찾는다."""
    dd = drawdown(values)
    periods: list[DrawdownPeriod] = []
    in_dd = False
    peak = trough = None
    for date, d in dd.items():
        if d < 0 and not in_dd:
            in_dd = True
            peak = values.index[values.index.get_loc(date) - 1]
            trough = date
        elif in_dd and d < 0:
            if d < dd[trough]:
                trough = date
        elif in_dd and d >= 0:
            periods.append(DrawdownPeriod(peak, trough, date, float(dd[trough])))
            in_dd = False
    if in_dd:
        periods.append(DrawdownPeriod(peak, trough, None, float(dd[trough])))
    return periods


def period_returns(values: pd.Series, freq: str) -> pd.Series:
    """기간별 수익률. freq: 'ME'(월말) 또는 'YE'(연말).

    첫 기간은 시작일 가치에서, 마지막 기간은 마지막 날 가치까지 계산한다 (일부 기간 포함).
    """
    ends = values.resample(freq).last().dropna()
    start = pd.Series([values.iloc[0]], index=[values.index[0] - pd.Timedelta(days=1)])
    return pd.concat([start, ends]).pct_change().dropna()


@dataclass
class Metrics:
    start: pd.Timestamp
    end: pd.Timestamp
    initial: float
    final: float
    total_return: float
    cagr: float
    volatility: float | None  # 연 변동성
    sharpe: float | None
    max_drawdown: float
    max_drawdown_period: DrawdownPeriod | None
    longest_drawdown: DrawdownPeriod | None
    best_year: tuple[int, float] | None
    worst_year: tuple[int, float] | None

    @property
    def years(self) -> float:
        return (self.end - self.start).days / 365.25


def compute_metrics(values: pd.Series, risk_free: float = 0.0, invested: float | None = None) -> Metrics:
    """주요 성과 지표를 계산한다.

    risk_free: 연 무위험 수익률 (0.03 = 3%).
    invested: 실제로 넣은 돈. 첫날 가치는 매수 비용이 빠진 금액이라, 수익률은 넣은 돈 기준으로 계산한다.
    """
    start, end = values.index[0], values.index[-1]
    initial = float(invested) if invested is not None else float(values.iloc[0])
    final = float(values.iloc[-1])
    years = (end - start).days / 365.25
    total_return = final / initial - 1
    cagr = (final / initial) ** (1 / years) - 1 if years > 0 else 0.0

    # 변동성·샤프 비율은 월 수익률로 계산한다. 한국/미국 시장의 휴장일이 달라
    # 일별 수익률을 쓰면 값이 왜곡되기 쉽기 때문.
    monthly = period_returns(values, "ME")
    volatility = sharpe = None
    if len(monthly) >= 3:
        volatility = float(monthly.std(ddof=1) * np.sqrt(12))
        rf_monthly = (1 + risk_free) ** (1 / 12) - 1
        if volatility > 0:
            sharpe = float((monthly - rf_monthly).mean() * 12 / volatility)

    periods = drawdown_periods(values)
    deepest = min(periods, key=lambda p: p.depth) if periods else None
    longest = max(periods, key=lambda p: p.days(end)) if periods else None

    yearly = period_returns(values, "YE")
    best = worst = None
    if len(yearly):
        best = (int(yearly.idxmax().year), float(yearly.max()))
        worst = (int(yearly.idxmin().year), float(yearly.min()))

    return Metrics(
        start=start,
        end=end,
        initial=initial,
        final=final,
        total_return=total_return,
        cagr=cagr,
        volatility=volatility,
        sharpe=sharpe,
        max_drawdown=deepest.depth if deepest else 0.0,
        max_drawdown_period=deepest,
        longest_drawdown=longest,
        best_year=best,
        worst_year=worst,
    )
