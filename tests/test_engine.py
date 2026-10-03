import numpy as np
import pandas as pd
import pytest

from backtester.engine import (
    compute_metrics,
    drawdown,
    drawdown_periods,
    period_returns,
    rebalance_positions,
    run_backtest,
)


def prices_from(data: dict[str, list[float]], dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame(data, index=pd.to_datetime(dates))


def test_rebalance_positions_first_trading_day_of_each_period():
    idx = pd.to_datetime(["2020-01-02", "2020-01-31", "2020-02-03", "2020-03-02", "2020-04-01", "2021-01-04"])
    assert rebalance_positions(idx, "monthly") == [2, 3, 4, 5]
    assert rebalance_positions(idx, "quarterly") == [4, 5]
    assert rebalance_positions(idx, "yearly") == [5]
    assert rebalance_positions(idx, "none") == []
    with pytest.raises(ValueError):
        rebalance_positions(idx, "weekly")


def test_single_asset_tracks_price():
    prices = prices_from({"A": [100, 110, 99]}, ["2020-01-02", "2020-01-03", "2020-01-06"])
    values = run_backtest(prices, {"A": 1.0}, "none", initial=1000)
    assert values.tolist() == pytest.approx([1000, 1100, 990])


def test_yearly_rebalance_restores_target_weights():
    # A는 첫해에 2배, B는 그대로. 둘째 해 첫 거래일에 50:50으로 다시 맞춘다.
    dates = ["2020-01-02", "2020-12-31", "2021-01-04", "2021-12-31"]
    prices = prices_from({"A": [10, 20, 20, 40], "B": [10, 10, 10, 10]}, dates)

    rebalanced = run_backtest(prices, {"A": 0.5, "B": 0.5}, "yearly", initial=100)
    # 첫해 말: A 100 + B 50 = 150 -> 75:75로 재조정 -> A가 다시 2배: 150 + 75 = 225
    assert rebalanced.tolist() == pytest.approx([100, 150, 150, 225])

    held = run_backtest(prices, {"A": 0.5, "B": 0.5}, "none", initial=100)
    # 그대로 두면 A 200 + B 50 = 250
    assert held.tolist() == pytest.approx([100, 150, 150, 250])


def test_fees_on_initial_purchase_and_rebalance():
    dates = ["2020-01-02", "2020-12-31", "2021-01-04"]
    prices = prices_from({"A": [10, 20, 20], "B": [10, 10, 10]}, dates)
    values = run_backtest(prices, {"A": 0.5, "B": 0.5}, "yearly", initial=100, fee_rate=0.01)
    assert values.iloc[0] == pytest.approx(99)  # 처음 살 때 1%
    # 첫해 말 A 99, B 49.5 -> 합 148.5, 목표 74.25씩 -> 24.75씩 사고팔아 49.5 거래 -> 비용 0.495
    assert values.iloc[1] == pytest.approx(148.5)
    assert values.iloc[2] == pytest.approx(148.5 - 0.495)


def test_rejects_bad_weights_and_prices():
    prices = prices_from({"A": [10, 11], "B": [10, np.nan]}, ["2020-01-02", "2020-01-03"])
    with pytest.raises(ValueError, match="100%"):
        run_backtest(prices, {"A": 0.5, "B": 0.4})
    with pytest.raises(ValueError, match="빈칸"):
        run_backtest(prices, {"A": 0.5, "B": 0.5})


def test_drawdown_periods_peak_trough_recovery():
    values = pd.Series(
        [100, 150, 90, 120, 160, 150],
        index=pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01", "2020-05-01", "2020-06-01"]),
    )
    assert drawdown(values).min() == pytest.approx(-0.4)
    first, second = drawdown_periods(values)
    assert (first.peak, first.trough, first.recovery) == tuple(pd.to_datetime(["2020-02-01", "2020-03-01", "2020-05-01"]))
    assert first.depth == pytest.approx(-0.4)
    assert second.recovery is None  # 마지막 하락은 아직 회복 전
    assert second.days(values.index[-1]) == 31


def test_period_returns_includes_partial_first_and_last_year():
    values = pd.Series(
        [100, 110, 121, 133.1],
        index=pd.to_datetime(["2020-06-01", "2020-12-31", "2021-12-31", "2022-03-01"]),
    )
    yearly = period_returns(values, "YE")
    assert list(yearly.index.year) == [2020, 2021, 2022]
    assert yearly.tolist() == pytest.approx([0.1, 0.1, 0.1])


def test_compute_metrics_known_values():
    # 정확히 10년 동안 매일 같은 비율로 올라 2배가 되는 경우
    dates = pd.bdate_range("2010-01-01", "2020-01-01")
    years = (dates[-1] - dates[0]).days / 365.25
    t = np.array([(d - dates[0]).days / 365.25 for d in dates])
    values = pd.Series(100 * 2 ** (t / years), index=dates)
    m = compute_metrics(values, risk_free=0.0)
    assert m.total_return == pytest.approx(1.0)
    assert m.cagr == pytest.approx(2 ** (1 / years) - 1)
    assert m.max_drawdown == 0
    assert m.max_drawdown_period is None
    assert m.volatility < 0.01  # 거의 일정하게 오름 (달마다 날 수가 달라 0은 아님)


def test_sharpe_matches_formula():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2015-01-01", "2020-12-31")
    values = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, len(dates)))), index=dates)
    m = compute_metrics(values, risk_free=0.02)
    monthly = period_returns(values, "ME")
    rf_m = 1.02 ** (1 / 12) - 1
    expected = (monthly - rf_m).mean() * 12 / (monthly.std() * np.sqrt(12))
    assert m.sharpe == pytest.approx(expected)
    assert m.best_year[1] >= m.worst_year[1]


def test_short_period_has_no_volatility_or_sharpe():
    values = pd.Series([100, 101, 102], index=pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]))
    m = compute_metrics(values)
    assert m.volatility is None and m.sharpe is None
