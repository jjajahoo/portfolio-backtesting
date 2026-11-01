from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import data_sources as ds
import simulation as sim
from sim_charts import build_band_chart, build_technical_chart

SIMULATOR = Path(__file__).resolve().parents[1] / "simulator.py"


@pytest.fixture(scope="module")
def market():
    return sim.make_demo_market("2019-01-01", "2024-12-31")


@pytest.fixture(scope="module")
def ind(market):
    return sim.add_technicals(market)


def _next_day(market, price):
    """다음 거래일 종가를 price 로 붙였을 때의 지표 (해법 검증용)."""
    return sim.simulate(market, price, 1).iloc[-1]


def test_rsi_matches_wilder_definition():
    close = pd.Series([44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08,
                       45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64])
    out = sim.add_technicals(pd.DataFrame({"close": close}))
    d = close.diff()
    up, dn = d.clip(lower=0), (-d).clip(lower=0)
    au, ad = up.iloc[1], dn.iloc[1]
    for i in range(2, len(close)):  # Wilder: 새 평균 = 이전 평균 × 13/14 + 오늘 값 / 14
        au, ad = au * 13 / 14 + up.iloc[i] / 14, ad * 13 / 14 + dn.iloc[i] / 14
    assert out["rsi"].iloc[-1] == pytest.approx(100 * au / (au + ad))
    assert out["rsi"].iloc[:13].isna().all()


def test_indicator_ranges(ind):
    for col in ("rsi", "stoch_k", "stoch_d", "psy"):
        s = ind[col].dropna()
        assert ((s >= 0) & (s <= 100)).all(), col
    assert (ind["bb_up"].dropna() >= ind["bb_low"].dropna()).all()
    assert (ind["from_high"] <= 1e-9).all()
    assert np.allclose(ind["macd_hist"], ind["macd"] - ind["macd_signal"])


@pytest.mark.parametrize("vol", [0.0, 0.012])
@pytest.mark.parametrize("days", [1, 20])
def test_path_ends_at_target(market, days, vol):
    last = market["close"].iloc[-1]
    path = sim.make_path(last, last * 1.1, days, market.index[-1], daily_vol=vol, seed=3)
    assert len(path) == days
    assert path["close"].iloc[-1] == pytest.approx(last * 1.1)
    assert path.index[0] > market.index[-1] and path.index[0].dayofweek < 5
    assert (path["high"] >= path[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (path["low"] <= path[["open", "close"]].min(axis=1) + 1e-9).all()
    assert path["open"].iloc[0] == pytest.approx(last)


def test_straight_path_is_geometric(market):
    last = market["close"].iloc[-1]
    path = sim.make_path(last, last * 0.8, 10, market.index[-1])
    ratios = path["close"] / path["close"].shift(1).fillna(last)
    assert np.allclose(ratios, ratios.iloc[0])


def test_simulate_keeps_history_and_flags_hypo(market):
    out = sim.simulate(market, market["close"].iloc[-1] * 1.05, 15)
    assert out["hypo"].sum() == 15
    assert not out["hypo"].iloc[: len(market)].any()
    real = sim.add_technicals(market)
    assert np.allclose(out["rsi"].iloc[: len(market)], real["rsi"], equal_nan=True)


def test_next_day_solvers_hit_their_targets(market, ind):
    closes = ind["close"].to_numpy()
    row = ind.iloc[-1]

    p = sim.ma_touch_price(closes, 20)
    assert _next_day(market, p)["ma20"] == pytest.approx(p)

    p = sim.ma_cross_price(closes, 5, 20)
    r = _next_day(market, p)
    assert r["ma5"] == pytest.approx(r["ma20"])

    for target in (70, 30):
        p = sim.rsi_target_price(closes[-1], row["rsi_up"], row["rsi_down"], target)
        assert p is not None
        assert _next_day(market, p)["rsi"] == pytest.approx(target, abs=1e-6)

    zero_p, sig_p = sim.macd_target_prices(row["ema_fast"], row["ema_slow"], row["macd_signal"])
    assert _next_day(market, zero_p)["macd"] == pytest.approx(0, abs=1e-6)
    assert _next_day(market, sig_p)["macd_hist"] == pytest.approx(0, abs=1e-6)

    up, low = sim.bollinger_touch_prices(closes)
    assert _next_day(market, up)["bb_up"] == pytest.approx(up)
    assert _next_day(market, low)["bb_low"] == pytest.approx(low)
    assert low < closes[-1] < up


def test_next_day_levels_sorted_and_in_range(ind):
    levels = sim.next_day_levels(ind, max_move=0.3)
    prices = [lv.price for lv in levels]
    last = ind["close"].iloc[-1]
    assert prices == sorted(prices, reverse=True)
    assert all(last * 0.7 <= p <= last * 1.3 for p in prices)
    assert any(lv.label.startswith("RSI 70") for lv in levels)


def test_path_events_detects_golden_cross():
    # 오래 하락하다 멈춘 뒤 급등하면 5일선이 20일선을 뚫고 올라간다.
    idx = pd.bdate_range("2024-01-01", periods=80)
    close = pd.Series(np.r_[np.linspace(3000, 2500, 70), np.full(10, 2500.0)], index=idx)
    out = sim.simulate(pd.DataFrame({"close": close}), 2800, 10)
    events = [text for _, text in sim.path_events(out)]
    assert any("골든크로스: 5일선이 20일선" in t for t in events)
    assert any("RSI 70 돌파" in t for t in events)


def test_valuation_scales_with_level():
    f = sim.Fundamentals.from_ratios(2500, per=12.5, pbr=1.0, div_yield=2.0, fwd_per=10, mcap=2000e12)
    assert f.eps == pytest.approx(200) and f.bps == pytest.approx(2500) and f.dps == pytest.approx(50)
    assert f.roe == pytest.approx(8.0)
    v = sim.valuation_at(2750, f, bond_yield=3.0, gdp=2500e12)
    assert v["per"] == pytest.approx(13.75)
    assert v["pbr"] == pytest.approx(1.1)
    assert v["div_yield"] == pytest.approx(50 / 2750 * 100)
    assert v["yield_gap"] == pytest.approx(200 / 2750 * 100 - 3.0)
    assert v["mcap"] == pytest.approx(2200)  # 조원
    assert v["buffett"] == pytest.approx(88)
    assert sim.level_for(f, "pbr", 0.9) == pytest.approx(2250)
    assert sim.level_for(f, "per", 10) == pytest.approx(2000)
    assert sim.level_for(f, "div_yield", 2.5) == pytest.approx(2000)
    assert sim.level_for(f, "fwd_per", 12) == pytest.approx(3000)


def test_missing_ratios_are_skipped():
    f = sim.Fundamentals.from_ratios(2500, per=None, pbr=1.0, div_yield=float("nan"))
    v = sim.valuation_at(2600, f)
    assert "per" not in v and "div_yield" not in v and "pbr" in v
    assert sim.level_for(f, "per", 10) is None


def test_percentile_and_multiples():
    h = pd.Series(np.linspace(0.8, 1.2, 101))
    assert sim.percentile_of(h, 1.0) == pytest.approx(50, abs=1)
    assert sim.percentile_of(h.head(5), 1.0) is None
    mult = sim.nice_multiples(1.0, h, 6)
    assert mult == sorted(mult) and min(mult) <= 0.82 and max(mult) >= 1.18
    assert len(sim.nice_multiples(12.0, None, 6)) >= 3


def test_compare_table_and_charts(market, ind):
    out = sim.simulate(market, market["close"].iloc[-1] * 0.9, 20, daily_vol=0.01, seed=1)
    table = sim.compare_table(ind.iloc[-1], out.iloc[-1])
    assert len(table) == len(sim.INDICATORS)
    assert set(table.columns) == {"지표", "현재", "가상", "변화", "현재 상태", "가상 상태"}
    fig = build_technical_chart(out)
    assert any(t.name == "가상 경로" for t in fig.data)
    bps = sim.implied_per_point(market)["bps"]
    band = build_band_chart(out, bps, [0.8, 1.0, 1.2], "PBR", start=market.index[-250])
    assert len(band.data) >= 5


def test_fetch_kospi_fundamental_krx_marks_zero_as_missing(monkeypatch):
    class FakeStock:
        @staticmethod
        def get_index_fundamental(start, end, ticker):
            idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
            return pd.DataFrame({"종가": [2669.8, 2607.3], "등락률": [0, -2.3], "PER": [19.1, 0.0],
                                 "PBR": [0.94, 0.92], "배당수익률": [1.96, 2.01]}, index=idx)

    monkeypatch.setattr(ds, "_pykrx_stock", lambda: FakeStock)
    df = ds.fetch_kospi_fundamental_krx("2024-01-02", "2024-01-03")
    assert list(df.columns) == ["per", "pbr", "div_yield"]
    assert np.isnan(df["per"].iloc[1]) and df["pbr"].iloc[1] == pytest.approx(0.92)


def test_load_kospi_valuation_without_krx(monkeypatch):
    idx = pd.bdate_range("2024-01-01", periods=5)
    monkeypatch.setattr(ds, "fetch_kospi_naver", lambda s, e: pd.DataFrame(
        {"kospi_open": 1.0, "kospi_high": 2.0, "kospi_low": 0.5, "kospi_close": range(5)}, index=idx))
    df, report = ds.load_kospi_valuation("2024-01-01", "2024-01-10", use_krx=False)
    assert list(df.columns) == ["open", "high", "low", "close"]
    status = {name: ok for name, ok, _ in report.items}
    assert status == {"KOSPI 지수": True, "PER·PBR·배당수익률": False}


def test_simulator_demo_mode_runs():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(SIMULATOR), default_timeout=90)
    at.run()
    at.sidebar.radio[0].set_value("데모 (가상 데이터)").run()
    assert not at.exception, at.exception
    assert len(at.metric) == 6

    at.button(key="preset_-10").click().run()
    assert not at.exception, at.exception
    last = at.session_state.target_pt / 0.9
    assert at.session_state.target_pct == pytest.approx(-10.0)

    at.number_input(key="target_pt").set_value(round(last * 1.2, 2)).run()
    assert at.session_state.target_pct == pytest.approx(20.0, abs=0.1)

    at.radio(key="path_mode").set_value("변동성 반영 (무작위)").run()
    at.slider(key="days").set_value(1).run()
    assert not at.exception, at.exception
    for period in ("3년", "10년"):
        at.sidebar.selectbox[0].set_value(period).run()
        assert not at.exception, at.exception
