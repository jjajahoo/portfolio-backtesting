import json

import numpy as np
import pandas as pd
import pytest

from qqq_score import conditions as cm
from qqq_score import data, presets, stats
from qqq_score.conditions import Condition
from qqq_score.indicators import BY_KEY, HIGH_GOOD, INDICATORS, LOW_GOOD, RANGE, log_trend_gap, rsi
from qqq_score.scoring import (
    IndicatorSetting, composite_score, default_settings, percentile_score, range_score, run, score_series,
)
from tests.synthetic import make_market_data, truncate


@pytest.fixture(scope="module")
def md():
    m = make_market_data()
    m.custom = pd.Series(np.linspace(10, 30, 200), index=pd.date_range("2005-01-31", periods=200, freq="ME"))
    return m


def all_on():
    s = default_settings()
    for v in s.values():
        v.enabled = True
        v.in_score = True
    return s


# ---------------------------------------------------------------- 데이터


def test_parse_fred_csv_handles_missing_and_header_names():
    for header in ("DATE", "observation_date"):
        text = f"{header},DGS10\n2024-01-02,3.9\n2024-01-03,.\n2024-01-04,4.0\n"
        s = data.parse_fred_csv(text, "DGS10")
        assert list(s.index) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-04")]
        assert s.tolist() == [3.9, 4.0]


def test_to_daily_lag_moves_weekend_release_to_monday():
    # 금요일 관측, 1일 지연 → 토요일 공개 → 월요일부터 사용
    s = pd.Series([1.0, 2.0], index=pd.to_datetime(["2024-01-04", "2024-01-05"]))
    d = data.to_daily(s, lag_days=1, end=pd.Timestamp("2024-01-10"))
    assert d[pd.Timestamp("2024-01-05")] == 1.0
    assert d[pd.Timestamp("2024-01-08")] == 2.0
    assert d.index.max() == pd.Timestamp("2024-01-10")


def test_fed_target_rate_splices_old_and_new():
    old = pd.Series([1.0, 1.0], index=pd.to_datetime(["2008-12-01", "2008-12-15"]))
    new = pd.Series([0.25], index=pd.to_datetime(["2008-12-16"]))
    s = data.fed_target_rate({"DFEDTAR": old, "DFEDTARU": new})
    assert s.tolist() == [1.0, 1.0, 0.25]


def test_parse_custom_csv():
    s = data.parse_custom_csv(b"date,pe\n2020-01-31,25\n2020-02-29,x\n2020-03-31,20\n")
    assert s.tolist() == [25.0, 20.0]


# ---------------------------------------------------------------- 지표·점수


def test_rsi_extremes():
    up = pd.Series(np.arange(1, 60, dtype=float))
    assert rsi(up, 14).dropna().eq(100).all()
    down = pd.Series(np.arange(60, 1, -1, dtype=float))
    assert rsi(down, 14).dropna().eq(0).all()


def test_range_score_direction():
    raw = pd.Series([20.0, 30.0, 50.0, 70.0, 80.0])
    st = IndicatorSetting(True, True, 1, RANGE, LOW_GOOD, 10, 30, 70)
    assert score_series(raw, st).tolist() == [100, 100, 50, 0, 0]
    st.direction = HIGH_GOOD
    assert score_series(raw, st).tolist() == [0, 0, 50, 100, 100]
    # 하한/상한을 거꾸로 넣어도 같은 결과
    assert range_score(raw, 70, 30).tolist() == range_score(raw, 30, 70).tolist()


def test_percentile_score_bounds_and_monotone():
    raw = pd.Series(np.arange(1000, dtype=float))
    p = percentile_score(raw, 1).dropna()
    assert p.eq(100).all()  # 매일 신고가 → 항상 최고 백분위
    noisy = pd.Series(np.random.default_rng(0).normal(size=3000))
    q = percentile_score(noisy, 0).dropna()
    assert q.between(0, 100).all()


def test_log_trend_gap_zero_on_exact_trend():
    idx = pd.bdate_range("1990-01-01", periods=4000)
    years = (idx - idx[0]).days / 365.25
    price = pd.Series(100 * np.exp(0.1 * years), index=idx)
    gap = log_trend_gap(price, 5, min_years=2).dropna()
    assert len(gap) > 0
    assert gap.abs().max() < 1e-6


def test_composite_weighting_and_require_all():
    idx = pd.RangeIndex(3)
    scores = pd.DataFrame({"a": [100.0, 0.0, np.nan], "b": [0.0, 0.0, 50.0]}, index=idx)
    comp = composite_score(scores, {"a": 3, "b": 1}, require_all=False)
    assert comp.tolist() == [75.0, 0.0, 50.0]
    comp_all = composite_score(scores, {"a": 3, "b": 1}, require_all=True)
    assert np.isnan(comp_all.iloc[2])


def test_all_indicators_compute(md):
    res = run(md, all_on(), require_all=False)
    assert not res.errors, res.errors
    assert set(res.raw.columns) == {i.key for i in INDICATORS}
    for key in res.scores.columns:
        s = res.scores[key].dropna()
        assert len(s) > 1000, key
        assert s.between(0, 100).all(), key
    assert res.composite.dropna().between(0, 100).all()


def test_no_lookahead(md):
    """T 이후 데이터를 지워도 T까지의 원값·점수가 같아야 한다 (미래 정보 누수 없음)."""
    settings = all_on()
    full = run(md, settings, require_all=False)
    for cut in ["2008-06-13", "2016-03-01"]:
        t = pd.Timestamp(cut)
        part = run(truncate(md, t), settings, require_all=False)
        for frame_full, frame_part in [(full.raw, part.raw), (full.scores, part.scores)]:
            a = frame_full.loc[:t]
            b = frame_part.loc[:t]
            pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9, atol=1e-9)
        pd.testing.assert_series_equal(full.composite.loc[:t], part.composite.loc[:t], rtol=1e-9, atol=1e-9)


def test_disabled_indicator_not_computed(md):
    s = default_settings()
    for v in s.values():
        v.enabled = False
    s["rsi"].enabled = True
    res = run(md, s)
    assert list(res.raw.columns) == ["rsi"]


# ---------------------------------------------------------------- 조건·통계


def _result_with(series: dict[str, list[float]]):
    idx = pd.bdate_range("2020-01-01", periods=len(next(iter(series.values()))))
    raw = pd.DataFrame(series, index=idx)
    from qqq_score.scoring import EngineResult
    return EngineResult(raw, raw * 0 + 50, pd.Series(50.0, index=idx), {}, {}, {k: k for k in series}, {})


def test_condition_ops_and_combination():
    res = _result_with({"a": [1, 2, 3, 4, np.nan], "b": [5, 5, 0, 0, 5]})
    ge = Condition("raw:a", ">=", 3)
    hit, valid = cm.evaluate(res, [ge])
    assert hit.tolist() == [False, False, True, True, False]
    assert valid.tolist() == [True, True, True, True, False]
    btw = Condition("raw:a", "between", 3.5, 1.5)  # 순서 무관
    assert cm.evaluate(res, [btw])[0].tolist() == [False, True, True, False, False]
    out = Condition("raw:a", "outside", 1.5, 3.5)
    assert cm.evaluate(res, [out])[0].tolist() == [True, False, False, True, False]
    b5 = Condition("raw:b", ">", 1)
    assert cm.evaluate(res, [ge, b5], "and")[0].tolist() == [False, False, False, False, False]
    assert cm.evaluate(res, [ge, b5], "or")[0].tolist() == [True, True, True, True, False]
    assert cm.evaluate(res, [Condition("raw:zzz", ">", 1)]) is None


def test_periods_and_merge():
    idx = pd.bdate_range("2020-01-01", periods=12)
    m = pd.Series([0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0], index=idx).astype(bool)
    p = stats.periods(m)
    assert p["거래일"].tolist() == [2, 1, 3]
    assert p["start"].tolist() == [idx[1], idx[4], idx[8]]
    merged = stats.periods(m, merge_gap=1)
    assert merged["거래일"].tolist() == [4, 3]
    assert merged["해당일"].tolist() == [3, 3]
    assert stats.periods(pd.Series(False, index=idx)).empty


def test_forward_returns_and_worst():
    close = pd.Series([100.0, 90.0, 120.0, 110.0], index=pd.bdate_range("2020-01-01", periods=4))
    fwd = stats.forward_returns(close, {"2d": 2})
    assert fwd["2d"].iloc[0] == pytest.approx(20.0)
    assert np.isnan(fwd["2d"].iloc[2])
    worst = stats.forward_worst(close, {"2d": 2})
    assert worst["2d"].iloc[0] == pytest.approx(-10.0)
    assert worst["2d"].iloc[1] == pytest.approx(0.0)


def test_in_sample_excludes_windows_crossing_split():
    idx = pd.bdate_range("2020-01-01", periods=100)
    split = stats.Split(True, idx[50])
    is_m = stats.sample_mask(idx, split, stats.IN_SAMPLE, 10)
    oos = stats.sample_mask(idx, split, stats.OUT_SAMPLE, 10)
    assert is_m.sum() == 40          # 0..39: 40+10=50 은 검증 구간에 걸침
    assert is_m.iloc[39] and not is_m.iloc[40]
    assert oos.sum() == 40           # 50..89 (90부터는 10일 뒤 가격 없음)
    assert not (is_m & oos).any()


def test_condition_table_shapes(md):
    res = run(md, default_settings())
    hit, valid = cm.evaluate(res, [Condition("raw:fed_rate", ">=", 4)])
    split = stats.Split(True, pd.Timestamp("2016-01-01"))
    t = stats.condition_table(md.close, hit, valid, split)
    assert len(t) == 4 * 2 * 4
    base = t[(t["그룹"] == "전체") & (t["기간"] == "12개월") & (t["표본구간"] == "조정 구간")]
    assert base["표본수"].iloc[0] > 3000
    per = stats.periods(hit, 5)
    pt = stats.period_table(md.close, per)
    assert len(pt) == len(per)


def test_rank_ic_detects_signal():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2000-01-01", periods=3000)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), index=idx)
    fwd = stats.forward_returns(close, {"1m": 21})["1m"]
    score = fwd.rank(pct=True) * 100  # 미래를 아는 점수 → IC≈1
    ic = stats.rank_ic(score, fwd, pd.Series(True, index=idx))
    assert ic > 0.99


def test_score_bucket_table(md):
    res = run(md, default_settings())
    t = stats.score_bucket_table(res.composite, md.close, stats.Split(False, pd.Timestamp("2016-01-01")), 4)
    assert set(t["점수 구간"]) == {"0~25", "25~50", "50~75", "75~100"}
    assert t["표본수"].sum() > 0


# ---------------------------------------------------------------- 설정 파일


def test_preset_roundtrip_and_sanitize():
    s = default_settings()
    s["bb"].params["k"] = 2.5
    s["rsi"].enabled = False
    conds = [Condition("raw:fed_rate", "between", 3, 5)]
    text = presets.to_json(s, conds, "or", {"split_on": False})
    s2, c2, how, opts = presets.from_json(text)
    assert s2["bb"].params["k"] == 2.5 and s2["rsi"].enabled is False
    assert c2 == conds and how == "or" and opts == {"split_on": False}

    bad = json.loads(text)
    bad["indicators"]["rsi"]["params"]["n"] = 9999          # 범위 밖 → 최대값으로
    bad["indicators"]["rsi"]["method"] = "nonsense"         # 모르는 값 → 기본값
    bad["indicators"]["unknown"] = {"enabled": True}         # 모르는 지표 → 무시
    bad["conditions"].append({"operand": "composite", "op": "~~", "value": 1})
    s3, c3, _, _ = presets.from_json(json.dumps(bad))
    assert s3["rsi"].params["n"] == BY_KEY["rsi"].params[0].max
    assert s3["rsi"].method == BY_KEY["rsi"].score.method
    assert len(c3) == 1


def test_chart_keeps_shading_next_to_threshold_lines(md):
    from qqq_score import charts

    res = run(md, default_settings())
    hit, _ = cm.evaluate(res, [Condition("raw:fed_rate", ">=", 4)])
    shade = stats.periods(hit)
    assert len(shade) > 0
    sub = charts.Subplot("기준금리", res.raw["fed_rate"], [4.0])
    fig = charts.main_chart(md.close, res.overlays, [sub], shade, pd.Timestamp("2016-01-01"), True, "light")
    rects = [s for s in fig.layout.shapes if s.type == "rect"]
    lines = [s for s in fig.layout.shapes if s.type == "line"]
    assert len(rects) == len(shade)
    assert len(lines) == 2  # 기준선 + 검증 구간 시작선
