from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import data_sources as ds
from charts import PANELS, available_panels, build_compare_chart, build_main_chart
from indicators import add_indicators, correlation_table, make_demo_data

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_parse_naver_chart():
    xml = (
        '<?xml version="1.0" encoding="EUC-KR" ?><protocol><chartdata symbol="KOSPI">'
        '<item data="20240102|2645.47|2675.80|2641.84|2669.81|414000" />'
        '<item data="20240103|2669.06|2672.80|2604.79|2607.31|557000" /></chartdata></protocol>'
    )
    df = ds.parse_naver_chart(xml)
    assert list(df.columns) == ["kospi_open", "kospi_high", "kospi_low", "kospi_close"]
    assert df.loc["2024-01-03", "kospi_close"] == pytest.approx(2607.31)


def test_parse_naver_deposit():
    html = """<table class="type_1"><thead>
    <tr><th rowspan=2>날짜</th><th colspan=2>고객예탁금</th><th colspan=2>신용잔고</th></tr>
    <tr><th>금액</th><th>증감</th><th>금액</th><th>증감</th></tr></thead>
    <tr><td>24.10.11</td><td>537,105</td><td>1,234</td><td>178,900</td><td>-120</td></tr>
    <tr><td></td><td></td><td></td><td></td><td></td></tr>
    <tr><td>24.10.10</td><td>535,871</td><td>-500</td><td>179,020</td><td>30</td></tr>
    </table>"""
    df = ds.parse_naver_deposit(html)
    assert len(df) == 2
    assert df.loc["2024-10-11", "deposit"] == pytest.approx(537105e8)
    assert df.loc["2024-10-10", "credit"] == pytest.approx(179020e8)


def test_parse_naver_investor():
    html = """<table><tr><th>날짜</th><th>개인</th><th>외국인</th><th>기관계</th></tr>
    <tr><td>24.10.11</td><td>-1,234</td><td>2,000</td><td>-700</td></tr></table>"""
    df = ds.parse_naver_investor(html)
    assert df.loc["2024-10-11", "retail_net"] == pytest.approx(-1234e8)
    assert df.loc["2024-10-11", "foreign_net"] == pytest.approx(2000e8)
    assert df.loc["2024-10-11", "inst_net"] == pytest.approx(-700e8)


def test_parse_naver_daily_quote():
    html = """<table><thead><tr><th>날짜</th><th>매매기준율</th><th>전일대비</th></tr></thead>
    <tr><td>2024.10.11</td><td>1,350.50</td><td>2.5</td></tr>
    <tr><td>2024.10.10</td><td>1,348.00</td><td>-1.0</td></tr></table>"""
    df = ds.parse_naver_daily_quote(html, "usdkrw")
    assert df.loc["2024-10-11", "usdkrw"] == pytest.approx(1350.5)
    assert len(df) == 2


def test_parse_ecos():
    payload = {"StatisticSearch": {"row": [
        {"TIME": "20241011", "DATA_VALUE": "2.95"}, {"TIME": "20241010", "DATA_VALUE": "2.97"},
    ]}}
    df = ds.parse_ecos(payload, "bond_10y")
    assert df.index.is_monotonic_increasing
    assert df.loc["2024-10-11", "bond_10y"] == pytest.approx(2.95)
    with pytest.raises(ds.SourceError, match="인증키"):
        ds.parse_ecos({"RESULT": {"CODE": "INFO-100", "MESSAGE": "인증키가 유효하지 않습니다."}}, "x")


def test_parse_kofia_scales_million_won_to_won():
    payload = {"ds1": [
        {"TMPV1": "20241010", "TMPV2": "53587100", "TMPV3": "9300000"},
        {"TMPV1": "20241011", "TMPV2": "53710500", "TMPV3": "9310000"},
    ]}
    df = ds.parse_kofia(payload, {"TMPV2": "deposit", "TMPV3": "credit_kospi"})
    assert df.loc["2024-10-11", "deposit"] == pytest.approx(53710500e6)  # 약 53.7조원
    assert df.index.is_monotonic_increasing


def test_align_uses_kospi_trading_days_and_ffills_levels():
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    kospi = pd.DataFrame({"kospi_close": [1.0, 2.0, 3.0]}, index=idx)
    funds = pd.DataFrame({"deposit": [5e13]}, index=pd.to_datetime(["2024-01-02"]))
    extra = pd.DataFrame({"retail_net": [1.0]}, index=pd.to_datetime(["2024-01-06"]))  # 휴장일
    out = ds.align([kospi, funds, extra])
    assert list(out.index) == list(idx)
    assert out["deposit"].tolist() == [5e13] * 3


def test_load_market_data_falls_back(monkeypatch):
    idx = pd.bdate_range("2024-01-01", periods=5)

    def boom(*a, **k):
        raise ConnectionError("down")

    monkeypatch.setattr(ds, "fetch_kospi_naver", lambda s, e: pd.DataFrame({"kospi_close": range(5)}, index=idx))
    monkeypatch.setattr(ds, "fetch_investor_naver", boom)
    monkeypatch.setattr(ds, "fetch_funds_kofia", boom)
    monkeypatch.setattr(ds, "fetch_funds_naver", lambda s, e: pd.DataFrame({"deposit": [5e13] * 5}, index=idx))
    monkeypatch.setattr(ds, "fetch_bond_naver", lambda s, e: pd.DataFrame({"bond_3y": [3.0] * 5}, index=idx))
    monkeypatch.setattr(ds, "fetch_fx_naver", boom)
    monkeypatch.delenv("ECOS_KEY", raising=False)
    df, report = ds.load_market_data("2024-01-01", "2024-01-10", use_krx=False)
    status = {name: ok for name, ok, _ in report.items}
    assert status == {
        "KOSPI 지수": True, "개인 수급": False, "예탁금·신용융자": True,
        "PER·PBR·배당수익률": False, "국고채 금리": True, "원/달러 환율": False,
    }
    assert {"kospi_close", "deposit", "bond_3y"} <= set(df.columns)


def test_load_market_data_prefers_ecos(monkeypatch):
    idx = pd.bdate_range("2024-01-01", periods=5)
    calls = []

    def fake_ecos(key, names, s, e):
        calls.append((key, tuple(names)))
        return pd.DataFrame({n: [1.0] * 5 for n in names}, index=idx)

    def boom(*a, **k):
        raise ConnectionError("down")

    monkeypatch.setattr(ds, "fetch_kospi_naver", lambda s, e: pd.DataFrame({"kospi_close": range(5)}, index=idx))
    for name in ("fetch_investor_naver", "fetch_funds_kofia", "fetch_funds_naver", "fetch_bond_naver", "fetch_fx_naver"):
        monkeypatch.setattr(ds, name, boom)
    monkeypatch.setattr(ds, "fetch_ecos", fake_ecos)
    df, report = ds.load_market_data("2024-01-01", "2024-01-10", use_krx=False, ecos_key="KEY")
    assert calls == [("KEY", ("bond_3y", "bond_10y")), ("KEY", ("usdkrw",))]
    assert {"bond_10y", "usdkrw"} <= set(df.columns)


def test_indicators_on_demo():
    df = add_indicators(make_demo_data("2022-01-01", "2024-12-31"), flow_window=20, z_window=120)
    share = df["retail_share"].dropna()
    assert ((share > 0) & (share < 100)).all()
    assert np.allclose(df["retail_net_cum"].iloc[-1], df["retail_net"].sum())
    assert df["sentiment"].notna().sum() > 400
    assert set(available_panels(df)) == set(PANELS)
    corr = correlation_table(df)
    assert corr.abs().le(1).all().all()


def test_indicators_with_partial_data():
    """KRX 없이 (순매수·예탁금·신용만) 있어도 동작해야 한다."""
    demo = make_demo_data("2023-01-01", "2024-06-30")[["kospi_close", "retail_net", "deposit", "credit"]]
    df = add_indicators(demo)
    assert "retail_share" not in df and "retail_volume" not in df
    assert df["sentiment"].notna().any()
    assert "개인 거래량" not in available_panels(df)


def test_charts_build():
    df = add_indicators(make_demo_data("2024-01-01", "2024-12-31"))
    fig = build_main_chart(df, list(PANELS), candle=True, log_scale=True)
    assert len(fig.data) >= len(PANELS) + 1
    cmp = build_compare_chart(df, ["kospi_close", "deposit", "credit"], "index", 120)
    assert len(cmp.data) == 3


def test_csv_roundtrip(tmp_path):
    demo = make_demo_data("2024-01-01", "2024-03-31")
    path = tmp_path / "x.csv"
    demo.rename(columns=ds.LABELS).rename_axis("날짜").to_csv(path)
    out = ds.read_uploaded_csv(path)
    assert np.allclose(out["deposit"], demo["deposit"])


def test_app_demo_mode_runs():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value("데모 (가상 데이터)").run()
    assert not at.exception, at.exception
    assert len(at.metric) >= 5 + 1 + 4  # 상단 요약 + 시장 위치(종합 + 카테고리 4)
    assert any("종합 점수" in m.label for m in at.metric)
    for period in ["6개월", "5년"]:
        at.sidebar.selectbox[0].set_value(period).run()
        assert not at.exception, at.exception
    at.sidebar.selectbox[1].set_value("10년").run()  # 시장 위치 비교 기간
    assert not at.exception, at.exception


# ---------------------------------------------------------------------------
# 시장 위치
# ---------------------------------------------------------------------------
import position as pos  # noqa: E402


def test_rolling_percentile():
    s = pd.Series(np.arange(10, dtype=float))
    p = pos.rolling_percentile(s, window=5, min_periods=5)
    assert p.iloc[:4].isna().all()
    assert (p.iloc[4:] == 100).all()  # 계속 오르면 항상 최고
    p = pos.rolling_percentile(-s, window=5, min_periods=5)
    assert (p.iloc[4:] == 0).all()
    flat = pos.rolling_percentile(pd.Series([1.0] * 6), window=5, min_periods=5)
    assert flat.iloc[-1] == 50  # 동률은 중간


def test_rsi_bounds():
    close = pd.Series(np.r_[np.linspace(100, 200, 40), np.linspace(200, 120, 40)])
    r = pos.rsi(close).dropna()
    assert r.between(0, 100).all()
    assert r.iloc[25] == pytest.approx(100)  # 상승만 있으면 100
    assert r.iloc[-1] < 10


def test_position_on_demo():
    df = add_indicators(make_demo_data("2018-01-01", "2024-12-31"))
    full, scores, snap, summ, levels = pos.build_position(df, lookback_years=5)
    assert set(snap["key"]) == {i.key for i in pos.INDICATORS}
    heat_cols = [c for c in scores if c.startswith("heat_")] + ["overall", *pos.CATEGORIES]
    vals = scores[heat_cols].stack()
    assert vals.between(0, 100).all()
    # 역방향 지표는 온도 = 100 − 백분위
    row = snap.set_index("key").loc["div_yield"]
    assert row["온도"] == pytest.approx(100 - row["백분위"])
    assert set(summ) == {"overall", *pos.CATEGORIES}
    assert summ["overall"]["label"] == pos.zone_label(summ["overall"]["score"])
    assert levels.iloc[0]["구분"] == "현재 KOSPI"
    # 일드갭 = 1/PER − 10년물
    last = full.dropna(subset=["yield_gap"]).iloc[-1]
    assert last["yield_gap"] == pytest.approx(100 / last["per"] - last["bond_10y"])


def test_position_without_valuation():
    demo = make_demo_data("2019-01-01", "2024-06-30")[["kospi_close", "retail_net", "deposit", "credit"]]
    _, scores, snap, summ, _ = pos.build_position(add_indicators(demo), lookback_years=3)
    assert "밸류에이션" not in summ and "추세" in summ
    assert "per" not in set(snap["key"])
    assert scores["overall"].dropna().between(0, 100).all()


def test_zone_label():
    assert pos.zone_label(5) == "침체·저평가"
    assert pos.zone_label(50) == "중립"
    assert pos.zone_label(100) == "과열·고평가"
    assert pos.zone_label(float("nan")) == "-"


def test_position_charts_build():
    from charts import build_position_bars, build_position_history

    df = add_indicators(make_demo_data("2020-01-01", "2024-12-31"))
    full, scores, snap, _, _ = pos.build_position(df, 3)
    assert len(build_position_bars(snap).data) == 1
    fig = build_position_history(full, scores, pos.CATEGORIES)
    assert len(fig.data) == 1 + len(pos.CATEGORIES) + 1


def test_cli_demo(tmp_path, capsys):
    import kospi_position

    out = tmp_path / "pos.csv"
    hist = tmp_path / "hist.csv"
    md = tmp_path / "report.md"
    code = kospi_position.main(["--demo", "--end", "2024-12-31", "--csv", str(out), "--history-csv", str(hist),
                                "--markdown", str(md)])
    assert code == 0
    text = capsys.readouterr().out
    assert "종합 점수" in text and "밸류에이션" in text and "주요 가격 레벨" in text
    table = pd.read_csv(out)
    assert {"지표", "현재값", "백분위", "온도", "해석"} <= set(table.columns)
    assert "overall" in pd.read_csv(hist).columns
    report = md.read_text(encoding="utf-8")
    assert "| **종합** |" in report and "### 지표별" in report
