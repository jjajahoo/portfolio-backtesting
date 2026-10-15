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
    df, report = ds.load_market_data("2024-01-01", "2024-01-10", use_krx=False)
    status = {name: ok for name, ok, _ in report.items}
    assert status == {"KOSPI 지수": True, "개인 수급": False, "예탁금·신용융자": True}
    assert {"kospi_close", "deposit"} <= set(df.columns)


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
    assert len(at.metric) >= 5
    for period in ["6개월", "5년"]:
        at.sidebar.selectbox[0].set_value(period).run()
        assert not at.exception, at.exception
