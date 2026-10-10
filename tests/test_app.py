"""앱 전체를 가짜 데이터로 실행해 보는 스모크 테스트."""
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from qqq_score import data
from tests.synthetic import make_market_data

MD = make_market_data()
APP = Path(__file__).resolve().parent.parent / "streamlit_app.py"


@pytest.fixture
def app(monkeypatch):
    def fake_prices(ticker):
        return MD.prices if ticker == data.PRICE_TICKER else MD.trend_prices

    def fake_fred(series_id):
        return MD.fred[series_id]

    def fake_quote(ticker):
        return {"price": 123.45, "time": pd.Timestamp("2026-10-09 15:59", tz="America/New_York"),
                "prev_close": 120.0, "change_pct": 2.875}

    monkeypatch.setattr(data, "load_prices", fake_prices)
    monkeypatch.setattr(data, "load_fred", fake_fred)
    monkeypatch.setattr(data, "load_latest_quote", fake_quote)
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    yield at
    import streamlit as st
    st.cache_data.clear()


def test_app_runs_without_errors(app):
    assert not app.exception, app.exception
    assert app.title[0].value == "QQQ 점수 실험실"
    labels = [m.label for m in app.metric]
    assert "QQQ 현재가 (약 15분 지연)" in labels
    assert "현재 조건 충족" in labels


def test_change_condition_and_add_row(app):
    app.number_input(key="cond.0.v1").set_value(2.0).run()
    assert not app.exception, app.exception
    add = [b for b in app.sidebar.button if b.label == "＋ 조건 추가"][0]
    add.click().run()
    assert not app.exception, app.exception
    assert app.session_state["cond_ids"] == [0, 1]
    app.radio(key="cond_how").set_value("or").run()
    assert not app.exception, app.exception
    # '사이' 조건: 값 2 입력칸이 나타나고 통계가 계산되어야 한다
    app.selectbox(key="cond.0.op").set_value("between").run()
    app.number_input(key="cond.0.v2").set_value(5.0).run()
    assert not app.exception, app.exception
    assert any("이상 \\~" in m.value for m in app.markdown)


def test_toggle_indicator_and_bollinger_params(app):
    app.number_input(key="ind.bb.p.k").set_value(2.5).run()
    assert not app.exception, app.exception
    app.toggle(key="ind.macd.enabled").set_value(True).run()
    assert not app.exception, app.exception
    app.toggle(key="ind.fed_rate.enabled").set_value(False).run()
    # 조건이 쓰던 지표를 끄면 조건은 빠지고 앱은 계속 동작해야 한다
    assert not app.exception, app.exception
    app.toggle(key="split_on").set_value(False).run()
    assert not app.exception, app.exception
