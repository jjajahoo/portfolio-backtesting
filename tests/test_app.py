from pathlib import Path

import pytest
import streamlit as st
from markdown_it import MarkdownIt
from streamlit.testing.v1 import AppTest

from backtester import market_data
from backtester.catalog import PRESETS
from conftest import fake_close

APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")


@pytest.fixture(autouse=True)
def offline_prices(monkeypatch):
    monkeypatch.setattr(market_data, "fetch_close", fake_close)
    st.cache_data.clear()


def run_app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def assert_bold_renders(at: AppTest) -> None:
    """화면에 '**' 가 그대로 보이는 곳이 없어야 한다."""
    md = MarkdownIt()
    texts = [el.value for kind in (at.markdown, at.caption, at.info, at.warning, at.error) for el in kind]
    broken = [t for t in texts if "**" in md.render(t)]
    assert not broken, broken


def test_default_preset_shows_results_and_explanations():
    at = run_app()
    labels = [m.label for m in at.metric]
    assert "샤프 비율" in labels and "최대 낙폭 (MDD)" in labels
    assert not at.error
    markdown = " ".join(md.value for md in at.markdown)
    assert "얼마가 됐나요?" in markdown and "샤프 비율" in markdown


@pytest.mark.parametrize("preset", [p.name for p in PRESETS])
def test_every_preset_runs(preset):
    at = run_app()
    at.selectbox[0].select(preset).run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    assert_bold_renders(at)


def test_usd_base_and_no_rebalance():
    at = run_app()
    at.radio[0].set_value("USD").run()
    at.selectbox[1].set_value("none").run()
    assert not at.exception, at.exception
    assert any(m.value.startswith("$") for m in at.metric)


def test_unknown_benchmark_warns_but_still_runs():
    at = run_app()
    at.text_input[0].set_value("NOPE").run()
    assert not at.exception
    assert any("NOPE" in w.value for w in at.warning)
    assert at.metric


def test_start_after_end_is_an_error():
    at = run_app()
    at.date_input[0].set_value(at.date_input[1].value).run()
    assert any("시작일" in e.value for e in at.error)
