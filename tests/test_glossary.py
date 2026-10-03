import numpy as np
import pandas as pd

from backtester.engine import compute_metrics
from backtester.formatting import duration, money, pct
from backtester.glossary import TERMS, explain, sharpe_grade


def test_money_reads_like_korean():
    assert money(10_000_000, "KRW") == "1,000만 원"
    assert money(123_456_789, "KRW") == "1억 2,346만 원"
    assert money(200_000_000, "KRW") == "2억 원"
    assert money(9_800, "KRW") == "9,800원"
    assert money(-5_000_000, "KRW") == "-500만 원"
    assert money(12_345.6, "USD") == "$12,346"


def test_duration_and_pct():
    assert duration(10) == "10일"
    assert duration(150) == "5개월"
    assert duration(365) == "1년"
    assert duration(550) == "1년 6개월"
    assert pct(0.123) == "12.3%"
    assert pct(0.05, signed=True) == "+5.0%"
    assert pct(None) == "-"


def test_sharpe_grade():
    assert "못했" in sharpe_grade(-0.1)
    assert "아쉬운" in sharpe_grade(0.3)
    assert "괜찮은" in sharpe_grade(0.7)
    assert "훌륭한" in sharpe_grade(1.2)


def test_every_term_has_an_example():
    for term in TERMS:
        assert term.summary and term.example


def _values(seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2012-01-01", "2020-12-31")
    return pd.Series(1e7 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, len(dates)))), index=dates)


def test_explain_uses_actual_numbers():
    m = compute_metrics(_values(1), 0.03)
    bm = compute_metrics(_values(2), 0.03)
    items = dict(explain(m, "KRW", 0.03, bm, "KODEX 200 (069500)"))
    assert money(m.final, "KRW") in items["💰 얼마가 됐나요?"]
    assert "KODEX 200 (069500)" in items["💰 얼마가 됐나요?"]
    assert pct(m.max_drawdown) in items["📉 최대 낙폭 (MDD)"]
    assert f"{m.sharpe:.2f}" in items["⚖️ 샤프 비율"]


def test_explain_without_benchmark():
    m = compute_metrics(_values(3), 0.0)
    titles = [title for title, _ in explain(m, "USD", 0.0)]
    assert "📈 연평균 수익률 (CAGR)" in titles


def test_bold_markers_render():
    """'**4.7%**씩' 처럼 기호 바로 뒤에 한글이 붙으면 굵은 글씨가 안 되고 ** 가 그대로 보인다."""
    from markdown_it import MarkdownIt

    from backtester.glossary import CAVEATS, HELP

    texts = list(HELP.values()) + CAVEATS
    for term in TERMS:
        texts += [x for x in (term.summary, term.analogy, term.example, term.note) if x]
    m, bm = compute_metrics(_values(1), 0.03, 1e7), compute_metrics(_values(2), 0.03, 1e7)
    for currency in ("KRW", "USD"):
        texts += [body for _, body in explain(m, currency, 0.03, bm, "SPDR S&P 500 (SPY)")]
    md = MarkdownIt()
    broken = [t for t in texts if "**" in md.render(t)]
    assert not broken, broken


def test_explain_reports_amount_invested_not_after_fees():
    values = _values(4) * 0.999  # 처음 살 때 0.1% 비용이 빠진 상태
    m = compute_metrics(values, 0.03, invested=1e7)
    first = explain(m, "KRW", 0.03)[0][1]
    assert "1,000만 원을 넣고" in first
