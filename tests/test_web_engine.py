"""웹 버전 계산(web/engine.js)이 파이썬 엔진과 같은 결과를 내는지 Node로 확인한다."""

import json
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from backtester.engine import compute_metrics, run_backtest
from backtester.glossary import explain
from backtester.market_data import load_prices
from conftest import fake_close

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "web" / "engine.js"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node가 없어 웹 엔진을 확인할 수 없음")

JS_BACKTEST = """
const E = require(process.argv[1]);
const p = JSON.parse(require("fs").readFileSync(0, "utf8"));
const prices = Object.fromEntries(Object.entries(p.prices).map(([k, v]) => [k, Float64Array.from(v)]));
const run = (w, reb) => E.runBacktest(prices, p.dates, w, reb, p.initial, p.fee);
const values = run(p.weights, p.rebalance);
const m = E.computeMetrics(values, p.dates, p.rf, p.initial);
const bm = E.computeMetrics(run({ [p.bench]: 1 }, "none"), p.dates, p.rf, p.initial);
console.log(JSON.stringify({
  values: Array.from(values),
  m: { cagr: m.cagr, total: m.totalReturn, vol: m.volatility, sharpe: m.sharpe, mdd: m.maxDrawdown },
  explain: E.explain(m, p.dates, p.currency, p.rf, bm, p.benchName).map((x) => x.body),
}));
"""

JS_BUILD = """
const E = require(process.argv[1]);
const p = JSON.parse(require("fs").readFileSync(0, "utf8"));
const dates = E.decodeDates(p.index.base, p.index.days);
const series = { "KRW=X": { start: p.index.fx.s, close: E.decodeLevels(p.index.fx.d), currency: "KRW" } };
for (const row of p.index.tickers) {
  const packed = p.chunks[row[6]][row[0]];
  series[row[0]] = { start: packed.s, close: E.decodeLevels(packed.d), currency: row[5] };
}
const r = E.buildPrices({ dates, series }, p.tickers, p.base, p.start, p.end);
console.log(JSON.stringify({ dates: r.dates, prices: Object.fromEntries(
  Object.entries(r.prices).map(([k, v]) => [k, Array.from(v)])), limitedBy: r.limitedBy }));
"""


def node(script: str, payload: dict) -> dict:
    out = subprocess.run(
        [NODE, "-e", script, str(ENGINE)], input=json.dumps(payload), capture_output=True, text=True, check=True
    )
    return json.loads(out.stdout)


@pytest.fixture(scope="module")
def market():
    return load_prices(["SPY", "IEF", "069500"], date(2012, 1, 1), date(2020, 12, 31), "KRW", fetch=fake_close)


@pytest.mark.parametrize("rebalance", ["none", "monthly", "quarterly", "yearly"])
def test_js_engine_matches_python(market, rebalance):
    prices = market.prices
    weights = {"SPY": 0.5, "IEF": 0.3, "069500": 0.2}
    initial, fee, rf = 10_000_000, 0.001, 0.03
    values = run_backtest(prices, weights, rebalance, initial, fee)
    m = compute_metrics(values, rf, initial)
    bm = compute_metrics(run_backtest(prices, {"SPY": 1.0}, "none", initial, fee), rf, initial)

    js = node(JS_BACKTEST, {
        "dates": [d.strftime("%Y-%m-%d") for d in prices.index],
        "prices": {t: prices[t].tolist() for t in prices},
        "weights": weights, "rebalance": rebalance, "initial": initial, "fee": fee, "rf": rf,
        "bench": "SPY", "benchName": "SPDR S&P 500 (SPY)", "currency": "KRW",
    })
    np.testing.assert_allclose(js["values"], values.to_numpy(), rtol=1e-10)
    assert js["m"]["cagr"] == pytest.approx(m.cagr, rel=1e-9)
    assert js["m"]["total"] == pytest.approx(m.total_return, rel=1e-9)
    assert js["m"]["vol"] == pytest.approx(m.volatility, rel=1e-9)
    assert js["m"]["sharpe"] == pytest.approx(m.sharpe, rel=1e-9)
    assert js["m"]["mdd"] == pytest.approx(m.max_drawdown, rel=1e-9)
    python_text = [body for _, body in explain(m, "KRW", rf, bm, "SPDR S&P 500 (SPY)")]
    assert js["explain"] == python_text


def test_js_reads_packed_prices(monkeypatch):
    """scripts/fetch_universe.py가 만든 파일을 웹 엔진이 같은 가격으로 풀고, 환율·시작일을 맞추는지."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import fetch_universe
    import pandas as pd

    monkeypatch.setattr(fetch_universe, "CHUNK", 2)
    items = [{"code": c, "name": c, "kind": k, "desc": "", "alias": "", "symbol": s}
             for c, k, s in [("SPY", "미국 ETF", "SPY"), ("LATE", "미국 주식", "LATE"), ("069500", "한국 ETF", "069500.KS")]]
    closes = {it["symbol"]: fake_close(it["symbol"], "2014-01-01", "2016-12-31") for it in items}
    index, chunks = fetch_universe.build(items, closes, fake_close("KRW=X", "2014-01-01", "2016-12-31"))
    js = node(JS_BUILD, {"index": index, "chunks": chunks, "tickers": ["SPY", "LATE", "069500"], "base": "KRW",
                         "start": "2014-01-01", "end": "2016-12-31"})

    dates = pd.Timestamp(index["base"]) + pd.to_timedelta(np.cumsum(index["days"]), unit="D")
    assert js["dates"][0] == "2015-06-01"  # LATE가 상장한 날부터
    assert js["limitedBy"]["label"] == "LATE"
    first = list(dates).index(pd.Timestamp("2015-06-01"))
    fx = fetch_universe.decode(index["fx"]["d"])
    def native(code):
        packed = chunks[[r[0] for r in index["tickers"]].index(code) // 2][code]
        return fetch_universe.decode(packed["d"]), packed["s"]
    for code, usd in (("SPY", True), ("069500", False)):
        values, start = native(code)
        expected = [values[i - start] * (fx[i - index["fx"]["s"]] if usd else 1) for i in range(first, len(dates))]
        np.testing.assert_allclose(js["prices"][code], expected, rtol=1e-9)


def test_built_page_is_up_to_date():
    """web/index.html은 scripts/build_web.py로 만든 파일이다. 템플릿·엔진·설명을 고친 뒤 다시 빌드해야 한다."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_web

    assert (ROOT / "web" / "index.html").read_text() == build_web.build(), "python scripts/build_web.py 를 실행하세요"
