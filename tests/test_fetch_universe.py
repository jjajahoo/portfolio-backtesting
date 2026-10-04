"""웹 버전 시세 파일 만들기 (scripts/fetch_universe.py) 확인. 인터넷 없이 가짜 시세로."""

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fetch_universe  # noqa: E402
from conftest import fake_close  # noqa: E402


def test_encode_keeps_precision_without_drift():
    rng = np.random.default_rng(0)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 7000)))
    back = fetch_universe.decode(fetch_universe.encode(prices))
    assert np.max(np.abs(back / prices - 1)) < 5.1e-5  # 0.01% 단위로 반올림, 오차가 쌓이지 않는다


def sample(start="2014-01-01", end="2016-12-31"):
    items = [
        {"code": "SPY", "name": "SPDR S&P 500", "kind": "미국 ETF", "desc": "", "alias": "", "symbol": "SPY"},
        {"code": "LATE", "name": "Late Inc", "kind": "미국 주식", "desc": "", "alias": "", "symbol": "LATE"},
        {"code": "069500", "name": "KODEX 200", "kind": "한국 ETF", "desc": "", "alias": "", "symbol": "069500.KS"},
        {"code": "NOPE", "name": "없음", "kind": "미국 주식", "desc": "", "alias": "", "symbol": "NOPE"},
    ]
    closes = {s: fake_close(s, start, end) for s in ("SPY", "LATE", "069500.KS")}
    closes["069500.KS"] = closes["069500.KS"].drop(pd.Timestamp("2015-03-02"))  # 한국만 쉬는 날
    return items, closes, fake_close("KRW=X", start, end)


def test_build_index_and_chunks(monkeypatch):
    monkeypatch.setattr(fetch_universe, "CHUNK", 2)
    items, closes, fx = sample()
    index, chunks = fetch_universe.build(items, closes, fx)
    codes = [row[0] for row in index["tickers"]]
    assert codes == ["SPY", "LATE", "069500"]  # 시세가 없는 종목은 빠진다
    assert [row[6] for row in index["tickers"]] == [0, 0, 1]
    assert [row[5] for row in index["tickers"]] == ["USD", "USD", "KRW"]

    dates = pd.Timestamp(index["base"]) + pd.to_timedelta(np.cumsum(index["days"]), unit="D")
    kr = chunks[1]["069500"]
    values = fetch_universe.decode(kr["d"])
    assert len(values) == len(dates) - kr["s"]
    day = list(dates).index(pd.Timestamp("2015-03-02"))
    assert values[day - kr["s"]] == pytest.approx(closes["069500.KS"][:"2015-02-27"].iloc[-1], rel=1e-4)  # 직전 값
    late = chunks[0]["LATE"]
    assert dates[late["s"]] == pd.Timestamp("2015-06-01")
