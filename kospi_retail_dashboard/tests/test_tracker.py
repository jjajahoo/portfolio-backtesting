import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TRACKER = Path(__file__).resolve().parents[1] / "tracker"
sys.path.insert(0, str(TRACKER))

import build  # noqa: E402
import collect  # noqa: E402
import data_sources as ds  # noqa: E402


def _frame(dates, **cols):
    return pd.DataFrame(cols, index=pd.DatetimeIndex(pd.to_datetime(dates), name="date"))


def test_parse_ecos_rows_and_errors():
    payload = {"StatisticSearch": {"list_total_count": 2, "row": [
        {"TIME": "20261030", "DATA_VALUE": "3.105"}, {"TIME": "20261029", "DATA_VALUE": "3.12"}]}}
    s = collect.parse_ecos(payload)
    assert list(s.index) == list(pd.to_datetime(["2026-10-29", "2026-10-30"]))
    assert s.iloc[-1] == pytest.approx(3.105)
    with pytest.raises(ds.SourceError, match="INFO-200"):
        collect.parse_ecos({"RESULT": {"CODE": "INFO-200", "MESSAGE": "해당하는 데이터가 없습니다."}})


def test_collect_combines_sources(monkeypatch):
    d = ["2026-10-28", "2026-10-29", "2026-10-30"]
    monkeypatch.setattr(ds, "fetch_kospi_krx", lambda s, e: _frame(d, kospi_close=[1.0, 2.0, 3.0], kospi_mcap=[3.0e15, 3.1e15, 3.2e15]))
    monkeypatch.setattr(ds, "fetch_kospi_fundamental_krx", lambda s, e: _frame(d, per=[10.0, 11.0, np.nan], pbr=[1.0, 1.1, 1.2], div_yield=[2.0, 2.0, 2.0]))
    monkeypatch.setattr(collect, "fetch_ktb10y_ecos", lambda s, e, k: _frame(d[:2], ktb10y=[3.0, 3.1]))
    monkeypatch.setenv("ECOS_API_KEY", "x")
    report = ds.LoadReport()
    df = collect.collect("2026-10-28", "2026-10-30", report)
    assert list(df.columns) == collect.COLUMNS
    assert df["mcap"].iloc[-1] == pytest.approx(3200)  # 조원
    assert df["ktb10y"].iloc[-1] == pytest.approx(3.1)  # 휴장일은 직전 값
    assert all(ok for _, ok, _ in report.items)


def test_collect_falls_back_to_naver_and_skips_bond(monkeypatch):
    d = ["2026-10-29", "2026-10-30"]

    def boom(*a, **k):
        raise ConnectionError("down")

    monkeypatch.setattr(ds, "fetch_kospi_krx", boom)
    monkeypatch.setattr(ds, "fetch_kospi_fundamental_krx", boom)
    monkeypatch.setattr(ds, "fetch_kospi_naver", lambda s, e: _frame(d, kospi_close=[1.0, 2.0]))
    monkeypatch.delenv("ECOS_API_KEY", raising=False)
    report = ds.LoadReport()
    df = collect.collect("2026-10-29", "2026-10-30", report)
    assert df["close"].tolist() == [1.0, 2.0] and df["per"].isna().all()
    status = {n: ok for n, ok, _ in report.items}
    assert status == {"KOSPI 지수·시가총액": True, "PER·PBR·배당수익률": False, "국고채 10년 금리": False}


def test_merge_keeps_old_values_when_new_is_missing():
    old = _frame(["2026-10-29", "2026-10-30"], close=[1.0, 2.0], per=[10.0, 11.0])
    new = _frame(["2026-10-30", "2026-11-02"], close=[2.5, 3.0], per=[np.nan, 12.0])
    out = collect.merge(old.reindex(columns=collect.COLUMNS), new.reindex(columns=collect.COLUMNS))
    assert out["close"].tolist() == [1.0, 2.5, 3.0]
    assert out["per"].tolist() == [10.0, 11.0, 12.0]


def test_save_load_and_build(tmp_path):
    df = _frame(["2026-10-29", "2026-10-30"], close=[4000.123, 4010.0], mcap=[3300.0, 3310.0], per=[17.6, 17.7],
                pbr=[1.28, 1.29], div_yield=[1.7, 1.69], ktb10y=[3.1, np.nan]).reindex(columns=collect.COLUMNS)
    report = ds.LoadReport()
    report.ok("KOSPI 지수·시가총액", "KRX (2행)")
    csv, meta = tmp_path / "v.csv", tmp_path / "m.json"
    collect.save(df, report, csv, meta)
    back = collect.load_csv(csv)
    assert back["close"].iloc[0] == pytest.approx(4000.12)
    m = json.loads(meta.read_text())
    assert m["last"] == "2026-10-30" and m["sources"][0]["ok"] is True

    html = build.build(back, m)
    assert html.startswith("<title>KOSPI 가치 추적기</title>")
    data = json.loads(re.search(r'id="tracker-data">(.*?)</script>', html, re.S).group(1))
    assert data["rows"][1] == ["2026-10-30", 4010.0, 3310.0, 17.7, 1.29, 1.69, None]
    assert build.build(back, m, standalone=True).startswith("<!doctype html>")


@pytest.mark.skipif(shutil.which("node") is None, reason="node 가 없음")
def test_page_math_with_node(tmp_path):
    page = (TRACKER / "tracker_page.html").read_text(encoding="utf-8")
    core = re.search(r'<script id="tracker-core">(.*?)</script>', page, re.S).group(1)
    (tmp_path / "core.js").write_text(core, encoding="utf-8")
    script = """
    const T = require('./core.js');
    const rows = [];
    for (let i = 0; i < 100; i++) {
      const d = new Date(Date.UTC(2016, 0, 4 + i * 30)).toISOString().slice(0, 10);
      rows.push({ d, c: 1000 + i, per: 10 + i * 0.1, pbr: 1 + i * 0.01, dy: 2.5 - i * 0.01, ktb: 3, mcap: 2000 });
    }
    const st = T.stats(T.series(rows, 'pbr'), 1.99);
    const comp = T.composite(rows, 'all');
    const fair = T.fairLevels(rows, 'all');
    console.log(JSON.stringify({ pct: st.pct, q50: st.q50, score: comp.score, parts: comp.parts.length,
      fairMid: fair.out.pbr[2].level, base: fair.base.pbr, start5: T.windowStart(rows, '5'), verdict: T.verdict(comp.score)[0] }));
    """
    (tmp_path / "t.js").write_text(script, encoding="utf-8")
    out = json.loads(subprocess.run(["node", "t.js"], cwd=tmp_path, capture_output=True, text=True, check=True).stdout)
    assert out["pct"] == pytest.approx(99.0)          # 마지막 값은 100개 중 99개보다 큼
    assert out["q50"] == pytest.approx(1.495)
    assert out["score"] == pytest.approx(99.5)        # PER·PBR 99, 배당·일드갭은 최저라 100 → 평균 99.5
    assert out["parts"] == 4 and out["verdict"] == "매우 비싼 편"
    assert out["base"] == pytest.approx(1099 / 1.99)
    assert out["fairMid"] == pytest.approx(1099 / 1.99 * 1.495)
    assert 0 < out["start5"] < 100
