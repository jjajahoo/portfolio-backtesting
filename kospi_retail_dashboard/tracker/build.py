"""tracker/data/kospi_value.csv → KOSPI 가치 추적기 페이지(HTML 한 파일).

    python kospi_retail_dashboard/tracker/build.py --out kospi_value_tracker.html
    python kospi_retail_dashboard/tracker/build.py --out kospi_value_tracker.html --standalone

--standalone 은 브라우저로 바로 여는 완전한 HTML 문서를 만든다.
기본 출력은 Claude 아티팩트로 올리는 본문 형식(<!doctype> 없이 <title> 부터)이다.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "tracker_page.html"
PLACEHOLDER = '<script type="application/json" id="tracker-data">null</script>'
SKELETON_HEAD = (
    '<!doctype html><html lang="ko"><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
    "<style>:root{color-scheme:light}body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>"
    "</head><body>"
)
FIELDS = [("close", 2), ("mcap", 1), ("per", 2), ("pbr", 3), ("div_yield", 3), ("ktb10y", 3)]


def _num(v, digits):
    if v is None:
        return None
    v = float(v)
    return None if math.isnan(v) else round(v, digits)


def to_payload(df: pd.DataFrame, meta: dict | None = None) -> dict:
    """[날짜, 종가, 시가총액(조원), PER, PBR, 배당수익률, 국고채10년] 배열."""
    rows = [
        [f"{pd.Timestamp(d):%Y-%m-%d}", *[_num(r.get(col), dg) for col, dg in FIELDS]]
        for d, r in df.iterrows()
        if pd.notna(r.get("close"))
    ]
    meta = meta or {}
    return {"updated_at": meta.get("updated_at"), "sources": meta.get("sources", []), "rows": rows}


def build(df: pd.DataFrame, meta: dict | None = None, standalone: bool = False, template: str | None = None) -> str:
    page = template if template is not None else TEMPLATE.read_text(encoding="utf-8")
    if PLACEHOLDER not in page:
        raise ValueError("템플릿에서 데이터 자리를 찾지 못했습니다")
    data = json.dumps(to_payload(df, meta), ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    page = page.replace(PLACEHOLDER, f'<script type="application/json" id="tracker-data">{data}</script>')
    return SKELETON_HEAD + page + "</body></html>" if standalone else page


def main() -> None:
    sys.path.insert(0, str(HERE))
    from collect import CSV_PATH, META_PATH, load_csv

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="kospi_value_tracker.html")
    ap.add_argument("--standalone", action="store_true", help="브라우저로 바로 여는 완전한 HTML")
    args = ap.parse_args()

    df = load_csv()
    if df is None or df.empty:
        sys.exit(f"{CSV_PATH} 가 없습니다. 먼저 collect.py 를 실행하세요.")
    meta = json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else {}
    Path(args.out).write_text(build(df, meta, args.standalone), encoding="utf-8")
    print(f"{args.out} 저장 ({df.index.min():%Y-%m-%d} ~ {df.index.max():%Y-%m-%d}, {len(df):,}행)")


if __name__ == "__main__":
    main()
