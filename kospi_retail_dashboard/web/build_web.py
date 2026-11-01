"""실제 KOSPI 데이터를 넣은 웹 시뮬레이터(HTML 파일 하나)를 만든다.

    python kospi_retail_dashboard/web/build_web.py --years 5 --out kospi_simulator.html

- 가격: KRX(로그인 시) → 네이버 순서로 받는다.
- KRX_ID / KRX_PW 환경변수가 있으면 PER·PBR·배당수익률·시가총액 이력도 넣는다.
- 만든 파일은 브라우저로 바로 열면 된다 (차트 라이브러리만 인터넷에서 받음).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from data_sources import load_kospi_valuation, set_krx_credentials  # noqa: E402

TEMPLATE = HERE / "simulator_page.html"
PLACEHOLDER = '<script type="application/json" id="embedded-data">null</script>'
SKELETON_HEAD = (
    '<!doctype html><html lang="ko"><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
    "<style>:root{color-scheme:light}body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>"
    "</head><body>"
)


def _num(v, digits: int):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), digits)


def to_payload(df: pd.DataFrame, source: str) -> dict:
    """load_kospi_valuation 결과 → 페이지가 읽는 형식 [날짜, 시, 고, 저, 종, PER, PBR, 배당, 시총(조원)]."""
    rows = []
    for d, r in df.iterrows():
        mcap = r.get("mcap")
        rows.append([
            pd.Timestamp(d).strftime("%Y-%m-%d"),
            _num(r.get("open"), 2), _num(r.get("high"), 2), _num(r.get("low"), 2), _num(r["close"], 2),
            _num(r.get("per"), 2), _num(r.get("pbr"), 3), _num(r.get("div_yield"), 3),
            _num(mcap / 1e12 if mcap is not None else None, 1),
        ])
    return {"source": source, "rows": rows}


def build_html(df: pd.DataFrame, source: str, template: str | None = None) -> str:
    page = template if template is not None else TEMPLATE.read_text(encoding="utf-8")
    if PLACEHOLDER not in page:
        raise ValueError("템플릿에서 데이터 자리를 찾지 못했습니다")
    payload = json.dumps(to_payload(df, source), ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    page = page.replace(PLACEHOLDER, f'<script type="application/json" id="embedded-data">{payload}</script>')
    return SKELETON_HEAD + page + "</body></html>"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", type=int, default=5, help="받을 기간(년). 지표 계산용으로 1년 남짓 더 받습니다")
    ap.add_argument("--out", default="kospi_simulator.html")
    args = ap.parse_args()

    use_krx = set_krx_credentials(None, None)
    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=round(365.25 * args.years) + 420)
    df, report = load_kospi_valuation(start, end, use_krx)
    for name, ok, msg in report.items:
        print(("OK  " if ok else "--  ") + f"{name}: {msg}")
    if df.empty:
        sys.exit("KOSPI 가격을 받지 못했습니다.")
    source = "KRX" if use_krx and "per" in df else "네이버 금융"
    Path(args.out).write_text(build_html(df, f"{source} 일별 시세"), encoding="utf-8")
    print(f"{args.out} 저장: {df.index[0]:%Y-%m-%d} ~ {df.index[-1]:%Y-%m-%d}, {len(df):,}거래일")


if __name__ == "__main__":
    main()
