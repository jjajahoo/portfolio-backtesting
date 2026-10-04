"""web/template.html + web/engine.js + 종목·용어 설명(파이썬 모듈)을 합쳐 web/index.html을 만든다.

Claude 앱에서 바로 여는 웹 버전이다. 시세는 data/prices.json을 'prices.json'이라는 이름으로 함께 싣는다.
실행: python scripts/build_web.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtester.catalog import PRESETS, TICKERS  # noqa: E402
from backtester.glossary import CAVEATS, HELP, TERMS  # noqa: E402

WEB = ROOT / "web"


def content() -> dict:
    return {
        "tickers": [{"code": c, "name": n, "desc": d, "kind": k} for c, (n, d, k) in TICKERS.items()],
        "presets": [asdict(p) for p in PRESETS],
        "help": HELP,
        "terms": [asdict(t) for t in TERMS],
        "caveats": CAVEATS,
    }


def build() -> str:
    page = (WEB / "template.html").read_text()
    engine = (WEB / "engine.js").read_text()
    data = json.dumps(content(), ensure_ascii=False).replace("</", "<\\/")
    assert "<!--ENGINE-->" in page and "/*__CONTENT__*/null" in page
    page = page.replace("<!--ENGINE-->", f"<script>\n{engine}</script>")
    return page.replace("/*__CONTENT__*/null", data)


if __name__ == "__main__":
    out = WEB / "index.html"
    out.write_text(build())
    print(f"{out.relative_to(ROOT)} ({out.stat().st_size / 1000:.0f}KB)")
