"""catalog에 있는 모든 종목의 시세를 Yahoo Finance에서 받아 data/prices.json으로 저장한다.

Claude 앱에서 바로 여는 웹 버전(web/)은 인터넷에서 시세를 직접 받을 수 없어서 이 파일을 함께 싣는다.
GitHub Actions(.github/workflows/update-prices.yml)에서 실행된다.

저장 형식:
  dates: 모든 종목의 거래일을 합친 날짜 목록
  series[코드]: symbol(Yahoo 심볼), currency, name(Yahoo 이름, 확인용),
                start(dates 안의 첫 위치), close(start부터의 수정 종가, 휴장일은 직전 값)
  KRW=X 는 1달러당 원화 환율.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtester.catalog import TICKERS  # noqa: E402
from backtester.market_data import FX_TICKER, candidates, currency_of, fetch_close  # noqa: E402

START = date(2000, 1, 1)
OUT = ROOT / "data" / "prices.json"


def fetch_with_retry(symbol: str, fetch=fetch_close) -> pd.Series:
    for attempt in range(3):
        try:
            close = fetch(symbol, START, date.today())
            if not close.empty:
                return close
        except Exception as e:  # Yahoo가 가끔 요청을 거절한다
            print(f"  {symbol}: {e}")
        time.sleep(2 * (attempt + 1))
    return pd.Series(dtype=float)


def yahoo_name(symbol: str) -> str:
    import yfinance as yf

    try:
        info = yf.Ticker(symbol).info
        return info.get("shortName") or info.get("longName") or ""
    except Exception:
        return ""


def build(fetch=fetch_close, name_of=yahoo_name) -> dict:
    series: dict[str, pd.Series] = {}
    meta: dict[str, dict] = {}
    missing: list[str] = []
    for code in [*TICKERS, FX_TICKER]:
        for symbol in candidates(code) if code != FX_TICKER else [FX_TICKER]:
            close = fetch_with_retry(symbol, fetch)
            if not close.empty:
                series[code] = close
                meta[code] = {
                    "symbol": symbol,
                    "currency": "KRW" if code == FX_TICKER else currency_of(symbol),
                    "name": name_of(symbol),
                }
                break
        else:
            missing.append(code)

    frame = pd.DataFrame(series).sort_index()
    out = {}
    for code in frame:
        s = frame[code]
        first = frame.index.get_loc(s.first_valid_index())
        values = s.iloc[first:].ffill()
        out[code] = {**meta[code], "start": int(first), "close": [float(f"{v:.6g}") for v in values]}
    return {
        "updated": date.today().isoformat(),
        "dates": [d.strftime("%Y-%m-%d") for d in frame.index],
        "series": out,
        "missing": missing,
    }


def main() -> None:
    data = build()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    dates = data["dates"]
    print(f"{len(data['series'])}개 종목, {dates[0]} ~ {dates[-1]}, {OUT.stat().st_size / 1e6:.1f}MB")
    for code, s in data["series"].items():
        last = s["close"][-1]
        print(f"{code:8} {s['symbol']:10} {dates[s['start']]} {len(s['close']):6}일 {last:>12,.2f}  {s['name']}")
    if data["missing"]:
        print("찾지 못한 종목:", ", ".join(data["missing"]))


if __name__ == "__main__":
    main()
