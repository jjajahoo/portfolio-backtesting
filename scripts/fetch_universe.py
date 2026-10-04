"""웹 버전(web/)에 싣는 종목 목록과 시세를 만든다. GitHub Actions에서 실행된다.

종목: catalog의 TICKERS·WEB_EXTRA + S&P500 전 종목 + 코스피·코스닥 시가총액 상위 + 한국 ETF 시가총액 상위.
웹 버전은 인터넷에서 시세를 직접 받을 수 없어서, 여기서 받은 시세를 페이지와 함께 싣는다.

출력 (data/web/):
  index.json  종목 목록, 날짜, 환율
  c/<n>.json  종목 몇 개씩 묶은 시세. 페이지는 고른 종목이 든 묶음만 받는다.

시세 저장 방식: 휴장일은 직전 값으로 채운 수정 종가를 log(가격)×10000 정수로 바꾸고,
첫 값과 그다음부터의 차이만 적는다 (0.01% 단위, 오차가 쌓이지 않는다).
"""

from __future__ import annotations

import io
import json
import re
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtester.catalog import KO_ALIASES, TICKERS, WEB_EXTRA  # noqa: E402
from backtester.market_data import FX_TICKER  # noqa: E402

START = date(2000, 1, 1)
OUT = ROOT / "data" / "web"
CHUNK = 7  # 묶음 하나에 담는 종목 수
SCALE = 10000
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
SP500_CSV = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
SECTORS = {
    "Information Technology": "정보기술", "Health Care": "헬스케어", "Financials": "금융",
    "Consumer Discretionary": "경기소비재", "Communication Services": "커뮤니케이션", "Industrials": "산업재",
    "Consumer Staples": "필수소비재", "Energy": "에너지", "Utilities": "유틸리티", "Real Estate": "부동산",
    "Materials": "소재",
}


# ---------------------------------------------------------------- 시세 압축
def encode(values: np.ndarray) -> list[int]:
    levels = np.rint(np.log(values) * SCALE).astype(np.int64)
    return [int(levels[0])] + np.diff(levels).astype(int).tolist()


def decode(encoded: list[int]) -> np.ndarray:
    return np.exp(np.cumsum(encoded) / SCALE)


# ---------------------------------------------------------------- 종목 목록
def get(url: str):
    import requests

    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    return r


def sp500() -> list[dict]:
    df = pd.read_csv(io.StringIO(get(SP500_CSV).text))
    return [
        {"code": str(symbol).replace(".", "-"), "name": str(name), "kind": "미국 주식",
         "desc": "S&P500 · " + SECTORS.get(str(sector), str(sector))}
        for symbol, name, sector in df[["Symbol", "Security", "GICS Sector"]].itertuples(index=False)
    ]


def naver_market_cap(sosok: int, count: int) -> list[tuple[str, str]]:
    """네이버 금융 시가총액 순위. sosok 0=코스피, 1=코스닥."""
    out: list[tuple[str, str]] = []
    page = 1
    while len(out) < count and page <= 20:
        r = get(f"https://finance.naver.com/sise/sise_market_sum.naver?sosok={sosok}&page={page}")
        html = r.content.decode("euc-kr", errors="replace")
        found = re.findall(r'href="/item/main\.naver\?code=(\w{6})" class="tltle">([^<]+)</a>', html)
        if not found:
            break
        out += [(c, n.strip()) for c, n in found]
        page += 1
    return out[:count]


def fdr_market_cap(market: str, count: int) -> list[tuple[str, str]]:
    import FinanceDataReader as fdr

    df = fdr.StockListing(market).sort_values("Marcap", ascending=False)
    return list(zip(df["Code"].astype(str), df["Name"].astype(str)))[:count]


def korean_stocks(market: str, count: int) -> list[tuple[str, str]]:
    for source in (lambda: naver_market_cap(0 if market == "KOSPI" else 1, count), lambda: fdr_market_cap(market, count)):
        try:
            found = source()
            if len(found) >= count * 0.8:
                return found
            print(f"  {market}: {len(found)}개만 찾음, 다른 곳에서 다시 찾는다")
        except Exception as e:
            print(f"  {market} 목록 실패: {e}")
    return []


def korean_etfs(count: int) -> list[tuple[str, str]]:
    try:
        r = get("https://finance.naver.com/api/sise/etfItemList.nhn?etfType=0&targetColumn=market_sum&sortOrder=desc")
        try:
            data = r.json()
        except ValueError:
            data = json.loads(r.content.decode("euc-kr"))
        items = sorted(data["result"]["etfItemList"], key=lambda x: -(x.get("marketSum") or 0))
        return [(str(x["itemcode"]), str(x["itemname"])) for x in items][:count]
    except Exception as e:
        print(f"  ETF 목록(네이버) 실패: {e}")
    try:
        import FinanceDataReader as fdr

        df = fdr.StockListing("ETF/KR").sort_values("MarCap", ascending=False)
        return list(zip(df["Symbol"].astype(str), df["Name"].astype(str)))[:count]
    except Exception as e:
        print(f"  ETF 목록(FDR) 실패: {e}")
    return []


def universe() -> list[dict]:
    """인기 순서대로. 같은 코드가 또 나오면 앞의 것을 쓴다."""
    items = [{"code": c, "name": n, "desc": d, "kind": k} for c, (n, d, k) in {**TICKERS, **WEB_EXTRA}.items()]
    items += sp500()
    for market, count in (("KOSPI", 300), ("KOSDAQ", 150)):
        label = "코스피" if market == "KOSPI" else "코스닥"
        for rank, (code, name) in enumerate(korean_stocks(market, count), 1):
            items.append({"code": code, "name": name, "kind": "한국 주식", "desc": f"{label} 시가총액 {rank}위",
                          "symbol": code + (".KS" if market == "KOSPI" else ".KQ")})
    etf_codes = set()
    for code, name in korean_etfs(300):
        etf_codes.add(code)
        items.append({"code": code, "name": name, "kind": "한국 ETF", "desc": ""})

    seen, out = set(), []
    for it in items:
        if it["code"] in seen:
            continue
        seen.add(it["code"])
        if it["kind"] == "한국 주식" and it["code"] in etf_codes:
            it["kind"] = "한국 ETF"
        it.setdefault("symbol", it["code"] + ".KS" if re.match(r"^\d[0-9A-Z]{5}$", it["code"]) else it["code"])
        it["alias"] = KO_ALIASES.get(it["code"], "")
        out.append(it)
    return out


# ---------------------------------------------------------------- 시세 받기
def download(symbols: list[str]) -> dict[str, pd.Series]:
    import yfinance as yf

    closes: dict[str, pd.Series] = {}
    end = date.today() + timedelta(days=1)
    for i in range(0, len(symbols), 40):
        batch = symbols[i : i + 40]
        for attempt in range(3):
            try:
                df = yf.download(batch, start=START, end=end, auto_adjust=True, progress=False, threads=True,
                                 group_by="column", multi_level_index=True)
                close = df["Close"]
                break
            except Exception as e:
                print(f"  묶음 {i // 40} 실패 ({e}), 다시 시도")
                close = None
                time.sleep(5 * (attempt + 1))
        for s in batch:
            if close is not None and s in close and close[s].notna().any():
                series = close[s].dropna()
                series.index = pd.to_datetime(series.index).tz_localize(None).normalize()
                closes[s] = series[series > 0]
        time.sleep(1)
    # 코스피로 못 찾은 한국 코드는 코스닥으로 한 번 더
    retry = [s for s in symbols if s not in closes and s.endswith(".KS")]
    if retry:
        found = download_kq([s[:-3] + ".KQ" for s in retry])
        for s in retry:
            if s[:-3] + ".KQ" in found:
                closes[s] = found[s[:-3] + ".KQ"]
    return closes


def download_kq(symbols: list[str]) -> dict[str, pd.Series]:
    import yfinance as yf

    if not symbols:
        return {}
    df = yf.download(symbols, start=START, auto_adjust=True, progress=False, threads=True, group_by="column",
                     multi_level_index=True)
    out = {}
    for s in symbols:
        if s in df["Close"] and df["Close"][s].notna().any():
            series = df["Close"][s].dropna()
            series.index = pd.to_datetime(series.index).tz_localize(None).normalize()
            out[s] = series[series > 0]
    return out


# ---------------------------------------------------------------- 저장
def build(items: list[dict], closes: dict[str, pd.Series], fx: pd.Series) -> tuple[dict, list[dict]]:
    items = [it for it in items if it["symbol"] in closes and len(closes[it["symbol"]]) > 20]
    frame = pd.DataFrame({it["code"]: closes[it["symbol"]] for it in items}).sort_index()
    dates = frame.index.union(fx.index).sort_values()
    frame = frame.reindex(dates)
    base = pd.Timestamp(dates[0])
    day_numbers = [(d - base).days for d in dates]

    def pack(series: pd.Series) -> dict:
        start = int(np.argmax(series.notna().to_numpy()))
        values = series.iloc[start:].ffill().to_numpy(dtype=float)
        return {"s": start, "d": encode(values)}

    chunks: list[dict] = []
    rows = []
    for n, it in enumerate(items):
        chunk = n // CHUNK
        if chunk == len(chunks):
            chunks.append({})
        packed = pack(frame[it["code"]])
        chunks[chunk][it["code"]] = packed
        currency = "KRW" if it["symbol"].endswith((".KS", ".KQ")) else "USD"
        rows.append([it["code"], it["name"], it["kind"], it["desc"], it["alias"], currency, chunk, packed["s"]])

    index = {
        "updated": date.today().isoformat(),
        "base": base.strftime("%Y-%m-%d"),
        "days": [day_numbers[0]] + np.diff(day_numbers).tolist(),
        "fx": pack(fx.reindex(dates)),
        "fields": ["code", "name", "kind", "desc", "alias", "currency", "chunk", "start"],
        "tickers": rows,
    }
    return index, chunks


def main() -> None:
    items = universe()
    print(f"종목 후보 {len(items)}개 " + ", ".join(
        f"{k} {sum(it['kind'] == k for it in items)}" for k in ("미국 ETF", "미국 주식", "한국 ETF", "한국 주식")))
    closes = download([it["symbol"] for it in items] + [FX_TICKER])
    fx = closes.pop(FX_TICKER)
    missing = [it["code"] for it in items if it["symbol"] not in closes]
    print(f"시세를 받은 종목 {len(closes)}개, 못 받은 종목 {len(missing)}개: {', '.join(missing[:60])}")

    index, chunks = build(items, closes, fx)
    if (OUT / "c").exists():
        for f in (OUT / "c").glob("*.json"):
            f.unlink()
    (OUT / "c").mkdir(parents=True, exist_ok=True)
    dump = lambda obj: json.dumps(obj, ensure_ascii=False, separators=(",", ":"))  # noqa: E731
    (OUT / "index.json").write_text(dump(index))
    for n, chunk in enumerate(chunks):
        (OUT / "c" / f"{n}.json").write_text(dump(chunk))
    size = sum(f.stat().st_size for f in OUT.rglob("*.json")) / 1e6
    print(f"{len(index['tickers'])}개 종목, 묶음 {len(chunks)}개, 모두 {size:.1f}MB, 마지막 날 {index['base']} + {sum(index['days'])}일")
    for row in index["tickers"][:5] + index["tickers"][-5:]:
        print("  ", row[:4])


if __name__ == "__main__":
    main()
