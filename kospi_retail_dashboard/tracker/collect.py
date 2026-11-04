"""KOSPI 가치 추적용 데이터 수집.

일별 KOSPI 종가·시가총액·PER·PBR·배당수익률과 국고채 10년 금리를 모아
tracker/data/kospi_value.csv 에 쌓는다. 이미 파일이 있으면 최근 구간만 다시 받아 합친다.

    python kospi_retail_dashboard/tracker/collect.py            # 갱신
    python kospi_retail_dashboard/tracker/collect.py --full     # 처음부터 다시

환경변수
- KRX_ID, KRX_PW : KRX 정보데이터시스템 계정 (KRX 가 로그인을 요구할 때)
- ECOS_API_KEY   : 한국은행 ECOS 오픈 API 키 (국고채 10년 금리, 없으면 건너뜀)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import data_sources as ds  # noqa: E402

DATA_DIR = HERE / "data"
CSV_PATH = DATA_DIR / "kospi_value.csv"
META_PATH = DATA_DIR / "sources.json"
START = "2005-01-03"
COLUMNS = ["close", "mcap", "per", "pbr", "div_yield", "ktb10y"]
REFETCH_DAYS = 20  # 최근 구간은 정정·지연 공표가 있어 다시 받는다

ECOS_URL = "https://ecos.bok.or.kr/api/StatisticSearch/{key}/json/kr/{first}/{last}/{stat}/{cycle}/{start}/{end}/{item}"
ECOS_KTB10Y = ("817Y002", "D", "010210000")  # 시장금리(일별) > 국고채(10년)


def parse_ecos(payload: dict) -> pd.Series:
    """ECOS StatisticSearch 응답 → 날짜별 값."""
    if "RESULT" in payload:
        raise ds.SourceError(f"ECOS: {payload['RESULT'].get('CODE')} {payload['RESULT'].get('MESSAGE')}")
    rows = payload.get("StatisticSearch", {}).get("row") or []
    if not rows:
        raise ds.SourceError("ECOS 응답이 비어 있습니다")
    s = pd.Series(
        {pd.to_datetime(r["TIME"], format="%Y%m%d"): pd.to_numeric(r["DATA_VALUE"], errors="coerce") for r in rows}
    ).sort_index()
    s.index.name = "date"
    return s


def fetch_ktb10y_ecos(start, end, key: str) -> pd.DataFrame:
    stat, cycle, item = ECOS_KTB10Y
    frames = []
    for s, e in ds._date_chunks(start, end, days=3650):
        url = ECOS_URL.format(key=key, first=1, last=100000, stat=stat, cycle=cycle,
                              start=ds._ymd(s), end=ds._ymd(e), item=item)
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        try:
            frames.append(parse_ecos(r.json()))
        except ds.SourceError as err:
            if "INFO-200" not in str(err):  # INFO-200 = 해당 기간 데이터 없음
                raise
    if not frames:
        raise ds.SourceError("ECOS 에서 국고채 10년 금리를 받지 못했습니다")
    return pd.concat(frames).rename("ktb10y").to_frame()


def _price(start, end, report: ds.LoadReport) -> pd.DataFrame | None:
    df = ds._first_success(report, "KOSPI 지수·시가총액", [
        ("KRX", lambda: ds.fetch_kospi_krx(start, end)),
        ("네이버", lambda: ds.fetch_kospi_naver(start, end)),
    ])
    if df is None:
        return None
    out = pd.DataFrame({"close": df["kospi_close"]})
    if "kospi_mcap" in df:
        out["mcap"] = df["kospi_mcap"] / 1e12  # 조원
    return out


def collect(start, end, report: ds.LoadReport) -> pd.DataFrame:
    price = _price(start, end, report)
    fund = ds._first_success(report, "PER·PBR·배당수익률", [
        ("KRX", lambda: ds.fetch_kospi_fundamental_krx(start, end)),
    ])
    key = os.getenv("ECOS_API_KEY", "").strip()
    if key:
        bond = ds._first_success(report, "국고채 10년 금리", [("한국은행 ECOS", lambda: fetch_ktb10y_ecos(start, end, key))])
    else:
        bond = None
        report.fail("국고채 10년 금리", "ECOS_API_KEY 가 없어 건너뜀 (페이지에서 직접 입력)")
    frames = [f for f in (price, fund) if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.concat(frames, axis=1).sort_index()
    if bond is not None:
        df = df.join(bond, how="left")
        df["ktb10y"] = df["ktb10y"].ffill(limit=5)  # 채권시장 휴장일은 직전 값
    df = df[df["close"].notna()] if "close" in df else df
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df.reindex(columns=COLUMNS)


def merge(existing: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """새로 받은 값을 우선하되, 이번에 받지 못한 칸은 기존 값을 유지한다."""
    if existing is None or existing.empty:
        return new.sort_index()
    out = new.combine_first(existing)
    return out.reindex(columns=COLUMNS).sort_index()


def load_csv(path: Path = CSV_PATH) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["date"], index_col="date")


def save(df: pd.DataFrame, report: ds.LoadReport, path: Path = CSV_PATH, meta_path: Path = META_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.round({"close": 2, "mcap": 1, "per": 2, "pbr": 3, "div_yield": 3, "ktb10y": 3}).to_csv(path, date_format="%Y-%m-%d")
    meta = {
        "updated_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(timespec="minutes"),
        "first": f"{df.index.min():%Y-%m-%d}",
        "last": f"{df.index.max():%Y-%m-%d}",
        "sources": [{"name": n, "ok": ok, "detail": msg} for n, ok, msg in report.items],
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help=f"{START} 부터 전부 다시 받기")
    args = ap.parse_args()

    existing = None if args.full else load_csv()
    end = pd.Timestamp.today().normalize()
    start = (existing.index.max() - pd.Timedelta(days=REFETCH_DAYS)) if existing is not None and len(existing) else pd.Timestamp(START)
    print(f"수집 구간: {start:%Y-%m-%d} ~ {end:%Y-%m-%d}")
    report = ds.LoadReport()
    new = collect(start, end, report)
    for name, ok, msg in report.items:
        print(("OK  " if ok else "--  ") + f"{name}: {msg}")
    if new.empty:
        sys.exit("KOSPI 가격을 받지 못해 파일을 바꾸지 않았습니다.")
    df = merge(existing, new)
    save(df, report)
    last = df.iloc[-1]
    print(f"저장: {CSV_PATH} ({df.index.min():%Y-%m-%d} ~ {df.index.max():%Y-%m-%d}, {len(df):,}행)")
    print("최근:", f"{df.index[-1]:%Y-%m-%d}", last.to_dict())


if __name__ == "__main__":
    main()
