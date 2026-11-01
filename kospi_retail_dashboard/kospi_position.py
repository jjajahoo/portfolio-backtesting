"""KOSPI 현재 위치 리포트 (터미널용).

사용 예
    python kospi_position.py                 # 실제 데이터 (네이버·금융투자협회, 가능한 만큼)
    python kospi_position.py --demo          # 가상 데이터로 출력 형식 확인
    python kospi_position.py --years 10 --csv position.csv

KRX 로그인(KRX_ID / KRX_PW)이 있으면 PER·PBR·배당수익률·시가총액이, ECOS_KEY 가 있으면
국고채 10년물이 추가됩니다. 옵션(--krx-id 등) 또는 환경변수로 넣을 수 있습니다.
"""

from __future__ import annotations

import argparse
import os
import sys
import unicodedata
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from data_sources import load_market_data, set_krx_credentials  # noqa: E402
from indicators import add_indicators, make_demo_data  # noqa: E402
from position import CATEGORIES, build_position  # noqa: E402


def _width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _pad(text: str, width: int, right: bool = False) -> str:
    gap = " " * max(width - _width(text), 0)
    return gap + text if right else text + gap


def _bar(score: float, width: int = 20) -> str:
    filled = round(score / 100 * width)
    return "█" * filled + "·" * (width - filled)


def render_report(snap: pd.DataFrame, summ: dict, levels: pd.DataFrame, lookback_years: float,
                  report_items=None) -> str:
    lines = []
    if "overall" not in summ:
        return "시장 위치를 계산할 데이터가 부족합니다."
    ov = summ["overall"]
    lines.append(f"KOSPI 시장 위치 리포트 — 기준일 {ov['date']:%Y-%m-%d}, 비교 기간 최근 {lookback_years:g}년")
    lines.append("=" * 72)
    if not levels.empty:
        lines.append(f"KOSPI {levels.iloc[0]['지수']:,.2f}")
    chg = "" if ov["change_1m"] is None else f"  (1개월 전 대비 {ov['change_1m']:+.0f})"
    lines.append(f"종합 점수  {ov['score']:5.1f} / 100  [{_bar(ov['score'])}]  {ov['label']}{chg}")
    for cat in CATEGORIES:
        if cat in summ:
            c = summ[cat]
            lines.append(f"  {_pad(cat, 12)}{c['score']:5.1f}  [{_bar(c['score'])}]  {c['label']}")
        else:
            lines.append(f"  {_pad(cat, 12)}  -    (데이터 없음)")
    lines.append("")
    lines.append("0 = 침체·저평가 · 50 = 과거 평균 수준 · 100 = 과열·고평가")
    lines.append("")

    head = (_pad("카테고리", 12) + _pad("지표", 24) + _pad("현재값", 16, True)
            + _pad("백분위", 8, True) + _pad("온도", 6, True) + "  해석")
    lines.append(head)
    lines.append("-" * 72)
    for _, r in snap.iterrows():
        value = f"{r['현재값']:,.{int(r['digits'])}f}{r['단위']}"
        lines.append(
            _pad(r["카테고리"], 12) + _pad(r["지표"], 24) + _pad(value, 16, True)
            + _pad(f"{r['백분위']:.0f}%", 8, True) + _pad(f"{r['온도']:.0f}", 6, True) + f"  {r['해석']}"
        )
    stale = snap[snap["기준일"] < snap["기준일"].max()]
    if not stale.empty:
        lines.append("  ※ 기준일이 다른 지표: " + ", ".join(
            f"{r['지표']}({r['기준일']:%m-%d})" for _, r in stale.iterrows()))
    lines.append("")

    if not levels.empty:
        lines.append("주요 가격 레벨")
        lines.append("-" * 72)
        for _, r in levels.iterrows():
            lines.append(f"  {_pad(r['구분'], 20)}{r['지수']:>12,.2f}   현재가 대비 {r['현재가 대비(%)']:+6.1f}%")
        if "alignment" in levels.attrs:
            lines.append(f"  이동평균 배열: {levels.attrs['alignment']}")
        lines.append("")

    if report_items:
        lines.append("데이터 소스")
        lines.append("-" * 72)
        for name, ok, msg in report_items:
            lines.append(f"  {'OK ' if ok else '실패'} {name}: {msg}")
        lines.append("")
    lines.append("참고용 지표이며 투자 판단의 근거가 아닙니다.")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="KOSPI 현재 위치(밸류에이션·추세·수급·심리) 리포트")
    ap.add_argument("--demo", action="store_true", help="가상 데이터 사용 (네트워크 불필요)")
    ap.add_argument("--years", type=float, default=5, help="비교 기간(년), 기본 5")
    ap.add_argument("--flow-window", type=int, default=20, help="수급 이동합 기간(거래일), 기본 20")
    ap.add_argument("--end", default=None, help="기준일 YYYY-MM-DD (기본: 오늘)")
    ap.add_argument("--krx-id", default=os.getenv("KRX_ID"))
    ap.add_argument("--krx-pw", default=os.getenv("KRX_PW"))
    ap.add_argument("--ecos-key", default=os.getenv("ECOS_KEY"))
    ap.add_argument("--csv", help="지표 표를 CSV 로 저장할 경로")
    ap.add_argument("--history-csv", help="날짜별 점수(종합·카테고리·지표 온도)를 CSV 로 저장할 경로")
    args = ap.parse_args(argv)

    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=int(args.years * 365 + 400))
    report_items = None
    if args.demo:
        raw = make_demo_data(start, end)
        print("※ 데모 모드: 가상 데이터입니다. 실제 시장 데이터가 아닙니다.\n")
    else:
        use_krx = set_krx_credentials(args.krx_id, args.krx_pw)
        print("데이터를 불러오는 중… (네이버 소스만 쓰면 수 분 걸릴 수 있습니다)", file=sys.stderr)
        raw, report = load_market_data(start, end, use_krx, args.ecos_key)
        report_items = report.items
        if raw.empty:
            print("불러온 데이터가 없습니다.")
            for name, ok, msg in report_items:
                print(f"  {'OK ' if ok else '실패'} {name}: {msg}")
            return 1

    full = add_indicators(raw, flow_window=args.flow_window)
    _, scores, snap, summ, levels = build_position(full, args.years)
    if snap.empty:
        print("시장 위치를 계산할 데이터가 부족합니다.")
        return 1
    print(render_report(snap, summ, levels, args.years, report_items))

    if args.csv:
        snap.drop(columns=["key", "digits"]).to_csv(args.csv, index=False, encoding="utf-8-sig")
        print(f"\n저장: {args.csv}")
    if args.history_csv:
        scores.dropna(how="all").round(2).to_csv(args.history_csv, encoding="utf-8-sig")
        print(f"저장: {args.history_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
