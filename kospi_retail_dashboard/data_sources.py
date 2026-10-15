"""KOSPI·개인 수급·증시자금 데이터 수집 모듈.

모든 금액은 '원', 거래량은 '주', 지수는 'pt' 단위로 통일해 반환한다.
날짜는 DatetimeIndex(이름: "date")로 반환한다.

소스 우선순위
- KOSPI 지수      : KRX(pykrx, 로그인 필요) → 네이버 차트
- 개인 순매수     : KRX(pykrx, 로그인 필요) → 네이버 투자자별 매매동향
- 개인 거래량/대금 : KRX(pykrx, 로그인 필요)  ※ 대체 소스 없음
- 예탁금/신용융자  : 금융투자협회 FreeSIS → 네이버 증시자금동향
"""

from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass, field

import pandas as pd
import requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"
    )
}
TIMEOUT = 20

KOFIA_URL = "https://freesis.kofia.or.kr/meta/getMetaDataList.do"
KOFIA_FUNDS = "STATSCU0100000060BO"  # 증시자금추이 (투자자예탁금 등)
KOFIA_CREDIT = "STATSCU0100000070BO"  # 신용공여 잔고 추이

# 데이터프레임 컬럼 → 한글 이름
LABELS = {
    "kospi_open": "KOSPI 시가",
    "kospi_high": "KOSPI 고가",
    "kospi_low": "KOSPI 저가",
    "kospi_close": "KOSPI 종가",
    "kospi_value": "KOSPI 거래대금(원)",
    "retail_net": "개인 순매수(원)",
    "retail_buy_value": "개인 매수대금(원)",
    "retail_sell_value": "개인 매도대금(원)",
    "market_value": "시장 전체 거래대금(원)",
    "retail_buy_volume": "개인 매수량(주)",
    "retail_sell_volume": "개인 매도량(주)",
    "deposit": "투자자예탁금(원)",
    "credit": "신용융자 잔고 전체(원)",
    "credit_kospi": "신용융자 잔고 유가증권(원)",
}
RAW_COLUMNS = list(LABELS)
# 잔고(수준) 데이터: 거래일 기준으로 맞출 때 앞 값으로 채운다.
LEVEL_COLUMNS = ["deposit", "credit", "credit_kospi"]


class SourceError(RuntimeError):
    """데이터 소스 하나가 실패했을 때."""


@dataclass
class LoadReport:
    """어떤 소스가 성공/실패했는지 기록한다."""

    items: list[tuple[str, bool, str]] = field(default_factory=list)

    def ok(self, name: str, msg: str = "") -> None:
        self.items.append((name, True, msg))

    def fail(self, name: str, msg: str) -> None:
        self.items.append((name, False, msg))


def _ymd(d) -> str:
    return pd.Timestamp(d).strftime("%Y%m%d")


def _to_number(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce"
    )


def _date_chunks(start, end, days: int = 365):
    """긴 기간을 days 단위로 나눈다 (KRX 조회 기간 제한 대응)."""
    cur, end = pd.Timestamp(start), pd.Timestamp(end)
    while cur <= end:
        nxt = min(cur + pd.Timedelta(days=days - 1), end)
        yield cur, nxt
        cur = nxt + pd.Timedelta(days=1)


# ---------------------------------------------------------------------------
# KRX (pykrx)
# ---------------------------------------------------------------------------
def set_krx_credentials(krx_id: str | None, krx_pw: str | None) -> bool:
    """pykrx 는 환경변수 KRX_ID / KRX_PW 로 로그인한다."""
    if krx_id and krx_pw:
        os.environ["KRX_ID"] = krx_id
        os.environ["KRX_PW"] = krx_pw
    return bool(os.getenv("KRX_ID") and os.getenv("KRX_PW"))


def _pykrx_stock():
    try:
        from pykrx import stock  # 지연 import: 로그인 시도가 import 시점에 일어남
    except ImportError as e:  # pragma: no cover
        raise SourceError("pykrx 가 설치되어 있지 않습니다 (pip install pykrx)") from e
    return stock


def _krx_concat(fn, start, end, **kwargs) -> pd.DataFrame:
    frames = []
    for s, e in _date_chunks(start, end):
        df = fn(_ymd(s), _ymd(e), **kwargs)
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        raise SourceError("KRX 응답이 비어 있습니다 (로그인 정보 또는 기간을 확인하세요)")
    out = pd.concat(frames)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out.index = pd.to_datetime(out.index)
    out.index.name = "date"
    return out


def fetch_kospi_krx(start, end) -> pd.DataFrame:
    stock = _pykrx_stock()
    df = _krx_concat(stock.get_index_ohlcv, start, end, ticker="1001")
    out = pd.DataFrame(
        {
            "kospi_open": df["시가"],
            "kospi_high": df["고가"],
            "kospi_low": df["저가"],
            "kospi_close": df["종가"],
        }
    )
    if "거래대금" in df:
        out["kospi_value"] = df["거래대금"]
    return out.astype(float)


def fetch_investor_krx(start, end) -> pd.DataFrame:
    """KOSPI 시장 개인 순매수·매수/매도 대금·거래량."""
    stock = _pykrx_stock()
    val = stock.get_market_trading_value_by_date
    vol = stock.get_market_trading_volume_by_date
    net = _krx_concat(val, start, end, ticker="KOSPI", on="순매수")
    buy = _krx_concat(val, start, end, ticker="KOSPI", on="매수")
    sell = _krx_concat(val, start, end, ticker="KOSPI", on="매도")
    buy_v = _krx_concat(vol, start, end, ticker="KOSPI", on="매수")
    sell_v = _krx_concat(vol, start, end, ticker="KOSPI", on="매도")
    out = pd.DataFrame(
        {
            "retail_net": net["개인"],
            "retail_buy_value": buy["개인"],
            "retail_sell_value": sell["개인"],
            "market_value": buy["전체"],
            "retail_buy_volume": buy_v["개인"],
            "retail_sell_volume": sell_v["개인"],
        }
    )
    return out.astype(float)


# ---------------------------------------------------------------------------
# 네이버 금융 (로그인 불필요, 대체 소스)
# ---------------------------------------------------------------------------
def parse_naver_chart(text: str) -> pd.DataFrame:
    """fchart XML: <item data="20240102|시가|고가|저가|종가|거래량"/>"""
    rows = [r.split("|") for r in re.findall(r'data="([^"]+)"', text)]
    if not rows:
        raise SourceError("네이버 차트 응답을 해석하지 못했습니다")
    df = pd.DataFrame([r[:5] for r in rows], columns=["date", "o", "h", "l", "c"])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.set_index("date").astype(float)
    df.columns = ["kospi_open", "kospi_high", "kospi_low", "kospi_close"]
    return df


def fetch_kospi_naver(start, end) -> pd.DataFrame:
    count = (pd.Timestamp.today() - pd.Timestamp(start)).days + 10
    r = requests.get(
        "https://fchart.stock.naver.com/sise.nhn",
        params={"symbol": "KOSPI", "timeframe": "day", "count": count, "requestType": 0},
        headers=HEADERS,
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    df = parse_naver_chart(r.text)
    return df.loc[pd.Timestamp(start) : pd.Timestamp(end)]


def _read_naver_table(html: str, must_contain: str) -> pd.DataFrame:
    tables = pd.read_html(io.StringIO(html), thousands=",")
    for t in tables:
        if must_contain in " ".join(map(str, t.columns.to_flat_index())):
            t = t.copy()
            t.columns = range(t.shape[1])
            t["date"] = pd.to_datetime(t[0], format="%y.%m.%d", errors="coerce")
            return t.dropna(subset=["date"])
    raise SourceError(f"'{must_contain}' 표를 찾지 못했습니다")


def parse_naver_investor(html: str) -> pd.DataFrame:
    """투자자별 매매동향(일별): 날짜 | 개인 | 외국인 | 기관계 ... (억원)"""
    t = _read_naver_table(html, "개인")
    out = pd.DataFrame({"retail_net": _to_number(t[1]).values * 1e8}, index=t["date"])
    out.index.name = "date"
    return out


def parse_naver_deposit(html: str) -> pd.DataFrame:
    """증시자금동향: 날짜 | 고객예탁금 | 증감 | 신용잔고 | 증감 | ... (억원)"""
    t = _read_naver_table(html, "고객예탁금")
    out = pd.DataFrame(
        {"deposit": _to_number(t[1]).values * 1e8, "credit": _to_number(t[3]).values * 1e8},
        index=t["date"],
    )
    out.index.name = "date"
    return out


def _naver_paged(url, params, parser, start, max_pages: int) -> pd.DataFrame:
    frames, seen_first = [], set()
    for page in range(1, max_pages + 1):
        r = requests.get(url, params={**params, "page": page}, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        r.encoding = "euc-kr"
        df = parser(r.text)
        if df.empty or df.index[0] in seen_first:  # 마지막 페이지 이후 같은 페이지가 반복됨
            break
        seen_first.add(df.index[0])
        frames.append(df)
        if df.index.min() <= pd.Timestamp(start):
            break
    if not frames:
        raise SourceError("네이버 응답이 비어 있습니다")
    out = pd.concat(frames).sort_index()
    return out[~out.index.duplicated(keep="first")]


def fetch_investor_naver(start, end, max_pages: int = 300) -> pd.DataFrame:
    df = _naver_paged(
        "https://finance.naver.com/sise/investorDealTrendDay.naver",
        {"bizdate": _ymd(end), "sosok": ""},  # sosok 빈 값 = 코스피
        parse_naver_investor,
        start,
        max_pages,
    )
    return df.loc[pd.Timestamp(start) : pd.Timestamp(end)]


def fetch_funds_naver(start, end, max_pages: int = 300) -> pd.DataFrame:
    df = _naver_paged(
        "https://finance.naver.com/sise/sise_deposit.naver", {}, parse_naver_deposit, start, max_pages
    )
    return df.loc[pd.Timestamp(start) : pd.Timestamp(end)]


# ---------------------------------------------------------------------------
# 금융투자협회 FreeSIS
# ---------------------------------------------------------------------------
def _scale_to_won(s: pd.Series, lo: float = 1e12, hi: float = 1e15) -> pd.Series:
    """FreeSIS 는 백만원 단위로 응답한다. 단위가 바뀌어도 '조원' 대역에 들도록 보정한다."""
    med = s.dropna().abs().median()
    if not med:
        return s
    for factor in (1e6, 1e8, 1e3, 1.0):
        if lo <= med * factor <= hi:
            return s * factor
    return s * 1e6


def parse_kofia(payload: dict, mapping: dict[str, str]) -> pd.DataFrame:
    rows = payload.get("ds1") or []
    if not rows:
        raise SourceError("FreeSIS 응답이 비어 있습니다")
    df = pd.DataFrame(rows)
    df.columns = [c.upper() for c in df.columns]
    df["date"] = pd.to_datetime(df["TMPV1"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    out = pd.DataFrame({new: _scale_to_won(_to_number(df[old])) for old, new in mapping.items()})
    return out


def _kofia_request(obj_nm: str, start, end) -> dict:
    body = {
        "dmSearch": {
            "tmpV40": "1000000",  # 백만원 단위
            "tmpV41": "1",
            "tmpV1": "D",  # 일별
            "tmpV45": _ymd(start),
            "tmpV46": _ymd(end),
            "OBJ_NM": obj_nm,
        }
    }
    r = requests.post(KOFIA_URL, json=body, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def fetch_funds_kofia(start, end) -> pd.DataFrame:
    frames = []
    for s, e in _date_chunks(start, end, days=730):
        dep = parse_kofia(_kofia_request(KOFIA_FUNDS, s, e), {"TMPV2": "deposit"})
        cre = parse_kofia(
            _kofia_request(KOFIA_CREDIT, s, e), {"TMPV2": "credit", "TMPV3": "credit_kospi"}
        )
        frames.append(dep.join(cre, how="outer"))
    out = pd.concat(frames).sort_index()
    return out[~out.index.duplicated(keep="last")]


# ---------------------------------------------------------------------------
# 통합
# ---------------------------------------------------------------------------
def _first_success(report: LoadReport, label: str, attempts) -> pd.DataFrame | None:
    errors = []
    for name, fn in attempts:
        try:
            df = fn()
            if df is None or df.empty:
                raise SourceError("데이터 없음")
            report.ok(label, f"{name} ({len(df):,}행)")
            return df
        except Exception as e:  # 네트워크/파싱 오류 모두 다음 소스로 넘긴다
            errors.append(f"{name}: {type(e).__name__}: {e}"[:200])
    report.fail(label, " / ".join(errors) if errors else "사용 가능한 소스 없음")
    return None


def align(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """여러 소스를 하나의 일별 표로 합친다. KOSPI 가 있으면 KOSPI 거래일을 기준으로 한다."""
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame(columns=RAW_COLUMNS)
    df = pd.concat(frames, axis=1).sort_index()
    df = df.loc[:, ~df.columns.duplicated()]
    levels = [c for c in LEVEL_COLUMNS if c in df]
    if levels:
        df[levels] = df[levels].ffill(limit=5)
    if "kospi_close" in df:
        df = df[df["kospi_close"].notna()]
    df.index.name = "date"
    return df


def load_market_data(start, end, use_krx: bool) -> tuple[pd.DataFrame, LoadReport]:
    report = LoadReport()
    kospi_attempts, investor_attempts = [], []
    if use_krx:
        kospi_attempts.append(("KRX", lambda: fetch_kospi_krx(start, end)))
        investor_attempts.append(("KRX", lambda: fetch_investor_krx(start, end)))
    kospi_attempts.append(("네이버", lambda: fetch_kospi_naver(start, end)))
    investor_attempts.append(("네이버", lambda: fetch_investor_naver(start, end)))
    funds_attempts = [
        ("금융투자협회", lambda: fetch_funds_kofia(start, end)),
        ("네이버", lambda: fetch_funds_naver(start, end)),
    ]
    kospi = _first_success(report, "KOSPI 지수", kospi_attempts)
    investor = _first_success(report, "개인 수급", investor_attempts)
    funds = _first_success(report, "예탁금·신용융자", funds_attempts)
    return align([kospi, investor, funds]), report


def read_uploaded_csv(buf) -> pd.DataFrame:
    """사용자 CSV: 'date' 컬럼 + RAW_COLUMNS(영문 키 또는 한글 이름) 중 일부."""
    df = pd.read_csv(buf)
    rename = {v: k for k, v in LABELS.items()}
    rename.update({"날짜": "date", "일자": "date"})
    df = df.rename(columns=rename)
    if "date" not in df:
        raise SourceError("CSV 에 'date'(또는 '날짜') 컬럼이 필요합니다")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    cols = [c for c in RAW_COLUMNS if c in df]
    if not cols:
        raise SourceError("인식할 수 있는 데이터 컬럼이 없습니다")
    return align([df[cols].apply(_to_number)])
