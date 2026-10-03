"""숫자를 읽기 쉬운 한국어로 바꾸는 함수들."""

from __future__ import annotations


def money(value: float, currency: str) -> str:
    """원화는 '1억 2,345만 원', 달러는 '$12,345' 처럼."""
    if currency == "USD":
        return f"${value:,.0f}"
    sign = "-" if value < 0 else ""
    won = round(abs(value))
    if won < 10_000:
        return f"{sign}{won:,}원"
    eok, man = divmod(round(won / 10_000), 10_000)
    if eok and man:
        return f"{sign}{eok:,}억 {man:,}만 원"
    if eok:
        return f"{sign}{eok:,}억 원"
    return f"{sign}{man:,}만 원"


def pct(value: float | None, digits: int = 1, signed: bool = False) -> str:
    if value is None:
        return "-"
    return f"{value * 100:{'+' if signed else ''}.{digits}f}%"


def pp(value: float, digits: int = 1) -> str:
    """%포인트 차이."""
    return f"{value * 100:+.{digits}f}%p"


def ratio(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def duration(days: int) -> str:
    """일 수를 '1년 3개월' 처럼."""
    months = round(days / 30.44)
    if months < 1:
        return f"{days}일"
    years, months = divmod(months, 12)
    if years and months:
        return f"{years}년 {months}개월"
    if years:
        return f"{years}년"
    return f"{months}개월"


def ym(ts) -> str:
    return f"{ts.year}년 {ts.month}월"
