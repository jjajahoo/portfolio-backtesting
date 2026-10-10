"""조건식 평가와 해당 구간(연속된 날짜 묶음) 추출."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from .scoring import EngineResult

OPS = {
    ">=": "≥",
    "<=": "≤",
    ">": ">",
    "<": "<",
    "between": "사이",
    "outside": "바깥",
}

COMPOSITE = "composite"


@dataclass
class Condition:
    operand: str            # "raw:<지표key>", "score:<지표key>", "composite"
    op: str
    value: float
    value2: float = 0.0     # between/outside 의 두 번째 값

    def to_dict(self) -> dict:
        return asdict(self)


def operand_options(result: EngineResult) -> dict[str, str]:
    """조건에 쓸 수 있는 항목: operand → 표시 이름."""
    opts = {COMPOSITE: "종합점수"}
    for key, label in result.labels.items():
        if key in result.raw.columns:
            opts[f"raw:{key}"] = f"{label} 값"
            opts[f"score:{key}"] = f"{label} 점수"
    return opts


def operand_series(result: EngineResult, operand: str) -> pd.Series | None:
    if operand == COMPOSITE:
        return result.composite
    kind, _, key = operand.partition(":")
    frame = result.raw if kind == "raw" else result.scores if kind == "score" else None
    if frame is None or key not in frame.columns:
        return None
    return frame[key]


def evaluate_one(s: pd.Series, cond: Condition) -> pd.Series:
    v, v2 = float(cond.value), float(cond.value2)
    lo, hi = min(v, v2), max(v, v2)
    if cond.op == ">=":
        m = s >= v
    elif cond.op == "<=":
        m = s <= v
    elif cond.op == ">":
        m = s > v
    elif cond.op == "<":
        m = s < v
    elif cond.op == "between":
        m = (s >= lo) & (s <= hi)
    elif cond.op == "outside":
        m = (s < lo) | (s > hi)
    else:
        raise ValueError(f"알 수 없는 연산자: {cond.op}")
    return m.fillna(False).astype(bool)


def evaluate(
    result: EngineResult, conditions: list[Condition], how: str = "and"
) -> tuple[pd.Series, pd.Series] | None:
    """조건을 AND/OR로 묶는다. 유효한 조건이 없으면 None.

    반환: (hit, valid) - hit은 조건 충족일, valid는 조건에 쓰인 값이 모두 있는 날.
    값이 없는 날(지표 시작 전 등)은 hit=False, valid=False.
    """
    hits, valids = [], []
    for cond in conditions:
        s = operand_series(result, cond.operand)
        if s is not None:
            hits.append(evaluate_one(s, cond))
            valids.append(s.notna())
    if not hits:
        return None
    hit, valid = hits[0], valids[0]
    for h, v in zip(hits[1:], valids[1:]):
        hit = (hit & h) if how == "and" else (hit | h)
        valid = valid & v
    hit = hit & valid
    return hit.rename("condition"), valid.rename("valid")


def describe(cond: Condition, options: dict[str, str]) -> str:
    name = options.get(cond.operand, cond.operand)
    if cond.op in ("between", "outside"):
        lo, hi = sorted((cond.value, cond.value2))
        word = "이상 ~" if cond.op == "between" else "미만 또는"
        tail = "이하" if cond.op == "between" else "초과"
        return f"{name} {lo:g} {word} {hi:g} {tail}"
    return f"{name} {OPS[cond.op]} {cond.value:g}"
