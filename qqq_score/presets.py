"""설정 저장/불러오기 (JSON)."""
from __future__ import annotations

import json

from .conditions import OPS, Condition
from .indicators import BY_KEY, HIGH_GOOD, LOW_GOOD, PERCENTILE, RANGE
from .scoring import IndicatorSetting, default_setting

VERSION = 1


def to_json(
    settings: dict[str, IndicatorSetting],
    conditions: list[Condition],
    how: str,
    options: dict,
) -> str:
    payload = {
        "version": VERSION,
        "indicators": {k: s.to_dict() for k, s in settings.items()},
        "conditions": [c.to_dict() for c in conditions],
        "how": how,
        "options": options,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def from_json(text: str) -> tuple[dict[str, IndicatorSetting], list[Condition], str, dict]:
    """알 수 없는 지표·잘못된 값은 버리고 기본값으로 채운다."""
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("설정 파일 형식이 아닙니다.")
    settings = {}
    for key, ind in BY_KEY.items():
        base = default_setting(key)
        raw = payload.get("indicators", {}).get(key)
        if isinstance(raw, dict):
            base = _merge_setting(base, raw, ind)
        settings[key] = base
    conditions = []
    for c in payload.get("conditions", []):
        try:
            cond = Condition(str(c["operand"]), str(c["op"]), float(c["value"]), float(c.get("value2", 0)))
        except (KeyError, TypeError, ValueError):
            continue
        if cond.op in OPS:
            conditions.append(cond)
    how = payload.get("how", "and")
    how = how if how in ("and", "or") else "and"
    options = payload.get("options", {})
    return settings, conditions, how, options if isinstance(options, dict) else {}


def _merge_setting(base: IndicatorSetting, raw: dict, ind) -> IndicatorSetting:
    def num(name, cur):
        try:
            return float(raw.get(name, cur))
        except (TypeError, ValueError):
            return cur

    params = dict(base.params)
    for p in ind.params:
        val = raw.get("params", {}).get(p.key, params[p.key])
        try:
            val = min(max(float(val), p.min), p.max)
            params[p.key] = int(val) if p.integer else val
        except (TypeError, ValueError):
            pass
    method = raw.get("method", base.method)
    direction = raw.get("direction", base.direction)
    return IndicatorSetting(
        enabled=bool(raw.get("enabled", base.enabled)),
        in_score=bool(raw.get("in_score", base.in_score)),
        weight=max(num("weight", base.weight), 0.0),
        method=method if method in (PERCENTILE, RANGE) else base.method,
        direction=direction if direction in (LOW_GOOD, HIGH_GOOD) else base.direction,
        window_years=max(num("window_years", base.window_years), 0.0),
        low=num("low", base.low),
        high=num("high", base.high),
        params=params,
    )
