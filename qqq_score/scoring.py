"""원값 → 0~100점 변환과 종합점수.

점수는 모두 '그 시점까지의 과거 데이터'만으로 계산한다.
- 백분위: 과거 N년(0이면 전체 누적) 분포에서 현재 값의 위치
- 구간: 하한~상한 사이를 선형으로 0~100에 대응 (범위 밖은 0 또는 100)
방향이 '낮을수록 고득점'이면 점수를 뒤집는다.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .data import MarketData, align
from .indicators import BY_KEY, INDICATORS, LOW_GOOD, PERCENTILE

TRADING_DAYS = 252


@dataclass
class IndicatorSetting:
    enabled: bool
    in_score: bool
    weight: float
    method: str
    direction: str
    window_years: float
    low: float
    high: float
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def default_setting(key: str) -> IndicatorSetting:
    ind = BY_KEY[key]
    sd = ind.score
    return IndicatorSetting(
        enabled=ind.enabled,
        in_score=ind.in_score,
        weight=ind.weight,
        method=sd.method,
        direction=sd.direction,
        window_years=sd.window_years,
        low=sd.low,
        high=sd.high,
        params=ind.default_params(),
    )


def default_settings() -> dict[str, IndicatorSetting]:
    return {ind.key: default_setting(ind.key) for ind in INDICATORS}


def percentile_score(raw: pd.Series, window_years: float) -> pd.Series:
    """과거 창 안에서 현재 값의 백분위(0=최저, 100=최고)."""
    s = raw.replace([np.inf, -np.inf], np.nan)
    w = int(round(window_years * TRADING_DAYS))
    minp = min(TRADING_DAYS, w) if w > 0 else TRADING_DAYS
    minp = max(minp, 2)
    roll = s.expanding(min_periods=minp) if w <= 0 else s.rolling(w, min_periods=minp)
    rank = roll.rank()
    count = roll.count()
    return (rank - 1) / (count - 1) * 100


def range_score(raw: pd.Series, low: float, high: float) -> pd.Series:
    """하한 이하 0, 상한 이상 100, 사이는 선형."""
    s = raw.replace([np.inf, -np.inf], np.nan)
    if high == low:
        return (s >= high).astype(float).where(s.notna()) * 100
    lo, hi = min(low, high), max(low, high)
    return ((s - lo) / (hi - lo)).clip(0, 1) * 100


def score_series(raw: pd.Series, setting: IndicatorSetting) -> pd.Series:
    if setting.method == PERCENTILE:
        pct = percentile_score(raw, setting.window_years)
    else:
        pct = range_score(raw, setting.low, setting.high)
    return 100 - pct if setting.direction == LOW_GOOD else pct


def composite_score(scores: pd.DataFrame, weights: dict[str, float], require_all: bool = True) -> pd.Series:
    """가중평균 종합점수. require_all이면 반영 지표가 하나라도 비는 날은 계산하지 않는다."""
    cols = [c for c in scores.columns if weights.get(c, 0) > 0]
    if not cols:
        return pd.Series(np.nan, index=scores.index, name="composite")
    s = scores[cols]
    w = pd.Series({c: float(weights[c]) for c in cols})
    num = s.mul(w, axis=1).sum(axis=1, min_count=1)
    den = s.notna().mul(w, axis=1).sum(axis=1)
    comp = (num / den).where(den > 0)
    if require_all:
        comp = comp.where(s.notna().all(axis=1))
    return comp.rename("composite")


@dataclass
class EngineResult:
    raw: pd.DataFrame                 # 지표 원값 (QQQ 거래일)
    scores: pd.DataFrame              # 지표 점수 (QQQ 거래일)
    composite: pd.Series              # 종합점수
    weights: dict[str, float]         # 종합점수에 들어간 지표와 가중치
    overlays: dict[str, dict[str, pd.Series]]  # 지표 key → 가격 차트 위에 그릴 선들
    labels: dict[str, str]            # 지표 key → 표시 이름
    errors: dict[str, str]            # 계산 실패한 지표와 사유


def run(data: MarketData, settings: dict[str, IndicatorSetting], require_all: bool = True) -> EngineResult:
    idx = data.index
    raw, scores, overlays, labels, errors = {}, {}, {}, {}, {}
    for ind in INDICATORS:
        st = settings[ind.key]
        if not st.enabled:
            continue
        params = {**ind.default_params(), **st.params}
        labels[ind.key] = ind.label(params)
        try:
            values = ind.compute(data, params).replace([np.inf, -np.inf], np.nan)
            if values.dropna().empty:
                errors[ind.key] = "데이터 없음"
                continue
            # 점수는 원값의 고유 인덱스에서 계산한 뒤 거래일로 맞춘다
            raw[ind.key] = align(values, idx)
            scores[ind.key] = align(score_series(values, st), idx)
            if ind.overlay is not None:
                overlays[ind.key] = {k: v.reindex(idx) for k, v in ind.overlay(data, params).items()}
        except Exception as exc:  # 한 지표 실패가 앱 전체를 멈추지 않게
            errors[ind.key] = f"{type(exc).__name__}: {exc}"
    raw_df = pd.DataFrame(raw, index=idx)
    score_df = pd.DataFrame(scores, index=idx)
    weights = {
        k: float(settings[k].weight)
        for k in score_df.columns
        if settings[k].in_score and settings[k].weight > 0
    }
    comp = composite_score(score_df, weights, require_all)
    return EngineResult(raw_df, score_df, comp, weights, overlays, labels, errors)
