"""이후 수익률 통계.

모든 수익률은 QQQ 수정종가(배당 재투자 포함) 기준, 거래일 수로 기간을 잡는다.
조정/검증 구간을 나눌 때, 조정 구간 표본은 '이후 수익률 계산 기간'이 검증 구간 시작 전에
끝나는 날만 쓴다(검증 구간 가격이 조정 구간 통계에 새어 들어가지 않게).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

HORIZONS: dict[str, int] = {"1개월": 21, "3개월": 63, "6개월": 126, "12개월": 252}

ALL, IN_SAMPLE, OUT_SAMPLE = "전체", "조정 구간", "검증 구간"


def forward_returns(close: pd.Series, horizons: dict[str, int] = HORIZONS) -> pd.DataFrame:
    """각 날짜 종가에 사서 h거래일 뒤 종가에 판 수익률(%)."""
    return pd.DataFrame({name: (close.shift(-h) / close - 1) * 100 for name, h in horizons.items()})


def forward_worst(close: pd.Series, horizons: dict[str, int] = HORIZONS) -> pd.DataFrame:
    """h거래일 안에서 겪은 최악의 평가손(%, 0 이하)."""
    out = {}
    nxt = close.shift(-1)
    for name, h in horizons.items():
        future_min = nxt[::-1].rolling(h, min_periods=h).min()[::-1]
        out[name] = ((future_min / close - 1) * 100).clip(upper=0)
    return pd.DataFrame(out)


@dataclass
class Split:
    enabled: bool
    date: pd.Timestamp

    def parts(self) -> list[str]:
        return [IN_SAMPLE, OUT_SAMPLE] if self.enabled else [ALL]


def sample_mask(index: pd.DatetimeIndex, split: Split, part: str, horizon: int) -> pd.Series:
    """part에 속하면서 h거래일 뒤 수익률을 쓸 수 있는 날짜."""
    n = len(index)
    pos = np.arange(n)
    has_fwd = pos + horizon < n
    if part == ALL:
        m = has_fwd
    else:
        split_pos = index.searchsorted(split.date)
        if part == IN_SAMPLE:
            m = pos + horizon < split_pos
        else:
            m = (pos >= split_pos) & has_fwd
    return pd.Series(m, index=index)


def summarize(values: pd.Series) -> dict:
    v = values.dropna()
    if v.empty:
        return {"표본수": 0, "평균(%)": np.nan, "중앙값(%)": np.nan, "승률(%)": np.nan,
                "하위10%(%)": np.nan, "상위10%(%)": np.nan}
    return {
        "표본수": int(len(v)),
        "평균(%)": v.mean(),
        "중앙값(%)": v.median(),
        "승률(%)": (v > 0).mean() * 100,
        "하위10%(%)": v.quantile(0.1),
        "상위10%(%)": v.quantile(0.9),
    }


def periods(mask: pd.Series, merge_gap: int = 0) -> pd.DataFrame:
    """조건이 True인 연속 구간. merge_gap 거래일 이하로 끊긴 구간은 하나로 합친다.

    반환: start, end, 거래일(구간 길이), 해당일(구간 안에서 조건이 참인 날 수)
    """
    m = mask.fillna(False).astype(bool).to_numpy()
    cols = ["start", "end", "거래일", "해당일"]
    if not m.any():
        return pd.DataFrame(columns=cols)
    d = np.diff(np.r_[0, m.astype(int), 0])
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    if merge_gap > 0 and len(starts) > 1:
        keep = (starts[1:] - ends[:-1] - 1) > merge_gap
        starts = np.r_[starts[0], starts[1:][keep]]
        ends = np.r_[ends[:-1][keep], ends[-1]]
    csum = np.r_[0, np.cumsum(m)]
    idx = mask.index
    return pd.DataFrame({
        "start": idx[starts],
        "end": idx[ends],
        "거래일": ends - starts + 1,
        "해당일": csum[ends + 1] - csum[starts],
    })


def condition_overview(hit_mask: pd.Series, valid: pd.Series, per: pd.DataFrame) -> dict:
    total = int(valid.sum())
    hit = int(hit_mask.sum())
    return {
        "해당일": hit,
        "비율(%)": hit / total * 100 if total else np.nan,
        "구간 수": len(per),
        "평균 구간 길이(거래일)": float(per["거래일"].mean()) if len(per) else np.nan,
        "최장 구간(거래일)": int(per["거래일"].max()) if len(per) else 0,
    }


GROUP_HIT, GROUP_EVENT, GROUP_MISS, GROUP_BASE = "조건 해당일", "구간 시작일", "조건 외", "전체"


def condition_table(
    close: pd.Series,
    hit: pd.Series,
    valid: pd.Series,
    split: Split,
    merge_gap: int = 5,
    horizons: dict[str, int] = HORIZONS,
) -> pd.DataFrame:
    """기간 × 표본구간 × 그룹별 이후 수익률 요약.

    그룹: 조건 해당일(매일), 구간 시작일(구간마다 첫날 1번), 조건 외, 전체.
    '조건 외'와 '전체'는 조건에 쓰인 지표 값이 있는 날만 센다.
    """
    fwd = forward_returns(close, horizons)
    worst = forward_worst(close, horizons)
    per = periods(hit, merge_gap)
    event = pd.Series(False, index=close.index)
    if len(per):
        event.loc[pd.DatetimeIndex(per["start"])] = True
    groups = {
        GROUP_HIT: hit,
        GROUP_EVENT: event,
        GROUP_MISS: valid & ~hit,
        GROUP_BASE: valid,
    }
    rows = []
    for name, h in horizons.items():
        for part in split.parts():
            sm = sample_mask(close.index, split, part, h)
            for gname, g in groups.items():
                sel = sm & g
                row = {"기간": name, "표본구간": part, "그룹": gname}
                row.update(summarize(fwd[name][sel]))
                row["평균 최대하락(%)"] = worst[name][sel].mean() if sel.any() else np.nan
                rows.append(row)
    return pd.DataFrame(rows)


def period_table(close: pd.Series, per: pd.DataFrame, horizons: dict[str, int] = HORIZONS) -> pd.DataFrame:
    """구간별 상세: 구간 중 수익률, 구간 시작 후 이후 수익률."""
    if per.empty:
        return pd.DataFrame()
    fwd = forward_returns(close, horizons)
    out = per.copy()
    out["구간 중 수익률(%)"] = (close.loc[per["end"]].to_numpy() / close.loc[per["start"]].to_numpy() - 1) * 100
    for name in horizons:
        out[f"시작 후 {name}(%)"] = fwd[name].loc[per["start"]].to_numpy()
    out["start"] = pd.DatetimeIndex(out["start"]).date
    out["end"] = pd.DatetimeIndex(out["end"]).date
    return out.rename(columns={"start": "시작", "end": "종료"})


def rank_ic(score: pd.Series, fwd: pd.Series, sel: pd.Series) -> float:
    """점수와 이후 수익률의 순위상관(스피어만). 1에 가까울수록 점수가 높을 때 수익도 높았다."""
    df = pd.DataFrame({"s": score, "r": fwd})[sel].dropna()
    if len(df) < 30 or df["s"].nunique() < 2:
        return np.nan
    return float(df["s"].rank().corr(df["r"].rank()))


def score_bucket_table(
    score: pd.Series,
    close: pd.Series,
    split: Split,
    n_bins: int = 5,
    horizons: dict[str, int] = HORIZONS,
) -> pd.DataFrame:
    """점수 구간별 이후 수익률."""
    fwd = forward_returns(close, horizons)
    edges = np.linspace(0, 100, n_bins + 1)
    labels = [f"{edges[i]:.0f}~{edges[i + 1]:.0f}" for i in range(n_bins)]
    bucket = pd.cut(score, edges, labels=labels, include_lowest=True)
    rows = []
    for name, h in horizons.items():
        for part in split.parts():
            sm = sample_mask(close.index, split, part, h)
            for lab in labels:
                sel = sm & (bucket == lab)
                row = {"기간": name, "표본구간": part, "점수 구간": lab}
                row.update(summarize(fwd[name][sel]))
                rows.append(row)
    return pd.DataFrame(rows)


def ic_table(
    scores: dict[str, pd.Series],
    close: pd.Series,
    split: Split,
    horizons: dict[str, int] = HORIZONS,
) -> pd.DataFrame:
    """지표 점수별 순위상관. 행=지표, 열=기간·표본구간."""
    fwd = forward_returns(close, horizons)
    rows = {}
    for label, s in scores.items():
        row = {}
        for name, h in horizons.items():
            for part in split.parts():
                sm = sample_mask(close.index, split, part, h)
                col = name if part == ALL else f"{name} {part}"
                row[col] = rank_ic(s, fwd[name], sm)
        rows[label] = row
    return pd.DataFrame.from_dict(rows, orient="index")
