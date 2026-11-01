"""KOSPI '시장 위치' 지표와 종합 점수.

각 지표를 과거 N년 분포 안의 백분위(0~100)로 바꾼 뒤, 방향을 맞춰
'온도'(0 = 침체·저평가, 100 = 과열·고평가)로 통일한다.

- 밸류에이션 : PER, PBR, 배당수익률(역), 일드갭(역)
- 추세       : 60·200일선 이격도, 52주 고저 내 위치, RSI(14)
- 수급       : 외국인·기관·개인 순매수 이동합
- 심리·리스크 : 개인 투자심리 지수, 신용융자/예탁금, 신용융자/시가총액, 실현변동성(역), 원/달러 환율(역)

카테고리 점수 = 해당 지표 온도의 평균, 종합 점수 = 카테고리 점수의 평균 (데이터가 있는 것만).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass(frozen=True)
class Indicator:
    key: str
    name: str
    category: str
    unit: str
    higher_is_hot: bool  # 값이 클수록 과열·고평가 쪽이면 True
    hot_word: str  # 온도가 높을 때의 해석
    cold_word: str  # 온도가 낮을 때의 해석
    desc: str
    scale: float = 1.0  # 표시용 배율 (원 → 조원 등)
    digits: int = 2
    abs_levels: tuple[float, float] | None = None  # 해석을 백분위 대신 절대 기준(낮음, 높음)으로


INDICATORS: list[Indicator] = [
    # 밸류에이션
    Indicator("per", "PER", "밸류에이션", "배", True, "고평가", "저평가",
              "주가 ÷ 주당순이익 (KRX, 후행 12개월)"),
    Indicator("pbr", "PBR", "밸류에이션", "배", True, "고평가", "저평가",
              "주가 ÷ 주당순자산 (KRX)"),
    Indicator("div_yield", "배당수익률", "밸류에이션", "%", False, "고평가", "저평가",
              "높을수록 주가가 싸다는 뜻 (역방향)"),
    Indicator("yield_gap", "일드갭", "밸류에이션", "%p", False, "주식 매력 낮음", "주식 매력 높음",
              "이익수익률(1/PER) − 국고채 금리. 클수록 채권 대비 주식이 싸다 (역방향)"),
    # 추세
    Indicator("ma60_gap", "60일선 이격도", "추세", "%", True, "단기 강세", "단기 약세",
              "KOSPI ÷ 60일 이동평균 − 1"),
    Indicator("ma200_gap", "200일선 이격도", "추세", "%", True, "장기 과열", "장기 약세",
              "KOSPI ÷ 200일 이동평균 − 1. 0 위면 장기 상승 국면"),
    Indicator("pos_52w", "52주 고저 내 위치", "추세", "%", True, "고점권", "저점권",
              "(현재 − 52주 저가) ÷ (52주 고가 − 52주 저가)"),
    Indicator("rsi14", "RSI(14)", "추세", "", True, "과매수", "과매도",
              "14일 상대강도지수. 보통 70 이상 과매수, 30 이하 과매도", digits=1, abs_levels=(30, 70)),
    # 수급
    Indicator("foreign_net_roll", "외국인 순매수(이동합)", "수급", "조원", True, "강한 순매수", "강한 순매도",
              "외국인 순매수 N일 합계", scale=1e-12),
    Indicator("inst_net_roll", "기관 순매수(이동합)", "수급", "조원", True, "강한 순매수", "강한 순매도",
              "기관 순매수 N일 합계", scale=1e-12),
    Indicator("retail_net_roll", "개인 순매수(이동합)", "수급", "조원", True, "강한 순매수", "강한 순매도",
              "개인 순매수 N일 합계", scale=1e-12),
    # 심리·리스크
    Indicator("sentiment", "개인 투자심리 지수", "심리·리스크", "σ", True, "과열", "위축",
              "개인 순매수·거래 비중·예탁금·신용융자 z-score 평균"),
    Indicator("credit_to_deposit", "신용융자/예탁금", "심리·리스크", "%", True, "레버리지 과열", "레버리지 낮음",
              "대기 자금 대비 빚투 규모", digits=1),
    Indicator("credit_to_mcap", "신용융자/시가총액", "심리·리스크", "%", True, "레버리지 과열", "레버리지 낮음",
              "유가증권시장 신용융자 ÷ KOSPI 시가총액", digits=3),
    Indicator("vol20", "20일 변동성", "심리·리스크", "%", False, "안도(낮은 변동성)", "공포(높은 변동성)",
              "일간 수익률 표준편차 × √252. 높을수록 불안 (역방향, VKOSPI 대용)", digits=1),
    Indicator("usdkrw", "원/달러 환율", "심리·리스크", "원", False, "원화 강세", "원화 약세(위험회피)",
              "높을수록 외국인 이탈 압력 (역방향)", digits=1),
]
IND = {i.key: i for i in INDICATORS}
CATEGORIES = ["밸류에이션", "추세", "수급", "심리·리스크"]

ZONES = [(20, "침체·저평가"), (40, "낮음"), (60, "중립"), (80, "높음"), (100.01, "과열·고평가")]


def zone_label(score: float | None) -> str:
    if score is None or pd.isna(score):
        return "-"
    return next(label for upper, label in ZONES if score < upper)


# ---------------------------------------------------------------------------
# 지표 계산
# ---------------------------------------------------------------------------
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI."""
    diff = close.diff()
    gain = diff.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-diff.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(loss != 0, 100.0).where(gain.notna())


def add_position_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """add_indicators() 결과에 시장 위치용 파생 지표를 더한다."""
    out = df.copy()
    if "kospi_close" in out:
        close = out["kospi_close"]
        for w in (20, 60, 120, 200):
            out[f"ma{w}"] = close.rolling(w, min_periods=w).mean()
        out["ma60_gap"] = (close / out["ma60"] - 1) * 100
        out["ma200_gap"] = (close / out["ma200"] - 1) * 100
        high = (out["kospi_high"] if "kospi_high" in out else close).fillna(close)
        low = (out["kospi_low"] if "kospi_low" in out else close).fillna(close)
        out["high_52w"] = high.rolling(TRADING_DAYS, min_periods=TRADING_DAYS // 2).max()
        out["low_52w"] = low.rolling(TRADING_DAYS, min_periods=TRADING_DAYS // 2).min()
        rng = (out["high_52w"] - out["low_52w"]).replace(0, np.nan)
        out["pos_52w"] = (close - out["low_52w"]) / rng * 100
        out["drawdown"] = (close / close.cummax() - 1) * 100
        out["rsi14"] = rsi(close)
        logret = np.log(close).diff()
        out["vol20"] = logret.rolling(20, min_periods=15).std() * np.sqrt(TRADING_DAYS) * 100
    if "per" in out:
        bond = None
        for col in ("bond_10y", "bond_3y"):
            if col in out and out[col].notna().any():
                bond = out[col]
                break
        if bond is not None:
            out["yield_gap"] = 100 / out["per"] - bond
    return out


def rolling_percentile(s: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """각 시점 값이 직전 window 개(자기 포함) 값 중 몇 % 위치인지 (0~100, 동률은 절반)."""
    min_periods = min_periods or min(window, TRADING_DAYS)
    valid = s.dropna()
    if len(valid) < min_periods:
        return pd.Series(np.nan, index=s.index)

    def pct(a: np.ndarray) -> float:
        last, hist = a[-1], a[:-1]
        if len(hist) == 0:
            return np.nan
        return ((hist < last).sum() + 0.5 * (hist == last).sum()) / len(hist) * 100

    out = valid.rolling(window, min_periods=min_periods).apply(pct, raw=True)
    return out.reindex(s.index).ffill(limit=5)


def position_scores(df: pd.DataFrame, lookback_years: float = 5) -> pd.DataFrame:
    """지표별 온도(heat_<key>), 백분위(pct_<key>), 카테고리 점수, 종합 점수(overall)."""
    window = int(lookback_years * TRADING_DAYS)
    out = pd.DataFrame(index=df.index)
    for ind in INDICATORS:
        if ind.key not in df or df[ind.key].notna().sum() < TRADING_DAYS // 2:
            continue
        pct = rolling_percentile(df[ind.key], window, min_periods=min(window, TRADING_DAYS // 2))
        out[f"pct_{ind.key}"] = pct
        out[f"heat_{ind.key}"] = pct if ind.higher_is_hot else 100 - pct
    for cat in CATEGORIES:
        cols = [f"heat_{i.key}" for i in INDICATORS if i.category == cat and f"heat_{i.key}" in out]
        if cols:
            out[cat] = out[cols].mean(axis=1, skipna=True)
    cats = [c for c in CATEGORIES if c in out]
    out["overall"] = out[cats].mean(axis=1, skipna=True) if cats else np.nan
    return out


def snapshot(df: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """지표별 최신 값·기준일·백분위·온도·해석 표."""
    rows = []
    for ind in INDICATORS:
        hcol = f"heat_{ind.key}"
        if hcol not in scores:
            continue
        s = df[ind.key].dropna()
        h = scores[hcol].dropna()
        if s.empty or h.empty:
            continue
        heat = float(h.iloc[-1])
        rows.append(
            {
                "key": ind.key,
                "카테고리": ind.category,
                "지표": ind.name,
                "현재값": float(s.iloc[-1]) * ind.scale,
                "단위": ind.unit,
                "기준일": s.index[-1],
                "백분위": float(scores[f"pct_{ind.key}"].dropna().iloc[-1]),
                "온도": heat,
                "해석": interpret(ind, float(s.iloc[-1]), heat),
                "설명": ind.desc,
                "digits": ind.digits,
            }
        )
    return pd.DataFrame(rows)


def interpret(ind: Indicator, value: float, heat: float) -> str:
    if ind.abs_levels:
        lo, hi = ind.abs_levels
        return ind.hot_word if value >= hi else ind.cold_word if value <= lo else "보통"
    if 40 < heat < 60:
        return "보통"
    word = ind.hot_word if heat >= 60 else ind.cold_word
    return "매우 " + word if heat >= 80 or heat <= 20 else word


def summary(scores: pd.DataFrame) -> dict:
    """최신 종합·카테고리 점수와 1개월(20거래일) 전 대비 변화."""
    out = {}
    for col in ["overall", *CATEGORIES]:
        if col not in scores:
            continue
        s = scores[col].dropna()
        if s.empty:
            continue
        out[col] = {
            "score": float(s.iloc[-1]),
            "date": s.index[-1],
            "change_1m": float(s.iloc[-1] - s.iloc[-21]) if len(s) > 21 else None,
            "label": zone_label(s.iloc[-1]),
        }
    return out


def price_levels(df: pd.DataFrame) -> pd.DataFrame:
    """주요 가격 레벨과 현재가 대비 거리."""
    if "kospi_close" not in df or df["kospi_close"].dropna().empty:
        return pd.DataFrame()
    last = df.dropna(subset=["kospi_close"]).iloc[-1]
    close = last["kospi_close"]
    items = [("현재 KOSPI", close)]
    items += [(f"{w}일 이동평균", last.get(f"ma{w}")) for w in (20, 60, 120, 200)]
    items += [("52주 최고", last.get("high_52w")), ("52주 최저", last.get("low_52w")),
              ("불러온 기간 최고(종가)", df["kospi_close"].max())]
    rows = [
        {"구분": name, "지수": float(v), "현재가 대비(%)": (close / v - 1) * 100}
        for name, v in items
        if v is not None and not pd.isna(v)
    ]
    table = pd.DataFrame(rows)
    ma = [last.get(f"ma{w}") for w in (20, 60, 120, 200)]
    if all(m is not None and not pd.isna(m) for m in ma):
        aligned = "정배열(상승 추세)" if ma[0] > ma[1] > ma[2] > ma[3] else \
            "역배열(하락 추세)" if ma[0] < ma[1] < ma[2] < ma[3] else "혼조"
        table.attrs["alignment"] = aligned
    table.attrs["drawdown"] = float(last.get("drawdown", np.nan))
    return table


def build_position(df: pd.DataFrame, lookback_years: float = 5):
    """add_indicators() 결과를 받아 (지표 포함 df, 점수, 스냅샷, 요약, 가격 레벨)을 돌려준다."""
    full = add_position_indicators(df)
    scores = position_scores(full, lookback_years)
    return full, scores, snapshot(full, scores), summary(scores), price_levels(full)
