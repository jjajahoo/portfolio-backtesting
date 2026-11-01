"""KOSPI 가상 가격 시뮬레이션.

사용자가 정한 가상 KOSPI 수준(과 그 수준까지 가는 경로)을 실제 가격 이력 뒤에 붙여
기술적 지표와 밸류에이션이 어떻게 바뀌는지 계산한다. 화면(Streamlit)과 분리된 순수 계산 모듈.

입력 가격표 컬럼: close (필수), open / high / low (있으면 사용)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

MA_WINDOWS = (5, 20, 60, 120)
MA_CROSS_PAIRS = ((5, 20), (20, 60), (60, 120))
RSI_N = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
BB_N, BB_K = 20, 2.0
STOCH_N, STOCH_SMOOTH = 14, 3
PSY_N = 10
YEAR_DAYS = 250  # 52주 ≈ 250 거래일

PRICE_COLS = ["open", "high", "low", "close"]


# ---------------------------------------------------------------------------
# 기술적 지표
# ---------------------------------------------------------------------------
def _alpha(span: int) -> float:
    return 2.0 / (span + 1)


def add_technicals(prices: pd.DataFrame) -> pd.DataFrame:
    """가격표에 기술적 지표 컬럼을 더해 돌려준다. 계산식은 국내 HTS 기본값을 따른다."""
    out = prices.copy()
    c = out["close"].astype(float)
    h = out["high"].fillna(c) if "high" in out else c
    lo = out["low"].fillna(c) if "low" in out else c

    for n in MA_WINDOWS:
        out[f"ma{n}"] = c.rolling(n).mean()
        out[f"disp{n}"] = c / out[f"ma{n}"] * 100  # 이격도

    # RSI (Wilder 평활). 다음 날 가격 역산에 쓰려고 평균 상승·하락폭도 남긴다.
    d = c.diff()
    a = 1.0 / RSI_N
    out["rsi_up"] = d.clip(lower=0).ewm(alpha=a, adjust=False, min_periods=RSI_N).mean()
    out["rsi_down"] = (-d).clip(lower=0).ewm(alpha=a, adjust=False, min_periods=RSI_N).mean()
    total = out["rsi_up"] + out["rsi_down"]
    out["rsi"] = (100 * out["rsi_up"] / total).where(total > 0, 50.0).where(total.notna())

    out["ema_fast"] = c.ewm(span=MACD_FAST, adjust=False).mean()
    out["ema_slow"] = c.ewm(span=MACD_SLOW, adjust=False).mean()
    out["macd"] = out["ema_fast"] - out["ema_slow"]
    out["macd_signal"] = out["macd"].ewm(span=MACD_SIGNAL, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    mid = c.rolling(BB_N).mean()
    sd = c.rolling(BB_N).std(ddof=0)
    out["bb_mid"], out["bb_up"], out["bb_low"] = mid, mid + BB_K * sd, mid - BB_K * sd
    width = out["bb_up"] - out["bb_low"]
    out["bb_pctb"] = (c - out["bb_low"]) / width.where(width > 0)
    out["bb_width"] = width / mid * 100

    ll, hh = lo.rolling(STOCH_N).min(), h.rolling(STOCH_N).max()
    fast_k = (c - ll) / (hh - ll).where(hh > ll) * 100
    out["stoch_k"] = fast_k.rolling(STOCH_SMOOTH).mean()
    out["stoch_d"] = out["stoch_k"].rolling(STOCH_SMOOTH).mean()

    out["psy"] = (d > 0).astype(float).where(d.notna()).rolling(PSY_N).mean() * 100  # 투자심리선

    out["high52"] = h.rolling(YEAR_DAYS, min_periods=1).max()
    out["low52"] = lo.rolling(YEAR_DAYS, min_periods=1).min()
    out["from_high"] = (c / out["high52"] - 1) * 100
    return out


def daily_volatility(close: pd.Series, window: int = 60) -> float:
    """최근 window 거래일 일간 로그수익률 표준편차."""
    r = np.log(close.astype(float)).diff().dropna().tail(window)
    return float(r.std()) if len(r) > 5 else 0.01


# ---------------------------------------------------------------------------
# 가상 경로
# ---------------------------------------------------------------------------
def make_path(
    last_close: float,
    target: float,
    days: int,
    last_date,
    daily_vol: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """마지막 종가에서 target 까지 days 거래일 동안 가는 가상 OHLC.

    daily_vol = 0 이면 매일 같은 비율로 움직이는 직선 경로,
    daily_vol > 0 이면 끝점이 target 으로 고정된 무작위 경로(브라운 브리지)를 만든다.
    날짜는 주말만 건너뛴 가상 거래일이다 (공휴일 미반영).
    """
    days = max(1, int(days))
    t = np.arange(1, days + 1) / days
    log_close = math.log(last_close) + t * math.log(target / last_close)
    rng = np.random.default_rng(seed)
    if daily_vol > 0 and days > 1:
        walk = np.cumsum(rng.normal(0, daily_vol, days))
        log_close = log_close + walk - t * walk[-1]
    close = np.exp(log_close)
    close[-1] = target  # 부동소수 오차 제거
    open_ = np.r_[last_close, close[:-1]]
    high, low = np.maximum(open_, close), np.minimum(open_, close)
    if daily_vol > 0:  # 장중 꼬리
        wick = np.abs(rng.normal(0, daily_vol * 0.5, (2, days)))
        high, low = high * np.exp(wick[0]), low * np.exp(-wick[1])
    idx = pd.bdate_range(pd.Timestamp(last_date) + pd.Timedelta(days=1), periods=days, name="date")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


def simulate(
    prices: pd.DataFrame, target: float, days: int, daily_vol: float = 0.0, seed: int = 0
) -> pd.DataFrame:
    """실제 가격 뒤에 가상 경로를 붙이고 지표를 다시 계산한다. hypo 컬럼이 True 인 행이 가상 구간."""
    base = prices[[c for c in PRICE_COLS if c in prices]].astype(float)
    path = make_path(base["close"].iloc[-1], target, days, base.index[-1], daily_vol, seed)
    combined = pd.concat([base.assign(hypo=False), path.assign(hypo=True)])
    combined.index.name = "date"
    return add_technicals(combined)


# ---------------------------------------------------------------------------
# 지표 비교·해석
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Indicator:
    key: str
    label: str
    fmt: str
    unit: str = ""


INDICATORS = [
    Indicator("close", "KOSPI 종가", "{:,.2f}"),
    *[Indicator(f"ma{n}", f"{n}일 이동평균", "{:,.2f}") for n in MA_WINDOWS],
    *[Indicator(f"disp{n}", f"{n}일 이격도", "{:.1f}", "%") for n in MA_WINDOWS],
    Indicator("rsi", f"RSI({RSI_N})", "{:.1f}"),
    Indicator("macd", f"MACD({MACD_FAST},{MACD_SLOW})", "{:+.2f}"),
    Indicator("macd_signal", f"MACD 시그널({MACD_SIGNAL})", "{:+.2f}"),
    Indicator("macd_hist", "MACD 오실레이터", "{:+.2f}"),
    Indicator("bb_up", f"볼린저 상단({BB_N},{BB_K:g})", "{:,.2f}"),
    Indicator("bb_low", "볼린저 하단", "{:,.2f}"),
    Indicator("bb_pctb", "볼린저 %B", "{:.2f}"),
    Indicator("bb_width", "볼린저 밴드폭", "{:.1f}", "%"),
    Indicator("stoch_k", "스토캐스틱 %K", "{:.1f}"),
    Indicator("stoch_d", "스토캐스틱 %D", "{:.1f}"),
    Indicator("psy", f"투자심리선({PSY_N})", "{:.0f}", "%"),
    Indicator("high52", "52주 최고", "{:,.2f}"),
    Indicator("from_high", "52주 최고 대비", "{:+.1f}", "%"),
]


def describe(key: str, row: pd.Series) -> str:
    """지표 값의 일반적인 해석. 기준선은 널리 쓰이는 관례값."""
    v = row.get(key, np.nan)
    if v is None or pd.isna(v):
        return ""
    c = row["close"]
    if key.startswith("ma") and key[2:].isdigit():
        return "종가가 위" if c >= v else "종가가 아래"
    if key.startswith("disp"):
        return "평균보다 위" if v >= 100 else "평균보다 아래"
    if key == "rsi":
        return "과매수(70↑)" if v >= 70 else "과매도(30↓)" if v <= 30 else "중립"
    if key == "macd":
        return "0선 위" if v >= 0 else "0선 아래"
    if key == "macd_hist":
        return "시그널 위" if v >= 0 else "시그널 아래"
    if key == "bb_pctb":
        return "상단 돌파" if v > 1 else "하단 이탈" if v < 0 else "밴드 안"
    if key == "stoch_k":
        return "과매수(80↑)" if v >= 80 else "과매도(20↓)" if v <= 20 else "중립"
    if key == "psy":
        return "과열(75↑)" if v >= 75 else "침체(25↓)" if v <= 25 else "중립"
    if key == "from_high":
        if v >= -0.005:
            return "52주 신고가"
        return "약세장(-20%↓)" if v <= -20 else "조정(-10%↓)" if v <= -10 else "고점 부근"
    return ""


def _fmt(ind: Indicator, v) -> str:
    return "–" if v is None or pd.isna(v) else ind.fmt.format(v) + ind.unit


def compare_table(current: pd.Series, simulated: pd.Series) -> pd.DataFrame:
    rows = []
    for ind in INDICATORS:
        a, b = current.get(ind.key, np.nan), simulated.get(ind.key, np.nan)
        diff = b - a if pd.notna(a) and pd.notna(b) else np.nan
        rows.append(
            {
                "지표": ind.label,
                "현재": _fmt(ind, a),
                "가상": _fmt(ind, b),
                "변화": "–" if pd.isna(diff) else ind.fmt.replace("{:", "{:+").replace("++", "+").format(diff),
                "현재 상태": describe(ind.key, current),
                "가상 상태": describe(ind.key, simulated),
            }
        )
    return pd.DataFrame(rows)


def path_events(sim: pd.DataFrame) -> list[tuple[pd.Timestamp, str]]:
    """가상 구간에서 새로 생기는 신호(교차·돌파)를 날짜 순으로 찾는다."""
    if not sim["hypo"].any():
        return []
    first = int(np.argmax(sim["hypo"].to_numpy()))
    seg = sim.iloc[max(first - 1, 0):]
    events: list[tuple[pd.Timestamp, str]] = []

    def crossings(s: pd.Series, up_text: str, down_text: str, level: float = 0.0):
        sign = np.sign(s - level)
        prev = sign.shift(1)
        for dt in seg.index[1:]:
            if pd.isna(prev[dt]) or pd.isna(sign[dt]) or sign[dt] == prev[dt] or sign[dt] == 0:
                continue
            events.append((dt, up_text if sign[dt] > 0 else down_text))

    for a, b in MA_CROSS_PAIRS:
        crossings(seg[f"ma{a}"] - seg[f"ma{b}"],
                  f"골든크로스: {a}일선이 {b}일선 위로", f"데드크로스: {a}일선이 {b}일선 아래로")
    crossings(seg["close"] - seg["ma120"], "120일선 회복", "120일선 이탈")
    crossings(seg["macd_hist"], "MACD 시그널 상향 교차", "MACD 시그널 하향 교차")
    crossings(seg["macd"], "MACD 0선 상향 돌파", "MACD 0선 하향 이탈")
    crossings(seg["rsi"], "RSI 70 돌파 (과매수 진입)", "RSI 70 아래로 (과매수 해소)", 70)
    crossings(seg["rsi"], "RSI 30 위로 (과매도 탈출)", "RSI 30 이탈 (과매도 진입)", 30)
    crossings(seg["from_high"], "", "52주 고점 대비 -10% (조정 국면)", -10)
    crossings(seg["from_high"], "", "52주 고점 대비 -20% (약세장 기준)", -20)

    prev_high = seg["high52"].shift(1)
    for dt in seg.index[1:]:
        if seg.at[dt, "close"] > prev_high[dt]:
            events.append((dt, "52주 신고가 경신"))
            break  # 첫 경신일만 표시
    events = [(d, t) for d, t in events if t]
    return sorted(events, key=lambda e: e[0])


# ---------------------------------------------------------------------------
# 다음 거래일 종가 역산 (주요 가격대)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Level:
    price: float
    label: str
    meaning: str


def ma_touch_price(closes: np.ndarray, n: int) -> float:
    """다음 종가 P 가 P 를 포함한 n일 이동평균과 같아지는 가격 = 직전 n-1개 종가 평균."""
    return float(np.mean(closes[-(n - 1):]))


def ma_cross_price(closes: np.ndarray, a: int, b: int) -> float:
    """다음 종가 P 에서 a일선과 b일선이 같아지는 가격 (a < b)."""
    sa, sb = closes[-(a - 1):].sum(), closes[-(b - 1):].sum()
    return float((sb / b - sa / a) / (1 / a - 1 / b))


def rsi_target_price(last: float, avg_up: float, avg_down: float, target: float) -> float | None:
    """다음 날 RSI 가 target 이 되는 종가. RSI 는 다음 종가에 대해 단조증가한다."""
    a = 1.0 / RSI_N
    up, dn = (1 - a) * avg_up, (1 - a) * avg_down
    if not (0 < target < 100) or up + dn <= 0:
        return None
    rs = target / (100 - target)
    current = 100 * avg_up / (avg_up + avg_down)
    if target >= current:  # 올라야 도달: (up + a·d) / dn = rs
        d = (rs * dn - up) / a
        return last + d if d >= 0 else None
    d = (up / rs - dn) / a  # 내려야 도달: up / (dn + a·d) = rs
    return last - d if d >= 0 else None


def macd_target_prices(ema_fast: float, ema_slow: float, signal: float) -> tuple[float, float]:
    """다음 종가 P 에 대해 MACD 가 선형이므로 (0선 교차 가격, 시그널 교차 가격)을 바로 푼다."""
    af, aslow = _alpha(MACD_FAST), _alpha(MACD_SLOW)
    const = (1 - af) * ema_fast - (1 - aslow) * ema_slow  # MACD(P) = const + (af - aslow)·P
    slope = af - aslow
    return float(-const / slope), float((signal - const) / slope)


def bollinger_touch_prices(closes: np.ndarray, n: int = BB_N, k: float = BB_K) -> tuple[float, float]:
    """다음 종가가 그 종가를 포함한 볼린저 상단/하단과 같아지는 가격 (2차방정식의 두 근)."""
    prev = closes[-(n - 1):].astype(float)
    s, q = prev.sum(), (prev ** 2).sum()
    a = (n - 1) ** 2 - k * k * (n - 1)
    b = -2 * (n - 1) * s + 2 * k * k * s
    c = s * s - k * k * (n * q - s * s)
    disc = max(b * b - 4 * a * c, 0.0)
    r1, r2 = (-b + math.sqrt(disc)) / (2 * a), (-b - math.sqrt(disc)) / (2 * a)
    return max(r1, r2), min(r1, r2)


def next_day_levels(ind: pd.DataFrame, max_move: float = 0.3) -> list[Level]:
    """다음 거래일 하루 만에 이 종가로 마감하면 어떤 신호가 생기는지, 가격대별로 정리한다.

    ind: add_technicals 결과 (실제 데이터만). 현재가 ±max_move 범위 밖의 가격은 뺀다.
    """
    closes = ind["close"].to_numpy(dtype=float)
    last = closes[-1]
    row = ind.iloc[-1]
    levels: list[Level] = []

    for n in MA_WINDOWS:
        if len(closes) >= n:
            p = ma_touch_price(closes, n)
            above = last >= row[f"ma{n}"]
            levels.append(Level(p, f"{n}일선", f"이 가격 {'아래' if above else '위'}로 마감하면 {n}일선 "
                                                + ("이탈" if above else "회복")))
    for a, b in MA_CROSS_PAIRS:
        if len(closes) >= b:
            p = ma_cross_price(closes, a, b)
            if row[f"ma{a}"] >= row[f"ma{b}"]:
                levels.append(Level(p, f"{a}·{b}일선 데드크로스", f"이 가격 아래로 마감하면 {a}일선이 {b}일선 밑으로"))
            else:
                levels.append(Level(p, f"{a}·{b}일선 골든크로스", f"이 가격 위로 마감하면 {a}일선이 {b}일선 위로"))
    if pd.notna(row["rsi_up"]):
        for target, text in ((70, "RSI 70 (과매수 기준)"), (30, "RSI 30 (과매도 기준)")):
            p = rsi_target_price(last, row["rsi_up"], row["rsi_down"], target)
            if p is not None:
                side = "위" if target == 70 else "아래"
                levels.append(Level(p, text, f"이 가격 {side}로 마감하면 RSI {target} {'초과' if target == 70 else '미만'}"))
    zero_p, sig_p = macd_target_prices(row["ema_fast"], row["ema_slow"], row["macd_signal"])
    hist_up = row["macd_hist"] >= 0
    levels.append(Level(sig_p, "MACD 시그널 교차",
                        f"이 가격 {'아래' if hist_up else '위'}로 마감하면 MACD가 시그널 {'밑으로' if hist_up else '위로'}"))
    macd_up = row["macd"] >= 0
    levels.append(Level(zero_p, "MACD 0선",
                        f"이 가격 {'아래' if macd_up else '위'}로 마감하면 MACD가 0선 {'밑으로' if macd_up else '위로'}"))
    if len(closes) >= BB_N:
        up, low = bollinger_touch_prices(closes)
        levels.append(Level(up, "볼린저 상단", "이 가격 위로 마감하면 밴드 상단 돌파"))
        levels.append(Level(low, "볼린저 하단", "이 가격 아래로 마감하면 밴드 하단 이탈"))
    hi = row["high52"]
    levels.append(Level(hi, "52주 최고가", "이 가격 위로 마감하면 52주 신고가"))
    levels.append(Level(hi * 0.9, "고점 대비 -10%", "이 가격 아래면 조정 국면 (흔히 쓰는 기준)"))
    levels.append(Level(hi * 0.8, "고점 대비 -20%", "이 가격 아래면 약세장 (흔히 쓰는 기준)"))

    lo, hi_lim = last * (1 - max_move), last * (1 + max_move)
    levels = [lv for lv in levels if lo <= lv.price <= hi_lim and math.isfinite(lv.price)]
    return sorted(levels, key=lambda lv: -lv.price)


# ---------------------------------------------------------------------------
# 밸류에이션
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Fundamentals:
    """지수 1포인트당 이익·순자산·배당. 가격이 바뀌어도 이 값들은 그대로라고 가정한다."""

    ref_close: float
    eps: float | None = None  # PER 역산 (후행)
    bps: float | None = None
    dps: float | None = None
    fwd_eps: float | None = None
    mcap_per_point: float | None = None  # 원 / pt
    asof: pd.Timestamp | None = None
    source: str = ""

    @classmethod
    def from_ratios(cls, close, per=None, pbr=None, div_yield=None, fwd_per=None, mcap=None,
                    asof=None, source=""):
        def inv(ratio):
            return close / ratio if ratio and ratio > 0 else None

        return cls(
            ref_close=float(close), eps=inv(per), bps=inv(pbr),
            dps=close * div_yield / 100 if div_yield and div_yield > 0 else None,
            fwd_eps=inv(fwd_per), mcap_per_point=mcap / close if mcap and mcap > 0 else None,
            asof=asof, source=source,
        )

    @property
    def roe(self) -> float | None:
        return self.eps / self.bps * 100 if self.eps and self.bps else None


VALUATION_ROWS = {
    "level": ("KOSPI", "{:,.2f}", ""),
    "per": ("PER (후행)", "{:.2f}", "배"),
    "fwd_per": ("PER (선행, 직접 입력)", "{:.2f}", "배"),
    "pbr": ("PBR", "{:.2f}", "배"),
    "div_yield": ("배당수익률", "{:.2f}", "%"),
    "earnings_yield": ("이익수익률 (1/PER)", "{:.2f}", "%"),
    "yield_gap": ("일드갭 (vs 국고채 10년)", "{:+.2f}", "%p"),
    "mcap": ("시가총액 (추정)", "{:,.0f}", "조원"),
    "buffett": ("시가총액 / 명목 GDP", "{:.0f}", "%"),
}


def valuation_at(level: float, f: Fundamentals, bond_yield: float | None = None,
                 gdp: float | None = None) -> dict[str, float]:
    """bond_yield: %, gdp: 원."""
    v: dict[str, float] = {"level": level}
    if f.eps:
        v["per"] = level / f.eps
        v["earnings_yield"] = f.eps / level * 100
        if bond_yield is not None:
            v["yield_gap"] = v["earnings_yield"] - bond_yield
    if f.fwd_eps:
        v["fwd_per"] = level / f.fwd_eps
    if f.bps:
        v["pbr"] = level / f.bps
    if f.dps:
        v["div_yield"] = f.dps / level * 100
    if f.mcap_per_point:
        mcap = f.mcap_per_point * level
        v["mcap"] = mcap / 1e12
        if gdp:
            v["buffett"] = mcap / gdp * 100
    return v


def level_for(f: Fundamentals, metric: str, value: float) -> float | None:
    """목표 배수(또는 수익률)에 해당하는 KOSPI 수준."""
    if value is None or value <= 0:
        return None
    if metric == "per" and f.eps:
        return value * f.eps
    if metric == "fwd_per" and f.fwd_eps:
        return value * f.fwd_eps
    if metric == "pbr" and f.bps:
        return value * f.bps
    if metric == "div_yield" and f.dps:
        return f.dps / value * 100
    return None


def percentile_of(history: pd.Series, value: float) -> float | None:
    """과거 값 중 value 보다 낮았던 날의 비율(%)."""
    h = history.dropna()
    if len(h) < 20 or value is None or pd.isna(value):
        return None
    return float((h < value).mean() * 100)


def implied_per_point(data: pd.DataFrame) -> pd.DataFrame:
    """KRX 의 PER·PBR·배당수익률 이력에서 1포인트당 EPS·BPS·DPS 를 역산한다."""
    out = pd.DataFrame(index=data.index)
    c = data["close"]
    if "per" in data:
        out["eps"] = c / data["per"]
    if "pbr" in data:
        out["bps"] = c / data["pbr"]
    if "div_yield" in data:
        out["dps"] = c * data["div_yield"] / 100
    return out


def nice_multiples(center: float, history: pd.Series | None = None, count: int = 5) -> list[float]:
    """밴드 차트용 '깔끔한' 배수 목록. 과거 분포가 있으면 5~95% 구간을, 없으면 현재값 ±30% 를 덮는다."""
    h = history.dropna() if history is not None else pd.Series(dtype=float)
    lo, hi = (h.quantile(0.05), h.quantile(0.95)) if len(h) >= 20 else (center * 0.7, center * 1.3)
    lo, hi = min(lo, center), max(hi, center)
    raw = (hi - lo) / max(count - 1, 1)
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 0.1
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    first, last = math.floor(lo / step + 1e-9), math.ceil(hi / step - 1e-9)  # 구간을 바깥쪽으로 덮는다
    return [round(i * step, 4) for i in range(first, last + 1) if i * step > 0]


# ---------------------------------------------------------------------------
# 데모 데이터
# ---------------------------------------------------------------------------
def make_demo_market(start="2016-01-01", end=None, seed: int = 11) -> pd.DataFrame:
    """네트워크 없이 화면을 보기 위한 가상 KOSPI·PER·PBR·배당수익률·시가총액 (실제 데이터 아님)."""
    from indicators import make_demo_data

    px = make_demo_data(start, end, seed=seed)
    n = len(px)
    rng = np.random.default_rng(seed)
    years = np.arange(n) / YEAR_DAYS
    bps = 2300 * np.exp(0.055 * years)  # 순자산은 꾸준히 증가
    roe = 0.085 + 0.025 * np.sin(np.linspace(0, 3 * np.pi, n)) + rng.normal(0, 0.0005, n).cumsum() / 50
    eps = bps * np.clip(roe, 0.03, None)
    # KRX 지수 PER·PBR 은 실적 공시 시점에만 바뀌므로 분기별 계단으로 만든다.
    quarter = pd.Series(np.arange(n) // 63, index=px.index)
    bps = pd.Series(bps, index=px.index).groupby(quarter).transform("first")
    eps = pd.Series(eps, index=px.index).groupby(quarter).transform("first")
    close = px["kospi_close"]
    return pd.DataFrame(
        {
            "open": px["kospi_open"],
            "high": px["kospi_high"],
            "low": px["kospi_low"],
            "close": close,
            "mcap": close * 7.9e11,
            "per": close / eps,
            "pbr": close / bps,
            "div_yield": eps * 0.3 / close * 100,
        }
    )
