"""KOSPI 가격 시뮬레이터

KOSPI 를 원하는 수준으로 옮겨 보면 PER·PBR 같은 밸류에이션과
이동평균·RSI·MACD 같은 기술적 지표가 어떻게 바뀌는지 보여 준다.

실행:  streamlit run kospi_retail_dashboard/simulator.py
"""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

import simulation as sim  # noqa: E402
from data_sources import load_kospi_valuation, set_krx_credentials  # noqa: E402
from sim_charts import build_band_chart, build_technical_chart  # noqa: E402

st.set_page_config(page_title="KOSPI 가격 시뮬레이터", page_icon="🎚️", layout="wide")

SOURCE_LIVE = "실제 데이터 (KRX·네이버)"
SOURCE_DEMO = "데모 (가상 데이터)"
HISTORY_YEARS = {"3년": 3, "5년": 5, "10년": 10}
PRESETS = (-20, -10, -5, 5, 10, 20)
PATH_LINE = "일정한 비율로"
PATH_RANDOM = "변동성 반영 (무작위)"
METRIC_NAMES = {"pbr": "PBR", "per": "PER", "fwd_per": "선행 PER", "div_yield": "배당수익률"}
METRIC_UNITS = {"pbr": "배", "per": "배", "fwd_per": "배", "div_yield": "%"}

ss = st.session_state


def _secret(key: str) -> str:
    try:
        return st.secrets.get(key, "") or os.getenv(key, "")
    except Exception:  # secrets.toml 이 없을 때
        return os.getenv(key, "")


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def cached_live(start: pd.Timestamp, end: pd.Timestamp, use_krx: bool, krx_id: str):
    # krx_id 는 계정이 바뀌면 캐시를 새로 만들기 위한 키로만 쓴다.
    df, report = load_kospi_valuation(start, end, use_krx)
    return df, report.items


@st.cache_data(show_spinner=False)
def cached_demo(start: pd.Timestamp, end: pd.Timestamp):
    return sim.make_demo_market("2010-01-01", end).loc[start:end]


# 가상 KOSPI 입력: 지수(pt)와 변동률(%)을 서로 맞춰 준다. 콜백은 다음 실행 전에 돌아서 위젯 값을 바꿔도 안전하다.
def _from_pct(base: float) -> None:
    ss.target_pt = round(base * (1 + ss.target_pct / 100), 2)


def _from_pt(base: float) -> None:
    ss.target_pct = float(np.clip(round((ss.target_pt / base - 1) * 100, 1), -50.0, 50.0))


def _set_pct(base: float, pct: float) -> None:
    ss.target_pct = float(pct)
    _from_pct(base)


def _set_level(base: float, level: float, next_day: bool = False) -> None:
    ss.target_pt = round(float(np.clip(level, base * 0.3, base * 3)), 2)
    _from_pt(base)
    if next_day:
        ss.days = 1


def _new_seed() -> None:
    ss.seed = ss.get("seed", 0) + 1


def _fmt(value, fmt: str, unit: str = "") -> str:
    return "–" if value is None or pd.isna(value) else fmt.format(value) + unit


# ---------------------------------------------------------------------------
# 사이드바: 데이터
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("설정")
    source = st.radio("데이터", [SOURCE_LIVE, SOURCE_DEMO])
    years = HISTORY_YEARS[st.selectbox("비교할 과거 기간", list(HISTORY_YEARS), index=1,
                                       help="PER·PBR이 과거 대비 어디쯤인지 계산하고 밴드 차트를 그릴 기간")]
    use_krx, krx_id = False, ""
    if source == SOURCE_LIVE:
        with st.expander("KRX 로그인 (PER·PBR 이력)", expanded=not _secret("KRX_ID")):
            st.caption(
                "KRX 정보데이터시스템(data.krx.co.kr) 계정이 있으면 KOSPI의 PER·PBR·배당수익률 이력과 "
                "시가총액을 자동으로 가져옵니다. 없으면 가격은 네이버에서 받고, 밸류에이션 값은 아래에 직접 입력합니다."
            )
            krx_id = st.text_input("KRX ID", value=_secret("KRX_ID"))
            krx_pw = st.text_input("KRX 비밀번호", value=_secret("KRX_PW"), type="password")
        use_krx = set_krx_credentials(krx_id, krx_pw)
        if st.button("데이터 새로고침", width="stretch"):
            cached_live.clear()


# ---------------------------------------------------------------------------
# 데이터 불러오기
# ---------------------------------------------------------------------------
st.title("KOSPI 가격 시뮬레이터")

today = pd.Timestamp.today().normalize()
# 52주 고점·120일선 등이 비교 기간 첫날부터 계산되도록 1년 남짓 더 받는다.
start = today - pd.Timedelta(days=round(365.25 * years) + 420)
report_items: list[tuple[str, bool, str]] = []
if source == SOURCE_DEMO:
    data = cached_demo(start, today)
    st.warning("데모 모드: 화면 구성을 보기 위한 **가상 데이터**입니다. 실제 KOSPI가 아닙니다.", icon="⚠️")
else:
    with st.spinner("KOSPI 데이터를 불러오는 중… (처음 한 번은 오래 걸릴 수 있습니다)"):
        data, report_items = cached_live(start, today, use_krx, krx_id if use_krx else "")

if data is None or data.empty or "close" not in data or data["close"].dropna().empty:
    st.error("KOSPI 가격을 불러오지 못했습니다. 아래 소스 상태를 확인하거나 사이드바에서 데모 모드를 선택해 보세요.")
    for name, ok, msg in report_items:
        st.write(("✅ " if ok else "❌ ") + f"**{name}** — {msg}")
    st.stop()

data = data[data["close"].notna()]
if report_items:
    failed = [i for i in report_items if not i[1]]
    with st.expander("데이터 소스 상태" + (f" — {len(failed)}개 항목은 직접 입력값 사용" if failed else " — 모두 정상")):
        for name, ok, msg in report_items:
            st.write(("✅ " if ok else "⚠️ ") + f"**{name}** — {msg}")

last_date = data.index[-1]
last_close = float(data["close"].iloc[-1])
hist_from = last_date - pd.Timedelta(days=round(365.25 * years))
hist = data.loc[hist_from:]
has_fund = all(c in data and data[c].notna().any() for c in ("per", "pbr"))


# ---------------------------------------------------------------------------
# 사이드바: 밸류에이션 가정
# ---------------------------------------------------------------------------
with st.sidebar:
    st.divider()
    st.subheader("밸류에이션 가정")
    if has_fund:
        fund = data.reindex(columns=["close", "per", "pbr", "div_yield"]).dropna(subset=["per", "pbr"])
        fund_date, fund_row = fund.index[-1], fund.iloc[-1]
        f = sim.Fundamentals.from_ratios(
            fund_row["close"], fund_row["per"], fund_row["pbr"], fund_row.get("div_yield"),
            asof=fund_date, source="KRX" if source == SOURCE_LIVE else "데모",
        )
        st.caption(
            f"{f.source} {fund_date:%Y-%m-%d} 기준 — PER {fund_row['per']:.2f}배 · PBR {fund_row['pbr']:.2f}배 · "
            f"배당수익률 {_fmt(fund_row.get('div_yield'), '{:.2f}', '%')}"
        )
    else:
        st.caption("현재 KOSPI의 PER·PBR·배당수익률을 입력하세요. KRX 정보데이터시스템이나 증권사 앱에서 "
                   "확인할 수 있습니다. **처음 값은 예시입니다.**")
        per_in = st.number_input("현재 PER (배)", 1.0, 200.0, 15.0, 0.1, key="per_in")
        pbr_in = st.number_input("현재 PBR (배)", 0.1, 20.0, 1.2, 0.01, key="pbr_in")
        dy_in = st.number_input("현재 배당수익률 (%)", 0.0, 20.0, 1.5, 0.05, key="dy_in")
        f = sim.Fundamentals.from_ratios(last_close, per_in, pbr_in, dy_in, asof=last_date, source="직접 입력")

    fwd_per_in = st.number_input("선행 PER (배, 선택)", 0.0, 200.0, 0.0, 0.1, key="fwd_per_in",
                                 help="증권사 컨센서스 기준 12개월 선행 PER. 0이면 표시하지 않습니다.")
    bond = st.number_input("국고채 10년 금리 (%)", 0.0, 20.0, 3.0, 0.05, key="bond",
                           help="일드갭(이익수익률 − 금리) 계산에 씁니다. 현재 금리로 바꿔 넣으세요.")
    gdp = st.number_input("명목 GDP (조원)", 0.0, 20000.0, 2550.0, 10.0, key="gdp",
                          help="시가총액/GDP(버핏지수)에 씁니다. 2024년 한국 명목 GDP는 약 2,550조원입니다.")
    mcap_series = data["mcap"].dropna() if "mcap" in data else pd.Series(dtype=float)
    if len(mcap_series):
        mcap_per_point = mcap_series.iloc[-1] / data.loc[mcap_series.index[-1], "close"]
        st.caption(f"KOSPI 시가총액 {mcap_series.iloc[-1] / 1e12:,.0f}조원 ({mcap_series.index[-1]:%Y-%m-%d})")
    else:
        mcap_in = st.number_input("현재 KOSPI 시가총액 (조원, 선택)", 0.0, 50000.0, 0.0, 10.0, key="mcap_in",
                                  help="0이면 시가총액과 시가총액/GDP를 표시하지 않습니다.")
        mcap_per_point = mcap_in * 1e12 / last_close if mcap_in > 0 else None
    f = dataclasses.replace(
        f, fwd_eps=last_close / fwd_per_in if fwd_per_in > 0 else None, mcap_per_point=mcap_per_point
    )


# ---------------------------------------------------------------------------
# 가상 KOSPI 입력
# ---------------------------------------------------------------------------
base_key = f"{source}|{last_date:%Y%m%d}|{last_close:.2f}"
if ss.get("base_key") != base_key:  # 데이터가 바뀌면 입력값을 기준가에 맞춰 다시 잡는다
    ss.base_key = base_key
    ss.target_pct = 5.0
    ss.target_pt = round(last_close * 1.05, 2)
    ss.seed = 0
ss.setdefault("days", 10)
ss.setdefault("path_mode", PATH_LINE)

st.caption(
    f"기준: **{last_date:%Y-%m-%d} 종가 {last_close:,.2f}** · 가상 KOSPI를 정하면 그 수준까지 가는 경로를 "
    "실제 가격 뒤에 붙여 지표를 다시 계산합니다."
)
with st.container(border=True):
    c1, c2, c3 = st.columns([1.2, 2.4, 1.6])
    c1.number_input("가상 KOSPI (pt)", min_value=round(last_close * 0.3, 2), max_value=round(last_close * 3, 2),
                    step=10.0, format="%.2f", key="target_pt", on_change=_from_pt, args=(last_close,))
    c2.slider("현재 대비 변동률 (%)", -50.0, 50.0, step=0.1, format="%+.1f%%", key="target_pct",
              on_change=_from_pct, args=(last_close,))
    c3.slider("도달 기간 (거래일)", 1, 120, key="days",
              help="1이면 다음 거래일 하루 만에 그 가격으로 마감한다고 봅니다. 20거래일 ≈ 1개월.")

    p_cols = st.columns([1] * len(PRESETS) + [2.2, 1.3])
    for col, pct in zip(p_cols, PRESETS):
        col.button(f"{pct:+d}%", on_click=_set_pct, args=(last_close, pct), width="stretch",
                   key=f"preset_{pct}")
    p_cols[-2].radio("경로 모양", [PATH_LINE, PATH_RANDOM], key="path_mode", horizontal=True,
                     label_visibility="collapsed",
                     help="일정한 비율: 매일 같은 비율로 움직임 / 변동성 반영: 최근 60일 변동성으로 오르내리다 목표에서 끝남")
    p_cols[-1].button("다른 경로", on_click=_new_seed, disabled=ss.path_mode != PATH_RANDOM,
                      width="stretch", help="변동성 반영 경로를 새로 뽑습니다")

target = float(ss.target_pt)
days = int(ss.days)
random_path = ss.path_mode == PATH_RANDOM
prices = data[[c for c in sim.PRICE_COLS if c in data]]
ind = sim.add_technicals(prices)
vol = sim.daily_volatility(prices["close"]) if random_path else 0.0
sim_df = sim.simulate(prices, target, days, vol, ss.get("seed", 0))
cur, end = ind.iloc[-1], sim_df.iloc[-1]
gdp_won = gdp * 1e12 if gdp > 0 else None
v_now = sim.valuation_at(last_close, f, bond, gdp_won)
v_sim = sim.valuation_at(target, f, bond, gdp_won)
end_date = sim_df.index[-1]


# ---------------------------------------------------------------------------
# 요약
# ---------------------------------------------------------------------------
def _metric(col, label, now, new, fmt, unit="", help=None):
    if new is None or pd.isna(new):
        col.metric(label, "–", help=help)
        return
    delta = None if now is None or pd.isna(now) else f"{new - now:+{fmt}} (현재 {now:{fmt}}{unit})"
    col.metric(label, f"{new:{fmt}}{unit}", delta, delta_color="off", help=help)


when = "다음 거래일" if days == 1 else f"{days}거래일 뒤"
m = st.columns(6)
m[0].metric("가상 KOSPI", f"{target:,.2f}", f"{(target / last_close - 1) * 100:+.1f}% ({when})", delta_color="off",
            help=f"가상 경로의 마지막 날: {end_date:%Y-%m-%d}")
_metric(m[1], "PER", v_now.get("per"), v_sim.get("per"), ".2f", "배", "지수 ÷ 1포인트당 이익. 이익은 그대로라고 가정")
_metric(m[2], "PBR", v_now.get("pbr"), v_sim.get("pbr"), ".2f", "배", "지수 ÷ 1포인트당 순자산")
_metric(m[3], "배당수익률", v_now.get("div_yield"), v_sim.get("div_yield"), ".2f", "%")
_metric(m[4], "RSI (14)", cur["rsi"], end["rsi"], ".1f", help=sim.describe("rsi", end))
_metric(m[5], "20일 이격도", cur["disp20"], end["disp20"], ".1f", "%", "종가 ÷ 20일 이동평균 × 100")


# ---------------------------------------------------------------------------
# 탭
# ---------------------------------------------------------------------------
tab_val, tab_tech, tab_levels, tab_help = st.tabs(["📊 밸류에이션", "📈 기술적 지표", "🪜 주요 가격대", "ℹ️ 설명"])

with tab_val:
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.markdown("##### 현재 vs 가상")
        rows = []
        for key, (label, fmt, unit) in sim.VALUATION_ROWS.items():
            if key not in v_sim:
                continue
            a, b = v_now.get(key), v_sim[key]
            col = {"level": "close"}.get(key, key)
            past = hist[col] / (1e12 if key == "mcap" else 1) if col in hist else None
            pos = sim.percentile_of(past, b) if past is not None else None
            rows.append({
                "항목": label,
                "현재": _fmt(a, fmt, unit),
                "가상": _fmt(b, fmt, unit),
                "변화": _fmt(b - a if a is not None else None, fmt.replace("{:", "{:+").replace("++", "+")),
                f"과거 {years}년 중 위치": "–" if pos is None else f"{pos:.0f}%",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        notes = [
            "기업 이익·순자산·배당은 그대로이고 지수만 움직인다고 가정합니다.",
            f"'과거 {years}년 중 위치'는 그 기간에 이 값보다 낮았던 날의 비율입니다. "
            "PER·PBR은 높을수록 비싼 쪽, 배당수익률은 낮을수록 비싼 쪽입니다.",
        ]
        if f.roe:
            notes.append(f"현재 PER·PBR로 역산한 ROE는 약 {f.roe:.1f}%입니다 (PBR = PER × ROE).")
        if not has_fund:
            notes.append("PER·PBR 이력이 없어 과거 대비 위치는 계산하지 않습니다 (KRX 로그인 시 표시).")
        st.caption(" ".join(notes))

    with right:
        st.markdown("##### 목표 배수 → KOSPI 환산")
        options = [k for k in METRIC_NAMES if sim.level_for(f, k, 1.0)]
        metric = st.radio("기준", options, format_func=METRIC_NAMES.get, horizontal=True,
                          label_visibility="collapsed")
        unit = METRIC_UNITS[metric]
        now_val = v_now.get(metric)
        series = hist[metric].dropna() if metric in hist else pd.Series(dtype=float)
        if len(series) >= 20:
            points = [("과거 최저", series.min()), ("하위 10%", series.quantile(0.1)),
                      ("하위 25%", series.quantile(0.25)), ("중앙값", series.median()),
                      ("상위 25%", series.quantile(0.75)), ("상위 10%", series.quantile(0.9)),
                      ("과거 최고", series.max())]
            st.caption(f"과거 {years}년 {METRIC_NAMES[metric]} 분포의 각 지점이 되려면 KOSPI가 얼마여야 하는지")
        else:
            points = [(f"현재 {r:+.0%}", now_val * (1 + r)) for r in (-0.2, -0.1, 0.0, 0.1, 0.2)]
            st.caption(f"현재 {METRIC_NAMES[metric]}에서 배수를 바꿨을 때 KOSPI")
        rev = []
        for name, value in points:
            lvl = sim.level_for(f, metric, value)
            if lvl:
                rev.append({"기준": name, METRIC_NAMES[metric]: f"{value:.2f}{unit}",
                            "KOSPI": f"{lvl:,.0f}", "현재 대비": f"{(lvl / last_close - 1) * 100:+.1f}%"})
        st.dataframe(pd.DataFrame(rev), hide_index=True, width="stretch")

        c_in, c_btn = st.columns([1, 1.3], vertical_alignment="bottom")
        want = c_in.number_input(f"원하는 {METRIC_NAMES[metric]} ({unit})", 0.01, 200.0,
                                 float(round(now_val, 2)) if now_val else 1.0, 0.05, key=f"want_{metric}")
        want_level = sim.level_for(f, metric, want)
        c_btn.button(f"KOSPI {want_level:,.0f}로 시뮬레이션" if want_level else "계산 불가",
                     on_click=_set_level, args=(last_close, want_level or last_close),
                     disabled=not want_level, width="stretch")

    st.markdown("##### 밴드 차트")
    band = st.radio("밴드", ["PBR", "PER"], horizontal=True, key="band_metric", label_visibility="collapsed")
    band_key = band.lower()
    if v_now.get(band_key):
        if has_fund:
            per_point = sim.implied_per_point(data)["bps" if band_key == "pbr" else "eps"]
            band_start = hist_from
        else:
            per_point = pd.Series(f.bps if band_key == "pbr" else f.eps, index=data.index)
            band_start = last_date - pd.Timedelta(days=365)
        multiples = sim.nice_multiples(v_now[band_key], hist[band_key] if band_key in hist else None, 6)
        st.plotly_chart(build_band_chart(sim_df, per_point, multiples, band, band_start),
                        theme="streamlit")
        st.caption(
            f"점선은 1포인트당 {'순자산(BPS)' if band_key == 'pbr' else '이익(EPS)'} × 배수입니다. "
            + ("KRX가 발표한 PER·PBR로 역산했으며, 실적 공시 시점에 계단처럼 바뀝니다. " if has_fund else
               "직접 입력한 값으로 계산해 수평선으로 표시합니다. ")
            + "가상 구간은 마지막 값을 그대로 이어 씁니다."
        )
    else:
        st.info(f"{band} 값이 없어 밴드 차트를 그릴 수 없습니다.")

with tab_tech:
    st.plotly_chart(build_technical_chart(sim_df), theme="streamlit")
    t_left, t_right = st.columns([1.7, 1], gap="large")
    with t_left:
        st.markdown(f"##### 지표 비교: 현재({last_date:%m/%d}) → 가상({end_date:%m/%d})")
        table = sim.compare_table(cur, end)
        st.dataframe(table, hide_index=True, width="stretch", height=35 * (len(table) + 1) + 3)
    with t_right:
        st.markdown("##### 가상 구간에서 생기는 신호")
        events = sim.path_events(sim_df)
        if events:
            st.markdown("\n".join(f"- `{d:%m/%d}` {text}" for d, text in events))
        else:
            st.caption("새로 생기는 교차·돌파 신호가 없습니다.")
        st.caption(
            "가상 거래일은 주말만 뺀 날짜입니다 (공휴일 미반영). 가상 구간의 시가는 전날 종가, "
            "고가·저가는 시가와 종가 중 큰 값·작은 값입니다. "
            + ("변동성 반영 경로는 최근 60거래일 일간 변동성"
               f"({vol * 100:.2f}%)으로 오르내리다 목표가에서 끝납니다. " if random_path else
               "'일정한 비율로' 경로는 매일 같은 방향으로 움직여 RSI·스토캐스틱·투자심리선이 극단값으로 가기 쉽습니다. "
               "'변동성 반영'을 고르면 더 현실적인 경로로 볼 수 있습니다.")
        )

with tab_levels:
    st.markdown("##### 다음 거래일 종가가 얼마면 어떤 신호가 나오나")
    st.caption(f"{last_date:%Y-%m-%d} 종가 {last_close:,.2f}에서 다음 거래일 하루 만에 그 가격으로 마감한다고 보고 "
               "역산한 값입니다. 현재가 ±30% 범위만 표시합니다.")
    levels = sim.next_day_levels(ind)
    ladder = [{"가격": lv.price, "현재 대비": (lv.price / last_close - 1) * 100, "기준": lv.label, "의미": lv.meaning}
              for lv in levels]
    ladder.append({"가격": last_close, "현재 대비": 0.0, "기준": "현재 종가", "의미": f"{last_date:%Y-%m-%d} 마감"})
    ladder.append({"가격": target, "현재 대비": (target / last_close - 1) * 100, "기준": "가상 KOSPI",
                   "의미": "위에서 정한 값"})
    ladder_df = pd.DataFrame(ladder).sort_values("가격", ascending=False, kind="stable")

    def _highlight(row):
        if row["기준"] == "현재 종가":
            return ["background-color: rgba(42,120,214,0.16); font-weight: 600"] * len(row)
        if row["기준"] == "가상 KOSPI":
            return ["background-color: rgba(235,104,52,0.20); font-weight: 600"] * len(row)
        return [""] * len(row)

    st.dataframe(
        ladder_df.style.apply(_highlight, axis=1).format({"가격": "{:,.2f}", "현재 대비": "{:+.2f}%"}),
        hide_index=True, width="stretch", height=35 * (len(ladder_df) + 1) + 3,
    )
    if levels:
        c_sel, c_go = st.columns([2, 1], vertical_alignment="bottom")
        pick = c_sel.selectbox("가격대를 골라 시뮬레이션", range(len(levels)),
                               format_func=lambda i: f"{levels[i].label} — {levels[i].price:,.2f}")
        c_go.button("다음 거래일 이 가격으로", on_click=_set_level, args=(last_close, levels[pick].price, True),
                    width="stretch")

with tab_help:
    st.markdown(
        f"""
**이 화면이 하는 일**: 가상 KOSPI와 도달 기간을 정하면, 마지막 실제 종가에서 그 값까지 가는 가상 일봉을 만들어
실제 가격 뒤에 붙인 다음 모든 지표를 처음부터 다시 계산합니다.

**밸류에이션** (기업 이익·순자산·배당은 그대로, 지수만 움직인다고 가정)

| 지표 | 계산 | 읽는 법 |
|---|---|---|
| PER | 지수 ÷ 1포인트당 이익(EPS) | 이익 대비 몇 배에 거래되는지. KRX 지수 PER은 공시된 연간 실적 기준이라 실적 변화가 늦게 반영됨 |
| 선행 PER | 지수 ÷ 12개월 예상 이익 | 증권사 컨센서스를 직접 입력 |
| PBR | 지수 ÷ 1포인트당 순자산(BPS) | 1배 = 장부가치. KOSPI는 오랫동안 1배 안팎에서 움직였음 |
| 배당수익률 | 1포인트당 배당 ÷ 지수 | 지수가 오르면 낮아짐 |
| 일드갭 | 이익수익률(1/PER) − 국고채 10년 금리 | 클수록 채권 대비 주식이 싸다는 해석 |
| 시가총액 / GDP | 시가총액 ÷ 명목 GDP | 경제 규모 대비 증시 크기 (버핏지수) |

1포인트당 EPS·BPS·배당은 KRX가 발표한 PER·PBR·배당수익률로 역산합니다 (EPS = 종가 ÷ PER).

**기술적 지표** (국내 HTS 기본 설정)

| 지표 | 설정 | 흔히 쓰는 기준 |
|---|---|---|
| 이동평균·이격도 | 5·20·60·120일 | 이격도 = 종가 ÷ 이동평균 × 100 |
| RSI | 14일, Wilder 평활 | 70 이상 과매수, 30 이하 과매도 |
| MACD | 12·26일 지수이동평균, 시그널 9일 | 오실레이터(MACD − 시그널)가 0을 넘으면 상승 우위 |
| 볼린저밴드 | 20일, 2표준편차 | %B > 1 상단 돌파, < 0 하단 이탈 |
| 스토캐스틱 슬로우 | 14·3·3 | 80 이상 과매수, 20 이하 과매도 |
| 투자심리선 | 10일 중 상승일 비율 | 75% 이상 과열, 25% 이하 침체 |
| 52주 고점 대비 | 250거래일 최고가 기준 | −10% 조정, −20% 약세장 |

**주요 가격대**는 다음 거래일 종가 하나만 바뀐다고 보고 각 지표가 기준선에 닿는 가격을 식으로 풀어 낸 값입니다.

모든 결과는 가정에 따른 계산이며 투자 판단의 근거가 아닙니다.
"""
    )
