"""KOSPI × 개인 투자자 수급 대시보드

실행:  streamlit run kospi_retail_dashboard/app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from charts import (  # noqa: E402
    COMPARE_SERIES,
    available_panels,
    build_compare_chart,
    build_corr_heatmap,
    build_main_chart,
)
from data_sources import (  # noqa: E402
    LABELS,
    RAW_COLUMNS,
    load_market_data,
    read_uploaded_csv,
    set_krx_credentials,
)
from indicators import (  # noqa: E402
    DERIVED_LABELS,
    SENTIMENT_COMPONENTS,
    add_indicators,
    correlation_table,
    make_demo_data,
)

st.set_page_config(page_title="KOSPI 개인 수급 대시보드", page_icon="📈", layout="wide")

SOURCE_LIVE = "실제 데이터 (KRX·금융투자협회·네이버)"
SOURCE_DEMO = "데모 (가상 데이터)"
SOURCE_CSV = "CSV 업로드"
PERIODS = {"6개월": 182, "1년": 365, "3년": 365 * 3, "5년": 365 * 5, "10년": 365 * 10}


def _secret(key: str) -> str:
    try:
        return st.secrets.get(key, "") or os.getenv(key, "")
    except Exception:  # secrets.toml 이 없을 때
        return os.getenv(key, "")


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def cached_live(start: pd.Timestamp, end: pd.Timestamp, use_krx: bool, krx_id: str):
    # krx_id 는 계정이 바뀌면 캐시를 새로 만들기 위한 키로만 쓴다.
    df, report = load_market_data(start, end, use_krx)
    return df, report.items


@st.cache_data(show_spinner=False)
def cached_demo(start: pd.Timestamp, end: pd.Timestamp):
    return make_demo_data("2010-01-01", end).loc[start:end]


# ---------------------------------------------------------------------------
# 사이드바
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("설정")
    source = st.radio("데이터", [SOURCE_LIVE, SOURCE_DEMO, SOURCE_CSV])

    today = pd.Timestamp.today().normalize()
    period = st.selectbox("기간", [*PERIODS, "직접 입력"], index=1)
    if period == "직접 입력":
        picked = st.date_input("시작 ~ 종료", (today - pd.Timedelta(days=365), today), max_value=today)
        start, end = (pd.Timestamp(picked[0]), pd.Timestamp(picked[-1]))
    else:
        start, end = today - pd.Timedelta(days=PERIODS[period]), today

    uploaded = None
    if source == SOURCE_LIVE:
        with st.expander("KRX 로그인 (개인 순매수·거래량)", expanded=not _secret("KRX_ID")):
            st.caption(
                "KRX 정보데이터시스템(data.krx.co.kr) 계정이 필요합니다. "
                "없으면 개인 순매수는 네이버에서 가져오고, 개인 거래량·거래대금 비중은 표시되지 않습니다."
            )
            krx_id = st.text_input("KRX ID", value=_secret("KRX_ID"))
            krx_pw = st.text_input("KRX 비밀번호", value=_secret("KRX_PW"), type="password")
        use_krx = set_krx_credentials(krx_id, krx_pw)
        if st.button("데이터 새로고침", use_container_width=True):
            cached_live.clear()
    elif source == SOURCE_CSV:
        uploaded = st.file_uploader("CSV 파일", type="csv")
        template = pd.DataFrame(columns=["date", *RAW_COLUMNS])
        st.download_button(
            "CSV 양식 받기", template.to_csv(index=False).encode("utf-8-sig"),
            "kospi_retail_template.csv", "text/csv", use_container_width=True,
        )
        st.caption("금액은 원, 거래량은 주 단위. 컬럼은 영문 키 또는 한글 이름 모두 인식합니다.")

    st.divider()
    flow_window = st.slider("수급 이동 기간 (거래일)", 5, 60, 20, 5,
                            help="개인 순매수 이동합, 예탁금·신용융자 증가율, KOSPI 수익률 계산 기간")
    z_window = st.slider("심리지수 표준화 기간 (거래일)", 60, 250, 120, 10,
                         help="각 지표를 이 기간의 평균·표준편차로 z-score 화합니다")


# ---------------------------------------------------------------------------
# 데이터 불러오기
# ---------------------------------------------------------------------------
st.title("KOSPI × 개인 투자자 수급 대시보드")

# 이동평균·z-score 가 기간 첫날부터 계산되도록 앞쪽 여유 기간을 더 받아 둔다.
warmup = pd.Timedelta(days=int((flow_window + z_window) * 1.6))
report_items: list[tuple[str, bool, str]] = []
if source == SOURCE_DEMO:
    raw = cached_demo(start - warmup, end)
    st.warning("데모 모드: 화면 구성을 보기 위한 **가상 데이터**입니다. 실제 시장 데이터가 아닙니다.", icon="⚠️")
elif source == SOURCE_CSV:
    if uploaded is None:
        st.info("왼쪽에서 CSV 파일을 올려 주세요. 양식은 'CSV 양식 받기'로 받을 수 있습니다.")
        st.stop()
    try:
        raw = read_uploaded_csv(uploaded).loc[start - warmup : end]
    except Exception as e:
        st.error(f"CSV 를 읽지 못했습니다: {e}")
        st.stop()
else:
    with st.spinner("데이터를 불러오는 중… (처음 한 번은 오래 걸릴 수 있습니다)"):
        raw, report_items = cached_live(start - warmup, end, use_krx, krx_id if use_krx else "")

if raw is None or raw.empty:
    st.error("불러온 데이터가 없습니다. 아래 소스 상태를 확인하거나 데모 모드로 화면을 확인해 보세요.")
    for name, ok, msg in report_items:
        st.write(("✅ " if ok else "❌ ") + f"**{name}** — {msg}")
    st.stop()

full = add_indicators(raw, flow_window=flow_window, z_window=z_window)
df = full.loc[start:end].copy()
if df.empty:
    st.error("선택한 기간에 데이터가 없습니다.")
    st.stop()
if "retail_net" in df:  # 누적 순매수는 화면 기간 첫날부터 다시 누적
    df["retail_net_cum"] = df["retail_net"].fillna(0).cumsum().where(df["retail_net"].notna().cummax())

if report_items:
    failed = [i for i in report_items if not i[1]]
    with st.expander(
        "데이터 소스 상태" + (f" — ⚠️ {len(failed)}개 실패" if failed else " — 모두 정상"),
        expanded=bool(failed),
    ):
        for name, ok, msg in report_items:
            st.write(("✅ " if ok else "❌ ") + f"**{name}** — {msg}")
        if not use_krx:
            st.caption("KRX 로그인 정보가 없어 개인 거래량·거래대금 비중은 표시되지 않습니다.")


# ---------------------------------------------------------------------------
# 요약 지표
# ---------------------------------------------------------------------------
def last_valid(col: str):
    s = df[col].dropna() if col in df else pd.Series(dtype=float)
    return (s.iloc[-1], s.index[-1]) if len(s) else (None, None)


def change_over(col: str, periods: int):
    s = df[col].dropna() if col in df else pd.Series(dtype=float)
    return s.iloc[-1] - s.iloc[-1 - periods] if len(s) > periods else None


cols = st.columns(6)
kospi, kdate = last_valid("kospi_close")
if kospi is not None:
    first = df["kospi_close"].dropna().iloc[0]
    cols[0].metric("KOSPI", f"{kospi:,.2f}", f"{(kospi / first - 1) * 100:+.1f}% (기간)",
                   delta_color="off", help=f"기준일 {kdate:%Y-%m-%d}")
if "retail_net" in df and df["retail_net"].notna().any():
    total = df["retail_net"].sum()
    recent = df["retail_net"].tail(flow_window).sum()
    cols[1].metric("개인 순매수 (기간 합계)", f"{total / 1e12:+,.1f}조원",
                   f"최근 {flow_window}일 {recent / 1e12:+,.2f}조원", delta_color="off")
for i, (col, name) in enumerate([("deposit", "투자자예탁금"), ("credit", "신용융자 잔고")], start=2):
    val, vdate = last_valid(col)
    if val is not None:
        chg = change_over(col, flow_window)
        cols[i].metric(name, f"{val / 1e12:,.1f}조원",
                       None if chg is None else f"{chg / 1e12:+,.2f}조원 ({flow_window}일)",
                       delta_color="off", help=f"기준일 {vdate:%Y-%m-%d}")
val, _ = last_valid("credit_to_deposit")
if val is not None:
    cols[4].metric("신용융자 / 예탁금", f"{val:.1f}%",
                   help="높을수록 빚을 낸 투자 비중이 큼 (과열 신호로 해석되는 경우가 많음)")
val, _ = last_valid("sentiment")
if val is not None:
    mood = "과열" if val >= 1 else "위축" if val <= -1 else "중립"
    cols[5].metric("개인 투자심리 지수", f"{val:+.2f}", mood, delta_color="off",
                   help="+1 이상: 평소보다 개인 매수세·레버리지가 강함 / -1 이하: 약함")


# ---------------------------------------------------------------------------
# 탭
# ---------------------------------------------------------------------------
tab_main, tab_cmp, tab_corr, tab_data, tab_help = st.tabs(
    ["📈 KOSPI + 지표", "🔀 한 차트로 비교", "🔗 상관관계", "🗂 데이터", "ℹ️ 지표 설명"]
)

with tab_main:
    avail = available_panels(df)
    c1, c2, c3 = st.columns([6, 1, 1])
    panels = c1.multiselect("함께 볼 지표", avail, default=avail, label_visibility="collapsed",
                            placeholder="KOSPI 아래에 표시할 지표 선택")
    candle = c2.toggle("캔들", value=False)
    log_scale = c3.toggle("로그", value=False)
    fig = build_main_chart(df, panels, candle=candle, log_scale=log_scale, flow_window=flow_window)
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    st.caption("모든 패널은 같은 날짜 축을 공유합니다. 마우스를 올리면 같은 날의 값이 함께 표시됩니다. "
               "막대 색: 빨강 = 개인 순매수, 파랑 = 개인 순매도.")

with tab_cmp:
    mode_label = st.radio("비교 방식", ["z-score (평소 대비 위치)", "지수화 (기간 시작 = 100)"],
                          horizontal=True)
    mode = "index" if mode_label.startswith("지수화") else "z"
    options = [c for c in COMPARE_SERIES if c in df and df[c].notna().any()]
    if mode == "index":
        options = [c for c in options if (df[c].dropna() > 0).all()]
    default = [c for c in ("kospi_close", "retail_net_cum", "deposit", "credit") if c in options]
    picked = st.multiselect("비교할 시계열", options, default=default,
                            format_func=lambda c: COMPARE_SERIES[c])
    if picked:
        # z-score 는 앞쪽 여유 기간까지 포함해 계산한 뒤 화면 기간만 잘라 보여 준다.
        source_df = full if mode == "z" else df
        cmp = build_compare_chart(source_df, picked, mode, z_window)
        cmp.update_xaxes(range=[df.index.min(), df.index.max()])
        st.plotly_chart(cmp, use_container_width=True, theme="streamlit")
    st.caption("단위가 다른 지표를 한 축에 겹칠 때는 이중 y축 대신 공통 기준(z-score 또는 시작=100)으로 맞춥니다.")

with tab_corr:
    st.markdown(f"각 지표와 **KOSPI {flow_window}일 수익률**(같은 기간)·**향후 {flow_window}일 수익률**의 상관계수입니다.")
    corr = correlation_table(df)
    if corr.empty:
        st.info("상관관계를 계산할 만큼 데이터가 충분하지 않습니다 (30거래일 이상 필요).")
    else:
        st.plotly_chart(build_corr_heatmap(corr), use_container_width=True, theme="streamlit")
        st.caption("향후 수익률과의 상관은 과거 표본에 대한 통계일 뿐 예측을 보장하지 않습니다. "
                   "이동 기간이 겹치는 일별 표본이라 실제 유의성은 표보다 낮습니다.")
    if "sentiment" in df:
        z_cols = [f"z_{c}" for c in SENTIMENT_COMPONENTS if f"z_{c}" in df]
        latest = df[z_cols].dropna(how="all").tail(1)
        if not latest.empty:
            st.markdown("**최근 심리지수 구성 요소 (z-score)**")
            show = latest.T.iloc[:, 0].rename(lambda c: SENTIMENT_COMPONENTS[c[2:]]).rename("z-score")
            st.bar_chart(show, horizontal=True)

with tab_data:
    labels = {**LABELS, **DERIVED_LABELS}
    table = df[[c for c in labels if c in df]].sort_index(ascending=False)
    st.dataframe(table.rename(columns=labels), use_container_width=True, height=420)
    st.download_button(
        "CSV 다운로드", table.to_csv().encode("utf-8-sig"),
        f"kospi_retail_{start:%Y%m%d}_{end:%Y%m%d}.csv", "text/csv",
    )
    latest_dates = {
        labels[c]: f"{df[c].last_valid_index():%Y-%m-%d}"
        for c in ("kospi_close", "retail_net", "retail_volume", "deposit", "credit")
        if c in df and df[c].notna().any()
    }
    st.caption("최신 데이터 기준일 — " + " · ".join(f"{k}: {v}" for k, v in latest_dates.items()))

with tab_help:
    st.markdown(
        f"""
| 지표 | 의미 | 출처 |
|---|---|---|
| **개인 순매수** | 개인의 KOSPI 매수대금 − 매도대금. (+)면 개인이 순매수 | KRX / 네이버 |
| **개인 누적 순매수** | 선택 기간 첫날부터 순매수를 누적한 값 | 계산 |
| **개인 거래량** | 개인의 매수 수량 + 매도 수량 (주) | KRX |
| **개인 거래대금 비중** | (개인 매수+매도) ÷ (시장 전체 매수+매도). 개인 참여 열기를 보여 줌 | KRX |
| **투자자예탁금** | 주식 투자를 위해 증권사 계좌에 넣어 둔 대기 자금 | 금융투자협회 / 네이버 |
| **신용융자 잔고** | 증권사에서 돈을 빌려 산 주식 금액(빚투). 점선은 유가증권시장분 | 금융투자협회 / 네이버 |
| **신용융자 / 예탁금** | 대기 자금 대비 빚투 규모. 높을수록 레버리지 과열 | 계산 |
| **개인 투자심리 지수** | 아래 요소의 {z_window}일 z-score 평균 | 계산 |

**개인 투자심리 지수 구성** (사용 가능한 요소만 반영, 최소 2개)
- 개인 순매수 {flow_window}일 이동합
- 개인 거래대금 비중
- 투자자예탁금 {flow_window}일 증가율
- 신용융자 {flow_window}일 증가율

z-score 가 **+1 이상**이면 최근 {z_window}거래일 평균보다 1 표준편차 이상 개인 매수세·자금 유입·레버리지가 강하다는 뜻,
**−1 이하**이면 그 반대입니다. 투자 판단의 근거가 아닌 참고용 지표입니다.

**데이터 기준일 주의**: 예탁금·신용융자는 금융투자협회가 보통 1~2 영업일 늦게 공표합니다.
최근 며칠은 직전 공표값이 이어서 표시될 수 있습니다.
"""
    )
