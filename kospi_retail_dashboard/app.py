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
    build_position_bars,
    build_position_history,
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
    cumulative,
    make_demo_data,
)
from position import CATEGORIES, build_position, zone_label  # noqa: E402

st.set_page_config(page_title="KOSPI 개인 수급 대시보드", page_icon="📈", layout="wide")

SOURCE_LIVE = "실제 데이터 (KRX·금융투자협회·네이버)"
SOURCE_DEMO = "데모 (가상 데이터)"
SOURCE_CSV = "CSV 업로드"
PERIODS = {"6개월": 182, "1년": 365, "3년": 365 * 3, "5년": 365 * 5, "10년": 365 * 10}
LOOKBACKS = {"3년": 3, "5년": 5, "10년": 10}


def _secret(key: str) -> str:
    try:
        return st.secrets.get(key, "") or os.getenv(key, "")
    except Exception:  # secrets.toml 이 없을 때
        return os.getenv(key, "")


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def cached_live(start: pd.Timestamp, end: pd.Timestamp, use_krx: bool, krx_id: str, ecos_key: str):
    # krx_id 는 계정이 바뀌면 캐시를 새로 만들기 위한 키로만 쓴다.
    df, report = load_market_data(start, end, use_krx, ecos_key)
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
        with st.expander("한국은행 ECOS API 키 (금리·환율)", expanded=False):
            st.caption("없어도 네이버에서 국고채 3년·환율을 가져옵니다. "
                       "키가 있으면 국고채 10년물로 일드갭을 계산합니다 (ecos.bok.or.kr 무료 발급).")
            ecos_key = st.text_input("ECOS API 키", value=_secret("ECOS_KEY"), type="password")
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
    lookback_label = st.selectbox("시장 위치 비교 기간", list(LOOKBACKS), index=1,
                                  help="각 지표의 현재 값을 이 기간의 과거 분포와 비교해 백분위를 매깁니다")
    lookback_years = LOOKBACKS[lookback_label]


# ---------------------------------------------------------------------------
# 데이터 불러오기
# ---------------------------------------------------------------------------
st.title("KOSPI × 개인 투자자 수급 대시보드")

# 이동평균·z-score 가 기간 첫날부터 계산되도록 앞쪽 여유 기간을 더 받아 둔다.
warmup = pd.Timedelta(days=int((flow_window + z_window) * 1.6))
# 시장 위치 백분위는 화면 기간과 상관없이 '최근 N년' 분포가 필요하다.
load_start = min(start - warmup, end - pd.Timedelta(days=int(lookback_years * 365 + 300)))
report_items: list[tuple[str, bool, str]] = []
if source == SOURCE_DEMO:
    raw = cached_demo(load_start, end)
    st.warning("데모 모드: 화면 구성을 보기 위한 **가상 데이터**입니다. 실제 시장 데이터가 아닙니다.", icon="⚠️")
elif source == SOURCE_CSV:
    if uploaded is None:
        st.info("왼쪽에서 CSV 파일을 올려 주세요. 양식은 'CSV 양식 받기'로 받을 수 있습니다.")
        st.stop()
    try:
        raw = read_uploaded_csv(uploaded).loc[load_start:end]
    except Exception as e:
        st.error(f"CSV 를 읽지 못했습니다: {e}")
        st.stop()
else:
    with st.spinner("데이터를 불러오는 중… (처음 한 번은 오래 걸릴 수 있습니다)"):
        raw, report_items = cached_live(load_start, end, use_krx, krx_id if use_krx else "", ecos_key)

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
for who in ("retail", "foreign", "inst"):  # 누적 순매수는 화면 기간 첫날부터 다시 누적
    if f"{who}_net" in df:
        df[f"{who}_net_cum"] = cumulative(df[f"{who}_net"])

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
tab_pos, tab_main, tab_cmp, tab_corr, tab_data, tab_help = st.tabs(
    ["🧭 시장 위치", "📈 KOSPI + 지표", "🔀 한 차트로 비교", "🔗 상관관계", "🗂 데이터", "ℹ️ 지표 설명"]
)

with tab_pos:
    pos_full, scores, snap, summ, levels = build_position(full, lookback_years)
    if snap.empty:
        st.info("시장 위치를 계산할 만큼 데이터가 충분하지 않습니다 (지표별로 최소 6개월 필요).")
    else:
        st.markdown(
            f"각 지표의 현재 값이 **최근 {lookback_years}년 분포**에서 어디쯤인지(백분위)를 "
            "**온도 0~100**으로 맞춘 결과입니다. 0에 가까울수록 침체·저평가, 100에 가까울수록 과열·고평가입니다."
        )
        mcols = st.columns(1 + len(CATEGORIES))
        for c, key in zip(mcols, ["overall", *CATEGORIES]):
            if key not in summ:
                c.metric("종합 점수" if key == "overall" else key, "-", "데이터 없음", delta_color="off")
                continue
            item = summ[key]
            delta = item["label"] + ("" if item["change_1m"] is None else f" · 1개월 {item['change_1m']:+.0f}")
            c.metric("종합 점수" if key == "overall" else key, f"{item['score']:.0f} / 100",
                     delta, delta_color="off", help=f"기준일 {item['date']:%Y-%m-%d}")

        st.plotly_chart(build_position_bars(snap), use_container_width=True, theme="streamlit")
        st.caption("막대 = 온도. 역방향 지표(배당수익률·일드갭·변동성·환율)는 값이 낮을수록 온도가 높습니다. "
                   "막대에 마우스를 올리면 실제 값과 백분위가 보입니다.")

        left, right = st.columns([3, 2])
        with left:
            st.markdown("**지표별 현재 값**")
            table = snap.assign(
                현재값=[f"{v:,.{int(d)}f} {u}".strip() for v, d, u in zip(snap["현재값"], snap["digits"], snap["단위"])],
                기준일=snap["기준일"].dt.strftime("%Y-%m-%d"),
                백분위=snap["백분위"].round(0).astype(int),
                온도=snap["온도"].round(0).astype(int),
            )[["카테고리", "지표", "현재값", "백분위", "온도", "해석", "기준일"]]
            st.dataframe(table, hide_index=True, use_container_width=True)
        with right:
            st.markdown("**주요 가격 레벨**")
            if not levels.empty:
                st.dataframe(
                    levels, hide_index=True, use_container_width=True,
                    column_config={
                        "지수": st.column_config.NumberColumn(format="%.2f"),
                        "현재가 대비(%)": st.column_config.NumberColumn(format="%+.1f%%"),
                    },
                )
                notes = []
                if "alignment" in levels.attrs:
                    notes.append(f"이동평균 배열: **{levels.attrs['alignment']}**")
                if not pd.isna(levels.attrs.get("drawdown", float("nan"))):
                    notes.append(f"불러온 기간 고점 대비: **{levels.attrs['drawdown']:+.1f}%**")
                st.markdown(" · ".join(notes))
            missing = [n for n, k in (("PER·PBR·배당", "per"), ("외국인·기관 수급", "foreign_net"),
                                      ("금리", "bond_3y"), ("환율", "usdkrw"), ("시가총액", "kospi_mcap"))
                       if k not in raw or raw[k].notna().sum() == 0]
            if missing:
                st.caption("빠진 데이터: " + ", ".join(missing) +
                           " — 해당 지표는 점수에서 제외됩니다 (KRX 로그인·ECOS 키 또는 CSV 로 보완).")

        st.markdown("**점수 추이**")
        hist_scores = scores.loc[start:end]
        st.plotly_chart(
            build_position_history(pos_full.loc[start:end], hist_scores, CATEGORIES),
            use_container_width=True, theme="streamlit",
        )
        st.caption("점수는 각 날짜 기준 직전 비교 기간의 분포로 계산합니다 (미래 데이터 사용 없음). "
                   "과거에 점수가 극단(20 이하·80 이상)이었던 시점과 이후 KOSPI 흐름을 비교해 보세요. "
                   "참고용이며 매매 시점을 보장하지 않습니다.")
        st.download_button(
            "시장 위치 표 CSV", snap.drop(columns=["key", "digits"]).to_csv(index=False).encode("utf-8-sig"),
            f"kospi_position_{end:%Y%m%d}.csv", "text/csv",
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
| **외국인·기관 순매수** | 외국인 합계·기관 합계의 순매수 | KRX / 네이버 |
| **PER · PBR · 배당수익률** | KOSPI 지수 밸류에이션 | KRX |
| **국고채 금리** | 10년물(ECOS 키 있을 때) 또는 3년물 | 한국은행 ECOS / 네이버 |
| **원/달러 환율** | 매매기준율 | 한국은행 ECOS / 네이버 |

**개인 투자심리 지수 구성** (사용 가능한 요소만 반영, 최소 2개)
- 개인 순매수 {flow_window}일 이동합
- 개인 거래대금 비중
- 투자자예탁금 {flow_window}일 증가율
- 신용융자 {flow_window}일 증가율

z-score 가 **+1 이상**이면 최근 {z_window}거래일 평균보다 1 표준편차 이상 개인 매수세·자금 유입·레버리지가 강하다는 뜻,
**−1 이하**이면 그 반대입니다. 투자 판단의 근거가 아닌 참고용 지표입니다.

**🧭 시장 위치 점수** (비교 기간 = 사이드바 '시장 위치 비교 기간')

각 지표를 비교 기간 분포의 백분위(0~100)로 바꾸고, 값이 낮을수록 과열인 지표(배당수익률, 일드갭, 변동성, 환율)는
뒤집어 **온도**로 통일합니다. 카테고리 점수 = 지표 온도 평균, 종합 점수 = 카테고리 점수 평균.

| 카테고리 | 지표 |
|---|---|
| 밸류에이션 | PER, PBR, 배당수익률(역), 일드갭 = 1/PER − 국고채 금리(역) |
| 추세 | 60·200일선 이격도, 52주 고저 내 위치, RSI(14) |
| 수급 | 외국인·기관·개인 순매수 {flow_window}일 합 |
| 심리·리스크 | 개인 투자심리 지수, 신용융자/예탁금, 신용융자/시가총액, 20일 변동성(역), 원/달러 환율(역) |

0~20 침체·저평가 · 20~40 낮음 · 40~60 중립 · 60~80 높음 · 80~100 과열·고평가.
백분위는 '최근 N년 중 상대적 위치'이므로, 구조적으로 수준이 바뀐 지표(예: 밸류업 이후 PBR)는 해석에 주의하세요.

**데이터 기준일 주의**: 예탁금·신용융자는 금융투자협회가 보통 1~2 영업일 늦게 공표합니다.
최근 며칠은 직전 공표값이 이어서 표시될 수 있습니다.
"""
    )
