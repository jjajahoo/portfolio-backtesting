"""포트폴리오 백테스터 (Streamlit 앱).

실행: streamlit run streamlit_app.py
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from backtester import charts
from backtester.catalog import PRESETS, TICKERS, display_name
from backtester.engine import compute_metrics, drawdown, run_backtest
from backtester.formatting import duration, money, pct, pp, ratio
from backtester.glossary import CAVEATS, HELP, TERMS, explain
from backtester.market_data import PriceData, clean_ticker, load_prices

st.set_page_config(page_title="포트폴리오 백테스터", page_icon="📈", layout="wide")

REBALANCE_LABELS = {
    "yearly": "매년",
    "quarterly": "분기마다",
    "monthly": "매월",
    "none": "안 함 (처음 비중 그대로 두기)",
}
CURRENCY_LABELS = {"KRW": "원화 (₩)", "USD": "달러 ($)"}


def current_theme() -> str:
    try:
        return st.context.theme.type or "light"
    except AttributeError:
        return "light"


def short_name(ticker: str) -> str:
    """지표 옆에 붙일 짧은 이름. 한국 코드는 이름이 더 알아보기 쉽다."""
    info = TICKERS.get(ticker)
    return info[0] if info and ticker[0].isdigit() else ticker


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def cached_prices(tickers: tuple[str, ...], start: date, end: date, currency: str) -> PriceData:
    return load_prices(list(tickers), start, end, currency)


# ---------------------------------------------------------------- 사이드바: 입력
with st.sidebar:
    st.header("① 포트폴리오 만들기")
    preset_name = st.selectbox("예시에서 고르기", [p.name for p in PRESETS])
    preset = next(p for p in PRESETS if p.name == preset_name)
    st.caption(preset.description)

    edited = st.data_editor(
        pd.DataFrame({"종목코드": list(preset.weights), "비중(%)": list(preset.weights.values())}),
        key=f"editor_{preset_name}",
        num_rows="dynamic",
        hide_index=True,
        column_config={
            "종목코드": st.column_config.TextColumn(
                "종목코드", help="한국: 6자리 코드 (예: 069500) · 미국: 티커 (예: SPY)", required=True
            ),
            "비중(%)": st.column_config.NumberColumn(
                "비중(%)", min_value=0.0, max_value=100.0, step=0.5, format="%.1f", required=True
            ),
        },
    )
    rows = edited.dropna(subset=["종목코드"]).assign(종목코드=lambda d: d["종목코드"].map(clean_ticker))
    rows = rows[rows["종목코드"] != ""]
    weights_pct = rows.groupby("종목코드", sort=False)["비중(%)"].sum().fillna(0)
    weights_pct = weights_pct[weights_pct > 0]
    total_pct = float(weights_pct.sum())

    if abs(total_pct - 100) < 1e-6:
        st.caption(f"✅ 비중 합계 {total_pct:g}%")
    else:
        st.caption(f"⚠️ 비중 합계 {total_pct:g}% — 100%가 되도록 맞춰 주세요")
    if len(weights_pct):
        st.caption("구성: " + " · ".join(f"{display_name(t)} {w:g}%" for t, w in weights_pct.items()))

    with st.expander("🔎 종목코드 찾기"):
        st.dataframe(
            pd.DataFrame(
                [(code, name, desc, kind) for code, (name, desc, kind) in TICKERS.items()],
                columns=["코드", "이름", "설명", "분류"],
            ),
            hide_index=True,
        )
        st.caption("목록에 없어도 Yahoo Finance에 있는 한국·미국 종목이면 대부분 쓸 수 있어요.")

    st.header("② 기간과 금액")
    col1, col2 = st.columns(2)
    start = col1.date_input("시작일", date(2010, 1, 1), min_value=date(1990, 1, 1), max_value=date.today())
    end = col2.date_input("종료일", date.today(), min_value=date(1990, 1, 1), max_value=date.today())
    currency = st.radio(
        "기준 통화", list(CURRENCY_LABELS), format_func=CURRENCY_LABELS.get, horizontal=True,
        help=HELP["base_currency"],
    )
    initial = st.number_input(
        "처음 투자금 (원)" if currency == "KRW" else "처음 투자금 (달러)",
        min_value=1.0,
        value=10_000_000.0 if currency == "KRW" else 10_000.0,
        step=1_000_000.0 if currency == "KRW" else 1_000.0,
        format="%.0f",
        key=f"initial_{currency}",
    )
    st.caption(f"= {money(initial, currency)}")

    st.header("③ 세부 설정")
    rebalance = st.selectbox(
        "리밸런싱 (비중 다시 맞추기)", list(REBALANCE_LABELS), format_func=REBALANCE_LABELS.get,
        help=HELP["rebalance"],
    )
    bench_ticker = clean_ticker(
        st.text_input("비교 대상 (벤치마크)", preset.benchmark, key=f"bench_{preset_name}", help=HELP["benchmark"])
    )
    fee_pct = st.number_input(
        "거래비용 (%)", min_value=0.0, max_value=5.0, value=0.1, step=0.05, format="%.2f", help=HELP["fee"]
    )
    rf_pct = st.number_input(
        "예금 금리 (무위험 수익률, %)", min_value=0.0, max_value=20.0, value=3.0, step=0.25, format="%.2f",
        help=HELP["risk_free"],
    )


# ---------------------------------------------------------------- 본문
st.title("📈 포트폴리오 백테스터")
st.caption(
    "과거 데이터로 내 투자 방법을 미리 시험해 보세요. 한국·미국 주식과 ETF를 섞어서 넣을 수 있어요. "
    "왼쪽에서 설정을 바꾸면 결과가 바로 다시 계산돼요."
)
tab_result, tab_terms, tab_help = st.tabs(["📊 결과", "📖 용어 사전", "❓ 사용법 · 주의사항"])


def render_results() -> None:
    if weights_pct.empty:
        st.info("왼쪽 표에 종목코드와 비중을 입력해 주세요.")
        return
    if abs(total_pct - 100) > 1e-6:
        st.error(f"비중 합계가 100%가 되어야 해요. 지금은 {total_pct:g}%예요.")
        return
    if start >= end:
        st.error("시작일이 종료일보다 앞이어야 해요.")
        return

    tickers = list(weights_pct.index)
    bench = bench_ticker or None
    load_list = tickers + ([bench] if bench and bench not in tickers else [])
    with st.spinner("가격 데이터를 불러오는 중..."):
        try:
            data = cached_prices(tuple(load_list), start, end, currency)
        except Exception as e:  # 네트워크 오류 등
            st.error(f"데이터를 불러오지 못했어요: {e}")
            return

    missing = [t for t in data.missing if t in tickers]
    if data.prices.empty and len(load_list) > 1:
        st.error(
            "가격 데이터를 하나도 받지 못했어요. 인터넷 연결을 확인하거나 잠시 후 다시 시도해 주세요. "
            "(Yahoo Finance가 일시적으로 응답하지 않을 때가 있어요)"
        )
        return
    if missing:
        st.error(
            f"다음 종목의 데이터를 찾지 못했어요: **{', '.join(missing)}**. "
            "종목코드를 확인해 주세요. (한국은 6자리 숫자, 미국은 영문 티커)"
        )
        return
    if bench and bench in data.missing:
        st.warning(f"비교 대상 **{bench}**의 데이터를 찾지 못해 비교 없이 계산했어요.")
        bench = None

    prices = data.prices[tickers + ([bench] if bench and bench not in tickers else [])]
    if len(prices) < 2:
        st.error("선택한 종목들이 함께 거래된 기간이 없어요. 기간이나 종목을 바꿔 보세요.")
        return

    actual_start = prices.index[0]
    if (actual_start - pd.Timestamp(start)).days > 10:
        starts = {display_name(t): data.first_dates[t] for t in prices.columns}
        if data.fx_first_date is not None:
            starts["환율(원/달러)"] = data.fx_first_date
        late = max(starts, key=starts.get)
        st.info(
            f"📅 **{late}** 데이터가 {starts[late]:%Y년 %m월 %d일}부터 있어서, "
            f"모든 종목을 공평하게 비교하려고 **{actual_start:%Y년 %m월 %d일}부터** 계산했어요."
        )
    if len(prices) < 60:
        st.warning("기간이 3개월도 안 돼서 숫자가 크게 흔들릴 수 있어요. 기간을 늘려 보세요.")

    fee, rf = fee_pct / 100, rf_pct / 100
    weights = {t: w / 100 for t, w in weights_pct.items()}
    values = run_backtest(prices, weights, rebalance, initial, fee)
    m = compute_metrics(values, rf, initial)
    bench_values = bm = None
    if bench:
        bench_values = run_backtest(prices, {bench: 1.0}, "none", initial, fee)
        bm = compute_metrics(bench_values, rf, initial)
    bname = display_name(bench) if bench else None
    vs = f"{short_name(bench)} 대비" if bench else None
    theme = current_theme()

    def versus(mine: float | None, theirs: float | None, fmt) -> str | None:
        """비교 대상과의 차이. 비교할 값이 없으면 None."""
        if mine is None or theirs is None:
            return None
        return fmt(mine - theirs)

    def bench_value(attr: str) -> float | None:
        return getattr(bm, attr) if bm else None

    st.markdown(
        f"**{m.start:%Y.%m.%d} ~ {m.end:%Y.%m.%d}** ({duration((m.end - m.start).days)}) · "
        f"리밸런싱 {REBALANCE_LABELS[rebalance].split(' ')[0]} · 기준 통화 {CURRENCY_LABELS[currency]}"
        + (" · 환율 반영" if data.fx_first_date is not None else "")
    )

    # 핵심 숫자
    c = st.columns(4)
    c[0].metric(
        "최종 금액", money(m.final, currency),
        versus(m.final, bench_value("final"), lambda d: money(d, currency)),
        delta_description=vs, help=HELP["final"], border=True,
    )
    c[1].metric(
        "연평균 수익률 (CAGR)", pct(m.cagr), versus(m.cagr, bench_value("cagr"), pp),
        delta_description=vs, help=HELP["cagr"], border=True,
    )
    c[2].metric(
        "최대 낙폭 (MDD)", pct(m.max_drawdown), versus(m.max_drawdown, bench_value("max_drawdown"), pp),
        delta_description=vs, help=HELP["mdd"], border=True,
    )
    c[3].metric(
        "샤프 비율", ratio(m.sharpe), versus(m.sharpe, bench_value("sharpe"), lambda d: f"{d:+.2f}"),
        delta_description=vs, help=HELP["sharpe"], border=True,
    )

    c = st.columns(4)
    c[0].metric(
        "총 수익률", pct(m.total_return, signed=True),
        versus(m.total_return, bench_value("total_return"), lambda d: pp(d, 0)),
        delta_description=vs, help=HELP["total_return"], border=True,
    )
    c[1].metric(
        "변동성", pct(m.volatility), versus(m.volatility, bench_value("volatility"), pp),
        delta_color="inverse", delta_description=vs, help=HELP["volatility"], border=True,
    )
    longest = m.longest_drawdown
    c[2].metric(
        "가장 긴 회복 기간", duration(longest.days(m.end)) if longest else "없음",
        "아직 회복 중" if longest and longest.recovery is None else None,
        delta_color="off", delta_arrow="off", help=HELP["recovery"], border=True,
    )
    c[3].metric(
        "최악의 해", pct(m.worst_year[1], signed=True) if m.worst_year else "-",
        f"{m.worst_year[0]}년" if m.worst_year else None,
        delta_color="off", delta_arrow="off", help=HELP["worst_year"], border=True,
    )
    if bm:
        st.caption(
            f"작은 글씨는 내 포트폴리오에서 비교 대상 {bname}의 값을 뺀 차이예요. 초록색이면 내 포트폴리오가 더 나은 거예요. "
            "제목 옆 (?) 아이콘에 마우스를 올리면 (휴대폰은 누르면) 뜻과 예시가 나와요."
        )

    # 맞춤 설명
    st.subheader("📖 내 결과 쉽게 읽기")
    st.caption("위 숫자들을 내 결과에 맞춰 풀어 썼어요. 더 자세한 뜻은 '용어 사전' 탭에 있어요.")
    items = explain(m, currency, rf, bm, bname)
    cols = st.columns(2)
    for i, (title, body) in enumerate(items):
        with cols[i % 2].container(border=True):
            st.markdown(f"**{title}**")
            st.markdown(body)

    # 그래프
    st.subheader("자산이 어떻게 불어났나요?")
    st.caption(
        "선이 오른쪽 위로 갈수록 좋아요. 중간에 푹 꺼진 곳이 '낙폭'이에요. 그래프에 마우스를 올리면 그날의 금액이 보여요."
    )
    log_scale = st.toggle(
        "로그 스케일로 보기",
        help="기간이 길면 뒤쪽 금액이 커서 앞쪽 움직임이 납작하게 보여요. 로그 스케일에서는 "
        "'10% 오름'이 언제든 같은 높이로 보여서 기간 전체를 공평하게 볼 수 있어요.",
    )
    st.plotly_chart(charts.growth_chart(values, bench_values, bname, currency, theme, log_scale))

    st.subheader("고점 대비 얼마나 빠져 있었나요? (낙폭)")
    st.caption(
        "0%는 '최고점 그대로'라는 뜻이고, −20%는 '최고점보다 20% 빠진 상태'예요. "
        "파란 영역이 깊고 넓을수록 견디기 힘든 시기였어요."
    )
    st.plotly_chart(charts.drawdown_chart(values, m, bench_values, bname, theme))

    st.subheader("연도별 수익률")
    st.caption("막대가 0% 선 위면 그해 이익, 아래면 손해예요. 첫해와 마지막 해는 1년이 다 안 될 수 있어요.")
    st.plotly_chart(charts.annual_chart(values, bench_values, bname, theme))

    # 종목별 비교
    st.subheader("종목별로 따로 투자했다면?")
    st.caption(
        "각 종목에 돈을 전부 넣었을 때와 비교해 보세요. 여러 종목을 섞었을 때 MDD(최대 낙폭)가 줄어드는 것이 "
        "'분산 투자'의 효과예요."
    )
    table = [("⭐ 내 포트폴리오", "100%", m)]
    if bm and bench not in tickers:
        table.append((f"비교 대상: {bname}", "-", bm))
    for t in tickers:
        alone = compute_metrics(run_backtest(prices, {t: 1.0}, "none", initial, fee), rf, initial)
        table.append((display_name(t), f"{weights_pct[t]:g}%", alone))
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "이름": name,
                    "비중": w,
                    "연평균 수익률": pct(x.cagr),
                    "최대 낙폭": pct(x.max_drawdown),
                    "변동성": pct(x.volatility),
                    "샤프 비율": ratio(x.sharpe),
                    "최종 금액": money(x.final, currency),
                }
                for name, w, x in table
            ]
        ),
        hide_index=True,
    )

    with st.expander("📋 그래프 데이터를 표로 보기 · 내려받기"):
        frame = pd.DataFrame({"내 포트폴리오": values, "내 포트폴리오 낙폭": drawdown(values)})
        if bench_values is not None:
            frame[bname] = bench_values
            frame[f"{bname} 낙폭"] = drawdown(bench_values)
        monthly = frame.resample("ME").last()
        monthly.index = monthly.index.strftime("%Y-%m")
        st.caption("월말 기준 평가 금액과 낙폭")
        st.dataframe(
            monthly.style.format({col: "{:,.0f}" if "낙폭" not in col else "{:.1%}" for col in monthly.columns})
        )
        st.download_button(
            "일별 데이터 CSV 내려받기",
            frame.to_csv(index_label="날짜").encode("utf-8-sig"),
            file_name="backtest.csv",
            mime="text/csv",
        )


with tab_result:
    render_results()

with tab_terms:
    st.markdown("어려운 말을 쉬운 말과 예시로 풀었어요. 결과 화면의 숫자 제목 옆 (?) 아이콘에도 짧은 설명이 있어요.")
    cols = st.columns(2)
    for i, term in enumerate(TERMS):
        with cols[i % 2].container(border=True):
            st.markdown(f"#### {term.name}")
            st.markdown(f"**{term.summary}**")
            if term.analogy:
                st.markdown(f"🧩 **비유** — {term.analogy}")
            st.markdown(f"🔢 **예시** — {term.example}")
            if term.note:
                st.markdown(f"💡 {term.note}")

with tab_help:
    st.markdown(
        """
#### 사용법
1. **왼쪽 메뉴**에서 예시 포트폴리오를 고르거나 '**직접 만들기**'를 선택하세요. (휴대폰에서는 왼쪽 위 `»` 버튼)
2. 표에 **종목코드**와 **비중**(%)을 넣으세요. 비중 합계는 100%여야 해요.
   - 한국 종목·ETF: 6자리 코드 (예: `069500` KODEX 200, `005930` 삼성전자)
   - 미국 종목·ETF: 영문 티커 (예: `SPY`, `QQQ`, `AAPL`)
   - 행을 추가하려면 표 아래 `+`, 지우려면 행을 선택하고 휴지통 아이콘을 누르세요.
3. **기간, 투자금, 리밸런싱 주기**를 정하세요.
4. **결과** 탭에서 숫자와 그래프를 보세요. 모르는 말은 (?) 아이콘에 마우스를 올리거나 **용어 사전** 탭을 보세요.

#### 이런 걸 시험해 보세요
- 'S&P500 하나만'과 '주식 60 : 채권 40'의 **최대 낙폭** 차이
- 같은 포트폴리오에서 **리밸런싱 '안 함' vs '매년'** 비교
- 기준 통화를 **원화 ↔ 달러**로 바꿔서 환율이 수익에 준 영향 보기
"""
    )
    st.markdown("#### 꼭 알아두세요")
    for caveat in CAVEATS:
        st.markdown(f"- {caveat}")
