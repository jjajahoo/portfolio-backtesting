"""QQQ 점수 실험실.

나스닥100(QQQ)을 기술적·금리·경기·가치평가 지표로 점수화하고,
조건에 해당했던 구간을 차트에 표시한 뒤 그 이후 수익률을 통계로 보여준다.
"""
from __future__ import annotations

import datetime as dt
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import streamlit as st

from qqq_score import charts, data, presets, stats
from qqq_score import conditions as cond_mod
from qqq_score.conditions import OPS, Condition
from qqq_score.indicators import (
    BY_KEY, CATEGORIES, HIGH_GOOD, INDICATORS, LOW_GOOD, PERCENTILE, RANGE,
)
from qqq_score.scoring import EngineResult, IndicatorSetting, default_settings, run

DEFAULT_CONDITIONS = [Condition("raw:fed_rate", ">=", 4.0)]
DEFAULT_OPTIONS = {
    "split_on": True,
    "split_date": "2016-01-01",
    "require_all": True,
    "merge_gap": 5,
    "log_scale": True,
    "show_overlays": True,
    "subplots": ["raw:fed_rate", "raw:rsi", "composite"],
}
METHOD_LABELS = {PERCENTILE: "백분위", RANGE: "구간"}
DIRECTION_LABELS = {LOW_GOOD: "낮을수록 고득점", HIGH_GOOD: "높을수록 고득점"}
HOW_LABELS = {"and": "모두 만족 (AND)", "or": "하나라도 (OR)"}
PCT = st.column_config.NumberColumn(format="%.2f")
PCT1 = st.column_config.NumberColumn(format="%.1f")
COUNT = st.column_config.NumberColumn(format="%d")


# ---------------------------------------------------------------- 데이터


@st.cache_data(ttl=3600, show_spinner=False)
def cached_prices(ticker: str) -> pd.DataFrame:
    return data.load_prices(ticker)


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def cached_fred_all(ids: tuple[str, ...]) -> tuple[dict[str, pd.Series], list[str]]:
    out, errors = {}, []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {sid: ex.submit(data.load_fred, sid) for sid in ids}
    for sid, fut in futures.items():
        try:
            out[sid] = fut.result()
        except Exception as exc:
            errors.append(f"FRED {sid}: {exc}")
    return out, errors


@st.cache_data(ttl=55, show_spinner=False)
def cached_quote(ticker: str) -> dict:
    return data.load_latest_quote(ticker)


def load_market_data() -> data.MarketData:
    try:
        prices = cached_prices(data.PRICE_TICKER)
    except Exception as exc:
        st.error(f"QQQ 가격 데이터를 불러오지 못했습니다: {exc}")
        st.button("다시 시도", on_click=st.cache_data.clear)
        st.stop()
    errors = []
    try:
        trend = cached_prices(data.TREND_TICKER)
    except Exception as exc:
        trend = None
        errors.append(f"{data.TREND_TICKER}: {exc} (장기추세 괴리는 QQQ로 대신 계산)")
    fred, fred_errors = cached_fred_all(tuple(data.FRED_SERIES))
    custom = None
    upload = st.session_state.get("custom_csv")
    if upload is not None:
        try:
            custom = data.parse_custom_csv(upload.getvalue())
        except Exception as exc:
            errors.append(f"사용자 CSV: {exc}")
    return data.MarketData(prices, trend, fred, custom, errors + fred_errors)


# ---------------------------------------------------------------- 상태


def split_date_bounds() -> tuple[dt.date, dt.date]:
    return dt.date(2002, 1, 1), dt.date.today() - dt.timedelta(days=365)


def k(ind_key: str, name: str) -> str:
    return f"ind.{ind_key}.{name}"


def pk(ind_key: str, param: str) -> str:
    return f"ind.{ind_key}.p.{param}"


def write_settings(settings: dict[str, IndicatorSetting], overwrite: bool) -> None:
    ss = st.session_state
    for key, s in settings.items():
        vals = {
            k(key, "enabled"): bool(s.enabled),
            k(key, "in_score"): bool(s.in_score),
            k(key, "weight"): float(s.weight),
            k(key, "method"): s.method,
            k(key, "direction"): s.direction,
            k(key, "window"): int(s.window_years),
            k(key, "low"): float(s.low),
            k(key, "high"): float(s.high),
        }
        for p in BY_KEY[key].params:
            v = s.params.get(p.key, p.default)
            vals[pk(key, p.key)] = int(v) if p.integer else float(v)
        for name, v in vals.items():
            if overwrite or name not in ss:
                ss[name] = v


def write_conditions(conds: list[Condition], how: str) -> None:
    ss = st.session_state
    ss["cond_ids"] = list(range(len(conds)))
    ss["cond_next"] = len(conds)
    for i, c in enumerate(conds):
        _write_condition_row(i, c)
    ss["cond_how"] = how


def _write_condition_row(i: int, c: Condition) -> None:
    ss = st.session_state
    ss[f"cond.{i}.operand"] = c.operand
    ss[f"cond.{i}.op"] = c.op
    ss[f"cond.{i}.v1"] = float(c.value)
    ss[f"cond.{i}.v2"] = float(c.value2)


def write_options(options: dict, overwrite: bool) -> None:
    ss = st.session_state
    merged = {**DEFAULT_OPTIONS, **options}
    try:
        merged["split_date"] = pd.Timestamp(merged["split_date"]).date()
    except (TypeError, ValueError):
        merged["split_date"] = pd.Timestamp(DEFAULT_OPTIONS["split_date"]).date()
    lo, hi = split_date_bounds()
    merged["split_date"] = min(max(merged["split_date"], lo), hi)
    merged["merge_gap"] = int(min(max(int(merged["merge_gap"]), 0), 60))
    merged["subplots"] = [str(x) for x in merged.get("subplots", [])]
    for name in DEFAULT_OPTIONS:
        if overwrite or name not in ss:
            ss[name] = merged[name]


def init_state() -> None:
    ss = st.session_state
    first = "initialized" not in ss
    write_settings(default_settings(), overwrite=first)
    write_options({}, overwrite=first)
    if first or "cond_ids" not in ss:
        write_conditions(DEFAULT_CONDITIONS, "and")
    for i in ss["cond_ids"]:
        if f"cond.{i}.operand" not in ss:
            _write_condition_row(i, Condition(cond_mod.COMPOSITE, ">=", 70.0))
        # '값 2'는 사이/바깥일 때만 보이므로 숨겨진 동안 지워질 수 있다
        ss.setdefault(f"cond.{i}.v2", 0.0)
    ss.setdefault("auto_refresh", False)
    ss["initialized"] = True


def current_settings() -> dict[str, IndicatorSetting]:
    ss = st.session_state
    out = {}
    for ind in INDICATORS:
        out[ind.key] = IndicatorSetting(
            enabled=bool(ss[k(ind.key, "enabled")]),
            in_score=bool(ss[k(ind.key, "in_score")]),
            weight=float(ss[k(ind.key, "weight")]),
            method=ss[k(ind.key, "method")],
            direction=ss[k(ind.key, "direction")],
            window_years=float(ss[k(ind.key, "window")]),
            low=float(ss[k(ind.key, "low")]),
            high=float(ss[k(ind.key, "high")]),
            params={p.key: ss[pk(ind.key, p.key)] for p in ind.params},
        )
    return out


def current_conditions() -> list[Condition]:
    ss = st.session_state
    return [
        Condition(ss[f"cond.{i}.operand"], ss[f"cond.{i}.op"], float(ss[f"cond.{i}.v1"]), float(ss[f"cond.{i}.v2"]))
        for i in ss["cond_ids"]
    ]


def current_options() -> dict:
    ss = st.session_state
    out = {name: ss[name] for name in DEFAULT_OPTIONS}
    out["split_date"] = str(ss["split_date"])
    return out


# 콜백 (위젯이 그려지기 전에 실행되므로 상태를 바꿔도 된다)


def on_preset_upload() -> None:
    f = st.session_state.get("preset_file")
    if f is None:
        return
    try:
        settings, conds, how, options = presets.from_json(f.getvalue().decode("utf-8"))
    except Exception as exc:
        st.session_state["flash"] = ("error", f"설정 파일을 읽지 못했습니다: {exc}")
        return
    write_settings(settings, overwrite=True)
    write_conditions(conds, how)
    write_options(options, overwrite=True)
    st.session_state["flash"] = ("success", f"'{f.name}' 설정을 불러왔습니다.")


def on_reset() -> None:
    write_settings(default_settings(), overwrite=True)
    write_conditions(DEFAULT_CONDITIONS, "and")
    write_options({}, overwrite=True)
    st.session_state["flash"] = ("success", "기본 설정으로 되돌렸습니다.")


def add_condition() -> None:
    ss = st.session_state
    i = ss["cond_next"]
    ss["cond_next"] = i + 1
    ss["cond_ids"] = [*ss["cond_ids"], i]
    _write_condition_row(i, Condition(cond_mod.COMPOSITE, ">=", 70.0))


def remove_condition(i: int) -> None:
    ss = st.session_state
    ss["cond_ids"] = [x for x in ss["cond_ids"] if x != i]


def on_custom_upload() -> None:
    if st.session_state.get("custom_csv") is not None:
        st.session_state[k("custom", "enabled")] = True


def refresh_data() -> None:
    st.cache_data.clear()


def esc_md(text: str) -> str:
    """마크다운에서 ~ 두 개가 취소선이 되지 않게."""
    return text.replace("~", "\\~")


def show_table(df: pd.DataFrame, max_rows: int | None = None, **kwargs) -> None:
    """빈 값은 '–'로, 높이는 행 수에 맞춰 (max_rows를 넘으면 스크롤)."""
    rows = len(df) if max_rows is None else min(len(df), max_rows)
    kwargs.setdefault("hide_index", True)
    st.dataframe(df, height=35 * (rows + 1) + 3, placeholder="–", **kwargs)


def theme_type() -> str:
    try:
        t = st.context.theme.type
    except Exception:
        return "light"
    return t if t in ("light", "dark") else "light"


# ---------------------------------------------------------------- 화면 조각


def render_sidebar_conditions(opts: dict[str, str]) -> None:
    ss = st.session_state
    st.header("조건")
    st.caption("조건에 맞는 날이 차트에 음영으로 표시되고, 그 이후 수익률을 '조건 통계' 탭에서 볼 수 있습니다.")
    st.radio("결합 방식", list(HOW_LABELS), format_func=HOW_LABELS.get, key="cond_how", horizontal=True)
    for i in list(ss["cond_ids"]):
        with st.container(border=True):
            operand_key = f"cond.{i}.operand"
            options = dict(opts)
            cur = ss.get(operand_key)
            if cur not in options:
                options[cur] = f"⚠ 꺼진 지표: {cur}"
            st.selectbox("항목", list(options), format_func=options.get, key=operand_key)
            two = ss[f"cond.{i}.op"] in ("between", "outside")
            row = st.columns(2)
            row[0].selectbox("연산", list(OPS), format_func=OPS.get, key=f"cond.{i}.op")
            row[1].number_input("값 1" if two else "값", step=0.1, format="%.2f", key=f"cond.{i}.v1")
            row = st.columns(2, vertical_alignment="bottom")
            if two:
                row[1].number_input("값 2", step=0.1, format="%.2f", key=f"cond.{i}.v2")
            row[0].button("삭제", key=f"cond.{i}.del", icon=":material/delete:", type="tertiary",
                          on_click=remove_condition, args=(i,))
            if cur not in opts:
                st.caption("이 지표가 꺼져 있어 조건에서 빠집니다. '지표 설정'에서 켜세요.")
    st.button("＋ 조건 추가", on_click=add_condition)


def render_sidebar_options() -> None:
    st.header("검증 구간 분리")
    st.toggle("조정/검증 구간 나누기", key="split_on",
              help="설정을 조정할 때 쓰는 기간과, 그 설정이 처음 보는 기간에서도 통하는지 확인하는 기간을 나눕니다.")
    lo, hi = split_date_bounds()
    st.date_input("검증 구간 시작일", key="split_date", min_value=lo, max_value=hi,
                  disabled=not st.session_state["split_on"])
    st.caption("조정 구간 통계는 이후 수익률 계산 기간이 검증 구간 시작 전에 끝나는 날만 씁니다.")

    st.header("통계 옵션")
    st.number_input("구간 병합 간격(거래일)", min_value=0, max_value=60, step=1, key="merge_gap",
                    help="조건이 이 일수 이하로 잠깐 끊겼다 다시 맞으면 같은 구간으로 봅니다. '구간 시작일' 통계와 구간 목록에 쓰입니다.")
    st.checkbox("반영 지표가 모두 있는 날만 종합점수 계산", key="require_all",
                help="끄면 데이터가 늦게 시작하는 지표는 그 전까지 빼고 나머지로 가중평균합니다.")

    st.header("시세")
    st.toggle("현재가 1분마다 새로고침", key="auto_refresh")


def render_sidebar_files() -> None:
    st.header("설정 저장 / 불러오기")
    payload = presets.to_json(current_settings(), current_conditions(), st.session_state["cond_how"], current_options())
    st.download_button("현재 설정 저장 (JSON)", payload, file_name="qqq_score_preset.json", mime="application/json")
    st.file_uploader("설정 불러오기", type=["json"], key="preset_file", on_change=on_preset_upload)
    st.button("기본 설정으로 초기화", on_click=on_reset)

    st.header("사용자 지표 CSV")
    st.file_uploader("CSV (첫 열 날짜, 둘째 열 값)", type=["csv"], key="custom_csv", on_change=on_custom_upload,
                     help="예: 나스닥100 PER. 올리면 '지표 설정 > 가치평가 > 사용자 지표'가 켜집니다.")

    st.header("데이터")
    st.button("데이터 새로 받기", on_click=refresh_data, help="캐시를 비우고 QQQ와 FRED 데이터를 다시 받습니다.")


def render_header(md: data.MarketData, result: EngineResult, hit: pd.Series | None, cond_text: str) -> None:
    st.title("QQQ 점수 실험실")
    st.caption("나스닥100(QQQ)을 여러 지표로 점수화하고, 조건에 해당했던 구간과 그 이후 수익률을 확인합니다. "
               "통계는 과거 기록일 뿐 미래 수익을 보장하지 않습니다.")
    c1, c2, c3, c4 = st.columns(4)

    def quote() -> None:
        try:
            q = cached_quote(data.PRICE_TICKER)
            st.metric("QQQ 현재가 (약 15분 지연)", f"${q['price']:,.2f}", f"{q['change_pct']:+.2f}%")
            st.caption(f"{q['time']:%m/%d %H:%M} 뉴욕 시간")
        except Exception:
            close = md.close
            st.metric("QQQ 종가", f"${close.iloc[-1]:,.2f}", f"{(close.iloc[-1] / close.iloc[-2] - 1) * 100:+.2f}%")
            st.caption(f"{close.index[-1]:%Y-%m-%d} 기준 (현재가를 받지 못함)")

    with c1:
        st.fragment(run_every="60s" if st.session_state["auto_refresh"] else None)(quote)()
    comp = result.composite.dropna()
    with c2:
        if comp.empty:
            st.metric("종합점수", "–")
        else:
            prev = comp.iloc[-22] if len(comp) > 22 else None
            st.metric("종합점수 (0~100)", f"{comp.iloc[-1]:.0f}",
                      f"{comp.iloc[-1] - prev:+.1f} (1개월 전 대비)" if prev is not None else None)
    rate = data.fed_target_rate(md.fred).dropna()
    with c3:
        st.metric("기준금리 (목표 상단)", f"{rate.iloc[-1]:.2f}%" if len(rate) else "–")
        if len(rate):
            st.caption(f"{rate.index[-1]:%Y-%m-%d} 기준")
    with c4:
        if hit is None:
            st.metric("현재 조건 충족", "조건 없음")
        else:
            st.metric("현재 조건 충족", "예" if bool(hit.iloc[-1]) else "아니오", help=esc_md(cond_text))


def render_chart_tab(md, result, opts, conds, hit, split, theme) -> None:
    ss = st.session_state
    ss["subplots"] = [x for x in ss["subplots"] if x in opts]
    c1, c2, c3 = st.columns([4, 1, 1], vertical_alignment="bottom")
    c1.multiselect("아래에 함께 볼 지표", list(opts), format_func=opts.get, key="subplots", max_selections=6)
    c2.toggle("로그 스케일", key="log_scale")
    c3.toggle("가격 보조선", key="show_overlays", help="이동평균, 볼린저밴드 등 켜진 지표의 선")

    subplots = []
    for op in ss["subplots"]:
        s = cond_mod.operand_series(result, op)
        if s is None:
            continue
        lines = []
        for c in conds:
            if c.operand == op:
                lines += [c.value, c.value2] if c.op in ("between", "outside") else [c.value]
        yrange = (0, 100) if op == cond_mod.COMPOSITE or op.startswith("score:") else None
        subplots.append(charts.Subplot(opts[op], s, sorted(set(lines)), yrange))

    shade = None
    if hit is not None:
        shade = stats.periods(hit, 0)
        if len(shade) > 400:
            shade = stats.periods(hit, max(int(ss["merge_gap"]), 5))
            st.caption(f"구간이 너무 많아 {max(int(ss['merge_gap']), 5)}거래일 이내로 끊긴 구간은 합쳐서 표시합니다.")
    overlays = result.overlays if ss["show_overlays"] else {}
    fig = charts.main_chart(md.close, overlays, subplots, shade, split.date if split.enabled else None,
                            ss["log_scale"], theme)
    st.plotly_chart(fig, key="main_chart")
    if conds and hit is not None:
        st.caption("점선은 조건에 넣은 기준값입니다. 음영은 조건을 만족한 날입니다.")


def render_stats_tab(md, hit, valid, split, cond_text, theme) -> None:
    if hit is None:
        st.info("사이드바에서 조건을 추가하면 조건에 해당한 구간의 이후 수익률 통계가 여기에 나옵니다.")
        return
    gap = int(st.session_state["merge_gap"])
    per = stats.periods(hit, gap)
    ov = stats.condition_overview(hit, valid, per)
    st.markdown(f"**조건:** {esc_md(cond_text)}")
    c = st.columns(4)
    c[0].metric("해당일", f"{ov['해당일']:,}일")
    c[1].metric("전체 대비 비율", f"{ov['비율(%)']:.1f}%" if pd.notna(ov["비율(%)"]) else "–")
    c[2].metric("구간 수", f"{ov['구간 수']:,}")
    c[3].metric("평균 구간 길이", f"{ov['평균 구간 길이(거래일)']:.0f}거래일" if ov["구간 수"] else "–")
    if ov["해당일"] == 0:
        st.warning("조건을 만족한 날이 없습니다. 기준값을 바꿔 보세요.")
        return

    table = stats.condition_table(md.close, hit, valid, split, gap)
    st.plotly_chart(charts.condition_bars(table, split.parts(), theme), key="cond_bars")
    st.caption(
        "**조건 해당일**: 조건을 만족한 모든 날 각각에 샀다고 가정(겹치는 기간이 많아 표본수가 실제보다 커 보임). "
        "**구간 시작일**: 구간마다 첫날 한 번만 샀다고 가정(더 보수적인 비교). "
        "**전체**: 조건에 쓰인 지표 값이 있는 모든 날. 평균 최대하락은 보유 기간 중 겪은 가장 큰 평가손의 평균."
    )
    cfg = {"표본수": COUNT, "평균(%)": PCT, "중앙값(%)": PCT, "승률(%)": PCT1,
           "하위10%(%)": PCT, "상위10%(%)": PCT, "평균 최대하락(%)": PCT}
    for part in split.parts():
        if split.enabled:
            st.subheader(part)
        sub = table[table["표본구간"] == part].drop(columns="표본구간")
        show_table(sub, column_config=cfg)

    st.subheader("구간 목록")
    pt = stats.period_table(md.close, per)
    pcfg = {c: PCT for c in pt.columns if c.endswith("(%)")}
    show_table(pt.iloc[::-1], max_rows=15, column_config=pcfg)


def render_score_tab(md, result, split, theme) -> None:
    if not result.weights:
        st.info("'지표 설정' 탭에서 '점수 반영'을 켠 지표가 없습니다.")
        return
    comp = result.composite
    if comp.dropna().empty:
        st.warning("종합점수를 계산할 수 있는 날이 없습니다. 반영 지표의 데이터 기간을 확인하세요.")
        return
    st.markdown(
        f"종합점수 = 반영 지표 {len(result.weights)}개의 가중평균. "
        f"계산 가능 기간 {comp.first_valid_index():%Y-%m-%d} – {comp.last_valid_index():%Y-%m-%d}"
    )
    c1, c2 = st.columns([3, 1])
    horizon = c1.radio("이후 수익률 기간", list(stats.HORIZONS), index=3, horizontal=True, key="bucket_horizon")
    n_bins = c2.number_input("점수 구간 수", 2, 10, 5, key="bucket_bins")
    table = stats.score_bucket_table(comp, md.close, split, int(n_bins))
    st.plotly_chart(charts.bucket_bars(table, horizon, split.parts(), theme), key="bucket_bars")
    st.caption("점수가 높은 구간일수록 이후 수익률이 높게 나오면 점수가 의미 있다는 신호입니다. 표본수가 적은 구간은 믿기 어렵습니다.")
    cfg = {"표본수": COUNT, "평균(%)": PCT, "중앙값(%)": PCT, "승률(%)": PCT1, "하위10%(%)": PCT, "상위10%(%)": PCT}
    sub = table[table["기간"] == horizon].drop(columns="기간")
    show_table(sub, column_config=cfg)

    st.subheader("지표별 순위상관 (IC)")
    st.caption(
        "각 점수와 이후 수익률의 순위상관입니다. +면 점수가 높을 때 수익도 높았고, 0 근처면 관계가 없었다는 뜻입니다. "
        "조정 구간에서 좋았던 지표가 검증 구간에서도 같은 부호를 유지하면 비교적 견고하고, "
        "검증 구간에서 0 근처나 반대 부호로 무너지면 과최적화를 의심해 볼 수 있습니다."
    )
    scores = {"종합점수": comp}
    scores.update({result.labels[key]: result.scores[key] for key in result.scores.columns})
    ic = stats.ic_table(scores, md.close, split)
    show_table(ic, hide_index=False, column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ic.columns})


def render_settings_tab(result: EngineResult) -> None:
    st.caption(
        "지표를 켜면 계산되어 차트·조건에 쓸 수 있고, '점수 반영'까지 켜면 종합점수에 들어갑니다. "
        "**백분위**: 과거 N년 중 현재 값의 위치(그 시점까지의 데이터만 사용). "
        "**구간**: 하한–상한 사이를 0–100점으로 선형 변환."
    )
    tabs = st.tabs(CATEGORIES)
    for cat, tab in zip(CATEGORIES, tabs):
        with tab:
            for ind in [i for i in INDICATORS if i.category == cat]:
                render_indicator_card(ind, result)


def render_indicator_card(ind, result: EngineResult) -> None:
    ss = st.session_state
    on = ss[k(ind.key, "enabled")]
    with st.container(border=True):
        c1, c2, c3 = st.columns([3, 1.2, 1.2], vertical_alignment="bottom")
        title = f"**{ind.name}**" + (f" ({ind.unit})" if ind.unit else "")
        c1.toggle(title, key=k(ind.key, "enabled"))
        c2.checkbox("점수 반영", key=k(ind.key, "in_score"), disabled=not on)
        c3.number_input("가중치", min_value=0.0, max_value=10.0, step=0.5, key=k(ind.key, "weight"),
                        disabled=not (on and ss[k(ind.key, "in_score")]))
        st.caption(esc_md(ind.description))
        if ind.key in result.errors:
            st.warning(f"계산하지 못했습니다: {result.errors[ind.key]}")
        if ind.params:
            cols = st.columns(max(len(ind.params), 3))
            for col, p in zip(cols, ind.params):
                if p.integer:
                    col.number_input(p.label, min_value=int(p.min), max_value=int(p.max), step=int(p.step),
                                     key=pk(ind.key, p.key), disabled=not on)
                else:
                    col.number_input(p.label, min_value=float(p.min), max_value=float(p.max), step=float(p.step),
                                     format="%.1f", key=pk(ind.key, p.key), disabled=not on)
        method = ss[k(ind.key, "method")]
        cols = st.columns([1, 1.3, 1, 1, 1])
        cols[0].selectbox("점수 방식", [PERCENTILE, RANGE], format_func=METHOD_LABELS.get,
                          key=k(ind.key, "method"), disabled=not on)
        cols[1].selectbox("방향", [LOW_GOOD, HIGH_GOOD], format_func=DIRECTION_LABELS.get,
                          key=k(ind.key, "direction"), disabled=not on)
        cols[2].number_input("백분위 기간(년)", min_value=0, max_value=40, step=1, key=k(ind.key, "window"),
                             disabled=not on or method != PERCENTILE, help="0이면 데이터 처음부터 누적")
        cols[3].number_input("구간 하한", step=0.1, format="%.2f", key=k(ind.key, "low"),
                             disabled=not on or method != RANGE)
        cols[4].number_input("구간 상한", step=0.1, format="%.2f", key=k(ind.key, "high"),
                             disabled=not on or method != RANGE)


def render_status_tab(md: data.MarketData, result: EngineResult, hit: pd.Series | None) -> None:
    st.subheader("지표 현재 값")
    total_w = sum(result.weights.values())
    rows = []
    for key in result.raw.columns:
        raw = result.raw[key].dropna()
        sc = result.scores[key].dropna()
        w = result.weights.get(key, 0.0)
        last_score = sc.iloc[-1] if len(sc) else float("nan")
        rows.append({
            "분류": BY_KEY[key].category,
            "지표": result.labels[key],
            "현재 값": raw.iloc[-1] if len(raw) else float("nan"),
            "단위": BY_KEY[key].unit,
            "점수": last_score,
            "가중치": w,
            "종합점수 기여": last_score * w / total_w if total_w and w else float("nan"),
        })
    show_table(pd.DataFrame(rows), column_config={"현재 값": PCT, "점수": PCT1, "가중치": PCT1, "종합점수 기여": PCT1})
    for key, err in result.errors.items():
        st.warning(f"{BY_KEY[key].name}: {err}")

    st.subheader("데이터 기준일")
    fresh = [{"데이터": "QQQ (야후 파이낸스)", "시작": md.index[0].date(), "마지막": md.index[-1].date()}]
    if md.trend_prices is not None:
        tp = md.trend_prices.index
        fresh.append({"데이터": "나스닥100 지수 ^NDX", "시작": tp[0].date(), "마지막": tp[-1].date()})
    for sid in data.FRED_SERIES:
        s = md.fred.get(sid)
        if s is not None and len(s):
            fresh.append({"데이터": f"FRED {sid}", "시작": s.index[0].date(), "마지막": s.index[-1].date()})
    show_table(pd.DataFrame(fresh))
    st.caption("경제지표는 발표 지연(물가 45일, 실업률 35일, M2 60일 등)을 반영해 실제로 알 수 있었던 날부터 씁니다.")
    for err in md.errors:
        st.warning(err)

    st.subheader("내려받기")
    out = pd.DataFrame({"QQQ 수정종가": md.close})
    for key in result.raw.columns:
        out[f"{result.labels[key]} 값"] = result.raw[key]
        out[f"{result.labels[key]} 점수"] = result.scores[key]
    out["종합점수"] = result.composite
    if hit is not None:
        out["조건 충족"] = hit
    st.download_button("계산 결과 CSV", out.to_csv(index_label="날짜").encode("utf-8-sig"),
                       file_name="qqq_score_data.csv", mime="text/csv")


# ---------------------------------------------------------------- 메인


def main() -> None:
    st.set_page_config(page_title="QQQ 점수 실험실", page_icon="📈", layout="wide")
    init_state()
    ss = st.session_state

    sb_cond = st.sidebar.container()
    sb_opts = st.sidebar.container()
    sb_files = st.sidebar.container()
    with sb_files:
        render_sidebar_files()

    flash = ss.pop("flash", None)
    if flash:
        getattr(st, flash[0])(flash[1])

    with st.spinner("데이터 불러오는 중…"):
        md = load_market_data()
    result = run(md, current_settings(), require_all=ss["require_all"])
    opts = cond_mod.operand_options(result)

    with sb_cond:
        render_sidebar_conditions(opts)
    with sb_opts:
        render_sidebar_options()

    conds = current_conditions()
    evaluated = cond_mod.evaluate(result, conds, ss["cond_how"])
    hit, valid = evaluated if evaluated is not None else (None, None)
    joiner = " 그리고 " if ss["cond_how"] == "and" else " 또는 "
    cond_text = joiner.join(cond_mod.describe(c, opts) for c in conds if c.operand in opts) or "조건 없음"
    split = stats.Split(bool(ss["split_on"]), pd.Timestamp(ss["split_date"]))
    theme = theme_type()

    render_header(md, result, hit, cond_text)
    if md.errors:
        st.warning(f"일부 데이터를 받지 못했습니다 ({len(md.errors)}건). '현재 상태' 탭에서 확인하세요.")

    tabs = st.tabs(["📈 차트", "📊 조건 통계", "🎯 점수 검증", "⚙️ 지표 설정", "📋 현재 상태"])
    with tabs[0]:
        render_chart_tab(md, result, opts, conds, hit, split, theme)
    with tabs[1]:
        render_stats_tab(md, hit, valid, split, cond_text, theme)
    with tabs[2]:
        render_score_tab(md, result, split, theme)
    with tabs[3]:
        render_settings_tab(result)
    with tabs[4]:
        render_status_tab(md, result, hit)


if __name__ == "__main__":
    main()
