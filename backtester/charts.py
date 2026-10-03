"""결과 그래프 (Plotly).

내 포트폴리오는 파란색, 비교 대상(벤치마크)은 회색으로 그려 내 결과가 먼저 눈에 들어오게 한다.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from .engine import Metrics, drawdown, period_returns
from .formatting import money, pct

_THEMES = {
    "light": {
        "portfolio": "#2a78d6",
        "bench": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "text": "#52514e",
        "fill": "rgba(42, 120, 214, 0.10)",
    },
    "dark": {
        "portfolio": "#3987e5",
        "bench": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "text": "#c3c2b7",
        "fill": "rgba(57, 135, 229, 0.12)",
    },
}


def _base_layout(fig: go.Figure, c: dict, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=36, b=8),
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", color=c["text"]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, linecolor=c["axis"], ticks="", title=None)
    fig.update_yaxes(gridcolor=c["grid"], gridwidth=1, zeroline=False, linecolor=c["axis"], title=None)
    return fig


def growth_chart(
    values: pd.Series,
    bench: pd.Series | None,
    bench_name: str | None,
    currency: str,
    theme: str = "light",
    log_scale: bool = False,
) -> go.Figure:
    c = _THEMES.get(theme, _THEMES["light"])
    # 원화는 '만 원' 단위로 그려 축 숫자를 짧게 만든다
    scale, unit = (10_000, "만 원") if currency == "KRW" else (1, "달러")
    fig = go.Figure()
    series = [("내 포트폴리오", values, c["portfolio"])]
    if bench is not None:
        series.append((bench_name or "벤치마크", bench, c["bench"]))
    for name, s, color in reversed(series):  # 내 포트폴리오를 맨 위에 그린다
        fig.add_trace(
            go.Scatter(
                x=s.index,
                y=s / scale,
                name=name,
                mode="lines",
                line=dict(color=color, width=2),
                customdata=[money(v, currency) for v in s],
                hovertemplate="%{customdata}<extra>" + name + "</extra>",
            )
        )
    _base_layout(fig, c)
    fig.update_layout(legend_traceorder="reversed")
    fig.update_yaxes(
        type="log" if log_scale else "linear",
        tickformat=",.0f",
        title=dict(text=f"평가 금액 ({unit})", font=dict(size=12)),
    )
    return fig


def drawdown_chart(
    values: pd.Series,
    metrics: Metrics,
    bench: pd.Series | None,
    bench_name: str | None,
    theme: str = "light",
) -> go.Figure:
    c = _THEMES.get(theme, _THEMES["light"])
    fig = go.Figure()
    if bench is not None:
        fig.add_trace(
            go.Scatter(
                x=bench.index,
                y=drawdown(bench),
                name=bench_name or "벤치마크",
                mode="lines",
                line=dict(color=c["bench"], width=1.5),
                hovertemplate="%{y:.1%}<extra>" + (bench_name or "벤치마크") + "</extra>",
            )
        )
    dd = drawdown(values)
    fig.add_trace(
        go.Scatter(
            x=dd.index,
            y=dd,
            name="내 포트폴리오",
            mode="lines",
            line=dict(color=c["portfolio"], width=2),
            fill="tozeroy",
            fillcolor=c["fill"],
            hovertemplate="%{y:.1%}<extra>내 포트폴리오</extra>",
        )
    )
    p = metrics.max_drawdown_period
    if p is not None:
        fig.add_annotation(
            x=p.trough,
            y=p.depth,
            text=f"최대 낙폭 {pct(p.depth)}<br>({p.trough:%Y-%m-%d})",
            showarrow=True,
            arrowhead=0,
            arrowcolor=c["text"],
            ax=40,
            ay=-10,
            xanchor="left",
            font=dict(size=12, color=c["text"]),
        )
    _base_layout(fig, c, height=300)
    fig.update_layout(legend_traceorder="reversed")
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    return fig


def yearly_returns(values: pd.Series) -> pd.Series:
    yearly = period_returns(values, "YE")
    yearly.index = yearly.index.year
    return yearly


def annual_chart(
    values: pd.Series,
    bench: pd.Series | None,
    bench_name: str | None,
    theme: str = "light",
) -> go.Figure:
    c = _THEMES.get(theme, _THEMES["light"])
    fig = go.Figure()
    series = [("내 포트폴리오", values, c["portfolio"])]
    if bench is not None:
        series.append((bench_name or "벤치마크", bench, c["bench"]))
    for name, s, color in series:
        yearly = yearly_returns(s)
        fig.add_trace(
            go.Bar(
                x=yearly.index.astype(str),
                y=yearly,
                name=name,
                marker=dict(color=color, line=dict(width=0)),
                hovertemplate="%{y:+.1%}<extra>" + name + "</extra>",
            )
        )
    _base_layout(fig, c, height=320)
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08, barcornerradius=3)
    fig.update_xaxes(type="category")
    fig.update_yaxes(tickformat=".0%", zeroline=True, zerolinecolor=c["axis"], zerolinewidth=1)
    return fig
