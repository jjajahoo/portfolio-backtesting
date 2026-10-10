"""Plotly 차트. 배경·글꼴은 Streamlit 테마에 맡기고 선/막대 색만 지정한다."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .stats import ALL, GROUP_BASE, GROUP_EVENT, GROUP_HIT


@dataclass(frozen=True)
class Palette:
    series: tuple[str, ...]
    ink2: str
    muted: str
    shade: str
    shade_key: str  # 범례 견본 (음영보다 진하게)
    band: str


LIGHT = Palette(
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"),
    ink2="#52514e",
    muted="#898781",
    shade="rgba(237, 161, 0, 0.18)",
    shade_key="rgba(237, 161, 0, 0.55)",
    band="rgba(137, 135, 129, 0.10)",
)
DARK = Palette(
    series=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"),
    ink2="#c3c2b7",
    muted="#898781",
    shade="rgba(201, 133, 0, 0.22)",
    shade_key="rgba(201, 133, 0, 0.65)",
    band="rgba(137, 135, 129, 0.14)",
)

# 가격 위 보조선 색: 지표별로 고정 (켜고 끄는 순서와 무관하게 같은 색)
OVERLAY_SLOTS = {"ma_gap": (1,), "ma_cross": (2, 6)}


def palette(theme: str) -> Palette:
    return DARK if theme == "dark" else LIGHT


@dataclass
class Subplot:
    title: str
    series: pd.Series
    hlines: list[float] = field(default_factory=list)
    yrange: tuple[float, float] | None = None


def main_chart(
    close: pd.Series,
    overlays: dict[str, dict[str, pd.Series]],
    subplots: list[Subplot],
    shade: pd.DataFrame | None,
    split_date: pd.Timestamp | None,
    log_y: bool,
    theme: str,
) -> go.Figure:
    pal = palette(theme)
    k = len(subplots)
    heights = [0.55] + [0.45 / k] * k if k else [1.0]
    fig = make_subplots(
        rows=1 + k, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=heights,
        subplot_titles=["QQQ 수정주가 (배당 재투자)"] + [s.title for s in subplots],
    )
    for key, lines in overlays.items():
        if key == "bb":
            _add_bollinger(fig, lines, pal)
            continue
        slots = OVERLAY_SLOTS.get(key, (3,))
        for i, (name, s) in enumerate(lines.items()):
            fig.add_trace(go.Scatter(
                x=s.index, y=s, name=name, mode="lines",
                line=dict(color=pal.series[slots[i % len(slots)]], width=1.2),
                hovertemplate="%{y:,.2f}",
            ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=close.index, y=close, name="QQQ", mode="lines",
        line=dict(color=pal.series[0], width=1.8), hovertemplate="%{y:,.2f}",
    ), row=1, col=1)

    for i, sp in enumerate(subplots, start=2):
        fig.add_trace(go.Scatter(
            x=sp.series.index, y=sp.series, name=sp.title, mode="lines", showlegend=False,
            line=dict(color=pal.series[0], width=1.4), hovertemplate="%{y:,.2f}",
        ), row=i, col=1)
        for y in sp.hlines:
            fig.add_hline(y=y, line=dict(color=pal.muted, width=1, dash="dot"), row=i, col=1)
        if sp.yrange:
            fig.update_yaxes(range=list(sp.yrange), row=i, col=1)

    shapes = []
    if shade is not None and len(shade):
        idx = close.index
        for start, end in zip(shade["start"], shade["end"]):
            pos = idx.searchsorted(end) + 1
            x1 = idx[pos] if pos < len(idx) else end + pd.Timedelta(days=1)
            shapes.append(dict(
                type="rect", xref="x", yref="paper", x0=start, x1=x1, y0=0, y1=1,
                fillcolor=pal.shade, line=dict(width=0), layer="below",
            ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name="조건 충족 구간",
            marker=dict(symbol="square", size=12, color=pal.shade_key),
        ), row=1, col=1)
    if split_date is not None:
        shapes.append(dict(
            type="line", xref="x", yref="paper", x0=split_date, x1=split_date, y0=0, y1=1,
            line=dict(color=pal.ink2, width=1),
        ))
        fig.add_annotation(
            x=split_date, y=1, xref="x", yref="paper", yanchor="bottom", xanchor="left",
            text=" 검증 구간 →", showarrow=False, font=dict(color=pal.ink2, size=11),
        )
    # add_hline이 이미 넣은 기준선 뒤에 이어 붙인다 (update_layout(shapes=)는 기존 도형과 섞여 버림)
    fig.layout.shapes = (*fig.layout.shapes, *(go.layout.Shape(**sh) for sh in shapes))
    fig.update_layout(
        height=430 + 170 * k,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1),
    )
    if log_y:
        fig.update_yaxes(type="log", row=1, col=1)
    fig.update_xaxes(
        rangeselector=dict(buttons=[
            dict(count=1, label="1년", step="year", stepmode="backward"),
            dict(count=3, label="3년", step="year", stepmode="backward"),
            dict(count=5, label="5년", step="year", stepmode="backward"),
            dict(count=10, label="10년", step="year", stepmode="backward"),
            dict(step="all", label="전체"),
        ], x=0, y=1.03, yanchor="bottom"),
        row=1, col=1,
    )
    return fig


def _add_bollinger(fig: go.Figure, lines: dict[str, pd.Series], pal: Palette) -> None:
    upper, mid, lower = lines["볼린저 상단"], lines["볼린저 중심"], lines["볼린저 하단"]
    fig.add_trace(go.Scatter(
        x=upper.index, y=upper, name="볼린저 상단", mode="lines", legendgroup="bb", showlegend=False,
        line=dict(color=pal.muted, width=0.8), hovertemplate="%{y:,.2f}",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=lower.index, y=lower, name="볼린저 하단", mode="lines", legendgroup="bb", showlegend=False,
        line=dict(color=pal.muted, width=0.8), fill="tonexty", fillcolor=pal.band,
        hovertemplate="%{y:,.2f}",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=mid.index, y=mid, name="볼린저밴드", mode="lines", legendgroup="bb",
        line=dict(color=pal.muted, width=0.8, dash="dot"), hovertemplate="%{y:,.2f}<extra>볼린저 중심</extra>",
    ), row=1, col=1)


def condition_bars(table: pd.DataFrame, parts: list[str], theme: str) -> go.Figure:
    """기간별 평균 이후 수익률: 조건 해당일 / 구간 시작일 / 전체."""
    pal = palette(theme)
    groups = [(GROUP_HIT, pal.series[0]), (GROUP_EVENT, pal.series[1]), (GROUP_BASE, pal.muted)]
    fig = make_subplots(
        rows=1, cols=len(parts), shared_yaxes=True,
        subplot_titles=parts if parts != [ALL] else None,
    )
    for ci, part in enumerate(parts, start=1):
        for g, color in groups:
            sub = table[(table["표본구간"] == part) & (table["그룹"] == g)]
            fig.add_trace(go.Bar(
                x=sub["기간"], y=sub["평균(%)"], name=g, legendgroup=g, showlegend=ci == 1,
                marker=dict(color=color, line=dict(width=0)),
                customdata=sub[["표본수", "승률(%)"]].to_numpy(),
                hovertemplate=f"<b>{g}</b> %{{x}}<br>평균 %{{y:.2f}}%<br>"
                              "표본 %{customdata[0]:,}<br>승률 %{customdata[1]:.1f}%<extra></extra>",
            ), row=1, col=ci)
    fig.update_layout(
        barmode="group", bargap=0.3, bargroupgap=0.1, height=360,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="평균 이후 수익률(%)", zeroline=True, row=1, col=1)
    return fig


def bucket_bars(table: pd.DataFrame, horizon: str, parts: list[str], theme: str) -> go.Figure:
    """점수 구간별 평균 이후 수익률 (조정/검증 구간 나란히)."""
    pal = palette(theme)
    fig = go.Figure()
    for i, part in enumerate(parts):
        sub = table[(table["기간"] == horizon) & (table["표본구간"] == part)]
        fig.add_trace(go.Bar(
            x=sub["점수 구간"], y=sub["평균(%)"], name=part,
            marker=dict(color=pal.series[i], line=dict(width=0)),
            customdata=sub[["표본수", "승률(%)"]].to_numpy(),
            hovertemplate=f"<b>{part}</b> 점수 %{{x}}<br>평균 %{{y:.2f}}%<br>"
                          "표본 %{customdata[0]:,}<br>승률 %{customdata[1]:.1f}%<extra></extra>",
        ))
    fig.update_layout(
        barmode="group", bargap=0.3, bargroupgap=0.1, height=340,
        margin=dict(l=10, r=10, t=30, b=10), showlegend=len(parts) > 1,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="종합점수 구간", yaxis_title=f"평균 {horizon} 이후 수익률(%)",
    )
    return fig
