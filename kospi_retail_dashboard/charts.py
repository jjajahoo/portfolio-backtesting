"""Plotly 차트 생성. KOSPI 를 맨 위에 두고 지표 패널을 같은 x축으로 아래에 쌓는다.

두 개의 y축을 겹치는 대신(오해 소지), 지표마다 별도 패널을 두어 시점만 맞춘다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from indicators import rolling_zscore

# 한국 시장 관례: 상승/순매수 = 빨강, 하락/순매도 = 파랑
UP = "#e34948"
DOWN = "#2a78d6"
COLORS = {
    "kospi_close": "#2a78d6",
    "retail_net_roll": "#eb6834",
    "retail_net_cum": "#eb6834",
    "retail_volume": "#1baf7a",
    "retail_share": "#1baf7a",
    "deposit": "#008300",
    "credit": "#6250d6",
    "credit_kospi": "#e87ba4",
    "credit_to_deposit": "#eda100",
    "sentiment": "#e87ba4",
}

TRILLION = 1e12  # 조원
HUNDRED_MILLION = 1e8  # 억원


@dataclass
class Panel:
    title: str
    needs: tuple[str, ...]
    draw: Callable[[go.Figure, pd.DataFrame, int, dict], None]
    height: float = 1.0


def _line(fig, x, y, name, color, row, unit, dash=None, fill=None):
    fig.add_trace(
        go.Scatter(
            x=x, y=y, name=name, mode="lines", fill=fill,
            line=dict(color=color, width=2, dash=dash),
            hovertemplate=f"%{{y:,.2f}}{unit}",
        ),
        row=row, col=1,
    )


def _draw_net(fig, df, row, opt):
    net = df["retail_net"] / HUNDRED_MILLION
    fig.add_trace(
        go.Bar(
            x=df.index, y=net, name="개인 순매수(일별)",
            marker_color=[UP if v >= 0 else DOWN for v in net.fillna(0)],
            hovertemplate="%{y:,.0f}억원",
        ),
        row=row, col=1,
    )
    if "retail_net_roll" in df:
        avg = df["retail_net_roll"] / opt["flow_window"] / HUNDRED_MILLION
        _line(fig, df.index, avg, f"{opt['flow_window']}일 평균", COLORS["retail_net_roll"], row, "억원")


def _draw_cum(fig, df, row, opt):
    _line(fig, df.index, df["retail_net_cum"] / TRILLION, "개인 누적 순매수",
          COLORS["retail_net_cum"], row, "조원", fill="tozeroy")


def _draw_volume(fig, df, row, opt):
    fig.add_trace(
        go.Bar(
            x=df.index, y=df["retail_volume"] / 1e6, name="개인 거래량(매수+매도)",
            marker_color=COLORS["retail_volume"], hovertemplate="%{y:,.0f}백만주",
        ),
        row=row, col=1,
    )


def _draw_share(fig, df, row, opt):
    _line(fig, df.index, df["retail_share"], "개인 거래대금 비중", COLORS["retail_share"], row, "%")


def _draw_deposit(fig, df, row, opt):
    _line(fig, df.index, df["deposit"] / TRILLION, "투자자예탁금", COLORS["deposit"], row, "조원")


def _draw_credit(fig, df, row, opt):
    if "credit" in df:
        _line(fig, df.index, df["credit"] / TRILLION, "신용융자(전체)", COLORS["credit"], row, "조원")
    if "credit_kospi" in df and df["credit_kospi"].notna().any():
        _line(fig, df.index, df["credit_kospi"] / TRILLION, "신용융자(유가증권)",
              COLORS["credit_kospi"], row, "조원", dash="dot")


def _draw_ratio(fig, df, row, opt):
    _line(fig, df.index, df["credit_to_deposit"], "신용융자/예탁금", COLORS["credit_to_deposit"], row, "%")


def _draw_sentiment(fig, df, row, opt):
    for level in (1, -1):
        fig.add_hline(y=level, line=dict(color="#898781", width=1, dash="dot"), row=row, col=1)
    fig.add_hline(y=0, line=dict(color="#c3c2b7", width=1), row=row, col=1)
    _line(fig, df.index, df["sentiment"], "개인 투자심리 지수", COLORS["sentiment"], row, "")


PANELS: dict[str, Panel] = {
    "개인 순매수(일별)": Panel("개인 순매수 (억원) · 막대 빨강 순매수/파랑 순매도, 선 = 이동평균", ("retail_net",), _draw_net),
    "개인 누적 순매수": Panel("개인 누적 순매수 (조원)", ("retail_net_cum",), _draw_cum),
    "개인 거래량": Panel("개인 거래량 (백만주, 매수+매도)", ("retail_volume",), _draw_volume),
    "개인 거래대금 비중": Panel("개인 거래대금 비중 (%)", ("retail_share",), _draw_share),
    "투자자예탁금": Panel("투자자예탁금 (조원)", ("deposit",), _draw_deposit),
    "신용융자 잔고": Panel("신용융자 잔고 (조원) · 실선 전체, 점선 유가증권", ("credit",), _draw_credit),
    "신용융자/예탁금 비율": Panel("신용융자 / 예탁금 (%)", ("credit_to_deposit",), _draw_ratio),
    "개인 투자심리 지수": Panel("개인 투자심리 지수 (z, ±1 점선)", ("sentiment",), _draw_sentiment),
}


def available_panels(df: pd.DataFrame) -> list[str]:
    def ok(col):
        if col == "credit":  # 전체 또는 유가증권 중 하나만 있어도 그린다
            return any(c in df and df[c].notna().any() for c in ("credit", "credit_kospi"))
        return col in df and df[col].notna().any()

    return [k for k, p in PANELS.items() if all(ok(c) for c in p.needs)]


def _holiday_breaks(index: pd.DatetimeIndex) -> list:
    if len(index) < 2:
        return []
    weekdays = pd.bdate_range(index.min(), index.max())
    missing = weekdays.difference(index)
    return [dict(bounds=["sat", "mon"]), dict(values=missing.strftime("%Y-%m-%d").tolist())]


def build_main_chart(
    df: pd.DataFrame,
    panels: list[str],
    candle: bool = False,
    log_scale: bool = False,
    flow_window: int = 20,
) -> go.Figure:
    panels = [p for p in panels if p in PANELS]
    has_kospi = "kospi_close" in df and df["kospi_close"].notna().any()
    titles = (["KOSPI"] if has_kospi else []) + [PANELS[p].title for p in panels]
    n = len(titles)
    if n == 0:
        return go.Figure()
    heights = ([2.2] if has_kospi else []) + [1.0] * len(panels)
    fig = make_subplots(
        rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.035 if n > 4 else 0.06,
        subplot_titles=titles, row_heights=heights,
    )
    row = 1
    if has_kospi:
        ohlc = all(c in df and df[c].notna().any() for c in ("kospi_open", "kospi_high", "kospi_low"))
        if candle and ohlc:
            fig.add_trace(
                go.Candlestick(
                    x=df.index, open=df["kospi_open"], high=df["kospi_high"],
                    low=df["kospi_low"], close=df["kospi_close"], name="KOSPI",
                    increasing=dict(line=dict(color=UP), fillcolor=UP),
                    decreasing=dict(line=dict(color=DOWN), fillcolor=DOWN),
                ),
                row=1, col=1,
            )
        else:
            _line(fig, df.index, df["kospi_close"], "KOSPI", COLORS["kospi_close"], 1, "pt")
        if log_scale:
            fig.update_yaxes(type="log", row=1, col=1)
        row = 2
    opt = {"flow_window": flow_window}
    for p in panels:
        PANELS[p].draw(fig, df, row, opt)
        row += 1

    fig.update_layout(
        height=int(280 * heights[0] / 1.6 + 170 * len(panels)) + 80,
        hovermode="x unified",
        showlegend=False,  # 패널 제목이 계열 이름을 대신한다
        margin=dict(l=10, r=10, t=30, b=10),
        bargap=0.15,
    )
    fig.update_xaxes(rangebreaks=_holiday_breaks(df.index), showspikes=True,
                     spikemode="across", spikethickness=1, spikedash="dot")
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_yaxes(tickformat=",")
    for ann in fig.layout.annotations:  # 패널 제목은 왼쪽 정렬
        ann.update(x=0, xanchor="left", font=dict(size=13))
    return fig


COMPARE_SERIES = {
    "kospi_close": "KOSPI",
    "retail_net_cum": "개인 누적 순매수",
    "retail_share": "개인 거래대금 비중",
    "deposit": "투자자예탁금",
    "credit": "신용융자(전체)",
    "credit_to_deposit": "신용융자/예탁금",
}


def build_compare_chart(df: pd.DataFrame, cols: list[str], mode: str, z_window: int) -> go.Figure:
    """서로 단위가 다른 지표를 하나의 축에서 비교: 시작=100 지수화 또는 z-score."""
    fig = go.Figure()
    for c in cols:
        s = df[c].dropna()
        if s.empty:
            continue
        if mode == "index":
            base = s.iloc[0]
            if base <= 0 or (s <= 0).any():  # 음수가 생기는 시계열은 지수화가 의미 없음
                continue
            y, unit = s / base * 100, ""
        else:
            y, unit = rolling_zscore(s, z_window), "σ"
        fig.add_trace(
            go.Scatter(
                x=y.index, y=y, name=COMPARE_SERIES.get(c, c), mode="lines",
                line=dict(color=COLORS.get(c, "#898781"), width=3 if c == "kospi_close" else 2),
                hovertemplate=f"%{{y:,.1f}}{unit}",
            )
        )
    ref = 100 if mode == "index" else 0
    fig.add_hline(y=ref, line=dict(color="#c3c2b7", width=1))
    fig.update_layout(
        height=460, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis_title="기간 시작 = 100" if mode == "index" else f"{z_window}일 z-score",
    )
    fig.update_xaxes(rangebreaks=_holiday_breaks(df.index))
    return fig


def build_corr_heatmap(corr: pd.DataFrame) -> go.Figure:
    """상관계수 표: 파랑(−) ↔ 회색(0) ↔ 빨강(+)."""
    fig = go.Figure(
        go.Heatmap(
            z=corr.values, x=list(corr.columns), y=list(corr.index),
            zmin=-1, zmax=1, zmid=0,
            colorscale=[[0, DOWN], [0.5, "#f0efec"], [1, UP]],
            text=corr.map(lambda v: f"{v:+.2f}").values, texttemplate="%{text}",
            hovertemplate="%{y} × %{x}: %{z:+.2f}<extra></extra>", xgap=2, ygap=2,
        )
    )
    fig.update_layout(height=80 + 48 * len(corr), margin=dict(l=10, r=10, t=10, b=10))
    fig.update_yaxes(autorange="reversed")
    return fig
