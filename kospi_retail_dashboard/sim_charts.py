"""KOSPI 가격 시뮬레이터용 Plotly 차트.

실제 구간은 실선, 가상 구간은 점선 + 옅은 주황 배경으로 구분한다.
지표마다 별도 패널을 두고 x축(날짜)만 공유한다 (이중 y축을 쓰지 않음).
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from charts import DOWN, UP

PRICE = "#2a78d6"  # 실제 KOSPI
HYPO = "#eb6834"  # 가상 경로
MA_COLORS = {5: "#1baf7a", 20: "#eda100", 60: "#e87ba4", 120: "#6250d6"}
BB_FILL = "rgba(42,120,214,0.10)"
BB_LINE = "rgba(42,120,214,0.35)"
GUIDE = "rgba(128,128,128,0.6)"
BAND_LINE = "rgba(128,128,128,0.75)"
WEEKENDS = [dict(bounds=["sat", "mon"])]


def _split_line(fig, df, col, name, color, row, width=1.6, fmt=",.2f", legend=True, group=None):
    """실제 구간은 실선, 가상 구간(마지막 실제 점부터)은 같은 색 점선으로 그린다."""
    group = group or name
    hypo = df["hypo"].to_numpy()
    real = df.loc[~hypo, col]
    fig.add_trace(
        go.Scatter(x=real.index, y=real, name=name, legendgroup=group, showlegend=legend,
                   mode="lines", line=dict(color=color, width=width),
                   hovertemplate=f"%{{y:{fmt}}}"),
        row=row, col=1,
    )
    if hypo.any():
        first = int(hypo.argmax())
        seg = df[col].iloc[max(first - 1, 0):]
        fig.add_trace(
            go.Scatter(x=seg.index, y=seg, name=f"{name} (가상)", legendgroup=group, showlegend=False,
                       mode="lines", line=dict(color=color, width=width, dash="dot"),
                       hovertemplate=f"%{{y:{fmt}}}"),
            row=row, col=1,
        )


def _shade_hypo(fig, df, rows: int):
    if not df["hypo"].any():
        return
    hypo_idx = df.index[df["hypo"].to_numpy()]
    last_real = df.index[~df["hypo"].to_numpy()][-1]
    x0 = last_real + (hypo_idx[0] - last_real) / 2
    for r in range(1, rows + 1):
        fig.add_vrect(x0=x0, x1=hypo_idx[-1] + pd.Timedelta(hours=12), fillcolor=HYPO, opacity=0.08,
                      line_width=0, layer="below", row=r, col=1)
    fig.add_annotation(x=x0, y=1, xref="x", yref="y domain", text="가상 구간", showarrow=False,
                       xanchor="left", yanchor="top", font=dict(size=11, color=HYPO))


def _end_marker(fig, df, row=None):
    end = df.iloc[-1]
    grid = dict(row=row, col=1) if row else {}
    fig.add_trace(
        go.Scatter(x=[df.index[-1]], y=[end["close"]], mode="markers+text", showlegend=False,
                   marker=dict(size=10, color=HYPO, line=dict(width=2, color="white")),
                   text=[f"{end['close']:,.0f}"], textposition="middle right",
                   textfont=dict(size=12, color=HYPO), hoverinfo="skip"),
        **grid,
    )


def build_technical_chart(sim_df: pd.DataFrame, window: int = 250, height: int = 860) -> go.Figure:
    """가격(+이동평균·볼린저) / RSI / MACD / 스토캐스틱 4단 차트."""
    n_hypo = int(sim_df["hypo"].sum())
    df = sim_df.iloc[-(window + n_hypo):]
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.035,
        row_heights=[0.46, 0.17, 0.2, 0.17],
        subplot_titles=("KOSPI · 이동평균 · 볼린저밴드(20, 2)", "RSI (14)",
                        "MACD (12, 26, 9)", "스토캐스틱 슬로우 (14, 3, 3)"),
    )

    # 1) 가격
    fig.add_trace(go.Scatter(x=df.index, y=df["bb_up"], line=dict(color=BB_LINE, width=1),
                             name="볼린저밴드", legendgroup="bb", hovertemplate="%{y:,.2f}"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["bb_low"], line=dict(color=BB_LINE, width=1),
                             fill="tonexty", fillcolor=BB_FILL, name="볼린저 하단", legendgroup="bb",
                             showlegend=False, hovertemplate="%{y:,.2f}"), row=1, col=1)
    for n, color in MA_COLORS.items():
        _split_line(fig, df, f"ma{n}", f"{n}일선", color, 1, width=1.3)
    real = df[~df["hypo"]]
    fig.add_trace(go.Scatter(x=real.index, y=real["close"], name="KOSPI", mode="lines",
                             line=dict(color=PRICE, width=2.2), hovertemplate="%{y:,.2f}"), row=1, col=1)
    if n_hypo:
        seg = df["close"].iloc[-(n_hypo + 1):]
        fig.add_trace(go.Scatter(x=seg.index, y=seg, name="가상 경로", mode="lines",
                                 line=dict(color=HYPO, width=2.6), hovertemplate="%{y:,.2f}"),
                      row=1, col=1)
        _end_marker(fig, df, row=1)

    # 2) RSI
    _split_line(fig, df, "rsi", "RSI", PRICE, 2, fmt=".1f", legend=False)
    for y in (70, 30):
        fig.add_hline(y=y, line=dict(color=GUIDE, width=1, dash="dash"), row=2, col=1)

    # 3) MACD
    hist = df["macd_hist"]
    colors = [UP if v >= 0 else DOWN for v in hist.fillna(0)]
    opacity = [0.45 if h else 0.85 for h in df["hypo"]]
    fig.add_trace(go.Bar(x=df.index, y=hist, name="MACD 오실레이터", showlegend=False,
                         marker=dict(color=colors, opacity=opacity), hovertemplate="%{y:+.2f}"),
                  row=3, col=1)
    _split_line(fig, df, "macd", "MACD", PRICE, 3, fmt="+.2f", legend=False)
    _split_line(fig, df, "macd_signal", "시그널", "#6250d6", 3, fmt="+.2f", legend=False)

    # 4) 스토캐스틱
    _split_line(fig, df, "stoch_k", "%K", PRICE, 4, fmt=".1f", legend=False)
    _split_line(fig, df, "stoch_d", "%D", "#6250d6", 4, fmt=".1f", legend=False)
    for y in (80, 20):
        fig.add_hline(y=y, line=dict(color=GUIDE, width=1, dash="dash"), row=4, col=1)

    _shade_hypo(fig, df, 4)
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    fig.update_yaxes(range=[0, 100], row=4, col=1)
    fig.update_xaxes(rangebreaks=WEEKENDS, showspikes=True, spikemode="across", spikethickness=1)
    fig.update_layout(
        height=height, hovermode="x unified", bargap=0.15,
        margin=dict(l=10, r=60, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0),
    )
    for ann in fig.layout.annotations[:4]:  # 패널 제목은 왼쪽 정렬
        ann.update(x=0, xanchor="left", font=dict(size=13))
    return fig


def build_band_chart(
    sim_df: pd.DataFrame,
    per_point: pd.Series,
    multiples: list[float],
    metric_label: str,
    start=None,
    height: int = 460,
) -> go.Figure:
    """PER·PBR 밴드: 1포인트당 이익(또는 순자산) × 배수 선들 위에 KOSPI 와 가상 경로를 그린다.

    per_point: 날짜별 EPS 또는 BPS (가상 구간은 마지막 값을 이어 쓴다).
    """
    df = sim_df if start is None else sim_df.loc[pd.Timestamp(start):]
    base = per_point.reindex(df.index).ffill().bfill()
    fig = go.Figure()
    for m in multiples:
        y = base * m
        fig.add_trace(go.Scatter(x=df.index, y=y, mode="lines", name=f"{metric_label} {m:g}배",
                                 line=dict(color=BAND_LINE, width=1, dash="dot"),
                                 hovertemplate="%{y:,.0f}", showlegend=False))
        fig.add_annotation(x=df.index[-1], y=y.iloc[-1], text=f"{m:g}배", showarrow=False,
                           xanchor="left", xshift=6, font=dict(size=11))
    real = df[~df["hypo"]]
    fig.add_trace(go.Scatter(x=real.index, y=real["close"], name="KOSPI", mode="lines",
                             line=dict(color=PRICE, width=2), hovertemplate="%{y:,.2f}"))
    n_hypo = int(df["hypo"].sum())
    if n_hypo:
        seg = df["close"].iloc[-(n_hypo + 1):]
        fig.add_trace(go.Scatter(x=seg.index, y=seg, name="가상 경로", mode="lines",
                                 line=dict(color=HYPO, width=2.6), hovertemplate="%{y:,.2f}"))
        _end_marker(fig, df)
    fig.update_xaxes(rangebreaks=WEEKENDS)
    fig.update_layout(
        height=height, hovermode="x unified", margin=dict(l=10, r=60, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        yaxis_title="KOSPI (pt)",
    )
    return fig
