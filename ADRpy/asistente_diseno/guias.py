# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import ipywidgets as w
from IPython.display import display, clear_output

# Reusamos IQR y config
from .outliers import compute_iqr_bounds

try:
    from . import config as _cfg

    SEGMENT_COL = getattr(_cfg, "SEGMENT_COL", "Misión")
    SEGMENT_LABELS = getattr(_cfg, "SEGMENT_LABELS", {})
    NAME_COL = getattr(_cfg, "NAME_COL", None)
except Exception:
    SEGMENT_COL, SEGMENT_LABELS, NAME_COL = "Misión", {}, None


# ------------------------ helpers generales ------------------------ #
def _resolve_name_col(df: pd.DataFrame) -> Optional[str]:
    if NAME_COL and NAME_COL in df.columns:
        return NAME_COL
    for c in ["Aeronave", "Nombre", "UAV", "Modelo", "Name"]:
        if c in df.columns:
            return c
    return None


def supports_emoji() -> bool:
    # Heurística simple: Windows 10/11 con fuentes modernas suele OK; dejamos toggle
    return True


def fmt(v, nd=2):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "—"
    if isinstance(v, (int, np.integer)):
        return f"{int(v)}"
    return f"{float(v):.{nd}f}"


@dataclass
class ParamStats:
    n_total: int
    n_val: int
    pct_nan: float
    vmin: float
    q1: float
    median: float
    mean: float
    q3: float
    vmax: float
    iqr: float
    low: float
    high: float
    out_low: List[Tuple[str, float]]  # (nombre, valor)
    out_high: List[Tuple[str, float]]


def compute_param_stats(
    df: pd.DataFrame, col: str, factor: float = 1.5, max_list: int = 5
) -> ParamStats:
    s = pd.to_numeric(df[col], errors="coerce")
    n_total = len(s)
    n_val = int(s.notna().sum())
    pct_nan = 0.0 if n_total == 0 else 100.0 * (n_total - n_val) / n_total
    if n_val == 0:
        return ParamStats(
            n_total,
            0,
            pct_nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            [],
            [],
        )
    info = compute_iqr_bounds(s, factor=factor, min_n=5)
    name_col = _resolve_name_col(df)
    names = (
        df[name_col].astype(str)
        if name_col
        else pd.Series([str(i) for i in df.index], index=df.index)
    )
    out_low = names[s < info["low"]].head(max_list).index.tolist()
    out_high = names[s > info["high"]].head(max_list).index.tolist()
    # recuperar nombres+valores
    low_list = (
        [(names.loc[i], float(s.loc[i])) for i in out_low] if len(out_low) else []
    )
    high_list = (
        [(names.loc[i], float(s.loc[i])) for i in out_high] if len(out_high) else []
    )
    return ParamStats(
        n_total=n_total,
        n_val=n_val,
        pct_nan=pct_nan,
        vmin=float(np.nanmin(s)),
        q1=float(info["Q1"]),
        median=float(np.nanmedian(s)),
        mean=float(np.nanmean(s)),
        q3=float(info["Q3"]),
        vmax=float(np.nanmax(s)),
        iqr=float(info["IQR"]),
        low=float(info["low"]),
        high=float(info["high"]),
        out_low=low_list,
        out_high=high_list,
    )


# ------------------------ plotly mini-chart ------------------------ #
def fig_param_distribution(
    df: pd.DataFrame,
    col: str,
    objetivo: Optional[float] = None,
    sugerido: Optional[float] = None,
    low: Optional[float] = None,
    high: Optional[float] = None,
) -> go.Figure:
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    fig = go.Figure()
    if len(s) == 0:
        fig.update_layout(height=240)
        fig.add_annotation(text="Sin datos numéricos", x=0.5, y=0.5, showarrow=False)
        return fig
    # hist
    fig.add_trace(go.Histogram(x=s, name="Distribución", opacity=0.7))
    # box horizontal
    fig.add_trace(go.Box(x=s, name="Box", boxmean=True, orientation="h"))
    # líneas LOW/HIGH
    if low is not None:
        fig.add_vline(x=low, line_width=1.5, line_dash="dot")
    if high is not None:
        fig.add_vline(x=high, line_width=1.5, line_dash="dot")
    # objetivo/sugerido
    if objetivo is not None and np.isfinite(objetivo):
        fig.add_vline(x=float(objetivo), line_width=2.2)
    if sugerido is not None and np.isfinite(sugerido):
        fig.add_vline(x=float(sugerido), line_width=2.2)
    fig.update_layout(
        height=300,
        margin=dict(l=50, r=10, t=30, b=40),
        showlegend=False,
        title=f"Distribución: {col}",
        xaxis_title=col,
        yaxis_title="",
    )
    return fig


# ------------------------ widget “i” por parámetro ------------------------ #
def widget_info_param(
    df: pd.DataFrame,
    col: str,
    *,
    get_objetivo: Optional[Callable[[str], Optional[float]]] = None,
    get_sugerido: Optional[Callable[[str], Optional[float]]] = None,
    factor_iqr: float = 1.5,
    on_open_outliers: Optional[Callable[[str], None]] = None,
    on_open_xy: Optional[Callable[[str], None]] = None,
    on_open_sugerencias: Optional[Callable[[str], None]] = None,
) -> w.Accordion:
    obj = get_objetivo(col) if get_objetivo else None
    sug = get_sugerido(col) if get_sugerido else None
    stats = compute_param_stats(df, col, factor=factor_iqr, max_list=5)
    # alertas
    alert_emoji = "⚠️" if supports_emoji() else "(!)"
    alerta = None
    if stats.n_val >= 5 and obj is not None and np.isfinite(obj):
        if obj < stats.low or obj > stats.high:
            alerta = f"{alert_emoji} El objetivo está fuera del IQR [{fmt(stats.low)}, {fmt(stats.high)}]"

    # Sección 1: Resumen
    html1 = f"""
    <b>{col}</b><br>
    Cobertura: {stats.n_val}/{stats.n_total} (NaN {fmt(stats.pct_nan,1)}%)<br>
    Mín: {fmt(stats.vmin)} • Q1: {fmt(stats.q1)} • Mediana: {fmt(stats.median)} • Media: {fmt(stats.mean)} • Q3: {fmt(stats.q3)} • Máx: {fmt(stats.vmax)}
    """
    if alerta:
        html1 += f"<br><span style='color:#b30'>{alerta}</span>"
    if sug is not None and np.isfinite(sug):
        html1 += f"<br>Sugerido (Top-K): <b>{fmt(sug)}</b>"
    if obj is not None and np.isfinite(obj):
        html1 += f"<br>Objetivo: <b>{fmt(obj)}</b>"
    box1 = w.HTML(html1)

    # Sección 2: Outliers (resumen)
    def _list_outs(lst: List[Tuple[str, float]]) -> str:
        if not lst:
            return "(ninguno)"
        return ", ".join([f"{n}: {fmt(v)}" for n, v in lst])

    html2 = f"""
    <b>Atípicos (IQR×{factor_iqr}):</b><br>
    Bajos: {_list_outs(stats.out_low)}<br>
    Altos: {_list_outs(stats.out_high)}<br>
    Rango IQR confiable: [{fmt(stats.low)}, {fmt(stats.high)}]
    """
    btn_out = w.Button(description="Ver en Outliers", icon="search")
    if on_open_outliers:
        btn_out.on_click(lambda _: on_open_outliers(col))
    sec2 = w.VBox([w.HTML(html2), btn_out])

    # Sección 3: Distribución (Plotly)
    fig = fig_param_distribution(
        df, col, objetivo=obj, sugerido=sug, low=stats.low, high=stats.high
    )
    try:
        import plotly.io as pio

        out_fig = w.Output()
        with out_fig:
            clear_output(wait=True)
            pio.show(fig)
    except Exception:
        out_fig = w.HTML("<i>No se pudo renderizar el gráfico Plotly.</i>")

    # Accesos a X–Y y Sugerencias
    btn_xy = w.Button(description="Abrir X–Y", icon="line-chart")
    btn_sug = w.Button(description="Ir a Sugerencias", icon="bullseye")
    if on_open_xy:
        btn_xy.on_click(lambda _: on_open_xy(col))
    if on_open_sugerencias:
        btn_sug.on_click(lambda _: on_open_sugerencias(col))
    accesos = w.HBox([btn_xy, btn_sug])

    acc = w.Accordion(
        children=[w.VBox([box1]), sec2, w.VBox([out_fig, w.HTML("<hr>"), accesos])]
    )
    acc.set_title(0, "Resumen")
    acc.set_title(1, "Rangos & atípicos")
    acc.set_title(2, "Distribución & accesos")
    acc.selected_index = 0
    return acc
