# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Mapping
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import ipywidgets as w

# Reusamos IQR y config
from .outliers import compute_iqr_bounds
from .mplutils import numeric_2dec_styler, plotly_apply_2dec

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
    # Histograma (transparente) + Box horizontal
    fig.add_trace(go.Histogram(x=s, name="Distribución", opacity=0.55, nbinsx=None))
    fig.add_trace(go.Box(x=s, name="Box", boxmean=True, orientation="h"))
    # Líneas LOW/HIGH
    if low is not None:
        fig.add_vline(x=low, line_width=1.5, line_dash="dot")
    if high is not None:
        fig.add_vline(x=high, line_width=1.5, line_dash="dot")
    # Objetivo / sugerido
    if objetivo is not None and np.isfinite(objetivo):
        fig.add_vline(x=float(objetivo), line_color="#0d6efd", line_width=2)
    if sugerido is not None and np.isfinite(sugerido):
        fig.add_vline(x=float(sugerido), line_color="#198754", line_width=2)
    fig.update_layout(
        height=260,
        margin=dict(l=8, r=8, t=28, b=24),
        showlegend=False,
        template="plotly_white",
        bargap=0.07,
    )
    fig.update_xaxes(title=col)
    # Formato 2 decimales (ticks/hover) con helper común
    try:
        plotly_apply_2dec(fig)
    except Exception:
        pass
    return fig


# ------------------------ piezas de render reutilizables ------------------------ #
def render_resumen_parametro(
    df: pd.DataFrame,
    col: str,
    *,
    get_objetivo: Optional[Callable[[str], Optional[float]]] = None,
    get_sugerido: Optional[Callable[[str], Optional[float]]] = None,
    factor_iqr: float = 1.5,
) -> w.Widget:
    """Devuelve un widget con una tabla resumen estilizada a 2 decimales para 'col'."""
    obj = get_objetivo(col) if get_objetivo else None
    sug = get_sugerido(col) if get_sugerido else None
    st = compute_param_stats(df, col, factor=factor_iqr, max_list=5)
    data = {
        "n_total": [st.n_total],
        "n_val": [st.n_val],
        "%NaN": [st.pct_nan],
        "min": [st.vmin],
        "Q1": [st.q1],
        "mediana": [st.median],
        "media": [st.mean],
        "Q3": [st.q3],
        "max": [st.vmax],
        "LOW(IQR)": [st.low],
        "HIGH(IQR)": [st.high],
        "objetivo": [obj],
        "sugerido": [sug],
    }
    df_sum = pd.DataFrame(data)
    try:
        sty = numeric_2dec_styler(df_sum)
        html = sty.to_html()
        return w.HTML(html)
    except Exception:
        return w.HTML(df_sum.to_html(index=False))


def render_rangos_iqr(
    df: pd.DataFrame,
    col: str,
    *,
    factor_iqr: float = 1.5,
    max_list: int = 5,
) -> w.Widget:
    """Devuelve un widget con tabla de outliers (bajos/altos) y rangos IQR estilizados."""
    st = compute_param_stats(df, col, factor=factor_iqr, max_list=max_list)
    rows = []
    for n, v in st.out_low:
        rows.append({"tipo": "bajo", "nombre": n, "valor": v})
    for n, v in st.out_high:
        rows.append({"tipo": "alto", "nombre": n, "valor": v})
    if not rows:
        rows.append({"tipo": "—", "nombre": "(sin atípicos)", "valor": np.nan})
    df_out = pd.DataFrame(rows)
    # Agregar fila de rangos
    df_rng = pd.DataFrame(
        [{"tipo": "rango", "nombre": "IQR confiable", "valor": np.nan}]
    )
    try:
        sty = numeric_2dec_styler(df_out)
        html = sty.to_html() + "<br>" + df_rng.to_html(index=False)
        return w.HTML(html)
    except Exception:
        return w.HTML(
            df_out.to_html(index=False) + "<br>" + df_rng.to_html(index=False)
        )


def render_distribucion(
    df: pd.DataFrame,
    col: str,
    *,
    objetivo: Optional[float] = None,
    sugerido: Optional[float] = None,
    factor_iqr: float = 1.5,
) -> go.Figure:
    st = compute_param_stats(df, col, factor=factor_iqr, max_list=5)
    fig = fig_param_distribution(
        df, col, objetivo=objetivo, sugerido=sugerido, low=st.low, high=st.high
    )
    try:
        fig = plotly_apply_2dec(fig)
    except Exception:
        pass
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

    # Sección 3: Distribución (Plotly) -> HTML embebido (sin autodisplay)
    fig = fig_param_distribution(
        df, col, objetivo=obj, sugerido=sug, low=stats.low, high=stats.high
    )
    try:
        fig = plotly_apply_2dec(fig)
        html_fig = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")  # type: ignore[arg-type]
        out_fig = w.HTML(html_fig)
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


# === Tooltips unificados ===
# Export público de HELP/apply_tooltip junto con las guías
__all__ = [
    "compute_param_stats",
    "fig_param_distribution",
    "render_resumen_parametro",
    "render_rangos_iqr",
    "render_distribucion",
    "widget_info_param",
    "apply_tooltip",
    "HELP",
]

# Textos cortos y claros para tooltips de controles.
HELP: dict[str, str] = {
    # Panel izquierdo (ranking / similitud)
    "modo_param": (
        "Cómo usar cada parámetro en la comparación: • ignorar: no participa • mínimo/máximo: actúa como restricción blanda • "
        "fijo: busca cercanía al valor indicado."
    ),
    "valor_param": "Valor objetivo del parámetro (se usa si Modo=fijo/máximo/mínimo).",
    "peso_param": "Peso relativo del parámetro al combinar similitudes (1=neutral).",
    "alpha_sim": "α (agregación): α=1 promedio; α>1 penaliza desvíos grandes; α<1 suaviza.",
    "penalizar_nan": "Si hay NaN en un parámetro, agrega penalidad para no favorecer filas incompletas.",
    "penalidad_nan": "Cuánto sumar/restar a la distancia cuando hay NaN (sólo si Penalizar NaN está activo).",
    "segmentar_por": "Corte del dataset para trabajar por segmentos (p.ej. misión). (ninguno)=global.",
    "modo_global_familia": "Global: un único set. Por misión: agrupa y compara sólo dentro de cada misión.",
    "top_k": "Cantidad de vecinos más similares para calcular el sugerido.",
    # X–Y
    "t_x": "Variable en el eje X.",
    "t_y": "Variable en el eje Y.",
    "iqr_on": "Si está activo, dibuja rangos IQR y marca atípicos.",
    "iqr_factor": "Factor multiplicador del IQR (1.5 por defecto).",
    "min_n": "Mínimo de datos válidos para considerar el IQR confiable.",
    "t_logx": "Aplica log a X si procede (sólo valores positivos).",
    # Varios
    "auto": "Actualiza el panel automáticamente al cambiar opciones.",
}


def apply_tooltip(
    widget: w.Widget, key: str | None, defaults: Mapping[str, str] = HELP
) -> None:
    """Asigna tooltips de forma robusta a distintos tipos de widgets, sin autodisplay."""
    if key is None:
        return
    txt = defaults.get(key)
    if not txt:
        return
    # Widgets con 'description_tooltip' (Dropdown, Checkbox, FloatText, Sliders, etc.)
    if hasattr(widget, "description_tooltip"):
        try:
            setattr(widget, "description_tooltip", txt)
            return
        except Exception:
            pass
    # Botones y otros widgets
    if hasattr(widget, "tooltip"):
        try:
            setattr(widget, "tooltip", txt)
        except Exception:
            pass


# __all__ ya definido explícitamente arriba
