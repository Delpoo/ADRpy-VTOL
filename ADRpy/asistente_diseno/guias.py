# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Mapping
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import ipywidgets as w

from .datos import to_numeric_locale
from .mplutils import f2

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


# ----------------------------------------------------------------------
# Tooltips unificados (movidos desde guias_tooltips.py)
# ----------------------------------------------------------------------
HELP: dict[str, str] = {
    # Panel izquierdo (ranking / similitud)
    "panel_similitud": "El ranking de similitud ordena TODAS las aeronaves según su cercanía al objetivo (no filtra).",
    "panel_topk": "Top-K toma los K más parecidos del ranking y calcula sugerencias/estadísticos robustos sobre ese subconjunto.",
    "panel_similitud_help": "Similitud (ranking): ordena todas las aeronaves por cercanía al objetivo, sin filtrar el dataset.",
    "panel_topk_help": "Top-K (vecinos): usa sólo los K más cercanos para estadísticas y sugerencias robustas (IQR, medianas, ponderaciones).",
    "df_original": "Incluye todas las aeronaves del dataset (sin filtrar por las restricciones actuales).",
    "df_filtrado": "Sólo aeronaves que cumplen la selección actual (p. ej., máximos/mínimos, rangos u objetivos ± tolerancia).",
    "modo_param": (
        "Cómo usar cada parámetro en la comparación: • ignorar: no participa • "
        "mínimo/máximo: actúa como restricción blanda • fijo: busca cercanía al valor indicado."
    ),
    "valor_param": "Valor objetivo del parámetro (se usa si Modo=fijo/máximo/mínimo).",
    "peso_param": "Peso relativo del parámetro al combinar similitudes (1=neutral).",
    "alpha_sim": "α (agregación): α=1 promedio; α>1 penaliza desvíos grandes; α<1 suaviza.",
    "penalizar_nan": "Si hay NaN en un parámetro, agrega penalidad para no favorecer filas incompletas.",
    "penalidad_nan": "Cuánto sumar/restar a la distancia cuando hay NaN (sólo si Penalizar NaN está activo).",
    "segmentar_por": "Corte del dataset para trabajar por segmentos (p.ej. misión). (ninguno)=global.",
    "modo_global_familia": "Global: un único set. Por misión: calcula por cada segmento con n>=min_n.",
    "top_k": "Cantidad de vecinos más parecidos a considerar (lista Top-K).",
    "factor_prefer": "Multiplicador opcional para priorizar un segmento (si corresponde).",
    "iqr_on": "Quitar atípicos IQR antes de calcular ajustes/métricas.",
    "iqr_factor": "Factor IQR (1.5 típico). Mayor=recorta menos.",
    "min_n": "Mínimo de muestras válidas para ajustar una curva/métrica en cada segmento.",
    "auto": "Si está ON, recalcula automáticamente al cambiar un control.",
    # Ranking y tabla de resultados
    "ranking_sim": "Similitud: 1 es idéntico al objetivo, valores mayores indican peor ajuste (según agregación α y pesos).",
    "ranking_dist": "Distancia agregada entre parámetros (antes de normalización a similitud). Útil para diagnóstico.",
    "ranking_alerta": "Alerta en la fila 'Objetivo (usuario)': indica si el objetivo queda fuera del IQR de los Top‑K para ese parámetro (LOW/HIGH).",
    "segmentar_valor": "Valor específico del segmento cuando 'Segmentar por' está activo. Se muestran etiquetas legibles si hay mapeo.",
    "info_button": "Botón 'i': abre un panel con distribución, outliers, objetivo y valor sugerido para ese parámetro.",
    "report_button": "Generar informe: crea informe_diseno.md/html con resumen de objetivos, sugeridos, cobertura IQR y ranking.",
    # Tendencias (X–Y)
    "t_x": "Variable X (independiente) para la nube y la curva tendencia.",
    "t_y": "Variable Y (dependiente) para la nube y la curva tendencia.",
    "t_logx": "Escala logarítmica en el eje X (útil si X tiene varias órdenes de magnitud).",
    "t_obj_line": "Línea vertical con el valor objetivo de X (si está definido).",
    "r2_adj": (
        "R² ajustado: penaliza la complejidad del modelo (n vs parámetros). "
        "Se calcula como 1 - (1-R²)*(n-1)/(n-p-1); más alto es mejor."
    ),
    "dispersion_indicator": (
        "Indicador de dispersión: diagnóstico rápido de variabilidad relativa en la nube/vecinos. "
        "Útil para interpretar la confiabilidad de tendencias o sugerencias."
    ),
    # Sugerencias Top-K (histograma & box)
    "suger_box": (
        "Box overlay: rango intercuartil (Q1–Q3). Bigotes: hasta 1.5×IQR. Puntos fuera: atípicos."
    ),
    "suger_obj_line": "Línea vertical del valor objetivo actual del parámetro.",
    "suger_low_high": "LOW/HIGH (IQR): límites internos del box (Q1 y Q3). La mediana se indica como línea central.",
    "suger_w_mediana": "w_mediana: mediana ponderada por similitud (Top‑K más parecidos pesan más).",
    "suger_n_efectivo": (
        "n_efectivo: suma de pesos normalizados (0..1) de los vecinos usados; "
        "se interpreta como 'cantidad equivalente' de vecinos útiles."
    ),
    "suger_pesos": (
        "Pesos: w_dist proviene de un kernel acotado en [0,1] (p.ej., 1/(1+d) o exp(-γ·d)); "
        "w_conf en [0,1]; w_total=(w_dist^βdist)*(w_conf^βconf)."
    ),
    # Outliers
    "out_iqr_explain": "IQR=Q3−Q1; se marcan atípicos fuera de [Q1−k·IQR, Q3+k·IQR].",
    # Panel de detalle e informe
    "panel_info": "Panel de detalle del parámetro: muestra distribución, outliers y valores objetivo/sugeridos con acciones de navegación.",
    "narrativa": "Informe narrativo: resumen en Markdown/HTML con tablas y explicaciones listo para compartir.",
    # Glosario por tablas (encabezados y columnas)
    "col_dv": (
        "dv_*: aporte de la columna a la distancia total (ya normalizado por escala robusta e incluido el peso)."
    ),
    "col_viol": (
        "viol_*: indicador de violación de la restricción (True si está fuera de la regla definida para el parámetro)."
    ),
    "ranking_cols": (
        "Ranking: distancia/similitud (y sus medias) resumen la concordancia global con el objetivo; 'alerta' marca objetivos fuera de LOW/HIGH(IQR)."
    ),
    "neighbors_gloss": (
        "Top‑K vecinos: w_total combina w_dist (kernel de distancia 0..1) y w_conf (0..1). Se ordenan por w_total; 'valor' es el del parámetro."
    ),
    # Tendencias: guías orientativas
    "r2_adj_ranges": (
        "R²_ajustado (orientativo): ≥0.8 alto, 0.5–0.8 medio, <0.5 bajo. Interpretar junto con n y dispersión."
    ),
    "mape_guide": (
        "MAPE (orientativo): <10% bueno, 10–20% aceptable, >20% alto (posible ruido o no linealidad)."
    ),
}


# Helper mínimo para aplicar tooltip a un widget sin romper si el backend no lo soporta
def apply_tooltip(
    widget: w.Widget,
    key_or_text: str,
    mapping: Mapping[str, str] | None = None,
) -> w.Widget:
    """Asigna un tooltip usando una clave del diccionario HELP o un texto directo."""

    if widget is None:
        return widget

    src: Mapping[str, str] | None = HELP if mapping is None else mapping
    txt = ""
    if src is not None:
        txt = src.get(key_or_text, "")
    if not txt:
        txt = str(key_or_text)

    try:
        if hasattr(widget, "description_tooltip"):
            setattr(widget, "description_tooltip", txt)
        elif hasattr(widget, "tooltip"):
            setattr(widget, "tooltip", txt)
    except Exception:
        # fallback no-op si el widget no soporta tooltip
        pass
    return widget


__all__ = ["widget_info_param", "HELP", "apply_tooltip"]


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


def _serie_numerica(df: pd.DataFrame, col: str) -> pd.Series:
    """Devuelve la columna convertida a numérico usando reglas locale-aware."""
    if not isinstance(df, pd.DataFrame) or col not in df.columns:
        return pd.Series(dtype=float)
    try:
        serie = to_numeric_locale(df[col])
    except Exception:
        serie = pd.Series(df[col])
    serie = pd.to_numeric(serie, errors="coerce")
    try:
        return serie.astype(float)
    except Exception:
        return serie


def _stats_basicos(s: pd.Series) -> dict[str, float | int]:
    """Calcula estadísticos simples y cuenta de válidos/total."""
    serie = pd.to_numeric(s, errors="coerce")
    n_total = int(len(serie))
    serie_val = serie.dropna()
    if serie_val.empty:
        return {
            "min": np.nan,
            "q1": np.nan,
            "mediana": np.nan,
            "media": np.nan,
            "q3": np.nan,
            "max": np.nan,
            "n_validos": 0,
            "n_total": n_total,
        }
    return {
        "min": float(serie_val.min()),
        "q1": float(serie_val.quantile(0.25)),
        "mediana": float(serie_val.median()),
        "media": float(serie_val.mean()),
        "q3": float(serie_val.quantile(0.75)),
        "max": float(serie_val.max()),
        "n_validos": int(len(serie_val)),
        "n_total": n_total,
    }


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
    s = to_numeric_locale(df[col])
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
    s = to_numeric_locale(df[col]).dropna()
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
    df_filtrado: Optional[pd.DataFrame] = None,
    on_open_outliers: Optional[Callable[[str], None]] = None,
    on_open_xy: Optional[Callable[[str], None]] = None,
    on_open_sugerencias: Optional[Callable[[str], None]] = None,
) -> w.Accordion:
    # Guardas de robustez
    if col not in df.columns:
        return w.Accordion(
            children=[w.HTML(f"<i>La columna '{col}' no existe en el dataset.</i>")]
        )
    try:
        n_valid = int(to_numeric_locale(df[col]).notna().sum())
    except Exception:
        n_valid = 0
    if n_valid < 3:
        msg = f"'{col}' tiene muy pocos datos válidos para un resumen confiable (n={n_valid})."
        return w.Accordion(children=[w.HTML(f"<i>{msg}</i>")])

    obj = get_objetivo(col) if get_objetivo else None
    sug = get_sugerido(col) if get_sugerido else None
    stats = compute_param_stats(df, col, factor=factor_iqr, max_list=5)
    stats_f: Optional[ParamStats] = None
    n_filtrado = 0
    if (
        df_filtrado is not None
        and isinstance(df_filtrado, pd.DataFrame)
        and (col in df_filtrado.columns)
    ):
        try:
            n_filtrado = len(df_filtrado)
        except Exception:
            n_filtrado = 0
        if n_filtrado > 0:
            try:
                stats_f = compute_param_stats(
                    df_filtrado, col, factor=factor_iqr, max_list=5
                )
            except Exception:
                stats_f = None
    elif df_filtrado is not None:
        try:
            n_filtrado = len(df_filtrado)
        except Exception:
            n_filtrado = 0
    # alertas
    alert_emoji = "⚠️" if supports_emoji() else "(!)"
    alerta = None
    if stats.n_val >= 5 and obj is not None and np.isfinite(obj):
        if obj < stats.low or obj > stats.high:
            alerta = f"{alert_emoji} El objetivo está fuera del IQR [{fmt(stats.low)}, {fmt(stats.high)}]"

    # Sección 1: Resumen global / filtrado
    serie_global = _serie_numerica(df, col)
    stats_global = _stats_basicos(serie_global)
    pct_na_global = (
        100.0
        * (stats_global["n_total"] - stats_global["n_validos"])
        / stats_global["n_total"]
        if stats_global["n_total"]
        else np.nan
    )
    html1 = (
        "<b>Resumen (global)</b><br>"
        f"Cobertura: {stats_global['n_validos']}/{stats_global['n_total']} (NaN {f2(pct_na_global)}%)<br>"
        f"Min: {f2(stats_global['min'])} • Q1: {f2(stats_global['q1'])} • Mediana: {f2(stats_global['mediana'])} • "
        f"Media: {f2(stats_global['media'])} • Q3: {f2(stats_global['q3'])} • Max: {f2(stats_global['max'])}<br>"
        f"Rango IQR: [{fmt(stats.low)}, {fmt(stats.high)}]"
    )
    if alerta:
        html1 += f"<br><span style='color:#b30'>{alerta}</span>"
    if sug is not None and np.isfinite(sug):
        html1 += f"<br>Sugerido (Top-K): <b>{fmt(sug)}</b>"
    if obj is not None and np.isfinite(obj):
        html1 += f"<br>Objetivo: <b>{fmt(obj)}</b>"
    box_global = w.HTML(html1)

    if isinstance(df_filtrado, pd.DataFrame) and (col in df_filtrado.columns):
        serie_filtrada = _serie_numerica(df_filtrado, col)
    else:
        serie_filtrada = pd.Series(dtype=float)
    stats_filtrado = _stats_basicos(serie_filtrada)
    pct_na_filtrado = (
        100.0
        * (stats_filtrado["n_total"] - stats_filtrado["n_validos"])
        / stats_filtrado["n_total"]
        if stats_filtrado["n_total"]
        else np.nan
    )

    if stats_filtrado["n_total"] == 0:
        html1b = (
            "<b>Resumen (filtrado)</b><br>"
            f"Cobertura: {stats_filtrado['n_validos']}/{stats_filtrado['n_total']} (NaN {f2(pct_na_filtrado)}%)<br>"
            "<i>Sin datos filtrados disponibles.</i>"
        )
    elif stats_filtrado["n_validos"] == 0:
        diag_html = ""
        try:
            if isinstance(df_filtrado, pd.DataFrame) and (col in df_filtrado.columns):
                s_raw = df_filtrado[col]
                n_nonnull = int(s_raw.notna().sum())
                s_num = to_numeric_locale(s_raw)
                n_numeric = int(s_num.notna().sum())
                if n_nonnull > 0 and n_numeric == 0:
                    ejemplos = s_raw.dropna().astype(str).unique().tolist()[:5]
                    ejemplos_txt = ", ".join(ejemplos) if ejemplos else "—"
                    diag_html = (
                        "<br><small><i>Diagnóstico:</i> "
                        f"{n_nonnull} valores no nulos, pero ninguno se pudo interpretar como numérico. "
                        f"Ejemplos crudos: <code>{ejemplos_txt}</code></small>"
                    )
        except Exception:
            pass
        html1b = (
            "<b>Resumen (filtrado)</b><br>"
            f"Cobertura: {stats_filtrado['n_validos']}/{stats_filtrado['n_total']} (NaN {f2(pct_na_filtrado)}%)<br>"
            "<i>Sin datos numéricos válidos en el subconjunto.</i>"
            f"{diag_html}"
        )
    else:
        rango_iqr = (
            f"[{fmt(stats_f.low)}, {fmt(stats_f.high)}]"
            if stats_f is not None and stats_f.n_val > 0
            else "[—, —]"
        )
        html1b = (
            "<b>Resumen (filtrado)</b><br>"
            f"Cobertura: {stats_filtrado['n_validos']}/{stats_filtrado['n_total']} (NaN {f2(pct_na_filtrado)}%)<br>"
            f"Min: {f2(stats_filtrado['min'])} • Q1: {f2(stats_filtrado['q1'])} • Mediana: {f2(stats_filtrado['mediana'])} • "
            f"Media: {f2(stats_filtrado['media'])} • Q3: {f2(stats_filtrado['q3'])} • Max: {f2(stats_filtrado['max'])}<br>"
            f"Rango IQR: {rango_iqr}"
        )
    box_filtrado = w.HTML(html1b)

    try:
        tooltip_original = "Todo el dataset, sin restricciones."
        tooltip_filtrado = (
            "Subconjunto que cumple la selección (mín./máx./rango/fijo/objetivo)."
        )
    except Exception:
        tooltip_original = ""
        tooltip_filtrado = ""

    columnas = w.HBox(
        [
            w.VBox(
                [w.HTML("<b>DataFrame original</b>"), box_global],
                layout=w.Layout(width="50%"),
            ),
            w.VBox(
                [w.HTML("<b>DataFrame filtrado</b>"), box_filtrado],
                layout=w.Layout(width="50%"),
            ),
        ],
        layout=w.Layout(width="100%", gap="12px"),
    )
    try:
        columnas.children[0].children[0].tooltip = tooltip_original
        columnas.children[1].children[0].tooltip = tooltip_filtrado
    except Exception:
        pass

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
        children=[w.VBox([columnas]), sec2, w.VBox([out_fig, w.HTML("<hr>"), accesos])]
    )
    acc.set_title(0, f"Resumen global/filtrado: {col}")
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
