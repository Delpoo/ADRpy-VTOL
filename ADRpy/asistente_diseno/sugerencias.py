"""
Sugerencias robustas a partir del Top-K del ranking de similitud.

Uso típico
---------
1) Obtener el ranking:
    df_rank = rank(df, restricciones, ...)

2) Calcular sugerencias del Top-K:
    from asistente_diseno.sugerencias import sugerencias_topk, vista_sugerencias_resumen, widget_sugerencias_param

    sug = sugerencias_topk(
        df_ranked=df_rank,
        params=None,                 # autodetecta numéricas (o lista explícita)
        top_k=10,
        remove_outliers=True,        # toggle ON/OFF
        iqr_factor=1.5,
        use_distance_weights=True,   # 1/(distancia+eps)
        use_confidence_weights=False,# si tenés columnas de confianza
        confidence_cols=None,        # dict opcional: {param->col_conf}
        name_objetivo="Objetivo (usuario)"
    )

3) Mostrar resumen y el detalle interactivo:
    display(vista_sugerencias_resumen(sug["summary"]))
    ui = widget_sugerencias_param(sug); display(ui)

Qué hace
-------
- Trabaja sobre el ranking ya ordenado (no recalcula similitud).
- Toma los Top-K vecinos (excluyendo la fila "Objetivo (usuario)" si existe).
- Aplica filtro de outliers (IQR) por parámetro si remove_outliers=True.
- Calcula estadísticos robustos con y sin ponderación por:
    * distancia: w_dist = 1/(distancia+eps)
    * confianza: detectada por mapping {param -> columna_confianza} o ignorada
- Entrega:
    * summary (DataFrame por parámetro)
    * details (dict: param -> DataFrame con cada vecino, valor, pesos, flags)
    * config usada (para reproducibilidad)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple, Any, Callable

import numpy as np
import pandas as pd

# matplotlib se importa via helper centralizado
import plotly.graph_objects as go
import ipywidgets as w

# sin autodisplay: usar pio.to_html y widgets.HTML
import plotly.io as pio

from asistente_diseno.outliers import compute_iqr_bounds
from pandas.api.types import is_bool_dtype
from asistente_diseno.mplutils import numeric_2dec_styler, plotly_apply_2dec, fmt2

# =============================================================================
# NUEVO: utilidades y tipos para Top-K enriquecido (panel independiente)
# =============================================================================

ALERT = "⚠️"  # Si tu entorno no muestra emoji, cambia por "(!)" o "▲"


@dataclass
class TopKStats:
    n: int
    median: float
    q1: float
    q3: float
    iqr: float
    p10: float
    p90: float
    mean: float
    std: float
    cv: float  # σ / media


def _topk_stats(x: pd.Series) -> TopKStats:
    """Calcula métricas clave sobre la serie Top-K (numérica)."""
    x = pd.to_numeric(x, errors="coerce").dropna()
    n = int(x.size)
    if n == 0:
        return TopKStats(0, *(np.nan,) * 8)
    info = compute_iqr_bounds(pd.Series(x), factor=1.5, min_n=5)
    q1 = float(info.get("Q1", np.nan))
    q3 = float(info.get("Q3", np.nan))
    iqr = float(info.get("IQR", np.nan))
    mean = float(np.nanmean(x))
    std = float(np.nanstd(x, ddof=1)) if n > 1 else np.nan
    cv = (std / mean) if (n > 1 and np.isfinite(mean) and mean != 0.0) else np.nan
    return TopKStats(
        n=n,
        median=float(np.nanmedian(x)),
        q1=q1,
        q3=q3,
        iqr=iqr,
        p10=float(np.nanpercentile(x, 10)),
        p90=float(np.nanpercentile(x, 90)),
        mean=mean,
        std=std,
        cv=cv,
    )


# =============================================================================
# Pesos normalizados y helpers de render scrollable
# =============================================================================

# Kernel para pesos por distancia (evita explosiones con dist≈0 y acota a [0,1])
W_KERNEL = "one_plus"  # opciones: "one_plus" | "exp" | "inv"
EPS = 1e-6
GAMMA = 4.0  # usado si W_KERNEL == "exp"
W_MAX = 100.0  # tope duro si se usa el inverso antes de normalizar


def _w_from_dist(dist_arr: np.ndarray) -> np.ndarray:
    d = np.asarray(dist_arr, dtype=float)
    d = np.where(np.isfinite(d), d, np.nan)
    if W_KERNEL == "exp":
        w = np.exp(-float(GAMMA) * d)
    elif W_KERNEL == "inv":
        w_raw = 1.0 / np.maximum(d, float(EPS))
        # tope visual/numérico para evitar dominancias extremas aun antes de normalizar
        w_raw = np.minimum(w_raw, float(W_MAX))
        mx = np.nanmax(w_raw) if np.any(np.isfinite(w_raw)) else np.nan
        w = w_raw / mx if (mx and np.isfinite(mx) and mx > 0) else w_raw
    else:
        # por defecto: 1/(1 + d)  →  d=0 => 1.0 ; d grande => ~0
        w = 1.0 / (1.0 + d)
    # asegurar rango [0,1]
    w = np.clip(w, 0.0, 1.0)
    # NaN -> 0
    w = np.where(np.isfinite(w), w, 0.0)
    return w


def _wrap_scrollable_html(html_table: str, *, height: int = 420) -> str:
    """Render robusto: encapsula la tabla en un <iframe> con su propio CSS.

    Asegura barras de scroll visibles, encabezados sticky, primera columna fija,
    bordes, zebra y truncado de headers con ellipsis.
    """
    inner = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<style>\n"
        "html,body{margin:0;padding:8px;overflow:auto;font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Arial,sans-serif;font-size:12px;color:#111;}\n"
        ".vecinos{table-layout:auto;width:max-content !important;min-width:100%;max-width:100%;border-collapse:separate;border-spacing:0;border:1px solid #ddd;}\n"
        ".vecinos th, .vecinos td{min-width:12ch;}\n"
        ".vecinos td{white-space:nowrap;padding:6px 8px;border-right:1px solid #eee;border-bottom:1px solid #eee;}\n"
        ".vecinos th{padding:6px 8px;border-right:1px solid #eee;border-bottom:1px solid #eee;}\n"
        ".vecinos thead th{position:sticky;top:0;background:#fff;z-index:3;box-shadow:0 1px 0 rgba(0,0,0,0.08);font-weight:600;}\n"
        ".vecinos tbody tr:nth-child(odd) td{background:#fafafa;}\n"
        ".vecinos tbody tr:hover td{background:#f0f7ff;}\n"
        ".vecinos th:first-child, .vecinos td:first-child{position:sticky;left:0;background:#fff;z-index:2;box-shadow:1px 0 0 rgba(0,0,0,0.08);}\n"
        ".vecinos td:nth-child(n+2){text-align:right;}\n"
        ".vecinos .hdr{display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;white-space:normal;word-break:break-word;}\n"
        ".vecinos thead th:hover .hdr{-webkit-line-clamp:unset;max-height:none;}\n"
        "</style></head><body>"
        "<div id='tbl-wrap'>"
        f"{html_table}"
        "</div>"
        "<script>\n"
        "(function(){\n"
        " var tbl=document.querySelector('.vecinos'); if(!tbl) return;\n"
        " // Envolver texto de headers en un span para aplicar ellipsis\n"
        " tbl.querySelectorAll('thead th').forEach(function(th){\n"
        "   var txt=(th.textContent||'').trim(); th.setAttribute('title', txt);\n"
        "   var span=document.createElement('span'); span.className='hdr'; span.textContent=txt;\n"
        "   while(th.firstChild){ th.removeChild(th.firstChild);} th.appendChild(span);\n"
        " });\n"
        " // Ordenamiento simple al click del header\n"
        " function parseVal(s){ var x=parseFloat(String(s).replace(',','.')); return isNaN(x)? null : x; }\n"
        " function sortTable(tbl,col,asc){\n"
        "   var tb=tbl.tBodies[0]; if(!tb) return;\n"
        "   var rows=Array.prototype.slice.call(tb.querySelectorAll('tr'));\n"
        "   rows.sort(function(a,b){\n"
        "     var ta=(a.cells[col]&&a.cells[col].innerText||'').trim();\n"
        "     var tbv=(b.cells[col]&&b.cells[col].innerText||'').trim();\n"
        "     var na=parseVal(ta), nb=parseVal(tbv);\n"
        "     var cmp=0;\n"
        "     if(na!==null && nb!==null){ cmp = na-nb; } else { cmp = ta.localeCompare(tbv); }\n"
        "     return asc? cmp : -cmp;\n"
        "   });\n"
        "   rows.forEach(function(r){ tb.appendChild(r); });\n"
        " }\n"
        " tbl.querySelectorAll('thead th').forEach(function(th,idx){\n"
        "   th.style.cursor='pointer';\n"
        "   th.addEventListener('click', function(){\n"
        "     var asc = th.getAttribute('data-asc') !== 'true';\n"
        "     sortTable(tbl, idx, asc);\n"
        "     tbl.querySelectorAll('thead th').forEach(function(t){ t.removeAttribute('data-asc'); });\n"
        "     th.setAttribute('data-asc', asc?'true':'false');\n"
        "   });\n"
        " });\n"
        "})();\n"
        "</script>"
        "</body></html>"
    )
    # IMPORTANTE: usar comillas simples en el atributo srcdoc y escapar comillas simples dentro del HTML
    # para evitar que el contenido rompa el atributo y deje el iframe vacío.
    escaped = inner.replace("'", "&#39;")
    iframe = (
        f'<iframe style="width:100%;height:{int(height)}px;border:1px solid #eee;border-radius:4px;" '
        f"srcdoc='{escaped}' loading=\"lazy\"></iframe>"
    )
    return iframe


def _fig_topk_hist(
    x: pd.Series,
    *,
    param_label: str,
    objetivo: Optional[float],
    min_bins: int = 12,
) -> tuple[go.Figure, TopKStats, Dict[str, float]]:
    """
    Devuelve (fig, stats, iqr_info):
      - fig: histograma con overlay IQR (dentro) + líneas LOW/HIGH, Sugerido (mediana) y Objetivo.
      - stats: métricas Top-K (mediana, IQR, p10/p90, media, σ, CV, n).
      - iqr_info: dict {'low','high','usable'} coherente con compute_iqr_bounds.
    """
    x = pd.to_numeric(x, errors="coerce").dropna()
    stats = _topk_stats(x)

    iqr_info = compute_iqr_bounds(x, factor=1.5, min_n=5)
    low = float(iqr_info.get("low", np.nan))
    high = float(iqr_info.get("high", np.nan))
    usable = bool(iqr_info.get("usable", False))

    nbins = min_bins if x.nunique() >= min_bins else max(6, int(x.nunique()))
    fig = go.Figure()
    fig.add_histogram(
        x=x,
        nbinsx=nbins,
        marker=dict(color="rgba(66, 99, 255, 0.65)"),
        name="Top-K",
        hovertemplate="valor=%{x:.2f}<br>frec=%{y:.2f}<extra></extra>",
    )

    # Overlay IQR dentro del rango (si usable) + guías Low/High
    if usable and np.isfinite(low) and np.isfinite(high) and (high > low):
        fig.add_shape(
            type="rect",
            xref="x",
            yref="paper",
            x0=low,
            x1=high,
            y0=0.0,
            y1=1.0,
            line=dict(width=0),
            fillcolor="rgba(255, 99, 71, 0.18)",
            layer="below",
        )
        fig.add_vline(x=low, line=dict(color="crimson", dash="dash", width=2))
        fig.add_vline(x=high, line=dict(color="crimson", dash="dash", width=2))
        fig.add_annotation(
            x=low,
            y=1.02,
            yref="paper",
            showarrow=False,
            text="LOW(IQR)",
            font=dict(color="crimson"),
        )
        fig.add_annotation(
            x=high,
            y=1.02,
            yref="paper",
            showarrow=False,
            text="HIGH(IQR)",
            font=dict(color="crimson"),
        )

    # Línea sugerida = mediana Top-K
    if np.isfinite(stats.median):
        fig.add_vline(x=stats.median, line=dict(color="seagreen", dash="dash", width=3))
        fig.add_annotation(
            x=stats.median,
            y=1.02,
            yref="paper",
            showarrow=False,
            text="Sugerido",
            font=dict(color="seagreen"),
        )

    # Línea objetivo
    if objetivo is not None and np.isfinite(objetivo):
        fig.add_vline(x=float(objetivo), line=dict(color="royalblue", width=2))
        fig.add_annotation(
            x=float(objetivo),
            y=1.02,
            yref="paper",
            showarrow=False,
            text="Objetivo",
            font=dict(color="royalblue"),
        )

    fig.update_layout(
        title=dict(
            text=f"Distribución Top-K: {param_label}", x=0.02, y=0.94, xanchor="left"
        ),
        xaxis_title=param_label,
        yaxis_title="frecuencia",
        xaxis_title_standoff=6,
        yaxis_title_standoff=6,
        bargap=0.05,
        margin=dict(l=50, r=16, t=50, b=48),
        template="plotly_white",
        showlegend=False,
    )
    try:
        plotly_apply_2dec(fig)
    except Exception:
        pass
    return fig, stats, {"low": low, "high": high, "usable": usable}


def _guia_imputacion_html(
    param_label: str,
    stats: TopKStats,
    iqr: Dict[str, float],
    objetivo: Optional[float],
) -> w.HTML:
    """Construye una guía textual con sugerencia/avisos para imputación del parámetro."""
    low = iqr.get("low", np.nan)
    high = iqr.get("high", np.nan)
    usable = bool(iqr.get("usable", False))
    sug = stats.median

    msg = []
    msg.append(f"<b>Parámetro:</b> {param_label}")
    msg.append(f"<b>Sugerido (mediana Top-K):</b> <code>{fmt2(sug)}</code>")

    if usable and np.isfinite(low) and np.isfinite(high):
        msg.append(
            f"<b>Rango confiable (IQR):</b> <code>[{fmt2(low)}, {fmt2(high)}]</code>"
        )
    else:
        msg.append(
            "<b>Rango confiable (IQR):</b> <i>no disponible (Top-K insuficiente o muy disperso)</i>"
        )

    if (
        objetivo is not None
        and np.isfinite(objetivo)
        and usable
        and np.isfinite(low)
        and np.isfinite(high)
    ):
        obj = float(objetivo)
        if obj < low:
            prox = low
            msg.append(
                f"{ALERT} <b>Aviso:</b> el objetivo (<code>{fmt2(obj)}</code>) <b>está por debajo</b> del IQR. "
                f"Considera aproximarlo a <code>{fmt2(prox)}</code> para alinear con la evidencia Top-K."
            )
        elif obj > high:
            prox = high
            msg.append(
                f"{ALERT} <b>Aviso:</b> el objetivo (<code>{fmt2(obj)}</code>) <b>está por encima</b> del IQR. "
                f"Considera aproximarlo a <code>{fmt2(prox)}</code> para alinear con la evidencia Top-K."
            )
        else:
            msg.append("✅ El objetivo está dentro del IQR (coherente con Top-K).")

    msg.append(
        f"<small>n={stats.n} · media={fmt2(stats.mean)} · σ={fmt2(stats.std)} · CV={fmt2(stats.cv)} · "
        f"p10={fmt2(stats.p10)} · p90={fmt2(stats.p90)}</small>"
    )
    html = "<br>".join(msg)
    return w.HTML(html)


def widget_topk_param(
    df_topk: pd.DataFrame,
    *,
    param_cols: Dict[str, str],
    get_objetivo: Optional[Callable[[str], Optional[float]]] = None,
    objetivo_widget: Optional[w.Widget] = None,
    title: str = "Detalle por parámetro",
) -> w.Accordion:
    """
    Panel standalone para explorar Top-K por parámetro con métricas y guía.
    df_topk: DataFrame con filas Top-K (pre-filtradas) para el objetivo actual.
    param_cols: mapping etiqueta legible -> nombre de columna en df_topk.
    get_objetivo: función opcional: (col) -> valor objetivo actual.
    objetivo_widget: si se pasa, vincula un observe para refresco automático.
    """
    dd = w.Dropdown(
        options=[(k, v) for k, v in param_cols.items()],
        description="Parámetro:",
        layout=w.Layout(width="60%"),
    )
    html_fig = w.HTML()
    html_tbl = w.HTML()
    html_info = w.HTML()
    hdr = w.HTML(f"<b>{title}</b>")

    box = w.VBox([hdr, dd, html_fig, html_tbl, html_info])

    def _render(*_):
        col = dd.value
        label = next((k for k, v in param_cols.items() if v == col), col)
        x = df_topk[col] if col in df_topk.columns else pd.Series(dtype=float)

        objetivo = get_objetivo(col) if callable(get_objetivo) else None
        fig, stats, iqri = _fig_topk_hist(x, param_label=label, objetivo=objetivo)

        dfm = pd.DataFrame(
            [
                {
                    "n_topk": stats.n,
                    "sugerido (mediana)": stats.median,
                    "Q1": stats.q1,
                    "Q3": stats.q3,
                    "IQR": stats.iqr,
                    "p10": stats.p10,
                    "p90": stats.p90,
                    "media": stats.mean,
                    "σ": stats.std,
                    "CV": stats.cv,
                }
            ]
        )

        try:
            plotly_apply_2dec(fig)
        except Exception:
            pass
        # Embebido Plotly -> HTML
        try:
            html_fig.value = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")  # type: ignore[arg-type]
        except Exception:
            # Fallback: mensaje simple
            html_fig.value = "<i>No se pudo renderizar la figura.</i>"

        try:
            html_tbl.value = numeric_2dec_styler(dfm).to_html()
        except Exception:
            try:
                df_f = pd.DataFrame(dfm).copy()
                for _c in df_f.columns:
                    try:
                        if pd.api.types.is_numeric_dtype(df_f[_c]):
                            df_f[_c] = df_f[_c].apply(lambda x: fmt2(x))
                    except Exception:
                        continue
                html_tbl.value = df_f.to_html(index=False)
            except Exception:
                html_tbl.value = dfm.to_html(index=False)

        try:
            guia = _guia_imputacion_html(label, stats, iqri, objetivo)
            # _guia_imputacion_html retorna un w.HTML; usamos su .value
            html_info.value = getattr(guia, "value", str(guia))
        except Exception:
            html_info.value = ""

    dd.observe(_render, "value")
    _render()

    if objetivo_widget is not None:
        objetivo_widget.observe(lambda *_: _render(), "value")

    acc = w.Accordion(children=[box])
    acc.set_title(0, "Sugerencias Top-K · Resumen")
    acc.selected_index = None
    return acc


# =============================================================================
# Helpers de pesos y estadísticos
# =============================================================================


def _nanmean_safe(x: pd.Series) -> float:
    """Media segura que retorna NaN si no hay datos válidos (evita RuntimeWarning)."""
    x = pd.to_numeric(x, errors="coerce")
    x = x.dropna()
    if x.empty:
        return np.nan
    return float(np.mean(x))


def _wmean_safe(x: pd.Series, w: pd.Series) -> float:
    """Media ponderada segura que retorna NaN si no hay pesos/datos válidos."""
    x = pd.to_numeric(x, errors="coerce")
    w = pd.to_numeric(w, errors="coerce")
    m = (~x.isna()) & (~w.isna()) & (w > 0)
    if m.sum() == 0:
        return np.nan
    return float(np.average(x[m], weights=w[m]))


def _normalize_conf_series(s: pd.Series) -> pd.Series:
    """Normaliza posibles columnas de confianza a [0,1]. Si max>1, asume 0..100."""
    s = pd.to_numeric(s, errors="coerce")
    if s.dropna().empty:
        return pd.Series(np.nan, index=s.index)
    mx = float(s.max(skipna=True))
    if mx > 1.0:
        s = s / 100.0
    return s.clip(lower=0.0, upper=1.0)


def _weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    wsum = np.sum(w)
    if wsum <= 0 or x.size == 0:
        return np.nan
    return float(np.sum(x * w) / wsum)


def _weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    """Cuantil ponderado simple (0..1)."""
    if x.size == 0:
        return np.nan
    # Ordenar por x y acumular pesos
    order = np.argsort(x)
    x_sorted = x[order]
    w_sorted = w[order]
    cw = np.cumsum(w_sorted)
    if cw.size == 0 or cw[-1] <= 0:
        # Fallback: mediana simple si no hay pesos válidos
        return float(np.nanmedian(x))
    target = float(q) * float(cw[-1])
    idx = int(np.searchsorted(cw, target, side="left"))
    idx = int(np.clip(idx, 0, len(x_sorted) - 1))
    return float(x_sorted[idx])


def _weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    return _weighted_quantile(x, w, 0.5)


def _stats_unweighted(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return dict(
            n=0,
            min=np.nan,
            q1=np.nan,
            median=np.nan,
            mean=np.nan,
            q3=np.nan,
            max=np.nan,
        )
    return dict(
        n=int(s.size),
        min=float(s.min()),
        q1=float(s.quantile(0.25)),
        median=float(s.median()),
        mean=float(s.mean()),
        q3=float(s.quantile(0.75)),
        max=float(s.max()),
    )


def _stats_weighted(s: pd.Series, w: pd.Series) -> dict:
    """Estadísticos básicos ponderados; si no hay pesos válidos, devuelve NaN."""
    x = pd.to_numeric(s, errors="coerce").to_numpy()
    ww = pd.to_numeric(w, errors="coerce").to_numpy()
    mask = np.isfinite(x) & np.isfinite(ww) & (ww > 0)
    x, ww = x[mask], ww[mask]
    if x.size == 0:
        return dict(n_eff=0.0, w_mean=np.nan, w_median=np.nan)
    return dict(
        n_eff=float(np.sum(ww)),
        w_mean=_weighted_mean(x, ww),
        w_median=_weighted_median(x, ww),
    )


# =============================================================================
# Núcleo de sugerencias
# =============================================================================


@dataclass
class SugerenciasConfig:
    params: list[str]
    top_k: int
    remove_outliers: bool
    iqr_factor: float
    use_distance_weights: bool
    use_confidence_weights: bool
    beta_dist: float
    beta_conf: float
    confidence_cols: dict | None
    name_objetivo: str
    eps: float = 1e-9


def _autodetect_numeric_params(df: pd.DataFrame, min_n: int = 3) -> list[str]:
    """
    Devuelve columnas numéricas 'sugeribles', excluyendo:
      - auxiliares del ranking/sanitización: distancia, similitud, *_media
      - columnas por-prefijo: dv_*, viol_*, is_outlier_*
      - booleanas
    """
    deny_exact = {
        "distancia",
        "similitud",
        "distancia_media",
        "similitud_media",
        "aeronave",
        "segmento",
    }
    deny_prefix = ("dv_", "viol_", "is_outlier_", "Δ_", "delta_")

    cols: list[str] = []
    for c in df.columns:
        if c in deny_exact:
            continue
        if any(c.startswith(p) for p in deny_prefix):
            continue
        if is_bool_dtype(df[c]):
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= min_n:
            cols.append(c)
    return cols


def _build_weights_for_param(
    df_sub: pd.DataFrame, col_param: str, cfg: SugerenciasConfig
) -> pd.Series:
    """w_total = (w_dist ** beta_dist) * (w_conf ** beta_conf)."""
    w_dist = pd.Series(1.0, index=df_sub.index)
    if cfg.use_distance_weights and "distancia" in df_sub.columns:
        dist_arr = pd.to_numeric(df_sub["distancia"], errors="coerce").to_numpy(
            dtype=float
        )
        w_arr = _w_from_dist(dist_arr)
        w_dist = pd.Series(w_arr, index=df_sub.index)

    w_conf = pd.Series(1.0, index=df_sub.index)
    if cfg.use_confidence_weights and cfg.confidence_cols:
        col_conf = cfg.confidence_cols.get(col_param)
        if col_conf and col_conf in df_sub.columns:
            w_conf = _normalize_conf_series(df_sub[col_conf])
            w_conf = w_conf.fillna(0.0)

    # potencias (por si querés reforzar/atenuar)
    w_total = (w_dist ** float(cfg.beta_dist)) * (w_conf ** float(cfg.beta_conf))
    # si todos dieron 0 → poner 1 como fallback para no vaciar el set
    if (w_total <= 0).all():
        w_total = pd.Series(1.0, index=df_sub.index)
    return w_total


def _mask_no_outliers(s: pd.Series, factor: float = 1.5, min_n: int = 5) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").astype(float)
    info = compute_iqr_bounds(x, factor=factor, min_n=min_n)
    if not info["usable"]:
        return pd.Series(True, index=s.index)
    low, high = info["low"], info["high"]
    mask = (x >= low) & (x <= high)
    return mask


def _prepare_topk_neighbors(
    df_ranked: pd.DataFrame, cfg: SugerenciasConfig
) -> pd.DataFrame:
    df2 = df_ranked.copy()
    # excluir la fila objetivo si existe
    if "aeronave" in df2.columns:
        df2 = df2[df2["aeronave"] != cfg.name_objetivo]
    # top K
    if cfg.top_k is not None and cfg.top_k > 0:
        df2 = df2.head(cfg.top_k)
    return df2


def sugerencias_topk(
    df_ranked: pd.DataFrame,
    *,
    params: list[str] | None = None,
    top_k: int = 10,
    remove_outliers: bool = True,
    iqr_factor: float = 1.5,
    use_distance_weights: bool = True,
    use_confidence_weights: bool = False,
    confidence_cols: dict | None = None,  # ejemplo: {"Payload": "conf_Payload", ...}
    beta_dist: float = 1.0,
    beta_conf: float = 1.0,
    name_objetivo: str = "Objetivo (usuario)",
) -> dict:
    """
    Calcula sugerencias de parámetros usando los Top-K vecinos del ranking.

    Parámetros
    ----------
    df_ranked : DataFrame (ranking ya ordenado)
    params : lista de columnas a sugerir; si None, autodetecta numéricas (>=3 válidos)
    top_k : cuántos vecinos usar
    remove_outliers : si True, aplica IQR (factor) por parámetro
    iqr_factor : ancho de bigotes del IQR (1.5 por defecto)
    use_distance_weights : ponderar por 1/(distancia+eps)
    use_confidence_weights : ponderar también por columna de confianza
    confidence_cols : dict {param->col_conf} si usás confianza
    beta_dist, beta_conf : exponentes para “reforzar” o “atenuar” cada peso
    name_objetivo : nombre de la fila objetivo para excluirla del cómputo

    Returns
    -------
    dict con:
      - 'summary' (DataFrame por parámetro)
      - 'details' (dict param -> DataFrame de vecinos con valor, pesos, flags)
      - 'neighbors' (Top-K usado, ya sin la fila objetivo)
      - 'config' (SugerenciasConfig)
    """
    if params is None:
        params = _autodetect_numeric_params(df_ranked)

    cfg = SugerenciasConfig(
        params=list(params),
        top_k=top_k,
        remove_outliers=remove_outliers,
        iqr_factor=iqr_factor,
        use_distance_weights=use_distance_weights,
        use_confidence_weights=use_confidence_weights,
        beta_dist=beta_dist,
        beta_conf=beta_conf,
        confidence_cols=confidence_cols,
        name_objetivo=name_objetivo,
    )

    neighbors = _prepare_topk_neighbors(df_ranked, cfg)

    summary_rows = []
    # Detalles por parámetro: cada entrada contiene 'usados' y 'excluidos'
    details: Dict[str, dict] = {}

    for col in cfg.params:
        if col not in neighbors.columns:
            continue
        # Evitar columnas auxiliares o booleanas aunque params venga mal
        if col in {
            "distancia",
            "similitud",
            "distancia_media",
            "similitud_media",
            "aeronave",
            "segmento",
        }:
            continue
        if col.startswith(("dv_", "viol_", "is_outlier_")):
            continue
        if is_bool_dtype(neighbors[col]):
            continue

        s = pd.to_numeric(neighbors[col], errors="coerce")
        base_mask = s.notna()

        mask_ok = base_mask.copy()
        low = high = np.nan
        if cfg.remove_outliers:
            m = _mask_no_outliers(s, factor=cfg.iqr_factor, min_n=5)
            mask_ok = base_mask & m
            # para el resumen, guardamos los límites si eran “usables”
            info = compute_iqr_bounds(s, factor=cfg.iqr_factor, min_n=5)
            if info["usable"]:
                low, high = info["low"], info["high"]

        df_sub = neighbors.loc[mask_ok, :].copy()
        n_in = int(base_mask.sum())
        n_ok = int(mask_ok.sum())
        n_out = max(n_in - n_ok, 0)

        # pesos (sobre el subconjunto válido)
        w_total = _build_weights_for_param(df_sub, col, cfg)

        # valores válidos para estadísticas
        vals = pd.to_numeric(df_sub[col], errors="coerce")
        n_used = int(vals.notna().sum())

        if n_used == 0:
            # Resumen limpio sin warnings
            summary_rows.append(
                {
                    "parametro": col,
                    "top_k": cfg.top_k,
                    "n_vecinos": n_in,
                    "n_usados": 0,
                    "n_outliers": n_out,
                    "low": low,
                    "high": high,
                    "min": np.nan,
                    "q1": np.nan,
                    "mediana": np.nan,
                    "media": np.nan,
                    "q3": np.nan,
                    "max": np.nan,
                    "w_mediana": np.nan,
                    "w_media": np.nan,
                    "n_efectivo": 0.0,
                }
            )

            # detalle (vacío pero con estructura coherente)
            det = (
                df_sub[["aeronave", "distancia"]].copy()
                if "aeronave" in df_sub.columns
                else df_sub[["distancia"]].copy()
            )
            det["valor"] = df_sub[col]
            det["w_total"] = w_total
            if cfg.use_distance_weights and "distancia" in df_sub.columns:
                _s = df_sub.get("distancia", None)
                if _s is not None and hasattr(_s, "to_numpy"):
                    arr = pd.to_numeric(_s, errors="coerce").to_numpy(dtype=float)
                else:
                    arr = np.asarray([], dtype=float)
                det["w_dist"] = _w_from_dist(arr)
            else:
                det["w_dist"] = np.nan
            if (
                cfg.use_confidence_weights
                and cfg.confidence_cols
                and cfg.confidence_cols.get(col) in df_sub.columns
            ):
                det["w_conf"] = _normalize_conf_series(df_sub[cfg.confidence_cols[col]])
            else:
                det["w_conf"] = np.nan

            if cfg.remove_outliers:
                m_out = _mask_no_outliers(
                    neighbors[col], factor=cfg.iqr_factor, min_n=5
                )
                excl = neighbors.loc[base_mask & (~m_out)]
                excl = (
                    excl[["aeronave"]].copy()
                    if "aeronave" in excl.columns
                    else excl[[]]
                )
                excl["valor_outlier"] = neighbors.loc[base_mask & (~m_out), col]
                details[col] = {"usados": det, "excluidos": excl}
            else:
                details[col] = {
                    "usados": det,
                    "excluidos": pd.DataFrame(columns=["aeronave", "valor_outlier"]),
                }
            continue

        # estadísticas robustas sin pesos
        desc = vals.describe(percentiles=[0.25, 0.5, 0.75])
        q1 = float(desc.get("25%", np.nan))
        med = float(desc.get("50%", np.nan))
        q3 = float(desc.get("75%", np.nan))
        w_media = _wmean_safe(vals, w_total)
        w_mediana = med  # mediana simple como recomendación robusta

        # n_efectivo: suma de pesos NORMALIZADOS (0..1) donde hay datos
        #   - Interpretación: "cantidad equivalente" de vecinos útiles
        #   - Norma usada: escalar por el máximo de w_total (>0) en los usados
        w_num = pd.to_numeric(w_total, errors="coerce").fillna(0.0)
        m_valid = vals.notna() & (w_num > 0)
        if m_valid.any():
            max_w = float(w_num[m_valid].max())
            if max_w > 0:
                w_scaled = (w_num[m_valid] / max_w).clip(lower=0.0, upper=1.0)
                n_eff = float(w_scaled.sum())
            else:
                n_eff = 0.0
        else:
            n_eff = 0.0

        summary_rows.append(
            {
                "parametro": col,
                "top_k": cfg.top_k,
                "n_vecinos": n_in,
                "n_usados": n_ok,
                "n_outliers": n_out,
                "low": low,
                "high": high,
                "min": float(vals.min()),
                "q1": q1,
                "mediana": med,
                "media": _nanmean_safe(vals),
                "q3": q3,
                "max": float(vals.max()),
                "w_mediana": w_mediana,
                "w_media": w_media,
                "n_efectivo": n_eff,
            }
        )

        # detalle por vecino
        det = (
            df_sub[["aeronave", "distancia"]].copy()
            if "aeronave" in df_sub.columns
            else df_sub[["distancia"]].copy()
        )
        det["valor"] = df_sub[col]
        det["w_total"] = w_total
        if cfg.use_distance_weights and "distancia" in df_sub.columns:
            det["w_dist"] = _w_from_dist(
                pd.to_numeric(df_sub["distancia"], errors="coerce").to_numpy(
                    dtype=float
                )
            )
        else:
            det["w_dist"] = np.nan
        if (
            cfg.use_confidence_weights
            and cfg.confidence_cols
            and cfg.confidence_cols.get(col) in df_sub.columns
        ):
            det["w_conf"] = _normalize_conf_series(df_sub[cfg.confidence_cols[col]])
        else:
            det["w_conf"] = np.nan

        # marcar cuáles fueron excluidos por outlier (para transparencia)
        if cfg.remove_outliers:
            m_out = _mask_no_outliers(neighbors[col], factor=cfg.iqr_factor, min_n=5)
            excl = neighbors.loc[base_mask & (~m_out)]
            excl = excl[["aeronave"]].copy() if "aeronave" in excl.columns else excl[[]]
            excl["valor_outlier"] = neighbors.loc[base_mask & (~m_out), col]
            details[col] = {"usados": det, "excluidos": excl}
        else:
            details[col] = {
                "usados": det,
                "excluidos": pd.DataFrame(columns=["aeronave", "valor_outlier"]),
            }

    summary = pd.DataFrame(summary_rows)

    # tag simple de calidad por n_usados
    def _tag(n):
        if n >= 10:
            return "buena"
        if n >= 5:
            return "aceptable"
        if n >= 3:
            return "baja"
        return "insuficiente"

    if not summary.empty:
        summary["calidad_n"] = summary["n_usados"].map(_tag)

    return dict(summary=summary, details=details, neighbors=neighbors, config=cfg)


# =============================================================================
# Vistas para Notebook
# =============================================================================


def vista_sugerencias_resumen(summary: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """Styler con columnas clave y gradientes para lectura rápida."""
    if summary.empty:
        return summary.style  # vacío
    cols = [
        "parametro",
        "top_k",
        "n_vecinos",
        "n_usados",
        "n_outliers",
        "calidad_n",
        "min",
        "q1",
        "mediana",
        "media",
        "q3",
        "max",
        "w_mediana",
        "w_media",
        "n_efectivo",
    ]
    cols = [c for c in cols if c in summary.columns]
    dfv = summary[cols].copy()
    fmt = {
        c: "{:.2f}"
        for c in dfv.columns
        if c
        not in {
            "parametro",
            "calidad_n",
            "top_k",
            "n_vecinos",
            "n_usados",
            "n_outliers",
        }
    }
    sty = dfv.style
    for col, f in fmt.items():
        if col in dfv.columns:
            sty = sty.format({col: f})
    sty = sty.background_gradient(subset=["n_usados"], cmap="Greens")
    cols_blues = [c for c in ["w_media", "w_mediana"] if c in dfv.columns]
    if cols_blues:
        sty = sty.background_gradient(subset=cols_blues, cmap="Blues")
    return numeric_2dec_styler(dfv).background_gradient(
        subset=["n_usados"], cmap="Greens"
    )


def widget_sugerencias_param(
    sug: dict,
    *,
    bins: int = 20,
    titulo: str = "Sugerencias por parámetro (Top-K)",
    get_objetivo: Optional[Callable[[str], Optional[float]]] = None,
    df_scope: Optional[pd.DataFrame] = None,
    ambito: str = "global",
    topk_index: Optional[list] = None,
):

    try:
        import ipywidgets as w  # local import to keep compatibility if used elsewhere
    except Exception as e:
        raise RuntimeError("Este widget requiere 'ipywidgets' instalado.") from e

    params = [d for d in sug["details"].keys()]
    if not params:
        raise ValueError("No hay parámetros en 'details'.")

    dd = w.Dropdown(
        options=params, description="Parámetro:", layout=w.Layout(width="45%")
    )
    html_out = w.HTML("")
    # Badge N_efectivo / n_usados
    badge = w.HTML("")
    head = w.HBox(
        [w.HTML(f"<h4 style='margin:0'>{titulo}</h4>"), w.HTML("&nbsp;"), badge]
    )

    def _render(*args):
        par = dd.value
        pack = sug["details"][par]
        usados = pack["usados"]
        excl = pack["excluidos"]

            # Tabla de vecinos usados ahora vive únicamente en la sección dedicada
            # del panel principal para evitar duplicados. Dejamos una referencia clara.
            if not usados.empty:
                display(
                    w.HTML(
                        "<i>Consultá el Excel original (botón &quot;Abrir Excel original&quot; en la parte superior) para ver la lista completa de vecinos ordenados por peso.</i>"
                    )
                )
            else:
                display(w.HTML("<i>Sin vecinos disponibles para este parámetro.</i>"))

            # Histograma Top-K con overlay IQR consistente (_fig_parametro_hist_box)
            vals = pd.to_numeric(usados["valor"], errors="coerce").dropna()
            resumen = (
                sug["summary"].set_index("parametro")
                if isinstance(sug.get("summary"), pd.DataFrame)
                else None
            )
            sugerido_val = (
                resumen.loc[par, "w_mediana"]
                if resumen is not None
                and par in resumen.index
                and "w_mediana" in resumen.columns
                else None
            )
            low = (
                resumen.loc[par, "low"]
                if resumen is not None
                and par in resumen.index
                and "low" in resumen.columns
                else None
            )
            high = (
                resumen.loc[par, "high"]
                if resumen is not None
                and par in resumen.index
                and "high" in resumen.columns
                else None
            )
            objetivo_val = None
            if callable(get_objetivo):
                try:
                    objetivo_val = get_objetivo(par)
                except Exception:
                    objetivo_val = None
            fig = _fig_parametro_hist_box(
                vals,
                titulo=f"Distribución Top-K: {par}",
                objetivo_val=objetivo_val,
                sugerido_val=sugerido_val,
                low=low,
                high=high,
            )
            apply_tickformat_2dec(fig)
            display(fig)

        # Badge update (N_efectivo / n_usados)
        try:
            n_eff = (
                resumen.loc[par, "n_efectivo"]
                if resumen is not None and "n_efectivo" in resumen.columns
                else None
            )
            n_used = (
                resumen.loc[par, "n_usados"]
                if resumen is not None and "n_usados" in resumen.columns
                else None
            )
            if n_eff is not None and n_used is not None:
                badge.value = f"<span style='background:#eef7ff;border:1px solid #cde3ff;border-radius:10px;padding:2px 8px;font-size:12px;'>N_efectivo={float(n_eff):.2f} / n_usados={int(n_used)}</span>"
            else:
                badge.value = ""
        except Exception:
            badge.value = ""

        # Mostrar posibles outliers excluidos
        if excl is not None and not isinstance(excl, dict) and not excl.empty:
            html_parts.append("Valores excluidos como outliers:")
            try:
                html_parts.append(numeric_2dec_styler(excl.head(20)).to_html())
            except Exception:
                html_parts.append(excl.head(20).to_html())

        html_out.value = "<br>".join(html_parts)

    _render()
    dd.observe(_render, names="value")
    box = w.VBox([head, dd, html_out])
    return box


def _fig_parametro_hist_box(
    serie_val: pd.Series,
    *,
    titulo: str,
    objetivo_val: float | None = None,
    sugerido_val: float | None = None,
    low: float | None = None,
    high: float | None = None,
) -> go.Figure:
    """Histograma + mejoras de lectura (IQR box overlay, vlines, avisos)."""
    s = pd.to_numeric(serie_val, errors="coerce").dropna()
    fig = go.Figure()

    # Histograma base (bins estables y tooltip limpio)
    fig.add_histogram(
        x=s,
        name="Top-K",
        nbinsx=12,
        opacity=0.75,
        hovertemplate="valor=%{x:.2f}<br>frec=%{y:.2f}<extra></extra>",
    )

    # BoxPlot adicional (referencia robusta) — mantenido como antes
    if len(s) >= 5:
        fig.add_trace(
            go.Box(
                x=s,
                name="Box",
                boxpoints="outliers",
                jitter=0,
                opacity=0.35,
                orientation="h",
            )
        )

    # Variables requeridas por el parche
    try:
        param = titulo.split(":", 1)[1].strip() if ":" in titulo else titulo
    except Exception:
        param = titulo
    vals_topk = s.to_numpy()
    min_n = 5
    info_iqr = compute_iqr_bounds(pd.Series(s), factor=1.5, min_n=min_n)
    q1_val = float(info_iqr.get("Q1", np.nan))
    q3_val = float(info_iqr.get("Q3", np.nan))
    iqr_val = float(info_iqr.get("IQR", np.nan))
    usable_iqr = bool(info_iqr.get("usable", False))
    fence_low = float(info_iqr.get("low", np.nan)) if usable_iqr else np.nan
    fence_high = float(info_iqr.get("high", np.nan)) if usable_iqr else np.nan
    sugerido = float(sugerido_val) if (sugerido_val is not None) else np.nan
    objetivo = float(objetivo_val) if (objetivo_val is not None) else np.nan
    n_topk = int(s.shape[0])

    # --- 1) Título sin solaparse (con subtítulo en <sup>) y margen superior amplio
    fig.update_layout(
        title=dict(
            text=f"Distribución Top-K: {param}<br>"
            f"<sup>Q1={f2(q1_val)}  ·  Q3={f2(q3_val)}  ·  n_topk={n_topk}</sup>",
            x=0.02,
            xanchor="left",
            y=0.97,
            font=dict(size=16),
        ),
        margin=dict(t=110),  # deja espacio para título y posibles avisos
    )

    # --- 2) Overlay IQR (LOW-HIGH) como "box" siempre visible y líneas guía

    # Sombrear fuera del rango IQR extendido (fences) como marco visual
    if usable_iqr and np.isfinite(fence_low) and np.isfinite(fence_high) and s.size:
        data_min = float(np.nanmin(s))
        data_max = float(np.nanmax(s))
        if data_min < float(fence_low):
            fig.add_shape(
                type="rect",
                xref="x",
                yref="paper",
                x0=data_min,
                x1=float(fence_low),
                y0=0.0,
                y1=1.0,
                line=dict(width=0),
                fillcolor="rgba(255, 127, 80, 0.12)",
                layer="below",
            )
        if data_max > float(fence_high):
            fig.add_shape(
                type="rect",
                xref="x",
                yref="paper",
                x0=float(fence_high),
                x1=data_max,
                y0=0.0,
                y1=1.0,
                line=dict(width=0),
                fillcolor="rgba(255, 127, 80, 0.12)",
                layer="below",
            )

    # líneas verticales Q1/Q3
    if np.isfinite(q1_val):
        fig.add_vline(
            x=float(q1_val), line=dict(color="indianred", width=1, dash="dot")
        )
        fig.add_annotation(
            x=float(q1_val),
            y=1.02,
            yref="paper",
            xref="x",
            text="Q1",
            showarrow=False,
            font=dict(color="indianred"),
        )
    if np.isfinite(q3_val):
        fig.add_vline(
            x=float(q3_val), line=dict(color="indianred", width=1, dash="dot")
        )
        fig.add_annotation(
            x=float(q3_val),
            y=1.02,
            yref="paper",
            xref="x",
            text="Q3",
            showarrow=False,
            font=dict(color="indianred"),
        )

    # Fences (Q1 ± 1.5*IQR)
    if usable_iqr and np.isfinite(fence_low):
        fig.add_vline(
            x=float(fence_low), line=dict(color="crimson", width=2, dash="dash")
        )
        fig.add_annotation(
            x=float(fence_low),
            y=1.08,
            yref="paper",
            xref="x",
            text="LOW fence (IQR)",
            showarrow=False,
            font=dict(color="crimson"),
        )
    if usable_iqr and np.isfinite(fence_high):
        fig.add_vline(
            x=float(fence_high), line=dict(color="crimson", width=2, dash="dash")
        )
        fig.add_annotation(
            x=float(fence_high),
            y=1.08,
            yref="paper",
            xref="x",
            text="HIGH fence (IQR)",
            showarrow=False,
            font=dict(color="crimson"),
        )

    # línea sugerido
    if np.isfinite(sugerido):
        fig.add_vline(
            x=float(sugerido), line=dict(color="seagreen", width=2, dash="dash")
        )
        fig.add_annotation(
            x=float(sugerido),
            y=1.02,
            yref="paper",
            xref="x",
            text="Sugerido",
            showarrow=False,
            font=dict(size=11, color="seagreen"),
        )

    # línea objetivo (si existe)
    if np.isfinite(objetivo):
        fig.add_vline(x=float(objetivo), line=dict(color="royalblue", width=2))
        fig.add_annotation(
            x=float(objetivo),
            y=1.02,
            yref="paper",
            xref="x",
            text="Objetivo",
            showarrow=False,
            font=dict(size=11, color="royalblue"),
        )

    # --- 3) Avisos (⚠️) con reglas extendidas
    def _pct(x):
        return f"{100*x:.0f}%"

    warnings = []

    # Parámetros (ajustables)
    edge_tol = 0.10  # 10% del ancho del IQR --> "cerca del límite"
    diff_tol = 0.20  # 20% del ancho del IQR --> brecha objetivo–sugerido
    out_tol = 0.30  # 30% de Top-K fuera del IQR
    min_unique = 4  # diversidad mínima en Top-K

    # Magnitudes básicas
    iqr_w = iqr_val if np.isfinite(iqr_val) else np.nan
    vals = np.asarray(vals_topk, dtype=float)
    vals = vals[np.isfinite(vals)]

    # 1) objetivo fuera del IQR
    if np.isfinite(objetivo) and np.isfinite(q1_val) and np.isfinite(q3_val):
        if objetivo < q1_val or objetivo > q3_val:
            warnings.append("⚠️ objetivo fuera del IQR")

    # 2) objetivo “cerca del límite” del IQR
    if np.isfinite(objetivo) and np.isfinite(iqr_w) and iqr_w > 0:
        dist_edge = min(abs(objetivo - q1_val), abs(q3_val - objetivo))
        if dist_edge <= edge_tol * iqr_w:
            warnings.append("⚠️ objetivo cerca del límite IQR")

    # 3) sugerido fuera del IQR
    if np.isfinite(sugerido) and np.isfinite(q1_val) and np.isfinite(q3_val):
        if sugerido < q1_val or sugerido > q3_val:
            warnings.append("⚠️ sugerido fuera del IQR")

    # 4) brecha grande entre objetivo y sugerido (relativa al IQR)
    if (
        np.isfinite(objetivo)
        and np.isfinite(sugerido)
        and np.isfinite(iqr_w)
        and iqr_w > 0
    ):
        if abs(objetivo - sugerido) >= diff_tol * iqr_w:
            warnings.append("⚠️ gran brecha objetivo–sugerido")

    # 5) alto % de atípicos entre los Top-K
    if vals.size and np.isfinite(q1_val) and np.isfinite(q3_val):
        out_frac = np.mean((vals < q1_val) | (vals > q3_val))
        if out_frac >= out_tol:
            warnings.append(f"⚠️ { _pct(out_frac) } de Top-K fuera del IQR")

    # 6) poca diversidad en Top-K
    if vals.size and len(np.unique(vals)) < min_unique:
        warnings.append("⚠️ poca diversidad en Top-K")

    # 7) n_topk chico (regla original)
    if n_topk is not None and min_n is not None and n_topk < min_n:
        warnings.append(f"⚠️ muestra Top-K pequeña (n={n_topk} < {min_n})")

    # Mostrar si hay algo
    if warnings:
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.01,
            y=1.13,
            text=" &nbsp; ".join(warnings),
            align="left",
            showarrow=False,
            font=dict(size=12),
            bgcolor="rgba(255,245,204,.9)",
            bordercolor="rgba(0,0,0,.1)",
            borderwidth=1,
        )
        fig.update_layout(margin=dict(t=150))

    # estéticas suaves
    fig.update_layout(
        bargap=0.15,
        template="plotly_white",
        hoverlabel=dict(namelength=-1),  # no truncar nombres
        xaxis_title=str(param),
        yaxis_title="frecuencia",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    try:
        plotly_apply_2dec(fig)
    except Exception:
        pass
    return fig


def widget_sugerencias_panel_plotly(
    df_ranked: pd.DataFrame,
    *,
    params: list[str] | None = None,
    top_k: int = 10,
    remove_outliers: bool = True,
    iqr_factor: float = 1.5,
    use_distance_weights: bool = True,
    use_confidence_weights: bool = False,
    confidence_cols: dict[str, str] | None = None,
    beta_dist: float = 1.0,
    beta_conf: float = 1.0,
    name_objetivo: str = "Objetivo (usuario)",
    df_scope: Optional[pd.DataFrame] = None,
    ambito: str = "global",
    topk_index: Optional[list] = None,
) -> w.Accordion:
    """Panel interactivo con resumen y distribuciones por parámetro."""
    dd_k = w.BoundedIntText(
        value=top_k,
        min=3,
        max=50,
        step=1,
        description="Top-K:",
        layout=w.Layout(width="160px"),
    )
    ch_out = w.Checkbox(value=remove_outliers, description="Quitar atípicos (IQR)")
    ft_iqr = w.FloatText(
        value=iqr_factor, description="factor IQR", layout=w.Layout(width="170px")
    )
    ch_wd = w.Checkbox(value=use_distance_weights, description="Ponderar 1/(dist)")
    btn = w.Button(description="Recalcular")
    btn.style.button_color = "#28a745"
    header = w.HBox([dd_k, ch_out, ft_iqr, ch_wd, btn])

    out_resumen = w.Output()
    out_detalle = w.Output()

    if params is None:
        candidates = [
            c
            for c in df_ranked.columns
            if pd.to_numeric(df_ranked[c], errors="coerce").notna().sum() > 0
            and c.lower() not in ("dist", "similitud")
        ]
        params = sorted(candidates)
    dd_param = w.Dropdown(
        options=params, description="Parámetro:", layout=w.Layout(width="60%")
    )

    def _run_calc():
        return sugerencias_topk(
            df_ranked=df_ranked,
            params=params,
            top_k=int(dd_k.value),
            remove_outliers=bool(ch_out.value),
            iqr_factor=float(ft_iqr.value),
            use_distance_weights=bool(ch_wd.value),
            use_confidence_weights=use_confidence_weights,
            confidence_cols=confidence_cols,
            beta_dist=beta_dist,
            beta_conf=beta_conf,
            name_objetivo=name_objetivo,
        )

    def _render(*_):
        sug = _run_calc()

        # 1) Resumen
        df_summary = sug.get("summary")
        if isinstance(df_summary, pd.DataFrame):
            try:
                out_resumen.value = numeric_2dec_styler(df_summary).to_html()
            except Exception:
                try:
                    df_f = df_summary.copy()
                    for _c in df_f.columns:
                        try:
                            if pd.api.types.is_numeric_dtype(df_f[_c]):
                                df_f[_c] = df_f[_c].map(
                                    lambda v: fmt2(v) if pd.notna(v) else v
                                )
                        except Exception:
                            continue
                    out_resumen.value = df_f.to_html(index=False)
                except Exception:
                    out_resumen.value = df_summary.to_html(index=False)
        else:
            out_resumen.value = "<i>Sin datos de resumen.</i>"

        # 2) Detalle por parámetro
        par = dd_param.value
        if par not in sug.get("details", {}):
            out_detalle.value = "<i>Sin datos para el parámetro elegido.</i>"
        else:
            # serie top-k y referencias
            det_pack = sug["details"][par]
            usados = det_pack.get("usados", pd.DataFrame())
            # s_vals según alcance: Global/Filtrado usan df_scope; Top‑K usa vecinos
            if df_scope is not None and ambito in {"global", "filtrado"}:
                if par in df_scope.columns:
                    s_vals = pd.to_numeric(df_scope[par], errors="coerce").dropna()
                else:
                    s_vals = pd.Series(dtype=float)
            else:
                s_vals = usados.get("valor", pd.Series(dtype=float))

                # Intentar recuperar objetivo/sugerido/low/high si existen en summary
                objetivo_val = None  # no está explícito en output actual
                resumen = (
                    sug["summary"].set_index("parametro")
                    if "parametro" in sug["summary"].columns
                    else None
                )
                q1_val = (
                    resumen.loc[par, "q1"]
                    if resumen is not None
                    and par in resumen.index
                    and "q1" in resumen.columns
                    else None
                )
                q3_val = (
                    resumen.loc[par, "q3"]
                    if resumen is not None
                    and par in resumen.index
                    and "q3" in resumen.columns
                    else None
                )
                sugerido_val = (
                    resumen.loc[par, "w_mediana"]
                    if resumen is not None
                    and par in resumen.index
                    and "w_mediana" in resumen.columns
                    else None
                )
                low = (
                    resumen.loc[par, "low"]
                    if resumen is not None
                    and par in resumen.index
                    and "low" in resumen.columns
                    else None
                )
                high = (
                    resumen.loc[par, "high"]
                    if resumen is not None
                    and par in resumen.index
                    and "high" in resumen.columns
                    else None
                )

            fig = _fig_parametro_hist_box(
                s_vals,
                titulo=f"Distribución Top-K: {par}",
                objetivo_val=objetivo_val,
                sugerido_val=sugerido_val,
                low=low,
                high=high,
            )
            try:
                plotly_apply_2dec(fig)
            except Exception:
                pass
            out_detalle.value = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")  # type: ignore[arg-type]

                # Números clave
                rows = [
                    {
                        "parámetro": par,
                        "n_topk": int(
                            pd.to_numeric(s_vals, errors="coerce").dropna().shape[0]
                        ),
                        "sugerido (w_mediana)": sugerido_val,
                        "Q1": q1_val,
                        "Q3": q3_val,
                        "LOW fence": low,
                        "HIGH fence": high,
                    }
                ]
                try:
                    display(style_df_2dec(pd.DataFrame(rows)))
                except Exception:
                    display(pd.DataFrame(rows))
                display(
                    w.HTML(
                        "<i>Para auditar los vecinos detallados abrí el Excel original con el botón superior.</i>"
                    )
                )

    btn.on_click(lambda _: _render())

    # Render inicial
    _render()

    title_main = w.HTML("<b>Sugerencias por parámetro (Top‑K)</b>")
    help_html = w.HTML(
        "<div style='font-size:12px;line-height:1.35em;margin:4px 0 6px 0;'>"
        "<b>Similitud</b>: ranking global que ordena todo el dataset por cercanía. "
        "<b>Top-K</b>: trabaja sólo con los K vecinos más parecidos para sugerencias y estadísticas robustas (IQR, medianas, ponderaciones)."
        "</div>"
    )
    title_res = w.HTML("<b>Resumen por parámetro (Top‑K)</b>")
    title_det = w.HTML("<b>Distribución por parámetro</b>")
    body = w.VBox(
        [
            title_main,
            help_html,
            header,
            w.HTML("<hr>"),
            title_res,
            out_resumen,
            w.HTML("<hr>"),
            title_det,
            dd_param,
            out_detalle,
        ]
    )
    acc = w.Accordion(children=[body])
    acc.set_title(0, "Sugerencias por parámetro (Top‑K)")
    acc.selected_index = None
    return acc


def widget_sugerencias_panel(
    data,
    *,
    titulo: str = "Sugerencias por parámetro (Top-K)",
    collapsed: bool = True,
    bins: int = 20,
    get_objetivo: Optional[Callable[[str], Optional[float]]] = None,
    params: list[str] | None = None,
    top_k: int = 10,
    remove_outliers: bool = True,
    iqr_factor: float = 1.5,
    use_distance_weights: bool = True,
    use_confidence_weights: bool = False,
    confidence_cols: dict[str, str] | None = None,
    beta_dist: float = 1.0,
    beta_conf: float = 1.0,
    name_objetivo: str = "Objetivo (usuario)",
    # Alcance actual
    df_scope: Optional[pd.DataFrame] = None,
    ambito: str = "global",
    topk_index: Optional[list] = None,
) -> w.Accordion:
    """Compatibilidad: acepta un dict ya calculado o un DataFrame y delega al panel interactivo."""
    if isinstance(data, dict) and {"summary", "details"}.issubset(data.keys()):
        try:
            import ipywidgets as w  # local import
        except Exception as e:
            raise RuntimeError("Este widget requiere 'ipywidgets' instalado.") from e

        out_resumen = w.HTML()
        if isinstance(data.get("summary"), pd.DataFrame):
            try:
                out_resumen.value = numeric_2dec_styler(data["summary"]).to_html()
            except Exception:
                try:
                    df_f = data["summary"].copy()
                    for _c in df_f.columns:
                        try:
                            if pd.api.types.is_numeric_dtype(df_f[_c]):
                                df_f[_c] = df_f[_c].apply(lambda x: fmt2(x))
                        except Exception:
                            continue
                    out_resumen.value = df_f.to_html(index=False)
                except Exception:
                    out_resumen.value = data["summary"].to_html(index=False)
        else:
            out_resumen.value = "<i>Sin resumen disponible.</i>"

        box_detalle = widget_sugerencias_param(
            data,
            bins=bins,
            titulo=titulo,
            get_objetivo=get_objetivo,
            df_scope=df_scope,
            ambito=ambito,
            topk_index=topk_index,
        )

        nota_excel = w.HTML(
            "<i>Para revisar los vecinos Top-K podés abrir el Excel original con el botón superior.</i>"
        )

        title_main = w.HTML(f"<b>{titulo}</b>")
        title_res = w.HTML("<b>Resumen por parámetro (Top-K)</b>")
        title_det = w.HTML("<b>Distribución por parámetro</b>")
        body = w.VBox(
            [
                title_main,
                title_res,
                out_resumen,
                w.HTML("<hr>"),
                title_det,
                box_detalle,
                nota_excel,
            ]
        )
        acc = w.Accordion(children=[body])
        acc.set_title(0, titulo)
        acc.selected_index = None if collapsed else 0
        return acc

    if isinstance(data, pd.DataFrame):
        return widget_sugerencias_panel_plotly(
            df_ranked=data,
            params=params,
            top_k=top_k,
            remove_outliers=remove_outliers,
            iqr_factor=iqr_factor,
            use_distance_weights=use_distance_weights,
            use_confidence_weights=use_confidence_weights,
            confidence_cols=confidence_cols,
            beta_dist=beta_dist,
            beta_conf=beta_conf,
            name_objetivo=name_objetivo,
            df_scope=df_scope,
            ambito=ambito,
            topk_index=topk_index,
        )

    raise TypeError(
        "widget_sugerencias_panel: primer argumento debe ser dict de sugerencias o DataFrame (ranking)."
    )
