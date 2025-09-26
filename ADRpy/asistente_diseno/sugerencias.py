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
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

# matplotlib se importa via helper centralizado
import plotly.graph_objects as go
import ipywidgets as w
from IPython.display import display, clear_output

from asistente_diseno.outliers import compute_iqr_bounds
from pandas.api.types import is_bool_dtype


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
    order = np.argsort(x)
    x_sorted = x[order]
    w_sorted = w[order]
    cw = np.cumsum(w_sorted)
    if cw[-1] <= 0:
        return float(np.nanmedian(x))
    target = q * cw[-1]
    idx = np.searchsorted(cw, target, side="left")
    idx = np.clip(idx, 0, len(x_sorted) - 1)
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
        w_dist = 1.0 / (dist_arr + float(cfg.eps))
        w_dist = pd.Series(w_dist, index=df_sub.index)

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
            det["w_dist"] = (
                1.0
                / (
                    pd.to_numeric(df_sub.get("distancia", np.nan), errors="coerce")
                    + float(cfg.eps)
                )
                if (cfg.use_distance_weights and "distancia" in df_sub.columns)
                else np.nan
            )
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

        # n_efectivo: suma de pesos positivos donde hay datos
        w_num = pd.to_numeric(w_total, errors="coerce").fillna(0.0)
        m_valid = vals.notna() & (w_num > 0)
        n_eff = float(w_num[m_valid].sum())

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
            dist_arr2 = pd.to_numeric(df_sub["distancia"], errors="coerce").to_numpy(
                dtype=float
            )
            det["w_dist"] = 1.0 / (dist_arr2 + float(cfg.eps))
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
        c: "{:.3f}"
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
    return sty


def widget_sugerencias_param(
    sug: dict, *, bins: int = 20, titulo: str = "Sugerencias por parámetro (Top-K)"
):

    try:
        import ipywidgets as w  # local import to keep compatibility if used elsewhere
        from IPython.display import display, clear_output
    except Exception as e:
        raise RuntimeError("Este widget requiere 'ipywidgets' instalado.") from e

    params = [d for d in sug["details"].keys()]
    if not params:
        raise ValueError("No hay parámetros en 'details'.")

    dd = w.Dropdown(
        options=params, description="Parámetro:", layout=w.Layout(width="45%")
    )
    out = w.Output()
    head = w.HTML(f"<h4 style='margin:0'>{titulo}</h4>")

    def _render(*args):
        with out:
            clear_output(wait=True)
            par = dd.value
            pack = sug["details"][par]
            usados = pack["usados"]
            excl = pack["excluidos"]

            # Tabla de vecinos usados (orden por peso total desc)
            if "w_total" in usados.columns:
                usados = usados.sort_values(by="w_total", ascending=False)
            display(
                usados.head(30).style.format(
                    {
                        "distancia": "{:.3f}",
                        "w_total": "{:.3f}",
                        "w_dist": "{:.3f}",
                        "w_conf": "{:.3f}",
                    }
                )
            )

            # Histograma de valores
            vals = pd.to_numeric(usados["valor"], errors="coerce").dropna()
            # Migración ligera: usar Plotly si está disponible; si no, tabla únicamente
            try:
                import plotly.graph_objects as _go

                summary = sug["summary"].set_index("parametro")
                med_w = (
                    summary.loc[par, "w_mediana"]
                    if par in summary.index and "w_mediana" in summary.columns
                    else np.nan
                )
                mean_w = (
                    summary.loc[par, "w_media"]
                    if par in summary.index and "w_media" in summary.columns
                    else np.nan
                )
                fig = _go.Figure()
                fig.add_histogram(x=vals, nbinsx=bins, name="Usados", opacity=0.8)
                if np.isfinite(med_w):
                    fig.add_vline(
                        x=float(med_w),
                        line_width=2,
                        line_dash="dash",
                        line_color="green",
                        annotation_text="w_mediana",
                        annotation_position="top",
                    )
                if np.isfinite(mean_w):
                    fig.add_vline(
                        x=float(mean_w),
                        line_width=2,
                        line_dash="dot",
                        line_color="royalblue",
                        annotation_text="w_media",
                        annotation_position="top",
                    )
                fig.update_layout(
                    template="plotly_white",
                    title=par,
                    xaxis_title=par,
                    yaxis_title="frecuencia",
                    margin=dict(l=40, r=10, t=40, b=40),
                )
                display(fig)
            except Exception:
                display(
                    w.HTML(
                        "<i>No se pudo renderizar el histograma (Plotly no disponible).</i>"
                    )
                )

            # Mostrar posibles outliers excluidos
            if excl is not None and not isinstance(excl, dict) and not excl.empty:
                print("Valores excluidos como outliers:")
                display(excl.head(20))

    _render()
    dd.observe(_render, names="value")
    box = w.VBox([head, dd, out])
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

    # Histograma base
    fig.add_histogram(x=s, name="Top-K", nbinsx=30, opacity=0.75)

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
    low_iqr = float(low) if (low is not None) else np.nan
    high_iqr = float(high) if (high is not None) else np.nan
    sugerido = float(sugerido_val) if (sugerido_val is not None) else np.nan
    objetivo = float(objetivo_val) if (objetivo_val is not None) else np.nan
    n_topk = int(s.shape[0])
    min_n = 5

    # --- 1) Título sin solaparse (con subtítulo en <sup>) y margen superior amplio
    fig.update_layout(
        title=dict(
            text=f"Distribución Top-K: {param}<br>"
            f"<sup>LOW(IQR)={low_iqr:.3g}  ·  HIGH(IQR)={high_iqr:.3g}  ·  n_topk={n_topk}</sup>",
            x=0.02,
            xanchor="left",
            y=0.97,
            font=dict(size=16),
        ),
        margin=dict(t=110),  # deja espacio para título y posibles avisos
    )

    # --- 2) “Box overlay” del IQR (Q1–Q3) y líneas guía
    vals = np.asarray(vals_topk, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size:
        q1, q3 = np.nanpercentile(vals, [25, 75])
        # estimar altura del hist para dimensionar el rectángulo
        try:
            import numpy as _np

            ymax = max(1, int(_np.histogram(vals, bins="auto")[0].max()))
        except Exception:
            ymax = 1
        fig.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=float(q1),
            x1=float(q3),
            y0=0,
            y1=ymax,
            line=dict(width=0),
            fillcolor="rgba(255,127,80,0.15)",  # coral suave
        )

    # líneas verticales LOW/HIGH (IQR)
    if np.isfinite(low_iqr):
        fig.add_vline(
            x=float(low_iqr), line=dict(color="crimson", width=2, dash="dash")
        )
    if np.isfinite(high_iqr):
        fig.add_vline(
            x=float(high_iqr), line=dict(color="crimson", width=2, dash="dash")
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

    # --- 3) Avisos (⚠️) bien posicionados y legibles
    warnings = []
    if np.isfinite(objetivo) and (objetivo < low_iqr or objetivo > high_iqr):
        warnings.append("⚠️ objetivo fuera del IQR")
    if (n_topk is not None) and (min_n is not None) and (n_topk < min_n):
        warnings.append(f"⚠️ muestra Top-K pequeña (n={n_topk} < {min_n})")

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
            bgcolor="rgba(255,245,204,.9)",  # amarillo muy suave
            bordercolor="rgba(0,0,0,.1)",
            borderwidth=1,
        )
        fig.update_layout(margin=dict(t=150))  # un poco más de aire si hay avisos

    # estéticas suaves
    fig.update_layout(
        bargap=0.15,
        template="plotly_white",
        hoverlabel=dict(namelength=-1),  # no truncar nombres
        xaxis_title="valor",
        yaxis_title="frecuencia",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
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
) -> w.Accordion:
    """
    Accordion con:
      1) Resumen de sugerencias por parámetro (tabla)
      2) Detalle por parámetro (Dropdown + histograma/box Plotly + referencias)
      3) Insumos Top-K (tabla de vecinos usados)
    """
    cfg = dict(
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
    )

    # Controles arriba
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

    # Salidas
    out_resumen = w.Output()
    out_detalle = w.Output()
    out_insumos = w.Output()

    # Dropdown de parámetro en el panel de Detalle
    # Si no se pasa `params`, autodetectamos numéricas comunes del ranking (excluyendo auxiliares)
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
        with out_resumen:
            clear_output(wait=True)
            df_summary = sug["summary"].copy()
            display(df_summary)

        # 2) Detalle por parámetro
        with out_detalle:
            clear_output(wait=True)
            par = dd_param.value
            if par not in sug["details"]:
                display(w.HTML("<i>Sin datos para el parámetro elegido.</i>"))
            else:
                # serie top-k y referencias
                det_pack = sug["details"][par]
                usados = det_pack.get("usados", pd.DataFrame())
                s_vals = usados.get("valor", pd.Series(dtype=float))

                # Intentar recuperar objetivo/sugerido/low/high si existen en summary
                objetivo_val = None  # no está explícito en output actual
                resumen = (
                    sug["summary"].set_index("parametro")
                    if "parametro" in sug["summary"].columns
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
                display(fig)

                # Números clave
                rows = [
                    {
                        "parámetro": par,
                        "n_topk": int(
                            pd.to_numeric(s_vals, errors="coerce").dropna().shape[0]
                        ),
                        "sugerido (w_mediana)": sugerido_val,
                        "LOW(IQR)": low,
                        "HIGH(IQR)": high,
                    }
                ]
                display(pd.DataFrame(rows))

        # 3) Insumos (vecinos usados)
        with out_insumos:
            clear_output(wait=True)
            df_k = sug.get("neighbors", None)
            if df_k is None or df_k.empty:
                display(w.HTML("<i>Sin vecinos disponibles.</i>"))
            else:
                display(df_k)

    btn.on_click(lambda _: _render())

    # Render inicial
    _render()

    # Armar Accordion
    box_resumen = w.VBox([header, w.HTML("<hr>"), out_resumen])
    box_detalle = w.VBox([dd_param, w.HTML("<hr>"), out_detalle])
    box_insumos = w.VBox([out_insumos])

    acc = w.Accordion(children=[box_resumen, box_detalle, box_insumos])
    acc.set_title(0, "Sugerencias Top-K · Resumen")
    acc.set_title(1, "Detalle por parámetro")
    acc.set_title(2, "Vecinos (insumos)")
    acc.selected_index = None  # colapsado por defecto
    return acc


def widget_sugerencias_panel(
    data,
    *,
    # Back-compat args (cuando se pasa un dict 'sug'):
    titulo: str = "Sugerencias (Top-K)",
    collapsed: bool = True,
    bins: int = 20,
    # Nueva API (cuando se pasa df_ranked):
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
) -> w.Accordion:
    """
    Wrapper compatible:
    - Si 'data' es un dict de sugerencias (keys: summary, details, neighbors, config),
      arma un Accordion simple con Resumen + Detalle (Plotly) + Vecinos.
    - Si 'data' es un DataFrame (df_ranked), delega al panel Plotly recalculable
      (widget_sugerencias_panel_plotly) con los parámetros provistos.
    """
    # Caso 1: dict de sugerencias ya calculadas
    if isinstance(data, dict) and {"summary", "details"}.issubset(set(data.keys())):
        try:
            import ipywidgets as w  # local import
            from IPython.display import display, clear_output  # noqa: F401
        except Exception as e:
            raise RuntimeError("Este widget requiere 'ipywidgets' instalado.") from e

        out_resumen = w.Output()
        with out_resumen:
            if isinstance(data.get("summary"), pd.DataFrame):
                display(vista_sugerencias_resumen(data["summary"]))
            else:
                display(w.HTML("<i>Sin resumen disponible.</i>"))

        box_resumen = w.VBox([w.HTML(f"<b>{titulo} — resumen</b>"), out_resumen])
        box_detalle = widget_sugerencias_param(data, bins=bins, titulo=titulo)

        out_insumos = w.Output()
        with out_insumos:
            neigh = data.get("neighbors", None)
            if isinstance(neigh, pd.DataFrame) and not neigh.empty:
                display(neigh)
            else:
                display(w.HTML("<i>Sin vecinos Top-K disponibles.</i>"))
        box_insumos = w.VBox([out_insumos])

        acc = w.Accordion(children=[box_resumen, box_detalle, box_insumos])
        acc.set_title(0, "Resumen")
        acc.set_title(1, "Detalle por parámetro")
        acc.set_title(2, "Vecinos (insumos)")
        acc.selected_index = None if collapsed else 0
        return acc

    # Caso 2: DataFrame del ranking → delegar al panel Plotly modular
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
        )

    # Tipo no soportado
    raise TypeError(
        "widget_sugerencias_panel: primer argumento debe ser dict de sugerencias o DataFrame (ranking)."
    )
