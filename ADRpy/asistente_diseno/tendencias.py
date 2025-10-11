# -*- coding: utf-8 -*-
"""
TENDENCIAS X–Y (comparador bivariado)
-------------------------------------
- Selección de variables X e Y (drop-downs)
- Modo GLOBAL: un ajuste sobre todo el set (opcionalmente sin atípicos IQR)
- Modo POR MISIÓN: un ajuste por cada misión (segmento) con n suficiente
- Modelos candidatos (automático): lineal, cuadrático, log, exponencial, potencia
- Métricas: n, R² ajustado, MAPE (%), etiqueta de calidad por n
- Gráfico: nube + curva(s) de tendencia + leyenda con ecuación y métricas

No hace predicciones para imputar. Es una herramienta **descriptiva**.
"""

from __future__ import annotations
import copy
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Reutilizamos límites IQR del módulo de atípicos
from .outliers import compute_iqr_bounds

# Para UI
import ipywidgets as w
from IPython.display import display, clear_output

# Config (nombre de columna de misión + etiquetas legibles si existen)
from .config import SEGMENT_COL, SEGMENT_LABELS
from .guias import apply_tooltip
from .mplutils import apply_tickformat_2dec, style_df_2dec, f2

# Cache ligera para recomputaciones cercanas
_CACHE_PREP: dict[tuple, dict] = {}


def _prep_key(
    df: pd.DataFrame,
    x: str,
    y: str,
    modo: str,
    remove_outliers: bool,
    iqr_factor: float,
    min_n: int,
) -> tuple:
    return (
        id(df),
        x,
        y,
        str(modo),
        bool(remove_outliers),
        round(float(iqr_factor), 4),
        int(min_n),
        df.shape,
    )


# ------------------------------- Utilidades -------------------------------- #


def _safe_log(v: pd.Series) -> pd.Series:
    """ln(x) con seguridad; x<=0 -> NaN."""
    x = pd.to_numeric(v, errors="coerce")
    x = x.where(x > 0, np.nan)
    # Aseguramos devolver un pandas.Series para satisfacer tipado estático
    logged = np.log(x)
    return pd.Series(logged, index=x.index, name=getattr(v, "name", None))


def _adj_r2(y_true: np.ndarray, y_hat: np.ndarray, p: int) -> float:
    """R² ajustado robusto a NaN."""
    m = (~np.isnan(y_true)) & (~np.isnan(y_hat))
    n = int(np.sum(m))
    if n <= p + 1:
        return np.nan
    y = y_true[m]
    yb = y_hat[m]
    ss_res = float(np.sum((y - yb) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0:
        return np.nan
    r2 = 1.0 - ss_res / ss_tot
    return 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)


def _mape(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    """MAPE (%) robusto (ignora y=0 o NaN)."""
    m = (~np.isnan(y_true)) & (~np.isnan(y_hat)) & (np.abs(y_true) > 1e-12)
    if not np.any(m):
        return np.nan
    ape = np.abs((y_true[m] - y_hat[m]) / y_true[m])
    return float(np.mean(ape) * 100.0)


def fig_tendencias_plotly(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    segment_col: str = SEGMENT_COL,
    modo: str = "global",
    remove_outliers: bool = True,
    iqr_factor: float = 1.5,
    min_n: int = 5,
) -> Tuple[Any, pd.DataFrame]:
    """Genera figura Plotly y DataFrame de métricas para las tendencias."""
    import plotly.graph_objects as go

    info = preparar_tendencias(
        df,
        x_col,
        y_col,
        segment_col=segment_col,
        modo=modo,
        remove_outliers=remove_outliers,
        iqr_factor=iqr_factor,
        min_n=min_n,
    )
    datos = info["datos"]
    sub = datos[datos["mask"]]

    try:
        detected_col = _detect_name_col(df)
        if detected_col:
            name_series = df[detected_col].astype(str)
        else:
            name_series = pd.Series(df.index.astype(str), index=df.index)
    except Exception:
        name_series = pd.Series(df.index.astype(str), index=df.index)

    fig = go.Figure()
    if sub.empty:
        fig.add_annotation(
            text="Sin datos válidos con los parámetros actuales",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
        )
    else:
        xvals = sub["x"].to_numpy()
        yvals = sub["y"].to_numpy()
        names_vals = name_series.loc[sub.index].to_numpy()

        if modo == "global":
            fig.add_scatter(
                x=xvals,
                y=yvals,
                mode="markers",
                name="Datos",
                customdata=names_vals.reshape(-1, 1),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>X=%{x:.2f}<br>Y=%{y:.2f}<extra></extra>"
                ),
                marker=dict(size=8, opacity=0.85),
            )
            best = info.get("global", {}).get("fit")
            if best is not None:
                xs = np.linspace(np.nanmin(xvals), np.nanmax(xvals), 220)
                ys = _fit_predict(best, xs)
                if ys is not None:
                    fig.add_scatter(
                        x=xs,
                        y=ys,
                        mode="lines",
                        name=f"Tendencia ({best.nombre})",
                        line=dict(width=3),
                    )
        else:
            por_mision = info.get("por_mision", {}) or {}
            for label, grupo in sub.groupby("segmento"):
                xv = grupo["x"].to_numpy()
                yv = grupo["y"].to_numpy()
                nm = name_series.loc[grupo.index].to_numpy()
                fig.add_scatter(
                    x=xv,
                    y=yv,
                    mode="markers",
                    name=f"Datos: {label}",
                    customdata=nm.reshape(-1, 1),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>X=%{x:.2f}<br>Y=%{y:.2f}<extra></extra>"
                    ),
                    marker=dict(size=8, opacity=0.85),
                )
                fit = por_mision.get(label, {}).get("fit")
                if fit is not None:
                    xs = np.linspace(np.nanmin(xv), np.nanmax(xv), 180)
                    ys = _fit_predict(fit, xs)
                    if ys is not None:
                        fig.add_scatter(
                            x=xs,
                            y=ys,
                            mode="lines",
                            name=f"Tendencia: {label}",
                            line=dict(width=2.5),
                        )

    fig.update_layout(
        height=420,
        template="plotly_white",
        margin=dict(l=60, r=20, t=50, b=60),
        title=f"Tendencias (X–Y): {x_col} vs {y_col}",
        title_x=0.01,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_title=x_col,
        yaxis_title=y_col,
    )
    try:
        apply_tickformat_2dec(fig)
    except Exception:
        pass

    rows: List[Dict[str, Any]] = []
    if modo == "global":
        g = info.get("global", {})
        r = g.get("resumen")
        if r:
            rows.append(
                {
                    "ámbito": "Global",
                    "n": r.n,
                    "modelo": r.modelo,
                    "ecuación": r.ecuacion,
                    "R²_adj": None if r.r2_adj is None else round(r.r2_adj, 4),
                    "MAPE_%": None if r.mape is None else round(r.mape, 2),
                    "calidad_n": r.calidad_n,
                    "IQR": "ON" if remove_outliers else "OFF",
                    "min_n": int(min_n),
                }
            )
    else:
        for lab, obj in (info.get("por_mision", {}) or {}).items():
            r = obj["resumen"]
            rows.append(
                {
                    "ámbito": lab,
                    "n": r.n,
                    "modelo": r.modelo,
                    "ecuación": r.ecuacion,
                    "R²_adj": None if r.r2_adj is None else round(r.r2_adj, 4),
                    "MAPE_%": None if r.mape is None else round(r.mape, 2),
                    "calidad_n": r.calidad_n,
                    "IQR": "ON" if remove_outliers else "OFF",
                    "min_n": int(min_n),
                }
            )

    df_metrics = pd.DataFrame(rows)
    return fig, df_metrics


@dataclass
class FitResult:
    nombre: str
    params: Tuple[float, ...]
    ecuacion: str
    r2_adj: float
    mape: float


def _fmt_coef(value: float) -> str:
    try:
        return f2(value)
    except Exception:
        return f"{value:.3f}"


def _equation_from_params(nombre: str, params: Tuple[float, ...]) -> str:
    if nombre == "lineal":
        a, b = params
        return f"y = {_fmt_coef(a)}·x + {_fmt_coef(b)}"
    if nombre == "cuadrático":
        a, b, c = params
        return f"y = {_fmt_coef(a)}·x² + {_fmt_coef(b)}·x + {_fmt_coef(c)}"
    if nombre == "log":
        a, b = params
        return f"y = {_fmt_coef(a)}·ln(x) + {_fmt_coef(b)}"
    if nombre == "exp":
        a, b = params
        return f"y = {_fmt_coef(a)}·e^({_fmt_coef(b)}·x)"
    if nombre == "potencia":
        a, b = params
        return f"y = {_fmt_coef(a)}·x^{_fmt_coef(b)}"
    return ""


def _fit_predict(result: Optional[FitResult], xs: np.ndarray) -> Optional[np.ndarray]:
    if result is None:
        return None
    nombre = result.nombre
    params = result.params
    if nombre == "lineal":
        a, b = params
        return a * xs + b
    if nombre == "cuadrático":
        a, b, c = params
        return a * xs**2 + b * xs + c
    if nombre == "log":
        a, b = params
        xs = np.asarray(xs, dtype=float)
        xs_pos = np.where(xs > 0, xs, np.nan)
        return a * np.log(xs_pos) + b
    if nombre == "exp":
        a, b = params
        return a * np.exp(b * xs)
    if nombre == "potencia":
        a, b = params
        xs = np.asarray(xs, dtype=float)
        xs_pos = np.where(xs > 0, xs, np.nan)
        return a * np.power(xs_pos, b)
    return None


def _build_fit(
    nombre: str, params: Tuple[float, ...], x: np.ndarray, y: np.ndarray
) -> Optional[FitResult]:
    try:
        y_hat = _fit_predict(FitResult(nombre, params, "", np.nan, np.nan), x)
    except Exception:
        return None
    if y_hat is None:
        return None
    r2_adj = _adj_r2(y, y_hat, p=len(params))
    if not np.isfinite(r2_adj):
        return None
    mape = _mape(y, y_hat)
    ecuacion = _equation_from_params(nombre, params)
    return FitResult(
        nombre=nombre, params=params, ecuacion=ecuacion, r2_adj=r2_adj, mape=mape
    )


def _best_fit(x: pd.Series, y: pd.Series) -> Optional[FitResult]:
    x_num = pd.to_numeric(x, errors="coerce")
    y_num = pd.to_numeric(y, errors="coerce")
    mask = x_num.notna() & y_num.notna()
    if not bool(mask.any()):
        return None
    x_vals = x_num[mask].to_numpy(dtype=float)
    y_vals = y_num[mask].to_numpy(dtype=float)
    n = len(x_vals)
    if n < 3:
        return None

    candidates: List[FitResult] = []

    try:
        params_lin = tuple(np.polyfit(x_vals, y_vals, 1))
        fit = _build_fit("lineal", params_lin, x_vals, y_vals)
        if fit is not None:
            candidates.append(fit)
    except Exception:
        pass

    if n >= 3:
        try:
            params_quad = tuple(np.polyfit(x_vals, y_vals, 2))
            fit = _build_fit("cuadrático", params_quad, x_vals, y_vals)
            if fit is not None:
                candidates.append(fit)
        except Exception:
            pass

    try:
        mask_log = x_vals > 0
        if np.count_nonzero(mask_log) >= 3:
            xv = np.log(x_vals[mask_log])
            params_log = tuple(np.polyfit(xv, y_vals[mask_log], 1))
            fit = _build_fit("log", params_log, x_vals, y_vals)
            if fit is not None:
                candidates.append(fit)
    except Exception:
        pass

    try:
        mask_exp = y_vals > 0
        if np.count_nonzero(mask_exp) >= 3:
            coeffs = np.polyfit(x_vals[mask_exp], np.log(y_vals[mask_exp]), 1)
            b = float(coeffs[0])
            a = float(np.exp(coeffs[1]))
            fit = _build_fit("exp", (a, b), x_vals, y_vals)
            if fit is not None:
                candidates.append(fit)
    except Exception:
        pass

    try:
        mask_pow = (x_vals > 0) & (y_vals > 0)
        if np.count_nonzero(mask_pow) >= 3:
            coeffs = np.polyfit(np.log(x_vals[mask_pow]), np.log(y_vals[mask_pow]), 1)
            b = float(coeffs[0])
            a = float(np.exp(coeffs[1]))
            fit = _build_fit("potencia", (a, b), x_vals, y_vals)
            if fit is not None:
                candidates.append(fit)
    except Exception:
        pass

    valid = [c for c in candidates if c is not None and np.isfinite(c.r2_adj)]
    if not valid:
        return None

    def _key(res: FitResult) -> Tuple[float, float]:
        r = res.r2_adj if np.isfinite(res.r2_adj) else -np.inf
        m = res.mape if res.mape is not None and np.isfinite(res.mape) else np.inf
        return (r, -m)

    best = max(valid, key=_key)
    return best


def _iqr_mask_pair(
    x: pd.Series,
    y: pd.Series,
    *,
    factor: float,
    min_n: int,
) -> pd.Series:
    x_num = pd.to_numeric(x, errors="coerce")
    y_num = pd.to_numeric(y, errors="coerce")
    base_mask = x_num.notna() & y_num.notna()
    if base_mask.sum() < max(min_n, 3):
        return base_mask

    info_x = compute_iqr_bounds(x_num[base_mask], factor=factor, min_n=min_n)
    info_y = compute_iqr_bounds(y_num[base_mask], factor=factor, min_n=min_n)

    mask = base_mask.copy()
    if info_x.get("usable"):
        mask &= (x_num >= info_x["low"]) & (x_num <= info_x["high"])
    if info_y.get("usable"):
        mask &= (y_num >= info_y["low"]) & (y_num <= info_y["high"])
    return mask


def _detect_name_col(df: pd.DataFrame) -> Optional[str]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    candidates = [c for c in df.columns if df[c].dtype == object]
    if not candidates:
        return None

    def _score(col: str) -> int:
        score = 0
        name = col.lower()
        if any(token in name for token in ("nombre", "name", "modelo", "description")):
            score += 2
        try:
            unique_ratio = df[col].astype(str).nunique(dropna=True) / max(len(df), 1)
            if unique_ratio >= 0.6:
                score += 1
        except Exception:
            pass
        return score

    ranked = sorted(candidates, key=lambda c: (_score(c), len(c)), reverse=True)
    return ranked[0] if ranked else None


# ----------------------------- API de cálculo ------------------------------ #


@dataclass
class TendenciaResumen:
    n: int
    modelo: Optional[str]
    ecuacion: Optional[str]
    r2_adj: Optional[float]
    mape: Optional[float]
    calidad_n: str  # "mala" | "aceptable" | "buena"


def _calidad_n(n: int) -> str:
    return "buena" if n >= 30 else ("aceptable" if n >= 12 else "mala")


def preparar_tendencias(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    segment_col: str = SEGMENT_COL,
    modo: str = "global",  # "global" | "familia"
    remove_outliers: bool = True,
    iqr_factor: float = 1.5,
    min_n: int = 5,
) -> Dict[str, Any]:
    """
    Devuelve un dict con:
      - 'datos': DF limpio (x,y,segmento,mask)
      - 'global': TendenciaResumen + FitResult
      - 'por_mision': dict[label] -> TendenciaResumen + FitResult
    """
    cache_key = _prep_key(
        df,
        x_col,
        y_col,
        modo,
        remove_outliers,
        iqr_factor,
        min_n,
    )
    cached = _CACHE_PREP.get(cache_key)
    if cached is not None:
        return copy.deepcopy(cached)

    # Datos base
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    seg = (
        df[segment_col]
        if segment_col in df.columns
        else pd.Series(["(sin)"] * len(df), index=df.index)
    )

    mask = x.notna() & y.notna()
    if remove_outliers:
        mask &= _iqr_mask_pair(x, y, factor=iqr_factor, min_n=min_n)

    datos = pd.DataFrame(
        {
            "x": x,
            "y": y,
            "segmento": seg.astype(str)
            .map({str(k): v for k, v in SEGMENT_LABELS.items()})
            .fillna(seg.astype(str)),
            "mask": mask,
        }
    )

    resp: Dict[str, Any] = {"datos": datos}

    # GLOBAL
    if modo == "global":
        sub = datos[datos["mask"]]
        n = int(len(sub))
        best = _best_fit(sub["x"], sub["y"]) if n >= min_n else None
        resumen = TendenciaResumen(
            n=n,
            modelo=None if best is None else best.nombre,
            ecuacion=None if best is None else best.ecuacion,
            r2_adj=None if best is None else best.r2_adj,
            mape=None if best is None else best.mape,
            calidad_n=_calidad_n(n),
        )
        resp["global"] = {"resumen": resumen, "fit": best}

    # POR FAMILIA (MISIÓN)
    if modo == "familia":
        por = {}
        for etiqueta, sub in datos[datos["mask"]].groupby("segmento"):
            n = int(len(sub))
            best = _best_fit(sub["x"], sub["y"]) if n >= min_n else None
            resumen = TendenciaResumen(
                n=n,
                modelo=None if best is None else best.nombre,
                ecuacion=None if best is None else best.ecuacion,
                r2_adj=None if best is None else best.r2_adj,
                mape=None if best is None else best.mape,
                calidad_n=_calidad_n(n),
            )
            por[etiqueta] = {"resumen": resumen, "fit": best}
        resp["por_mision"] = por

    if len(_CACHE_PREP) >= 16:
        _CACHE_PREP.clear()
    stored = copy.deepcopy(resp)
    _CACHE_PREP[cache_key] = stored
    return copy.deepcopy(stored)


# ------------------------------ Widget (UI) -------------------------------- #


def widget_tendencias(
    df: pd.DataFrame,
    *,
    x_default: Optional[str] = None,
    y_default: Optional[str] = None,
    modo_default: str = "global",
    remove_outliers_default: bool = True,
    iqr_factor_default: float = 1.5,
    min_n_default: int = 5,
    logx_default: bool = False,
    # mantener API (enlace externo opcional)
    x_obj_col_name: Optional[str] = None,
    x_obj_widget: Optional[w.Widget] = None,
    auto_from_widget: bool = True,
) -> w.Accordion:
    """Compatibilidad: redirige a widget_tendencias_plotly (única implementación)."""
    return widget_tendencias_plotly(
        df,
        x_default=x_default,
        y_default=y_default,
        modo_default=modo_default,
        remove_outliers_default=remove_outliers_default,
        iqr_factor_default=iqr_factor_default,
        min_n_default=min_n_default,
        x_obj_col_name=x_obj_col_name,
        x_obj_widget=x_obj_widget,
    )


def widget_tendencias_plotly(
    df: pd.DataFrame,
    *,
    df_filtrado: Optional[pd.DataFrame] = None,
    x_default: Optional[str] = None,
    y_default: Optional[str] = None,
    modo_default: str = "global",
    remove_outliers_default: bool = True,
    iqr_factor_default: float = 1.5,
    min_n_default: int = 5,
    # enlace opcional de línea vertical
    x_obj_col_name: Optional[str] = None,
    x_obj_widget: Optional[w.Widget] = None,
    # NUEVO: callback de objetivo centralizado (columna -> valor)
    get_objetivo: Optional[Callable[[str], Optional[float]]] = None,
) -> w.Accordion:
    """Widget Plotly puro que usa fig_tendencias_plotly para renderizar."""
    import plotly.graph_objects as go

    # columnas numéricas candidatas considerando DF original y filtrado
    frames_for_columns: list[pd.DataFrame] = []
    if isinstance(df, pd.DataFrame):
        frames_for_columns.append(df)
    filtered_valid = (
        df_filtrado
        if (
            df_filtrado is not None
            and isinstance(df_filtrado, pd.DataFrame)
            and not df_filtrado.empty
        )
        else None
    )
    if filtered_valid is not None:
        frames_for_columns.append(filtered_valid)

    num_cols: list[str] = []
    for frame in frames_for_columns:
        for c in frame.columns:
            if c in num_cols:
                continue
            try:
                valid = pd.to_numeric(frame[c], errors="coerce").notna().sum()
            except Exception:
                valid = 0
            if valid >= 5:
                num_cols.append(c)
    if not num_cols:
        return w.Accordion(
            children=[w.HTML("<b>No hay columnas numéricas suficientes.</b>")]
        )

    dd_x = w.Dropdown(
        options=num_cols,
        value=(x_default or num_cols[0]),
        description="X:",
        layout=w.Layout(width="34%"),
    )
    dd_y = w.Dropdown(
        options=num_cols,
        value=(y_default or (num_cols[1] if len(num_cols) > 1 else num_cols[0])),
        description="Y:",
        layout=w.Layout(width="34%"),
    )
    dd_modo = w.Dropdown(
        options=[("Global", "global"), ("Por misión", "familia")],
        value=("familia" if modo_default == "familia" else "global"),
        description="Modo:",
        layout=w.Layout(width="22%"),
    )
    ch_out = w.Checkbox(
        value=remove_outliers_default, description="Quitar atípicos (IQR)"
    )
    ft_iqr = w.FloatText(
        value=iqr_factor_default,
        description="factor IQR",
        layout=w.Layout(width="150px"),
    )
    it_min = w.IntText(
        value=min_n_default, description="min_n", layout=w.Layout(width="110px")
    )
    btn = w.Button(description="Recalcular")
    btn.style.button_color = "#28a745"

    # tooltips
    apply_tooltip(dd_x, "t_x")
    apply_tooltip(dd_y, "t_y")
    apply_tooltip(dd_modo, "modo_global_familia")
    apply_tooltip(ch_out, "iqr_on")
    apply_tooltip(ft_iqr, "iqr_factor")
    apply_tooltip(it_min, "min_n")
    apply_tooltip(btn, "auto")

    # Controles en dos filas para evitar scroll horizontal
    top1 = w.HBox([dd_x, dd_y, dd_modo])
    top2 = w.HBox([ch_out, ft_iqr, it_min, btn])
    out_plot = w.Output()
    out_plot.layout = w.Layout(width="100%", height="420px")
    out_plot_filtrado = w.Output()
    out_plot_filtrado.layout = w.Layout(width="100%", height="420px")
    out_tbl = w.Output()
    n_left = len(df) if isinstance(df, pd.DataFrame) else 0
    n_right = len(filtered_valid) if isinstance(filtered_valid, pd.DataFrame) else 0
    hdr_left = w.HTML(f"<b>DataFrame original</b> <small>(n={n_left})</small>")
    hdr_right = w.HTML(f"<b>DataFrame filtrado</b> <small>(n={n_right})</small>")
    try:
        hdr_left.tooltip = "Todo el dataset, sin restricciones."
        hdr_right.tooltip = (
            "Subconjunto que cumple la selección (mín./máx./rango/fijo/objetivo)."
        )
    except Exception:
        pass
    state: Dict[str, Optional[pd.DataFrame]] = {"df_filtrado": filtered_valid}

    col_left = w.VBox([hdr_left, out_plot])
    col_left.layout = w.Layout(width="50%", flex="1 1 0%", min_width="0")
    col_right = w.VBox([hdr_right, out_plot_filtrado])
    col_right.layout = w.Layout(width="50%", flex="1 1 0%", min_width="0")

    def _valor_objetivo(col_name: str) -> Optional[float]:
        vline_val: Optional[float] = None
        if callable(get_objetivo):
            try:
                ov = get_objetivo(col_name)
                if ov is not None and np.isfinite(float(ov)):
                    vline_val = float(ov)
            except Exception:
                vline_val = None
        if vline_val is None:
            if (
                (x_obj_widget is not None)
                and (x_obj_col_name is not None)
                and col_name == x_obj_col_name
            ):
                try:
                    v = float(getattr(x_obj_widget, "value", np.nan))
                    if np.isfinite(v):
                        vline_val = float(v)
                except Exception:
                    vline_val = None
        return vline_val

    def _postprocess_fig(fig: Any, col_name: str) -> None:
        try:
            vline_val = _valor_objetivo(col_name)
            if vline_val is not None:
                fig.add_vline(
                    x=vline_val, line=dict(color="gray", width=1, dash="dash")
                )
        except Exception:
            pass
        try:
            apply_tickformat_2dec(fig)
        except Exception:
            pass

    def _update_headers(
        orig_df: Optional[pd.DataFrame], filt_df: Optional[pd.DataFrame]
    ) -> None:
        n_orig = len(orig_df) if isinstance(orig_df, pd.DataFrame) else 0
        n_filt = len(filt_df) if isinstance(filt_df, pd.DataFrame) else 0
        hdr_left.value = f"<b>DataFrame original</b> <small>(n={n_orig})</small>"
        hdr_right.value = f"<b>DataFrame filtrado</b> <small>(n={n_filt})</small>"

    def _render(*_):
        dfm_main: Optional[pd.DataFrame] = None

        current_filtered = state.get("df_filtrado")
        _update_headers(
            df, current_filtered if isinstance(current_filtered, pd.DataFrame) else None
        )

        with out_plot:
            clear_output(wait=True)
            if not isinstance(df, pd.DataFrame) or df.empty:
                display(
                    w.HTML(
                        "<i>No se pudo generar la tendencia: DataFrame original vacío o inválido.</i>"
                    )
                )
            else:
                try:
                    fig, dfm_main = fig_tendencias_plotly(
                        df,
                        dd_x.value,
                        dd_y.value,
                        modo=dd_modo.value,
                        remove_outliers=bool(ch_out.value),
                        iqr_factor=float(ft_iqr.value),
                        min_n=int(it_min.value),
                    )
                except Exception as exc:
                    display(
                        w.HTML(
                            f"<i>No se pudo generar la tendencia con el DataFrame original:</i> {exc}"
                        )
                    )
                else:
                    _postprocess_fig(fig, dd_x.value)
                    fig.show()

        current_filtered = state.get("df_filtrado")
        if (
            current_filtered is None
            or not isinstance(current_filtered, pd.DataFrame)
            or current_filtered.empty
        ):
            col_right.layout.display = ""
            with out_plot_filtrado:
                clear_output(wait=True)
                display(
                    w.HTML(
                        "<i>No hay datos filtrados disponibles para este conjunto de restricciones.</i>"
                    )
                )
        else:
            col_right.layout.display = ""
            with out_plot_filtrado:
                clear_output(wait=True)
                try:
                    fig_f, _ = fig_tendencias_plotly(
                        current_filtered,
                        dd_x.value,
                        dd_y.value,
                        modo=dd_modo.value,
                        remove_outliers=bool(ch_out.value),
                        iqr_factor=float(ft_iqr.value),
                        min_n=int(it_min.value),
                    )
                except Exception as exc:
                    display(
                        w.HTML(
                            f"<i>No se pudo generar la tendencia con el DataFrame filtrado:</i> {exc}"
                        )
                    )
                else:
                    _postprocess_fig(fig_f, dd_x.value)
                    fig_f.show()

        with out_tbl:
            clear_output(wait=True)
            if isinstance(dfm_main, pd.DataFrame) and not dfm_main.empty:
                try:
                    display(style_df_2dec(dfm_main))
                except Exception:
                    display(dfm_main)

    btn.on_click(lambda _: _render())

    plots_box: w.Widget = w.HBox(
        [col_left, col_right],
        layout=w.Layout(
            width="100%",
            display="flex",
            justify_content="space-between",
            gap="12px",
        ),
    )

    acc = w.Accordion(
        children=[
            w.VBox([top1, top2, w.HTML("<hr>"), plots_box, w.HTML("<hr>"), out_tbl])
        ]
    )
    acc.set_title(0, "Tendencias (X–Y)")
    acc.selected_index = None

    def _set_df_filtrado(new_df: Optional[pd.DataFrame]) -> None:
        sanitized_df = (
            new_df if isinstance(new_df, pd.DataFrame) and not new_df.empty else None
        )
        state["df_filtrado"] = sanitized_df
        hdr_right.value = (
            f"<b>DataFrame filtrado</b> <small>(n={len(sanitized_df)})</small>"
            if isinstance(sanitized_df, pd.DataFrame)
            else "<b>DataFrame filtrado</b> <small>(n=0)</small>"
        )
        _render()

    setattr(acc, "set_df_filtrado", _set_df_filtrado)

    _render()
    return acc


# ============================================================================
# NUEVO: Comparador Global vs Filtrado (dos columnas simétricas)
# ============================================================================
def widget_tendencias_comparador(
    df: pd.DataFrame,
    *,
    df_filtrado: Optional[pd.DataFrame] = None,
    x_default: Optional[str] = None,
    y_default: Optional[str] = None,
    modo_default: str = "global",
    remove_outliers_default: bool = True,
    iqr_factor_default: float = 1.5,
    min_n_default: int = 5,
    # enlace opcional de línea vertical (por ejemplo, objetivo en X)
    x_obj_col_name: Optional[str] = None,
    x_obj_widget: Optional[w.Widget] = None,
    get_objetivo: Optional[Callable[[str], Optional[float]]] = None,
) -> w.Accordion:
    """Renderiza tendencias comparando DF original vs filtrado, con métricas en paralelo."""

    frames_for_columns: list[pd.DataFrame] = []
    if isinstance(df, pd.DataFrame):
        frames_for_columns.append(df)
    filtered_valid = (
        df_filtrado
        if (
            df_filtrado is not None
            and isinstance(df_filtrado, pd.DataFrame)
            and not df_filtrado.empty
        )
        else None
    )
    if filtered_valid is not None:
        frames_for_columns.append(filtered_valid)

    num_cols: list[str] = []
    for frame in frames_for_columns:
        for c in frame.columns:
            if c in num_cols:
                continue
            try:
                valid = pd.to_numeric(frame[c], errors="coerce").notna().sum()
            except Exception:
                valid = 0
            if valid >= 5:
                num_cols.append(c)
    if not num_cols:
        return w.Accordion(
            children=[w.HTML("<b>No hay columnas numéricas suficientes.</b>")]
        )

    def _default_value(preferred: Optional[str], candidates: list[str]) -> str:
        if preferred and preferred in candidates:
            return preferred
        return candidates[0]

    x_initial = _default_value(x_default, num_cols)
    y_candidates = [c for c in num_cols if c != x_initial] or [x_initial]
    y_initial = _default_value(y_default, y_candidates)

    dd_x = w.Dropdown(
        options=num_cols,
        value=x_initial,
        description="X:",
        layout=w.Layout(width="34%"),
    )
    dd_y = w.Dropdown(
        options=y_candidates,
        value=y_initial,
        description="Y:",
        layout=w.Layout(width="34%"),
    )

    def _sync_y(*_):
        opts = [c for c in num_cols if c != dd_x.value]
        if not opts:
            opts = [dd_x.value]
        dd_y.options = opts
        if dd_y.value not in opts:
            dd_y.value = opts[0]

    dd_x.observe(lambda *_: _sync_y(), names="value")

    dd_modo = w.Dropdown(
        options=[("Global", "global"), ("Por misión", "familia")],
        value=("familia" if modo_default == "familia" else "global"),
        description="Modo:",
        layout=w.Layout(width="22%"),
    )
    ch_out = w.Checkbox(
        value=remove_outliers_default, description="Quitar atípicos (IQR)"
    )
    ft_iqr = w.FloatText(
        value=iqr_factor_default,
        description="factor IQR",
        layout=w.Layout(width="150px"),
    )
    it_min = w.IntText(
        value=min_n_default, description="min_n", layout=w.Layout(width="110px")
    )
    btn = w.Button(description="Recalcular")
    btn.style.button_color = "#28a745"

    apply_tooltip(dd_x, "t_x")
    apply_tooltip(dd_y, "t_y")
    apply_tooltip(dd_modo, "modo_global_familia")
    apply_tooltip(ch_out, "iqr_on")
    apply_tooltip(ft_iqr, "iqr_factor")
    apply_tooltip(it_min, "min_n")
    apply_tooltip(btn, "auto")

    top1 = w.HBox([dd_x, dd_y, dd_modo])
    top2 = w.HBox([ch_out, ft_iqr, it_min, btn])

    out_plot_left, out_tbl_left = w.Output(), w.Output()
    out_plot_right, out_tbl_right = w.Output(), w.Output()

    hdr_left = w.HTML("<b>DataFrame original</b> <small>(n=0)</small>")
    hdr_right = w.HTML("<b>DataFrame filtrado</b> <small>(n=0)</small>")

    try:
        hdr_left.tooltip = "Todo el dataset, sin restricciones."
        hdr_right.tooltip = (
            "Subconjunto que cumple la selección (mín./máx./rango/fijo/objetivo)."
        )
    except Exception:
        pass

    state: Dict[str, Optional[pd.DataFrame]] = {
        "df_filtrado": (
            filtered_valid if isinstance(filtered_valid, pd.DataFrame) else None
        )
    }

    def _valor_objetivo(col_name: str) -> Optional[float]:
        target: Optional[float] = None
        if callable(get_objetivo):
            try:
                target = get_objetivo(col_name)
            except Exception:
                target = None
        if target is None and x_obj_widget is not None and x_obj_col_name == col_name:
            try:
                target = float(getattr(x_obj_widget, "value", np.nan))
            except Exception:
                target = None
        if target is None:
            return None
        try:
            val = float(target)
        except Exception:
            return None
        return val if np.isfinite(val) else None

    def _add_vline(fig: Any, x_value: Optional[float]) -> None:
        if fig is None or x_value is None or not np.isfinite(x_value):
            return
        try:
            fig.add_vline(x=x_value, line=dict(color="gray", width=1, dash="dash"))
        except Exception:
            pass

    def _update_headers(
        orig_df: Optional[pd.DataFrame], filt_df: Optional[pd.DataFrame]
    ) -> None:
        n_orig = len(orig_df) if isinstance(orig_df, pd.DataFrame) else 0
        n_filt = len(filt_df) if isinstance(filt_df, pd.DataFrame) else 0
        hdr_left.value = f"<b>DataFrame original</b> <small>(n={n_orig})</small>"
        hdr_right.value = f"<b>DataFrame filtrado</b> <small>(n={n_filt})</small>"

    def _render(*_):
        metrics_left: Optional[pd.DataFrame] = None
        metrics_right: Optional[pd.DataFrame] = None

        current_filtered = state.get("df_filtrado")
        valid_filtered = (
            current_filtered if isinstance(current_filtered, pd.DataFrame) else None
        )
        _update_headers(df, valid_filtered)

        vline_val = _valor_objetivo(dd_x.value)

        with out_plot_left:
            clear_output(wait=True)
            if not isinstance(df, pd.DataFrame) or df.empty:
                display(
                    w.HTML(
                        "<i>No se pudo generar la tendencia: DataFrame original vacío o inválido.</i>"
                    )
                )
            else:
                try:
                    fig_left, metrics_left = fig_tendencias_plotly(
                        df,
                        dd_x.value,
                        dd_y.value,
                        modo=dd_modo.value,
                        remove_outliers=bool(ch_out.value),
                        iqr_factor=float(ft_iqr.value),
                        min_n=int(it_min.value),
                    )
                except Exception as exc:
                    display(
                        w.HTML(
                            f"<i>No se pudo generar la tendencia con el DataFrame original:</i> {exc}"
                        )
                    )
                    metrics_left = None
                else:
                    _add_vline(fig_left, vline_val)
                    fig_left.show()

        with out_plot_right:
            clear_output(wait=True)
            if (
                valid_filtered is None
                or not isinstance(valid_filtered, pd.DataFrame)
                or valid_filtered.empty
            ):
                display(
                    w.HTML(
                        "<i>Sin datos filtrados disponibles para este conjunto de restricciones.</i>"
                    )
                )
            else:
                try:
                    fig_right, metrics_right = fig_tendencias_plotly(
                        valid_filtered,
                        dd_x.value,
                        dd_y.value,
                        modo=dd_modo.value,
                        remove_outliers=bool(ch_out.value),
                        iqr_factor=float(ft_iqr.value),
                        min_n=int(it_min.value),
                    )
                except Exception as exc:
                    display(
                        w.HTML(
                            f"<i>No se pudo generar la tendencia con el DataFrame filtrado:</i> {exc}"
                        )
                    )
                    metrics_right = None
                else:
                    _add_vline(fig_right, vline_val)
                    fig_right.show()

        with out_tbl_left:
            clear_output(wait=True)
            if isinstance(metrics_left, pd.DataFrame) and not metrics_left.empty:
                try:
                    display(style_df_2dec(metrics_left))
                except Exception:
                    display(metrics_left)
            else:
                display(
                    w.HTML(
                        "<small><i>Sin métricas calculables para el DataFrame original.</i></small>"
                    )
                )

        with out_tbl_right:
            clear_output(wait=True)
            if isinstance(metrics_right, pd.DataFrame) and not metrics_right.empty:
                try:
                    display(style_df_2dec(metrics_right))
                except Exception:
                    display(metrics_right)
            else:
                display(
                    w.HTML(
                        "<small><i>Sin métricas calculables para el DataFrame filtrado.</i></small>"
                    )
                )

    btn.on_click(lambda *_: _render())

    col_left = w.VBox(
        [hdr_left, out_plot_left, w.HTML("<hr>"), out_tbl_left],
        layout=w.Layout(width="50%", flex="1 1 0%", min_width="0"),
    )
    col_right = w.VBox(
        [hdr_right, out_plot_right, w.HTML("<hr>"), out_tbl_right],
        layout=w.Layout(width="50%", flex="1 1 0%", min_width="0"),
    )

    body = w.VBox(
        [
            top1,
            top2,
            w.HTML("<hr>"),
            w.HBox([col_left, col_right], layout=w.Layout(width="100%", gap="12px")),
        ]
    )

    acc = w.Accordion(children=[body])
    acc.set_title(0, "Tendencias (X–Y) — Global vs Filtrado")
    acc.selected_index = None

    def _set_df_filtrado(new_df: Optional[pd.DataFrame]) -> None:
        sanitized = (
            new_df if isinstance(new_df, pd.DataFrame) and not new_df.empty else None
        )
        state["df_filtrado"] = sanitized
        _update_headers(df, sanitized)
        _render()

    setattr(acc, "set_df_filtrado", _set_df_filtrado)

    _render()
    return acc
