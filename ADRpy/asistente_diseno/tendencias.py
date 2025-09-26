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


def _iqr_mask_pair(
    x: pd.Series, y: pd.Series, factor: float = 1.5, min_n: int = 5
) -> pd.Series:
    """Máscara True para pares (x,y) dentro de límites IQR **en ambas** variables."""
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < min_n:
        return mask
    bx = compute_iqr_bounds(x[mask], factor=factor, min_n=min_n)
    by = compute_iqr_bounds(y[mask], factor=factor, min_n=min_n)
    if bx["usable"]:
        mask &= (x >= bx["low"]) & (x <= bx["high"])
    if by["usable"]:
        mask &= (y >= by["low"]) & (y <= by["high"])
    return mask


def _detect_name_col(df: pd.DataFrame) -> Optional[str]:
    """Devuelve la columna 'nombre' más probable para hover."""
    candidatos = ["Modelo", "modelo", "aeronave", "Aeronave", "Nombre", "name"]
    for c in candidatos:
        if c in df.columns:
            return c
    return None


# ---------------------------- Ajustes candidatos --------------------------- #


@dataclass
class FitResult:
    nombre: str  # "lineal", "cuadrático", "log", "exp", "potencia"
    ecuacion: str  # texto amigable
    p: int  # nº de parámetros (para R² ajustado)
    y_hat: np.ndarray  # predicción (dominio original)
    r2_adj: float
    mape: float
    params: Tuple[float, ...]  # coeficientes guardados (para leyenda)


def _fit_lineal(x: np.ndarray, y: np.ndarray) -> Optional[FitResult]:
    # y = a*x + b
    if np.sum(~np.isnan(x) & ~np.isnan(y)) < 3:
        return None
    coef = np.polyfit(x, y, 1)  # a, b
    y_hat = coef[0] * x + coef[1]
    return FitResult(
        nombre="lineal",
        ecuacion=f"y = {coef[0]:.4g}·x + {coef[1]:.4g}",
        p=1,
        y_hat=y_hat,
        r2_adj=_adj_r2(y, y_hat, p=1),
        mape=_mape(y, y_hat),
        params=(coef[0], coef[1]),
    )


def _fit_cuadratico(x: np.ndarray, y: np.ndarray) -> Optional[FitResult]:
    # y = a*x^2 + b*x + c
    if np.sum(~np.isnan(x) & ~np.isnan(y)) < 4:
        return None
    coef = np.polyfit(x, y, 2)  # a, b, c
    y_hat = coef[0] * x**2 + coef[1] * x + coef[2]
    return FitResult(
        nombre="cuadrático",
        ecuacion=f"y = {coef[0]:.4g}·x² + {coef[1]:.4g}·x + {coef[2]:.4g}",
        p=2,
        y_hat=y_hat,
        r2_adj=_adj_r2(y, y_hat, p=2),
        mape=_mape(y, y_hat),
        params=(coef[0], coef[1], coef[2]),
    )


def _fit_log(x: np.ndarray, y: np.ndarray) -> Optional[FitResult]:
    # y = a·ln(x) + b   (x>0)
    if not np.any(x > 0):
        return None
    lx = np.log(np.where(x > 0, x, np.nan))
    m = ~np.isnan(lx) & ~np.isnan(y)
    if np.sum(m) < 3:
        return None
    a, b = np.polyfit(lx[m], y[m], 1)
    y_hat = a * lx + b
    return FitResult(
        nombre="log",
        ecuacion=f"y = {a:.4g}·ln(x) + {b:.4g}",
        p=1,
        y_hat=y_hat,
        r2_adj=_adj_r2(y, y_hat, p=1),
        mape=_mape(y, y_hat),
        params=(a, b),
    )


def _fit_exp(x: np.ndarray, y: np.ndarray) -> Optional[FitResult]:
    # y = a·e^(b·x)  <=> ln(y) = ln(a) + b·x   (y>0)
    if not np.any(y > 0):
        return None
    ly = np.log(np.where(y > 0, y, np.nan))
    m = ~np.isnan(ly) & ~np.isnan(x)
    if np.sum(m) < 3:
        return None
    b, ln_a = np.polyfit(x[m], ly[m], 1)
    a = math.exp(ln_a)
    y_hat = a * np.exp(b * x)
    return FitResult(
        nombre="exp",
        ecuacion=f"y = {a:.4g}·e^({b:.4g}·x)",
        p=1,
        y_hat=y_hat,
        r2_adj=_adj_r2(y, y_hat, p=1),
        mape=_mape(y, y_hat),
        params=(a, b),
    )


def _fit_potencia(x: np.ndarray, y: np.ndarray) -> Optional[FitResult]:
    # y = a·x^b  <=> ln(y) = ln(a) + b·ln(x)   (x>0, y>0)
    if not (np.any(x > 0) and np.any(y > 0)):
        return None
    lx = np.log(np.where(x > 0, x, np.nan))
    ly = np.log(np.where(y > 0, y, np.nan))
    m = ~np.isnan(lx) & ~np.isnan(ly)
    if np.sum(m) < 3:
        return None
    b, ln_a = np.polyfit(lx[m], ly[m], 1)
    a = math.exp(ln_a)
    y_hat = a * np.power(x, b)
    return FitResult(
        nombre="potencia",
        ecuacion=f"y = {a:.4g}·x^{b:.4g}",
        p=1,
        y_hat=y_hat,
        r2_adj=_adj_r2(y, y_hat, p=1),
        mape=_mape(y, y_hat),
        params=(a, b),
    )


CANDIDATOS = (_fit_lineal, _fit_cuadratico, _fit_log, _fit_exp, _fit_potencia)


def _best_fit(x: pd.Series, y: pd.Series) -> Optional[FitResult]:
    """Prueba todos los modelos y devuelve el mejor por R² ajustado."""
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    m = x.notna() & y.notna()
    if m.sum() < 3:
        return None
    X = x[m].to_numpy()
    Y = y[m].to_numpy()
    resultados: List[FitResult] = []
    for f in CANDIDATOS:
        try:
            r = f(X, Y)
            if r is not None and np.isfinite(r.r2_adj):
                resultados.append(r)
        except Exception:
            continue
    if not resultados:
        return None
    resultados.sort(key=lambda r: (np.nan_to_num(r.r2_adj, nan=-1e9)), reverse=True)
    return resultados[0]


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

    return resp


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
    """Versión Plotly interactiva (sin matplotlib)."""
    import plotly.graph_objects as go
    from IPython.display import display, clear_output

    # columnas numéricas candidatas
    num_cols = [
        c
        for c in df.columns
        if pd.to_numeric(df[c], errors="coerce").notna().sum() >= 5
    ]
    if not num_cols:
        return w.Accordion(
            children=[w.HTML("<b>No hay columnas numéricas suficientes.</b>")]
        )

    # heurística para nombre visible
    name_col = _detect_name_col(df)
    if name_col:
        nombres = df[name_col].astype(str)
    else:
        # usar índice como nombre, pero como Serie para permitir .loc
        nombres = pd.Series(df.index.astype(str), index=df.index)

    # widgets
    dd_x = w.Dropdown(
        options=num_cols,
        value=(x_default or num_cols[0]),
        description="X:",
        layout=w.Layout(width="35%"),
    )
    dd_y = w.Dropdown(
        options=num_cols,
        value=(y_default or (num_cols[1] if len(num_cols) > 1 else num_cols[0])),
        description="Y:",
        layout=w.Layout(width="35%"),
    )
    dd_modo = w.Dropdown(
        options=[("Global", "global"), ("Por misión", "familia")],
        value=("familia" if modo_default == "familia" else "global"),
        description="Modo:",
        layout=w.Layout(width="20%"),
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
    ch_logx = w.Checkbox(value=logx_default, description="log X")
    btn = w.Button(description="Recalcular")
    btn.style.button_color = "#28a745"

    top = w.HBox([dd_x, dd_y, dd_modo, ch_out, ft_iqr, it_min, ch_logx, btn])
    out_plot = w.Output()
    out_tbl = w.Output()

    def _fit_to_xs(best, xs):
        if best is None or not np.isfinite(best.r2_adj):
            return None
        n = best.nombre
        p = best.params
        if n == "lineal":
            a, b = p
            return a * xs + b
        if n == "cuadrático":
            a, b, c = p
            return a * xs**2 + b * xs + c
        if n == "log":
            a, b = p
            xs_pos = np.where(xs > 0, xs, np.nan)
            return a * np.log(xs_pos) + b
        if n == "exp":
            a, b = p
            return a * np.exp(b * xs)
        if n == "potencia":
            a, b = p
            xs_pos = np.where(xs > 0, xs, np.nan)
            return a * np.power(xs_pos, b)
        return None

    def _render(*_):
        with out_plot:
            clear_output(wait=True)
            info = preparar_tendencias(
                df,
                dd_x.value,
                dd_y.value,
                segment_col=SEGMENT_COL,
                modo=dd_modo.value,
                remove_outliers=bool(ch_out.value),
                iqr_factor=float(ft_iqr.value),
                min_n=int(it_min.value),
            )
            datos = info["datos"]
            sub = datos[datos["mask"]]
            if sub.empty:
                display(w.HTML("<i>Sin datos válidos con los filtros actuales.</i>"))
                with out_tbl:
                    clear_output(wait=True)
                return

            # figura
            fig = go.Figure()
            xvals = sub["x"].to_numpy()
            yvals = sub["y"].to_numpy()
            names = nombres.loc[sub.index].to_numpy()

            if dd_modo.value == "global":
                # nube global
                fig.add_scatter(
                    x=xvals,
                    y=yvals,
                    mode="markers",
                    name="Datos",
                    hovertemplate="<b>%{customdata}</b><br>X=%{x:.3g}<br>Y=%{y:.3g}<extra></extra>",
                    customdata=names.reshape(-1, 1),
                    marker=dict(size=8, opacity=0.8),
                )
                # mejor ajuste
                best = info.get("global", {}).get("fit", None)
                if best is not None and np.isfinite(best.r2_adj):
                    xs = np.linspace(np.nanmin(xvals), np.nanmax(xvals), 220)
                    ys = _fit_to_xs(best, xs)
                    if ys is not None:
                        fig.add_scatter(
                            x=xs,
                            y=ys,
                            mode="lines",
                            name=f"Tendencia ({best.nombre})",
                            line=dict(width=3),
                        )
            else:
                # por misión
                for lab, grupo in sub.groupby("segmento"):
                    xv = grupo["x"].to_numpy()
                    yv = grupo["y"].to_numpy()
                    nm = nombres.loc[grupo.index].to_numpy()
                    fig.add_scatter(
                        x=xv,
                        y=yv,
                        mode="markers",
                        name=f"Datos: {lab}",
                        hovertemplate="<b>%{customdata}</b><br>X=%{x:.3g}<br>Y=%{y:.3g}<extra></extra>",
                        customdata=nm.reshape(-1, 1),
                        marker=dict(size=8, opacity=0.85),
                    )
                    obj = info.get("por_mision", {}).get(lab, None)
                    if (
                        obj
                        and obj.get("fit", None) is not None
                        and np.isfinite(obj["fit"].r2_adj)
                    ):
                        xs = np.linspace(np.nanmin(xv), np.nanmax(xv), 180)
                        ys = _fit_to_xs(obj["fit"], xs)
                        if ys is not None:
                            fig.add_scatter(
                                x=xs,
                                y=ys,
                                mode="lines",
                                name=f"Tendencia: {lab}",
                                line=dict(width=2.5),
                            )

            # Línea vertical vinculada (si corresponde)
            if (
                (x_obj_widget is not None)
                and (x_obj_col_name is not None)
                and dd_x.value == x_obj_col_name
            ):
                try:
                    v = float(getattr(x_obj_widget, "value", np.nan))
                    if np.isfinite(v):
                        fig.add_vline(
                            x=float(v), line=dict(color="gray", width=1, dash="dash")
                        )
                except Exception:
                    pass

            fig.update_layout(
                template="plotly_white",
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0
                ),
                margin=dict(l=40, r=10, t=30, b=40),
                xaxis_title=dd_x.value,
                yaxis_title=dd_y.value,
            )
            if ch_logx.value:
                fig.update_xaxes(type="log")

            display(fig)

        # tabla de métricas
        with out_tbl:
            clear_output(wait=True)
            rows = []
            if dd_modo.value == "global":
                g = info.get("global", {})
                r = g.get("resumen", None)
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
                            "IQR": "ON" if ch_out.value else "OFF",
                            "min_n": int(it_min.value),
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
                            "IQR": "ON" if ch_out.value else "OFF",
                            "min_n": int(it_min.value),
                        }
                    )
            if rows:
                display(pd.DataFrame(rows))

    # redibujar automáticamente si cambia el control externo (opcional)
    if (x_obj_widget is not None) and bool(auto_from_widget):

        def _on_ext(change):
            if change.get("name") == "value":
                _render()

        try:
            x_obj_widget.observe(_on_ext, names="value")
        except Exception:
            pass

    btn.on_click(lambda _: _render())
    acc = w.Accordion(
        children=[w.VBox([top, w.HTML("<hr>"), out_plot, w.HTML("<hr>"), out_tbl])]
    )
    acc.set_title(0, "Tendencias (X–Y)")
    acc.selected_index = None
    _render()
    return acc
