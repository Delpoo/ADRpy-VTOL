"""
Módulo de outliers (atípicos) para ADRpy.

Objetivo
--------
Brindar utilidades robustas para detectar y visualizar outliers sin modificar
el DataFrame original. Incluye:
  - IQR (Interquartile Range) y MAD (Median Absolute Deviation)
  - Máscaras booleans para series/filas
  - Anotaciones 'is_outlier_<col>'
  - Vistas listas para el notebook:
      * Resumen por columna (Q1, Q3, IQR, límites, % outliers) con Styler
      * "Quicklook" que empaqueta todo
    * Histograma por columna con límites IQR (Plotly)

Reglas base
-----------
- IQR: outlier si valor < Q1 - factor*IQR o > Q3 + factor*IQR (factor=1.5 típico)
- MAD: z_robusto = 0.6745*(x - mediana)/MAD; outlier si |z| > k (k≈3.5 típico)
- Casos límite (IQR≈0 o MAD≈0 o n pequeño): no marcamos outliers (seguro)
- NaN: por defecto se conservan (keep_na=True)

Uso rápido en notebook
----------------------
from asistente_diseno.datos import leer_excel
from asistente_diseno.outliers import outliers_quicklook, plot_outliers_hist

df = leer_excel()
report = outliers_quicklook(df, columns=["Envergadura","Cuerda"], metodo="IQR", factor=1.5)
display(report["summary_styler"])          # tabla resumen estilada
display(report["annotated_flags"].head())  # DF con columnas is_outlier_*
report["outliers_by_col"]["Envergadura"].head()  # filas outlier en esa columna
fig = plot_outliers_hist(df, "Envergadura") ; fig  # histograma con límites IQR
"""

from __future__ import annotations
from typing import Dict, Iterable, List, Tuple, Optional
import numpy as np
import pandas as pd

# from .mplutils import import_matplotlib  # <- eliminar este import
from .config import SEGMENT_COL, SEGMENT_LABELS
import plotly.graph_objects as go
import ipywidgets as w
from IPython.display import display, clear_output


# =============================================================================
# Utilidades internas
# =============================================================================


def _to_numeric_series(s: pd.Series) -> pd.Series:
    """Convierte a numérico con errors='coerce' y devuelve una copia."""
    if not isinstance(s, pd.Series):
        s = pd.Series(s)
    return pd.to_numeric(s.copy(), errors="coerce")


def _valid_numeric(s: pd.Series) -> pd.Series:
    """Retorna sólo valores numéricos finitos (sin NaN/inf)."""
    s_num = _to_numeric_series(s)
    s_num = s_num.replace([np.inf, -np.inf], np.nan).dropna()
    return s_num


# Mapeo de etiquetas de segmento con claves en str para evitar conflictos de tipo
try:
    SEGMENT_LABELS_STR = {str(k): v for k, v in SEGMENT_LABELS.items()}
except Exception:
    SEGMENT_LABELS_STR = {}


# =============================================================================
# IQR
# =============================================================================


def compute_iqr_bounds(
    serie: pd.Series, factor: float = 1.5, min_n: int = 5
) -> Dict[str, float]:
    """
    Calcula Q1, Q3, IQR y límites IQR para una serie.

    Parámetros
    ----------
    serie : pd.Series
        Serie de entrada (se convierte a numérica; NaN ignorados).
    factor : float
        Factor multiplicativo para el IQR (típico 1.5).
    min_n : int
        Mínimo de observaciones válidas requerido para intentar detectar atípicos.

    Returns
    -------
    dict con:
      - n_valido: int, cantidad de valores numéricos no NaN
      - Q1, Q3, IQR
      - low, high: límites inferior y superior
      - usable: bool, True si IQR>0 y n_valido>=min_n
    """
    s = _valid_numeric(serie)
    n = int(s.size)
    if n == 0:
        return dict(
            n_valido=0,
            Q1=np.nan,
            Q3=np.nan,
            IQR=np.nan,
            low=np.nan,
            high=np.nan,
            usable=False,
        )
    Q1 = float(np.nanpercentile(s, 25))
    Q3 = float(np.nanpercentile(s, 75))
    IQR = Q3 - Q1
    if n < min_n or IQR <= 0 or not np.isfinite(IQR):
        return dict(
            n_valido=n, Q1=Q1, Q3=Q3, IQR=IQR, low=np.nan, high=np.nan, usable=False
        )
    low = Q1 - factor * IQR
    high = Q3 + factor * IQR
    return dict(
        n_valido=n, Q1=Q1, Q3=Q3, IQR=IQR, low=float(low), high=float(high), usable=True
    )


def iqr_mask(
    serie: pd.Series, factor: float = 1.5, min_n: int = 5, keep_na: bool = True
) -> pd.Series:
    """
    Máscara de no-atípicos por IQR (True=conservar).

    Reglas:
    - Si n_valido < min_n o IQR ≤ 0 → no se marcan outliers (todo True excepto NaN si keep_na=False).
    - NaN: si keep_na=True → True (se conservan); si False → False (se eliminan).

    Returns
    -------
    pd.Series[bool] alineada con la serie de entrada.
    """
    s = _to_numeric_series(serie)
    info = compute_iqr_bounds(s, factor=factor, min_n=min_n)
    mask = pd.Series(True, index=s.index)
    if not info["usable"]:
        if not keep_na:
            mask = mask & s.notna()
        return mask
    low, high = info["low"], info["high"]
    in_range = (s >= low) & (s <= high)
    if keep_na:
        in_range = in_range | s.isna()
    return in_range


def mad_mask(
    serie: pd.Series, k: float = 3.5, min_n: int = 5, keep_na: bool = True
) -> pd.Series:
    """
    Máscara de no-atípicos por MAD (True=conservar).
    - z_robusto = 0.6745*(x - mediana)/MAD
    - Se marcan outliers si |z_robusto| > k

    Reglas:
    - Si n_valido < min_n o MAD ≈ 0 → no se marcan outliers (todo True excepto NaN si keep_na=False).
    """
    s = _to_numeric_series(serie)
    s_valid = _valid_numeric(s)
    n = int(s_valid.size)
    mask = pd.Series(True, index=s.index)
    if n < min_n:
        if not keep_na:
            mask = mask & s.notna()
        return mask
    med = float(np.nanmedian(s_valid))
    abs_dev = np.abs(s - med)
    abs_dev_series = pd.Series(abs_dev, index=s.index)
    MAD = float(np.nanmedian(abs_dev_series.dropna()))
    if MAD <= 0 or not np.isfinite(MAD):
        if not keep_na:
            mask = mask & s.notna()
        return mask
    z = 0.6745 * (s - med) / MAD
    keep = np.abs(z) <= k
    keep = pd.Series(keep, index=s.index)
    if keep_na:
        keep = keep | s.isna()
    return keep


# =============================================================================
# Operar por DataFrame (filas/columnas)
# =============================================================================


def mask_dataframe(
    df: pd.DataFrame,
    columnas: Iterable[str],
    *,
    metodo: str = "IQR",
    modo_filas: str = "todas",
    keep_na: bool = True,
    factor: float = 1.5,
    k: float = 3.5,
    min_n: int = 5,
) -> pd.Series:
    """
    Retorna una máscara booleana por FILA para un DataFrame, combinando columnas.

    modo_filas:
      - "todas": conserva filas donde TODAS las columnas son no-atípicas (AND).
      - "cualquiera": conserva filas donde AL MENOS una columna es no-atípica (OR).
    """
    if metodo.upper() not in {"IQR", "MAD"}:
        raise ValueError("metodo debe ser 'IQR' o 'MAD'")
    if modo_filas not in {"todas", "cualquiera"}:
        raise ValueError("modo_filas debe ser 'todas' o 'cualquiera'")

    masks = []
    for col in columnas:
        if col not in df.columns:
            masks.append(pd.Series(True, index=df.index))
            continue
        s = df[col]
        if metodo.upper() == "IQR":
            m = iqr_mask(s, factor=factor, min_n=min_n, keep_na=keep_na)
        else:
            m = mad_mask(s, k=k, min_n=min_n, keep_na=keep_na)
        masks.append(m.astype(bool))

    if not masks:
        return pd.Series(True, index=df.index)

    if modo_filas == "todas":
        out = masks[0].copy()
        for m in masks[1:]:
            out &= m
        return out

    out = masks[0].copy()
    for m in masks[1:]:
        out |= m
    return out


def annotate_outliers(
    df: pd.DataFrame,
    columnas: Iterable[str],
    *,
    metodo: str = "IQR",
    keep_na: bool = True,
    factor: float = 1.5,
    k: float = 3.5,
    min_n: int = 5,
    suffix: str = "is_outlier",
) -> pd.DataFrame:
    """
    Devuelve una copia del DataFrame con columnas booleanas 'is_outlier_<col>'.

    True = atípico, False = no-atípico.
    Si una columna no existe, se omite su anotación.
    """
    df2 = df.copy()
    for col in columnas:
        if col not in df2.columns:
            continue
        if metodo.upper() == "IQR":
            keep = iqr_mask(df2[col], factor=factor, min_n=min_n, keep_na=keep_na)
        else:
            keep = mad_mask(df2[col], k=k, min_n=min_n, keep_na=keep_na)
        df2[f"{suffix}_{col}"] = ~keep  # True si es atípico
    return df2


# =============================================================================
# Vistas listas para el notebook (tablas y figuras)
# =============================================================================


def iqr_summary_table(
    df: pd.DataFrame,
    columns: Iterable[str],
    *,
    factor: float = 1.5,
    min_n: int = 5,
    keep_na: bool = True,
) -> pd.DataFrame:
    """
    Tabla resumen por columna con Q1, Q3, IQR, límites IQR y % de outliers.
    No modifica df.
    """
    rows = []
    for col in columns:
        if col not in df.columns:
            rows.append(
                {
                    "columna": col,
                    "n_validos": 0,
                    "Q1": np.nan,
                    "Q3": np.nan,
                    "IQR": np.nan,
                    "low": np.nan,
                    "high": np.nan,
                    "usable": False,
                    "n_outliers": 0,
                    "%_outliers": np.nan,
                }
            )
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        info = compute_iqr_bounds(s, factor=factor, min_n=min_n)
        keep = iqr_mask(s, factor=factor, min_n=min_n, keep_na=keep_na)
        n_total = s.notna().sum()
        n_out = int((~keep & s.notna()).sum())
        rows.append(
            {
                "columna": col,
                "n_validos": int(info["n_valido"]),
                "Q1": info["Q1"],
                "Q3": info["Q3"],
                "IQR": info["IQR"],
                "low": info["low"],
                "high": info["high"],
                "usable": bool(info["usable"]),
                "n_outliers": n_out,
                "%_outliers": (100.0 * n_out / n_total) if n_total else np.nan,
            }
        )
    out = pd.DataFrame(rows).sort_values("%_outliers", ascending=False)
    return out


def style_iqr_summary(summary_df: pd.DataFrame):
    """
    Devuelve un Styler con formato para la tabla resumen IQR.
    """
    try:
        # Aplicamos formato columna por columna
        sty = summary_df.style
        numeric_cols = ["Q1", "Q3", "IQR", "low", "high"]
        percent_cols = ["%_outliers"]

        for col in numeric_cols:
            if col in summary_df.columns:
                sty = sty.format({col: "{:.3f}"})

        for col in percent_cols:
            if col in summary_df.columns:
                sty = sty.format({col: "{:.1f}"})

        return sty
    except Exception:
        # Si hay problemas con el styling, devolver DataFrame sin formato
        return summary_df


def annotate_and_list_outliers(
    df: pd.DataFrame,
    columns: Iterable[str],
    *,
    metodo: str = "IQR",
    factor: float = 1.5,
    k: float = 3.5,
    min_n: int = 5,
    keep_na: bool = True,
) -> Dict[str, object]:
    """
    Devuelve:
      - annotated_flags: DataFrame con columnas 'is_outlier_<col>'
      - outliers_by_col: dict[col -> DataFrame con filas que son outlier en esa col]
    """
    df_annot = annotate_outliers(
        df,
        columnas=columns,
        metodo=metodo,
        keep_na=keep_na,
        factor=factor,
        k=k,
        min_n=min_n,
    )
    flag_cols = [c for c in df_annot.columns if c.startswith("is_outlier_")]
    out_dict: Dict[str, pd.DataFrame] = {}
    for col in columns:
        flag_col = f"is_outlier_{col}"
        if flag_col in df_annot.columns:
            mask = df_annot[flag_col] & pd.to_numeric(df[col], errors="coerce").notna()
            out_dict[col] = df_annot.loc[mask, [col]].copy().sort_values(by=col)
    return {"annotated_flags": df_annot, "outliers_by_col": out_dict}


def plot_outliers_hist(
    df: pd.DataFrame,
    column: str,
    *,
    factor: float = 1.5,
    min_n: int = 5,
    bins: int = 20,
) -> go.Figure:
    """
    Histograma simple de una columna con líneas verticales en los límites IQR (Plotly).
    """
    if column not in df.columns:
        raise KeyError(f"La columna '{column}' no existe en el DataFrame.")
    s = pd.to_numeric(df[column], errors="coerce").dropna()
    info = compute_iqr_bounds(s, factor=factor, min_n=min_n)

    fig = go.Figure()
    fig.add_histogram(x=s, nbinsx=int(bins), name=str(column), opacity=0.85)
    if info["usable"]:
        for xline, label in [(info["low"], "LOW"), (info["high"], "HIGH")]:
            fig.add_vline(
                x=float(xline),
                line_width=2,
                line_dash="dash",
                line_color="red",
                annotation_text=label,
                annotation_position="top",
            )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=10, t=30, b=40),
        xaxis_title=column,
        yaxis_title="frecuencia",
        showlegend=False,
    )
    return fig


def widget_outliers_plotly(
    df: pd.DataFrame,
    *,
    segment_col: str = SEGMENT_COL,
    iqr_factor: float = 1.5,
    min_n: int = 5,
) -> w.Accordion:
    """
    Explorador interactivo (Plotly + ipywidgets) de outliers por columna:
      - Dropdown de columna numérica
      - Slider del factor IQR y bins
      - Opcional: filtro por segmento (si existe segment_col)
      - Tabla con Q1/Q3/IQR/LOW/HIGH y listado de outliers
    Devuelve un Accordion con la vista.
    """
    # Columnas numéricas con suficiente N
    cols = [
        c
        for c in df.columns
        if pd.to_numeric(df[c], errors="coerce").notna().sum() >= min_n
    ]
    if not cols:
        return w.Accordion(
            children=[w.HTML("<b>No hay columnas numéricas suficientes.</b>")]
        )

    # Detección de nombre para la tabla de outliers
    name_col = _detectar_columna_nombre(df)

    # Widgets
    dd_col = w.Dropdown(
        options=sorted(cols), description="Parámetro:", layout=w.Layout(width="45%")
    )
    sl_factor = w.FloatSlider(
        value=float(iqr_factor),
        min=0.5,
        max=3.0,
        step=0.1,
        description="factor IQR:",
        readout_format=".1f",
    )
    sl_bins = w.IntSlider(value=25, min=10, max=80, step=1, description="bins:")
    ch_out = w.Checkbox(value=True, description="Listar outliers")

    seg_widget = None
    seg_label_to_raw: dict[str, object] = {}
    seg_raw_to_label: dict[str, str] = {}
    if segment_col and segment_col in df.columns:
        seg_vals = pd.Series(df[segment_col]).dropna().astype(str).unique().tolist()
        seg_vals = [v for v in seg_vals if str(v).strip() != ""]
        if len(seg_vals) > 1:
            # Build label maps using SEGMENT_LABELS when available, else use raw string as label
            for raw in seg_vals:
                label = SEGMENT_LABELS_STR.get(str(raw), str(raw))
                seg_label_to_raw[label] = raw
                seg_raw_to_label[str(raw)] = label
            seg_options = ["Todos"] + sorted(list(seg_label_to_raw.keys()))
            seg_widget = w.Dropdown(options=seg_options, description="Segmento:")

    # Outputs
    out_plot = w.Output()
    out_tbl = w.Output()

    # Layout superior
    controls_left = [dd_col, sl_factor, sl_bins]
    controls_right = [seg_widget] if seg_widget is not None else []
    controls_right.append(ch_out)
    top = w.HBox(
        [
            w.HBox(controls_left, layout=w.Layout(flex="3")),
            w.HBox(
                controls_right, layout=w.Layout(flex="2", justify_content="flex-end")
            ),
        ]
    )

    def _render():
        # Filtrar por segmento si corresponde
        df_sel = df
        if seg_widget is not None and seg_widget.value and seg_widget.value != "Todos":
            try:
                # Map displayed label back to raw segment value (string comparison)
                chosen_label = str(seg_widget.value)
                raw_str = seg_label_to_raw.get(chosen_label, chosen_label)
                df_sel = df[df[segment_col].astype(str) == str(raw_str)]
            except Exception:
                df_sel = df

        col = dd_col.value
        s = pd.to_numeric(df_sel[col], errors="coerce")
        info = compute_iqr_bounds(s, factor=float(sl_factor.value), min_n=min_n)

        with out_plot:
            clear_output(wait=True)
            fig = go.Figure()
            fig.add_histogram(
                x=s.dropna(), nbinsx=int(sl_bins.value), name=str(col), opacity=0.85
            )
            if info["usable"]:
                for xline, label in [(info["low"], "LOW"), (info["high"], "HIGH")]:
                    if pd.notna(xline):
                        fig.add_vline(
                            x=float(xline),
                            line_width=2,
                            line_dash="dash",
                            line_color="red",
                            annotation_text=label,
                            annotation_position="top",
                        )
            title_suffix = ""
            if (
                seg_widget is not None
                and seg_widget.value
                and seg_widget.value != "Todos"
            ):
                # Always show the human-readable label in title
                chosen_label = str(seg_widget.value)
                title_suffix = f" — seg: {chosen_label}"
            fig.update_layout(
                template="plotly_white",
                margin=dict(l=40, r=10, t=35, b=40),
                xaxis_title=str(col),
                yaxis_title="frecuencia",
                showlegend=False,
                title=f"Histograma: {col}{title_suffix}",
            )
            display(fig)

        with out_tbl:
            clear_output(wait=True)
            rows = [
                {
                    "columna": col,
                    "n_valido": info["n_valido"],
                    "Q1": info["Q1"],
                    "Q3": info["Q3"],
                    "IQR": info["IQR"],
                    "LOW": info["low"],
                    "HIGH": info["high"],
                    "usable": info["usable"],
                }
            ]
            df_info = pd.DataFrame(rows)
            display(df_info)

            if ch_out.value and info["usable"]:
                mask_low = s < info["low"]
                mask_high = s > info["high"]
                cols_out = [col]
                if name_col and name_col in df_sel.columns:
                    cols_out.append(name_col)
                outs = df_sel.loc[(mask_low | mask_high) & s.notna(), cols_out].copy()
                if not outs.empty:
                    if name_col and name_col in outs.columns:
                        outs.rename(columns={name_col: "aeronave"}, inplace=True)
                    outs["tipo_outlier"] = np.where(
                        outs[col] < info["low"], "LOW", "HIGH"
                    )
                    display(w.HTML(f"<b>Outliers ({len(outs)} filas):</b>"))
                    display(outs.sort_values(col))

    # Render inicial y eventos
    _render()
    dd_col.observe(lambda _: _render(), names="value")
    sl_factor.observe(lambda _: _render(), names="value")
    sl_bins.observe(lambda _: _render(), names="value")
    if seg_widget is not None:
        seg_widget.observe(lambda _: _render(), names="value")

    acc = w.Accordion(
        children=[w.VBox([top, w.HTML("<hr>"), out_plot, w.HTML("<hr>"), out_tbl])]
    )
    acc.set_title(0, "Outliers (histograma interactivo)")
    acc.selected_index = None
    return acc


def outliers_quicklook(
    df: pd.DataFrame,
    *,
    segment_col: str = SEGMENT_COL,
    iqr_factor: float = 1.5,
    min_n: int = 5,
) -> w.Accordion:
    """Wrapper a la UI plotly para mantener compatibilidad con el notebook."""
    return widget_outliers_plotly(
        df, segment_col=segment_col, iqr_factor=iqr_factor, min_n=min_n
    )


# =========================
# NOMBRE DE AERONAVE + TABLA GLOBAL
# =========================


def _detectar_columna_nombre(
    df: pd.DataFrame, candidatos: list[str] | None = None
) -> str | None:
    """
    Heurística simple para encontrar la columna que tiene el "nombre" del modelo/aeronave.
    Devuelve el nombre de columna si encuentra una, o None si no.
    """
    if candidatos is None:
        candidatos = [
            "Modelo",
            "Modelo/Designación",
            "Nombre",
            "Aeronave",
            "Aircraft",
            "Model",
            "Designation",
            "Name",
        ]
    lower_map = {str(c).lower(): c for c in df.columns}
    for cand in candidatos:
        key = cand.lower()
        for col_lower, col_real in lower_map.items():
            # coincidencia laxa por contiene palabra clave
            if key in col_lower:
                return col_real
    return None


def outliers_tabla_global_con_nombre(
    df: pd.DataFrame, report: dict, *, name_col: str | None = None
) -> pd.DataFrame:
    """
    Igual que outliers_tabla_global(), pero reemplaza el número de fila por el nombre de la aeronave.

    Parámetros
    ----------
    df : DataFrame original (para tomar la columna de nombre)
    report : dict devuelto por outliers_quicklook()
    name_col : str | None
        Si None, intenta detectar automáticamente una columna de nombre (Modelo/Nombre/Aeronave/etc.).

    Returns
    -------
    DataFrame con columnas: ['columna', 'aeronave', 'valor'].
    """
    # Determinar columna de nombre
    if name_col is None:
        name_col = _detectar_columna_nombre(df)
    usar_indice = False
    if not name_col or name_col not in df.columns:
        usar_indice = True  # fallback: usar índice numérico

    flat = []
    out_by_col = report.get("outliers_by_col", {})
    for col, dfcol in out_by_col.items():
        if dfcol is None or dfcol.empty:
            continue
        tmp = dfcol.reset_index()[["index", col]]
        tmp = tmp.rename(columns={"index": "fila", col: "valor"})
        tmp["columna"] = col
        # Mapear fila -> nombre si es posible
        if not usar_indice:
            # ojo: 'fila' es índice original del DF
            tmp["aeronave"] = tmp["fila"].map(lambda i: df.loc[i, name_col])
        else:
            tmp["aeronave"] = tmp["fila"]
        tmp = tmp[["columna", "aeronave", "valor"]]
        flat.append(tmp)

    if not flat:
        return pd.DataFrame(columns=["columna", "aeronave", "valor"])

    return pd.concat(flat, ignore_index=True)


def outliers_tabla_global(report: dict) -> pd.DataFrame:
    """
    Aplana report['outliers_by_col'] a una sola tabla con columnas:
    ['columna', 'fila', 'valor'].
    Si no hay outliers, devuelve DF vacío con esas columnas.

    NOTA: Esta función se mantiene por compatibilidad. Para ver nombres de aeronave
    usar outliers_tabla_global_con_nombre() en su lugar.
    """
    flat = []
    out_by_col = report.get("outliers_by_col", {})
    for col, dfcol in out_by_col.items():
        if dfcol is None or dfcol.empty:
            continue
        tmp = dfcol.reset_index()[["index", col]]
        tmp = tmp.rename(columns={"index": "fila", col: "valor"})
        tmp["columna"] = col
        tmp = tmp[["columna", "fila", "valor"]]
        flat.append(tmp)
    if not flat:
        return pd.DataFrame(columns=["columna", "fila", "valor"])
    return pd.concat(flat, ignore_index=True)


def vista_outliers_en_notebook(
    df: pd.DataFrame,
    *,
    metodo: str = "IQR",
    columns: list[str] | None = None,
    auto_detect_numeric: bool = True,
    factor: float = 1.5,
    k: float = 3.5,
    min_n: int = 5,
    keep_na: bool = True,
) -> dict:
    """
    Empaqueta todo para verlo directo en el notebook sin armar nada:
      - 'summary_styler': tabla resumen estilada por columna
      - 'annotated_flags': DF original con columnas is_outlier_*
      - 'tabla_outliers': tabla aplanada (columna, fila, valor)
    """
    # No usar outliers_quicklook (ahora retorna UI). Reproducimos cálculos aquí.
    # Selección de columnas
    if columns is None and auto_detect_numeric:
        cols = []
        for c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() >= min_n:
                cols.append(c)
        columns = cols
    elif columns is None:
        columns = list(df.columns)
    columns = [c for c in columns if c in df.columns]

    summary = iqr_summary_table(
        df, columns, factor=factor, min_n=min_n, keep_na=keep_na
    )
    summary_styler = style_iqr_summary(summary)
    annot = annotate_and_list_outliers(
        df, columns, metodo=metodo, factor=factor, k=k, min_n=min_n, keep_na=keep_na
    )
    tabla = outliers_tabla_global(
        {
            "outliers_by_col": annot["outliers_by_col"],
        }
    )
    return {
        "summary_styler": summary_styler,
        "annotated_flags": annot["annotated_flags"],
        "tabla_outliers": tabla,
    }


# =========================
# WIDGET INTERACTIVO (JUPYTER) PARA HISTOGRAMAS CON IQR
# =========================


def widget_outliers_hist(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    *,
    factor: float = 1.5,
    min_n: int = 5,
    bins: int = 20,
    titulo: str = "Outliers • Histograma con límites IQR",
):
    """
    Devuelve un widget interactivo (ipywidgets) con un dropdown para elegir la columna
    y dibuja el histograma con límites IQR. Ideal para no llenar el notebook de gráficos.

    Uso en notebook:
    >>> ui = widget_outliers_hist(df, columns=None)   # autodetecta numéricas útiles
    >>> ui                                           # mostrar el widget

    Requisitos: ipywidgets instalado y habilitado en Jupyter (conda/pip).
    """
    try:
        import ipywidgets as w
        from IPython.display import display, clear_output
    except Exception as e:
        raise RuntimeError(
            "Este widget requiere 'ipywidgets' instalado y habilitado en Jupyter."
            " Instalación típica: 'conda install ipywidgets' o 'pip install ipywidgets'."
        ) from e

    # Autodetectar columnas numéricas con >= min_n valores válidos
    if columns is None:
        cols = []
        for c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() >= min_n:
                cols.append(c)
        columns = cols

    if not columns:
        raise ValueError("No hay columnas numéricas suficientes para graficar.")

    dd = w.Dropdown(
        options=columns, description="Parámetro:", layout=w.Layout(width="50%")
    )
    sl_factor = w.FloatSlider(
        value=factor,
        min=0.5,
        max=3.0,
        step=0.1,
        description="factor IQR:",
        readout_format=".1f",
    )
    sl_bins = w.IntSlider(value=bins, min=10, max=60, step=1, description="bins:")
    out = w.Output()

    title = w.HTML(f"<h4 style='margin:0'>{titulo}</h4>")

    def _plot(col: str, factor_val: float, bins_val: int):
        s = pd.to_numeric(df[col], errors="coerce")
        info = compute_iqr_bounds(s, factor=factor_val, min_n=min_n)
        fig = go.Figure()
        fig.add_histogram(
            x=s.dropna(), nbinsx=int(bins_val), name=str(col), opacity=0.85
        )
        if info["usable"]:
            for xline, label in [(info["low"], "LOW"), (info["high"], "HIGH")]:
                if pd.notna(xline):
                    fig.add_vline(
                        x=float(xline),
                        line_width=2,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=label,
                        annotation_position="top",
                    )
        n_low = int((s < info["low"]).sum()) if info["usable"] else 0
        n_high = int((s > info["high"]).sum()) if info["usable"] else 0
        title = (
            f"{col}  |  IQR usable={info['usable']}  |  outliers: low={n_low}, high={n_high}"
            if info["usable"]
            else f"{col}  |  IQR no usable (n<{min_n} o IQR≈0)"
        )
        fig.update_layout(
            template="plotly_white",
            margin=dict(l=40, r=10, t=40, b=40),
            xaxis_title=str(col),
            yaxis_title="frecuencia",
            showlegend=False,
            title=title,
        )
        return fig

    def _on_change(*args):
        with out:
            clear_output(wait=True)
            fig = _plot(dd.value, sl_factor.value, sl_bins.value)
            display(fig)

    # primera render
    _on_change()

    dd.observe(_on_change, names="value")
    sl_factor.observe(_on_change, names="value")
    sl_bins.observe(_on_change, names="value")

    controls = w.HBox([dd, sl_factor, sl_bins])
    box = w.VBox([title, controls, out])
    return box


def widget_outliers_panel(
    df: pd.DataFrame,
    *,
    factor: float = 1.5,
    min_n: int = 5,
    titulo: str = "Outliers (IQR)",
    collapsed: bool = True,
):
    """
    Panel colapsable con:
      - Resumen global (Styler) en Output
      - Selector + histograma por columna
    """
    try:
        import ipywidgets as w
        from IPython.display import display, clear_output
    except Exception as e:
        raise RuntimeError("Este widget requiere 'ipywidgets' instalado.") from e

    # 1) Resumen (Styler) → Output (sin depender de outliers_quicklook)
    out_resumen = w.Output()
    with out_resumen:
        cols_sum = []
        for c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() >= min_n:
                cols_sum.append(c)
        summary = iqr_summary_table(
            df, cols_sum, factor=factor, min_n=min_n, keep_na=True
        )
        display(style_iqr_summary(summary))

    box_resumen = w.VBox([w.HTML(f"<b>{titulo} — resumen</b>"), out_resumen])

    # 2) Explorador Plotly
    box_detalle = widget_outliers_plotly(
        df, segment_col=SEGMENT_COL, iqr_factor=factor, min_n=min_n
    )

    acc = w.Accordion(children=[box_resumen, box_detalle])
    acc.set_title(0, "Resumen")
    acc.set_title(1, "Histograma por columna")
    acc.selected_index = None if collapsed else 0
    return acc
