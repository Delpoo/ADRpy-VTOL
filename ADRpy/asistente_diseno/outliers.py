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
from typing import Dict, Iterable, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from .datos import to_numeric_locale

# from .mplutils import import_matplotlib  # <- eliminar este import
from .config import SEGMENT_COL, SEGMENT_LABELS
import plotly.graph_objects as go

# Importar ipywidgets/IPython de forma segura para que el módulo cargue aunque no haya UI
try:
    import ipywidgets as w  # type: ignore
    from IPython.display import display, clear_output  # type: ignore
except Exception:  # pragma: no cover

    class _WidgetStub:
        def __getattr__(self, _name: str) -> Any:
            raise RuntimeError(
                "Este módulo requiere 'ipywidgets'/'IPython' para la UI."
            )

    w = _WidgetStub()  # type: ignore

    def display(*_args, **_kwargs):  # type: ignore[override]
        raise RuntimeError("Este módulo requiere 'ipywidgets'/'IPython' para la UI.")

    def clear_output(*_args, **_kwargs):  # type: ignore[override]
        pass


from .mplutils import style_df_2dec, apply_tickformat_2dec, f2

# Cache ligera para el resumen IQR
_SUMMARY_CACHE: dict[tuple, pd.DataFrame] = {}


def _summary_key(df: pd.DataFrame, factor: float, min_n: int) -> tuple:
    return (
        id(df),
        round(float(factor), 4),
        int(min_n),
        tuple(df.columns),
        len(df),
    )


# =============================================================================
# Utilidades internas
# =============================================================================


def _to_numeric_series(s: pd.Series) -> pd.Series:
    """Convierte a numérico con errors='coerce' y devuelve una copia."""
    if not isinstance(s, pd.Series):
        s = pd.Series(s)
    return to_numeric_locale(s.copy())


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
    df_view: pd.DataFrame,
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
        if col not in df_view.columns:
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
        s = to_numeric_locale(df[col])
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
    if not rows:
        return pd.DataFrame(
            columns=[
                "columna",
                "n_validos",
                "Q1",
                "Q3",
                "IQR",
                "low",
                "high",
                "usable",
                "n_outliers",
                "%_outliers",
            ]
        )
    out = pd.DataFrame(rows)
    if "%_outliers" in out.columns:
        out = out.sort_values("%_outliers", ascending=False)
    return out


def style_iqr_summary(summary_df: pd.DataFrame):
    """
    Devuelve un Styler con formato para la tabla resumen IQR.
    Aplica .2f a TODAS las columnas numéricas (incluye %_outliers) para armonizar.
    """
    try:
        # Formato base a 2 decimales para columnas numéricas
        sty = numeric_2dec_styler(summary_df)
        # Asegurar específicamente .2f para el porcentaje si existe
        if "%_outliers" in summary_df.columns:
            sty = sty.format(lambda v: fmt2(v), subset=["%_outliers"])
        return sty
    except Exception:
        # Si hay problemas con el styling, devolver DataFrame sin formato
        return summary_df


def annotate_and_list_outliers(
    df_view: pd.DataFrame,
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
        df_view,
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
            mask = df_annot[flag_col] & to_numeric_locale(df[col]).notna()
            out_dict[col] = df_annot.loc[mask, [col]].copy().sort_values(by=col)
    return {"annotated_flags": df_annot, "outliers_by_col": out_dict}


def plot_outliers_hist(
    df_view: pd.DataFrame,
    column: str,
    *,
    factor: float = 1.5,
    min_n: int = 5,
    bins: int = 20,
) -> go.Figure:
    """
    Histograma simple de una columna con líneas verticales en los límites IQR (Plotly).
    """
    if column not in df_view.columns:
        raise KeyError(f"La columna '{column}' no existe en el DataFrame.")
    s = to_numeric_locale(df[column]).dropna()
    info = compute_iqr_bounds(s, factor=factor, min_n=min_n)

    fig = go.Figure()
    fig.add_histogram(
        x=s,
        nbinsx=int(bins),
        name=str(column),
        opacity=0.85,
        hovertemplate="valor=%{x:.2f}<br>freq=%{y:.2f}<extra></extra>",
    )
    if info["usable"]:
        for xline, label in [(info["low"], "LOW"), (info["high"], "HIGH")]:
            fig.add_vline(
                x=float(xline),
                line_width=2,
                line_dash="dash",
                line_color="red",
                annotation_text=f"{label} {fmt2(xline)}",
                annotation_position="top",
            )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=10, t=30, b=40),
        xaxis_title=column,
        yaxis_title="frecuencia",
        showlegend=False,
    )
    plotly_apply_2dec(fig)
    return fig


def widget_outliers_plotly(
    df_view: pd.DataFrame,
    *,
    segment_col: str = SEGMENT_COL,
    iqr_factor: float = 1.5,
    min_n: int = 5,
) -> Any:
    if w is None:
        raise RuntimeError("ipywidgets no está disponible en este entorno.")
    """
    Explorador interactivo (Plotly + ipywidgets) de outliers por columna:
      - Dropdown de columna numérica
      - Slider del factor IQR y bins
      - Opcional: filtro por segmento (si existe segment_col)
      - Tabla con Q1/Q3/IQR/LOW/HIGH y listado de outliers
    Devuelve un Accordion con la vista.
    """
    # Columnas numéricas con suficiente N
    cols = [c for c in df.columns if to_numeric_locale(df[c]).notna().sum() >= min_n]
    if not cols:
        return w.Accordion(
            children=[w.HTML("<b>No hay columnas numéricas suficientes.</b>")]
        )

    # Detección de nombre para la tabla de outliers
    name_col = _detectar_columna_nombre(df_view)

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
        readout_format=".2f",
    )
    sl_bins = w.IntSlider(value=25, min=10, max=80, step=1, description="bins:")
    ch_out = w.Checkbox(value=True, description="Listar outliers")

    seg_widget = None
    seg_label_to_raw: dict[str, object] = {}
    seg_raw_to_label: dict[str, str] = {}
    if segment_col and segment_col in df_view.columns:
        seg_vals = (
            pd.Series(df_view[segment_col]).dropna().astype(str).unique().tolist()
        )
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
    out_plot = w.HTML()
    out_tbl = w.HTML()

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
        df_sel = df_view
        if seg_widget is not None and seg_widget.value and seg_widget.value != "Todos":
            try:
                # Map displayed label back to raw segment value (string comparison)
                chosen_label = str(seg_widget.value)
                raw_str = seg_label_to_raw.get(chosen_label, chosen_label)
                df_sel = df_view[df_view[segment_col].astype(str) == str(raw_str)]
            except Exception:
                df_sel = df_view

        col = dd_col.value
        s = to_numeric_locale(df_sel[col])
        info = compute_iqr_bounds(s, factor=float(sl_factor.value), min_n=min_n)

        # Histograma -> HTML embebido
        fig = go.Figure()
        fig.add_histogram(
            x=s.dropna(),
            nbinsx=int(sl_bins.value),
            name=str(col),
            opacity=0.85,
            hovertemplate="valor=%{x:.2f}<br>freq=%{y:.2f}<extra></extra>",
        )
        if info["usable"]:
            for xline, label in [(info["low"], "LOW"), (info["high"], "HIGH")]:
                if pd.notna(xline):
                    fig.add_vline(
                        x=float(xline),
                        line_width=2,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"{label} {fmt2(xline)}",
                        annotation_position="top",
                    )
        title_suffix = ""
        if seg_widget is not None and seg_widget.value and seg_widget.value != "Todos":
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
        # Formateo homogéneo de ejes a .2f
        try:
            plotly_apply_2dec(fig)
        except Exception:
            pass
        out_plot.value = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")  # type: ignore[arg-type]

        # Tabla resumen + listado de outliers (en HTML)
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
        try:
            out_html = numeric_2dec_styler(format_df_2dec(df_info)).to_html()
        except Exception:
            try:
                out_html = format_df_2dec(df_info).to_html()
            except Exception:
                out_html = df_info.to_html()

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
                outs["tipo_outlier"] = np.where(outs[col] < info["low"], "LOW", "HIGH")
                out_html += f"<br><b>Outliers ({len(outs)} filas):</b>"
                try:
                    out_html += numeric_2dec_styler(
                        format_df_2dec(outs.sort_values(col))
                    ).to_html()
                except Exception:
                    try:
                        out_html += format_df_2dec(outs.sort_values(col)).to_html()
                    except Exception:
                        out_html += outs.sort_values(col).to_html()
        out_tbl.value = out_html

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
    df_view: pd.DataFrame,
    *,
    segment_col: str = SEGMENT_COL,
    iqr_factor: float = 1.5,
    min_n: int = 5,
) -> Any:
    """Wrapper a la UI plotly para mantener compatibilidad con el notebook."""
    return widget_outliers_plotly(
        df_view, segment_col=segment_col, iqr_factor=iqr_factor, min_n=min_n
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
    df_view: pd.DataFrame, report: dict, *, name_col: str | None = None
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
        name_col = _detectar_columna_nombre(df_view)
    usar_indice = False
    if not name_col or name_col not in df_view.columns:
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
            tmp["aeronave"] = tmp["fila"].map(lambda i: df_view.loc[i, name_col])
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
    df_view: pd.DataFrame,
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
            s = to_numeric_locale(df[c])
            if s.notna().sum() >= min_n:
                cols.append(c)
        columns = cols
    elif columns is None:
        columns = list(df_view.columns)
    columns = [c for c in columns if c in df_view.columns]

    summary = iqr_summary_table(
        df_view, columns, factor=factor, min_n=min_n, keep_na=keep_na
    )
    summary_styler = style_iqr_summary(summary)
    annot = annotate_and_list_outliers(
        df_view,
        columns,
        metodo=metodo,
        factor=factor,
        k=k,
        min_n=min_n,
        keep_na=keep_na,
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
    y dibuja el histograma con límites IQR.

    Ruta HTML-only: sin usar display()/clear_output(); renderizamos con pio.to_html.
    """

    # Autodetectar columnas numéricas con >= min_n valores válidos
    if columns is None:
        cols = []
        for c in df.columns:
            s = to_numeric_locale(df[c])
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
        readout_format=".2f",
    )
    sl_bins = w.IntSlider(value=bins, min=10, max=60, step=1, description="bins:")
    out_fig = w.HTML()

    title = w.HTML(f"<h4 style='margin:0'>{titulo}</h4>")

    def _plot(col: str, factor_val: float, bins_val: int):
        s = to_numeric_locale(df[col])
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
            f"{col}  |  IQR usable={info['usable']}  |  outliers: low={n_low:.2f}, high={n_high:.2f}"
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
        # Aplicar formato de ejes a .2f en todos los histogramas
        try:
            plotly_apply_2dec(fig)
        except Exception:
            pass
        return fig

    def _on_change(*args):
        fig = _plot(dd.value, sl_factor.value, sl_bins.value)
        try:
            out_fig.value = _to_html(fig, full_html=False, include_plotlyjs="cdn")
        except Exception:
            out_fig.value = "<i>No se pudo renderizar el histograma.</i>"

    # primera render
    _on_change()

    dd.observe(_on_change, names="value")
    sl_factor.observe(_on_change, names="value")
    sl_bins.observe(_on_change, names="value")

    controls = w.HBox([dd, sl_factor, sl_bins])
    box = w.VBox([title, controls, out_fig])
    return box


def widget_outliers_panel(
    df: pd.DataFrame,
    *,
    df_filtrado: Optional[pd.DataFrame] = None,
    factor: float = 1.5,
    min_n: int = 5,
    titulo: str = "Outliers (IQR)",
    collapsed: bool = True,
):
    """
    Panel colapsable con un ÚNICO módulo expandible que contiene secciones tituladas:
      - "Resumen (IQR k=…)" con la tabla resumen y selector de columna
      - "Histogramas (IQR k=…)" con controles y gráfico/tabla por columna
    """
    try:
        import ipywidgets as w
        from IPython.display import display, clear_output
    except Exception as e:
        raise RuntimeError("Este widget requiere 'ipywidgets' instalado.") from e

    df_view = (
        df_filtrado
        if isinstance(df_filtrado, pd.DataFrame) and not df_filtrado.empty
        else df
    )

    # Columnas elegibles (>= min_n válidos)
    cols_sum = []
    for c in df_view.columns:
        s = to_numeric_locale(df_view[c])
        if s.notna().sum() >= min_n:
            cols_sum.append(c)
    key = _summary_key(df_view, factor, min_n)
    summary_cached = _SUMMARY_CACHE.get(key)
    if summary_cached is None:
        summary = iqr_summary_table(
            df_view, cols_sum, factor=factor, min_n=min_n, keep_na=True
        )
        _SUMMARY_CACHE[key] = summary.copy()
    else:
        summary = summary_cached.copy()

    # Si no hay columnas numéricas suficientes, devolver mensaje claro
    if summary.empty:
        body = w.VBox(
            [
                w.HTML(
                    f"<i>Sin columnas numéricas suficientes (min_n={min_n}). Ajusta filtros o elige otro dataset.</i>"
                )
            ]
        )
        acc = w.Accordion(children=[body])
        acc.set_title(0, f"{titulo}")
        acc.selected_index = None if collapsed else 0
        return acc

    # Selector de columna compartido por ambas secciones
    dd_col = w.Dropdown(
        options=sorted(summary["columna"].tolist()) if not summary.empty else [],
        description="Parámetro:",
        layout=w.Layout(width="45%"),
    )

    # Títulos de sección (actualizables con el factor)
    title_main = w.HTML(f"<b>{titulo}</b>")
    title_res = w.HTML(f"<b>Resumen (IQR k={factor:.2f})</b>")
    title_hist = w.HTML(f"<b>Histogramas (IQR k={factor:.2f})</b>")

    # Salida del resumen
    out_resumen = w.HTML()
    try:
        sty_or_df = style_iqr_summary(summary)
        if hasattr(sty_or_df, "to_html"):
            out_resumen.value = sty_or_df.to_html()  # type: ignore[union-attr]
        else:
            out_resumen.value = format_df_2dec(summary).to_html()
    except Exception:
        try:
            out_resumen.value = format_df_2dec(summary).to_html()
        except Exception:
            out_resumen.value = summary.to_html()

    # Controles del histograma (reutilizamos lógica de widget_outliers_plotly)
    sl_factor = w.FloatSlider(
        value=float(factor),
        min=0.5,
        max=3.0,
        step=0.1,
        description="factor IQR:",
        readout_format=".2f",
    )
    sl_bins = w.IntSlider(value=25, min=10, max=80, step=1, description="bins:")
    ch_out = w.Checkbox(value=True, description="Listar outliers")

    seg_widget = None
    seg_label_to_raw: dict[str, object] = {}
    seg_raw_to_label: dict[str, str] = {}
    if SEGMENT_COL and SEGMENT_COL in df_view.columns:
        seg_vals = (
            pd.Series(df_view[SEGMENT_COL]).dropna().astype(str).unique().tolist()
        )
        seg_vals = [v for v in seg_vals if str(v).strip() != ""]
        if len(seg_vals) > 1:
            for raw in seg_vals:
                label = SEGMENT_LABELS_STR.get(str(raw), str(raw))
                seg_label_to_raw[label] = raw
                seg_raw_to_label[str(raw)] = label
            seg_options = ["Todos"] + sorted(list(seg_label_to_raw.keys()))
            seg_widget = w.Dropdown(options=seg_options, description="Segmento:")

    out_plot = w.HTML()
    out_tbl = w.HTML()

    controls_left = [dd_col, sl_factor, sl_bins]
    controls_right = [seg_widget] if seg_widget is not None else []
    controls_right.append(ch_out)
    top_controls = w.HBox(
        [
            w.HBox(controls_left, layout=w.Layout(flex="3")),
            w.HBox(
                controls_right, layout=w.Layout(flex="2", justify_content="flex-end")
            ),
        ]
    )

    def _render_hist():
        # Filtrar por segmento si corresponde
        df_sel = df_view
        if seg_widget is not None and seg_widget.value and seg_widget.value != "Todos":
            try:
                chosen_label = str(seg_widget.value)
                raw_str = seg_label_to_raw.get(chosen_label, chosen_label)
                df_sel = df_view[df_view[SEGMENT_COL].astype(str) == str(raw_str)]
            except Exception:
                df_sel = df_view

        col = (
            dd_col.value
            if dd_col.value
            else (sorted(summary["columna"].tolist())[0] if not summary.empty else None)
        )
        if col is None:
            out_plot.value = "<i>Sin columnas numéricas suficientes.</i>"
            out_tbl.value = ""
            return

        s = to_numeric_locale(df_sel[col])
        info = compute_iqr_bounds(s, factor=float(sl_factor.value), min_n=min_n)

        fig = go.Figure()
        fig.add_histogram(
            x=s.dropna(),
            nbinsx=int(sl_bins.value),
            name=str(col),
            opacity=0.85,
            hovertemplate="valor=%{x:.2f}<br>freq=%{y:.2f}<extra></extra>",
        )
        if info["usable"]:
            for xline, label in [(info["low"], "LOW"), (info["high"], "HIGH")]:
                if pd.notna(xline):
                    fig.add_vline(
                        x=float(xline),
                        line_width=2,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"{label} {fmt2(xline)}",
                        annotation_position="top",
                    )
        title_suffix = ""
        if seg_widget is not None and seg_widget.value and seg_widget.value != "Todos":
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
        try:
            plotly_apply_2dec(fig)
        except Exception:
            pass
        try:
            out_plot.value = _to_html(fig, full_html=False, include_plotlyjs="cdn")
        except Exception:
            out_plot.value = "<i>No se pudo renderizar el histograma.</i>"

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
        try:
            out_html = numeric_2dec_styler(format_df_2dec(df_info)).to_html()
        except Exception:
            try:
                out_html = format_df_2dec(df_info).to_html()
            except Exception:
                out_html = df_info.to_html()

        if ch_out.value and info["usable"]:
            mask_low = s < info["low"]
            mask_high = s > info["high"]
            cols_out = [col]
            name_col = _detectar_columna_nombre(df_sel)
            if name_col and name_col in df_sel.columns:
                cols_out.append(name_col)
            outs = df_sel.loc[(mask_low | mask_high) & s.notna(), cols_out].copy()
            if not outs.empty:
                if name_col and name_col in outs.columns:
                    outs.rename(columns={name_col: "aeronave"}, inplace=True)
                outs["tipo_outlier"] = np.where(outs[col] < info["low"], "LOW", "HIGH")
                try:
                    outs_html = numeric_2dec_styler(
                        format_df_2dec(outs.sort_values(col))
                    ).to_html()
                except Exception:
                    try:
                        outs_html = format_df_2dec(outs.sort_values(col)).to_html()
                    except Exception:
                        outs_html = outs.sort_values(col).to_html()
                out_html += f"<br><b>Outliers ({len(outs)} filas):</b><br>" + outs_html
        out_tbl.value = out_html

    # Eventos: cualquier cambio re-renderiza histograma y actualiza títulos
    def _on_any_change(*_):
        try:
            title_res.value = f"<b>Resumen (IQR k={float(sl_factor.value):.2f})</b>"
            title_hist.value = (
                f"<b>Histogramas (IQR k={float(sl_factor.value):.2f})</b>"
            )
        except Exception:
            pass
        _render_hist()

    dd_col.observe(lambda *_: _on_any_change(), names="value")
    sl_factor.observe(lambda *_: _on_any_change(), names="value")
    sl_bins.observe(lambda *_: _on_any_change(), names="value")
    ch_out.observe(lambda *_: _on_any_change(), names="value")
    if seg_widget is not None:
        seg_widget.observe(lambda *_: _on_any_change(), names="value")

    # Render inicial
    _on_any_change()

    # Composición en un único módulo con secciones
    body = w.VBox(
        [
            title_main,
            title_res,
            dd_col,
            out_resumen,
            w.HTML("<hr>"),
            title_hist,
            top_controls,
            w.HTML("<hr>"),
            out_plot,
            w.HTML("<hr>"),
            out_tbl,
        ]
    )

    out_acc = w.Accordion(children=[body])
    out_acc.set_title(0, f"Outliers (IQR k={float(sl_factor.value):.2f})")
    out_acc.selected_index = None if collapsed else 0

    if df_filtrado is not None and isinstance(df_filtrado, pd.DataFrame):
        try:
            n_orig = len(df)
        except Exception:
            n_orig = 0
        try:
            if df_filtrado.empty:
                left_label = w.HTML(
                    f"<b>DataFrame original</b> <small>(n={int(n_orig)})</small>"
                )
                right_label = w.HTML("<b>DataFrame filtrado</b> <small>(n=0)</small>")
                try:
                    left_label.tooltip = "Todo el dataset, sin restricciones."
                    right_label.tooltip = (
                        "No hay filas que cumplan las reglas de filtrado actuales."
                    )
                except Exception:
                    pass
                msg = w.HTML("<i>Sin datos filtrados con las reglas actuales.</i>")
                return w.HBox(
                    [w.VBox([left_label, out_acc]), w.VBox([right_label, msg])]
                )

            right_panel = widget_outliers_panel(
                df_filtrado,
                factor=factor,
                min_n=min_n,
                titulo="Outliers (IQR) – DataFrame filtrado",
                collapsed=collapsed,
            )
            left_label = w.HTML(
                f"<b>DataFrame original</b> <small>(n={int(n_orig)})</small>"
            )
            right_label = w.HTML(
                f"<b>DataFrame filtrado</b> <small>(n={len(df_filtrado)})</small>"
            )
            try:
                left_label.tooltip = "Todo el dataset, sin restricciones."
                right_label.tooltip = "Subconjunto que cumple la selección (mín./máx./rango/fijo/objetivo)."
            except Exception:
                pass
            return w.HBox(
                [w.VBox([left_label, out_acc]), w.VBox([right_label, right_panel])]
            )
        except Exception:
            pass
    return out_acc


def widget_outliers_panel_dual(
    df: pd.DataFrame,
    *,
    df_filtrado: Optional[pd.DataFrame] = None,
    factor: float = 1.5,
    min_n: int = 5,
    titulo: str = "Outliers (IQR) — Global vs Filtrado",
    collapsed: bool = True,
) -> w.Accordion:
    """Panel doble que permite refrescar el subconjunto filtrado in-place."""

    try:
        import ipywidgets as w  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Este widget requiere 'ipywidgets' instalado.") from exc

    state: dict[str, Optional[pd.DataFrame]] = {"df_filtrado": None}

    n_orig = len(df) if isinstance(df, pd.DataFrame) else 0
    left_label = w.HTML(f"<b>DataFrame original</b> <small>(n={int(n_orig)})</small>")
    try:
        left_label.tooltip = "Todo el dataset, sin restricciones."
    except Exception:
        pass

    left_panel = widget_outliers_panel(
        df,
        factor=factor,
        min_n=min_n,
        titulo="Outliers (IQR) – DataFrame original",
        collapsed=False,
    )

    left_container = w.VBox(
        [left_label, left_panel], layout=w.Layout(width="50%", min_width="0")
    )

    right_label = w.HTML("")
    right_container = w.VBox(layout=w.Layout(width="50%", min_width="0"))

    def _render_filtrado(new_df: Optional[pd.DataFrame]) -> None:
        sanitized = (
            new_df if isinstance(new_df, pd.DataFrame) and not new_df.empty else None
        )
        state["df_filtrado"] = sanitized
        if sanitized is None:
            right_label.value = "<b>DataFrame filtrado</b> <small>(n=0)</small>"
            try:
                right_label.tooltip = (
                    "No hay filas que cumplan las reglas de filtrado actuales."
                )
            except Exception:
                pass
            msg = w.HTML(
                "<i>Sin datos filtrados disponibles para el conjunto actual de restricciones.</i>"
            )
            right_container.children = [right_label, msg]
            return

        right_label.value = (
            f"<b>DataFrame filtrado</b> <small>(n={len(sanitized)})</small>"
        )
        try:
            right_label.tooltip = (
                "Subconjunto que cumple la selección (mín./máx./rango/fijo/objetivo)."
            )
        except Exception:
            pass

        panel_filtrado = widget_outliers_panel(
            sanitized,
            factor=factor,
            min_n=min_n,
            titulo="Outliers (IQR) – DataFrame filtrado",
            collapsed=False,
        )
        right_container.children = [right_label, panel_filtrado]

    _render_filtrado(df_filtrado)

    box = w.HBox(
        [left_container, right_container],
        layout=w.Layout(width="100%", gap="12px", justify_content="space-between"),
    )

    acc = w.Accordion(children=[box])
    acc.set_title(0, titulo)
    acc.selected_index = None if collapsed else 0

    def _set_df_filtrado(new_df: Optional[pd.DataFrame]) -> None:
        _render_filtrado(new_df)

    setattr(acc, "set_df_filtrado", _set_df_filtrado)
    return acc
