"""
Asistente ADRpy – UI integrada en Jupyter (panel izquierdo + ranking + sugerencias + outliers).

Uso desde notebook:
        %run "C:/Users/delpi/OneDrive/Tesis/ADRpy-VTOL/ADRpy/asistente_diseno/main.py"

Notas
-----
- No elimina ni reemplaza tus celdas individuales de análisis (B: outliers, C: similitud, D: sugerencias).
    Este 'main' sólo orquesta los módulos existentes en una vista unificada con paneles colapsables.
- Requiere ipywidgets.
"""

from __future__ import annotations
import os, sys, shutil, importlib, warnings
from IPython.display import display, clear_output
import ipywidgets as w  # widgets used throughout

# ---------------------------------------------------------------------
# Asegurar que el parent (ADRpy) esté en sys.path para imports absolutos
# ---------------------------------------------------------------------
HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# Limpieza de caches para recarga "en caliente"
def _purge_pycache(base):
    for dirpath, dirnames, filenames in os.walk(base):
        if "__pycache__" in dirnames:
            shutil.rmtree(os.path.join(dirpath, "__pycache__"), ignore_errors=True)


_purge_pycache(os.path.join(PROJECT_ROOT, "asistente_diseno"))
importlib.invalidate_caches()

# ---------------------------------------------------------------------
# Imports de nuestros módulos (no se tocan tus scripts existentes)
# ---------------------------------------------------------------------
from asistente_diseno.datos import leer_excel
from asistente_diseno.similitud import (
    rank,
    insertar_objetivo_en_ranking,
    widget_filtrado_ranking,
)
from asistente_diseno.sugerencias import sugerencias_topk, widget_sugerencias_panel
from asistente_diseno.outliers import widget_outliers_panel
from asistente_diseno.config import SEGMENT_LABELS, SEGMENT_COL

# ---------------------------------------------------------------------
# UI (ipywidgets)
# ---------------------------------------------------------------------
import pandas as pd

# Parámetros a exponer en el panel izquierdo (podés ampliar)
PARAMS_DEFAULT = [
    "Peso máximo al despegue (MTOW)",
    "Payload",
    "Velocidad a la que se realiza el crucero (m/s TAS)",
    "Autonomía de la aeronave (h)",
]


def _bloque_param(nombre_param: str, val: float = 0.0) -> dict:
    tipos = ["ignorar", "fijo", "objetivo", "maximo", "minimo", "rango"]
    dd_tipo = w.Dropdown(
        options=tipos,
        value="ignorar",
        description=nombre_param,
        layout=w.Layout(width="420px"),
    )
    ft_val = w.FloatText(value=val, description="valor", layout=w.Layout(width="200px"))
    ft_tol = w.FloatText(value=0.5, description="±tol", layout=w.Layout(width="160px"))
    ft_min = w.FloatText(value=0.0, description="min", layout=w.Layout(width="160px"))
    ft_max = w.FloatText(value=0.0, description="max", layout=w.Layout(width="160px"))
    sl_peso = w.FloatSlider(
        value=1.0,
        min=0.0,
        max=2.0,
        step=0.1,
        description="peso",
        readout_format=".1f",
        layout=w.Layout(width="300px"),
    )
    box = w.HBox([dd_tipo, ft_val, ft_tol, ft_min, ft_max, sl_peso])

    # Mostrar/ocultar campos según tipo
    def _toggle(*args):
        t = dd_tipo.value
        ft_val.layout.display = "none"
        ft_tol.layout.display = "none"
        ft_min.layout.display = "none"
        ft_max.layout.display = "none"
        if t in ("fijo", "objetivo", "maximo", "minimo"):
            ft_val.layout.display = ""
        if t == "objetivo":
            ft_tol.layout.display = ""
        if t == "rango":
            ft_min.layout.display = ""
            ft_max.layout.display = ""

    dd_tipo.observe(_toggle, names="value")
    _toggle()

    return dict(
        nombre=nombre_param,
        cont=box,
        dd_tipo=dd_tipo,
        ft_val=ft_val,
        ft_tol=ft_tol,
        ft_min=ft_min,
        ft_max=ft_max,
        sl_peso=sl_peso,
    )


def _armar_restricciones(bloques: list[dict]) -> dict:
    restr = {}
    for b in bloques:
        t = b["dd_tipo"].value
        if t == "ignorar":
            continue
        d = {"tipo": t, "peso": float(b["sl_peso"].value)}
        if t in ("fijo", "objetivo", "maximo", "minimo"):
            d["valor"] = float(b["ft_val"].value)
        if t == "objetivo":
            d["tol"] = float(b["ft_tol"].value)
        if t == "rango":
            d["min"] = float(b["ft_min"].value)
            d["max"] = float(b["ft_max"].value)
        restr[b["nombre"]] = d
    return restr


def build_ui(df: pd.DataFrame, params: list[str] | None = None) -> w.VBox:
    """
    Construye la UI completa (panel izquierdo + ranking + paneles colapsables).
    Devuelve un contenedor para display().
    """
    params = params or PARAMS_DEFAULT

    # --- Bloques del panel izquierdo
    bloques = [
        _bloque_param(params[0], 25.0 if len(params) > 0 else 0.0),
        _bloque_param(params[1], 5.0 if len(params) > 1 else 0.0),
        _bloque_param(params[2], 22.0 if len(params) > 2 else 0.0),
        _bloque_param(params[3], 4.0 if len(params) > 3 else 0.0),
    ]
    box_params = w.VBox([b["cont"] for b in bloques])

    # --- Controles globales
    sl_alpha = w.FloatSlider(
        value=1.0,
        min=0.2,
        max=3.0,
        step=0.05,
        description="α similitud",
        layout=w.Layout(width="300px"),
    )
    ch_nan = w.Checkbox(value=True, description="Penalizar NaN")
    ft_pen_nan = w.FloatText(
        value=1.0, description="penalidad NaN", layout=w.Layout(width="220px")
    )
    sl_topk = w.IntSlider(
        value=10,
        min=3,
        max=30,
        step=1,
        description="Top-K",
        layout=w.Layout(width="250px"),
    )
    ch_out = w.Checkbox(value=True, description="Quitar atípicos (IQR)")
    ft_iqrf = w.FloatText(
        value=1.5, description="factor IQR", layout=w.Layout(width="180px")
    )

    # Segmentación: columna + modo + valor + factor prefer
    candidatas_seg = ["(ninguno)"]
    if SEGMENT_COL in df.columns:
        candidatas_seg.append(SEGMENT_COL)
    candidatas_seg += [
        c
        for c in df.columns
        if c not in candidatas_seg
        and c.lower().startswith(("misi", "tipo", "class", "categoria", "famil"))
    ]
    dd_segcol = w.Dropdown(
        options=candidatas_seg,
        value=candidatas_seg[0],
        description="Segmentar por",
        layout=w.Layout(width="300px"),
    )
    dd_segm_modo = w.Dropdown(
        options=[
            ("Global (sin segmentar)", "off"),
            ("Filtrar por valor", "filter"),
            ("Preferir valor (penalizar otros)", "prefer"),
        ],
        value="off",
        description="Modo",
        layout=w.Layout(width="330px"),
    )
    dd_segm_val = w.Dropdown(
        options=["(seleccioná columna)"],
        value="(seleccioná columna)",
        description="Valor",
        layout=w.Layout(width="300px"),
    )
    sl_pref_fac = w.FloatSlider(
        value=1.3,
        min=1.0,
        max=2.0,
        step=0.05,
        description="factor prefer",
        layout=w.Layout(width="300px"),
    )

    # Botones de acción + modo Auto
    btn_run = w.Button(
        description="▶ Recalcular", tooltip="Recalcula ranking y sugerencias"
    )
    btn_clear = w.Button(description="🧹 Limpiar", tooltip="Restablece parámetros")
    ch_auto = w.Checkbox(
        value=False, description="Auto", tooltip="Recalcular al cambiar"
    )
    # Contraste de botones (evita que “desaparezcan” en temas oscuros)
    btn_run.style.button_color = "#28a745"  # verde
    btn_clear.style.button_color = "#f0ad4e"  # naranja

    panel_global_1 = w.HBox([sl_alpha, ch_nan, ft_pen_nan, sl_topk, ch_out, ft_iqrf])
    panel_global_2 = w.HBox(
        [dd_segcol, dd_segm_modo, dd_segm_val, sl_pref_fac, btn_run, btn_clear, ch_auto]
    )

    # --- Salidas
    out_rank = w.Output()
    out_sug = w.Output()
    out_outl = w.Output()

    # --- Helpers internos
    def _populate_segment_values(*args):
        col = dd_segcol.value
        if col == "(ninguno)" or col not in df.columns:
            dd_segm_val.options = ["(ninguno)"]
            dd_segm_val.value = "(ninguno)"
            return
        if col == SEGMENT_COL and SEGMENT_LABELS:
            opciones = ["(ninguno)"] + list(SEGMENT_LABELS.values())
            dd_segm_val.options = opciones
            dd_segm_val.value = opciones[1] if len(opciones) > 1 else "(ninguno)"
        else:
            vals = df[col].dropna().astype(str).unique().tolist()
            dd_segm_val.options = ["(ninguno)"] + sorted(vals)
            dd_segm_val.value = sorted(vals)[0] if vals else "(ninguno)"

    def _render(*_):
        """Recalcula Ranking + Sugerencias + Outliers panel (colapsables)."""
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=RuntimeWarning, message="Mean of empty slice"
            )
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message="invalid value encountered in scalar divide",
            )

            restricciones = _armar_restricciones(bloques)

            with out_rank:
                clear_output(wait=True)
                if not restricciones:
                    display(
                        w.HTML(
                            "<b>Definí al menos un parámetro (no 'ignorar') para calcular el ranking.</b>"
                        )
                    )
                    return

                seg_col = None if dd_segcol.value == "(ninguno)" else dd_segcol.value
                seg_modo = dd_segm_modo.value
                seg_val = None
                if seg_col and dd_segm_val.value and dd_segm_val.value != "(ninguno)":
                    val_ui = dd_segm_val.value
                    # rank() compara contra etiquetas legibles al filtrar (mapea raw->label antes),
                    # por lo tanto debemos pasar el NOMBRE (label) seleccionado, no el código.
                    seg_val = str(val_ui)

                df_rank = rank(
                    df,
                    restricciones,
                    metodo_escala="IQR",
                    min_n=5,
                    penalizar_nan=bool(ch_nan.value),
                    penalidad_nan=float(ft_pen_nan.value),
                    alpha=float(sl_alpha.value),
                    segmentar_por=seg_col,
                    segment_labels=SEGMENT_LABELS if seg_col == SEGMENT_COL else None,
                    segmentar_modo=seg_modo,
                    segmentar_valor=seg_val,
                    prefer_factor=float(sl_pref_fac.value),
                    top_n=None,
                )
                # Asegurar que la tabla muestre etiquetas legibles aunque no segmentemos por SEGMENT_COL
                if (SEGMENT_COL in df.columns) and SEGMENT_LABELS:
                    seg_map = {str(k): v for k, v in SEGMENT_LABELS.items()}
                    seg_raw = df.loc[df_rank.index, SEGMENT_COL].astype(str)
                    df_rank["segmento"] = seg_raw.map(seg_map).fillna(seg_raw)
                df_rank_obj = insertar_objetivo_en_ranking(
                    df_rank,
                    restricciones,
                    name="Objetivo (usuario)",
                    segment_label="(selección)",
                )

                ui_rank = widget_filtrado_ranking(
                    df_rank_obj,
                    restricciones,
                    sort_default="similitud",
                    top_n_default=15,
                )
                display(w.HTML("<h4>Ranking por similitud</h4>"))
                display(ui_rank)

            with out_sug:
                clear_output(wait=True)
                params_sug = list(restricciones.keys())
                sug = sugerencias_topk(
                    df_ranked=df_rank_obj,
                    params=params_sug,
                    top_k=int(sl_topk.value),
                    remove_outliers=bool(ch_out.value),
                    iqr_factor=float(ft_iqrf.value),
                    use_distance_weights=True,
                    use_confidence_weights=False,
                    confidence_cols=None,
                    beta_dist=1.0,
                    beta_conf=1.0,
                    name_objetivo="Objetivo (usuario)",
                )
                acc_sug = widget_sugerencias_panel(
                    sug, titulo="Sugerencias (Top-K)", collapsed=True, bins=20
                )
                display(acc_sug)

            with out_outl:
                clear_output(wait=True)
                acc_out = widget_outliers_panel(
                    df,
                    factor=float(ft_iqrf.value),
                    min_n=5,
                    titulo="Outliers en el dataset (IQR)",
                    collapsed=True,
                )
                display(acc_out)

    def _clear(_):
        for b in bloques:
            b["dd_tipo"].value = "ignorar"
        sl_alpha.value = 1.0
        ch_nan.value = True
        ft_pen_nan.value = 1.0
        sl_topk.value = 10
        ch_out.value = True
        ft_iqrf.value = 1.5
        dd_segcol.value = "(ninguno)"
        dd_segm_modo.value = "off"
        dd_segm_val.options = ["(seleccioná columna)"]
        dd_segm_val.value = "(seleccioná columna)"
        _render()

    def _maybe_auto(change):
        if ch_auto.value:
            _render()

    # Wiring de eventos
    btn_run.on_click(lambda _: _render())
    btn_clear.on_click(_clear)
    dd_segcol.observe(_populate_segment_values, names="value")

    for b in bloques:
        for wdg in (
            b["dd_tipo"],
            b["ft_val"],
            b["ft_tol"],
            b["ft_min"],
            b["ft_max"],
            b["sl_peso"],
        ):
            wdg.observe(_maybe_auto, names="value")
    for wdg in (
        sl_alpha,
        ch_nan,
        ft_pen_nan,
        sl_topk,
        ch_out,
        ft_iqrf,
        dd_segm_modo,
        dd_segm_val,
        sl_pref_fac,
    ):
        wdg.observe(_maybe_auto, names="value")

    # Render inicial
    _populate_segment_values()
    header = w.HTML("<h4>Panel izquierdo (prototipo en notebook)</h4>")
    container = w.VBox(
        [
            header,
            box_params,
            panel_global_1,
            panel_global_2,
            w.HTML("<hr>"),
            out_rank,
            w.HTML("<hr>"),
            out_sug,
            w.HTML("<hr>"),
            out_outl,
        ]
    )
    _render()
    return container


def run_demo():
    """Carga datos y muestra la UI integrada."""
    df = leer_excel()
    ui = build_ui(df, params=PARAMS_DEFAULT)
    display(ui)


if __name__ == "__main__":
    run_demo()
