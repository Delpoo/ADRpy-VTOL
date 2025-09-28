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
import os, sys, shutil, importlib, warnings, threading
from IPython.display import display, clear_output
from typing import Optional, Any
from dataclasses import dataclass
import ipywidgets as w  # widgets used throughout
import numpy as np

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
    ParamSpec,
)
from asistente_diseno.sugerencias import sugerencias_topk, widget_sugerencias_panel
from asistente_diseno.outliers import widget_outliers_panel
from asistente_diseno.config import SEGMENT_LABELS, SEGMENT_COL
from asistente_diseno.tendencias import widget_tendencias_plotly
from asistente_diseno.guias import widget_info_param
from asistente_diseno.narrativa import narrativa_informe, export_markdown, export_html
from asistente_diseno.guias_tooltips import apply_tooltip, HELP
from asistente_diseno.datos import columnas_numericas_utiles
from asistente_diseno.config import (
    DISPLAY_LABELS,
    PARAM_DEFAULTS,
    PREFERRED_ORDER,
    SEGMENT_COL,
)
import asistente_diseno.config as _cfg

# --- Forzar reload de módulos clave para evitar versiones "stale" al re-ejecutar desde notebook
try:
    import asistente_diseno.similitud as _sim
    import asistente_diseno.sugerencias as _sug
    import asistente_diseno.mplutils as _mpl

    _sim = importlib.reload(_sim)
    _sug = importlib.reload(_sug)
    _mpl = importlib.reload(_mpl)
    # Reasignar símbolos usados a las versiones recién recargadas
    rank = _sim.rank
    insertar_objetivo_en_ranking = _sim.insertar_objetivo_en_ranking
    widget_filtrado_ranking = _sim.widget_filtrado_ranking
    sugerencias_topk = _sug.sugerencias_topk
    widget_sugerencias_panel = _sug.widget_sugerencias_panel
except Exception:
    # Si algo falla, seguimos con los ya importados arriba
    pass

PARAM_GROUPS = getattr(_cfg, "PARAM_GROUPS", {})

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


@dataclass
class ParamRow:
    col: str
    label: str
    ch_active: w.Checkbox
    dd_mode: w.Dropdown
    ft_value: w.FloatText
    ft_min: w.FloatText
    ft_max: w.FloatText
    sl_weight: w.FloatSlider
    btn_info: w.Button
    box: w.HBox
    # Commit C: vista compacta + toggler por fila
    btn_toggle: Optional[w.Button] = None
    expanded: bool = False


class ParamPanel(w.VBox):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df
        self.rows: list[ParamRow] = []
        self.stats: dict[str, dict] = {}
        self._filtered_rows: list[ParamRow] = []
        self.view_mode: str = "Detallada"
        # Paginación (Commit B)
        self.page_size = 15
        self.page = 1
        # Contenedor de filas: sin scroll interno, que use el scroll principal del notebook
        self.rows_container = w.VBox(layout=w.Layout(overflow="visible"))
        self._build()

    # Recalcular estadísticas según un subconjunto del dataset actual
    def set_stats_from_df(self, df_subset: pd.DataFrame):
        def _param_stats(df0: pd.DataFrame, col: str) -> dict:
            s = (
                pd.to_numeric(df0[col], errors="coerce")
                if col in df0.columns
                else pd.Series(dtype=float)
            )
            n = int(s.notna().sum())
            miss = 1.0 - (n / max(int(len(s)), 1)) if len(s) else 1.0
            arr = s.to_numpy()
            try:
                var = float(np.nanvar(arr, ddof=1)) if n > 1 else 0.0
            except Exception:
                var = 0.0
            try:
                vmin = float(np.nanmin(arr)) if n else float("nan")
            except Exception:
                vmin = float("nan")
            try:
                vmax = float(np.nanmax(arr)) if n else float("nan")
            except Exception:
                vmax = float("nan")
            return {"n": n, "miss": miss, "var": var, "min": vmin, "max": vmax}

        for r in self.rows:
            try:
                self.stats[r.col] = _param_stats(df_subset, r.col)
            except Exception:
                self.stats[r.col] = {
                    "n": 0,
                    "miss": 1.0,
                    "var": 0.0,
                    "min": float("nan"),
                    "max": float("nan"),
                }
        # Prefill de min/max para filas en modo 'rango' que aún no tengan valores cargados
        self._update_range_defaults()

    def _prefill_range_from_stats(self, pr: "ParamRow"):
        st = self.stats.get(pr.col, {})
        vmin = st.get("min")
        vmax = st.get("max")
        try:
            if pr.dd_mode.value == "rango":
                if (
                    pr.ft_min.value in (None, "")
                    and vmin is not None
                    and np.isfinite(vmin)
                ):
                    pr.ft_min.value = float(vmin)
                if (
                    pr.ft_max.value in (None, "")
                    and vmax is not None
                    and np.isfinite(vmax)
                ):
                    pr.ft_max.value = float(vmax)
        except Exception:
            pass

    def _update_range_defaults(self):
        for rr in self.rows:
            self._prefill_range_from_stats(rr)

    def _build(self):
        # Controles de filtro/orden/visibilidad
        self.txt_buscar = w.Text(
            placeholder="Buscar parámetro…", layout=w.Layout(width="40%")
        )
        self.ch_solo_activos = w.Checkbox(value=False, description="Solo activos")
        self.btn_todos_on = w.Button(
            description="Activar visibles", button_style="success"
        )
        self.btn_todos_off = w.Button(
            description="Desactivar todos", button_style="warning"
        )

        self.dd_sort = w.Dropdown(
            options=[
                "Alfabético",
                "Activos primero",
                "Menos faltantes",
                "Más variabilidad",
            ],
            value="Alfabético",
            description="Orden:",
            layout=w.Layout(width="35%"),
        )
        grupos = sorted(set(PARAM_GROUPS.values())) if PARAM_GROUPS else []
        opts_group = ["Todos"] + grupos if grupos else ["Todos"]
        self.dd_group = w.Dropdown(
            options=opts_group,
            value=opts_group[0],
            description="Grupo:",
            layout=w.Layout(width="35%"),
        )
        # Vista como Dropdown (alineado con el resto)
        self.dd_view = w.Dropdown(
            options=["Detallada", "Compacta"],
            value="Detallada",
            description="Vista:",
            layout=w.Layout(width="35%"),
        )

        # Controles de paginación (Commit B)
        self.dd_page_size = w.Dropdown(
            options=[5, 10, 15, 20, 30, 50],
            value=15,
            description="Por página:",
            layout=w.Layout(width="220px"),
        )
        self.btn_prev = w.Button(description="◀", tooltip="Página anterior")
        self.btn_next = w.Button(description="▶", tooltip="Página siguiente")
        self.lbl_page = w.Label("Página 1/1 (0 items)")

        header1 = w.HBox(
            [
                self.txt_buscar,
                self.ch_solo_activos,
                self.btn_todos_on,
                self.btn_todos_off,
            ]
        )
        header2 = w.HBox(
            [
                self.dd_sort,
                self.dd_group,
                self.dd_view,
                self.dd_page_size,
                self.btn_prev,
                self.btn_next,
                self.lbl_page,
            ]
        )

        # Commit C: edición masiva (modo/peso) sobre la página visible
        self.dd_bulk_mode = w.Dropdown(
            options=[
                ("(sin cambio)", "__nochange__"),
                ("Ignorar", "ignorar"),
                ("Fijo (= valor)", "fijo"),
                ("Máximo (<= valor)", "maximo"),
                ("Mínimo (>= valor)", "minimo"),
                ("Rango [min..max]", "rango"),
            ],
            value="__nochange__",
            description="Modo masivo:",
            layout=w.Layout(width="280px"),
        )
        self.ft_bulk_weight = w.FloatText(
            value=float(PARAM_DEFAULTS.get("weight", 1.0)),
            description="Peso masivo:",
            layout=w.Layout(width="220px"),
        )
        # Valores de rango para edición masiva (opcionales)
        self.ft_bulk_min = w.FloatText(
            value=None, placeholder="min (masivo)", layout=w.Layout(width="160px")
        )
        self.ft_bulk_max = w.FloatText(
            value=None, placeholder="max (masivo)", layout=w.Layout(width="160px")
        )
        self.btn_aplicar_bulk = w.Button(
            description="Aplicar a visibles", button_style="info"
        )
        header3 = w.HBox(
            [
                self.dd_bulk_mode,
                self.ft_bulk_weight,
                self.ft_bulk_min,
                self.ft_bulk_max,
                self.btn_aplicar_bulk,
            ]
        )

        # Construir filas para columnas numéricas útiles
        cols = columnas_numericas_utiles(self.df)

        def _key(c: str):
            return (
                c not in PREFERRED_ORDER,
                (PREFERRED_ORDER.index(c) if c in PREFERRED_ORDER else 10**6),
                c.lower(),
            )

        try:
            cols.sort(key=_key)
        except Exception:
            cols.sort(key=lambda c: c.lower())

        # Stats por columna
        def _param_stats(df: pd.DataFrame, col: str) -> dict:
            s = pd.to_numeric(df[col], errors="coerce")
            n = int(s.notna().sum())
            miss = 1.0 - (n / max(int(len(s)), 1))
            arr = s.to_numpy()
            try:
                var = float(np.nanvar(arr, ddof=1)) if n > 1 else 0.0
            except Exception:
                var = 0.0
            try:
                vmin = float(np.nanmin(arr)) if n else float("nan")
            except Exception:
                vmin = float("nan")
            try:
                vmax = float(np.nanmax(arr)) if n else float("nan")
            except Exception:
                vmax = float("nan")
            return {"n": n, "miss": miss, "var": var, "min": vmin, "max": vmax}

        for c in cols:
            try:
                self.stats[c] = _param_stats(self.df, c)
            except Exception:
                self.stats[c] = {
                    "n": 0,
                    "miss": 1.0,
                    "var": 0.0,
                    "min": float("nan"),
                    "max": float("nan"),
                }

        items: list[w.HBox] = []
        for c in cols:
            label = str(DISPLAY_LABELS.get(c, c))
            ch = w.Checkbox(
                value=False,
                description=label,
                indent=False,
                layout=w.Layout(width="36%"),
            )
            dd = w.Dropdown(
                options=[
                    ("Ignorar", "ignorar"),
                    ("Fijo (= valor)", "fijo"),
                    ("Máximo (<= valor)", "maximo"),
                    ("Mínimo (>= valor)", "minimo"),
                    ("Rango [min..max]", "rango"),
                ],
                value="ignorar",
                layout=w.Layout(width="16%"),
            )
            ft = w.FloatText(
                value=None, placeholder="valor", layout=w.Layout(width="14%")
            )
            ft.layout.display = "none"  # oculto por defecto
            ft_min = w.FloatText(
                value=None, placeholder="min", layout=w.Layout(width="12%")
            )
            ft_min.layout.display = "none"
            ft_max = w.FloatText(
                value=None, placeholder="max", layout=w.Layout(width="12%")
            )
            ft_max.layout.display = "none"
            sl = w.FloatSlider(
                value=float(PARAM_DEFAULTS.get("weight", 1.0)),
                min=0.0,
                max=3.0,
                step=0.1,
                readout=True,
                layout=w.Layout(width="14%"),
            )
            btn = w.Button(
                description="i",
                tooltip=f"Info de {label}",
                layout=w.Layout(width="40px"),
            )
            # Commit C: per-row toggler (detalles) para vista compacta
            btn_toggle = w.Button(
                description="⋯",
                tooltip="Mostrar/ocultar detalle",
                layout=w.Layout(width="36px"),
            )
            row = w.HBox([ch, dd, ft, ft_min, ft_max, sl, btn, btn_toggle])
            row.layout = w.Layout(align_items="center")
            pr = ParamRow(
                col=c,
                label=label,
                ch_active=ch,
                dd_mode=dd,
                ft_value=ft,
                ft_min=ft_min,
                ft_max=ft_max,
                sl_weight=sl,
                btn_info=btn,
                box=row,
                btn_toggle=btn_toggle,
            )
            self.rows.append(pr)
            items.append(row)

            # tooltips y toggles
            apply_tooltip(dd, "modo_param")
            apply_tooltip(ft, "valor_param")
            apply_tooltip(ft_min, "valor_param")
            apply_tooltip(ft_max, "valor_param")
            apply_tooltip(sl, "peso_param")

            def _toggle_value(change, pr=pr):
                # Mostrar ft_value para fijo/min/max; ft_min/ft_max para rango.
                detailed = (self.view_mode == "Detallada") or pr.expanded
                mode = change["new"]
                wants_value = mode in {"minimo", "maximo", "fijo"}
                wants_range = mode == "rango"
                pr.ft_value.layout.display = (
                    "" if (wants_value and detailed) else "none"
                )
                pr.ft_min.layout.display = "" if (wants_range and detailed) else "none"
                pr.ft_max.layout.display = "" if (wants_range and detailed) else "none"
                # Si cambió a 'rango', pre-cargar min/max desde stats si están vacíos
                if wants_range:
                    self._prefill_range_from_stats(pr)

            dd.observe(_toggle_value, names="value")
            _toggle_value({"new": dd.value})

            # Toggle per-row for compact view
            def _on_toggle_row(_btn, pr=pr):
                pr.expanded = not pr.expanded
                _update_row_view(pr)

            btn_toggle.on_click(_on_toggle_row)

        # Helpers de vista (detallada/compacta)
        def _update_row_view(pr: ParamRow):
            is_compact = self.view_mode == "Compacta"
            # dd_mode y peso visibles si no compacta o si expandido
            show_detail = (not is_compact) or pr.expanded
            pr.dd_mode.layout.display = "" if show_detail else "none"
            pr.sl_weight.layout.display = "" if show_detail else "none"
            # ft_value depende del modo y de show_detail; reutilizamos regla de _toggle_value
            mode = pr.dd_mode.value
            wants_value = mode in {"minimo", "maximo", "fijo"}
            wants_range = mode == "rango"
            pr.ft_value.layout.display = "" if (wants_value and show_detail) else "none"
            pr.ft_min.layout.display = "" if (wants_range and show_detail) else "none"
            pr.ft_max.layout.display = "" if (wants_range and show_detail) else "none"
            # el botón de toggle solo tiene sentido en compacta
            if pr.btn_toggle is not None:
                pr.btn_toggle.layout.display = "" if is_compact else "none"

        def _update_all_rows_view():
            for rr in self.rows:
                _update_row_view(rr)

        # Manejo de filtro/orden y acciones masivas + paginación
        def _total_pages() -> int:
            n = len(self._filtered_rows)
            ps = max(int(self.page_size), 1)
            return max((n + ps - 1) // ps, 1)

        def _visible_rows() -> list[ParamRow]:
            ps = max(int(self.page_size), 1)
            p = max(min(int(self.page), _total_pages()), 1)
            start = (p - 1) * ps
            end = start + ps
            return self._filtered_rows[start:end]

        def _update_page_label():
            self.lbl_page.value = f"Página {self.page}/{_total_pages()} ({len(self._filtered_rows)} items)"

        def _render_page():
            vis = _visible_rows()
            self.rows_container.children = (
                [r.box for r in vis] if vis else [w.HTML("<i>Sin coincidencias</i>")]
            )
            _update_page_label()

        def _reset_pagination():
            self.page = 1
            _update_page_label()

        def _apply_filter(*_):
            q = self.txt_buscar.value.strip().lower()
            solo = bool(self.ch_solo_activos.value)
            grupo = str(self.dd_group.value) if hasattr(self, "dd_group") else "Todos"
            # filtrar
            filtered: list[ParamRow] = []
            for r in self.rows:
                ok_q = (q in r.label.lower() or q in r.col.lower()) if q else True
                ok_a = r.ch_active.value or not solo
                ok_g = True
                if grupo and grupo != "Todos" and PARAM_GROUPS:
                    ok_g = PARAM_GROUPS.get(r.col) == grupo
                if ok_q and ok_a and ok_g:
                    filtered.append(r)
            # ordenar
            sort_mode = str(self.dd_sort.value)
            if sort_mode == "Alfabético":
                filtered.sort(key=lambda r: r.label.lower())
            elif sort_mode == "Activos primero":
                filtered.sort(
                    key=lambda r: (not bool(r.ch_active.value), r.label.lower())
                )
            elif sort_mode == "Menos faltantes":
                filtered.sort(
                    key=lambda r: float(self.stats.get(r.col, {}).get("miss", 1.0))
                )
            elif sort_mode == "Más variabilidad":
                filtered.sort(
                    key=lambda r: -float(self.stats.get(r.col, {}).get("var", 0.0))
                )
            # guardar y montar (paginado)
            self._filtered_rows = filtered
            _reset_pagination()
            _render_page()

        self.txt_buscar.observe(lambda _: _apply_filter(), names="value")
        self.ch_solo_activos.observe(lambda _: _apply_filter(), names="value")
        self.dd_sort.observe(lambda _: _apply_filter(), names="value")
        self.dd_group.observe(lambda _: _apply_filter(), names="value")

        def _on_view_change(c):
            self.view_mode = str(c.get("new", "Detallada"))
            _update_all_rows_view()

        self.dd_view.observe(_on_view_change, names="value")

        # Paginación events
        def _on_page_size(_):
            try:
                self.page_size = int(self.dd_page_size.value)
            except Exception:
                self.page_size = 10
            _reset_pagination()
            _render_page()

        def _go_prev(_):
            if self.page > 1:
                self.page -= 1
                _render_page()

        def _go_next(_):
            if self.page < _total_pages():
                self.page += 1
                _render_page()

        self.dd_page_size.observe(_on_page_size, names="value")
        self.btn_prev.on_click(_go_prev)
        self.btn_next.on_click(_go_next)

        # Acciones masivas sobre la página visible
        self.btn_todos_on.on_click(
            lambda _: [setattr(r.ch_active, "value", True) for r in _visible_rows()]
        )
        self.btn_todos_off.on_click(
            lambda _: [setattr(r.ch_active, "value", False) for r in _visible_rows()]
        )

        # Aplicar edición masiva (modo/peso) a visibles
        def _apply_bulk(_):
            mode_sel = str(self.dd_bulk_mode.value)
            try:
                w_val = float(self.ft_bulk_weight.value)
            except Exception:
                w_val = float(PARAM_DEFAULTS.get("weight", 1.0))
            for r in _visible_rows():
                if mode_sel != "__nochange__":
                    r.dd_mode.value = mode_sel
                    # si es rango: setear min/max desde inputs masivos (si hay) o usar stats
                    if mode_sel == "rango":
                        try:
                            if self.ft_bulk_min.value not in (None, ""):
                                r.ft_min.value = float(self.ft_bulk_min.value)
                            else:
                                # prefill desde stats si vacío
                                if r.ft_min.value in (None, ""):
                                    self._prefill_range_from_stats(r)
                            if self.ft_bulk_max.value not in (None, ""):
                                r.ft_max.value = float(self.ft_bulk_max.value)
                            else:
                                if r.ft_max.value in (None, ""):
                                    self._prefill_range_from_stats(r)
                        except Exception:
                            pass
                r.sl_weight.value = w_val
                _update_row_view(r)

        self.btn_aplicar_bulk.on_click(_apply_bulk)

        self.children = [
            header1,
            header2,
            header3,
            w.HTML("<hr>"),
            self.rows_container,
        ]
        # Inicializar paginación y primer render
        self.page_size = int(self.dd_page_size.value)
        _apply_filter()
        _update_all_rows_view()

    def collect_params(self) -> list[ParamSpec]:
        out: list[ParamSpec] = []
        for r in self.rows:
            mode = str(r.dd_mode.value)
            try:
                val = (
                    None
                    if (mode == "ignorar" or r.ft_value.value in (None, ""))
                    else float(r.ft_value.value)
                )
            except Exception:
                val = None
            out.append(
                ParamSpec(
                    col=r.col,
                    label=r.label,
                    active=bool(r.ch_active.value),
                    mode=mode,  # type: ignore[arg-type]
                    value=val,
                    weight=float(r.sl_weight.value),
                )
            )
        return out

    def collect_restricciones(self) -> dict:
        restr: dict[str, dict] = {}
        for r in self.rows:
            mode = str(r.dd_mode.value)
            if (
                not bool(r.ch_active.value)
                or mode == "ignorar"
                or float(r.sl_weight.value) <= 0
            ):
                continue
            d: dict = {"tipo": mode, "peso": float(r.sl_weight.value)}
            if mode in {"minimo", "maximo", "fijo"}:
                try:
                    if r.ft_value.value not in (None, ""):
                        d["valor"] = float(r.ft_value.value)
                except Exception:
                    pass
            elif mode == "rango":
                try:
                    vmin = (
                        None if r.ft_min.value in (None, "") else float(r.ft_min.value)
                    )
                except Exception:
                    vmin = None
                try:
                    vmax = (
                        None if r.ft_max.value in (None, "") else float(r.ft_max.value)
                    )
                except Exception:
                    vmax = None
                if vmin is not None:
                    d["min"] = vmin
                if vmax is not None:
                    d["max"] = vmax
            restr[r.col] = d
        return restr


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
    # botón de información (se cablea en build_ui)
    btn_info = w.Button(description="i", tooltip=f"Info de {nombre_param}")
    btn_info.layout.width = "36px"
    # tooltips por control
    apply_tooltip(dd_tipo, "modo_param")
    apply_tooltip(ft_val, "valor_param")
    apply_tooltip(sl_peso, "peso_param")
    box = w.HBox([dd_tipo, ft_val, ft_tol, ft_min, ft_max, sl_peso, btn_info])

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
        btn_info=btn_info,
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

    # --- Panel dinámico de parámetros (reemplaza a los 4 fijos)
    param_panel = ParamPanel(df)

    # Panel derecho: detalle del parámetro + botones (informe/export/guardar) + ayuda
    out_info = w.Output(layout=w.Layout(border="1px solid #ddd", padding="6px"))
    btn_informe = w.Button(description="Generar informe", icon="file")
    # Exportaciones/tablas + persistencia de sesión
    btn_export = w.Button(description="Exportar Excel/CSV", icon="download")
    btn_save = w.Button(description="Guardar sesión", icon="save")
    dd_load = w.Dropdown(
        options=["(cargar sesión…)"], value="(cargar sesión…)", description="Sesión:"
    )
    ch_autoload = w.Checkbox(
        value=True,
        description="Autocargar última",
        tooltip="Carga automáticamente la última sesión guardada al abrir",
    )
    # Ayuda global (toggle) + panel de ayuda (Output)
    btn_ayuda = w.ToggleButton(value=False, description="? Ayuda", icon="question")
    try:
        btn_ayuda.tooltip = "Mostrar/ocultar ayuda de controles"
    except Exception:
        pass
    out_help = w.Output(
        layout=w.Layout(
            border="1px solid #ddd", padding="6px", max_height="260px", overflow="auto"
        )
    )
    # inicialmente oculto
    out_help.layout.display = "none"

    # --- Controles globales
    sl_alpha = w.FloatSlider(
        value=1.0,
        min=0.2,
        max=3.0,
        step=0.05,
        description="α similitud",
        layout=w.Layout(width="300px"),
    )
    apply_tooltip(sl_alpha, "alpha_sim")
    ch_nan = w.Checkbox(value=True, description="Penalizar NaN")
    apply_tooltip(ch_nan, "penalizar_nan")
    ft_pen_nan = w.FloatText(
        value=1.0, description="penalidad NaN", layout=w.Layout(width="220px")
    )
    apply_tooltip(ft_pen_nan, "penalidad_nan")
    sl_topk = w.IntSlider(
        value=10,
        min=3,
        max=30,
        step=1,
        description="Top-K",
        layout=w.Layout(width="250px"),
    )
    apply_tooltip(sl_topk, "top_k")
    ch_out = w.Checkbox(value=True, description="Quitar atípicos (IQR)")
    apply_tooltip(ch_out, "iqr_on")
    ft_iqrf = w.FloatText(
        value=1.5, description="factor IQR", layout=w.Layout(width="180px")
    )
    apply_tooltip(ft_iqrf, "iqr_factor")

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
    apply_tooltip(dd_segcol, "segmentar_por")
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
    apply_tooltip(dd_segm_modo, "modo_global_familia")
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
    apply_tooltip(sl_pref_fac, "factor_prefer")

    # Botones de acción + modo Auto
    btn_run = w.Button(
        description="▶ Recalcular", tooltip="Recalcula ranking y sugerencias"
    )
    btn_clear = w.Button(description="🧹 Limpiar", tooltip="Restablece parámetros")
    ch_auto = w.Checkbox(
        value=False, description="Auto", tooltip="Recalcular al cambiar"
    )
    apply_tooltip(ch_auto, "auto")
    # Contraste de botones (evita que “desaparezcan” en temas oscuros)
    btn_run.style.button_color = "#28a745"  # verde
    btn_clear.style.button_color = "#f0ad4e"  # naranja

    panel_global_1 = w.HBox([sl_alpha, ch_nan, ft_pen_nan, sl_topk, ch_out, ft_iqrf])
    panel_global_1.layout = w.Layout(
        flex_flow="row wrap", align_items="center", width="100%"
    )
    panel_global_2 = w.HBox(
        [dd_segcol, dd_segm_modo, dd_segm_val, sl_pref_fac, btn_run, btn_clear, ch_auto]
    )
    panel_global_2.layout = w.Layout(
        flex_flow="row wrap", align_items="center", width="100%"
    )
    # Hacer la barra de controles "sticky" para mejorar visibilidad
    sticky_bar = w.VBox([panel_global_1, panel_global_2])
    sticky_bar.layout = w.Layout(
        position="sticky",
        top="0",
        z_index="10",
        border="1px solid #e0e0e0",
        padding="6px 8px",
        background_color="#fafafa",
        width="100%",
    )

    # --- Salidas
    out_rank = w.Output()
    out_sug = w.Output()
    out_outl = w.Output()
    # referencias a acordeones para callbacks
    acc_out_ref: dict[str, Optional[w.Accordion]] = {"acc": None}
    acc_sug_ref: dict[str, Optional[w.Accordion]] = {"acc": None}
    # Tendencias se arma una vez (no depende del ranking); lo colocamos antes de Outliers
    try:
        x_obj_col_name = params[0] if (params and params[0] in df.columns) else None
    except Exception:
        x_obj_col_name = None

    # Callback centralizado: dado un nombre de columna, devolver el "objetivo" actual (si aplica)
    def _get_objetivo(col: str) -> float | None:
        try:
            # Buscar fila en panel dinámico
            for r in param_panel.rows:
                if r.col == col:
                    t = str(r.dd_mode.value)
                    if t in {"fijo", "maximo", "minimo"}:
                        try:
                            return float(r.ft_value.value)
                        except Exception:
                            return None
            return None
        except Exception:
            return None

    acc_tend = widget_tendencias_plotly(
        df,
        x_obj_col_name=x_obj_col_name,
        x_obj_widget=None,
        get_objetivo=_get_objetivo,
    )

    # helpers para mostrar info del parámetro
    estado_sugerencias: dict[str, float | None] = {}
    # estado del último ranking (para generar informe)
    df_rank_obj_state: dict[str, pd.DataFrame | None] = {"df": None}

    def _get_sugerido(col: str) -> float | None:
        return estado_sugerencias.get(col)

    def abrir_outliers(col: str):
        try:
            acc = acc_out_ref["acc"]
            if acc is not None:
                acc.selected_index = 0
        except Exception:
            pass

    def abrir_xy(col: str):
        try:
            acc_tend.selected_index = 0
        except Exception:
            pass

    def abrir_sugerencias(col: str):
        try:
            acc = acc_sug_ref["acc"]
            if acc is not None:
                acc.selected_index = 0
        except Exception:
            pass

    def show_info(col: str):
        with out_info:
            clear_output(wait=True)
            acc = widget_info_param(
                df,
                col,
                get_objetivo=_get_objetivo,
                get_sugerido=_get_sugerido,
                factor_iqr=1.5,
                on_open_outliers=lambda c: abrir_outliers(c),
                on_open_xy=lambda c: abrir_xy(c),
                on_open_sugerencias=lambda c: abrir_sugerencias(c),
            )
            display(acc)

    # cablear botones info en panel dinámico
    for r in param_panel.rows:
        r.btn_info.on_click(lambda _btn, c=r.col: show_info(c))

    # --- Helpers internos
    def _populate_segment_values(*args):
        col = dd_segcol.value
        if col == "(ninguno)" or col not in df.columns:
            dd_segm_val.options = ["(ninguno)"]
            dd_segm_val.value = "(ninguno)"
            # Actualizar stats con dataset global
            param_panel.set_stats_from_df(df)
            return
        if col == SEGMENT_COL and SEGMENT_LABELS:
            opciones = ["(ninguno)"] + list(SEGMENT_LABELS.values())
            dd_segm_val.options = opciones
            dd_segm_val.value = opciones[1] if len(opciones) > 1 else "(ninguno)"
        else:
            vals = df[col].dropna().astype(str).unique().tolist()
            dd_segm_val.options = ["(ninguno)"] + sorted(vals)
            dd_segm_val.value = sorted(vals)[0] if vals else "(ninguno)"
        # Cuando cambia la columna de segmentación, si aún no hay valor, usar global; si hay valor, filtrar
        _update_param_stats_for_current_segment()

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

            params_dyn = param_panel.collect_params()
            restricciones = param_panel.collect_restricciones()

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
                    restricciones=restricciones,
                    params=params_dyn,
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
                # guardar para el informe
                df_rank_obj_state["df"] = df_rank_obj.copy()

                # Calcular un resumen de sugerencias rápido para construir 'alerta' del objetivo
                try:
                    params_sug = [
                        p.col
                        for p in params_dyn
                        if p.active and p.mode != "ignorar" and p.col in df.columns
                    ]
                    sug_small = sugerencias_topk(
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
                    alerts: list[str] = []
                    summary = sug_small.get("summary")
                    if isinstance(summary, pd.DataFrame) and not summary.empty:
                        for _, row in summary.iterrows():
                            par = row.get("parametro")
                            if par is None:
                                continue
                            objv = _get_objetivo(str(par))
                            low = row.get("low", None)
                            high = row.get("high", None)
                            if (
                                objv is not None
                                and pd.notna(objv)
                                and pd.notna(low)
                                and pd.notna(high)
                            ):
                                if float(objv) < float(low):
                                    alerts.append(f"{par}: objetivo < LOW(IQR)")
                                elif float(objv) > float(high):
                                    alerts.append(f"{par}: objetivo > HIGH(IQR)")
                    alerta_txt = " | ".join(alerts) if alerts else ""
                    if "alerta" not in df_rank_obj.columns:
                        df_rank_obj.insert(2, "alerta", "")
                    if len(df_rank_obj.index) > 0:
                        df_rank_obj.loc[df_rank_obj.index[0], "alerta"] = alerta_txt
                    # actualizar estado de sugerencias (valor sugerido por parámetro)
                    try:
                        estado_sugerencias.clear()
                        if isinstance(summary, pd.DataFrame) and not summary.empty:
                            for _, row in summary.iterrows():
                                par = row.get("parametro")
                                if par is None:
                                    continue
                                val = row.get("w_mediana")
                                if pd.isna(val):
                                    val = row.get("mediana")
                                try:
                                    estado_sugerencias[str(par)] = (
                                        None if pd.isna(val) else float(val)
                                    )
                                except Exception:
                                    estado_sugerencias[str(par)] = None
                    except Exception:
                        pass
                except Exception:
                    pass

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
                params_sug = [
                    p.col
                    for p in params_dyn
                    if p.active and p.mode != "ignorar" and p.col in df.columns
                ]
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
                    sug,
                    titulo="Sugerencias (Top-K)",
                    collapsed=True,
                    bins=20,
                    get_objetivo=_get_objetivo,
                )
                # guardar referencia para callbacks
                acc_sug_ref["acc"] = acc_sug
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
                # guardar referencia para callbacks
                acc_out_ref["acc"] = acc_out
                display(acc_out)

    # Mantener stats del panel sincronizadas con el subset del dataset según segmentación actual
    def _current_subset_df() -> pd.DataFrame:
        col = dd_segcol.value
        if not col or col == "(ninguno)" or col not in df.columns:
            return df
        # Mapear el valor seleccionado si es SEGMENT_COL con etiquetas
        if dd_segm_val.value and dd_segm_val.value != "(ninguno)":
            val_ui = str(dd_segm_val.value)
            if col == SEGMENT_COL and SEGMENT_LABELS:
                # SEGMENT_LABELS: raw->label ; necesitamos raw que tenga ese label
                inv = {v: str(k) for k, v in SEGMENT_LABELS.items()}
                raw_val = inv.get(val_ui, val_ui)
            else:
                raw_val = val_ui
            try:
                return df[df[col].astype(str) == str(raw_val)]
            except Exception:
                return df
        return df

    def _update_param_stats_for_current_segment(*_):
        subset = _current_subset_df()
        param_panel.set_stats_from_df(subset)

    # Helpers: serializar/recuperar configuración de objetivo y globals
    def _collect_config_dict() -> dict:
        cfg = {
            "globals": {
                "alpha": float(sl_alpha.value),
                "top_k": int(sl_topk.value),
                "iqr_on": bool(ch_out.value),
                "iqr_factor": float(ft_iqrf.value),
                "penalizar_nan": bool(ch_nan.value),
                "penalidad_nan": float(ft_pen_nan.value),
                "segmentar_por": (
                    None if dd_segcol.value == "(ninguno)" else dd_segcol.value
                ),
                "segmentar_modo": dd_segm_modo.value,
                "segmentar_valor": (
                    None
                    if dd_segm_val.value in (None, "(ninguno)")
                    else dd_segm_val.value
                ),
                "prefer_factor": float(sl_pref_fac.value),
            },
            "params": [],
        }
        for r in param_panel.rows:
            cfg["params"].append(
                {
                    "col": r.col,
                    "label": r.label,
                    "active": bool(r.ch_active.value),
                    "mode": str(r.dd_mode.value),
                    "value": (
                        None
                        if r.ft_value.value in (None, "")
                        else float(r.ft_value.value)
                    ),
                    "min": (
                        None if r.ft_min.value in (None, "") else float(r.ft_min.value)
                    ),
                    "max": (
                        None if r.ft_max.value in (None, "") else float(r.ft_max.value)
                    ),
                    "weight": float(r.sl_weight.value),
                }
            )
        return cfg

    def _apply_config_dict(cfg: dict) -> None:
        try:
            g = cfg.get("globals", {})
            if g:
                sl_alpha.value = float(g.get("alpha", sl_alpha.value))
                sl_topk.value = int(g.get("top_k", sl_topk.value))
                ch_out.value = bool(g.get("iqr_on", ch_out.value))
                ft_iqrf.value = float(g.get("iqr_factor", ft_iqrf.value))
                ch_nan.value = bool(g.get("penalizar_nan", ch_nan.value))
                ft_pen_nan.value = float(g.get("penalidad_nan", ft_pen_nan.value))
                segcol = g.get("segmentar_por", None)
                dd_segcol.value = segcol if segcol in dd_segcol.options else "(ninguno)"
                dd_segm_modo.value = str(g.get("segmentar_modo", dd_segm_modo.value))
                val = g.get("segmentar_valor", None)
                if val and (val in dd_segm_val.options):
                    dd_segm_val.value = val
                sl_pref_fac.value = float(g.get("prefer_factor", sl_pref_fac.value))
        except Exception:
            pass
        # parámetros
        plist = cfg.get("params", []) or []
        row_by_col = {r.col: r for r in param_panel.rows}
        for p in plist:
            r = row_by_col.get(p.get("col"))
            if not r:
                continue
            try:
                r.ch_active.value = bool(p.get("active", r.ch_active.value))
                md = str(p.get("mode", r.dd_mode.value))
                if md in {"ignorar", "fijo", "maximo", "minimo", "rango"}:
                    r.dd_mode.value = md
                if p.get("value") not in (None, ""):
                    r.ft_value.value = float(p.get("value"))
                if p.get("min") not in (None, ""):
                    r.ft_min.value = float(p.get("min"))
                if p.get("max") not in (None, ""):
                    r.ft_max.value = float(p.get("max"))
                r.sl_weight.value = float(p.get("weight", r.sl_weight.value))
            except Exception:
                continue

    # Persistencia de sesiones (JSON/Excel) en analisis/Results/perfiles/sesiones
    perfiles_dir = os.path.join(
        PROJECT_ROOT, "analisis", "Results", "perfiles", "sesiones"
    )
    os.makedirs(perfiles_dir, exist_ok=True)

    def _refresh_profiles_dropdown():
        try:
            files = [f for f in os.listdir(perfiles_dir) if f.lower().endswith(".json")]
            opts = ["(cargar sesión…)"] + sorted(files)
            dd_load.options = opts
            dd_load.value = "(cargar sesión…)"
        except Exception:
            dd_load.options = ["(cargar sesión…)"]
            dd_load.value = "(cargar sesión…)"

    _refresh_profiles_dropdown()

    def on_save_profile(_):  # rename kept to minimize diff; acts as 'save session'
        import json, datetime

        cfg = _collect_config_dict()
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"sesion_{stamp}.json"
        fpath = os.path.join(perfiles_dir, fname)
        try:
            with open(fpath, "w", encoding="utf-8") as fh:
                json.dump(cfg, fh, ensure_ascii=False, indent=2)
            # Excel amigable (dos hojas: globals, params)
            try:
                import pandas as _pd

                with pd.ExcelWriter(
                    os.path.join(perfiles_dir, f"sesion_{stamp}.xlsx")
                ) as xw:
                    _pd.DataFrame([cfg.get("globals", {})]).to_excel(
                        xw, sheet_name="globals", index=False
                    )
                    _pd.DataFrame(cfg.get("params", [])).to_excel(
                        xw, sheet_name="params", index=False
                    )
            except Exception:
                pass
            _refresh_profiles_dropdown()
            with out_info:
                clear_output(wait=True)
                display(w.HTML(f"<b>Sesión guardada:</b> {fpath}"))
        except Exception as e:
            with out_info:
                clear_output(wait=True)
                display(w.HTML(f"<b>Error al guardar sesión:</b> {e}"))

    def on_load_profile(change):  # rename kept; acts as 'load session'
        if change.get("new") in (None, "(cargar sesión…)"):
            return
        sel = str(change.get("new"))
        fpath = os.path.join(perfiles_dir, sel)
        try:
            import json

            with open(fpath, "r", encoding="utf-8") as fh:
                cfg = json.load(fh)
            _apply_config_dict(cfg)
            with out_info:
                clear_output(wait=True)
                display(w.HTML(f"<b>Sesión cargada:</b> {sel}"))
        except Exception as e:
            with out_info:
                clear_output(wait=True)
                display(w.HTML(f"<b>Error al cargar sesión:</b> {e}"))

    # Autocargar la última sesión si está habilitado
    def _autoload_last_if_enabled():
        if not ch_autoload.value:
            return
        try:
            files = [f for f in os.listdir(perfiles_dir) if f.lower().endswith(".json")]
            if not files:
                return
            paths = [os.path.join(perfiles_dir, f) for f in files]
            last = max(paths, key=lambda p: os.path.getmtime(p))
            import json

            with open(last, "r", encoding="utf-8") as fh:
                cfg = json.load(fh)
            _apply_config_dict(cfg)
            with out_info:
                clear_output(wait=True)
                display(w.HTML(f"<b>Sesión auto-cargada:</b> {os.path.basename(last)}"))
        except Exception:
            pass

    btn_save.on_click(on_save_profile)
    dd_load.observe(on_load_profile, names="value")

    # reporte narrativo + exportaciones (Excel/CSV)
    def on_generar_informe(_):
        # construir mapas de objetivo/sugerido
        objetivo_map = {}
        try:
            for r in param_panel.rows:
                t = str(r.dd_mode.value)
                if not r.ch_active.value or t == "ignorar":
                    continue
                v = None
                if t in {"fijo", "maximo", "minimo"} and r.ft_value.value not in (
                    None,
                    "",
                ):
                    try:
                        v = float(r.ft_value.value)
                    except Exception:
                        v = None
                objetivo_map[r.col] = v
        except Exception:
            pass
        sugerido_map = dict(estado_sugerencias)
        df_rank_for_report = df_rank_obj_state["df"]

        md = narrativa_informe(
            df,
            objetivo_map,
            sugerido_map,
            df_rank=df_rank_for_report,
            top_k=int(sl_topk.value),
            titulo="Resumen de diseño (asistente)",
            iqr_factor=float(ft_iqrf.value),
        )
        # guardar en analisis/Results
        try:
            results_dir = os.path.join(PROJECT_ROOT, "analisis", "Results")
            os.makedirs(results_dir, exist_ok=True)
            p_md = os.path.join(results_dir, "informe_diseno.md")
            p_ht = os.path.join(results_dir, "informe_diseno.html")
            export_markdown(md, p_md)
            export_html(md, p_ht)
            # Además exportar Excel con hojas: ranking, sugerencias, topk_<param>
            try:
                import pandas as _pd

                sug_df = None
                try:
                    # Recalcular un resumen de sugerencias coherente al informe
                    params_sug = [
                        p.col
                        for p in (param_panel.collect_params())
                        if p.active and p.mode != "ignorar" and p.col in df.columns
                    ]
                    base_rank = (
                        df_rank_for_report
                        if isinstance(df_rank_for_report, pd.DataFrame)
                        else None
                    )
                    if base_rank is not None:
                        sug_pack = sugerencias_topk(
                            df_ranked=base_rank,
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
                    else:
                        sug_pack = {}
                    sug_df = sug_pack.get("summary")
                except Exception:
                    sug_df = None
                p_xlsx = os.path.join(results_dir, "informe_diseno.xlsx")
                with pd.ExcelWriter(p_xlsx) as xw:
                    if isinstance(df_rank_for_report, pd.DataFrame):
                        df_rank_for_report.to_excel(
                            xw, sheet_name="ranking", index=False
                        )
                    if isinstance(sug_df, pd.DataFrame):
                        sug_df.to_excel(xw, sheet_name="sugerencias", index=False)
                    if sug_df is not None and isinstance(sug_pack, dict):
                        dets = sug_pack.get("details", {}) or {}
                        for par, pack in dets.items():
                            usados = pack.get("usados")
                            if isinstance(usados, pd.DataFrame) and not usados.empty:
                                # Evitar columnas duplicadas de nombre si existieran
                                dfu = usados.copy()
                                cols = []
                                seen = set()
                                for c in dfu.columns:
                                    cc = c if c not in seen else f"{c}_1"
                                    seen.add(cc)
                                    cols.append(cc)
                                dfu.columns = cols
                                safe_sheet = "topk_" + str(par)[:25].replace(
                                    "/", "_"
                                ).replace("\\", "_")
                                dfu.to_excel(xw, sheet_name=safe_sheet, index=False)
                # CSVs básicos como fallback
                try:
                    if isinstance(df_rank_for_report, pd.DataFrame):
                        df_rank_for_report.to_csv(
                            os.path.join(results_dir, "ranking.csv"), index=False
                        )
                    if isinstance(sug_df, pd.DataFrame):
                        sug_df.to_csv(
                            os.path.join(results_dir, "sugerencias.csv"), index=False
                        )
                except Exception:
                    pass
            except Exception:
                pass
            with out_info:
                clear_output(wait=True)
                display(
                    w.HTML(
                        f"<b>Informe generado</b><br>MD: {p_md}<br>HTML: {p_ht}<br>Excel: {os.path.join(results_dir, 'informe_diseno.xlsx')}"
                    )
                )
        except Exception as e:
            with out_info:
                clear_output(wait=True)
                display(w.HTML(f"<b>Error al generar informe:</b> {e}"))

    btn_informe.on_click(on_generar_informe)
    btn_export.on_click(on_generar_informe)  # exporta junto con informe
    # ejecutar autoload al iniciar
    _autoload_last_if_enabled()

    # Panel de ayuda: índice + acordeón con secciones
    def _render_help():
        with out_help:
            clear_output(wait=True)
            # Etiquetas legibles por clave
            labels = {
                "modo_param": "Modo por parámetro",
                "valor_param": "Valor del parámetro",
                "peso_param": "Peso relativo",
                "alpha_sim": "α similitud (agregación)",
                "penalizar_nan": "Penalizar NaN",
                "penalidad_nan": "Penalidad por NaN",
                "segmentar_por": "Segmentar por",
                "segmentar_valor": "Valor del segmento",
                "modo_global_familia": "Modo de segmentación",
                "top_k": "Top‑K",
                "factor_prefer": "Factor prefer",
                "iqr_on": "Quitar atípicos (IQR)",
                "iqr_factor": "Factor IQR",
                "min_n": "Mínimo n",
                "auto": "Auto‑recalcular",
                # Tabla de ranking
                "ranking_sim": "Similitud",
                "ranking_dist": "Distancia",
                "ranking_alerta": "Alerta",
                "info_button": "Botón de info",
                "report_button": "Generar informe",
                # Tendencias
                "t_x": "X (independiente)",
                "t_y": "Y (dependiente)",
                "t_logx": "Log(X)",
                "t_obj_line": "Línea de objetivo",
                # Sugerencias
                "suger_box": "Boxplot e IQR",
                "suger_low_high": "LOW/HIGH (IQR)",
                "suger_w_mediana": "Mediana ponderada",
                "suger_n_efectivo": "n_efectivo (vecinos útiles)",
                "suger_pesos": "Definición de pesos",
                "suger_obj_line": "Línea de objetivo",
                # Outliers
                "out_iqr_explain": "Definición IQR",
                # Paneles
                "panel_info": "Panel de detalle",
                "narrativa": "Informe narrativo",
            }

            def make_list(keys: list[str]) -> w.HTML:
                items = []
                for k in keys:
                    if k in HELP:
                        label = labels.get(k, k)
                        items.append(f"<li><b>{label}:</b> {HELP[k]}</li>")
                return w.HTML(
                    "<ul style='margin:0 0 0 16px'>" + "".join(items) + "</ul>"
                )

            # Secciones del acordeón
            sections = [
                (
                    "Ranking y similitud",
                    [
                        "modo_param",
                        "valor_param",
                        "peso_param",
                        "alpha_sim",
                        "penalizar_nan",
                        "penalidad_nan",
                        "segmentar_por",
                        "segmentar_valor",
                        "modo_global_familia",
                        "top_k",
                        "factor_prefer",
                        "iqr_on",
                        "iqr_factor",
                        "auto",
                        "ranking_sim",
                        "ranking_dist",
                        "ranking_alerta",
                        "info_button",
                        "report_button",
                    ],
                ),
                (
                    "Tendencias X–Y",
                    ["t_x", "t_y", "t_logx", "t_obj_line", "r2_adj", "min_n"],
                ),
                (
                    "Sugerencias (Top‑K)",
                    [
                        "suger_box",
                        "suger_low_high",
                        "suger_w_mediana",
                        "suger_n_efectivo",
                        "suger_pesos",
                        "suger_obj_line",
                        "dispersion_indicator",
                    ],
                ),
                ("Outliers", ["out_iqr_explain"]),
                ("Panel e informe", ["panel_info", "narrativa"]),
            ]

            # Construir acordeón con todas las secciones colapsadas
            children = [make_list(keys) for _, keys in sections]
            acc = w.Accordion(children=children)
            for i, (title, _keys) in enumerate(sections):
                acc.set_title(i, title)
            acc.selected_index = None  # todo colapsado por defecto

            # Índice de navegación superior
            btns = []
            for i, (title, _keys) in enumerate(sections):
                b = w.Button(description=title, layout=w.Layout(width="auto"))
                b.style.button_color = "#e9ecef"

                def on_click_factory(idx: int):
                    def _go(_):
                        # abrir/cerrar: si ya está abierto, colapsar; si no, abrir
                        if acc.selected_index == idx:
                            acc.selected_index = None
                        else:
                            acc.selected_index = idx

                    return _go

                b.on_click(on_click_factory(i))
                btns.append(b)
            idx_box = w.HBox(btns)

            display(
                w.VBox(
                    [
                        w.HTML("<h4 style='margin:0 0 8px 0'>Ayuda de controles</h4>"),
                        idx_box,
                        w.HTML("<div style='height:6px'></div>"),
                        acc,
                    ]
                )
            )

    def _toggle_help(change):
        val = bool(change.get("new"))
        out_help.layout.display = "" if val else "none"
        if val:
            _render_help()
        else:
            with out_help:
                clear_output(wait=True)

    btn_ayuda.observe(_toggle_help, names="value")

    def _clear(_):
        # reset panel dinámico
        for r in param_panel.rows:
            r.dd_mode.value = "ignorar"
            r.ch_active.value = False
            r.ft_value.value = None
            r.sl_weight.value = float(PARAM_DEFAULTS.get("weight", 1.0))
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

    # Debounce para auto-recalcular
    _debounce_timer: dict[str, Any] = {"t": None}

    def _debounced_render(delay: float = 0.35):
        try:
            t: Optional[threading.Timer] = _debounce_timer.get("t")  # type: ignore[assignment]
            if t is not None:
                t.cancel()
        except Exception:
            pass

        def _do():
            try:
                _render()
            except Exception:
                pass

        try:
            timer = threading.Timer(delay, _do)
            _debounce_timer["t"] = timer
            timer.daemon = True
            timer.start()
        except Exception:
            _render()

    def on_any_change(change):
        if ch_auto.value:
            _debounced_render(0.35)

    def wire_observers():
        # Botones de acción
        btn_run.on_click(lambda _: _render())
        btn_clear.on_click(_clear)
        # Poblado/estadísticas de segmentación (estos no disparan render, sólo stats)
        dd_segcol.observe(_populate_segment_values, names="value")
        dd_segm_val.observe(_update_param_stats_for_current_segment, names="value")
        dd_segm_modo.observe(_update_param_stats_for_current_segment, names="value")
        # Observers unificados (globales)
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
            try:
                wdg.observe(on_any_change, names="value")
            except Exception:
                pass
        # Observers unificados (panel dinámico de parámetros)
        for r in getattr(param_panel, "rows", []):
            for wdg in (
                r.ch_active,
                r.dd_mode,
                r.ft_value,
                r.ft_min,
                r.ft_max,
                r.sl_weight,
            ):
                try:
                    wdg.observe(on_any_change, names="value")
                except Exception:
                    pass

    # Render inicial
    _populate_segment_values()
    # Inicializar stats (globales) para prefill de rangos
    param_panel.set_stats_from_df(df)
    header = w.HTML("<h4>Parámetros (dinámico)</h4>")
    # Orden: sticky bar arriba (siempre visible), luego panel de parámetros
    left_panel = w.VBox(
        [
            sticky_bar,
            header,
            param_panel,
        ],
        layout=w.Layout(width="100%"),
    )
    right_panel = w.VBox(
        [
            w.HBox(
                [btn_informe, btn_export, btn_save, dd_load, ch_autoload, btn_ayuda]
            ),
            w.HTML("<b>Detalle del parámetro</b>"),
            out_info,
            out_help,
        ]
    )
    top_row = w.HBox([left_panel, w.VBox([right_panel], layout=w.Layout(width="40%"))])
    container = w.VBox(
        [
            top_row,
            w.HTML("<hr>"),
            out_rank,
            w.HTML("<hr>"),
            out_sug,
            w.HTML("<hr>"),
            acc_tend,
            w.HTML("<hr>"),
            out_outl,
        ]
    )
    # Conectar observers centralizados y render inicial
    wire_observers()
    _render()
    return container


def run_demo():
    """Carga datos y muestra la UI integrada."""
    df = leer_excel()
    ui = build_ui(df, params=PARAMS_DEFAULT)
    display(ui)


if __name__ == "__main__":
    run_demo()
