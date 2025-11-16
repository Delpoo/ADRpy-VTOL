# === Crear/actualizar: Modulos/ux_notebook_panel.py ===
# Objetivo: construir el panel con ipywidgets (tabs) que edita CONFIG y, con botones,
# guarda overrides/snapshot y ejecuta el pipeline a través de Modulos.controller.
# Este módulo NO contiene lógica de negocio; sólo UI y llamadas al controlador.

from __future__ import annotations
import sys
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# imports del controlador y la config
from Modulos.controller import (
    default_config,
    get_config,
    load_effective_config,
    save_overrides,
    snapshot_config,
    run_pipeline,
)

import ipywidgets as W
from IPython.display import display


def _w_bool(val):
    return W.Checkbox(value=bool(val), indent=False)


def _w_int(val):
    return W.BoundedIntText(
        value=int(val), min=-(10**6), max=10**6, layout=W.Layout(width="140px")
    )


def _w_float(val):
    return W.BoundedFloatText(
        value=float(val), min=-1e9, max=1e9, step=0.01, layout=W.Layout(width="160px")
    )


def _w_str(val):
    return W.Text(value=str(val), layout=W.Layout(width="420px"))


def _w_color(val):
    return W.Text(value=str(val), placeholder="#RRGGBB", layout=W.Layout(width="120px"))


def _w_color_picker(val):
    try:
        return W.ColorPicker(
            value=str(val), concise=True, description="", layout=W.Layout(width="160px")
        )
    except Exception:
        # Fallback si ColorPicker no está disponible
        return _w_color(val)


# === Similitud (Avanzado) ===
def _build_sim_advanced(cfg: dict):
    sim = cfg.get("similitud", {})
    familias = sim.get("familias", {})
    familias_usadas_def = sim.get(
        "familias_usadas",
        list(familias.keys()) or ["fisica", "geometrica", "prestacional"],
    )

    import ipywidgets as W

    def _csv_text(lst):
        if isinstance(lst, (list, tuple)):
            return ", ".join(map(str, lst))
        return str(lst or "")

    # --- BÁSICOS (antes en la pestaña "Similitud") ---
    w_umbral = W.BoundedFloatText(
        value=float(sim.get("umbral_pct_diferencia", 0.20)),
        min=0.0,
        max=1.0,
        step=0.01,
        layout=W.Layout(width="120px"),
    )
    w_minfam = W.BoundedIntText(
        value=int(sim.get("min_familias", 3)),
        min=1,
        max=10**6,
        layout=W.Layout(width="120px"),
    )
    exc = sim.get("excepcion_min_familias", {"min_familias": 2, "min_parametros": 6})
    w_exc_fam = W.BoundedIntText(
        value=int(exc.get("min_familias", 2)),
        min=1,
        max=10**6,
        layout=W.Layout(width="120px"),
    )
    w_exc_par = W.BoundedIntText(
        value=int(exc.get("min_parametros", 6)),
        min=1,
        max=10**6,
        layout=W.Layout(width="120px"),
    )
    w_kmin = W.BoundedIntText(
        value=int(sim.get("k_min", 3)),
        min=1,
        max=10**6,
        layout=W.Layout(width="120px"),
    )
    w_kmax = W.BoundedIntText(
        value=int(sim.get("k_max", 10)),
        min=1,
        max=10**6,
        layout=W.Layout(width="120px"),
    )
    w_wsim = W.BoundedFloatText(
        value=float(sim.get("peso_confianza_similitud", 0.7)),
        min=0.0,
        max=1.0,
        step=0.01,
        layout=W.Layout(width="120px"),
    )
    w_wcv = W.BoundedFloatText(
        value=float(sim.get("peso_confianza_cv", 0.3)),
        min=0.0,
        max=1.0,
        step=0.01,
        layout=W.Layout(width="120px"),
    )
    w_verb = W.BoundedIntText(
        value=int(sim.get("verbosidad", 0)),
        min=0,
        max=3,
        layout=W.Layout(width="120px"),
    )

    ui_basicos = W.VBox(
        [
            W.HTML("<b>Básicos</b>"),
            W.HBox(
                [
                    W.Label(
                        "Umbral % diferencia (0–1)", layout=W.Layout(width="240px")
                    ),
                    w_umbral,
                ]
            ),
            W.HBox(
                [
                    W.Label("Mínimo familias", layout=W.Layout(width="240px")),
                    w_minfam,
                ]
            ),
            W.HBox(
                [
                    W.Label("Excepción: min familias", layout=W.Layout(width="240px")),
                    w_exc_fam,
                ]
            ),
            W.HBox(
                [
                    W.Label(
                        "Excepción: min parámetros", layout=W.Layout(width="240px")
                    ),
                    w_exc_par,
                ]
            ),
            W.HBox(
                [
                    W.Label("k mínimo", layout=W.Layout(width="240px")),
                    w_kmin,
                ]
            ),
            W.HBox(
                [
                    W.Label("k máximo", layout=W.Layout(width="240px")),
                    w_kmax,
                ]
            ),
            W.HBox(
                [
                    W.Label("Peso confianza similitud", layout=W.Layout(width="240px")),
                    w_wsim,
                ]
            ),
            W.HBox(
                [
                    W.Label("Peso confianza CV", layout=W.Layout(width="240px")),
                    w_wcv,
                ]
            ),
            W.HBox(
                [
                    W.Label("Verbosidad (0/1/2/3)", layout=W.Layout(width="240px")),
                    w_verb,
                ]
            ),
            W.HTML("<hr>"),
        ]
    )

    # --- Familias: editor CSV por familia ---
    fam_keys = list(familias.keys()) or ["fisica", "geometrica", "prestacional"]
    fam_boxes = {}
    for fk in fam_keys:
        fam_boxes[fk] = W.Text(
            value=_csv_text(familias.get(fk, [])),
            layout=W.Layout(width="600px"),
            placeholder="característica1, característica2, ...",
        )

    sel_familias_usadas = W.SelectMultiple(
        options=fam_keys,
        value=tuple([f for f in familias_usadas_def if f in fam_keys])
        or tuple(fam_keys),
        rows=min(6, len(fam_keys)),
        layout=W.Layout(width="260px", height="120px"),
    )

    ui_familias = W.VBox(
        [
            W.HTML("<b>Familias y características (CSV por familia)</b>"),
            W.HBox(
                [
                    W.VBox(
                        [
                            W.HBox(
                                [
                                    W.Label(f"{fk}", layout=W.Layout(width="140px")),
                                    fam_boxes[fk],
                                ]
                            )
                            for fk in fam_keys
                        ]
                    )
                ]
            ),
            W.HBox(
                [
                    W.Label("familias_usadas", layout=W.Layout(width="140px")),
                    sel_familias_usadas,
                ]
            ),
        ]
    )

    # --- Función de similitud (polinomio a2 x^2 + a1 x + a0) ---
    fun = sim.get(
        "funcion_similitud",
        {
            "tipo": "polinomica",
            "coef": {"a2": -0.002, "a1": -0.01, "a0": 1.0},
            "dominio_max_pct": 20.0,
        },
    )
    a2 = W.BoundedFloatText(
        value=float(fun.get("coef", {}).get("a2", -0.002)),
        min=-1e3,
        max=1e3,
        step=0.001,
        layout=W.Layout(width="120px"),
    )
    a1 = W.BoundedFloatText(
        value=float(fun.get("coef", {}).get("a1", -0.01)),
        min=-1e3,
        max=1e3,
        step=0.001,
        layout=W.Layout(width="120px"),
    )
    a0 = W.BoundedFloatText(
        value=float(fun.get("coef", {}).get("a0", 1.0)),
        min=-1e3,
        max=1e3,
        step=0.001,
        layout=W.Layout(width="120px"),
    )
    domax = W.BoundedFloatText(
        value=float(fun.get("dominio_max_pct", 20.0)),
        min=0.0,
        max=1e6,
        step=0.5,
        layout=W.Layout(width="120px"),
    )

    ui_fun = W.VBox(
        [
            W.HTML(
                "<b>Función de similitud</b> &nbsp; <i>(x = diferencia %, 0≤x≤dominio)</i>"
            ),
            W.HBox(
                [
                    W.Label("a2"),
                    a2,
                    W.Label("a1"),
                    a1,
                    W.Label("a0"),
                    a0,
                    W.Label("dominio_max_pct"),
                    domax,
                ]
            ),
        ]
    )

    # --- Selección de vecinos ---
    vec = sim.get("vecinos", {"modo": "todos", "top_k": 10, "enforce_k_min": True})
    modo = W.Dropdown(
        options=["todos", "top_k", "k_en_rango"],
        value=vec.get("modo", "todos"),
        description="modo",
    )
    topk = W.BoundedIntText(
        value=int(vec.get("top_k", 10)),
        min=1,
        max=10**6,
        layout=W.Layout(width="120px"),
    )
    enforce = W.Checkbox(
        value=bool(vec.get("enforce_k_min", True)),
        description="enforce k_min",
        indent=False,
    )

    ui_vec = W.VBox(
        [
            W.HTML("<b>Selección de vecinos</b>"),
            W.HBox([modo, W.Label("top_k"), topk, enforce]),
        ]
    )

    # --- Confianza: cv_ref y penalización por k (coef. polinomio opcional) ---
    conf = sim.get(
        "confianza",
        {"cv_ref": 0.5, "penalizacion_k": {"tipo": "polinomica", "params": {}}},
    )
    cv_ref = W.BoundedFloatText(
        value=float(conf.get("cv_ref", 0.5)),
        min=1e-6,
        max=1e3,
        step=0.01,
        layout=W.Layout(width="120px"),
    )
    pk = conf.get("penalizacion_k", {}).get("params", {})
    a5 = W.FloatText(
        value=float(pk.get("a5", 0.00002281)), layout=W.Layout(width="120px")
    )
    a4 = W.FloatText(
        value=float(pk.get("a4", -0.00024)), layout=W.Layout(width="120px")
    )
    a3 = W.FloatText(value=float(pk.get("a3", -0.0036)), layout=W.Layout(width="120px"))
    a2k = W.FloatText(value=float(pk.get("a2", 0.046)), layout=W.Layout(width="120px"))
    a1k = W.FloatText(value=float(pk.get("a1", 0.0095)), layout=W.Layout(width="120px"))
    a0k = W.FloatText(value=float(pk.get("a0", 0.024)), layout=W.Layout(width="120px"))

    acc_conf = W.Accordion(
        children=[
            W.VBox(
                [
                    W.HBox([W.Label("a5"), a5, W.Label("a4"), a4, W.Label("a3"), a3]),
                    W.HBox(
                        [W.Label("a2"), a2k, W.Label("a1"), a1k, W.Label("a0"), a0k]
                    ),
                ]
            )
        ]
    )
    acc_conf.set_title(0, "Penalización por k (coef. polinomio)")

    ui_conf = W.VBox([W.HBox([W.Label("cv_ref"), cv_ref]), acc_conf])

    # --- Umbral por familia (0–1 o vacío para None) ---
    um_pf = sim.get(
        "umbral_pct_por_familia",
        {"fisica": None, "geometrica": None, "prestacional": None},
    )
    um_inputs = {}
    for fk in fam_keys:
        val = "" if um_pf.get(fk) in (None, "") else str(um_pf.get(fk))
        um_inputs[fk] = W.Text(
            value=val, placeholder="None o 0–1", layout=W.Layout(width="140px")
        )
    ui_um = W.VBox(
        [
            W.HTML(
                "<b>Umbral por familia</b> &nbsp; <i>(usar fracción 0–1; vacío = None)</i>"
            ),
            W.VBox(
                [
                    W.HBox([W.Label(fk, layout=W.Layout(width="140px")), um_inputs[fk]])
                    for fk in fam_keys
                ]
            ),
        ]
    )

    # --- Outliers para vecinos de similitud ---
    out = sim.get(
        "outliers",
        {
            "usar": False,
            "umbral_z_suave": 3.0,
            "umbral_z_duro": 6.0,
            "alpha_pesos": 0.5,
            "w_min": 0.2,
            "remover_duro": False,
        },
    )
    w_usar = W.Checkbox(
        value=bool(out.get("usar", False)),
        description="Usar outliers en vecinos",
        indent=False,
    )
    w_zs = W.BoundedFloatText(
        value=float(out.get("umbral_z_suave", 3.0)),
        min=0.0,
        max=1e9,
        step=0.1,
        layout=W.Layout(width="120px"),
    )
    w_zd = W.BoundedFloatText(
        value=float(out.get("umbral_z_duro", 6.0)),
        min=0.0,
        max=1e9,
        step=0.1,
        layout=W.Layout(width="120px"),
    )
    w_al = W.BoundedFloatText(
        value=float(out.get("alpha_pesos", 0.5)),
        min=0.0,
        max=1.0,
        step=0.01,
        layout=W.Layout(width="120px"),
    )
    w_wm = W.BoundedFloatText(
        value=float(out.get("w_min", 0.2)),
        min=0.0,
        max=1.0,
        step=0.01,
        layout=W.Layout(width="120px"),
    )
    w_rd = W.Checkbox(
        value=bool(out.get("remover_duro", False)),
        description="remover_duro",
        indent=False,
    )

    ui_out = W.VBox(
        [
            W.HTML("<b>Outliers (Similitud)</b>"),
            W.HBox(
                [
                    w_usar,
                    W.Label("z_suave"),
                    w_zs,
                    W.Label("z_duro"),
                    w_zd,
                    W.Label("alpha"),
                    w_al,
                    W.Label("w_min"),
                    w_wm,
                    w_rd,
                ]
            ),
        ]
    )

    # --- empaquetado en acordeón ---
    acc = W.Accordion(
        children=[ui_basicos, ui_familias, ui_fun, ui_vec, ui_conf, ui_um, ui_out]
    )
    acc.set_title(0, "Básicos")
    acc.set_title(1, "Familias")
    acc.set_title(2, "Función de similitud")
    acc.set_title(3, "Selección de vecinos")
    acc.set_title(4, "Confianza (cv_ref & penalización k)")
    acc.set_title(5, "Umbral por familia")
    acc.set_title(6, "Outliers (Similitud)")

    # --- collect() ---
    def collect():
        basicos_out = {
            "umbral_pct_diferencia": float(w_umbral.value),
            "min_familias": int(w_minfam.value),
            "excepcion_min_familias": {
                "min_familias": int(w_exc_fam.value),
                "min_parametros": int(w_exc_par.value),
            },
            "k_min": int(w_kmin.value),
            "k_max": int(w_kmax.value),
            "peso_confianza_similitud": float(w_wsim.value),
            "peso_confianza_cv": float(w_wcv.value),
            "verbosidad": int(w_verb.value),
        }
        # familias (CSV -> lista)
        fam_out = {}
        for fk in fam_keys:
            raw = fam_boxes[fk].value
            items = [s.strip() for s in raw.split(",") if s.strip()]
            fam_out[fk] = items

        # familias usadas
        usadas = list(sel_familias_usadas.value) or fam_keys

        # función
        fun_out = {
            "tipo": "polinomica",
            "coef": {
                "a2": float(a2.value),
                "a1": float(a1.value),
                "a0": float(a0.value),
            },
            "dominio_max_pct": float(domax.value),
        }

        # vecinos
        vec_out = {
            "modo": modo.value,
            "top_k": int(topk.value),
            "enforce_k_min": bool(enforce.value),
        }

        # confianza
        pk_out = {
            "a5": float(a5.value),
            "a4": float(a4.value),
            "a3": float(a3.value),
            "a2": float(a2k.value),
            "a1": float(a1k.value),
            "a0": float(a0k.value),
        }
        conf_out = {
            "cv_ref": float(cv_ref.value),
            "penalizacion_k": {"tipo": "polinomica", "params": pk_out},
        }

        # umbral por familia (Text -> None/float)
        umpf = {}
        for fk in fam_keys:
            txt = (um_inputs[fk].value or "").strip().lower()
            if txt in ("", "none", "null"):
                umpf[fk] = None
            else:
                try:
                    umpf[fk] = float(txt)
                except Exception:
                    umpf[fk] = None

        # outliers similitud
        out_sim = {
            "usar": bool(w_usar.value),
            "umbral_z_suave": float(w_zs.value),
            "umbral_z_duro": float(w_zd.value),
            "alpha_pesos": float(w_al.value),
            "w_min": float(w_wm.value),
            "remover_duro": bool(w_rd.value),
        }

        return {
            **basicos_out,
            "familias": fam_out,
            "familias_usadas": usadas,
            "funcion_similitud": fun_out,
            "vecinos": vec_out,
            "confianza": conf_out,
            "umbral_pct_por_familia": umpf,
            "outliers": out_sim,
        }, usadas

    return acc, collect


# === Loop / Orquestación avanzada ===
def _build_loop_advanced(cfg: dict):
    loop = cfg.get("loop", {})
    import ipywidgets as W

    # Orden (primero/segundo)
    primero = (
        loop.get("orden", ["similitud", "correlacion"])[0]
        if loop.get("orden")
        else "similitud"
    )
    dd_first = W.Dropdown(
        options=["similitud", "correlacion"], value=primero, description="Primero"
    )

    def _second_of(first):
        return "correlacion" if first == "similitud" else "similitud"

    lbl_second = W.HTML(f"<b>Segundo:</b> {_second_of(dd_first.value)}")

    def _on_first_change(change):
        if change["name"] == "value":
            lbl_second.value = f"<b>Segundo:</b> {_second_of(change['new'])}"

    dd_first.observe(_on_first_change)

    ui_order = W.HBox([dd_first, lbl_second])

    # Combinación
    comb = loop.get("combinacion", {"metodo": "promedio_ponderado", "conf_min": 0.0})
    dd_met = W.Dropdown(
        options=[
            "promedio_ponderado",
            "mejor_confianza",
            "prioridad_correlacion",
            "prioridad_similitud",
        ],
        value=comb.get("metodo", "promedio_ponderado"),
        description="comb.",
    )
    conf_min = W.BoundedFloatText(
        value=float(comb.get("conf_min", 0.0)),
        min=0.0,
        max=1.0,
        step=0.01,
        layout=W.Layout(width="120px"),
    )
    ui_comb = W.HBox([dd_met, W.Label("conf_min"), conf_min])

    # Stop
    stop = loop.get("stop", {"min_nuevas_por_iter": 1, "sin_mejora_consecutivas": 1})
    min_new = W.BoundedIntText(
        value=int(stop.get("min_nuevas_por_iter", 1)),
        min=0,
        max=10**6,
        layout=W.Layout(width="140px"),
    )
    sin_mej = W.BoundedIntText(
        value=int(stop.get("sin_mejora_consecutivas", 1)),
        min=0,
        max=10**6,
        layout=W.Layout(width="140px"),
    )
    ui_stop = W.HBox(
        [
            W.Label("min_nuevas_por_iter"),
            min_new,
            W.Label("sin_mejora_consecutivas"),
            sin_mej,
        ]
    )

    # Export
    exp = loop.get(
        "export",
        {
            "json": {
                "enabled": True,
                "dir": "Results",
                "fname": "modelos_completos_por_celda.json",
            },
            "html_df_base": {"mostrar": True},
        },
    )
    json_enabled = W.Checkbox(
        value=bool(exp.get("json", {}).get("enabled", True)),
        description="Export JSON",
        indent=False,
    )
    json_dir = W.Text(
        value=str(exp.get("json", {}).get("dir", "Results")),
        layout=W.Layout(width="240px"),
    )
    json_fname = W.Text(
        value=str(exp.get("json", {}).get("fname", "modelos_completos_por_celda.json")),
        layout=W.Layout(width="320px"),
    )
    html_show = W.Checkbox(
        value=bool(exp.get("html_df_base", {}).get("mostrar", True)),
        description="Mostrar HTML df_base",
        indent=False,
    )

    ui_export = W.VBox(
        [
            W.HBox(
                [json_enabled, W.Label("dir"), json_dir, W.Label("fname"), json_fname]
            ),
            W.HBox([html_show]),
        ]
    )

    # Empaquetado
    acc = W.Accordion(
        children=[
            W.VBox([ui_order]),
            W.VBox([ui_comb]),
            W.VBox([ui_stop]),
            W.VBox([ui_export]),
        ]
    )
    acc.set_title(0, "Orden de ejecución")
    acc.set_title(1, "Combinación de métodos")
    acc.set_title(2, "Criterios de parada")
    acc.set_title(3, "Export/HTML")

    # collect
    def collect():
        orden = [dd_first.value, _second_of(dd_first.value)]
        return {
            "orden": orden,
            "combinacion": {"metodo": dd_met.value, "conf_min": float(conf_min.value)},
            "stop": {
                "min_nuevas_por_iter": int(min_new.value),
                "sin_mejora_consecutivas": int(sin_mej.value),
            },
            "export": {
                "json": {
                    "enabled": bool(json_enabled.value),
                    "dir": json_dir.value,
                    "fname": json_fname.value,
                },
                "html_df_base": {"mostrar": bool(html_show.value)},
            },
        }

    return acc, collect


# === agregar en Modulos/ux_notebook_panel.py (debajo de helpers de widgets) ===
def _build_corr_advanced(cfg: dict):
    c = cfg.get("correlacion", {})
    out = W.Output()

    # ---- Checks 2D ----
    chk = c.get("checks_2d", {})
    w_chk_enabled = W.Checkbox(
        value=bool(chk.get("enabled", True)), description="Activar checks 2D"
    )
    # (Se eliminó la pestaña antigua Correlación/Outliers: todo está en Correlación (Avanzado))

    def _wb(val):
        return W.Checkbox(value=bool(val), indent=False)

    def _wf(val):
        return W.BoundedFloatText(
            value=float(val),
            min=0.0,
            max=1e9,
            step=0.01,
            layout=W.Layout(width="120px"),
        )

    def _wi(val):
        return W.BoundedIntText(
            value=int(val), min=0, max=10**6, layout=W.Layout(width="120px")
        )

    # switches + umbrales
    rows = []

    def row_check(name, subkey, label, default, kind="float"):
        cfgk = chk.get(name, {})
        on = _wb(cfgk.get("enabled", True))
        if kind == "float":
            val = _wf(cfgk.get(subkey, default))
        elif kind == "int":
            val = _wi(cfgk.get(subkey, default))
        else:
            val = _wf(cfgk.get(subkey, default))
        rows.append((name, subkey, on, val))
        return W.HBox([on, W.Label(label, layout=W.Layout(width="240px")), val])

    ui_checks = W.VBox(
        [
            w_chk_enabled,
            row_check("pearson", "abs_r_max", "|r| máx", 0.90, "float"),
            row_check("vif", "max", "VIF máx", 10.0, "float"),
            row_check("pc2", "ratio_min", "PC2 ratio mín", 0.03, "float"),
            row_check("rank", "min", "Rango mínimo", 2, "int"),
            row_check("cond", "max", "Condición máx", 1e5, "float"),
            row_check(
                "coverage_unique_pair",
                "ratio_min",
                "Cobertura pares únicos mín",
                0.60,
                "float",
            ),
            row_check(
                "coverage_hull", "ratio_min", "Cobertura hull mín", 0.15, "float"
            ),
            row_check(
                "coverage_ellipse", "ratio_min", "Cobertura elipse mín", 0.10, "float"
            ),
            row_check("n_per_param", "linear2_min", "n/param (linear-2)", 8, "int"),
            row_check("n_per_param", "poly2_min", "n/param (poly-2)", 10, "int"),
            row_check("agresivo", "abs_r_min", "Modo agresivo |r| mín", 0.95, "float"),
        ]
    )

    # ---- Diversidad mínima ----
    div = c.get("diversidad_minima", {})
    mu = div.get("min_unicos", {})
    mm = div.get("min_muestras", {})
    tipos = ["exp-1", "log-1", "pot-1", "linear-1", "poly-1", "linear-2", "poly-2"]
    ui_min = []
    for t in tipos:
        ui_min.append(
            W.HBox(
                [
                    W.Label(t, layout=W.Layout(width="90px")),
                    _wi(mu.get(t, 5)),
                    _wi(mm.get(t, 6)),
                ]
            )
        )
    ui_div = W.VBox(
        [
            W.HBox(
                [
                    W.Label("tipo", layout=W.Layout(width="90px")),
                    W.Label("min_unicos"),
                    W.Label("min_muestras"),
                ]
            )
        ]
        + ui_min
    )

    # ---- Extrapolación ----
    ex = c.get("extrapolacion", {})
    w_modo_pred = W.Dropdown(
        options=["eliminar", "permitir_con_tolerancia"],
        value=ex.get("modo_predictores", "eliminar"),
        description="Predictores",
    )
    w_tol_pct = _wf(ex.get("tolerancia_pct", 0.0))
    w_modo2d = W.Dropdown(
        options=["marginal", "convex_hull"],
        value=ex.get("modo_2d", "marginal"),
        description="2D",
    )
    w_hull_pad = _wf(ex.get("tolerancia_hull_pad", 0.0))
    ui_ex = W.VBox(
        [
            W.HBox([w_modo_pred, W.Label("tolerancia_pct"), w_tol_pct]),
            W.HBox([w_modo2d, W.Label("hull_pad"), w_hull_pad]),
        ]
    )

    # ---- Modelos (1D/2D + catálogo disponible) ----
    md = c.get("modelos", {})
    modelos_base = cfg.get("modelos", {})
    habilitados_cfg = md.get("habilitados", modelos_base.get("habilitados", {}))

    w_m1d = _wb(md.get("permitir_1d", True))
    w_m2d = _wb(md.get("permitir_2d", True))
    w_pdeg = _wi(md.get("poly_grado", 2))

    w_mod_lineal = W.Checkbox(
        value=bool(habilitados_cfg.get("lineal", True)), description="lineal"
    )
    w_mod_poly2 = W.Checkbox(
        value=bool(habilitados_cfg.get("polinomico2", True)), description="poly2"
    )
    w_mod_log = W.Checkbox(
        value=bool(habilitados_cfg.get("log", True)), description="log"
    )
    w_mod_pot = W.Checkbox(
        value=bool(habilitados_cfg.get("potencia", True)), description="pot"
    )
    w_mod_exp = W.Checkbox(
        value=bool(habilitados_cfg.get("exponencial", True)), description="exp"
    )

    w_mod_mape = W.BoundedFloatText(
        value=float(
            md.get("umbral_mape_max", modelos_base.get("umbral_mape_max", 0.35))
        ),
        min=0.0,
        max=100.0,
        step=0.01,
        layout=W.Layout(width="140px"),
    )
    w_mod_loocv = W.Checkbox(
        value=bool(md.get("usar_loocv", modelos_base.get("usar_loocv", True))),
        description="Usar LOOCV",
    )
    w_mod_loocv_w = W.Checkbox(
        value=bool(
            md.get("loocv_usa_pesos", modelos_base.get("loocv_usa_pesos", False))
        ),
        description="LOOCV usa pesos",
    )

    pond_cfg = md.get(
        "ponderaciones_seleccion",
        modelos_base.get(
            "ponderaciones_seleccion",
            {"mape": 0.5, "r2": 0.2, "corr": 0.2, "confianza": 0.1},
        ),
    )
    w_mod_w_mape = W.BoundedFloatText(
        value=float(pond_cfg.get("mape", 0.5)),
        min=0.0,
        max=1.0,
        step=0.01,
        layout=W.Layout(width="90px"),
    )
    w_mod_w_r2 = W.BoundedFloatText(
        value=float(pond_cfg.get("r2", 0.2)),
        min=0.0,
        max=1.0,
        step=0.01,
        layout=W.Layout(width="90px"),
    )
    w_mod_w_corr = W.BoundedFloatText(
        value=float(pond_cfg.get("corr", 0.2)),
        min=0.0,
        max=1.0,
        step=0.01,
        layout=W.Layout(width="90px"),
    )
    w_mod_w_conf = W.BoundedFloatText(
        value=float(pond_cfg.get("confianza", 0.1)),
        min=0.0,
        max=1.0,
        step=0.01,
        layout=W.Layout(width="90px"),
    )

    ui_modelos = W.VBox(
        [
            W.HTML("<b>Modelos</b>"),
            W.HBox(
                [
                    W.Label("Permitir 1D", layout=W.Layout(width="120px")),
                    w_m1d,
                    W.Label("Permitir 2D", layout=W.Layout(width="120px")),
                    w_m2d,
                    W.Label("Grado polinómico", layout=W.Layout(width="140px")),
                    w_pdeg,
                ]
            ),
            W.HBox(
                [
                    W.Label("Habilitados:", layout=W.Layout(width="120px")),
                    w_mod_lineal,
                    w_mod_poly2,
                    w_mod_log,
                    w_mod_pot,
                    w_mod_exp,
                ]
            ),
            W.HBox(
                [
                    W.Label(
                        "Umbral MAPE máx. (%)",
                        layout=W.Layout(width="200px"),
                    ),
                    w_mod_mape,
                ]
            ),
            W.HBox(
                [
                    w_mod_loocv,
                    w_mod_loocv_w,
                ]
            ),
            W.HTML("<b>Ponderaciones de selección</b>"),
            W.HBox(
                [
                    W.Label("MAPE", layout=W.Layout(width="60px")),
                    w_mod_w_mape,
                    W.Label("R²", layout=W.Layout(width="60px")),
                    w_mod_w_r2,
                    W.Label("Corr", layout=W.Layout(width="60px")),
                    w_mod_w_corr,
                    W.Label("Conf.", layout=W.Layout(width="60px")),
                    w_mod_w_conf,
                ]
            ),
        ]
    )

    # ---- Confianza ----
    cf = c.get("confianza", {})
    w_wr2 = _wf(cf.get("w_r2", 0.5))
    w_wmp = _wf(cf.get("w_mape", 0.5))
    w_div = _wf(cf.get("mape_divisor", 15.0))
    ui_conf = W.VBox(
        [
            W.HBox(
                [
                    W.Label("w_r2"),
                    w_wr2,
                    W.Label("w_mape"),
                    w_wmp,
                    W.Label("mape_divisor"),
                    w_div,
                ]
            ),
        ]
    )

    # ---- NUEVO: Penalización por k (polinomio a5..a0) ----
    pk_corr = cf.get("penalizacion_k", {"tipo": "polinomica", "params": {}})
    _ka5 = W.FloatText(
        value=float(pk_corr.get("params", {}).get("a5", 0.00002281)),
        layout=W.Layout(width="120px"),
    )
    _ka4 = W.FloatText(
        value=float(pk_corr.get("params", {}).get("a4", -0.00024)),
        layout=W.Layout(width="120px"),
    )
    _ka3 = W.FloatText(
        value=float(pk_corr.get("params", {}).get("a3", -0.0036)),
        layout=W.Layout(width="120px"),
    )
    _ka2 = W.FloatText(
        value=float(pk_corr.get("params", {}).get("a2", 0.046)),
        layout=W.Layout(width="120px"),
    )
    _ka1 = W.FloatText(
        value=float(pk_corr.get("params", {}).get("a1", 0.0095)),
        layout=W.Layout(width="120px"),
    )
    _ka0 = W.FloatText(
        value=float(pk_corr.get("params", {}).get("a0", 0.024)),
        layout=W.Layout(width="120px"),
    )
    acc_pk = W.Accordion(
        children=[
            W.HBox(
                [
                    W.Label("a5"),
                    _ka5,
                    W.Label("a4"),
                    _ka4,
                    W.Label("a3"),
                    _ka3,
                    W.Label("a2"),
                    _ka2,
                    W.Label("a1"),
                    _ka1,
                    W.Label("a0"),
                    _ka0,
                ]
            )
        ]
    )
    acc_pk.set_title(0, "Penalización por k (polinomio)")

    # ---- NUEVO: Penalización por N (polinomio b3*N^3 + b2*N^2 + b1*N + b0) ----
    pn = cf.get("penalizacion_n", {"tipo": "polinomica", "params": {}})
    _b3 = W.FloatText(
        value=float(pn.get("params", {}).get("b3", 0.0)), layout=W.Layout(width="120px")
    )
    _b2 = W.FloatText(
        value=float(pn.get("params", {}).get("b2", 0.0025)),
        layout=W.Layout(width="120px"),
    )
    _b1 = W.FloatText(
        value=float(pn.get("params", {}).get("b1", 0.02)),
        layout=W.Layout(width="120px"),
    )
    _b0 = W.FloatText(
        value=float(pn.get("params", {}).get("b0", 0.10)),
        layout=W.Layout(width="120px"),
    )
    acc_pn = W.Accordion(
        children=[
            W.HBox(
                [
                    W.Label("b3"),
                    _b3,
                    W.Label("b2"),
                    _b2,
                    W.Label("b1"),
                    _b1,
                    W.Label("b0"),
                    _b0,
                ]
            )
        ]
    )
    acc_pn.set_title(0, "Penalización por N (polinomio)")

    # ---- NUEVO: Penalizaciones por métricas 2D (usar + c2,c1,c0 por métrica) ----
    pm = cf.get("penalizaciones_metricas", {})

    def _row_metric(name, defaults):
        cfgm = pm.get(name, defaults)
        w_on = W.Checkbox(
            value=bool(cfgm.get("usar", defaults.get("usar", True))),
            description=name,
            indent=False,
        )
        p = cfgm.get("params", {})
        c2 = W.FloatText(
            value=float(p.get("c2", defaults.get("params", {}).get("c2", 0.0))),
            layout=W.Layout(width="110px"),
        )
        c1 = W.FloatText(
            value=float(p.get("c1", defaults.get("params", {}).get("c1", 0.0))),
            layout=W.Layout(width="110px"),
        )
        c0 = W.FloatText(
            value=float(p.get("c0", defaults.get("params", {}).get("c0", 1.0))),
            layout=W.Layout(width="110px"),
        )
        return (name, w_on, c2, c1, c0), W.HBox(
            [w_on, W.Label("c2"), c2, W.Label("c1"), c1, W.Label("c0"), c0]
        )

    rows_metrics = []
    ui_rows = []
    for nm, dflt in [
        ("pearson_abs", {"usar": True, "params": {"c2": -1.2, "c1": 1.2, "c0": 0.2}}),
        ("vif", {"usar": True, "params": {"c2": -0.02, "c1": -0.10, "c0": 1.2}}),
        ("cond", {"usar": True, "params": {"c2": -1e-10, "c1": -1e-5, "c0": 1.0}}),
        ("pc2_ratio", {"usar": True, "params": {"c2": 2.0, "c1": 0.0, "c0": 0.0}}),
        (
            "coverage_unique_pair",
            {"usar": True, "params": {"c2": 0.0, "c1": 0.8, "c0": 0.2}},
        ),
        ("coverage_hull", {"usar": True, "params": {"c2": 0.0, "c1": 0.8, "c0": 0.2}}),
        (
            "coverage_ellipse",
            {"usar": True, "params": {"c2": 0.0, "c1": 0.8, "c0": 0.2}},
        ),
    ]:
        row, uirow = _row_metric(nm, dflt)
        rows_metrics.append(row)
        ui_rows.append(uirow)
    acc_pm = W.Accordion(children=[W.VBox(ui_rows)])
    acc_pm.set_title(0, "Penalizaciones por métricas 2D")

    # ---- NUEVO: Aporte LOOCV a la confianza ----
    la = cf.get(
        "loocv_aporte",
        {
            "w": 0.2,
            "factor_por_clase": {"robusto": 1.0, "no_robusto": 0.85, "rechazado": 0.6},
        },
    )
    # Nota: evitar colisión de nombres con checkbox LOOCV más abajo
    w_lo_w_aporte = W.BoundedFloatText(
        value=float(la.get("w", 0.2)),
        min=0.0,
        max=1.0,
        step=0.01,
        layout=W.Layout(width="120px"),
    )
    w_lo_fr = W.BoundedFloatText(
        value=float(la.get("factor_por_clase", {}).get("robusto", 1.0)),
        min=0.0,
        max=2.0,
        step=0.01,
        layout=W.Layout(width="120px"),
    )
    w_lo_fnr = W.BoundedFloatText(
        value=float(la.get("factor_por_clase", {}).get("no_robusto", 0.85)),
        min=0.0,
        max=2.0,
        step=0.01,
        layout=W.Layout(width="120px"),
    )
    w_lo_fre = W.BoundedFloatText(
        value=float(la.get("factor_por_clase", {}).get("rechazado", 0.6)),
        min=0.0,
        max=2.0,
        step=0.01,
        layout=W.Layout(width="120px"),
    )
    ui_loocv = W.VBox(
        [
            W.HBox([W.Label("w (peso LOOCV)"), w_lo_w_aporte]),
            W.HBox(
                [
                    W.Label("factor ROBUSTO"),
                    w_lo_fr,
                    W.Label("NO ROBUSTO"),
                    w_lo_fnr,
                    W.Label("RECHAZADO"),
                    w_lo_fre,
                ]
            ),
        ]
    )

    # ---- Selección (train) / LOOCV ----
    sel = c.get("seleccion_modelos", {})
    tr = sel.get("train", {"mape_max": 7.5, "r2_min": 0.6})
    pf = sel.get("pre_filtro", {"mape_max": 18.0, "r2_min": 0.4})
    w_tr_m = _wf(tr.get("mape_max", 7.5))
    w_tr_r = _wf(tr.get("r2_min", 0.6))
    w_pf_m = _wf(pf.get("mape_max", 18.0))
    w_pf_r = _wf(pf.get("r2_min", 0.4))

    lo = c.get("loocv", {})
    w_lo_use = _wb(lo.get("usar", True))
    # Renombrado para evitar colisión con peso LOOCV de arriba
    w_lo_use_pesos = _wb(lo.get("usar_pesos_outliers", False))
    cr = lo.get("criterios", {})
    rob = cr.get("robusto", {"mape_max": 7.5, "r2_min": 0.6})
    nrb = cr.get("no_robusto", {"mape_max": 12.5, "r2_min": 0.45})
    w_rb_m = _wf(rob.get("mape_max", 7.5))
    w_rb_r = _wf(rob.get("r2_min", 0.6))
    w_nb_m = _wf(nrb.get("mape_max", 12.5))
    w_nb_r = _wf(nrb.get("r2_min", 0.45))
    w_ratio = _wf(lo.get("ratio_val_train_alerta", 5.0))

    ui_sel = W.VBox(
        [
            W.HTML("<b>Entrenamiento</b>"),
            W.HBox(
                [W.Label("train MAPE máx"), w_tr_m, W.Label("train R2 mín"), w_tr_r]
            ),
            W.HBox(
                [
                    W.Label("pre-filtro MAPE máx"),
                    w_pf_m,
                    W.Label("pre-filtro R2 mín"),
                    w_pf_r,
                ]
            ),
            W.HTML("<b>LOOCV</b>"),
            W.HBox(
                [
                    W.Label("usar LOOCV"),
                    w_lo_use,
                    W.Label("LOOCV usa pesos outliers"),
                    w_lo_use_pesos,
                ]
            ),
            W.HBox(
                [W.Label("robusto MAPE máx"), w_rb_m, W.Label("robusto R2 mín"), w_rb_r]
            ),
            W.HBox(
                [
                    W.Label("no robusto MAPE máx"),
                    w_nb_m,
                    W.Label("no robusto R2 mín"),
                    w_nb_r,
                ]
            ),
            W.HBox([W.Label("ratio val/train alerta"), w_ratio]),
        ]
    )

    # ---- Outliers (subsección dentro de Correlación) ----
    o = cfg.get("correlacion_outliers", {})
    w_om = _wb(o.get("manejar_outliers", True))
    w_zs = _wf(o.get("umbral_z_suave", 3.0))
    w_zd = _wf(o.get("umbral_z_duro", 6.0))
    w_al = _wf(o.get("alpha_pesos", 0.5))
    w_wm = _wf(o.get("w_min", 0.2))
    w_rd = _wb(o.get("remover_duro", False))
    ui_out = W.VBox(
        [
            W.HBox([W.Label("Manejar outliers"), w_om, W.Label("remover_duro"), w_rd]),
            W.HBox(
                [
                    W.Label("z_suave"),
                    w_zs,
                    W.Label("z_duro"),
                    w_zd,
                    W.Label("alpha"),
                    w_al,
                    W.Label("w_min"),
                    w_wm,
                ]
            ),
        ]
    )

    # --- Serializer de la pestaña ---
    def collect():
        c2 = {"checks_2d": {"enabled": w_chk_enabled.value}}
        for name, subkey, on, val in rows:
            d = c2["checks_2d"].setdefault(name, {})
            d["enabled"] = on.value
            d[subkey] = float(val.value)
        # diversidad
        tipos_loc = tipos
        mu2, mm2 = {}, {}
        # fila 0 son labels
        for hbox in ui_div.children[1:]:
            t = hbox.children[0].value
            mu2[t] = int(hbox.children[1].value)
            mm2[t] = int(hbox.children[2].value)
        # retorno
        return {
            "checks_2d": c2["checks_2d"],
            "diversidad_minima": {"min_unicos": mu2, "min_muestras": mm2},
            "extrapolacion": {
                "modo_predictores": w_modo_pred.value,
                "tolerancia_pct": float(w_tol_pct.value),
                "modo_2d": w_modo2d.value,
                "tolerancia_hull_pad": float(w_hull_pad.value),
            },
            "modelos": {
                "permitir_1d": bool(w_m1d.value),
                "permitir_2d": bool(w_m2d.value),
                "poly_grado": int(w_pdeg.value),
                "habilitados": {
                    "lineal": bool(w_mod_lineal.value),
                    "polinomico2": bool(w_mod_poly2.value),
                    "log": bool(w_mod_log.value),
                    "potencia": bool(w_mod_pot.value),
                    "exponencial": bool(w_mod_exp.value),
                },
                "umbral_mape_max": float(w_mod_mape.value),
                "usar_loocv": bool(w_mod_loocv.value),
                "loocv_usa_pesos": bool(w_mod_loocv_w.value),
                "ponderaciones_seleccion": {
                    "mape": float(w_mod_w_mape.value),
                    "r2": float(w_mod_w_r2.value),
                    "corr": float(w_mod_w_corr.value),
                    "confianza": float(w_mod_w_conf.value),
                },
            },
            "confianza": {
                "w_r2": float(w_wr2.value),
                "w_mape": float(w_wmp.value),
                "mape_divisor": float(w_div.value),
                "penalizacion_k": {
                    "tipo": "polinomica",
                    "params": {
                        "a5": float(_ka5.value),
                        "a4": float(_ka4.value),
                        "a3": float(_ka3.value),
                        "a2": float(_ka2.value),
                        "a1": float(_ka1.value),
                        "a0": float(_ka0.value),
                    },
                },
                "penalizacion_n": {
                    "tipo": "polinomica",
                    "params": {
                        "b3": float(_b3.value),
                        "b2": float(_b2.value),
                        "b1": float(_b1.value),
                        "b0": float(_b0.value),
                    },
                },
                "penalizaciones_metricas": {
                    nm: {
                        "usar": bool(on.value),
                        "tipo": "polinomica",
                        "params": {
                            "c2": float(c2.value),
                            "c1": float(c1.value),
                            "c0": float(c0.value),
                        },
                    }
                    for (nm, on, c2, c1, c0) in rows_metrics
                },
                "loocv_aporte": {
                    "w": float(w_lo_w_aporte.value),
                    "factor_por_clase": {
                        "robusto": float(w_lo_fr.value),
                        "no_robusto": float(w_lo_fnr.value),
                        "rechazado": float(w_lo_fre.value),
                    },
                },
            },
            "seleccion_modelos": {
                "train": {
                    "mape_max": float(w_tr_m.value),
                    "r2_min": float(w_tr_r.value),
                },
                "pre_filtro": {
                    "mape_max": float(w_pf_m.value),
                    "r2_min": float(w_pf_r.value),
                },
            },
            "loocv": {
                "usar": w_lo_use.value,
                "usar_pesos_outliers": w_lo_use_pesos.value,
                "criterios": {
                    "robusto": {
                        "mape_max": float(w_rb_m.value),
                        "r2_min": float(w_rb_r.value),
                    },
                    "no_robusto": {
                        "mape_max": float(w_nb_m.value),
                        "r2_min": float(w_nb_r.value),
                    },
                },
                "ratio_val_train_alerta": float(w_ratio.value),
            },
            # Outliers (se guarda en raíz porque otras pestañas lo usan)
            "_outliers_embed": {
                "manejar_outliers": w_om.value,
                "umbral_z_suave": float(w_zs.value),
                "umbral_z_duro": float(w_zd.value),
                "alpha_pesos": float(w_al.value),
                "w_min": float(w_wm.value),
                "remover_duro": w_rd.value,
            },
        }

    # UI final
    ui = W.Accordion(
        children=[
            W.VBox([ui_checks]),
            W.VBox([ui_div]),
            W.VBox([ui_ex]),
            W.VBox([ui_modelos]),
            W.VBox([ui_conf, acc_pk, acc_pn, acc_pm, ui_loocv]),
            W.VBox([ui_sel]),
            W.VBox([ui_out]),
        ]
    )
    for i, title in enumerate(
        [
            "Checks 2D",
            "Diversidad mínima",
            "Extrapolación",
            "Modelos",
            "Confianza",
            "Selección/LOOCV",
            "Outliers",
        ]
    ):
        ui.set_title(i, title)

    return ui, collect


def _build_orq(cfg: dict):
    orq = cfg.get("orquestacion", {})
    ent = cfg.get("entorno", {})
    loop = cfg.get("loop", {})
    export_cfg = loop.get("export", {})

    default_dir = export_cfg.get("dir")
    if not default_dir:
        default_dir = export_cfg.get("json", {}).get("dir")
    if not default_dir:
        default_dir = "salidas"

    w_ruta = W.Text(
        value=str(ent.get("ruta_excel", "")), layout=W.Layout(width="600px")
    )
    w_dir_salida = W.Text(value=str(default_dir), layout=W.Layout(width="600px"))

    w_it = W.BoundedIntText(value=int(orq.get("max_iteraciones", 5)), min=1, max=999)
    w_sim = W.Checkbox(
        value=bool(orq.get("ejecutar_similitud", True)),
        description="Ejecutar similitud",
    )
    w_cor = W.Checkbox(
        value=bool(orq.get("ejecutar_correlacion", True)),
        description="Ejecutar correlación",
    )
    w_sin = W.Checkbox(
        value=bool(orq.get("permitir_sin_filtro", False)),
        description="Permitir correlación sin filtro estricto",
    )
    w_minv = W.BoundedIntText(
        value=int(orq.get("min_datos_validos", 5)), min=1, max=10**6
    )
    w_show = W.Checkbox(
        value=bool(orq.get("mostrar_consola", True)), description="Mostrar consola"
    )

    w_rows = W.BoundedIntText(value=int(ent.get("max_rows", 200)), min=10, max=9999)
    w_cols = W.BoundedIntText(value=int(ent.get("max_columns", 120)), min=10, max=9999)

    conf = orq.get("confirmaciones", {})
    w_conf_modo = W.Dropdown(
        options=[
            ("Automático (sin preguntas)", "automatico"),
            ("Mostrar resumen antes de ejecutar", "resumen"),
            ("Preguntar por consola (modo avanzado)", "consola"),
        ],
        value=conf.get("modo", "automatico"),
        description="Modo",
        layout=W.Layout(width="400px"),
    )
    w_conf_cols = W.Checkbox(
        value=bool(conf.get("confirmar_columnas", False)),
        description="Pedir confirmación de columnas seleccionadas",
    )
    w_conf_rutas = W.Checkbox(
        value=bool(conf.get("confirmar_rutas", False)),
        description="Pedir confirmación de rutas (entrada/salida)",
    )

    ui_confirmaciones = W.VBox(
        [
            W.HTML("<b>Confirmaciones</b>"),
            W.HTML(
                "<small>Elegí cómo querés validar antes de ejecutar. En 'Automático' no se pedirá nada;"
                " en 'Resumen' verás un resumen para confirmar; en 'Consola' se intentará preguntar por teclado.</small>"
            ),
            w_conf_modo,
            w_conf_cols,
            w_conf_rutas,
        ]
    )

    ui = W.VBox(
        [
            W.HBox(
                [
                    W.Label("Ruta Excel", layout=W.Layout(width="120px")),
                    w_ruta,
                ]
            ),
            W.HBox(
                [
                    W.Label("Carpeta de salida", layout=W.Layout(width="120px")),
                    w_dir_salida,
                ]
            ),
            W.HBox(
                [
                    W.Label("Máx. iteraciones", layout=W.Layout(width="120px")),
                    w_it,
                ]
            ),
            w_sim,
            w_cor,
            w_sin,
            W.HBox(
                [
                    W.Label("Mín. datos válidos", layout=W.Layout(width="120px")),
                    w_minv,
                ]
            ),
            w_show,
            W.HBox(
                [
                    W.Label("display.max_rows", layout=W.Layout(width="120px")),
                    w_rows,
                ]
            ),
            W.HBox(
                [
                    W.Label("display.max_columns", layout=W.Layout(width="120px")),
                    w_cols,
                ]
            ),
            W.HTML("<hr>"),
            ui_confirmaciones,
        ]
    )

    def collect():
        ent_out = {
            "ruta_excel": str(w_ruta.value),
            "max_rows": int(w_rows.value),
            "max_columns": int(w_cols.value),
        }
        orq_out = {
            "max_iteraciones": int(w_it.value),
            "ejecutar_similitud": bool(w_sim.value),
            "ejecutar_correlacion": bool(w_cor.value),
            "permitir_sin_filtro": bool(w_sin.value),
            "min_datos_validos": int(w_minv.value),
            "mostrar_consola": bool(w_show.value),
            "confirmaciones": {
                "modo": w_conf_modo.value,
                "confirmar_columnas": bool(w_conf_cols.value),
                "confirmar_rutas": bool(w_conf_rutas.value),
            },
        }
        exp_out = {"dir": str(w_dir_salida.value)}
        return ent_out, orq_out, {"export": exp_out}

    return ui, collect


def _build_tabs(cfg: dict):
    out = W.Output()

    tab_orq, orq_collect = _build_orq(cfg)
    loop_export_override: dict[str, str] = {}
    cfg_preview = cfg

    # -------- Excel --------
    e = cfg["excel"]
    w_xl_col_sim = _w_color_picker(e["colores"]["similitud"])
    w_xl_col_cor = _w_color_picker(e["colores"]["correlacion"])
    w_xl_col_comb = _w_color_picker(e["colores"]["combinado"])
    w_xl_col_eval = _w_color_picker(e["colores"]["evaluado"])
    w_xl_cmt_gr = _w_bool(e["comentarios_grandes"])
    w_xl_overwr = _w_bool(e["permitir_sobrescribir"])
    w_xl_freeze = _w_bool(e["congelar_panes"])
    w_xl_dec = _w_int(e["decimales"])

    tab_xl = W.VBox(
        [
            W.HBox(
                [
                    W.Label("Color similitud", layout=W.Layout(width="180px")),
                    w_xl_col_sim,
                ]
            ),
            W.HBox(
                [
                    W.Label("Color correlación", layout=W.Layout(width="180px")),
                    w_xl_col_cor,
                ]
            ),
            W.HBox(
                [
                    W.Label("Color combinado", layout=W.Layout(width="180px")),
                    w_xl_col_comb,
                ]
            ),
            W.HBox(
                [
                    W.Label("Color evaluado", layout=W.Layout(width="180px")),
                    w_xl_col_eval,
                ]
            ),
            W.HBox(
                [
                    W.Label("Comentarios grandes", layout=W.Layout(width="180px")),
                    w_xl_cmt_gr,
                ]
            ),
            W.HBox(
                [
                    W.Label("Permitir sobrescribir", layout=W.Layout(width="180px")),
                    w_xl_overwr,
                ]
            ),
            W.HBox(
                [
                    W.Label("Congelar paneles", layout=W.Layout(width="180px")),
                    w_xl_freeze,
                ]
            ),
            W.HBox(
                [
                    W.Label("Decimales exportación", layout=W.Layout(width="180px")),
                    w_xl_dec,
                ]
            ),
        ]
    )

    # -------- HTML --------
    h = cfg["html"]
    w_html_dec = _w_int(h["decimales"])
    w_html_w = _w_int(h["ancho_px"])
    w_html_h = _w_int(h["alto_px"])

    tab_html = W.VBox(
        [
            W.HBox(
                [W.Label("HTML decimales", layout=W.Layout(width="180px")), w_html_dec]
            ),
            W.HBox(
                [W.Label("HTML ancho (px)", layout=W.Layout(width="180px")), w_html_w]
            ),
            W.HBox(
                [W.Label("HTML alto (px)", layout=W.Layout(width="180px")), w_html_h]
            ),
        ]
    )

    # --- construir pestañas avanzadas ---
    sim_adv_ui, sim_collect = _build_sim_advanced(cfg)
    corr_adv_ui, corr_collect = _build_corr_advanced(cfg)
    loop_adv_ui, loop_collect = _build_loop_advanced(cfg)

    # --- armar el Tab general ---
    tabs = W.Tab(
        children=[
            tab_orq,  # Orquestación
            sim_adv_ui,  # Similitud (única)
            corr_adv_ui,  # Correlación
            loop_adv_ui,  # Loop/Orquestación
            tab_xl,
            tab_html,
        ]
    )
    titles = [
        "Orquestación",
        "Similitud",
        "Correlación",
        "Loop/Orquestación",
        "Excel",
        "HTML",
    ]
    for i, name in enumerate(titles):
        tabs.set_title(i, name)

    # Botones y salida
    btn_save = W.Button(description="Guardar overrides", button_style="")
    btn_run = W.Button(description="Guardar y ejecutar", button_style="primary")

    def _collect():
        nonlocal loop_export_override
        cfg_new = default_config()
        ent_out, orq_out, loop_exp_out = orq_collect()
        loop_export_override = dict(loop_exp_out.get("export", {}))
        cfg_new["entorno"].update(ent_out)
        cfg_new["orquestacion"].update(orq_out)
        if loop_export_override:
            cfg_new.setdefault("loop", {}).setdefault("export", {}).update(
                loop_export_override
            )
        # Excel
        cfg_new["excel"]["colores"]["similitud"] = str(w_xl_col_sim.value)
        cfg_new["excel"]["colores"]["correlacion"] = str(w_xl_col_cor.value)
        cfg_new["excel"]["colores"]["combinado"] = str(w_xl_col_comb.value)
        cfg_new["excel"]["colores"]["evaluado"] = str(w_xl_col_eval.value)
        cfg_new["excel"]["comentarios_grandes"] = bool(w_xl_cmt_gr.value)
        cfg_new["excel"]["permitir_sobrescribir"] = bool(w_xl_overwr.value)
        cfg_new["excel"]["congelar_panes"] = bool(w_xl_freeze.value)
        cfg_new["excel"]["decimales"] = int(w_xl_dec.value)
        # HTML
        cfg_new["html"]["decimales"] = int(w_html_dec.value)
        cfg_new["html"]["ancho_px"] = int(w_html_w.value)
        cfg_new["html"]["alto_px"] = int(w_html_h.value)
        return cfg_new

    def _save_clicked(_):
        nonlocal cfg_preview, loop_export_override
        with out:
            out.clear_output()
            cfg_new = _collect()
            # mezclar Similitud (Avanzado)
            sim_adv, _familias_usadas = sim_collect()
            cfg_new.setdefault("similitud", {}).update(sim_adv)
            # mezclar correlacion avanzada
            corr_adv = corr_collect()
            cfg_new.setdefault("correlacion", {}).update(
                {k: v for k, v in corr_adv.items() if k != "_outliers_embed"}
            )
            modelos_corr = cfg_new.get("correlacion", {}).get("modelos")
            if modelos_corr:
                cfg_new["modelos"] = deepcopy(
                    modelos_corr
                )  # compat con módulos que leen raíz
            if "_outliers_embed" in corr_adv:
                cfg_new.setdefault("correlacion_outliers", {}).update(
                    corr_adv["_outliers_embed"]
                )
            # mezclar Loop/Orquestación avanzada
            loop_adv = loop_collect()
            if loop_export_override:
                loop_adv.setdefault("export", {}).update(loop_export_override)
            cfg_new.setdefault("loop", {}).update(loop_adv)
            # Compat: reflejar extrapolación avanzada en correlacion_outliers
            try:
                ex = cfg_new.get("correlacion", {}).get("extrapolacion", {})
                permitir = (
                    ex.get("modo_predictores", "eliminar") == "permitir_con_tolerancia"
                )
                tol = float(ex.get("tolerancia_pct", 0.0)) if permitir else None
                cfg_new.setdefault("correlacion_outliers", {})[
                    "permitir_extrapolacion"
                ] = permitir
                cfg_new["correlacion_outliers"]["tolerancia_fuera_rango"] = tol
            except Exception:
                pass
            # Compat: reflejar LOOCV avanzado en modelos
            try:
                lo = cfg_new.get("correlacion", {}).get("loocv", {})
                cfg_new.setdefault("modelos", {})["usar_loocv"] = bool(
                    lo.get("usar", cfg_new["modelos"]["usar_loocv"])
                )
                cfg_new["modelos"]["loocv_usa_pesos"] = bool(
                    lo.get("usar_pesos_outliers", cfg_new["modelos"]["loocv_usa_pesos"])
                )
            except Exception:
                pass
            try:
                cfg_new = get_config(cfg_new)
                cfg_preview = cfg_new
                save_overrides(cfg_new)
                snapshot_config(cfg_new)
                print(
                    "✅ Overrides guardados en config_overrides.json y snapshot en ./salidas"
                )
            except Exception as e:
                print("❌ Error de configuración:", e)

    def _run_clicked(_):
        nonlocal cfg_preview
        _save_clicked(None)
        modo = (
            cfg_preview.get("orquestacion", {})
            .get("confirmaciones", {})
            .get("modo", "automatico")
        )
        if modo == "resumen":
            from IPython.display import clear_output, display

            box = W.VBox([])
            resumen = (
                "<b>Resumen antes de ejecutar</b><br>"
                "• Se usará la ruta de Excel indicada y la carpeta de salida seleccionada.<br>"
                "• Se ejecutarán las etapas marcadas (Similitud/Correlación) con los valores del panel.<br>"
                "• No se pedirá escribir nada por teclado durante el proceso.<br>"
            )
            btn_ok = W.Button(
                description="Confirmar y ejecutar", button_style="success"
            )
            btn_no = W.Button(description="Cancelar", button_style="warning")
            out_resumen = W.Output()
            box.children = [W.HTML(resumen), W.HBox([btn_ok, btn_no]), out_resumen]
            display(box)

            def _do_run(_b):
                with out_resumen:
                    clear_output(wait=True)
                    print("\n--- Ejecutando pipeline ---")
                    logs = run_pipeline()
                    if logs:
                        print(logs)
                    print("\n✅ Pipeline finalizado.")

            def _cancel(_b):
                with out_resumen:
                    clear_output(wait=True)
                    print("Ejecución cancelada por el usuario.")

            btn_ok.on_click(_do_run)
            btn_no.on_click(_cancel)
            return

        with out:
            print("\n--- Ejecutando pipeline ---")
            logs = run_pipeline()
            if logs:
                print(logs)
            print("\n✅ Pipeline finalizado.")

    btn_save.on_click(_save_clicked)
    btn_run.on_click(_run_clicked)

    notice = W.HTML(
        "<b>Importante:</b> Por defecto no se pedirán confirmaciones manuales; podés ajustar el comportamiento en la sección Confirmaciones."
    )
    ui = W.VBox([notice, tabs, W.HBox([btn_save, btn_run]), W.HTML("<hr>"), out])
    return ui


def show_ux():
    """Renderiza el panel en el notebook."""
    cfg = load_effective_config()  # defaults + overrides previos si existen
    ui = _build_tabs(cfg)
    display(ui)
