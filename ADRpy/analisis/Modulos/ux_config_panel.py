from __future__ import annotations
import json
import re
from typing import Any, Dict

# Import relative to this package
from .config_and_loading import default_config, deep_update, validate_config

HEX_COLOR_RE = re.compile(r"^#([0-9a-fA-F]{6})$")


def _ask_bool(prompt: str, default: bool) -> bool:
    val = input(f"{prompt} [{'y' if default else 'n'}]: ").strip().lower()
    if val == "":
        return default
    if val in ("y", "yes", "s", "si", "sí", "true", "1"):  # type: ignore
        return True
    if val in ("n", "no", "false", "0"):  # type: ignore
        return False
    return default


essential_error = ""


def _ask_int(
    prompt: str, default: int, min_val: int | None = None, max_val: int | None = None
) -> int:
    raw = input(f"{prompt} [{default}]: ").strip()
    if raw == "":
        return default
    try:
        v = int(raw)
        if min_val is not None and v < min_val:
            return default
        if max_val is not None and v > max_val:
            return default
        return v
    except Exception:
        return default


def _ask_float(
    prompt: str,
    default: float,
    min_val: float | None = None,
    max_val: float | None = None,
) -> float:
    raw = input(f"{prompt} [{default}]: ").strip()
    if raw == "":
        return default
    try:
        v = float(raw)
        if min_val is not None and v < min_val:
            return default
        if max_val is not None and v > max_val:
            return default
        return v
    except Exception:
        return default


def _ask_color(prompt: str, default_hex: str) -> str:
    raw = input(f"{prompt} [{default_hex}]: ").strip()
    if raw == "":
        return default_hex
    return raw if HEX_COLOR_RE.match(raw) else default_hex


# ---- Secciones de edición ----


def _edit_similitud(cfg: Dict[str, Any]) -> Dict[str, Any]:
    s = cfg["similitud"].copy()
    s["umbral_pct_diferencia"] = _ask_float(
        "Umbral % diferencia (0-1)", s["umbral_pct_diferencia"], 0.0, 1.0
    )
    s["min_familias"] = _ask_int("Mínimo de familias", s["min_familias"], 1, 5)
    s["excepcion_min_familias"]["min_familias"] = _ask_int(
        "Excepción familias (min)", s["excepcion_min_familias"]["min_familias"], 1, 5
    )
    s["excepcion_min_familias"]["min_parametros"] = _ask_int(
        "Excepción parámetros (min)",
        s["excepcion_min_familias"]["min_parametros"],
        1,
        20,
    )
    s["k_min"] = _ask_int("k mínimo", s["k_min"], 1, 50)
    s["k_max"] = _ask_int("k máximo", s["k_max"], 1, 200)
    s["peso_confianza_similitud"] = _ask_float(
        "Peso confianza similitud", s["peso_confianza_similitud"], 0.0, 1.0
    )
    s["peso_confianza_cv"] = _ask_float(
        "Peso confianza CV", s["peso_confianza_cv"], 0.0, 1.0
    )
    s["verbosidad"] = _ask_int(
        "Verbosidad (0 normal, 1 detallado)", s["verbosidad"], 0, 1
    )
    return {"similitud": s}


def _edit_correlacion_outliers(cfg: Dict[str, Any]) -> Dict[str, Any]:
    c = cfg["correlacion_outliers"].copy()
    c["manejar_outliers"] = _ask_bool("Manejar outliers (pesos)", c["manejar_outliers"])
    c["umbral_z_suave"] = _ask_float("Umbral z suave", c["umbral_z_suave"], 0.0, 10.0)
    c["umbral_z_duro"] = _ask_float("Umbral z duro", c["umbral_z_duro"], 0.0, 20.0)
    c["alpha_pesos"] = _ask_float("Alpha de pesos", c["alpha_pesos"], 0.0, 10.0)
    c["w_min"] = _ask_float("Peso mínimo w_min", c["w_min"], 0.01, 1.0)
    c["remover_duro"] = _ask_bool("Remover outliers duros", c["remover_duro"])
    c["permitir_extrapolacion"] = _ask_bool(
        "Permitir extrapolación (NO recomendado)", c["permitir_extrapolacion"]
    )
    if c["permitir_extrapolacion"]:
        c["tolerancia_fuera_rango"] = _ask_float(
            "Tolerancia fuera de rango (0-1)", 0.15, 0.0, 1.0
        )
    else:
        c["tolerancia_fuera_rango"] = None
    return {"correlacion_outliers": c}


def _edit_modelos(cfg: Dict[str, Any]) -> Dict[str, Any]:
    m = cfg["modelos"].copy()
    for k in list(m["habilitados"].keys()):
        m["habilitados"][k] = _ask_bool(f"Habilitar modelo {k}", m["habilitados"][k])
    m["umbral_mape_max"] = _ask_float(
        "Umbral MAPE máximo", m["umbral_mape_max"], 0.0, 5.0
    )
    m["ponderaciones_seleccion"]["mape"] = _ask_float(
        "Peso selección MAPE", m["ponderaciones_seleccion"]["mape"], 0.0, 1.0
    )
    m["ponderaciones_seleccion"]["r2"] = _ask_float(
        "Peso selección R2", m["ponderaciones_seleccion"]["r2"], 0.0, 1.0
    )
    m["ponderaciones_seleccion"]["corr"] = _ask_float(
        "Peso selección Corr", m["ponderaciones_seleccion"]["corr"], 0.0, 1.0
    )
    m["ponderaciones_seleccion"]["confianza"] = _ask_float(
        "Peso selección Confianza", m["ponderaciones_seleccion"]["confianza"], 0.0, 1.0
    )
    m["usar_loocv"] = _ask_bool("Usar LOOCV", m["usar_loocv"])
    m["loocv_usa_pesos"] = _ask_bool(
        "LOOCV usa pesos (cuando esté implementado)", m["loocv_usa_pesos"]
    )
    return {"modelos": m}


def _edit_orquestacion(cfg: Dict[str, Any]) -> Dict[str, Any]:
    o = cfg["orquestacion"].copy()
    o["max_iteraciones"] = _ask_int("Máx. iteraciones", o["max_iteraciones"], 1, 50)
    o["ejecutar_similitud"] = _ask_bool("Ejecutar similitud", o["ejecutar_similitud"])
    o["ejecutar_correlacion"] = _ask_bool(
        "Ejecutar correlación", o["ejecutar_correlacion"]
    )
    o["permitir_sin_filtro"] = _ask_bool(
        "Permitir correlación sin filtro estricto", o["permitir_sin_filtro"]
    )
    o["min_datos_validos"] = _ask_int(
        "Mínimo de datos válidos", o["min_datos_validos"], 2, 100
    )
    o["mostrar_consola"] = _ask_bool(
        "Mostrar mensajes en consola", o["mostrar_consola"]
    )
    return {"orquestacion": o}


def _edit_excel(cfg: Dict[str, Any]) -> Dict[str, Any]:
    e = cfg["excel"].copy()
    e["colores"]["similitud"] = _ask_color("Color similitud", e["colores"]["similitud"])
    e["colores"]["correlacion"] = _ask_color(
        "Color correlación", e["colores"]["correlacion"]
    )
    e["colores"]["combinado"] = _ask_color("Color combinado", e["colores"]["combinado"])
    e["colores"]["evaluado"] = _ask_color("Color evaluado", e["colores"]["evaluado"])
    e["comentarios_grandes"] = _ask_bool(
        "Comentarios grandes", e["comentarios_grandes"]
    )
    e["permitir_sobrescribir"] = _ask_bool(
        "Permitir sobrescribir celdas con valor", e["permitir_sobrescribir"]
    )
    e["congelar_panes"] = _ask_bool("Congelar paneles en Excel", e["congelar_panes"])
    e["decimales"] = _ask_int("Decimales al exportar", e["decimales"], 0, 6)
    return {"excel": e}


def _edit_html(cfg: Dict[str, Any]) -> Dict[str, Any]:
    h = cfg["html"].copy()
    h["decimales"] = _ask_int("HTML: decimales", h["decimales"], 0, 6)
    h["ancho_px"] = _ask_int("HTML: ancho px", h["ancho_px"], 400, 4000)
    h["alto_px"] = _ask_int("HTML: alto px", h["alto_px"], 300, 4000)
    return {"html": h}


def open_config_panel(cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    base = cfg or default_config()
    overrides: Dict[str, Any] = {}

    print(
        "\n=== ADRpy • Panel de Configuración ===\n(Enter mantiene el valor por defecto)\n"
    )

    # Edición por secciones:
    if _ask_bool("Editar ORQUESTACIÓN", True):
        overrides = deep_update(overrides, _edit_orquestacion(base))
    if _ask_bool("Editar SIMILITUD", True):
        overrides = deep_update(overrides, _edit_similitud(base))
    # Deshabilitado para evitar duplicación con el panel de Notebook (Correlación/Outliers)
    if _ask_bool(
        "Editar CORRELACIÓN / OUTLIERS (deshabilitado, usar panel Notebook)", False
    ):
        print(
            "[Info] Edición de 'Correlación/Outliers' deshabilitada en CLI. Use el panel de Notebook: 'Correlación (Avanzado)'."
        )
    if _ask_bool("Editar MODELOS", True):
        overrides = deep_update(overrides, _edit_modelos(base))
    if _ask_bool("Editar EXCEL", True):
        overrides = deep_update(overrides, _edit_excel(base))
    if _ask_bool("Editar HTML", False):
        overrides = deep_update(overrides, _edit_html(base))

    # Validar y devolver mergeado
    merged = deep_update(base, overrides)
    validate_config(merged)

    # Resumen y opción de guardado
    print("\nResumen final de configuración (merge con defaults):")
    print(json.dumps(merged, indent=2, ensure_ascii=False))
    if _ask_bool("¿Guardar overrides en config_overrides.json?", True):
        with open("config_overrides.json", "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        print("Overrides guardados en config_overrides.json")

    return merged


def run_standalone(path_out: str = "config_overrides.json") -> None:
    base = default_config()
    merged = open_config_panel(base)
    # open_config_panel ya pregunta por guardar; forzamos guardado si no existe
    try:
        with open(path_out, "x", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
            print(f"Archivo creado: {path_out}")
    except FileExistsError:
        # Ya guardado
        pass


if __name__ == "__main__":
    run_standalone()
