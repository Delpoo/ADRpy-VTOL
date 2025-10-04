from __future__ import annotations
import sys, os, io, contextlib, json, runpy
from pathlib import Path
from copy import deepcopy
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# -------------------- CONFIG CENTRAL --------------------
def default_config() -> dict:
    return {
        "entorno": {
            "debug_mode": False,
            "seed": 42,
            "ruta_excel": "Datos_aeronaves.xlsx",
            "max_rows": 200,
            "max_columns": 120,
        },
        "orquestacion": {
            "max_iteraciones": 5,
            "ejecutar_similitud": True,
            "ejecutar_correlacion": True,
            "permitir_sin_filtro": False,
            "min_datos_validos": 5,
            "mostrar_consola": True,
        },
        "similitud": {
            "umbral_pct_diferencia": 0.20,
            "min_familias": 3,
            "excepcion_min_familias": {"min_familias": 2, "min_parametros": 6},
            "k_min": 3,
            "k_max": 10,
            "peso_confianza_similitud": 0.7,
            "peso_confianza_cv": 0.3,
            "verbosidad": 0,
            # NUEVO: familias configurables (por defecto, las actuales)
            "familias": {
                "fisica": [
                    "Peso máximo al despegue (MTOW)",
                    "Peso Vacio (MTOW - payload)",
                    "Payload",
                ],
                "geometrica": [
                    "Área del ala",
                    "Envergadura",
                    "Longitud del fuselaje",
                    "Ancho del fuselaje",
                    "Relación de aspecto del ala",
                ],
                "prestacional": [
                    "Potencia específica (P/W)",
                    "Autonomía de la aeronave",
                    "Alcance de la aeronave",
                    "Velocidad a la que se realiza el crucero (m/s TAS)",
                    "Velocidad máxima (m/s IAS)",
                    "Rango de comunicación",
                    "Potencia HP",
                    "Potencia Watts",
                ],
            },
            "familias_usadas": [
                "fisica",
                "geometrica",
                "prestacional",
            ],  # subset permitido
            # NUEVO: función de similitud por parámetro (x=diferencia %)
            "funcion_similitud": {
                "tipo": "polinomica",
                "coef": {
                    "a2": -0.002,
                    "a1": -0.01,
                    "a0": 1.0,
                },  # -0.002 x^2 - 0.01 x + 1
                "dominio_max_pct": 20.0,
            },
            # NUEVO: selección de vecinos
            "vecinos": {
                "modo": "todos",  # todos | top_k | k_en_rango
                "top_k": 10,  # sólo si modo=top_k
                "enforce_k_min": True,  # si hay <k_min, falla la imputación
            },
            # NUEVO: confianza por datos en similitud
            "confianza": {
                "cv_ref": 0.5,  # fue fijo; ahora knob
                "penalizacion_k": {  # misma idea que correlacion.confianza.penalizacion_k
                    "tipo": "polinomica",
                    "params": {
                        "a5": 0.00002281,
                        "a4": -0.00024,
                        "a3": -0.0036,
                        "a2": 0.046,
                        "a1": 0.0095,
                        "a0": 0.024,
                    },
                },
            },
            # NUEVO: umbral por familia (opcional, si no se usa cae al global umbral_pct_diferencia)
            "umbral_pct_por_familia": {
                "fisica": None,
                "geometrica": None,
                "prestacional": None,
            },
            # NUEVO: outliers en vecinos de similitud
            "outliers": {
                "usar": False,
                "umbral_z_suave": 3.0,
                "umbral_z_duro": 6.0,
                "alpha_pesos": 0.5,
                "w_min": 0.2,
                "remover_duro": False,
            },
        },
        # Sub-sección Outliers (compartida por correlación ahora y similitud mañana)
        "correlacion_outliers": {
            "manejar_outliers": True,
            "umbral_z_suave": 3.0,
            "umbral_z_duro": 6.0,
            "alpha_pesos": 0.5,
            "w_min": 0.2,
            "remover_duro": False,
            "permitir_extrapolacion": False,  # política general (desactivada)
            "tolerancia_fuera_rango": None,  # None => 0 %
        },
        "modelos": {
            "habilitados": {
                "lineal": True,
                "polinomico2": True,
                "log": True,
                "potencia": True,
                "exponencial": True,
            },
            "umbral_mape_max": 7.5,  # % (coherente con tu motor)
            "ponderaciones_seleccion": {
                "mape": 0.5,
                "r2": 0.2,
                "corr": 0.2,
                "confianza": 0.1,
            },
            "usar_loocv": True,
            "loocv_usa_pesos": False,  # ahora lo implementamos en el motor
        },
        "excel": {
            "colores": {
                "similitud": "#FFF59D",
                "correlacion": "#A5D6A7",
                "combinado": "#90CAF9",
                "evaluado": "#FFCC80",
            },
            "comentarios_grandes": True,
            "permitir_sobrescribir": False,
            "congelar_panes": True,
            "decimales": 2,
        },
        "html": {"decimales": 3, "ancho_px": 1200, "alto_px": 600},
        # -------------------- NUEVO: CORRELACIÓN (Avanzado) --------------------
        "correlacion": {
            "checks_2d": {
                "enabled": True,
                "pearson": {"enabled": True, "abs_r_max": 0.90},
                "vif": {"enabled": True, "max": 10.0},
                "pc2": {"enabled": True, "ratio_min": 0.03},
                "rank": {"enabled": True, "min": 2},
                "cond": {"enabled": True, "max": 1e5},
                "coverage_unique_pair": {"enabled": True, "ratio_min": 0.60},
                "coverage_hull": {"enabled": True, "ratio_min": 0.15},
                "coverage_ellipse": {"enabled": True, "ratio_min": 0.10},
                "n_per_param": {"enabled": True, "linear2_min": 8, "poly2_min": 10},
                "agresivo": {"enabled": True, "abs_r_min": 0.95},
            },
            "diversidad_minima": {
                "min_unicos": {
                    "exp-1": 5,
                    "log-1": 5,
                    "pot-1": 5,
                    "linear-1": 5,
                    "poly-1": 7,
                    "linear-2": 8,
                    "poly-2": 10,
                },
                "min_muestras": {
                    "exp-1": 6,
                    "log-1": 6,
                    "pot-1": 6,
                    "linear-1": 6,
                    "poly-1": 10,
                    "linear-2": 10,
                    "poly-2": 12,
                },
            },
            "extrapolacion": {
                "modo_predictores": "eliminar",  # eliminar | permitir_con_tolerancia
                "tolerancia_pct": 0.0,  # 0-1 del rango (min,max)
                "modo_2d": "marginal",  # marginal | convex_hull
                "tolerancia_hull_pad": 0.0,  # padding (0-1) sobre el hull
            },
            "modelos": {"permitir_1d": True, "permitir_2d": True, "poly_grado": 2},
            "constante": {
                "habilitar": True,
                "respaldo_min": 8,
                "porcentaje_min_total_validos": 0.5,
            },
            "pesos_predictores": {
                "metodo": "abs_coef"  # abs_coef | abs_coef_normalizado | var_importance
            },
            "confianza": {
                "w_r2": 0.5,
                "w_mape": 0.5,
                "mape_divisor": 15.0,
                "penalizacion_k": {
                    "tipo": "polinomica",
                    "params": {
                        "a5": 0.00002281,
                        "a4": -0.00024,
                        "a3": -0.0036,
                        "a2": 0.046,
                        "a1": 0.0095,
                        "a0": 0.024,
                    },
                },
                # NUEVO: penalización por N (número de muestras de entrenamiento)
                "penalizacion_n": {
                    "tipo": "polinomica",
                    "params": {"b3": 0.0, "b2": 0.0025, "b1": 0.02, "b0": 0.10},
                },
                # NUEVO: penalizaciones por métricas 2D (todas opcionales, multiplicativas)
                "penalizaciones_metricas": {
                    "pearson_abs": {
                        "usar": True,
                        "tipo": "polinomica",
                        "params": {"c2": -1.2, "c1": 1.2, "c0": 0.2},
                    },
                    "vif": {
                        "usar": True,
                        "tipo": "polinomica",
                        "params": {"c2": -0.02, "c1": -0.10, "c0": 1.20},
                    },
                    "cond": {
                        "usar": True,
                        "tipo": "polinomica",
                        "params": {"c2": -1e-10, "c1": -1e-5, "c0": 1.0},
                    },
                    "pc2_ratio": {
                        "usar": True,
                        "tipo": "polinomica",
                        "params": {"c2": 2.0, "c1": 0.0, "c0": 0.0},
                    },
                    "coverage_unique_pair": {
                        "usar": True,
                        "tipo": "polinomica",
                        "params": {"c2": 0.0, "c1": 0.8, "c0": 0.2},
                    },
                    "coverage_hull": {
                        "usar": True,
                        "tipo": "polinomica",
                        "params": {"c2": 0.0, "c1": 0.8, "c0": 0.2},
                    },
                    "coverage_ellipse": {
                        "usar": True,
                        "tipo": "polinomica",
                        "params": {"c2": 0.0, "c1": 0.8, "c0": 0.2},
                    },
                },
                # NUEVO: aporte LOOCV a la confianza
                "loocv_aporte": {
                    "w": 0.20,
                    "factor_por_clase": {
                        "robusto": 1.00,
                        "no_robusto": 0.85,
                        "rechazado": 0.60,
                    },
                },
            },
            "loocv": {
                "usar": True,
                "usar_pesos_outliers": False,
                "criterios": {
                    "robusto": {"mape_max": 7.5, "r2_min": 0.6},
                    "no_robusto": {"mape_max": 12.5, "r2_min": 0.45},
                },
                "ratio_val_train_alerta": 5.0,
            },
            "seleccion_modelos": {
                "train": {"mape_max": 7.5, "r2_min": 0.6},
                "pre_filtro": {"mape_max": 18.0, "r2_min": 0.4},
            },
        },
        # --- NUEVO bloque para el loop/orquestación avanzada ---
        "loop": {
            "orden": ["similitud", "correlacion"],
            "combinacion": {
                "metodo": "promedio_ponderado",  # promedio_ponderado | mejor_confianza | prioridad_correlacion | prioridad_similitud
                "conf_min": 0.0,  # si ambos < conf_min, no imputar
            },
            "stop": {"min_nuevas_por_iter": 1, "sin_mejora_consecutivas": 1},
            "export": {
                "json": {
                    "enabled": True,
                    "dir": "Results",
                    "fname": "modelos_completos_por_celda.json",
                },
                "html_df_base": {"mostrar": True},
            },
        },
    }


def _deep_update(base: dict, updates: dict) -> dict:
    out = deepcopy(base)
    for k, v in (updates or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def get_config(overrides: dict | None = None) -> dict:
    cfg = _deep_update(default_config(), overrides or {})
    # Validaciones mínimas:
    if cfg["orquestacion"]["min_datos_validos"] < 2:
        raise ValueError("min_datos_validos debe ser >= 2")
    if not (0 <= cfg["similitud"]["umbral_pct_diferencia"] <= 1):
        raise ValueError("umbral_pct_diferencia debe estar entre 0 y 1.")
    if not (0 < cfg["correlacion_outliers"]["w_min"] <= 1):
        raise ValueError("w_min debe estar en (0,1].")
    if cfg["correlacion_outliers"]["permitir_extrapolacion"] is False:
        cfg["correlacion_outliers"]["tolerancia_fuera_rango"] = None
    # Validaciones correlación (rangos básicos)
    c = cfg["correlacion"]
    if c["modelos"]["poly_grado"] < 1:
        c["modelos"]["poly_grado"] = 1
    # Validaciones nuevas de similitud
    try:
        cvs = cfg["similitud"]["confianza"]["cv_ref"]
        if cvs <= 0:
            cfg["similitud"]["confianza"]["cv_ref"] = 0.5
    except Exception:
        # Asegurar valor por defecto si falta la clave completa
        cfg.setdefault("similitud", {}).setdefault("confianza", {})["cv_ref"] = 0.5
    try:
        if cfg["similitud"]["vecinos"]["modo"] not in ("todos", "top_k", "k_en_rango"):
            cfg["similitud"]["vecinos"]["modo"] = "todos"
    except Exception:
        cfg.setdefault("similitud", {}).setdefault("vecinos", {})["modo"] = "todos"
    # Validaciones nuevas de correlacion.confianza
    try:
        conf = cfg["correlacion"]["confianza"]
        la = conf.setdefault("loocv_aporte", {})
        w = float(la.get("w", 0.2))
        la["w"] = min(1.0, max(0.0, w))
        fac = la.setdefault("factor_por_clase", {})
        fac.setdefault("robusto", 1.0)
        fac.setdefault("no_robusto", 0.85)
        fac.setdefault("rechazado", 0.6)
    except Exception:
        cfg.setdefault("correlacion", {}).setdefault("confianza", {}).setdefault(
            "loocv_aporte",
            {
                "w": 0.2,
                "factor_por_clase": {
                    "robusto": 1.0,
                    "no_robusto": 0.85,
                    "rechazado": 0.6,
                },
            },
        )
    return cfg


# Helpers de carga/guardado/snapshot
def load_effective_config(
    overrides_path: str | os.PathLike = "config_overrides.json",
) -> dict:
    cfg = default_config()
    p = PROJECT_ROOT / Path(overrides_path)
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                raw = json.load(f)
            # Normalizar _outliers_embed -> correlacion_outliers (compatibilidad de panel avanzado)
            try:
                corr_raw = raw.get("correlacion", {}) if isinstance(raw, dict) else {}
                out_embed = corr_raw.get("_outliers_embed")
                if isinstance(out_embed, dict):
                    raw.setdefault("correlacion_outliers", {}).update(out_embed)
                    # Es opcional eliminar la clave interna; la ignoramos al construir cfg
            except Exception:
                pass
            cfg = get_config(raw)
        except Exception:
            cfg = get_config()
    else:
        cfg = get_config()
    # Opciones de display pandas (si aplica)
    try:
        import pandas as pd

        pd.set_option("display.max_rows", cfg["entorno"]["max_rows"])
        pd.set_option("display.max_columns", cfg["entorno"]["max_columns"])
    except Exception:
        pass
    return cfg


def save_overrides(
    cfg: dict, overrides_path: str | os.PathLike = "config_overrides.json"
) -> None:
    # Normalizar _outliers_embed -> correlacion_outliers antes de persistir
    try:
        corr = cfg.get("correlacion", {}) if isinstance(cfg, dict) else {}
        out_embed = corr.get("_outliers_embed")
        if isinstance(out_embed, dict):
            cfg.setdefault("correlacion_outliers", {}).update(out_embed)
            # No guardar la clave interna en el archivo
            try:
                del corr["_outliers_embed"]
            except Exception:
                pass
    except Exception:
        pass
    p = PROJECT_ROOT / Path(overrides_path)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def snapshot_config(
    cfg: dict, out_dir: str | os.PathLike = "salidas", prefix: str = "config_usada"
) -> str:
    out = PROJECT_ROOT / Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = out / f"{prefix}_{ts}.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    return str(p)


# Ejecución del pipeline con captura de logs
def _import_pipeline():
    try:
        from Modulos.imputation_loop import ejecutar_pipeline

        return ("loop", ejecutar_pipeline)
    except Exception:
        try:
            from imputation_loop import ejecutar_pipeline

            return ("loop", ejecutar_pipeline)
        except Exception:
            return ("main", None)


pipeline_mode, ejecutar_pipeline = _import_pipeline()


def run_pipeline(cfg: dict | None = None) -> str:
    if cfg is None:
        cfg = load_effective_config()
    cfg = get_config(cfg)
    save_overrides(cfg)
    snapshot_config(cfg)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            # Re-importar en cada ejecución para evitar referencias obsoletas
            _mode, _entry = _import_pipeline()
            # Preferimos SIEMPRE el loop del proyecto
            if _entry is not None:
                try:
                    _entry(cfg)  # pasar cfg para asegurar uso de overrides
                except TypeError:
                    # Compatibilidad con firma sin cfg
                    _entry()
            else:
                # Fallback extremo (no recomendado)
                print("[WARN] ejecutar_pipeline no disponible; intentando main.py")
                # Sanitizar argv en entornos Jupyter para evitar argumentos desconocidos
                _argv_backup = sys.argv[:]
                try:
                    sys.argv = [str(PROJECT_ROOT / "main.py")]  # limpiar args
                    runpy.run_path(str(PROJECT_ROOT / "main.py"), run_name="__main__")
                finally:
                    sys.argv = _argv_backup
        except SystemExit:
            pass
        except Exception as e:
            print(f"[ERROR] {e}")
    return buf.getvalue()
