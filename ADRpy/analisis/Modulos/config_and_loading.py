from __future__ import annotations
import sys
import unicodedata
from copy import deepcopy
import json
import os
from datetime import datetime

try:
    import pandas as pd
except Exception:
    pd = None  # permitir correr sin pandas en validaciones


# === CONFIG centralizado con defaults (replica comportamiento actual) ===
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
            "permitir_sin_filtro": False,  # política estricta por defecto
            "min_datos_validos": 5,
            "mostrar_consola": True,
        },
        "similitud": {
            "umbral_pct_diferencia": 0.20,  # 20%
            "min_familias": 3,
            "excepcion_min_familias": {"min_familias": 2, "min_parametros": 6},
            "k_min": 3,
            "k_max": 10,
            "peso_confianza_similitud": 0.7,
            "peso_confianza_cv": 0.3,
            "verbosidad": 0,  # 0=normal, 1=detallado
        },
        "correlacion_outliers": {
            "manejar_outliers": True,
            "umbral_z_suave": 3.0,
            "umbral_z_duro": 6.0,
            "alpha_pesos": 0.5,
            "w_min": 0.2,
            "remover_duro": False,
            "permitir_extrapolacion": False,  # sigue desactivado
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
            "umbral_mape_max": 0.35,
            "ponderaciones_seleccion": {
                "mape": 0.5,
                "r2": 0.2,
                "corr": 0.2,
                "confianza": 0.1,
            },
            "usar_loocv": True,
            "loocv_usa_pesos": False,  # TODO: activar cuando implementemos pesos en LOOCV
        },
        "excel": {
            "colores": {
                "similitud": "#FFF59D",  # amarillo claro
                "correlacion": "#A5D6A7",  # verde claro
                "combinado": "#90CAF9",  # azul claro
                "evaluado": "#FFCC80",  # naranja claro
            },
            "comentarios_grandes": True,
            "permitir_sobrescribir": False,
            "congelar_panes": True,
            "decimales": 2,
        },
        "html": {
            "decimales": 3,
            "ancho_px": 1200,
            "alto_px": 600,
        },
    }


CONFIG: dict = default_config()


def deep_update(base: dict, updates: dict) -> dict:
    out = deepcopy(base)
    for k, v in (updates or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def validate_config(cfg: dict) -> None:
    # Validaciones esenciales y simples (evitar valores imposibles)
    if cfg["orquestacion"]["min_datos_validos"] < 2:
        raise ValueError("min_datos_validos debe ser >= 2")
    if not (0 <= cfg["similitud"]["umbral_pct_diferencia"] <= 1):
        raise ValueError("umbral_pct_diferencia debe estar entre 0 y 1 (proporción).")
    wmin = cfg["correlacion_outliers"]["w_min"]
    if wmin <= 0 or wmin > 1:
        raise ValueError("w_min debe estar en (0,1].")
    # Política de extrapolación: si está desactivada, tolerancia debe ser None
    if cfg["correlacion_outliers"]["permitir_extrapolacion"] is False:
        cfg["correlacion_outliers"]["tolerancia_fuera_rango"] = None


def get_config(overrides: dict | None = None) -> dict:
    global CONFIG
    merged = deep_update(CONFIG, overrides or {})
    validate_config(merged)
    return merged


def load_config_file(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cfg = get_config(data)  # merge + validate
    return cfg


def save_config_snapshot(
    cfg: dict, out_dir: str, prefix: str = "config_snapshot"
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"{prefix}_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    return path


def apply_env_display_limits(cfg: dict) -> None:
    if pd is None:
        return
    try:
        pd.set_option("display.max_rows", cfg["entorno"]["max_rows"])
        pd.set_option("display.max_columns", cfg["entorno"]["max_columns"])
    except Exception:
        pass


import tkinter as tk
from tkinter import simpledialog, messagebox


def configurar_entorno(max_rows=20, max_columns=10):
    """
    Configura el entorno para mostrar más datos en la consola.
    :param max_rows: Número máximo de filas para mostrar en consola.
    :param max_columns: Número máximo de columnas para mostrar en consola.
    """
    if pd is None:
        return
    pd.set_option("display.max_rows", max_rows)
    pd.set_option("display.max_columns", max_columns)


def cargar_datos(ruta_archivo=None):
    """
    Carga los datos desde un archivo Excel y realiza validaciones.
    Devuelve el DataFrame cargado y la ruta utilizada.
    """
    # Detectar si se está ejecutando en modo debug
    modo_debug = "--debug_mode" in sys.argv

    if pd is None:
        raise ImportError("Se requiere 'pandas' para cargar archivos Excel.")

    if not ruta_archivo:
        if modo_debug:
            ruta_archivo = r"C:\Users\delpi\OneDrive\Tesis\ADRpy-VTOL\ADRpy\analisis\Data\Datos_aeronaves.xlsx"
            print(f"DEBUG MODE ACTIVADO: usando ruta predeterminada: {ruta_archivo}")
        else:
            ruta_archivo = (
                input(
                    r"Ingrese la ruta del archivo Excel original (o presione Enter para usar la predeterminada): "
                ).strip()
                or r"C:\Users\delpi\OneDrive\Tesis\ADRpy-VTOL\ADRpy\analisis\Data\Datos_aeronaves.xlsx"
            )

    print(f"DEBUG: ruta_archivo antes de validar: '{ruta_archivo}'")

    # Validar el formato del archivo
    if not ruta_archivo.endswith((".xlsx", ".xlsm")):
        raise ValueError(
            "El archivo debe estar en formato .xlsx o .xlsm compatible con openpyxl."
        )

    # Mostrar mensaje de carga
    print(f"=== Cargando datos desde el archivo: {ruta_archivo} ===")

    try:
        # Cargar el archivo con encabezado e índice configurados
        df = pd.read_excel(ruta_archivo, header=0, index_col=0)

        # Validaciones adicionales
        if df.empty:
            raise ValueError("El archivo cargado está vacío.")

        # Manejar índices nulos
        if df.index.isnull().any():
            print(
                "Advertencia: Índices nulos encontrados. Reemplazando por 'indice_desconocido'."
            )
            df.index = df.index.fillna("indice_desconocido")

        # Manejar columnas nulas
        if df.columns.isnull().any():
            print(
                "Advertencia: Columnas nulas encontradas. Reemplazando por 'columna_desconocida'."
            )
            df.columns = df.columns.fillna("columna_desconocida")

        # Mostrar información básica del DataFrame cargado
        print("\n=== Resumen inicial del DataFrame cargado ===")
        print(df.info())
        # print("\n=== Vista previa de índices y columnas ===")
        # print(f"Primeros índices: {df.index.tolist()[:10]}")
        # print(f"Primeras columnas: {df.columns.tolist()[:10]}")

        return df, ruta_archivo
    except FileNotFoundError:
        raise ValueError("Error: Archivo no encontrado.")
    except Exception as e:
        raise ValueError(f"Error al cargar el archivo: {e}")


def normalizar_encabezados(df):
    """
    Normaliza los encabezados del DataFrame:
    - Elimina espacios al inicio y al final.
    - Convierte a minúsculas.
    - Elimina caracteres especiales o no ASCII.
    :param df: DataFrame a normalizar.
    :return: DataFrame con encabezados normalizados.
    """

    def normalizar(texto):
        if isinstance(texto, str):
            # Eliminar caracteres no ASCII y convertir a minúsculas
            texto = "".join(
                c
                for c in unicodedata.normalize("NFD", texto)
                if unicodedata.category(c) != "Mn"
            )
            return texto.strip().lower()  # Eliminar espacios y convertir a minúsculas
        return texto

    # Normalizar columnas e índices
    df.columns = [normalizar(col) for col in df.columns]
    df.index = [normalizar(idx) for idx in df.index]
    return df
