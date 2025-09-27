"""
Gestión mínima de datos y utilidades:

- Lectura del Excel desde la PRIMERA hoja (sheet 0).
- Validación mínima de columnas requeridas.
- Conversión numérica segura (sin tocar el DataFrame original).

Aquí NO hay lógica de negocio; sólo utilidades para que los demás módulos
partan de un DataFrame "usable".
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
import numpy as np

from . import config


def leer_excel(
    ruta: Path | str | None = None, hoja: int | str | None = None
) -> pd.DataFrame:
    """
    Lee el Excel del proyecto y devuelve un DataFrame sin modificar.
    - Por defecto usa config.DATA_XLSX y la PRIMERA hoja (config.EXCEL_SHEET = 0).
    - No castea columnas; la conversión se hace columna a columna cuando haga falta.

    Parameters
    ----------
    ruta : Path | str | None
        Ruta del Excel. Si None, usa config.DATA_XLSX.
    hoja : int | str | None
        Hoja a cargar. Si None, usa config.EXCEL_SHEET (0).

    Returns
    -------
    pd.DataFrame
    """
    ruta = Path(ruta) if ruta is not None else config.DATA_XLSX
    hoja = hoja if hoja is not None else config.EXCEL_SHEET
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo Excel: {ruta}")
    df = pd.read_excel(ruta, sheet_name=hoja)
    return df


def columnas_faltantes(df: pd.DataFrame, requeridas: Iterable[str]) -> List[str]:
    """
    Devuelve la lista de nombres requeridos que NO están presentes en el DataFrame.
    No cambia el DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
    requeridas : Iterable[str]

    Returns
    -------
    List[str]
    """
    faltan = [c for c in requeridas if c not in df.columns]
    return faltan


def a_numerico_seguro(serie: pd.Series) -> pd.Series:
    """
    Intenta convertir una serie a numérico.
    - Errores → NaN
    - No muta la serie original (devuelve copia convertida)

    Returns
    -------
    pd.Series
    """
    return pd.to_numeric(serie.copy(), errors="coerce")


# --- NUEVO: detección de columnas numéricas "útiles" ---
def columnas_numericas_utiles(
    df: pd.DataFrame, min_valid: int | None = None
) -> List[str]:
    """
    Devuelve columnas potencialmente útiles para la UI dinámica:
    - Excluye nombres configurados en EXCLUDE_COLS
    - Requiere al menos 'min_valid' valores no nulos
    - Requiere varianza > 0 (evita columnas constantes)
    """
    if min_valid is None:
        min_valid = int(getattr(config, "MIN_VALID_NUMERIC", 5))
    excl = set(getattr(config, "EXCLUDE_COLS", set()))
    out: List[str] = []
    for c in df.columns:
        if c in excl:
            continue
        s = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        n_valid = int(s.notna().sum())
        # varianza sobre valores no nulos
        arr = s.dropna().to_numpy(dtype=float)
        var = float(np.var(arr, ddof=1)) if arr.size > 1 else 0.0
        if n_valid >= int(min_valid) and var > 0.0:
            out.append(c)
    return out
