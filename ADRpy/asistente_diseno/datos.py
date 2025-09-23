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
