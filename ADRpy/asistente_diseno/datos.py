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


# --------- Normalización numérica (coma decimal, miles, unidades, etc.) --------- #
def to_numeric_locale(s: pd.Series) -> pd.Series:
    """
    Convierte una Serie a float soportando:
    - coma decimal (0,32 -> 0.32)
    - separadores de miles (1.234,56 -> 1234.56 ; 1,234.56 -> 1234.56)
    - espacios/nbspace y unidades sueltas (p.ej. '0,32 m')
    - signo y notación científica
    Cualquier valor imposible -> NaN.
    """
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")

    s2 = s.astype(str).str.strip()
    s2 = s2.str.replace("\u00a0", "", regex=False).str.replace(
        r"[^0-9,\.\-\+\(\)eE]", "", regex=True
    )
    s2 = s2.str.replace(r"^\((.*)\)$", r"-\1", regex=True)

    has_comma = s2.str.contains(",", na=False)
    has_dot = s2.str.contains(r"\.", na=False)
    both = has_comma & has_dot

    last_comma = s2.str.rfind(",")
    last_dot = s2.str.rfind(".")
    comma_as_decimal = both & (last_comma > last_dot)

    mask_comma_decimal = comma_as_decimal | (has_comma & ~has_dot)
    if mask_comma_decimal.any():
        s2.loc[mask_comma_decimal] = (
            s2.loc[mask_comma_decimal]
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
        )

    mask_dot_decimal = has_dot & (~has_comma | (has_comma & ~comma_as_decimal))
    mask_dot_decimal_comma_thousands = mask_dot_decimal & has_comma
    if mask_dot_decimal_comma_thousands.any():
        s2.loc[mask_dot_decimal_comma_thousands] = s2.loc[
            mask_dot_decimal_comma_thousands
        ].str.replace(",", "", regex=False)

    converted = pd.to_numeric(s2, errors="coerce")

    mask_retry = converted.isna() & s2.str.contains(r"[0-9]", na=False)
    if mask_retry.any():
        idx_retry = mask_retry[mask_retry].index
        retry_series = s2.loc[idx_retry]

        # Fallback 1: conservar solo el último separador decimal y normalizar a punto
        retry_last_sep = retry_series.str.replace(r"[.,](?=.*[.,])", "", regex=True)
        retry_last_sep = retry_last_sep.str.replace(",", ".", regex=False)
        fallback_last_sep = pd.to_numeric(retry_last_sep, errors="coerce")
        converted.loc[idx_retry] = converted.loc[idx_retry].combine_first(
            fallback_last_sep
        )

        mask_retry2 = converted.loc[idx_retry].isna()
        if mask_retry2.any():
            idx_retry2 = converted.loc[idx_retry][mask_retry2].index
            retry_plain = retry_series.loc[idx_retry2].str.replace(",", "", regex=False)
            retry_plain = retry_plain.str.replace(".", "", regex=False)
            fallback_plain = pd.to_numeric(retry_plain, errors="coerce")
            converted.loc[idx_retry2] = fallback_plain

    return converted


def normalize_numeric_df(
    df: pd.DataFrame, columns: list[str] | None = None
) -> pd.DataFrame:
    """Aplica to_numeric_locale a todas las columnas indicadas o a las object que parezcan numéricas."""

    if columns is None:
        patt = r"^[\s\(\)\-\+]*[0-9]+([.,][0-9]+)?([eE][\-\+]?[0-9]+)?[\s\)]*$"
        candidates = [
            c
            for c in df.columns
            if df[c].dtype == object
            and df[c].astype(str).str.strip().str.match(patt, na=False).mean() >= 0.7
        ]
    else:
        candidates = [c for c in columns if c in df.columns]

    for c in candidates:
        try:
            df[c] = to_numeric_locale(df[c])
        except Exception:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


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


def cargar_dataset(path_xlsx: str) -> pd.DataFrame:
    """Lee el Excel principal, luego normaliza columnas numéricas usando heurística local-aware."""

    df = pd.read_excel(path_xlsx, sheet_name=0)
    df = normalize_numeric_df(df)
    return df
