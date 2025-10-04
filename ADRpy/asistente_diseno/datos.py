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


# ------------------------------------------------------------
# Alcance de datos (Ámbito): helpers para filtrar según "state"
# ------------------------------------------------------------
from typing import Any, Dict, Tuple as _Tuple


def _fullmask(df: pd.DataFrame) -> pd.Series:
    return pd.Series(True, index=df.index, dtype=bool)


def _safe_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def _apply_param_mask(
    df: pd.DataFrame, base_mask: pd.Series, p: Dict[str, Any], penalizar_nan: bool
) -> pd.Series:
    """Aplicar una restricción de parámetro sobre base_mask.

    p admite claves flexibles:
      - col (str), active (bool, opcional), mode (str)
      - value (float, opcional), tol/tolerance (float, opcional)
      - min (float, opcional), max (float, opcional)

    Modos soportados: ignorar, fijo|objetivo, minimo, maximo, rango
    """
    try:
        col = str(p.get("col"))
        if not col or col not in df.columns:
            return base_mask
        mode = str(p.get("mode", "ignorar"))
        # Sólo aplicar si está activo o el modo no es ignorar
        if not bool(p.get("active", mode != "ignorar")):
            return base_mask

        s_num = _safe_numeric_series(df, col)
        s_raw = df[col]

        m = pd.Series(True, index=df.index, dtype=bool)
        if mode in {"fijo", "objetivo"}:
            v = p.get("value", None)
            tol = p.get("tol", p.get("tolerance", None))
            if v is None:
                return base_mask
            try:
                vv = float(v)
            except Exception:
                # Si no es numérico, comparar por string exacto
                m &= s_raw.astype(str) == str(v)
            else:
                if tol not in (None, ""):
                    try:
                        tt = float(tol)
                    except Exception:
                        tt = 0.0
                    m &= (s_num - vv).abs() <= tt
                else:
                    # Igualdad exacta para numéricos (puede ser estricta)
                    m &= s_num == vv
        elif mode == "minimo":
            v = p.get("value", None)
            if v in (None, ""):
                return base_mask
            try:
                vv = float(v)
            except Exception:
                return base_mask
            m &= s_num >= vv
        elif mode == "maximo":
            v = p.get("value", None)
            if v in (None, ""):
                return base_mask
            try:
                vv = float(v)
            except Exception:
                return base_mask
            m &= s_num <= vv
        elif mode == "rango":
            vmin = p.get("min", None)
            vmax = p.get("max", None)
            if vmin not in (None, ""):
                try:
                    m &= s_num >= float(vmin)
                except Exception:
                    pass
            if vmax not in (None, ""):
                try:
                    m &= s_num <= float(vmax)
                except Exception:
                    pass
        else:
            # ignorar u otros: no afectan
            return base_mask

        if penalizar_nan:
            m &= s_raw.notna()

        return base_mask & m
    except Exception:
        return base_mask


def _apply_segment_mask(
    df: pd.DataFrame, base_mask: pd.Series, state: Dict[str, Any]
) -> pd.Series:
    col = state.get("segmentar_por")
    modo = state.get("segmentar_modo", "off")
    val = state.get("segmentar_valor")
    if not col or col not in df.columns:
        return base_mask
    if val in (None, "(ninguno)"):
        return base_mask
    if str(modo) == "off":
        return base_mask

    seg_labels = state.get("segment_labels")
    s_raw = df[col].astype(str)
    target = str(val)

    if isinstance(seg_labels, dict) and seg_labels:
        # seg_labels: raw -> label; permitir match por label
        s_map = s_raw.map(lambda x: str(seg_labels.get(x, x)))
        m = (s_map == target) | (s_raw == target)
    else:
        m = s_raw == target

    return base_mask & m


def get_scope_view(
    df_base: pd.DataFrame, state: Dict[str, Any]
) -> _Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Construye una vista del DataFrame según el alcance (Ámbito) indicado en 'state'.

    Devuelve (df_view, mask, info) donde:
      - df_view = df_base[mask]
      - mask = Serie booleana alineada al índice de df_base
      - info = {"n": len(df_view), "scope": ambito | "topk-empty"}
    """
    try:
        ambito = str(state.get("ambito", "global"))
    except Exception:
        ambito = "global"

    if ambito == "global":
        mask = _fullmask(df_base)
        return df_base, mask, {"n": len(df_base), "scope": "global"}

    if ambito == "topk":
        topk_idx = state.get("topk_index")
        if not topk_idx:
            mask = _fullmask(df_base)
            return df_base, mask, {"n": len(df_base), "scope": "topk-empty"}
        try:
            mask = df_base.index.isin(list(topk_idx))
            mask = pd.Series(mask, index=df_base.index)
        except Exception:
            mask = _fullmask(df_base)
            return df_base, mask, {"n": len(df_base), "scope": "topk-empty"}
        df_view = df_base[mask]
        return df_view, mask, {"n": len(df_view), "scope": "topk"}

    # ambito == "filtrado" (o cualquier otro → tratar como filtrado)
    mask = _fullmask(df_base)
    penalizar_nan = bool(state.get("penalizar_nan", False))
    # Parametrización proveniente del panel dinámico
    param_config = state.get("param_config") or []
    # Aceptar también 'restricciones' (formato dict) como respaldo
    if not param_config:
        restr_any = state.get("restricciones")
        if isinstance(restr_any, dict):
            restr_dict: Dict[str, Any] = restr_any
            for col, d in restr_dict.items():
                if not isinstance(d, dict):
                    continue
                mode = str(d.get("tipo", d.get("mode", "ignorar")))
                item = {
                    "col": col,
                    "active": True,
                    "mode": mode,
                    "value": d.get("valor", d.get("value")),
                    "min": d.get("min"),
                    "max": d.get("max"),
                    "tol": d.get("tol"),
                }
                param_config.append(item)

    # Aplicar parámetros activos (modo != ignorar)
    if isinstance(param_config, list):
        for p in param_config:
            try:
                if str(p.get("mode", "ignorar")) == "ignorar":
                    continue
            except Exception:
                continue
            mask = _apply_param_mask(df_base, mask, p, penalizar_nan)

    # Intersectar con segmentación/misión si existe
    mask = _apply_segment_mask(df_base, mask, state)

    df_view = df_base[mask]
    return df_view, mask, {"n": len(df_view), "scope": "filtrado"}


def scope_or_global(
    df_base: pd.DataFrame, state: Dict[str, Any]
) -> _Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """Devuelve (df_view, mask, info) según Ámbito, con fallback al global si queda vacío."""
    df_view, mask, info = get_scope_view(df_base, state)
    if len(df_view) == 0:
        full = _fullmask(df_base)
        return df_base, full, {"n": len(df_base), "scope": "fallback-global"}
    return df_view, mask, info
