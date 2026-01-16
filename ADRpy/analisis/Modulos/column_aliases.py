# -*- coding: utf-8 -*-
"""column_aliases.py

Pequeña utilidad para tolerar alias de encabezados (por ejemplo con unidades)
SIN exigir renombrar columnas en el Excel.

La idea es que el resto del código trabaje con nombres canónicos y esta capa
resuelva a la columna real presente en el DataFrame.
"""

from __future__ import annotations

from typing import Iterable, Optional

import difflib
import re
import unicodedata


# Nombres canónicos -> posibles variantes en archivos reales
ALIASES: dict[str, list[str]] = {
    "Alcance de la aeronave": [
        "Alcance de la aeronave (km)",
        "Alcance de la aeronave [km]",
    ],
    "Autonomía de la aeronave": [
        "Autonomía de la aeronave (h)",
        "Autonomía de la aeronave [h]",
    ],
    "Velocidad crucero m/s": [
        "Velocidad crucero (m/s IAS)",
        "Velocidad crucero m/s",
    ],
    "Altitud de crucero": [
        "Altitud a la que se realiza el crucero",
        "Altitud a la que se realiza el crucero (ft)",
    ],
    # Variantes por problemas de encoding en algunos Excels (mojibake)
    "Relación de aspecto del ala": [
        "Relaci�n de aspecto del ala",
    ],
    "Peso máximo al despegue (MTOW)": [
        "Peso m�ximo al despegue (MTOW)",
    ],
    "Área del ala": [
        "�rea del ala",
    ],
    "Potencia específica (P/W)": [
        "Potencia espec�fica (P/W)",
    ],
    "Rango de comunicación": [
        "Rango de comunicaci�n",
    ],
    "Misión": [
        "Misi�n",
    ],
    "Propulsión horizontal": [
        "Propulsi�n horizontal",
    ],
    "Propulsión vertical": [
        "Propulsi�n vertical",
    ],
}


def _build_alias_to_canonical() -> dict[str, str]:
    out: dict[str, str] = {}
    for canon, alts in ALIASES.items():
        out[canon] = canon
        for a in alts:
            out[a] = canon
    return out


ALIAS_TO_CANONICAL = _build_alias_to_canonical()


def _normalize(s: str) -> str:
    # Normaliza: minúsculas, quita acentos, colapsa espacios,
    # y elimina caracteres raros para tolerar mojibake.
    s0 = str(s)
    s0 = unicodedata.normalize("NFKD", s0)
    s0 = s0.encode("ascii", "ignore").decode("ascii")
    s0 = s0.lower().strip()
    s0 = re.sub(r"\s+", " ", s0)
    return s0


_NORM_ALIAS_TO_CANON: dict[str, str] = {
    _normalize(k): v for k, v in ALIAS_TO_CANONICAL.items()
}


def canonicalize_parametro(nombre: str) -> str:
    """Devuelve el nombre canónico de un parámetro si es alias."""
    if nombre in ALIAS_TO_CANONICAL:
        return ALIAS_TO_CANONICAL[nombre]

    n = _normalize(nombre)
    if n in _NORM_ALIAS_TO_CANON:
        return _NORM_ALIAS_TO_CANON[n]

    # Fuzzy: por si el texto viene con mojibake (p.ej. Autonom�a)
    keys = list(_NORM_ALIAS_TO_CANON.keys())
    if keys:
        best = difflib.get_close_matches(n, keys, n=1, cutoff=0.92)
        if best:
            return _NORM_ALIAS_TO_CANON[best[0]]

    return nombre


def resolve_name_in_columns(nombre: str, columnas: Iterable[str]) -> Optional[str]:
    """Resuelve 'nombre' (canónico o alias) al nombre real existente en columnas.

    - Preferencia: match exacto, luego buscar en alias del canónico.
    - Si no existe nada, devuelve None.
    """
    cols = list(columnas)
    if nombre in cols:
        return nombre

    canon = canonicalize_parametro(nombre)
    if canon in cols:
        return canon

    for alt in ALIASES.get(canon, []):
        if alt in cols:
            return alt

    # Fuzzy match contra nombres reales (tolerar unidades/acentos/mojibake)
    candidates = [canon] + list(ALIASES.get(canon, []))
    cols_norm = {_normalize(c): c for c in cols}
    cols_norm_keys = list(cols_norm.keys())
    best_match = None
    best_ratio = 0.0
    for cand in candidates:
        cn = _normalize(cand)
        if cn in cols_norm:
            return cols_norm[cn]
        if cols_norm_keys:
            m = difflib.get_close_matches(cn, cols_norm_keys, n=1, cutoff=0.90)
            if m:
                ratio = difflib.SequenceMatcher(None, cn, m[0]).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = cols_norm[m[0]]

    return best_match if best_ratio >= 0.90 else None
