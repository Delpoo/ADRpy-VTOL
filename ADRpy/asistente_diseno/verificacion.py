"""
Verificaciones físicas básicas (sin imputar), con umbrales:
- OK si Δrel < 2%
- Ajuste sugerido si 2% ≤ Δrel ≤ 10%
- No confiable si Δrel > 10%

Identidades:
1) MTOW ≈ W0 + Payload                           (positividad y Payload ≤ MTOW)
2) AR ≈ b / c                                    (sin usar área alar)
3) VTAS ≈ IAS / sqrt(σ_5000ft)                   (σ constante de verificación)
4) R ≈ (h * 3600 * V) / 1000, con V = TAS crucero

Modo por defecto: agregado (usa MEDIANAS de filas válidas para cada ecuación),
para estabilizar redondeos y evitar ruido. No modifica el DataFrame.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Literal, Any

import numpy as np
import pandas as pd

from . import config
from .tipos import VerificationCard
from .datos import a_numerico_seguro


# =========================
# Helpers internos
# =========================


def _rel_error(a: float, b: float) -> Optional[float]:
    """Δrel = |a-b| / max(|a|,|b|,1e-12). None si no se puede evaluar."""
    try:
        A, B = float(a), float(b)
    except Exception:
        return None
    denom = max(abs(A), abs(B), 1e-12)
    return abs(A - B) / denom


def _dictamen(delta_rel: Optional[float]) -> str:
    """Clasifica Δrel según umbrales de config."""
    if delta_rel is None:
        return "no_evaluable"
    if delta_rel < config.DELTA_REL_OK:
        return "OK"
    if delta_rel <= config.DELTA_REL_AJUSTE:
        return "Ajuste sugerido"
    return "No confiable"


def _mediana_segura(s: pd.Series) -> Optional[float]:
    s_num = a_numerico_seguro(s).dropna()
    if s_num.empty:
        return None
    return float(s_num.median())


# =========================
# Verificaciones principales
# =========================


def verificar_masas(
    df: pd.DataFrame,
    *,
    col_mtow: str = config.COL_MTOW,
    col_w0: str = config.COL_W0,
    col_payload: str = config.COL_PAYLOAD,
    modo: str = "aggregate",
) -> VerificationCard:
    """
    Verifica: MTOW ≈ W0 + Payload. También audita positividad y Payload ≤ MTOW.
    - modo="aggregate": usa MEDIANAS de las filas válidas (recomendado para UI).
    - modo="row": no implementado aquí (para auditoría fila a fila más adelante).

    Returns
    -------
    VerificationCard
    """
    # Subconjunto válido
    cols = [col_mtow, col_w0, col_payload]
    sub = df[cols].copy()
    for c in cols:
        sub[c] = a_numerico_seguro(sub[c])
    sub = sub.dropna()

    if sub.empty:
        return VerificationCard(
            nombre="MTOW = W0 + Payload",
            ecuacion="MTOW ≈ W0 + Payload",
            entradas={},
            delta_rel=None,
            dictamen="no_evaluable",
            notas="Sin filas válidas (faltan datos numéricos).",
        )

    # Modo agregado: medianas
    mtow = float(sub[col_mtow].median())
    w0 = float(sub[col_w0].median())
    payload = float(sub[col_payload].median())

    lhs = mtow
    rhs = w0 + payload
    drel = _rel_error(lhs, rhs)

    # Auditoría simple de violaciones duras (conteo, no detiene)
    violaciones = 0
    if (
        (sub[col_mtow] <= 0).any()
        or (sub[col_w0] <= 0).any()
        or (sub[col_payload] <= 0).any()
    ):
        violaciones += 1
    if (sub[col_payload] > sub[col_mtow]).any():
        violaciones += 1

    notas = "Medianas; reglas: valores > 0 y Payload ≤ MTOW."
    if violaciones > 0:
        notas += f" Se detectaron {violaciones} violación(es) en el conjunto."

    return VerificationCard(
        nombre="MTOW = W0 + Payload",
        ecuacion="MTOW ≈ W0 + Payload",
        entradas={"MTOW_med": mtow, "W0_med": w0, "Payload_med": payload},
        delta_rel=drel,
        dictamen=_dictamen(drel),
        notas=notas,
    )


def verificar_ar_bc(
    df: pd.DataFrame,
    *,
    col_ar: str = config.COL_AR,
    col_b: str = config.COL_B,
    col_c: str = config.COL_C,
    modo: str = "aggregate",
) -> VerificationCard:
    """
    Verifica: AR ≈ b / c. Si no hay AR en el DF, reporta AR_ref=b/c y dictamen 'no_evaluable'.
    Returns
    -------
    VerificationCard
    """
    cols_bc = [col_b, col_c]
    sub_bc = df[cols_bc].copy()
    for c in cols_bc:
        sub_bc[c] = a_numerico_seguro(sub_bc[c])
    sub_bc = sub_bc.dropna()

    if sub_bc.empty:
        return VerificationCard(
            nombre="AR = b / c",
            ecuacion="AR ≈ b / c",
            entradas={},
            delta_rel=None,
            dictamen="no_evaluable",
            notas="Sin filas válidas para b y c.",
        )

    b_med = float(sub_bc[col_b].median())
    c_med = float(sub_bc[col_c].median())
    ar_ref = b_med / c_med if c_med != 0 else np.nan

    if col_ar not in df.columns:
        return VerificationCard(
            nombre="AR = b / c",
            ecuacion="AR ≈ b / c",
            entradas={"b_med": b_med, "c_med": c_med, "AR_ref": ar_ref},
            delta_rel=None,
            dictamen="no_evaluable",
            notas="Columna AR no existe en el dataset; se informa AR_ref = b/c como referencia.",
        )

    ar_med = _mediana_segura(df[col_ar])
    if ar_med is None or not np.isfinite(ar_ref):
        return VerificationCard(
            nombre="AR = b / c",
            ecuacion="AR ≈ b / c",
            entradas={
                "b_med": b_med,
                "c_med": c_med,
                "AR_med": ar_med if ar_med is not None else float("nan"),
            },
            delta_rel=None,
            dictamen="no_evaluable",
            notas="No fue posible evaluar Δrel (datos insuficientes o división por cero).",
        )

    drel = _rel_error(ar_med, ar_ref)
    return VerificationCard(
        nombre="AR = b / c",
        ecuacion="AR ≈ b / c",
        entradas={"AR_med": ar_med, "b_med": b_med, "c_med": c_med, "AR_ref": ar_ref},
        delta_rel=drel,
        dictamen=_dictamen(drel),
        notas="Medianas; no se usa área alar (S).",
    )


def verificar_ias_tas(
    df: pd.DataFrame,
    *,
    col_ias: str = config.COL_IAS,
    col_tas: str = config.COL_TAS,
    sigma_5000: float = config.SIGMA_5000_FT,
    modo: str = "aggregate",
) -> VerificationCard:
    """
    Verifica: VTAS ≈ IAS / sqrt(σ_5000ft). σ es constante (no atmósfera).
    Returns
    -------
    VerificationCard
    """
    cols = [col_ias, col_tas]
    sub = df[cols].copy()
    for c in cols:
        sub[c] = a_numerico_seguro(sub[c])
    sub = sub.dropna()

    if sub.empty:
        return VerificationCard(
            nombre="IAS ↔ TAS (σ fija)",
            ecuacion="TAS ≈ IAS / sqrt(σ_5000ft)",
            entradas={},
            delta_rel=None,
            dictamen="no_evaluable",
            notas="Sin filas válidas para IAS y TAS.",
        )

    ias_med = float(sub[col_ias].median())
    tas_med = float(sub[col_tas].median())

    if sigma_5000 <= 0:
        return VerificationCard(
            nombre="IAS ↔ TAS (σ fija)",
            ecuacion="TAS ≈ IAS / sqrt(σ_5000ft)",
            entradas={"IAS_med": ias_med, "TAS_med": tas_med, "σ": sigma_5000},
            delta_rel=None,
            dictamen="no_evaluable",
            notas="σ inválida (≤0). Debe ser positiva.",
        )

    tas_ref = ias_med / np.sqrt(sigma_5000)
    drel = _rel_error(tas_med, tas_ref)

    return VerificationCard(
        nombre="IAS ↔ TAS (σ fija)",
        ecuacion="TAS ≈ IAS / sqrt(σ_5000ft)",
        entradas={
            "IAS_med": ias_med,
            "TAS_med": tas_med,
            "TAS_ref": tas_ref,
            "σ": sigma_5000,
        },
        delta_rel=drel,
        dictamen=_dictamen(drel),
        notas="Medianas; σ constante (5000 ft).",
    )


def verificar_alcance(
    df: pd.DataFrame,
    *,
    col_R: str = config.COL_R,
    col_h: str = config.COL_H,
    col_V: Optional[str] = None,  # preferir IAS; fallback a TAS
    modo: str = "aggregate",
) -> VerificationCard:
    """
    Verifica: R ≈ (h * 3600 * V) / 1000, con V = IAS de crucero (preferente; fallback TAS).
    - h en horas, V en m/s, R en km.
    """
    # Resolver columna de velocidad
    if col_V is None:
        col_V = (
            getattr(config, "COL_V_IAS", None)
            or getattr(config, "COL_V_TAS", None)
            or getattr(config, "COL_TAS", None)
        )

    cols = [col_R, col_h, col_V]
    sub = df[cols].copy()
    for c in cols:
        sub[c] = a_numerico_seguro(sub[c])
    sub = sub.dropna()

    if sub.empty:
        return VerificationCard(
            nombre="R = h·V (coherencia unitaria)",
            ecuacion="R ≈ (h * 3600 * V) / 1000",
            entradas={},
            delta_rel=None,
            dictamen="no_evaluable",
            notas=f"Sin filas válidas para R, h y V (usando {'IAS' if col_V == getattr(config, 'COL_V_IAS', None) else 'TAS'}).",
        )

    R_med = float(sub[col_R].median())
    h_med = float(sub[col_h].median())
    V_med = float(sub[col_V].median())

    R_ref = (h_med * 3600.0 * V_med) / 1000.0
    drel = _rel_error(R_med, R_ref)

    return VerificationCard(
        nombre="R = h·V (coherencia unitaria)",
        ecuacion=f"R ≈ (h * 3600 * V) / 1000  (V = {'IAS' if col_V == getattr(config, 'COL_V_IAS', None) else 'TAS'} crucero)",
        entradas={"R_med": R_med, "h_med": h_med, "V_med": V_med, "R_ref": R_ref},
        delta_rel=drel,
        dictamen=_dictamen(drel),
        notas="Medianas; h en horas, V en m/s, R en km.",
    )


def verificaciones_resumen(df: pd.DataFrame) -> List[VerificationCard]:
    """
    Ejecuta todas las verificaciones en modo agregado (medianas) y devuelve la lista de tarjetas.

    Returns
    -------
    List[VerificationCard]
    """
    cards: List[VerificationCard] = []
    cards.append(verificar_masas(df))
    cards.append(verificar_ar_bc(df))
    cards.append(verificar_ias_tas(df))
    cards.append(verificar_alcance(df))
    return cards


# =========================
# Exportación fila a fila con subrayados en Excel
# =========================
from pathlib import Path
from openpyxl.styles import Font
from openpyxl.comments import Comment
from openpyxl import load_workbook
from openpyxl.cell.cell import MergedCell
import shutil


def exportar_verificacion_fila_a_excel(
    df: pd.DataFrame,
    ruta_salida: Path | str | None = None,
    *,
    sigma_5000: float | None = None,
    umbral_ok: float | None = None,
    umbral_ajuste: float | None = None,
    mostrar_comentarios: bool = True,
) -> Path:
    """
    Exporta un Excel con verificación FILA A FILA de identidades físicas y celdas subrayadas.

    Identidades verificadas por fila:
      1) MTOW ≈ W0 + Payload
      2) AR ≈ b / c
      3) TAS ≈ IAS / sqrt(σ)     (σ fija; no atmósfera)
      4) R ≈ (h * 3600 * V) / 1000, con V = TAS de crucero

    Subrayado en hoja 'Datos':
      - Subrayado simple: Δrel en [2%, 10%]
      - Doble subrayado : Δrel > 10%
    (Los umbrales se leen de config por defecto: <2% OK, 2–10% ajuste, >10% no confiable.)

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame original (no se modifica).
    ruta_salida : Path | str | None
        Ruta del .xlsx a generar. Si None, escribe en:
        {config.BASE_DIR}/analisis/Results/Verificacion_fila_fila.xlsx
    sigma_5000 : float | None
        σ fija para IAS↔TAS. Si None, usa config.SIGMA_5000_FT.
    umbral_ok : float | None
        Umbral de OK (Δrel < umbral_ok). Si None, usa config.DELTA_REL_OK (0.02).
    umbral_ajuste : float | None
        Umbral superior de "ajuste" (Δrel ≤ umbral_ajuste). Si None, usa config.DELTA_REL_AJUSTE (0.10).

    Returns
    -------
    Path
        Ruta del archivo .xlsx generado.

    Notas
    -----
    - Si falta alguna columna (según config.COL_*), esa identidad queda 'no_evaluable' para esas filas.
    - No se hace imputación. Sólo verificación y marcado.
    - mostrar_comentarios: si True, deja visibles los comentarios para facilitar su lectura.
    """
    # ---- Config efectiva
    sigma = config.SIGMA_5000_FT if sigma_5000 is None else float(sigma_5000)
    thr_ok = config.DELTA_REL_OK if umbral_ok is None else float(umbral_ok)
    thr_adj = config.DELTA_REL_AJUSTE if umbral_ajuste is None else float(umbral_ajuste)

    # ---- Ruta de salida por defecto
    if ruta_salida is None:
        ruta_salida = (
            config.BASE_DIR / "analisis" / "Results" / "Verificacion_fila_fila.xlsx"
        )
    ruta_salida = Path(ruta_salida)

    # ---- Helper: convertir a numérico
    def _num(s: pd.Series) -> pd.Series:
        return pd.to_numeric(s, errors="coerce")

    # ---- Helper: Δrel serie
    def _rel_err(a: pd.Series, b: pd.Series) -> pd.Series:
        A = _num(a)
        B = _num(b)
        # Denominador como Series para preservar índices y evitar ndarray
        denom = pd.concat([A.abs(), B.abs()], axis=1).max(axis=1).clip(lower=1e-12)
        out = (A.subtract(B).abs().divide(denom)).astype(float)
        out[~np.isfinite(out)] = np.nan
        return out

    # ---- Mapeo de columnas (desde config)
    cols = {
        "MTOW": getattr(config, "COL_MTOW", None),
        "W0": getattr(config, "COL_W0", None),
        "Payload": getattr(config, "COL_PAYLOAD", None),
        "AR": getattr(config, "COL_AR", None),
        "b": getattr(config, "COL_B", None),
        "c": getattr(config, "COL_C", None),
        "IAS": getattr(config, "COL_IAS", None),
        "TAS": getattr(config, "COL_TAS", None),
        "R": getattr(config, "COL_R", None),
        "h": getattr(config, "COL_H", None),
    }
    # V = IAS si existe en el DF; si no, TAS
    cols["V"] = cols["IAS"] if (cols.get("IAS") in df.columns) else cols.get("TAS")
    speed_label = "IAS" if cols["V"] == cols.get("IAS") else "TAS"

    # ---- Copia numérica de las columnas usadas
    num_df = df.copy()
    for col in cols.values():
        if col is not None and col in num_df.columns:
            num_df[col] = _num(num_df[col])

    # ---- Cálculo Δrel por identidad
    results = pd.DataFrame(index=df.index)

    # 1) MTOW ≈ W0 + Payload
    if (
        cols["MTOW"] in df.columns
        and cols["W0"] in df.columns
        and cols["Payload"] in df.columns
    ):
        lhs = num_df[cols["MTOW"]]
        rhs = num_df[cols["W0"]] + num_df[cols["Payload"]]
        results["drel_MTOW"] = _rel_err(lhs, rhs)
    else:
        results["drel_MTOW"] = np.nan

    # 2) AR ≈ b / c
    if cols["AR"] in df.columns and cols["b"] in df.columns and cols["c"] in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            ar_ref = num_df[cols["b"]] / num_df[cols["c"]]
        ar = num_df[cols["AR"]]
        results["drel_AR"] = _rel_err(ar, ar_ref)
    else:
        results["drel_AR"] = np.nan

    # 3) TAS ≈ IAS / sqrt(σ)
    if cols["IAS"] in df.columns and cols["TAS"] in df.columns and sigma > 0:
        tas_ref = num_df[cols["IAS"]] / np.sqrt(sigma)
        tas = num_df[cols["TAS"]]
        results["drel_IAS_TAS"] = _rel_err(tas, tas_ref)
    else:
        results["drel_IAS_TAS"] = np.nan

    # 4) R ≈ (h * 3600 * V) / 1000, con V = TAS
    if cols["R"] in df.columns and cols["h"] in df.columns and cols["V"] in df.columns:
        R_ref = (num_df[cols["h"]] * 3600.0 * num_df[cols["V"]]) / 1000.0
        R = num_df[cols["R"]]
        results["drel_R"] = _rel_err(R, R_ref)
    else:
        results["drel_R"] = np.nan

    # ---- Severidad por celda (fila a fila)
    def _sev(x: float) -> str:
        if pd.isna(x):
            return "no_evaluable"
        if x < thr_ok:
            return "OK"
        if x <= thr_adj:
            return "ajuste"
        return "no_confiable"

    sev_df = pd.DataFrame(index=df.index)
    sev_df["sev_MTOW"] = results["drel_MTOW"].apply(_sev)
    sev_df["sev_AR"] = results["drel_AR"].apply(_sev)
    sev_df["sev_IAS_TAS"] = results["drel_IAS_TAS"].apply(_sev)
    sev_df["sev_R"] = results["drel_R"].apply(_sev)

    # ---- Crear carpeta salida y copiar archivo original para preservar formato
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    origen = config.DATA_XLSX
    shutil.copyfile(origen, ruta_salida)

    # ---- Abrir libro copiado y subrayar en la hoja original
    wb = load_workbook(ruta_salida)
    sheet_spec = getattr(config, "EXCEL_SHEET", 0)
    if isinstance(sheet_spec, int):
        ws = wb.worksheets[sheet_spec]
    else:
        ws = wb[sheet_spec]

    # Congelar paneles en B2 (fila 1 y columna A visibles)
    try:
        ws.freeze_panes = ws["B2"]
    except Exception:
        pass

    # Header mapping desde fila 1
    col_to_idx: Dict[str, int] = {}
    for j, cell in enumerate(ws[1], start=1):
        header = cell.value
        if header is None:
            continue
        col_to_idx[str(header)] = j

    def _set_underline(cell, style: Literal["single", "double"]):
        f = cell.font or Font()
        cell.font = Font(
            name=f.name,
            size=f.size,
            bold=f.bold,
            italic=f.italic,
            vertAlign=f.vertAlign,
            underline=style,
            color=f.color,
        )

    # Helper: si la celda pertenece a un rango combinado, usar la celda topleft
    def _top_left_coords(row: int, col: int) -> tuple[int, int]:
        try:
            for rng in ws.merged_cells.ranges:
                if (rng.min_row <= row <= rng.max_row) and (
                    rng.min_col <= col <= rng.max_col
                ):
                    return rng.min_row, rng.min_col
        except Exception:
            pass
        return row, col

    def _cell_at(row: int, col: int):
        r, c = _top_left_coords(row, col)
        return ws.cell(row=r, column=c)

    def create_large_comment(text, author="Asistente Diseño"):
        """Create a comment with enlarged size for better visibility"""
        comment = Comment(text, author)
        # Usar EXACTAMENTE los mismos valores que excel_export.py
        comment.width = 500  # Default is around 100, making it 4x
        comment.height = 1000  # Default is around 50, making it 12x
        return comment

    # Acumular severidad por celda: 0 nada, 1 single, 2 double
    sev_map: dict[tuple[int, int], int] = {}
    # Acumular comentarios por celda (posible combinación de varias identidades)
    comments_map: dict[tuple[int, int], list[str]] = {}

    def _mark(row_excel: int, cols_names: list[str | None], level: int):
        for cname in cols_names:
            if cname is None:
                continue
            if cname not in col_to_idx:
                continue
            cidx = col_to_idx[cname]
            key = (row_excel, cidx)
            prev = sev_map.get(key, 0)
            sev_map[key] = max(prev, level)

    def _add_comment(row_excel: int, cols_names: list[str | None], text: str):
        for cname in cols_names:
            if cname is None or cname not in col_to_idx:
                continue
            cidx = col_to_idx[cname]
            key = (row_excel, cidx)
            comments_map.setdefault(key, []).append(text)

    n_rows = df.shape[0]

    def _fmt(x: Any) -> str:
        try:
            if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
                return "NaN"
            return f"{float(x):.6g}"
        except Exception:
            return str(x)

    # Helpers para obtener valores numéricos de manera segura
    def _to_float(x: Any) -> float:
        try:
            xf = float(x)
            return xf
        except Exception:
            return float("nan")

    def _get_num(row_i: int, colname: Optional[str]) -> float:
        if isinstance(colname, str) and colname in num_df.columns:
            val = num_df.at[df.index[row_i], colname]
            return _to_float(val) if pd.notna(val) else float("nan")
        return float("nan")

    def _safe0(x: Any) -> float:
        try:
            xf = float(x)
            return xf if np.isfinite(xf) else 0.0
        except Exception:
            return 0.0

    for i in range(n_rows):
        excel_row = i + 2  # datos empiezan en fila 2
        # MTOW = W0 + Payload
        s = sev_df.at[df.index[i], "sev_MTOW"]
        if s == "ajuste":
            _mark(excel_row, [cols["MTOW"], cols["W0"], cols["Payload"]], 1)
            mt = _get_num(i, cols["MTOW"])
            w0 = _get_num(i, cols["W0"])
            pl = _get_num(i, cols["Payload"])
            rhs = _safe0(w0) + _safe0(pl)
            drel = _to_float(results.at[df.index[i], "drel_MTOW"])
            pct = "" if not np.isfinite(drel) else f"{drel*100:.2f}%"
            txt = (
                "[Verificación] MTOW ≈ W0 + Payload\n"
                "[Fórmula] Δrel = |A − B| / max(|A|, |B|, 1e−12) con A=MTOW, B=W0+Payload\n"
                "[Normalización] max(|A|,|B|) hace el error relativo, simétrico y estable (evita división por 0).\n"
                f"[Sustitución] |{_fmt(mt)} − ({_fmt(w0)} + {_fmt(pl)})| / max(|{_fmt(mt)}|, |{_fmt(rhs)}|, 1e−12)\n"
                f"[Resultado] Δrel={_fmt(drel)} ({pct}); severidad: {s}"
            )
            _add_comment(excel_row, [cols["MTOW"], cols["W0"], cols["Payload"]], txt)
        elif s == "no_confiable":
            _mark(excel_row, [cols["MTOW"], cols["W0"], cols["Payload"]], 2)
            mt = _get_num(i, cols["MTOW"])
            w0 = _get_num(i, cols["W0"])
            pl = _get_num(i, cols["Payload"])
            rhs = _safe0(w0) + _safe0(pl)
            drel = _to_float(results.at[df.index[i], "drel_MTOW"])
            pct = "" if not np.isfinite(drel) else f"{drel*100:.2f}%"
            txt = (
                "[Verificación] MTOW ≈ W0 + Payload\n"
                "[Fórmula] Δrel = |A − B| / max(|A|, |B|, 1e−12) con A=MTOW, B=W0+Payload\n"
                "[Normalización] max(|A|,|B|) hace el error relativo, simétrico y estable (evita división por 0).\n"
                f"[Sustitución] |{_fmt(mt)} − ({_fmt(w0)} + {_fmt(pl)})| / max(|{_fmt(mt)}|, |{_fmt(rhs)}|, 1e−12)\n"
                f"[Resultado] Δrel={_fmt(drel)} ({pct}); severidad: {s}"
            )
            _add_comment(excel_row, [cols["MTOW"], cols["W0"], cols["Payload"]], txt)
        # AR = b / c
        s = sev_df.at[df.index[i], "sev_AR"]
        if s == "ajuste":
            _mark(excel_row, [cols["AR"], cols["b"], cols["c"]], 1)
            ar = _get_num(i, cols["AR"])
            b = _get_num(i, cols["b"])
            c = _get_num(i, cols["c"])
            with np.errstate(divide="ignore", invalid="ignore"):
                ref = ar if np.isfinite(ar) else np.nan  # placeholder to keep type
                if np.isfinite(b) and np.isfinite(c) and c != 0:
                    ref = b / c
                else:
                    ref = np.nan
            drel = _to_float(results.at[df.index[i], "drel_AR"])
            pct = "" if not np.isfinite(drel) else f"{drel*100:.2f}%"
            txt = (
                "[Verificación] AR ≈ b / c\n"
                "[Fórmula] Δrel = |A − B| / max(|A|, |B|, 1e−12) con A=AR, B=b/c\n"
                "[Normalización] Error relativo simétrico para escalas comparables.\n"
                f"[Sustitución] |{_fmt(ar)} − ({_fmt(b)}/{_fmt(c)})| / max(|{_fmt(ar)}|, |{_fmt(ref)}|, 1e−12)\n"
                f"[Resultado] Δrel={_fmt(drel)} ({pct}); severidad: {s}"
            )
            _add_comment(excel_row, [cols["AR"], cols["b"], cols["c"]], txt)
        elif s == "no_confiable":
            _mark(excel_row, [cols["AR"], cols["b"], cols["c"]], 2)
            ar = _get_num(i, cols["AR"])
            b = _get_num(i, cols["b"])
            c = _get_num(i, cols["c"])
            with np.errstate(divide="ignore", invalid="ignore"):
                if np.isfinite(b) and np.isfinite(c) and c != 0:
                    ref = b / c
                else:
                    ref = np.nan
            drel = _to_float(results.at[df.index[i], "drel_AR"])
            pct = "" if not np.isfinite(drel) else f"{drel*100:.2f}%"
            txt = (
                "[Verificación] AR ≈ b / c\n"
                "[Fórmula] Δrel = |A − B| / max(|A|, |B|, 1e−12) con A=AR, B=b/c\n"
                "[Normalización] Error relativo simétrico para escalas comparables.\n"
                f"[Sustitución] |{_fmt(ar)} − ({_fmt(b)}/{_fmt(c)})| / max(|{_fmt(ar)}|, |{_fmt(ref)}|, 1e−12)\n"
                f"[Resultado] Δrel={_fmt(drel)} ({pct}); severidad: {s}"
            )
            _add_comment(excel_row, [cols["AR"], cols["b"], cols["c"]], txt)
        # IAS ↔ TAS
        s = sev_df.at[df.index[i], "sev_IAS_TAS"]
        if s == "ajuste":
            _mark(excel_row, [cols["IAS"], cols["TAS"]], 1)
            ias = _get_num(i, cols["IAS"])
            tas = _get_num(i, cols["TAS"])
            ref = (
                (ias / np.sqrt(sigma))
                if (sigma and sigma > 0 and np.isfinite(ias))
                else np.nan
            )
            drel = _to_float(results.at[df.index[i], "drel_IAS_TAS"])
            pct = "" if not np.isfinite(drel) else f"{drel*100:.2f}%"
            txt = (
                "[Verificación] TAS ≈ IAS / √σ (σ fija)\n"
                "[Fórmula] Δrel = |A − B| / max(|A|, |B|, 1e−12) con A=TAS, B=IAS/√σ\n"
                "[Normalización] Comparación en escala relativa (σ fija evita usar atmósfera).\n"
                f"[Sustitución] |{_fmt(tas)} − {_fmt(ias)}/√{_fmt(sigma)}| / max(|{_fmt(tas)}|, |{_fmt(ref)}|, 1e−12)\n"
                f"[Resultado] Δrel={_fmt(drel)} ({pct}); severidad: {s}"
            )
            _add_comment(excel_row, [cols["IAS"], cols["TAS"]], txt)
        elif s == "no_confiable":
            _mark(excel_row, [cols["IAS"], cols["TAS"]], 2)
            ias = _get_num(i, cols["IAS"])
            tas = _get_num(i, cols["TAS"])
            ref = (
                (ias / np.sqrt(sigma))
                if (sigma and sigma > 0 and np.isfinite(ias))
                else np.nan
            )
            drel = _to_float(results.at[df.index[i], "drel_IAS_TAS"])
            pct = "" if not np.isfinite(drel) else f"{drel*100:.2f}%"
            txt = (
                "[Verificación] TAS ≈ IAS / √σ (σ fija)\n"
                "[Fórmula] Δrel = |A − B| / max(|A|, |B|, 1e−12) con A=TAS, B=IAS/√σ\n"
                "[Normalización] Comparación en escala relativa (σ fija evita usar atmósfera).\n"
                f"[Sustitución] |{_fmt(tas)} − {_fmt(ias)}/√{_fmt(sigma)}| / max(|{_fmt(tas)}|, |{_fmt(ref)}|, 1e−12)\n"
                f"[Resultado] Δrel={_fmt(drel)} ({pct}); severidad: {s}"
            )
            _add_comment(excel_row, [cols["IAS"], cols["TAS"]], txt)
        # R = h·V
        s = sev_df.at[df.index[i], "sev_R"]
        if s == "ajuste":
            _mark(excel_row, [cols["R"], cols["h"], cols["V"]], 1)
            Rv = _get_num(i, cols["R"])
            hv = _get_num(i, cols["h"])
            vv = _get_num(i, cols["V"])
            ref = (
                ((hv * 3600.0 * vv) / 1000.0)
                if (np.isfinite(hv) and np.isfinite(vv))
                else np.nan
            )
            drel = _to_float(results.at[df.index[i], "drel_R"])
            pct = "" if not np.isfinite(drel) else f"{drel*100:.2f}%"
            txt = (
                f"[Verificación] R ≈ (h · 3600 · {speed_label}) / 1000\n"
                "[Fórmula] Δrel = |A − B| / max(|A|, |B|, 1e−12) con A=R, B=(h·3600·V)/1000\n"
                "[Normalización] Escala el error a porcentaje relativo de la magnitud observada.\n"
                f"[Sustitución] |{_fmt(Rv)} − ({_fmt(hv)}·3600·{_fmt(vv)})/1000| / max(|{_fmt(Rv)}|, |{_fmt(ref)}|, 1e−12)\n"
                f"[Resultado] Δrel={_fmt(drel)} ({pct}); severidad: {s}"
            )
            _add_comment(excel_row, [cols["R"], cols["h"], cols["V"]], txt)
        elif s == "no_confiable":
            _mark(excel_row, [cols["R"], cols["h"], cols["V"]], 2)
            Rv = _get_num(i, cols["R"])
            hv = _get_num(i, cols["h"])
            vv = _get_num(i, cols["V"])
            ref = (
                ((hv * 3600.0 * vv) / 1000.0)
                if (np.isfinite(hv) and np.isfinite(vv))
                else np.nan
            )
            drel = _to_float(results.at[df.index[i], "drel_R"])
            pct = "" if not np.isfinite(drel) else f"{drel*100:.2f}%"
            txt = (
                f"[Verificación] R ≈ (h · 3600 · {speed_label}) / 1000\n"
                "[Fórmula] Δrel = |A − B| / max(|A|, |B|, 1e−12) con A=R, B=(h·3600·V)/1000\n"
                "[Normalización] Escala el error a porcentaje relativo de la magnitud observada.\n"
                f"[Sustitución] |{_fmt(Rv)} − ({_fmt(hv)}·3600·{_fmt(vv)})/1000| / max(|{_fmt(Rv)}|, |{_fmt(ref)}|, 1e−12)\n"
                f"[Resultado] Δrel={_fmt(drel)} ({pct}); severidad: {s}"
            )
            _add_comment(excel_row, [cols["R"], cols["h"], cols["V"]], txt)

    # Aplicar subrayados
    for (ridx, cidx), lvl in sev_map.items():
        cell = _cell_at(ridx, cidx)
        if isinstance(cell, MergedCell):
            # Seguridad extra (aunque _cell_at ya intenta ir al topleft)
            continue
        if lvl == 1:
            _set_underline(cell, "single")
        elif lvl == 2:
            _set_underline(cell, "double")

    # Aplicar comentarios
    for (ridx, cidx), texts in comments_map.items():
        cell = _cell_at(ridx, cidx)
        if isinstance(cell, MergedCell):
            # Evitar escribir en celdas combinadas que no sean topleft
            continue
        try:
            combined = "\n\n".join(texts)
            # Si ya había comentario, concatenar en lugar de reemplazar
            if cell.comment and getattr(cell.comment, "text", None):
                prev_text = cell.comment.text or ""
                author = cell.comment.author or "Asistente Diseño"
                new_text = prev_text + "\n" + combined
                cell.comment = create_large_comment(new_text, author)
            else:
                cell.comment = create_large_comment(combined, "Asistente Diseño")
            # No forzar "visible"; el objetivo es que el tooltip (hover) respete el tamaño
        except Exception:
            # Si por alguna razón no se puede asignar el comentario, continuar sin bloquear
            pass

    wb.save(ruta_salida)

    # ---- Añadir hojas Verificacion y Resumen (sin tocar hoja original)
    with pd.ExcelWriter(
        ruta_salida, engine="openpyxl", mode="a", if_sheet_exists="replace"
    ) as writer:
        # Verificacion
        verif_sheet = pd.concat([results, sev_df], axis=1)
        verif_sheet.to_excel(writer, index=False, sheet_name="Verificacion")
        # Resumen
        resumen_rows = [
            {
                "Clave": k,
                "Columna utilizada": (v if v in df.columns else "(no encontrada)"),
            }
            for k, v in cols.items()
        ]
        resumen_df = pd.DataFrame(resumen_rows)
        resumen_df.to_excel(writer, index=False, sheet_name="Resumen")
        ws_res = writer.book["Resumen"]
        counts = {
            "MTOW - ajuste (2%-10%)": int((sev_df["sev_MTOW"] == "ajuste").sum()),
            "MTOW - no confiable (>10%)": int(
                (sev_df["sev_MTOW"] == "no_confiable").sum()
            ),
            "AR - ajuste (2%-10%)": int((sev_df["sev_AR"] == "ajuste").sum()),
            "AR - no confiable (>10%)": int((sev_df["sev_AR"] == "no_confiable").sum()),
            "IAS_TAS - ajuste (2%-10%)": int((sev_df["sev_IAS_TAS"] == "ajuste").sum()),
            "IAS_TAS - no confiable (>10%)": int(
                (sev_df["sev_IAS_TAS"] == "no_confiable").sum()
            ),
            "R - ajuste (2%-10%)": int((sev_df["sev_R"] == "ajuste").sum()),
            "R - no confiable (>10%)": int((sev_df["sev_R"] == "no_confiable").sum()),
        }
        start_row = resumen_df.shape[0] + 3
        ws_res.cell(row=start_row, column=1, value="Conteos por severidad")
        r = start_row + 1
        for k, v in counts.items():
            ws_res.cell(row=r, column=1, value=k)
            ws_res.cell(row=r, column=2, value=v)
            r += 1

    return ruta_salida
