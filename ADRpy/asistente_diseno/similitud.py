"""
Motor de similitud (ranking) para ADRpy.

Propósito
---------
Ordenar el dataset por "parecido" al diseño parcial del usuario SIN filtrar filas.
Se apoya en una distancia robusta (normalizada por IQR/MAD) y tolera NaN.

Ideas clave
-----------
- El usuario define, por parámetro (columna), una *restricción* con un "tipo":
    * 'ignorar': no contribuye a la distancia.
    * 'fijo'   : igualar al valor (distancia = |x - v| / escala).
    * 'objetivo': 0 dentro de ±tol; fuera: (|x - v| - tol) / escala.
    * 'maximo' : 0 si x ≤ v; penaliza exceso (x - v)/escala si x>v.
    * 'minimo' : 0 si x ≥ v; penaliza faltante (v - x)/escala si x<v.
    * 'rango'  : 0 si min ≤ x ≤ max; penaliza la distancia al borde más cercano si está fuera.
- La distancia total es suma ponderada de contribuciones por columna.
- Se normaliza por una **escala robusta** por columna (IQR por defecto; fallback a MAD/STD).
- Si un valor del dataset es NaN en una columna usada, se aplica una penalidad configurable.
- Similitud = exp(-alpha * distancia), en [0,1].

API principal
-------------
- preparar_escalas(df, columnas, metodo='IQR'): escala robusta por columna.
- rank(df, restricciones, ...): devuelve df ordenado por distancia con detalles.
- vista_topn_en_notebook(...): Styler para ver el Top-N prolijo en el notebook.

Ejemplo rápido
--------------
restricciones = {
    "Peso máximo al despegue (MTOW)": {"tipo": "fijo", "valor": 20.0, "peso": 1.0},
    "Payload": {"tipo": "objetivo", "valor": 5.0, "tol": 0.5, "peso": 1.0},
    "Velocidad a la que se realiza el crucero (m/s TAS)": {"tipo": "minimo", "valor": 22.0, "peso": 0.7},
    "Autonomía de la aeronave (h)": {"tipo": "rango", "min": 2.0, "max": 8.0, "peso": 0.5},
}

df_rank = rank(df, restricciones, top_n=10, name_col="Modelo")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple, List, Literal

import numpy as np
import pandas as pd

# Fix matplotlib backend issues before pandas styling operations
from asistente_diseno.mplutils import fix_matplotlib_backend, style_df_2dec

fix_matplotlib_backend()

# Reutilizamos el módulo de outliers para IQR/MAD robustos
from asistente_diseno.outliers import compute_iqr_bounds, mad_mask


# =============================================================================
# Utilidades de nombres y tipos
# =============================================================================

# Representación de un parámetro dinámico seleccionado en la UI (opcional)
# Ampliamos modos para cubrir lo que usa la UI: objetivo (valor/tol) y rango (min/max)
Modo = Literal["ignorar", "minimo", "maximo", "fijo", "objetivo", "rango"]


@dataclass
class ParamSpec:
    col: str
    label: str
    active: bool
    mode: Modo
    value: Optional[float]
    weight: float
    # nuevos campos opcionales para modos richer
    tol: Optional[float] = None  # para 'objetivo'
    min_value: Optional[float] = None  # para 'rango'
    max_value: Optional[float] = None  # para 'rango'


def _params_to_restricciones(params: Optional[List[ParamSpec]]) -> Dict[str, dict]:
    restr: Dict[str, dict] = {}
    if not params:
        return restr
    for p in params:
        if (
            not p.active
            or p.mode == "ignorar"
            or (p.weight is not None and p.weight <= 0)
        ):
            continue
        r = {"tipo": p.mode, "peso": float(p.weight)}
        try:
            if p.mode in ("minimo", "maximo", "fijo") and p.value is not None:
                r["valor"] = float(p.value)
            elif p.mode == "objetivo" and p.value is not None:
                r["valor"] = float(p.value)
                if p.tol is not None:
                    r["tol"] = float(p.tol)
            elif p.mode == "rango":
                if p.min_value is not None:
                    r["min"] = float(p.min_value)
                if p.max_value is not None:
                    r["max"] = float(p.max_value)
        except Exception:
            # si algún campo viene mal tipado, ignoramos silenciosamente ese extra
            pass
        restr[p.col] = r
    return restr


def _detectar_columna_nombre(
    df: pd.DataFrame, candidatos: list[str] | None = None
) -> str | None:
    """
    Heurística para detectar la columna con el "nombre" del modelo/aeronave.
    """
    if candidatos is None:
        candidatos = [
            "Modelo",
            "Modelo/Designación",
            "Nombre",
            "Aeronave",
            "Aircraft",
            "Model",
            "Designation",
            "Name",
        ]
    lower_map = {str(c).lower(): c for c in df.columns}
    for cand in candidatos:
        key = cand.lower()
        for col_lower, col_real in lower_map.items():
            if key in col_lower:
                return col_real
    return None


# =============================================================================
# Escalas robustas por columna
# =============================================================================


@dataclass
class Escala:
    centro: float  # típico: mediana
    escala: float  # típico: IQR (o MAD/STD si IQR≈0)
    metodo: str  # 'IQR' | 'MAD' | 'STD'
    usable: bool  # True si escala > 0


def _fallback_scale(s: pd.Series) -> Tuple[float, float, str]:
    """
    Fallback si IQR no sirve: intenta MAD; si tampoco, STD; si no, 1.0.
    """
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return (0.0, 1.0, "CONST1")
    med = float(np.nanmedian(s))
    mad = float(np.nanmedian(np.abs(s - med)))
    if np.isfinite(mad) and mad > 0:
        # MAD no es directamente una escala "en unidades" como IQR; usamos MAD*1.4826 aprox. a sigma
        return (med, mad * 1.4826, "MAD")
    std = float(np.nanstd(s, ddof=1)) if s.size >= 2 else 0.0
    if np.isfinite(std) and std > 0:
        return (float(np.nanmedian(s)), std, "STD")
    # Todo cero o constante
    val = float(np.nanmedian(s)) if s.size else 0.0
    return (val, 1.0, "CONST1")


def preparar_escalas(
    df: pd.DataFrame, columnas: Iterable[str], *, metodo: str = "IQR", min_n: int = 5
) -> Dict[str, Escala]:
    """
    Calcula 'centro' y 'escala' por columna de forma robusta.
    - metodo='IQR': intenta IQR; si IQR<=0 o n<min_n → fallback MAD/STD/1.0
    """
    out: Dict[str, Escala] = {}
    for col in columnas:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        centro = float(np.nanmedian(s))
        if metodo.upper() == "IQR":
            info = compute_iqr_bounds(s, factor=1.5, min_n=min_n)
            if info["usable"] and np.isfinite(info["IQR"]) and info["IQR"] > 0:
                esc = float(info["IQR"])
                out[col] = Escala(centro=centro, escala=esc, metodo="IQR", usable=True)
                continue
        # Fallbacks
        c, e, m = _fallback_scale(s)
        usable = e > 0
        out[col] = Escala(centro=c, escala=e, metodo=m, usable=usable)
    return out


# =============================================================================
# Distancia por columna según restricción
# =============================================================================


def _distancia_col(
    x: float | None, restr: dict, esc: Escala, *, eps: float = 1e-9
) -> Tuple[float, bool]:
    """
    Calcula la contribución a la distancia de UNA columna, y si hay violación de restricción.

    Returns
    -------
    (dv, violado)
      dv: contribución (>=0) ya normalizada por escala y ponderada por 'peso' si viene en restricción.
      violado: True si la regla se incumple (sirve para marcar).
    """
    peso = float(restr.get("peso", 1.0))
    tipo = str(restr.get("tipo", "ignorar")).lower().strip()

    if tipo == "ignorar" or peso <= 0:
        return (0.0, False)

    # Si x no está (NaN)
    if x is None or (isinstance(x, float) and (not np.isfinite(x))):
        # Penalización por ausencia se maneja afuera (para poder diferenciar según config)
        return (0.0, False)

    val = restr.get("valor", None)
    tol = restr.get("tol", 0.0)
    minv = restr.get("min", None)
    maxv = restr.get("max", None)

    escala = max(float(esc.escala), eps)

    dv_base = 0.0
    viol = False

    if tipo == "fijo":
        if val is None:
            return (0.0, False)
        dv_base = abs(x - float(val)) / escala
        # No hay "violación" estricta; todo se mide por distancia
    elif tipo == "objetivo":
        if val is None:
            return (0.0, False)
        tol = abs(float(tol)) if tol is not None else 0.0
        d = abs(x - float(val))
        if d <= tol:
            dv_base = 0.0
        else:
            dv_base = (d - tol) / escala
            viol = True
    elif tipo == "maximo":
        if val is None:
            return (0.0, False)
        exceso = x - float(val)
        if exceso <= 0:
            dv_base = 0.0
        else:
            dv_base = exceso / escala
            viol = True
    elif tipo == "minimo":
        if val is None:
            return (0.0, False)
        faltante = float(val) - x
        if faltante <= 0:
            dv_base = 0.0
        else:
            dv_base = faltante / escala
            viol = True
    elif tipo == "rango":
        if minv is None or maxv is None:
            return (0.0, False)
        if minv <= x <= maxv:
            dv_base = 0.0
        elif x < minv:
            dv_base = (float(minv) - x) / escala
            viol = True
        else:  # x > maxv
            dv_base = (x - float(maxv)) / escala
            viol = True
    else:
        # Tipo desconocido → ignorar
        return (0.0, False)

    return (peso * max(dv_base, 0.0), viol)


# =============================================================================
# Ranking principal
# =============================================================================


# =============================================================================
# Vista para notebook (Top-N bonito)
# =============================================================================


def vista_ranking(df_rank: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """Styler del ranking con formato homogéneo a 2 decimales en todas las columnas numéricas.

    - No altera los valores del DataFrame, solo la presentación.
    - Aplica gradientes suaves para 'distancia' (Reds_r) y 'similitud' (Greens) si existen.
    """
    dfv = df_rank.copy()

    # Asegurar matplotlib para Styler (colormaps/norm) y backend válido
    try:
        from asistente_diseno.mplutils import ensure_matplotlib_for_styler

        ensure_matplotlib_for_styler()
    except Exception:
        pass

    sty = style_df_2dec(dfv)
    cols_rojas = [c for c in ["distancia", "distancia_media"] if c in dfv.columns]
    cols_verdes = [c for c in ["similitud", "similitud_media"] if c in dfv.columns]
    if cols_rojas:
        sty = sty.background_gradient(subset=cols_rojas, cmap="Reds_r")
    if cols_verdes:
        sty = sty.background_gradient(subset=cols_verdes, cmap="Greens")
    return sty


def vista_topn_en_notebook(
    df_ranked: pd.DataFrame,
    *,
    mostrar_cols: Optional[List[str]] = None,
    top_n: int = 10,
) -> "pd.io.formats.style.Styler":
    """
    Devuelve un Styler con las columnas clave (aeronave, distancia, similitud) y
    las columnas que vos elijas (mostrar_cols). Resalta distancia baja y similitud alta.
    """
    cols_base = [
        c
        for c in ["aeronave", "segmento", "distancia", "similitud"]
        if c in df_ranked.columns
    ]
    cols = cols_base[:]
    if mostrar_cols:
        for c in mostrar_cols:
            if c in df_ranked.columns and c not in cols:
                cols.append(c)

    dfv = df_ranked.head(top_n)[cols].copy()

    def _format_sim(v):
        try:
            return f"{v:.2f}"
        except Exception:
            return v

    # Asegurar matplotlib para Styler (colormaps/norm) y backend válido
    try:
        import os as _os

        # Sanitizar backend inline inválido en entornos no soportados
        if _os.environ.get("MPLBACKEND", "").startswith("module://"):
            _os.environ["MPLBACKEND"] = "Agg"
        from asistente_diseno.mplutils import (
            ensure_matplotlib_for_styler as _ens,
            fix_matplotlib_backend as _fix,
        )

        _fix()
        _ens()
    except Exception:
        pass

    sty = style_df_2dec(dfv)
    # Forzar 2 decimales en similitud y distancia
    try:
        sty = sty.format({"similitud": _format_sim, "distancia": "{:.2f}"})
    except Exception:
        pass
    sty = sty.background_gradient(subset=["similitud"], cmap="Greens")
    sty = sty.background_gradient(subset=["distancia"], cmap="Reds_r")
    return sty


# =============================================================================
# Mapeo de segmento y utilidades de presentación
# =============================================================================


def _aplicar_segment_labels(
    df: pd.DataFrame, seg_col: str | None, segment_labels: dict | None
) -> pd.Series | None:
    """
    Devuelve una serie 'segmento' legible si hay columna y labels; si no, None.
    """
    if not seg_col or seg_col not in df.columns:
        return None
    if segment_labels is None:
        return df[seg_col]
    return df[seg_col].map(lambda v: segment_labels.get(v, v))


# === Mejora: medias normalizadas y labels de segmento directamente en rank() ===
#    (si ya definiste rank antes, podés reemplazar la función por esta versión;
#     si no querés tocar rank, podés saltar esto y usar sólo las vistas de abajo.)


def rank(
    df: pd.DataFrame,
    restricciones: Optional[Dict[str, dict]] = None,
    *,
    params: Optional[List[ParamSpec]] = None,
    columnas_activas: Optional[Iterable[str]] = None,
    metodo_escala: str = "IQR",
    min_n: int = 5,
    penalizar_nan: bool = True,
    penalidad_nan: float = 1.0,
    alpha: float = 1.0,
    # Segmentación y etiquetas
    segmentar_por: Optional[str] = None,  # nombre de columna p/mostrar "segmento"
    segment_labels: Optional[dict] = None,  # mapeo opcional (p.ej. 1->"Vigilancia")
    # NUEVO: modo de segmentación en el cálculo
    segmentar_modo: str = "off",  # "off" | "filter" | "prefer"
    segmentar_valor: Optional[
        object
    ] = None,  # valor de segmento a usar en "filter"/"prefer"
    prefer_factor: float = 1.3,  # factor multiplicativo para "prefer" (dist *= factor si no coincide)
    name_col: Optional[str] = None,
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """
    Calcula ranking por similitud.
    - 'off': usa todo el dataset (por defecto).
    - 'filter': restringe el dataset a filas cuyo 'segmentar_por' == 'segmentar_valor'.
    - 'prefer': usa todo el dataset, pero multiplica la distancia por 'prefer_factor'
                a las filas cuyo segmento != 'segmentar_valor'.

    El resto del comportamiento es idéntico a la V2 (distancia_media/similitud_media, etc.).
    """
    # Si vienen params dinámicos y NO se pasaron 'restricciones' explícitas,
    # generamos el dict desde params. Si ya hay 'restricciones', las respetamos.
    if params is not None and not restricciones:
        restricciones = _params_to_restricciones(params)
    else:
        restricciones = restricciones or {}

    # Determinar columnas a usar
    cols_restr = [c for c in restricciones.keys() if c in df.columns]
    if columnas_activas is not None:
        columnas_activas = [c for c in columnas_activas if c in cols_restr]
    else:
        columnas_activas = cols_restr[:]

    # Modo de segmentación: prefiltrado si corresponde
    base = df
    seg_col = segmentar_por if (segmentar_por in df.columns) else None

    # Si hay mapeo de labels, lo prearmamos para comparar en "filter"/"prefer"
    def _seg_value_raw_to_label(v):
        return segment_labels.get(v, v) if segment_labels else v

    if seg_col and segmentar_modo == "filter" and segmentar_valor is not None:
        # comparar contra el valor ya mapeado si corresponde
        mask = base[seg_col].map(_seg_value_raw_to_label) == segmentar_valor
        base = base.loc[mask].copy()

    # Escalas robustas
    escalas = preparar_escalas(
        base, columnas_activas, metodo=metodo_escala, min_n=min_n
    )

    # Detección de nombre y segmento
    if name_col is None:
        name_col = _detectar_columna_nombre(base)

    out = base.copy()
    if name_col in out.columns:
        out["aeronave"] = out[name_col]
    elif name_col is not None:
        out["aeronave"] = out.index.astype(str)

    if seg_col:
        out["segmento"] = _aplicar_segment_labels(out, seg_col, segment_labels)
        # uniformar tipo a str para facilitar filtros posteriores
        out["segmento"] = out["segmento"].astype(str)

    # Cálculo fila a fila
    dist_total = np.zeros(len(out), dtype=float)
    nan_count = np.zeros(len(out), dtype=int)

    for col in columnas_activas:
        s = pd.to_numeric(out[col], errors="coerce")
        e = escalas.get(
            col,
            Escala(
                centro=float(np.nanmedian(s)), escala=1.0, metodo="CONST1", usable=True
            ),
        )
        restr = restricciones[col]

        d_col = np.zeros(len(out), dtype=float)
        viol_col = np.zeros(len(out), dtype=bool)

        for i, x in enumerate(s.values):
            if pd.isna(x):
                nan_count[i] += 1
                continue
            dv, viol = _distancia_col(float(x), restr, e)
            d_col[i] = dv
            viol_col[i] = viol

        out[f"dv_{col}"] = d_col
        out[f"viol_{col}"] = viol_col
        dist_total += d_col

    if penalizar_nan and penalidad_nan > 0:
        dist_total += penalidad_nan * nan_count

    # Preferencia por segmento (penalización multiplicativa)
    if seg_col and segmentar_modo == "prefer" and segmentar_valor is not None:
        seg_series = out["segmento"].astype(str)
        penal_mask = seg_series != str(segmentar_valor)
        dist_total = dist_total * np.where(penal_mask, float(prefer_factor), 1.0)

    out["distancia"] = dist_total
    out["similitud"] = np.exp(-alpha * out["distancia"])

    n_activos = max(len(columnas_activas), 1)
    out["distancia_media"] = out["distancia"] / n_activos
    out["similitud_media"] = np.exp(-alpha * out["distancia_media"])

    out_sorted = out.sort_values(by=["distancia", "similitud"], ascending=[True, False])
    if top_n is not None and top_n > 0:
        out_sorted = out_sorted.head(top_n).copy()
    return out_sorted


# =============================================================================
# Insertar fila "Objetivo (usuario)" arriba de la tabla
# =============================================================================


def _valor_objetivo_para_tabla(restr: dict) -> float | None:
    """
    Devuelve un número representativo para mostrar en la fila objetivo:
      - fijo/objetivo → valor
      - maximo/minimo → valor
      - rango → (min+max)/2
    Si no hay datos, None.
    """
    t = str(restr.get("tipo", "ignorar")).lower().strip()
    if t in {"fijo", "objetivo"} and "valor" in restr:
        return float(restr["valor"])
    if t in {"maximo", "minimo"} and "valor" in restr:
        return float(restr["valor"])
    if t == "rango" and ("min" in restr and "max" in restr):
        return (float(restr["min"]) + float(restr["max"])) / 2.0
    return None


def insertar_objetivo_en_ranking(
    df_ranked: pd.DataFrame,
    restricciones: Dict[str, dict],
    *,
    name: str = "Objetivo (usuario)",
    segment_label: str | None = None,
) -> pd.DataFrame:
    """
    Inserta como primera fila una "aeronave virtual" con tus objetivos.
    distancia=0, similitud=1, y columnas de parámetros con valores representativos.
    """
    # columnas a rellenar
    cols_nums = [c for c in restricciones.keys() if c in df_ranked.columns]
    # Construimos la fila como dict[str, object] para permitir mezcla de tipos
    fila: dict[str, object] = {
        c: _valor_objetivo_para_tabla(restricciones[c]) for c in cols_nums
    }
    fila["aeronave"] = name
    fila["distancia"] = 0.0
    fila["similitud"] = 1.0
    fila["distancia_media"] = 0.0
    fila["similitud_media"] = 1.0
    if "segmento" in df_ranked.columns:
        fila["segmento"] = segment_label if segment_label is not None else "—"

    # dv_/viol_ a cero/Falso si existen
    for c in cols_nums:
        col_dv = f"dv_{c}"
        col_viol = f"viol_{c}"
        if col_dv in df_ranked.columns:
            fila[col_dv] = 0.0
        if col_viol in df_ranked.columns:
            fila[col_viol] = False

    # armar DF y concatenar arriba (alineando columnas para evitar FutureWarning)
    df_obj = pd.DataFrame([fila])
    # si el ranking está vacío, devolver solo la fila objetivo
    if df_ranked is None or df_ranked.empty:
        # reordenamos columnas si fuera posible
        try:
            df_obj = df_obj.reindex(columns=df_ranked.columns)
        except Exception:
            pass
        return df_obj.copy()

    df_obj_aligned = df_obj.reindex(columns=df_ranked.columns)
    out = pd.concat([df_obj_aligned, df_ranked], ignore_index=True)
    return out


# =============================================================================
# Vista detallada Top-N con Δ y violaciones por parámetro
# =============================================================================


def _delta_parametro(x: float | None, restr: dict) -> float:
    """
    Devuelve desvío con signo respecto al objetivo:
      - fijo/objetivo: x - valor (pero 0 si |x-valor|<=tol)
      - maximo: max(0, x - valor)
      - minimo: min(0, x - valor)   (negativo si está por debajo)
      - rango : 0 dentro; x-min (negativo) si bajo; x-max (positivo) si alto
    Si x es None/NaN → 0 (no contamina la vista).
    """
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return 0.0
    t = str(restr.get("tipo", "ignorar")).lower().strip()
    if t in {"ignorar"}:
        return 0.0
    v = restr.get("valor", None)
    tol = float(restr.get("tol", 0.0) or 0.0)
    if t in {"fijo", "objetivo"} and v is not None:
        d = float(x) - float(v)
        return 0.0 if abs(d) <= tol else d
    if t == "maximo" and v is not None:
        d = float(x) - float(v)
        return d if d > 0 else 0.0
    if t == "minimo" and v is not None:
        d = float(x) - float(v)
        return d if d < 0 else 0.0
    if t == "rango" and ("min" in restr and "max" in restr):
        lo, hi = float(restr["min"]), float(restr["max"])
        if lo <= float(x) <= hi:
            return 0.0
        return float(x) - hi if x > hi else float(x) - lo
    return 0.0


def vista_topn_detallada(
    df_ranked: pd.DataFrame, restricciones: Dict[str, dict], *, top_n: int = 10
) -> "pd.io.formats.style.Styler":
    """
    Muestra:
      - aeronave, segmento, distancia/similitud y *también* distancia_media/similitud_media
      - por parámetro activo: Δ_<col> y ⚠_<col> (violación)
      - los Δ se colorean por |dv_<col>| (cuanto más aporta a la distancia, más fuerte)
    """
    activos = [
        c
        for c, r in restricciones.items()
        if r.get("tipo", "ignorar") != "ignorar" and c in df_ranked.columns
    ]
    base_cols = [
        c
        for c in [
            "aeronave",
            "segmento",
            "alerta",  # NUEVO: si main inyecta un resumen de alertas por objetivo
            "distancia",
            "distancia_media",
            "similitud",
            "similitud_media",
        ]
        if c in df_ranked.columns
    ]
    cols = base_cols[:]

    dfv = df_ranked.head(top_n).copy()

    # Agregar Δ y ⚠ por parámetro
    for c in activos:
        delta_col = f"Δ_{c}"
        flag_col = f"⚠_{c}"
        dv_col = f"dv_{c}"
        viol_col = f"viol_{c}"

        deltas = []
        for x in dfv[c].values:
            deltas.append(
                _delta_parametro(float(x) if pd.notna(x) else None, restricciones[c])
            )
        dfv[delta_col] = deltas
        if viol_col in dfv.columns:
            dfv[flag_col] = dfv[viol_col].map(lambda b: "⚠" if bool(b) else "")
        else:
            dfv[flag_col] = ""

        # Orden de columnas: valor original, Δ, ⚠
        if c not in cols:
            cols.append(c)
        cols.append(delta_col)
        cols.append(flag_col)

    dfv = dfv[cols]

    # Estilos
    fmt_cols = {
        "distancia": "{:.2f}",
        "distancia_media": "{:.2f}",
        "similitud": "{:.2f}",
        "similitud_media": "{:.2f}",
    }
    for c in activos:
        fmt_cols[f"Δ_{c}"] = "{:.2f}"
        # también formatear la columna de valor original a 2 decimales, incluso si es dtype 'object'
        if c in dfv.columns:
            # usar callable para tolerar strings/no numéricos
            try:
                from asistente_diseno.mplutils import f2 as _f2

                dfv[c]  # touch
                # aplicaremos el callable más abajo vía sty.format
            except Exception:
                pass

    # Asegurar matplotlib para Styler (colormaps/norm)
    try:
        from asistente_diseno.mplutils import ensure_matplotlib_for_styler

        ensure_matplotlib_for_styler()
    except Exception:
        pass

    # Aplicar formato a .2f + columnas especiales
    sty = style_df_2dec(dfv)
    # aplicar formatos fijos y callables
    for col, fmt in fmt_cols.items():
        if col in dfv.columns:
            sty = sty.format({col: fmt})
    # para columnas de valor de parámetros (posibles 'object'), aplicar f2 callable
    try:
        from asistente_diseno.mplutils import f2 as _f2

        for c in activos:
            if c in dfv.columns:
                sty = sty.format({c: (lambda v, _c=c: _f2(v))})
    except Exception:
        pass

    # Tooltip sobre 'alerta' para explicar causas (usa el mismo texto de la celda)
    if "alerta" in dfv.columns:
        try:
            import pandas as _pd  # safe local import name

            tt = _pd.DataFrame("", index=dfv.index, columns=dfv.columns)
            tt["alerta"] = dfv["alerta"].fillna("")
            sty = sty.set_tooltips(tt)
        except Exception:
            pass

    # Colorear por aporte |dv_col| si existe
    for c in activos:
        dv_col = f"dv_{c}"
        delta_col = f"Δ_{c}"
        if dv_col in df_ranked.columns:
            # crear una serie de pesos (0..1) normalizados para colorear Δ
            pesos = df_ranked.loc[dfv.index, dv_col].abs()
            maxp = float(pesos.max()) if len(pesos) else 0.0
            if maxp > 0:
                norm = (pesos / maxp).clip(0, 1)
                # aplicar color en función de la intensidad
                sty = sty.background_gradient(
                    subset=[delta_col], cmap="OrRd", gmap=norm
                )
        # marcar visualmente ⚠
        flag_col = f"⚠_{c}"
        if flag_col in dfv.columns:
            sty = sty.set_properties(
                subset=[flag_col], **{"color": "#b00", "font-weight": "bold"}
            )
    # Gradientes globales
    for col, cmap in [
        ("similitud", "Greens"),
        ("similitud_media", "Greens"),
        ("distancia", "Reds_r"),
        ("distancia_media", "Reds_r"),
    ]:
        if col in dfv.columns:
            sty = sty.background_gradient(subset=[col], cmap=cmap)
    # Re-aplicar formato de 2 decimales al final (algunos estilos pueden sobrescribir cache interno)
    try:
        from asistente_diseno.mplutils import f2 as _f2

        fmt_cols_final = {}
        for c in activos:
            if f"Δ_{c}" in dfv.columns:
                fmt_cols_final[f"Δ_{c}"] = "{:.2f}"
        for col, fmt in fmt_cols_final.items():
            if col in dfv.columns:
                sty = sty.format({col: fmt})
        # Aplicar f2 a columnas clave y a valores originales por parámetro
        for col in ["distancia", "distancia_media", "similitud", "similitud_media"]:
            if col in dfv.columns:
                sty = sty.format({col: (lambda v, _c=col: _f2(v))})
        for c in activos:
            if c in dfv.columns:
                sty = sty.format({c: (lambda v, _c=c: _f2(v))})
    except Exception:
        pass
    return sty


# =============================================================================
# Widget de filtrado/orden simple para el ranking
# =============================================================================


def widget_filtrado_ranking(
    df_ranked: pd.DataFrame,
    restricciones: Dict[str, dict],
    *,
    sort_default: str = "similitud",
    top_n_default: int = 15,
    max_height_px: int = 600,
):
    """
    Widget interactivo para explorar el ranking de similitud.

    Usa ipydatagrid (DataGrid) para la tabla, lo que provee nativamente:
      - sort por click en encabezados
      - resize de columnas arrastrando bordes
      - scroll suave horizontal/vertical

    Controles ipywidgets arriba:
      - filtro por segmento, similitud mínima, búsqueda texto
      - orden por columna + dirección
      - top-N
      - panel colapsable para ocultar/reordenar columnas
    """
    try:
        import ipywidgets as w
        from IPython.display import display, clear_output
    except Exception as e:
        raise RuntimeError("Este widget requiere 'ipywidgets' instalado.") from e

    # Intentar importar ipydatagrid; si no está, fallback a tabla HTML
    try:
        from ipydatagrid import DataGrid, TextRenderer, BarRenderer, Expr

        _HAS_DATAGRID = True
    except ImportError:
        _HAS_DATAGRID = False

    activos = [
        c
        for c, r in restricciones.items()
        if r.get("tipo", "ignorar") != "ignorar" and c in df_ranked.columns
    ]

    # ── Enriquecer df_ranked con Δ_ y ⚠_ (misma lógica que vista_topn_detallada) ─
    for c in activos:
        delta_col = f"Δ_{c}"
        flag_col = f"⚠_{c}"
        viol_col = f"viol_{c}"
        if delta_col not in df_ranked.columns:
            df_ranked[delta_col] = [
                _delta_parametro(float(x) if pd.notna(x) else None, restricciones[c])
                for x in df_ranked[c].values
            ]
        if flag_col not in df_ranked.columns:
            if viol_col in df_ranked.columns:
                df_ranked[flag_col] = df_ranked[viol_col].map(
                    lambda b: "⚠" if bool(b) else ""
                )
            else:
                df_ranked[flag_col] = ""

    # ── Clasificación de columnas por grupo ──────────────────────────────
    _COL_ID = [c for c in ["aeronave", "segmento"] if c in df_ranked.columns]
    _has_alerta = "alerta" in df_ranked.columns
    _COL_METRICS = [
        c
        for c in ["distancia", "distancia_media", "similitud", "similitud_media"]
        if c in df_ranked.columns
    ]
    _COL_PARAMS = [c for c in activos if c in df_ranked.columns]
    _COL_DELTA = [f"Δ_{c}" for c in activos if f"Δ_{c}" in df_ranked.columns]
    _COL_VIOL_FLAG = [f"⚠_{c}" for c in activos if f"⚠_{c}" in df_ranked.columns]
    _COL_DV = [f"dv_{c}" for c in activos if f"dv_{c}" in df_ranked.columns]
    _COL_VIOL_BOOL = [f"viol_{c}" for c in activos if f"viol_{c}" in df_ranked.columns]
    _all_known = set(
        _COL_ID
        + _COL_METRICS
        + _COL_PARAMS
        + _COL_DELTA
        + _COL_VIOL_FLAG
        + _COL_DV
        + _COL_VIOL_BOOL
    )
    _COL_OTHER = [c for c in df_ranked.columns if c not in _all_known]

    _default_order: list[str] = (
        _COL_ID
        + _COL_METRICS
        + _COL_PARAMS
        + _COL_DELTA
        + _COL_VIOL_FLAG
        + _COL_DV
        + _COL_VIOL_BOOL
        + _COL_OTHER
    )
    # Ocultar por defecto: dv_, viol_ (internals) y Otras
    _default_hidden: set[str] = set(_COL_DV + _COL_VIOL_BOOL + _COL_OTHER)
    _default_visible: list[str] = [
        c for c in _default_order if c not in _default_hidden
    ]

    # Estado mutable de visibilidad y orden
    _col_state: dict = {
        "visible": list(_default_visible),
        "order": list(_default_order),
    }

    # opciones de orden
    opciones_orden = [
        "similitud",
        "distancia",
        "similitud_media",
        "distancia_media",
    ] + activos
    opciones_orden = [c for c in opciones_orden if c in df_ranked.columns]
    sort_default = sort_default if sort_default in opciones_orden else opciones_orden[0]

    # segmento (forzar str para evitar mezcla de tipos)
    tiene_seg = "segmento" in df_ranked.columns
    if tiene_seg:
        seg_series = df_ranked["segmento"].dropna().astype(str)
        seg_values = ["Todos"] + sorted(seg_series.unique().tolist())
    else:
        seg_values = ["Todos"]

    # ── Controles de filtrado ────────────────────────────────────────────
    dd_seg = w.Dropdown(
        options=seg_values,
        description="Segmento:",
        value="Todos",
        layout=w.Layout(width="35%"),
    )
    sl_sim = w.FloatSlider(
        value=0.0,
        min=0.0,
        max=1.0,
        step=0.01,
        description="Sim mín:",
        readout_format=".2f",
        layout=w.Layout(width="45%"),
    )
    txt_busca = w.Text(value="", description="Buscar:", layout=w.Layout(width="45%"))
    dd_sort = w.Dropdown(
        options=opciones_orden,
        value=sort_default,
        description="Orden:",
        layout=w.Layout(width="35%"),
    )
    dd_sort_dir = w.Dropdown(
        options=["Asc", "Desc"],
        value="Desc",
        description="Dirección:",
        layout=w.Layout(width="25%"),
    )
    sl_top = w.IntSlider(
        value=top_n_default,
        min=5,
        max=50,
        step=1,
        description="Top-N:",
        layout=w.Layout(width="35%"),
    )
    btn_clear = w.Button(
        description="Limpiar", button_style="warning", layout=w.Layout(width="15%")
    )

    # ── Panel de control de columnas (checkboxes por columna) ──────────
    _updating = {"flag": False}
    _default_visible_set = set(_default_visible)

    # Crear un checkbox por cada columna
    _col_cbs: dict = {}  # col_name → Checkbox
    for col in _default_order:
        _col_cbs[col] = w.Checkbox(
            value=(col in _default_visible_set),
            description=col,
            indent=False,
            layout=w.Layout(width="auto", min_width="140px", max_width="300px"),
        )

    # Definición de grupos: (label, col_list, default_on)
    _GROUPS_DEF = [
        ("Métricas", _COL_METRICS, True),
        ("Parámetros", _COL_PARAMS, True),
        ("Δ Desvíos", _COL_DELTA, True),
        ("⚠ Violaciones", _COL_VIOL_FLAG, True),
        ("dv_ Aportes distancia", _COL_DV, False),
        ("viol_ Booleanos", _COL_VIOL_BOOL, False),
        ("Otras", _COL_OTHER, False),
    ]

    # Para cada grupo: checkbox cabecera + checkboxes hijos en flow layout
    _grp_cbs: list = []  # [(grp_checkbox, grp_cols), ...]
    _group_sections: list = []

    for grp_label, grp_cols, grp_default in _GROUPS_DEF:
        if not grp_cols:
            continue
        grp_cb = w.Checkbox(
            value=grp_default,
            indent=False,
            layout=w.Layout(width="auto"),
        )
        grp_header = w.HBox(
            [
                grp_cb,
                w.HTML(
                    f"<b style='font-size:12px;'>{grp_label}</b>"
                    f" <span style='color:#888;font-size:11px;'>({len(grp_cols)})</span>"
                ),
            ],
            layout=w.Layout(margin="4px 0 0 0"),
        )

        col_flow = w.HBox(
            [_col_cbs[c] for c in grp_cols if c in _col_cbs],
            layout=w.Layout(
                flex_flow="row wrap",
                padding="0 0 0 24px",
            ),
        )
        section = w.VBox([grp_header, col_flow], layout=w.Layout(margin="0"))
        _grp_cbs.append((grp_cb, grp_cols))
        _group_sections.append(section)

    # Botones globales
    btn_show_all = w.Button(
        description="✓ Mostrar todo",
        button_style="success",
        layout=w.Layout(width="auto"),
    )
    btn_hide_all = w.Button(
        description="✗ Ocultar todo",
        button_style="danger",
        layout=w.Layout(width="auto"),
    )
    btn_col_reset = w.Button(
        description="⟲ Restablecer",
        button_style="warning",
        layout=w.Layout(width="auto"),
    )

    def _sync_visible():
        """Recalcula _col_state['visible'] desde checkboxes y re-renderiza."""
        vis = [c for c in _col_state["order"] if c in _col_cbs and _col_cbs[c].value]
        # Forzar columnas ID siempre
        for c in reversed(_COL_ID):
            if c not in vis:
                vis.insert(0, c)
        _col_state["visible"] = vis
        _render()

    def _on_grp_toggle(change, grp_cols):
        if _updating["flag"]:
            return
        _updating["flag"] = True
        try:
            for c in grp_cols:
                if c in _col_cbs:
                    _col_cbs[c].value = change["new"]
        finally:
            _updating["flag"] = False
        _sync_visible()

    def _on_col_toggle(change):
        if _updating["flag"]:
            return
        _updating["flag"] = True
        try:
            for grp_cb, grp_cols in _grp_cbs:
                grp_cb.value = any(_col_cbs[c].value for c in grp_cols if c in _col_cbs)
        finally:
            _updating["flag"] = False
        _sync_visible()

    # Conectar eventos
    for grp_cb, grp_cols in _grp_cbs:
        grp_cb.observe(
            lambda change, gc=grp_cols: _on_grp_toggle(change, gc), names="value"
        )
    for col, cb in _col_cbs.items():
        cb.observe(_on_col_toggle, names="value")

    def _on_show_all(_btn):
        _updating["flag"] = True
        try:
            for cb in _col_cbs.values():
                cb.value = True
            for grp_cb, _ in _grp_cbs:
                grp_cb.value = True
        finally:
            _updating["flag"] = False
        _sync_visible()

    def _on_hide_all(_btn):
        _updating["flag"] = True
        try:
            for c, cb in _col_cbs.items():
                cb.value = c in _COL_ID
            for grp_cb, _ in _grp_cbs:
                grp_cb.value = False
        finally:
            _updating["flag"] = False
        _sync_visible()

    def _on_col_reset(_btn):
        _updating["flag"] = True
        try:
            _col_state["order"] = list(_default_order)
            for c, cb in _col_cbs.items():
                cb.value = c in _default_visible_set
            for grp_cb, grp_cols in _grp_cbs:
                grp_cb.value = any(c in _default_visible_set for c in grp_cols)
        finally:
            _updating["flag"] = False
        _sync_visible()

    btn_show_all.on_click(_on_show_all)
    btn_hide_all.on_click(_on_hide_all)
    btn_col_reset.on_click(_on_col_reset)

    # ── Panel de alertas (extraído de columna 'alerta') ──────────────────
    html_alert = w.HTML(value="", layout=w.Layout(width="100%"))

    def _update_alert(dfv: pd.DataFrame):
        if not _has_alerta or "alerta" not in dfv.columns:
            html_alert.value = ""
            return
        parts: list[str] = []
        for _, row in dfv.iterrows():
            a = str(row.get("alerta", "") or "").strip()
            if a:
                name = str(row.get("aeronave", ""))
                # Separar alertas individuales (delimitadas por |)
                items = [x.strip() for x in a.split("|") if x.strip()]
                for item in items:
                    parts.append(f"• <b>{name}</b>: {item}")
        if parts:
            html_alert.value = (
                "<div style='background:#fff3cd;border:1px solid #ffc107;"
                "border-radius:4px;padding:6px 10px;margin:4px 0;font-size:12px;"
                "line-height:1.4em;'>"
                "<b>⚠ Alertas del objetivo:</b><br>" + "<br>".join(parts) + "</div>"
            )
        else:
            html_alert.value = ""

    # ── Contenedor de salida ─────────────────────────────────────────────
    out = w.Output(layout=w.Layout(width="100%"))
    out_container = w.Box(
        [out],
        layout=w.Layout(
            width="100%",
            max_height=f"{int(max_height_px)}px",
            overflow_y="visible",
            overflow_x="visible",
            border="0",
        ),
    )

    # Índice de la fila "fijada"
    pinned_idx = df_ranked.index[0] if len(df_ranked.index) > 0 else None

    def _filtrar():
        dfv = df_ranked.copy()
        if tiene_seg and dd_seg.value != "Todos":
            dfv = dfv[dfv["segmento"].astype(str) == dd_seg.value]
        if "similitud" in dfv.columns:
            dfv = dfv[dfv["similitud"] >= sl_sim.value]
        if txt_busca.value.strip():
            cad = txt_busca.value.strip().lower()
            dfv = dfv[
                dfv["aeronave"].astype(str).str.lower().str.contains(cad, na=False)
            ]
        if dd_sort.value in dfv.columns:
            asc = dd_sort_dir.value == "Asc"
            if pinned_idx is not None and pinned_idx in dfv.index:
                pinned_row = dfv.loc[[pinned_idx]]
                resto = dfv.drop(index=pinned_idx)
                resto_sorted = resto.sort_values(by=dd_sort.value, ascending=asc)
                n = max(int(sl_top.value), 1)
                take = max(n - 1, 0)
                dfv = pd.concat([pinned_row, resto_sorted.head(take)])
            else:
                dfv = dfv.sort_values(by=dd_sort.value, ascending=asc)
                dfv = dfv.head(sl_top.value)
        else:
            dfv = dfv.head(sl_top.value)
        return dfv

    def _render_datagrid(dfv: pd.DataFrame):
        """Renderizar con ipydatagrid — columnas compactas."""
        vis_cols = [c for c in _col_state["visible"] if c in dfv.columns]
        if not vis_cols:
            vis_cols = [c for c in _COL_ID if c in dfv.columns] or list(dfv.columns[:3])

        # Excluir 'alerta' de la grilla (se muestra en el banner)
        vis_cols = [c for c in vis_cols if c != "alerta"]

        df_show = dfv[vis_cols].copy()

        # Redondear numéricas
        for c in df_show.columns:
            if pd.api.types.is_numeric_dtype(df_show[c]):
                df_show[c] = df_show[c].round(2)

        # ── Renderers ────────────────────────────────────────────────────
        renderers = {}
        try:
            for sim_col in ["similitud", "similitud_media"]:
                if sim_col in vis_cols:
                    renderers[sim_col] = BarRenderer(
                        horizontal_alignment="center",
                        bar_color=Expr(
                            '"#28a745" if cell.value >= 0.7 '
                            'else ("#ffc107" if cell.value >= 0.4 else "#dc3545")'
                        ),
                        bar_value=Expr("cell.value"),
                        show_text=True,
                        text_color="black",
                        format=".2f",
                    )
            dist_renderer = TextRenderer(
                text_color="black",
                background_color=Expr(
                    '"#f8d7da" if cell.value > 1.0 '
                    'else ("#fff3cd" if cell.value > 0.3 else "#d4edda")'
                ),
                format=".2f",
                horizontal_alignment="right",
            )
            for dc in ["distancia", "distancia_media"]:
                if dc in vis_cols:
                    renderers[dc] = dist_renderer

            delta_renderer = TextRenderer(
                text_color="black",
                background_color=Expr(
                    '"#f8d7da" if abs(cell.value) > 1.0 '
                    'else ("#fff3cd" if abs(cell.value) > 0.3 else "#d4edda")'
                ),
                format=".2f",
                horizontal_alignment="right",
            )
            for c in vis_cols:
                if c.startswith("Δ_"):
                    renderers[c] = delta_renderer

            dv_renderer = TextRenderer(
                text_color="#555",
                background_color=Expr(
                    '"#f8d7da" if cell.value > 0.5 '
                    'else ("#fff3cd" if cell.value > 0.1 else "#f5f5f5")'
                ),
                format=".3f",
                horizontal_alignment="right",
            )
            for c in vis_cols:
                if c.startswith("dv_"):
                    renderers[c] = dv_renderer

            viol_bool_renderer = TextRenderer(
                text_color=Expr('"#b00" if cell.value else "#888"'),
                horizontal_alignment="center",
            )
            for c in vis_cols:
                if c.startswith("viol_"):
                    renderers[c] = viol_bool_renderer

            viol_flag_renderer = TextRenderer(
                text_color="#b00",
                font=Expr(
                    '"bold 12px sans-serif" if cell.value == "⚠" '
                    'else "12px sans-serif"'
                ),
                horizontal_alignment="center",
            )
            for c in vis_cols:
                if c.startswith("⚠_"):
                    renderers[c] = viol_flag_renderer

            num_renderer = TextRenderer(format=".2f", horizontal_alignment="right")
            for c in vis_cols:
                if c not in renderers and pd.api.types.is_numeric_dtype(df_show[c]):
                    renderers[c] = num_renderer
        except Exception:
            renderers = {}

        df_show = df_show.reset_index(drop=True)

        # ── Tamaños de columna compactos ─────────────────────────────────
        # auto_fit_columns=False → ancho uniforme salvo 'aeronave'
        n_rows = len(df_show)
        grid_h = min(max_height_px, max(200, 32 * (n_rows + 2)))

        grid = DataGrid(
            df_show,
            editable=False,
            selection_mode="row",
            auto_fit_columns=False,
            base_row_size=28,
            base_column_size=90,
            base_column_header_size=40,
            header_visibility="all",
            renderers=renderers,
            layout={"height": f"{grid_h}px", "width": "100%"},
            grid_style={
                "header_background_color": "#f0f0f0",
                "header_grid_line_color": "#ccc",
            },
        )
        display(grid)

    def _render_html_fallback(dfv: pd.DataFrame):
        """Fallback a tabla HTML estática."""
        vis_cols = [c for c in _col_state["visible"] if c in dfv.columns]
        vis_cols = [c for c in vis_cols if c != "alerta"]
        if not vis_cols:
            vis_cols = [c for c in _COL_ID if c in dfv.columns] or list(dfv.columns[:3])

        df_show = dfv[vis_cols].copy()
        sty = vista_topn_detallada(df_show, restricciones, top_n=len(df_show))
        try:
            num_cols = [
                c for c in df_show.columns if pd.api.types.is_numeric_dtype(df_show[c])
            ]
            if num_cols:
                sty = sty.format("{:.2f}", subset=num_cols)
        except Exception:
            pass
        try:
            sty = sty.set_table_attributes('class="ranktbl"')
            html_tbl = sty.to_html()
            inner = (
                "<!DOCTYPE html><html><head><meta charset='utf-8'>"
                "<style>\n"
                "html,body{margin:0;padding:8px;overflow:auto;"
                "font-family:system-ui,sans-serif;font-size:12px;color:#111;}\n"
                ".ranktbl{border-collapse:separate;border-spacing:0;"
                "border:1px solid #ddd;width:max-content;min-width:100%;}\n"
                ".ranktbl th,.ranktbl td{padding:6px 8px;"
                "border-right:1px solid #eee;border-bottom:1px solid #eee;"
                "white-space:nowrap;}\n"
                ".ranktbl thead th{position:sticky;top:0;background:#fff;"
                "z-index:3;font-weight:600;cursor:pointer;}\n"
                ".ranktbl tbody tr:nth-child(odd) td{background:#fafafa;}\n"
                ".ranktbl tbody tr:hover td{background:#f0f7ff;}\n"
                ".ranktbl td:nth-child(n+2){text-align:right;}\n"
                "</style></head><body>" + html_tbl + "</body></html>"
            )
            escaped = inner.replace("'", "&#39;")
            iframe = (
                f'<iframe style="width:100%;height:{int(max_height_px)}px;'
                f'border:1px solid #eee;border-radius:4px;" '
                f"srcdoc='{escaped}' loading=\"lazy\"></iframe>"
            )
            display(w.HTML(value=iframe))
        except Exception:
            display(sty)

    def _render(*args):
        with out:
            clear_output(wait=True)
            dfv = _filtrar()
            _update_alert(dfv)
            if _HAS_DATAGRID:
                _render_datagrid(dfv)
            else:
                _render_html_fallback(dfv)

    def _clear(_):
        dd_seg.value = "Todos"
        sl_sim.value = 0.0
        txt_busca.value = ""
        dd_sort.value = sort_default
        sl_top.value = top_n_default
        dd_sort_dir.value = "Desc"

    # ── Eventos de filtrado ──────────────────────────────────────────────
    for wdg in [dd_seg, sl_sim, txt_busca, dd_sort, dd_sort_dir, sl_top]:
        wdg.observe(_render, names="value")
    btn_clear.on_click(_clear)

    # Primera render
    _render()

    # ── Layout final ─────────────────────────────────────────────────────
    controls1 = w.HBox([dd_seg, sl_sim])
    controls2 = w.HBox([txt_busca, dd_sort, dd_sort_dir, sl_top, btn_clear])

    btn_row = w.HBox(
        [btn_show_all, btn_hide_all, btn_col_reset],
        layout=w.Layout(margin="4px 0"),
    )
    col_panel_inner = w.VBox(
        [btn_row] + _group_sections,
        layout=w.Layout(
            max_height="320px",
            overflow_y="auto",
            padding="4px",
        ),
    )
    acc_cols = w.Accordion(children=[col_panel_inner])
    acc_cols.set_title(0, "⚙ Personalizar columnas")
    acc_cols.selected_index = None  # colapsado

    help_html = w.HTML(
        "<div style='font-size:12px;line-height:1.35em;margin:4px 0 6px 0;'>"
        "<b>Similitud (ranking)</b>: ordena aeronaves por cercanía al objetivo. "
        "<b>Top-K</b>: vecinos más cercanos para sugerencias locales. "
        "Click en encabezados para ordenar · arrastrá bordes de columna "
        "para redimensionar."
        "</div>"
    )
    box = w.VBox(
        [
            help_html,
            controls1,
            controls2,
            acc_cols,
            html_alert,
            out_container,
        ]
    )
    return box
