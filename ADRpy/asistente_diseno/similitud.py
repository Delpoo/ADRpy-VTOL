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
from typing import Dict, Iterable, Optional, Tuple, List

import numpy as np
import pandas as pd

# Reutilizamos el módulo de outliers para IQR/MAD robustos
from asistente_diseno.outliers import compute_iqr_bounds, mad_mask


# =============================================================================
# Utilidades de nombres y tipos
# =============================================================================


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
            return f"{v:.3f}"
        except Exception:
            return v

    sty = (
        dfv.style.format({"distancia": "{:.3f}", "similitud": _format_sim})
        .background_gradient(subset=["similitud"], cmap="Greens")
        .background_gradient(subset=["distancia"], cmap="Reds_r")
    )
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
    restricciones: Dict[str, dict],
    *,
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
        "distancia": "{:.3f}",
        "distancia_media": "{:.3f}",
        "similitud": "{:.3f}",
        "similitud_media": "{:.3f}",
    }
    for c in activos:
        fmt_cols[f"Δ_{c}"] = "{:.3f}"

    # Aplicar formato columna por columna para mayor compatibilidad de tipos
    sty = dfv.style
    for col, fmt in fmt_cols.items():
        if col in dfv.columns:
            sty = sty.format({col: fmt})

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
    Widget (ipywidgets) para filtrar/ordenar el ranking sin recalcular:
      - filtro por segmento (si existe)
      - similitud mínima
      - búsqueda por texto en 'aeronave'
      - orden por 'similitud'/'distancia' o cualquier columna activa
      - top-N
      - botón 'Limpiar'
    (Arreglo: convierte 'segmento' a str para evitar TypeError en sorted/int vs str)
    Además: salida con scroll vertical/horizontal para ver tablas grandes sin desbordes.
    """
    try:
        import ipywidgets as w
        from IPython.display import display, clear_output
    except Exception as e:
        raise RuntimeError("Este widget requiere 'ipywidgets' instalado.") from e

    activos = [
        c
        for c, r in restricciones.items()
        if r.get("tipo", "ignorar") != "ignorar" and c in df_ranked.columns
    ]

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
    # Selector de dirección de orden (Dropdown para coincidir con la UI del screenshot)
    dd_sort_dir = w.Dropdown(
        options=["Asc", "Desc"],
        value="Desc",  # por defecto descendente (útil para 'similitud')
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
    out = w.Output(layout=w.Layout(width="100%"))
    # Contenedor con scroll para tablas grandes
    out_container = w.Box(
        [out],
        layout=w.Layout(
            width="100%",
            max_height=f"{int(max_height_px)}px",
            overflow_y="auto",
            overflow_x="auto",
            border="1px solid #ddd",
        ),
    )

    # Índice de la fila "fijada" (primer fila del ranking, p.ej., "Objetivo (usuario)")
    pinned_idx = df_ranked.index[0] if len(df_ranked.index) > 0 else None

    def _filtrar():
        dfv = df_ranked.copy()
        if tiene_seg and dd_seg.value != "Todos":
            # forzar str en el DF para comparar contra el valor del dropdown (str)
            dfv = dfv[dfv["segmento"].astype(str) == dd_seg.value]
        if "similitud" in dfv.columns:
            dfv = dfv[dfv["similitud"] >= sl_sim.value]
        if txt_busca.value.strip():
            cad = txt_busca.value.strip().lower()
            dfv = dfv[
                dfv["aeronave"].astype(str).str.lower().str.contains(cad, na=False)
            ]
        # Ordenar respetando la primera fila "fijada" si está presente tras los filtros
        if dd_sort.value in dfv.columns:
            asc = dd_sort_dir.value == "Asc"
            if pinned_idx is not None and pinned_idx in dfv.index:
                pinned_row = dfv.loc[[pinned_idx]]
                resto = dfv.drop(index=pinned_idx)
                resto_sorted = resto.sort_values(by=dd_sort.value, ascending=asc)
                # Armar respetando Top-N
                n = max(int(sl_top.value), 1)
                take = max(n - 1, 0)
                dfv = pd.concat([pinned_row, resto_sorted.head(take)])
            else:
                dfv = dfv.sort_values(by=dd_sort.value, ascending=asc)
                dfv = dfv.head(sl_top.value)
        else:
            dfv = dfv.head(sl_top.value)
        return dfv

    def _render(*args):
        with out:
            clear_output(wait=True)
            dfv = _filtrar()
            sty = vista_topn_detallada(dfv, restricciones, top_n=len(dfv))
            display(sty)

    def _clear(_):
        dd_seg.value = "Todos"
        sl_sim.value = 0.0
        txt_busca.value = ""
        dd_sort.value = sort_default
        sl_top.value = top_n_default
        dd_sort_dir.value = "Desc"

    # eventos
    for wdg in [dd_seg, sl_sim, txt_busca, dd_sort, dd_sort_dir, sl_top]:
        wdg.observe(_render, names="value")
    btn_clear.on_click(_clear)

    # primera render
    _render()

    controls1 = w.HBox([dd_seg, sl_sim])
    controls2 = w.HBox([txt_busca, dd_sort, dd_sort_dir, sl_top, btn_clear])
    box = w.VBox([controls1, controls2, out_container])
    return box
