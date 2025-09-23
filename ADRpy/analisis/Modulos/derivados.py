# -*- coding: utf-8 -*-
"""
derivados.py — Cálculo de campos derivados para completar DataFrame.

Módulo simplificado que solo se encarga de completar campos derivados
usando fórmulas, sin lógica de validación. El objetivo es enriquecer
el DataFrame después de df_procesado para tener datos más completos
antes de la imputación por correlación y similitud.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any


def completar_campos_derivados(
    df: pd.DataFrame,
    decimales: int = 2,                  # Decimales fijos después de la coma (16.59 en lugar de cifras significativas)
    sigma_5000: float = 0.8617,          # densidad relativa ISA a 5000 ft (constante)
    altitud_crucero: float = 5000,       # altitud de crucero asumida en pies para cálculos
    solo_completar_vacios: bool = True,
    usar_IAS_para_alcance: bool = True   # RESPETA tu fórmula: alcance con IAS
) -> Tuple[pd.DataFrame, Dict[tuple, Dict[str, Any]]]:
    """
    Completa columnas derivadas usando fórmulas (sin verificación).
    Devuelve (df_actualizado, origen_por_celda).

    - IAS -> TAS@5000: TAS_5000 = IAS / sqrt(sigma_5000)
    - Altitud de crucero asumida: valor fijo usado para cálculos
    - Alcance siempre en km. Si usar_IAS_para_alcance=True, usa IAS.
    - No calcular 'Área del ala' (S). Solo AR = b/c y combinaciones con b, c, AR.
    - Solo se rellenan NaN salvo que 'solo_completar_vacios' sea False.
    """
    df = df.copy()

    def round_decimals(x: float, decimales: int) -> float:
        """Redondea a decimales fijos después de la coma"""
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return x
        if x == 0:
            return 0.0
        return float(round(x, decimales))

    # === Nombres de columnas (compatibles con tu archivo) ===
    COL_B    = "Envergadura"
    COL_C    = "Cuerda"
    COL_AR   = "Relación de aspecto del ala"
    COL_W0   = "Peso Vacio (MTOW - payload)"
    COL_PL   = "Payload"
    COL_MTOW = "Peso máximo al despegue (MTOW)"
    COL_VIAS = "Velocidad crucero m/s"  # IAS (entrada)
    COL_VTAS = "Velocidad a la que se realiza el crucero (m/s TAS)"  # salida
    COL_AUTO = "Autonomía de la aeronave"  # horas
    COL_RNG  = "Alcance de la aeronave"    # salida: km
    COL_ALT  = "Altitud de crucero"        # altitud asumida en pies

    for col in [COL_AR, COL_MTOW, COL_VTAS, COL_RNG, COL_ALT]:
        if col not in df.columns:
            df[col] = np.nan

    origen_por_celda: Dict[tuple, Dict[str, Any]] = {}

    def registrar_calculado(idx, col, orig, val, fuente, formula, inputs):
        key = (idx, col)
        origen_por_celda[key] = {
            "Estado": "CALCULADO",
            "Fuente": fuente,
            "formula": formula,
            "inputs": inputs,
            "calculado": True
        }

    def escribir(idx, col, nuevo, fuente, formula, inputs):
        nuevo_r = round_decimals(nuevo, decimales) if pd.notna(nuevo) else nuevo
        orig = df.at[idx, col] if col in df.columns else np.nan
        if solo_completar_vacios and pd.notna(orig):
            return False
        # escribir el nuevo valor redondeado
        df.at[idx, col] = nuevo_r
        # Registrar como CALCULADO
        registrar_calculado(idx, col, orig, nuevo_r, fuente, formula, inputs)
        return True

    # === Utilidad: resolver exactamente una variable faltante en una relación ===
    def try_solve_relation(idx: Any, cols: list, compute_map: Dict[str, Any], fuente: str = "fórmula derivada") -> bool:
        """Si en 'cols' hay exactamente una celda NaN y existe compute_map para esa columna,
        calcula el valor faltante usando exclusivamente las demás columnas de la relación y lo escribe.
        compute_map[col] debe retornar (valor, formula_text, inputs_dict). Devuelve True si escribió algo."""
        # Verificar presencia de columnas
        for c in cols:
            if c not in df.columns:
                return False
        values = {c: df.at[idx, c] for c in cols}
        missing = [c for c in cols if pd.isna(values[c])]
        # Resolver solo cuando hay exactamente una variable faltante
        if len(missing) != 1:
            return False
        mcol = missing[0]
        func = compute_map.get(mcol)
        if func is None:
            return False
        try:
            val, formula_text, inputs = func(values)
        except Exception:
            return False
        # Validar el valor
        if val is None:
            return False
        try:
            if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                return False
        except Exception:
            return False
        return escribir(idx, mcol, val, fuente, formula_text, inputs)

    # === Inferencias iterativas entre b, c y AR (NO calcular Área del ala) ===
    def inferir_geom(idx) -> bool:
        changed = False
        cols = [COL_B, COL_C, COL_AR]
        def safe_div(a, b):
            try:
                return None if b == 0 else a / b
            except Exception:
                return None
        compute_map = {
            COL_AR: lambda v: (safe_div(v[COL_B], v[COL_C]), "AR = b/c", {"b": v[COL_B], "c": v[COL_C]}),
            COL_C:  lambda v: (safe_div(v[COL_B], v[COL_AR]), "c = b/AR", {"b": v[COL_B], "AR": v[COL_AR]}),
            COL_B:  lambda v: (None if pd.isna(v[COL_AR]) or pd.isna(v[COL_C]) else v[COL_AR] * v[COL_C], "b = AR*c", {"AR": v[COL_AR], "c": v[COL_C]}),
        }
        if try_solve_relation(idx, cols, compute_map, "fórmula derivada"):
            changed = True
        return changed

    for idx, _ in df.iterrows():
        # Repetir inferencias hasta convergencia para aprovechar los valores nuevos
        max_iter = 10
        iter_count = 0
        while True:
            iter_count += 1
            wrote = False

            # 1) Geometría (sin Área)
            if inferir_geom(idx):
                wrote = True

            # 2) Pesos: W0, Payload y MTOW (directas e inversas) con resolución por variable
            def inferir_pesos():
                ch = False
                cols_pesos = [COL_MTOW, COL_W0, COL_PL]
                compute_map_pesos = {
                    COL_MTOW: lambda v: (None if pd.isna(v[COL_W0]) or pd.isna(v[COL_PL]) else v[COL_W0] + v[COL_PL], "MTOW = W0 + Payload", {"W0": v[COL_W0], "Payload": v[COL_PL]}),
                    COL_W0:   lambda v: (None if pd.isna(v[COL_MTOW]) or pd.isna(v[COL_PL]) else v[COL_MTOW] - v[COL_PL], "W0 = MTOW - Payload", {"MTOW": v[COL_MTOW], "Payload": v[COL_PL]}),
                    COL_PL:   lambda v: (None if pd.isna(v[COL_MTOW]) or pd.isna(v[COL_W0]) else v[COL_MTOW] - v[COL_W0], "Payload = MTOW - W0", {"MTOW": v[COL_MTOW], "W0": v[COL_W0]}),
                }
                if try_solve_relation(idx, cols_pesos, compute_map_pesos, "fórmula derivada"):
                    ch = True
                return ch

            if inferir_pesos():
                wrote = True

            # 3) Velocidades: IAS <-> VTAS
            def inferir_velocidades():
                ch = False
                cols_vel = [COL_VIAS, COL_VTAS]
                root_sigma = np.sqrt(sigma_5000)
                compute_map_vel = {
                    COL_VTAS: lambda v: (None if pd.isna(v[COL_VIAS]) else v[COL_VIAS] / root_sigma, "TAS_5000 = IAS / sqrt(σ_5000)", {"IAS": v[COL_VIAS], "σ_5000": sigma_5000}),
                    COL_VIAS: lambda v: (None if pd.isna(v[COL_VTAS]) else v[COL_VTAS] * root_sigma, "IAS = VTAS * sqrt(σ_5000)", {"VTAS": v[COL_VTAS], "σ_5000": sigma_5000}),
                }
                if try_solve_relation(idx, cols_vel, compute_map_vel, "fórmula derivada"):
                    ch = True
                return ch

            if inferir_velocidades():
                wrote = True

            # 4) Alcance/autonomía/velocidad base
            def inferir_alcance():
                ch = False
                col_vel = COL_VIAS if usar_IAS_para_alcance else COL_VTAS
                cols_rng = [COL_RNG, COL_AUTO, col_vel]
                def calc_R(v):
                    return (v[COL_AUTO] * 3600.0 * v[col_vel]) / 1000.0
                def calc_h(v):
                    return (v[COL_RNG] * 1000.0) / (v[col_vel] * 3600.0)
                def calc_V(v):
                    return (v[COL_RNG] * 1000.0) / (v[COL_AUTO] * 3600.0)
                def safe_val(x):
                    try:
                        return None if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) else x
                    except Exception:
                        return None
                compute_map_rng = {
                    COL_RNG:  lambda v: (safe_val(calc_R(v)), "R[km] = h*3600*V/1000", {"Autonomía[h]": v[COL_AUTO], "Velocidad[m/s]": v[col_vel]}),
                    COL_AUTO: lambda v: (safe_val(calc_h(v)), "h = R*1000/(V*3600)", {"R[km]": v[COL_RNG], "Velocidad[m/s]": v[col_vel]}),
                    col_vel:  lambda v: (safe_val(calc_V(v)), "V = R*1000/(h*3600)", {"R[km]": v[COL_RNG], "Autonomía[h]": v[COL_AUTO]}),
                }
                if try_solve_relation(idx, cols_rng, compute_map_rng, "fórmula derivada"):
                    ch = True
                return ch

            if inferir_alcance():
                wrote = True

            # 5) Asignar altitud de crucero asumida
            def asignar_altitud_crucero():
                ch = False
                alt_val = df.at[idx, COL_ALT]
                if pd.isna(alt_val):
                    if escribir(idx, COL_ALT, altitud_crucero, "valor asumido", f"Altitud asumida = {altitud_crucero} pies", {}):
                        ch = True
                return ch

            if asignar_altitud_crucero():
                wrote = True

            # Salir si no se escribió nada o alcanzamos iteraciones máximas
            if not wrote or iter_count >= max_iter:
                break

    return df, origen_por_celda
