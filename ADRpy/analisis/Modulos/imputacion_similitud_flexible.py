"""
imputacion_similitud_flexible.py
--------------------------------
Implementa la lógica de K‑NN con 3 ejes obligatorios (físico, geométrico, prestacional)
y filtrado progresivo de familia (F0‑F3).  Diseñado para integrarse sin romper
los nombres ni los flujos que ya existen en tu proyecto ADRpy.

Uso rápido:
    python imputacion_similitud_flexible.py --ruta_excel Datos_aeronaves.xlsx \
        --aeronave "Stalker XE" \
        --parametro "Velocidad a la que se realiza el crucero (KTAS)"
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# ------------------------------------------------------------------ #
#  CONFIGURACIÓN DE BLOQUES Y CAPAS DE FAMILIA
# ------------------------------------------------------------------ #

def configurar_similitud():
    """
    Devuelve las configuraciones necesarias para ejecutar la imputación por similitud flexible:
    - bloques de rasgos
    - filas de familia
    - capas de filtrado jerárquico de familia
    """

    bloques_rasgos = {
        "fisico": [
            "Peso máximo al despegue (MTOW)",
            "Peso vacío",
        ],
        "geom": [
            "Área del ala",
            "envergadura",
            "Longitud del fuselaje",
            "Relación de aspecto del ala",
        ],
        "prest": [
            "Potencia específica (P/W)",
            "Autonomía de la aeronave",
            "Alcance de la aeronave",
            "Velocidad a la que se realiza el crucero (KTAS)",
            "Velocidad máxima (KIAS)",
        ],
    }

    filas_familia = [
        "Misión",
        "Despegue",
        "Propulsión vertical",
        "Propulsión horizontal",
        "Cantidad de motores propulsión vertical",
        "Cantidad de motores propulsión horizontal",
    ]

    capas_familia = [
        {
            "Misión": "equals",
            "Despegue": "equals",
            "Propulsión vertical": "equals",
            "Propulsión horizontal": "equals",
            "Cantidad de motores propulsión vertical": "equals",
            "Cantidad de motores propulsión horizontal": "equals",
        },
        {
            "Misión": "equals",
            "Despegue": "equals",
            "Propulsión vertical": "equals",
            "Propulsión horizontal": "equals",
        },
        {
            "Misión": "equals",
            "Despegue": "equals",
        },
        {
            "Misión": "equals",
        },
    ]

    return bloques_rasgos, filas_familia, capas_familia

# ------------------------------------------------------------------ #
#  HELPERS
# ------------------------------------------------------------------ #

def imprimir(msg, bold=False):
    prefix = "\033[1m" if bold else ""
    suffix = "\033[0m"  if bold else ""
    print(f"{prefix}{msg}{suffix}")

import pandas as pd

def zscore(arr: pd.Series) -> pd.Series:
    """
    Si la desviación estándar es cero, devolvemos un
    vector de ceros (todos idénticos a la media).
    En caso contrario, el Z‐score habitual.
    """
    std = arr.std(ddof=0)
    if std == 0 or pd.isna(std):
        # arr.index preserva el nombre de filas
        return pd.Series(0.0, index=arr.index)
    return (arr - arr.mean()) / std


# ------------------------------------------------------------------ #
#  FUNCIÓN PRINCIPAL
# ------------------------------------------------------------------ #
def imputar_por_similitud(
    df_parametros: pd.DataFrame,
    df_atributos: pd.DataFrame,
    aeronave_obj: str,
    parametro_objetivo: str,
    bloques_rasgos: dict,
    capas_familia: list
):

    imprimir(f"\n=== Iniciando imputación por similitud de aeronave {aeronave_obj} y parametro {parametro_objetivo} ===", True)

    # ------------------------ Paso 0. Validaciones ------------------
    if parametro_objetivo not in df_parametros.index:
        imprimir(f"Parámetro '{parametro_objetivo}' no encontrado.", True)
        return None

    if aeronave_obj not in df_parametros.columns:
        imprimir(f"Aeronave '{aeronave_obj}' no encontrada.", True)
        return None

    # ------------------------ Paso 1. Vector de X -------------------
    vector_objetivo = {}
    for bloque, lista in bloques_rasgos.items():
        for rasgo in lista:
            if not pd.isna(df_parametros.at[rasgo, aeronave_obj]):
                vector_objetivo[bloque] = rasgo
                break
        else:  # ningún rasgo disponible
            imprimir(f"⚠️  Bloque {bloque.upper()} sin datos en '{aeronave_obj}'.", True)
            return None

    imprimir("Vector objetivo seleccionado (se buscan parámetros con valores no nulos en celdas para realizar la imputación):")
    for bloque, rasgo in vector_objetivo.items():
        imprimir(f"  {bloque}: {rasgo}")

    # ------------------------ Paso 2. Iterar capas familia ----------
    for capa_idx, criterios in enumerate(capas_familia):
        imprimir(f"\n=== Capa F{capa_idx} ===", True)

        # Filtrado fila a fila
        mascara_cols = np.array([True] * df_parametros.shape[1])
        for fila, modo in criterios.items():
            val_obj = df_atributos.at[fila, aeronave_obj]
            if modo == "equals":
                mascara_cols &= df_atributos.loc[fila] == val_obj

        df_familia = df_parametros.loc[:, mascara_cols]
        imprimir("Realizando selección de aeronaves con valores en todos los parámetros físicos, geométricos y prestacionales:")
        n_fam = df_familia.shape[1]
        imprimir(f"Drones en familia: {n_fam}")

        if n_fam == 0:
            imprimir("❌  Sin drones en esta capa. Relajando…")
            continue

        # Filtrar los que poseen el parámetro objetivo
        cols_with_param = df_familia.columns[
            df_familia.loc[parametro_objetivo].notna()
        ]
        if cols_with_param.empty:
            imprimir(f"❌  Ningún dron en F{capa_idx} tiene '{parametro_objetivo}'.")
            continue

        # Además deben poseer los tres ejes
        filtros_ejes = [
            df_familia.loc[vector_objetivo[bloque]].notna()
            for bloque in ("fisico", "geom", "prest")
        ]
        mask_all_ejes = filtros_ejes[0] & filtros_ejes[1] & filtros_ejes[2]
        cols_validas = cols_with_param[mask_all_ejes[cols_with_param]]

        k = len(cols_validas)
        imprimir(f"Vecinos válidos (k)......................: {k}")

        if k == 0:
            imprimir("❌  No quedan drones con los 3 ejes. Relajando…")
            continue

        imprimir(f"Vecinos válidos nombres: {list(cols_validas)}")

        # ---------------- Paso 3. Distancias y media ponderada -------
        # Matriz con los tres ejes
        data_ejes = pd.DataFrame({
            col: [
                df_familia.at[vector_objetivo["fisico"], col],
                df_familia.at[vector_objetivo["geom"], col],
                df_familia.at[vector_objetivo["prest"], col],
            ]
            for col in list(cols_validas) + [aeronave_obj]
        }, index=["fis", "geo", "pre"]).T

        # Z‑score columna a columna
        imprimir("Matriz de datos antes de z-score (data_ejes):")
        imprimir(data_ejes)
        data_z = data_ejes.apply(zscore)
        imprimir("Matriz de datos después de z-score (data_z):")
        imprimir(data_z)

        # Distancias
        diffs = data_z.loc[cols_validas].values - data_z.loc[aeronave_obj].values
        dist = np.linalg.norm(diffs, axis=1)
        weights = 1 / (1 + dist)

        valores_vecinos = df_familia.loc[parametro_objetivo, cols_validas].values
        valor_imputado = np.sum(weights * valores_vecinos) / np.sum(weights)

        # Confianza simple (CV + distancia media)
        cv = valores_vecinos.std(ddof=0) / valores_vecinos.mean()
        conf = max(0, 1 - cv - 0.1 * dist.mean())

        imprimir("Detalle del cálculo del valor imputado (se calcula el valor ponderado basado en la distancia entre aeronaves):")
        imprimir(f"  Pesos (weights): {weights}")
        imprimir(f"  Valores vecinos: {valores_vecinos}")

        productos = weights * valores_vecinos
        suma_ponderada = np.sum(productos)
        suma_pesos = np.sum(weights)

        imprimir(f"  Producto elemento a elemento (weights * valores_vecinos): {productos}")
        imprimir(f"  Suma ponderada (Σ weights * valores_vecinos): {suma_ponderada:.3f}")
        imprimir(f"  Suma de pesos (Σ weights): {suma_pesos:.3f}")
        imprimir(f"  División final: {suma_ponderada:.3f} / {suma_pesos:.3f} = {valor_imputado:.3f}")

        imprimir(f"✅  Valor imputado: {valor_imputado:.3f}")
        imprimir(f"    Confianza.....: {conf:.2f}")
        imprimir(f"    Vecinos usados: {list(cols_validas)}")

        return {
            "valor": valor_imputado,
            "confianza": conf,
            "vecinos": list(cols_validas)
        }

    # Si ninguna capa funcionó
    imprimir("⚠️  Sin vecinos en ninguna capa. Se delega a correlación.", True)
    return None



