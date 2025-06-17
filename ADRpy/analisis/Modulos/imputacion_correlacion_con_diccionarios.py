"""
VERSIÓN MODIFICADA DE IMPUTACION_CORRELACION.PY QUE GENERA DICCIONARIOS DE MODELOS

Esta versión extiende la función original para generar y retornar diccionarios
completos de modelos que pueden ser usados por el sistema de análisis visual.
"""

import pandas as pd
import numpy as np
from itertools import combinations
from collections import defaultdict
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_percentage_error, r2_score

# Importar funciones originales
from .imputacion_correlacion import (
    is_missing, cargar_y_validar_datos, penalizacion_por_k,
    seleccionar_predictores_validos, generar_combinaciones, entrenar_modelo,
    imputar_valores_celda, imputaciones_correlacion, filtrar_mejores_modelos
)

# Importar función de validación LOOCV
try:
    from .imputacion_correlacion import validar_con_loocv
except ImportError:
    print("⚠️ Función validar_con_loocv no encontrada, se implementará localmente")

def imputaciones_correlacion_con_diccionarios(df, path: str | None = None):
    """
    Versión extendida de imputaciones_correlacion que TAMBIÉN genera diccionarios
    de modelos para ser usados por el sistema de análisis visual.
    
    Returns:
        df_resultado: DataFrame con valores imputados
        reporte: Lista de reportes de imputación  
        diccionarios_modelos: Dict con todos los modelos entrenados por celda
    """
    if isinstance(df, str):
        df = pd.read_excel(df)
    df = df.rename(columns=lambda c: str(c).strip())
    df.replace("", np.nan, inplace=True)
    
    df_original = df
    df_resultado = df_original.copy()
    reporte = []
    diccionarios_modelos = {}  # ← NUEVO: Almacenar modelos por celda
    
    print("🔧 Ejecutando imputación por correlación CON generación de diccionarios...")
    
    for objetivo in [c for c in df_original.columns if df_original[c].isna().any()]:
        faltantes = df_original[df_original[objetivo].isna()].index
        
        for idx in faltantes:
            clave_celda = f"aeronave_{idx}_parametro_{objetivo}"
            print(f"  📊 Procesando: {clave_celda}")
            
            # Seleccionar predictores válidos
            df_filtrado, familia_usada, filtro_aplicado = seleccionar_predictores_validos(
                df_original, objetivo, idx
            )
            
            if df_filtrado.empty:
                print(f"    ❌ Sin datos válidos para {clave_celda}")
                continue
            
            # Excluir primera columna y objetivo
            predictores = [
                col for col in df_filtrado.columns 
                if col != df_filtrado.columns[0] and col != objetivo
            ]
            
            if not predictores:
                print(f"    ❌ Sin predictores válidos para {clave_celda}")
                # Agregar entrada vacía al diccionario
                diccionarios_modelos[clave_celda] = {
                    "mejor_modelo": None,
                    "todos_los_modelos": [],
                    "df_filtrado": df_filtrado,
                    "predictores": [],
                    "familia_usada": familia_usada,
                    "filtro_aplicado": filtro_aplicado,
                    "error": "Sin predictores válidos"
                }
                
                reporte.append({
                    "Aeronave": idx,
                    "Parámetro": objetivo,
                    "Valor imputado": "NAN",
                    "Confianza": 0,
                    "Tipo Modelo": "N/A",
                    "Predictores": "N/A",
                    "k": 0,
                    "Penalizacion_k": 0,
                    "Corr": 0,
                    "Familia": familia_usada,
                    "Método predictivo": "Correlacion",
                    "Advertencia": "No se pudo imputar por falta de parámetros válidos."
                })
                continue
            
            try:
                # Generar combinaciones y entrenar TODOS los modelos
                combos = generar_combinaciones(predictores)
                todos_los_modelos = []
                
                print(f"    🔄 Entrenando {len(combos)} combinaciones...")
                
                for combo in combos:
                    # Modelos normales
                    for poly in [False, True]:
                        try:
                            modelo = entrenar_modelo(df_filtrado, objetivo, combo, poly, idx)
                            if modelo:
                                todos_los_modelos.append(modelo)
                        except Exception as e:
                            print(f"      ⚠️ Error modelo {combo}, poly={poly}: {e}")
                    
                    # Modelos especiales (solo para 1 predictor)
                    if len(combo) == 1:
                        for modelo_tipo in ["log", "power", "exp"]:
                            try:
                                modelo = entrenar_modelo(df_filtrado, objetivo, combo, False, idx, modelo_tipo)
                                if modelo:
                                    todos_los_modelos.append(modelo)
                            except Exception as e:
                                print(f"      ⚠️ Error modelo {combo}, tipo={modelo_tipo}: {e}")
                
                if not todos_los_modelos:
                    print(f"    ❌ No se pudo entrenar ningún modelo para {clave_celda}")
                    diccionarios_modelos[clave_celda] = {
                        "mejor_modelo": None,
                        "todos_los_modelos": [],
                        "df_filtrado": df_filtrado,
                        "predictores": predictores,
                        "familia_usada": familia_usada,
                        "filtro_aplicado": filtro_aplicado,
                        "error": "No se pudo entrenar ningún modelo"
                    }
                    continue
                  # Evaluar todos los modelos usando la misma lógica que el original
                # Solo pasar a LOOCV los modelos con MAPE <= 7.5% y R2 >= 0.6
                validos = [m for m in todos_los_modelos 
                          if m is not None and not m.get("descartado", False) 
                          and m.get("mape", 100) <= 7.5 and m.get("r2", 0) >= 0.6]
                
                if not validos:
                    print(f"    ❌ Ningún modelo válido para {clave_celda}")
                    diccionarios_modelos[clave_celda] = {
                        "mejor_modelo": None,
                        "todos_los_modelos": todos_los_modelos,
                        "df_filtrado": df_filtrado,
                        "predictores": predictores,
                        "familia_usada": familia_usada,
                        "filtro_aplicado": filtro_aplicado,
                        "error": "Ningún modelo pasó validación inicial"
                    }
                    continue
                
                # Validar con LOOCV y calcular confianza promedio
                try:
                    from .imputacion_correlacion import validar_con_loocv
                    for m in validos:
                        m.update(validar_con_loocv(df_filtrado, objetivo, m))
                        m["Confianza_promedio"] = (m["Confianza"] + m["Confianza_LOOCV"]) / 2
                except ImportError:
                    # Fallback: usar solo confianza inicial
                    for m in validos:
                        m["Confianza_promedio"] = m.get("Confianza", 0)
                
                # Filtrar modelos robustos por LOOCV (misma lógica que original)
                robustos = [m for m in validos 
                           if m.get("MAPE_LOOCV", 100) <= 15 and m.get("R2_LOOCV", 0) >= 0.6]
                
                if robustos:
                    mejor_modelo = max(robustos, key=lambda x: x["Confianza_promedio"])
                    warning_text = "Modelo robusto"
                else:
                    mejor_modelo = max(validos, key=lambda x: x["Confianza_promedio"])
                    warning_text = "Modelo no robusto"
                
                if not filtro_aplicado:
                    warning_text += "; modelo sin filtrado por familia"
                
                mejor_modelo["warning"] = warning_text
                mejor_modelo["Familia"] = familia_usada
                
                if not mejor_modelo:
                    print(f"    ❌ No se encontró mejor modelo para {clave_celda}")
                    diccionarios_modelos[clave_celda] = {
                        "mejor_modelo": None,
                        "todos_los_modelos": todos_los_modelos,
                        "df_filtrado": df_filtrado,
                        "predictores": predictores,
                        "familia_usada": familia_usada,
                        "filtro_aplicado": filtro_aplicado,
                        "error": "No se pudo evaluar modelos"
                    }
                    continue
                
                # ✅ GUARDAR DICCIONARIO COMPLETO
                diccionarios_modelos[clave_celda] = {
                    "mejor_modelo": mejor_modelo,
                    "todos_los_modelos": todos_los_modelos,
                    "df_filtrado": df_filtrado,
                    "predictores": predictores,
                    "familia_usada": familia_usada,
                    "filtro_aplicado": filtro_aplicado
                }
                
                print(f"    ✅ Mejor modelo: {mejor_modelo['tipo']} - R²={mejor_modelo.get('r2', 0):.3f}")
                
                # Imputar valor usando el mejor modelo
                df_resultado, imputacion = imputar_valores_celda(
                    df_resultado, df_filtrado, objetivo, mejor_modelo, idx
                )
                
                imputacion["Familia"] = familia_usada
                reporte.append(imputacion)
                
            except Exception as e:
                print(f"    ❌ Error procesando {clave_celda}: {e}")
                diccionarios_modelos[clave_celda] = {
                    "mejor_modelo": None,
                    "todos_los_modelos": [],
                    "df_filtrado": df_filtrado,
                    "predictores": predictores,
                    "familia_usada": familia_usada,
                    "filtro_aplicado": filtro_aplicado,
                    "error": str(e)
                }
    
    print(f"✅ Diccionarios generados para {len(diccionarios_modelos)} celdas")
    
    return df_resultado, reporte, diccionarios_modelos


def filtrar_solo_correlacion_desde_diccionarios(diccionarios_modelos: dict, detalles_para_excel: list) -> dict:
    """
    Filtra los diccionarios para quedarse solo con las imputaciones por correlación,
    excluyendo similitud y promedios ponderados.
    
    Args:
        diccionarios_modelos: Diccionarios completos de modelos
        detalles_para_excel: Lista de detalles que van al Excel final
        
    Returns:
        dict: Diccionarios filtrados solo con correlación
    """
    diccionarios_correlacion = {}
    
    # Extraer claves de celdas que fueron imputadas por correlación
    celdas_correlacion = set()
    for detalle in detalles_para_excel:
        if detalle.get("Método predictivo") == "Correlacion":
            aeronave = detalle.get("Aeronave")
            parametro = detalle.get("Parámetro")
            clave = f"aeronave_{aeronave}_parametro_{parametro}"
            celdas_correlacion.add(clave)
    
    # Filtrar diccionarios
    for clave, datos in diccionarios_modelos.items():
        if clave in celdas_correlacion:
            diccionarios_correlacion[clave] = datos
            print(f"  ✅ Incluido: {clave}")
        else:
            print(f"  ❌ Excluido (no es correlación): {clave}")
    
    print(f"📊 Filtrados: {len(diccionarios_correlacion)}/{len(diccionarios_modelos)} celdas")
    return diccionarios_correlacion
