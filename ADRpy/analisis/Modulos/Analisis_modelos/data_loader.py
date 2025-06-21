"""
Cargador y Procesador de Datos para Análisis de Modelos
======================================================

Este módulo se encarga de cargar y procesar los datos del archivo JSON
que contiene los modelos de imputación generados por la pipeline.

Funciones principales:
- load_models_data: Carga los datos del JSON
- extract_unique_values: Extrae valores únicos para filtros
- filter_models: Filtra modelos según criterios
- prepare_plot_data: Prepara datos para visualización
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_models_data(json_path: str) -> Tuple[Dict, Dict]:
    """
    Carga los datos del archivo JSON de modelos.
    
    Parameters:
    -----------
    json_path : str
        Ruta al archivo JSON con los modelos
        
    Returns:
    --------
    Tuple[Dict, Dict]
        Tupla con (modelos_por_celda, detalles_por_celda)
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            # Leer como texto primero para reemplazar NaN problemático
            json_text = f.read()
            # Reemplazar NaN que no es válido en JSON estándar
            json_text = json_text.replace(': NaN', ': "NaN"')
            
        # Parsear JSON corregido
        data = json.loads(json_text)
        
        modelos_por_celda = data.get('modelos_por_celda', {})
        detalles_por_celda = data.get('detalles_por_celda', {})
        
        logger.info(f"Cargados {len(modelos_por_celda)} celdas con modelos")
        logger.info(f"Cargados {len(detalles_por_celda)} celdas con detalles")
        
        return modelos_por_celda, detalles_por_celda
        
    except Exception as e:
        logger.error(f"Error cargando datos: {e}")
        # Intentar carga estándar como fallback
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            modelos_por_celda = data.get('modelos_por_celda', {})
            detalles_por_celda = data.get('detalles_por_celda', {})
            logger.info("Carga fallback exitosa")
            return modelos_por_celda, detalles_por_celda
        except Exception as e2:
            logger.error(f"Error en carga fallback: {e2}")
            raise


def extract_unique_values(modelos_por_celda: Dict) -> Dict[str, List]:
    """
    Extrae valores únicos para los filtros de la interfaz.
    
    Parameters:
    -----------
    modelos_por_celda : Dict
        Diccionario con todos los modelos por celda
        
    Returns:
    --------
    Dict[str, List]
        Diccionario con listas de valores únicos para cada filtro
    """
    aeronaves = set()
    parametros = set()
    tipos_modelo = set()
    predictores = set()
    n_predictores = set()
    
    for celda_key, modelos in modelos_por_celda.items():
        # Parsear la clave de celda
        if '|' in celda_key:
            aeronave, parametro = celda_key.split('|', 1)
            aeronaves.add(aeronave)
            parametros.add(parametro)
        
        # Procesar cada modelo
        for modelo in modelos:
            if isinstance(modelo, dict):
                # Tipos de modelo
                tipo = modelo.get('tipo')
                if tipo:
                    tipos_modelo.add(tipo)
                
                # Número de predictores
                n_pred = modelo.get('n_predictores')
                if n_pred:
                    n_predictores.add(n_pred)
                
                # Predictores individuales
                pred_list = modelo.get('predictores', [])
                if isinstance(pred_list, list):
                    predictores.update(pred_list)
    
    return {
        'aeronaves': sorted(list(aeronaves)),
        'parametros': sorted(list(parametros)),
        'tipos_modelo': sorted(list(tipos_modelo)),
        'predictores': sorted(list(predictores)),
        'n_predictores': sorted(list(n_predictores))
    }


def get_parametros_for_aeronave(modelos_por_celda: Dict, aeronave: str) -> List[str]:
    """
    Obtiene los parámetros disponibles para una aeronave específica.
    
    Parameters:
    -----------
    modelos_por_celda : Dict
        Diccionario con todos los modelos
    aeronave : str
        Nombre de la aeronave
        
    Returns:
    --------
    List[str]
        Lista de parámetros disponibles para la aeronave
    """
    parametros = set()
    
    for celda_key in modelos_por_celda.keys():
        if '|' in celda_key:
            aero, param = celda_key.split('|', 1)
            if aero == aeronave:
                parametros.add(param)
    
    return sorted(list(parametros))


def filter_models(modelos_por_celda: Dict, 
                 aeronave: Optional[str] = None,
                 parametro: Optional[str] = None,
                 tipos_modelo: Optional[List[str]] = None,
                 predictores: Optional[List[str]] = None,
                 only_real_curves: bool = False,
                 comparison_type: str = 'by_type',
                 mejores: bool = False) -> Dict:
    """
    Filtra los modelos según los criterios especificados y el modo de comparación.
    Mejor modelo: mayor confianza promedio (entrenamiento + validación), solo si tiene validación.
    Permite filtrar por nombre de predictor (no por número).
    comparison_type:
        - 'by_type': Todos los modelos filtrados (default)
        - 'best_overall': Solo el modelo de mayor confianza promedio (si empate, mayor r2)
        - 'by_predictors': Mejor modelo por cada combinación única de predictores
    only_real_curves:
        - Si True, solo modelos con datos reales en y_original
    """
    filtered_models = {}
    target_key = None
    if aeronave and parametro:
        target_key = f"{aeronave}|{parametro}"
    for celda_key, modelos in modelos_por_celda.items():
        if target_key and celda_key != target_key:
            continue
        if aeronave and not celda_key.startswith(f"{aeronave}|"):
            continue
        filtered_models_celda = []
        for modelo in modelos:
            if not isinstance(modelo, dict):
                continue
            # Filtrar por tipo de modelo
            if tipos_modelo:
                tipo = modelo.get('tipo')
                if tipo not in tipos_modelo:
                    continue
            # Filtrar por nombre de predictor (al menos uno debe estar en la lista)
            if predictores:
                modelo_preds = set(modelo.get('predictores', []))
                if not any(pred in modelo_preds for pred in predictores):
                    continue
            # Filtro: solo curvas con datos reales
            if only_real_curves:
                datos_entrenamiento = modelo.get('datos_entrenamiento', {})
                y_original = datos_entrenamiento.get('y_original')
                if not y_original or (isinstance(y_original, list) and len([y for y in y_original if y is not None]) == 0):
                    continue
            filtered_models_celda.append(modelo)
        # Modos de comparación
        if filtered_models_celda:
            if comparison_type == 'best_overall':
                # Solo el modelo de mayor confianza promedio (entrenamiento + validación), solo si tiene validación
                def confianza_promedio(m):
                    conf_train = m.get('Confianza', 0)
                    conf_val = m.get('Confianza_validacion')
                    if conf_val is None:
                        return -1  # Descarta modelos sin validación
                    return (conf_train + conf_val) / 2
                # Filtrar solo modelos con validación
                modelos_validos = [m for m in filtered_models_celda if m.get('Confianza_validacion') is not None]
                if modelos_validos:
                    best = max(modelos_validos, key=lambda m: (confianza_promedio(m), m.get('r2', 0)))
                    filtered_models_celda = [best]
                else:
                    filtered_models_celda = []
            elif comparison_type == 'by_predictors':
                # Mejor modelo por cada combinación única de predictores (por nombre)
                best_by_pred = {}
                def confianza_promedio(m):
                    conf_train = m.get('Confianza', 0)
                    conf_val = m.get('Confianza_validacion')
                    if conf_val is None:
                        return -1
                    return (conf_train + conf_val) / 2
                for m in filtered_models_celda:
                    preds = tuple(sorted(m.get('predictores', [])))
                    conf = confianza_promedio(m)
                    r2 = m.get('r2', 0)
                    if preds not in best_by_pred:
                        best_by_pred[preds] = m
                    else:
                        best = best_by_pred[preds]
                        best_conf = confianza_promedio(best)
                        if conf > best_conf or (conf == best_conf and r2 > best.get('r2', 0)):
                            best_by_pred[preds] = m
                # Solo modelos con validación
                filtered_models_celda = [m for m in best_by_pred.values() if m.get('Confianza_validacion') is not None]
            # else: by_type (default): no cambio, todos los modelos filtrados
            filtered_models[celda_key] = filtered_models_celda
    return filtered_models


def prepare_plot_data(modelo: Dict) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Prepara los datos de un modelo para visualización.
    Asegura que los DataFrames devueltos contengan SOLO el predictor y parámetro del modelo actual.
    
    Parameters:
    -----------
    modelo : Dict
        Diccionario con información del modelo
        
    Returns:
    --------
    Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]
        Tupla con (df_original, df_filtrado) o (None, None) si hay error
    """
    try:
        datos_entrenamiento = modelo.get('datos_entrenamiento', {})
        predictor = None
        parametro = None
        # Obtener nombre del predictor y parámetro
        if 'predictores' in modelo and isinstance(modelo['predictores'], list) and len(modelo['predictores']) == 1:
            predictor = modelo['predictores'][0]
        parametro = modelo.get('Parámetro') or modelo.get('parametro')

        # Datos originales
        df_original_dict = datos_entrenamiento.get('df_original')
        df_original = None
        if df_original_dict:
            df_original = pd.DataFrame(df_original_dict)
            df_original = df_original.replace('NaN', np.nan)
            # Filtrar solo columnas relevantes
            if predictor and parametro:
                cols = [c for c in [predictor, parametro] if c in df_original.columns]
                df_original = df_original[cols]

        # Datos filtrados (entrenamiento)
        df_filtrado_dict = datos_entrenamiento.get('df_filtrado')
        df_filtrado = None
        if df_filtrado_dict:
            df_filtrado = pd.DataFrame(df_filtrado_dict)
            df_filtrado = df_filtrado.replace('NaN', np.nan)
            # Filtrar solo columnas relevantes
            if predictor and parametro:
                cols = [c for c in [predictor, parametro] if c in df_filtrado.columns]
                df_filtrado = df_filtrado[cols]

        return df_original, df_filtrado
        
    except Exception as e:
        logger.error(f"Error preparando datos de plot: {e}")
        return None, None


def get_model_predictions(modelo: Dict, x_range: np.ndarray) -> Optional[np.ndarray]:
    """
    Genera predicciones del modelo para un rango de valores X.
    
    Parameters:
    -----------
    modelo : Dict
        Diccionario con información del modelo
    x_range : np.ndarray
        Rango de valores X para predecir
        
    Returns:
    --------
    Optional[np.ndarray]
        Array con las predicciones o None si hay error
    """
    try:
        tipo = modelo.get('tipo', '')
        coefs = modelo.get('coeficientes_originales', [])
        intercept = modelo.get('intercepto_original', 0)
        
        if not coefs:
            return None
        
        # Solo manejar modelos de 1 predictor por ahora
        if modelo.get('n_predictores', 0) != 1:
            return None
        
        coef = coefs[0]
        
        if 'linear' in tipo:
            # Modelo lineal: y = intercept + coef * x
            predictions = intercept + coef * x_range
            
        elif 'poly' in tipo:
            # Para modelos polinómicos, necesitaríamos más información
            # Por ahora, tratarlo como lineal
            predictions = intercept + coef * x_range
            
        elif 'log' in tipo:
            # Modelo logarítmico: y = intercept + coef * log(x)
            # Evitar log de valores <= 0
            x_safe = np.where(x_range > 0, x_range, 1e-10)
            predictions = intercept + coef * np.log(x_safe)
            
        elif 'exp' in tipo:
            # Modelo exponencial: y = intercept + coef * exp(x)
            # Limitar para evitar overflow
            x_limited = np.clip(x_range, -50, 50)
            predictions = intercept + coef * np.exp(x_limited)
            
        elif 'pot' in tipo:
            # Modelo de potencia: y = intercept + coef * x^poder
            # Por simplicidad, asumimos potencia 2
            predictions = intercept + coef * np.power(np.abs(x_range), 2) * np.sign(x_range)
            
        else:
            # Tipo desconocido, usar lineal como fallback
            predictions = intercept + coef * x_range
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error generando predicciones: {e}")
        return None


def get_model_info_text(modelo: Dict) -> str:
    """
    Genera texto informativo sobre un modelo para mostrar en hover.
    
    Parameters:
    -----------
    modelo : Dict
        Diccionario con información del modelo
        
    Returns:
    --------
    str
        Texto formateado con información del modelo
    """
    info_lines = []
    
    # Información básica
    aeronave = modelo.get('Aeronave', 'N/A')
    parametro = modelo.get('Parámetro', 'N/A')
    tipo = modelo.get('tipo', 'N/A')
    
    info_lines.append(f"<b>Aeronave:</b> {aeronave}")
    info_lines.append(f"<b>Parámetro:</b> {parametro}")
    info_lines.append(f"<b>Tipo:</b> {tipo}")
    
    # Predictores
    predictores = modelo.get('predictores', [])
    if predictores:
        pred_text = ', '.join(predictores)
        info_lines.append(f"<b>Predictores:</b> {pred_text}")
    
    # Ecuación
    ecuacion = modelo.get('ecuacion_string', '')
    if ecuacion:
        info_lines.append(f"<b>Ecuación:</b> {ecuacion}")
    
    # Métricas
    mape = modelo.get('mape')
    r2 = modelo.get('r2')
    corr = modelo.get('corr')
    confianza = modelo.get('Confianza')
    
    if mape is not None:
        info_lines.append(f"<b>MAPE:</b> {mape:.3f}%")
    if r2 is not None:
        info_lines.append(f"<b>R²:</b> {r2:.3f}")
    if corr is not None:
        info_lines.append(f"<b>Correlación:</b> {corr:.3f}")
    if confianza is not None:
        info_lines.append(f"<b>Confianza:</b> {confianza:.3f}")
    
    # Entrenamiento
    n_muestras = modelo.get('n_muestras_entrenamiento')
    if n_muestras:
        info_lines.append(f"<b>N° muestras:</b> {n_muestras}")
    
    # Advertencias
    advertencia = modelo.get('Advertencia')
    if advertencia:
        info_lines.append(f"<b>Advertencia:</b> {advertencia}")
    
    return '<br>'.join(info_lines)
