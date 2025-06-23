"""
plot_data_access.py

Funciones de obtención y manipulación de datos crudos para la visualización de modelos:
- Acceso a datos originales y de entrenamiento
- Extracción de información relevante de los modelos
"""

from typing import Dict, Optional
import logging
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

def get_model_original_data(modelo: Dict) -> Optional[pd.DataFrame]:
    """
    Obtiene el DataFrame original asociado al modelo.
    Si no existe, intenta reconstruirlo desde datos disponibles.
    
    Parameters:
    -----------
    modelo : Dict
        Diccionario con información del modelo
        
    Returns:
    --------
    Optional[pd.DataFrame]
        DataFrame original o None si no se encuentra
    """
    try:
        # Intentar obtener datos desde el modelo
        if 'df_original' in modelo and modelo['df_original'] is not None:
            df = modelo['df_original']
            if isinstance(df, dict):
                return pd.DataFrame(df)
            elif isinstance(df, pd.DataFrame):
                return df
        
        # Intentar desde datos_entrenamiento
        if 'datos_entrenamiento' in modelo:
            datos = modelo['datos_entrenamiento']
            if isinstance(datos, dict):
                # Intentar reconstruir desde df_original dentro de datos_entrenamiento
                if 'df_original' in datos and datos['df_original'] is not None:
                    df_dict = datos['df_original']
                    if isinstance(df_dict, dict):
                        return pd.DataFrame(df_dict)
                
                # Intentar desde X_original, y_original
                if 'X_original' in datos and 'y_original' in datos:
                    X = datos['X_original']
                    y = datos['y_original']
                    predictores = modelo.get('predictores', [])
                    parametro = modelo.get('Parámetro', 'target')
                    
                    if predictores and len(predictores) == 1:
                        # Crear DataFrame para modelo de 1 predictor
                        df_data = {}
                        
                        # Añadir predictor
                        if hasattr(X, '__iter__') and not isinstance(X, str):
                            df_data[predictores[0]] = list(X)
                        
                        # Añadir variable objetivo
                        if hasattr(y, '__iter__') and not isinstance(y, str):
                            df_data[parametro] = list(y)
                        
                        if df_data:
                            return pd.DataFrame(df_data)
        
        # Si no hay datos disponibles, retornar None
        logger.warning(f"No se encontraron datos originales para modelo de tipo {modelo.get('tipo', 'unknown')}")
        return None
        
    except Exception as e:
        logger.error(f"Error obteniendo datos originales: {e}")
        return None


def get_model_training_data(modelo: Dict) -> Optional[pd.DataFrame]:
    """
    Obtiene el DataFrame de entrenamiento asociado al modelo.
    Si no existe, intenta reconstruirlo desde datos disponibles.
    
    Parameters:
    -----------
    modelo : Dict
        Diccionario con información del modelo
        
    Returns:
    --------
    Optional[pd.DataFrame]
        DataFrame de entrenamiento o None si no se encuentra
    """
    try:
        # Intentar obtener datos desde el modelo
        if 'df_filtrado' in modelo and modelo['df_filtrado'] is not None:
            df = modelo['df_filtrado']
            if isinstance(df, dict):
                return pd.DataFrame(df)
            elif isinstance(df, pd.DataFrame):
                return df
        
        # Intentar desde datos_entrenamiento
        if 'datos_entrenamiento' in modelo:
            datos = modelo['datos_entrenamiento']
            if isinstance(datos, dict):
                # Intentar reconstruir desde df_filtrado dentro de datos_entrenamiento
                if 'df_filtrado' in datos and datos['df_filtrado'] is not None:
                    df_dict = datos['df_filtrado']
                    if isinstance(df_dict, dict):
                        return pd.DataFrame(df_dict)
                
                # Intentar desde X, y (datos de entrenamiento)
                if 'X' in datos and 'y' in datos:
                    X = datos['X']
                    y = datos['y']
                    predictores = modelo.get('predictores', [])
                    parametro = modelo.get('Parámetro', 'target')
                    
                    if predictores and len(predictores) == 1:
                        # Crear DataFrame para modelo de 1 predictor
                        df_data = {}
                        
                        # Añadir predictor
                        if hasattr(X, '__iter__') and not isinstance(X, str):
                            df_data[predictores[0]] = list(X)
                        
                        # Añadir variable objetivo
                        if hasattr(y, '__iter__') and not isinstance(y, str):
                            df_data[parametro] = list(y)
                        
                        if df_data:
                            return pd.DataFrame(df_data)
        
        # Si no hay datos de entrenamiento, retornar None
        logger.warning(f"No se encontraron datos de entrenamiento para modelo de tipo {modelo.get('tipo', 'unknown')}")
        return None
        
    except Exception as e:
        logger.error(f"Error obteniendo datos de entrenamiento: {e}")
        return None

