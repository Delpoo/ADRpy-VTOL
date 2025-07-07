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

def get_model_original_data(modelo: Dict, parametro_objetivo: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Obtiene el DataFrame original asociado al modelo.
    Si no existe, intenta reconstruirlo desde datos disponibles.
    
    Parameters:
    -----------
    modelo : Dict
        Diccionario con información del modelo
    parametro_objetivo : str, optional
        Nombre real del parámetro objetivo. Si no se provee, se intentará detectar.
        
    Returns:
    --------
    Optional[pd.DataFrame]
        DataFrame original o None si no se encuentra
    """
    # Usar helper robusto para obtener DataFrame original
    from .json_data_helpers import get_full_dataframe_from_celda
    df, warnings = get_full_dataframe_from_celda({'informacion_generica_celda': {'df_original': modelo.get('df_original')}})
    if df is not None:
        return df
    logger.warning(f"No se encontraron datos originales para modelo de tipo {modelo.get('tipo', 'unknown')}. Warnings: {warnings}")
    return None


def get_model_training_data(modelo: Dict, parametro_objetivo: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Obtiene el DataFrame de entrenamiento asociado al modelo.
    Si no existe, intenta reconstruirlo desde datos disponibles.
    
    Parameters:
    -----------
    modelo : Dict
        Diccionario con información del modelo
    parametro_objetivo : str, optional
        Nombre real del parámetro objetivo. Si no se provee, se intentará detectar.
        
    Returns:
    --------
    Optional[pd.DataFrame]
        DataFrame de entrenamiento o None si no se encuentra
    """
    # Usar helper robusto para obtener datos de entrenamiento
    from .json_data_helpers import get_model_specific_data
    X_data, y_data, df_filtrado, warnings = get_model_specific_data({}, modelo)
    if df_filtrado is not None:
        return df_filtrado
    logger.warning(f"No se encontraron datos de entrenamiento para modelo de tipo {modelo.get('tipo', 'unknown')}. Warnings: {warnings}")
    return None

