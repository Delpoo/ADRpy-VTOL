"""
Utilidades Comunes para el Análisis de Modelos
==============================================

Este módulo contiene funciones utilitarias comunes para todo el sistema
de análisis de modelos, incluyendo manejo de NaN, validaciones, y más.
"""

import numpy as np
import pandas as pd
from typing import Any, Union, Optional, List, Tuple, Dict


def is_valid_numeric(value: Any) -> bool:
    """
    Verifica si un valor es numérico válido (no None, no NaN, no Inf).
    
    Parameters:
    -----------
    value : Any
        Valor a verificar
        
    Returns:
    --------
    bool
        True si el valor es numérico válido, False en caso contrario
    """
    if value is None:
        return False
    
    try:
        # Convertir a float para verificar
        numeric_value = float(value)
        return not (np.isnan(numeric_value) or np.isinf(numeric_value))
    except (ValueError, TypeError):
        return False


def clean_numeric_value(value: Any, default: Optional[float] = None) -> Optional[float]:
    """
    Limpia un valor numérico, reemplazando valores inválidos por un default.
    
    Parameters:
    -----------
    value : Any
        Valor a limpiar
    default : Optional[float]
        Valor por defecto a usar si el valor es inválido
        
    Returns:
    --------
    Optional[float]
        Valor limpio o None/default si es inválido
    """
    if is_valid_numeric(value):
        return float(value)
    return default


def clean_numeric_series(series: pd.Series, 
                        drop_invalid: bool = True,
                        fill_value: Optional[float] = None) -> pd.Series:
    """
    Limpia una serie de pandas eliminando o reemplazando valores inválidos.
    
    Parameters:
    -----------
    series : pd.Series
        Serie a limpiar
    drop_invalid : bool
        Si True, elimina valores inválidos. Si False, los reemplaza con fill_value
    fill_value : Optional[float]
        Valor con el que reemplazar valores inválidos (si drop_invalid=False)
        
    Returns:
    --------
    pd.Series
        Serie limpia
    """
    if drop_invalid:
        return series.dropna()
    else:
        if fill_value is not None:
            return series.fillna(fill_value)
        else:
            return series


def validate_numeric_array(arr: np.ndarray, 
                          name: str = "array") -> Tuple[bool, str]:
    """
    Valida que un array numérico no contenga valores inválidos.
    
    Parameters:
    -----------
    arr : np.ndarray
        Array a validar
    name : str
        Nombre del array para el mensaje de error
        
    Returns:
    --------
    Tuple[bool, str]
        (is_valid, error_message)
    """
    if arr is None or len(arr) == 0:
        return False, f"{name} está vacío o es None"
    
    if np.any(np.isnan(arr)):
        return False, f"{name} contiene valores NaN"
    
    if np.any(np.isinf(arr)):
        return False, f"{name} contiene valores infinitos"
    
    return True, ""


def safe_json_load(json_text: str) -> Any:
    """
    Carga JSON de manera segura, manejando valores NaN problemáticos.
    
    Parameters:
    -----------
    json_text : str
        Texto JSON a cargar
        
    Returns:
    --------
    Any
        Objeto Python cargado desde JSON
    """
    import json
    
    # Reemplazar NaN problemáticos que no son válidos en JSON estándar
    json_text = json_text.replace(': NaN', ': null')
    json_text = json_text.replace(':NaN', ': null')
    json_text = json_text.replace('[NaN', '[null')
    json_text = json_text.replace(',NaN', ',null')
    json_text = json_text.replace('NaN,', 'null,')
    json_text = json_text.replace('NaN]', 'null]')
    
    return json.loads(json_text)


def generate_safe_range(start: float, 
                       end: float, 
                       num_points: int = 100,
                       expand_factor: float = 0.1) -> Optional[np.ndarray]:
    """
    Genera un rango seguro para predicciones, validando los parámetros de entrada.
    
    Parameters:
    -----------
    start : float
        Valor inicial del rango
    end : float
        Valor final del rango
    num_points : int
        Número de puntos a generar
    expand_factor : float
        Factor de expansión del rango (0.1 = 10% a cada lado)
        
    Returns:
    --------
    Optional[np.ndarray]
        Array con el rango generado, o None si los parámetros son inválidos
    """
    # Validar parámetros de entrada
    if not is_valid_numeric(start) or not is_valid_numeric(end):
        return None
    
    if not isinstance(num_points, int) or num_points <= 0:
        return None
    
    if not is_valid_numeric(expand_factor) or expand_factor < 0:
        expand_factor = 0.1
    
    # Asegurar que start <= end
    if start > end:
        start, end = end, start
    
    # Si start == end, crear un pequeño rango alrededor del valor
    if start == end:
        margin = abs(start) * 0.1 if start != 0 else 1.0
        start = start - margin
        end = end + margin
    
    # Expandir el rango
    range_size = end - start
    expansion = range_size * expand_factor
    expanded_start = start - expansion
    expanded_end = end + expansion
    
    # Generar el rango
    try:
        return np.linspace(expanded_start, expanded_end, num_points)
    except Exception:
        return None


def safe_format_number(value: Any, 
                      decimal_places: int = 3,
                      default: str = "N/A") -> str:
    """
    Formatea un número de manera segura, manejando valores inválidos.
    
    Parameters:
    -----------
    value : Any
        Valor a formatear
    decimal_places : int
        Número de decimales
    default : str
        Texto a mostrar si el valor es inválido
        
    Returns:
    --------
    str
        Valor formateado
    """
    if is_valid_numeric(value):
        try:
            return f"{float(value):.{decimal_places}f}"
        except Exception:
            return default
    return default


def log_nan_warning(context: str, details: str = "") -> None:
    """
    Registra una advertencia sobre valores NaN encontrados.
    
    Parameters:
    -----------
    context : str
        Contexto donde se encontró el NaN
    details : str
        Detalles adicionales
    """
    import logging
    logger = logging.getLogger(__name__)
    
    message = f"Valores NaN encontrados en {context}"
    if details:
        message += f": {details}"
    
    logger.warning(message)


def generate_synthetic_range_for_model(model_type: str, 
                                     coefficients: List[float],
                                     intercept: float = 0,
                                     num_points: int = 100) -> Tuple[Optional[np.ndarray], bool, str]:
    """
    Genera un rango sintético apropiado para un tipo específico de modelo.
    
    Parameters:
    -----------
    model_type : str
        Tipo de modelo (linear, log, exp, pot, etc.)
    coefficients : List[float]
        Coeficientes del modelo
    intercept : float
        Intercepto del modelo
    num_points : int
        Número de puntos a generar
        
    Returns:
    --------
    Tuple[Optional[np.ndarray], bool, str]
        (rango_generado, es_sintético, mensaje_advertencia)
    """
    if not coefficients or len(coefficients) == 0:
        return None, True, "No hay coeficientes válidos para generar rango sintético"
      # Validar coeficientes e intercepto
    coef = clean_numeric_value(coefficients[0], 1.0)
    if coef is None:
        coef = 1.0
    
    intercept_clean = clean_numeric_value(intercept, 0.0)
    if intercept_clean is None:
        intercept_clean = 0.0
    
    try:
        # Determinar rango según tipo de modelo
        if model_type.startswith('linear'):
            # Para modelos lineales: rango estándar
            x_min, x_max = 0, 10
            
        elif model_type.startswith('log'):
            # Para modelos logarítmicos: evitar valores <= 0
            x_min, x_max = 0.1, 10
            # Expandir si el coeficiente es muy grande/pequeño
            if abs(coef) > 10:
                x_max = 100
            elif abs(coef) < 0.1:
                x_max = 1
                
        elif model_type.startswith('exp'):
            # Para modelos exponenciales: rango limitado para evitar overflow
            x_min, x_max = 0, 5
            # Ajustar según coeficiente para evitar explosión exponencial
            if abs(coef) > 1:
                x_max = min(3, 700 / abs(coef))  # Limitar exp(coef * x) < e^700
            elif abs(coef) < 0.1:
                x_max = 10
                
        elif model_type.startswith('pot'):
            # Para modelos de potencia: rango positivo
            x_min, x_max = 0.1, 10
            # Ajustar según exponente
            if abs(coef) > 3:
                x_max = 3  # Evitar crecimiento muy rápido
            elif abs(coef) < 1:
                x_max = 100
                
        else:
            # Tipo no reconocido: usar rango por defecto
            x_min, x_max = 0, 10
            
        # Validar que el rango es válido
        if not is_valid_numeric(x_min) or not is_valid_numeric(x_max) or x_max <= x_min:
            return None, True, f"Rango generado inválido: [{x_min}, {x_max}]"
        
        # Generar el rango usando la función segura
        x_range = generate_safe_range(x_min, x_max, num_points, expand_factor=0.0)
        
        if x_range is None:
            return None, True, f"Error generando rango sintético para modelo {model_type}"
        
        warning_msg = f"Rango sintético para {model_type}: [{x_min:.2f}, {x_max:.2f}]"
        return x_range, True, warning_msg
        
    except Exception as e:
        return None, True, f"Error en generación de rango sintético: {str(e)}"


def validate_model_coefficients(model: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Valida que los coeficientes de un modelo sean válidos para generar predicciones.
    
    Parameters:
    -----------
    model : Dict[str, Any]
        Diccionario del modelo con coeficientes e intercepto
        
    Returns:
    --------
    Tuple[bool, str]
        (es_válido, mensaje_error)
    """
    tipo = model.get('tipo', 'unknown')
    coefs = model.get('coeficientes_originales', [])
    intercept = model.get('intercepto_original', 0)
    
    # Verificar que existen coeficientes
    if not coefs or len(coefs) == 0:
        return False, f"Modelo {tipo} no tiene coeficientes"
    
    # Validar primer coeficiente (el más importante)
    if not is_valid_numeric(coefs[0]):
        return False, f"Coeficiente principal inválido para modelo {tipo}: {coefs[0]}"
    
    # Validar intercepto
    if not is_valid_numeric(intercept):
        return False, f"Intercepto inválido para modelo {tipo}: {intercept}"
    
    # Validaciones específicas por tipo
    coef = float(coefs[0])
    
    if tipo.startswith('exp'):
        # Para exponenciales, evitar coeficientes que causen overflow
        if abs(coef) > 100:
            return False, f"Coeficiente exponencial demasiado grande: {coef} (riesgo de overflow)"
    
    elif tipo.startswith('log'):
        # Para logarítmicos, el coeficiente no puede ser 0
        if abs(coef) < 1e-10:
            return False, f"Coeficiente logarítmico demasiado pequeño: {coef}"
    
    elif tipo.startswith('pot'):
        # Para potenciales, validar exponente razonable
        if abs(coef) > 10:
            return False, f"Exponente de potencia demasiado grande: {coef}"
    
    return True, ""


def compute_range_and_warning(model: Dict[str, Any], 
                            predictor: str,
                            df_original: Optional[pd.DataFrame] = None) -> Tuple[Optional[np.ndarray], Optional[float], Optional[float], bool, str]:
    """
    Computa el rango para un modelo y predictor, con validaciones completas.
    
    Parameters:
    -----------
    model : Dict[str, Any]
        Diccionario del modelo
    predictor : str
        Nombre del predictor
    df_original : Optional[pd.DataFrame]
        DataFrame con datos originales (si existe)
          Returns:
    --------
    Tuple[Optional[np.ndarray], Optional[float], Optional[float], bool, str]
        (x_range, x_min, x_max, es_sintético, mensaje_advertencia)
    """
    # Intentar usar datos originales primero
    x_range = None
    x_min = None
    x_max = None
    is_synthetic = False
    warning = ""
    
    if df_original is not None and not df_original.empty and predictor in df_original.columns:
        x_data = clean_numeric_series(df_original[predictor], drop_invalid=True)
        
        if len(x_data) > 0:
            x_min = x_data.min()
            x_max = x_data.max()
            
            if is_valid_numeric(x_min) and is_valid_numeric(x_max) and x_max > x_min:
                x_range = generate_safe_range(x_min, x_max, 100)
                if x_range is not None:
                    return x_range, x_min, x_max, False, ""
    
    # Si no hay datos originales válidos, generar rango sintético
    is_valid_model, validation_error = validate_model_coefficients(model)
    if not is_valid_model:
        return None, None, None, True, f"Modelo inválido: {validation_error}"
    
    tipo = model.get('tipo', '')
    coefs = model.get('coeficientes_originales', [])
    intercept = model.get('intercepto_original', 0)
    
    x_range, is_synthetic, warning = generate_synthetic_range_for_model(
        tipo, coefs, intercept
    )
    
    if x_range is not None:
        x_min = x_range.min()
        x_max = x_range.max()
    
    return x_range, x_min, x_max, is_synthetic, warning
