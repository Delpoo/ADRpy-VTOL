"""
Visualización 3D para Modelos con 2 Predictores
===============================================

Este módulo contiene funciones para crear gráficos 3D que muestran modelos
de regresión con 2 predictores, incluyendo tanto modelos lineales como 
polinómicos de segundo grado.

Funciones principales:
- create_3d_plot: Crea gráfico 3D interactivo
- extract_coefficients_from_equation: Extrae coeficientes de ecuación LaTeX normalizada
- generate_model_surface: Genera superficie del modelo 3D
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Optional, Any, Tuple, Union
import re
import logging

# Importar configuración de colores
try:
    from .plot_config import COLORS
except ImportError:
    from plot_config import COLORS

logger = logging.getLogger(__name__)


def extract_coefficients_from_model(modelo: Dict[str, Any]) -> Optional[List[float]]:
    """
    Extrae coeficientes del modelo usando los campos directos.
    
    Para modelos polinómicos de 2 predictores, se usa:
    coeficientes_originales + intercepto_original
    
    Parameters:
    -----------
    modelo : Dict[str, Any]
        Diccionario del modelo con coeficientes_originales e intercepto_original
        
    Returns:
    --------
    Optional[List[float]]
        Lista de coeficientes [c0, c1, c2, c3, c4, c5] o None si hay error
        donde c0 es el intercepto
    """
    try:
        coefs = modelo.get('coeficientes_originales', [])
        intercepto = modelo.get('intercepto_original', 0)
        n_predictores = modelo.get('n_predictores', 0)
        
        if n_predictores != 2:
            return None
        if not isinstance(coefs, list) or len(coefs) == 0:
            return None
            
        # Para modelos de 2 predictores, esperamos diferentes números de coeficientes
        # Linear-2: 2 coeficientes (c1, c2) + intercepto
        # Poly-2: 5 coeficientes (c1, c2, c3, c4, c5) + intercepto
        # Retornar [intercepto, c1, c2, ...] 
        result = [float(intercepto)] + [float(c) for c in coefs]
        return result
        
    except Exception as e:
        logger.error(f"Error extrayendo coeficientes del modelo: {e}")
        return None





def normalize_training_data(X_original: List[List[float]], y_original: List[float], 
                           x0_range: Tuple[float, float], x1_range: Tuple[float, float],
                           y_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normaliza los datos de entrenamiento usando los rangos del modelo.
    
    Parameters:
    -----------
    X_original : List[List[float]]
        Datos X originales [[x0, x1], ...]
    y_original : List[float]
        Datos Y originales
    x0_range : Tuple[float, float]
        Rango [min, max] del primer predictor
    x1_range : Tuple[float, float]
        Rango [min, max] del segundo predictor  
    y_range : Tuple[float, float]
        Rango [min, max] de la variable objetivo
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Arrays normalizados x0, x1, y
    """
    X_array = np.array(X_original)
    y_array = np.array(y_original)
    
    # Normalizar cada predictor
    x0_norm = (X_array[:, 0] - x0_range[0]) / (x0_range[1] - x0_range[0])
    x1_norm = (X_array[:, 1] - x1_range[0]) / (x1_range[1] - x1_range[0])
    # 🔧 CAMBIO: No normalizar variable dependiente (Y) para mostrar valores originales
    y_norm = y_array  # Mantener valores originales
    
    return x0_norm, x1_norm, y_norm


def get_model_ranges(model: Dict[str, Any]) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """
    Obtiene los rangos de normalización del modelo de forma robusta usando helpers.
    """
    from .json_data_helpers import get_model_specific_data
    X_data, y_data, df_filtrado, warnings = get_model_specific_data({}, model)
    if X_data is None or y_data is None or len(X_data) == 0:
        return (None, None, None)
    X_array = np.array(X_data)
    y_array = np.array(y_data)
    x0_range = (X_array[:, 0].min(), X_array[:, 0].max())
    x1_range = (X_array[:, 1].min(), X_array[:, 1].max())
    y_range = (y_array.min(), y_array.max())
    return x0_range, x1_range, y_range





def filter_models_for_3d(modelos_por_celda: Dict[str, List[Dict]], aeronave: str, 
                        parametro: str, tipos_modelo: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Filtra modelos para vista 3D (solo modelos con 2 predictores).
    
    Parameters:
    -----------
    modelos_por_celda : Dict[str, List[Dict]]
        Diccionario con todos los modelos
    aeronave : str
        Aeronave seleccionada
    parametro : str
        Parámetro seleccionado
    tipos_modelo : Optional[List[str]]
        Tipos de modelo a incluir
        
    Returns:
    --------
    List[Dict[str, Any]]
        Lista de modelos filtrados con 2 predictores
    """
    from .data_loader import filter_models
    
    # Usar filter_models con parámetros para incluir modelos sin LOOCV y solo 2 predictores
    modelos_filtrados = filter_models(
        modelos_por_celda,
        aeronave=aeronave,
        parametro=parametro,
        tipos_modelo=tipos_modelo,
        require_loocv=False  # No filtrar por confianza LOOCV para vista 3D
    )
    
    celda_key = f"{aeronave}|{parametro}"
    modelos = modelos_filtrados.get(celda_key, [])
    
    # Filtrar solo modelos con 2 predictores
    modelos_2pred = []
    for model in modelos:
        if not isinstance(model, dict):
            continue
            
        n_predictores = model.get('n_predictores', 0)
        
        # Solo modelos con 2 predictores
        if n_predictores == 2:
            modelos_2pred.append(model)
    
    return modelos_2pred


def add_theoretical_points_3d(fig: go.Figure, 
                             theoretical_points_3d: List[Dict],
                             show_theoretical_points: bool = True) -> None:
    """
    Añade puntos de imputación teóricos al gráfico 3D.
    
    Parameters:
    -----------
    fig : go.Figure
        Figura 3D de Plotly donde añadir los puntos
    theoretical_points_3d : List[Dict]
        Lista de puntos teóricos 3D calculados
    show_theoretical_points : bool
        Si mostrar los puntos teóricos
    """
    if not show_theoretical_points or not theoretical_points_3d:
        logger.info(f"Puntos teóricos 3D no mostrados: show_theoretical_points={show_theoretical_points}, len(theoretical_points_3d)={len(theoretical_points_3d) if theoretical_points_3d else 0}")
        return
        
    logger.info(f"Añadiendo {len(theoretical_points_3d)} puntos teóricos 3D al gráfico")
    
    # Filtrar solo puntos 3D (modelos de 2 predictores)
    points_3d = [p for p in theoretical_points_3d if p.get('n_predictores', 1) == 2]
    
    if not points_3d:
        return
    
    # Extraer coordenadas
    x_coords = [p['x'] for p in points_3d]  # Variable independiente 1 normalizada
    y_coords = [p['y'] for p in points_3d]  # Variable independiente 2 normalizada  
    z_coords = [p['z'] for p in points_3d]  # Variable objetivo normalizada
    
    # Crear texto de hover
    hover_texts = []
    for p in points_3d:
        hover_parts = [
            f"<b>Punto Teórico - Modelo {p['modelo_idx']+1}</b>",
            f"<b>Predictor 1:</b> {p['predictor_1']}",
            f"<b>Predictor 2:</b> {p['predictor_2']}",
            f"<b>Tipo:</b> {p['tipo_modelo']}",
            f"<b>X1 original:</b> {p['x_original']:.3f}",
            f"<b>X2 original:</b> {p['y_original']:.3f}",
            f"<b>Z calculado:</b> {p['z_original']:.3f}",
            f"<b>Ecuación:</b> {p['ecuacion']}"
        ]
        
        if p.get('r2') is not None:
            hover_parts.append(f"<b>R²:</b> {p['r2']:.3f}")
        if p.get('mape') is not None:
            hover_parts.append(f"<b>MAPE:</b> {p['mape']:.1f}%")
        if p.get('confianza') is not None:
            hover_parts.append(f"<b>Confianza:</b> {p['confianza']:.3f}")
            
        hover_texts.append("<br>".join(hover_parts))
    
    # Añadir puntos teóricos al gráfico 3D
    fig.add_trace(go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        mode='markers',
        name='Puntos Teóricos 3D',
        marker=dict(
            symbol='diamond',
            size=8,
            color='purple',
            line=dict(width=1, color='darkviolet')
        ),
        hovertemplate='%{hovertext}<extra></extra>',
        hovertext=hover_texts,
        legendgroup='theoretical_3d',
        showlegend=True
    ))
    
    logger.info(f"Añadidos {len(points_3d)} puntos teóricos 3D al gráfico con coordenadas: x={x_coords[:3]}..., y={y_coords[:3]}..., z={z_coords[:3]}...")


# =============================================================================
# 🔧 FUNCIONES PARA SUPERFICIES CON ECUACIONES NORMALIZADAS
# =============================================================================

def generate_normalized_surface_3d(ecuacion_original: str, rango_x: Tuple[float, float], 
                                   rango_y: Tuple[float, float], pred_names: List[str],
                                   resolution: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Genera superficie 3D usando ecuación normalizada (estilo asdasd.py).
    
    La metodología es:
    1. Crear malla normalizada [0,1] x [0,1] para X*, Y*
    2. Crear ecuación que acepta X*, Y* y desnormaliza internamente
    3. Evaluar ecuación para obtener Z en escala original
    4. Retornar X*, Y* normalizadas y Z original para gráfico coherente
    
    Parameters:
    -----------
    ecuacion_original : str
        Ecuación que usa variables en escala original
    rango_x : Tuple[float, float]
        Rango [min, max] del primer predictor
    rango_y : Tuple[float, float]
        Rango [min, max] del segundo predictor
    pred_names : List[str]
        Nombres de los predictores [pred_x, pred_y]
    resolution : int
        Resolución de la malla
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Arrays X*, Y* normalizados y Z original para superficie coherente
    """
    try:
        # Importar funciones de normalización
        try:
            from .normalization_engine import (
                create_normalized_equation_3d, 
                evaluate_normalized_equation_3d
            )
        except ImportError:
            from normalization_engine import (
                create_normalized_equation_3d, 
                evaluate_normalized_equation_3d
            )
        
        # Crear malla normalizada [0,1] x [0,1]
        x_star = np.linspace(0, 1, resolution)
        y_star = np.linspace(0, 1, resolution)
        X_star, Y_star = np.meshgrid(x_star, y_star)
        
        # Crear ecuación normalizada que acepta x_star, y_star
        ecuacion_normalizada = create_normalized_equation_3d(
            ecuacion_original, rango_x, rango_y, pred_names
        )
        
        # Evaluar ecuación normalizada para obtener Z en escala original
        Z_original = evaluate_normalized_equation_3d(
            ecuacion_normalizada, X_star, Y_star
        )
        
        logger.info(f"Superficie 3D normalizada generada: "
                   f"X*=[0,1], Y*=[0,1], Z=[{np.nanmin(Z_original):.3f}, {np.nanmax(Z_original):.3f}]")
        
        return X_star, Y_star, Z_original
        
    except Exception as e:
        logger.error(f"Error generando superficie normalizada 3D: {e}")
        # Fallback a método tradicional
        return generate_model_surface_fallback(rango_x, rango_y, resolution)


def generate_model_surface_fallback(rango_x: Tuple[float, float], 
                                   rango_y: Tuple[float, float], 
                                   resolution: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Superficie de fallback en caso de error.
    """
    x_star = np.linspace(0, 1, resolution)
    y_star = np.linspace(0, 1, resolution)
    X_star, Y_star = np.meshgrid(x_star, y_star)
    Z_fallback = np.zeros_like(X_star)
    
    logger.warning("Usando superficie de fallback (plana)")
    return X_star, Y_star, Z_fallback


# =============================================================================
# 🔧 FUNCIONES PARA SUPERFICIES SIMBIÓTICAS (MIGRADAS DESDE symbiotic_surface.py)
# =============================================================================

def generate_symbiotic_surface_3d(
    model_data: Union[str, Dict],
    rango_x: Tuple[float, float],
    rango_y: Tuple[float, float],
    pred_names: List[str],
    resolution: int = 50
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Genera una superficie 3D usando la transformada simbiótica:
    - X*, Y* normalizados (0-1) para visualización
    - Desnormaliza internamente X*, Y* a escala original
    - Evalúa el modelo en escala original
    - Devuelve X*, Y* (normalizados) y Z (escala original)
    
    Args:
        model_data: String de ecuación (modelos lineales) o dict con coeficientes (modelos polinomiales)
        rango_x: (min, max) para variable x0
        rango_y: (min, max) para variable x1  
        pred_names: Nombres de las variables predictoras
        resolution: Número de puntos por eje
        
    Returns:
        X_star, Y_star, Z: Mallas normalizadas (X,Y) y valores originales (Z)
    """
    x_star = np.linspace(0, 1, resolution)
    y_star = np.linspace(0, 1, resolution)
    X_star, Y_star = np.meshgrid(x_star, y_star)

    x0_min, x0_max = rango_x
    x1_min, x1_max = rango_y
    
    # Desnormalizar a valores originales para evaluar modelo
    x_orig = X_star * (x0_max - x0_min) + x0_min
    y_orig = Y_star * (x1_max - x1_min) + x1_min

    # Determinar tipo de modelo y evaluar apropiadamente
    if isinstance(model_data, str):
        # Modelo lineal: usar string de ecuación
        Z = _evaluate_equation_model(model_data, x_orig, y_orig, pred_names)
    elif isinstance(model_data, dict) and 'coeficientes_originales' in model_data:
        # Modelo polinomial: usar coeficientes directamente
        Z = _evaluate_polynomial_model(model_data, x_orig, y_orig)
    else:
        logger.warning(f"Tipo de modelo no reconocido: {type(model_data)}")
        if isinstance(model_data, dict):
            logger.warning(f"Keys disponibles: {list(model_data.keys())[:5]}...")
        Z = np.full_like(X_star, 0.0)
    
    return X_star, Y_star, Z


def _evaluate_equation_model(equation_string: str, x_orig: np.ndarray, y_orig: np.ndarray, pred_names: List[str]) -> np.ndarray:
    """Evalúa modelos lineales usando string de ecuación"""
    # Preparar variables para diferentes formatos de ecuación
    local_vars = {
        # Variables clásicas
        'x': x_orig, 'y': y_orig,
        # Variables indexadas (x0, x1) - formato común en modelos
        'x0': x_orig, 'x1': y_orig,
        # Nombres de predictores específicos
        pred_names[0]: x_orig, pred_names[1]: y_orig,
        # Funciones matemáticas
        'np': np, 'pow': np.power, 'exp': np.exp, 'log': np.log
    }
    
    try:
        # Limpiar y preparar ecuación
        eq = equation_string.replace('^', '**')
        
        # Si la ecuación tiene formato "y = ...", extraer solo la parte derecha
        if ' = ' in eq:
            eq = eq.split(' = ', 1)[1]
        
        # Evaluar ecuación con variables desnormalizadas
        Z = eval(eq, {"__builtins__": {}}, local_vars)
        
    except Exception as e:
        logger.warning(f"Error evaluando ecuación '{equation_string}': {e}")
        # En caso de error, generar superficie plana
        Z = np.full_like(x_orig, 0.0)
    
    return Z


def _evaluate_polynomial_model(model_data: Dict, x_orig: np.ndarray, y_orig: np.ndarray) -> np.ndarray:
    """
    Evalúa modelos polinomiales usando coeficientes directamente.
    Orden de términos PolynomialFeatures: [x0, x1, x0², x0*x1, x1²]
    """
    try:
        intercept = model_data['intercepto_original']
        coefs = model_data['coeficientes_originales']
        
        # Verificar que tengamos 5 coeficientes para modelo poly-2
        if len(coefs) != 5:
            logger.warning(f"Se esperaban 5 coeficientes para modelo poly-2, se encontraron {len(coefs)}")
            return np.full_like(x_orig, intercept)
        
        # Calcular términos polinomiales según orden de PolynomialFeatures
        # [x0, x1, x0², x0*x1, x1²]
        Z = (intercept + 
             coefs[0] * x_orig +                    # x0
             coefs[1] * y_orig +                    # x1 
             coefs[2] * (x_orig ** 2) +             # x0²
             coefs[3] * x_orig * y_orig +           # x0*x1
             coefs[4] * (y_orig ** 2))              # x1²
        
        return Z
        
    except Exception as e:
        logger.warning(f"Error evaluando modelo polinomial: {e}")
        return np.full_like(x_orig, 0.0)



