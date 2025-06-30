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
from typing import List, Dict, Optional, Any, Tuple
import re
import logging

logger = logging.getLogger(__name__)


def extract_coefficients_from_equation(ecuacion_normalizada_latex: str, n_predictores: int) -> Optional[List[float]]:
    """
    Extrae coeficientes de la ecuación LaTeX normalizada.
    
    Para modelos polinómicos de 2 predictores, la ecuación tiene la forma:
    y = c0 + c1*x0 + c2*x1 + c3*x0² + c4*x1² + c5*x0*x1
    
    Parameters:
    -----------
    ecuacion_normalizada_latex : str
        Ecuación en formato LaTeX normalizada
    n_predictores : int
        Número de predictores (debe ser 2)
        
    Returns:
    --------
    Optional[List[float]]
        Lista de coeficientes [c0, c1, c2, c3, c4, c5] o None si hay error
    """
    if n_predictores != 2:
        return None
        
    try:
        # Remover espacios y limpiar la ecuación
        ecuacion = ecuacion_normalizada_latex.replace(' ', '')
        # Limpiar signos problemáticos
        ecuacion = ecuacion.replace('+-', '-')
        
        # Buscar coeficientes usando regex más robusto
        # Patrón para intercepto (después del =)
        intercepto_match = re.search(r'y=([+-]?[\d\.e-]+)', ecuacion)
        c0 = float(intercepto_match.group(1)) if intercepto_match else 0.0
        
        # Patrón para términos lineales x_{0}
        x0_linear_pattern = r'([+-]?[\d\.e-]+)x_\{0\}(?![²\^])'  # No seguido por ² o ^
        x0_match = re.search(x0_linear_pattern, ecuacion)
        c1 = float(x0_match.group(1)) if x0_match else 0.0
        
        # Patrón para términos lineales x_{1}
        x1_linear_pattern = r'([+-]?[\d\.e-]+)x_\{1\}(?![²\^])'  # No seguido por ² o ^
        x1_match = re.search(x1_linear_pattern, ecuacion)
        c2 = float(x1_match.group(1)) if x1_match else 0.0
        
        # Para modelos polinómicos, buscar términos cuadráticos y cruzados
        # x_{2} = x_{0}²
        x0_sq_pattern = r'([+-]?[\d\.e-]+)x_\{2\}'
        x0_sq_match = re.search(x0_sq_pattern, ecuacion)
        c3 = float(x0_sq_match.group(1)) if x0_sq_match else 0.0
        
        # x_{3} = x_{1}²
        x1_sq_pattern = r'([+-]?[\d\.e-]+)x_\{3\}'
        x1_sq_match = re.search(x1_sq_pattern, ecuacion)
        c4 = float(x1_sq_match.group(1)) if x1_sq_match else 0.0
        
        # x_{4} = x_{0}*x_{1}
        x0x1_pattern = r'([+-]?[\d\.e-]+)x_\{4\}'
        x0x1_match = re.search(x0x1_pattern, ecuacion)
        c5 = float(x0x1_match.group(1)) if x0x1_match else 0.0
        
        return [c0, c1, c2, c3, c4, c5]
        
    except Exception as e:
        logger.warning(f"Error extrayendo coeficientes de ecuación: {e}")
        return None


def generate_model_surface(coefficients: List[float], x_range: Tuple[float, float], 
                          y_range: Tuple[float, float], resolution: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Genera superficie 3D del modelo usando los coeficientes.
    
    Parameters:
    -----------
    coefficients : List[float]
        Coeficientes [c0, c1, c2, c3, c4, c5] donde:
        y = c0 + c1*x0 + c2*x1 + c3*x0² + c4*x1² + c5*x0*x1
    x_range : Tuple[float, float]
        Rango para el primer predictor (x0)
    y_range : Tuple[float, float]
        Rango para el segundo predictor (x1)
    resolution : int
        Resolución de la malla
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Arrays X, Y, Z para la superficie
    """
    c0, c1, c2, c3, c4, c5 = coefficients
    
    # Crear malla
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Calcular Z usando la ecuación polinómica
    Z = c0 + c1*X + c2*Y + c3*X**2 + c4*Y**2 + c5*X*Y
    
    return X, Y, Z


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
    y_norm = (y_array - y_range[0]) / (y_range[1] - y_range[0])
    
    return x0_norm, x1_norm, y_norm


def get_model_ranges(model: Dict[str, Any]) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """
    Obtiene los rangos de normalización del modelo.
    
    Parameters:
    -----------
    model : Dict[str, Any]
        Diccionario del modelo con datos de entrenamiento
        
    Returns:
    --------
    Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
        Rangos (x0_range, x1_range, y_range)
    """
    X_original = model['datos_entrenamiento']['X_original']
    y_original = model['datos_entrenamiento']['y_original']
    
    X_array = np.array(X_original)
    y_array = np.array(y_original)
    
    x0_range = (X_array[:, 0].min(), X_array[:, 0].max())
    x1_range = (X_array[:, 1].min(), X_array[:, 1].max())
    y_range = (y_array.min(), y_array.max())
    
    return x0_range, x1_range, y_range


def create_3d_plot(modelos_2pred: List[Dict[str, Any]], aeronave: str, parametro: str,
                   show_training_points: bool = True, show_model_surface: bool = True,
                   highlight_model_idx: Optional[int] = None) -> go.Figure:
    """
    Crea gráfico 3D interactivo para modelos con 2 predictores.
    
    Parameters:
    -----------
    modelos_2pred : List[Dict[str, Any]]
        Lista de modelos con 2 predictores
    aeronave : str
        Nombre de la aeronave
    parametro : str
        Nombre del parámetro
    show_training_points : bool
        Si mostrar puntos de entrenamiento
    show_model_surface : bool
        Si mostrar superficie del modelo
    highlight_model_idx : Optional[int]
        Índice del modelo a resaltar
        
    Returns:
    --------
    go.Figure
        Figura 3D de Plotly
    """
    if not modelos_2pred:
        fig = go.Figure()
        fig.add_annotation(
            text="No hay modelos de 2 predictores disponibles",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig
    
    fig = go.Figure()
    
    # Colores para diferentes tipos de modelo
    color_map = {
        'linear-2': '#1f77b4',  # Azul
        'poly-2': '#ff7f0e',    # Naranja
    }
    
    # Procesar cada modelo
    for idx, model in enumerate(modelos_2pred):
        if not isinstance(model, dict):
            continue
            
        tipo = model.get('tipo', 'unknown')
        predictores = model.get('predictores', [])
        
        # Verificar que tengamos 2 predictores
        if len(predictores) != 2:
            continue
            
        # Obtener datos de entrenamiento
        try:
            X_original = model['datos_entrenamiento']['X_original']
            y_original = model['datos_entrenamiento']['y_original']
        except KeyError:
            logger.warning(f"Datos de entrenamiento faltantes en modelo {idx}")
            continue
            
        # Obtener rangos de normalización
        x0_range, x1_range, y_range = get_model_ranges(model)
        
        # Normalizar datos de entrenamiento
        x0_norm, x1_norm, y_norm = normalize_training_data(
            X_original, y_original, x0_range, x1_range, y_range
        )
        
        # Color y opacidad según si está resaltado
        is_highlighted = (highlight_model_idx is not None and idx == highlight_model_idx)
        color = color_map.get(tipo, '#2ca02c')
        opacity = 1.0 if is_highlighted else 0.7
        size = 8 if is_highlighted else 6
        
        # Agregar puntos de entrenamiento
        if show_training_points:
            hover_text = [
                f"Modelo {idx+1}: {tipo}<br>" +
                f"{predictores[0]}: {X_original[i][0]:.3f}<br>" +
                f"{predictores[1]}: {X_original[i][1]:.3f}<br>" +
                f"{parametro}: {y_original[i]:.3f}<br>" +
                f"R²: {model.get('r2', 0):.3f}<br>" +
                f"MAPE: {model.get('mape', 0):.2f}%"
                for i in range(len(X_original))
            ]
            
            fig.add_trace(go.Scatter3d(
                x=x0_norm,
                y=x1_norm, 
                z=y_norm,
                mode='markers',
                marker=dict(
                    size=size,
                    color=color,
                    opacity=opacity,
                    line=dict(width=2, color='black' if is_highlighted else 'white')
                ),
                name=f"Datos {tipo} (Modelo {idx+1})",
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                showlegend=True
            ))
        
        # Agregar superficie del modelo
        if show_model_surface:
            # Extraer coeficientes de la ecuación normalizada
            ecuacion_normalizada_latex = model.get('ecuacion_normalizada_latex', '')
            coefficients = extract_coefficients_from_equation(ecuacion_normalizada_latex, 2)
            
            if coefficients:
                # Generar superficie
                try:
                    X_surf, Y_surf, Z_surf = generate_model_surface(
                        coefficients, 
                        x_range=(0, 1),  # Rango normalizado
                        y_range=(0, 1),  # Rango normalizado
                        resolution=30
                    )
                    
                    # Ajustar opacidad de la superficie
                    surf_opacity = 0.6 if is_highlighted else 0.3
                    
                    fig.add_trace(go.Surface(
                        x=X_surf,
                        y=Y_surf,
                        z=Z_surf,
                        colorscale='Viridis',
                        opacity=surf_opacity,
                        name=f"Superficie {tipo} (Modelo {idx+1})",
                        showscale=is_highlighted,
                        hovertemplate=(
                            f"Modelo {idx+1}: {tipo}<br>" +
                            f"X0 (norm): %{{x:.3f}}<br>" +
                            f"X1 (norm): %{{y:.3f}}<br>" +
                            f"Y (norm): %{{z:.3f}}<br>" +
                            f"R²: {model.get('r2', 0):.3f}<br>" +
                            f"MAPE: {model.get('mape', 0):.2f}%<br>" +
                            "<extra></extra>"
                        )
                    ))
                    
                except Exception as e:
                    logger.warning(f"Error creando superficie para modelo {idx}: {e}")
            else:
                logger.warning(f"No se pudieron extraer coeficientes del modelo {idx}")
    
    # Configurar layout 3D
    fig.update_layout(
        title=f"Vista 3D - {aeronave} | {parametro}",
        scene=dict(
            xaxis_title=f"{modelos_2pred[0]['predictores'][0]} (normalizado)" if modelos_2pred else "X0",
            yaxis_title=f"{modelos_2pred[0]['predictores'][1]} (normalizado)" if modelos_2pred else "X1", 
            zaxis_title=f"{parametro} (normalizado)",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            aspectmode='cube'
        ),
        width=800,
        height=600,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig


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
