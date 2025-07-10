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


def get_model_ranges(model: Dict[str, Any]) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
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


def create_3d_plot(modelos_2pred: List[Dict[str, Any]], aeronave: str, parametro: str,
                   show_training_points: bool = True, show_model_surface: bool = True,
                   highlight_model_idx: Optional[int] = None, 
                   detalles_por_celda: Optional[Dict] = None) -> go.Figure:
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
    # Inicializar figura
    fig = go.Figure()
    
    if not modelos_2pred:
        fig.add_annotation(
            text="No hay modelos de 2 predictores disponibles",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig
        
    # Colores para diferentes tipos de modelo
    color_map = {
        'linear-2': '#1f77b4',  # Azul
        'poly-2': '#ff7f0e',    # Naranja
    }
    
    # Procesar cada modelo
    from .json_data_helpers import get_model_specific_data
    for idx, model in enumerate(modelos_2pred):
        if not isinstance(model, dict):
            continue
        tipo = model.get('tipo', 'unknown')
        predictores = model.get('predictores', [])
        if len(predictores) != 2:
            continue
        # Usar helper robusto para extraer datos de entrenamiento
        X_data, y_data, df_filtrado, warnings = get_model_specific_data({}, model)
        if X_data is None or y_data is None or len(X_data) == 0:
            logger.warning(f"Datos de entrenamiento faltantes o inválidos en modelo {idx}: {warnings}")
            continue
        X_array = np.array(X_data)
        y_array = np.array(y_data)
        # Rangos normalizados (0,1) para superficie
        x0_range = (X_array[:, 0].min(), X_array[:, 0].max())
        x1_range = (X_array[:, 1].min(), X_array[:, 1].max())
        y_range = (y_array.min(), y_array.max())
        # Normalizar datos de entrenamiento
        x0_norm = (X_array[:, 0] - x0_range[0]) / (x0_range[1] - x0_range[0]) if x0_range[1] > x0_range[0] else X_array[:, 0]
        x1_norm = (X_array[:, 1] - x1_range[0]) / (x1_range[1] - x1_range[0]) if x1_range[1] > x1_range[0] else X_array[:, 1]
        # 🔧 CAMBIO: No normalizar variable dependiente (Y) para mostrar valores originales
        y_norm = y_array  # Mantener valores originales
        is_highlighted = (highlight_model_idx is not None and idx == highlight_model_idx)
        color = color_map.get(tipo, '#2ca02c')
        opacity = 1.0 if is_highlighted else 0.7
        size = 8 if is_highlighted else 6
        if show_training_points:
            hover_text = [
                f"Modelo {idx+1}: {tipo}<br>" +
                f"{predictores[0]}: {X_array[i,0]:.3f}<br>" +
                f"{predictores[1]}: {X_array[i,1]:.3f}<br>" +
                f"{parametro}: {y_array[i]:.3f}<br>" +
                f"R²: {model.get('r2', 0):.3f}<br>" +
                f"MAPE: {model.get('mape', 0):.2f}%"
                for i in range(len(X_array))
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
        if show_model_surface:
            ecuacion_original = model.get('ecuacion_string', '')
            pred_names = model.get('predictores', [])
            x0_range = (X_array[:, 0].min(), X_array[:, 0].max())
            x1_range = (X_array[:, 1].min(), X_array[:, 1].max())
            try:
                from .symbiotic_surface import generate_symbiotic_surface_3d
                
                # Determinar qué datos pasar según el tipo de modelo
                if tipo.startswith('poly-'):
                    # Para modelos polinomiales, pasar el diccionario completo
                    model_data = model
                else:
                    # Para modelos lineales, pasar la ecuación string
                    model_data = ecuacion_original
                
                X_surf, Y_surf, Z_surf = generate_symbiotic_surface_3d(
                    model_data,
                    x0_range,
                    x1_range,
                    pred_names,
                    resolution=30
                )
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
                        f"Y (escala original): %{{z:.3f}}<br>" +
                        f"R²: {model.get('r2', 0):.3f}<br>" +
                        f"MAPE: {model.get('mape', 0):.2f}%<br>" +
                        "<extra></extra>"
                    )
                ))
            except Exception as e:
                logger.warning(f"Error creando superficie normalizada para modelo {idx}: {e}")
    
    # --- AÑADIR PUNTOS TEÓRICOS 3D ---
    # Extraer y añadir puntos teóricos de imputación para modelos de 2 predictores
    from .plot_model_curves import extract_theoretical_imputation_points
    
    theoretical_points_3d = extract_theoretical_imputation_points(
        modelos_2pred, 
        f"{aeronave}|{parametro}", 
        n_predictores_filter=2  # Solo modelos de 2 predictores para gráficos 3D
    )
    
    # Añadir los puntos teóricos 3D al gráfico
    add_theoretical_points_3d(fig, theoretical_points_3d, show_theoretical_points=True)
    
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


def update_3d_plot_with_normalized_equations(fig, modelos_por_celda, celda_key, 
                                            highlight_model_idx=None):
    """
    Actualiza un gráfico 3D existente para usar ecuaciones normalizadas coherentes.
    
    Esta función reemplaza las superficies existentes con versiones que usan
    la metodología normalizada (estilo asdasd.py) para máxima coherencia visual.
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        Figura 3D existente para actualizar
    modelos_por_celda : dict
        Datos de modelos
    celda_key : str
        Clave de la celda (ej: "A7|Payload")
    highlight_model_idx : int, optional
        Índice del modelo a destacar
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Figura actualizada con superficies normalizadas
    """
    try:
        # Obtener datos de la celda
        if celda_key not in modelos_por_celda:
            logger.warning(f"Celda {celda_key} no encontrada")
            return fig
        
        celda_data = modelos_por_celda[celda_key]
        modelos = celda_data.get('modelos', [])
        
        if not modelos:
            logger.warning(f"No hay modelos en celda {celda_key}")
            return fig
        
        # Obtener rangos de la celda
        try:
            from .plot_model_curves import get_best_model_ranges
        except ImportError:
            from plot_model_curves import get_best_model_ranges
            
        rango_x, rango_y, rango_z, pred_names = get_best_model_ranges(
            modelos_por_celda, celda_key
        )
        
        # Actualizar cada modelo con superficie normalizada
        traces_to_remove = []
        traces_to_add = []
        
        for trace_idx, trace in enumerate(fig.data):
            # Buscar trazas de superficie para reemplazar
            if hasattr(trace, 'type') and trace.type == 'surface':
                model_name = getattr(trace, 'name', '')
                if 'Modelo' in model_name:
                    traces_to_remove.append(trace_idx)
        
        # Remover trazas de superficie antiguas
        for idx in reversed(traces_to_remove):
            fig.data = list(fig.data[:idx]) + list(fig.data[idx+1:])
        
        # Añadir nuevas superficies normalizadas
        for i, modelo in enumerate(modelos):
            try:
                ecuacion_original = modelo.get('ecuacion_string', '')
                if not ecuacion_original:
                    continue
                
                # Generar superficie normalizada
                X_star, Y_star, Z_original = generate_normalized_surface_3d(
                    ecuacion_original, rango_x, rango_y, pred_names
                )
                
                # Configuración visual
                opacity = 0.7 if highlight_model_idx is None or i == highlight_model_idx else 0.3
                color = COLORS['model_lines'][i % len(COLORS['model_lines'])]
                
                # Añadir superficie normalizada
                fig.add_trace(go.Surface(
                    x=X_star,  # Coordenadas normalizadas [0,1]
                    y=Y_star,  # Coordenadas normalizadas [0,1]
                    z=Z_original,  # Valores en escala original
                    opacity=opacity,
                    colorscale=[[0, color], [1, color]],
                    showscale=False,
                    name=f'Modelo {i+1} (Normalizado)',
                    hovertemplate=(
                        f"<b>Modelo {i+1}</b><br>"
                        f"X* (norm): %{{x:.3f}}<br>"
                        f"Y* (norm): %{{y:.3f}}<br>"
                        f"Z (escala original): %{{z:.3f}}<br>"
                        f"<b>Ecuación normalizada activa</b>"
                        "<extra></extra>"
                    )
                ))
                
                logger.info(f"Superficie normalizada añadida para Modelo {i+1}")
                
            except Exception as e:
                logger.error(f"Error añadiendo superficie normalizada para modelo {i}: {e}")
        
        # Actualizar título para indicar uso de ecuaciones normalizadas
        current_title = fig.layout.title.text if fig.layout.title else ""
        if "Ecuaciones Normalizadas" not in current_title:
            fig.update_layout(
                title=f"{current_title} - Ecuaciones Normalizadas Activas"
            )
        
        logger.info(f"Gráfico 3D actualizado con {len(modelos)} superficies normalizadas")
        return fig
        
    except Exception as e:
        logger.error(f"Error actualizando gráfico 3D con ecuaciones normalizadas: {e}")
        return fig
