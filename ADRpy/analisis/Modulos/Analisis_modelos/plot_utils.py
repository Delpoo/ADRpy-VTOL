"""
Utilidades para Generación de Gráficos Interactivos
===================================================

Este módulo contiene funciones para crear visualizaciones interactivas
de los modelos de imputación usando Plotly.

Funciones principales:
- create_interactive_plot: Crea el gráfico principal interactivo
- add_model_traces: Añade trazas de modelos al gráfico
- add_data_points: Añade puntos de datos al gráfico
- create_hover_info: Crea información de hover personalizada
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

# Configuración de colores
COLORS = {
    'original_data': '#B0BEC5',  # Gris claro para datos originales
    'training_data': '#FF6B6B',  # Rojo para datos de entrenamiento
    'model_lines': px.colors.qualitative.Set1,  # Colores para líneas de modelos
    'selected_model': '#2E8B57'  # Verde para modelo seleccionado
}

# Configuración de símbolos para diferentes tipos de modelo
SYMBOLS = {
    'linear': 'circle',
    'poly': 'square',
    'log': 'diamond',
    'exp': 'triangle-up',
    'pot': 'star'
}


def create_interactive_plot(modelos_filtrados: Dict, 
                          aeronave: str, 
                          parametro: str,
                          show_training_points: bool = True,
                          show_model_curves: bool = True) -> go.Figure:
    """
    Crea un gráfico interactivo para visualizar modelos y datos.
    
    Parameters:
    -----------
    modelos_filtrados : Dict
        Diccionario con modelos filtrados
    aeronave : str
        Nombre de la aeronave
    parametro : str
        Parámetro objetivo
    show_training_points : bool
        Si mostrar puntos de entrenamiento
    show_model_curves : bool
        Si mostrar curvas de modelos
        
    Returns:
    --------
    go.Figure
        Figura de Plotly con el gráfico interactivo
    """
    fig = go.Figure()
    
    celda_key = f"{aeronave}|{parametro}"
    
    if celda_key not in modelos_filtrados:
        # Crear gráfico vacío con mensaje
        fig.add_annotation(
            text="No hay modelos disponibles para la selección actual",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig
    
    modelos = modelos_filtrados[celda_key]
    
    # Obtener datos de referencia del primer modelo válido
    df_original = None
    df_filtrado = None
    predictores = None
    
    for modelo in modelos:
        if isinstance(modelo, dict):
            datos_entrenamiento = modelo.get('datos_entrenamiento', {})
            
            # Datos originales
            df_original_dict = datos_entrenamiento.get('df_original')
            if df_original_dict:
                df_original = pd.DataFrame(df_original_dict)
            
            # Datos filtrados
            df_filtrado_dict = datos_entrenamiento.get('df_filtrado')
            if df_filtrado_dict:
                df_filtrado = pd.DataFrame(df_filtrado_dict)
            
            # Predictores
            predictores = modelo.get('predictores', [])
            
            if df_original is not None and predictores:
                break
    
    # Solo manejar modelos de 1 predictor por ahora
    if not predictores or len(predictores) != 1:
        fig.add_annotation(
            text="Visualización disponible solo para modelos de 1 predictor",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig
    
    predictor = predictores[0]
      # Verificar que las columnas existen y limpiar datos NaN
    if df_original is None:
        fig.add_annotation(
            text="No hay datos originales disponibles",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig
    
    # Limpiar NaN strings del JSON
    df_original = df_original.replace('NaN', np.nan)
    
    if (predictor not in df_original.columns or 
        parametro not in df_original.columns):
        fig.add_annotation(
            text=f"Columnas faltantes: {predictor} o {parametro}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig
    
    # Filtrar datos válidos (no NaN) para la visualización
    df_plot = df_original[[predictor, parametro]].dropna()
    
    if df_plot.empty:
        fig.add_annotation(
            text="No hay datos válidos para visualizar",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig
      # Añadir puntos de datos originales
    fig.add_trace(go.Scatter(
        x=df_plot[predictor],
        y=df_plot[parametro],
        mode='markers',
        name='Datos originales',
        marker=dict(
            color=COLORS['original_data'],
            size=8,
            opacity=0.6
        ),
        hovertemplate=f'<b>Datos originales</b><br>' +
                     f'{predictor}: %{{x}}<br>' +
                     f'{parametro}: %{{y}}<extra></extra>'
    ))
      # Añadir puntos de entrenamiento si están disponibles y se solicita
    if show_training_points and df_filtrado is not None:
        # Limpiar NaN strings también en df_filtrado
        df_filtrado = df_filtrado.replace('NaN', np.nan)
        
        if predictor in df_filtrado.columns and parametro in df_filtrado.columns:
            df_training = df_filtrado[[predictor, parametro]].dropna()
            
            if not df_training.empty:
                fig.add_trace(go.Scatter(
                    x=df_training[predictor],
                    y=df_training[parametro],
                    mode='markers',
                    name='Datos de entrenamiento',
                    marker=dict(
                        color=COLORS['training_data'],
                        size=10,
                        opacity=0.8,
                        symbol='diamond'
                    ),
                    hovertemplate=f'<b>Datos de entrenamiento</b><br>' +
                                 f'{predictor}: %{{x}}<br>' +
                                 f'{parametro}: %{{y}}<extra></extra>'
                ))
      # Añadir curvas de modelos si se solicita
    if show_model_curves:
        add_model_curves(fig, modelos, predictor, parametro, df_plot)
    
    # Configurar layout
    fig.update_layout(
        title=f'Análisis de Modelos - {aeronave}: {parametro}',
        xaxis_title=predictor,
        yaxis_title=parametro,
        hovermode='closest',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        template='plotly_white',
        width=900,
        height=600
    )
    
    return fig


def add_model_curves(fig: go.Figure, 
                    modelos: List[Dict], 
                    predictor: str, 
                    parametro: str,
                    df_original: pd.DataFrame) -> None:
    """
    Añade curvas de modelos al gráfico.
    
    Parameters:
    -----------
    fig : go.Figure
        Figura de Plotly
    modelos : List[Dict]
        Lista de modelos
    predictor : str
        Nombre del predictor
    parametro : str
        Nombre del parámetro objetivo
    df_original : pd.DataFrame
        DataFrame con datos originales
    """
    if df_original.empty:
        return
    
    # Crear rango de valores X para las curvas basado en datos válidos
    x_data = df_original[predictor].dropna()
    if x_data.empty:
        return
        
    x_min = x_data.min()
    x_max = x_data.max()
    x_range = np.linspace(x_min, x_max, 100)
    
    color_idx = 0
    curves_added = 0
    
    for modelo in modelos:
        if not isinstance(modelo, dict):
            continue
        
        # Solo manejar modelos de 1 predictor
        if modelo.get('n_predictores', 0) != 1:
            continue
        
        # Verificar que el predictor coincide
        model_predictors = modelo.get('predictores', [])
        if not model_predictors or model_predictors[0] != predictor:
            continue
        
        # Generar predicciones
        predictions = get_model_predictions_safe(modelo, x_range)
        if predictions is None:
            continue
        
        # Información del modelo
        tipo = modelo.get('tipo', 'unknown')
        ecuacion = modelo.get('ecuacion_string', '')
        mape = modelo.get('mape', 0)
        r2 = modelo.get('r2', 0)
        
        # Información de hover
        hover_info = create_model_hover_info(modelo)
        
        # Determinar color y símbolo
        tipo_base = tipo.split('-')[0] if '-' in tipo else tipo
        symbol = SYMBOLS.get(tipo_base, 'circle')
        color = COLORS['model_lines'][color_idx % len(COLORS['model_lines'])]
        
        # Añadir curva del modelo
        fig.add_trace(go.Scatter(
            x=x_range,
            y=predictions,
            mode='lines',
            name=f'{tipo} (R²={r2:.3f})',
            line=dict(color=color, width=2),
            hovertemplate=hover_info + '<extra></extra>',
            legendgroup=tipo
        ))
        
        color_idx += 1
        curves_added += 1
    
    # Log información para debugging
    logger.info(f"Añadidas {curves_added} curvas de modelos para {predictor} -> {parametro}")


def get_model_predictions_safe(modelo: Dict, x_range: np.ndarray) -> Optional[np.ndarray]:
    """
    Genera predicciones del modelo de forma segura.
    
    Parameters:
    -----------
    modelo : Dict
        Diccionario con información del modelo
    x_range : np.ndarray
        Rango de valores X
        
    Returns:
    --------
    Optional[np.ndarray]
        Predicciones o None si hay error
    """
    try:
        tipo = modelo.get('tipo', '')
        coefs = modelo.get('coeficientes_originales', [])
        intercept = modelo.get('intercepto_original', 0)
        
        # Validaciones básicas
        if not coefs or len(coefs) == 0:
            logger.warning(f"No hay coeficientes para modelo tipo {tipo}")
            return None
        
        if intercept is None:
            intercept = 0
            
        coef = coefs[0]
        
        # Validar que los coeficientes son números válidos
        if not isinstance(coef, (int, float)) or np.isnan(coef) or np.isinf(coef):
            logger.warning(f"Coeficiente inválido para modelo tipo {tipo}: {coef}")
            return None
        
        if not isinstance(intercept, (int, float)) or np.isnan(intercept) or np.isinf(intercept):
            logger.warning(f"Intercepto inválido para modelo tipo {tipo}: {intercept}")
            return None
        
        # Generar predicciones según el tipo
        if 'linear' in tipo:
            predictions = intercept + coef * x_range
            
        elif 'poly' in tipo:
            # Para modelos polinómicos de grado 2
            if len(coefs) >= 2:
                predictions = intercept + coefs[0] * x_range + coefs[1] * (x_range ** 2)
            else:
                predictions = intercept + coef * x_range
            
        elif 'log' in tipo:
            # Modelo logarítmico: y = intercept + coef * log(x)
            x_safe = np.where(x_range > 0, x_range, 1e-10)
            predictions = intercept + coef * np.log(x_safe)
            
        elif 'exp' in tipo:
            # Modelo exponencial: y = intercept + coef * exp(x)
            # Limitar para evitar overflow
            x_limited = np.clip(x_range, -10, 10)  # Más conservador
            try:
                predictions = intercept + coef * np.exp(x_limited)
            except OverflowError:
                logger.warning(f"Overflow en modelo exponencial: {tipo}")
                return None
            
        elif 'pot' in tipo:
            # Modelo de potencia: y = intercept + coef * x^2
            predictions = intercept + coef * (x_range ** 2)
            
        else:
            # Tipo desconocido, usar lineal como fallback
            logger.info(f"Tipo de modelo desconocido: {tipo}, usando lineal")
            predictions = intercept + coef * x_range
        
        # Verificar que las predicciones son válidas
        if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
            logger.warning(f"Predicciones inválidas para modelo tipo {tipo}")
            return None
        
        # Verificar rangos razonables
        if np.any(np.abs(predictions) > 1e6):
            logger.warning(f"Predicciones fuera de rango para modelo tipo {tipo}")
            return None
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error generando predicciones para tipo {modelo.get('tipo', 'unknown')}: {e}")
        return None


def create_model_hover_info(modelo: Dict) -> str:
    """
    Crea información de hover para un modelo.
    
    Parameters:
    -----------
    modelo : Dict
        Diccionario con información del modelo
        
    Returns:
    --------
    str
        HTML formateado para hover
    """
    info_lines = []
    
    # Información básica
    tipo = modelo.get('tipo', 'N/A')
    predictores = modelo.get('predictores', [])
    
    info_lines.append(f"<b>Tipo:</b> {tipo}")
    
    if predictores:
        pred_text = ', '.join(predictores)
        info_lines.append(f"<b>Predictores:</b> {pred_text}")
    
    # Ecuación
    ecuacion = modelo.get('ecuacion_string', '')
    if ecuacion and len(ecuacion) < 100:  # Limitar longitud
        info_lines.append(f"<b>Ecuación:</b> {ecuacion}")
    
    # Métricas
    mape = modelo.get('mape')
    r2 = modelo.get('r2')
    confianza = modelo.get('Confianza')
    
    if mape is not None:
        info_lines.append(f"<b>MAPE:</b> {mape:.3f}%")
    if r2 is not None:
        info_lines.append(f"<b>R²:</b> {r2:.3f}")
    if confianza is not None:
        info_lines.append(f"<b>Confianza:</b> {confianza:.3f}")
    
    # Entrenamiento
    n_muestras = modelo.get('n_muestras_entrenamiento')
    if n_muestras:
        info_lines.append(f"<b>N° muestras:</b> {n_muestras}")
    
    return '<br>'.join(info_lines)


def create_comparison_plot(modelos_filtrados: Dict, 
                          aeronave: str, 
                          parametro: str,
                          comparison_type: str = 'by_type') -> go.Figure:
    """
    Crea un gráfico de comparación entre modelos.
    
    Parameters:
    -----------
    modelos_filtrados : Dict
        Diccionario con modelos filtrados
    aeronave : str
        Nombre de la aeronave
    parametro : str
        Parámetro objetivo
    comparison_type : str
        Tipo de comparación ('by_type', 'best_overall', 'by_predictors')
        
    Returns:
    --------
    go.Figure
        Figura de comparación
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Métricas de Rendimiento', 'Distribución de Errores'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    celda_key = f"{aeronave}|{parametro}"
    
    if celda_key not in modelos_filtrados:
        return fig
    
    modelos = modelos_filtrados[celda_key]
    
    # Extraer métricas
    tipos = []
    mapes = []
    r2s = []
    confianzas = []
    
    for modelo in modelos:
        if isinstance(modelo, dict):
            tipos.append(modelo.get('tipo', 'unknown'))
            mapes.append(modelo.get('mape', 0))
            r2s.append(modelo.get('r2', 0))
            confianzas.append(modelo.get('Confianza', 0))
    
    # Gráfico de barras para métricas
    fig.add_trace(
        go.Bar(x=tipos, y=r2s, name='R²', marker_color='blue'),
        row=1, col=1
    )
    
    # Gráfico de dispersión para MAPE vs Confianza
    fig.add_trace(
        go.Scatter(
            x=mapes, y=confianzas, mode='markers+text',
            text=tipos, textposition='top center',
            name='MAPE vs Confianza',
            marker=dict(size=10, color='red')
        ),
        row=1, col=2
    )
    
    # Actualizar layout
    fig.update_xaxes(title_text="Tipo de Modelo", row=1, col=1)
    fig.update_yaxes(title_text="R²", row=1, col=1)
    fig.update_xaxes(title_text="MAPE (%)", row=1, col=2)
    fig.update_yaxes(title_text="Confianza", row=1, col=2)
    
    fig.update_layout(
        title=f'Comparación de Modelos - {aeronave}: {parametro}',
        showlegend=True,
        template='plotly_white'
    )
    
    return fig


def create_metrics_summary_table(modelos_filtrados: Dict, 
                                aeronave: str, 
                                parametro: str) -> pd.DataFrame:
    """
    Crea una tabla resumen con las métricas de todos los modelos.
    
    Parameters:
    -----------
    modelos_filtrados : Dict
        Diccionario con modelos filtrados
    aeronave : str
        Nombre de la aeronave
    parametro : str
        Parámetro objetivo
        
    Returns:
    --------
    pd.DataFrame
        DataFrame con resumen de métricas
    """
    celda_key = f"{aeronave}|{parametro}"
    
    if celda_key not in modelos_filtrados:
        return pd.DataFrame()
    
    modelos = modelos_filtrados[celda_key]
    
    summary_data = []
    
    for i, modelo in enumerate(modelos):
        if isinstance(modelo, dict):
            row = {
                'ID': i + 1,
                'Tipo': modelo.get('tipo', 'N/A'),
                'Predictores': ', '.join(modelo.get('predictores', [])),
                'N° Predictores': modelo.get('n_predictores', 0),
                'MAPE (%)': round(modelo.get('mape', 0), 3),
                'R²': round(modelo.get('r2', 0), 3),
                'Correlación': round(modelo.get('corr', 0), 3),
                'Confianza': round(modelo.get('Confianza', 0), 3),
                'N° Muestras': modelo.get('n_muestras_entrenamiento', 0),
                'Advertencia': modelo.get('Advertencia', '')
            }
            summary_data.append(row)
    
    return pd.DataFrame(summary_data)
