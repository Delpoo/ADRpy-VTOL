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


def get_model_original_data(modelo: Dict) -> Optional[pd.DataFrame]:
    """
    Obtiene el DataFrame original asociado al modelo.
    
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
        if 'df_original' in modelo:
            return modelo['df_original']
        
        # Si no están en el modelo, podríamos intentar reconstruir o cargar
        # desde otras fuentes, pero por ahora retornamos None
        logger.warning("No se encontraron datos originales en el modelo")
        return None
        
    except Exception as e:
        logger.error(f"Error obteniendo datos originales: {e}")
        return None


def get_model_training_data(modelo: Dict) -> Optional[pd.DataFrame]:
    """
    Obtiene el DataFrame de entrenamiento asociado al modelo.
    
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
        if 'df_filtrado' in modelo:
            return modelo['df_filtrado']
        
        # Si no están en el modelo, intentar reconstruir desde datos de entrenamiento
        if 'datos_entrenamiento' in modelo:
            datos = modelo['datos_entrenamiento']
            if isinstance(datos, dict) and 'X' in datos and 'y' in datos:
                # Reconstruir DataFrame desde X e y
                X = datos['X']
                y = datos['y']
                
                if hasattr(X, 'shape') and hasattr(y, 'shape'):
                    # Asumiendo que X es array-like y y es array-like
                    predictor = modelo.get('predictores', ['X'])[0]
                    df = pd.DataFrame({predictor: X.flatten() if hasattr(X, 'flatten') else X})
                    
                    # Añadir y si es posible determinar el nombre del parámetro
                    # (esto podríamos mejorarlo pasando el parámetro como argumento)
                    return df
        
        logger.warning("No se encontraron datos de entrenamiento en el modelo")
        return None
        
    except Exception as e:
        logger.error(f"Error obteniendo datos de entrenamiento: {e}")
        return None


def create_interactive_plot(modelos_filtrados: Dict, 
                          aeronave: str, 
                          parametro: str,
                          show_training_points: bool = True,
                          show_model_curves: bool = True) -> go.Figure:
    """
    Crea un gráfico interactivo para visualizar modelos y datos.
    Permite superposición de modelos con diferentes predictores usando normalización X.
    
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
    
    # Filtrar solo modelos de 1 predictor
    modelos_1_pred = [m for m in modelos if isinstance(m, dict) and m.get('n_predictores', 0) == 1]
    
    if not modelos_1_pred:
        fig.add_annotation(
            text="Visualización disponible solo para modelos de 1 predictor",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig
    
    # Añadir puntos de datos originales y de entrenamiento para cada modelo
    if show_training_points:
        add_model_data_points(fig, modelos_1_pred, parametro, show_training_points)
    
    # Añadir curvas de modelos si se solicita
    if show_model_curves:
        add_normalized_model_curves(fig, modelos_1_pred, parametro)
    
    # Configurar layout con eje X normalizado
    fig.update_layout(
        title=f'Análisis de Modelos - {aeronave}: {parametro}',
        xaxis_title="Input normalizado (por predictor)",
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


def add_model_data_points(fig: go.Figure, 
                         modelos: List[Dict], 
                         parametro: str,
                         show_training_points: bool = True) -> None:
    """
    Añade puntos de datos originales y de entrenamiento para cada modelo de 1 predictor.
    Los puntos se normalizan al rango [0, 1] para permitir superposición.
    
    Parameters:
    -----------
    fig : go.Figure
        Figura de Plotly
    modelos : List[Dict]
        Lista de modelos de 1 predictor
    parametro : str
        Nombre del parámetro objetivo
    show_training_points : bool
        Si mostrar puntos de entrenamiento
    """
    
    # Recopilar todos los datos de modelos para normalización global opcional
    all_x_values = []
    all_y_values = []
    model_data = []
    
    for i, modelo in enumerate(modelos):
        if not isinstance(modelo, dict) or modelo.get('n_predictores', 0) != 1:
            continue
            
        predictor = modelo.get('predictores', [None])[0]
        if not predictor:
            continue
            
        # Obtener datos originales y de entrenamiento del modelo
        df_original = get_model_original_data(modelo)
        df_filtrado = get_model_training_data(modelo)
        
        if df_original is None or df_original.empty:
            logger.warning(f"No se pudieron obtener datos originales para modelo con predictor {predictor}")
            continue
              # Extraer datos válidos
        if predictor in df_original.columns and parametro in df_original.columns:
            # Crear máscaras para valores válidos (no NaN)
            x_valid_mask = df_original[predictor].notna()
            y_valid_mask = df_original[parametro].notna()
            both_valid_mask = x_valid_mask & y_valid_mask
            
            x_orig_valid = df_original.loc[both_valid_mask, predictor]
            y_orig_valid = df_original.loc[both_valid_mask, parametro]
        else:
            x_orig_valid = pd.Series(dtype=float)
            y_orig_valid = pd.Series(dtype=float)
            
        if len(x_orig_valid) == 0:
            continue
          # Datos de entrenamiento
        x_train_valid = pd.Series(dtype=float)
        y_train_valid = pd.Series(dtype=float)
        
        if show_training_points and df_filtrado is not None and not df_filtrado.empty:
            if predictor in df_filtrado.columns and parametro in df_filtrado.columns:
                # Crear máscaras para valores válidos (no NaN) en datos de entrenamiento
                x_train_valid_mask = df_filtrado[predictor].notna()
                y_train_valid_mask = df_filtrado[parametro].notna()
                both_train_valid_mask = x_train_valid_mask & y_train_valid_mask
                
                x_train_valid = df_filtrado.loc[both_train_valid_mask, predictor]
                y_train_valid = df_filtrado.loc[both_train_valid_mask, parametro]
            else:
                x_train_valid = pd.Series(dtype=float)
                y_train_valid = pd.Series(dtype=float)
        
        model_data.append({
            'modelo': modelo,
            'predictor': predictor,
            'x_orig': x_orig_valid,
            'y_orig': y_orig_valid,
            'x_train': x_train_valid,
            'y_train': y_train_valid,
            'index': i
        })
          # Recopilar todos los valores para referencia (opcional)
        try:
            if len(x_orig_valid) > 0:
                all_x_values.extend(list(x_orig_valid))
                all_y_values.extend(list(y_orig_valid))
        except Exception:
            # Ignorar errores en la recopilación de valores
            pass
    
    # Añadir puntos al gráfico con normalización individual por modelo
    for data in model_data:
        modelo = data['modelo']
        predictor = data['predictor']
        x_orig = data['x_orig']
        y_orig = data['y_orig']
        x_train = data['x_train']
        y_train = data['y_train']
        model_idx = data['index']
        
        if len(x_orig) == 0:
            continue
            
        # Normalizar X para este modelo específico al rango [0, 1]
        x_min = x_orig.min()
        x_max = x_orig.max()
        
        if x_max == x_min:
            # Si todos los valores X son iguales, centrar en 0.5
            x_orig_norm = pd.Series([0.5] * len(x_orig), index=x_orig.index)
            x_train_norm = pd.Series([0.5] * len(x_train), index=x_train.index) if len(x_train) > 0 else pd.Series()
        else:
            x_orig_norm = (x_orig - x_min) / (x_max - x_min)
            x_train_norm = (x_train - x_min) / (x_max - x_min) if len(x_train) > 0 else pd.Series()
        
        # Color único por modelo
        color_idx = model_idx % len(COLORS['model_lines'])
        model_color = COLORS['model_lines'][color_idx]
        
        # Información del modelo para hover
        tipo = modelo.get('tipo', 'unknown')
        mape = modelo.get('mape', 0)
        r2 = modelo.get('r2', 0)
        
        # Añadir puntos originales
        if len(x_orig_norm) > 0:
            hover_text = [
                f"Predictor: {predictor}<br>" +
                f"Valor original X: {x_orig.iloc[i]:.3f}<br>" +
                f"X normalizado: {x_orig_norm.iloc[i]:.3f}<br>" +
                f"Y: {y_orig.iloc[i]:.3f}<br>" +
                f"Modelo: {tipo}<br>" +
                f"MAPE: {mape:.3f}%<br>" +
                f"R²: {r2:.3f}"
                for i in range(len(x_orig_norm))
            ]
            
            fig.add_trace(go.Scatter(
                x=x_orig_norm,
                y=y_orig,
                mode='markers',
                name=f'Datos orig. - {predictor} ({tipo})',
                marker=dict(
                    color=model_color,
                    size=6,
                    opacity=0.6,
                    symbol='circle'
                ),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                legendgroup=f'model_{model_idx}',
                showlegend=True
            ))
        
        # Añadir puntos de entrenamiento si están disponibles
        if show_training_points and len(x_train_norm) > 0:
            hover_text_train = [
                f"Predictor: {predictor}<br>" +
                f"Valor original X: {x_train.iloc[i]:.3f}<br>" +
                f"X normalizado: {x_train_norm.iloc[i]:.3f}<br>" +
                f"Y: {y_train.iloc[i]:.3f}<br>" +
                f"Modelo: {tipo}<br>" +
                f"MAPE: {mape:.3f}%<br>" +
                f"R²: {r2:.3f}<br>" +
                f"[Punto entrenamiento]"
                for i in range(len(x_train_norm))
            ]
            
            fig.add_trace(go.Scatter(
                x=x_train_norm,
                y=y_train,
                mode='markers',
                name=f'Entren. - {predictor} ({tipo})',
                marker=dict(
                    color=model_color,
                    size=8,
                    opacity=0.9,
                    symbol='diamond',
                    line=dict(color='black', width=1)
                ),
                text=hover_text_train,
                hovertemplate='%{text}<extra></extra>',
                legendgroup=f'model_{model_idx}',
                showlegend=True
            ))


def add_normalized_model_curves(fig: go.Figure, 
                               modelos: List[Dict], 
                               parametro: str) -> None:
    """
    Añade curvas de modelos normalizadas al gráfico.
    Cada modelo usa su propio rango de datos normalizado a [0, 1].
    
    Parameters:
    -----------
    fig : go.Figure
        Figura de Plotly
    modelos : List[Dict]
        Lista de modelos de 1 predictor
    parametro : str
        Nombre del parámetro objetivo
    """
    
    curves_added = 0
    warnings_added = []
    
    for i, modelo in enumerate(modelos):
        if not isinstance(modelo, dict) or modelo.get('n_predictores', 0) != 1:
            continue
            
        predictor = modelo.get('predictores', [None])[0]
        if not predictor:
            continue
            
        # Obtener datos originales del modelo para definir el rango
        df_original = get_model_original_data(modelo)
        
        if df_original is None or df_original.empty:
            warning_msg = f"No hay datos para modelo {predictor}"
            warnings_added.append(warning_msg)
            logger.warning(warning_msg)
            continue
            
        x_data = df_original[predictor].dropna() if predictor in df_original.columns else pd.Series()
        
        if len(x_data) == 0:
            warning_msg = f"No hay datos válidos de X para predictor {predictor}"
            warnings_added.append(warning_msg)
            continue
        
        # Definir rango de valores X original para este modelo
        x_min = x_data.min()
        x_max = x_data.max()
        
        if x_max == x_min:
            warning_msg = f"Rango de X constante para predictor {predictor}"
            warnings_added.append(warning_msg)
            continue
       