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
        Modelos filtrados por celda
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
        Figura de Plotly
    """
    
    # Crear figura vacía
    fig = go.Figure()
    
    celda_key = f"{aeronave}|{parametro}"
    
    if celda_key not in modelos_filtrados:
        fig.add_annotation(
            text="No hay modelos disponibles para esta combinación",
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


def add_model_data_points(fig: go.Figure, 
                         modelos: List[Dict], 
                         parametro: str,
                         show_training_points: bool = True) -> None:
    """
    Añade puntos de datos originales y de entrenamiento para cada modelo de 1 predictor.
    Los puntos se normalizan al rango [0, 1] para permitir superposición.
    
    Si no hay datos disponibles, añade una nota informativa al gráfico.
    
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
    
    model_data = []
    models_without_data = []
    
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
            models_without_data.append(f"{predictor} ({modelo.get('tipo', 'unknown')})")
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
            models_without_data.append(f"{predictor} ({modelo.get('tipo', 'unknown')})")
            continue
            
        if len(x_orig_valid) == 0:
            models_without_data.append(f"{predictor} ({modelo.get('tipo', 'unknown')})")
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
        
        model_data.append({
            'modelo': modelo,
            'predictor': predictor,
            'x_orig': x_orig_valid,
            'y_orig': y_orig_valid,
            'x_train': x_train_valid,
            'y_train': y_train_valid,
            'index': i
        })
    
    # Si no hay modelos con datos, mostrar mensaje informativo
    if not model_data:
        if models_without_data:
            warning_text = "No se encontraron datos para los puntos.<br>Modelos sin datos:<br>" + "<br>".join(models_without_data[:5])
            if len(models_without_data) > 5:
                warning_text += f"<br>... y {len(models_without_data) - 5} más"
        else:
            warning_text = "No hay modelos válidos para mostrar puntos de datos"
            
        fig.add_annotation(
            text=warning_text,
            xref="paper", yref="paper",
            x=0.5, y=0.3,
            xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=12, color="orange"),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="orange",
            borderwidth=1
        )
        return
    
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
    
    # Añadir nota sobre modelos sin datos si los hay
    if models_without_data:
        note_text = f"Nota: {len(models_without_data)} modelo(s) sin datos de puntos"
        fig.add_annotation(
            text=note_text,
            xref="paper", yref="paper",
            x=0.02, y=0.02,
            xanchor='left', yanchor='bottom',
            showarrow=False,
            font=dict(size=9, color="gray"),
            bgcolor="rgba(255, 255, 255, 0.8)"
        )


def add_normalized_model_curves(fig: go.Figure, 
                               modelos: List[Dict], 
                               parametro: str) -> None:
    """
    Añade curvas de modelos normalizadas al gráfico.
    Cada modelo usa su propio rango de datos normalizado a [0, 1].
    Si no hay datos originales, genera un rango sintético para mostrar la ecuación.
    
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
    synthetic_ranges_used = 0
    
    for i, modelo in enumerate(modelos):
        if not isinstance(modelo, dict) or modelo.get('n_predictores', 0) != 1:
            continue
            
        predictor = modelo.get('predictores', [None])[0]
        if not predictor:
            continue
            
        # Obtener datos originales del modelo para definir el rango
        df_original = get_model_original_data(modelo)
        
        x_range_orig = None
        x_min = None
        x_max = None
        using_synthetic_range = False
        
        if df_original is not None and not df_original.empty:
            x_data = df_original[predictor].dropna() if predictor in df_original.columns else pd.Series()
            
            if len(x_data) > 0:
                x_min = x_data.min()
                x_max = x_data.max()
                
                if x_max > x_min:
                    x_range_orig = np.linspace(x_min, x_max, 100)
        
        # Si no hay datos originales, crear un rango sintético
        if x_range_orig is None:
            # Generar un rango sintético basado en el tipo de modelo y sus coeficientes
            tipo = modelo.get('tipo', '')
            coefs = modelo.get('coeficientes_originales', [0])
            intercept = modelo.get('intercepto_original', 0)
            
            # Estimar un rango razonable basado en el tipo de modelo
            if tipo.startswith('linear'):
                # Para modelos lineales, usar un rango estándar
                x_min, x_max = 0, 10
            elif tipo.startswith('log'):
                # Para modelos logarítmicos, evitar valores cercanos a 0
                x_min, x_max = 0.1, 10
            elif tipo.startswith('exp'):
                # Para modelos exponenciales, usar un rango más pequeño
                x_min, x_max = 0, 5
            else:
                # Rango por defecto
                x_min, x_max = 0, 10
            
            x_range_orig = np.linspace(x_min, x_max, 100)
            using_synthetic_range = True
            synthetic_ranges_used += 1
            
            logger.info(f"Usando rango sintético para modelo {predictor} ({tipo}): [{x_min}, {x_max}]")
          # Verificar que tenemos un rango válido
        if x_max is None or x_min is None or x_max == x_min:
            warning_msg = f"Rango de X inválido para predictor {predictor}"
            warnings_added.append(warning_msg)
            continue
        
        # Generar predicciones en el rango original
        predictions = get_model_predictions_safe(modelo, x_range_orig)
        
        if predictions is None:
            warning_msg = f"Error generando predicciones para {predictor}"
            warnings_added.append(warning_msg)
            continue
        
        # Normalizar X al rango [0, 1]
        x_range_norm = (x_range_orig - float(x_min)) / (float(x_max) - float(x_min))
        
        # Color único por modelo
        color_idx = i % len(COLORS['model_lines'])
        model_color = COLORS['model_lines'][color_idx]
        
        # Información del modelo
        tipo = modelo.get('tipo', 'unknown')
        mape = modelo.get('mape', 0)
        r2 = modelo.get('r2', 0)
        ecuacion = modelo.get('ecuacion_string', '')
        
        # Crear información de hover para la curva
        hover_text = [
            f"Predictor: {predictor}<br>" +
            f"Valor original X: {x_range_orig[j]:.3f}<br>" +
            f"X normalizado: {x_range_norm[j]:.3f}<br>" +
            f"Predicción Y: {predictions[j]:.3f}<br>" +
            f"Modelo: {tipo}<br>" +
            f"Ecuación: {ecuacion}<br>" +
            f"MAPE: {mape:.3f}%<br>" +
            f"R²: {r2:.3f}" +
            (f"<br>[Rango sintético: {x_min:.1f}-{x_max:.1f}]" if using_synthetic_range else "")
            for j in range(len(x_range_norm))
        ]
        
        # Determinar el estilo de línea
        line_style = 'dash' if using_synthetic_range else 'solid'
        line_width = 2 if using_synthetic_range else 3
        
        # Añadir curva del modelo
        fig.add_trace(go.Scatter(
            x=x_range_norm,
            y=predictions,
            mode='lines',
            name=f'Curva - {predictor} ({tipo})' + (' [sintética]' if using_synthetic_range else ''),
            line=dict(
                color=model_color, 
                width=line_width,
                dash=line_style
            ),
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
            legendgroup=f'model_{i}',
            showlegend=True
        ))
        
        curves_added += 1
    
    # Añadir advertencias y notas al gráfico
    note_lines = []
    
    if warnings_added:
        note_lines.append("Advertencias:")
        note_lines.extend(warnings_added[:3])  # Limitar a 3 advertencias
        if len(warnings_added) > 3:
            note_lines.append(f"... y {len(warnings_added) - 3} más")
    
    if synthetic_ranges_used > 0:
        if note_lines:
            note_lines.append("")  # Línea vacía como separador
        note_lines.append(f"Nota: {synthetic_ranges_used} curva(s) con rango sintético")
        note_lines.append("(líneas punteadas, sin datos originales)")
    
    if note_lines:
        note_text = "<br>".join(note_lines)
        
        fig.add_annotation(
            text=note_text,
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            xanchor='left', yanchor='top',
            showarrow=False,
            font=dict(size=10, color="blue"),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="blue",
            borderwidth=1
        )
    
    logger.info(f"Añadidas {curves_added} curvas normalizadas para parámetro {parametro} ({synthetic_ranges_used} sintéticas)")


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
        
        # Generar predicciones según el tipo de modelo
        if tipo.startswith('linear'):
            predictions = coef * x_range + intercept
        elif tipo.startswith('poly'):
            # Para modelos polinomiales, necesitamos más coeficientes
            if len(coefs) >= 2:
                predictions = coefs[1] * x_range**2 + coef * x_range + intercept
            else:
                predictions = coef * x_range + intercept
        elif tipo.startswith('log'):
            # Logarítmico: y = a * log(x) + b
            # Evitar log de valores <= 0
            x_safe = np.where(x_range > 0, x_range, 1e-10)
            predictions = coef * np.log(x_safe) + intercept
        elif tipo.startswith('exp'):
            # Exponencial: y = a * exp(b * x) + c
            # Limitar el exponente para evitar overflow
            exp_arg = np.clip(coef * x_range, -700, 700)
            predictions = intercept * np.exp(exp_arg)
        elif tipo.startswith('pot'):
            # Potencial: y = a * x^b + c
            # Evitar valores negativos para exponentes no enteros
            x_safe = np.abs(x_range)
            x_safe = np.where(x_safe > 0, x_safe, 1e-10)
            predictions = intercept * (x_safe ** coef)
        else:
            # Por defecto, usar modelo lineal
            predictions = coef * x_range + intercept
        
        # Verificar que las predicciones son válidas
        if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
            logger.warning(f"Predicciones inválidas para modelo tipo {tipo}")
            return None
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error generando predicciones para modelo tipo {modelo.get('tipo', 'unknown')}: {e}")
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
        String formateado para hover
    """
    try:
        tipo = modelo.get('tipo', 'N/A')
        predictores = ', '.join(modelo.get('predictores', []))
        mape = modelo.get('mape', 0)
        r2 = modelo.get('r2', 0)
        corr = modelo.get('corr', 0)
        ecuacion = modelo.get('ecuacion_string', '')
        
        hover_info = (
            f"<b>Tipo:</b> {tipo}<br>"
            f"<b>Predictores:</b> {predictores}<br>"
            f"<b>MAPE:</b> {mape:.3f}%<br>"
            f"<b>R²:</b> {r2:.3f}<br>"
            f"<b>Correlación:</b> {corr:.3f}<br>"
        )
        
        if ecuacion:
            hover_info += f"<b>Ecuación:</b> {ecuacion}<br>"
        
        return hover_info
        
    except Exception as e:
        logger.error(f"Error creando hover info: {e}")
        return "Error en información del modelo"


def add_model_curves(fig: go.Figure, 
                    modelos: List[Dict], 
                    predictor: str, 
                    parametro: str,
                    df_original: pd.DataFrame) -> None:
    """
    Añade curvas de modelos al gráfico (función legacy).
    
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


def create_comparison_plot(modelos_filtrados: Dict, 
                         aeronave: str, 
                         parametro: str,
                         predictor_seleccionado: str,
                         show_training_points: bool = True,
                         show_model_curves: bool = True,
                         df_original: Optional[pd.DataFrame] = None,
                         df_filtrado: Optional[pd.DataFrame] = None) -> go.Figure:
    """
    Crea un gráfico de comparación entre datos originales, imputados y modelos (función legacy).
    """
    # Esta función mantiene la funcionalidad anterior para compatibilidad
    # En el nuevo sistema, se usa create_interactive_plot
    return create_interactive_plot(modelos_filtrados, aeronave, parametro, 
                                 show_training_points, show_model_curves)


def create_metrics_summary_table(modelos_filtrados: Dict, 
                                aeronave: str, 
                                parametro: str) -> pd.DataFrame:
    """
    Crea una tabla resumen con las métricas de los modelos.
    
    Parameters:
    -----------
    modelos_filtrados : Dict
        Modelos filtrados por celda
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
