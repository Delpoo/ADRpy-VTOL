"""
plot_interactive.py

Funciones de composición de la visualización y lógica interactiva:
- Composición de la figura principal
- Agregado de puntos originales y de entrenamiento
- Gráficos de comparación y métricas
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import logging
from .plot_config import COLORS, SYMBOLS, _ensure_list
from .plot_data_access import get_model_original_data, get_model_training_data
from .plot_model_curves import get_model_predictions_safe, add_normalized_model_curves, create_model_hover_info


def _to_list_safe(val):
    if val is None:
        return []
    if isinstance(val, (list, np.ndarray, pd.Series)):
        return list(val)
    return [val]


logger = logging.getLogger(__name__)


def create_interactive_plot(
    modelos_filtrados: Dict,
    aeronave: str,
    parametro: str,
    show_training_points: bool = True,
    show_model_curves: bool = True,
    show_synthetic_curves: bool = True,
    highlight_model_idx: Optional[int] = None,
    detalles_por_celda: Optional[Dict] = None,
    selected_imputation_methods: Optional[List[str]] = None,
    show_only_real_curves: bool = False
) -> go.Figure:
    """
    Crea un gráfico interactivo para visualizar modelos y datos.
    Todos los valores de X (curvas y puntos) se normalizan (min-max scaling) a [0, 1] POR PREDICTOR, nunca global ni individual por punto.
    El eje X se rotula como "X adimensional".
    Además, agrega marcadores para los valores imputados por correlación, similitud y promedio ponderado (final),
    usando los datos de detalles_por_celda si están disponibles.
    Si show_only_real_curves=True, solo se grafican curvas con datos reales.
    Si highlight_model_idx está definido, resalta ese modelo y baja la opacidad de los demás.
    """
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
    # --- Normalización SIEMPRE por predictor, nunca global ni individual por punto ---
    for i, modelo in enumerate(modelos_1_pred):
        predictor = modelo.get('predictores', [None])[0]
        if not predictor:
            continue
        # Determinar resaltado y opacidad para el modelo actual
        is_highlighted = highlight_model_idx is not None and i == highlight_model_idx
        is_dimmed = highlight_model_idx is not None and i != highlight_model_idx
        marker_opacity = 1.0 if is_highlighted else (0.4 if is_dimmed else 0.6)
        marker_size = 8 if is_highlighted else 6
        df_original = get_model_original_data(modelo)
        df_filtrado = get_model_training_data(modelo)
        # Calcular min y max SOLO de ese predictor usando df_original
        x_min, x_max = None, None
        y_min, y_max = None, None
        x_data = None
        y_data = None
        if df_original is not None and not df_original.empty and predictor in df_original.columns and parametro in df_original.columns:
            x_data = df_original[predictor].dropna()
            y_data = df_original[parametro].dropna()
            if len(x_data) > 0:
                x_min = x_data.min()
                x_max = x_data.max()
            if len(y_data) > 0:
                y_min = y_data.min()
                y_max = y_data.max()
        # --- PUNTOS ORIGINALES ---
        if df_original is not None and not df_original.empty and predictor in df_original.columns and parametro in df_original.columns and x_min is not None and x_max is not None and y_min is not None and y_max is not None:
            mask = df_original[predictor].notna() & df_original[parametro].notna()
            x_orig = df_original.loc[mask, predictor]
            y_orig = df_original.loc[mask, parametro]
            # Asegurar que sean Series de 1D y no DataFrame
            if isinstance(x_orig, pd.DataFrame):
                x_orig = x_orig.squeeze()
                if isinstance(x_orig, pd.DataFrame):
                    x_orig = x_orig.iloc[:,0]
            if not isinstance(x_orig, pd.Series):
                x_orig = pd.Series(x_orig)
            if isinstance(y_orig, pd.DataFrame):
                y_orig = y_orig.squeeze()
                if isinstance(y_orig, pd.DataFrame):
                    y_orig = y_orig.iloc[:,0]
            if not isinstance(y_orig, pd.Series):
                y_orig = pd.Series(y_orig)
            if x_max != x_min:
                x_orig_norm = (x_orig - x_min) / (x_max - x_min)
            else:
                x_orig_norm = pd.Series([0.5] * len(x_orig), index=x_orig.index if hasattr(x_orig, 'index') else None)
            if y_max != y_min:
                y_orig_norm = (y_orig - y_min) / (y_max - y_min)
            else:
                y_orig_norm = pd.Series([0.5] * len(y_orig), index=y_orig.index if hasattr(y_orig, 'index') else None)
            x_orig_list = _to_list_safe(x_orig)
            x_orig_norm_list = _to_list_safe(x_orig_norm)
            y_orig_norm_list = _to_list_safe(y_orig_norm)
            fig.add_trace(go.Scatter(
                x=x_orig_norm_list,
                y=y_orig_norm_list,
                mode='markers',
                name=f'Datos orig. - {predictor} ({modelo.get("tipo", "unknown")})',
                marker=dict(
                    color=COLORS['model_lines'][i % len(COLORS['model_lines'])],
                    size=marker_size,
                    opacity=marker_opacity,
                    symbol='circle',
                    line=dict(color='darkgray', width=1)  # Borde para mejor visibilidad
                ),
                # Información personalizada para identificar el modelo en callbacks
                customdata=[i] * len(x_orig_norm_list),
                text=[
                    f"Aeronave: {aeronave}<br>Parámetro: {parametro}<br>Predictor: {predictor}<br>Valor original X: {xv:.3f}<br>X adimensional: {xn:.3f}<br>Y normalizado: {yn:.3f}" for xv, xn, yn in zip(x_orig_list, x_orig_norm_list, y_orig_norm_list)
                ],
                hovertemplate='%{text}<extra></extra>',
                legendgroup=f'model_{i}',
                showlegend=True,
                # Información adicional para el callback
                meta=dict(
                    model_idx=i,
                    model_type=modelo.get("tipo", "unknown"),
                    predictor=predictor,
                    aeronave=aeronave,
                    parametro=parametro,
                    data_type='original'
                )
            ))
        elif df_original is None or df_original.empty or x_min is None or x_max is None:
            # Mostrar advertencia si no hay datos originales válidos para este predictor
            fig.add_annotation(
                text=f"Sin datos originales para el predictor '{predictor}'. Solo se mostrará la curva teórica si es posible.",
                xref="paper", yref="paper",
                x=0.5, y=0.15, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=12, color="orange"),
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="orange",
                borderwidth=1
            )
        # --- PUNTOS DE ENTRENAMIENTO ---
        if show_training_points and df_filtrado is not None and not df_filtrado.empty and predictor in df_filtrado.columns and parametro in df_filtrado.columns and x_min is not None and x_max is not None and y_min is not None and y_max is not None:
            mask = df_filtrado[predictor].notna() & df_filtrado[parametro].notna()
            x_train = df_filtrado.loc[mask, predictor]
            y_train = df_filtrado.loc[mask, parametro]
            if x_max != x_min:
                x_train_norm = (x_train - x_min) / (x_max - x_min)
            else:
                x_train_norm = pd.Series([0.5] * len(x_train), index=x_train.index)
            if y_max != y_min:
                y_train_norm = (y_train - y_min) / (y_max - y_min)
            else:
                y_train_norm = pd.Series([0.5] * len(y_train), index=y_train.index if hasattr(y_train, 'index') else None)
            x_train_list = _to_list_safe(x_train)
            x_train_norm_list = _to_list_safe(x_train_norm)
            y_train_norm_list = _to_list_safe(y_train_norm)
            # Determinar resaltado y opacidad para el modelo actual
            training_opacity = 1.0 if is_highlighted else (0.5 if is_dimmed else 0.9)
            training_size = 10 if is_highlighted else 8
            fig.add_trace(go.Scatter(
                x=x_train_norm_list,
                y=y_train_norm_list,
                mode='markers',
                name=f'Entren. - {predictor} ({modelo.get("tipo", "unknown")})',
                marker=dict(
                    color=COLORS['model_lines'][i % len(COLORS['model_lines'])],
                    size=training_size,
                    opacity=training_opacity,
                    symbol='diamond',
                    line=dict(color='black', width=1)
                ),
                # Información personalizada para identificar el modelo en callbacks
                customdata=[i] * len(x_train_norm_list),
                text=[
                    f"Aeronave: {aeronave}<br>Parámetro: {parametro}<br>Predictor: {predictor}<br>Valor original X: {xv:.3f}<br>X adimensional: {xn:.3f}<br>Y normalizado: {yn:.3f}" for xv, xn, yn in zip(x_train_list, x_train_norm_list, y_train_norm_list)
                ],
                hovertemplate='%{text}<extra></extra>',
                legendgroup=f'model_{i}',
                showlegend=True,
                # Información adicional para el callback
                meta=dict(
                    model_idx=i,
                    model_type=modelo.get("tipo", "unknown"),
                    predictor=predictor,
                    aeronave=aeronave,
                    parametro=parametro,
                    data_type='training'
                )
            ))
        # --- CURVA DEL MODELO ---
        # Si no hay datos originales válidos, usar rango sintético SOLO para la curva
        using_synthetic_range = False
        if x_min is None or x_max is None:
            tipo = modelo.get('tipo', '')
            if tipo.startswith('log'):
                x_min, x_max = 0.1, 10
            elif tipo.startswith('exp'):
                x_min, x_max = 0, 5
            else:
                x_min, x_max = 0, 10
            using_synthetic_range = True
        if show_model_curves:
            if x_max != x_min:
                x_range_orig = np.linspace(x_min, x_max, 100)
                x_range_norm = (x_range_orig - x_min) / (x_max - x_min)
            else:
                x_range_orig = np.array([x_min])
                x_range_norm = np.array([0.5])
            predictions = get_model_predictions_safe(modelo, x_range_orig)
            if predictions is None:
                continue
            # Normalizar predicciones
            if y_min is not None and y_max is not None and y_max != y_min:
                predictions_norm = (predictions - y_min) / (y_max - y_min)
            else:
                predictions_norm = np.full_like(predictions, 0.5)
            color_idx = i % len(COLORS['model_lines'])
            model_color = COLORS['selected_model'] if (highlight_model_idx is not None and i == highlight_model_idx) else COLORS['model_lines'][color_idx]
            line_width = 5 if (highlight_model_idx is not None and i == highlight_model_idx) else 2
            opacity = 1.0 if (highlight_model_idx is not None and i == highlight_model_idx) else (0.3 if highlight_model_idx is not None else 1.0)
            line_style = 'dash' if using_synthetic_range else 'solid'
            hover_extra = "<br><b>ADVERTENCIA:</b> Curva generada con valores sintéticos por falta de datos originales" if using_synthetic_range else ""
            model_info = create_model_hover_info(modelo)
            fig.add_trace(go.Scatter(
                x=x_range_norm,
                y=predictions_norm,
                mode='lines',
                name=f'Curva - {predictor} ({modelo.get("tipo", "unknown")})' + (" [sintética]" if using_synthetic_range else ""),
                line=dict(
                    color=model_color, 
                    width=line_width,
                    dash=line_style
                ),
                opacity=opacity,
                # Información personalizada para identificar el modelo en callbacks
                customdata=[i] * len(predictions),  # Índice del modelo para identificarlo
                text=[
                    f"Predictor: {predictor}<br>Valor original X: {xv:.3f}<br>X adimensional: {xn:.3f}<br>Predicción Y normalizada: {yv:.3f}{hover_extra}<br>{model_info}" for xv, xn, yv in zip(x_range_orig, x_range_norm, predictions_norm)
                ],
                hovertemplate='%{text}<extra></extra>',
                legendgroup=f'model_{i}',
                showlegend=True,
                # Configuración para hacer la línea más clickeable
                connectgaps=True,
                # Información adicional para el callback
                meta=dict(
                    model_idx=i,
                    model_type=modelo.get("tipo", "unknown"),
                    predictor=predictor,
                    aeronave=aeronave,
                    parametro=parametro
                )
            ))
            # Si la curva es sintética, agregar advertencia visible
            if using_synthetic_range:
                fig.add_annotation(
                    text=f"Curva generada con valores sintéticos para '{predictor}' (sin datos originales)",
                    xref="paper", yref="paper",
                    x=0.5, y=0.08, xanchor='center', yanchor='middle',
                    showarrow=False,
                    font=dict(size=12, color="red"),
                    bgcolor="rgba(255,255,255,0.95)",
                    bordercolor="red",
                    borderwidth=1
                )
    # --- Agregar marcadores de imputación si hay detalles disponibles ---    # --- PUNTOS IMPUTADOS ---
    # Usar la nueva estructura de datos y filtrado por métodos
    if detalles_por_celda and celda_key in detalles_por_celda:
        # Asegurar que y_min y y_max estén definidos
        if 'y_min' not in locals() or 'y_max' not in locals() or y_min is None or y_max is None:
            # Buscar en los modelos de 1 predictor
            y_min, y_max = None, None
            for modelo in modelos_1_pred:
                df_original = get_model_original_data(modelo)
                if df_original is not None and parametro in df_original.columns:
                    y_data = df_original[parametro].dropna()
                    if len(y_data) > 0:
                        y_min = y_data.min() if y_min is None else min(y_min, y_data.min())
                        y_max = y_data.max() if y_max is None else max(y_max, y_data.max())
        from .plot_model_curves import extract_imputed_values_from_details, filter_imputed_points_by_method
        methods_to_show = selected_imputation_methods or ['final', 'similitud', 'correlacion']
        imputed_points = extract_imputed_values_from_details(
            detalles_por_celda, 
            celda_key, 
            modelos_1_pred
        )
        filtered_points = filter_imputed_points_by_method(imputed_points, methods_to_show)
        for point in filtered_points:
            metodo = point.get('imputation_method', 'unknown')
            x_norm = point.get('x_normalized', 0.5)
            y_value = point.get('y_value', 0)
            # Normalizar el valor imputado usando el mismo rango que los datos originales
            if y_min is not None and y_max is not None and y_max != y_min:
                y_value_norm = (y_value - y_min) / (y_max - y_min)
            else:
                y_value_norm = 0.5
            confidence = point.get('confidence', '')
            iteration = point.get('iteration', '')
            warning = point.get('warning', '')
            tooltip = f"Método: {metodo.capitalize()}<br>Valor Y normalizado: {y_value_norm:.3f}<br>Confianza: {confidence}<br>Iteración: {iteration}"
            if warning:
                tooltip += f"<br><b>Advertencia:</b> {warning}"
            symbol = point.get('symbol', 'circle')
            size = point.get('size', 10)
            color_map = {
                'final': 'black',
                'similitud': 'orange', 
                'correlacion': 'blue'
            }
            color = color_map.get(metodo, 'gray')
            if warning:
                symbol = 'x'
                color = 'red'
            fig.add_trace(go.Scatter(
                x=[x_norm],
                y=[y_value_norm],
                mode='markers',
                name=f'Imputación {metodo.capitalize()}',
                marker=dict(
                    color=color,
                    size=size,
                    symbol=symbol,
                    line=dict(color='black', width=1)
                ),
                text=[tooltip],
                hovertemplate='%{text}<extra></extra>',
                legendgroup=f'imputacion_{metodo}',
                showlegend=True
            ))
    fig.update_layout(
        title=f'Análisis de Modelos - {aeronave}: {parametro}',
        xaxis_title="X adimensional",
        yaxis_title=parametro,
        hovermode='x',  # Cambia a 'x' para anclar el hover a un lateral
        hoverlabel=dict(
            bgcolor="white",
            font_size=9,  # Disminuye el tamaño de la letra en un 30% aprox (de 13 a 9)
            font_family="Arial",
            align="left",
            namelength=-1
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1
        ),
        template='plotly_white',
        # Configuración clave para mantener el zoom y la interactividad
        uirevision=f"{aeronave}_{parametro}",  # Mantener estado de UI para la misma combinación
        clickmode='event+select',
        dragmode='zoom',
        # Configuración del zoom y pan
        xaxis=dict(
            fixedrange=False,  # Permitir zoom en X
            autorange=True,
            gridcolor='lightgray',
            gridwidth=1,
            zeroline=True,
            zerolinecolor='gray',
            zerolinewidth=1,
            hoverformat='.3f',
            side='top'
        ),
        yaxis=dict(
            fixedrange=False,  # Permitir zoom en Y
            autorange=True,
            gridcolor='lightgray',
            gridwidth=1,
            zeroline=True,
            zerolinecolor='gray',
            zerolinewidth=1
        ),
        # Mejorar la configuración del plot
        plot_bgcolor='white',
        paper_bgcolor='white',
        # Configuración para mejor responsividad
        margin=dict(l=60, r=20, t=60, b=60),
        autosize=True
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
        
        if show_training_points and df_filtrado is not None and df_filtrado.empty:
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
        
        # Hover para puntos originales
        hover_text = [
            f"Predictor: {predictor}<br>" +
            f"Valor original X: {x_orig.iloc[i]:.3f}<br>" +
            f"X normalizado: {x_orig_norm.iloc[i]:.3f}<br>" +
            f"Y: {y_orig.iloc[i]:.3f}<br>" +
            f"Modelo: {tipo}<br>" +
            f"MAPE: {mape:.3f}%<br>" +
            f"R²: {r2:.3f}<br>" +
            f"Fuente de datos: original"
            for i in range(len(x_orig_norm))
        ]
        
        if len(x_orig_norm) > 0:
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
        
        # Hover para puntos de entrenamiento
        if show_training_points and len(x_train_norm) > 0:
            hover_text_train = [
                f"Predictor: {predictor}<br>" +
                f"Valor original X: {x_train.iloc[i]:.3f}<br>" +
                f"X normalizado: {x_train_norm.iloc[i]:.3f}<br>" +
                f"Y: {y_train.iloc[i]:.3f}<br>" +
                f"Modelo: {tipo}<br>" +
                f"MAPE: {mape:.3f}%<br>" +
                f"R²: {r2:.3f}<br>" +
                f"Fuente de datos: entrenamiento"
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
            confianza = modelo.get('Confianza', 0)
            confianza_loocv = modelo.get('Confianza_LOOCV', 0)
            # Calcular confianza final como promedio (si ambos existen y son numéricos)
            if confianza is not None and confianza_loocv is not None:
                try:
                    confianza_final = round((float(confianza) + float(confianza_loocv)) / 2, 3)
                except Exception:
                    confianza_final = ''
            else:
                confianza_final = ''
            row = {
                'ID': i + 1,
                'Tipo': modelo.get('tipo', 'N/A'),
                'Predictores': ', '.join(modelo.get('predictores', [])),
                'N° Predictores': modelo.get('n_predictores', 0),
                'MAPE (%)': round(modelo.get('mape', 0), 3),
                'R²': round(modelo.get('r2', 0), 3),
                'Correlación': round(modelo.get('corr', 0), 3),
                'Confianza': round(confianza, 3),
                'Confianza_LOOCV': round(confianza_loocv, 3),
                'Confianza Final': confianza_final,
                'N° Muestras': modelo.get('n_muestras_entrenamiento', 0),
                'MAPE_LOOCV': round(modelo.get('MAPE_LOOCV', 0), 3),
                'R2_LOOCV': round(modelo.get('R2_LOOCV', 0), 3),
                'Corr_LOOCV': round(modelo.get('Corr_LOOCV', 0), 3),
                'k_LOOCV': modelo.get('k_LOOCV', ''),
                'Advertencia': modelo.get('Advertencia', '')
            }
            summary_data.append(row)
    
    return pd.DataFrame(summary_data)

def create_interactive_plot_3d(
    modelos_2_pred: list,
    aeronave: str,
    parametro: str,
    show_training_points: bool = True,
    show_model_curves: bool = True,
    highlight_model_idx: Optional[int] = None,
    detalles_por_celda: Optional[Dict] = None,
    selected_imputation_methods: Optional[list] = None
) -> go.Figure:
    """
    Visualización 3D de modelos de 2 predictores (lineales o polinómicos) para Dash.
    Utiliza la función create_3d_plot para graficar los modelos filtrados y normalizados.
    """
    # Si no hay modelos, mostrar mensaje claro
    if not modelos_2_pred or len(modelos_2_pred) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No hay modelos de 2 predictores para mostrar.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            title=f"Modelos de 2 Predictores (3D) - {aeronave}: {parametro}",
            template='plotly_white',
            autosize=True,
            margin=dict(l=10, r=10, t=40, b=10),
            uirevision=f"{aeronave}_{parametro}_3d"  # <-- Forzar uirevision único para 3D
        )
        return fig
    # Llama a la función principal de visualización 3D
    fig = create_3d_plot(
        modelos_2_pred,
        modelo_seleccionado_idx=highlight_model_idx,
        aeronave=aeronave,
        parametro=parametro
    )
    # Título y ejes personalizados
    fig.update_layout(
        title=f"Modelos de 2 Predictores (3D) - {aeronave}: {parametro}",
        scene=dict(
            zaxis_title=f"{parametro} (normalizado)",
        ),
        uirevision=f"{aeronave}_{parametro}_3d"  # <-- Forzar uirevision único para 3D
    )
    return fig

def create_3d_plot(modelos, modelo_seleccionado_idx=None, aeronave=None, parametro=None):
    """
    Genera un gráfico 3D interactivo con Plotly para modelos de 2 predictores (linear-2 y poly-2).
    Visualiza simultáneamente todos los modelos filtrados, mostrando su plano/superficie y puntos de entrenamiento normalizados.
    El modelo seleccionado se destaca con mayor opacidad.
    """
    import plotly.graph_objects as go
    import numpy as np

    fig = go.Figure()
    colores = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
        '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52'
    ]
    n_colores = len(colores)
    n_modelos = len(modelos)
    x0_name, x1_name = None, None
    # Malla normalizada
    grid_n = 30
    x0_grid = np.linspace(0, 1, grid_n)
    x1_grid = np.linspace(0, 1, grid_n)
    X0, X1 = np.meshgrid(x0_grid, x1_grid)
    # Graficar cada modelo
    for idx, modelo in enumerate(modelos):
        tipo = modelo.get('tipo', '').lower()
        if not (tipo.startswith('linear-2') or tipo.startswith('poly-2')):
            continue
        coefs = modelo.get('coeficientes_originales')
        intercepto = modelo.get('intercepto_original')
        predictores = modelo.get('predictores', [])
        ecuacion_latex = modelo.get('ecuacion_normalizada_latex', '')
        mape = modelo.get('mape', None)
        r2 = modelo.get('r2', None)
        if coefs is None or intercepto is None or len(predictores) != 2:
            continue
        x0_name, x1_name = predictores[0], predictores[1]
        # Calcular Z para la malla
        if tipo.startswith('linear-2') and len(coefs) >= 2:
            Z = intercepto + coefs[0]*X0 + coefs[1]*X1
        elif tipo.startswith('poly-2') and len(coefs) >= 5:
            Z = (intercepto + coefs[0]*X0 + coefs[1]*X1 +
                 coefs[2]*X0**2 + coefs[3]*X0*X1 + coefs[4]*X1**2)
        else:
            continue
        # Normalizar Z usando el mismo rango que los puntos de entrenamiento
        datos_entrenamiento = modelo.get('datos_entrenamiento', {})
        X_train = datos_entrenamiento.get('X_original')
        y_train = datos_entrenamiento.get('y_original')
        if y_train is not None and len(y_train) > 0:
            y_train = np.array(y_train)
            y_min, y_max = np.min(y_train), np.max(y_train)
            if y_max != y_min:
                Z_norm = (Z - y_min) / (y_max - y_min)
            else:
                Z_norm = np.full_like(Z, 0.5)
        else:
            # Si no hay datos de entrenamiento, no normalizar
            Z_norm = Z
        # Color y opacidad
        color = colores[idx % n_colores]
        opacity = 0.85 if idx == modelo_seleccionado_idx else 0.45
        # Hover info
        aeronave = modelo.get('Aeronave', 'N/A')
        parametro_obj = modelo.get('Parámetro', modelo.get('parametro', 'N/A'))
        ecuacion_latex = modelo.get('ecuacion_normalizada_latex', '')
        hovertext = (
            f"<b>Aeronave:</b> {aeronave}<br>"
            f"<b>Parámetro:</b> {parametro_obj}<br>"
            f"<b>Tipo:</b> {modelo.get('tipo','')}<br>"
            f"<b>Predictores:</b> {x0_name}, {x1_name}<br>"
            f"<b>Ecuación:</b> {ecuacion_latex}<br>"
            f"<b>MAPE:</b> {mape:.3f}%<br>"
            f"<b>R²:</b> {r2:.3f}<br>"
            f"<b>Z normalizado:</b> %{{z:.3f}}"
        )
        fig.add_trace(go.Surface(
            x=X0, y=X1, z=Z_norm,
            name=f"{modelo.get('tipo','')} [{x0_name}, {x1_name}]",
            showscale=False,
            opacity=opacity,
            surfacecolor=None,
            hovertemplate=hovertext + "<extra></extra>",
            legendgroup=f"modelo_{idx}",
            visible=True,
            colorscale=[[0, color], [1, color]],
            # Para click: customdata con el índice
            customdata=np.full(X0.shape, idx)
        ))
        # Puntos de entrenamiento
        datos_entrenamiento = modelo.get('datos_entrenamiento', {})
        X_train = datos_entrenamiento.get('X_original')
        y_train = datos_entrenamiento.get('y_original')
        if X_train is not None and y_train is not None and len(X_train) > 0 and len(y_train) == len(X_train):
            # Normalizar X_train por columna (igual que en 2D)
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            # Normalización por predictor
            x0_vals = X_train[:,0]
            x1_vals = X_train[:,1]
            x0_min, x0_max = np.min(x0_vals), np.max(x0_vals)
            x1_min, x1_max = np.min(x1_vals), np.max(x1_vals)
            x0_norm = (x0_vals - x0_min) / (x0_max - x0_min) if x0_max != x0_min else np.full_like(x0_vals, 0.5)
            x1_norm = (x1_vals - x1_min) / (x1_max - x1_min) if x1_max != x1_min else np.full_like(x1_vals, 0.5)
            # Normalización del eje z (y_train)
            y_min, y_max = np.min(y_train), np.max(y_train)
            y_train_norm = (y_train - y_min) / (y_max - y_min) if y_max != y_min else np.full_like(y_train, 0.5)
            fig.add_trace(go.Scatter3d(
                x=x0_norm, y=x1_norm, z=y_train_norm,
                mode='markers',
                name=f"Entrenamiento {idx+1}",
                marker=dict(
                    size=5 if idx != modelo_seleccionado_idx else 8,
                    color=color,
                    opacity=1.0 if idx == modelo_seleccionado_idx else 0.7
                ),
                # Información personalizada para identificar el modelo en callbacks
                customdata=np.full(X_train.shape[0], idx),
                text=[
                    f"Predictor 1: {x0_name}<br>Predictor 2: {x1_name}<br>" +
                    f"Valor original X1: {x0_vals[i]:.3f}<br>Valor original X2: {x1_vals[i]:.3f}<br>" +
                    f"X1 normalizado: {x0_norm[i]:.3f}<br>X2 normalizado: {x1_norm[i]:.3f}<br>" +
                    f"Y: {y_train[i]:.3f}<br>" +
                    f"Modelo: {tipo}<br>" +
                    f"MAPE: {mape:.3f}%<br>" +
                    f"R²: {r2:.3f}<br>" +
                    f"Fuente de datos: entrenamiento"
                    for i in range(len(y_train_norm))
                ],
                hovertemplate='%{text}<extra></extra>',
                legendgroup=f'model_{idx}',
                showlegend=True
            ))

    # Ajustes finales de la figura
    # Usar los argumentos recibidos, o valores por defecto si no se pasan
    safe_parametro = parametro if parametro else 'Parámetro'
    safe_aeronave = aeronave if aeronave else 'Aeronave'
    fig.update_layout(
        scene=dict(
            xaxis_title=f"{x0_name} (normalizado)",
            yaxis_title=f"{x1_name} (normalizado)",
            zaxis_title=f"{safe_parametro} (normalizado)",
            camera=dict(
                eye=dict(x=1.2, y=1.2, z=0.8)
            )
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        title=f"Modelos de 2 Predictores (3D) - {safe_aeronave}: {safe_parametro}",
        uirevision=f"{safe_aeronave}_{safe_parametro}_3d",
        template='plotly_white'
    )

    return fig
