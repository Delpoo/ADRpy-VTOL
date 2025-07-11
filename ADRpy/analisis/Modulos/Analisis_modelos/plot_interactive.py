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
try:
    from .plot_config import COLORS, SYMBOLS, _ensure_list
    from .plot_model_curves import add_normalized_model_curves, create_model_hover_info
    from .json_data_helpers import get_full_dataframe_from_celda
    from .normalization_engine import ModelNormalizationEngine
except ImportError:
    from plot_config import COLORS, SYMBOLS, _ensure_list
    from plot_model_curves import add_normalized_model_curves, create_model_hover_info
    from json_data_helpers import get_full_dataframe_from_celda
    from normalization_engine import ModelNormalizationEngine


def _to_list_safe(val):
    if val is None:
        return []
    if isinstance(val, (list, np.ndarray, pd.Series)):
        return list(val)
    return [val]


logger = logging.getLogger(__name__)


def validate_model_for_plotting(modelo: dict) -> tuple[bool, list[str]]:
    """
    Valida si un modelo puede ser graficado de manera segura.
    
    Returns:
        tuple: (es_valido, lista_de_warnings)
        - es_valido: True si el modelo puede ser graficado (solo se bloquea por errores críticos)
        - lista_de_warnings: Lista de problemas informativos o menores detectados
    """
    if not isinstance(modelo, dict):
        return False, ["modelo_no_es_dict"]
    
    warnings = []
    
    # Verificar datos básicos (solo informativos, no críticos)
    if not modelo.get('tipo'):
        warnings.append("sin_tipo")
    
    predictores = modelo.get('predictores', [])
    if not predictores:
        warnings.append("sin_predictores")
    
    # Método de imputación faltante - solo informativo
    if not modelo.get('metodo_imputacion'):
        warnings.append("sin_metodo_imputacion")
    
    # LOOCV faltante - solo informativo
    if modelo.get('Confianza_LOOCV') is None:
        warnings.append("sin_loocv")
    
    # Verificar datos de entrenamiento si existen
    datos_entrenamiento = modelo.get('datos_entrenamiento', {})
    if datos_entrenamiento:
        y_original = datos_entrenamiento.get('y_original')
        x_original = datos_entrenamiento.get('X_original')
        
        if y_original is not None:
            if not isinstance(y_original, (list, tuple)):
                return False, ["y_original_formato_invalido"]  # CRÍTICO
            elif len(y_original) == 0 or len([y for y in y_original if y is not None and not (isinstance(y, float) and y != y)]) == 0:
                return False, ["y_original_sin_datos_validos"]  # CRÍTICO
        
        if x_original is not None:
            if not isinstance(x_original, (list, tuple)):
                return False, ["x_original_formato_invalido"]  # CRÍTICO
            elif len(x_original) == 0:
                return False, ["x_original_vacio"]  # CRÍTICO
    
    # Verificar métricas básicas - solo informativos
    r2 = modelo.get('r2')
    if r2 is not None and isinstance(r2, (int, float)):
        if r2 != r2 or abs(r2) == float('inf'):  # NaN o infinito
            warnings.append("r2_invalido")
    
    confianza = modelo.get('Confianza')
    if confianza is not None and isinstance(confianza, (int, float)):
        if confianza != confianza:  # NaN
            warnings.append("confianza_nan")
    
    # Si no tiene n_predictores válido, es problema menor
    n_pred = modelo.get('n_predictores', 0)
    if not isinstance(n_pred, int) or n_pred <= 0:
        warnings.append("n_predictores_invalido")
    
    # Todos los modelos son válidos para mostrar a menos que haya errores críticos de datos
    return True, warnings


def create_interactive_plot(
    modelos_filtrados: Dict,
    aeronave: str,
    parametro: str,
    show_training_points: bool = True,
    show_theoretical_points: bool = True,
    show_imputation_points: bool = True,
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
    
    # Validar y filtrar modelos problemáticos
    modelos_validos = []
    modelos_con_problemas = []
    
    for i, modelo in enumerate(modelos):
        if isinstance(modelo, dict) and modelo.get('n_predictores', 0) == 1:
            es_valido, problemas = validate_model_for_plotting(modelo)
            if es_valido:
                modelos_validos.append(modelo)
            else:
                modelos_con_problemas.append({
                    'modelo': modelo,
                    'problemas': problemas,
                    'indice': i
                })
    
    modelos_1_pred = modelos_validos
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

        # --- CONSTRUIR DATAFRAME DESDE DATOS DEL MODELO ACTUAL ---
        df_original = None
        df_filtrado = None
        
        # Usar los datos de entrenamiento del modelo actual
        datos_entrenamiento = modelo.get('datos_entrenamiento', {})
        X_data = datos_entrenamiento.get('X_original')
        y_data = datos_entrenamiento.get('y_original')
        columnas_pred = datos_entrenamiento.get('columnas_predictores')
        
        if X_data is not None and y_data is not None and columnas_pred is not None:
            try:
                # Construir DataFrame desde datos del modelo
                df_model = pd.DataFrame(X_data, columns=columnas_pred)
                df_model[parametro] = y_data
                
                # Verificar que el predictor está en las columnas
                if predictor in df_model.columns and parametro in df_model.columns:
                    mask = df_model[predictor].notna() & df_model[parametro].notna()
                    df_original = df_model.loc[mask, [predictor, parametro]].copy()
                    df_filtrado = df_original.copy() if df_original is not None else None
            except Exception as e:
                # Fallback: usar detalles_por_celda si está disponible
                if detalles_por_celda and celda_key in detalles_por_celda:
                    # Crear estructura completa de celda para get_full_dataframe_from_celda
                    celda_completa = {
                        'informacion_generica_celda': detalles_por_celda[celda_key],
                        'informacion_modelos_celda': {'modelos': modelos_filtrados[celda_key]}
                    }
                    df_celda, warnings_df = get_full_dataframe_from_celda(celda_completa)
                    if isinstance(df_celda, pd.DataFrame):
                        if predictor in df_celda.columns and parametro in df_celda.columns:
                            mask = df_celda[predictor].notna() & df_celda[parametro].notna()
                            df_original = df_celda.loc[mask, [predictor, parametro]].copy()
                            df_filtrado = df_original.copy() if df_original is not None else None
        
        # Si aún no hay datos, usar detalles_por_celda como último recurso
        if df_original is None and detalles_por_celda and celda_key in detalles_por_celda:
            celda_completa = {
                'informacion_generica_celda': detalles_por_celda[celda_key],
                'informacion_modelos_celda': {'modelos': modelos_filtrados[celda_key]}
            }
            df_celda, warnings_df = get_full_dataframe_from_celda(celda_completa)
            if isinstance(df_celda, pd.DataFrame):
                if predictor in df_celda.columns and parametro in df_celda.columns:
                    mask = df_celda[predictor].notna() & df_celda[parametro].notna()
                    df_original = df_celda.loc[mask, [predictor, parametro]].copy()
                    df_filtrado = df_original.copy() if df_original is not None else None
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
            # Asegurar que sean arrays de numpy 1D
            if isinstance(x_orig, pd.DataFrame):
                x_orig = x_orig.squeeze()
                if isinstance(x_orig, pd.DataFrame):
                    x_orig = x_orig.iloc[:,0]
            if not isinstance(x_orig, (pd.Series, np.ndarray, list)):
                x_orig = pd.Series(x_orig)
            x_orig = np.asarray(x_orig).flatten()
            if isinstance(y_orig, pd.DataFrame):
                y_orig = y_orig.squeeze()
                if isinstance(y_orig, pd.DataFrame):
                    y_orig = y_orig.iloc[:,0]
            if not isinstance(y_orig, (pd.Series, np.ndarray, list)):
                y_orig = pd.Series(y_orig)
            y_orig = np.asarray(y_orig).flatten()
            if x_max != x_min:
                x_orig_norm = (x_orig - x_min) / (x_max - x_min)
            else:
                x_orig_norm = np.full_like(x_orig, 0.5, dtype=float)
            # 🔧 CAMBIO: No normalizar variable dependiente (Y) para mostrar valores originales
            y_orig_norm = y_orig  # Mantener valores originales
            x_orig_list = x_orig.tolist()
            x_orig_norm_list = x_orig_norm.tolist()
            y_orig_norm_list = y_orig_norm.tolist()
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
                    f"Aeronave: {aeronave}<br>Parámetro: {parametro}<br>Predictor: {predictor}<br>Valor original X: {xv:.3f}<br>X adimensional: {xn:.3f}<br>Y (escala original): {yn:.3f}" for xv, xn, yn in zip(x_orig_list, x_orig_norm_list, y_orig_norm_list)
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
            x_train = np.asarray(x_train).flatten()
            y_train = np.asarray(y_train).flatten()
            if x_max != x_min:
                x_train_norm = (x_train - x_min) / (x_max - x_min)
            else:
                x_train_norm = np.full_like(x_train, 0.5, dtype=float)
            # 🔧 CAMBIO: No normalizar variable dependiente (Y) para mostrar valores originales
            y_train_norm = y_train  # Mantener valores originales
            x_train_list = x_train.tolist()
            x_train_norm_list = x_train_norm.tolist()
            y_train_norm_list = y_train_norm.tolist()
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
                    f"Aeronave: {aeronave}<br>Parámetro: {parametro}<br>Predictor: {predictor}<br>Valor original X: {xv:.3f}<br>X adimensional: {xn:.3f}<br>Y (escala original): {yn:.3f}" for xv, xn, yn in zip(x_train_list, x_train_norm_list, y_train_norm_list)
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
            # Usar normalization_engine para generar curvas correctas
            try:
                normalizer = ModelNormalizationEngine()
                
                # Generar curva normalizada usando el motor de normalización
                x_range_norm = np.linspace(0, 1, 100)
                x_norm_result, y_norm_result, metadata = normalizer.generate_normalized_curve_from_model(
                    modelo, x_range_normalized=x_range_norm, resolution=100
                )
                
                if x_norm_result is not None and y_norm_result is not None:
                    x_range_norm = np.asarray(x_norm_result).flatten()
                    predictions_norm = np.asarray(y_norm_result).flatten()
                    
                    # Reconstruir x_range_orig para hover info
                    if x_min is not None and x_max is not None and x_max != x_min:
                        x_range_orig = x_range_norm * (x_max - x_min) + x_min
                    else:
                        x_range_orig = x_range_norm  # Fallback
                    
                    # Advertencias adicionales
                    warnings_text = ""
                    if metadata.get("synthetic_range", False):
                        warnings_text += "<br><b>ADVERTENCIA:</b> Rango sintético utilizado por falta de datos"
                        using_synthetic_range = True
                    if metadata.get("multi_predictor_warning"):
                        warnings_text += f"<br><b>INFO:</b> {metadata['multi_predictor_warning']}"
                    if metadata.get("poly_fallback_warning"):
                        warnings_text += f"<br><b>ADVERTENCIA:</b> {metadata['poly_fallback_warning']}"
                    
                    color_idx = i % len(COLORS['model_lines'])
                    model_color = COLORS['selected_model'] if (highlight_model_idx is not None and i == highlight_model_idx) else COLORS['model_lines'][color_idx]
                    line_width = 5 if (highlight_model_idx is not None and i == highlight_model_idx) else 2
                    opacity = 1.0 if (highlight_model_idx is not None and i == highlight_model_idx) else (0.3 if highlight_model_idx is not None else 1.0)
                    line_style = 'dash' if using_synthetic_range else 'solid'
                    
                    model_info = create_model_hover_info(modelo)
                    fig.add_trace(go.Scatter(
                        x=x_range_norm.tolist(),
                        y=predictions_norm.tolist(),
                        mode='lines',
                        name=f'Curva - {predictor} ({modelo.get("tipo", "unknown")})' + (" [sintética]" if using_synthetic_range else ""),
                        line=dict(
                            color=model_color, 
                            width=line_width,
                            dash=line_style
                        ),
                        opacity=opacity,
                        # Información personalizada para identificar el modelo en callbacks
                        customdata=[i] * len(predictions_norm),  # Índice del modelo para identificarlo
                        text=[
                            f"Predictor: {predictor}<br>Valor original X: {xv:.3f}<br>X adimensional: {xn:.3f}<br>Predicción Y normalizada: {yv:.3f}{warnings_text}<br>{model_info}" 
                            for xv, xn, yv in zip(x_range_orig.tolist(), x_range_norm.tolist(), predictions_norm.tolist())
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
                            parametro=parametro,
                            data_type='curve'
                        )
                    ))
                else:
                    # Fallback si falla la generación de curva
                    logger.warning(f"No se pudo generar curva para modelo {i}: {metadata.get('error', 'Error desconocido')}")
                    continue
                    
            except Exception as e:
                logger.error(f"Error generando curva para modelo {i}: {e}")
                # Fallback a la lógica anterior solo en caso de error crítico
                if x_max != x_min:
                    x_range_orig = np.linspace(x_min, x_max, 100)
                    x_range_norm = (x_range_orig - x_min) / (x_max - x_min)
                else:
                    x_range_orig = np.array([x_min])
                    x_range_norm = np.array([0.5])
                x_range_orig = np.asarray(x_range_orig).flatten()
                x_range_norm = np.asarray(x_range_norm).flatten()
                
                # Use normalized model data instead of obsolete function
                try:
                    from .normalization_engine import get_normalized_model_data
                    vis_data = get_normalized_model_data(modelo)
                    curva_data = vis_data.get('curva', {})
                    predictions = np.array(curva_data.get('y_normalized', []))
                    if len(predictions) == 0:
                        continue
                except:
                    continue
                    
                predictions = np.asarray(predictions).flatten()
                # 🔧 CORRECCIÓN: No normalizar predicciones Y para mostrar valores originales
                # if y_min is not None and y_max is not None and y_max != y_min:
                #     predictions_norm = (predictions - y_min) / (y_max - y_min)
                # else:
                #     predictions_norm = np.full_like(predictions, 0.5, dtype=float)
                # Usar valores originales de Y directamente
                predictions_norm = predictions
                color_idx = i % len(COLORS['model_lines'])
                model_color = COLORS['selected_model'] if (highlight_model_idx is not None and i == highlight_model_idx) else COLORS['model_lines'][color_idx]
                line_width = 5 if (highlight_model_idx is not None and i == highlight_model_idx) else 2
                opacity = 1.0 if (highlight_model_idx is not None and i == highlight_model_idx) else (0.3 if highlight_model_idx is not None else 1.0)
                line_style = 'dash' if using_synthetic_range else 'solid'
                hover_extra = "<br><b>ADVERTENCIA:</b> Curva generada con valores sintéticos por falta de datos originales" if using_synthetic_range else ""
                model_info = create_model_hover_info(modelo)
                fig.add_trace(go.Scatter(
                    x=x_range_norm.tolist(),
                    y=predictions_norm.tolist(),
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
                        f"Predictor: {predictor}<br>Valor original X: {xv:.3f}<br>X adimensional: {xn:.3f}<br>Predicción Y (original): {yv:.3f}{hover_extra}<br>{model_info}" for xv, xn, yv in zip(x_range_orig.tolist(), x_range_norm.tolist(), predictions_norm.tolist())
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
        # Asegurar que y_min y y_max estén definidos correctamente
        y_min, y_max = None, None
        # Buscar en los modelos de 1 predictor (usando el DataFrame de la celda si es posible)
        if detalles_por_celda[celda_key].get('df_original') is not None:
            df_celda = detalles_por_celda[celda_key]['df_original']
            if isinstance(df_celda, dict):
                df_celda = pd.DataFrame(df_celda)
            if isinstance(df_celda, pd.DataFrame) and parametro in df_celda.columns:
                y_data = df_celda[parametro].dropna()
                if len(y_data) > 0:
                    y_min = y_data.min()
                    y_max = y_data.max()
        # Fallback: buscar en los modelos si no se encontró nada
        if y_min is None or y_max is None:
            for modelo in modelos_1_pred:
                # Usar solo DataFrame centralizado si está disponible
                if detalles_por_celda and celda_key in detalles_por_celda:
                    # Crear estructura completa de celda
                    celda_completa = {
                        'informacion_generica_celda': detalles_por_celda[celda_key],
                        'informacion_modelos_celda': {'modelos': modelos_filtrados[celda_key]}
                    }
                    df_celda, warnings_df = get_full_dataframe_from_celda(celda_completa)
                    if isinstance(df_celda, pd.DataFrame) and parametro in df_celda.columns:
                        y_data = df_celda[parametro].dropna()
                        if len(y_data) > 0:
                            y_min = y_data.min() if y_min is None else min(y_min, y_data.min())
                            y_max = y_data.max() if y_max is None else max(y_max, y_data.max())
        # Si aún no están definidos, usar valores por defecto
        if y_min is None or y_max is None or y_max == y_min:
            y_min, y_max = 0.0, 1.0
        
    # --- PUNTOS TEÓRICOS DE IMPUTACIÓN ---
    # Añadir puntos teóricos calculados usando los valores de las variables independientes
    from .plot_model_curves import extract_theoretical_imputation_points, add_theoretical_imputation_points_to_plot
    
    # Extraer puntos teóricos para modelos de 1 predictor (para gráficos 2D)
    theoretical_points = extract_theoretical_imputation_points(
        modelos_1_pred, 
        celda_key, 
        n_predictores_filter=1,  # Solo modelos de 1 predictor para gráficos 2D
        modelos_por_celda=modelos_filtrados  # Pasar datos completos para rangos globales
    )
    
    # Añadir los puntos teóricos al gráfico
    add_theoretical_imputation_points_to_plot(fig, theoretical_points, show_theoretical_points=show_theoretical_points)
    
    # --- AÑADIR PUNTOS DE IMPUTACIÓN ---
    # Añadir puntos de imputación (similitud, correlación, final) usando coordenadas sin normalizar por ahora
    if show_imputation_points and detalles_por_celda:
        try:
            from .plot_model_curves import add_normalized_imputation_points
            # Por ahora pasar global_ranges=None para usar coordenadas originales
            add_normalized_imputation_points(
                fig=fig,
                detalles_por_celda=detalles_por_celda,
                celda_key=celda_key,
                show_imputation_points=show_imputation_points,
                n_predictores_filter=1,  # Solo modelos de 1 predictor para gráficos 2D
                global_ranges=None,  # Sin rangos globales, usaremos los del mejor modelo
                modelos_por_celda=modelos_filtrados  # Pasar modelos para encontrar el mejor
            )
        except Exception as e:
            logger.error(f"Error añadiendo puntos de imputación: {e}")
    
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
                         detalles_por_celda: Optional[Dict] = None,
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
        # Obtener datos originales y de entrenamiento del modelo usando helpers centralizados
        df_original = None
        df_filtrado = None
        if detalles_por_celda:
            celda_key = f"{modelo.get('Aeronave', '')}|{parametro}"
            if celda_key in detalles_por_celda:
                # Crear estructura completa de celda
                celda_completa = {
                    'informacion_generica_celda': detalles_por_celda[celda_key],
                    'informacion_modelos_celda': {'modelos': [modelo]}
                }
                df_celda, warnings_df = get_full_dataframe_from_celda(celda_completa)
                if isinstance(df_celda, pd.DataFrame) and predictor in df_celda.columns and parametro in df_celda.columns:
                    mask = df_celda[predictor].notna() & df_celda[parametro].notna()
                    x_orig_valid = df_celda.loc[mask, predictor]
                    y_orig_valid = df_celda.loc[mask, parametro]
                    df_original = df_celda.loc[mask, [predictor, parametro]].copy()
                    # Para entrenamiento, puedes adaptar aquí si hay lógica especial
                    x_train_valid = x_orig_valid.copy() if isinstance(x_orig_valid, (pd.Series, pd.DataFrame)) else x_orig_valid
                    y_train_valid = y_orig_valid.copy() if isinstance(y_orig_valid, (pd.Series, pd.DataFrame)) else y_orig_valid
                    df_filtrado = df_original.copy() if isinstance(df_original, (pd.Series, pd.DataFrame)) else df_original
                else:
                    x_orig_valid = pd.Series(dtype=float)
                    y_orig_valid = pd.Series(dtype=float)
                    x_train_valid = pd.Series(dtype=float)
                    y_train_valid = pd.Series(dtype=float)
            else:
                x_orig_valid = pd.Series(dtype=float)
                y_orig_valid = pd.Series(dtype=float)
                x_train_valid = pd.Series(dtype=float)
                y_train_valid = pd.Series(dtype=float)
        else:
            x_orig_valid = pd.Series(dtype=float)
            y_orig_valid = pd.Series(dtype=float)
            x_train_valid = pd.Series(dtype=float)
            y_train_valid = pd.Series(dtype=float)

        if df_original is None or (hasattr(df_original, 'empty') and df_original.empty) or len(x_orig_valid) == 0:
            models_without_data.append(f"{predictor} ({modelo.get('tipo', 'unknown')})")
            logger.warning(f"No se pudieron obtener datos originales para modelo con predictor {predictor}")
            continue

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
    modelos_validos = 0
    modelos_con_problemas = 0
    problemas_encontrados = {}
    
    for i, modelo in enumerate(modelos):
        if isinstance(modelo, dict):
            # Validar modelo antes de procesarlo
            es_valido, warnings = validate_model_for_plotting(modelo)
            
            # Validar modelo antes de procesarlo
            es_valido, warnings = validate_model_for_plotting(modelo)
            
            if not es_valido:
                modelos_con_problemas += 1
                for warning in warnings:
                    problemas_encontrados[warning] = problemas_encontrados.get(warning, 0) + 1
            elif warnings:
                modelos_con_problemas += 1
                for warning in warnings:
                    problemas_encontrados[warning] = problemas_encontrados.get(warning, 0) + 1
            else:
                modelos_validos += 1
            
            confianza = modelo.get('Confianza')
            confianza_loocv = modelo.get('Confianza_LOOCV')
            
            # Calcular confianza final como promedio (solo si ambos existen y son numéricos)
            if (confianza is not None and confianza_loocv is not None and 
                isinstance(confianza, (int, float)) and isinstance(confianza_loocv, (int, float))):
                try:
                    confianza_final = round((float(confianza) + float(confianza_loocv)) / 2, 3)
                except Exception:
                    confianza_final = ''
            else:
                # Si no hay LOOCV, usar solo la confianza básica
                if confianza is not None and isinstance(confianza, (int, float)):
                    confianza_final = round(float(confianza), 3)
                else:
                    confianza_final = ''
            
            # Procesar valores con manejo seguro de None
            def safe_round(value, decimals=3, default=0):
                if value is None:
                    return default
                try:
                    return round(float(value), decimals)
                except (ValueError, TypeError):
                    return default
            
            def safe_round_display(value, decimals=3, default='N/A'):
                if value is None:
                    return default
                try:
                    return round(float(value), decimals)
                except (ValueError, TypeError):
                    return default
            
            # Determinar estado de validación
            if not es_valido:
                # Errores críticos (rojo) - modelo no se puede graficar
                estado_validacion = f"❌ Error crítico: {', '.join(warnings[:2])}"
                if len(warnings) > 2:
                    estado_validacion += f" (+{len(warnings)-2})"
            elif not warnings:
                estado_validacion = "✅ Completo"
            elif any(w in ["sin_loocv", "sin_metodo_imputacion"] for w in warnings):
                # Avisos informativos (amarillo)
                estado_validacion = f"⚠️ Incompleto: {', '.join([w.replace('sin_', '') for w in warnings[:2]])}"
                if len(warnings) > 2:
                    estado_validacion += f" (+{len(warnings)-2})"
            else:
                # Otros problemas menores (amarillo)
                estado_validacion = f"⚠️ Advertencia: {', '.join(warnings[:2])}"
                if len(warnings) > 2:
                    estado_validacion += f" (+{len(warnings)-2})"
                
            row = {
                'ID': i + 1,
                'Estado': estado_validacion,
                'Tipo': modelo.get('tipo', 'N/A'),
                'Predictores': ', '.join(modelo.get('predictores', [])),
                'N° Predictores': modelo.get('n_predictores', 0),
                'MAPE (%)': safe_round(modelo.get('mape')),
                'R²': safe_round(modelo.get('r2')),
                'Correlación': safe_round(modelo.get('corr')),
                'Confianza': safe_round(confianza),
                'Confianza_LOOCV': safe_round_display(confianza_loocv),
                'Confianza Final': confianza_final,
                'N° Muestras': modelo.get('n_muestras_entrenamiento', 0),
                'MAPE_LOOCV': safe_round_display(modelo.get('MAPE_LOOCV')),
                'R2_LOOCV': safe_round_display(modelo.get('R2_LOOCV')),
                'Corr_LOOCV': safe_round_display(modelo.get('Corr_LOOCV')),
                'k_LOOCV': modelo.get('k_LOOCV', ''),
                'Advertencia': modelo.get('Advertencia', '')
            }
            summary_data.append(row)
    
    df_result = pd.DataFrame(summary_data)
    
    # Agregar información de validación como atributos del DataFrame
    df_result.attrs['modelos_validos'] = modelos_validos
    df_result.attrs['modelos_con_problemas'] = modelos_con_problemas
    df_result.attrs['problemas_encontrados'] = problemas_encontrados
    
    return df_result

def create_interactive_plot_3d(
    modelos_2_pred: list,
    aeronave: str,
    parametro: str,
    show_training_points: bool = True,
    show_theoretical_points: bool = True,
    show_imputation_points: bool = True,
    show_model_curves: bool = True,
    highlight_model_idx: Optional[int] = None,
    detalles_por_celda: Optional[Dict] = None,
    selected_imputation_methods: Optional[list] = None,
    modelos_filtrados: Optional[Dict] = None
) -> go.Figure:
    """
    Visualización 3D de modelos de 2 predictores (lineales o polinómicos) para Dash.
    Utiliza la función create_3d_plot para graficar los modelos filtrados con Z en escala original.
    """
    # Importar funciones necesarias para evitar circular imports
    from .plot_model_curves import extract_theoretical_imputation_points
    from .plot_3d import add_theoretical_points_3d
    
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
        parametro=parametro,
        detalles_por_celda=detalles_por_celda
    )
    
    # --- AÑADIR PUNTOS TEÓRICOS 3D ---
    # Añadir puntos teóricos para modelos de 2 predictores si está habilitado
    if show_theoretical_points:
        theoretical_points_3d = extract_theoretical_imputation_points(
            modelos_2_pred, 
            f"{aeronave}|{parametro}", 
            n_predictores_filter=2,  # Solo modelos de 2 predictores para gráficos 3D
            modelos_por_celda=modelos_filtrados  # Pasar datos completos para rangos globales
        )
        
        # Añadir los puntos teóricos 3D al gráfico
        add_theoretical_points_3d(fig, theoretical_points_3d, show_theoretical_points=show_theoretical_points)
    
    # --- AÑADIR PUNTOS DE IMPUTACIÓN 3D ---
    # Añadir puntos de imputación (similitud, correlación, final) usando coordenadas sin normalizar por ahora para 3D
    if show_imputation_points and detalles_por_celda:
        try:
            from .plot_model_curves import add_normalized_imputation_points
            # Añadir puntos de imputación usando el mejor modelo para normalización
            add_normalized_imputation_points(
                fig=fig,
                detalles_por_celda=detalles_por_celda,
                celda_key=f"{aeronave}|{parametro}",
                show_imputation_points=show_imputation_points,
                n_predictores_filter=2,  # Solo modelos de 2 predictores para gráficos 3D
                global_ranges=None,  # Sin rangos globales, usaremos los del mejor modelo
                modelos_por_celda=modelos_filtrados  # Pasar modelos para encontrar el mejor
            )
        except Exception as e:
            logger.error(f"Error añadiendo puntos de imputación 3D: {e}")
    
    # --- IMPUTED POINTS LOGIC REMOVED ---
    
    # Título y ejes personalizados
    fig.update_layout(
        title=f"Modelos de 2 Predictores (3D) - {aeronave}: {parametro}",
        scene=dict(
            zaxis_title=f"{parametro} (escala original)",
        ),
        uirevision=f"{aeronave}_{parametro}_3d"  # <-- Forzar uirevision único para 3D
    )
    return fig

def create_3d_plot(modelos, modelo_seleccionado_idx=None, aeronave=None, parametro=None, detalles_por_celda=None):
    """
    Genera un gráfico 3D interactivo con Plotly para modelos de 2 predictores (linear-2 y poly-2).
    Visualiza simultáneamente todos los modelos filtrados, mostrando su plano/superficie y puntos de entrenamiento normalizados.
    El modelo seleccionado se destaca con mayor opacidad.
    """
    import plotly.graph_objects as go
    import numpy as np
    from .plot_3d import generate_symbiotic_surface_3d

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
    grid_n = 30
    # Graficar cada modelo
    for idx, modelo in enumerate(modelos):
        tipo = modelo.get('tipo', '').lower()
        if not (tipo.startswith('linear-2') or tipo.startswith('poly-2')):
            continue
        coefs = modelo.get('coeficientes_originales')
        intercepto = modelo.get('intercepto_original')
        predictores = modelo.get('predictores', [])
        ecuacion_string = modelo.get('ecuacion_string', '')
        mape = modelo.get('mape', None)
        r2 = modelo.get('r2', None)
        if coefs is None or intercepto is None or len(predictores) != 2:
            continue
        x0_name, x1_name = predictores[0], predictores[1]
        # Obtener datos de entrenamiento para rangos
        datos_entrenamiento = modelo.get('datos_entrenamiento', {})
        X_train = datos_entrenamiento.get('X_original')
        y_train = datos_entrenamiento.get('y_original')
        if X_train is not None and len(X_train) > 0:
            X_train = np.array(X_train)
            x0_min, x0_max = np.min(X_train[:,0]), np.max(X_train[:,0])
            x1_min, x1_max = np.min(X_train[:,1]), np.max(X_train[:,1])
        else:
            x0_min, x0_max = 0, 1
            x1_min, x1_max = 0, 1
        # Definir color por defecto para este modelo
        color = colores[idx % n_colores]
        opacity = 0.85 if idx == modelo_seleccionado_idx else 0.45
        
        # --- SUPERFICIE: usar lógica simbiotica ---
        try:
            # Determinar qué datos pasar según el tipo de modelo
            if tipo.startswith('poly-'):
                # Para modelos polinomiales, pasar el diccionario completo
                model_data = modelo
            else:
                # Para modelos lineales, pasar la ecuación string
                model_data = ecuacion_string
            
            X_surf, Y_surf, Z_surf = generate_symbiotic_surface_3d(
                model_data,
                (x0_min, x0_max),
                (x1_min, x1_max),
                [x0_name, x1_name],
                resolution=grid_n
            )
            hovertext = (
                f"<b>Aeronave:</b> {modelo.get('Aeronave','N/A')}<br>"
                f"<b>Parámetro:</b> {modelo.get('Parámetro', modelo.get('parametro','N/A'))}<br>"
                f"<b>Tipo:</b> {modelo.get('tipo','')}<br>"
                f"<b>Predictores:</b> {x0_name}, {x1_name}<br>"
                f"<b>Ecuación:</b> {ecuacion_string}<br>"
                f"<b>MAPE:</b> {mape:.3f}%<br>"
                f"<b>R²:</b> {r2:.3f}<br>"
                f"<b>Z (escala original):</b> %{{z:.3f}}"
            )
            fig.add_trace(go.Surface(
                x=X_surf, y=Y_surf, z=Z_surf,
                name=f"{modelo.get('tipo','')} [{x0_name}, {x1_name}]",
                showscale=False,
                opacity=opacity,
                surfacecolor=None,
                hovertemplate=hovertext + "<extra></extra>",
                legendgroup=f"modelo_{idx}",
                visible=True,
                colorscale=[[0, color], [1, color]],
                customdata=np.full(X_surf.shape, idx)
            ))
        except Exception as e:
            print(f"Error generando superficie simbiotica para modelo {idx}: {e}")
        # --- PUNTOS DE ENTRENAMIENTO ---
        if X_train is not None and y_train is not None and len(X_train) > 0 and len(y_train) == len(X_train):
            x0_vals = X_train[:,0]
            x1_vals = X_train[:,1]
            x0_norm = (x0_vals - x0_min) / (x0_max - x0_min) if x0_max != x0_min else np.full_like(x0_vals, 0.5)
            x1_norm = (x1_vals - x1_min) / (x1_max - x1_min) if x1_max != x1_min else np.full_like(x1_vals, 0.5)
            y_train = np.array(y_train)
            fig.add_trace(go.Scatter3d(
                x=x0_norm, y=x1_norm, z=y_train,
                mode='markers',
                name=f"Entrenamiento {idx+1}",
                marker=dict(
                    size=5 if idx != modelo_seleccionado_idx else 8,
                    color=color,
                    opacity=1.0 if idx == modelo_seleccionado_idx else 0.7
                ),
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
                    for i in range(len(y_train))
                ],
                hovertemplate='%{text}<extra></extra>',
                legendgroup=f'model_{idx}',
                showlegend=True
            ))
    # Ajustes finales de la figura
    safe_parametro = parametro if parametro else 'Parámetro'
    safe_aeronave = aeronave if aeronave else 'Aeronave'
    fig.update_layout(
        scene=dict(
            xaxis_title=f"{x0_name} (normalizado)",
            yaxis_title=f"{x1_name} (normalizado)",
            zaxis_title=f"{safe_parametro} (escala original)",
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
