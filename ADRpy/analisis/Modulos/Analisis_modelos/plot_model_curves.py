"""
plot_model_curves.py

Funciones para:
- Cálculo y predicción de curvas de modelos
- Normalización de predictores
- Construcción de hovers para modelos y puntos
- Preparación de datos para la UI de Dash

FUNCIONES PRINCIPALES PARA UI:
------------------------------

1. prepare_normalized_model_curves_data(): 
   - Retorna datos de curvas preparados para la UI
   - Solo modelos de 1 predictor para gráficos 2D
   - Incluye información sobre curvas sintéticas (is_synthetic, warning)
   - Retorna listas con x_normalized, y_predicted, etiquetas, estilos

2. prepare_imputed_points_data():
   - Retorna datos de puntos imputados preparados para scatter plotting
   - Incluye tooltips completos, símbolos según método, tamaños según confianza
   - Campos: value, confidence, iteration, warning para filtrado en UI

3. filter_single_predictor_models():
   - Filtra modelos para retornar solo los de 1 predictor
   - Usar antes de cualquier visualización 2D

CAMPOS PARA UI CALLBACKS:
------------------------
- Curvas: is_synthetic, warning -> para controlar visibilidad
- Puntos: imputation_method, confidence, warning -> para filtros/tooltips
- line_style: 'solid' | 'dash' -> para estilo visual de curvas sintéticas
- symbol, size, color_method -> para scatter points styling

NOTA: Estas funciones solo preparan datos, no incluyen lógica de layout o callbacks.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import logging
from .plot_config import COLORS, SYMBOLS, _ensure_list
from .plot_data_access import get_model_original_data, get_model_training_data

logger = logging.getLogger(__name__)


def add_normalized_model_curves(fig: go.Figure, 
                               modelos: List[Dict], 
                               parametro: str,
                               show_synthetic_curves: bool = True) -> None:
    """
    Añade curvas de modelos normalizadas al gráfico.
    Solo procesa modelos de 1 predictor para gráficos 2D.
    Cada modelo usa su propio rango de datos normalizado a [0, 1].
    Si no hay datos originales, genera un rango sintético para mostrar la ecuación.
    Las curvas sintéticas se muestran con líneas punteadas y etiquetas apropiadas.
    
    Parameters:
    -----------
    fig : go.Figure
        Figura de Plotly donde añadir las curvas
    modelos : List[Dict]
        Lista de modelos (solo se procesan los de 1 predictor)
    parametro : str
        Nombre del parámetro objetivo
    show_synthetic_curves : bool
        Si mostrar curvas generadas con rangos sintéticos (líneas punteadas)
    """
    
    curves_added = 0
    warnings_added = []
    synthetic_ranges_used = 0
    omitted_synthetic = 0
    
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
        
        # Si el usuario no quiere mostrar curvas sintéticas y solo hay rango sintético, omitir
        if using_synthetic_range and not show_synthetic_curves:
            omitted_synthetic += 1
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
        fuente = "sintético" if using_synthetic_range else "original"
        
        # Crear información de hover para la curva
        hover_text = [
            f"Predictor: {predictor}<br>" +
            f"Valor original X: {x_range_orig[j]:.3f}<br>" +
            f"X normalizado: {x_range_norm[j]:.3f}<br>" +
            f"Predicción Y: {predictions[j]:.3f}<br>" +
            f"Modelo: {tipo}<br>" +
            f"Ecuación: {ecuacion}<br>" +
            f"MAPE: {mape:.3f}%<br>" +
            f"R²: {r2:.3f}<br>" +
            f"Fuente de datos: {fuente}"
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
    
    if synthetic_ranges_used > 0 and show_synthetic_curves:
        if note_lines:
            note_lines.append("")  # Línea vacía como separador
        note_lines.append(f"Nota: {synthetic_ranges_used} curva(s) con rango sintético")
        note_lines.append("(líneas punteadas, sin datos originales)")
    
    if omitted_synthetic > 0 and not show_synthetic_curves:
        note_lines.append(f"{omitted_synthetic} modelo(s) omitidos por falta de datos reales")
    
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
    
    logger.info(f"Añadidas {curves_added} curvas normalizadas para parámetro {parametro} ({synthetic_ranges_used} sintéticas, {omitted_synthetic} omitidas)")


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
    Solo procesa modelos de 1 predictor para gráficos 2D.
    
    Parameters:
    -----------
    fig : go.Figure
        Figura de Plotly donde añadir las curvas
    modelos : List[Dict]
        Lista de modelos (solo se procesan los de 1 predictor)
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


def prepare_normalized_model_curves_data(modelos: List[Dict], 
                                        parametro: str,
                                        include_synthetic: bool = True) -> List[Dict]:
    """
    Prepara datos de curvas de modelos normalizadas para la UI.
    Solo retorna modelos de 1 predictor para gráficos 2D.
    
    Parameters:
    -----------
    modelos : List[Dict]
        Lista de modelos
    parametro : str
        Nombre del parámetro objetivo
    include_synthetic : bool
        Si incluir curvas con rangos sintéticos
        
    Returns:
    --------
    List[Dict]
        Lista de diccionarios con datos de curvas preparados para la UI
    """
    curves_data = []
    
    for i, modelo in enumerate(modelos):
        # Solo modelos de 1 predictor para gráficos 2D
        if not isinstance(modelo, dict) or modelo.get('n_predictores', 0) != 1:
            continue
            
        predictor = modelo.get('predictores', [None])[0]
        if not predictor:
            continue
          # Obtener rango de datos y determinar si es sintético
        x_range, x_min, x_max, is_synthetic, warning = _compute_range_and_warning(modelo, predictor)
        
        if x_range is None or x_min is None or x_max is None:
            continue
            
        # Si no incluir sintéticos y este es sintético, omitir
        if is_synthetic and not include_synthetic:
            continue
        
        # Generar predicciones
        predictions = get_model_predictions_safe(modelo, x_range)
        if predictions is None:
            continue
        
        # Normalizar X al rango [0, 1]
        x_range_norm = (x_range - x_min) / (x_max - x_min)
        
        # Información del modelo
        tipo = modelo.get('tipo', 'unknown')
        mape = modelo.get('mape', 0)
        r2 = modelo.get('r2', 0)
        ecuacion = modelo.get('ecuacion_string', '')
        
        # Preparar datos para la UI
        curve_data = {
            'x_normalized': x_range_norm.tolist(),
            'y_predicted': predictions.tolist(),
            'x_original': x_range.tolist(),
            'predictor': predictor,
            'model_type': tipo,
            'mape': mape,
            'r2': r2,
            'equation': ecuacion,
            'is_synthetic': is_synthetic,
            'warning': warning,
            'model_index': i,
            'label': f'Curva - {predictor} ({tipo})' + (' [sintética]' if is_synthetic else ''),
            'line_style': 'dash' if is_synthetic else 'solid',
            'line_width': 2 if is_synthetic else 3,
            'color_index': i % len(COLORS['model_lines'])
        }
        
        curves_data.append(curve_data)
    
    logger.info(f"Preparados {len(curves_data)} curvas para UI del parámetro {parametro}")
    return curves_data


def prepare_imputed_points_data(modelos: List[Dict], 
                               parametro: str,
                               datos_originales: Optional[pd.DataFrame] = None,
                               detalles_por_celda: Optional[Dict] = None,
                               celda_key: Optional[str] = None) -> List[Dict]:
    """
    Prepara datos de puntos imputados para scatter plotting con tooltips.
    Utiliza detalles_por_celda si está disponible, sino intenta extraer
    de los datos de entrenamiento de los modelos.
    
    Parameters:
    -----------
    modelos : List[Dict]
        Lista de modelos
    parametro : str
        Nombre del parámetro objetivo
    datos_originales : Optional[pd.DataFrame]
        DataFrame con datos originales para contexto
    detalles_por_celda : Optional[Dict]
        Diccionario con detalles de imputación por celda
    celda_key : Optional[str]
        Clave de celda en formato "aeronave|parametro"
        
    Returns:
    --------
    List[Dict]
        Lista de diccionarios con datos de puntos imputados preparados para la UI
    """
    # Si tenemos detalles_por_celda y celda_key, extraer de ahí (método preferido)
    if detalles_por_celda is not None and celda_key is not None:
        # Filtrar solo modelos de 1 predictor
        modelos_1pred = filter_single_predictor_models(modelos)
        return extract_imputed_values_from_details(detalles_por_celda, celda_key, modelos_1pred)
    
    # Método alternativo: extraer de datos de entrenamiento de modelos
    logger.info("Usando método alternativo de extracción de datos imputados desde datos de entrenamiento")
    imputed_points = []
    
    for modelo in modelos:
        if not isinstance(modelo, dict):
            continue
        
        # Solo modelos de 1 predictor para gráficos 2D
        if modelo.get('n_predictores', 0) != 1:
            continue
            
        predictor = modelo.get('predictores', [None])[0]
        if not predictor:
            continue
        
        # Obtener datos de entrenamiento del modelo
        df_training = get_model_training_data(modelo)
        if df_training is None or df_training.empty:
            continue
        
        # Buscar valores imputados en los datos de entrenamiento
        imputed_mask = df_training.get('es_imputado', pd.Series([False] * len(df_training)))
        metodo_imputacion = df_training.get('metodo_imputacion', pd.Series([''] * len(df_training)))
        confianza_imputacion = df_training.get('confianza_imputacion', pd.Series([0.0] * len(df_training)))
        iteracion_imputacion = df_training.get('iteracion_imputacion', pd.Series([0] * len(df_training)))
        
        if not any(imputed_mask):
            continue
        
        # Filtrar solo puntos imputados
        df_imputed = df_training[imputed_mask].copy()
        
        if df_imputed.empty:
            continue
        
        # Obtener rango original para normalización
        x_range, x_min, x_max, _, _ = _compute_range_and_warning(modelo, predictor)
        if x_range is None or x_min is None or x_max is None or x_max == x_min:
            continue
        
        # Procesar cada punto imputado
        for idx, row in df_imputed.iterrows():
            x_value = row.get(predictor)
            y_value = row.get(parametro)
            metodo = row.get('metodo_imputacion', '')
            confianza = row.get('confianza_imputacion', 0.0)
            iteracion = row.get('iteracion_imputacion', 0)
            
            if pd.isna(x_value) or pd.isna(y_value):
                continue
            
            # Normalizar X
            x_normalized = (x_value - x_min) / (x_max - x_min)
            
            # Determinar warning basado en el método y confianza
            warning = ""
            if metodo == 'similitud' and confianza < 0.7:
                warning = "Baja confianza en similitud"
            elif metodo == 'correlacion' and abs(confianza) < 0.5:
                warning = "Baja correlación"
            elif metodo == 'promedio_ponderado':
                warning = "Valor promedio ponderado"
            
            # Preparar datos para la UI
            point_data = {
                'x_normalized': x_normalized,
                'y_value': y_value,
                'x_original': x_value,
                'predictor': predictor,
                'parameter': parametro,
                'imputation_method': metodo,
                'confidence': confianza,
                'iteration': iteracion,
                'warning': warning,
                'tooltip': (
                    f"Predictor: {predictor}<br>"
                    f"Valor X: {x_value:.3f}<br>"
                    f"X normalizado: {x_normalized:.3f}<br>"
                    f"Valor Y: {y_value:.3f}<br>"
                    f"Método: {metodo}<br>"
                    f"Confianza: {confianza:.3f}<br>"
                    f"Iteración: {iteracion}<br>"
                    + (f"Advertencia: {warning}<br>" if warning else "")
                ),
                'symbol': 'diamond' if metodo == 'similitud' else 'triangle-up' if metodo == 'correlacion' else 'circle',
                'size': max(6, min(15, 6 + confianza * 9)),  # Tamaño basado en confianza
                'color_method': metodo
            }
            
            imputed_points.append(point_data)
    
    logger.info(f"Preparados {len(imputed_points)} puntos imputados para UI del parámetro {parametro}")
    return imputed_points


def _compute_range_and_warning(modelo: Dict, predictor: str) -> Tuple[Optional[np.ndarray], Optional[float], Optional[float], bool, str]:
    """
    Calcula el rango de datos para un modelo y determina si es sintético.
    
    Parameters:
    -----------
    modelo : Dict
        Diccionario con información del modelo
    predictor : str
        Nombre del predictor
        
    Returns:
    --------
    tuple
        (x_range, x_min, x_max, is_synthetic, warning)
    """
    # Obtener datos originales del modelo
    df_original = get_model_original_data(modelo)
    
    x_range = None
    x_min = None
    x_max = None
    is_synthetic = False
    warning = ""
    
    if df_original is not None and not df_original.empty:
        x_data = df_original[predictor].dropna() if predictor in df_original.columns else pd.Series()
        
        if len(x_data) > 0:
            x_min = x_data.min()
            x_max = x_data.max()
            
            if x_max > x_min:
                x_range = np.linspace(x_min, x_max, 100)
    
    # Si no hay datos originales, crear un rango sintético
    if x_range is None:
        tipo = modelo.get('tipo', '')
        
        # Estimar un rango razonable basado en el tipo de modelo
        if tipo.startswith('linear'):
            x_min, x_max = 0, 10
        elif tipo.startswith('log'):
            x_min, x_max = 0.1, 10
        elif tipo.startswith('exp'):
            x_min, x_max = 0, 5
        else:
            x_min, x_max = 0, 10
        
        x_range = np.linspace(x_min, x_max, 100)
        is_synthetic = True
        warning = f"Rango sintético generado para {predictor} (sin datos originales)"
        
        logger.info(f"Usando rango sintético para modelo {predictor} ({tipo}): [{x_min}, x_max]")
    
    # Verificar que tenemos un rango válido
    if x_max is None or x_min is None or x_max == x_min:
        warning = f"Rango de X inválido para predictor {predictor}"
        return None, None, None, False, warning
    
    return x_range, x_min, x_max, is_synthetic, warning


def filter_single_predictor_models(modelos: List[Dict]) -> List[Dict]:
    """
    Filtra modelos para retornar solo aquellos con 1 predictor (para gráficos 2D).
    
    Parameters:
    -----------
    modelos : List[Dict]
        Lista de modelos
        
    Returns:
    --------
    List[Dict]
        Lista de modelos con 1 predictor únicamente
    """
    filtered_models = []
    
    for modelo in modelos:
        if isinstance(modelo, dict) and modelo.get('n_predictores', 0) == 1:
            filtered_models.append(modelo)
    
    logger.info(f"Filtrados {len(filtered_models)} modelos de 1 predictor de {len(modelos)} totales")
    return filtered_models


def extract_imputed_values_from_details(detalles_por_celda: Dict, 
                                       celda_key: str, 
                                       modelos_1pred: List[Dict]) -> List[Dict]:
    """
    Extrae los valores imputados desde detalles_por_celda para la visualización.
    
    Parameters:
    -----------
    detalles_por_celda : Dict
        Diccionario con los detalles de imputación por celda
    celda_key : str
        Clave de la celda (aeronave|parametro)
    modelos_1pred : List[Dict]
        Lista de modelos de 1 predictor para la celda
        
    Returns:
    --------
    List[Dict]
        Lista de diccionarios con los datos para la visualización de puntos imputados
    """
    imputed_points = []
    
    if celda_key not in detalles_por_celda:
        logger.warning(f"No se encontraron detalles para la celda: {celda_key}")
        return imputed_points
    
    detalles = detalles_por_celda[celda_key]
    
    # Verificar estructura esperada
    metodos_imputacion = ["final", "similitud", "correlacion"]
    metodos_encontrados = [m for m in metodos_imputacion if m in detalles]
    
    if not metodos_encontrados:
        logger.warning(f"No se encontraron métodos de imputación en los detalles para: {celda_key}")
        return imputed_points
    
    # Constantes para visualización
    SYMBOLS = {
        "final": "star",
        "similitud": "circle",
        "correlacion": "square"
    }
    
    SIZES = {
        "final": 12,
        "similitud": 10,
        "correlacion": 10    }
    
    # Usar el PRIMER modelo (mejor modelo) para determinar el rango de normalización
    # Esto asegura que la normalización sea consistente con las curvas mostradas
    if not modelos_1pred:
        logger.warning(f"No hay modelos de 1 predictor para normalizar puntos imputados en celda {celda_key}")
        return imputed_points
    
    modelo_referencia = modelos_1pred[0]  # Usar el mejor modelo como referencia
    predictor = modelo_referencia.get('predictores', [None])[0]
    if not predictor:
        logger.warning(f"No se encontró predictor válido en el mejor modelo para celda {celda_key}")
        return imputed_points
        
    # Obtener rango original para normalización del mejor modelo seleccionado
    x_range, x_min, x_max, _, _ = _compute_range_and_warning(modelo_referencia, predictor)
    if x_range is None or x_min is None or x_max is None or x_max == x_min:
        logger.warning(f"No se pudo determinar rango válido del mejor modelo para celda {celda_key}")
        return imputed_points
      # CORRECCIÓN: Usar el rango del mejor modelo seleccionado (no los X_visualizacion de la celda)
    # Esto asegura que la normalización sea consistente con las curvas del modelo
    
    # Para cada método de imputación, extraer valores
    for metodo in metodos_encontrados:
        datos_metodo = detalles[metodo]
        if "Valor imputado" not in datos_metodo or "Confianza" not in datos_metodo:
            continue
        
        valor_imputado = datos_metodo["Valor imputado"]
        confianza = datos_metodo["Confianza"]
        iteracion = datos_metodo.get("Iteración imputación", "N/A")
        advertencia = datos_metodo.get("Advertencia", "")
        x_value = datos_metodo.get("X_visualizacion")
        
        # Solo mostrar si existen ambos valores y no son None/NaN
        if x_value is None or pd.isna(x_value) or valor_imputado is None or pd.isna(valor_imputado):
            continue
        
        # Normalizar X usando el rango del mejor modelo seleccionado (consistente con curvas)
        if x_max != x_min:
            x_normalized = (x_value - x_min) / (x_max - x_min)
        else:
            x_normalized = 0.5
            
        # Determinar warning basado en el método y confianza
        if not advertencia:
            if metodo == 'similitud' and confianza < 0.7:
                advertencia = "Baja confianza en similitud"
            elif metodo == 'correlacion' and abs(confianza) < 0.5:
                advertencia = "Baja correlación"
        
        # Crear punto para visualización
        imputed_point = {
            'predictor': predictor,
            'x_normalized': x_normalized,
            'y_value': valor_imputado,
            'x_original': x_value,
            'parameter': celda_key.split('|')[1] if '|' in celda_key else '',
            'imputation_method': metodo,
            'confidence': confianza,
            'iteration': iteracion,
            'warning': advertencia,
            'symbol': SYMBOLS.get(metodo, "circle"),
            'size': SIZES.get(metodo, 10),
            'tooltip': (
                f"Predictor: {predictor}<br>"
                f"Valor X: {x_value:.3f}<br>"
                f"X normalizado: {x_normalized:.3f}<br>"
                f"Valor imputado: {valor_imputado:.3f}<br>"
                f"Método: {metodo}<br>"
                f"Confianza: {confianza:.3f}<br>"
                f"Iteración: {iteracion}"
                + (f"<br>⚠️ {advertencia}" if advertencia else "")
            )
        }
        imputed_points.append(imputed_point)
    
    logger.info(f"Extraídos {len(imputed_points)} puntos imputados de detalles para celda {celda_key}")
    return imputed_points


def filter_imputed_points_by_method(imputed_points_data: List[Dict], 
                                  selected_methods: List[str]) -> List[Dict]:
    """
    Filtra los puntos imputados según los métodos seleccionados por el usuario.
    
    Parameters:
    -----------
    imputed_points_data : List[Dict]
        Lista de datos de puntos imputados preparados
    selected_methods : List[str]
        Lista de métodos de imputación a mostrar ('final', 'similitud', 'correlacion')
        
    Returns:
    --------
    List[Dict]
        Lista filtrada de puntos imputados
    """
    if not selected_methods:
        return []
    
    filtered_points = []
    for point in imputed_points_data:
        method = point.get('imputation_method', '')
        if method in selected_methods:
            filtered_points.append(point)
    
    logger.info(f"Filtrados {len(filtered_points)} puntos imputados por métodos: {selected_methods}")
    return filtered_points

