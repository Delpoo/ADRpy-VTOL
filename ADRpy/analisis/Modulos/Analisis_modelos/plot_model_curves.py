"""
plot_model_curves.py

Funciones para curvas de modelos usando el motor de normalización unificado.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import logging
try:
    from .plot_config import COLORS, SYMBOLS, _ensure_list
    from .plot_data_access import get_model_original_data, get_model_training_data
    from .normalization_engine import normalization_engine, get_normalized_model_data
except ImportError:
    from plot_config import COLORS, SYMBOLS, _ensure_list
    from plot_data_access import get_model_original_data, get_model_training_data
    from normalization_engine import normalization_engine, get_normalized_model_data

logger = logging.getLogger(__name__)


def add_normalized_model_curves(fig: go.Figure, 
                               modelos: List[Dict], 
                               parametro: str,
                               show_synthetic_curves: bool = True,
                               show_only_real_curves: bool = False,
                               highlight_model_idx: Optional[int] = None) -> None:
    """
    Añade curvas de modelos normalizadas al gráfico usando el motor de normalización unificado.
    Solo procesa modelos de 1 predictor para gráficos 2D.
    
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
    show_only_real_curves : bool
        Si mostrar solo curvas con datos reales (omitir sintéticas)
    highlight_model_idx : int
        Índice del modelo a resaltar (opacidad y grosor de línea)
    """
    
    curves_added = 0
    warnings_added = []
    synthetic_ranges_used = 0
    omitted_synthetic = 0
    
    for i, modelo in enumerate(modelos):
        try:
            # Filtrar solo modelos de 1 predictor
            if not isinstance(modelo, dict) or modelo.get('n_predictores', 0) != 1:
                continue
                
            predictor = modelo.get('predictores', [None])[0]
            if not predictor:
                continue
            
            # Usar el motor de normalización para obtener datos
            vis_data = get_normalized_model_data(modelo, curve_resolution=100)
            
            if not vis_data.get('modelo_valido', False):
                warnings_added.append(f"Modelo {i+1} inválido: {vis_data.get('error', 'Error desconocido')}")
                continue
            
            # Obtener datos de la curva
            curva_data = vis_data.get('curva', {})
            x_normalized = curva_data.get('x_normalized')
            y_normalized = curva_data.get('y_normalized')
            curve_metadata = curva_data.get('metadata', {})
            
            if x_normalized is None or y_normalized is None:
                warnings_added.append(f"Modelo {i+1}: No se pudo generar curva")
                continue
            
            # Verificar si es rango sintético
            is_synthetic = curve_metadata.get('synthetic_range', False)
            
            if is_synthetic:
                synthetic_ranges_used += 1
                
            # Aplicar filtros de visualización
            if is_synthetic and not show_synthetic_curves:
                omitted_synthetic += 1
                continue
                
            if show_only_real_curves and is_synthetic:
                omitted_synthetic += 1
                continue
            
            # Configurar estilo de línea
            line_style = 'dash' if is_synthetic else 'solid'
            
            # Configurar colores y opacidad
            color = COLORS['model_lines'][i % len(COLORS['model_lines'])]
            opacity = 1.0
            line_width = 2
            
            if highlight_model_idx is not None:
                if i == highlight_model_idx:
                    opacity = 1.0
                    line_width = 3
                else:
                    opacity = 0.3
                    line_width = 1
            
            # Crear información de hover
            tipo_modelo = vis_data.get('tipo', 'unknown')
            ecuacion = vis_data.get('ecuacion_string', 'N/A')
            r2 = vis_data.get('metrica_r2')
            mape = vis_data.get('metrica_mape')
            confianza = vis_data.get('confianza')
            
            hover_parts = [
                f"<b>Modelo {i+1}:</b> {tipo_modelo}",
                f"<b>Predictor:</b> {predictor}",
                f"<b>Ecuación:</b> {ecuacion}"
            ]
            
            if r2 is not None:
                hover_parts.append(f"<b>R²:</b> {r2:.3f}")
            if mape is not None:
                hover_parts.append(f"<b>MAPE:</b> {mape:.1f}%")
            if confianza is not None:
                hover_parts.append(f"<b>Confianza:</b> {confianza:.3f}")
                
            if is_synthetic:
                hover_parts.append("<b>⚠️ Rango sintético</b>")
            
            hovertemplate = "<br>".join(hover_parts) + "<extra></extra>"
            
            # Añadir curva al gráfico
            fig.add_trace(go.Scatter(
                x=x_normalized,
                y=y_normalized,
                mode='lines',
                name=f"{tipo_modelo} - {predictor}" + (" (sintético)" if is_synthetic else ""),
                line=dict(
                    color=color,
                    width=line_width,
                    dash=line_style
                ),
                opacity=opacity,
                hovertemplate=hovertemplate,
                legendgroup=f"modelo_{i}",
                showlegend=True
            ))
            
            curves_added += 1
            
        except Exception as e:
            logger.error(f"Error procesando modelo {i}: {e}")
            warnings_added.append(f"Error en modelo {i+1}: {str(e)}")
    
    # Añadir anotaciones informativas si es necesario
    if warnings_added:
        warning_text = "⚠️ Advertencias: " + "; ".join(warnings_added[:3])
        if len(warnings_added) > 3:
            warning_text += f" (y {len(warnings_added)-3} más)"
        
        fig.add_annotation(
            text=warning_text,
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            font=dict(size=10, color="orange"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="orange",
            borderwidth=1
        )
    
    if synthetic_ranges_used > 0 and show_synthetic_curves:
        fig.add_annotation(
            text=f"📊 {synthetic_ranges_used} curva(s) con rango sintético (líneas punteadas)",
            xref="paper", yref="paper",
            x=0.02, y=0.02,
            showarrow=False,
            font=dict(size=9, color="blue"),
            bgcolor="rgba(255,255,255,0.8)"
        )
    
    if omitted_synthetic > 0:
        fig.add_annotation(
            text=f"🚫 {omitted_synthetic} curva(s) sintética(s) oculta(s)",
            xref="paper", yref="paper",
            x=0.98, y=0.02,
            showarrow=False,
            font=dict(size=9, color="gray"),
            bgcolor="rgba(255,255,255,0.8)",
            xanchor="right"
        )
    
    logger.info(f"Añadidas {curves_added} curvas normalizadas para parámetro {parametro}")
    logger.info(f"Estadísticas: {synthetic_ranges_used} sintéticas, {omitted_synthetic} omitidas")


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
        n_predictores = modelo.get('n_predictores', 0)
        if n_predictores == 1:
            filtered_models.append(modelo)
    
    logger.info(f"Filtrados {len(filtered_models)} modelos de 1 predictor de {len(modelos)} totales")
    return filtered_models


def extract_imputed_values_from_details(detalles_por_celda: Dict, 
                                       celda_key: str, 
                                       modelos_1pred: List[Dict]) -> List[Dict]:
    """
    Extrae los valores imputados desde detalles_por_celda para la visualización.
    Usa el motor de normalización para consistencia.
    Adaptado a la nueva estructura JSON.
    
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
    
    if celda_key not in detalles_por_celda or not modelos_1pred:
        logger.warning(f"No hay datos de imputación para celda {celda_key}")
        return imputed_points
    
    detalles = detalles_por_celda[celda_key]
    
    # Usar el primer modelo (mejor modelo) para normalización
    modelo_referencia = modelos_1pred[0]
    predictor = modelo_referencia.get('predictores', [None])[0]
    
    if not predictor:
        logger.warning(f"Modelo de referencia sin predictor válido")
        return imputed_points
    
    # Obtener rangos usando el motor de normalización
    rangos_x, rango_y = normalization_engine.get_model_data_ranges(modelo_referencia)
    
    if not rangos_x or not rango_y:
        logger.warning(f"No se pudieron obtener rangos para normalización")
        return imputed_points
    
    # Constantes para visualización
    SYMBOLS_MAP = {
        "final": "star",
        "similitud": "circle", 
        "correlacion": "square"
    }
    
    SIZES_MAP = {
        "final": 12,
        "similitud": 10,
        "correlacion": 10
    }
    
    # Procesar cada método de imputación disponible en los detalles
    for metodo_key in ['final', 'similitud', 'correlacion']:
        if metodo_key in detalles:
            metodo_data = detalles[metodo_key]
            
            # Verificar si hay datos válidos para este método
            if not isinstance(metodo_data, dict) or not metodo_data:
                continue
            
            # Obtener valor imputado y coordenadas de visualización
            valor_y = metodo_data.get('Valor imputado')
            valor_x = metodo_data.get('X_visualizacion')
            
            # Si no hay X_visualizacion en el método, usar el global de la celda
            if valor_x is None:
                valor_x = detalles.get('X_visualizacion')
            
            if valor_y is not None and valor_x is not None:
                # Crear punto de imputación
                punto = {
                    'x': valor_x,
                    'y': valor_y,
                    'metodo': metodo_key,
                    'symbol': SYMBOLS_MAP[metodo_key],
                    'size': SIZES_MAP[metodo_key],
                    'confianza': metodo_data.get('Confianza', 0),
                    'iteracion': metodo_data.get('Iteración imputación', 1),
                    'metodo_predictivo': metodo_data.get('Método predictivo', metodo_key),
                    'advertencia': metodo_data.get('Advertencia', None)
                }
                
                imputed_points.append(punto)
                logger.debug(f"Punto imputado añadido: {metodo_key} -> x={valor_x}, y={valor_y}")
    
    logger.info(f"Extraídos {len(imputed_points)} puntos imputados para celda {celda_key}")
    return imputed_points
    metodos_imputacion = ["final", "similitud", "correlacion"]
    
    for metodo in metodos_imputacion:
        if metodo not in detalles:
            continue
            
        datos_metodo = detalles[metodo]
        if not isinstance(datos_metodo, dict):
            continue
            
        # Extraer valores para el predictor específico
        if predictor in datos_metodo:
            valores_predictor = datos_metodo[predictor]
            
            for entrada in valores_predictor:
                if not isinstance(entrada, dict):
                    continue
                    
                valor_imputado = entrada.get('valor_imputado')
                confianza = entrada.get('confianza', 0.5)
                x_value = entrada.get('x_value', 0)
                
                if valor_imputado is None:
                    continue
                
                try:
                    # Normalizar usando el motor
                    x_normalized = normalization_engine.normalize_x_values([x_value], rangos_x, 0)[0]
                    y_normalized = normalization_engine.normalize_y_values([valor_imputado], rango_y)[0]
                    
                    # Crear entrada para visualización
                    point_data = {
                        'x_normalized': float(x_normalized),
                        'y_normalized': float(y_normalized),
                        'x_original': float(x_value),
                        'y_original': float(valor_imputado),
                        'imputation_method': metodo,
                        'confidence': float(confianza),
                        'symbol': SYMBOLS_MAP.get(metodo, 'circle'),
                        'size': SIZES_MAP.get(metodo, 10),
                        'predictor': predictor,
                        'hover_info': f"Método: {metodo}<br>Valor: {valor_imputado:.3f}<br>Confianza: {confianza:.3f}"
                    }
                    
                    imputed_points.append(point_data)
                    
                except Exception as e:
                    logger.error(f"Error normalizando punto imputado: {e}")
                    continue
    
    logger.info(f"Extraídos {len(imputed_points)} puntos imputados de detalles para celda {celda_key}")
    return imputed_points


def filter_imputed_points_by_method(imputed_points_data: List[Dict], 
                                  selected_methods: List[str]) -> List[Dict]:
    """
    Filtra puntos imputados según métodos seleccionados.
    
    Parameters:
    -----------
    imputed_points_data : List[Dict]
        Lista de puntos imputados
    selected_methods : List[str]
        Lista de métodos seleccionados para mostrar
        
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
    
    return filtered_points


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
        tipo = modelo.get('tipo', 'unknown')
        ecuacion = modelo.get('ecuacion_string', 'N/A')
        r2 = modelo.get('r2')
        mape = modelo.get('mape')
        confianza = modelo.get('Confianza')
        
        hover_parts = [
            f"<b>Tipo:</b> {tipo}",
            f"<b>Ecuación:</b> {ecuacion}"
        ]
        
        if r2 is not None:
            hover_parts.append(f"<b>R²:</b> {r2:.3f}")
        if mape is not None:
            hover_parts.append(f"<b>MAPE:</b> {mape:.1f}%")
        if confianza is not None:
            hover_parts.append(f"<b>Confianza:</b> {confianza:.3f}")
        
        return "<br>".join(hover_parts)
        
    except Exception as e:
        logger.error(f"Error creando hover info: {e}")
        return "Error en información del modelo"


def get_model_predictions_safe(modelo: Dict, x_range: np.ndarray) -> Optional[np.ndarray]:
    """
    Función de compatibilidad - usar normalization_engine en su lugar.
    """
    logger.warning("get_model_predictions_safe está obsoleta, usar normalization_engine")
    try:
        vis_data = get_normalized_model_data(modelo)
        curva_data = vis_data.get('curva', {})
        return np.array(curva_data.get('y_normalized', []))
    except:
        return None

