"""
plot_stability.py

Funciones y configuraciones para mejorar la estabilidad y interactividad 
de las visualizaciones de Plotly en la aplicación Dash.
"""

from typing import Dict, Any, Optional
import plotly.graph_objects as go


def get_stable_plot_config() -> Dict[str, Any]:
    """
    Retorna la configuración de Plotly para plots más estables.
    Esta configuración mejora la interactividad y previene resets inesperados.
    """
    return {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': [
            'select2d', 
            'lasso2d',
            'autoScale2d',
            'resetScale2d'
        ],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'modelo_analisis',
            'height': 600,
            'width': 1000,
            'scale': 2
        },
        'scrollZoom': True,
        'doubleClick': 'reset+autosize',
        'showTips': True,
        'responsive': True,
        # Configuración adicional para estabilidad
        'staticPlot': False,
        'editable': False,
        'showEditInChartStudio': False
    }


def get_stable_layout_config(aeronave: str, parametro: str) -> Dict[str, Any]:
    """
    Retorna la configuración de layout que mantiene el estado del zoom.
    Versión simplificada para evitar errores de compatibilidad.
    
    Parameters:
    -----------
    aeronave : str
        Nombre de la aeronave (para uirevision)
    parametro : str
        Nombre del parámetro (para uirevision)
    """
    return {
        'hovermode': 'closest',
        'clickmode': 'event+select',
        'dragmode': 'zoom',
        
        # Configuración clave para mantener el estado de UI
        'uirevision': f"{aeronave}_{parametro}_stable",
        
        # Configuración de ejes para permitir zoom estable
        'xaxis': {
            'fixedrange': False,
            'autorange': True,
            'showgrid': True,
            'gridcolor': 'lightgray',
            'zeroline': True,
            'zerolinecolor': 'gray',
            'showline': True,
            'linecolor': 'black',
            'mirror': False
        },
        
        'yaxis': {
            'fixedrange': False,
            'autorange': True,
            'showgrid': True,
            'gridcolor': 'lightgray',
            'zeroline': True,
            'zerolinecolor': 'gray',
            'showline': True,
            'linecolor': 'black',
            'mirror': False
        },
        
        # Configuración visual básica
        'plot_bgcolor': 'white',
        'paper_bgcolor': 'white',
        
        # Márgenes básicos
        'margin': {
            'l': 70,
            'r': 30,
            't': 80,
            'b': 70
        },
        
        'autosize': True,
        'showlegend': True,
        
        # Configuración de leyenda básica
        'legend': {
            'yanchor': "top",
            'y': 0.99,
            'xanchor': "left",
            'x': 0.01,
            'bgcolor': "rgba(255,255,255,0.9)",
            'bordercolor': "rgba(0,0,0,0.2)",
            'borderwidth': 1
        }
    }


def enhance_trace_interactivity(trace: go.Scatter, model_idx: int, model_info: Dict) -> go.Scatter:
    """
    Mejora la interactividad de una traza agregando metadatos e información de hover.
    
    Parameters:
    -----------
    trace : go.Scatter
        La traza a mejorar
    model_idx : int
        Índice del modelo
    model_info : Dict
        Información del modelo
    """
    # Agregar información para callbacks
    meta_info = {
        'model_idx': model_idx,
        'model_type': model_info.get('tipo', 'unknown'),
        'predictor': model_info.get('predictores', [None])[0],
        'interactive': True
    }
    
    # Actualizar trace con meta información
    trace.update(meta=meta_info)
    
    # Mejorar hover template si no existe
    if not hasattr(trace, 'hovertemplate') or not trace.hovertemplate or trace.hovertemplate == '%{text}<extra></extra>':
        # Crear hover template más informativo
        base_info = f"Modelo {model_idx + 1}: {model_info.get('tipo', 'Unknown')}"
        if 'mape' in model_info:
            base_info += f"<br>MAPE: {model_info['mape']:.3f}%"
        if 'r2' in model_info:
            base_info += f"<br>R²: {model_info['r2']:.3f}"
        
        trace.update(hovertemplate=f"{base_info}<br>%{{text}}<extra></extra>")
    
    return trace


def should_preserve_zoom(trigger_info: Dict, current_selection: Dict) -> bool:
    """
    Determina si se debe preservar el zoom actual basado en el trigger.
    
    Parameters:
    -----------
    trigger_info : Dict
        Información sobre qué disparó el callback
    current_selection : Dict
        Selección actual (aeronave, parámetro)
    
    Returns:
    --------
    bool
        True si se debe preservar el zoom
    """
    if not trigger_info or not current_selection:
        return False
    
    # Preservar zoom para cambios menores (selección, checkboxes, etc.)
    preserve_triggers = [
        'selected-model-store',
        'hide-plot-legend',
        'show-training-points',
        'show-model-curves',
        'show-only-real-curves',
        'imputation-methods-checklist'
    ]
    
    triggered_ids = trigger_info.get('triggered_ids', [])
    
    # Si solo se dispararon triggers "menores", preservar zoom
    if all(tid in preserve_triggers for tid in triggered_ids):
        return True
    
    # Si cambió aeronave o parámetro, no preservar zoom
    major_changes = ['aeronave-dropdown', 'parametro-dropdown', 'predictor-dropdown']
    if any(tid in major_changes for tid in triggered_ids):
        return False
    
    return True


def apply_stable_configuration(fig: go.Figure, aeronave: str, parametro: str, preserve_zoom: bool = True) -> go.Figure:
    """
    Aplica configuración estable a una figura de Plotly.
    
    Parameters:
    -----------
    fig : go.Figure
        Figura a configurar
    aeronave : str
        Nombre de la aeronave
    parametro : str
        Nombre del parámetro
    preserve_zoom : bool
        Si preservar el estado del zoom
    
    Returns:
    --------
    go.Figure
        Figura configurada
    """
    layout_config = get_stable_layout_config(aeronave, parametro)
    
    if not preserve_zoom:
        # Forzar autorange si no se debe preservar zoom
        layout_config['xaxis']['autorange'] = True
        layout_config['yaxis']['autorange'] = True
        # Cambiar uirevision para forzar reset
        layout_config['uirevision'] = f"{aeronave}_{parametro}_reset"
    
    fig.update_layout(**layout_config)
    
    return fig
