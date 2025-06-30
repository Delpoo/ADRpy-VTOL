"""
advanced_filters.py

Sistema de filtros avanzados para la aplicación de visualización.
Incluye filtros por rangos, búsqueda de texto y filtros inteligentes.
"""

from typing import Dict, List, Any, Optional, Tuple
import re
import pandas as pd
from dash import html, dcc


class AdvancedFilterSystem:
    """Sistema de filtros avanzados para modelos."""
    
    def __init__(self):
        """Inicializar el sistema de filtros."""
        self.active_filters = {}
    
    def filter_models_by_text(self, modelos: List[Dict], search_text: str) -> List[Dict]:
        """
        Filtrar modelos por texto de búsqueda.
        
        Parameters:
        -----------
        modelos : List[Dict]
            Lista de modelos
        search_text : str
            Texto a buscar
            
        Returns:
        --------
        List[Dict]
            Modelos filtrados
        """
        if not search_text:
            return modelos
        
        search_lower = search_text.lower()
        filtered = []
        
        for modelo in modelos:
            if not isinstance(modelo, dict):
                continue
            
            # Buscar en tipo de modelo
            if search_lower in modelo.get('tipo', '').lower():
                filtered.append(modelo)
                continue
            
            # Buscar en predictores
            predictores = modelo.get('predictores', [])
            if any(search_lower in pred.lower() for pred in predictores):
                filtered.append(modelo)
                continue
            
            # Buscar en ecuación
            ecuacion = modelo.get('ecuacion', '')
            if search_lower in ecuacion.lower():
                filtered.append(modelo)
                continue
            
            # Buscar en métricas (formato texto)
            mape = modelo.get('mape', 0)
            r2 = modelo.get('r2', 0)
            if (search_lower in f"{mape:.3f}" or 
                search_lower in f"{r2:.3f}"):
                filtered.append(modelo)
                continue
        
        return filtered
    
    def filter_models_by_mape_range(self, modelos: List[Dict], 
                                   min_mape: float, max_mape: float) -> List[Dict]:
        """
        Filtrar modelos por rango de MAPE.
        
        Parameters:
        -----------
        modelos : List[Dict]
            Lista de modelos
        min_mape : float
            MAPE mínimo
        max_mape : float
            MAPE máximo
            
        Returns:
        --------
        List[Dict]
            Modelos filtrados
        """
        filtered = []
        for modelo in modelos:
            if not isinstance(modelo, dict):
                continue
            
            mape = modelo.get('mape', float('inf'))
            if min_mape <= mape <= max_mape:
                filtered.append(modelo)
        
        return filtered
    
    def filter_models_by_r2_range(self, modelos: List[Dict], 
                                 min_r2: float, max_r2: float) -> List[Dict]:
        """
        Filtrar modelos por rango de R².
        
        Parameters:
        -----------
        modelos : List[Dict]
            Lista de modelos
        min_r2 : float
            R² mínimo
        max_r2 : float
            R² máximo
            
        Returns:
        --------
        List[Dict]
            Modelos filtrados
        """
        filtered = []
        for modelo in modelos:
            if not isinstance(modelo, dict):
                continue
            
            r2 = modelo.get('r2', -1)
            if min_r2 <= r2 <= max_r2:
                filtered.append(modelo)
        
        return filtered
    
    def filter_models_by_predictor_count(self, modelos: List[Dict], 
                                        min_count: int, max_count: int) -> List[Dict]:
        """
        Filtrar modelos por número de predictores.
        
        Parameters:
        -----------
        modelos : List[Dict]
            Lista de modelos
        min_count : int
            Número mínimo de predictores
        max_count : int
            Número máximo de predictores
            
        Returns:
        --------
        List[Dict]
            Modelos filtrados
        """
        filtered = []
        for modelo in modelos:
            if not isinstance(modelo, dict):
                continue
            
            n_pred = modelo.get('n_predictores', 0)
            if min_count <= n_pred <= max_count:
                filtered.append(modelo)
        
        return filtered
    
    def apply_combined_filters(self, modelos: List[Dict], filters: Dict) -> List[Dict]:
        """
        Aplicar múltiples filtros de forma combinada.
        
        Parameters:
        -----------
        modelos : List[Dict]
            Lista de modelos original
        filters : Dict
            Diccionario con todos los filtros a aplicar
            
        Returns:
        --------
        List[Dict]
            Modelos filtrados
        """
        result = modelos.copy()
        
        # Filtro de texto
        if filters.get('search_text'):
            result = self.filter_models_by_text(result, filters['search_text'])
        
        # Filtro de MAPE
        if filters.get('mape_range'):
            min_mape, max_mape = filters['mape_range']
            result = self.filter_models_by_mape_range(result, min_mape, max_mape)
        
        # Filtro de R²
        if filters.get('r2_range'):
            min_r2, max_r2 = filters['r2_range']
            result = self.filter_models_by_r2_range(result, min_r2, max_r2)
        
        # Filtro de número de predictores
        if filters.get('predictor_count_range'):
            min_count, max_count = filters['predictor_count_range']
            result = self.filter_models_by_predictor_count(result, min_count, max_count)
        
        # Filtro de tipos específicos
        if filters.get('model_types'):
            selected_types = filters['model_types']
            result = [m for m in result if isinstance(m, dict) and 
                     m.get('tipo', '') in selected_types]
        
        return result
    
    def get_filter_statistics(self, modelos: List[Dict]) -> Dict:
        """
        Obtener estadísticas para configurar los rangos de filtros.
        
        Parameters:
        -----------
        modelos : List[Dict]
            Lista de modelos
            
        Returns:
        --------
        Dict
            Estadísticas de los modelos
        """
        if not modelos:
            return {}
        
        valid_models = [m for m in modelos if isinstance(m, dict)]
        
        mape_values = [m.get('mape', 0) for m in valid_models if m.get('mape') is not None]
        r2_values = [m.get('r2', 0) for m in valid_models if m.get('r2') is not None]
        predictor_counts = [m.get('n_predictores', 0) for m in valid_models]
        
        stats = {
            'mape': {
                'min': min(mape_values) if mape_values else 0,
                'max': max(mape_values) if mape_values else 100,
                'mean': sum(mape_values) / len(mape_values) if mape_values else 0
            },
            'r2': {
                'min': min(r2_values) if r2_values else 0,
                'max': max(r2_values) if r2_values else 1,
                'mean': sum(r2_values) / len(r2_values) if r2_values else 0
            },
            'predictor_count': {
                'min': min(predictor_counts) if predictor_counts else 1,
                'max': max(predictor_counts) if predictor_counts else 5
            }
        }
        
        return stats


def create_advanced_filters_ui(filter_stats: Optional[Dict] = None) -> html.Div:
    """
    Crear la interfaz de filtros avanzados.
    
    Parameters:
    -----------
    filter_stats : Dict
        Estadísticas para configurar rangos
        
    Returns:
    --------
    html.Div
        Componente de filtros avanzados
    """
    if not filter_stats:
        filter_stats = {
            'mape': {'min': 0, 'max': 100, 'mean': 50},
            'r2': {'min': 0, 'max': 1, 'mean': 0.5},
            'predictor_count': {'min': 1, 'max': 5}
        }
    
    return html.Div([
        # Encabezado de filtros
        html.Div([
            html.H4([
                html.I(className="fas fa-filter", style={'margin-right': '10px'}),
                "🔍 Filtros Avanzados"
            ], style={'margin-bottom': '20px', 'color': '#2c3e50'}),
            
            # Botón para limpiar filtros
            html.Button([
                html.I(className="fas fa-eraser", style={'margin-right': '5px'}),
                "Limpiar Filtros"
            ], 
            id="clear-filters-btn", 
            className="btn btn-outline-secondary btn-sm",
            style={'float': 'right', 'margin-bottom': '10px'})
        ], style={'clearfix': True}),
        
        # Búsqueda de texto
        html.Div([
            html.Label([
                html.I(className="fas fa-search", style={'margin-right': '5px'}),
                "Búsqueda:"
            ], style={'font-weight': 'bold', 'margin-bottom': '5px'}),
            dcc.Input(
                id="search-models-input",
                type="text",
                placeholder="🔍 Buscar por tipo, predictor, ecuación o métrica...",
                style={
                    'width': '100%',
                    'padding': '8px',
                    'border': '1px solid #ddd',
                    'border-radius': '4px',
                    'font-size': '14px'
                },
                debounce=True  # Esperar que el usuario termine de escribir
            )
        ], style={'margin-bottom': '20px'}),
        
        # Filtro de MAPE
        html.Div([
            html.Label([
                html.I(className="fas fa-percentage", style={'margin-right': '5px'}),
                "Rango MAPE:"
            ], style={'font-weight': 'bold', 'margin-bottom': '5px'}),
            dcc.RangeSlider(
                id="mape-range-slider",
                min=0,
                max=max(100, filter_stats['mape']['max'] * 1.1),
                step=0.1,
                value=[0, filter_stats['mape']['max']],
                marks={
                    0: '0%',
                    filter_stats['mape']['mean']: f"{filter_stats['mape']['mean']:.1f}%",
                    filter_stats['mape']['max']: f"{filter_stats['mape']['max']:.1f}%"
                },
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], style={'margin-bottom': '25px'}),
        
        # Filtro de R²
        html.Div([
            html.Label([
                html.I(className="fas fa-chart-line", style={'margin-right': '5px'}),
                "Rango R²:"
            ], style={'font-weight': 'bold', 'margin-bottom': '5px'}),
            dcc.RangeSlider(
                id="r2-range-slider",
                min=0,
                max=1,
                step=0.01,
                value=[0, 1],
                marks={
                    0: '0.0',
                    0.5: '0.5',
                    filter_stats['r2']['mean']: f"{filter_stats['r2']['mean']:.2f}",
                    1: '1.0'
                },
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], style={'margin-bottom': '25px'}),
        
        # Filtro de número de predictores
        html.Div([
            html.Label([
                html.I(className="fas fa-layer-group", style={'margin-right': '5px'}),
                "Número de Predictores:"
            ], style={'font-weight': 'bold', 'margin-bottom': '5px'}),
            dcc.RangeSlider(
                id="predictor-count-slider",
                min=1,
                max=filter_stats['predictor_count']['max'],
                step=1,
                value=[1, filter_stats['predictor_count']['max']],
                marks={i: str(i) for i in range(1, filter_stats['predictor_count']['max'] + 1)},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], style={'margin-bottom': '25px'}),
        
        # Filtro de calidad del modelo
        html.Div([
            html.Label([
                html.I(className="fas fa-star", style={'margin-right': '5px'}),
                "Calidad del Modelo:"
            ], style={'font-weight': 'bold', 'margin-bottom': '5px'}),
            dcc.Dropdown(
                id="quality-filter-dropdown",
                options=[
                    {'label': '🌟 Todos los modelos', 'value': 'all'},
                    {'label': '⭐ Solo modelos buenos (MAPE < 10%)', 'value': 'good'},
                    {'label': '🏆 Solo modelos excelentes (MAPE < 5%)', 'value': 'excellent'},
                    {'label': '🥇 Solo el mejor modelo', 'value': 'best'}
                ],
                value='all',
                style={'margin-bottom': '10px'}
            )
        ], style={'margin-bottom': '20px'}),
        
        # Indicador de filtros activos
        html.Div(
            id="active-filters-indicator",
            style={'margin-top': '15px', 'padding': '10px', 'background-color': '#f8f9fa', 'border-radius': '4px'}
        ),
        
        # Estadísticas de filtrado
        html.Div(
            id="filter-stats-display",
            style={'margin-top': '10px', 'font-size': '12px', 'color': '#6c757d'}
        )
        
    ], style={
        'padding': '20px',
        'background-color': '#ffffff',
        'border': '1px solid #e0e0e0',
        'border-radius': '8px',
        'box-shadow': '0 2px 4px rgba(0,0,0,0.1)'
    })


def create_filter_callbacks(app, filter_system: AdvancedFilterSystem):
    """Crear callbacks para el sistema de filtros avanzados."""
    
    from dash import Input, Output, State, html
    import dash
    
    @app.callback(
        [Output('filtered-models-store', 'data'),
         Output('active-filters-indicator', 'children'),
         Output('filter-stats-display', 'children')],
        [Input('search-models-input', 'value'),
         Input('mape-range-slider', 'value'),
         Input('r2-range-slider', 'value'),
         Input('predictor-count-slider', 'value'),
         Input('quality-filter-dropdown', 'value'),
         Input('clear-filters-btn', 'n_clicks')],
        [State('models-data-store', 'data'),
         State('aeronave-dropdown', 'value'),
         State('parametro-dropdown', 'value')]
    )
    def apply_advanced_filters(search_text, mape_range, r2_range, predictor_range, 
                              quality_filter, clear_clicks, models_data, aeronave, parametro):
        """Aplicar filtros avanzados a los modelos."""
        
        ctx = dash.callback_context
        
        # Si se presionó limpiar filtros
        if ctx.triggered and ctx.triggered[0]['prop_id'] == 'clear-filters-btn.n_clicks':
            return None, "", ""
        
        if not models_data or not aeronave or not parametro:
            return None, "", ""
        
        # Obtener modelos de la celda actual
        celda_key = f"{aeronave}|{parametro}"
        modelos = models_data.get('modelos', {}).get(celda_key, [])
        
        if not modelos:
            return None, "", "No hay modelos disponibles"
        
        # Preparar filtros
        filters = {}
        active_filters_list = []
        
        if search_text:
            filters['search_text'] = search_text
            active_filters_list.append(f"Texto: '{search_text}'")
        
        if mape_range and mape_range != [0, 100]:
            filters['mape_range'] = mape_range
            active_filters_list.append(f"MAPE: {mape_range[0]:.1f}% - {mape_range[1]:.1f}%")
        
        if r2_range and r2_range != [0, 1]:
            filters['r2_range'] = r2_range
            active_filters_list.append(f"R²: {r2_range[0]:.2f} - {r2_range[1]:.2f}")
        
        if predictor_range and predictor_range[0] > 1:
            filters['predictor_count_range'] = predictor_range
            active_filters_list.append(f"Predictores: {predictor_range[0]} - {predictor_range[1]}")
        
        # Aplicar filtro de calidad
        if quality_filter and quality_filter != 'all':
            if quality_filter == 'good':
                filters['mape_range'] = [0, 10]
                active_filters_list.append("Solo modelos buenos (MAPE < 10%)")
            elif quality_filter == 'excellent':
                filters['mape_range'] = [0, 5]
                active_filters_list.append("Solo modelos excelentes (MAPE < 5%)")
            elif quality_filter == 'best':
                # Encontrar el mejor modelo
                valid_models = [m for m in modelos if isinstance(m, dict) and m.get('mape') is not None]
                if valid_models:
                    best_mape = min(m['mape'] for m in valid_models)
                    filters['mape_range'] = [best_mape - 0.001, best_mape + 0.001]
                    active_filters_list.append("Solo el mejor modelo")
        
        # Aplicar filtros combinados
        filtered_models = filter_system.apply_combined_filters(modelos, filters)
        
        # Crear indicador de filtros activos
        if active_filters_list:
            active_indicator = html.Div([
                html.Strong("🎯 Filtros activos: "),
                html.Ul([html.Li(f) for f in active_filters_list])
            ])
        else:
            active_indicator = html.Div("Sin filtros activos", style={'font-style': 'italic'})
        
        # Estadísticas de filtrado
        total_models = len([m for m in modelos if isinstance(m, dict)])
        filtered_count = len([m for m in filtered_models if isinstance(m, dict)])
        stats_text = f"Mostrando {filtered_count} de {total_models} modelos"
        
        return {celda_key: filtered_models}, active_indicator, stats_text
