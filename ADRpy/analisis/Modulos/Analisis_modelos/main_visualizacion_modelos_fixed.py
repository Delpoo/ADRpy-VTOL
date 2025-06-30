"""
Aplicación Principal de Visualización de Modelos - VERSIÓN CORREGIDA
==================================================================

Esta versión incluye todas las funcionalidades principales sin los sistemas
adicionales que causan problemas. Se agregan paso a paso para identificar
dónde está el conflicto.
"""

import os
import sys
from typing import Optional
import logging

# Añadir el directorio padre al path para imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    import dash
    from dash import dcc, html, dash_table
    from dash.dependencies import Input, Output, State
    import plotly.graph_objects as go
    import pandas as pd
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    dash = None
    dcc = None
    html = None
    dash_table = None
    Input = None
    Output = None
    State = None
    go = None

# Importar pandas por separado (siempre necesario)
import pandas as pd

from .data_loader import (
    load_models_data, 
    extract_unique_values, 
    filter_models,
    get_parametros_for_aeronave
)

from .ui_components import (
    create_aeronave_dropdown,
    create_parametro_dropdown,
    create_tipo_modelo_checklist,
    create_visualization_options,
    create_summary_table,
    format_model_info,
    create_predictor_dropdown,
    create_imputation_methods_checklist,
    create_integrated_advanced_filters
)

from .plot_stability import (
    get_stable_plot_config,
    apply_stable_configuration,
    should_preserve_zoom
)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_main_layout_fixed() -> html.Div:
    """
    Crea el layout principal CORREGIDO sin sistemas problemáticos.
    
    Returns:
    --------
    html.Div
        Layout principal corregido
    """
    return html.Div([
        html.H1("Análisis Interactivo de Modelos de Imputación", 
               style={'textAlign': 'center', 'marginBottom': '30px'}),
        
        # Barra de herramientas simple
        html.Div([
            html.Button(
                id='toggle-filters-btn',
                children='👁️ Ocultar/Mostrar Filtros',
                n_clicks=0,
                style={
                    'padding': '8px 16px',
                    'backgroundColor': '#007bff',
                    'color': 'white',
                    'border': 'none',
                    'borderRadius': '5px',
                    'cursor': 'pointer',
                    'marginRight': '10px',
                    'boxShadow': '0 2px 6px rgba(0,0,0,0.1)'
                }
            )
        ], style={
            'textAlign': 'center',
            'marginBottom': '20px',
            'padding': '10px',
            'backgroundColor': '#f8f9fa',
            'borderRadius': '5px'
        }),
        
        html.Div([
            # Panel de filtros (colapsable)
            html.Div(id='filters-panel', children=[
                html.H3("Filtros y Controles"),
                html.Label("Aeronave:"),
                html.Div(id='aeronave-dropdown-container'),
                html.Label("Parámetro:"),
                html.Div(id='parametro-dropdown-container'),
                html.Label("Predictor:"),
                html.Div(id='predictor-dropdown-container'),
                html.Label("Tipos de Modelo:"),
                html.Div(id='tipo-modelo-container'),
                html.Div(id='visualization-options-container'),
                html.Label("Métodos de Imputación:"),
                html.Div(id='imputation-methods-container'),
                
                # Sección de filtros avanzados
                html.Div([
                    html.Button(
                        id='toggle-advanced-filters-section',
                        children='🔽 Mostrar Filtros Avanzados',
                        n_clicks=0,
                        style={
                            'padding': '5px 10px',
                            'backgroundColor': '#6c757d',
                            'color': 'white',
                            'border': 'none',
                            'borderRadius': '3px',
                            'cursor': 'pointer',
                            'fontSize': '12px',
                            'marginTop': '10px',
                            'width': '100%'
                        }
                    ),
                    html.Div(id='advanced-filters-section', 
                            style={'display': 'none'})
                ]),
                
                html.Button('Actualizar Visualización', 
                           id='update-button',
                           style={
                               'marginTop': '20px',
                               'padding': '10px 20px',
                               'backgroundColor': '#007bff',
                               'color': 'white',
                               'border': 'none',
                               'borderRadius': '5px',
                               'cursor': 'pointer'
                           })
            ], style={
                'width': '22%',
                'minWidth': '220px',
                'maxWidth': '320px',
                'display': 'block',
                'verticalAlign': 'top',
                'padding': '20px',
                'backgroundColor': '#f8f9fa',
                'borderRadius': '5px',
                'margin': '10px',
                'boxSizing': 'border-box',
                'transition': 'all 0.3s ease',
                'overflowY': 'auto',
                'height': 'fit-content'
            }),
            
            # Área de contenido principal (centro, expandible)
            html.Div(id='main-content-container', children=[
                # Pestañas en la parte superior
                dcc.Tabs(id='plot-tabs', value='main-view', children=[
                    dcc.Tab(label='📊 Vista Principal', value='main-view'),
                    dcc.Tab(label='🔄 Comparación', value='comparison-view'),
                    dcc.Tab(label='📋 Métricas', value='metrics-view')
                ], style={
                    'height': '44px',
                    'marginBottom': '10px'
                }),
                # Contenido de las pestañas
                html.Div(id='tab-content', style={
                    'height': '65vh',
                    'width': '100%',
                    'position': 'relative'
                })
            ], style={
                'width': '56%',
                'minWidth': '320px',
                'display': 'inline-block',
                'verticalAlign': 'top',
                'margin': '10px',
                'boxSizing': 'border-box',
                'transition': 'all 0.3s ease',
            }),
            
            # Panel de información (derecha)
            html.Div([
                html.H3("Información del Modelo"),
                html.Div(id='model-info-content', children=[
                    html.P("Haga hover o click en un modelo para ver su información.")
                ])
            ], style={
                'width': '22%',
                'minWidth': '220px',
                'maxWidth': '340px',
                'display': 'inline-block',
                'verticalAlign': 'top',
                'padding': '20px',
                'backgroundColor': '#f8f9fa',
                'borderRadius': '5px',
                'margin': '10px',
                'boxSizing': 'border-box',
                'transition': 'all 0.3s ease'
            })
        ], style={
            'display': 'flex',
            'flexWrap': 'nowrap',
            'alignItems': 'flex-start',
            'justifyContent': 'space-between',
            'width': '100%',
            'position': 'relative',
            'minHeight': '650px'
        }),
        
        # Tabla de resumen
        html.Div([
            html.H3("Resumen de Modelos"),
            html.Div(id='summary-table-container')
        ], style={
            'margin': '20px',
            'padding': '20px',
            'backgroundColor': '#f8f9fa',
            'borderRadius': '5px'
        }),
        
        # Stores necesarios
        dcc.Store(id='models-data-store'),
        dcc.Store(id='unique-values-store'),
        dcc.Store(id='selected-model-store', data=None),
        dcc.Store(id='comparison-models-store', data=[])
    ])


def main_visualizacion_modelos_fixed(json_path: Optional[str] = None, 
                                   use_dash: bool = True,
                                   port: int = 8050,
                                   debug: bool = False) -> None:
    """
    Función principal CORREGIDA para ejecutar la visualización de modelos.
    
    Parameters:
    -----------
    json_path : Optional[str]
        Ruta al archivo JSON. Si es None, usa la ruta por defecto.
    use_dash : bool
        Si usar Dash (True) o matplotlib (False)
    port : int
        Puerto para la aplicación Dash
    debug : bool
        Modo debug para Dash
    """
    # Ruta por defecto del JSON
    if json_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(script_dir, '..', '..', 'Results', 'modelos_completos_por_celda.json')
        json_path = os.path.normpath(json_path)
    
    if not os.path.exists(json_path):
        logger.error(f"No se encontró el archivo JSON: {json_path}")
        print(f"Error: No se encontró el archivo JSON en: {json_path}")
        return
    
    logger.info(f"Cargando datos desde: {json_path}")
    
    try:
        # Cargar datos
        modelos_por_celda, detalles_por_celda = load_models_data(json_path)
        unique_values = extract_unique_values(modelos_por_celda)
        
        logger.info("Datos cargados exitosamente")
        
        if use_dash and DASH_AVAILABLE:
            _run_dash_app_fixed(modelos_por_celda, detalles_por_celda, unique_values, port, debug)
        else:
            print("Dash no disponible o deshabilitado")
            
    except Exception as e:
        logger.error(f"Error en la aplicación: {e}")
        print(f"Error ejecutando la aplicación: {e}")


def _run_dash_app_fixed(modelos_por_celda, detalles_por_celda, unique_values, port, debug):
    """Ejecuta la aplicación CORREGIDA con Dash."""
    if not DASH_AVAILABLE:
        print("Dash no está disponible. No se puede ejecutar la aplicación interactiva.")
        return

    # Re-importar los componentes de Dash
    try:
        import dash
        from dash import dcc, html, dash_table
        from dash.dependencies import Input, Output, State
        import plotly.graph_objects as go
        from .plot_interactive import (
            create_interactive_plot,
            create_metrics_summary_table,
        )
    except ImportError as e:
        print(f"Error: No se pueden importar los componentes necesarios de Dash: {e}")
        return

    # Crear aplicación Dash con configuración para callbacks dinámicos
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    app.title = "Análisis de Modelos de Imputación"
    
    # Layout principal CORREGIDO
    app.layout = create_main_layout_fixed()
    
    # CALLBACKS BÁSICOS Y FUNCIONALES
    
    # 1. Callback para toggle de filtros
    @app.callback(
        [Output('filters-panel', 'style'),
         Output('main-content-container', 'style')],
        [Input('toggle-filters-btn', 'n_clicks')],
        [State('filters-panel', 'style')]
    )
    def toggle_filters_panel(n_clicks, current_style):
        if n_clicks is None or n_clicks == 0:
            # Estado inicial - filtros visibles
            filters_style = {
                'width': '22%',
                'minWidth': '220px',
                'maxWidth': '320px',
                'display': 'block',
                'verticalAlign': 'top',
                'padding': '20px',
                'backgroundColor': '#f8f9fa',
                'borderRadius': '5px',
                'margin': '10px',
                'boxSizing': 'border-box',
                'transition': 'all 0.3s ease',
                'overflowY': 'auto',
                'height': 'fit-content'
            }
            main_style = {
                'width': '56%',
                'minWidth': '320px',
                'display': 'inline-block',
                'verticalAlign': 'top',
                'margin': '10px',
                'boxSizing': 'border-box',
                'transition': 'all 0.3s ease',
            }
            return filters_style, main_style
        
        is_visible = current_style.get('display', 'block') != 'none'
        
        if is_visible:
            # Ocultar filtros - expandir contenido principal
            filters_style = {'display': 'none'}
            main_style = {
                'width': '78%',
                'minWidth': '320px',
                'display': 'inline-block',
                'verticalAlign': 'top',
                'margin': '10px auto',
                'boxSizing': 'border-box',
                'transition': 'all 0.3s ease',
            }
        else:
            # Mostrar filtros - reducir contenido principal
            filters_style = {
                'width': '22%',
                'minWidth': '220px',
                'maxWidth': '320px',
                'display': 'block',
                'verticalAlign': 'top',
                'padding': '20px',
                'backgroundColor': '#f8f9fa',
                'borderRadius': '5px',
                'margin': '10px',
                'boxSizing': 'border-box',
                'transition': 'all 0.3s ease',
                'overflowY': 'auto',
                'height': 'fit-content'
            }
            main_style = {
                'width': '56%',
                'minWidth': '320px',
                'display': 'inline-block',
                'verticalAlign': 'top',
                'margin': '10px',
                'boxSizing': 'border-box',
                'transition': 'all 0.3s ease',
            }
        
        return filters_style, main_style
    
    # 2. Callback para filtros avanzados toggle
    @app.callback(
        [Output('advanced-filters-section', 'style'),
         Output('advanced-filters-section', 'children')],
        [Input('toggle-advanced-filters-section', 'n_clicks')],
        [State('advanced-filters-section', 'style')]
    )
    def toggle_advanced_filters_section(n_clicks, current_style):
        if n_clicks is None or n_clicks == 0:
            return {'display': 'none'}, []
        
        is_visible = current_style.get('display', 'none') != 'none'
        
        if is_visible:
            return {'display': 'none'}, []
        else:
            return {
                'display': 'block',
                'marginTop': '10px'
            }, [create_integrated_advanced_filters()]
    
    # 3. Callback de inicialización de controles
    @app.callback(
        [Output('aeronave-dropdown-container', 'children'),
         Output('tipo-modelo-container', 'children'),
         Output('visualization-options-container', 'children'),
         Output('imputation-methods-container', 'children'),
         Output('models-data-store', 'data'),
         Output('unique-values-store', 'data')],
        [Input('update-button', 'id')]
    )
    def initialize_controls(_):
        return (
            create_aeronave_dropdown(unique_values['aeronaves']),
            create_tipo_modelo_checklist(unique_values['tipos_modelo']),
            create_visualization_options(),
            create_imputation_methods_checklist(),
            {'modelos': modelos_por_celda, 'detalles': detalles_por_celda},
            unique_values
        )
    
    # 4. Callback para actualizar dropdown de parámetros
    @app.callback(
        Output('parametro-dropdown-container', 'children'),
        [Input('aeronave-dropdown', 'value')],
        [State('models-data-store', 'data')]
    )
    def update_parametro_dropdown(aeronave_selected, models_data):
        if not aeronave_selected:
            return create_parametro_dropdown([])
        
        parametros = get_parametros_for_aeronave(modelos_por_celda, aeronave_selected)
        return create_parametro_dropdown(parametros)
    
    # 5. Callback para actualizar dropdown de predictores
    @app.callback(
        Output('predictor-dropdown-container', 'children'),
        [Input('parametro-dropdown', 'value'),
         Input('aeronave-dropdown', 'value')],
        [State('models-data-store', 'data')]
    )
    def update_predictor_dropdown(parametro, aeronave, models_data):
        if not parametro or not aeronave or not models_data:
            return create_predictor_dropdown([])
        celda_key = f"{aeronave}|{parametro}"
        modelos = models_data['modelos'].get(celda_key, [])
        all_preds = set()
        for m in modelos:
            if isinstance(m, dict):
                all_preds.update(m.get('predictores', []))
        return create_predictor_dropdown(sorted(all_preds))
    
    # 6. Callback para contenido de pestañas
    @app.callback(
        Output('tab-content', 'children'),
        [Input('plot-tabs', 'value')],
        [State('models-data-store', 'data'),
         State('aeronave-dropdown', 'value'),
         State('parametro-dropdown', 'value')]
    )
    def update_tab_content(active_tab, models_data, aeronave, parametro):
        if active_tab == 'main-view':
            return html.Div([
                dcc.Graph(
                    id='main-plot', 
                    style={'height': '100%', 'width': '100%'},
                    config={
                        'displayModeBar': True,
                        'displaylogo': False,
                        'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
                        'toImageButtonOptions': {
                            'format': 'png',
                            'filename': 'modelo_analisis',
                            'height': 600,
                            'width': 1000,
                            'scale': 1
                        },
                        'scrollZoom': True,
                        'doubleClick': 'reset+autosize',
                        'responsive': True,
                        'showTips': True
                    }
                )
            ], style={'height': '100%', 'width': '100%'})
        
        elif active_tab == 'comparison-view':
            return html.Div([
                html.H4("🔄 Comparación de Modelos", style={'marginBottom': '15px'}),
                html.P("Funcionalidad de comparación disponible en versión completa.", 
                      style={'textAlign': 'center', 'color': 'gray', 'padding': '20px'})
            ], style={'padding': '20px', 'height': '100%', 'boxSizing': 'border-box'})
        
        elif active_tab == 'metrics-view':
            if not models_data or not aeronave or not parametro:
                return html.P("Seleccione aeronave y parámetro para ver métricas",
                             style={'textAlign': 'center', 'color': 'gray', 'padding': '20px'})
            
            # Mostrar métricas agregadas
            celda_key = f"{aeronave}|{parametro}"
            modelos = models_data.get('modelos', {}).get(celda_key, [])
            
            if not modelos:
                return html.P("No hay modelos disponibles",
                             style={'textAlign': 'center', 'color': 'gray', 'padding': '20px'})
            
            # Calcular estadísticas
            modelos_validos = [m for m in modelos if isinstance(m, dict)]
            
            if not modelos_validos:
                return html.P("No hay modelos válidos para calcular métricas",
                             style={'textAlign': 'center', 'color': 'gray', 'padding': '20px'})
            
            # Estadísticas básicas
            mapes = [m.get('mape', 0) for m in modelos_validos if m.get('mape') is not None]
            r2s = [m.get('r2', 0) for m in modelos_validos if m.get('r2') is not None]
            
            stats_content = []
            
            if mapes:
                stats_content.extend([
                    html.H5("📊 Estadísticas MAPE"),
                    html.P(f"Promedio: {sum(mapes)/len(mapes):.3f}%"),
                    html.P(f"Mínimo: {min(mapes):.3f}%"),
                    html.P(f"Máximo: {max(mapes):.3f}%"),
                ])
            
            if r2s:
                stats_content.extend([
                    html.H5("📈 Estadísticas R²"),
                    html.P(f"Promedio: {sum(r2s)/len(r2s):.3f}"),
                    html.P(f"Mínimo: {min(r2s):.3f}"),
                    html.P(f"Máximo: {max(r2s):.3f}"),
                ])
            
            # Contar tipos de modelo
            tipos = {}
            for modelo in modelos_validos:
                tipo = modelo.get('tipo', 'Desconocido')
                tipos[tipo] = tipos.get(tipo, 0) + 1
            
            stats_content.extend([
                html.H5("🏷️ Distribución por tipo"),
                html.Ul([html.Li(f"{tipo}: {count} modelos") for tipo, count in tipos.items()])
            ])
            
            return html.Div([
                html.H4("📋 Métricas del Conjunto", style={'marginBottom': '15px'}),
                html.P(f"Total de modelos: {len(modelos_validos)}", 
                      style={'fontSize': '16px', 'fontWeight': 'bold'}),
                html.Div(stats_content, style={
                    'height': 'calc(100% - 80px)',
                    'overflowY': 'auto'
                })
            ], style={'padding': '20px', 'height': '100%', 'boxSizing': 'border-box'})
        
        return html.P("Pestaña no reconocida", style={'textAlign': 'center', 'color': 'red'})
    
    # 7. Callback principal del gráfico
    @app.callback(
        [Output('main-plot', 'figure'),
         Output('summary-table-container', 'children')],
        [Input('update-button', 'n_clicks'),
         Input('aeronave-dropdown', 'value'),
         Input('parametro-dropdown', 'value'),
         Input('predictor-dropdown', 'value'),
         Input('tipo-modelo-checklist', 'value'),
         Input('show-training-points', 'value'),
         Input('show-model-curves', 'value'),
         Input('show-only-real-curves', 'value'),
         Input('hide-plot-legend', 'value'),
         Input('imputation-methods-checklist', 'value')],
        [State('models-data-store', 'data')],
        prevent_initial_call=False
    )
    def update_main_plot(n_clicks, aeronave, parametro, predictor, tipos_modelo, 
                        show_training, show_curves, only_real_curves, hide_legend, 
                        imputation_methods, models_data):
        
        if not aeronave or not parametro or not models_data:
            empty_fig = go.Figure()
            empty_fig.add_annotation(
                text="Seleccione aeronave y parámetro",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=16, color="gray")
            )
            return empty_fig, html.P("Sin datos")

        # Filtro de predictores
        if predictor == '__all__':
            predictores = None
        elif predictor:
            predictores = [predictor]
        else:
            predictores = None

        modelos_filtrados = filter_models(
            models_data['modelos'],
            aeronave=aeronave,
            parametro=parametro,
            tipos_modelo=tipos_modelo,
            predictores=predictores
        )

        # Crear gráfico principal
        show_training_points = 'show' in (show_training or [])
        show_model_curves = 'show' in (show_curves or [])
        show_only_real = 'only_real' in (only_real_curves or [])
        
        fig = create_interactive_plot(
            modelos_filtrados,
            aeronave,
            parametro,
            show_training_points=show_training_points,
            show_model_curves=show_model_curves,
            show_only_real_curves=show_only_real,
            detalles_por_celda=models_data.get('detalles'),
            selected_imputation_methods=imputation_methods or ['final', 'similitud', 'correlacion']
        )
        
        # Configurar layout adicional
        fig.update_layout(
            showlegend=('hide' not in (hide_legend or []))
        )

        # Crear tabla resumen
        df_summary = create_metrics_summary_table(modelos_filtrados, aeronave, parametro)
        summary_table = create_summary_table(df_summary) if not df_summary.empty else html.P("Sin datos")
        
        return fig, summary_table
    
    # 8. Callback para panel de información
    @app.callback(
        Output('model-info-content', 'children'),
        [Input('main-plot', 'hoverData'),
         Input('main-plot', 'clickData'),
         Input('aeronave-dropdown', 'value'),
         Input('parametro-dropdown', 'value'),
         Input('predictor-dropdown', 'value'),
         Input('tipo-modelo-checklist', 'value')],
        [State('models-data-store', 'data')]
    )
    def update_info_panel(hoverData, clickData, aeronave, parametro, predictor, 
                         tipos_modelo, models_data):
        if not aeronave or not parametro or not models_data:
            return html.P("Seleccione una combinación válida para ver información del modelo.")
        
        # Aplicar los mismos filtros que en el callback principal
        if predictor == '__all__':
            predictores = None
        elif predictor:
            predictores = [predictor]
        else:
            predictores = None

        modelos_filtrados = filter_models(
            models_data['modelos'],
            aeronave=aeronave,
            parametro=parametro,
            tipos_modelo=tipos_modelo,
            predictores=predictores
        )
        
        celda_key = f"{aeronave}|{parametro}"
        modelos_celda = modelos_filtrados.get(celda_key, [])

        selected_model = None
        
        # Click en gráfica
        if clickData and 'points' in clickData and clickData['points']:
            idx = clickData['points'][0].get('curveNumber')
            if idx is not None and idx < len(modelos_celda):
                selected_model = modelos_celda[idx]
        # Hover en gráfica
        elif hoverData and 'points' in hoverData and hoverData['points']:
            idx = hoverData['points'][0].get('curveNumber')
            if idx is not None and idx < len(modelos_celda):
                selected_model = modelos_celda[idx]
        
        # Si no hay selección, mostrar el mejor modelo
        if not selected_model and modelos_celda:
            def confianza_promedio(m):
                if not isinstance(m, dict):
                    return 0
                c1 = m.get('Confianza', 0) or 0
                c2 = m.get('Confianza_LOOCV', 0) or 0
                return (c1 + c2) / 2
            selected_model = max(modelos_celda, key=confianza_promedio)
        
        if selected_model:
            return format_model_info(selected_model)
        return html.P("No hay información disponible para el modelo seleccionado.")
    
    # Ejecutar aplicación
    print(f"🚀 Iniciando aplicación CORREGIDA en http://localhost:{port}")
    print("📋 Funcionalidades incluidas:")
    print("   • Filtros básicos y avanzados")
    print("   • Gráfico principal interactivo")
    print("   • Panel de información de modelos")
    print("   • Tabla de resumen")
    print("   • Toggle de filtros")
    print("   • Vista de métricas")
    print("⚡ Presione Ctrl+C para detener la aplicación")
    print("="*60)
    
    try:
        app.run_server(debug=debug, port=port, host='127.0.0.1')
    except KeyboardInterrupt:
        print("\nAplicación detenida por el usuario")
    except Exception as e:
        print(f"Error ejecutando aplicación Dash: {e}")


if __name__ == "__main__":
    # Permitir ejecución directa del módulo
    import argparse
    
    parser = argparse.ArgumentParser(description='Análisis de Modelos de Imputación - VERSIÓN CORREGIDA')
    parser.add_argument('--json-path', type=str, help='Ruta al archivo JSON')
    parser.add_argument('--no-dash', action='store_true', help='No usar Dash')
    parser.add_argument('--port', type=int, default=8050, help='Puerto para Dash')
    parser.add_argument('--debug', action='store_true', help='Modo debug')
    
    args = parser.parse_args()
    
    main_visualizacion_modelos_fixed(
        json_path=args.json_path,
        use_dash=not args.no_dash,
        port=args.port,
        debug=args.debug
    )
