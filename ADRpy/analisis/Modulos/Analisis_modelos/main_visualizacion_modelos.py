"""
Aplicación Principal de Visualización de Modelos
===============================================

Este módulo contiene la función principal que ejecuta la aplicación 
interactiva de análisis de modelos usando Dash.

Función principal:
- main_visualizacion_modelos: Ejecuta la aplicación interactiva
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
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('TkAgg')  # Backend para Windows

from .data_loader import (
    load_models_data, 
    extract_unique_values, 
    filter_models,
    get_parametros_for_aeronave
)

if DASH_AVAILABLE:
    from .plot_utils import create_interactive_plot, create_comparison_plot, create_metrics_summary_table
    from .ui_components import (
        create_main_layout,
        create_aeronave_dropdown,
        create_parametro_dropdown,
        create_tipo_modelo_checklist,
        create_n_predictores_checklist,
        create_visualization_options,
        create_summary_table,
        format_model_info
    )

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main_visualizacion_modelos(json_path: Optional[str] = None, 
                             use_dash: bool = True,
                             port: int = 8050,
                             debug: bool = False) -> None:
    """
    Función principal para ejecutar la visualización de modelos.
    
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
            _run_dash_app(modelos_por_celda, detalles_por_celda, unique_values, port, debug)
        else:
            _run_matplotlib_app(modelos_por_celda, detalles_por_celda, unique_values)
            
    except Exception as e:
        logger.error(f"Error en la aplicación: {e}")
        print(f"Error ejecutando la aplicación: {e}")


def _run_dash_app(modelos_por_celda, detalles_por_celda, unique_values, port, debug):
    """Ejecuta la aplicación con Dash."""
    
    # Crear aplicación Dash
    app = dash.Dash(__name__)
    app.title = "Análisis de Modelos de Imputación"
    
    # Layout principal
    app.layout = create_main_layout()
    
    # Callback para actualizar dropdown de parámetros basado en aeronave
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
    
    # Callback para inicializar controles
    @app.callback(
        [Output('aeronave-dropdown-container', 'children'),
         Output('tipo-modelo-container', 'children'),
         Output('n-predictores-container', 'children'),
         Output('visualization-options-container', 'children'),
         Output('models-data-store', 'data'),
         Output('unique-values-store', 'data')],
        [Input('update-button', 'id')]  # Dummy input para trigger inicial
    )
    def initialize_controls(_):
        return (
            create_aeronave_dropdown(unique_values['aeronaves']),
            create_tipo_modelo_checklist(unique_values['tipos_modelo']),
            create_n_predictores_checklist(unique_values['n_predictores']),
            create_visualization_options(),
            {'modelos': modelos_por_celda, 'detalles': detalles_por_celda},
            unique_values
        )
    
    # Callback principal para actualizar gráfico
    @app.callback(
        [Output('main-plot', 'figure'),
         Output('summary-table-container', 'children'),
         Output('model-info-content', 'children')],
        [Input('update-button', 'n_clicks'),
         Input('aeronave-dropdown', 'value'),
         Input('parametro-dropdown', 'value')],
        [State('tipo-modelo-checklist', 'value'),
         State('n-predictores-checklist', 'value'),
         State('show-training-points', 'value'),
         State('show-model-curves', 'value'),
         State('models-data-store', 'data')]
    )
    def update_main_plot(n_clicks, aeronave, parametro, tipos_modelo, n_predictores,
                        show_training, show_curves, models_data):
        
        if not aeronave or not parametro or not models_data:
            empty_fig = go.Figure()
            empty_fig.add_annotation(
                text="Seleccione aeronave y parámetro",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=16, color="gray")
            )
            return empty_fig, html.P("Sin datos"), html.P("Sin información")
        
        # Filtrar modelos
        modelos_filtrados = filter_models(
            models_data['modelos'],
            aeronave=aeronave,
            parametro=parametro,
            tipos_modelo=tipos_modelo,
            n_predictores=n_predictores
        )
        
        # Crear gráfico principal
        show_training_points = 'show' in (show_training or [])
        show_model_curves = 'show' in (show_curves or [])
        
        fig = create_interactive_plot(
            modelos_filtrados,
            aeronave,
            parametro,
            show_training_points=show_training_points,
            show_model_curves=show_model_curves
        )
        
        # Crear tabla resumen
        df_summary = create_metrics_summary_table(modelos_filtrados, aeronave, parametro)
        summary_table = create_summary_table(df_summary) if not df_summary.empty else html.P("Sin datos")
        
        # Información del primer modelo
        celda_key = f"{aeronave}|{parametro}"
        model_info = html.P("Sin información disponible")
        
        if celda_key in modelos_filtrados and modelos_filtrados[celda_key]:
            primer_modelo = modelos_filtrados[celda_key][0]
            model_info = format_model_info(primer_modelo)
        
        return fig, summary_table, model_info
    
    # Ejecutar aplicación
    print(f"Iniciando aplicación Dash en http://localhost:{port}")
    print("Presione Ctrl+C para detener la aplicación")
    
    try:
        app.run_server(debug=debug, port=port, host='127.0.0.1')
    except KeyboardInterrupt:
        print("\nAplicación detenida por el usuario")
    except Exception as e:
        print(f"Error ejecutando aplicación Dash: {e}")


def _run_matplotlib_app(modelos_por_celda, detalles_por_celda, unique_values):
    """Ejecuta una versión simplificada con matplotlib."""
    
    print("Dash no está disponible. Usando visualización simplificada con matplotlib.")
    print(f"Aeronaves disponibles: {unique_values['aeronaves']}")
    print(f"Tipos de modelo disponibles: {unique_values['tipos_modelo']}")
    
    # Interfaz simplificada por consola
    while True:
        print("\n" + "="*50)
        print("ANÁLISIS DE MODELOS DE IMPUTACIÓN")
        print("="*50)
        
        # Seleccionar aeronave
        print(f"\nAeronaves disponibles: {', '.join(unique_values['aeronaves'])}")
        aeronave = input("Seleccione una aeronave (o 'quit' para salir): ").strip()
        
        if aeronave.lower() == 'quit':
            break
            
        if aeronave not in unique_values['aeronaves']:
            print("Aeronave no válida.")
            continue
        
        # Seleccionar parámetro
        parametros = get_parametros_for_aeronave(modelos_por_celda, aeronave)
        print(f"\nParámetros disponibles para {aeronave}: {', '.join(parametros)}")
        parametro = input("Seleccione un parámetro: ").strip()
        
        if parametro not in parametros:
            print("Parámetro no válido.")
            continue
        
        # Mostrar información de modelos
        celda_key = f"{aeronave}|{parametro}"
        if celda_key in modelos_por_celda:
            modelos = modelos_por_celda[celda_key]
            print(f"\n--- Modelos para {aeronave} - {parametro} ---")
            
            for i, modelo in enumerate(modelos):
                if isinstance(modelo, dict):
                    print(f"\nModelo {i+1}:")
                    print(f"  Tipo: {modelo.get('tipo', 'N/A')}")
                    print(f"  Predictores: {', '.join(modelo.get('predictores', []))}")
                    print(f"  MAPE: {modelo.get('mape', 0):.3f}%")
                    print(f"  R²: {modelo.get('r2', 0):.3f}")
                    print(f"  Confianza: {modelo.get('Confianza', 0):.3f}")
                    
                    ecuacion = modelo.get('ecuacion_string', '')
                    if ecuacion:
                        print(f"  Ecuación: {ecuacion}")
        else:
            print(f"No se encontraron modelos para {aeronave} - {parametro}")
        
        input("\nPresione Enter para continuar...")


if __name__ == "__main__":
    # Permitir ejecución directa del módulo
    import argparse
    
    parser = argparse.ArgumentParser(description='Análisis de Modelos de Imputación')
    parser.add_argument('--json-path', type=str, help='Ruta al archivo JSON')
    parser.add_argument('--no-dash', action='store_true', help='No usar Dash, usar matplotlib')
    parser.add_argument('--port', type=int, default=8050, help='Puerto para Dash')
    parser.add_argument('--debug', action='store_true', help='Modo debug')
    
    args = parser.parse_args()
    
    main_visualizacion_modelos(
        json_path=args.json_path,
        use_dash=not args.no_dash,
        port=args.port,
        debug=args.debug
    )
