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
    # Crear dummies para evitar errores de importación
    dash = None
    dcc = None
    html = None
    dash_table = None
    Input = None
    Output = None
    State = None
    go = None
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('TkAgg')  # Backend para Windows

# Importar pandas por separado (siempre necesario)
import pandas as pd

from .data_loader import (
    load_models_data, 
    extract_unique_values, 
    filter_models,
    get_parametros_for_aeronave
)

from .ui_components import (
    create_main_layout,
    create_aeronave_dropdown,
    create_parametro_dropdown,
    create_tipo_modelo_checklist,
    create_visualization_options,
    create_summary_table,
    format_model_info,
    create_predictor_dropdown
)

from .plot_stability import (
    get_stable_plot_config,
    apply_stable_configuration,
    should_preserve_zoom
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
    if not DASH_AVAILABLE:
        print("Dash no está disponible. No se puede ejecutar la aplicación interactiva.")
        return

    # Re-importar los componentes de Dash dentro de la función para evitar errores
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
    
    # Callback para inicializar controles (sin cambios)
    @app.callback(
        [Output('aeronave-dropdown-container', 'children'),
         Output('tipo-modelo-container', 'children'),
         Output('visualization-options-container', 'children'),
         Output('models-data-store', 'data'),
         Output('unique-values-store', 'data')],
        [Input('update-button', 'id')]
    )
    def initialize_controls(_):
        return (
            create_aeronave_dropdown(unique_values['aeronaves']),
            create_tipo_modelo_checklist(unique_values['tipos_modelo']),
            create_visualization_options(),
            {'modelos': modelos_por_celda, 'detalles': detalles_por_celda},
            unique_values
        )

    # Callback para actualizar el dropdown de predictores según el parámetro seleccionado
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
        # Obtener todos los predictores únicos de los modelos filtrados
        all_preds = set()
        for m in modelos:
            if isinstance(m, dict):
                all_preds.update(m.get('predictores', []))
        return create_predictor_dropdown(sorted(all_preds))    # Callback principal: actualiza gráfica y tabla resumen según TODOS los filtros
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
         Input('imputation-methods-checklist', 'value'),
         Input('comparison-type', 'value'),
         Input('selected-model-store', 'data')],  # Usar el store en lugar de selected_rows
        [State('models-data-store', 'data')],
        prevent_initial_call=False
    )
    def update_main_plot(n_clicks, aeronave, parametro, predictor, tipos_modelo, show_training, show_curves, only_real_curves, hide_legend, imputation_methods, comparison_type, selected_model_data, models_data):
        import copy
        ctx = dash.callback_context
        
        # Detectar si el trigger fue solo el selected-model-store
        triggered_only_selection = False
        if ctx.triggered:
            trigger_props = [t['prop_id'].split('.')[0] for t in ctx.triggered]
            if len(trigger_props) == 1 and trigger_props[0] == 'selected-model-store':
                triggered_only_selection = True
        
        logger.info(f"[DEBUG] Entrando a update_main_plot con aeronave={aeronave}, parametro={parametro}, predictor={predictor}")
        logger.info(f"[DEBUG] triggered_only_selection: {triggered_only_selection}")
        logger.info(f"[DEBUG] selected_model_data recibido: {selected_model_data}")
        
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
            predictores=predictores,
            comparison_type=comparison_type
        )
        celda_key = f"{aeronave}|{parametro}"
        modelos_celda = modelos_filtrados.get(celda_key, [])

        # Determinar modelo seleccionado desde el store
        highlight_idx = None
        logger.info(f"[DEBUG] selected_model_data: {selected_model_data}")
        logger.info(f"[DEBUG] modelos_celda (len={len(modelos_celda)}): {[m.get('tipo','?') if isinstance(m,dict) else str(m) for m in modelos_celda]}")
        if selected_model_data and isinstance(selected_model_data, dict):
            stored_aeronave = selected_model_data.get('aeronave')
            stored_parametro = selected_model_data.get('parametro')
            stored_idx = selected_model_data.get('model_idx')
            logger.info(f"[DEBUG] stored_aeronave={stored_aeronave}, stored_parametro={stored_parametro}, stored_idx={stored_idx}")
            # Solo aplicar si coincide la aeronave y parámetro actuales
            if (stored_aeronave == aeronave and stored_parametro == parametro and 
                stored_idx is not None and 0 <= stored_idx < len(modelos_celda)):
                highlight_idx = stored_idx
        logger.info(f"[DEBUG] highlight_idx usado en create_interactive_plot: {highlight_idx}")

        # Crear gráfico principal con resaltado
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
            highlight_model_idx=highlight_idx,
            detalles_por_celda=models_data.get('detalles') if models_data else None,
            selected_imputation_methods=imputation_methods or ['final', 'similitud', 'correlacion']
        )
        logger.info(f"[DEBUG] Llamada a create_interactive_plot con highlight_model_idx={highlight_idx}")
        
        # Determinar si preservar zoom basado en el trigger
        preserve_zoom = True
        if ctx.triggered:
            triggered_ids = [t['prop_id'].split('.')[0] for t in ctx.triggered]
            trigger_info = {'triggered_ids': triggered_ids}
            current_selection = {'aeronave': aeronave, 'parametro': parametro}
            preserve_zoom = should_preserve_zoom(trigger_info, current_selection)
        
        # Aplicar configuración estable
        fig = apply_stable_configuration(fig, aeronave, parametro, preserve_zoom)
        
        # Configurar layout adicional
        fig.update_layout(
            showlegend=('hide' not in (hide_legend or []))
        )

        # Crear tabla resumen con información de selección
        df_summary = create_metrics_summary_table(modelos_filtrados, aeronave, parametro)
        summary_table = create_summary_table(df_summary, highlight_idx) if not df_summary.empty else html.P("Sin datos")
        return fig, summary_table

    # Callback para el panel de información: hover/click, tabla seleccionada y filtros
    @app.callback(
        Output('model-info-content', 'children'),
        [Input('main-plot', 'hoverData'),
         Input('main-plot', 'clickData'),
         Input('summary-table', 'selected_rows'),  # Agregar selección de tabla
         Input('aeronave-dropdown', 'value'),
         Input('parametro-dropdown', 'value'),
         Input('predictor-dropdown', 'value'),
         Input('tipo-modelo-checklist', 'value'),
         Input('comparison-type', 'value'),         Input('models-data-store', 'data')]
    )
    def update_info_panel(hoverData, clickData, selected_rows, aeronave, parametro, predictor, tipos_modelo, comparison_type, models_data):
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
            predictores=predictores,
            comparison_type=comparison_type
        )
        
        celda_key = f"{aeronave}|{parametro}"
        modelos_celda = modelos_filtrados.get(celda_key, [])
        logger.info(f"[DEBUG] modelos_celda (len={len(modelos_celda)}): {[m.get('tipo','?') if isinstance(m,dict) else str(m) for m in modelos_celda]}")

        selected_model = None
        
        # Prioridad 1: Selección de tabla de resumen
        if selected_rows and len(selected_rows) > 0 and modelos_celda:
            selected_row_idx = selected_rows[0]
            if 0 <= selected_row_idx < len(modelos_celda):
                selected_model = modelos_celda[selected_row_idx]
        
        # Prioridad 2: Click en gráfica
        elif clickData and 'points' in clickData and clickData['points']:
            idx = clickData['points'][0].get('curveNumber')
            if idx is not None and idx < len(modelos_celda):
                selected_model = modelos_celda[idx]
          # Prioridad 3: Hover en gráfica
        elif hoverData and 'points' in hoverData and hoverData['points']:
            idx = hoverData['points'][0].get('curveNumber')
            if idx is not None and idx < len(modelos_celda):
                selected_model = modelos_celda[idx]
          # Si no hay selección, mostrar el mejor modelo por confianza promedio
        if not selected_model and modelos_celda:
            def confianza_promedio(m):
                if not isinstance(m, dict):
                    return 0
                c1 = m.get('Confianza', 0)
                c2 = m.get('Confianza_LOOCV', 0)
                # Manejar valores None de manera robusta
                if c1 is None:
                    c1 = 0
                if c2 is None:
                    c2 = 0
                return (c1 + c2) / 2
            selected_model = max(modelos_celda, key=confianza_promedio)
        
        if selected_model:
            return format_model_info(selected_model)
        return html.P("No hay información disponible para el modelo seleccionado.")
      # Callback adicional: sincronizar selección de tabla con resaltado en gráfica
    @app.callback(
        Output('selected-model-store', 'data'),
        [Input('summary-table', 'selected_rows'),
         Input('main-plot', 'clickData')],
        [State('aeronave-dropdown', 'value'),
         State('parametro-dropdown', 'value'),
         State('predictor-dropdown', 'value'),
         State('tipo-modelo-checklist', 'value'),
         State('comparison-type', 'value'),
         State('models-data-store', 'data'),
         State('selected-model-store', 'data')]
    )
    def sync_model_selection(selected_rows, clickData, aeronave, parametro, predictor, tipos_modelo, comparison_type, models_data, prev_store):
        """Sincroniza la selección entre tabla y gráfica"""
        if not aeronave or not parametro or not models_data:
            return prev_store
        
        ctx = dash.callback_context
        if not ctx.triggered:
            return prev_store
        
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        logger.info(f"[DEBUG] sync_model_selection: trigger_id={trigger_id}, selected_rows={selected_rows}, clickData={clickData}")
        
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
            predictores=predictores,
            comparison_type=comparison_type
        )
        celda_key = f"{aeronave}|{parametro}"
        modelos_celda = modelos_filtrados.get(celda_key, [])
        logger.info(f"[DEBUG] modelos_celda (len={len(modelos_celda)}): {[m.get('tipo','?') if isinstance(m,dict) else str(m) for m in modelos_celda]}")

        selected_idx = None
        if trigger_id == 'summary-table' and selected_rows:
            selected_idx = selected_rows[0]
        elif trigger_id == 'main-plot' and clickData:
            if clickData and 'points' in clickData and clickData['points']:
                point = clickData['points'][0]
                logger.info(f"[DEBUG] Click point data: {point}")
                
                # Intentar obtener el índice del modelo desde customdata
                if 'customdata' in point and point['customdata'] is not None:
                    selected_idx = point['customdata']
                    logger.info(f"[DEBUG] Modelo seleccionado por customdata: {selected_idx}")
                # Fallback: usar curveNumber si customdata no está disponible
                elif 'curveNumber' in point:
                    # Calcular el índice del modelo basado en curveNumber
                    # Cada modelo puede tener hasta 3 trazas (puntos originales, entrenamiento, curva)
                    curve_number = point['curveNumber']
                    logger.info(f"[DEBUG] curveNumber: {curve_number}, modelos disponibles: {len(modelos_celda)}")
                    
                    # Mapear curveNumber a índice de modelo
                    # Asumiendo que las trazas se agregan en orden: orig, train, curve para cada modelo
                    if curve_number < len(modelos_celda) * 3:  # Máximo 3 trazas por modelo
                        selected_idx = curve_number // 3
                        if selected_idx >= len(modelos_celda):
                            selected_idx = len(modelos_celda) - 1
                    logger.info(f"[DEBUG] Modelo inferido por curveNumber: {selected_idx}")
                
                # Validar que el índice esté dentro del rango
                if selected_idx is not None and (selected_idx < 0 or selected_idx >= len(modelos_celda)):
                    logger.warning(f"[DEBUG] Índice fuera de rango: {selected_idx}, ajustando...")
                    selected_idx = None
        else:
            # Si no hay trigger válido, mantener selección previa
            if prev_store and prev_store.get('aeronave') == aeronave and prev_store.get('parametro') == parametro:
                selected_idx = prev_store.get('model_idx')

        logger.info(f"[DEBUG] selected_idx calculado: {selected_idx}")
        return {
            'aeronave': aeronave,
            'parametro': parametro,
            'model_idx': selected_idx
        }
      # Callback para actualizar selected_rows cuando cambia el store
    @app.callback(
        Output('summary-table', 'selected_rows'),
        [Input('selected-model-store', 'data'),
         Input('aeronave-dropdown', 'value'),
         Input('parametro-dropdown', 'value')],
        prevent_initial_call=True
    )
    def update_table_selection(selected_model_data, aeronave, parametro):
        """Actualiza la selección de la tabla basada en el store"""
        if not selected_model_data or not aeronave or not parametro:
            return []
            
        stored_aeronave = selected_model_data.get('aeronave')
        stored_parametro = selected_model_data.get('parametro')
        stored_idx = selected_model_data.get('model_idx')
        
        # Solo aplicar si coincide la aeronave y parámetro actuales
        if (stored_aeronave == aeronave and stored_parametro == parametro and 
            stored_idx is not None):
            return [stored_idx]
        
        return []

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
