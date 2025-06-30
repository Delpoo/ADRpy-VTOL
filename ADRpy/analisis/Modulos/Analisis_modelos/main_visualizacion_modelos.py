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
    create_predictor_dropdown,
    create_imputation_methods_checklist,
    create_comparison_type_radioitems
)

from .plot_stability import (
    get_stable_plot_config,
    apply_stable_configuration,
    should_preserve_zoom
)

from .plot_3d import (
    create_3d_plot,
    filter_models_for_3d
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
    
    # Callback para inicializar controles principales
    @app.callback(
        [Output('aeronave-dropdown-container', 'children'),
         Output('tipo-modelo-container', 'children'),
         Output('visualization-options-container', 'children'),
         Output('imputation-methods-container', 'children'),
         Output('comparison-type-container', 'children'),
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
            create_comparison_type_radioitems(),
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
        return create_predictor_dropdown(sorted(all_preds))

    # Callback para el panel de información usando los gráficos separados
    @app.callback(
        Output('model-info-content', 'children'),
        [Input('plot-2d', 'hoverData'),
         Input('plot-2d', 'clickData'),
         Input('plot-3d', 'hoverData'),
         Input('plot-3d', 'clickData'),
         Input('unified-summary-table-container', 'children'),  # Para detectar cambios en tabla
         Input('aeronave-dropdown', 'value'),
         Input('parametro-dropdown', 'value')],
        [State('filtered-models-2d-store', 'data'),
         State('filtered-models-3d-store', 'data'),
         State('active-view-store', 'data'),
         State('selected-model-store', 'data')],
        prevent_initial_call=False
    )
    def update_info_panel(hover_2d, click_2d, hover_3d, click_3d, table_children, 
                          aeronave, parametro, modelos_2d, modelos_3d, active_view, selected_model_data):
        """
        Actualiza el panel de información basado en interacciones con gráficos 2D/3D o tabla.
        """
        if not aeronave or not parametro:
            return html.P("Seleccione aeronave y parámetro para ver información del modelo.")

        selected_model = None
        celda_key = f"{aeronave}|{parametro}"
        
        # Obtener modelos según la vista activa
        if active_view == '3d-view':
            modelos_activos = modelos_3d.get(celda_key, []) if modelos_3d else []
        else:
            modelos_activos = modelos_2d.get(celda_key, []) if modelos_2d else []

        # Prioridad 1: Modelo seleccionado en el store
        if selected_model_data and isinstance(selected_model_data, dict):
            stored_aeronave = selected_model_data.get('aeronave')
            stored_parametro = selected_model_data.get('parametro')
            stored_idx = selected_model_data.get('model_idx')
            
            if (stored_aeronave == aeronave and stored_parametro == parametro and 
                stored_idx is not None and 0 <= stored_idx < len(modelos_activos)):
                selected_model = modelos_activos[stored_idx]

        # Prioridad 2: Click en gráficos
        if not selected_model:
            if active_view == '3d-view' and click_3d and 'points' in click_3d and click_3d['points']:
                idx = click_3d['points'][0].get('curveNumber')
                if idx is not None and idx < len(modelos_activos):
                    selected_model = modelos_activos[idx]
            elif active_view == '2d-view' and click_2d and 'points' in click_2d and click_2d['points']:
                idx = click_2d['points'][0].get('curveNumber')
                if idx is not None and idx < len(modelos_activos):
                    selected_model = modelos_activos[idx]

        # Prioridad 3: Hover en gráficos
        if not selected_model:
            if active_view == '3d-view' and hover_3d and 'points' in hover_3d and hover_3d['points']:
                idx = hover_3d['points'][0].get('curveNumber')
                if idx is not None and idx < len(modelos_activos):
                    selected_model = modelos_activos[idx]
            elif active_view == '2d-view' and hover_2d and 'points' in hover_2d and hover_2d['points']:
                idx = hover_2d['points'][0].get('curveNumber')
                if idx is not None and idx < len(modelos_activos):
                    selected_model = modelos_activos[idx]

        # Si no hay selección, mostrar el mejor modelo por confianza promedio
        if not selected_model and modelos_activos:
            def confianza_promedio(m):
                if not isinstance(m, dict):
                    return 0
                c1 = m.get('Confianza', 0) or 0
                c2 = m.get('Confianza_LOOCV', 0) or 0
                return (c1 + c2) / 2
            selected_model = max(modelos_activos, key=confianza_promedio)

        if selected_model:
            return format_model_info(selected_model)
        return html.P("No hay información disponible para el modelo seleccionado.")

    # ===== CALLBACK PRINCIPAL: ALTERNANCIA DE VISTAS =====
    @app.callback(
        [Output('2d-plot-container', 'style'),
         Output('3d-plot-container', 'style'),
         Output('active-view-store', 'data'),
         Output('active-view-indicator', 'children')],
        [Input('view-tabs', 'value')],
        prevent_initial_call=False
    )
    def toggle_view_containers(active_tab):
        """
        Controla la alternancia entre vistas 2D y 3D.
        Solo una vista es visible a la vez.
        """
        # Estilos base para contenedores
        hidden_style = {'display': 'none'}
        visible_style = {'display': 'block'}
        
        # Nombres de vista para indicador
        view_names = {
            '2d-view': '📊 Vista 2D (1 Predictor)',
            '3d-view': '🧊 Vista 3D (2 Predictores)',
            'comparison-view': '📈 Comparación',
            'metrics-view': '📋 Métricas'
        }
        
        if active_tab == '2d-view':
            return (
                visible_style,    # 2D visible
                hidden_style,     # 3D oculto
                '2d-view',        # Store vista activa
                view_names['2d-view']  # Indicador
            )
        elif active_tab == '3d-view':
            return (
                hidden_style,     # 2D oculto
                visible_style,    # 3D visible
                '3d-view',        # Store vista activa
                view_names['3d-view']  # Indicador
            )
        else:  # comparison-view, metrics-view - por defecto mostrar 2D
            return (
                visible_style,    # 2D visible
                hidden_style,     # 3D oculto
                '2d-view',        # Store vista activa por defecto
                view_names.get('2d-view', 'Vista 2D')  # Indicador por defecto
            )

    # ===== CALLBACK PRINCIPAL: FILTRADO INTELIGENTE Y ACTUALIZACIÓN DE MODELOS =====
    @app.callback(
        [Output('filtered-models-2d-store', 'data'),
         Output('filtered-models-3d-store', 'data'),
         Output('models-2d-count', 'children'),
         Output('models-3d-count', 'children'),
         Output('total-models-count', 'children'),
         Output('filter-state-store', 'data')],
        [Input('aeronave-dropdown', 'value'),
         Input('parametro-dropdown', 'value'),
         Input('tipo-modelo-checklist', 'value'),
         Input('predictor-dropdown', 'value'),
         Input('show-training-points', 'value'),
         Input('show-model-curves', 'value'),
         Input('show-only-real-curves', 'value'),
         Input('imputation-methods-checklist', 'value'),
         Input('comparison-type', 'value')],
        [State('models-data-store', 'data'),
         State('active-view-store', 'data')],
        prevent_initial_call=False
    )
    def update_filtered_models(aeronave, parametro, tipos_modelo, predictor, 
                             show_training, show_curves, only_real_curves, 
                             imputation_methods, comparison_type, 
                             models_data, active_view):
        """
        Filtrado inteligente de modelos según vista activa y filtros globales.
        
        Lógica:
        - Filtros globales (aeronave, parámetro, tipo_modelo): Afectan AMBAS vistas
        - Filtros específicos (predictor): Solo afectan vista activa
        - Vista 2D: Solo modelos de 1 predictor + LOOCV > 0
        - Vista 3D: Solo modelos de 2 predictores + sin filtro LOOCV
        """
        
        if not aeronave or not parametro or not models_data:
            return {}, {}, "0", "0", "0", {}
        
        # Estado actual de filtros
        filter_state = {
            'aeronave': aeronave,
            'parametro': parametro,
            'tipos_modelo': tipos_modelo or [],
            'predictor': predictor,
            'active_view': active_view,
            'show_training': show_training,
            'show_curves': show_curves,
            'only_real_curves': only_real_curves,
            'imputation_methods': imputation_methods or ['final', 'similitud', 'correlacion'],
            'comparison_type': comparison_type
        }
        
        # Filtro de predictores (específico de vista)
        if predictor == '__all__' or not predictor:
            predictores = None
        else:
            predictores = [predictor]
        
        try:
            # === FILTRADO PARA VISTA 2D ===
            modelos_2d = filter_models(
                models_data['modelos'],
                aeronave=aeronave,
                parametro=parametro,
                tipos_modelo=tipos_modelo,
                predictores=predictores if active_view == '2d-view' else None,  # Solo aplicar si vista activa
                comparison_type=comparison_type,
                exclude_2pred_from_2d=True,  # Excluir modelos de 2 predictores
                require_loocv=True  # Requerir LOOCV > 0
            )
            
            # === FILTRADO PARA VISTA 3D ===
            modelos_3d_temp = filter_models(
                models_data['modelos'],
                aeronave=aeronave,
                parametro=parametro,
                tipos_modelo=tipos_modelo,
                predictores=predictores if active_view == '3d-view' else None,  # Solo aplicar si vista activa
                comparison_type=comparison_type,
                exclude_2pred_from_2d=False,  # No excluir
                require_loocv=False  # No requerir LOOCV
            )
            
            # Filtrar solo modelos de 2 predictores para 3D
            celda_key = f"{aeronave}|{parametro}"
            if isinstance(modelos_3d_temp, dict) and celda_key in modelos_3d_temp:
                modelos_lista_3d = modelos_3d_temp[celda_key]
                modelos_3d_filtrados = [
                    m for m in modelos_lista_3d 
                    if isinstance(m, dict) and len(m.get('predictores', [])) == 2
                ]
                modelos_3d = {celda_key: modelos_3d_filtrados}
            else:
                modelos_3d = {}
            
            # Contar modelos
            count_2d = len(modelos_2d.get(celda_key, [])) if isinstance(modelos_2d, dict) else 0
            count_3d = len(modelos_3d.get(celda_key, [])) if isinstance(modelos_3d, dict) else 0
            count_total = count_2d + count_3d
            
            logger.info(f"[FILTRADO] 2D: {count_2d}, 3D: {count_3d}, Total: {count_total}")
            
            return (
                modelos_2d,
                modelos_3d,
                str(count_2d),
                str(count_3d),
                str(count_total),
                filter_state
            )
            
        except Exception as e:
            logger.error(f"Error en filtrado de modelos: {e}")
            return {}, {}, "Error", "Error", "Error", filter_state

    # ===== CALLBACK: ACTUALIZACIÓN GRÁFICO 2D =====
    @app.callback(
        Output('plot-2d', 'figure'),
        [Input('filtered-models-2d-store', 'data'),
         Input('selected-model-store', 'data'),
         Input('show-training-points', 'value'),
         Input('show-model-curves', 'value'),
         Input('show-only-real-curves', 'value'),
         Input('hide-plot-legend', 'value'),
         Input('imputation-methods-checklist', 'value')],
        [State('aeronave-dropdown', 'value'),
         State('parametro-dropdown', 'value'),
         State('models-data-store', 'data')],
        prevent_initial_call=False
    )
    def update_2d_plot(modelos_2d, selected_model_data, show_training, show_curves, 
                       only_real_curves, hide_legend, imputation_methods,
                       aeronave, parametro, models_data):
        """
        Actualiza el gráfico 2D usando los modelos filtrados del store.
        """
        if not aeronave or not parametro or not modelos_2d:
            empty_fig = go.Figure()
            empty_fig.add_annotation(
                text="Seleccione aeronave y parámetro para ver la vista 2D",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=16, color="gray")
            )
            return empty_fig

        # Determinar modelo resaltado
        highlight_idx = None
        if selected_model_data and isinstance(selected_model_data, dict):
            stored_aeronave = selected_model_data.get('aeronave')
            stored_parametro = selected_model_data.get('parametro')
            stored_idx = selected_model_data.get('model_idx')
            
            if (stored_aeronave == aeronave and stored_parametro == parametro and 
                stored_idx is not None):
                celda_key = f"{aeronave}|{parametro}"
                modelos_lista = modelos_2d.get(celda_key, [])
                if 0 <= stored_idx < len(modelos_lista):
                    highlight_idx = stored_idx

        # Crear gráfico 2D
        show_training_points = 'show' in (show_training or [])
        show_model_curves = 'show' in (show_curves or [])
        show_only_real = 'only_real' in (only_real_curves or [])
        
        fig = create_interactive_plot(
            modelos_2d,
            aeronave,
            parametro,
            show_training_points=show_training_points,
            show_model_curves=show_model_curves,
            show_only_real_curves=show_only_real,
            highlight_model_idx=highlight_idx,
            detalles_por_celda=models_data.get('detalles') if models_data else None,
            selected_imputation_methods=imputation_methods or ['final', 'similitud', 'correlacion']
        )
        
        # Aplicar configuración estable y ocultar leyenda si es necesario
        fig = apply_stable_configuration(fig, aeronave, parametro, preserve_zoom=True)
        fig.update_layout(showlegend=('hide' not in (hide_legend or [])))
        
        return fig

    # ===== CALLBACK: ACTUALIZACIÓN GRÁFICO 3D =====
    @app.callback(
        Output('plot-3d', 'figure'),
        [Input('filtered-models-3d-store', 'data'),
         Input('selected-model-store', 'data'),
         Input('show-training-points', 'value'),
         Input('show-model-curves', 'value')],
        [State('aeronave-dropdown', 'value'),
         State('parametro-dropdown', 'value'),
         State('active-view-store', 'data')],
        prevent_initial_call=False
    )
    def update_3d_plot(modelos_3d, selected_model_data, show_training, show_curves,
                       aeronave, parametro, active_view):
        """
        Actualiza el gráfico 3D usando los modelos filtrados del store.
        """
        if not aeronave or not parametro or not modelos_3d:
            empty_fig = go.Figure()
            empty_fig.add_annotation(
                text="Seleccione aeronave y parámetro para ver la vista 3D",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=16, color="gray")
            )
            return empty_fig

        # Obtener lista de modelos 3D
        celda_key = f"{aeronave}|{parametro}"
        modelos_lista_3d = modelos_3d.get(celda_key, [])
        
        if not modelos_lista_3d:
            empty_fig = go.Figure()
            empty_fig.add_annotation(
                text=f"No hay modelos de 2 predictores para {aeronave} - {parametro}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=16, color="orange")
            )
            return empty_fig

        # Determinar modelo resaltado para 3D
        highlight_idx = None
        if selected_model_data and isinstance(selected_model_data, dict):
            stored_aeronave = selected_model_data.get('aeronave')
            stored_parametro = selected_model_data.get('parametro')
            stored_idx = selected_model_data.get('model_idx')
            
            if (stored_aeronave == aeronave and stored_parametro == parametro and 
                stored_idx is not None and 0 <= stored_idx < len(modelos_lista_3d)):
                highlight_idx = stored_idx

        # Crear gráfico 3D
        show_training_points = 'show' in (show_training or [])
        show_model_surface = 'show' in (show_curves or [])
        
        fig_3d = create_3d_plot(
            modelos_lista_3d,
            aeronave,
            parametro,
            show_training_points=show_training_points,
            show_model_surface=show_model_surface,
            highlight_model_idx=highlight_idx
        )
        
        return fig_3d

    # ===== CALLBACK: TABLA DE RESUMEN UNIFICADA =====
    @app.callback(
        Output('unified-summary-table-container', 'children'),
        [Input('filtered-models-2d-store', 'data'),
         Input('filtered-models-3d-store', 'data'),
         Input('summary-filter', 'value'),
         Input('selected-model-store', 'data')],
        [State('aeronave-dropdown', 'value'),
         State('parametro-dropdown', 'value'),
         State('active-view-store', 'data')],
        prevent_initial_call=False
    )
    def update_unified_summary_table(modelos_2d, modelos_3d, summary_filter, 
                                   selected_model_data, aeronave, parametro, active_view):
        """
        Actualiza la tabla de resumen unificada mostrando modelos 2D y/o 3D según el filtro.
        """
        if not aeronave or not parametro:
            return html.P("Seleccione aeronave y parámetro para ver el resumen.")

        celda_key = f"{aeronave}|{parametro}"
        
        # Obtener modelos según el filtro
        modelos_para_tabla = []
        
        if summary_filter == 'all':
            # Mostrar todos los modelos (2D + 3D)
            modelos_2d_lista = modelos_2d.get(celda_key, []) if modelos_2d else []
            modelos_3d_lista = modelos_3d.get(celda_key, []) if modelos_3d else []
            
            # Agregar etiquetas para distinguir
            for modelo in modelos_2d_lista:
                if isinstance(modelo, dict):
                    modelo_copia = modelo.copy()
                    modelo_copia['vista_origen'] = '2D'
                    modelos_para_tabla.append(modelo_copia)
            
            for modelo in modelos_3d_lista:
                if isinstance(modelo, dict):
                    modelo_copia = modelo.copy()
                    modelo_copia['vista_origen'] = '3D'
                    modelos_para_tabla.append(modelo_copia)
                    
        elif summary_filter == '2d':
            # Solo modelos 2D
            modelos_2d_lista = modelos_2d.get(celda_key, []) if modelos_2d else []
            for modelo in modelos_2d_lista:
                if isinstance(modelo, dict):
                    modelo_copia = modelo.copy()
                    modelo_copia['vista_origen'] = '2D'
                    modelos_para_tabla.append(modelo_copia)
                    
        elif summary_filter == '3d':
            # Solo modelos 3D
            modelos_3d_lista = modelos_3d.get(celda_key, []) if modelos_3d else []
            for modelo in modelos_3d_lista:
                if isinstance(modelo, dict):
                    modelo_copia = modelo.copy()
                    modelo_copia['vista_origen'] = '3D'
                    modelos_para_tabla.append(modelo_copia)
                    
        elif summary_filter == 'active':
            # Solo vista activa
            if active_view == '3d-view':
                modelos_3d_lista = modelos_3d.get(celda_key, []) if modelos_3d else []
                for modelo in modelos_3d_lista:
                    if isinstance(modelo, dict):
                        modelo_copia = modelo.copy()
                        modelo_copia['vista_origen'] = '3D'
                        modelos_para_tabla.append(modelo_copia)
            else:
                modelos_2d_lista = modelos_2d.get(celda_key, []) if modelos_2d else []
                for modelo in modelos_2d_lista:
                    if isinstance(modelo, dict):
                        modelo_copia = modelo.copy()
                        modelo_copia['vista_origen'] = '2D'
                        modelos_para_tabla.append(modelo_copia)

        if not modelos_para_tabla:
            return html.P("No hay modelos disponibles para mostrar.")

        # Crear DataFrame para la tabla
        df_summary = create_metrics_summary_table({celda_key: modelos_para_tabla}, aeronave, parametro)
        
        # Determinar fila seleccionada
        selected_row_idx = None
        if selected_model_data and isinstance(selected_model_data, dict):
            stored_aeronave = selected_model_data.get('aeronave')
            stored_parametro = selected_model_data.get('parametro')
            stored_idx = selected_model_data.get('model_idx')
            
            if (stored_aeronave == aeronave and stored_parametro == parametro and 
                stored_idx is not None and 0 <= stored_idx < len(modelos_para_tabla)):
                selected_row_idx = stored_idx

        return create_summary_table(df_summary, selected_row_idx)

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
