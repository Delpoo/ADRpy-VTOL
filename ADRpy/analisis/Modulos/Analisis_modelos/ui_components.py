"""
Componentes de Interfaz de Usuario Reutilizables
===============================================

Este módulo contiene componentes de interfaz reutilizables para 
la aplicación de análisis de modelos usando Dash.

Funciones principales:
- create_filter_controls: Crea controles de filtrado
- create_info_panel: Crea panel de información
- create_layout: Crea el layout principal de la aplicación
"""

from typing import List, Dict, Optional, Any
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import pandas as pd


def create_aeronave_dropdown(aeronaves: List[str], 
                           selected: Optional[str] = None) -> dcc.Dropdown:
    """
    Crea dropdown para selección de aeronave.
    
    Parameters:
    -----------
    aeronaves : List[str]
        Lista de aeronaves disponibles
    selected : Optional[str]
        Aeronave seleccionada por defecto
        
    Returns:
    --------
    dcc.Dropdown
        Componente dropdown de Dash
    """
    options = [{'label': aero, 'value': aero} for aero in aeronaves]
    
    return dcc.Dropdown(
        id='aeronave-dropdown',
        options=options,
        value=selected or (aeronaves[0] if aeronaves else None),
        placeholder="Seleccione una aeronave...",
        style={'marginBottom': '10px'}
    )


def create_parametro_dropdown(parametros: List[str], 
                            selected: Optional[str] = None) -> dcc.Dropdown:
    """
    Crea dropdown para selección de parámetro.
    
    Parameters:
    -----------
    parametros : List[str]
        Lista de parámetros disponibles
    selected : Optional[str]
        Parámetro seleccionado por defecto
        
    Returns:
    --------
    dcc.Dropdown
        Componente dropdown de Dash
    """
    options = [{'label': param, 'value': param} for param in parametros]
    
    return dcc.Dropdown(
        id='parametro-dropdown',
        options=options,
        value=selected or (parametros[0] if parametros else None),
        placeholder="Seleccione un parámetro...",
        style={'marginBottom': '10px'}
    )


def create_tipo_modelo_checklist(tipos_modelo: List[str]) -> dcc.Checklist:
    """
    Crea checklist para selección de tipos de modelo.
    
    Parameters:
    -----------
    tipos_modelo : List[str]
        Lista de tipos de modelo disponibles
        
    Returns:
    --------
    dcc.Checklist
        Componente checklist de Dash
    """
    options = [{'label': tipo, 'value': tipo} for tipo in tipos_modelo]
    
    return dcc.Checklist(
        id='tipo-modelo-checklist',
        options=options,
        value=tipos_modelo,  # Todos seleccionados por defecto
        style={'marginBottom': '10px'},
        inputStyle={"marginRight": "5px"}
    )


def create_predictor_dropdown(predictors: List[str], selected: Optional[str] = None) -> dcc.Dropdown:
    """
    Crea un dropdown para selección de predictor.
    
    Parameters:
    -----------
    predictors : List[str]
        Lista de predictores disponibles
    selected : Optional[str]
        Predictor seleccionado por defecto
        
    Returns:
    --------
    dcc.Dropdown
        Componente dropdown de Dash
    """
    options = [{'label': 'Todos los predictores', 'value': '__all__'}] + [
        {'label': pred, 'value': pred} for pred in predictors
    ]
    return dcc.Dropdown(
        id='predictor-dropdown',
        options=options,
        value=selected if selected is not None else '__all__',
        placeholder="Seleccione un predictor...",
        style={'marginBottom': '10px'}
    )


def create_visualization_options() -> html.Div:
    """
    Crea opciones de visualización.
    
    Returns:
    --------
    html.Div
        Div con opciones de visualización
    """
    return html.Div([
        html.H4("Opciones de Visualización", style={'marginTop': '20px'}),
        
        dcc.Checklist(
            id='show-training-points',
            options=[{'label': 'Mostrar puntos de entrenamiento', 'value': 'show'}],
            value=['show'],
            style={'marginBottom': '10px'},
            inputStyle={"marginRight": "5px"}
        ),
        
        dcc.Checklist(
            id='show-model-curves',
            options=[{'label': 'Mostrar curvas de modelos', 'value': 'show'}],
            value=['show'],
            style={'marginBottom': '10px'},
            inputStyle={"marginRight": "5px"}
        ),
        
        dcc.Checklist(
            id='show-only-real-curves',
            options=[{'label': 'Mostrar solo curvas con datos reales', 'value': 'only_real'}],
            value=[],
            style={'marginBottom': '10px'},
            inputStyle={"marginRight": "5px"}
        ),
          dcc.Checklist(
            id='hide-plot-legend',
            options=[{'label': 'Ocultar leyenda de la gráfica', 'value': 'hide'}],
            value=[],
            style={'marginBottom': '10px'},
            inputStyle={"marginRight": "5px"}
        ),
        
        html.H5("Métodos de Imputación", style={'marginTop': '15px', 'marginBottom': '5px'}),
        create_imputation_methods_checklist(),
        
        html.Div(id='predictor-checklist-container'),
        
        html.Label("Tipo de comparación:"),
        dcc.RadioItems(
            id='comparison-type',
            options=[
                {'label': 'Por tipo de modelo', 'value': 'by_type'},
                {'label': 'Mejores globales', 'value': 'best_overall'},
                {'label': 'Por número de predictores', 'value': 'by_predictors'}
            ],
            value='by_type',
            style={'marginBottom': '10px'}
        )
    ])


def create_info_panel() -> html.Div:
    """
    Crea panel de información lateral.
    
    Returns:
    --------
    html.Div
        Panel de información
    """
    return html.Div([
        html.H3("Información del Modelo", style={'marginBottom': '20px'}),
        
        html.Div(id='model-info-content', children=[
            html.P("Seleccione un modelo para ver información detallada.",
                  style={'color': 'gray', 'fontStyle': 'italic'})
        ]),
        
        html.Hr(),
        
        html.H4("Métricas de Comparación"),
        html.Div(id='metrics-comparison'),
        
        html.Hr(),
        
        html.H4("Detalles de Imputación"),
        html.Div(id='imputation-details')
        
    ], style={
        'padding': '20px',
        'backgroundColor': '#f8f9fa',
        'borderRadius': '5px',
        'margin': '10px'
    })


def create_main_layout() -> html.Div:
    """
    Crea el layout principal de la aplicación.
    
    Returns:
    --------
    html.Div
        Layout principal
    """
    return html.Div([
        html.H1("Análisis Interactivo de Modelos de Imputación", 
               style={'textAlign': 'center', 'marginBottom': '30px'}),
        html.Div([
            # Botón SIEMPRE visible, fuera del panel de filtros
            html.Button(
                id='toggle-filters-btn',
                children='Ocultar/Mostrar Filtros',
                n_clicks=0,
                style={
                    'position': 'absolute',
                    'top': '20px',
                    'left': '20px',
                    'zIndex': 10,
                    'padding': '8px 16px',
                    'backgroundColor': '#007bff',
                    'color': 'white',
                    'border': 'none',
                    'borderRadius': '5px',
                    'cursor': 'pointer',
                    'boxShadow': '0 2px 6px rgba(0,0,0,0.1)'
                }
            ),
            html.Div([
                # Panel de filtros (colapsable)
                html.Div(id='filters-panel', children=[
                    html.H3("Filtros y Controles"),
                    html.Label("Aeronave:"),
                    html.Div(id='aeronave-dropdown-container'),  # Contenedor para el dropdown de aeronave
                    html.Label("Parámetro:"),
                    html.Div(id='parametro-dropdown-container'),  # Contenedor para el dropdown de parámetro
                    html.Label("Predictor:"),
                    html.Div(id='predictor-dropdown-container'),  # Nuevo: Contenedor para el dropdown de predictor
                    html.Label("Tipos de Modelo:"),
                    html.Div(id='tipo-modelo-container'),  # Checklist de tipo de modelo
                    html.Div(id='visualization-options-container'),  # Opciones de visualización (toggles)
                    html.Label("Métodos de Imputación:"),
                    html.Div(id='imputation-methods-container'),  # Checklist de métodos de imputación
                    html.Label("Tipo de comparación:"),
                    html.Div(id='comparison-type-container'),  # RadioItems de tipo de comparación
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
                    'transition': 'width 0.3s, min-width 0.3s, max-width 0.3s, opacity 0.3s',
                    'overflowY': 'auto',
                    'height': 'fit-content'
                }),
                # Área de visualización principal (centro, expandible)
                html.Div(id='main-visualization-container', children=[
                    # Tabs de navegación - PARTE SUPERIOR
                    dcc.Tabs(id='view-tabs', value='2d-view', 
                            style={'marginBottom': '10px'}, children=[
                        dcc.Tab(label='📊 Vista 2D (1 Predictor)', value='2d-view', 
                               style={'fontWeight': 'bold', 'padding': '10px'}),
                        dcc.Tab(label='🧊 Vista 3D (2 Predictores)', value='3d-view', 
                               style={'fontWeight': 'bold', 'padding': '10px'}),
                        dcc.Tab(label='📈 Comparación', value='comparison-view'),
                        dcc.Tab(label='📋 Métricas', value='metrics-view')
                    ]),
                    
                    # Contenedor unificado para ambas vistas
                    html.Div(id='unified-plot-area', children=[
                        # Vista 2D - Inicialmente visible
                        html.Div(id='2d-plot-container', children=[
                            dcc.Graph(
                                id='plot-2d', 
                                style={'height': '65vh', 'width': '100%'},
                                config={
                                    'displayModeBar': True,
                                    'displaylogo': False,
                                    'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
                                    'toImageButtonOptions': {
                                        'format': 'png',
                                        'filename': 'modelo_2d_analisis',
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
                        ], style={'display': 'block'}),  # Inicialmente visible
                        
                        # Vista 3D - Inicialmente oculta
                        html.Div(id='3d-plot-container', children=[
                            dcc.Graph(
                                id='plot-3d',
                                style={'height': '65vh', 'width': '100%'},
                                config={
                                    'displayModeBar': True,
                                    'displaylogo': False,
                                    'toImageButtonOptions': {
                                        'format': 'png',
                                        'filename': 'modelo_3d_analisis',
                                        'height': 600,
                                        'width': 1000,
                                        'scale': 1
                                    },
                                    'scrollZoom': True,
                                    'doubleClick': 'reset+autosize',
                                    'responsive': True
                                }
                            )
                        ], style={'display': 'none'}),  # Inicialmente oculto
                        
                        # Contenedor para otras vistas
                        html.Div(id='other-views-container', 
                                style={'display': 'none'})  # Inicialmente oculto
                    ])
                ], style={
                    'width': '56%',  # Se ajustará dinámicamente
                    'minWidth': '320px',
                    'display': 'inline-block',
                    'verticalAlign': 'top',
                    'margin': '10px',
                    'boxSizing': 'border-box',
                    'transition': 'width 0.3s',
                }),
                # Panel de información (derecha)
                html.Div([
                    create_info_panel()
                ], style={
                    'width': '22%',
                    'minWidth': '220px',
                    'maxWidth': '340px',
                    'display': 'inline-block',
                    'verticalAlign': 'top',
                    'boxSizing': 'border-box',
                    'margin': '10px',
                    'transition': 'width 0.3s'
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
        ], style={'position': 'relative', 'width': '100%'}),
        
        # Panel de resumen unificado (2D + 3D)
        html.Div([
            html.H3("📊 Resumen Unificado de Modelos (2D + 3D)", 
                   style={'marginBottom': '15px', 'color': '#007bff'}),
            
            # Indicadores de estado
            html.Div([
                html.Div([
                    html.Span("🎯 Vista Activa: ", style={'fontWeight': 'bold'}),
                    html.Span(id='active-view-indicator', 
                             style={'color': '#007bff', 'fontWeight': 'bold', 'fontSize': '16px'})
                ], style={'display': 'inline-block', 'marginRight': '30px'}),
                
                html.Div([
                    html.Span("📈 Modelos 2D: ", style={'fontWeight': 'bold'}),
                    html.Span(id='models-2d-count', 
                             style={'color': '#28a745', 'fontWeight': 'bold'})
                ], style={'display': 'inline-block', 'marginRight': '30px'}),
                
                html.Div([
                    html.Span("🧊 Modelos 3D: ", style={'fontWeight': 'bold'}),
                    html.Span(id='models-3d-count', 
                             style={'color': '#17a2b8', 'fontWeight': 'bold'})
                ], style={'display': 'inline-block', 'marginRight': '30px'}),
                
                html.Div([
                    html.Span("📊 Total Visible: ", style={'fontWeight': 'bold'}),
                    html.Span(id='total-models-count', 
                             style={'color': '#dc3545', 'fontWeight': 'bold'})
                ], style={'display': 'inline-block'})
                
            ], style={
                'marginBottom': '15px', 
                'padding': '12px', 
                'backgroundColor': '#e9ecef', 
                'borderRadius': '5px',
                'borderLeft': '4px solid #007bff'
            }),
            
            # Controles de filtrado del resumen
            html.Div([
                html.Label("🔍 Filtrar tabla por vista:", 
                          style={'fontWeight': 'bold', 'marginRight': '10px'}),
                dcc.RadioItems(
                    id='summary-filter',
                    options=[
                        {'label': '📊 Mostrar Todos', 'value': 'all'},
                        {'label': '📈 Solo 2D', 'value': '2d'},
                        {'label': '🧊 Solo 3D', 'value': '3d'},
                        {'label': '🎯 Solo Vista Activa', 'value': 'active'}
                    ],
                    value='all',
                    inline=True,
                    style={'marginTop': '5px'}
                )
            ], style={'marginBottom': '15px'}),
            
            # Contenedor de la tabla
            html.Div(id='unified-summary-table-container')
        ], style={
            'margin': '20px',
            'padding': '20px',
            'backgroundColor': '#f8f9fa',
            'borderRadius': '5px'
        }),
        
        # Stores para manejo de estado
        dcc.Store(id='models-data-store'),
        dcc.Store(id='filtered-models-2d-store'),  # Modelos filtrados para 2D
        dcc.Store(id='filtered-models-3d-store'),  # Modelos filtrados para 3D
        dcc.Store(id='unique-values-store'),
        dcc.Store(id='selected-model-store', data=None),  # Modelo seleccionado
        dcc.Store(id='active-view-store', data='2d-view'),  # Vista activa
        dcc.Store(id='filter-state-store', data={})  # Estado de filtros
    ])


def create_summary_table(df_summary: 'pd.DataFrame', selected_row_idx: Optional[int] = None):
    """
    Crea tabla de resumen de modelos con resaltado opcional.
    
    Parameters:
    -----------
    df_summary : pd.DataFrame
        DataFrame con resumen de modelos
    selected_row_idx : Optional[int]
        Índice de la fila seleccionada para resaltar
        
    Returns:
    --------
    dash_table.DataTable
        Tabla de Dash
    """
    if df_summary.empty:
        return html.P("No hay datos para mostrar.")

    # Estilo condicional para resaltar fila seleccionada
    style_data_conditional = [
        {
            'if': {'row_index': 'odd'},
            'backgroundColor': 'rgb(248, 248, 248)'
        },
        {
            'if': {'state': 'selected'},
            'backgroundColor': '#ffe082',  # Amarillo suave para fila seleccionada
            'color': 'black',
        }
    ]
    
    # Agregar resaltado específico si hay una fila seleccionada
    if selected_row_idx is not None:
        style_data_conditional.append({
            'if': {'row_index': selected_row_idx},
            'backgroundColor': '#ffb74d',  # Naranja más fuerte para fila activamente seleccionada
            'color': 'black',
            'fontWeight': 'bold'
        })

    return dash_table.DataTable(
        id='summary-table',
        data=df_summary.to_dict('records'),
        columns=[{"name": col, "id": col} for col in df_summary.columns if col != '_selected_'],
        row_selectable='single',  # Habilita selección de filas
        style_cell={
            'textAlign': 'left',
            'padding': '10px',
            'fontFamily': 'Arial'
        },
        style_header={
            'backgroundColor': '#007bff',
            'color': 'white',
            'fontWeight': 'bold'
        },
        style_data_conditional=style_data_conditional,
        sort_action="native",
        filter_action="native",
        page_action="native",
        page_current=0,
        page_size=10,
        selected_rows=[selected_row_idx] if selected_row_idx is not None else []
    )


def format_model_info(modelo: Dict):
    """
    Formatea información de un modelo para mostrar en el panel.
    
    Parameters:
    -----------
    modelo : Dict
        Diccionario con información del modelo
        
    Returns:
    --------
    html.Div
        Div formateado con información del modelo
    """
    if not modelo:
        return html.P("No hay información disponible.")
    
    components = []
    
    # Información básica
    components.append(html.H5("Información Básica"))
    components.append(html.P(f"Tipo: {modelo.get('tipo', 'N/A')}"))
    components.append(html.P(f"Predictores: {', '.join(modelo.get('predictores', []))}"))
    components.append(html.P(f"N° Predictores: {modelo.get('n_predictores', 'N/A')}"))
    
    # Ecuación
    ecuacion = modelo.get('ecuacion_string', '')
    if ecuacion:
        components.append(html.H5("Ecuación"))
        components.append(html.Code(ecuacion, style={
            'backgroundColor': '#f1f1f1',
            'padding': '10px',
            'borderRadius': '3px',
            'display': 'block',
            'whiteSpace': 'pre-wrap'
        }))
    
    # Métricas
    components.append(html.H5("Métricas de Rendimiento"))
    
    mape = modelo.get('mape')
    if mape is not None:
        components.append(html.P(f"MAPE: {mape:.3f}%"))
    
    r2 = modelo.get('r2')
    if r2 is not None:
        components.append(html.P(f"R²: {r2:.3f}"))
    
    corr = modelo.get('corr')
    if corr is not None:
        components.append(html.P(f"Correlación: {corr:.3f}"))
    
    confianza = modelo.get('Confianza')
    if confianza is not None:
        components.append(html.P(f"Confianza: {confianza:.3f}"))
    
    # Métricas LOOCV
    mape_loocv = modelo.get('MAPE_LOOCV')
    if mape_loocv is not None:
        components.append(html.P(f"MAPE LOOCV: {mape_loocv:.3f}%"))
    
    r2_loocv = modelo.get('R2_LOOCV')
    if r2_loocv is not None:
        components.append(html.P(f"R² LOOCV: {r2_loocv:.3f}"))
    
    corr_loocv = modelo.get('Corr_LOOCV')
    if corr_loocv is not None:
        components.append(html.P(f"Correlación LOOCV: {corr_loocv:.3f}"))
    
    confianza_loocv = modelo.get('Confianza_LOOCV')
    if confianza_loocv is not None:
        components.append(html.P(f"Confianza LOOCV: {confianza_loocv:.3f}"))
    
    k_loocv = modelo.get('k_LOOCV')
    if k_loocv is not None and k_loocv != '':
        components.append(html.P(f"k LOOCV: {k_loocv}"))
    
    # Entrenamiento
    n_muestras = modelo.get('n_muestras_entrenamiento')
    if n_muestras:
        components.append(html.P(f"N° muestras entrenamiento: {n_muestras}"))
    
    # Advertencias
    advertencia = modelo.get('Advertencia')
    if advertencia:
        components.append(html.H5("Advertencias"))
        components.append(html.P(advertencia, style={'color': 'red'}))
    
    return html.Div(components)


def create_imputation_methods_checklist(selected: Optional[List[str]] = None) -> dcc.Checklist:
    """
    Crea checklist para selección de métodos de imputación a visualizar.
    
    Parameters:
    -----------
    selected : Optional[List[str]]
        Métodos seleccionados por defecto
        
    Returns:
    --------
    dcc.Checklist
        Componente checklist de Dash
    """
    default_selected = selected or ['final', 'similitud', 'correlacion']
    
    options = [
        {'label': 'Final (Promedio Ponderado)', 'value': 'final'},
        {'label': 'Similitud', 'value': 'similitud'},
        {'label': 'Correlación', 'value': 'correlacion'}
    ]
    
    return dcc.Checklist(
        id='imputation-methods-checklist',
        options=options,
        value=default_selected,
        style={'marginBottom': '10px'},
        inputStyle={"marginRight": "5px"}
    )


def create_filter_controls() -> html.Div:
    """
    Crea el panel de filtros optimizado para modelos de un predictor (2D).
    Incluye solo los controles relevantes y con IDs únicos y claros.
    """
    return html.Div([
        html.H3("Filtros y Controles"),
        html.Label("Aeronave:"),
        html.Div(id='aeronave-dropdown-container'),  # Contenedor para el dropdown de aeronave
        html.Label("Parámetro:"),
        html.Div(id='parametro-dropdown-container'),  # Contenedor para el dropdown de parámetro
        html.Label("Predictor:"),
        html.Div(id='predictor-dropdown-container'),  # Nuevo: Contenedor para el dropdown de predictor
        html.Label("Tipos de Modelo:"),
        html.Div(id='tipo-modelo-container'),  # Checklist de tipo de modelo
        html.Div(id='visualization-options-container'),  # Opciones de visualización (toggles)
        html.Label("Métodos de Imputación:"),
        html.Div(id='imputation-methods-container'),  # Checklist de métodos de imputación
        html.Label("Tipo de comparación:"),
        html.Div(id='comparison-type-container'),  # RadioItems de tipo de comparación
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
        'transition': 'width 0.3s, min-width 0.3s, max-width 0.3s, opacity 0.3s',
        'overflowY': 'auto',
        'height': 'fit-content'
    })

def create_comparison_type_radioitems(selected: Optional[str] = None) -> dcc.RadioItems:
    """
    Crea radio items para selección de tipo de comparación.
    
    Parameters:
    -----------
    selected : Optional[str]
        Tipo seleccionado por defecto
        
    Returns:
    --------
    dcc.RadioItems
        Componente radio items de Dash
    """
    default_selected = selected or 'all'
    
    options = [
        {'label': 'Todas las comparaciones', 'value': 'all'},
        {'label': 'Solo originales vs predichos', 'value': 'original_vs_predicted'},
        {'label': 'Solo LOOCV', 'value': 'loocv'}
    ]
    
    return dcc.RadioItems(
        id='comparison-type',
        options=options,
        value=default_selected,
        style={'marginBottom': '10px'},
        inputStyle={"marginRight": "5px"}
    )
