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
            id='show-models-without-loocv',
            options=[{'label': 'Mostrar modelos sin validación LOOCV', 'value': 'show_without_loocv'}],
            value=['show_without_loocv'],  # Por defecto mostrar todos los modelos
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
                    html.Div(id='aeronave-dropdown-container', children=[
                        create_aeronave_dropdown([])  # Inicializar con dropdown vacío
                    ]),
                    html.Label("Parámetro:"),
                    html.Div(id='parametro-dropdown-container', children=[
                        create_parametro_dropdown([])  # Inicializar con dropdown vacío
                    ]),
                    html.Label("Predictor:"),
                    html.Div(id='predictor-dropdown-container', children=[
                        create_predictor_dropdown([])  # Inicializar con dropdown vacío
                    ]),
                    html.Label("Tipos de Modelo:"),
                    html.Div(id='tipo-modelo-container', children=[
                        create_tipo_modelo_checklist([])  # Inicializar con checklist vacío
                    ]),
                    html.Div(id='visualization-options-container', children=[
                        create_visualization_options()  # Contiene todos los componentes: show-*, hide-*, comparison-type, etc.
                    ]),
                    html.Label("Métodos de Imputación:"),
                    html.Div(id='imputation-methods-container'),  # Se llena por callback debido a duplicación
                    html.Label("Tipo de comparación:"),
                    html.Div(id='comparison-type-container'),  # Se llena por callback debido a duplicación,
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
                    'width': '19%',
                    'minWidth': '200px',
                    'maxWidth': '300px',
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
                # Gráfico principal (centro, expandible)
                html.Div(id='main-plot-container', children=[
                    # El área central ahora acepta cualquier componente (gráfica o dashboard)
                    html.Div(
                        id='main-plot',
                        style={
                            'height': '100%',
                            'width': '100%',
                            'minHeight': '0',
                            'flex': '1 1 0%',
                            'display': 'flex',
                            'flexDirection': 'column',
                            'justifyContent': 'stretch',
                            'alignItems': 'stretch',
                            'overflow': 'hidden'
                        }
                    ),
                    # Tabs ahora van debajo del área central
                    dcc.Tabs(id='plot-tabs', value='main-view', children=[
                        dcc.Tab(label='2D', value='main-view'),
                        dcc.Tab(label='3D', value='3d-view'),
                        dcc.Tab(label='Comparación', value='comparison-view'),
                        dcc.Tab(label='Métricas', value='metrics-view')
                    ], style={'marginTop': '0'}),
                    html.Div(id='tab-content')
                ], style={
                    'width': '62%',  # Se ajustará dinámicamente
                    'minWidth': '320px',
                    'display': 'flex',
                    'flexDirection': 'column',
                    'verticalAlign': 'top',
                    'margin': '10px',
                    'boxSizing': 'border-box',
                    'transition': 'width 0.3s',
                    'height': '80vh',
                    'minHeight': '500px',
                    'maxHeight': '100vh',
                }),
                # Panel de información (derecha)
                html.Div([
                    create_info_panel()
                ], style={
                    'width': '19%',
                    'minWidth': '200px',
                    'maxWidth': '300px',
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
        html.Div([
            html.H3("Resumen de Modelos"),
            html.Div(id='summary-table-container')
        ], style={
            'margin': '20px',
            'padding': '20px',
            'backgroundColor': '#f8f9fa',
            'borderRadius': '5px'
        }),
        dcc.Store(id='models-data-store'),
        dcc.Store(id='filtered-models-store'),
        dcc.Store(id='unique-values-store'),
        dcc.Store(id='selected-model-store', data=None),  # Store para modelo seleccionado
        
        # Botón flotante de alertas
        create_floating_alerts_button()
    ])


def create_validation_alert(df_summary: 'pd.DataFrame') -> html.Div:
    """
    Crea alerta de validación si hay modelos problemáticos.
    
    Parameters:
    -----------
    df_summary : pd.DataFrame
        DataFrame con información de validación en attrs
        
    Returns:
    --------
    html.Div
        Componente de alerta o None si no hay problemas
    """
    if not hasattr(df_summary, 'attrs'):
        return html.Div()
    
    modelos_validos = df_summary.attrs.get('modelos_validos', 0)
    modelos_con_problemas = df_summary.attrs.get('modelos_con_problemas', 0)
    problemas_encontrados = df_summary.attrs.get('problemas_encontrados', {})
    
    if modelos_con_problemas == 0:
        return html.Div()
    
    total_modelos = modelos_validos + modelos_con_problemas
    
    # Crear mensaje de alerta
    mensaje_principal = f"⚠️ {modelos_con_problemas} de {total_modelos} modelos tienen problemas de datos:"
    
    # Listar los problemas más frecuentes
    problemas_texto = []
    for problema, cantidad in sorted(problemas_encontrados.items(), key=lambda x: x[1], reverse=True)[:5]:
        # Traducir nombres técnicos a mensajes más amigables
        problema_amigable = {
            'sin_tipo': 'Sin tipo de modelo',
            'sin_predictores': 'Sin predictores definidos', 
            'y_original_formato_invalido': 'Datos Y en formato inválido',
            'x_original_formato_invalido': 'Datos X en formato inválido',
            'y_original_sin_datos_validos': 'Datos Y sin valores válidos',
            'x_original_vacio': 'Datos X vacíos',
            'r2_invalido': 'R² inválido (NaN/infinito)',
            'confianza_nan': 'Confianza NaN'
        }.get(problema, problema)
        
        problemas_texto.append(f"• {problema_amigable}: {cantidad} modelo{'s' if cantidad > 1 else ''}")
    
    return html.Div([
        html.Div(mensaje_principal, style={'fontWeight': 'bold', 'marginBottom': '5px'}),
        html.Ul([html.Li(texto) for texto in problemas_texto[:3]], style={'margin': '0', 'paddingLeft': '20px'}),
        html.Small("Los modelos problemáticos pueden aparecer en la tabla pero no ser graficables.", 
                  style={'fontStyle': 'italic', 'color': '#666'}) if problemas_texto else None
    ], style={
        'backgroundColor': '#fff3cd',
        'border': '1px solid #ffeaa7', 
        'color': '#856404',
        'padding': '10px',
        'marginBottom': '10px',
        'borderRadius': '4px',
        'fontSize': '14px'
    })


def create_summary_table(df_summary: 'pd.DataFrame', selected_row_idx: Optional[int] = None):
    """
    Crea tabla de resumen de modelos con resaltado opcional y alertas de validación.
    
    Parameters:
    -----------
    df_summary : pd.DataFrame
        DataFrame con resumen de modelos
    selected_row_idx : Optional[int]
        Índice de la fila seleccionada para resaltar
        
    Returns:
    --------
    html.Div
        Contenedor con alerta (si aplica) y tabla de Dash
    """
    if df_summary.empty:
        return html.P("No hay datos para mostrar.")

    # Crear alerta de validación si hay problemas
    alert_component = create_validation_alert(df_summary)

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
        },
        # Resaltar filas con errores críticos en rojo suave
        {
            'if': {
                'filter_query': '{Estado} contains "❌"',
                'column_id': 'Estado'
            },
            'backgroundColor': '#ffebee',
            'color': '#d32f2f'
        },
        # Resaltar filas con advertencias en amarillo suave
        {
            'if': {
                'filter_query': '{Estado} contains "⚠️"',
                'column_id': 'Estado'
            },
            'backgroundColor': '#fff8e1',
            'color': '#f57c00'
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

    return html.Div([
        alert_component,  # Incluir componente de alerta al inicio
        dash_table.DataTable(
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
            page_size=50,  # Aumentado de 10 a 50 modelos por página
            selected_rows=[selected_row_idx] if selected_row_idx is not None else []
        )
    ])


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
    
    # Aeronave y parámetro objetivo
    aeronave = modelo.get('Aeronave', 'N/A')
    parametro = modelo.get('Parámetro', modelo.get('parametro', 'N/A'))
    components.append(html.H5(f"Aeronave: {aeronave}"))
    components.append(html.H5(f"Parámetro: {parametro}"))

    # Información básica
    components.append(html.H5("Información Básica"))
    components.append(html.P(f"Tipo: {modelo.get('tipo', 'N/A')}"))
    components.append(html.P(f"Predictores: {', '.join(modelo.get('predictores', []) )}"))
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
def create_floating_alerts_button() -> html.Div:
    """
    Crea un botón flotante para mostrar/ocultar alertas del sistema.
    
    Returns:
    --------
    html.Div
        Botón flotante con modal de alertas
    """
    return html.Div([
        # Botón flotante
        html.Button(
            "🚨",
            id="floating-alerts-button",
            style={
                'position': 'fixed',
                'bottom': '20px',
                'right': '20px',
                'width': '60px',
                'height': '60px',
                'borderRadius': '50%',
                'border': 'none',
                'backgroundColor': '#dc3545',
                'color': 'white',
                'fontSize': '24px',
                'cursor': 'pointer',
                'boxShadow': '0 4px 8px rgba(0,0,0,0.3)',
                'zIndex': '1000',
                'transition': 'all 0.3s ease'
            }
        ),
        
        # Modal de alertas (inicialmente oculto)
        html.Div(
            id="alerts-modal",
            children=[
                html.Div([
                    html.Div([
                        html.H4("🚨 Alertas del Sistema", style={'margin': '0 0 15px 0'}),
                        html.Button(
                            "×",
                            id="close-alerts-modal",
                            style={
                                'position': 'absolute',
                                'top': '10px',
                                'right': '15px',
                                'border': 'none',
                                'background': 'none',
                                'fontSize': '24px',
                                'cursor': 'pointer',
                                'color': '#999'
                            }
                        ),
                        html.Div(id="alerts-content", children=[
                            html.P("Cargando alertas del sistema...", style={'color': '#666'})
                        ])
                    ], style={
                        'backgroundColor': 'white',
                        'padding': '20px',
                        'borderRadius': '8px',
                        'maxWidth': '500px',
                        'maxHeight': '400px',
                        'overflowY': 'auto',
                        'position': 'relative',
                        'margin': 'auto',
                        'marginTop': '10vh'
                    })
                ], style={
                    'position': 'fixed',
                    'top': '0',
                    'left': '0',
                    'width': '100%',
                    'height': '100%',
                    'backgroundColor': 'rgba(0,0,0,0.5)',
                    'zIndex': '1001',
                    'display': 'flex',
                    'alignItems': 'flex-start',
                    'justifyContent': 'center'
                })
            ],
            style={'display': 'none'}  # Inicialmente oculto
        )
    ])


def update_alerts_content(df_summary: Optional['pd.DataFrame'] = None, modelos_por_celda: Optional[dict] = None) -> list:
    """
    Actualiza el contenido de alertas basado en el estado actual del sistema.
    
    Parameters:
    -----------
    df_summary : pd.DataFrame, optional
        DataFrame con información de validación
    modelos_por_celda : dict, optional
        Diccionario con todos los modelos
        
    Returns:
    --------
    list
        Lista de componentes HTML para mostrar en el modal
    """
    alertas = []
    
    # Alertas de modelos problemáticos
    if df_summary is not None and hasattr(df_summary, 'attrs'):
        modelos_con_errores = df_summary.attrs.get('modelos_con_problemas', 0)
        problemas_encontrados = df_summary.attrs.get('problemas_encontrados', {})
        
        if modelos_con_errores > 0:
            alertas.append(html.Div([
                html.H5("⚠️ Modelos con Problemas", style={'color': '#dc3545', 'marginBottom': '10px'}),
                html.P(f"{modelos_con_errores} modelos tienen problemas de datos:", style={'marginBottom': '8px'}),
                html.Ul([
                    html.Li(f"{problema.replace('_', ' ').title()}: {cantidad} modelo{'s' if cantidad > 1 else ''}")
                    for problema, cantidad in sorted(problemas_encontrados.items(), key=lambda x: x[1], reverse=True)[:5]
                ], style={'marginLeft': '15px', 'marginBottom': '10px'}),
                html.Small("Los modelos aparecen en la tabla pero pueden tener limitaciones de visualización.", 
                          style={'color': '#666', 'fontStyle': 'italic'})
            ], style={'marginBottom': '15px', 'padding': '10px', 'backgroundColor': '#fff3cd', 'borderRadius': '4px'}))
    
    # Alertas de filtros activos
    alertas.append(html.Div([
        html.H5("🔍 Filtros Activos", style={'color': '#007bff', 'marginBottom': '10px'}),
        html.P("Algunos modelos pueden estar ocultos por filtros activos:"),
        html.Ul([
            html.Li("Filtro LOOCV: Oculta modelos sin confianza LOOCV"),
            html.Li("Filtros de tipo: Pueden limitar tipos de modelo visibles"),
            html.Li("Filtros de método: Pueden filtrar por método de imputación")
        ], style={'marginLeft': '15px'})
    ], style={'marginBottom': '15px', 'padding': '10px', 'backgroundColor': '#e3f2fd', 'borderRadius': '4px'}))
    
    # Alertas de rendimiento
    if modelos_por_celda:
        total_modelos = sum(len(v) for v in modelos_por_celda.values())
        if total_modelos > 1000:
            alertas.append(html.Div([
                html.H5("⚡ Rendimiento", style={'color': '#ff9800', 'marginBottom': '10px'}),
                html.P(f"Se detectaron {total_modelos} modelos en total. Con datasets grandes:"),
                html.Ul([
                    html.Li("La carga inicial puede ser lenta"),
                    html.Li("Use filtros para mejorar la navegación"),
                    html.Li("Las tablas están paginadas para mejor rendimiento")
                ], style={'marginLeft': '15px'})
            ], style={'marginBottom': '15px', 'padding': '10px', 'backgroundColor': '#fff3e0', 'borderRadius': '4px'}))
    
    # Alertas generales del sistema
    alertas.append(html.Div([
        html.H5("💡 Consejos de Uso", style={'color': '#28a745', 'marginBottom': '10px'}),
        html.Ul([
            html.Li("Use la tabla de resumen para identificar modelos específicos"),
            html.Li("Los modelos marcados como ❌ pueden tener datos faltantes"),
            html.Li("Revise los KPIs en la pestaña Métricas para estadísticas globales"),
            html.Li("Active/desactive filtros para ver diferentes conjuntos de modelos")
        ], style={'marginLeft': '15px'})
    ], style={'padding': '10px', 'backgroundColor': '#e8f5e8', 'borderRadius': '4px'}))
    
    if not alertas:
        return [html.P("✅ No hay alertas activas en el sistema.", style={'color': '#28a745', 'textAlign': 'center'})]
    
    return alertas
