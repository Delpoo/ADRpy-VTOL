"""
comparison_system.py

Sistema de comparación avanzada de modelos.
Permite comparar múltiples modelos lado a lado con visualizaciones y métricas.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from dash import html, dcc, dash_table


class ModelComparisonSystem:
    """Sistema de comparación avanzada de modelos."""
    
    def __init__(self):
        """Inicializar el sistema de comparación."""
        self.max_models = 4  # Máximo 4 modelos para comparar
    
    def create_comparison_chart(self, models: List[Dict], 
                               aeronave: str, parametro: str) -> go.Figure:
        """
        Crear gráfico de comparación de múltiples modelos.
        
        Parameters:
        -----------
        models : List[Dict]
            Lista de modelos a comparar
        aeronave : str
            Aeronave seleccionada
        parametro : str
            Parámetro seleccionado
            
        Returns:
        --------
        go.Figure
            Figura con comparación de modelos
        """
        if not models:
            fig = go.Figure()
            fig.add_annotation(
                text="Seleccione modelos para comparar",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=16, color="gray")
            )
            return fig
        
        # Crear subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Curvas de Modelos', 'Métricas de Performance', 
                           'Distribución de Residuos', 'Comparación R² vs MAPE'),
            specs=[[{"secondary_y": False}, {"type": "bar"}],
                   [{"type": "histogram"}, {"type": "scatter"}]]
        )
        
        colors = px.colors.qualitative.Set1[:len(models)]
        
        # 1. Gráfico de curvas superpuestas
        for i, model in enumerate(models):
            if not isinstance(model, dict):
                continue
            
            # Obtener predicciones del modelo (simuladas para demo)
            x_range = np.linspace(0, 1, 100)
            predictions = self._get_model_predictions_demo(model, x_range)
            
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=predictions,
                    mode='lines',
                    name=f"{model.get('tipo', 'Unknown')} (M{i+1})",
                    line=dict(color=colors[i], width=3),
                    legendgroup=f"model_{i}"
                ),
                row=1, col=1
            )
        
        # 2. Gráfico de barras de métricas
        model_names = [f"M{i+1}: {m.get('tipo', 'Unknown')}" for i, m in enumerate(models)]
        mape_values = [m.get('mape', 0) for m in models]
        r2_values = [m.get('r2', 0) for m in models]
        
        fig.add_trace(
            go.Bar(
                x=model_names,
                y=mape_values,
                name='MAPE (%)',
                marker_color='red',
                opacity=0.7
            ),
            row=1, col=2
        )
        
        # 3. Histograma de residuos (simulado)
        for i, model in enumerate(models):
            residuals = np.random.normal(0, model.get('mape', 10), 100)
            fig.add_trace(
                go.Histogram(
                    x=residuals,
                    name=f"Residuos M{i+1}",
                    opacity=0.6,
                    marker_color=colors[i],
                    legendgroup=f"model_{i}",
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # 4. Scatter R² vs MAPE
        fig.add_trace(
            go.Scatter(
                x=mape_values,
                y=r2_values,
                mode='markers+text',
                text=model_names,
                textposition='top center',
                marker=dict(
                    size=15,
                    color=colors[:len(models)],
                    line=dict(width=2, color='white')
                ),
                name='Modelos',
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Actualizar layout
        fig.update_layout(
            title=f'Comparación de Modelos - {aeronave}: {parametro}',
            height=800,
            showlegend=True,
            legend=dict(x=1.05, y=1)
        )
        
        # Configurar ejes específicos
        fig.update_xaxes(title_text="X normalizado", row=1, col=1)
        fig.update_yaxes(title_text=parametro, row=1, col=1)
        
        fig.update_xaxes(title_text="Modelos", row=1, col=2)
        fig.update_yaxes(title_text="MAPE (%)", row=1, col=2)
        
        fig.update_xaxes(title_text="Residuos", row=2, col=1)
        fig.update_yaxes(title_text="Frecuencia", row=2, col=1)
        
        fig.update_xaxes(title_text="MAPE (%)", row=2, col=2)
        fig.update_yaxes(title_text="R²", row=2, col=2)
        
        return fig
    
    def _get_model_predictions_demo(self, model: Dict, x_range: np.ndarray) -> np.ndarray:
        """Generar predicciones demo para el modelo."""
        tipo = model.get('tipo', 'linear')
        
        if 'linear' in tipo:
            return 2 * x_range + np.random.normal(0, 0.1, len(x_range))
        elif 'poly' in tipo:
            return x_range ** 2 + 0.5 * x_range + np.random.normal(0, 0.1, len(x_range))
        elif 'exp' in tipo:
            return np.exp(x_range) + np.random.normal(0, 0.1, len(x_range))
        elif 'log' in tipo:
            return np.log(x_range + 0.1) + np.random.normal(0, 0.1, len(x_range))
        else:
            return x_range + np.random.normal(0, 0.1, len(x_range))
    
    def create_comparison_table(self, models: List[Dict]) -> dash_table.DataTable:
        """
        Crear tabla de comparación detallada.
        
        Parameters:
        -----------
        models : List[Dict]
            Lista de modelos a comparar
            
        Returns:
        --------
        dash_table.DataTable
            Tabla de comparación
        """
        if not models:
            return dash_table.DataTable(data=[], columns=[])
        
        # Preparar datos para la tabla
        table_data = []
        for i, model in enumerate(models):
            if not isinstance(model, dict):
                continue
            
            row = {
                'Modelo': f"M{i+1}",
                'Tipo': model.get('tipo', 'N/A'),
                'Predictores': ', '.join(model.get('predictores', [])),
                'N° Predictores': model.get('n_predictores', 0),
                'MAPE (%)': f"{model.get('mape', 0):.3f}",
                'R²': f"{model.get('r2', 0):.3f}",
                'Ecuación': model.get('ecuacion', 'N/A')
            }
            table_data.append(row)
        
        # Definir columnas
        columns = [
            {'name': 'Modelo', 'id': 'Modelo', 'type': 'text'},
            {'name': 'Tipo', 'id': 'Tipo', 'type': 'text'},
            {'name': 'Predictores', 'id': 'Predictores', 'type': 'text'},
            {'name': 'N° Pred.', 'id': 'N° Predictores', 'type': 'numeric'},
            {'name': 'MAPE (%)', 'id': 'MAPE (%)', 'type': 'numeric'},
            {'name': 'R²', 'id': 'R²', 'type': 'numeric'},
            {'name': 'Ecuación', 'id': 'Ecuación', 'type': 'text'}
        ]
        
        # Encontrar el mejor modelo (menor MAPE)
        best_model_idx = 0
        if len(models) > 1:
            best_mape = float('inf')
            for i, model in enumerate(models):
                if isinstance(model, dict) and model.get('mape', float('inf')) < best_mape:
                    best_mape = model.get('mape', float('inf'))
                    best_model_idx = i
        
        # Estilo condicional para resaltar el mejor modelo
        style_data_conditional = [
            {
                'if': {'row_index': best_model_idx},
                'backgroundColor': '#d5f4e6',
                'color': 'black',
                'fontWeight': 'bold'
            }
        ]
        
        return dash_table.DataTable(
            data=table_data,
            columns=columns,
            style_cell={
                'textAlign': 'left',
                'padding': '10px',
                'font-family': 'Arial, sans-serif'
            },
            style_header={
                'backgroundColor': '#3498db',
                'color': 'white',
                'fontWeight': 'bold',
                'textAlign': 'center'
            },
            style_data_conditional=style_data_conditional,
            style_table={'overflowX': 'auto'},
            tooltip_data=[
                {
                    'Ecuación': {'value': row['Ecuación'], 'type': 'markdown'}
                } for row in table_data
            ],
            tooltip_duration=None
        )
    
    def create_metrics_radar_chart(self, models: List[Dict]) -> go.Figure:
        """
        Crear gráfico de radar para comparar métricas.
        
        Parameters:
        -----------
        models : List[Dict]
            Lista de modelos a comparar
            
        Returns:
        --------
        go.Figure
            Gráfico de radar
        """
        if not models:
            return go.Figure()
        
        fig = go.Figure()
        
        for i, model in enumerate(models):
            if not isinstance(model, dict):
                continue
            
            # Normalizar métricas para el radar (0-1)
            mape = model.get('mape', 100)
            r2 = model.get('r2', 0)
            
            # Invertir MAPE para que menor sea mejor (1 - mape_normalizado)
            mape_normalized = max(0, 1 - (mape / 100))
            r2_normalized = max(0, r2)
            
            # Métricas adicionales (simuladas)
            complexity = 1 / (model.get('n_predictores', 1))  # Simplicidad
            stability = np.random.uniform(0.6, 0.9)  # Estabilidad simulada
            
            fig.add_trace(go.Scatterpolar(
                r=[mape_normalized, r2_normalized, complexity, stability],
                theta=['Precisión (1-MAPE)', 'R²', 'Simplicidad', 'Estabilidad'],
                fill='toself',
                name=f"M{i+1}: {model.get('tipo', 'Unknown')}",
                opacity=0.6
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Comparación de Métricas Normalizadas",
            showlegend=True
        )
        
        return fig


def create_comparison_ui() -> html.Div:
    """
    Crear la interfaz de comparación de modelos.
    
    Returns:
    --------
    html.Div
        Componente de comparación
    """
    return html.Div([
        # Encabezado
        html.Div([
            html.H4([
                html.I(className="fas fa-balance-scale", style={'margin-right': '10px'}),
                "🔄 Comparación de Modelos"
            ], style={'margin-bottom': '20px', 'color': '#2c3e50'}),
            
            html.P(
                "Seleccione hasta 4 modelos para comparar lado a lado:",
                style={'margin-bottom': '15px', 'color': '#7f8c8d'}
            )
        ]),
        
        # Selector de modelos para comparar
        html.Div([
            html.Label("Modelos a Comparar:", style={'font-weight': 'bold', 'margin-bottom': '5px'}),
            dcc.Dropdown(
                id="compare-models-dropdown",
                multi=True,
                placeholder="Seleccionar modelos (máximo 4)...",
                style={'margin-bottom': '20px'}
            )
        ]),
        
        # Tabs de comparación
        dcc.Tabs(id="comparison-tabs", value="curves-comparison", children=[
            # Tab 1: Comparación de curvas
            dcc.Tab(label="📈 Curvas y Métricas", value="curves-comparison", children=[
                html.Div([
                    dcc.Graph(id="comparison-chart", style={'height': '800px'})
                ], style={'padding': '20px'})
            ]),
            
            # Tab 2: Tabla detallada
            dcc.Tab(label="📋 Tabla Detallada", value="table-comparison", children=[
                html.Div([
                    html.Div(id="comparison-table-container"),
                    html.Div([
                        html.P("🏆 El mejor modelo se resalta en verde", 
                               style={'margin-top': '10px', 'font-style': 'italic', 'color': '#27ae60'})
                    ])
                ], style={'padding': '20px'})
            ]),
            
            # Tab 3: Radar de métricas
            dcc.Tab(label="🎯 Radar de Métricas", value="radar-comparison", children=[
                html.Div([
                    dcc.Graph(id="radar-comparison-chart", style={'height': '600px'}),
                    html.Div([
                        html.H5("📊 Interpretación del Radar:"),
                        html.Ul([
                            html.Li("Precisión: Mayor área = Menor MAPE (mejor)"),
                            html.Li("R²: Mayor valor = Mejor ajuste"),
                            html.Li("Simplicidad: Mayor valor = Menos predictores"),
                            html.Li("Estabilidad: Mayor valor = Más robusto")
                        ])
                    ], style={'margin-top': '20px', 'padding': '15px', 'background-color': '#f8f9fa', 'border-radius': '5px'})
                ], style={'padding': '20px'})
            ])
        ]),
        
        # Botones de acción
        html.Div([
            html.Button([
                html.I(className="fas fa-download", style={'margin-right': '5px'}),
                "Exportar Comparación"
            ], id="export-comparison-btn", className="btn btn-primary", style={'margin-right': '10px'}),
            
            html.Button([
                html.I(className="fas fa-file-alt", style={'margin-right': '5px'}),
                "Generar Reporte"
            ], id="generate-comparison-report-btn", className="btn btn-info")
        ], style={'margin-top': '20px', 'text-align': 'center'}),
        
        # Estado de la comparación
        html.Div(
            id="comparison-status",
            style={'margin-top': '15px', 'text-align': 'center'}
        )
        
    ], style={
        'padding': '20px',
        'background-color': '#ffffff',
        'border': '1px solid #e0e0e0',
        'border-radius': '8px',
        'box-shadow': '0 2px 4px rgba(0,0,0,0.1)'
    })


def create_comparison_callbacks(app, comparison_system: ModelComparisonSystem):
    """Crear callbacks para el sistema de comparación."""
    
    from dash import Input, Output, State, html
    import dash
    
    @app.callback(
        Output('compare-models-dropdown', 'options'),
        [Input('models-data-store', 'data'),
         Input('aeronave-dropdown', 'value'),
         Input('parametro-dropdown', 'value')]
    )
    def update_comparison_dropdown(models_data, aeronave, parametro):
        """Actualizar opciones del dropdown de comparación."""
        if not models_data or not aeronave or not parametro:
            return []
        
        celda_key = f"{aeronave}|{parametro}"
        modelos = models_data.get('modelos', {}).get(celda_key, [])
        
        options = []
        for i, modelo in enumerate(modelos):
            if isinstance(modelo, dict):
                label = f"M{i+1}: {modelo.get('tipo', 'Unknown')} (MAPE: {modelo.get('mape', 0):.2f}%)"
                options.append({'label': label, 'value': i})
        
        return options
    
    @app.callback(
        [Output('comparison-chart', 'figure'),
         Output('comparison-table-container', 'children'),
         Output('radar-comparison-chart', 'figure'),
         Output('comparison-status', 'children')],
        [Input('compare-models-dropdown', 'value')],
        [State('models-data-store', 'data'),
         State('aeronave-dropdown', 'value'),
         State('parametro-dropdown', 'value')]
    )
    def update_comparison_views(selected_model_indices, models_data, aeronave, parametro):
        """Actualizar todas las vistas de comparación."""
        if not selected_model_indices or not models_data or not aeronave or not parametro:
            empty_fig = go.Figure()
            return empty_fig, html.Div("Seleccione modelos para comparar"), empty_fig, ""
        
        # Limitar a 4 modelos máximo
        if len(selected_model_indices) > 4:
            selected_model_indices = selected_model_indices[:4]
        
        celda_key = f"{aeronave}|{parametro}"
        all_models = models_data.get('modelos', {}).get(celda_key, [])
        
        # Obtener modelos seleccionados
        selected_models = [all_models[i] for i in selected_model_indices if i < len(all_models)]
        
        # Crear visualizaciones
        comparison_chart = comparison_system.create_comparison_chart(selected_models, aeronave, parametro)
        comparison_table = comparison_system.create_comparison_table(selected_models)
        radar_chart = comparison_system.create_metrics_radar_chart(selected_models)
        
        # Estado
        status = f"Comparando {len(selected_models)} modelos"
        if len(selected_models) > 0:
            best_model = min(selected_models, key=lambda x: x.get('mape', float('inf')) if isinstance(x, dict) else float('inf'))
            if isinstance(best_model, dict):
                status += f" | Mejor: {best_model.get('tipo', 'Unknown')} (MAPE: {best_model.get('mape', 0):.2f}%)"
        
        return comparison_chart, comparison_table, radar_chart, status
