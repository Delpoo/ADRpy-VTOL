"""
Dashboard visual y moderno para la pestaña de métricas globales.
Incluye KPIs, tablas y gráficos para un resumen completo del estado de los modelos importados y filtrados.
"""
from dash import html, dcc, dash_table
import plotly.express as px
import pandas as pd
from typing import Dict, List, Any, Optional

def find_missing_models(
    modelos_por_celda: Dict[str, List[Dict]],
    detalles_por_celda: Dict[str, Any]
) -> List[Dict]:
    """
    Busca modelos que están en detalles pero no en modelos_por_celda, o viceversa.
    """
    missing = []
    for celda, modelos in modelos_por_celda.items():
        detalles = detalles_por_celda.get(celda, None)
        if detalles is None:
            missing.append({'celda': celda, 'motivo': 'Sin detalles en detalles_por_celda'})
        elif not modelos:
            missing.append({'celda': celda, 'motivo': 'Sin modelos en modelos_por_celda'})
    # También buscar celdas en detalles_por_celda que no están en modelos_por_celda
    for celda in detalles_por_celda:
        if celda not in modelos_por_celda:
            missing.append({'celda': celda, 'motivo': 'Presente solo en detalles_por_celda'})
    return missing

def generate_metrics_dashboard(
    modelos_por_celda: Dict[str, List[Dict]],
    detalles_por_celda: Dict[str, Any],
    modelos_filtrados: Optional[List[Dict]] = None,
    modelos_mostrados: Optional[List[Dict]] = None,
    celda_seleccionada: Optional[str] = None,
    modelos_no_mostrados: Optional[List[Dict]] = None
) -> html.Div:
    # Importar función de validación para contar modelos con errores
    from .plot_interactive import validate_model_for_plotting
    
    # KPIs globales
    total_celdas = len(modelos_por_celda)
    total_modelos = sum(len(v) for v in modelos_por_celda.values())
    total_modelos_filtrados = len(modelos_filtrados) if modelos_filtrados is not None else 0
    total_modelos_mostrados = len(modelos_mostrados) if modelos_mostrados is not None else 0
    total_celdas_con_modelos = sum(1 for v in modelos_por_celda.values() if len(v) > 0)
    porcentaje_celdas_cubiertas = 100 * total_celdas_con_modelos / total_celdas if total_celdas else 0
    
    # Calcular modelos no mostrados como: total disponibles - total mostrados
    # Nota: modelos_no_mostrados puede venir de find_missing_models() pero es otra métrica diferente
    total_modelos_no_mostrados = total_modelos - total_modelos_mostrados
    
    # NUEVO: Contar modelos con warnings vs modelos completos
    modelos_con_warnings = 0
    modelos_completos = 0
    modelos_criticos = 0
    tipos_warnings = {}
    
    for modelos_lista in modelos_por_celda.values():
        for modelo in modelos_lista:
            if isinstance(modelo, dict):
                es_valido, warnings = validate_model_for_plotting(modelo)
                if not es_valido:
                    modelos_criticos += 1
                elif warnings:
                    modelos_con_warnings += 1
                    for warning in warnings:
                        tipos_warnings[warning] = tipos_warnings.get(warning, 0) + 1
                else:
                    modelos_completos += 1
    
    porcentaje_con_warnings = 100 * modelos_con_warnings / total_modelos if total_modelos > 0 else 0
    porcentaje_criticos = 100 * modelos_criticos / total_modelos if total_modelos > 0 else 0

    # Distribución por tipo de modelo
    tipos = []
    for modelos in modelos_por_celda.values():
        for m in modelos:
            if isinstance(m, dict):
                tipos.append(m.get('tipo', 'N/A'))
    tipo_counts = pd.Series(tipos).value_counts().reset_index()
    tipo_counts.columns = ['Tipo de Modelo', 'Cantidad']
    fig_tipo = px.bar(tipo_counts, x='Tipo de Modelo', y='Cantidad', title='Distribución por Tipo de Modelo', color='Tipo de Modelo') if not tipo_counts.empty else None
    if fig_tipo:
        fig_tipo.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            title_x=0.5,
            font=dict(size=15),
            margin=dict(l=10, r=10, t=50, b=10)
        )

    # Top 5 modelos por MAPE (menor es mejor)
    modelos_flat = []
    for celda_key, ms in modelos_por_celda.items():
        if '|' in celda_key:
            aeronave, parametro = celda_key.split('|', 1)
        else:
            aeronave, parametro = celda_key, ''
        for m in ms:
            if isinstance(m, dict) and m.get('mape') is not None:
                m = dict(m)  # avoid mutating original
                m['Aeronave'] = aeronave
                m['Parámetro'] = parametro
                modelos_flat.append(m)
    top_mape = sorted(modelos_flat, key=lambda m: m.get('mape', 9999))[:5]
    df_top_mape = pd.DataFrame(top_mape)
    # Modelos por número de predictores
    n_preds = [m.get('n_predictores', 0) for m in modelos_flat]
    pred_counts = pd.Series(n_preds).value_counts().sort_index().reset_index()
    pred_counts.columns = ['N° Predictores', 'Cantidad']
    fig_preds = px.bar(pred_counts, x='N° Predictores', y='Cantidad', title='Modelos por N° de Predictores', color='N° Predictores') if not pred_counts.empty else None
    if fig_preds:
        fig_preds.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            title_x=0.5,
            font=dict(size=15),
            margin=dict(l=10, r=10, t=50, b=10)
        )

    # KPIs visuales compactos con tooltips
    kpi_style = {
        'background': '#f8f9fa', 'borderRadius': '8px', 'padding': '10px 12px', 'margin': '0 10px 10px 0',
        'boxShadow': '0 2px 8px rgba(0,0,0,0.07)', 'display': 'inline-block', 'minWidth': '120px', 'textAlign': 'center',
        'position': 'relative', 'fontSize': '15px', 'lineHeight': '1.1'
    }
    def kpi_box(value, label, color, tooltip):
        return html.Div([
            html.H3(f"{value}", style={'color': color, 'margin': 0, 'fontSize': '1.5em'}),
            html.P(label, style={'margin': 0, 'fontSize': '0.95em'}),
            html.Span(" ⓘ", title=tooltip, style={'cursor': 'help', 'color': '#888', 'fontSize': '1em'})
        ], style=kpi_style)

    kpis = html.Div([
        kpi_box(total_modelos, "Modelos importados", '#007bff', "Cantidad total de modelos presentes en el JSON, sin filtrar ni agrupar."),
        kpi_box(total_celdas, "Celdas importadas", '#28a745', "Cantidad de celdas (combinaciones aeronave-parámetro) importadas del JSON."),
        kpi_box(f"{porcentaje_celdas_cubiertas:.1f}%", "Cobertura de celdas", '#17a2b8', "Porcentaje de celdas que tienen al menos un modelo entrenado."),
        kpi_box(total_modelos_filtrados, "Modelos filtrados", '#6f42c1', "Cantidad de modelos que cumplen los filtros activos en la interfaz (tipo, predictores, etc)."),
        kpi_box(total_modelos_mostrados, "Modelos mostrados", '#fd7e14', "Modelos que efectivamente se visualizan en la gráfica principal para la celda seleccionada."),
        kpi_box(modelos_completos, "Modelos completos", '#28a745', f"Modelos sin problemas ni campos faltantes ({100-porcentaje_con_warnings-porcentaje_criticos:.1f}% del total)."),
        kpi_box(modelos_con_warnings, "Modelos incompletos", '#ff9800', f"Modelos con datos faltantes no críticos ({porcentaje_con_warnings:.1f}% del total). Ej: sin LOOCV, sin método imputación."),
        kpi_box(modelos_criticos, "Modelos con errores", '#dc3545', f"Modelos con errores críticos que impiden su graficado ({porcentaje_criticos:.1f}% del total)."),
        kpi_box(total_modelos_no_mostrados, "Modelos no mostrados", '#6c757d', "Diferencia entre modelos importados y modelos actualmente mostrados en la interfaz."),
    ], style={'display': 'flex', 'flexWrap': 'wrap', 'marginBottom': '18px', 'justifyContent': 'center', 'width': '100%'})

    # Visualización: Modelos por celda (heatmap compacto)
    modelos_por_celda_count = {k: len(v) for k, v in modelos_por_celda.items()}
    df_celdas = pd.DataFrame([
        {'Celda': k, 'Cantidad': v} for k, v in modelos_por_celda_count.items()
    ])
    heatmap_celdas = None
    if not df_celdas.empty:
        # Separar aeronave y parámetro si es posible
        if df_celdas['Celda'].str.contains('|', regex=False).all():
            df_celdas[['Aeronave', 'Parámetro']] = df_celdas['Celda'].str.split('|', expand=True)
        else:
            df_celdas['Aeronave'] = df_celdas['Celda']
            df_celdas['Parámetro'] = ''
        # Pivot para heatmap
        pivot = df_celdas.pivot_table(index='Aeronave', columns='Parámetro', values='Cantidad', fill_value=0)
        import plotly.graph_objects as go
        # Ajuste 100% ancho, alto igual al ancho (1:1), navegación óptima
        heatmap_fig = go.Figure(
            data=[go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale='Blues',
                colorbar=dict(title='Modelos'),
                hoverongaps=False,
                hovertemplate='Aeronave: %{y}<br>Parámetro: %{x}<br>Cantidad: %{z}<extra></extra>'
            )],
            layout=go.Layout(
                title='Cantidad de modelos por celda',
                title_x=0.5,
                autosize=True,
                width=None,
                height=None,
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=0, r=0, t=40, b=0),
                xaxis=dict(tickangle=45, automargin=True),
                yaxis=dict(automargin=True)
            )
        )
        heatmap_celdas = html.Div(
            dcc.Graph(
                id='heatmap-modelos',
                figure=heatmap_fig,
                style={
                    'width': '100%',
                    'aspectRatio': '1',
                    'minWidth': '400px',
                    'minHeight': '400px',
                    'maxWidth': '100%',
                    'maxHeight': '100vw',
                },
                config={
                    'responsive': True,
                    'scrollZoom': True,
                    'displayModeBar': 'hover',
                    'displaylogo': False,
                    # Asegura que los botones estándar estén presentes
                    'modeBarButtonsToRemove': [],
                    'modeBarButtonsToAdd': ['zoom2d', 'pan2d', 'resetScale2d', 'resetViewMapbox'],
                }
            ),
            style={
                'overflowX': 'auto',
                'overflowY': 'auto',
                'width': '100%',
                'maxWidth': '100%',
                'background': 'white',
                'border': '1px solid #eee',
                'marginBottom': '18px',
                'display': 'flex',
                'justifyContent': 'center',
                'alignItems': 'center',
            }
        )
    # Panel principal
    children = [
        html.H3("Dashboard de Métricas Globales", style={"marginBottom": "10px", 'fontSize': '1.3em', 'textAlign': 'center'}),
        kpis,
        html.Hr(style={'margin': '10px 0 18px 0'}),
    ]
    # Nueva fila: heatmap arriba, luego fila con tipo de modelo (65%) y predictores (35%)
    if heatmap_celdas is not None:
        children.append(html.Div([
            heatmap_celdas
        ], style={'marginBottom': '18px', 'height': '900px', 'minHeight': '600px', 'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center', 'alignItems': 'center', 'width': '100%', 'background': 'white', 'border': '1px solid #eee'}))
    # Fila con dos gráficos más altos y proporción 65/35
    row_graphs = []
    if fig_tipo:
        row_graphs.append(
            html.Div(
                dcc.Graph(figure=fig_tipo, style={'height': '400px', 'width': '100%', 'marginBottom': '0'}),
                style={'width': '65%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingRight': '8px'}
            )
        )
    if fig_preds:
        row_graphs.append(
            html.Div(
                dcc.Graph(figure=fig_preds, style={'height': '400px', 'width': '100%', 'marginBottom': '0'}),
                style={'width': '35%', 'display': 'inline-block', 'verticalAlign': 'top'}
            )
        )
    if row_graphs:
        children.append(html.Div(row_graphs, style={'width': '100%', 'display': 'flex', 'flexDirection': 'row', 'marginBottom': '18px', 'justifyContent': 'center', 'alignItems': 'center'}))
    # Tablas en su ancho estándar, una debajo de la otra, con más altura
    if not df_top_mape.empty:
        children.append(html.H5("Top 5 Modelos con Mejor MAPE (menor es mejor)", style={'marginTop': '10px', 'marginBottom': '5px'}))
        children.append(dash_table.DataTable(
            data=df_top_mape[['Aeronave','Parámetro','tipo','mape','r2','n_predictores']].to_dict('records'),
            columns=[{"name": c, "id": c} for c in ['Aeronave','Parámetro','tipo','mape','r2','n_predictores']],
            style_cell={'textAlign': 'center', 'fontFamily': 'Arial', 'fontSize': 14, 'padding': '8px'},
            style_header={'backgroundColor': '#007bff', 'color': 'white', 'fontWeight': 'bold'},
            style_data={'backgroundColor': '#f8f9fa'},
            page_size=5,
            style_table={'marginBottom': '18px', 'width': '100%'}
        ))
    if modelos_no_mostrados:
        df_no_mostrados = pd.DataFrame(modelos_no_mostrados)
        children.append(html.H5("Modelos/Celdas con inconsistencias", style={'marginTop': '10px', 'marginBottom': '5px'}))
        children.append(dash_table.DataTable(
            data=df_no_mostrados.to_dict('records'),
            columns=[{"name": c, "id": c} for c in df_no_mostrados.columns],
            style_cell={'textAlign': 'left', 'fontFamily': 'Arial', 'fontSize': 13, 'padding': '7px'},
            style_header={'backgroundColor': '#dc3545', 'color': 'white', 'fontWeight': 'bold'},
            style_data={'backgroundColor': '#fff5f5'},
            page_size=5,
            style_table={'marginBottom': '18px', 'width': '100%'}
        ))
    return html.Div(
        children,
        style={
            'padding': '18px 10px 10px 10px',
            'background': '#fff',
            'borderRadius': '10px',
            'boxShadow': '0 2px 12px rgba(0,0,0,0.06)',
            'overflowX': 'auto',
            'width': '100%',
            'maxWidth': '1200px',
            'margin': '0 auto',
            'display': 'block',  # Cambiado de flex a block
            # 'flexDirection': 'column',
            # 'alignItems': 'center',
            # 'justifyContent': 'center'
            # Eliminadas restricciones de flexbox que pueden colapsar hijos
        }
    )
