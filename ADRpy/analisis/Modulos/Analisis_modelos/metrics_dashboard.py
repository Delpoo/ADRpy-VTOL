"""
Dashboard visual y moderno para la pestaña de métricas globales.
Incluye KPIs, tablas y gráficos para un resumen completo del estado de los modelos importados y filtrados.
"""
from dash import html, dcc, dash_table
import plotly.express as px
import pandas as pd
from typing import Dict, List, Any, Optional

def generate_metrics_dashboard(
    modelos_por_celda: Dict[str, List[Dict]],
    detalles_por_celda: Dict[str, Any],
    modelos_filtrados: Optional[List[Dict]] = None,
    modelos_mostrados: Optional[List[Dict]] = None,
    celda_seleccionada: Optional[str] = None,
    modelos_no_mostrados: Optional[List[Dict]] = None
) -> html.Div:
    # KPIs globales
    total_celdas = len(modelos_por_celda)
    total_modelos = sum(len(v) for v in modelos_por_celda.values())
    total_modelos_filtrados = len(modelos_filtrados) if modelos_filtrados is not None else 0
    total_modelos_mostrados = len(modelos_mostrados) if modelos_mostrados is not None else 0
    total_modelos_no_mostrados = len(modelos_no_mostrados) if modelos_no_mostrados is not None else 0
    total_celdas_con_modelos = sum(1 for v in modelos_por_celda.values() if len(v) > 0)
    porcentaje_celdas_cubiertas = 100 * total_celdas_con_modelos / total_celdas if total_celdas else 0

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
    modelos_flat = [m for ms in modelos_por_celda.values() for m in ms if isinstance(m, dict) and m.get('mape') is not None]
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
        kpi_box(total_modelos_no_mostrados, "Modelos no mostrados", '#dc3545', "Modelos presentes en el JSON pero que no se visualizan por inconsistencias, errores o falta de datos requeridos."),
        kpi_box(total_modelos_mostrados, "Modelos mostrados", '#ff9800', "Modelos que efectivamente se visualizan en la gráfica principal para la celda seleccionada."),
    ], style={'display': 'flex', 'flexWrap': 'wrap', 'marginBottom': '18px'})

    # Visualización: Modelos por celda (heatmap compacto)
    modelos_por_celda_count = {k: len(v) for k, v in modelos_por_celda.items()}
    df_celdas = pd.DataFrame([
        {'Celda': k, 'Cantidad': v} for k, v in modelos_por_celda_count.items()
    ])
    # Si hay muchas celdas, mostrar un heatmap compacto y scroll horizontal
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
        heatmap_fig = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale='Blues',
                colorbar=dict(title='Modelos'),
                hoverongaps=False,
                hovertemplate='Aeronave: %{y}<br>Parámetro: %{x}<br>Cantidad: %{z}<extra></extra>'
            ),
            layout=go.Layout(
                title='Cantidad de modelos por celda',
                title_x=0.5,
                autosize=True,
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=0, r=0, t=40, b=0),
                height=320 if len(pivot) < 20 else min(600, 12*len(pivot)),
            )
        )
        heatmap_celdas = dcc.Graph(
            id='heatmap-modelos',
            figure=heatmap_fig,
            style={'height': '340px', 'width': '100%', 'overflowX': 'auto', 'marginBottom': '18px', 'background': 'white'},
            config={'responsive': True}
        )


    # Panel principal
    children = [
        html.H3("Dashboard de Métricas Globales", style={"marginBottom": "10px", 'fontSize': '1.3em', 'textAlign': 'center'}),
        html.Div(kpis, style={'display': 'flex', 'justifyContent': 'center', 'width': '100%'}),
        html.Hr(style={'margin': '10px 0 18px 0'}),
    ]
    # Nueva fila: heatmap arriba, luego fila con tipo de modelo (65%) y predictores (35%)
    if heatmap_celdas:
        children.append(html.Div(heatmap_celdas, style={'marginBottom': '18px', 'height': '380px', 'minHeight': '260px', 'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'width': '100%'}))
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
            'display': 'flex',
            'flexDirection': 'column',
            'alignItems': 'center',
            'justifyContent': 'center'
        }
    )
