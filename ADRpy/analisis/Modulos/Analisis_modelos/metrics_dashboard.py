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

    # Top 5 modelos por MAPE (menor es mejor)
    modelos_flat = [m for ms in modelos_por_celda.values() for m in ms if isinstance(m, dict) and m.get('mape') is not None]
    top_mape = sorted(modelos_flat, key=lambda m: m.get('mape', 9999))[:5]
    df_top_mape = pd.DataFrame(top_mape)
    # Modelos por número de predictores
    n_preds = [m.get('n_predictores', 0) for m in modelos_flat]
    pred_counts = pd.Series(n_preds).value_counts().sort_index().reset_index()
    pred_counts.columns = ['N° Predictores', 'Cantidad']
    fig_preds = px.bar(pred_counts, x='N° Predictores', y='Cantidad', title='Modelos por N° de Predictores', color='N° Predictores') if not pred_counts.empty else None

    # KPIs visuales
    kpi_style = {
        'background': '#f8f9fa', 'borderRadius': '8px', 'padding': '18px 24px', 'margin': '0 18px 18px 0',
        'boxShadow': '0 2px 8px rgba(0,0,0,0.07)', 'display': 'inline-block', 'minWidth': '180px', 'textAlign': 'center'
    }
    kpis = html.Div([
        html.Div([
            html.H2(f"{total_modelos}", style={'color': '#007bff', 'margin': 0}),
            html.P("Modelos importados")
        ], style=kpi_style),
        html.Div([
            html.H2(f"{total_celdas}", style={'color': '#28a745', 'margin': 0}),
            html.P("Celdas importadas")
        ], style=kpi_style),
        html.Div([
            html.H2(f"{porcentaje_celdas_cubiertas:.1f}%", style={'color': '#17a2b8', 'margin': 0}),
            html.P("Cobertura de celdas")
        ], style=kpi_style),
        html.Div([
            html.H2(f"{total_modelos_filtrados}", style={'color': '#6f42c1', 'margin': 0}),
            html.P("Modelos filtrados")
        ], style=kpi_style),
        html.Div([
            html.H2(f"{total_modelos_no_mostrados}", style={'color': '#dc3545', 'margin': 0}),
            html.P("Modelos no mostrados")
        ], style=kpi_style),
    ], style={'display': 'flex', 'flexWrap': 'wrap', 'marginBottom': '30px'})

    # Panel principal
    children = [
        html.H2("Dashboard de Métricas Globales", style={"marginBottom": "20px"}),
        kpis,
        html.Hr(),
    ]
    if fig_tipo:
        children.append(dcc.Graph(figure=fig_tipo, style={'marginBottom': '30px'}))
    if fig_preds:
        children.append(dcc.Graph(figure=fig_preds, style={'marginBottom': '30px'}))
    if not df_top_mape.empty:
        children.append(html.H4("Top 5 Modelos con Mejor MAPE (menor es mejor)"))
        children.append(dash_table.DataTable(
            data=df_top_mape[['Aeronave','Parámetro','tipo','mape','r2','n_predictores']].to_dict('records'),
            columns=[{"name": c, "id": c} for c in ['Aeronave','Parámetro','tipo','mape','r2','n_predictores']],
            style_cell={'textAlign': 'center', 'fontFamily': 'Arial', 'fontSize': 14},
            style_header={'backgroundColor': '#007bff', 'color': 'white', 'fontWeight': 'bold'},
            style_data={'backgroundColor': '#f8f9fa'},
            page_size=5
        ))
    if modelos_no_mostrados:
        children.append(html.H4("Modelos/Celdas con inconsistencias"))
        df_no_mostrados = pd.DataFrame(modelos_no_mostrados)
        children.append(dash_table.DataTable(
            data=df_no_mostrados.to_dict('records'),
            columns=[{"name": c, "id": c} for c in df_no_mostrados.columns],
            style_cell={'textAlign': 'left', 'fontFamily': 'Arial', 'fontSize': 13},
            style_header={'backgroundColor': '#dc3545', 'color': 'white', 'fontWeight': 'bold'},
            style_data={'backgroundColor': '#fff5f5'},
            page_size=5
        ))
    return html.Div(children, style={'padding': '30px', 'background': '#fff', 'borderRadius': '10px', 'boxShadow': '0 2px 12px rgba(0,0,0,0.06)'})
