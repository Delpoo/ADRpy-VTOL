"""
Pestaña de Métricas y Resumen de Modelos
========================================

Este módulo provee funciones para generar el contenido de la pestaña de métricas en la app Dash.
Muestra información relevante sobre los modelos importados, filtrados, mostrados y ausentes,
para facilitar la depuración y el análisis de cobertura de los datos.
"""

from typing import Dict, List, Any, Optional
from dash import html
from dash import dash_table
import pandas as pd

def generate_metrics_summary(
    modelos_por_celda: Dict[str, List[Dict]],
    detalles_por_celda: Dict[str, Any],
    modelos_filtrados: Optional[List[Dict]] = None,
    modelos_mostrados: Optional[List[Dict]] = None,
    celda_seleccionada: Optional[str] = None,
    modelos_no_mostrados: Optional[List[Dict]] = None
) -> html.Div:
    """
    Genera un resumen de métricas y conteos de modelos para la pestaña de métricas.
    """
    # Conteos globales
    total_celdas = len(modelos_por_celda)
    total_modelos_importados = sum(len(v) for v in modelos_por_celda.values())
    total_modelos_detalles = sum(len(v) if isinstance(v, list) else 0 for v in detalles_por_celda.values())
    total_modelos_filtrados = len(modelos_filtrados) if modelos_filtrados is not None else 0
    total_modelos_mostrados = len(modelos_mostrados) if modelos_mostrados is not None else 0
    total_modelos_no_mostrados = len(modelos_no_mostrados) if modelos_no_mostrados is not None else 0

    # Para la celda seleccionada
    modelos_celda = modelos_por_celda.get(celda_seleccionada, []) if celda_seleccionada else []
    detalles_celda = detalles_por_celda.get(celda_seleccionada, {}) if celda_seleccionada else {}
    n_modelos_celda = len(modelos_celda)
    n_detalles_celda = len(detalles_celda) if isinstance(detalles_celda, list) else (1 if detalles_celda else 0)

    resumen = [
        html.H3("Resumen de Modelos y Métricas"),
        html.Ul([
            html.Li(f"Total de celdas importadas: {total_celdas}"),
            html.Li(f"Total de modelos importados (JSON): {total_modelos_importados}"),
            html.Li(f"Total de modelos con detalles: {total_modelos_detalles}"),
            html.Li(f"Total de modelos filtrados: {total_modelos_filtrados}"),
            html.Li(f"Total de modelos mostrados: {total_modelos_mostrados}"),
            html.Li(f"Total de modelos no mostrados: {total_modelos_no_mostrados}"),
        ]),
        html.Hr(),
        html.H4("Celda seleccionada"),
        html.Ul([
            html.Li(f"Celda: {celda_seleccionada if celda_seleccionada else 'Ninguna'}"),
            html.Li(f"Modelos en celda: {n_modelos_celda}"),
            html.Li(f"Detalles en celda: {n_detalles_celda}"),
        ]),
    ]

    # Tabla de modelos no mostrados (si hay)
    if modelos_no_mostrados:
        df_no_mostrados = pd.DataFrame(modelos_no_mostrados)
        resumen.append(html.H5("Modelos no mostrados (posibles problemas):"))
        resumen.append(dash_table.DataTable(
            data=df_no_mostrados.to_dict('records'),
            columns=[{"name": col, "id": col} for col in df_no_mostrados.columns],
            page_size=5,
            style_cell={'textAlign': 'left', 'fontFamily': 'Arial', 'fontSize': 12},
            style_header={'backgroundColor': '#ffb74d', 'color': 'black', 'fontWeight': 'bold'},
            style_data={'backgroundColor': '#fffde7'},
        ))

    return html.Div(resumen, style={
        'padding': '20px',
        'backgroundColor': '#f8f9fa',
        'borderRadius': '5px',
        'margin': '10px',
        'boxShadow': '0 2px 6px rgba(0,0,0,0.07)'
    })


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
