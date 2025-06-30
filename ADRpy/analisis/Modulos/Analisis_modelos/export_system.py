"""
export_system.py

Sistema de exportación para gráficos, reportes y configuraciones.
Permite exportar visualizaciones en diferentes formatos y generar reportes automatizados.
"""

import os
import json
import base64
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from io import BytesIO
import zipfile


class ExportSystem:
    """Sistema de exportación completo para la aplicación."""
    
    def __init__(self, output_dir: str = "exports"):
        """
        Inicializar el sistema de exportación.
        
        Parameters:
        -----------
        output_dir : str
            Directorio donde guardar las exportaciones
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def export_chart(self, fig: go.Figure, filename: str, format: str = "png", 
                    width: int = 1200, height: int = 800, scale: int = 2) -> str:
        """
        Exportar gráfico en formato especificado.
        
        Parameters:
        -----------
        fig : go.Figure
            Figura de Plotly a exportar
        filename : str
            Nombre del archivo (sin extensión)
        format : str
            Formato de exportación ('png', 'svg', 'pdf', 'html')
        width : int
            Ancho en píxeles
        height : int
            Alto en píxeles
        scale : int
            Factor de escala para mejor calidad
            
        Returns:
        --------
        str
            Ruta del archivo exportado
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_filename = f"{filename}_{timestamp}.{format}"
        filepath = os.path.join(self.output_dir, full_filename)
        
        try:
            if format.lower() == 'png':
                pio.write_image(fig, filepath, format='png', 
                              width=width, height=height, scale=scale)
            elif format.lower() == 'svg':
                pio.write_image(fig, filepath, format='svg', 
                              width=width, height=height)
            elif format.lower() == 'pdf':
                pio.write_image(fig, filepath, format='pdf', 
                              width=width, height=height)
            elif format.lower() == 'html':
                pio.write_html(fig, filepath, include_plotlyjs=True)
            else:
                raise ValueError(f"Formato no soportado: {format}")
                
            return filepath
            
        except Exception as e:
            raise Exception(f"Error exportando gráfico: {str(e)}")
    
    def generate_report(self, modelos_data: Dict, aeronave: str, parametro: str, 
                       selected_models: Optional[List[Dict]] = None) -> str:
        """
        Generar reporte completo en formato HTML.
        
        Parameters:
        -----------
        modelos_data : Dict
            Datos de modelos
        aeronave : str
            Aeronave seleccionada
        parametro : str
            Parámetro seleccionado
        selected_models : List[Dict]
            Modelos seleccionados para el reporte
            
        Returns:
        --------
        str
            Ruta del archivo de reporte generado
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reporte_{aeronave}_{parametro}_{timestamp}.html"
        filepath = os.path.join(self.output_dir, filename)
        
        # Obtener datos de la celda
        celda_key = f"{aeronave}|{parametro}"
        modelos = modelos_data.get('modelos', {}).get(celda_key, [])
        
        if selected_models:
            modelos = selected_models
        
        # Generar estadísticas
        stats = self._generate_statistics(modelos)
        
        # Crear HTML del reporte
        html_content = self._create_report_html(aeronave, parametro, modelos, stats)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return filepath
    
    def _generate_statistics(self, modelos: List[Dict]) -> Dict:
        """Generar estadísticas de los modelos."""
        if not modelos:
            return {}
        
        mape_values = [m.get('mape', 0) for m in modelos if isinstance(m, dict)]
        r2_values = [m.get('r2', 0) for m in modelos if isinstance(m, dict)]
        
        stats = {
            'total_models': len(modelos),
            'mape_stats': {
                'mean': sum(mape_values) / len(mape_values) if mape_values else 0,
                'min': min(mape_values) if mape_values else 0,
                'max': max(mape_values) if mape_values else 0
            },
            'r2_stats': {
                'mean': sum(r2_values) / len(r2_values) if r2_values else 0,
                'min': min(r2_values) if r2_values else 0,
                'max': max(r2_values) if r2_values else 0
            },
            'model_types': list(set([m.get('tipo', 'unknown') for m in modelos if isinstance(m, dict)]))
        }
        
        return stats
    
    def _create_report_html(self, aeronave: str, parametro: str, 
                           modelos: List[Dict], stats: Dict) -> str:
        """Crear contenido HTML del reporte."""
        html = f"""
        <!DOCTYPE html>
        <html lang="es">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Reporte de Análisis - {aeronave}: {parametro}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .header {{ text-align: center; color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 20px; margin-bottom: 30px; }}
                .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 30px 0; }}
                .stat-card {{ background: #ecf0f1; padding: 20px; border-radius: 8px; text-align: center; }}
                .stat-value {{ font-size: 2em; font-weight: bold; color: #2980b9; }}
                .stat-label {{ color: #7f8c8d; margin-top: 5px; }}
                .models-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                .models-table th, .models-table td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                .models-table th {{ background-color: #3498db; color: white; }}
                .models-table tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .footer {{ text-align: center; color: #7f8c8d; margin-top: 30px; padding-top: 20px; border-top: 1px solid #ecf0f1; }}
                .best-model {{ background-color: #d5f4e6 !important; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📊 Reporte de Análisis de Modelos</h1>
                    <h2>{aeronave}: {parametro}</h2>
                    <p>Generado el: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}</p>
                </div>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value">{stats.get('total_models', 0)}</div>
                        <div class="stat-label">Total de Modelos</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{stats.get('mape_stats', {}).get('mean', 0):.2f}%</div>
                        <div class="stat-label">MAPE Promedio</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{stats.get('r2_stats', {}).get('mean', 0):.3f}</div>
                        <div class="stat-label">R² Promedio</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{len(stats.get('model_types', []))}</div>
                        <div class="stat-label">Tipos de Modelo</div>
                    </div>
                </div>
                
                <h3>📋 Detalle de Modelos</h3>
                <table class="models-table">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Tipo</th>
                            <th>Predictores</th>
                            <th>MAPE (%)</th>
                            <th>R²</th>
                            <th>Ecuación</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        # Ordenar modelos por MAPE (menor = mejor)
        modelos_sorted = sorted([m for m in modelos if isinstance(m, dict)], 
                               key=lambda x: x.get('mape', float('inf')))
        
        for i, modelo in enumerate(modelos_sorted):
            is_best = i == 0
            row_class = "best-model" if is_best else ""
            best_indicator = "🥇 " if is_best else ""
            
            predictores = ", ".join(modelo.get('predictores', []))
            ecuacion = modelo.get('ecuacion', 'N/A')
            
            html += f"""
                        <tr class="{row_class}">
                            <td>{best_indicator}{i+1}</td>
                            <td>{modelo.get('tipo', 'N/A')}</td>
                            <td>{predictores}</td>
                            <td>{modelo.get('mape', 0):.3f}</td>
                            <td>{modelo.get('r2', 0):.3f}</td>
                            <td style="font-family: monospace; font-size: 0.9em;">{ecuacion}</td>
                        </tr>
            """
        
        html += f"""
                    </tbody>
                </table>
                
                <div class="footer">
                    <p>🏆 El mejor modelo se destaca en verde</p>
                    <p>Generado por Sistema de Análisis de Modelos ADRpy-VTOL</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def save_configuration(self, config_data: Dict, name: str = "config") -> str:
        """
        Guardar configuración actual de la aplicación.
        
        Parameters:
        -----------
        config_data : Dict
            Datos de configuración a guardar
        name : str
            Nombre del archivo de configuración
            
        Returns:
        --------
        str
            Ruta del archivo guardado
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        # Agregar metadatos
        config_with_meta = {
            'metadata': {
                'created': datetime.now().isoformat(),
                'version': '1.0',
                'type': 'visualization_config'
            },
            'config': config_data
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_with_meta, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def load_configuration(self, filepath: str) -> Dict:
        """
        Cargar configuración desde archivo.
        
        Parameters:
        -----------
        filepath : str
            Ruta del archivo de configuración
            
        Returns:
        --------
        Dict
            Datos de configuración
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data.get('config', data)
    
    def create_export_package(self, fig: go.Figure, modelos_data: Dict, 
                             aeronave: str, parametro: str, 
                             config_data: Optional[Dict] = None) -> str:
        """
        Crear paquete completo de exportación con gráfico, reporte y configuración.
        
        Returns:
        --------
        str
            Ruta del archivo ZIP creado
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        package_name = f"analisis_completo_{aeronave}_{parametro}_{timestamp}"
        zip_filepath = os.path.join(self.output_dir, f"{package_name}.zip")
        
        with zipfile.ZipFile(zip_filepath, 'w') as zipf:
            # Exportar gráfico
            chart_path = self.export_chart(fig, f"{package_name}_grafico", "png")
            zipf.write(chart_path, f"grafico_{aeronave}_{parametro}.png")
            
            # Generar reporte
            report_path = self.generate_report(modelos_data, aeronave, parametro)
            zipf.write(report_path, f"reporte_{aeronave}_{parametro}.html")
            
            # Guardar configuración si se proporciona
            if config_data:
                config_path = self.save_configuration(config_data, f"{package_name}_config")
                zipf.write(config_path, f"configuracion_{aeronave}_{parametro}.json")
        
        return zip_filepath


# Funciones de utilidad para Dash callbacks
def create_export_callbacks(app, export_system: ExportSystem):
    """Crear callbacks para el sistema de exportación."""
    
    from dash import Input, Output, State, html, dcc
    import dash
    
    @app.callback(
        Output('export-status', 'children'),
        [Input('export-chart-btn', 'n_clicks'),
         Input('export-report-btn', 'n_clicks'),
         Input('export-package-btn', 'n_clicks')],
        [State('main-plot', 'figure'),
         State('models-data-store', 'data'),
         State('aeronave-dropdown', 'value'),
         State('parametro-dropdown', 'value'),
         State('export-format-dropdown', 'value')]
    )
    def handle_exports(chart_clicks, report_clicks, package_clicks, 
                      figure, models_data, aeronave, parametro, export_format):
        """Manejar diferentes tipos de exportación."""
        
        if not any([chart_clicks, report_clicks, package_clicks]):
            return ""
        
        ctx = dash.callback_context
        if not ctx.triggered:
            return ""
        
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        try:
            if trigger_id == 'export-chart-btn' and figure and aeronave and parametro:
                fig = go.Figure(figure)
                filepath = export_system.export_chart(
                    fig, f"grafico_{aeronave}_{parametro}", 
                    export_format or 'png'
                )
                return html.Div([
                    html.I(className="fas fa-check-circle", style={'color': 'green'}),
                    f" Gráfico exportado: {os.path.basename(filepath)}"
                ], style={'color': 'green', 'margin': '10px 0'})
            
            elif trigger_id == 'export-report-btn' and models_data and aeronave and parametro:
                filepath = export_system.generate_report(
                    models_data, aeronave, parametro
                )
                return html.Div([
                    html.I(className="fas fa-check-circle", style={'color': 'green'}),
                    f" Reporte generado: {os.path.basename(filepath)}"
                ], style={'color': 'green', 'margin': '10px 0'})
            
            elif trigger_id == 'export-package-btn' and figure and models_data and aeronave and parametro:
                fig = go.Figure(figure)
                filepath = export_system.create_export_package(
                    fig, models_data, aeronave, parametro
                )
                return html.Div([
                    html.I(className="fas fa-check-circle", style={'color': 'green'}),
                    f" Paquete completo creado: {os.path.basename(filepath)}"
                ], style={'color': 'green', 'margin': '10px 0'})
        
        except Exception as e:
            return html.Div([
                html.I(className="fas fa-exclamation-circle", style={'color': 'red'}),
                f" Error: {str(e)}"
            ], style={'color': 'red', 'margin': '10px 0'})
        
        return ""
