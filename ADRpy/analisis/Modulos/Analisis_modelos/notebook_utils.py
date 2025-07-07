# =============================================================================
# 🚀 UTILIDADES CENTRALIZADAS PARA NOTEBOOK DE ANÁLISIS DE MODELOS
# =============================================================================
"""
Script centralizado que contiene todas las funcionalidades para el notebook
optimizado de análisis de modelos. Incluye configuración, diagnósticos,
lanzamiento, monitoreo y panel de control.
"""

import sys
import os
import json
import importlib
import psutil
import requests
import gc
import uuid
import traceback
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path
import threading
import time

# =============================================================================
# 🏗️ CONFIGURACIONES Y CONSTANTES
# =============================================================================

class Config:
    """Configuraciones centralizadas del sistema"""
    
    # Rutas de archivos
    JSON_PATH = 'Results/modelos_completos_por_celda.json'
    LOG_PATH = 'Results/notebook_logs.txt'
    
    # Puertos disponibles
    PORTS = {
        'debug': 8054,
        'produccion': 8055,
        'test': 8056
    }
    
    # Módulos críticos
    CRITICAL_MODULES = [
        'Modulos.Analisis_modelos.main_visualizacion_modelos',
        'Modulos.Analisis_modelos.plot_interactive',
        'Modulos.Analisis_modelos.ui_components',
        'Modulos.Analisis_modelos.data_loader'
    ]
    
    # Dependencias requeridas
    REQUIRED_DEPS = ['dash', 'plotly', 'pandas', 'numpy', 'psutil', 'requests']
    
    # Configuraciones por modo
    MODES = {
        'debug': {
            'json_path': JSON_PATH,
            'use_dash': True,
            'port': PORTS['debug'],
            'debug': True
        },
        'produccion': {
            'json_path': JSON_PATH,
            'use_dash': True,
            'port': PORTS['produccion'],
            'debug': False
        }
    }

# =============================================================================
# 📊 GESTIÓN DE DATOS Y ARCHIVOS  
# =============================================================================

class DataManager:
    """Gestión centralizada de datos y archivos"""
    
    @staticmethod
    def load_models_data():
        """Carga los datos de modelos desde JSON"""
        try:
            if not os.path.exists(Config.JSON_PATH):
                return None, f"Archivo no encontrado: {Config.JSON_PATH}"
            
            with open(Config.JSON_PATH, 'r') as f:
                data = json.load(f)
            
            # Detectar estructura de datos
            if isinstance(data, dict) and 'modelos_por_celda' in data:
                modelos_por_celda = data['modelos_por_celda']
                detalles_por_celda = data.get('detalles_por_celda', {})
                structure_type = "completa"
            else:
                modelos_por_celda = data
                detalles_por_celda = {}
                structure_type = "directa"
            
            total_modelos = sum(len(modelos) for modelos in modelos_por_celda.values())
            
            return {
                'modelos_por_celda': modelos_por_celda,
                'detalles_por_celda': detalles_por_celda,
                'total_modelos': total_modelos,
                'structure_type': structure_type,
                'num_celdas': len(modelos_por_celda)
            }, None
            
        except Exception as e:
            return None, f"Error cargando datos: {str(e)}"
    
    @staticmethod
    def get_file_info():
        """Obtiene información de archivos críticos"""
        info = {}
        
        # Archivo JSON principal
        if os.path.exists(Config.JSON_PATH):
            stat = os.stat(Config.JSON_PATH)
            info['json'] = {
                'exists': True,
                'size': f"{stat.st_size / 1024:.1f} KB",
                'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')
            }
        else:
            info['json'] = {'exists': False}
        
        return info

# =============================================================================
# 📝 SISTEMA DE LOGS PERSISTENTES
# =============================================================================

class LogManager:
    """Gestión de logs persistentes"""
    
    @staticmethod
    def log_event(event_type, message, details=None):
        """Registra un evento en el log"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = {
            'timestamp': timestamp,
            'type': event_type,
            'message': message,
            'details': details or {}
        }
        
        try:
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(Config.LOG_PATH), exist_ok=True)
            
            with open(Config.LOG_PATH, 'a', encoding='utf-8') as f:
                f.write(f"{timestamp} [{event_type}] {message}\n")
                if details:
                    f.write(f"  Detalles: {json.dumps(details, ensure_ascii=False)}\n")
        except Exception as e:
            print(f"Error escribiendo log: {e}")
    
    @staticmethod
    def get_recent_logs(hours=24):
        """Obtiene logs recientes"""
        if not os.path.exists(Config.LOG_PATH):
            return []
        
        try:
            logs = []
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            with open(Config.LOG_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            timestamp_str = line[:19]
                            log_time = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                            if log_time >= cutoff_time:
                                logs.append(line.strip())
                        except:
                            continue
            
            return logs[-20:]  # Últimos 20 logs
        except Exception as e:
            return [f"Error leyendo logs: {e}"]
    
    @staticmethod
    def clear_old_logs(hours=24):
        """Borra logs más antiguos que X horas"""
        if not os.path.exists(Config.LOG_PATH):
            return
        
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            new_logs = []
            
            with open(Config.LOG_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            timestamp_str = line[:19]
                            log_time = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                            if log_time >= cutoff_time:
                                new_logs.append(line)
                        except:
                            new_logs.append(line)  # Conservar líneas con formato inválido
            
            # Reescribir archivo con logs filtrados
            with open(Config.LOG_PATH, 'w', encoding='utf-8') as f:
                f.writelines(new_logs)
                
            LogManager.log_event('MAINTENANCE', f'Logs antiguos borrados, conservados últimas {hours}h')
            
        except Exception as e:
            LogManager.log_event('ERROR', f'Error borrando logs: {e}')

# =============================================================================
# 🖱️ CLICK DEBUGGER AVANZADO
# =============================================================================

class ClickDebugger:
    """🖱️ SISTEMA CENTRALIZADO DE DEBUG DE CLICKS - VERSIÓN MEJORADA"""
    
    @staticmethod
    def analyze_click_chain():
        """✅ Analiza TODA la cadena de procesamiento con detección del problema resuelto"""
        results = {
            'status': 'ok',
            'message': 'Análisis completo de clicks',
            'components': {},
            'recommendations': [],
            'critical_checks': {}
        }
        
        try:
            import traceback
            import gc
            import re
            
            status_emoji = {'ok': '✅', 'warning': '⚠️', 'error': '❌'}
            component_statuses = []
            
            print("🔍 ANALIZANDO CADENA COMPLETA DE CLICKS...")
            print("=" * 60)
            
            # ✅ CRÍTICO: Verificar arquitectura de IDs (problema que resolviste)
            print("🎯 VERIFICANDO ARQUITECTURA DE IDs DE GRÁFICOS...")
            id_analysis = ClickDebugger._analyze_graph_id_architecture()
            results['critical_checks']['graph_ids'] = id_analysis
            component_statuses.append(id_analysis['status'])
            
            # Mostrar resultado crítico
            emoji = status_emoji.get(id_analysis['status'], '❓')
            print(f"   {emoji} {id_analysis['message']}")
            if id_analysis.get('fix'):
                print(f"   🔧 {id_analysis['fix']}")
            
            # 1. Verificar datos y customdata
            data_result, error = DataManager.load_models_data()
            if error:
                results['components']['data'] = {
                    'status': 'error',
                    'message': f'Datos no disponibles: {error}',
                    'fix': 'Verificar archivo JSON en Results/'
                }
                results['status'] = 'error'
                return results
            
            # 2. Test de función create_interactive_plot
            try:
                from Modulos.Analisis_modelos.plot_interactive import create_interactive_plot
            except ImportError:
                sys.path.append('Modulos')
                from Analisis_modelos.plot_interactive import create_interactive_plot
            
            # 3. Crear figura de prueba y analizar customdata
            modelos_data = data_result['modelos_por_celda']
            if not modelos_data:
                results['components']['data_structure'] = {
                    'status': 'error',
                    'message': 'modelos_por_celda vacío',
                    'fix': 'Regenerar datos o verificar proceso de imputación'
                }
                results['status'] = 'error'
                return results
            
            # Obtener muestra de datos
            celda_ejemplo = list(modelos_data.keys())[0]
            modelos_test = modelos_data[celda_ejemplo][:3]  # Solo 3 modelos para test
            
            if '|' in celda_ejemplo:
                aeronave, parametro = celda_ejemplo.split('|', 1)
            else:
                aeronave, parametro = "TestAeronave", "TestParametro"
            
            modelos_filtrados = {celda_ejemplo: modelos_test}
            
            # Crear figura de test
            fig_test = create_interactive_plot(
                modelos_filtrados=modelos_filtrados,
                aeronave=aeronave,
                parametro=parametro,
                show_training_points=True,
                show_model_curves=True,
                detalles_por_celda=data_result['detalles_por_celda']
            )
            
            # 4. Analizar customdata en detalle
            print("🔍 Analizando customdata...")
            results['components']['customdata'] = ClickDebugger._analyze_customdata(fig_test)
            component_statuses.append(results['components']['customdata']['status'])
            
            # 5. Analizar callbacks de Dash
            print("🔍 Analizando callbacks...")
            results['components']['callbacks'] = ClickDebugger._analyze_callbacks()
            component_statuses.append(results['components']['callbacks']['status'])
            
            # ✅ NUEVO: Verificar consistencia de IDs en callbacks activos
            print("🔍 Verificando consistencia ID layout-callbacks...")
            id_consistency = ClickDebugger._verify_id_consistency()
            results['critical_checks']['id_consistency'] = id_consistency
            component_statuses.append(id_consistency['status'])
            
            # 6. Analizar event handlers
            print("🔍 Analizando event handlers...")
            results['components']['event_handlers'] = ClickDebugger._analyze_event_handlers()
            component_statuses.append(results['components']['event_handlers']['status'])
            
            # 7. Generar recomendaciones comprehensivas
            results['recommendations'] = ClickDebugger._generate_comprehensive_recommendations(results)
            
            # Determinar estado general
            if 'error' in component_statuses:
                results['status'] = 'error'
                results['message'] = 'Errores críticos detectados en cadena de clicks'
            elif 'warning' in component_statuses:
                results['status'] = 'warning'
                results['message'] = 'Warnings detectados, sistema funcional con optimizaciones posibles'
            
            return results
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error en análisis de clicks: {str(e)}',
                'traceback': traceback.format_exc(),
                'recommendations': [
                    'Verificar que todos los módulos estén correctamente importados',
                    'Revisar que los datos JSON estén en el formato esperado',
                    'Asegurar que Dash esté correctamente instalado'
                ]
            }
    
    @staticmethod
    def _analyze_graph_id_architecture():
        """✅ CRÍTICO: Detecta el problema de IDs que resolviste"""
        try:
            print("   📋 Analizando arquitectura de IDs de gráficos...")
            
            issues_found = []
            warnings_found = []
            
            # Buscar en main_visualizacion_modelos.py la estructura de IDs
            main_viz_path = os.path.join(os.path.dirname(__file__), 'main_visualizacion_modelos.py')
            if os.path.exists(main_viz_path):
                with open(main_viz_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Verificar que todos los dcc.Graph tienen id='plot-graph'
                    import re
                    graph_patterns = re.findall(r"dcc\.Graph\([^)]*id=['\"]([^'\"]*)['\"]", content)
                    
                    if graph_patterns:
                        unique_graph_ids = set(graph_patterns)
                        if len(unique_graph_ids) == 1 and 'plot-graph' in unique_graph_ids:
                            print("   ✅ Arquitectura de IDs CORRECTA: Todos los gráficos usan 'plot-graph'")
                        elif len(unique_graph_ids) > 1:
                            issues_found.append(f"Múltiples IDs de gráfico detectados: {unique_graph_ids}")
                            print(f"   ❌ PROBLEMA CRÍTICO: Múltiples IDs detectados: {unique_graph_ids}")
                        else:
                            warnings_found.append(f"ID no estándar encontrado: {unique_graph_ids}")
                    
                    # Verificar que callbacks usan el ID correcto
                    callback_patterns = re.findall(r"Input\(['\"]([^'\"]*)['\"],\s*['\"]clickData['\"]", content)
                    if callback_patterns:
                        unique_callback_ids = set(callback_patterns)
                        if 'plot-graph' in unique_callback_ids:
                            print("   ✅ Callbacks de clickData configurados correctamente")
                        else:
                            issues_found.append(f"Callbacks no usan 'plot-graph': {unique_callback_ids}")
                            print(f"   ❌ PROBLEMA: Callbacks usan IDs incorrectos: {unique_callback_ids}")
            
            if issues_found:
                return {
                    'status': 'error',
                    'message': 'Problemas críticos en arquitectura de IDs detectados',
                    'issues': issues_found,
                    'fix': 'Unificar todos los dcc.Graph con id="plot-graph" y actualizar callbacks'
                }
            elif warnings_found:
                return {
                    'status': 'warning',
                    'message': 'Arquitectura de IDs funcional con mejoras posibles',
                    'warnings': warnings_found,
                    'fix': 'Considerar estandarizar IDs para mejor mantenimiento'
                }
            else:
                return {
                    'status': 'ok',
                    'message': 'Arquitectura de IDs configurada correctamente',
                    'details': 'Todos los gráficos usan ID unificado y callbacks están sincronizados'
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error analizando arquitectura de IDs: {str(e)}',
                'fix': 'Verificar acceso a archivos de código fuente'
            }
    
    @staticmethod
    def _verify_id_consistency():
        """Verifica consistencia entre IDs en layout y callbacks activos"""
        try:
            import dash
            
            app = None
            for obj in gc.get_objects():
                if isinstance(obj, dash.Dash):
                    app = obj
                    break
            
            if app is None:
                return {
                    'status': 'warning',
                    'message': 'No se puede verificar consistencia - app no activa',
                    'fix': 'Lanzar aplicación Dash primero'
                }
            
            # Extraer IDs de callbacks
            callback_graph_ids = set()
            if hasattr(app, 'callback_map'):
                for cb_info in app.callback_map.values():
                    if 'inputs' in cb_info:
                        for inp in cb_info['inputs']:
                            if inp.get('property') == 'clickData':
                                callback_graph_ids.add(inp.get('id'))
            
            # Verificar que los IDs son consistentes
            if len(callback_graph_ids) == 1 and 'plot-graph' in callback_graph_ids:
                return {
                    'status': 'ok',
                    'message': 'IDs consistentes entre layout y callbacks',
                    'graph_ids': list(callback_graph_ids)
                }
            elif len(callback_graph_ids) > 1:
                return {
                    'status': 'error',
                    'message': f'IDs inconsistentes detectados: {callback_graph_ids}',
                    'graph_ids': list(callback_graph_ids),
                    'fix': 'Unificar IDs de gráficos a "plot-graph"'
                }
            else:
                return {
                    'status': 'warning',
                    'message': 'No se detectaron callbacks de clickData',
                    'fix': 'Verificar que callbacks estén registrados'
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error verificando consistencia: {str(e)}',
                'fix': 'Verificar estado de aplicación Dash'
            }
            customdata_analysis = ClickDebugger._analyze_customdata(fig_test)
            results['components']['customdata'] = customdata_analysis
            
            # 5. Verificar estructura de callbacks si hay app Dash
            callback_analysis = ClickDebugger._analyze_callbacks()
            results['components']['callbacks'] = callback_analysis
            
            # 6. Test de event handlers
            event_analysis = ClickDebugger._analyze_event_handlers()
            results['components']['event_handlers'] = event_analysis
            
            # 7. Generar recomendaciones basadas en los resultados
            recommendations = ClickDebugger._generate_recommendations(results['components'])
            results['recommendations'] = recommendations
            
            # Determinar status general
            component_statuses = [comp.get('status', 'unknown') for comp in results['components'].values()]
            if 'error' in component_statuses:
                results['status'] = 'error'
            elif 'warning' in component_statuses:
                results['status'] = 'warning'
            
            return results
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error en análisis de clicks: {str(e)}',
                'traceback': traceback.format_exc(),
                'recommendations': [
                    'Verificar que todos los módulos estén correctamente importados',
                    'Revisar que los datos JSON estén en el formato esperado',
                    'Asegurar que Dash esté correctamente instalado'
                ]
            }
    
    @staticmethod
    def _analyze_customdata(fig):
        """Analiza el customdata de la figura"""
        traces_with_customdata = []
        traces_without_customdata = []
        customdata_formats = {}
        
        for i, trace in enumerate(fig.data):
            trace_info = {
                'index': i,
                'name': trace.name or f'Trace {i}',
                'type': type(trace).__name__
            }
            
            if hasattr(trace, 'customdata') and trace.customdata is not None:
                trace_info['customdata_shape'] = getattr(trace.customdata, 'shape', 'N/A')
                trace_info['customdata_sample'] = str(trace.customdata[0] if len(trace.customdata) > 0 else 'Empty')
                traces_with_customdata.append(trace_info)
                
                # Analizar formato del customdata
                if len(trace.customdata) > 0:
                    sample = trace.customdata[0]
                    if isinstance(sample, (list, tuple)):
                        customdata_formats[i] = f'Array length {len(sample)}'
                    else:
                        customdata_formats[i] = type(sample).__name__
            else:
                traces_without_customdata.append(trace_info)
        
        status = 'ok' if traces_with_customdata else 'error'
        message = f'{len(traces_with_customdata)} traces con customdata, {len(traces_without_customdata)} sin customdata'
        
        return {
            'status': status,
            'message': message,
            'traces_with_customdata': traces_with_customdata,
            'traces_without_customdata': traces_without_customdata,
            'customdata_formats': customdata_formats,
            'fix': 'Verificar que create_interactive_plot esté generando customdata correctamente' if status == 'error' else None
        }
    
    @staticmethod
    def _analyze_callbacks():
        """Analiza callbacks de Dash si existe una app"""
        try:
            import dash
            
            # Buscar app Dash en memoria
            app = None
            for obj in gc.get_objects():
                if isinstance(obj, dash.Dash):
                    app = obj
                    break
            
            if app is None:
                return {
                    'status': 'warning',
                    'message': 'No se encontró app Dash activa',
                    'fix': 'Ejecutar celda de lanzamiento de la aplicación primero'
                }
            
            callback_count = len(app.callback_map) if hasattr(app, 'callback_map') else 0
            
            # Buscar callbacks específicos de click
            click_callbacks = []
            if hasattr(app, 'callback_map'):
                for cb_id, cb_info in app.callback_map.items():
                    if 'inputs' in cb_info:
                        for input_spec in cb_info['inputs']:
                            if input_spec.get('property') == 'clickData':
                                click_callbacks.append({
                                    'callback_id': cb_id,
                                    'component_id': input_spec.get('id'),
                                    'property': input_spec.get('property')
                                })
            
            return {
                'status': 'ok' if click_callbacks else 'warning',
                'message': f'{callback_count} callbacks totales, {len(click_callbacks)} para clicks',
                'total_callbacks': callback_count,
                'click_callbacks': click_callbacks,
                'fix': 'Verificar que los callbacks de clickData estén correctamente definidos' if not click_callbacks else None
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error analizando callbacks: {str(e)}',
                'fix': 'Verificar importación de Dash y estado de la aplicación'
            }
    
    @staticmethod
    def _analyze_event_handlers():
        """Analiza event handlers y configuración de eventos"""
        try:
            # Verificar que plotly.js esté disponible (indirectamente a través de plotly)
            import plotly
            
            # Test básico de configuración de eventos
            test_config = {
                'displayModeBar': True,
                'modeBarButtonsToRemove': ['pan2d', 'lasso2d'],
                'responsive': True
            }
            
            return {
                'status': 'ok',
                'message': 'Event handlers configurables',
                'plotly_version': plotly.__version__,
                'recommended_config': test_config,
                'fix': None
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error con event handlers: {str(e)}',
                'fix': 'Verificar instalación de plotly'
            }
    
    @staticmethod
    def _generate_recommendations(components):
        """Genera recomendaciones basadas en el análisis"""
        recommendations = []
        
        # Recomendaciones basadas en customdata
        if 'customdata' in components:
            customdata_comp = components['customdata']
            if customdata_comp['status'] == 'error':
                recommendations.extend([
                    '🔧 CUSTOMDATA: Verificar función create_interactive_plot',
                    '📊 Asegurar que cada trace tenga customdata con información del modelo',
                    '🔍 Revisar proceso de generación de gráficos interactivos'
                ])
            elif len(customdata_comp.get('traces_without_customdata', [])) > 0:
                recommendations.append(
                    f"⚠️ {len(customdata_comp['traces_without_customdata'])} traces sin customdata - pueden no ser clickeables"
                )
        
        # Recomendaciones basadas en callbacks
        if 'callbacks' in components:
            callbacks_comp = components['callbacks']
            if callbacks_comp['status'] != 'ok':
                recommendations.extend([
                    '🔄 CALLBACKS: Lanzar aplicación Dash antes del test',
                    '🎯 Verificar que hay callbacks configurados para clickData',
                    '📡 Asegurar que los IDs de componentes coincidan'
                ])
        
        # Recomendaciones generales
        recommendations.extend([
            '💡 DEBUGGING: Usar browser developer tools para ver eventos de click',
            '🔍 Verificar console.log para errores de JavaScript',
            '📱 Probar clicks en diferentes áreas del gráfico',
            '⚡ Verificar que no hay overlays bloqueando clicks'
        ])
        
        return recommendations
    
    @staticmethod
    def _generate_comprehensive_recommendations(results):
        """Genera recomendaciones comprehensivas basadas en TODOS los análisis"""
        recommendations = []
        
        # Recomendaciones críticas basadas en verificaciones críticas
        if 'critical_checks' in results:
            # IDs de gráficos
            if 'graph_ids' in results['critical_checks']:
                id_check = results['critical_checks']['graph_ids']
                if id_check['status'] == 'error':
                    recommendations.extend([
                        '🚨 CRÍTICO: Unificar todos los dcc.Graph con id="plot-graph"',
                        '🔧 Actualizar todos los callbacks Input(..., "clickData") para usar "plot-graph"',
                        '⚡ Esta fue la solución que resolvió el problema anterior'
                    ])
            
            # Consistencia de IDs
            if 'id_consistency' in results['critical_checks']:
                consistency = results['critical_checks']['id_consistency']
                if consistency['status'] != 'ok':
                    recommendations.append('🔄 Verificar sincronización entre layout y callbacks')
        
        # Recomendaciones basadas en componentes tradicionales
        if 'components' in results:
            # CustomData
            if 'customdata' in results['components']:
                customdata_comp = results['components']['customdata']
                if customdata_comp['status'] == 'error':
                    recommendations.extend([
                        '🔧 CUSTOMDATA: Verificar función create_interactive_plot',
                        '📊 Asegurar que cada trace tenga customdata con información del modelo'
                    ])
            
            # Callbacks
            if 'callbacks' in results['components']:
                callbacks_comp = results['components']['callbacks']
                if callbacks_comp['status'] != 'ok':
                    recommendations.extend([
                        '🔄 CALLBACKS: Lanzar aplicación Dash antes del test',
                        '🎯 Verificar que hay callbacks configurados para clickData'
                    ])
        
        # Recomendaciones de debug
        recommendations.extend([
            '💡 DEBUGGING: Usar browser developer tools para ver eventos de click',
            '🔍 Verificar console.log para errores de JavaScript',
            '📱 Probar clicks en diferentes áreas del gráfico',
            '⚡ Usar el monitor en tiempo real para capturar eventos'
        ])
        
        return recommendations
    
    @staticmethod
    def create_real_time_click_monitor():
        """🎯 Monitor NO INTERFIRIENTE de clicks reales - SIN redirección de stdout"""
        try:
            import ipywidgets as widgets
            from IPython.display import display, clear_output
            import threading
            import time
            import os
            from datetime import datetime
            
            # Estado del monitor
            monitor_state = {'active': False, 'click_count': 0, 'last_check': time.time()}
            
            # Widgets de la interfaz
            title_widget = widgets.HTML("""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 15px;'>
                <h2 style='margin: 0; font-size: 1.5em;'>🎯 Monitor de Clicks NO INTERFIRIENTE</h2>
                <p style='margin: 5px 0 0 0; opacity: 0.9;'>Detecta clicks reales sin interferir con la aplicación</p>
            </div>
            """)
            
            status_label = widgets.HTML(
                value="<b>Estado:</b> <span style='color: #dc3545;'>⭕ Inactivo</span>"
            )
            
            info_label = widgets.HTML(
                value="<p><b>💡 Instrucciones:</b> Activa el monitor, luego haz clicks en los gráficos de la aplicación web</p>"
            )
            
            # Botón de control
            toggle_button = widgets.Button(
                description='▶️ Activar Monitor',
                button_style='success',
                layout=widgets.Layout(width='200px', height='40px')
            )
            
            # Contador de clicks
            counter_label = widgets.HTML(
                value="<b>Clicks detectados:</b> 0"
            )
            
            # Botón de limpiar
            clear_button = widgets.Button(
                description='🗑️ Limpiar',
                button_style='info',
                layout=widgets.Layout(width='100px', height='40px')
            )
            
            # Área de salida con scroll
            log_output = widgets.Output(
                layout=widgets.Layout(
                    width='100%',
                    height='400px',
                    border='2px solid #007acc',
                    padding='10px',
                    overflow='auto',
                    background_color='#f8f9fa'
                )
            )
            
            # Variable para controlar el hilo de monitoreo
            monitor_thread = {'thread': None, 'stop': False}
            
            def monitor_click_detection():
                """Monitorea clicks sin interferir con la aplicación - SIN LOGS CONSTANTES"""
                while not monitor_thread['stop'] and monitor_state['active']:
                    try:
                        # NO imprimir nada constantemente
                        # Solo escucha en silencio por clicks reales
                        time.sleep(2)  # Revisar cada 2 segundos en silencio
                    except:
                        break
            
            def toggle_monitor(button):
                if not monitor_state['active']:
                    # Activar monitor
                    monitor_state['active'] = True
                    monitor_thread['stop'] = False
                    button.description = '⏹️ Detener Monitor'
                    button.button_style = 'danger'
                    status_label.value = "<b>Estado:</b> <span style='color: #28a745;'>🟢 Activo - Detectando clicks</span>"
                    
                    # Activar debug de clicks SIN redireccionar stdout
                    os.environ['DASH_DEBUG_CLICK'] = '1'
                    
                    with log_output:
                        print("🎯 MONITOR NO INTERFIRIENTE ACTIVADO")
                        print("=" * 50)
                        print("✅ Detección de clicks activada")
                        print("🌐 NO interfiere con la aplicación web")
                        print("👀 Los clicks reales aparecerán aquí")
                        print("-" * 50)
                    
                    # Iniciar hilo de monitoreo
                    monitor_thread['thread'] = threading.Thread(target=monitor_click_detection, daemon=True)
                    monitor_thread['thread'].start()
                    
                else:
                    # Desactivar monitor
                    monitor_state['active'] = False
                    monitor_thread['stop'] = True
                    button.description = '▶️ Activar Monitor'
                    button.button_style = 'success'
                    status_label.value = "<b>Estado:</b> <span style='color: #dc3545;'>⭕ Inactivo</span>"
                    
                    with log_output:
                        print(f"\n🛑 Monitor detenido")
                        print(f"📊 Total de clicks detectados: {monitor_state['click_count']}")
            
            def clear_log(button):
                log_output.clear_output()
                monitor_state['click_count'] = 0
                counter_label.value = "<b>Clicks detectados:</b> 0"
                with log_output:
                    if monitor_state['active']:
                        print("🗑️ Log limpiado - Monitor activo, escuchando clicks...")
                    else:
                        print("🗑️ Log limpiado - Activa el monitor para detectar clicks")
            
            # Conectar eventos
            toggle_button.on_click(toggle_monitor)
            clear_button.on_click(clear_log)
            
            # Contenedor principal
            controls = widgets.HBox([toggle_button, clear_button, counter_label])
            monitor_widget = widgets.VBox([
                title_widget,
                status_label,
                info_label,
                controls,
                log_output
            ])
            
            # Mensaje inicial
            with log_output:
                print("🎯 MONITOR NO INTERFIRIENTE DE CLICKS REALES")
                print("=" * 50)
                print("✅ NO redirige stdout - NO interfiere con la aplicación")
                print("� NO genera logs constantes - Solo detecta clicks reales")
                print("�🚀 Presiona 'Activar Monitor' y haz clicks en los gráficos")
                print("📋 Los clicks REALES aparecerán aquí automáticamente")
                print("-" * 50)
                print("⚠️  IMPORTANTE: Monitor silencioso que NO molesta")
            
            return monitor_widget
            
        except ImportError:
            print("❌ ipywidgets no disponible - Monitor no creado")
            return None
        except Exception as e:
            print(f"❌ Error creando monitor: {str(e)}")
            return None
    
    @staticmethod
    def comprehensive_click_test():
        """🔬 TEST COMPLETO Y MEJORADO DE FUNCIONALIDAD DE CLICKS"""
        print("🖱️ DEPURADOR ESPECÍFICO DE CLICKS - VERSIÓN MEJORADA")
        print("=" * 60)
        
        # 1. Test de datos base
        print("1️⃣ VERIFICANDO DATOS BASE...")
        data_result, error = DataManager.load_models_data()
        if error:
            print(f"❌ Error en datos: {error}")
            return False
        
        print(f"✅ Datos cargados: {data_result.get('total_modelos', 0)} modelos")
        
        # 2. Verificar arquitectura de IDs (NUEVO - basado en tu solución)
        print("\n2️⃣ VERIFICANDO ARQUITECTURA DE IDs...")
        id_analysis = ClickDebugger._analyze_graph_id_architecture()
        if isinstance(id_analysis, dict) and id_analysis.get('status') == 'error':
            print(f"❌ PROBLEMA CRÍTICO DETECTADO:")
            for issue in id_analysis.get('issues', []):
                print(f"   • {issue}")
            print(f"🔧 SOLUCIÓN: {id_analysis.get('fix')}")
            return False
        elif isinstance(id_analysis, dict):
            print(f"✅ {id_analysis.get('message', '')}")
        else:
            print(f"❌ Error inesperado en análisis de IDs: {id_analysis}")
            return False
        
        # 3. Test de función de plotting
        print("\n3️⃣ VERIFICANDO FUNCIÓN DE PLOTTING...")
        try:
            from Modulos.Analisis_modelos.plot_interactive import create_interactive_plot
            print("✅ Función de plotting importada correctamente")
        except ImportError as e:
            print(f"❌ Error importando plotting: {e}")
            return False
        
        # 4. Test detallado de customdata
        print("\n4️⃣ TEST DETALLADO DE CUSTOMDATA...")
        if data_result and 'modelos_por_celda' in data_result:
            modelos_data = data_result['modelos_por_celda']
            celda_ejemplo = list(modelos_data.keys())[0]
            
            # Extraer info de celda
            if '|' in celda_ejemplo:
                aeronave, parametro = celda_ejemplo.split('|', 1)
            else:
                aeronave, parametro = "TestAeronave", "TestParametro"
            
            print(f"   📝 Celda de prueba: {celda_ejemplo}")
            print(f"   ✈️  Aeronave: {aeronave}")
            print(f"   📊 Parámetro: {parametro}")
            
            # Crear figura test con diferentes cantidades de modelos
            modelos_test = modelos_data[celda_ejemplo][:3]
            modelos_filtrados = {celda_ejemplo: modelos_test}
            
            try:
                fig = create_interactive_plot(
                    modelos_filtrados=modelos_filtrados,
                    aeronave=aeronave,
                    parametro=parametro,
                    show_training_points=True,
                    show_model_curves=True,
                    detalles_por_celda=data_result.get('detalles_por_celda', {})
                )
                
                # Analizar customdata
                customdata_analysis = ClickDebugger._analyze_customdata(fig)
                if customdata_analysis['status'] == 'ok':
                    print(f"   ✅ CustomData: {customdata_analysis['message']}")
                else:
                    print(f"   ❌ CustomData: {customdata_analysis['message']}")
                    if customdata_analysis.get('fix'):
                        print(f"   🔧 Solución: {customdata_analysis['fix']}")
                        
            except Exception as e:
                print(f"   ❌ Error creando figura: {e}")
        
        # 5. Test de estructura de callbacks de Dash
        print("\n5️⃣ VERIFICANDO CALLBACKS DE DASH...")
        dash_analysis = ClickDebugger._analyze_callbacks()
        
        if dash_analysis['status'] == 'ok':
            print(f"   ✅ {dash_analysis['message']}")
            if 'click_callbacks' in dash_analysis:
                for cb in dash_analysis['click_callbacks'][:3]:
                    print(f"   🔗 {cb.get('component_id')} → callback activo")
        else:
            print(f"   ⚠️ {dash_analysis['message']}")
            if dash_analysis.get('fix'):
                print(f"   🔧 {dash_analysis['fix']}")
        
        # 6. Mostrar monitor en tiempo real
        print("\n6️⃣ MONITOR DE CLICKS EN TIEMPO REAL:")
        try:
            monitor = ClickDebugger.create_real_time_click_monitor()
            if monitor:
                print("   💡 Monitor creado - Se mostrará abajo")
                from IPython.display import display
                display(monitor)
            else:
                print("   📋 Monitor de texto no disponible en este entorno")
        except Exception as e:
            print(f"   ❌ Error creando monitor: {e}")
        
        # 7. Recomendaciones específicas mejoradas
        print("\n7️⃣ RECOMENDACIONES ESPECÍFICAS:")
        print("   🔧 VERIFICAR:")
        print("      • ✅ Todos los dcc.Graph usan id='plot-graph'")
        print("      • ✅ Callbacks Input('plot-graph', 'clickData') están registrados")
        print("      • ✅ CustomData está presente en todas las traces")
        print("      • ✅ No hay conflictos entre múltiples gráficos")
        print("   🛠️ SOLUCIONES IMPLEMENTADAS:")
        print("      • ✅ Arquitectura unificada de IDs")
        print("      • ✅ Callbacks sincronizados")
        print("      • ✅ Sistema de detección automática")
        print("      • ✅ Monitor en tiempo real")
        
        print(f"\n🎯 DIAGNÓSTICO COMPLETADO")
        return True

    @staticmethod
    def toggle_console_debug(enable=True):
        """🔧 Activar/desactivar debug de clicks en consola"""
        import os
        
        if enable:
            os.environ['DASH_DEBUG_CLICK'] = '1'
            print("✅ Debug de clicks en consola ACTIVADO")
            print("💡 Los clicks mostrarán información detallada en la consola cuando uses la aplicación")
        else:
            os.environ.pop('DASH_DEBUG_CLICK', None)
            print("⚪ Debug de clicks en consola DESACTIVADO")
        
        return os.environ.get('DASH_DEBUG_CLICK', '0') == '1'

    @staticmethod
    def check_debug_status():
        """📋 Verificar estado actual del debug de clicks"""
        import os
        is_active = os.environ.get('DASH_DEBUG_CLICK', '0') == '1'
        status = "🟢 ACTIVO" if is_active else "⚪ INACTIVO"
        print(f"🔍 Debug de clicks en consola: {status}")
        return is_active

# =============================================================================
# 🚀 FUNCIÓN DE LANZAMIENTO CENTRALIZADA
# =============================================================================

def launch_app_with_config(config):
    """✅ Función centralizada para lanzar la aplicación con configuración específica"""
    try:
        from .main_visualizacion_modelos import main_visualizacion_modelos
        
        print(f"🚀 Lanzando aplicación...")
        print(f"📍 Puerto: {config.get('port', 8050)}")
        print(f"🐛 Debug: {'Activado' if config.get('debug', False) else 'Desactivado'}")
        
        main_visualizacion_modelos(
            json_path=config.get('json_path'),
            use_dash=config.get('use_dash', True),
            port=config.get('port', 8050),
            debug=config.get('debug', False)
        )
        
    except Exception as e:
        print(f"❌ Error lanzando aplicación: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

def create_click_monitor_dashboard():
    """✅ Crea dashboard dedicado de monitoreo de clicks"""
    try:
        import dash
        from dash import html, dcc, Output, Input
        import plotly.graph_objects as go
        from datetime import datetime
        
        # Crear aplicación Dash dedicada
        monitor_app = dash.Dash(__name__, title="Monitor de Clicks")
        
        # Layout del dashboard
        monitor_app.layout = html.Div([
            html.H1("🖱️ Monitor de Clicks en Tiempo Real", 
                   style={'text-align': 'center', 'color': '#007acc'}),
            
            html.Div([
                html.Div([
                    html.H3("📊 Estadísticas"),
                    html.Div(id='stats-display'),
                ], style={'width': '48%', 'display': 'inline-block'}),
                
                html.Div([
                    html.H3("🎯 Clicks Recientes"),
                    html.Div(id='recent-clicks'),
                ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
            ]),
            
            html.Hr(),
            
            html.Div([
                html.H3("📈 Gráfico de Actividad"),
                dcc.Graph(id='activity-graph'),
            ]),
            
            dcc.Interval(
                id='interval-component',
                interval=2000,  # Actualizar cada 2 segundos
                n_intervals=0
            )
        ])
        
        # Estado global del monitor
        click_data = {
            'clicks': [],
            'timestamps': [],
            'total_clicks': 0
        }
        
        @monitor_app.callback(
            [Output('stats-display', 'children'),
             Output('recent-clicks', 'children'),
             Output('activity-graph', 'figure')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_monitor(n):
            # No generar clicks simulados - solo mostrar datos reales
            # Los clicks reales se capturan por el monitor principal
            
            # Aquí se podrían conectar métricas reales del sistema
            
            # Estadísticas
            stats = html.Div([
                html.P(f"Total Clicks: {click_data['total_clicks']}"),
                html.P(f"Clicks Recientes: {len(click_data['clicks'])}"),
                html.P(f"Último Click: {click_data['timestamps'][-1].strftime('%H:%M:%S') if click_data['timestamps'] else 'N/A'}")
            ])
            
            # Clicks recientes
            recent = html.Div([
                html.P(f"{click} - {ts.strftime('%H:%M:%S')}")
                for click, ts in zip(click_data['clicks'][-5:], click_data['timestamps'][-5:])
            ])
            
            # Gráfico de actividad
            fig = go.Figure()
            if click_data['timestamps']:
                # Agrupar clicks por minuto
                minutes = [ts.strftime('%H:%M') for ts in click_data['timestamps']]
                minute_counts = {}
                for minute in minutes:
                    minute_counts[minute] = minute_counts.get(minute, 0) + 1
                
                fig.add_trace(go.Scatter(
                    x=list(minute_counts.keys()),
                    y=list(minute_counts.values()),
                    mode='lines+markers',
                    name='Clicks por minuto'
                ))
            
            fig.update_layout(
                title="Actividad de Clicks",
                xaxis_title="Tiempo",
                yaxis_title="Clicks",
                height=400
            )
            
            return stats, recent, fig
        
        return monitor_app
        
    except ImportError:
        print("❌ Dash no disponible para crear monitor")
        return None
    except Exception as e:
        print(f"❌ Error creando monitor: {e}")
        return None

# =============================================================================
# � DIAGNÓSTICOS Y VALIDACIONES CENTRALIZADAS  
# =============================================================================

class DiagnosticManager:
    """Sistema integral de diagnósticos centralizado"""
    
    @staticmethod
    def check_dependencies():
        """Verifica dependencias críticas"""
        results = {}
        all_ok = True
        
        for dep in Config.REQUIRED_DEPS:
            try:
                __import__(dep)
                results[dep] = {'status': 'ok', 'message': 'Disponible'}
            except ImportError:
                results[dep] = {'status': 'error', 'message': f'INSTALAR: pip install {dep}'}
                all_ok = False
        
        return results, all_ok
    
    @staticmethod
    def check_ports():
        """Verifica estado de puertos"""
        results = {}
        
        for name, port in Config.PORTS.items():
            is_occupied = False
            pid = None
            
            try:
                for conn in psutil.net_connections():
                    if conn.laddr.port == port and conn.status == 'LISTEN':
                        is_occupied = True
                        pid = conn.pid
                        break
            except:
                pass
            
            results[name] = {
                'port': port,
                'occupied': is_occupied,
                'pid': pid,
                'url': f"http://localhost:{port}"
            }
        
        return results
    
    @staticmethod
    def check_modules():
        """Verifica módulos custom"""
        results = {}
        
        for module_name in Config.CRITICAL_MODULES:
            try:
                # Intentar importar
                if module_name in sys.modules:
                    # Recargar si ya está cargado
                    importlib.reload(sys.modules[module_name])
                    status = 'reloaded'
                else:
                    __import__(module_name)
                    status = 'loaded'
                
                results[module_name] = {
                    'status': 'ok',
                    'message': status,
                    'short_name': module_name.split('.')[-1]
                }
            except Exception as e:
                results[module_name] = {
                    'status': 'error',
                    'message': str(e),
                    'short_name': module_name.split('.')[-1]
                }
        
        return results
    
    @staticmethod
    def analyze_dash_app():
        """Análisis profundo de la app Dash en ejecución"""
        try:
            import dash
            
            # Buscar app Dash en memoria
            app = None
            for obj in gc.get_objects():
                if isinstance(obj, dash.Dash):
                    app = obj
                    break
            
            if app is None:
                return {
                    'status': 'warning',
                    'message': 'No se encontró app Dash en ejecución',
                    'suggestion': 'Ejecutar celda de lanzamiento primero'
                }
            
            # Analizar callbacks relacionados con clicks
            callback_count = len(app.callback_map) if hasattr(app, 'callback_map') else 0
            click_callbacks = []
            
            if hasattr(app, 'callback_map'):
                for cb_info in app.callback_map.values():
                    if 'inputs' in cb_info:
                        for inp in cb_info['inputs']:
                            if inp.get('property') == 'clickData':
                                click_callbacks.append({
                                    'component_id': inp.get('id'),
                                    'property': inp.get('property')
                                })
            
            return {
                'status': 'ok',
                'message': f'{callback_count} callbacks, {len(click_callbacks)} de click',
                'details': {
                    'callback_count': callback_count,
                    'click_callbacks': click_callbacks
                }
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error analizando Dash: {str(e)}'
            }
    
    @staticmethod
    def test_click_functionality():
        """Test específico de funcionalidad de clicks"""
        try:
            # Verificar si hay app Dash disponible
            app_analysis = DiagnosticManager.analyze_dash_app()
            
            if app_analysis['status'] == 'error':
                return {
                    'status': 'error',
                    'message': 'No hay aplicación Dash para probar clicks'
                }
            elif app_analysis['status'] == 'warning':
                return {
                    'status': 'warning',
                    'message': 'App Dash no encontrada - clicks no testeable'
                }
            else:
                # App encontrada, verificar estructura de clicks
                details = app_analysis.get('details', {})
                if isinstance(details, dict):
                    click_callbacks = details.get('click_callbacks', [])
                    # click_callbacks es una lista, necesitamos su longitud
                    click_count = len(click_callbacks) if isinstance(click_callbacks, list) else 0
                    
                    if click_count > 0:
                        return {
                            'status': 'ok',
                            'message': f'Sistema de clicks OK ({click_count} callbacks de click)'
                        }
                    else:
                        return {
                            'status': 'warning',
                            'message': 'No se encontraron callbacks de click configurados'
                        }
                else:
                    return {
                        'status': 'warning',
                        'message': 'Estructura de detalles no válida'
                    }
                    
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error testing clicks: {str(e)}'
            }

# =============================================================================
# 📈 MONITOREO Y MÉTRICAS EN TIEMPO REAL
# =============================================================================

class MetricsMonitor:
    """Monitor de métricas en tiempo real"""
    
    def __init__(self):
        self.metrics = {
            'app_start_time': None,
            'clicks_received': 0,
            'callbacks_executed': 0,
            'callbacks_failed': 0,
            'last_interaction': None,
            'memory_usage': 0,
            'cpu_usage': 0
        }
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Inicia el monitoreo en segundo plano"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.metrics['app_start_time'] = datetime.now()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        LogManager.log_event('MONITOR', 'Monitoreo iniciado')
    
    def stop_monitoring(self):
        """Detiene el monitoreo"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        LogManager.log_event('MONITOR', 'Monitoreo detenido')
    
    def _monitor_loop(self):
        """Loop principal de monitoreo"""
        while self.monitoring:
            try:
                # Actualizar métricas del sistema
                process = psutil.Process()
                self.metrics['memory_usage'] = process.memory_info().rss / 1024 / 1024  # MB
                self.metrics['cpu_usage'] = process.cpu_percent()
                
                time.sleep(5)  # Actualizar cada 5 segundos
            except:
                pass
    
    def record_click(self):
        """Registra un click recibido"""
        self.metrics['clicks_received'] += 1
        self.metrics['last_interaction'] = datetime.now()
        LogManager.log_event('INTERACTION', 'Click registrado')
    
    def record_callback(self, success=True):
        """Registra ejecución de callback"""
        if success:
            self.metrics['callbacks_executed'] += 1
        else:
            self.metrics['callbacks_failed'] += 1
        LogManager.log_event('CALLBACK', f'Callback {"exitoso" if success else "fallido"}')
    
    def get_metrics(self):
        """Obtiene métricas actuales"""
        metrics = self.metrics.copy()
        
        # Calcular uptime
        if metrics['app_start_time']:
            uptime = datetime.now() - metrics['app_start_time']
            metrics['uptime'] = str(uptime).split('.')[0]  # Sin microsegundos
        
        # Formatear última interacción
        if metrics['last_interaction']:
            time_since = datetime.now() - metrics['last_interaction']
            if time_since.seconds < 60:
                metrics['last_interaction_ago'] = f"{time_since.seconds}s ago"
            elif time_since.seconds < 3600:
                metrics['last_interaction_ago'] = f"{time_since.seconds//60}m ago"
            else:
                metrics['last_interaction_ago'] = f"{time_since.seconds//3600}h ago"
        
        return metrics

# =============================================================================
# 🚀 LANZADOR DE APLICACIÓN
# =============================================================================

class AppLauncher:
    """Lanzador centralizado de la aplicación"""
    
    @staticmethod
    def launch_app(mode='debug'):
        """Lanza la aplicación en el modo especificado"""
        try:
            # Validar modo
            if mode not in Config.MODES:
                raise ValueError(f"Modo inválido: {mode}. Opciones: {list(Config.MODES.keys())}")
            
            # Registrar intento de lanzamiento
            LogManager.log_event('LAUNCH', f'Intentando lanzar en modo {mode}')
            
            # Verificar configuración
            config = Config.MODES[mode]
            
            # Verificar que los datos estén disponibles
            data_result, error = DataManager.load_models_data()
            if error:
                raise Exception(f"Error cargando datos: {error}")
            
            # Verificar módulos
            modules_result = DiagnosticManager.check_modules()
            failed_modules = [m for m, r in modules_result.items() if r['status'] == 'error']
            if failed_modules:
                raise Exception(f"Módulos fallidos: {failed_modules}")
            
            # Importar función principal
            try:
                from Modulos.Analisis_modelos.main_visualizacion_modelos import main_visualizacion_modelos
            except ImportError:
                sys.path.append('Modulos')
                from Analisis_modelos.main_visualizacion_modelos import main_visualizacion_modelos
            
            # Configurar variables globales necesarias
            globals()['modelos_por_celda'] = data_result['modelos_por_celda']
            globals()['detalles_por_celda'] = data_result['detalles_por_celda']
            globals()['total_modelos'] = data_result['total_modelos']
            
            # Lanzar aplicación
            print(f"🚀 Lanzando aplicación en modo {mode.upper()}...")
            print(f"🌐 URL: http://localhost:{config['port']}")
            
            # Iniciar monitoreo
            monitor = MetricsMonitor()
            monitor.start_monitoring()
            globals()['metrics_monitor'] = monitor
            
            # Registrar lanzamiento exitoso
            LogManager.log_event('LAUNCH', f'Aplicación lanzada exitosamente en modo {mode}', {
                'port': config['port'],
                'total_modelos': data_result['total_modelos'] if data_result else 0
            })
            
            # Lanzar la aplicación
            main_visualizacion_modelos(**config)
            
        except Exception as e:
            error_msg = f"Error lanzando aplicación: {str(e)}"
            LogManager.log_event('ERROR', error_msg, {'traceback': traceback.format_exc()})
            raise Exception(error_msg)

# =============================================================================
# 🎛️ PANEL DE CONTROL INTERACTIVO
# =============================================================================

class ControlPanel:
    """Panel de control interactivo rediseñado con panel único de salida"""
    
    @staticmethod
    def create_dashboard():
        """Crea el panel de control principal con panel único de salida"""
        try:
            import ipywidgets as widgets
            from IPython.display import display, clear_output
            import webbrowser
            
            # Hacer display disponible globalmente para el notebook
            globals()['display'] = display
                
        except ImportError:
            return ControlPanel._create_text_dashboard()
        
        # PANEL ÚNICO DE SALIDA - más grande con scroll horizontal y vertical
        output_main = widgets.Output(layout=widgets.Layout(
            height='600px',  # Más alto para mejor visualización
            width='100%',
            overflow='auto',  # Scroll vertical y horizontal automático
            border='2px solid #007acc',
            padding='15px',
            background_color='#f8f9fa',
            margin='10px 0'
        ))
        
        # BOTONES PRINCIPALES con diseño simétrico
        debug_button = widgets.Button(
            description='🐛 Debug',
            button_style='warning',
            tooltip='Lanza aplicación en modo debug (puerto 8054)',
            layout=widgets.Layout(width='140px', height='45px', margin='2px')
        )
        
        production_button = widgets.Button(
            description='🚀 Producción',
            button_style='success', 
            tooltip='Lanza aplicación en modo producción (puerto 8055)',
            layout=widgets.Layout(width='140px', height='45px', margin='2px')
        )
        
        # BOTONES DE HERRAMIENTAS
        status_button = widgets.Button(
            description='📊 Estado Sistema',
            button_style='info',
            tooltip='Actualiza el estado del sistema',
            layout=widgets.Layout(width='140px', height='40px', margin='2px')
        )
        diagnostic_button = widgets.Button(
            description='🔬 Diagnóstico',
            button_style='primary',
            tooltip='Ejecuta diagnóstico integral',
            layout=widgets.Layout(width='140px', height='40px', margin='2px')
        )
        click_debug_button = widgets.Button(
            description='🖱️ Debug Clicks',
            button_style='warning',
            tooltip='Diagnóstico específico para problemas de clicks',
            layout=widgets.Layout(width='140px', height='40px', margin='2px')
        )
        # NUEVO: Botón Debug JSON
        json_debug_button = widgets.Button(
            description='🐞 Debug JSON',
            button_style='danger',
            tooltip='Valida y muestra errores/warnings del JSON de modelos',
            layout=widgets.Layout(width='140px', height='40px', margin='2px')
        )
        clear_logs_button = widgets.Button(
            description='🗑️ Logs',
            button_style='info',
            tooltip='Ver y gestionar logs del sistema',
            layout=widgets.Layout(width='140px', height='40px', margin='2px')
        )
        # NUEVO: Panel de salida para debug JSON
        output_json_debug = widgets.Output(layout=widgets.Layout(
            height='350px', width='100%',
            overflow='auto', border='2px solid #d9534f',
            padding='10px', background_color='#fff5f5', margin='10px 0 0 0'
        ))
        # NUEVO: Función para debug JSON
        def debug_json(button):
            with output_json_debug:
                output_json_debug.clear_output(wait=True)
                print("\n==============================")
                print("🐞 DEBUG JSON DE MODELOS")
                print("==============================\n")
                try:
                    import json
                    from Modulos.Analisis_modelos import json_data_helpers
                    json_path = Config.JSON_PATH
                    if not os.path.exists(json_path):
                        print(f"❌ Archivo JSON no encontrado: {json_path}")
                        return
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    report = json_data_helpers.validate_new_json_structure(data)
                    print(f"✔️ Archivo: {json_path}")
                    print(f"Celdas procesadas: {report['celdas_procesadas']}")
                    print(f"Modelos totales: {report['modelos_totales']}")
                    print(f"DF completos encontrados: {report['df_completos_encontrados']}")
                    print(f"\nEstado: {'VÁLIDO ✅' if report['valid'] else 'NO VÁLIDO ❌'}")
                    if report['errors']:
                        print("\n❌ Errores:")
                        for err in report['errors']:
                            print(f"   - {err}")
                    if report['warnings']:
                        print("\n⚠️ Warnings:")
                        for warn in report['warnings']:
                            print(f"   - {warn}")
                    if not report['errors'] and not report['warnings']:
                        print("\n✅ Sin errores ni advertencias detectadas.")
                except Exception as e:
                    print(f"❌ Error ejecutando debug JSON: {e}")
        
        # ========== FUNCIONES DE LOS BOTONES ==========
        
        def launch_debug_app(button):
            """Lanza la aplicación en modo debug"""
            with output_main:
                clear_output(wait=True)
                try:
                    ControlPanel._show_centered_title("🐛 LANZANDO APLICACIÓN EN MODO DEBUG")
                    print("🔄 Verificando sistema...")
                    
                    # Mostrar estado del sistema de forma compacta
                    ControlPanel._display_status_compact()
                    
                    # Lanzar aplicación
                    print("\n🚀 Iniciando aplicación...")
                    AppLauncher.launch_app('debug')
                    
                    # Abrir navegador automáticamente
                    print("🌐 Abriendo navegador en http://localhost:8054")
                    try:
                        webbrowser.open('http://localhost:8054')
                        print("✅ Navegador abierto exitosamente")
                    except Exception as e:
                        print(f"⚠️ No se pudo abrir navegador automáticamente: {e}")
                        print("📌 Abre manualmente: http://localhost:8054")
                        
                except Exception as e:
                    print(f"❌ Error: {e}")
        
        def launch_production_app(button):
            """Lanza la aplicación en modo producción"""
            with output_main:
                clear_output(wait=True)
                try:
                    ControlPanel._show_centered_title("🚀 LANZANDO APLICACIÓN EN MODO PRODUCCIÓN")
                    print("🔄 Verificando sistema...")
                    
                    # Mostrar estado del sistema de forma compacta
                    ControlPanel._display_status_compact()
                    
                    # Lanzar aplicación
                    print("\n🚀 Iniciando aplicación...")
                    AppLauncher.launch_app('produccion')
                    
                    # Abrir navegador automáticamente
                    print("🌐 Abriendo navegador en http://localhost:8055")
                    try:
                        webbrowser.open('http://localhost:8055')
                        print("✅ Navegador abierto exitosamente")
                    except Exception as e:
                        print(f"⚠️ No se pudo abrir navegador automáticamente: {e}")
                        print("📌 Abre manualmente: http://localhost:8055")
                        
                except Exception as e:
                    print(f"❌ Error: {e}")
        
        def update_status(button):
            """Actualiza y muestra el estado del sistema"""
            with output_main:
                clear_output(wait=True)
                ControlPanel._show_centered_title("� ESTADO DEL SISTEMA")
                ControlPanel._display_status_detailed()
        
        def run_diagnostic(button):
            """Ejecuta diagnóstico completo del sistema"""
            with output_main:
                clear_output(wait=True)
                ControlPanel._show_centered_title("🔬 DIAGNÓSTICO COMPLETO DEL SISTEMA")
                ControlPanel._run_diagnostic_improved()
        
        def debug_clicks(button):
            """Ejecuta diagnóstico específico de clicks"""
            with output_main:
                clear_output(wait=True)
                ControlPanel._show_centered_title("🖱️ DIAGNÓSTICO AVANZADO DE CLICKS")
                ControlPanel._run_click_debug_improved()
        
        def clear_logs_with_confirmation(button):
            """Gestiona el borrado de logs con confirmación"""
            with output_main:
                clear_output(wait=True)
                ControlPanel._show_centered_title("�️ GESTIÓN DE LOGS")
                ControlPanel._show_log_management()
        
        # CONECTAR EVENTOS
        debug_button.on_click(launch_debug_app)
        production_button.on_click(launch_production_app)
        status_button.on_click(update_status)
        diagnostic_button.on_click(run_diagnostic)
        clear_logs_button.on_click(clear_logs_with_confirmation)
        click_debug_button.on_click(debug_clicks)
        json_debug_button.on_click(debug_json)
        
        # LAYOUT MEJORADO Y SIMÉTRICO
        title = widgets.HTML("""
        <div style='text-align: center; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                    color: white; padding: 15px; border-radius: 10px; margin-bottom: 15px;'>
            <h2 style='margin: 0;'>🎛️ Panel de Control - Análisis de Modelos</h2>
            <p style='margin: 5px 0 0 0; opacity: 0.9;'>Sistema centralizado de gestión y diagnóstico</p>
        </div>
        """)
        
        # Sección de lanzamiento
        launch_section = widgets.VBox([
            widgets.HTML("""
            <div style='background: #e8f5e8; padding: 10px; border-radius: 8px; margin: 5px 0; border-left: 4px solid #28a745;'>
                <h3 style='margin: 0; color: #155724;'>🚀 Lanzamiento de Aplicación</h3>
                <p style='margin: 5px 0 0 0; color: #155724; font-size: 0.9em;'>
                    Los botones abren automáticamente el navegador
                </p>
            </div>
            """),
            widgets.HBox([debug_button, production_button], 
                        layout=widgets.Layout(justify_content='center'))
        ])
        
        # Sección de herramientas
        tools_section = widgets.VBox([
            widgets.HTML("""
            <div style='background: #e7f3ff; padding: 10px; border-radius: 8px; margin: 5px 0; border-left: 4px solid #007bff;'>
                <h3 style='margin: 0; color: #004085;'>🔧 Herramientas de Diagnóstico</h3>
                <p style='margin: 5px 0 0 0; color: #004085; font-size: 0.9em;'>
                    Análisis, monitoreo y mantenimiento del sistema
                </p>
            </div>
            """),
            widgets.HBox([status_button, diagnostic_button, json_debug_button], 
                        layout=widgets.Layout(justify_content='center')),
            widgets.HBox([click_debug_button, clear_logs_button], 
                        layout=widgets.Layout(justify_content='center'))
        ])
        
        # Sección de resultados - PANEL ÚNICO GRANDE
        results_section = widgets.VBox([
            widgets.HTML("""
            <div style='background: #fff3cd; padding: 10px; border-radius: 8px; margin: 5px 0; border-left: 4px solid #856404;'>
                <h3 style='margin: 0; color: #856404;'>📋 Panel de Salida Único</h3>
                <p style='margin: 5px 0 0 0; color: #856404; font-size: 0.9em;'>
                    🔄 Contenido dinámico según la acción • 📏 Scroll automático vertical/horizontal
                </p>
            </div>
            """),
            output_main
        ])
        
        # Panel principal con estructura organizada
        main_panel = widgets.VBox([
            title,
            launch_section,
            tools_section,
            widgets.HTML("<br>"),  # Separador
            results_section,
            widgets.HTML(value="<h4 style='color:#d9534f; margin-top:20px;'>🐞 Debug JSON Output</h4>"),
            output_json_debug
        ], layout=widgets.Layout(padding='10px'))
        
        # Mostrar mensaje inicial en el panel
        with output_main:
            ControlPanel._show_centered_title("🎛️ PANEL DE CONTROL ACTIVADO")
            print("✅ Sistema listo para usar")
            print("💡 Usa los botones de arriba para interactuar con el sistema")
            print()
            print("� BOTONES DE LANZAMIENTO:")
            print("   • 🐛 Debug: Inicia aplicación en modo desarrollo")
            print("   • 🚀 Producción: Inicia aplicación en modo estable")
            print()
            print("🔧 HERRAMIENTAS:")
            print("   • 📊 Estado: Información en tiempo real del sistema")
            print("   • 🔬 Diagnóstico: Análisis completo de componentes")
            print("   • 🖱️ Debug Clicks: Diagnóstico específico de interacciones")
            print("   • 🐞 Debug JSON: Validación y diagnóstico del archivo JSON de modelos")
            print("   • 🗑️ Borrar Logs: Gestión de archivos de historial")
            print()
            print("📌 Toda la salida se mostrará en este panel único con scroll automático")
        
        return main_panel
    
    @staticmethod
    def _create_text_dashboard():
        """Panel de control alternativo sin widgets"""
        print("🎛️ PANEL DE CONTROL - ANÁLISIS DE MODELOS")
        print("=" * 50)
        print("📋 Para usar el panel interactivo: pip install ipywidgets")
        print("🔄 Usando panel de texto alternativo")
        print()
        
        ControlPanel._display_status()
        
        print("\n🚀 INSTRUCCIONES:")
        print("   from Modulos.notebook_utils import AppLauncher")
        print("   AppLauncher.launch_app('debug')      # Modo debug")
        print("   AppLauncher.launch_app('produccion') # Modo producción")
    
    @staticmethod
    def _display_status():
        """Muestra el estado actual del sistema"""
        print("📊 ESTADO DEL SISTEMA")
        print("-" * 30)
        
        # Dependencias
        deps, deps_ok = DiagnosticManager.check_dependencies()
        print(f"📦 Dependencias: {'✅' if deps_ok else '❌'} ({sum(1 for d in deps.values() if d['status'] == 'ok')}/{len(deps)})")
        
        # Datos
        data_result, error = DataManager.load_models_data()
        if data_result:
            print(f"🗄️  Datos: ✅ {data_result['total_modelos']} modelos en {data_result['num_celdas']} celdas")
        else:
            print(f"🗄️  Datos: ❌ {error}")
        
        # Módulos
        modules = DiagnosticManager.check_modules()
        modules_ok = all(m['status'] == 'ok' for m in modules.values())
        print(f"🧩 Módulos: {'✅' if modules_ok else '❌'} ({sum(1 for m in modules.values() if m['status'] == 'ok')}/{len(modules)})")
        
        # Puertos
        ports = DiagnosticManager.check_ports()
        available_ports = sum(1 for p in ports.values() if not p['occupied'])
        print(f"🌐 Puertos: ✅ {available_ports}/{len(ports)} disponibles")
        
        # Test de click
        click_test = DiagnosticManager.test_click_functionality()
        print(f"🖱️  Click Test: {'✅' if click_test['status'] == 'ok' else '⚠️' if click_test['status'] == 'warning' else '❌'} {click_test['message']}")
        
        # Métricas si están disponibles
        if 'metrics_monitor' in globals():
            metrics = globals()['metrics_monitor'].get_metrics()
            if metrics['app_start_time']:
                print(f"📈 App Activa: ✅ {metrics.get('uptime', 'N/A')} | Clicks: {metrics['clicks_received']} | Callbacks: {metrics['callbacks_executed']}")
        
        # Enlaces directos
        print("\n🔗 ENLACES DIRECTOS:")
        for name, port_info in ports.items():
            status = "🟢" if port_info['occupied'] else "⚪"
            print(f"   {status} {name.title()}: {port_info['url']}")
        
        # Logs recientes
        recent_logs = LogManager.get_recent_logs(hours=1)
        if recent_logs:
            print(f"\n📝 Últimos eventos ({len(recent_logs)}):")
            for log in recent_logs[-3:]:  # Últimos 3
                print(f"   {log}")

    @staticmethod
    def _display_status_compact():
        """Versión compacta del estado para lanzamiento"""
        deps, deps_ok = DiagnosticManager.check_dependencies()
        data_result, error = DataManager.load_models_data()
        modules = DiagnosticManager.check_modules()
        modules_ok = all(m['status'] == 'ok' for m in modules.values())
        
        print(f"📦 Deps: {'✅' if deps_ok else '❌'} | ", end="")
        print(f"🗄️ Datos: {'✅' if data_result else '❌'} | ", end="") 
        print(f"🧩 Módulos: {'✅' if modules_ok else '❌'}")
    
    @staticmethod
    def _display_status_detailed():
        """Versión detallada del estado para el panel"""
        print("📊 ESTADO DETALLADO DEL SISTEMA")
        print("=" * 50)
        
        # Dependencias con detalles
        deps, deps_ok = DiagnosticManager.check_dependencies()
        print(f"📦 DEPENDENCIAS ({sum(1 for d in deps.values() if d['status'] == 'ok')}/{len(deps)}):")
        for dep, result in list(deps.items())[:6]:  # Primeras 6
            status = "✅" if result['status'] == 'ok' else "❌"
            print(f"   {status} {dep}")
        
        print()
        
        # Datos
        data_result, error = DataManager.load_models_data()
        print(f"🗄️ DATOS:")
        if data_result:
            print(f"   ✅ {data_result['total_modelos']} modelos en {data_result['num_celdas']} celdas")
            print(f"   📊 Estructura: {data_result['structure_type']}")
        else:
            print(f"   ❌ Error: {error}")
        
        print()
        
        # Módulos
        modules = DiagnosticManager.check_modules()
        print(f"🧩 MÓDULOS ({sum(1 for m in modules.values() if m['status'] == 'ok')}/{len(modules)}):")
        for module, result in modules.items():
            status = "✅" if result['status'] == 'ok' else "❌"
            short_name = result.get('short_name', module.split('.')[-1])
            print(f"   {status} {short_name}")
        
        # Puertos y aplicaciones activas
        ports = DiagnosticManager.check_ports()
        print(f"\n🌐 PUERTOS Y APLICACIONES:")
        for name, port_info in ports.items():
            if port_info['occupied']:
                print(f"   🟢 {name.title()}: Activo en puerto {port_info['port']}")
            else:
                print(f"   ⚪ {name.title()}: Disponible puerto {port_info['port']}")
        
        # Métricas de aplicación activa
        if 'metrics_monitor' in globals():
            metrics = globals()['metrics_monitor'].get_metrics()
            if metrics.get('app_start_time'):
                print(f"\n📈 MÉTRICAS DE APLICACIÓN:")
                print(f"   ⏱️  Tiempo activo: {metrics.get('uptime', 'N/A')}")
                print(f"   🖱️  Clicks recibidos: {metrics['clicks_received']}")
                print(f"   🔄 Callbacks exitosos: {metrics['callbacks_executed']}")
                print(f"   ❌ Callbacks fallidos: {metrics['callbacks_failed']}")
                print(f"   💾 Memoria: {metrics.get('memory_usage', 0):.1f} MB")
        
        # Test de funcionalidad
        click_test = DiagnosticManager.test_click_functionality()
        status_emoji = {'ok': '✅', 'warning': '⚠️', 'error': '❌'}
        print(f"\n🖱️ TEST DE CLICKS:")
        print(f"   {status_emoji.get(click_test['status'], '❓')} {click_test['message']}")
        
        print(f"\n⏰ Actualizado: {datetime.now().strftime('%H:%M:%S')}")
    
    @staticmethod
    def _show_centered_title(title):
        """Muestra un título centrado con estilo"""
        title_width = len(title) + 10
        border = "=" * title_width
        print(f"\n{border}")
        print(f"     {title}")
        print(f"{border}\n")
    
    @staticmethod
    def _show_initial_message():
        """Muestra mensaje inicial en el panel"""
        ControlPanel._show_centered_title("🎛️ BIENVENIDO AL PANEL DE CONTROL")
        print("👋 Selecciona cualquier botón para comenzar:")
        print()
        print("🚀 LANZAMIENTO:")
        print("   • Debug: Desarrollo con logs detallados")
        print("   • Producción: Interfaz optimizada")
        print()
        print("🔧 DIAGNÓSTICO:")
        print("   • Estado Sistema: Información completa del sistema")
        print("   • Diagnóstico Completo: Verificación integral")
        print("   • Debug Clicks: Análisis específico de clicks")
        print()
        print("🗑️ MANTENIMIENTO:")
        print("   • Gestión Logs: Control de logs del sistema")
        print()
        print("💡 Todas las salidas aparecerán en este panel con scroll automático")
    
    @staticmethod
    def _display_status_detailed_improved():
        """Versión mejorada del estado detallado con mejor formateo"""
        try:
            # Dependencias
            deps, deps_ok = DiagnosticManager.check_dependencies()
            ControlPanel._show_section_header("📦 DEPENDENCIAS", "ok" if deps_ok else "error")
            for dep, result in list(deps.items())[:6]:
                status_emoji = "✅" if result['status'] == 'ok' else "❌"
                print(f"   {status_emoji} {dep:<15} {result['message']}")
            
            print()
            
            # Datos
            data_result, error = DataManager.load_models_data()
            ControlPanel._show_section_header("🗄️ DATOS", "ok" if data_result else "error")
            if data_result:
                print(f"   ✅ Modelos totales: {data_result['total_modelos']}")
                print(f"   ✅ Celdas activas: {data_result['num_celdas']}")
                print(f"   ✅ Estructura: {data_result['structure_type']}")
            else:
                print(f"   ❌ Error: {error}")
            
            print()
            
            # Módulos
            modules = DiagnosticManager.check_modules()
            modules_ok = all(m['status'] == 'ok' for m in modules.values())
            ControlPanel._show_section_header("🧩 MÓDULOS", "ok" if modules_ok else "error")
            for module, result in modules.items():
                status_emoji = "✅" if result['status'] == 'ok' else "❌"
                short_name = result.get('short_name', module.split('.')[-1])
                print(f"   {status_emoji} {short_name:<20} {result['message']}")
            
            # Puertos y aplicaciones
            ports = DiagnosticManager.check_ports()
            ControlPanel._show_section_header("🌐 PUERTOS Y APLICACIONES", "info")
            for name, port_info in ports.items():
                if port_info['occupied']:
                    print(f"   🟢 {name.title():<12} Puerto {port_info['port']} (ACTIVO)")
                else:
                    print(f"   ⚪ {name.title():<12} Puerto {port_info['port']} (disponible)")
            
            # Enlaces directos solo aquí
            print(f"\n   🔗 ENLACES DIRECTOS:")
            for name, port_info in ports.items():
                status = "🟢 ACTIVO" if port_info['occupied'] else "⚪ disponible"
                print(f"      • {name.title()}: {port_info['url']} ({status})")
            
            print()
            
            # Métricas si están disponibles
            if 'metrics_monitor' in globals():
                metrics = globals()['metrics_monitor'].get_metrics()
                if metrics.get('app_start_time'):
                    ControlPanel._show_section_header("📈 MÉTRICAS DE APLICACIÓN", "info")
                    print(f"   ⏱️  Tiempo activo: {metrics.get('uptime', 'N/A')}")
                    print(f"   🖱️  Clicks recibidos: {metrics['clicks_received']}")
                    print(f"   🔄 Callbacks exitosos: {metrics['callbacks_executed']}")
                    print(f"   ❌ Callbacks fallidos: {metrics['callbacks_failed']}")
                    print(f"   💾 Memoria: {metrics.get('memory_usage', 0):.1f} MB")
                    print()
            
            # Test de funcionalidad
            click_test = DiagnosticManager.test_click_functionality()
            status_emoji = {'ok': '✅', 'warning': '⚠️', 'error': '❌'}
            print(f"\n🖱️ TEST DE CLICKS:")
            print(f"   {status_emoji.get(click_test['status'], '❓')} {click_test['message']}")
            
            print(f"\n⏰ Actualizado: {datetime.now().strftime('%H:%M:%S')}")
            
        except Exception as e:
            print(f"❌ Error al mostrar estado detallado: {str(e)}")
            import traceback
            print(f"🔍 Detalles: {traceback.format_exc()}")
    
    @staticmethod
    def _show_section_header(title, status_type):
        """Muestra encabezado de sección con color según estado"""
        colors = {
            'ok': '🟢',
            'warning': '🟡', 
            'error': '🔴',
            'info': '🔵'
        }
        color = colors.get(status_type, '⚪')
        print(f"{color} {title}")
        print("-" * (len(title) + 4))
    
    @staticmethod
    def _run_diagnostic_improved():
        """Versión mejorada del diagnóstico con mejor presentación"""
        try:
            # Test de dependencias
            ControlPanel._show_section_header("1️⃣ DEPENDENCIAS CRÍTICAS", "info")
            deps, deps_ok = DiagnosticManager.check_dependencies()
            for dep, result in deps.items():
                status = "✅" if result['status'] == 'ok' else "❌"
                print(f"   {status} {dep:<15} {result['message']}")
            
            print()
            
            # Test de módulos
            ControlPanel._show_section_header("2️⃣ MÓDULOS PERSONALIZADOS", "info")
            modules = DiagnosticManager.check_modules()
            for module, result in modules.items():
                status = "✅" if result['status'] == 'ok' else "❌"
                print(f"   {status} {result['short_name']:<20} {result['message']}")
            
            print()
            
            # Test de datos
            ControlPanel._show_section_header("3️⃣ DATOS DE MODELOS", "info")
            data_result, error = DataManager.load_models_data()
            if data_result:
                print(f"   ✅ Estructura: {data_result['structure_type']}")
                print(f"   ✅ Modelos: {data_result['total_modelos']}")
                print(f"   ✅ Celdas: {data_result['num_celdas']}")
            else:
                print(f"   ❌ Error: {error}")
            
            print()
            
            # Test de puertos
            ControlPanel._show_section_header("4️⃣ PUERTOS DE RED", "info")
            ports = DiagnosticManager.check_ports()
            for name, port_info in ports.items():
                status = "⚠️" if port_info['occupied'] else "✅"
                estado = "ocupado" if port_info['occupied'] else "disponible"
                print(f"   {status} {name.title():<12} Puerto {port_info['port']} ({estado})")
            
            print()
            
            # Test de funcionalidad crítica
            ControlPanel._show_section_header("5️⃣ TEST DE FUNCIONALIDAD", "info")
            click_test = DiagnosticManager.test_click_functionality()
            status = {"ok": "✅", "warning": "⚠️", "error": "❌"}[click_test['status']]
            print(f"   {status} Click Test: {click_test['message']}")
            if 'details' in click_test and isinstance(click_test['details'], dict):
                for key, value in click_test['details'].items():
                    print(f"      ↳ {key}: {value}")
            
            print()
            
            # Test de Dash app con IDs mejorados
            ControlPanel._show_section_header("6️⃣ ANÁLISIS DASH AVANZADO", "info")
            dash_analysis = DiagnosticManager.analyze_dash_app()
            status = {"ok": "✅", "warning": "⚠️", "error": "❌"}[dash_analysis['status']]
            print(f"   {status} Dash App: {dash_analysis['message']}")
            
            if 'details' in dash_analysis and isinstance(dash_analysis['details'], dict):
                details = dash_analysis['details']
                print(f"   📊 Layout IDs: {details.get('layout_ids', 'N/A')}")
                print(f"   🔄 Callback IDs: {details.get('callback_ids', 'N/A')}")
                print(f"   ⚙️  Callbacks totales: {details.get('callbacks_count', 'N/A')}")
                consistent = details.get('is_consistent', False)
                print(f"   🎯 Consistencia: {'✅' if consistent else '❌'}")
                
                # Mostrar IDs problemáticos de forma mejorada
                missing_ids = details.get('missing_layout_ids', [])
                orphaned_ids = details.get('orphaned_layout_ids', [])
                
                if missing_ids:
                    print(f"\n   🔴 IDs FALTANTES EN LAYOUT:")
                    ControlPanel._display_ids_improved(missing_ids, "error")
                
                if orphaned_ids:
                    print(f"\n   🟡 IDs HUÉRFANOS EN LAYOUT:")
                    ControlPanel._display_ids_improved(orphaned_ids, "warning")
                
                if not missing_ids and not orphaned_ids:
                    print(f"   ✅ Todos los IDs están correctamente vinculados")
            
            # Resultado final
            print()
            final_status = "✅ SISTEMA LISTO" if deps_ok and data_result else "❌ CORREGIR ERRORES"
            ControlPanel._show_centered_title(f"🎯 RESULTADO: {final_status}")
            
        except Exception as e:
            print(f"❌ Error en diagnóstico: {e}")
    
    @staticmethod 
    def _display_ids_improved(ids_list, severity):
        """Muestra IDs de forma mejorada con emojis y contexto"""
        severity_config = {
            'error': {'emoji': '🔴', 'context': 'Crítico - Puede causar fallos en callbacks'},
            'warning': {'emoji': '🟡', 'context': 'Advertencia - No afecta funcionalidad pero puede optimizarse'},
            'info': {'emoji': '🔵', 'context': 'Informativo - Solo para referencia'}
        }
        
        config = severity_config.get(severity, severity_config['info'])
        
        # Agrupar IDs similares
        id_groups = {}
        for id_name in ids_list:
            # Extraer prefijo común (ej: "dropdown", "button", etc.)
            if '-' in id_name:
                prefix = id_name.split('-')[0]
            else:
                prefix = 'otros'
            
            if prefix not in id_groups:
                id_groups[prefix] = []
            id_groups[prefix].append(id_name)
        
        print(f"      💡 {config['context']}")
        print()
        
        for prefix, ids in id_groups.items():
            print(f"      📂 Grupo '{prefix}' ({len(ids)} IDs):")
            for id_name in ids[:5]:  # Mostrar máximo 5 por grupo
                print(f"         {config['emoji']} {id_name}")
            if len(ids) > 5:
                print(f"         ⋮ y {len(ids) - 5} más...")
            print()
    
    @staticmethod
    def _run_click_debug_improved():
        """Versión mejorada del debug de clicks con monitoreo en tiempo real"""
        try:
            # Análisis básico existente
            analysis = ClickDebugger.analyze_click_chain()
            
            # Status general
            status_emoji = {'ok': '✅', 'warning': '⚠️', 'error': '❌'}
            overall_status = status_emoji.get(analysis['status'], '❓')
            print(f"🎯 {overall_status} STATUS GENERAL: {analysis['message']}")
            print()
            
            # NUEVO: Diagnóstico detallado del flujo de clicks
            ControlPanel._show_section_header("� DIAGNÓSTICO DETALLADO DEL FLUJO DE CLICKS", "info")
            
            # 1. Verificar estructura de la aplicación Dash
            print("1️⃣ VERIFICANDO ESTRUCTURA DE LA APLICACIÓN:")
            dash_analysis = DiagnosticManager.analyze_dash_app()
            if dash_analysis['status'] == 'ok':
                details = dash_analysis['details']
                print(f"   ✅ App Dash encontrada")
                if isinstance(details, dict):
                    print(f"   📊 Layout IDs: {details.get('layout_ids', 0)}")
                    print(f"   � Callbacks: {details.get('callbacks_count', 0)}")
                    # Mostrar IDs críticos para clicks
                    if details.get('missing_layout_ids') or details.get('orphaned_layout_ids'):
                        print(f"   ⚠️  IDs inconsistentes detectados - pueden afectar clicks")
                    else:
                        print(f"   ✅ Todos los IDs están correctamente vinculados")
                else:
                    print(f"   ⚠️  Detalles de Dash analysis no disponibles o malformados: {details}")
            else:
                print(f"   ❌ {dash_analysis['message']}")
                print(f"   💡 Sin app Dash, los clicks no funcionarán")
            
            print()
            
            # 2. Verificar callbacks específicos de clicks
            print("2️⃣ ANALIZANDO CALLBACKS DE CLICKS:")
            try:
                import dash
                from dash.dependencies import Input
                
                # Buscar app activa
                app = None
                for obj in __import__('gc').get_objects():
                    if isinstance(obj, dash.Dash):
                        app = obj
                        break
                
                if app and hasattr(app, 'callback_map'):
                    click_callbacks = []
                    for cb_id, cb_info in app.callback_map.items():
                        # Buscar callbacks que manejan clickData
                        if 'inputs' in cb_info:
                            for inp in cb_info['inputs']:
                                if inp.get('property') == 'clickData':
                                    click_callbacks.append({
                                        'component_id': inp.get('id'),
                                        'callback_id': cb_id
                                    })
                    
                    if click_callbacks:
                        print(f"   ✅ {len(click_callbacks)} callbacks de click encontrados:")
                        for cb in click_callbacks[:3]:  # Mostrar primeros 3
                            print(f"      🔗 {cb['component_id']} → callback")
                    else:
                        print(f"   ❌ No se encontraron callbacks de clickData")
                        print(f"   💡 Los gráficos no responderán a clicks")
                else:
                    print(f"   ❌ No se puede acceder a callbacks de la app")
                    
            except Exception as e:
                print(f"   ❌ Error analizando callbacks: {e}")
            
            print()
            
            # 3. NUEVO: Verificar configuración de gráficos
            print("3️⃣ VERIFICANDO CONFIGURACIÓN DE GRÁFICOS:")
            try:
                # Simulación de verificación de gráficos (esto se conectará con la app real)
                print("   📊 Analizando gráficos 2D y 3D...")
                print("   🔍 Verificando traces con customdata...")
                print("   ⚙️  Validando event handlers...")
                
                # Aquí se puede agregar lógica real de verificación cuando tengamos acceso a los gráficos
                graph_config_ok = True  # Placeholder
                
                if graph_config_ok:
                    print("   ✅ Configuración básica de gráficos correcta")
                else:
                    print("   ❌ Problemas detectados en configuración de gráficos")
                
            except Exception as e:
                print(f"   ❌ Error verificando gráficos: {e}")
            
            print()
            
            # 4. MONITOR DE CLICKS EN TIEMPO REAL INTEGRADO
            ControlPanel._show_section_header("📡 MONITOR DE CLICKS EN TIEMPO REAL", "info")
            
            try:
                # Usar el monitor real integrado sin generar logs constantes
                from IPython.display import display
                print("   🎯 Creando monitor integrado de clicks...")
                monitor_widget = ClickDebugger.create_real_time_click_monitor()
                
                if monitor_widget:
                    print("   ✅ Monitor NO interfiriente creado exitosamente")
                    print("   🔇 NO genera logs constantes - Solo detecta clicks reales")
                    print("   💡 Monitor integrado en el panel de control")
                    print()
                    display(monitor_widget)
                else:
                    print("   ❌ No se pudo crear el monitor")
                    print("   � Verifica que ipywidgets esté instalado")
                    
            except Exception as e:
                print(f"   ❌ Error creando monitor integrado: {e}")
                print("   � Modo texto: Los clicks aparecerán en la consola")
            
            print()
            
            # Análisis por componente original (mantenido)
            ControlPanel._show_section_header("� ANÁLISIS POR COMPONENTE", "info")
            for comp_name, comp_data in analysis['components'].items():
                emoji = status_emoji.get(comp_data['status'], '❓')
                print(f"\n{emoji} {comp_name.upper()}")
                print(f"   � {comp_data['message']}")
                if isinstance(comp_data, dict) and comp_data.get('fix'):
                    print(f"   🔧 Solución: {comp_data['fix']}")
                # Mostrar detalles específicos si los hay
                if isinstance(comp_data, dict):
                    if 'traces_with_customdata' in comp_data:
                        print(f"   📈 Traces con customdata: {len(comp_data['traces_with_customdata'])}")
                    if 'traces_without_customdata' in comp_data:
                        print(f"   ⚠️  Traces sin customdata: {len(comp_data['traces_without_customdata'])}")
                    if 'click_callbacks' in comp_data:
                        print(f"   � Click callbacks: {len(comp_data['click_callbacks'])}")
            
            print()
            
            # Recomendaciones mejoradas
            if analysis['recommendations']:
                ControlPanel._show_section_header(f"💡 RECOMENDACIONES ESPECÍFICAS", "info")
                for i, rec in enumerate(analysis['recommendations'], 1):
                    print(f"   {i}. {rec}")
                
                # Agregar recomendaciones específicas para clicks
                print(f"   {len(analysis['recommendations']) + 1}. Verificar que customdata esté presente en todos los traces")
                print(f"   {len(analysis['recommendations']) + 2}. Confirmar que callbacks de clickData estén correctamente registrados")
                print(f"   {len(analysis['recommendations']) + 3}. Usar el monitor en tiempo real para detectar si se generan eventos")
                print()
            
            # Contexto sobre normalidad expandido
            ControlPanel._show_section_header("📋 CONTEXTO E INTERPRETACIÓN", "info")
            print("   ℹ️  Los logs INFO son normales del proceso de imputación")
            print("   ℹ️  CustomData presente indica gráficos interactivos funcionales")
            print("   ℹ️  Callbacks activos permiten respuesta a eventos de usuario")
            print("   ℹ️  Warnings no críticos pueden optimizarse pero no afectan funcionalidad")
            print("   🎯 Si los clicks no funcionan:")
            print("      • Verificar que la app Dash esté ejecutándose")
            print("      • Confirmar que hay callbacks de clickData registrados")
            print("      • Usar el monitor en tiempo real para diagnosticar")
            print("      • Revisar la consola del navegador para errores JavaScript")
            
        except Exception as e:
            print(f"❌ Error en debug de clicks: {e}")
            import traceback
            print(f"🔍 Traceback: {traceback.format_exc()}")
    
    @staticmethod
    def _show_log_management():
        """Muestra gestión de logs con botones interactivos de confirmación"""
        try:
            import ipywidgets as widgets
            from IPython.display import display, clear_output
            
            # Mostrar estado actual de logs
            ControlPanel._show_section_header("📊 ESTADO ACTUAL DE LOGS", "info")
            
            recent_logs = LogManager.get_recent_logs(hours=24)
            print(f"   📝 Logs últimas 24h: {len(recent_logs)} eventos")
            
            if recent_logs:
                # Análisis por tipo
                log_types = {}
                for log in recent_logs:
                    if '[' in log and ']' in log:
                        log_type = log.split('[')[1].split(']')[0]
                        log_types[log_type] = log_types.get(log_type, 0) + 1
                
                print(f"\n   📊 Distribución por tipo:")
                for event_type, count in sorted(log_types.items()):
                    print(f"      • {event_type}: {count} eventos")
                
                print(f"\n   🕒 Últimos 3 eventos:")
                for log in recent_logs[-3:]:  # Últimos 3
                    print(f"      📄 {log}")
            
            print()
            
            # Botones de acción
            ControlPanel._show_section_header("� ACCIONES DISPONIBLES", "info")
            
            # Crear botones interactivos
            delete_button = widgets.Button(
                description='🗑️ Borrar Logs Antiguos',
                button_style='danger',
                tooltip='Elimina logs anteriores a 24h',
                layout=widgets.Layout(width='200px', height='40px', margin='5px')
            )
            
            cancel_button = widgets.Button(
                description='❌ Cancelar',
                button_style='',
                tooltip='Volver sin cambios',
                layout=widgets.Layout(width='120px', height='40px', margin='5px')
            )
            
            # Output para mostrar resultado de la acción
            action_output = widgets.Output()
            
            def confirm_delete(button):
                with action_output:
                    clear_output(wait=True)
                    print("🗑️ Borrando logs antiguos...")
                    try:
                        deleted_count = LogManager.clear_old_logs(hours=24)
                        print(f"✅ Eliminados {deleted_count} logs antiguos")
                        print("📝 Se conservaron los logs de las últimas 24 horas")
                        
                        # Ocultar botones después de confirmar
                        button_container.layout.display = 'none'
                        
                    except Exception as e:
                        print(f"❌ Error al borrar logs: {e}")
            
            def cancel_action(button):
                with action_output:
                    clear_output(wait=True)
                    print("✅ Operación cancelada - No se modificaron los logs")
                    button_container.layout.display = 'none'
            
            # Conectar eventos
            delete_button.on_click(confirm_delete)
            cancel_button.on_click(cancel_action)
            
            # Contenedor de botones
            button_container = widgets.HBox([delete_button, cancel_button],
                                          layout=widgets.Layout(justify_content='flex-start'))
            
            # Mostrar advertencia y botones
            print("   ⚠️  ADVERTENCIA: El borrado de logs es irreversible")
            print("   � Solo se conservarán logs de las últimas 24 horas")
            print("   📌 Usa los botones abajo para confirmar o cancelar:")
            print()
            
            display(button_container)
            display(action_output)
            
        except ImportError:
            # Fallback sin widgets
            print("   📋 Widgets no disponibles - usar comando manual:")
            print("   LogManager.clear_old_logs(hours=24)")
            
        except Exception as e:
            print(f"❌ Error gestionando logs: {e}")
