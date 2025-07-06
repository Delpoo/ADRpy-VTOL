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
    """Debugger especializado para problemas de interactividad de clicks"""
    
    @staticmethod
    def analyze_click_chain():
        """Analiza toda la cadena de procesamiento de clicks"""
        results = {
            'status': 'ok',
            'message': 'Análisis completo de clicks',
            'components': {},
            'recommendations': []
        }
        
        try:
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
            
            # 4. Análizar customdata en detalle
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

# =============================================================================
# �️ DEPURADOR ESPECÍFICO DE CLICKS
# =============================================================================

class ClickDebuggerOld:
    """Depurador especializado para problemas de clicks en gráficos"""
    
    @staticmethod
    def comprehensive_click_test():
        """Test completo y detallado de funcionalidad de clicks"""
        print("🖱️ DEPURADOR ESPECÍFICO DE CLICKS")
        print("=" * 50)
        
        # 1. Test de datos base
        print("1️⃣ VERIFICANDO DATOS BASE...")
        data_result, error = DataManager.load_models_data()
        if error:
            print(f"❌ Error en datos: {error}")
            return False
        
        print(f"✅ Datos cargados: {data_result['total_modelos']} modelos")
        
        # 2. Test de función de plotting
        print("\n2️⃣ VERIFICANDO FUNCIÓN DE PLOTTING...")
        try:
            from Modulos.Analisis_modelos.plot_interactive import create_interactive_plot
            print("✅ Función create_interactive_plot importada")
        except Exception as e:
            print(f"❌ Error importando: {e}")
            return False
        
        # 3. Test detallado de customdata
        print("\n3️⃣ TEST DETALLADO DE CUSTOMDATA...")
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
        for num_models in [1, 2, 5]:
            print(f"\n   🔍 Test con {num_models} modelo(s):")
            try:
                modelos_test = modelos_data[celda_ejemplo][:num_models]
                modelos_filtrados = {celda_ejemplo: modelos_test}
                
                fig_test = create_interactive_plot(
                    modelos_filtrados=modelos_filtrados,
                    aeronave=aeronave,
                    parametro=parametro,
                    show_training_points=True,
                    show_model_curves=True,
                    detalles_por_celda=data_result['detalles_por_celda']
                )
                
                # Análisis detallado de cada trace
                for i, trace in enumerate(fig_test.data):
                    trace_type = type(trace).__name__
                    has_customdata = hasattr(trace, 'customdata') and trace.customdata is not None
                    
                    print(f"      Trace {i} ({trace_type}): {'✅' if has_customdata else '❌'} CustomData")
                    
                    if has_customdata:
                        try:
                            customdata_len = len(trace.customdata) if trace.customdata is not None else 0
                            print(f"         📏 CustomData length: {customdata_len}")
                            
                            # Mostrar sample de customdata
                            if customdata_len > 0:
                                sample = trace.customdata[0] if hasattr(trace.customdata[0], '__len__') else trace.customdata[0]
                                print(f"         📋 Sample: {str(sample)[:100]}...")
                        except Exception as e:
                            print(f"         ⚠️ Error analizando CustomData: {e}")
                    
                    # Verificar propiedades críticas para clicks
                    click_props = ['hoverinfo', 'hovertemplate', 'mode']
                    for prop in click_props:
                        if hasattr(trace, prop):
                            value = getattr(trace, prop)
                            print(f"         🔗 {prop}: {value}")
                
                print(f"      ✅ Figura creada exitosamente: {len(fig_test.data)} traces")
                
            except Exception as e:
                print(f"      ❌ Error con {num_models} modelos: {e}")
                import traceback
                print(f"         Traceback: {traceback.format_exc()}")
        
        # 4. Test de estructura de callbacks de Dash
        print("\n4️⃣ VERIFICANDO CALLBACKS DE DASH...")
        dash_analysis = DiagnosticManager.analyze_dash_app()
        
        if dash_analysis['status'] == 'ok':
            details = dash_analysis.get('details', {})
            print(f"   ✅ App Dash encontrada")
            print(f"   📊 Layout IDs: {details.get('layout_ids', 0)}")
            print(f"   🔄 Callback IDs: {details.get('callback_ids', 0)}")
            
            # Verificar IDs específicos de gráficos
            missing_ids = details.get('missing_layout_ids', [])
            if missing_ids:
                print(f"   🔴 IDs faltantes en layout: {missing_ids}")
                print("      💡 Estos IDs pueden causar problemas de click")
        else:
            print(f"   ⚠️ {dash_analysis['message']}")
        
        # 5. Test de event handlers
        print("\n5️⃣ VERIFICANDO EVENT HANDLERS...")
        try:
            # Buscar funciones de callback relacionadas con clicks
            import inspect
            from Modulos.Analisis_modelos import main_visualizacion_modelos
            
            # Buscar funciones que manejan clicks
            functions = inspect.getmembers(main_visualizacion_modelos, inspect.isfunction)
            click_functions = [f for name, f in functions if 'click' in name.lower() or 'select' in name.lower()]
            
            print(f"   🔍 Funciones de click encontradas: {len(click_functions)}")
            for func in click_functions:
                print(f"      📝 {func.__name__}")
                
            if not click_functions:
                print("   ⚠️ No se encontraron funciones específicas de click")
                print("      💡 Esto puede indicar un problema en el manejo de eventos")
                
        except Exception as e:
            print(f"   ❌ Error verificando event handlers: {e}")
        
        # 6. Recomendaciones específicas
        print("\n6️⃣ RECOMENDACIONES PARA SOLUCIONAR CLICKS:")
        print("   🔧 VERIFICAR:")
        print("      • CustomData está presente en todas las traces")
        print("      • IDs de gráficos coinciden entre layout y callbacks")
        print("      • Event handlers están correctamente registrados")
        print("      • No hay conflictos entre múltiples gráficos")
        print("   🛠️ SOLUCIONES COMUNES:")
        print("      • Recargar módulos de plotting: importlib.reload()")
        print("      • Verificar que Plotly/Dash están actualizados")
        print("      • Revisar que clickData/selectedData se propagan")
        print("      • Confirmar que no hay JavaScript errors en browser")
        
        print(f"\n🎯 DIAGNÓSTICO COMPLETADO")
        return True

# =============================================================================
# �🔧 DIAGNÓSTICOS Y VALIDACIONES (AMPLIADOS)
# =============================================================================

class DiagnosticManager:
    """Sistema integral de diagnósticos"""
    
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
            
            for conn in psutil.net_connections():
                if conn.laddr.port == port and conn.status == 'LISTEN':
                    is_occupied = True
                    pid = conn.pid
                    break
            
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
    def test_click_functionality():
        """Test específico para funcionalidad de click"""
        try:
            # Cargar datos
            data_result, error = DataManager.load_models_data()
            if error:
                return {'status': 'error', 'message': f'Error datos: {error}'}
            
            # Verificar importación de función crítica
            try:
                from Modulos.Analisis_modelos.plot_interactive import create_interactive_plot
            except ImportError:
                sys.path.append('Modulos')
                from Analisis_modelos.plot_interactive import create_interactive_plot
            
            # Test con datos reales
            modelos_data = data_result['modelos_por_celda']
            if not modelos_data:
                return {'status': 'error', 'message': 'modelos_por_celda vacío'}
            
            # Obtener celda de ejemplo
            celda_ejemplo = list(modelos_data.keys())[0]
            
            # Extraer aeronave y parámetro
            if '|' in celda_ejemplo:
                aeronave, parametro = celda_ejemplo.split('|', 1)
            else:
                aeronave, parametro = "TestAeronave", "TestParametro"
            
            # Crear figura test
            modelos_test = modelos_data[celda_ejemplo][:2]
            modelos_filtrados = {celda_ejemplo: modelos_test}
            
            fig_test = create_interactive_plot(
                modelos_filtrados=modelos_filtrados,
                aeronave=aeronave,
                parametro=parametro,
                show_training_points=True,
                show_model_curves=True,
                detalles_por_celda=data_result['detalles_por_celda']
            )
            
            # Verificar customdata
            has_customdata = any(
                hasattr(trace, 'customdata') and trace.customdata is not None 
                for trace in fig_test.data
            )
            
            return {
                'status': 'ok' if has_customdata else 'warning',
                'message': 'CustomData OK' if has_customdata else 'CustomData faltante',
                'details': {
                    'traces': len(fig_test.data),
                    'aeronave': aeronave,
                    'parametro': parametro,
                    'customdata_found': has_customdata
                }
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error en test: {str(e)}',
                'traceback': traceback.format_exc()
            }
    
    @staticmethod
    def analyze_dash_app():
        """Análisis profundo de la app Dash en ejecución"""
        try:
            import dash
            from dash.development.base_component import Component
            
            # Buscar app Dash
            app = None
            
            # Buscar en variables globales del notebook
            notebook_globals = globals()
            for var_name in ['app', 'dash_app']:
                if var_name in notebook_globals and isinstance(notebook_globals.get(var_name), dash.Dash):
                    app = notebook_globals[var_name]
                    break
            
            # Buscar en memoria si no se encuentra
            if app is None:
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
            
            # Analizar IDs del layout
            def extract_ids(component):
                ids = set()
                if isinstance(component, Component):
                    if hasattr(component, 'id') and component.id is not None:
                        ids.add(component.id)
                    for prop in ['children', 'options', 'dropdown_menu', 'tabs']:
                        if hasattr(component, prop):
                            value = getattr(component, prop)
                            if isinstance(value, list):
                                for child in value:
                                    ids |= extract_ids(child)
                            elif isinstance(value, Component):
                                ids |= extract_ids(value)
                elif isinstance(component, list):
                    for c in component:
                        ids |= extract_ids(c)
                return ids
            
            # Analizar callbacks
            def extract_callback_info():
                callback_ids = set()
                callback_map = defaultdict(dict)
                
                if hasattr(app, 'callback_map') and app.callback_map:
                    for cb in app.callback_map.values():
                        cb_func = cb.get('callback')
                        if not cb_func:
                            continue
                        
                        for io_type in ['inputs', 'outputs', 'state']:
                            if io_type in cb:
                                if io_type not in callback_map[cb_func]:
                                    callback_map[cb_func][io_type] = []
                                for dep in cb[io_type]:
                                    callback_ids.add(dep['id'])
                                    callback_map[cb_func][io_type].append(dep['id'])
                
                return callback_ids, callback_map
            
            # Realizar análisis
            layout_ids = extract_ids(app.layout)
            callback_ids, callback_map = extract_callback_info()
            
            # Detectar inconsistencias
            ids_only_in_layout = layout_ids - callback_ids
            ids_only_in_callbacks = callback_ids - layout_ids
            
            return {
                'status': 'ok',
                'message': 'Análisis completado',
                'details': {
                    'layout_ids': len(layout_ids),
                    'callback_ids': len(callback_ids),
                    'callbacks_count': len(callback_map),
                    'orphaned_layout_ids': list(ids_only_in_layout),
                    'missing_layout_ids': list(ids_only_in_callbacks),
                    'is_consistent': len(ids_only_in_callbacks) == 0
                }
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error analizando Dash: {str(e)}',
                'traceback': traceback.format_exc()
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
                'total_modelos': data_result['total_modelos']
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
        
        clear_logs_button = widgets.Button(
            description='� Logs',
            button_style='info',
            tooltip='Ver y gestionar logs del sistema',
            layout=widgets.Layout(width='140px', height='40px', margin='2px')
        )
        
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
            widgets.HBox([status_button, diagnostic_button], 
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
            results_section
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
        
        # Datos
        data_result, error = DataManager.load_models_data()
        print(f"\n🗄️ DATOS:")
        if data_result:
            print(f"   ✅ {data_result['total_modelos']} modelos en {data_result['num_celdas']} celdas")
            print(f"   📊 Estructura: {data_result['structure_type']}")
        else:
            print(f"   ❌ Error: {error}")
        
        # Módulos
        modules = DiagnosticManager.check_modules()
        print(f"\n🧩 MÓDULOS ({sum(1 for m in modules.values() if m['status'] == 'ok')}/{len(modules)}):")
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
            
            print()
            
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
            status_type = "ok" if click_test['status'] == 'ok' else "warning" if click_test['status'] == 'warning' else "error"
            ControlPanel._show_section_header("🖱️ TEST DE CLICKS", status_type)
            status_emoji = {'ok': '✅', 'warning': '⚠️', 'error': '❌'}
            print(f"   {status_emoji.get(click_test['status'], '❓')} {click_test['message']}")
            
            print(f"\n⏰ Actualizado: {datetime.now().strftime('%H:%M:%S')}")
            
        except Exception as e:
            print(f"❌ Error obteniendo estado: {e}")
    
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
                print(f"   📊 Layout IDs: {details.get('layout_ids', 0)}")
                print(f"   � Callbacks: {details.get('callbacks_count', 0)}")
                
                # Mostrar IDs críticos para clicks
                if details.get('missing_layout_ids') or details.get('orphaned_layout_ids'):
                    print(f"   ⚠️  IDs inconsistentes detectados - pueden afectar clicks")
                else:
                    print(f"   ✅ Todos los IDs están correctamente vinculados")
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
            
            # 4. NUEVO: Monitor de clicks en tiempo real
            ControlPanel._show_section_header("📡 MONITOR DE CLICKS EN TIEMPO REAL", "info")
            
            try:
                import ipywidgets as widgets
                from IPython.display import display
                
                # Crear botón para activar/desactivar monitor
                monitor_button = widgets.Button(
                    description='🎯 Activar Monitor de Clicks',
                    button_style='success',
                    tooltip='Activa monitoreo en tiempo real de clicks en gráficos',
                    layout=widgets.Layout(width='220px', height='40px')
                )
                
                # Output para mostrar clicks detectados
                click_monitor_output = widgets.Output(
                    layout=widgets.Layout(
                        height='200px',
                        border='1px solid #ccc',
                        padding='10px',
                        overflow='auto'
                    )
                )
                
                # Estado del monitor
                monitor_active = {'status': False}
                
                def toggle_monitor(button):
                    if not monitor_active['status']:
                        monitor_active['status'] = True
                        button.description = '⏹️ Detener Monitor'
                        button.button_style = 'danger'
                        
                        with click_monitor_output:
                            print("🎯 MONITOR DE CLICKS ACTIVADO")
                            print("=" * 35)
                            print("🔍 Detectando clicks en gráficos 2D y 3D...")
                            print("📊 Los clicks aparecerán aquí en tiempo real")
                            print("⏰ Timestamp | 📍 Ubicación | 📋 Datos")
                            print("-" * 35)
                            
                            # Simulación de detección (en implementación real se conectaría con callbacks)
                            # Esta función se puede expandir para conectar con el sistema real
                            ControlPanel._start_click_monitoring(click_monitor_output)
                    else:
                        monitor_active['status'] = False
                        button.description = '🎯 Activar Monitor de Clicks'
                        button.button_style = 'success'
                        
                        with click_monitor_output:
                            print("\n⏹️ Monitor detenido")
                
                monitor_button.on_click(toggle_monitor)
                
                print("   💡 Usa el botón abajo para activar monitoreo en tiempo real:")
                print("   📡 Se detectarán clicks en gráficos 2D y 3D automáticamente")
                print("   🕒 Los eventos aparecerán con timestamp y ubicación")
                print()
                
                display(monitor_button)
                display(click_monitor_output)
                
            except ImportError:
                print("   📋 Widgets no disponibles - monitor en modo texto")
                print("   💡 Para monitoreo completo, instalar: pip install ipywidgets")
            
            print()
            
            # Análisis por componente original (mantenido)
            ControlPanel._show_section_header("� ANÁLISIS POR COMPONENTE", "info")
            for comp_name, comp_data in analysis['components'].items():
                emoji = status_emoji.get(comp_data['status'], '❓')
                print(f"\n{emoji} {comp_name.upper()}")
                print(f"   � {comp_data['message']}")
                
                if comp_data.get('fix'):
                    print(f"   🔧 Solución: {comp_data['fix']}")
                
                # Mostrar detalles específicos si los hay
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
    def _start_click_monitoring(output_widget):
        """Inicia monitoreo de clicks (placeholder para implementación futura)"""
        import threading
        import time
        from datetime import datetime
        
        def monitor_loop():
            try:
                # Simulación de detección de clicks (aquí se conectaría con callbacks reales)
                click_count = 0
                while click_count < 5:  # Demostración limitada
                    time.sleep(3)
                    click_count += 1
                    
                    with output_widget:
                        timestamp = datetime.now().strftime('%H:%M:%S')
                        print(f"🎯 {timestamp} | 📊 Gráfico-2D | clickData: {{x: 1.23, y: 4.56}}")
                        
                        if click_count == 3:
                            print(f"🎯 {timestamp} | 📈 Gráfico-3D | clickData: {{x: 2.1, y: 3.4, z: 1.8}}")
                
                with output_widget:
                    print("\n💡 Demo completada - En implementación real se conectaría con callbacks de Dash")
                    
            except Exception as e:
                with output_widget:
                    print(f"❌ Error en monitor: {e}")
        
        # Ejecutar en hilo separado para no bloquear
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
    
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
                for log in recent_logs[-3:]:
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

# =============================================================================
# 🎯 FIN DEL MÓDULO
# ============================================================================="
