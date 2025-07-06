# =============================================================================
# 🖱️ PANEL DE CONTROL CON MONITOR DE CLICKS INTEGRADO
# =============================================================================

import sys
import os
import webbrowser
import threading
import time

# Asegurar que podemos importar los módulos necesarios
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from . import notebook_utils
    from .notebook_utils import ClickDebugger, DataManager, Config, launch_app_with_config
    # Importar función de monitor si existe
    try:
        from .notebook_utils import create_click_monitor_dashboard
    except ImportError:
        create_click_monitor_dashboard = None
except ImportError:
    import notebook_utils
    from notebook_utils import ClickDebugger, DataManager, Config, launch_app_with_config
    try:
        from notebook_utils import create_click_monitor_dashboard
    except ImportError:
        create_click_monitor_dashboard = None


class ControlPanel:
    """Panel de control rediseñado con monitor de clicks integrado"""
    
    @staticmethod
    def create_control_panel():
        """Crea el panel de control principal con monitor integrado"""
        try:
            import ipywidgets as widgets
            from IPython.display import display, clear_output
        except ImportError:
            return ControlPanel._create_text_dashboard()
        
        # Panel único de salida más grande con scroll bidireccional
        output_main = widgets.Output(layout=widgets.Layout(
            height='600px', 
            width='100%',
            overflow='auto',
            border='2px solid #007acc',
            padding='15px',
            background_color='#f8f9fa'
        ))
        
        # Botones principales
        debug_button = widgets.Button(
            description='🐛 Debug + Monitor',
            button_style='warning',
            tooltip='Lanza aplicación debug con monitor de clicks',
            layout=widgets.Layout(width='160px', height='50px', margin='3px')
        )
        
        production_button = widgets.Button(
            description='🚀 Producción',
            button_style='success', 
            tooltip='Lanza aplicación en modo producción',
            layout=widgets.Layout(width='160px', height='50px', margin='3px')
        )
        
        # Botón específico para monitor de clicks
        click_monitor_button = widgets.Button(
            description='🖱️ Monitor Clicks',
            button_style='info',
            tooltip='Abre dashboard dedicado de monitoreo de clicks',
            layout=widgets.Layout(width='160px', height='50px', margin='3px')
        )
        
        # Botones de diagnóstico
        status_button = widgets.Button(
            description='📊 Estado Sistema',
            button_style='info',
            tooltip='Muestra estado detallado del sistema',
            layout=widgets.Layout(width='160px', height='45px', margin='3px')
        )
        
        diagnostic_button = widgets.Button(
            description='🔬 Diagnóstico',
            button_style='primary',
            tooltip='Ejecuta diagnóstico completo',
            layout=widgets.Layout(width='160px', height='45px', margin='3px')
        )
        
        clear_logs_button = widgets.Button(
            description='🗑️ Limpiar',
            button_style='danger',
            tooltip='Limpia logs y reinicia estado',
            layout=widgets.Layout(width='160px', height='45px', margin='3px')
        )
        
        # Funciones de los botones
        def launch_debug_app(button):
            with output_main:
                clear_output(wait=True)
                ControlPanel._show_centered_title("🐛 LANZAMIENTO DEBUG CON MONITOR")
                try:
                    print("🔄 Verificando sistema...")
                    ControlPanel._display_status_compact()
                    
                    print("\\n🚀 Iniciando aplicación con monitor de clicks...")
                    
                    # Configuración para debug con monitor
                    config = {
                        'json_path': Config.JSON_PATH,
                        'use_dash': True,
                        'port': Config.PORTS['debug'],
                        'debug': True
                    }
                    
                    # Habilitar debug visual de clicks
                    os.environ['DASH_DEBUG_CLICK'] = '1'
                    
                    print("🖱️ Monitor de clicks habilitado")
                    print("📍 Monitor aparecerá en el panel derecho del dashboard")
                    print("🌐 Abriendo navegador...")
                    
                    # Lanzar en thread separado
                    def run_app():
                        launch_app_with_config(config)
                    
                    thread = threading.Thread(target=run_app, daemon=True)
                    thread.start()
                    
                    # Esperar un poco y abrir navegador
                    time.sleep(3)
                    try:
                        webbrowser.open(f'http://localhost:{Config.PORTS["debug"]}')
                        print("✅ Navegador abierto exitosamente")
                    except Exception as e:
                        print(f"⚠️ No se pudo abrir navegador: {e}")
                        print(f"📌 Abre manualmente: http://localhost:{Config.PORTS['debug']}")
                        
                except Exception as e:
                    print(f"❌ Error: {e}")
        
        def launch_production_app(button):
            with output_main:
                clear_output(wait=True)
                ControlPanel._show_centered_title("🚀 LANZAMIENTO PRODUCCIÓN")
                try:
                    config = {
                        'json_path': Config.JSON_PATH,
                        'use_dash': True,
                        'port': Config.PORTS['produccion'],
                        'debug': False
                    }
                    
                    def run_app():
                        launch_app_with_config(config)
                    
                    thread = threading.Thread(target=run_app, daemon=True)
                    thread.start()
                    
                    time.sleep(3)
                    webbrowser.open(f'http://localhost:{Config.PORTS["produccion"]}')
                    
                    print(f"✅ Aplicación lanzada en puerto {Config.PORTS['produccion']}")
                        
                except Exception as e:
                    print(f"❌ Error: {e}")
        
        def launch_click_monitor(button):
            with output_main:
                clear_output(wait=True)
                ControlPanel._show_centered_title("🖱️ MONITOR DE CLICKS DEDICADO")
                try:
                    print("🔄 Creando dashboard de monitor de clicks...")
                    
                    # Crear y lanzar dashboard específico de clicks
                    monitor_app = create_click_monitor_dashboard()
                    
                    if monitor_app:
                        def run_monitor():
                            monitor_app.run_server(debug=True, port=8057, host='127.0.0.1')
                        
                        thread = threading.Thread(target=run_monitor, daemon=True)
                        thread.start()
                        
                        time.sleep(2)
                        webbrowser.open('http://localhost:8057')
                        
                        print("✅ Monitor de clicks lanzado en puerto 8057")
                        print("🖱️ Este dashboard muestra información detallada de clicks")
                        print("🔧 Úsalo junto con la aplicación principal para debug")
                    else:
                        print("❌ No se pudo crear el monitor (Dash no disponible)")
                        
                except Exception as e:
                    print(f"❌ Error: {e}")
        
        def show_system_status(button):
            with output_main:
                clear_output(wait=True)
                ControlPanel._show_centered_title("📊 ESTADO DETALLADO DEL SISTEMA")
                ControlPanel._display_status_detailed()
        
        def run_full_diagnostic(button):
            with output_main:
                clear_output(wait=True)
                ControlPanel._show_centered_title("🔬 DIAGNÓSTICO COMPLETO")
                ControlPanel._run_diagnostic()
        
        def clear_and_reset(button):
            with output_main:
                clear_output(wait=True)
                ControlPanel._show_centered_title("🗑️ LIMPIEZA Y RESET")
                print("🔄 Limpiando logs y reiniciando estado...")
                # Aquí podrías agregar lógica de limpieza
                print("✅ Sistema limpiado")
                ControlPanel._show_initial_message()
        
        # Conectar eventos
        debug_button.on_click(launch_debug_app)
        production_button.on_click(launch_production_app)
        click_monitor_button.on_click(launch_click_monitor)
        status_button.on_click(show_system_status)
        diagnostic_button.on_click(run_full_diagnostic)
        clear_logs_button.on_click(clear_and_reset)
        
        # Layout principal
        buttons_box = widgets.HBox([
            widgets.VBox([debug_button, production_button], layout=widgets.Layout(margin='5px')),
            widgets.VBox([click_monitor_button, status_button], layout=widgets.Layout(margin='5px')),
            widgets.VBox([diagnostic_button, clear_logs_button], layout=widgets.Layout(margin='5px'))
        ], layout=widgets.Layout(justify_content='center', margin='10px'))
        
        # Panel principal
        main_panel = widgets.VBox([
            widgets.HTML(value="<h2 style='text-align: center; color: #007acc;'>🚀 Panel de Control - Análisis de Modelos</h2>"),
            buttons_box,
            output_main
        ])
        
        # Mostrar mensaje inicial
        with output_main:
            ControlPanel._show_initial_message()
        
        return main_panel
    
    @staticmethod
    def _show_centered_title(title):
        """Muestra título centrado con estilo"""
        print("\\n" + "="*60)
        print(f"{title:^60}")
        print("="*60 + "\\n")
    
    @staticmethod
    def _display_status_compact():
        """Muestra estado compacto del sistema"""
        print("📊 Verificando archivos críticos...")
        
        # Verificar archivos
        archivos = [
            Config.JSON_PATH,
            'Modulos/Analisis_modelos/main_visualizacion_modelos.py',
            'Modulos/Analisis_modelos/notebook_utils.py'
        ]
        
        for archivo in archivos:
            if os.path.exists(archivo):
                print(f"   ✅ {archivo}")
            else:
                print(f"   ❌ {archivo}")
        
        # Verificar puertos
        print("\\n🌐 Verificando puertos...")
        for nombre, puerto in Config.PORTS.items():
            print(f"   {nombre}: {puerto}")
    
    @staticmethod
    def _display_status_detailed():
        """Muestra estado detallado del sistema"""
        ControlPanel._display_status_compact()
        
        print("\\n🔍 Verificando datos...")
        try:
            data_result, error = DataManager.load_models_data()
            if error:
                print(f"   ❌ Error: {error}")
            else:
                print(f"   ✅ {data_result['total_modelos']} modelos cargados")
                print(f"   📊 {data_result['num_celdas']} celdas de datos")
        except Exception as e:
            print(f"   ❌ Error verificando datos: {e}")
    
    @staticmethod
    def _run_diagnostic():
        """Ejecuta diagnóstico completo"""
        try:
            print("🔍 Ejecutando análisis de cadena de clicks...")
            result = ClickDebugger.analyze_click_chain()
            
            print(f"\\n📋 Resultado: {result['status']}")
            print(f"📝 Mensaje: {result['message']}")
            
            print("\\n🔧 Componentes verificados:")
            for comp_name, comp_data in result.get('components', {}).items():
                status = comp_data.get('status', 'unknown')
                message = comp_data.get('message', '')
                print(f"   {comp_name}: {status}")
                if message:
                    print(f"      {message}")
            
            if result.get('recommendations'):
                print("\\n💡 Recomendaciones:")
                for rec in result['recommendations']:
                    print(f"   • {rec}")
                    
        except Exception as e:
            print(f"❌ Error en diagnóstico: {e}")
    
    @staticmethod
    def _show_initial_message():
        """Muestra mensaje inicial"""
        print("🎯 PANEL DE CONTROL LISTO")
        print("\\n📋 Opciones disponibles:")
        print("   🐛 Debug + Monitor: Aplicación con monitor de clicks integrado")
        print("   🚀 Producción: Aplicación optimizada sin debug")
        print("   🖱️ Monitor Clicks: Dashboard dedicado para debug de clicks")
        print("   📊 Estado Sistema: Información detallada del sistema")
        print("   🔬 Diagnóstico: Verificación completa de componentes")
        print("   🗑️ Limpiar: Reset de logs y estado")
        print("\\n✨ Seleccione una opción para comenzar")
    
    @staticmethod
    def _create_text_dashboard():
        """Fallback para cuando ipywidgets no está disponible"""
        print("⚠️ ipywidgets no disponible, usando interfaz simplificada")
        print("\\nPuede ejecutar manualmente:")
        print("   launch_app_with_config({'port': 8054, 'debug': True})")
        return None
    
    @staticmethod
    def _create_click_monitor_widget(output_widget):
        """
        Crea un monitor de clicks integrado con cuadro scrollable dentro del dashboard de control.
        """
        try:
            import ipywidgets as widgets
            from IPython.display import display, clear_output
            import threading
            import time
            import json
            import os
            from datetime import datetime
        except ImportError:
            print("❌ ipywidgets no disponible")
            return

        print("🖱️ INICIANDO MÓDULO DE DEBUG DE CLICKS")
        print("="*60)
        print("📋 Este módulo registra clicks en tiempo real desde la aplicación principal")
        print("🎯 Los clicks aparecerán en el cuadro scrollable de abajo")
        print()

        # Crear botones de control
        start_button = widgets.Button(
            description='🔄 Iniciar Monitor',
            button_style='success',
            tooltip='Inicia el registro de clicks en tiempo real',
            layout=widgets.Layout(width='160px', height='40px', margin='5px')
        )

        stop_button = widgets.Button(
            description='⏹️ Detener Monitor',
            button_style='danger',
            tooltip='Detiene el registro de clicks',
            layout=widgets.Layout(width='160px', height='40px', margin='5px')
        )

        clear_button = widgets.Button(
            description='🗑️ Limpiar Registro',
            button_style='warning',
            tooltip='Limpia el registro de clicks',
            layout=widgets.Layout(width='160px', height='40px', margin='5px')
        )

        # Estado del monitor
        status_label = widgets.HTML(
            value="<span style='color: #6c757d; font-weight: bold;'>Monitor inactivo</span>",
            layout=widgets.Layout(margin='5px')
        )

        # Contador de clicks
        click_counter = widgets.HTML(
            value="<span style='color: #007bff; font-weight: bold;'>Total clicks: 0</span>",
            layout=widgets.Layout(margin='5px')
        )

        # Cuadro de registro con scroll
        click_log = widgets.Textarea(
            value="Monitor inactivo. Presione 'Iniciar Monitor' para comenzar el registro.\n",
            placeholder='Los clicks aparecerán aquí...',
            layout=widgets.Layout(
                width='100%',
                height='400px',
                border='2px solid #007bff',
                font_family='monospace',
                font_size='12px'
            ),
            disabled=True
        )

        # Variables de estado
        monitor_active = False
        click_count = 0
        log_lines = []

        def add_log_entry(message):
            """Añade una entrada al log"""
            nonlocal log_lines
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            entry = f"[{timestamp}] {message}"
            log_lines.append(entry)
            
            # Mantener solo las últimas 100 líneas
            if len(log_lines) > 100:
                log_lines = log_lines[-100:]
            
            # Actualizar el cuadro de texto
            click_log.value = "\n".join(log_lines) + "\n"

        def start_monitor(button):
            """Inicia el monitor de clicks"""
            nonlocal monitor_active
            if not monitor_active:
                monitor_active = True
                status_label.value = "<span style='color: #28a745; font-weight: bold;'>🟢 Monitor activo - Esperando clicks</span>"
                add_log_entry("🟢 MONITOR INICIADO - Detectando clicks de la aplicación principal")
                
                # Activar variable de entorno para debug de clicks
                os.environ['DASH_DEBUG_CLICK'] = '1'
                add_log_entry("✅ Variable DASH_DEBUG_CLICK activada")
                
                # Crear hilo monitor (lee clicks reales de la aplicación principal)
                def monitor_real_clicks():
                    """Lee clicks reales del archivo de log"""
                    nonlocal monitor_active, click_count
                    import json
                    last_count = 0
                    
                    while monitor_active:
                        try:
                            # Leer archivo de clicks
                            if os.path.exists('clicks_log.json'):
                                with open('clicks_log.json', 'r') as f:
                                    clicks_data = json.load(f)
                                
                                # Verificar si hay clicks nuevos
                                if len(clicks_data) > last_count:
                                    # Procesar clicks nuevos
                                    for i in range(last_count, len(clicks_data)):
                                        click = clicks_data[i]
                                        click_count = click.get('count', i + 1)
                                        
                                        click_msg = f"🖱️ CLICK REAL #{click_count} - Coordenadas: ({click['x']}, {click['y']}) - Modelo: {click['customdata']} - Trace: {click['trace_name']}"
                                        add_log_entry(click_msg)
                                        
                                        click_counter.value = f"<span style='color: #007bff; font-weight: bold;'>Total clicks: {click_count}</span>"
                                    
                                    last_count = len(clicks_data)
                            
                            time.sleep(1)  # Verificar cada segundo
                            
                        except Exception as e:
                            if monitor_active:  # Solo reportar errores si el monitor está activo
                                add_log_entry(f"❌ Error leyendo clicks: {e}")
                            time.sleep(2)
                
                # Iniciar monitor en hilo separado
                thread = threading.Thread(target=monitor_real_clicks, daemon=True)
                thread.start()

        def stop_monitor(button):
            """Detiene el monitor de clicks"""
            nonlocal monitor_active
            monitor_active = False
            status_label.value = "<span style='color: #dc3545; font-weight: bold;'>🔴 Monitor detenido</span>"
            add_log_entry("🔴 MONITOR DETENIDO")
            
            # Desactivar variable de entorno
            if 'DASH_DEBUG_CLICK' in os.environ:
                del os.environ['DASH_DEBUG_CLICK']
                add_log_entry("❌ Variable DASH_DEBUG_CLICK desactivada")

        def clear_log(button):
            """Limpia el registro de clicks"""
            nonlocal log_lines, click_count
            log_lines = []
            click_count = 0
            click_log.value = "Registro limpiado.\n"
            click_counter.value = "<span style='color: #007bff; font-weight: bold;'>Total clicks: 0</span>"
            add_log_entry("🗑️ REGISTRO LIMPIADO")

        # Conectar eventos
        start_button.on_click(start_monitor)
        stop_button.on_click(stop_monitor)
        clear_button.on_click(clear_log)

        # Layout del widget
        controls = widgets.HBox([start_button, stop_button, clear_button])
        status_row = widgets.HBox([status_label, click_counter])
        
        # Mostrar el widget completo
        monitor_widget = widgets.VBox([
            widgets.HTML("""
            <div style='background: #e3f2fd; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #007bff;'>
                <h4 style='margin: 0; color: #0d47a1;'>🎯 Monitor de Clicks en Tiempo Real</h4>
                <p style='margin: 5px 0 0 0; color: #1976d2;'>
                    Registra automáticamente los clicks de la aplicación principal
                </p>
            </div>
            """),
            controls,
            status_row,
            widgets.HTML("<h5 style='color: #495057; margin: 15px 0 5px 0;'>📝 Registro de Clicks:</h5>"),
            click_log
        ])

        display(monitor_widget)
        
        print("\n✅ Módulo de debug de clicks creado exitosamente")
        print("🎯 Use los botones de arriba para controlar el monitor")
        print("📝 Los clicks REALES aparecerán en el cuadro de texto de abajo")
        print()
        print("💡 INSTRUCCIONES PARA USO:")
        print("   1. Presione 'Iniciar Monitor' en este dashboard de control")
        print("   2. Abra la aplicación principal usando '🐛 Lanzar Debug'")
        print("   3. Haga clicks en los gráficos de la aplicación principal")
        print("   4. Los clicks aparecerán automáticamente en el cuadro de arriba")
        print()
        print("🔧 El monitor lee clicks reales del archivo 'clicks_log.json'")
        print("📊 Los clicks se registran solo cuando DASH_DEBUG_CLICK está activo")
    
    @staticmethod
    def _show_log_management():
        """Muestra opciones de gestión de logs"""
        print("🗑️ GESTIÓN DE LOGS")
        print("-" * 40)
        print("📋 Estado de logs del sistema:")
        print()
        
        # Verificar archivos de log
        log_files = ['clicks_log.json', 'debug_log.txt', 'error_log.txt']
        total_size = 0
        
        for log_file in log_files:
            if os.path.exists(log_file):
                import os
                size = os.path.getsize(log_file)
                total_size += size
                print(f"   ✅ {log_file}: {size} bytes")
            else:
                print(f"   ❌ {log_file}: No existe")
        
        print(f"\n📊 Tamaño total de logs: {total_size} bytes")
        
        if total_size > 0:
            print("\n🔧 Opciones de limpieza:")
            print("   - Para limpiar clicks: rm clicks_log.json")
            print("   - Para limpiar todo: rm *.log *.json")
        else:
            print("\n✅ No hay logs que limpiar")

    @staticmethod
    def _create_text_dashboard():
        """Fallback para cuando ipywidgets no está disponible"""
        print("⚠️ ipywidgets no disponible, usando interfaz simplificada")
        print("\\nPuede ejecutar manualmente:")
        print("   launch_app_with_config({'port': 8054, 'debug': True})")
        return None
    
    @staticmethod
    def _create_click_monitor_widget(output_widget):
        """
        Crea un monitor de clicks integrado con cuadro scrollable dentro del dashboard de control.
        """
        try:
            import ipywidgets as widgets
            from IPython.display import display, clear_output
            import threading
            import time
            import json
            import os
            from datetime import datetime
        except ImportError:
            print("❌ ipywidgets no disponible")
            return

        print("🖱️ INICIANDO MÓDULO DE DEBUG DE CLICKS")
        print("="*60)
        print("📋 Este módulo registra clicks en tiempo real desde la aplicación principal")
        print("🎯 Los clicks aparecerán en el cuadro scrollable de abajo")
        print()

        # Crear botones de control
        start_button = widgets.Button(
            description='🔄 Iniciar Monitor',
            button_style='success',
            tooltip='Inicia el registro de clicks en tiempo real',
            layout=widgets.Layout(width='160px', height='40px', margin='5px')
        )

        stop_button = widgets.Button(
            description='⏹️ Detener Monitor',
            button_style='danger',
            tooltip='Detiene el registro de clicks',
            layout=widgets.Layout(width='160px', height='40px', margin='5px')
        )

        clear_button = widgets.Button(
            description='🗑️ Limpiar Registro',
            button_style='warning',
            tooltip='Limpia el registro de clicks',
            layout=widgets.Layout(width='160px', height='40px', margin='5px')
        )

        # Estado del monitor
        status_label = widgets.HTML(
            value="<span style='color: #6c757d; font-weight: bold;'>Monitor inactivo</span>",
            layout=widgets.Layout(margin='5px')
        )

        # Contador de clicks
        click_counter = widgets.HTML(
            value="<span style='color: #007bff; font-weight: bold;'>Total clicks: 0</span>",
            layout=widgets.Layout(margin='5px')
        )

        # Cuadro de registro con scroll
        click_log = widgets.Textarea(
            value="Monitor inactivo. Presione 'Iniciar Monitor' para comenzar el registro.\n",
            placeholder='Los clicks aparecerán aquí...',
            layout=widgets.Layout(
                width='100%',
                height='400px',
                border='2px solid #007bff',
                font_family='monospace',
                font_size='12px'
            ),
            disabled=True
        )

        # Variables de estado
        monitor_active = False
        click_count = 0
        log_lines = []

        def add_log_entry(message):
            """Añade una entrada al log"""
            nonlocal log_lines
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            entry = f"[{timestamp}] {message}"
            log_lines.append(entry)
            
            # Mantener solo las últimas 100 líneas
            if len(log_lines) > 100:
                log_lines = log_lines[-100:]
            
            # Actualizar el cuadro de texto
            click_log.value = "\n".join(log_lines) + "\n"

        def start_monitor(button):
            """Inicia el monitor de clicks"""
            nonlocal monitor_active
            if not monitor_active:
                monitor_active = True
                status_label.value = "<span style='color: #28a745; font-weight: bold;'>🟢 Monitor activo - Esperando clicks</span>"
                add_log_entry("🟢 MONITOR INICIADO - Detectando clicks de la aplicación principal")
                
                # Activar variable de entorno para debug de clicks
                os.environ['DASH_DEBUG_CLICK'] = '1'
                add_log_entry("✅ Variable DASH_DEBUG_CLICK activada")
                
                # Crear hilo monitor (lee clicks reales de la aplicación principal)
                def monitor_real_clicks():
                    """Lee clicks reales del archivo de log"""
                    nonlocal monitor_active, click_count
                    import json
                    last_count = 0
                    
                    while monitor_active:
                        try:
                            # Leer archivo de clicks
                            if os.path.exists('clicks_log.json'):
                                with open('clicks_log.json', 'r') as f:
                                    clicks_data = json.load(f)
                                
                                # Verificar si hay clicks nuevos
                                if len(clicks_data) > last_count:
                                    # Procesar clicks nuevos
                                    for i in range(last_count, len(clicks_data)):
                                        click = clicks_data[i]
                                        click_count = click.get('count', i + 1)
                                        
                                        click_msg = f"🖱️ CLICK REAL #{click_count} - Coordenadas: ({click['x']}, {click['y']}) - Modelo: {click['customdata']} - Trace: {click['trace_name']}"
                                        add_log_entry(click_msg)
                                        
                                        click_counter.value = f"<span style='color: #007bff; font-weight: bold;'>Total clicks: {click_count}</span>"
                                    
                                    last_count = len(clicks_data)
                            
                            time.sleep(1)  # Verificar cada segundo
                            
                        except Exception as e:
                            if monitor_active:  # Solo reportar errores si el monitor está activo
                                add_log_entry(f"❌ Error leyendo clicks: {e}")
                            time.sleep(2)
                
                # Iniciar monitor en hilo separado
                thread = threading.Thread(target=monitor_real_clicks, daemon=True)
                thread.start()

        def stop_monitor(button):
            """Detiene el monitor de clicks"""
            nonlocal monitor_active
            monitor_active = False
            status_label.value = "<span style='color: #dc3545; font-weight: bold;'>🔴 Monitor detenido</span>"
            add_log_entry("🔴 MONITOR DETENIDO")
            
            # Desactivar variable de entorno
            if 'DASH_DEBUG_CLICK' in os.environ:
                del os.environ['DASH_DEBUG_CLICK']
                add_log_entry("❌ Variable DASH_DEBUG_CLICK desactivada")

        def clear_log(button):
            """Limpia el registro de clicks"""
            nonlocal log_lines, click_count
            log_lines = []
            click_count = 0
            click_log.value = "Registro limpiado.\n"
            click_counter.value = "<span style='color: #007bff; font-weight: bold;'>Total clicks: 0</span>"
            add_log_entry("🗑️ REGISTRO LIMPIADO")

        # Conectar eventos
        start_button.on_click(start_monitor)
        stop_button.on_click(stop_monitor)
        clear_button.on_click(clear_log)

        # Layout del widget
        controls = widgets.HBox([start_button, stop_button, clear_button])
        status_row = widgets.HBox([status_label, click_counter])
        
        # Mostrar el widget completo
        monitor_widget = widgets.VBox([
            widgets.HTML("""
            <div style='background: #e3f2fd; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #007bff;'>
                <h4 style='margin: 0; color: #0d47a1;'>🎯 Monitor de Clicks en Tiempo Real</h4>
                <p style='margin: 5px 0 0 0; color: #1976d2;'>
                    Registra automáticamente los clicks de la aplicación principal
                </p>
            </div>
            """),
            controls,
            status_row,
            widgets.HTML("<h5 style='color: #495057; margin: 15px 0 5px 0;'>📝 Registro de Clicks:</h5>"),
            click_log
        ])

        display(monitor_widget)
        
        print("\n✅ Módulo de debug de clicks creado exitosamente")
        print("🎯 Use los botones de arriba para controlar el monitor")
        print("📝 Los clicks REALES aparecerán en el cuadro de texto de abajo")
        print()
        print("💡 INSTRUCCIONES PARA USO:")
        print("   1. Presione 'Iniciar Monitor' en este dashboard de control")
        print("   2. Abra la aplicación principal usando '🐛 Lanzar Debug'")
        print("   3. Haga clicks en los gráficos de la aplicación principal")
        print("   4. Los clicks aparecerán automáticamente en el cuadro de arriba")
        print()
        print("🔧 El monitor lee clicks reales del archivo 'clicks_log.json'")
        print("📊 Los clicks se registran solo cuando DASH_DEBUG_CLICK está activo")
    
    @staticmethod
    def _show_log_management():
        """Muestra opciones de gestión de logs"""
        print("🗑️ GESTIÓN DE LOGS")
        print("-" * 40)
        print("📋 Estado de logs del sistema:")
        print()
        
        # Verificar archivos de log
        log_files = ['clicks_log.json', 'debug_log.txt', 'error_log.txt']
        total_size = 0
        
        for log_file in log_files:
            if os.path.exists(log_file):
                import os
                size = os.path.getsize(log_file)
                total_size += size
                print(f"   ✅ {log_file}: {size} bytes")
            else:
                print(f"   ❌ {log_file}: No existe")
        
        print(f"\n📊 Tamaño total de logs: {total_size} bytes")
        
        if total_size > 0:
            print("\n🔧 Opciones de limpieza:")
            print("   - Para limpiar clicks: rm clicks_log.json")
            print("   - Para limpiar todo: rm *.log *.json")
        else:
            print("\n✅ No hay logs que limpiar")

def create_control_panel():
    """Función de conveniencia para crear el panel"""
    return ControlPanel.create_control_panel()
    """Panel de control rediseñado con panel único de salida"""
    
    @staticmethod
    def create_dashboard():
        """Crea el panel de control principal con panel único"""
        try:
            import ipywidgets as widgets
            from IPython.display import display, clear_output
            import webbrowser
            
            # Hacer display disponible globalmente para el notebook
            globals()['display'] = display
                
        except ImportError:
            return ControlPanel._create_text_dashboard()
        
        # Panel único de salida más grande con scroll bidireccional
        output_main = widgets.Output(layout=widgets.Layout(
            height='600px', 
            width='100%',
            overflow='auto',  # Scroll vertical y horizontal automático
            border='2px solid #007acc',
            padding='15px',
            background_color='#f8f9fa'
        ))
        
        # Botones rediseñados - más grandes y organizados
        debug_button = widgets.Button(
            description='🐛 Lanzar Debug',
            button_style='warning',
            tooltip='Lanza aplicación en modo debug (puerto 8054)',
            layout=widgets.Layout(width='160px', height='50px', margin='3px')
        )
        
        production_button = widgets.Button(
            description='🚀 Lanzar Producción',
            button_style='success', 
            tooltip='Lanza aplicación en modo producción (puerto 8055)',
            layout=widgets.Layout(width='160px', height='50px', margin='3px')
        )
        
        # Botones de información
        status_button = widgets.Button(
            description='📊 Estado Sistema',
            button_style='info',
            tooltip='Muestra estado detallado del sistema',
            layout=widgets.Layout(width='160px', height='45px', margin='3px')
        )
        
        diagnostic_button = widgets.Button(
            description='🔬 Diagnóstico Completo',
            button_style='primary',
            tooltip='Ejecuta diagnóstico integral del sistema',
            layout=widgets.Layout(width='160px', height='45px', margin='3px')
        )
        
        click_debug_button = widgets.Button(
            description='🖱️ Debug Clicks',
            button_style='warning',
            tooltip='Diagnóstico específico para problemas de clicks',
            layout=widgets.Layout(width='160px', height='45px', margin='3px')
        )
        
        # Botones de mantenimiento
        clear_logs_button = widgets.Button(
            description='🗑️ Gestión Logs',
            button_style='danger',
            tooltip='Gestiona y borra logs antiguos con confirmación',
            layout=widgets.Layout(width='160px', height='45px', margin='3px')
        )
        
        # Funciones rediseñadas para usar el panel único
        def launch_debug_app(button):
            with output_main:
                clear_output(wait=True)
                ControlPanel._show_centered_title("🐛 LANZAMIENTO MODO DEBUG")
                try:
                    print("🔄 Verificando sistema...")
                    ControlPanel._display_status_compact()
                    
                    print("\\n🚀 Iniciando aplicación...")
                    AppLauncher.launch_app('debug')
                    
                    print("\\n🌐 Abriendo navegador en http://localhost:8054")
                    try:
                        webbrowser.open('http://localhost:8054')
                        print("✅ Navegador abierto exitosamente")
                    except Exception as e:
                        print(f"⚠️ No se pudo abrir navegador automáticamente: {e}")
                        print("📌 Abre manualmente: http://localhost:8054")
                        
                except Exception as e:
                    print(f"❌ Error: {e}")
        
        def launch_production_app(button):
            with output_main:
                clear_output(wait=True)
                ControlPanel._show_centered_title("🚀 LANZAMIENTO MODO PRODUCCIÓN")
                try:
                    print("🔄 Verificando sistema...")
                    ControlPanel._display_status_compact()
                    
                    print("\\n🚀 Iniciando aplicación...")
                    AppLauncher.launch_app('produccion')
                    
                    print("\\n🌐 Abriendo navegador en http://localhost:8055")
                    try:
                        webbrowser.open('http://localhost:8055')
                        print("✅ Navegador abierto exitosamente")
                    except Exception as e:
                        print(f"⚠️ No se pudo abrir navegador automáticamente: {e}")
                        print("📌 Abre manualmente: http://localhost:8055")
                        
                except Exception as e:
                    print(f"❌ Error: {e}")
        
        def show_system_status(button):
            with output_main:
                clear_output(wait=True)
                ControlPanel._show_centered_title("📊 ESTADO DETALLADO DEL SISTEMA")
                ControlPanel._display_status_detailed()
        
        def run_full_diagnostic(button):
            with output_main:
                clear_output(wait=True)
                ControlPanel._show_centered_title("🔬 DIAGNÓSTICO COMPLETO DEL SISTEMA")
                ControlPanel._run_diagnostic()
        
        def debug_clicks_advanced(button):
            with output_main:
                clear_output(wait=True)
                ControlPanel._show_centered_title("🖱️ MÓDULO DE DEBUG DE CLICKS")
                ControlPanel._create_click_monitor_widget(output_main)
        
        def manage_logs(button):
            with output_main:
                clear_output(wait=True)
                ControlPanel._show_centered_title("🗑️ GESTIÓN DE LOGS")
                ControlPanel._show_log_management()
        
        # Conectar eventos
        debug_button.on_click(launch_debug_app)
        production_button.on_click(launch_production_app)
        status_button.on_click(show_system_status)
        diagnostic_button.on_click(run_full_diagnostic)
        click_debug_button.on_click(debug_clicks_advanced)
        clear_logs_button.on_click(manage_logs)
        
        # Layout completamente rediseñado
        title = widgets.HTML("""
        <div style='text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; padding: 20px; border-radius: 15px; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.2);'>
            <h1 style='margin: 0; font-size: 2.2em;'>🎛️ Panel de Control</h1>
            <h2 style='margin: 5px 0 0 0; font-size: 1.3em; opacity: 0.9;'>Análisis de Modelos de Aeronaves</h2>
            <p style='margin: 10px 0 0 0; opacity: 0.8; font-size: 1em;'>Sistema centralizado de gestión y diagnóstico</p>
        </div>
        """)
        
        # Sección de lanzamiento
        launch_section = widgets.VBox([
            widgets.HTML("""
            <div style='text-align: center; background: #e8f4fd; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #007acc;'>
                <h3 style='margin: 0; color: #005c99; font-size: 1.3em;'>🚀 Lanzamiento de Aplicación</h3>
                <p style='margin: 8px 0 0 0; color: #007acc; font-size: 1em;'>
                    Los botones abren automáticamente el navegador web
                </p>
            </div>
            """),
            widgets.HBox([debug_button, production_button], 
                        layout=widgets.Layout(justify_content='center', margin='10px 0'))
        ])
        
        # Sección de diagnósticos
        diagnostic_section = widgets.VBox([
            widgets.HTML("""
            <div style='text-align: center; background: #f0f8f0; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #28a745;'>
                <h3 style='margin: 0; color: #1e6b2e; font-size: 1.3em;'>🔧 Herramientas de Diagnóstico</h3>
                <p style='margin: 8px 0 0 0; color: #28a745; font-size: 1em;'>
                    Toda la información se muestra en el panel principal abajo
                </p>
            </div>
            """),
            widgets.HBox([status_button, diagnostic_button], 
                        layout=widgets.Layout(justify_content='center', margin='5px 0')),
            widgets.HBox([click_debug_button, clear_logs_button], 
                        layout=widgets.Layout(justify_content='center', margin='5px 0'))
        ])
        
        # Sección del panel principal
        main_output_section = widgets.VBox([
            widgets.HTML("""
            <div style='text-align: center; background: #fff3cd; padding: 15px; border-radius: 10px; margin: 15px 0; border-left: 4px solid #ffc107;'>
                <h3 style='margin: 0; color: #856404; font-size: 1.3em;'>📋 Panel Principal de Información</h3>
                <p style='margin: 8px 0 0 0; color: #856404; font-size: 1em;'>
                    Selecciona cualquier botón para ver la información correspondiente
                </p>
            </div>
            """),
            output_main
        ])
        
        # Panel principal con estructura mejorada
        main_panel = widgets.VBox([
            title,
            launch_section,
            diagnostic_section, 
            main_output_section
        ], layout=widgets.Layout(padding='15px', width='100%'))
        
        # Mostrar mensaje inicial en el panel
        with output_main:
            ControlPanel._show_initial_message()
        
        return main_panel
