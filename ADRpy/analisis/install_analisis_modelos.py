"""
Script de Instalación para Módulo de Análisis de Modelos
======================================================

Este script configura el entorno necesario para ejecutar el módulo
de análisis interactivo de modelos de imputación.
"""

import subprocess
import sys
import os
from pathlib import Path

def install_package(package):
    """Instala un paquete usando pip."""
    try:
        print(f"📦 Instalando {package}...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", package, "--upgrade"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando {package}: {e}")
        return False

def check_dependencies():
    """Verifica e instala dependencias necesarias."""
    required_packages = [
        'dash>=2.14.0',
        'plotly>=5.17.0', 
        'pandas>=1.5.0',
        'numpy>=1.21.0'
    ]
    
    print("🔍 Verificando dependencias...")
    
    installed_count = 0
    for package_spec in required_packages:
        package_name = package_spec.split('>=')[0]
        
        try:
            __import__(package_name)
            print(f"✅ {package_name} ya está instalado")
            installed_count += 1
        except ImportError:
            if install_package(package_spec):
                print(f"✅ {package_name} instalado correctamente")
                installed_count += 1
            else:
                print(f"❌ No se pudo instalar {package_name}")
    
    return installed_count == len(required_packages)

def verify_structure():
    """Verifica que la estructura de archivos sea correcta."""
    print("\n🗂️ Verificando estructura de archivos...")
    
    required_files = [
        'Modulos/Analisis_modelos/__init__.py',
        'Modulos/Analisis_modelos/main_visualizacion_modelos.py',
        'Modulos/Analisis_modelos/data_loader.py',
        'Modulos/Analisis_modelos/plot_utils.py',
        'Modulos/Analisis_modelos/ui_components.py',
        'Results/modelos_completos_por_celda.json'
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} (faltante)")
            missing_files.append(file_path)
    
    return len(missing_files) == 0, missing_files

def create_launcher_script():
    """Crea un script launcher para facilitar la ejecución."""
    launcher_content = '''#!/usr/bin/env python3
"""
Launcher para Aplicación de Análisis de Modelos
==============================================
"""

import sys
import os

# Añadir directorio de módulos al path
current_dir = os.path.dirname(os.path.abspath(__file__))
modulos_path = os.path.join(current_dir, 'Modulos')
if modulos_path not in sys.path:
    sys.path.append(modulos_path)

try:
    from Analisis_modelos.main_visualizacion_modelos import main_visualizacion_modelos
    
    print("🚀 Iniciando Aplicación de Análisis de Modelos")
    print("📍 URL: http://localhost:8050")
    print("⚡ Presione Ctrl+C para detener")
    
    main_visualizacion_modelos(
        json_path=None,  # Usar ruta automática
        use_dash=True,   # Interfaz web
        port=8050,       # Puerto estándar
        debug=False      # Modo producción
    )
    
except KeyboardInterrupt:
    print("\\n⏹️ Aplicación detenida")
except ImportError as e:
    print(f"❌ Error de importación: {e}")
    print("💡 Ejecute primero el script de instalación")
except Exception as e:
    print(f"❌ Error: {e}")
'''
    
    try:
        with open('launch_analisis_modelos.py', 'w', encoding='utf-8') as f:
            f.write(launcher_content)
        print("✅ Script launcher creado: launch_analisis_modelos.py")
        return True
    except Exception as e:
        print(f"❌ Error creando launcher: {e}")
        return False

def main():
    """Función principal de instalación."""
    print("=" * 60)
    print("🛠️ INSTALACIÓN - MÓDULO DE ANÁLISIS DE MODELOS")
    print("=" * 60)
    
    # Verificar dependencias
    deps_ok = check_dependencies()
    
    if not deps_ok:
        print("\n❌ No se pudieron instalar todas las dependencias")
        print("💡 Intente instalar manualmente: pip install dash plotly pandas numpy")
        return False
    
    # Verificar estructura
    structure_ok, missing = verify_structure()
    
    if not structure_ok:
        print(f"\n⚠️ Faltan {len(missing)} archivos necesarios:")
        for file_path in missing:
            print(f"   - {file_path}")
        
        if 'Results/modelos_completos_por_celda.json' in missing:
            print("\n💡 El archivo JSON de modelos debe ser generado primero")
            print("   Ejecute la pipeline de imputación para crear este archivo")
    
    # Crear launcher
    launcher_ok = create_launcher_script()
    
    # Resumen final
    print("\n" + "=" * 60)
    print("📋 RESUMEN DE INSTALACIÓN")
    print("=" * 60)
    print(f"✅ Dependencias: {'OK' if deps_ok else 'ERROR'}")
    print(f"✅ Estructura: {'OK' if structure_ok else 'INCOMPLETA'}")
    print(f"✅ Launcher: {'OK' if launcher_ok else 'ERROR'}")
    
    if deps_ok and structure_ok and launcher_ok:
        print("\n🎉 INSTALACIÓN COMPLETADA")
        print("🚀 Ejecute: python launch_analisis_modelos.py")
        print("🌐 O use el notebook: notebook_analisis_modelos.ipynb")
    else:
        print("\n⚠️ Instalación incompleta - revise los errores arriba")
    
    return deps_ok and structure_ok and launcher_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
