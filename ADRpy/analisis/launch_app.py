#!/usr/bin/env python3
"""
Launcher para la aplicación de visualización de modelos
Maneja los imports correctamente para evitar problemas con imports relativos
"""

import sys
import os
from pathlib import Path

# Configurar paths
current_dir = Path(__file__).parent
modulos_dir = current_dir / "Modulos"
analisis_modelos_dir = modulos_dir / "Analisis_modelos"

# Añadir directorios al path
sys.path.insert(0, str(modulos_dir))
sys.path.insert(0, str(analisis_modelos_dir))

# Configurar variables de entorno para debug
os.environ['DASH_DEBUG_CLICK'] = '1'

def main():
    """Función principal del launcher"""
    try:
        # Importar el módulo principal (ahora debería funcionar)
        import main_visualizacion_modelos as main_viz
        
        print("🚀 Iniciando aplicación de visualización de modelos...")
        print("📊 Con soporte para la nueva estructura JSON")
        
        # Ruta del JSON
        json_path = current_dir / "Results" / "modelos_completos_por_celda.json"
        
        if not json_path.exists():
            print(f"❌ Archivo JSON no encontrado: {json_path}")
            return
        
        print(f"📁 Usando archivo: {json_path}")
        
        # Ejecutar aplicación
        main_viz.main_visualizacion_modelos(
            json_path=str(json_path),
            use_dash=True,
            port=8050,
            debug=True
        )
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        print("🔧 Intentando método alternativo...")
        
        # Método alternativo: importar componentes individuales
        try:
            import data_loader
            import plot_interactive
            import ui_components
            print("✅ Componentes importados correctamente")
            print("🔧 Ejecute la aplicación manualmente desde el directorio Analisis_modelos")
            
        except ImportError as e2:
            print(f"❌ Error en método alternativo: {e2}")
            print("📋 Archivos disponibles en Modulos/Analisis_modelos:")
            
            if analisis_modelos_dir.exists():
                for file in analisis_modelos_dir.glob("*.py"):
                    print(f"   - {file.name}")
    
    except Exception as e:
        print(f"❌ Error general: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
