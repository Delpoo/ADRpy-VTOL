#!/usr/bin/env python3
"""
Script directo para ejecutar la aplicación desde el directorio correcto
"""

import sys
import os
from pathlib import Path

# Cambiar al directorio de Analisis_modelos
os.chdir(r"c:\Users\delpi\OneDrive\Tesis\ADRpy-VTOL\ADRpy\analisis\Modulos\Analisis_modelos")

# Añadir el directorio actual al path
sys.path.insert(0, os.getcwd())

def main():
    """Función principal"""
    try:
        # Ahora los imports relativos deberían funcionar
        import main_visualizacion_modelos
        
        print("🚀 Iniciando aplicación de visualización de modelos...")
        
        # Ruta del JSON (relativa al directorio de analisis)
        json_path = r"c:\Users\delpi\OneDrive\Tesis\ADRpy-VTOL\ADRpy\analisis\Results\modelos_completos_por_celda.json"
        
        if not Path(json_path).exists():
            print(f"❌ Archivo JSON no encontrado: {json_path}")
            return
        
        print(f"📁 Usando archivo: {json_path}")
        print(f"📁 Directorio de trabajo: {os.getcwd()}")
        
        # Ejecutar aplicación
        main_visualizacion_modelos.main_visualizacion_modelos(
            json_path=json_path,
            use_dash=True,
            port=8050,
            debug=True
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
