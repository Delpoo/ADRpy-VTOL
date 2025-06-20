"""
Módulo de Análisis de Modelos Imputados
=====================================

Este módulo proporciona herramientas interactivas para visualizar y analizar
los modelos de imputación generados por la pipeline de procesamiento de datos.

Módulos principales:
- data_loader: Carga y procesamiento de datos del JSON
- plot_utils: Utilidades para generación de gráficos interactivos  
- ui_components: Componentes de interfaz reutilizables
- main_visualizacion_modelos: Función principal de visualización
"""

__version__ = "1.0.0"
__author__ = "ADRpy VTOL Analysis Team"

# Imports condicionales para evitar errores si las dependencias no están instaladas
try:
    from .main_visualizacion_modelos import main_visualizacion_modelos
    __all__ = ['main_visualizacion_modelos']
except ImportError as e:
    print(f"Advertencia: No se pudieron importar todos los módulos: {e}")
    print("Instale las dependencias necesarias: pip install dash plotly pandas")
    __all__ = []
