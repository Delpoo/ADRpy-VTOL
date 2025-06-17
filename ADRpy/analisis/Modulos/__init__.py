"""
Paquete de módulos para el sistema de análisis de imputación por correlación.

Este paquete contiene todos los módulos necesarios para:
- Carga y configuración de datos
- Procesamiento de datos
- Bucles de imputación (similitud y correlación)
- Análisis visual de modelos
- Exportación de resultados
"""

__version__ = "1.0.0"
__author__ = "ADRpy Analysis System"

# Imports principales para facilitar el acceso
try:
    from .config_and_loading import *
    from .data_processing import *
    from .user_interaction import *
    from .imputation_loop import *
    from .html_utils import *
    from .excel_export import *
except ImportError:
    # En caso de dependencias faltantes, continuar sin errores
    pass
