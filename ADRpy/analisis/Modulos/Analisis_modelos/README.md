# Módulo de Análisis Interactivo de Modelos de Imputación

## 📋 Descripción

Este módulo proporciona una interfaz web interactiva para visualizar, analizar y comparar los modelos de imputación generados por la pipeline de procesamiento de datos de aeronaves VTOL.

## ✨ Características Principales

- **🌐 Interfaz web interactiva** con Plotly Dash
- **🔍 Filtros encadenados** por aeronave, parámetro, tipo de modelo y predictores
- **📊 Visualización avanzada** con puntos de datos y curvas de regresión
- **📈 Métricas comprensivas** (MAPE, R², Correlación, Confianza)
- **🎛️ Panel informativo** con ecuaciones y detalles de modelos
- **⚠️ Gestión robusta** de errores y advertencias
- **🔄 Alternativa por consola** si Dash no está disponible

## 🏗️ Arquitectura

```
Modulos/Analisis_modelos/
├── __init__.py                    # Punto de entrada del módulo
├── main_visualizacion_modelos.py  # Aplicación principal Dash
├── data_loader.py                 # Carga y procesamiento de datos
├── plot_utils.py                  # Utilidades de visualización
├── ui_components.py               # Componentes de interfaz
├── requirements.txt               # Dependencias
└── README.md                      # Este archivo
```

## 🚀 Instalación Rápida

### Opción 1: Script automático
```bash
cd ADRpy/analisis
python install_analisis_modelos.py
```

### Opción 2: Manual
```bash
pip install dash>=2.14.0 plotly>=5.17.0 pandas>=1.5.0 numpy>=1.21.0
```

## 💻 Uso

### Desde Notebook (Recomendado)
```python
# Abrir notebook_analisis_modelos.ipynb y ejecutar las celdas
```

### Desde Script
```python
from Modulos.Analisis_modelos import main_visualizacion_modelos

# Ejecutar con configuración por defecto
main_visualizacion_modelos()

# O con configuración personalizada
main_visualizacion_modelos(
    json_path="ruta/al/archivo.json",
    use_dash=True,
    port=8050,
    debug=False
)
```

### Desde Línea de Comandos
```bash
python launch_analisis_modelos.py
```

## 📊 Datos de Entrada

### Archivo JSON Requerido
- **Ubicación**: `ADRpy/analisis/Results/modelos_completos_por_celda.json`
- **Generado por**: Pipeline de imputación de la tesis

### Estructura de Datos
```json
{
  "modelos_por_celda": {
    "Aeronave|Parámetro": [
      {
        "tipo": "linear-1",
        "predictores": ["Potencia HP"],
        "ecuacion_string": "y = -4.29 + 1.43*x0",
        "mape": 3.626,
        "r2": 0.991,
        "Confianza": 0.387,
        "datos_entrenamiento": {...}
      }
    ]
  },
  "detalles_por_celda": {
    "Aeronave|Parámetro": {
      "final": {...},
      "similitud": {...},
      "correlacion": {...}
    }
  }
}
```

## 🎯 Funcionalidades Detalladas

### Filtros Disponibles

1. **Aeronave**: Selección única de aeronave de interés
2. **Parámetro**: Parámetro objetivo filtrado por aeronave
3. **Tipo de Modelo**: Múltiple selección (linear, poly, log, exp, pot)
4. **N° Predictores**: Filtro por cantidad de variables predictoras
5. **Predictores Específicos**: Selección de variables concretas

### Visualizaciones

- **Gráfico Principal**: Datos originales + datos de entrenamiento + curvas de modelos
- **Hover Detallado**: Ecuaciones, métricas, advertencias
- **Panel Lateral**: Información ampliada del modelo seleccionado
- **Tabla Comparativa**: Métricas de todos los modelos filtrados

### Métricas Mostradas

- **MAPE**: Error Absoluto Porcentual Medio
- **R²**: Coeficiente de Determinación
- **Correlación**: Coeficiente de correlación combinado
- **Confianza**: Medida ajustada por penalización de complejidad
- **N° Muestras**: Cantidad de datos de entrenamiento
- **Advertencias**: Validaciones y limitaciones del modelo

## 🛠️ Desarrollo y Extensión

### Añadir Nuevos Tipos de Modelo

1. Modificar `data_loader.py`:
```python
def get_model_predictions_safe(modelo, x_range):
    # Añadir nuevo caso en el if/elif
    elif 'nuevo_tipo' in tipo:
        predictions = nueva_formula(x_range, coef, intercept)
```

2. Actualizar `plot_utils.py`:
```python
SYMBOLS = {
    'nuevo_tipo': 'triangle-down'  # Añadir símbolo
}
```

### Añadir Nuevas Métricas

1. Modificar la función `get_model_info_text()` en `data_loader.py`
2. Actualizar `format_model_info()` en `ui_components.py`
3. Extender `create_metrics_summary_table()` en `plot_utils.py`

### Soporte para Múltiples Predictores

El módulo está preparado para extensión a modelos de 2+ predictores:

- Los filtros ya manejan múltiples predictores
- La estructura de datos soporta n predictores
- Se requiere implementar visualizaciones 3D en `plot_utils.py`

## 🐛 Solución de Problemas

### Error: Módulo no encontrado
```bash
# Verificar instalación
pip list | grep dash

# Reinstalar si es necesario
pip install --upgrade dash plotly pandas numpy
```

### Error: Archivo JSON no encontrado
- Verificar que existe `Results/modelos_completos_por_celda.json`
- Ejecutar la pipeline de imputación para generar el archivo
- Verificar permisos de lectura del archivo

### Error: Puerto ocupado
```python
# Cambiar puerto en la configuración
main_visualizacion_modelos(port=8051)
```

### Problemas de Rendimiento
- Usar filtros para reducir cantidad de datos mostrados
- Cerrar otras aplicaciones que usen memoria
- Considerar usar la versión por consola para debugging

## 📈 Limitaciones Actuales

- **Visualización**: Solo modelos de 1 predictor (2+ en desarrollo)
- **Exportación**: No implementada (próxima versión)
- **Caching**: Sin optimización para datasets muy grandes
- **Validación cruzada**: No implementada automáticamente

## 🤝 Contribución

Para contribuir al módulo:

1. Seguir la estructura modular existente
2. Documentar funciones con docstrings claros
3. Incluir manejo de errores robusto
4. Mantener compatibilidad con la estructura JSON actual
5. Añadir tests para nuevas funcionalidades

## 📞 Soporte

Para problemas o sugerencias:
- Revisar la documentación en el notebook
- Ejecutar el script de instalación
- Verificar logs de error en la consola
- Contactar al equipo de desarrollo del proyecto ADRpy-VTOL

---

## 📅 Historial de Versiones

### v1.0.0 (Actual)
- ✅ Implementación inicial completa
- ✅ Interfaz Dash con filtros encadenados
- ✅ Visualización de modelos de 1 predictor
- ✅ Panel de información detallada
- ✅ Manejo robusto de errores
- ✅ Alternativa por consola

### Próximas versiones
- 🔄 Soporte para modelos de múltiples predictores
- 🔄 Exportación de reportes
- 🔄 Análisis estadístico avanzado
- 🔄 Optimizaciones de rendimiento
