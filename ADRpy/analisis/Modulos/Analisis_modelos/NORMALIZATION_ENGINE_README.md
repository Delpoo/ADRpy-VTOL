# Motor de Normalización - Documentación Técnica

## Resumen

Este documento describe el nuevo sistema centralizado de normalización para la visualización de modelos de regresión en el proyecto ADRpy-VTOL. El sistema reemplaza la lógica dispersa de normalización y eliminó completamente las dependencias de campos LaTeX.

## Cambios Implementados

### 1. Eliminación de Campos LaTeX
- **Removidos**: `ecuacion_latex` y `ecuacion_normalizada_latex`
- **Conservado**: `ecuacion_string` (para display y hover en visualizaciones)
- **Beneficio**: Simplificación del código y eliminación de dependencias innecesarias

### 2. Creación del Motor de Normalización Central
**Archivo**: `normalization_engine.py`

#### Características principales:
- **Centralización**: Toda la lógica de normalización en un solo módulo
- **Coeficientes originales**: Usa solo `coeficientes_originales` e `intercepto_original` del JSON
- **Compatibilidad**: Funciona con modelos 1D (curvas) y 2D (superficies)
- **Tipos de modelo**: Soporta linear, polynomial, logarithmic, power, exponential

#### Métodos principales:
```python
# Predicción con coeficientes originales
predict_with_original_coefficients(modelo, X_values)

# Generación de datos para curvas 2D
generate_curve_data(modelo, predictor_name, celda_data, num_points=100)

# Generación de datos para superficies 3D
generate_surface_data(modelo, predictors, celda_data, grid_size=20)
```

### 3. Actualización de Scripts de Visualización

#### `plot_model_curves.py`
- **Recreado completamente** para usar el motor de normalización
- Eliminada lógica de normalización dispersa
- Todas las curvas generadas a través del motor central

#### `plot_interactive.py`
- Actualizado para usar el motor de normalización
- Eliminadas referencias a campos LaTeX
- Hover mejorado usando `ecuacion_string`

#### `plot_3d.py`
- Ya usaba `coeficientes_originales` correctamente
- Verificado que no tiene dependencias LaTeX

### 4. Actualización del JSON de Salida
**Archivo modificado**: `imputation_loop.py`

#### Estructura nueva del JSON:
```json
{
  "celda_id": {
    "info_general": {
      "variable_objetivo": "...",
      "predictores": [...],
      "n_predictores": 1
    },
    "imputaciones": {
      "correlacion": {...},
      "similitud": {...},
      "final": {...}
    },
    "modelos": [
      {
        "tipo_modelo": "linear",
        "n_predictores": 1,
        "coeficientes_originales": [...],
        "intercepto_original": 0.123,
        "ecuacion_string": "y = 0.123 + 0.456*x",
        "r2_score": 0.95,
        "loocv_r2_promedio": 0.89,
        "scaler_mean": [1.0, 2.0],
        "scaler_scale": [0.5, 0.8]
      }
    ]
  }
}
```

#### Campos eliminados:
- `ecuacion_latex`
- `ecuacion_normalizada_latex`
- `detalles_por_celda` (estructura antigua)

#### Campos conservados para visualización:
- `coeficientes_originales`: Coeficientes del modelo en escala original
- `intercepto_original`: Intercepto en escala original  
- `ecuacion_string`: Ecuación legible para display
- `scaler_mean` y `scaler_scale`: Para desnormalización cuando sea necesario

### 5. Limpieza del Notebook
- Actualizadas las celdas de diagnóstico
- Eliminadas las referencias LaTeX de outputs antiguos
- Añadidas celdas de prueba del motor de normalización

## Beneficios del Nuevo Sistema

### 1. Mantenibilidad
- **Un solo punto de control** para toda la lógica de normalización
- **Código más limpio** sin duplicación de lógica
- **Fácil debugging** centralizado

### 2. Consistencia
- **Misma lógica** para todas las visualizaciones (2D y 3D)
- **Mismos métodos** para todos los tipos de modelo
- **Resultados consistentes** entre diferentes vistas

### 3. Robustez
- **Manejo de errores** centralizado
- **Validación** de datos de entrada
- **Fallbacks** para casos edge

### 4. Rendimiento
- **Cache** interno para operaciones repetitivas
- **Optimización** de cálculos matriciales
- **Menor overhead** sin conversiones LaTeX

## Uso del Motor de Normalización

### Ejemplo básico:
```python
from normalization_engine import NormalizationEngine

# Crear instancia
engine = NormalizationEngine()

# Para un modelo 1D (curva)
curve_data = engine.generate_curve_data(
    modelo=modelo_dict,
    predictor_name="Potencia HP", 
    celda_data=celda_dict,
    num_points=100
)

# Para un modelo 2D (superficie)
surface_data = engine.generate_surface_data(
    modelo=modelo_dict,
    predictors=["Potencia HP", "Envergadura"],
    celda_data=celda_dict,
    grid_size=20
)
```

### Predicción directa:
```python
# Para datos específicos
X_test = [[100], [200], [300]]  # 1D
y_pred = engine.predict_with_original_coefficients(modelo, X_test)

# Para datos 2D
X_test_2d = [[100, 5], [200, 6], [300, 7]]  # 2D
y_pred_2d = engine.predict_with_original_coefficients(modelo, X_test_2d)
```

## Verificación del Sistema

### Tests realizados:
1. ✅ **Eliminación completa de campos LaTeX** del código y JSON
2. ✅ **Funcionamiento del motor de normalización** con modelos reales
3. ✅ **Consistencia de visualizaciones** 2D y 3D
4. ✅ **Compatibilidad con todos los tipos de modelo**
5. ✅ **Limpieza de archivos antiguos** con estructura obsoleta

### Scripts de diagnóstico:
- **Notebook células 2-5**: Verificación completa del sistema
- **Análisis de estructura JSON**: Sin campos LaTeX
- **Test del motor**: Funcionamiento correcto
- **Verificación de consistencia**: Todas las visualizaciones alineadas

## Consideraciones Futuras

### Posibles mejoras:
1. **Cache persistente** para modelos frecuentemente accedidos
2. **Validación más estricta** de entrada de datos
3. **Métricas de rendimiento** para optimización
4. **Documentación automática** de ecuaciones

### Mantenimiento:
- **Revisar el motor** ante cambios en estructura de modelos
- **Actualizar tests** si se añaden nuevos tipos de modelo
- **Monitorear rendimiento** en datasets grandes

## Estado Final

✅ **Sistema completamente migrado** al nuevo motor de normalización  
✅ **Eliminación total** de dependencias LaTeX  
✅ **Código centralizado y mantenible**  
✅ **Visualizaciones consistentes** en todo el proyecto  
✅ **JSON limpio** sin campos redundantes  
✅ **Documentación completa** del nuevo sistema

---

**Fecha de implementación**: Enero 2025  
**Autor**: Sistema automatizado de refactoring  
**Estado**: Completado y verificado
