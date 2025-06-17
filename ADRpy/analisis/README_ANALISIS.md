# 🎯 Sistema de Análisis Visual Dinámico para Imputación por Correlación

## 📋 Descripción

Este sistema proporciona una interfaz visual e interactiva para analizar y comparar modelos de imputación por correlación en datasets de aeronaves. Permite evaluar diferentes tipos de modelos (lineales, polinómicos, logarítmicos, potencia, exponenciales) y comparar su rendimiento de manera intuitiva.

## 🚀 Inicio Rápido

### 1. Prerrequisitos

Asegúrese de tener instaladas las siguientes librerías:

```bash
pip install pandas numpy plotly ipywidgets scipy scikit-learn openpyxl
```

### 2. Estructura de Archivos

```
ADRpy/analisis/
├── analisis_modelos_imputacion.ipynb  # Notebook principal
├── Modulos/
│   ├── imputacion_correlacion.py      # Módulo de imputación
│   └── ...                            # Otros módulos
├── Data/
│   ├── Datos_aeronaves.xlsx           # Datos principales
│   └── ...                            # Otros archivos de datos
└── README_ANALISIS.md                 # Este archivo
```

### 3. Ejecución

1. **Abrir Jupyter Notebook/Lab**:
   ```bash
   jupyter notebook
   # o
   jupyter lab
   ```

2. **Abrir el archivo**: `analisis_modelos_imputacion.ipynb`

3. **Ejecutar las celdas en orden**:
   - Sección 1: Importar librerías
   - Sección 2: Definir clases principales
   - Sección 3: Funciones de visualización
   - Sección 4: Interfaz interactiva
   - Sección 5: Inicialización
   - Sección 6: Ejecución de imputación
   - Sección 7: Interfaz principal

4. **Usar la interfaz interactiva** que aparecerá al final

## 🎮 Uso de la Interfaz

### Controles Principales

- **Selector de Celda**: Elige aeronave y parámetro a analizar
- **Selector de Tipo**: Selecciona tipo de modelo para análisis detallado
- **Selector de Criterio**: Cambia el criterio de ranking de modelos
- **Botones**: Actualizar, exportar, ayuda

### Gráficas

- **Gráfica Izquierda**: Comparación entre tipos de modelos
- **Gráfica Derecha**: Análisis detallado dentro del tipo seleccionado

### Pestañas de Análisis

- **📊 Métricas Comparativas**: Tabla de métricas por tipo
- **📈 Análisis de Residuos**: Detección de outliers y distribución
- **💡 Recomendaciones**: Sugerencias automáticas de mejora

## 📊 Funciones Avanzadas

### Análisis Ad-Hoc

```python
# Ver celdas disponibles
listar_celdas_disponibles()

# Análisis rápido de una celda específica
analisis_rapido_celda(aeronave_idx=5, parametro="Peso_Vacio")

# Comparar predictores para un parámetro
comparar_predictores("Potencia_Motor")

# Crear visualización 3D (modelos con 2 predictores)
crear_analisis_3d("aeronave_5_parametro_Peso_Vacio", "linear")

# Exportar reporte completo
exportar_reporte_completo("mi_analisis.xlsx")
```

### Test del Sistema

```python
# Verificar estado del sistema
test_sistema_completo()

# Información detallada
info_sistema()
```

## 📈 Interpretación de Métricas

### MAPE (Mean Absolute Percentage Error)
- **< 3%**: Excelente precisión
- **3-5%**: Buena precisión
- **5-7.5%**: Precisión aceptable
- **> 7.5%**: Precisión problemática

### R² (Coeficiente de Determinación)
- **> 0.9**: Excelente ajuste
- **0.8-0.9**: Buen ajuste
- **0.6-0.8**: Ajuste aceptable
- **< 0.6**: Ajuste insuficiente

### Confianza (Métrica Combinada)
- **> 0.8**: Alta confianza
- **0.6-0.8**: Confianza media
- **< 0.6**: Baja confianza

## 🔧 Características Técnicas

### Tipos de Modelos Soportados
- **Linear**: y = a + bx
- **Polinómico**: y = a + bx + cx²
- **Logarítmico**: y = a + b×ln(x)
- **Potencia**: y = a × x^b
- **Exponencial**: y = a × e^(bx)

### Validación
- **Validación Cruzada Leave-One-Out (LOOCV)**
- **Análisis de residuos**
- **Detección de outliers**
- **Test de normalidad**

### Exportación
- **Excel**: Reportes tabulares completos
- **HTML**: Gráficas interactivas
- **Métricas**: Comparaciones detalladas

## 🚨 Troubleshooting

### Problemas Comunes

**"No hay modelos disponibles"**
- Verificar que los datos se cargaron correctamente
- Ejecutar la imputación completa
- Revisar que hay valores faltantes en los datos

**"Error en visualización"**
- Verificar que la celda tiene modelos válidos
- Comprobar que el tipo seleccionado existe
- Actualizar la visualización manualmente

**"Error de importación"**
- Verificar que está en el directorio correcto
- Instalar dependencias faltantes
- Revisar la estructura de archivos

### Soluciones

1. **Reinstalar dependencias**:
   ```bash
   pip install --upgrade pandas numpy plotly ipywidgets
   ```

2. **Verificar estructura**:
   - Comprobar que existe `Modulos/imputacion_correlacion.py`
   - Verificar archivos en `Data/`

3. **Ejecutar en orden**:
   - No saltar celdas
   - Esperar que termine cada sección

## 📝 Notas Técnicas

- El sistema está optimizado para datasets de aeronaves
- Maneja automáticamente diferentes familias de aeronaves
- Implementa filtros de validez para modelos
- Usa unidades originales en todas las métricas
- Soporta análisis tanto 2D como 3D

## 🎯 Flujo de Trabajo Recomendado

1. **Exploración**: Usar interfaz para patrones generales
2. **Análisis**: Funciones avanzadas para casos específicos  
3. **Validación**: Revisar residuos y recomendaciones
4. **Documentación**: Exportar reportes finales

---

**Desarrollado para el proyecto ADRpy-VTOL**  
*Sistema de análisis avanzado para imputación de datos aeronáuticos*
