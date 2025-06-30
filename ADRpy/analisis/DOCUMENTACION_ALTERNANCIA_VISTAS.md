# Documentación: Reestructuración con Alternancia de Vistas 2D/3D

## Resumen de la Implementación

Se ha reestructurado completamente la aplicación Dash para que las vistas 2D y 3D alternen ocupando el mismo espacio, usando Tabs de Dash para navegar entre ellas.

## Componentes Principales

### 1. Layout Principal (ui_components.py)

#### Estructura de Alternancia:
```
view-tabs (dcc.Tabs)
├── 2d-view (📊 Vista 2D - 1 Predictor)
├── 3d-view (🧊 Vista 3D - 2 Predictores)  
├── comparison-view (📈 Comparación)
└── metrics-view (📋 Métricas)

unified-plot-area
├── 2d-plot-container (plot-2d)
└── 3d-plot-container (plot-3d)
```

#### Panel de Resumen Unificado:
- **Indicadores de Estado**: Vista activa, conteo de modelos 2D y 3D
- **Controles de Filtrado**: Radio items para filtrar tabla por vista
- **Tabla Unificada**: Muestra modelos de ambas vistas con etiquetas

#### Stores de Estado:
- `active-view-store`: Vista activa ('2d-view' | '3d-view')
- `filtered-models-2d-store`: Modelos filtrados para vista 2D
- `filtered-models-3d-store`: Modelos filtrados para vista 3D
- `selected-model-store`: Modelo seleccionado (aeronave, parámetro, índice)
- `filter-state-store`: Estado actual de todos los filtros

### 2. Lógica de Callbacks (main_visualizacion_modelos.py)

#### Callback de Alternancia de Vistas:
```python
@app.callback(
    [Output('2d-plot-container', 'style'),
     Output('3d-plot-container', 'style'),
     Output('active-view-store', 'data'),
     Output('active-view-indicator', 'children')],
    [Input('view-tabs', 'value')]
)
```
**Comportamiento**: Solo el contenedor correspondiente a la vista activa es visible.

#### Callback de Filtrado Inteligente:
```python
@app.callback(
    [Output('filtered-models-2d-store', 'data'),
     Output('filtered-models-3d-store', 'data'),
     Output('models-2d-count', 'children'),
     Output('models-3d-count', 'children'),
     Output('total-models-count', 'children'),
     Output('filter-state-store', 'data')],
    [Input('aeronave-dropdown', 'value'),
     Input('parametro-dropdown', 'value'),
     Input('tipo-modelo-checklist', 'value'),
     Input('predictor-dropdown', 'value'),
     # ... otros filtros
    ]
)
```

#### Callbacks de Actualización de Gráficos:
- **Gráfico 2D**: Usa `filtered-models-2d-store`
- **Gráfico 3D**: Usa `filtered-models-3d-store`

## Lógica de Actualización Cruzada

### Filtros Globales vs Específicos

#### Filtros Globales (Afectan Ambas Vistas):
1. **Aeronave**: Cambia los parámetros disponibles y filtra ambas vistas
2. **Parámetro**: Define la celda de datos activa para ambas vistas
3. **Tipo de Modelo**: Filtra modelos por tipo en ambas vistas
4. **Métodos de Imputación**: Afecta visualización en ambas vistas
5. **Opciones de Visualización**: Puntos de entrenamiento, curvas, etc.

#### Filtros Específicos (Solo Vista Activa):
1. **Predictor**: Solo afecta vista 2D (los modelos 3D siempre tienen 2 predictores)
2. **Tipo de Comparación**: Principalmente para vista 2D

### Flujo de Actualización

1. **Cambio de Filtro Global**:
   ```
   Filtro Global → Filtrado Inteligente → Actualiza ambos stores → 
   Actualiza ambos gráficos → Actualiza contadores → Actualiza tabla
   ```

2. **Cambio de Filtro Específico**:
   ```
   Filtro Específico → Filtrado Inteligente → Actualiza store correspondiente → 
   Actualiza gráfico activo → Actualiza contadores → Actualiza tabla
   ```

3. **Cambio de Vista Activa**:
   ```
   Tab Change → Alternancia de Containers → Actualiza Vista Store → 
   Gráfico correspondiente se hace visible → Panel se actualiza
   ```

### Separación de Modelos por Vista

#### Vista 2D:
- **Modelos incluidos**: Todos los modelos con 1 predictor
- **Filtrado**: Se excluyen explícitamente modelos con 2+ predictores
- **Gráfico**: Scatter plot 2D con curvas de modelos

#### Vista 3D:
- **Modelos incluidos**: Solo modelos con exactamente 2 predictores  
- **Filtrado**: Se filtran solo modelos de 2 predictores
- **Gráfico**: Superficie 3D con puntos de entrenamiento

## Panel de Resumen Unificado

### Indicadores de Estado:
- **Vista Activa**: Muestra qué vista está seleccionada
- **Modelos 2D**: Conteo de modelos disponibles en vista 2D
- **Modelos 3D**: Conteo de modelos disponibles en vista 3D  
- **Total Visible**: Suma de modelos visibles según filtros

### Filtros de Tabla:
- **Mostrar Todos**: Combina modelos 2D y 3D con etiquetas
- **Solo 2D**: Solo modelos de la vista 2D
- **Solo 3D**: Solo modelos de la vista 3D
- **Solo Vista Activa**: Modelos de la vista actualmente visible

### Sincronización de Selección:
- Selección en tabla actualiza el `selected-model-store`
- Modelo seleccionado se resalta en el gráfico correspondiente
- Click en gráfico actualiza selección en tabla
- Información del modelo se muestra en panel lateral

## Ventajas de la Nueva Arquitectura

1. **Espacio Optimizado**: Las vistas comparten el mismo espacio, maximizando el área de visualización
2. **Filtrado Inteligente**: Los filtros se aplican apropiadamente según su alcance (global vs específico)
3. **Estado Consistente**: Los stores mantienen sincronización entre vistas y controles
4. **Experiencia Unificada**: Panel de resumen combina información de ambas vistas
5. **Navegación Intuitiva**: Tabs claros para alternar entre vistas
6. **Rendimiento**: Solo se actualiza el gráfico de la vista activa

## Casos de Uso

### Análisis Comparativo:
1. Seleccionar aeronave y parámetro
2. Ver modelos de 1 predictor en vista 2D
3. Cambiar a vista 3D para ver modelos de 2 predictores
4. Usar tabla unificada para comparar métricas entre ambos tipos

### Filtrado Progresivo:
1. Aplicar filtros globales (aeronave, tipo de modelo)
2. Alternar entre vistas para ver efectos
3. Aplicar filtros específicos en vista activa
4. Observar cambios en contadores y tabla

### Selección y Detalle:
1. Seleccionar modelo en tabla o gráfico
2. Ver información detallada en panel lateral
3. Modelo se resalta en gráfico correspondiente
4. Navegación fluida entre vistas manteniendo selección

## Estructura de Archivos Modificados

```
ADRpy/analisis/Modulos/Analisis_modelos/
├── ui_components.py           # Layout con tabs y panel unificado
├── main_visualizacion_modelos.py  # Callbacks de alternancia y filtrado
└── DOCUMENTACION_ALTERNANCIA_VISTAS.md  # Esta documentación
```

## Pruebas Recomendadas

1. **Alternancia de Vistas**: Verificar que solo una vista es visible
2. **Filtros Globales**: Comprobar que afectan ambas vistas
3. **Filtros Específicos**: Verificar que solo afectan vista activa
4. **Sincronización**: Selección en tabla/gráfico se refleja correctamente
5. **Contadores**: Números de modelos son correctos para cada vista
6. **Performance**: Cambios de vista son fluidos sin re-cálculos innecesarios
