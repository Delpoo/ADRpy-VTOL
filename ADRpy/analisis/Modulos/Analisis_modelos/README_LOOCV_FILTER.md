# Filtro Opcional para Modelos sin Confianza LOOCV

## 📋 Resumen de Cambios

Se ha implementado un filtro opcional en el panel lateral para mostrar/ocultar modelos sin confianza LOOCV, eliminando el filtro rígido que los ocultaba por defecto.

## 🔄 Cambios Implementados

### 1. Nuevo Control de UI (`ui_components.py`)
- ✅ Agregado checkbox "Mostrar modelos sin validación LOOCV" en `create_visualization_options()`
- ✅ **Por defecto activado** - Muestra todos los modelos siempre
- ✅ ID del componente: `show-models-without-loocv`

### 2. Lógica de Filtrado Actualizada (`data_loader.py`)
- ✅ Cambiado valor por defecto de `require_loocv` de `True` a `False`
- ✅ Ahora por defecto muestra **todos los modelos** (con y sin LOOCV)
- ✅ Documentación actualizada para reflejar el nuevo comportamiento

### 3. Callbacks Actualizados (`main_visualizacion_modelos.py`)
- ✅ Agregado input `show-models-without-loocv` a todos los callbacks relevantes:
  - `update_main_plot()` - Gráfica principal y tabla resumen
  - `update_info_panel()` - Panel de información de modelos
  - `sync_model_selection()` - Sincronización de selección
- ✅ Lógica invertida: cuando checkbox está marcado = mostrar sin LOOCV = require_loocv = False

## 🎯 Comportamiento Final

### Estado por Defecto
- ✅ **Todos los modelos visibles** (con y sin confianza LOOCV)
- ✅ El filtro aparece como una opción en el panel lateral
- ✅ Usuario puede optar por ocultar modelos sin LOOCV desmarcando el checkbox

### Lógica del Filtro
```python
# Checkbox marcado: "Mostrar modelos sin validación LOOCV" = True
# → require_loocv = False → Muestra TODOS los modelos

# Checkbox desmarcado: No mostrar modelos sin LOOCV
# → require_loocv = True → Solo modelos con confianza LOOCV
```

## 🔍 Modelos Afectados

### Modelos CON Confianza LOOCV
```json
{
  "Confianza_LOOCV": 0.2806030928499668,
  "k_LOOCV": 7,
  "Corr_LOOCV": 0.6343244624016869,
  "MAPE_LOOCV": 9.928132767057278,
  "R2_LOOCV": 0.9305244426071924
}
```

### Modelos SIN Confianza LOOCV
```json
{
  "Confianza_LOOCV": null,
  "k_LOOCV": null,
  "Corr_LOOCV": null,
  "MAPE_LOOCV": null,
  "R2_LOOCV": null
}
```

## ✅ Verificación de Funcionamiento

1. **Panel de Control**: ✅ Funciona correctamente
2. **Filtro Visible**: ✅ Aparece en opciones de visualización
3. **Estado Predeterminado**: ✅ Todos los modelos visibles
4. **Filtrado Opcional**: ✅ Usuario puede ocultar modelos sin LOOCV
5. **Callbacks**: ✅ Todos actualizados y sincronizados

## 🎉 Beneficios

- **Flexibilidad**: Usuario decide qué modelos ver
- **Transparencia**: Todos los modelos disponibles por defecto
- **Control Granular**: Filtrado específico según necesidades
- **Retrocompatibilidad**: Mantiene funcionalidad existente
- **UX Mejorada**: Opciones claras y intuitivas

## 📝 Archivos Modificados

1. `ui_components.py` - Nuevo checkbox en panel lateral
2. `data_loader.py` - Valor por defecto del filtro cambiado
3. `main_visualizacion_modelos.py` - Callbacks actualizados con nueva lógica

## 🚀 Próximos Pasos

El filtro está completamente implementado y listo para usar. Los usuarios ahora pueden:
- Ver todos los modelos por defecto
- Opcionalmente filtrar modelos sin validación LOOCV
- Tener control completo sobre la visualización
