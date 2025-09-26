# Guía rápida (GUIA.md)

Este proyecto toma un **Excel con celdas faltantes** y genera un **Excel imputado** con valores predichos. El método principal es la **imputación por correlación/modelado**; opcionalmente puede alternar con **similitud** (modo mixto).

---

## Archivos que agregamos
- `requirements.txt`: librerías para instalar con `pip install -r requirements.txt`.
- `helper.py`: ejecutable/runner que orquesta carga → limpieza → (derivados) → imputación → exportación.

> **¿Guía vs helper?**  
> *Helper* (o *runner*) es un **módulo de utilidades/ejecución** que corre el pipeline.  
> *Guía* es **documentación** (este archivo) que explica **cómo** usar y **qué** parámetros tocar.

---

## Flujo de trabajo

1. **Cargar datos** desde Excel (`config_and_loading.cargar_datos`).         Esto prepara el `DataFrame` base para el pipeline. fileciteturn9file0L1-L6

2. **Procesar/limpiar** (manejo de duplicados, conversiones a numérico) con `data_processing.procesar_datos_y_manejar_duplicados`.         Si hay duplicados, se puede optar por conservar el primero automáticamente. fileciteturn9file0L8-L15 fileciteturn9file13L30-L41

3. **(Opcional) Derivados**: completar campos calculados consistentes antes de imputar (`derivados.completar_campos_derivados`).         Útil para que otros modelos dispongan de variables auxiliares coherentes. fileciteturn9file14L13-L21

4. **Imputación**:  
   - **Correlación** con `imputacion_correlacion.imputaciones_correlacion(...)` (recomendado por defecto).  
     Aplica familias F0→F1→F2 y *opcionalmente* `sin filtro` si activás el flag correspondiente. fileciteturn9file7L12-L18 fileciteturn9file12L33-L41
   - **Mixto (similitud↔correlación)** con `imputation_loop.bucle_imputacion_similitud_correlacion(...)`.           Alterna métodos y combina resultados por confianza, exportando también un JSON consolidado. fileciteturn9file1L21-L30 fileciteturn9file8L53-L61

5. **Exportar Excel**: `excel_export.exportar_excel_con_imputaciones(...)` agrega comentarios detallados y formatos.         Si falla o no está disponible, el helper hace `to_excel(...)` simple como respaldo. fileciteturn9file15L9-L16

---

## Parámetros clave para “tunear” el resultado

### 1) Umbrales de datos mínimos (correlación)
- **MIN_UNICOS**: cantidad mínima de valores **únicos** requerida por tipo de modelo (`linear-1`, `linear-2`, `poly-1`, `poly-2`, etc.).  
- **MIN_MUESTRAS**: cantidad mínima de **muestras** por tipo de modelo.  
Ambos se definen en `imputacion_correlacion.py`. (Este helper permite override en tiempo de ejecución). fileciteturn8file0L42-L50

**Por CLI** (sin tocar el archivo fuente):
```bash
python helper.py --input datos.xlsx --method correlacion \
  --min-unicos "linear:1=5,2=5;poly:1=5,2=10" \      --min-muestras "linear:1=12,2=18;poly:1=14,2=20"
```

### 2) Permitir “sin filtro” (último recurso)
Por defecto **no** se intenta “sin filtro”. Activá el flag si querés habilitar ese fallback:
```bash
python helper.py --input datos.xlsx --method correlacion --permitir-sin-filtro
```
(La política de familias es F0→F1→F2→(sin filtro si está permitido).) fileciteturn9file7L14-L18 fileciteturn9file10L55-L69

### 3) Debug en consola
Agregá `--debug` para ver mensajes, resúmenes de faltantes y pasos del loop en modo mixto:
```bash
python helper.py --input datos.xlsx --method mixto --max-iter 3 --debug
```
(El bucle imprime estados por iteración y un resumen final de imputaciones válidas.) fileciteturn9file8L44-L52

### 4) Derivados previos
Por defecto el helper intenta completar derivados antes de imputar. Para desactivar:
```bash
python helper.py --input datos.xlsx --sin-derivados
```

---

## Ejemplos de uso

**A) Imputación por correlación (default)**
```bash
python helper.py --input dataset.xlsx
```

**B) Modo “mixto” (similitud + correlación) con 5 iteraciones**
```bash
python helper.py --input dataset.xlsx --method mixto --max-iter 5
```

**C) Permitir fallback “sin filtro” + umbrales custom + debug**
```bash
python helper.py --input dataset.xlsx --permitir-sin-filtro \
  --min-unicos "linear:1=6,2=8;poly:1=6,2=12" \      --min-muestras "linear:1=14,2=22;poly:1=16,2=26" --debug
```

---

## Salida
- Excel `*.imputado.xlsx` (o el nombre que especifiques en `--output`).  
- Si usás el modo mixto, el loop también genera un **JSON consolidado por celda** con modelos/metrics, útil para visualizaciones posteriores. fileciteturn8file0L381-L390 fileciteturn8file0L520-L527

---

## Notas y buenas prácticas
- Asegurate de que los nombres de columnas/filas estén limpios y consistentes (el preprocesamiento se encarga en gran parte). fileciteturn9file0L18-L25
- Si querés **forzar** valores “mínimos” diferentes de los predefinidos sin editar código fuente, usá los flags `--min-unicos` y `--min-muestras` de este helper.
- Activar `--debug` ayuda a entender cuándo no aparecen ciertos símbolos/advertencias: si un modelo queda descartado por **falta de datos mínimos** o por criterios de **LOOCV/MAPE/R²**, el reporte lo explicita. fileciteturn9file11L5-L12
- Si el exportador avanzado falla por cambios de esquema, siempre tendrás el `to_excel(...)` de respaldo.

---

## Instalación rápida
```bash
pip install -r requirements.txt
```

## Ejecución rápida
```bash
python helper.py --input datos.xlsx
```
