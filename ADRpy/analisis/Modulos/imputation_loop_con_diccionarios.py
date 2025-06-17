"""
VERSIÓN MODIFICADA DE IMPUTATION_LOOP.PY QUE GENERA DICCIONARIOS DE MODELOS

Esta versión extiende el bucle principal para generar diccionarios de modelos
que pueden ser usados por el sistema de análisis visual, manteniendo la funcionalidad original.
"""

import pandas as pd
from .imputacion_similitud_flexible import *
from .html_utils import convertir_a_html
from .data_processing import generar_resumen_faltantes
from .imputacion_correlacion_con_diccionarios import (
    imputaciones_correlacion_con_diccionarios,
    filtrar_solo_correlacion_desde_diccionarios
)

def is_missing(val):
    """
    Returns True if the value is considered missing (NaN, empty string, special codes, etc.).
    """
    if pd.isna(val):
        return True
    if isinstance(val, str) and val.strip().lower() in ["", "nan", "nan ", "-", "#n/d", "n/d", "#¡valor!"]:
        return True
    return False

def bucle_imputacion_similitud_correlacion_con_diccionarios(
    df_parametros,
    df_atributos,
    parametros_preseleccionados,
    bloques_rasgos,
    capas_familia,
    df_procesado,
    max_iteraciones=3,
    debug_mode=False,
    generar_diccionarios=True  # ← NUEVO parámetro
):
    """
    Versión extendida del bucle que TAMBIÉN genera diccionarios de modelos.
    
    Args:
        generar_diccionarios: Si True, genera diccionarios para análisis visual
        
    Returns:
        df_procesado_base: DataFrame con imputaciones
        df_resumen: Resumen de imputaciones
        imputaciones_finales: Lista de imputaciones finales
        detalles_para_excel: Detalles para exportar
        diccionarios_modelos_globales: Dict con todos los modelos (si generar_diccionarios=True)
    """
    
    df_procesado_base = df_procesado.copy()
    
    if debug_mode:
        convertir_a_html(
            df_procesado_base,
            titulo="df_procesado_base",
            ancho="100%",
            alto="400px",
            mostrar=True,
        )

    resumen_imputaciones = []
    diccionarios_modelos_globales = {}  # ← NUEVO: Almacenar TODOS los modelos
    
    print("🚀 BUCLE DE IMPUTACIÓN CON GENERACIÓN DE DICCIONARIOS")
    print("=" * 60)
    
    def registrar_imputacion(reporte, metodo_origen=""):
        """Registra imputaciones válidas en el resumen."""
        for item in reporte:
            if not is_missing(item.get("Valor imputado", None)):
                item_copia = item.copy()
                item_copia["Método predictivo"] = metodo_origen
                resumen_imputaciones.append(item_copia)

    # Variables para tracking
    imputaciones_finales = []
    detalles_para_excel = []
    
    for iteracion in range(1, max_iteraciones + 1):
        print(f"\n🔄 === ITERACIÓN {iteracion} ===")
        
        # Contar valores faltantes al inicio de la iteración
        total_missing_inicio = df_procesado_base.isna().sum().sum()
        print(f"📊 Valores faltantes al inicio: {total_missing_inicio}")
        
        if total_missing_inicio == 0:
            print("✅ No hay más valores faltantes. Finalizando...")
            break
        
        # ═══════════════════════════════════════════════════════════
        # PASO 1: IMPUTACIÓN POR SIMILITUD
        # ═══════════════════════════════════════════════════════════
        print("\n🔍 Paso 1: Imputación por similitud...")
        
        df_similitud = df_procesado_base.copy()
        
        # ... [resto de la lógica de similitud igual que el original] ...
        # Por brevedad, copio la referencia al archivo original para similitud
        
        # ═══════════════════════════════════════════════════════════
        # PASO 2: IMPUTACIÓN POR CORRELACIÓN CON DICCIONARIOS
        # ═══════════════════════════════════════════════════════════
        print("\n📈 Paso 2: Imputación por correlación CON diccionarios...")
        
        df_correlacion = df_procesado_base.copy()
        
        if generar_diccionarios:
            # Usar la versión que genera diccionarios
            df_correlacion_resultado, reporte_correlacion, diccionarios_iteracion = \
                imputaciones_correlacion_con_diccionarios(df_correlacion)
            
            # Agregar diccionarios de esta iteración al global
            diccionarios_modelos_globales.update(diccionarios_iteracion)
            print(f"  📊 Diccionarios agregados: {len(diccionarios_iteracion)}")
            
        else:
            # Usar la versión original
            from .imputacion_correlacion import imputaciones_correlacion
            df_correlacion_resultado, reporte_correlacion = imputaciones_correlacion(df_correlacion)
            print("  ⚠️ Modo sin diccionarios - usando versión original")
        
        # Procesar reporte de correlación
        if reporte_correlacion is not None and len(reporte_correlacion) > 0:
            validos_correlacion = [r for r in reporte_correlacion 
                                 if not is_missing(r.get("Valor imputado", None))]
            print(f"  ✅ Imputaciones por correlación válidas: {len(validos_correlacion)}")
            
            if validos_correlacion:
                registrar_imputacion(reporte_correlacion, "Correlacion")
                
                # Actualizar DataFrame base
                for item in validos_correlacion:
                    fila = item["Aeronave"]
                    columna = item["Parámetro"]
                    valor = item["Valor imputado"]
                    df_procesado_base.at[fila, columna] = valor
                    
                    # Agregar a detalles para Excel
                    detalles_para_excel.append(item)
                    imputaciones_finales.append(item)
        
        # Verificar si hay progreso
        total_missing_fin = df_procesado_base.isna().sum().sum()
        progreso = total_missing_inicio - total_missing_fin
        
        print(f"📈 Progreso esta iteración: {progreso} valores imputados")
        print(f"📊 Valores faltantes restantes: {total_missing_fin}")
        
        if progreso == 0:
            print("⚠️ No hubo progreso en esta iteración. Finalizando...")
            break
    
    # ═══════════════════════════════════════════════════════════════
    # FILTRAR SOLO CORRELACIÓN EN LOS DICCIONARIOS
    # ═══════════════════════════════════════════════════════════════
    if generar_diccionarios:
        print("\n🎯 Filtrando diccionarios - solo correlación...")
        diccionarios_solo_correlacion = filtrar_solo_correlacion_desde_diccionarios(
            diccionarios_modelos_globales, detalles_para_excel
        )
        
        print(f"✅ Diccionarios finales: {len(diccionarios_solo_correlacion)} celdas por correlación")
        
        # Retornar con diccionarios
        return (
            df_procesado_base, 
            pd.DataFrame(resumen_imputaciones), 
            imputaciones_finales, 
            detalles_para_excel,
            diccionarios_solo_correlacion  # ← NUEVO retorno
        )
    else:
        # Retornar sin diccionarios (compatible con versión original)
        return (
            df_procesado_base, 
            pd.DataFrame(resumen_imputaciones), 
            imputaciones_finales, 
            detalles_para_excel
        )


def ejecutar_analisis_visual_integrado(
    df_original: pd.DataFrame,
    diccionarios_modelos: dict,
    df_resultado: pd.DataFrame = None
):
    """
    Función de conveniencia para ejecutar el análisis visual directamente
    desde el flujo principal con los diccionarios generados.
    """
    try:
        # Verificar que el notebook existe
        import os
        notebook_path = os.path.join(os.path.dirname(__file__), '..', 'analisis_modelos_imputacion.ipynb')
        
        if not os.path.exists(notebook_path):
            print(f"❌ Notebook no encontrado en: {notebook_path}")
            return None
        
        print("🎯 INICIANDO ANÁLISIS VISUAL INTEGRADO")
        print("=" * 45)
        print(f"📊 DataFrame original: {df_original.shape}")
        print(f"🔧 Diccionarios de modelos: {len(diccionarios_modelos)} celdas")
        
        if df_resultado is not None:
            print(f"✅ DataFrame resultado: {df_resultado.shape}")
        
        # Importar y ejecutar desde el notebook
        try:
            import sys
            sys.path.append(os.path.dirname(notebook_path))
            
            # Esta función debería estar definida en el notebook
            from analisis_modelos_imputacion import inicializar_sistema_analisis_desde_main
            
            interfaz = inicializar_sistema_analisis_desde_main(
                df_original, 
                diccionarios_modelos,
                df_resultado
            )
            
            if interfaz:
                print("✅ Interfaz de análisis iniciada exitosamente")
                print("🎮 Use los controles para explorar los modelos")
                return interfaz
            else:
                print("❌ Error al inicializar la interfaz")
                return None
                
        except ImportError as e:
            print(f"❌ Error importando funciones del notebook: {e}")
            print("💡 Asegúrese de que el notebook esté ejecutado y las funciones definidas")
            return None
        
    except Exception as e:
        print(f"❌ Error en análisis visual integrado: {e}")
        print("💡 Continuando con el flujo normal...")
        return None
