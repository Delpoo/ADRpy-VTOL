"""
EJEMPLO DE INTEGRACIÓN DEL SISTEMA DE ANÁLISIS VISUAL CON MAIN.PY

Este archivo muestra cómo integrar el sistema de análisis visual dinámico
con el flujo principal de imputación por correlación.
"""

# ================================================================================
# MODIFICACIONES NECESARIAS EN MAIN.PY
# ================================================================================

def main_con_analisis_visual():
    """
    Versión modificada de main() que incluye el análisis visual.
    Integra el notebook de análisis al final del proceso de imputación.
    """
    
    print("🚀 SISTEMA PRINCIPAL DE IMPUTACIÓN CON ANÁLISIS VISUAL")
    print("=" * 55)
    
    # 1. CARGA DE DATOS (como en main.py original)
    df = cargar_y_validar_datos("ruta/a/datos.xlsx")
    df_original = df.copy()  # ⚠️ IMPORTANTE: Guardar copia original
    
    # 2. EJECUTAR IMPUTACIÓN (como en main.py original)
    print("🔄 Ejecutando imputación por correlación...")
    df_resultado, reporte = imputaciones_correlacion(df)
    
    # 3. RECOPILAR DICCIONARIOS DE MODELOS (NUEVO)
    print("📊 Recopilando modelos para análisis visual...")
    diccionarios_modelos_globales = recopilar_diccionarios_modelos(df_original)
    
    # 4. ANÁLISIS VISUAL DINÁMICO (NUEVO)
    print("\n🎯 INICIANDO ANÁLISIS VISUAL DINÁMICO")
    print("=" * 40)
    
    try:
        # Importar el sistema de análisis
        from analisis_modelos_imputacion import ejecutar_analisis_visual
        
        # Ejecutar análisis visual
        interfaz = ejecutar_analisis_visual(
            df_original=df_original,
            diccionarios_modelos=diccionarios_modelos_globales,
            df_resultado=df_resultado
        )
        
        if interfaz:
            print("✅ Análisis visual iniciado exitosamente")
            print("🎮 Use la interfaz interactiva para explorar los modelos")
        else:
            print("❌ Error iniciando análisis visual")
            
    except Exception as e:
        print(f"❌ Error en análisis visual: {e}")
        print("💡 Continuando con el flujo normal...")
    
    # 5. CONTINUAR CON EL RESTO DEL MAIN.PY
    # Exportación, reportes, etc.
    
    return df_resultado, reporte

def recopilar_diccionarios_modelos(df_original):
    """
    Función para recopilar todos los diccionarios de modelos durante la imputación.
    Esta función debe modificarse para capturar los modelos durante el proceso.
    """
    
    diccionarios_globales = {}
    
    # Iterar por cada celda con valores faltantes
    for objetivo in [c for c in df_original.columns if df_original[c].isna().any()]:
        faltantes = df_original[df_original[objetivo].isna()].index
        
        for idx in faltantes:
            clave_celda = f"aeronave_{idx}_parametro_{objetivo}"
            
            # Ejecutar la misma lógica que en imputaciones_correlacion
            # pero guardando TODOS los modelos, no solo el mejor
            
            try:
                # Preparar datos (misma lógica que imputacion_correlacion.py)
                df_filtrado, familia_usada, filtro_aplicado = seleccionar_predictores_validos(
                    df_original, objetivo, idx
                )
                
                if df_filtrado.empty:
                    continue
                
                predictores = [col for col in df_filtrado.columns 
                              if col != df_filtrado.columns[0] and col != objetivo]
                
                if not predictores:
                    continue
                
                # Entrenar TODOS los modelos (no solo el mejor)
                todos_los_modelos = entrenar_todos_los_modelos_para_analisis(
                    df_filtrado, objetivo, predictores, idx
                )
                
                # Seleccionar el mejor modelo
                mejor_modelo = seleccionar_mejor_modelo(todos_los_modelos)
                
                # Guardar información completa
                diccionarios_globales[clave_celda] = {
                    "mejor_modelo": mejor_modelo,
                    "todos_los_modelos": todos_los_modelos,
                    "df_filtrado": df_filtrado,
                    "predictores": predictores,
                    "familia_usada": familia_usada,
                    "filtro_aplicado": filtro_aplicado
                }
                
            except Exception as e:
                print(f"⚠️ Error procesando {clave_celda}: {e}")
                continue
    
    return diccionarios_globales

def entrenar_todos_los_modelos_para_analisis(df_filtrado, objetivo, predictores, idx):
    """
    Función para entrenar TODOS los modelos posibles para una celda.
    Esta función es similar a la lógica en AnalizadorModelos._entrenar_todos_los_modelos
    pero adaptada para usar durante el flujo principal.
    """
    
    modelos = []
    
    # Generar todas las combinaciones de predictores
    for combo in generar_combinaciones(predictores):
        # Modelos lineales y polinómicos
        for poly in (False, True):
            modelo = entrenar_modelo(df_filtrado, objetivo, combo, poly, idx)
            if modelo and not modelo.get("descartado", False):
                # Agregar validación LOOCV
                modelo.update(validar_con_loocv(df_filtrado, objetivo, modelo))
                modelo["Confianza_promedio"] = (
                    modelo["Confianza"] + modelo.get("Confianza_LOOCV", 0)
                ) / 2
                modelos.append(modelo)
        
        # Modelos especiales solo para 1 predictor
        if len(combo) == 1:
            for tipo_especial in ["log", "potencia", "exp"]:
                modelo = entrenar_modelo(
                    df_filtrado, objetivo, combo, False, idx, 
                    modelo_extra=tipo_especial
                )
                if modelo and not modelo.get("descartado", False):
                    # Agregar validación LOOCV
                    modelo.update(validar_con_loocv(df_filtrado, objetivo, modelo))
                    modelo["Confianza_promedio"] = (
                        modelo["Confianza"] + modelo.get("Confianza_LOOCV", 0)
                    ) / 2
                    modelos.append(modelo)
    
    return modelos

def seleccionar_mejor_modelo(modelos):
    """
    Seleccionar el mejor modelo usando los mismos criterios que imputacion_correlacion.py
    """
    if not modelos:
        return None
    
    # Filtrar modelos válidos
    validos = [m for m in modelos if m["mape"] <= 7.5 and m["r2"] >= 0.6]
    
    if not validos:
        return None
    
    # Seleccionar por máxima confianza promedio
    mejor = max(validos, key=lambda x: x["Confianza_promedio"])
    return mejor

# ================================================================================
# INSTRUCCIONES DE IMPLEMENTACIÓN
# ================================================================================

"""
PASOS PARA INTEGRAR EN EL MAIN.PY ACTUAL:

1. MODIFICAR imputaciones_correlacion():
   - Agregar parámetro 'recopilar_modelos=False'
   - Si es True, guardar todos los modelos en lugar de solo el mejor
   - Retornar diccionarios_modelos además de df_resultado y reporte

2. MODIFICAR main():
   - Guardar df_original antes de la imputación
   - Llamar imputaciones_correlacion con recopilar_modelos=True
   - Después de la imputación, llamar ejecutar_analisis_visual()

3. OPCIONAL - Crear función wrapper:
   - main_con_analisis() que incluya todo
   - Mantener main() original para compatibilidad

EJEMPLO DE USO FINAL:

if __name__ == "__main__":
    # Opción 1: Main tradicional
    # main()
    
    # Opción 2: Main con análisis visual
    main_con_analisis_visual()
"""

# ================================================================================
# CÓDIGO DE EJEMPLO PARA TESTING
# ================================================================================

def test_integracion():
    """
    Función de prueba para verificar la integración.
    """
    print("🧪 TEST DE INTEGRACIÓN")
    print("=" * 25)
    
    # Simular datos del flujo principal
    import pandas as pd
    import numpy as np
    
    # Crear datos de ejemplo
    df_original = pd.DataFrame({
        'aeronave': [1, 2, 3, 4, 5],
        'parametro_A': [10, 20, np.nan, 40, 50],
        'parametro_B': [100, 200, 300, np.nan, 500],
        'predictor_1': [1, 2, 3, 4, 5],
        'predictor_2': [10, 20, 30, 40, 50]
    })
    
    # Simular diccionarios de modelos
    diccionarios_ejemplo = {
        "aeronave_2_parametro_parametro_A": {
            "mejor_modelo": {
                "tipo": "linear",
                "mape": 5.0,
                "r2": 0.95,
                "Confianza": 0.8,
                "Confianza_LOOCV": 0.75,
                "Confianza_promedio": 0.775,
                "predictores": ["predictor_1"],
                "coeficientes_originales": [2.0],
                "intercepto_original": 5.0
            },
            "todos_los_modelos": [],  # Lista de todos los modelos
            "df_filtrado": df_original,
            "predictores": ["predictor_1", "predictor_2"],
            "familia_usada": "sin filtro",
            "filtro_aplicado": "ninguno"
        }
    }
    
    try:
        # Test de la función de integración
        interfaz = inicializar_sistema_analisis_desde_main(
            df_original, 
            diccionarios_ejemplo
        )
        
        if interfaz:
            print("✅ Test de integración exitoso")
            return True
        else:
            print("❌ Test de integración falló")
            return False
            
    except Exception as e:
        print(f"❌ Error en test: {e}")
        return False

if __name__ == "__main__":
    test_integracion()
