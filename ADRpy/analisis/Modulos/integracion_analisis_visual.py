"""
INTEGRACIÓN PRINCIPAL: MAIN.PY CON ANÁLISIS VISUAL

Este archivo contiene las modificaciones necesarias para integrar el sistema de 
análisis visual con el flujo principal de imputación, generando diccionarios de 
modelos que pueden ser analizados en el notebook interactivo.
"""

import pandas as pd
import os
import sys

def modificar_main_con_analisis_visual():
    """
    Función que modifica el flujo principal de main.py para incluir la generación
    de diccionarios de modelos y el análisis visual integrado.
    
    Esta función debería llamarse AL FINAL de main.py, después de todas las 
    imputaciones pero ANTES de la exportación final.
    """
    
    print("\n" + "=" * 60)
    print("🎯 INTEGRACIÓN CON ANÁLISIS VISUAL DE MODELOS")
    print("=" * 60)
    
    # Verificar que tenemos los datos necesarios
    try:
        # Estas variables deberían existir en el scope de main.py
        # df_original debería haberse guardado al inicio
        # df_procesado_actualizado viene del bucle de imputación  
        # detalles_para_excel viene del bucle de imputación
        
        # Verificar variables en el scope global
        frame = sys._getframe(1)  # Frame del caller (main.py)
        local_vars = frame.f_locals
        global_vars = frame.f_globals
        
        # Buscar variables necesarias
        df_original = None
        df_resultado = None
        detalles_para_excel = None
        
        for var_name in ['df_original', 'df_inicial', 'df_procesado_original']:
            if var_name in local_vars:
                df_original = local_vars[var_name]
                print(f"✅ Encontrado {var_name} como DataFrame original")
                break
            elif var_name in global_vars:
                df_original = global_vars[var_name]
                print(f"✅ Encontrado {var_name} como DataFrame original (global)")
                break
        
        for var_name in ['df_procesado_actualizado', 'df_resultado', 'df_final']:
            if var_name in local_vars:
                df_resultado = local_vars[var_name]
                print(f"✅ Encontrado {var_name} como DataFrame resultado")
                break
            elif var_name in global_vars:
                df_resultado = global_vars[var_name]
                print(f"✅ Encontrado {var_name} como DataFrame resultado (global)")
                break
        
        for var_name in ['detalles_para_excel', 'imputaciones_finales', 'reporte']:
            if var_name in local_vars:
                detalles_para_excel = local_vars[var_name]
                print(f"✅ Encontrado {var_name} como detalles para Excel")
                break
            elif var_name in global_vars:
                detalles_para_excel = global_vars[var_name]
                print(f"✅ Encontrado {var_name} como detalles para Excel (global)")
                break
        
        if df_original is None:
            print("❌ No se encontró DataFrame original")
            print("💡 Asegúrese de guardar df_original = df.copy() al inicio de main.py")
            return None
            
        if df_resultado is None:
            print("❌ No se encontró DataFrame resultado")
            print("💡 Asegúrese de tener df_procesado_actualizado del bucle de imputación")
            return None
        
        print(f"📊 DataFrame original: {df_original.shape}")
        print(f"📊 DataFrame resultado: {df_resultado.shape}")
        
        # OPCIÓN A: Generar diccionarios post-hoc (reentrenando)
        print("\n🔧 OPCIÓN A: Generando diccionarios post-hoc...")
        diccionarios_modelos = generar_diccionarios_post_hoc(
            df_original, df_resultado, detalles_para_excel
        )
        
        if diccionarios_modelos and len(diccionarios_modelos) > 0:
            print(f"✅ Diccionarios generados: {len(diccionarios_modelos)} celdas")
            
            # Ejecutar análisis visual
            return ejecutar_analisis_visual_final(
                df_original, diccionarios_modelos, df_resultado
            )
        else:
            print("❌ No se pudieron generar diccionarios")
            return None
            
    except Exception as e:
        print(f"❌ Error en integración: {e}")
        print("💡 Continuando con el flujo normal...")
        return None


def generar_diccionarios_post_hoc(df_original, df_resultado, detalles_para_excel):
    """
    Genera diccionarios de modelos post-hoc reentrando solo las celdas que fueron
    imputadas por correlación en el flujo principal.
    """
    
    diccionarios_modelos = {}
    
    if not detalles_para_excel:
        print("⚠️ No hay detalles para Excel - no se pueden generar diccionarios")
        return diccionarios_modelos
    
    # Filtrar solo imputaciones por correlación
    celdas_correlacion = [
        detalle for detalle in detalles_para_excel
        if detalle.get("Método predictivo") == "Correlacion"
    ]
    
    print(f"🎯 Celdas imputadas por correlación: {len(celdas_correlacion)}")
    
    for detalle in celdas_correlacion:
        try:
            aeronave = detalle.get("Aeronave")
            parametro = detalle.get("Parámetro")
            valor_imputado = detalle.get("Valor imputado")
            
            if aeronave is None or parametro is None:
                continue
            
            clave_celda = f"aeronave_{aeronave}_parametro_{parametro}"
            print(f"  🔄 Procesando: {clave_celda}")
            
            # Reentrenar modelos para esta celda específica
            diccionario_celda = reentrenar_modelos_para_celda(
                df_original, aeronave, parametro, valor_imputado, detalle
            )
            
            if diccionario_celda:
                diccionarios_modelos[clave_celda] = diccionario_celda
                print(f"    ✅ Diccionario generado")
            else:
                print(f"    ❌ Error generando diccionario")
                
        except Exception as e:
            print(f"    ❌ Error procesando celda: {e}")
    
    return diccionarios_modelos


def reentrenar_modelos_para_celda(df_original, aeronave_idx, parametro, valor_imputado, detalle_original):
    """
    Reentrena modelos para una celda específica usando la misma lógica que 
    imputacion_correlacion.py pero guardando todos los modelos.
    """
    
    try:
        # Importar funciones necesarias
        from Modulos.imputacion_correlacion import (
            seleccionar_predictores_validos, generar_combinaciones, entrenar_modelo
        )
        
        # Seleccionar predictores válidos (misma lógica que el original)
        df_filtrado, familia_usada, filtro_aplicado = seleccionar_predictores_validos(
            df_original, parametro, aeronave_idx
        )
        
        if df_filtrado.empty:
            return None
        
        # Obtener predictores
        predictores = [
            col for col in df_filtrado.columns 
            if col != df_filtrado.columns[0] and col != parametro
        ]
        
        if not predictores:
            return None
        
        # Generar combinaciones y entrenar TODOS los modelos
        combos = generar_combinaciones(predictores)
        todos_los_modelos = []
        
        for combo in combos:
            # Modelos normales
            for poly in [False, True]:
                try:
                    modelo = entrenar_modelo(df_filtrado, parametro, combo, poly, aeronave_idx)
                    if modelo and not modelo.get("descartado", False):
                        todos_los_modelos.append(modelo)
                except:
                    pass
            
            # Modelos especiales (solo para 1 predictor)
            if len(combo) == 1:
                for modelo_tipo in ["log", "potencia", "exp"]:
                    try:
                        modelo = entrenar_modelo(df_filtrado, parametro, combo, False, aeronave_idx, modelo_tipo)
                        if modelo and not modelo.get("descartado", False):
                            todos_los_modelos.append(modelo)
                    except:
                        pass
        
        if not todos_los_modelos:
            return None
        
        # Crear modelo dummy del mejor (el que se usó en la imputación real)
        mejor_modelo_dummy = {
            "tipo": detalle_original.get("Tipo Modelo", "linear"),
            "predictores": detalle_original.get("Predictores", "").split(","),
            "mape": float(detalle_original.get("MAPE", 0)) if detalle_original.get("MAPE") else 0,
            "r2": float(detalle_original.get("R2", 0)) if detalle_original.get("R2") else 0,
            "Confianza": float(detalle_original.get("Confianza", 0)) if detalle_original.get("Confianza") else 0,
            "valor_imputado": valor_imputado,
            "es_mejor_original": True  # Marca para identificarlo
        }
        
        # Buscar el modelo que coincide o usar el dummy
        mejor_modelo = None
        for modelo in todos_los_modelos:
            if (modelo.get("tipo") == mejor_modelo_dummy["tipo"] and 
                set(modelo.get("predictores", [])) == set(mejor_modelo_dummy["predictores"])):
                mejor_modelo = modelo
                mejor_modelo["es_mejor_original"] = True
                break
        
        if not mejor_modelo:
            mejor_modelo = mejor_modelo_dummy
            todos_los_modelos.append(mejor_modelo)
        
        # Crear diccionario completo
        diccionario_celda = {
            "mejor_modelo": mejor_modelo,
            "todos_los_modelos": todos_los_modelos,
            "df_filtrado": df_filtrado,
            "predictores": predictores,
            "familia_usada": familia_usada,
            "filtro_aplicado": filtro_aplicado,
            "metodo_generacion": "post_hoc",
            "detalle_original": detalle_original
        }
        
        return diccionario_celda
        
    except Exception as e:
        print(f"      ❌ Error reentrando modelos: {e}")
        return None


def ejecutar_analisis_visual_final(df_original, diccionarios_modelos, df_resultado):
    """
    Ejecuta el análisis visual usando los diccionarios generados.
    """
    
    try:
        print("\n🎯 EJECUTANDO ANÁLISIS VISUAL")
        print("=" * 35)
        
        # Guardar variables en el namespace global para que el notebook las encuentre
        import __main__
        __main__.df_original_analisis = df_original
        __main__.diccionarios_modelos_analisis = diccionarios_modelos
        __main__.df_resultado_analisis = df_resultado
        
        print("✅ Variables guardadas en namespace global:")
        print("  - df_original_analisis")
        print("  - diccionarios_modelos_analisis") 
        print("  - df_resultado_analisis")
        
        print(f"\n📊 Resumen:")
        print(f"  - DataFrame original: {df_original.shape}")
        print(f"  - Diccionarios de modelos: {len(diccionarios_modelos)} celdas")
        print(f"  - DataFrame resultado: {df_resultado.shape}")
        
        print("\n💡 SIGUIENTE PASO:")
        print("=" * 20)
        print("1. Abra el notebook: analisis_modelos_imputacion.ipynb")
        print("2. Ejecute las celdas de definición de clases")
        print("3. Ejecute la siguiente celda:")
        print()
        print("```python")
        print("# Cargar desde el flujo principal")
        print("analizador.cargar_desde_bucle_imputacion(")
        print("    df_original_analisis,")
        print("    [],  # detalles_para_excel (no necesario)")
        print("    diccionarios_modelos_analisis")
        print(")")
        print()
        print("# Crear interfaz")
        print("interfaz = InterfazInteractiva(analizador)")
        print("interfaz.mostrar_interfaz_completa()")
        print("```")
        
        return True
        
    except Exception as e:
        print(f"❌ Error ejecutando análisis visual: {e}")
        return False


# FUNCIÓN PARA AGREGAR AL FINAL DE MAIN.PY
def integrar_analisis_visual_en_main():
    """
    Función que debe agregarse al final de main.py para integrar el análisis visual.
    
    INSTRUCCIONES DE USO:
    =====================
    
    1. Al INICIO de main.py, después de cargar datos, agregar:
       ```python
       df_original = df.copy()  # ← IMPORTANTE: Guardar copia original
       ```
    
    2. Al FINAL de main.py, ANTES de la exportación, agregar:
       ```python
       # Integración con análisis visual
       try:
           from Modulos.integracion_analisis_visual import integrar_analisis_visual_en_main
           integrar_analisis_visual_en_main()
       except Exception as e:
           print(f"⚠️ Error en análisis visual: {e}")
           print("💡 Continuando con el flujo normal...")
       ```
    """
    
    print("\n🚀 INTEGRACIÓN DE ANÁLISIS VISUAL ACTIVADA")
    return modificar_main_con_analisis_visual()


if __name__ == "__main__":
    print("📘 MÓDULO DE INTEGRACIÓN DE ANÁLISIS VISUAL")
    print("=" * 45)
    print("Este módulo integra el sistema de análisis visual con main.py")
    print("Para usar, siga las instrucciones en la función integrar_analisis_visual_en_main()")
