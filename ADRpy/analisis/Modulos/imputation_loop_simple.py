"""
VERSIÓN SIMPLIFICADA DEL BUCLE CON DICCIONARIOS
Solo para demostración y prueba del sistema de análisis visual.
"""

import pandas as pd
import sys
import os

# Agregar el directorio actual al path para importaciones
sys.path.append(os.path.dirname(__file__))

def bucle_imputacion_similitud_correlacion_con_diccionarios(
    df_parametros,
    df_atributos,
    parametros_preseleccionados,
    bloques_rasgos,
    capas_familia,
    df_procesado,
    max_iteraciones=3,
    debug_mode=False,
    generar_diccionarios=True
):
    """
    Versión simplificada que llama al bucle original pero intenta generar diccionarios.
    
    Si generar_diccionarios=True, intenta usar la versión con diccionarios.
    Si falla, hace fallback al bucle original.
    """
    
    print("🔧 INTENTANDO USAR BUCLE CON DICCIONARIOS...")
    
    # Intentar importar la versión con diccionarios
    if generar_diccionarios:
        try:
            from imputacion_correlacion_con_diccionarios import imputaciones_correlacion_con_diccionarios
            print("✅ Módulo con diccionarios disponible")
            
            # Por simplicidad, simular generación de diccionarios llamando a la función original
            # y luego generando diccionarios post-hoc
            
            # Llamar al bucle original primero
            from imputation_loop import bucle_imputacion_similitud_correlacion
            
            df_resultado, resumen, imputaciones, detalles = bucle_imputacion_similitud_correlacion(
                df_parametros=df_parametros,
                df_atributos=df_atributos,
                parametros_preseleccionados=parametros_preseleccionados,
                bloques_rasgos=bloques_rasgos,
                capas_familia=capas_familia,
                df_procesado=df_procesado,
                debug_mode=debug_mode
            )
            
            # Generar diccionarios post-hoc para las celdas imputadas por correlación
            diccionarios_modelos = generar_diccionarios_post_hoc_simple(
                df_procesado, df_resultado, detalles
            )
            
            if diccionarios_modelos:
                print(f"✅ Diccionarios generados post-hoc: {len(diccionarios_modelos)} celdas")
                return df_resultado, resumen, imputaciones, detalles, diccionarios_modelos
            else:
                print("⚠️ No se pudieron generar diccionarios - modo fallback")
                return df_resultado, resumen, imputaciones, detalles
                
        except Exception as e:
            print(f"❌ Error usando versión con diccionarios: {e}")
            print("🔄 Fallback al bucle original...")
    
    # Fallback al bucle original
    try:
        from imputation_loop import bucle_imputacion_similitud_correlacion
        
        resultado = bucle_imputacion_similitud_correlacion(
            df_parametros=df_parametros,
            df_atributos=df_atributos,
            parametros_preseleccionados=parametros_preseleccionados,
            bloques_rasgos=bloques_rasgos,
            capas_familia=capas_familia,
            df_procesado=df_procesado,
            debug_mode=debug_mode
        )
        
        print("✅ Bucle original ejecutado correctamente")
        return resultado
        
    except Exception as e:
        print(f"❌ Error crítico en bucle: {e}")
        raise


def generar_diccionarios_post_hoc_simple(df_original, df_resultado, detalles_para_excel):
    """
    Genera diccionarios post-hoc simplificados para demostración.
    """
    
    diccionarios = {}
    
    if not detalles_para_excel:
        return diccionarios
    
    # Debug: mostrar todos los métodos detectados
    metodos_detectados = [detalle.get("Método predictivo", "N/A") for detalle in detalles_para_excel]
    print(f"  🔍 DEBUG - Métodos detectados: {metodos_detectados}")
    
    # Filtrar imputaciones que involucren correlación (pura o híbrida)
    celdas_correlacion = [
        detalle for detalle in detalles_para_excel
        if "Correlacion" in detalle.get("Método predictivo", "")
    ]
    
    print(f"  🎯 Generando diccionarios para {len(celdas_correlacion)} celdas de correlación...")
    
    for detalle in celdas_correlacion:
        try:
            aeronave = detalle.get("Aeronave")
            parametro = detalle.get("Parámetro")
            
            if aeronave is None or parametro is None:
                continue
                
            clave_celda = f"aeronave_{aeronave}_parametro_{parametro}"
            
            # Crear un diccionario simplificado con la información disponible
            modelo_simple = {
                "tipo": detalle.get("Tipo Modelo", "linear"),
                "predictores": detalle.get("Predictores", "").split(",") if detalle.get("Predictores") else [],
                "mape": float(detalle.get("MAPE", 0)) if detalle.get("MAPE") else 0,
                "r2": float(detalle.get("R2", 0)) if detalle.get("R2") else 0,
                "Confianza": float(detalle.get("Confianza", 0)) if detalle.get("Confianza") else 0,
                "valor_imputado": detalle.get("Valor imputado"),
                "k": int(detalle.get("k", 0)) if detalle.get("k") else 0,
                "origen": "post_hoc_simple"
            }
            
            diccionario_celda = {
                "mejor_modelo": modelo_simple,
                "todos_los_modelos": [modelo_simple],  # Solo tenemos el modelo usado
                "df_filtrado": df_original,  # Simplificación: usar todo el DataFrame
                "predictores": modelo_simple["predictores"],
                "familia_usada": detalle.get("Familia", "sin_filtro"),
                "filtro_aplicado": True,
                "metodo_generacion": "post_hoc_simple",
                "detalle_original": detalle
            }
            
            diccionarios[clave_celda] = diccionario_celda
            
        except Exception as e:
            print(f"    ❌ Error procesando {detalle}: {e}")
            continue
    
    return diccionarios


if __name__ == "__main__":
    print("📘 MÓDULO SIMPLIFICADO DE BUCLE CON DICCIONARIOS")
    print("Este módulo proporciona una versión simplificada para testing.")
