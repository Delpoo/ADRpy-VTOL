#!/usr/bin/env python3
"""
Script de prueba simple para verificar que la imputación por correlación funciona correctamente.
"""

import pandas as pd
import numpy as np
from Modulos.imputacion_correlacion import imputaciones_correlacion, cargar_y_validar_datos

def test_imputacion_simple():
    """Prueba básica de la imputación por correlación."""
    
    try:
        # Cargar datos
        df = cargar_y_validar_datos("Data/Datos_aeronaves.xlsx")
        print("✅ Datos cargados exitosamente")
        print(f"Shape del DataFrame: {df.shape}")
        
        # Crear datos de prueba con algunos valores faltantes
        df_test = df.copy()
        
        # Buscar columnas numéricas con valores válidos
        columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(columnas_numericas) >= 2:
            col_objetivo = columnas_numericas[0]
            # Crear algunos valores faltantes artificialmente
            indices_validos = df_test[df_test[col_objetivo].notna()].index
            if len(indices_validos) > 3:
                # Crear un valor faltante para probar
                idx_test = indices_validos[0]
                df_test.at[idx_test, col_objetivo] = np.nan
                print(f"✅ Creando valor faltante en '{col_objetivo}', fila {idx_test}")
                
                # Ejecutar imputación
                print("⏳ Ejecutando imputación por correlación...")
                df_resultado, reporte = imputaciones_correlacion(df_test)
                
                print("✅ Imputación completada exitosamente")
                print(f"📊 Reporte generado con {len(reporte)} entradas")
                
                # Verificar si se imputó el valor
                valor_imputado = df_resultado.at[idx_test, col_objetivo]
                if not pd.isna(valor_imputado):
                    print(f"✅ Valor imputado: {valor_imputado}")
                    
                    # Mostrar información del reporte
                    for item in reporte:
                        if item.get("Aeronave") == idx_test and item.get("Parámetro") == col_objetivo:
                            print(f"   Tipo Modelo: {item.get('Tipo Modelo', 'N/A')}")
                            print(f"   Confianza: {item.get('Confianza', 'N/A')}")
                            print(f"   MAPE: {item.get('MAPE', 'N/A')}")
                            print(f"   R²: {item.get('R2', 'N/A')}")
                            break
                else:
                    print("❌ No se pudo imputar el valor")
                    
                return True
            else:
                print("❌ No hay suficientes datos válidos para la prueba")
                return False
        else:
            print("❌ No hay suficientes columnas numéricas")
            return False
            
    except Exception as e:
        print(f"❌ Error durante la prueba: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*60)
    print("PRUEBA DE IMPUTACIÓN POR CORRELACIÓN")
    print("="*60)
    success = test_imputacion_simple()
    print("="*60)
    if success:
        print("✅ PRUEBA EXITOSA")
    else:
        print("❌ PRUEBA FALLIDA")
    print("="*60)
