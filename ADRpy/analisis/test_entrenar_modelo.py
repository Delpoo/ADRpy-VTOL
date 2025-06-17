#!/usr/bin/env python3
"""
Script de prueba para verificar que la función entrenar_modelo 
exporta datos coherentes en unidades originales.
"""

import pandas as pd
import numpy as np
from Modulos.imputacion_correlacion import entrenar_modelo, cargar_y_validar_datos

def test_entrenar_modelo():
    """Prueba la función entrenar_modelo con diferentes tipos de modelos."""
    
    # Cargar datos de prueba
    try:
        df = cargar_y_validar_datos("Data/Datos_aeronaves.xlsx")
        print("Datos cargados exitosamente")
        print(f"Shape del DataFrame: {df.shape}")
        print(f"Columnas: {list(df.columns)}")
    except Exception as e:
        print(f"Error cargando datos: {e}")
        return
    
    # Seleccionar columnas numéricas válidas para las pruebas
    columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(columnas_numericas) < 2:
        print("Error: No hay suficientes columnas numéricas para la prueba")
        return
    
    objetivo = columnas_numericas[0]  # Primera columna como objetivo
    predictores_1 = (columnas_numericas[1],)  # Un predictor
    predictores_2 = (columnas_numericas[1], columnas_numericas[2]) if len(columnas_numericas) > 2 else predictores_1
    
    print(f"\nObjetivo: {objetivo}")
    print(f"Predictores (1 var): {predictores_1}")
    print(f"Predictores (2 var): {predictores_2}")
    
    # Encontrar una fila con datos válidos
    idx_prueba = None
    for i in df.index:
        if pd.notna(df.at[i, objetivo]) and all(pd.notna(df.at[i, p]) for p in predictores_1):
            idx_prueba = i
            break
    
    if idx_prueba is None:
        print("Error: No se encontró una fila con datos válidos para la prueba")
        return
        
    print(f"\nUsando fila índice: {idx_prueba}")
    
    # Simular datos faltantes en el objetivo para la prueba
    df_test = df.copy()
    df_test.at[idx_prueba, objetivo] = np.nan
    
    # Probar diferentes tipos de modelos
    modelos_prueba = [
        ("Lineal 1 var", predictores_1, False, None),
        ("Polinómico 1 var", predictores_1, True, None),
        ("Logarítmico", predictores_1, False, "log"),
        ("Potencia", predictores_1, False, "potencia"),
        ("Exponencial", predictores_1, False, "exp"),
    ]
    
    if len(predictores_2) == 2:
        modelos_prueba.extend([
            ("Lineal 2 var", predictores_2, False, None),
            ("Polinómico 2 var", predictores_2, True, None),
        ])
    
    print("\n" + "="*80)
    print("RESULTADOS DE LAS PRUEBAS")
    print("="*80)
    
    for nombre, predictores, poly, modelo_extra in modelos_prueba:
        print(f"\n{nombre}:")
        print("-" * len(nombre))
        
        try:
            resultado = entrenar_modelo(
                df_filtrado=df_test,
                objetivo=objetivo,
                predictores=predictores,
                poly=poly,
                idx=idx_prueba,
                modelo_extra=modelo_extra
            )
            
            if resultado is None:
                print("  ❌ Resultado: None (modelo no pudo ser entrenado)")
                continue
                
            if resultado.get("descartado", False):
                print(f"  ❌ Modelo descartado: {resultado.get('motivo', 'sin motivo')}")
                continue
            
            print("  ✅ Modelo entrenado exitosamente")
            print(f"     Tipo: {resultado.get('tipo', 'N/A')}")
            print(f"     Transformación: {resultado.get('tipo_transformacion', 'N/A')}")
            print(f"     N muestras: {resultado.get('n', 'N/A')}")
            print(f"     MAPE: {resultado.get('mape', 'N/A'):.2f}%")
            print(f"     R²: {resultado.get('r2', 'N/A'):.3f}")
            print(f"     Confianza: {resultado.get('Confianza', 'N/A'):.3f}")
            
            # Verificar datos originales
            datos_orig = resultado.get('datos_originales', {})
            if datos_orig:
                x_orig = datos_orig.get('X_original', [])
                y_orig = datos_orig.get('y_original', [])
                print(f"     Datos originales - X shape: {np.array(x_orig).shape}, Y shape: {np.array(y_orig).shape}")
            
            # Verificar coeficientes
            coef_orig = resultado.get('coeficientes_originales', [])
            intercept_orig = resultado.get('intercepto_original', 0)
            print(f"     Coeficientes originales: {coef_orig}")
            print(f"     Intercepto original: {intercept_orig}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print("\n" + "="*80)
    print("PRUEBA COMPLETADA")
    print("="*80)

if __name__ == "__main__":
    test_entrenar_modelo()
