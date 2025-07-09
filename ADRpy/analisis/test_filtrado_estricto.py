#!/usr/bin/env python3
"""
Test para verificar que la nueva lógica de filtrado estricto funciona correctamente.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'Modulos'))

import pandas as pd
import numpy as np
from Modulos.imputacion_correlacion import seleccionar_predictores_validos
import logging

# Configurar logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_filtrado_estricto():
    """
    Test para verificar que la nueva lógica de filtrado estricto elimina predictores
    cuando el valor objetivo está fuera del rango de entrenamiento.
    """
    print("=== TEST FILTRADO ESTRICTO ===")
    
    # Crear DataFrame de ejemplo
    df_test = pd.DataFrame({
        'Aeronave': ['A', 'B', 'C', 'D', 'E'],
        'MTOW': [1000, 1200, 1100, 1050, 5000],  # E tiene valor extremo
        'Potencia': [400, 500, 450, 425, 600],   # E ligeramente fuera
        'Velocidad': [100, 120, 110, 105, 115],  # E dentro del rango
        'Payload': [200, 250, 225, np.nan, 300] # E será el objetivo
    })
    
    print("DataFrame de prueba:")
    print(df_test)
    
    # Probar con aeronave E que tiene valores extremos
    objetivo = 'Payload'
    idx_objetivo = 4  # Aeronave E
    
    print(f"\n=== FILTRADO PARA AERONAVE {df_test.at[idx_objetivo, 'Aeronave']} ===")
    print(f"Objetivo: {objetivo}")
    print(f"Valores de entrenamiento para cada predictor:")
    
    for col in ['MTOW', 'Potencia', 'Velocidad']:
        valores_entrenamiento = df_test[col].drop(index=idx_objetivo).dropna()
        valor_objetivo = df_test.at[idx_objetivo, col]
        rango_min, rango_max = valores_entrenamiento.min(), valores_entrenamiento.max()
        
        print(f"  {col}:")
        print(f"    Rango de entrenamiento: [{rango_min}, {rango_max}]")
        print(f"    Valor objetivo: {valor_objetivo}")
        print(f"    ¿Dentro del rango?: {rango_min <= valor_objetivo <= rango_max}")
    
    # Aplicar filtrado
    try:
        df_filtrado, familia_usada, filtro_aplicado = seleccionar_predictores_validos(
            df_test, objetivo, idx_objetivo
        )
        
        print(f"\n=== RESULTADO DEL FILTRADO ===")
        print(f"Familia usada: {familia_usada}")
        print(f"Filtro aplicado: {filtro_aplicado}")
        print(f"Columnas antes del filtrado: {df_test.columns.tolist()}")
        print(f"Columnas después del filtrado: {df_filtrado.columns.tolist()}")
        
        # Verificar que se eliminaron las columnas esperadas
        predictores_esperados_eliminados = ['MTOW']  # MTOW debería eliminarse (5000 vs rango [1000-1200])
        predictores_esperados_conservados = ['Velocidad']  # Velocidad debería conservarse (115 vs rango [100-120])
        predictores_posiblemente_eliminados = ['Potencia']  # Potencia podría eliminarse (600 vs rango [400-500])
        
        for pred in predictores_esperados_eliminados:
            if pred in df_filtrado.columns:
                print(f"❌ ERROR: {pred} debería haber sido eliminado pero está presente")
            else:
                print(f"✅ CORRECTO: {pred} fue eliminado como se esperaba")
        
        for pred in predictores_esperados_conservados:
            if pred in df_filtrado.columns:
                print(f"✅ CORRECTO: {pred} fue conservado como se esperaba")
            else:
                print(f"❌ ERROR: {pred} debería haber sido conservado pero fue eliminado")
        
        for pred in predictores_posiblemente_eliminados:
            if pred in df_filtrado.columns:
                print(f"⚠️  NOTA: {pred} fue conservado (verificar si es correcto)")
            else:
                print(f"⚠️  NOTA: {pred} fue eliminado (verificar si es correcto)")
                
        print(f"\nDataFrame filtrado:")
        print(df_filtrado)
        
        return df_filtrado
        
    except Exception as e:
        print(f"❌ ERROR en el filtrado: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_casos_extremos():
    """
    Test con casos extremos para verificar robustez.
    """
    print("\n=== TEST CASOS EXTREMOS ===")
    
    # Caso 1: Todos los predictores fuera de rango
    df_extremo = pd.DataFrame({
        'Aeronave': ['A', 'B', 'C', 'D'],
        'Predictor1': [10, 20, 15, 1000],  # D extremo
        'Predictor2': [100, 200, 150, 2000],  # D extremo
        'Objetivo': [50, 60, 55, np.nan]  # D es objetivo
    })
    
    print("Caso 1: Todos los predictores extremos")
    try:
        df_filt, _, _ = seleccionar_predictores_validos(df_extremo, 'Objetivo', 3)
        predictores_finales = [col for col in df_filt.columns if col not in ['Aeronave', 'Objetivo']]
        print(f"  Predictores restantes: {predictores_finales}")
        if not predictores_finales:
            print("  ✅ CORRECTO: No quedan predictores válidos")
        else:
            print("  ❌ ERROR: Deberían eliminarse todos los predictores")
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
    
    # Caso 2: Todos los predictores dentro de rango
    df_normal = pd.DataFrame({
        'Aeronave': ['A', 'B', 'C', 'D'],
        'Predictor1': [10, 20, 15, 18],  # D dentro del rango
        'Predictor2': [100, 200, 150, 175],  # D dentro del rango
        'Objetivo': [50, 60, 55, np.nan]  # D es objetivo
    })
    
    print("\nCaso 2: Todos los predictores normales")
    try:
        df_filt, _, _ = seleccionar_predictores_validos(df_normal, 'Objetivo', 3)
        predictores_finales = [col for col in df_filt.columns if col not in ['Aeronave', 'Objetivo']]
        print(f"  Predictores restantes: {predictores_finales}")
        if len(predictores_finales) == 2:
            print("  ✅ CORRECTO: Se conservan todos los predictores")
        else:
            print("  ❌ ERROR: Deberían conservarse todos los predictores")
    except Exception as e:
        print(f"  ❌ ERROR: {e}")

if __name__ == "__main__":
    try:
        resultado = test_filtrado_estricto()
        test_casos_extremos()
        print("\n=== TEST COMPLETADO ===")
    except Exception as e:
        print(f"Error en test: {e}")
        import traceback
        traceback.print_exc()
