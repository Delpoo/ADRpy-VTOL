#!/usr/bin/env python3
"""
Script para debugear un modelo específico y verificar los valores de 
imputación teóricos calculados.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'Modulos'))

from Modulos.imputacion_correlacion import cargar_y_validar_datos, imputaciones_correlacion
import logging
import json

# Configurar logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def debug_modelo_especifico():
    """
    Carga los datos y ejecuta la imputación para debugear un modelo específico.
    """
    print("=== DEBUG MODELO ESPECÍFICO ===")
    
    # Cargar datos
    ruta_datos = r"C:\Users\delpi\OneDrive\Tesis\ADRpy-VTOL\ADRpy\analisis\Data\Datos_aeronaves.xlsx"
    df = cargar_y_validar_datos(ruta_datos)
    
    print(f"DataFrame cargado: {df.shape}")
    print(f"Columnas: {df.columns.tolist()}")
    
    # Ejecutar imputación con exportación de modelos
    df_resultado, reporte = imputaciones_correlacion(df, exportar_modelos=True)
    
    print(f"\nResultado de imputación: {len(reporte)} imputaciones")
    
    # Buscar un modelo específico para debugear
    modelo_ejemplo = None
    for item in reporte:
        if (item.get('informacion_modelos_celda') and 
            item['informacion_modelos_celda'].get('modelos') and 
            len(item['informacion_modelos_celda']['modelos']) > 0):
            modelo_ejemplo = item
            break
    
    if not modelo_ejemplo:
        print("No se encontró un modelo válido para debugear")
        return
    
    print(f"\n=== DEBUGEANDO MODELO EJEMPLO ===")
    print(f"Aeronave: {modelo_ejemplo.get('Aeronave')}")
    print(f"Parámetro: {modelo_ejemplo.get('Parámetro')}")
    
    modelos = modelo_ejemplo['informacion_modelos_celda']['modelos']
    print(f"Número de modelos: {len(modelos)}")
    
    for i, modelo in enumerate(modelos[:3]):  # Solo los primeros 3
        print(f"\n--- Modelo {i+1} ---")
        print(f"Tipo: {modelo.get('tipo')}")
        print(f"Predictores: {modelo.get('predictores')}")
        print(f"N predictores: {modelo.get('n_predictores')}")
        
        # Verificar datos de entrenamiento
        datos_entrenamiento = modelo.get('datos_entrenamiento', {})
        print(f"Datos de entrenamiento: {bool(datos_entrenamiento)}")
        
        if datos_entrenamiento:
            X_orig = datos_entrenamiento.get('X_original', [])
            y_orig = datos_entrenamiento.get('y_original', [])
            print(f"  X_original: {len(X_orig)} filas")
            print(f"  y_original: {len(y_orig)} filas")
            
            if X_orig and y_orig:
                print(f"  X_original[0]: {X_orig[0] if X_orig else 'N/A'}")
                print(f"  y_original[0]: {y_orig[0] if y_orig else 'N/A'}")
        
        # Verificar coeficientes
        coef_orig = modelo.get('coeficientes_originales', [])
        intercep_orig = modelo.get('intercepto_original', 0)
        print(f"Coeficientes originales: {coef_orig}")
        print(f"Intercepto original: {intercep_orig}")
        
        # Verificar variables independientes
        var_indep_1 = modelo.get('variable_independiente_1')
        var_indep_2 = modelo.get('variable_independiente_2')
        print(f"Variable independiente 1: {var_indep_1}")
        print(f"Variable independiente 2: {var_indep_2}")
        
        # Calcular valor teórico manualmente
        if var_indep_1 is not None and coef_orig and intercep_orig is not None:
            try:
                tipo_modelo = modelo.get('tipo', '')
                n_predictores = modelo.get('n_predictores', 0)
                
                if tipo_modelo == 'linear-1' and n_predictores == 1:
                    valor_teorico = intercep_orig + coef_orig[0] * var_indep_1
                    print(f"  Valor teórico calculado (linear-1): {valor_teorico}")
                    
                elif tipo_modelo == 'poly-1' and n_predictores == 1:
                    valor_teorico = intercep_orig
                    for grado, coef in enumerate(coef_orig):
                        valor_teorico += coef * (var_indep_1 ** (grado + 1))
                    print(f"  Valor teórico calculado (poly-1): {valor_teorico}")
                    
                elif tipo_modelo == 'linear-2' and n_predictores == 2 and var_indep_2 is not None:
                    valor_teorico = intercep_orig + coef_orig[0] * var_indep_1 + coef_orig[1] * var_indep_2
                    print(f"  Valor teórico calculado (linear-2): {valor_teorico}")
                    
                elif tipo_modelo == 'poly-2' and n_predictores == 2 and var_indep_2 is not None:
                    if len(coef_orig) >= 5:
                        valor_teorico = (intercep_orig + 
                                       coef_orig[0] * var_indep_1 + 
                                       coef_orig[1] * var_indep_2 + 
                                       coef_orig[2] * (var_indep_1 ** 2) + 
                                       coef_orig[3] * var_indep_1 * var_indep_2 + 
                                       coef_orig[4] * (var_indep_2 ** 2))
                        print(f"  Valor teórico calculado (poly-2): {valor_teorico}")
                    else:
                        print(f"  Modelo poly-2 sin suficientes coeficientes")
                        
                else:
                    print(f"  Tipo de modelo no soportado para cálculo manual: {tipo_modelo}")
                    
            except Exception as e:
                print(f"  Error calculando valor teórico: {e}")
    
    return modelo_ejemplo

if __name__ == "__main__":
    try:
        modelo_debug = debug_modelo_especifico()
        print("\n=== DEBUG COMPLETADO ===")
    except Exception as e:
        print(f"Error en debug: {e}")
        import traceback
        traceback.print_exc()
