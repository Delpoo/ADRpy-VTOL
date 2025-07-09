#!/usr/bin/env python3
"""
Test simple para verificar el cálculo de puntos teóricos después de las correcciones.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'Modulos'))

import numpy as np
import logging

# Configurar logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_punto_teorico_manual():
    """
    Test manual para verificar que el cálculo de puntos teóricos sea correcto.
    """
    print("=== TEST MANUAL DE PUNTO TEÓRICO ===")
    
    # Crear un modelo de ejemplo con datos conocidos
    modelo_ejemplo = {
        'tipo': 'poly-2',
        'n_predictores': 2,
        'predictores': ['Peso máximo al despegue (MTOW)', 'Potencia HP'],
        'coeficientes_originales': [0.1, 0.2, 0.01, 0.005, 0.02],  # [x1, x2, x1², x1*x2, x2²]
        'intercepto_original': 100.0,
        'variable_independiente_1': 1000.0,  # MTOW
        'variable_independiente_2': 500.0,   # Potencia HP
        'datos_entrenamiento': {
            'X_original': [[800, 400], [1200, 600], [900, 450], [1100, 550]],
            'y_original': [150, 200, 160, 180]
        }
    }
    
    print("Modelo de ejemplo creado:")
    print(f"  Tipo: {modelo_ejemplo['tipo']}")
    print(f"  Predictores: {modelo_ejemplo['predictores']}")
    print(f"  Coeficientes: {modelo_ejemplo['coeficientes_originales']}")
    print(f"  Intercepto: {modelo_ejemplo['intercepto_original']}")
    print(f"  Variable 1: {modelo_ejemplo['variable_independiente_1']}")
    print(f"  Variable 2: {modelo_ejemplo['variable_independiente_2']}")
    
    # Calcular valor teórico manualmente usando la ecuación correcta
    intercep = modelo_ejemplo['intercepto_original']
    coefs = modelo_ejemplo['coeficientes_originales']
    x1 = modelo_ejemplo['variable_independiente_1']
    x2 = modelo_ejemplo['variable_independiente_2']
    
    # Según PolynomialFeatures: [x1, x2, x1², x1*x2, x2²]
    valor_teorico = (intercep + 
                    coefs[0] * x1 +           # x1
                    coefs[1] * x2 +           # x2
                    coefs[2] * (x1 ** 2) +    # x1²
                    coefs[3] * x1 * x2 +      # x1*x2
                    coefs[4] * (x2 ** 2))     # x2²
    
    print(f"\nCálculo manual del valor teórico:")
    print(f"  y = {intercep} + {coefs[0]}*{x1} + {coefs[1]}*{x2} + {coefs[2]}*{x1**2} + {coefs[3]}*{x1*x2} + {coefs[4]}*{x2**2}")
    print(f"  y = {intercep} + {coefs[0]*x1} + {coefs[1]*x2} + {coefs[2]*x1**2} + {coefs[3]*x1*x2} + {coefs[4]*x2**2}")
    print(f"  y = {valor_teorico}")
    
    # Ahora usar la función de extracción de puntos teóricos
    try:
        from Modulos.Analisis_modelos.plot_model_curves import extract_theoretical_imputation_points
        
        puntos_teoricos = extract_theoretical_imputation_points(
            [modelo_ejemplo], 
            "TEST|PARAM", 
            n_predictores_filter=2
        )
        
        print(f"\nResultado de extract_theoretical_imputation_points:")
        print(f"  Número de puntos: {len(puntos_teoricos)}")
        
        if puntos_teoricos:
            punto = puntos_teoricos[0]
            print(f"  Punto extraído:")
            print(f"    x (normalizado): {punto.get('x')}")
            print(f"    y (normalizado): {punto.get('y')}")
            print(f"    z (valor original): {punto.get('z')}")
            print(f"    x_original: {punto.get('x_original')}")
            print(f"    y_original: {punto.get('y_original')}")
            print(f"    z_original: {punto.get('z_original')}")
            
            # Comparar el valor teórico calculado con el extraído
            z_extraido = punto.get('z_original', punto.get('z'))
            diferencia = abs(valor_teorico - z_extraido) if z_extraido is not None else float('inf')
            
            print(f"\nComparación:")
            print(f"  Valor teórico manual: {valor_teorico}")
            print(f"  Valor extraído: {z_extraido}")
            print(f"  Diferencia: {diferencia}")
            
            if diferencia < 0.001:
                print("  ✅ VALORES COINCIDEN - Cálculo correcto")
            else:
                print("  ❌ VALORES NO COINCIDEN - Hay un error en el cálculo")
        else:
            print("  ❌ No se extrajo ningún punto teórico")
            
    except Exception as e:
        print(f"Error al extraer puntos teóricos: {e}")
        import traceback
        traceback.print_exc()
    
    return valor_teorico

if __name__ == "__main__":
    try:
        resultado = test_punto_teorico_manual()
        print(f"\n=== TEST COMPLETADO ===")
        print(f"Valor teórico calculado: {resultado}")
    except Exception as e:
        print(f"Error en test: {e}")
        import traceback
        traceback.print_exc()
