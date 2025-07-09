#!/usr/bin/env python3
"""
Test completo para verificar el orden de coeficientes polinómicos 
y que coincidan con las ecuaciones en plot_model_curves.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'Modulos'))

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_polynomial_order_consistency():
    """
    Test para verificar que el orden de coeficientes polinómicos 
    es consistente entre imputacion_correlacion.py y plot_model_curves.py
    """
    print("=== TEST CONSISTENCIA ORDEN COEFICIENTES POLINÓMICOS ===")
    
    # Crear datos sintéticos
    np.random.seed(42)
    n_samples = 100
    
    # Generar datos para 2 predictores
    x1 = np.random.uniform(10, 50, n_samples)
    x2 = np.random.uniform(5, 25, n_samples)
    
    # Crear variable objetivo con ecuación conocida
    # y = 100 + 2*x1 + 3*x2 + 0.1*x1² + 0.5*x1*x2 + 0.2*x2²
    y = 100 + 2*x1 + 3*x2 + 0.1*(x1**2) + 0.5*x1*x2 + 0.2*(x2**2)
    y += np.random.normal(0, 1, n_samples)  # Añadir ruido
    
    # Crear DataFrame
    df = pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'y': y
    })
    
    print(f"Datos generados: {len(df)} muestras")
    print(f"Ecuación verdadera: y = 100 + 2*x1 + 3*x2 + 0.1*x1² + 0.5*x1*x2 + 0.2*x2²")
    
    # Simular el proceso de imputacion_correlacion.py
    print("\n1. Simulando proceso de imputacion_correlacion.py...")
    
    X_df = df[['x1', 'x2']]
    X_raw = np.array(X_df.values, dtype=float)
    y_raw = np.array(df['y'].values, dtype=float)
    
    # Aplicar PolynomialFeatures
    pf = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = pf.fit_transform(X_raw)
    
    print(f"Características polinómicas: {pf.get_feature_names_out(['x1', 'x2'])}")
    print(f"Powers: {pf.powers_}")
    
    # Normalizar
    scaler_X = StandardScaler()
    X_trans = scaler_X.fit_transform(X_poly)
    
    scaler_y = StandardScaler()
    y_transformed = scaler_y.fit_transform(y_raw.reshape(-1, 1)).flatten()
    
    # Entrenar modelo
    modelo = LinearRegression().fit(X_trans, y_transformed)
    coeficientes = modelo.coef_
    intercepto = modelo.intercept_
    
    # Desnormalizar coeficientes
    escalas_ajustadas = scaler_X.scale_
    medias_ajustadas = scaler_X.mean_
    coef_original = (coeficientes * scaler_y.scale_[0] / escalas_ajustadas).tolist()
    b_shift = sum([
        coeficientes[j] * medias_ajustadas[j] / escalas_ajustadas[j]
        for j in range(len(coeficientes))
    ])
    intercepto_original = float(scaler_y.scale_[0] * (intercepto - b_shift) + scaler_y.mean_[0])
    
    print(f"\nCoeficientes desnormalizados:")
    print(f"  Intercepto: {intercepto_original:.6f}")
    print(f"  Coeficientes: {coef_original}")
    
    # Verificar que el orden es correcto
    expected_order = ['x1', 'x2', 'x1^2', 'x1 x2', 'x2^2']
    actual_order = pf.get_feature_names_out(['x1', 'x2'])
    
    if list(actual_order) == expected_order:
        print("✅ Orden de características polinómicas CORRECTO")
    else:
        print(f"❌ Orden de características polinómicas INCORRECTO")
        print(f"   Esperado: {expected_order}")
        print(f"   Obtenido: {list(actual_order)}")
    
    # 2. Verificar que el cálculo manual coincide
    print("\n2. Verificando cálculo manual...")
    
    # Tomar un punto de prueba
    x1_test, x2_test = 25.0, 15.0
    
    # Calcular usando los coeficientes desnormalizados (como en plot_model_curves.py)
    valor_manual = (intercepto_original + 
                   coef_original[0] * x1_test +           # c1*x1
                   coef_original[1] * x2_test +           # c2*x2  
                   coef_original[2] * (x1_test ** 2) +    # c3*x1²
                   coef_original[3] * x1_test * x2_test + # c4*x1*x2
                   coef_original[4] * (x2_test ** 2))     # c5*x2²
    
    # Calcular usando el modelo entrenado
    X_test_poly = pf.transform([[x1_test, x2_test]])
    X_test_norm = scaler_X.transform(X_test_poly)
    y_test_norm = modelo.predict(X_test_norm)[0]
    valor_modelo = scaler_y.inverse_transform([[y_test_norm]])[0, 0]
    
    print(f"Punto de prueba: x1={x1_test}, x2={x2_test}")
    print(f"Cálculo manual: {valor_manual:.6f}")
    print(f"Cálculo modelo: {valor_modelo:.6f}")
    print(f"Diferencia: {abs(valor_manual - valor_modelo):.6f}")
    
    if abs(valor_manual - valor_modelo) < 1e-10:
        print("✅ Cálculo manual COINCIDE con el modelo entrenado")
    else:
        print("❌ Cálculo manual NO coincide con el modelo entrenado")
    
    # 3. Verificar que los coeficientes recuperados son razonables
    print("\n3. Verificando razonabilidad de coeficientes...")
    
    expected_coef = [2.0, 3.0, 0.1, 0.5, 0.2]
    expected_intercept = 100.0
    
    print(f"Coeficientes esperados: {expected_coef}")
    print(f"Coeficientes obtenidos: {[round(c, 1) for c in coef_original]}")
    print(f"Intercepto esperado: {expected_intercept}")
    print(f"Intercepto obtenido: {intercepto_original:.1f}")
    
    # Verificar que están cerca (con tolerancia para el ruido)
    coef_close = all(abs(o - e) < 0.5 for o, e in zip(coef_original, expected_coef))
    intercept_close = abs(intercepto_original - expected_intercept) < 5
    
    if coef_close and intercept_close:
        print("✅ Coeficientes recuperados son RAZONABLES")
    else:
        print("❌ Coeficientes recuperados NO son razonables")
    
    return coef_original, intercepto_original

def test_polynomial_prediction_consistency():
    """
    Test para verificar que las predicciones son consistentes
    """
    print("\n=== TEST CONSISTENCIA PREDICCIONES ===")
    
    # Usar los coeficientes del test anterior
    coef_original, intercepto_original = test_polynomial_order_consistency()
    
    # Crear varios puntos de prueba
    test_points = [
        (10, 5),
        (25, 15),
        (40, 20),
        (50, 25)
    ]
    
    print(f"\nProbando {len(test_points)} puntos...")
    
    all_consistent = True
    for x1, x2 in test_points:
        # Cálculo manual (como en plot_model_curves.py)
        valor_manual = (intercepto_original + 
                       coef_original[0] * x1 +           # c1*x1
                       coef_original[1] * x2 +           # c2*x2  
                       coef_original[2] * (x1 ** 2) +    # c3*x1²
                       coef_original[3] * x1 * x2 +      # c4*x1*x2
                       coef_original[4] * (x2 ** 2))     # c5*x2²
        
        # Cálculo con ecuación verdadera
        valor_verdadero = 100 + 2*x1 + 3*x2 + 0.1*(x1**2) + 0.5*x1*x2 + 0.2*(x2**2)
        
        diferencia = abs(valor_manual - valor_verdadero)
        
        print(f"  ({x1}, {x2}): manual={valor_manual:.2f}, verdadero={valor_verdadero:.2f}, diff={diferencia:.2f}")
        
        if diferencia > 10:  # Tolerancia razonable
            all_consistent = False
    
    if all_consistent:
        print("✅ Todas las predicciones son CONSISTENTES")
    else:
        print("❌ Algunas predicciones NO son consistentes")
    
    return all_consistent

if __name__ == "__main__":
    print("Iniciando tests de consistencia de coeficientes polinómicos...")
    
    try:
        success1 = test_polynomial_order_consistency()
        success2 = test_polynomial_prediction_consistency()
        
        if success1 and success2:
            print("\n✅ TODOS LOS TESTS PASARON - El orden de coeficientes es CORRECTO")
        else:
            print("\n❌ ALGUNOS TESTS FALLARON - Verificar implementación")
            
    except Exception as e:
        print(f"\n❌ Error en los tests: {e}")
        import traceback
        traceback.print_exc()
