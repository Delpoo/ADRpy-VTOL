#!/usr/bin/env python3
"""
Script de debug para verificar el cálculo correcto de ecuaciones polinómicas
en los puntos de imputación teóricos.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score
import logging

# Configurar logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_polynomial_coefficients():
    """
    Test para verificar el orden correcto de los coeficientes polinómicos
    y la evaluación de ecuaciones.
    """
    print("=== TEST DE COEFICIENTES POLINÓMICOS ===")
    
    # Datos de prueba sintéticos para 2 predictores
    np.random.seed(42)
    n_samples = 20
    
    # Variables independientes
    x1 = np.random.uniform(10, 100, n_samples)
    x2 = np.random.uniform(5, 50, n_samples)
    
    # Variable dependiente (función conocida para verificar)
    # y = 10 + 2*x1 + 3*x2 + 0.1*x1² + 0.2*x2² + 0.05*x1*x2 + ruido
    y = 10 + 2*x1 + 3*x2 + 0.1*x1**2 + 0.2*x2**2 + 0.05*x1*x2 + np.random.normal(0, 1, n_samples)
    
    print(f"Función original: y = 10 + 2*x1 + 3*x2 + 0.1*x1² + 0.2*x2² + 0.05*x1*x2")
    print(f"Coeficientes esperados: intercepto=10, c1=2, c2=3, c3=0.1, c4=0.05, c5=0.2")
    
    # Preparar datos
    X = np.column_stack([x1, x2])
    
    # === REPRODUCIR EL PIPELINE DE IMPUTACION_CORRELACION ===
    
    # 1. Crear features polinómicas
    pf = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = pf.fit_transform(X)
    
    print(f"\nFeatures polinómicas:")
    print(f"X_poly.shape: {X_poly.shape}")
    print(f"Feature names: {pf.get_feature_names_out(['x1', 'x2'])}")
    print(f"Powers: {pf.powers_}")
    
    # 2. Normalizar X polinómicas
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_poly)
    
    # 3. Normalizar y
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # 4. Entrenar modelo
    modelo = LinearRegression()
    modelo.fit(X_scaled, y_scaled)
    
    coef_scaled = modelo.coef_
    intercept_scaled = modelo.intercept_
    
    print(f"\nCoeficientes escalados del modelo:")
    print(f"Intercepto escalado: {intercept_scaled:.6f}")
    print(f"Coeficientes escalados: {coef_scaled}")
    
    # 5. Desnormalizar coeficientes (REPRODUCIR LÓGICA EXACTA DE imputacion_correlacion.py)
    if scaler_X.scale_ is not None and scaler_X.mean_ is not None:
        escalas_ajustadas = scaler_X.scale_
        medias_ajustadas = scaler_X.mean_
        coef_original = (coef_scaled * scaler_y.scale_[0] / escalas_ajustadas).tolist()
        b_shift = sum([
            coef_scaled[j] * medias_ajustadas[j] / escalas_ajustadas[j]
            for j in range(len(coef_scaled))
        ])
        intercepto_original = float(scaler_y.scale_[0] * (intercept_scaled - b_shift) + scaler_y.mean_[0])
    else:
        coef_original = coef_scaled.tolist()
        intercepto_original = float(intercept_scaled)
    
    print(f"\nCoeficientes desnormalizados:")
    print(f"Intercepto original: {intercepto_original:.6f}")
    print(f"Coeficientes originales: {coef_original}")
    
    # === VERIFICAR CON VALORES DE PRUEBA ===
    
    # Tomar primer punto de los datos como ejemplo
    test_x1, test_x2 = x1[0], x2[0]
    test_y_real = y[0]
    
    print(f"\n=== VERIFICACIÓN CON PUNTO DE PRUEBA ===")
    print(f"Punto de prueba: x1={test_x1:.3f}, x2={test_x2:.3f}")
    print(f"Valor y real: {test_y_real:.3f}")
    
    # Método 1: Usar el modelo completo (pipeline completo)
    test_X = np.array([[test_x1, test_x2]])
    test_X_poly = pf.transform(test_X)
    test_X_scaled = scaler_X.transform(test_X_poly)
    test_y_scaled_pred = modelo.predict(test_X_scaled)[0]
    test_y_pipeline = scaler_y.inverse_transform([[test_y_scaled_pred]])[0, 0]
    
    print(f"Método 1 (pipeline completo): {test_y_pipeline:.3f}")
    
    # Método 2: Usar coeficientes desnormalizados directamente (como en extract_theoretical_imputation_points)
    # Según PolynomialFeatures con degree=2, include_bias=False:
    # Para [x1, x2]: features = [x1, x2, x1², x1*x2, x2²]
    # Por lo tanto: coef_original = [c_x1, c_x2, c_x1_sq, c_x1_x2, c_x2_sq]
    
    test_y_manual = (intercepto_original + 
                    coef_original[0] * test_x1 +           # x1
                    coef_original[1] * test_x2 +           # x2
                    coef_original[2] * (test_x1 ** 2) +    # x1²
                    coef_original[3] * test_x1 * test_x2 + # x1*x2
                    coef_original[4] * (test_x2 ** 2))     # x2²
    
    print(f"Método 2 (coeficientes manuales): {test_y_manual:.3f}")
    
    # Calcular diferencias
    diff_pipeline_real = abs(test_y_pipeline - test_y_real)
    diff_manual_real = abs(test_y_manual - test_y_real)
    diff_pipeline_manual = abs(test_y_pipeline - test_y_manual)
    
    print(f"\nDiferencias:")
    print(f"Pipeline vs Real: {diff_pipeline_real:.3f}")
    print(f"Manual vs Real: {diff_manual_real:.3f}")
    print(f"Pipeline vs Manual: {diff_pipeline_manual:.3f}")
    
    # Si la diferencia entre pipeline y manual es muy grande, hay un problema
    if diff_pipeline_manual > 0.1:
        print(f"\n❌ ERROR: Gran diferencia entre métodos ({diff_pipeline_manual:.3f})")
        print("Esto indica un problema en la interpretación de coeficientes")
        
        print(f"\nDETALLE DEL CÁLCULO MANUAL:")
        print(f"  intercepto_original = {intercepto_original:.6f}")
        print(f"  + coef[0] * x1 = {coef_original[0]:.6f} * {test_x1:.3f} = {coef_original[0] * test_x1:.6f}")
        print(f"  + coef[1] * x2 = {coef_original[1]:.6f} * {test_x2:.3f} = {coef_original[1] * test_x2:.6f}")
        print(f"  + coef[2] * x1² = {coef_original[2]:.6f} * {test_x1**2:.3f} = {coef_original[2] * test_x1**2:.6f}")
        print(f"  + coef[3] * x1*x2 = {coef_original[3]:.6f} * {test_x1*test_x2:.3f} = {coef_original[3] * test_x1 * test_x2:.6f}")
        print(f"  + coef[4] * x2² = {coef_original[4]:.6f} * {test_x2**2:.3f} = {coef_original[4] * test_x2**2:.6f}")
        print(f"  = {test_y_manual:.6f}")
        
    else:
        print(f"\n✅ OK: Los métodos son consistentes (diferencia: {diff_pipeline_manual:.6f})")
    
    return {
        'test_x1': test_x1,
        'test_x2': test_x2,
        'test_y_real': test_y_real,
        'test_y_pipeline': test_y_pipeline,
        'test_y_manual': test_y_manual,
        'coef_original': coef_original,
        'intercepto_original': intercepto_original,
        'powers': pf.powers_,
        'feature_names': pf.get_feature_names_out(['x1', 'x2'])
    }

if __name__ == "__main__":
    resultado = test_polynomial_coefficients()
    print(f"\n=== RESUMEN ===")
    print(f"Orden de features: {resultado['feature_names']}")
    print(f"Powers: {resultado['powers']}")
    print(f"Coeficientes: {resultado['coef_original']}")
    print(f"Intercepto: {resultado['intercepto_original']}")
