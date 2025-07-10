import numpy as np
from typing import Tuple, List, Dict, Union

def generate_symbiotic_surface_3d(
    model_data: Union[str, Dict],
    rango_x: Tuple[float, float],
    rango_y: Tuple[float, float],
    pred_names: List[str],
    resolution: int = 50
):
    """
    Genera una superficie 3D usando la transformada simbiótica:
    - X*, Y* normalizados (0-1) para visualización
    - Desnormaliza internamente X*, Y* a escala original
    - Evalúa el modelo en escala original
    - Devuelve X*, Y* (normalizados) y Z (escala original)
    
    Args:
        model_data: String de ecuación (modelos lineales) o dict con coeficientes (modelos polinomiales)
        rango_x: (min, max) para variable x0
        rango_y: (min, max) para variable x1  
        pred_names: Nombres de las variables predictoras
        resolution: Número de puntos por eje
        
    Returns:
        X_star, Y_star, Z: Mallas normalizadas (X,Y) y valores originales (Z)
    """
    x_star = np.linspace(0, 1, resolution)
    y_star = np.linspace(0, 1, resolution)
    X_star, Y_star = np.meshgrid(x_star, y_star)

    x0_min, x0_max = rango_x
    x1_min, x1_max = rango_y
    
    # Desnormalizar a valores originales para evaluar modelo
    x_orig = X_star * (x0_max - x0_min) + x0_min
    y_orig = Y_star * (x1_max - x1_min) + x1_min

    # Determinar tipo de modelo y evaluar apropiadamente
    if isinstance(model_data, str):
        # Modelo lineal: usar string de ecuación
        Z = _evaluate_equation_model(model_data, x_orig, y_orig, pred_names)
    elif isinstance(model_data, dict) and 'coeficientes_originales' in model_data:
        # Modelo polinomial: usar coeficientes directamente
        Z = _evaluate_polynomial_model(model_data, x_orig, y_orig)
    else:
        print(f"⚠️ Tipo de modelo no reconocido: {type(model_data)}")
        if isinstance(model_data, dict):
            print(f"   Keys disponibles: {list(model_data.keys())[:5]}...")
        Z = np.full_like(X_star, 0.0)
    
    return X_star, Y_star, Z


def _evaluate_equation_model(equation_string: str, x_orig: np.ndarray, y_orig: np.ndarray, pred_names: List[str]) -> np.ndarray:
    """Evalúa modelos lineales usando string de ecuación"""
    # Preparar variables para diferentes formatos de ecuación
    local_vars = {
        # Variables clásicas
        'x': x_orig, 'y': y_orig,
        # Variables indexadas (x0, x1) - formato común en modelos
        'x0': x_orig, 'x1': y_orig,
        # Nombres de predictores específicos
        pred_names[0]: x_orig, pred_names[1]: y_orig,
        # Funciones matemáticas
        'np': np, 'pow': np.power, 'exp': np.exp, 'log': np.log
    }
    
    try:
        # Limpiar y preparar ecuación
        eq = equation_string.replace('^', '**')
        
        # Si la ecuación tiene formato "y = ...", extraer solo la parte derecha
        if ' = ' in eq:
            eq = eq.split(' = ', 1)[1]
        
        # Evaluar ecuación con variables desnormalizadas
        Z = eval(eq, {"__builtins__": {}}, local_vars)
        
    except Exception as e:
        print(f"⚠️ Error evaluando ecuación '{equation_string}': {e}")
        # En caso de error, generar superficie plana
        Z = np.full_like(x_orig, 0.0)
    
    return Z


def _evaluate_polynomial_model(model_data: Dict, x_orig: np.ndarray, y_orig: np.ndarray) -> np.ndarray:
    """
    Evalúa modelos polinomiales usando coeficientes directamente.
    Orden de términos PolynomialFeatures: [x0, x1, x0², x0*x1, x1²]
    """
    try:
        intercept = model_data['intercepto_original']
        coefs = model_data['coeficientes_originales']
        
        # Verificar que tengamos 5 coeficientes para modelo poly-2
        if len(coefs) != 5:
            print(f"⚠️ Se esperaban 5 coeficientes para modelo poly-2, se encontraron {len(coefs)}")
            return np.full_like(x_orig, intercept)
        
        # Calcular términos polinomiales según orden de PolynomialFeatures
        # [x0, x1, x0², x0*x1, x1²]
        Z = (intercept + 
             coefs[0] * x_orig +                    # x0
             coefs[1] * y_orig +                    # x1 
             coefs[2] * (x_orig ** 2) +             # x0²
             coefs[3] * x_orig * y_orig +           # x0*x1
             coefs[4] * (y_orig ** 2))              # x1²
        
        return Z
        
    except Exception as e:
        print(f"⚠️ Error evaluando modelo polinomial: {e}")
        return np.full_like(x_orig, 0.0)
