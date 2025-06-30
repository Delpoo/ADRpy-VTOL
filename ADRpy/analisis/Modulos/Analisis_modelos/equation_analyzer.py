"""
equation_analyzer.py

Módulo para analizar y validar información de ecuaciones en los datos de modelos.
Permite verificar la consistencia y disponibilidad de campos para visualización.
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable

def analyze_equation_data(modelos_por_celda: Dict, detalles_por_celda: Dict):
    """
    Analiza la información de ecuaciones disponible en los datos
    
    Args:
        modelos_por_celda: Diccionario con modelos por celda
        detalles_por_celda: Diccionario con detalles de imputación
    
    Returns:
        Dict con análisis de la información disponible
    """
    analysis = {
        'total_celdas': len(modelos_por_celda),
        'modelos_1_predictor': 0,
        'modelos_2_predictores': 0,
        'modelos_mas_predictores': 0,
        'tipos_ecuacion': set(),
        'campos_ecuacion': set(),
        'predictores_disponibles': set(),
        'ejemplos_por_tipo': {},
        'problemas_detectados': []
    }
    
    for celda_key, modelos in modelos_por_celda.items():
        if not isinstance(modelos, list):
            continue
            
        for modelo in modelos:
            if not isinstance(modelo, dict):
                continue
                
            n_pred = modelo.get('n_predictores', 0)
            tipo = modelo.get('tipo', 'unknown')
            predictores = modelo.get('predictores', [])
            
            # Contar por número de predictores
            if n_pred == 1:
                analysis['modelos_1_predictor'] += 1
            elif n_pred == 2:
                analysis['modelos_2_predictores'] += 1
            elif n_pred > 2:
                analysis['modelos_mas_predictores'] += 1
            
            # Recopilar tipos
            analysis['tipos_ecuacion'].add(tipo)
            
            # Recopilar predictores
            for pred in predictores:
                analysis['predictores_disponibles'].add(pred)
            
            # Recopilar campos disponibles
            for field in modelo.keys():
                if 'ecuacion' in field.lower() or 'coeficiente' in field.lower():
                    analysis['campos_ecuacion'].add(field)
            
            # Guardar ejemplos
            if tipo not in analysis['ejemplos_por_tipo']:
                analysis['ejemplos_por_tipo'][tipo] = {
                    'celda': celda_key,
                    'modelo': modelo,
                    'n_predictores': n_pred
                }
            
            # Detectar problemas
            if n_pred > 0:
                if not modelo.get('ecuacion_string'):
                    analysis['problemas_detectados'].append(
                        f"Modelo {celda_key} tipo {tipo}: Sin ecuacion_string"
                    )
                
                if len(predictores) != n_pred:
                    analysis['problemas_detectados'].append(
                        f"Modelo {celda_key}: n_predictores={n_pred} pero predictores={len(predictores)}"
                    )
    
    return analysis

def extract_equation_components(modelo: Dict) -> Dict[str, Any]:
    """
    Extrae los componentes de una ecuación de un modelo
    
    Args:
        modelo: Diccionario con información del modelo
    
    Returns:
        Dict con componentes de la ecuación
    """
    components = {
        'tipo': modelo.get('tipo', 'unknown'),
        'n_predictores': modelo.get('n_predictores', 0),
        'predictores': modelo.get('predictores', []),
        'coeficientes': modelo.get('coeficientes_originales', []),
        'intercepto': modelo.get('intercepto_original', 0),
        'ecuacion_string': modelo.get('ecuacion_string', ''),
        'ecuacion_normalizada': modelo.get('ecuacion_normalizada', ''),
        'r2': modelo.get('r2', 0),
        'mape': modelo.get('mape', float('inf')),
        'datos_entrenamiento': modelo.get('datos_entrenamiento', {}),
        'es_valido': False
    }
    
    # Validar componentes
    if (components['n_predictores'] > 0 and 
        len(components['predictores']) == components['n_predictores'] and
        components['ecuacion_string']):
        components['es_valido'] = True
    
    return components

def create_equation_function(components: Dict) -> Optional[Callable]:
    """
    Crea una función evaluable a partir de los componentes de la ecuación
    
    Args:
        components: Componentes de la ecuación extraídos
    
    Returns:
        Función que evalúa la ecuación o None si no es posible
    """
    if not components['es_valido']:
        return None
    
    tipo = components['tipo']
    coefs = components['coeficientes']
    intercept = components['intercepto']
    n_pred = components['n_predictores']
    
    try:
        if tipo.startswith('linear'):
            # Modelo lineal: y = a + b1*x1 + b2*x2 + ...
            def linear_func(*x_values):
                if len(x_values) != n_pred:
                    raise ValueError(f"Se esperan {n_pred} valores, se recibieron {len(x_values)}")
                result = intercept
                for i, (coef, x_val) in enumerate(zip(coefs, x_values)):
                    result += coef * x_val
                return result
            return linear_func
            
        elif tipo.startswith('exp'):
            # Modelo exponencial: y = a * exp(b1*x1 + b2*x2 + ...)
            def exp_func(*x_values):
                if len(x_values) != n_pred:
                    raise ValueError(f"Se esperan {n_pred} valores, se recibieron {len(x_values)}")
                exponent = 0
                for coef, x_val in zip(coefs, x_values):
                    exponent += coef * x_val
                return intercept * np.exp(exponent)
            return exp_func
            
        elif tipo.startswith('power'):
            # Modelo potencial: y = a * x1^b1 * x2^b2 * ...
            def power_func(*x_values):
                if len(x_values) != n_pred:
                    raise ValueError(f"Se esperan {n_pred} valores, se recibieron {len(x_values)}")
                result = intercept
                for coef, x_val in zip(coefs, x_values):
                    if x_val <= 0:
                        return float('nan')  # Evitar problemas con potencias
                    result *= (x_val ** coef)
                return result
            return power_func
            
        elif tipo.startswith('log'):
            # Modelo logarítmico: y = a + b1*log(x1) + b2*log(x2) + ...
            def log_func(*x_values):
                if len(x_values) != n_pred:
                    raise ValueError(f"Se esperan {n_pred} valores, se recibieron {len(x_values)}")
                result = intercept
                for coef, x_val in zip(coefs, x_values):
                    if x_val <= 0:
                        return float('nan')  # Evitar log de números negativos
                    result += coef * np.log(x_val)
                return result
            return log_func
            
        elif tipo in ['poly-1', 'poly-2'] or tipo.startswith('poly'):
            # Modelo polinómico: usar ecuación_string directamente
            ecuacion_string = components.get('ecuacion_string', '')
            if not ecuacion_string:
                return None
                
            def poly_func(*x_values):
                if len(x_values) != n_pred:
                    raise ValueError(f"Se esperan {n_pred} valores, se recibieron {len(x_values)}")
                
                # Crear diccionario para evaluar la ecuación
                # Para modelos con más variables que predictores, completar con 0
                variables = {}
                for i in range(max(5, len(x_values))):  # Asegurar hasta x4
                    if i < len(x_values):
                        variables[f'x{i}'] = x_values[i]
                    else:
                        variables[f'x{i}'] = 0.0  # Variables extra en 0
                
                # Extraer la parte derecha de la ecuación (después del '=')
                if '=' in ecuacion_string:
                    expression = ecuacion_string.split('=')[1].strip()
                else:
                    expression = ecuacion_string
                
                try:
                    # Evaluar la expresión de manera segura
                    # Permitir operaciones matemáticas básicas
                    safe_dict = {
                        "__builtins__": {},
                        "pow": pow,
                        "abs": abs,
                        "min": min,
                        "max": max
                    }
                    safe_dict.update(variables)
                    result = eval(expression, safe_dict)
                    return float(result)
                except Exception as e:
                    print(f"Error evaluando ecuación polinómica: {e}")
                    return float('nan')
                    
            return poly_func
            
    except Exception as e:
        print(f"Error creando función para tipo {tipo}: {e}")
        return None
    
    return None

def get_predictor_ranges(modelo: Dict) -> Dict[str, Dict[str, float]]:
    """
    Extrae los rangos de los predictores de los datos de entrenamiento
    
    Args:
        modelo: Diccionario con información del modelo
    
    Returns:
        Dict con rangos de cada predictor
    """
    ranges = {}
    datos_entrenamiento = modelo.get('datos_entrenamiento', {})
    predictores = modelo.get('predictores', [])
    
    if 'X_original' in datos_entrenamiento:
        X_data = datos_entrenamiento['X_original']
        
        for i, predictor in enumerate(predictores):
            if i < len(X_data[0]) if X_data else 0:
                # Extraer valores de la columna i
                valores = [fila[i] for fila in X_data if len(fila) > i]
                
                if valores:
                    ranges[predictor] = {
                        'min': min(valores),
                        'max': max(valores),
                        'mean': sum(valores) / len(valores),
                        'count': len(valores)
                    }
    
    return ranges

def validate_equation_for_plotting(components: Dict) -> Tuple[bool, str]:
    """
    Valida si una ecuación puede ser graficada
    
    Args:
        components: Componentes de la ecuación
    
    Returns:
        Tuple (es_valido, mensaje)
    """
    if not components['es_valido']:
        return False, "Componentes de ecuación inválidos"
    
    n_pred = components['n_predictores']
    
    if n_pred == 0:
        return False, "Sin predictores"
    elif n_pred == 1:
        return True, "Válido para gráfica 2D"
    elif n_pred == 2:
        return True, "Válido para gráfica 3D"
    else:
        return False, f"Demasiados predictores ({n_pred}) para visualización"

def print_equation_analysis(analysis: Dict):
    """Imprime un resumen del análisis de ecuaciones"""
    print("=" * 60)
    print("ANÁLISIS DE ECUACIONES EN LOS DATOS")
    print("=" * 60)
    
    print(f"📊 Total de celdas: {analysis['total_celdas']}")
    print(f"📈 Modelos con 1 predictor: {analysis['modelos_1_predictor']}")
    print(f"📈 Modelos con 2 predictores: {analysis['modelos_2_predictores']}")
    print(f"📈 Modelos con más predictores: {analysis['modelos_mas_predictores']}")
    
    print(f"\n🔧 Tipos de ecuación encontrados:")
    for tipo in sorted(analysis['tipos_ecuacion']):
        print(f"   - {tipo}")
    
    print(f"\n📝 Campos de ecuación disponibles:")
    for campo in sorted(analysis['campos_ecuacion']):
        print(f"   - {campo}")
    
    print(f"\n🎯 Predictores disponibles ({len(analysis['predictores_disponibles'])}):")
    for pred in sorted(list(analysis['predictores_disponibles'])[:10]):  # Mostrar solo los primeros 10
        print(f"   - {pred}")
    if len(analysis['predictores_disponibles']) > 10:
        print(f"   ... y {len(analysis['predictores_disponibles']) - 10} más")
    
    if analysis['problemas_detectados']:
        print(f"\n⚠️ Problemas detectados ({len(analysis['problemas_detectados'])}):")
        for problema in analysis['problemas_detectados'][:5]:  # Mostrar solo los primeros 5
            print(f"   - {problema}")
        if len(analysis['problemas_detectados']) > 5:
            print(f"   ... y {len(analysis['problemas_detectados']) - 5} más")
    
    print(f"\n📋 Ejemplos por tipo:")
    for tipo, ejemplo in analysis['ejemplos_por_tipo'].items():
        modelo = ejemplo['modelo']
        print(f"   {tipo} ({ejemplo['n_predictores']} pred.):")
        print(f"     Celda: {ejemplo['celda']}")
        ecuacion = modelo.get('ecuacion_string', 'N/A')
        if len(ecuacion) > 80:
            ecuacion = ecuacion[:80] + "..."
        print(f"     Ecuación: {ecuacion}")
        print(f"     R²: {modelo.get('r2', 0):.3f}")

def analyze_filtering_problem(modelos_por_celda: Dict) -> Dict:
    """
    Analiza el problema específico del filtrado entre gráficas 2D y 3D
    
    Args:
        modelos_por_celda: Diccionario con modelos por celda
    
    Returns:
        Dict con análisis del problema de filtrado
    """
    filtering_analysis = {
        'celdas_con_1pred': 0,
        'celdas_con_2pred': 0,
        'celdas_con_ambos': 0,
        'celdas_solo_1pred': 0,
        'celdas_solo_2pred': 0,
        'ejemplos_mixtos': [],
        'distribucion_por_celda': {}
    }
    
    for celda_key, modelos in modelos_por_celda.items():
        if not isinstance(modelos, list):
            continue
        
        modelos_1pred = []
        modelos_2pred = []
        
        for modelo in modelos:
            if isinstance(modelo, dict):
                n_pred = modelo.get('n_predictores', 0)
                if n_pred == 1:
                    modelos_1pred.append(modelo)
                elif n_pred == 2:
                    modelos_2pred.append(modelo)
        
        tiene_1pred = len(modelos_1pred) > 0
        tiene_2pred = len(modelos_2pred) > 0
        
        filtering_analysis['distribucion_por_celda'][celda_key] = {
            'modelos_1pred': len(modelos_1pred),
            'modelos_2pred': len(modelos_2pred),
            'total_modelos': len(modelos)
        }
        
        if tiene_1pred:
            filtering_analysis['celdas_con_1pred'] += 1
        if tiene_2pred:
            filtering_analysis['celdas_con_2pred'] += 1
        if tiene_1pred and tiene_2pred:
            filtering_analysis['celdas_con_ambos'] += 1
            filtering_analysis['ejemplos_mixtos'].append({
                'celda': celda_key,
                'modelos_1pred': len(modelos_1pred),
                'modelos_2pred': len(modelos_2pred)
            })
        elif tiene_1pred and not tiene_2pred:
            filtering_analysis['celdas_solo_1pred'] += 1
        elif tiene_2pred and not tiene_1pred:
            filtering_analysis['celdas_solo_2pred'] += 1
    
    return filtering_analysis

def print_filtering_analysis(filtering_analysis: Dict):
    """Imprime análisis del problema de filtrado"""
    print("\n" + "=" * 60)
    print("ANÁLISIS DEL PROBLEMA DE FILTRADO 2D vs 3D")
    print("=" * 60)
    
    print(f"📊 Distribución de celdas:")
    print(f"   Celdas con modelos de 1 predictor: {filtering_analysis['celdas_con_1pred']}")
    print(f"   Celdas con modelos de 2 predictores: {filtering_analysis['celdas_con_2pred']}")
    print(f"   Celdas con ambos tipos: {filtering_analysis['celdas_con_ambos']}")
    print(f"   Celdas solo con 1 predictor: {filtering_analysis['celdas_solo_1pred']}")
    print(f"   Celdas solo con 2 predictores: {filtering_analysis['celdas_solo_2pred']}")
    
    if filtering_analysis['ejemplos_mixtos']:
        print(f"\n🔄 Celdas con ambos tipos de modelos:")
        for ejemplo in filtering_analysis['ejemplos_mixtos']:
            print(f"   {ejemplo['celda']}: {ejemplo['modelos_1pred']} modelo(s) 2D, {ejemplo['modelos_2pred']} modelo(s) 3D")
    
    print(f"\n💡 IMPLICACIONES PARA EL FILTRADO:")
    print(f"   • Gráficas 2D deben filtrar: n_predictores == 1")
    print(f"   • Gráficas 3D deben filtrar: n_predictores == 2")
    print(f"   • {filtering_analysis['celdas_con_ambos']} celdas requieren filtrado diferente por vista")
    print(f"   • {filtering_analysis['celdas_solo_1pred']} celdas solo tienen modelos 2D")
    print(f"   • {filtering_analysis['celdas_solo_2pred']} celdas solo tienen modelos 3D")
