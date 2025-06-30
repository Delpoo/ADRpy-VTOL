"""
Módulo avanzado para validar la racionalidad de normalizaciones, 
compatibilidad de ejes y visibilidad de modelos polinómicos.
VERSIÓN SIMPLIFICADA Y ROBUSTA
"""

import json
import numpy as np
import pandas as pd
import re
from typing import Dict, List, Any, Tuple, Optional
import warnings


def analyze_normalization_simple(modelos_por_celda: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """Analiza problemas de normalización de manera simple."""
    print("📏 Analizando normalizaciones...")
    
    problems = {}
    stats_summary = {}
    
    for celda_key, modelos_list in modelos_por_celda.items():
        for i, modelo in enumerate(modelos_list):
            if not isinstance(modelo, dict):
                continue
                
            predictores = modelo.get('predictores', [])
            datos_entrenamiento = modelo.get('datos_entrenamiento', {})
            parametro = modelo.get('Parámetro', f'modelo_{i}')
            
            for predictor in predictores:
                if predictor in datos_entrenamiento:
                    try:
                        data = np.array(datos_entrenamiento[predictor])
                        if len(data) > 0:
                            data_min, data_max = np.min(data), np.max(data)
                            data_range = data_max - data_min
                            data_mean, data_std = np.mean(data), np.std(data)
                            
                            key = f"{celda_key}_{predictor}_{parametro}"
                            stats_summary[key] = {
                                'range': float(data_range),
                                'mean': float(data_mean),
                                'std': float(data_std),
                                'min': float(data_min),
                                'max': float(data_max)
                            }
                            
                            # Detectar problemas
                            issues = []
                            if data_range > 1000:
                                issues.append('rango_muy_amplio')
                            if abs(data_max) < 0.001 and abs(data_min) < 0.001:
                                issues.append('valores_muy_pequeños')
                            if data_std > 0 and abs(data_mean) > 10 * data_std:
                                issues.append('asimetria_extrema')
                                
                            if issues:
                                problems[key] = {
                                    'celda': celda_key,
                                    'predictor': predictor,
                                    'parametro': parametro,
                                    'issues': issues,
                                    'stats': stats_summary[key]
                                }
                    except Exception as e:
                        print(f"   ⚠️ Error procesando {predictor} en {celda_key}: {e}")
    
    print(f"   ✓ {len(stats_summary)} casos analizados, {len(problems)} con problemas")
    return {
        'problems': problems,
        'stats': stats_summary
    }


def analyze_polynomial_visibility_simple(modelos_por_celda: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """Analiza visibilidad de modelos polinómicos."""
    print("🔢 Analizando visibilidad de modelos polinómicos...")
    
    polynomial_models = {}
    problems = {}
    
    for celda_key, modelos_list in modelos_por_celda.items():
        for i, modelo in enumerate(modelos_list):
            if not isinstance(modelo, dict):
                continue
                
            tipo_modelo = modelo.get('tipo', '')
            if 'polynomial' in tipo_modelo.lower() or 'polinomial' in tipo_modelo.lower():
                
                key = f"{celda_key}_modelo_{i}"
                ecuacion = modelo.get('ecuacion_string', '')
                coeficientes = modelo.get('coeficientes_originales', [])
                
                polynomial_models[key] = {
                    'celda': celda_key,
                    'tipo': tipo_modelo,
                    'ecuacion': ecuacion,
                    'n_coeficientes': len(coeficientes)
                }
                
                # Analizar problemas de visibilidad
                issues = []
                
                if coeficientes:
                    try:
                        coef_array = np.array(coeficientes)
                        
                        # Coeficientes muy pequeños
                        if np.any(np.abs(coef_array) < 1e-6):
                            issues.append('coeficientes_muy_pequeños')
                        
                        # Órdenes de magnitud extremos
                        if len(coef_array) > 1:
                            max_coef = np.max(np.abs(coef_array))
                            non_zero = coef_array[coef_array != 0]
                            if len(non_zero) > 0:
                                min_coef = np.min(np.abs(non_zero))
                                if min_coef > 0 and max_coef / min_coef > 1e6:
                                    issues.append('ordenes_magnitud_extremos')
                    except Exception:
                        issues.append('error_procesando_coeficientes')
                
                # Analizar grado alto en ecuación
                if ecuacion:
                    high_degree_terms = re.findall(r'\*\*\s*(\d+)', ecuacion)
                    if high_degree_terms:
                        max_degree = max(int(d) for d in high_degree_terms)
                        if max_degree > 4:
                            issues.append('grado_muy_alto')
                
                if issues:
                    problems[key] = {
                        'celda': celda_key,
                        'issues': issues,
                        'modelo_info': polynomial_models[key]
                    }
    
    print(f"   ✓ {len(polynomial_models)} modelos polinómicos encontrados, {len(problems)} con problemas")
    return {
        'models': polynomial_models,
        'problems': problems
    }


def analyze_dimensional_consistency_simple(modelos_por_celda: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """Analiza consistencia dimensional básica."""
    print("📐 Analizando consistencia dimensional...")
    
    unit_keywords = {
        'masa': ['peso', 'mass', 'weight', 'kg'],
        'longitud': ['envergadura', 'length', 'span', 'chord', 'm'],
        'velocidad': ['velocity', 'speed', 'velocidad', 'mach'],
        'potencia': ['potencia', 'power', 'hp', 'kw'],
        'area': ['area', 'superficie', 'm2'],
        'adimensional': ['ratio', 'coefficient', 'factor', 'coeficiente']
    }
    
    dimensional_analysis = {}
    problems = {}
    
    for celda_key, modelos_list in modelos_por_celda.items():
        for i, modelo in enumerate(modelos_list):
            if not isinstance(modelo, dict):
                continue
                
            predictores = modelo.get('predictores', [])
            parametro = modelo.get('Parámetro', f'modelo_{i}')
            datos_entrenamiento = modelo.get('datos_entrenamiento', {})
            
            key = f"{celda_key}_{parametro}_{i}"
            
            analysis = {
                'celda': celda_key,
                'parametro': parametro,
                'predictores': predictores,
                'predictor_dims': {},
                'param_dim': 'desconocida'
            }
            
            # Clasificar dimensiones de predictores
            for pred in predictores:
                pred_lower = pred.lower()
                pred_dim = 'desconocida'
                for dim_type, keywords in unit_keywords.items():
                    if any(kw in pred_lower for kw in keywords):
                        pred_dim = dim_type
                        break
                analysis['predictor_dims'][pred] = pred_dim
            
            # Clasificar dimensión del parámetro
            param_lower = parametro.lower()
            for dim_type, keywords in unit_keywords.items():
                if any(kw in param_lower for kw in keywords):
                    analysis['param_dim'] = dim_type
                    break
            
            # Detectar problemas dimensionales
            issues = []
            
            # Verificar normalización extrema en variables con dimensión física
            for pred in predictores:
                if pred in datos_entrenamiento:
                    try:
                        data = np.array(datos_entrenamiento[pred])
                        if len(data) > 0:
                            data_range = np.max(data) - np.min(data)
                            pred_dim = analysis['predictor_dims'][pred]
                            
                            # Si el rango es muy pequeño pero la variable tiene dimensión física
                            if abs(data_range) < 10 and pred_dim not in ['adimensional', 'desconocida']:
                                issues.append({
                                    'tipo': 'posible_sobrenormalizacion',
                                    'predictor': pred,
                                    'rango': float(data_range),
                                    'dimension': pred_dim
                                })
                    except Exception:
                        pass
            
            dimensional_analysis[key] = analysis
            
            if issues:
                problems[key] = {
                    'celda': celda_key,
                    'issues': issues,
                    'analysis': analysis
                }
    
    print(f"   ✓ {len(dimensional_analysis)} casos analizados, {len(problems)} con problemas dimensionales")
    return {
        'analysis': dimensional_analysis,
        'problems': problems
    }


def generate_recommendations_simple(
    normalization_result: Dict,
    polynomial_result: Dict,
    dimensional_result: Dict
) -> List[Dict]:
    """Genera recomendaciones basadas en los análisis."""
    print("💡 Generando recomendaciones...")
    
    recommendations = []
    
    # Recomendaciones de normalización
    if normalization_result['problems']:
        recommendations.append({
            'categoria': 'normalización',
            'prioridad': 'alta',
            'n_problemas': len(normalization_result['problems']),
            'descripcion': 'Se detectaron problemas de normalización que pueden afectar la visualización',
            'acciones': [
                'Verificar escalado de predictores con rangos muy amplios (>1000)',
                'Revisar variables con valores extremadamente pequeños (<0.001)',
                'Considerar re-normalización para variables asimétricas',
                'Añadir advertencias de escala en la visualización'
            ]
        })
    
    # Recomendaciones polinómicas
    if polynomial_result['problems']:
        recommendations.append({
            'categoria': 'visibilidad_polinómica',
            'prioridad': 'media',
            'n_problemas': len(polynomial_result['problems']),
            'descripcion': 'Modelos polinómicos con posibles problemas de visualización',
            'acciones': [
                'Revisar coeficientes extremadamente pequeños o grandes',
                'Considerar reescalado para términos con órdenes de magnitud muy diferentes',
                'Implementar visualización adaptativa para polinómicos de alto grado',
                'Validar que los términos polinómicos sean numéricamente estables'
            ]
        })
    
    # Recomendaciones dimensionales
    if dimensional_result['problems']:
        recommendations.append({
            'categoria': 'consistencia_dimensional',
            'prioridad': 'media',
            'n_problemas': len(dimensional_result['problems']),
            'descripcion': 'Posibles inconsistencias en dimensionalidad y unidades',
            'acciones': [
                'Verificar unidades de variables con normalización extrema',
                'Añadir etiquetas de unidades en visualizaciones',
                'Considerar indicadores de estado de normalización',
                'Documentar transformaciones aplicadas a cada variable'
            ]
        })
    
    # Recomendación general
    recommendations.append({
        'categoria': 'mejoras_generales',
        'prioridad': 'baja',
        'descripcion': 'Mejoras para robustez general del sistema',
        'acciones': [
            'Implementar validación de rangos en tiempo real',
            'Añadir indicadores de calidad/confianza por modelo',
            'Proporcionar información contextual sobre transformaciones',
            'Considerar modo debug con estadísticas detalladas'
        ]
    })
    
    print(f"   ✓ {len(recommendations)} categorías de recomendaciones generadas")
    return recommendations


def main_simple_analysis(modelos_por_celda: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """Análisis principal simplificado."""
    print("🚀 Iniciando análisis avanzado simplificado...")
    
    # Análisis de normalización
    normalization_result = analyze_normalization_simple(modelos_por_celda)
    
    # Análisis de visibilidad polinómica
    polynomial_result = analyze_polynomial_visibility_simple(modelos_por_celda)
    
    # Análisis dimensional
    dimensional_result = analyze_dimensional_consistency_simple(modelos_por_celda)
    
    # Generar recomendaciones
    recommendations = generate_recommendations_simple(
        normalization_result, polynomial_result, dimensional_result
    )
    
    # Compilar reporte final
    report = {
        'normalization': normalization_result,
        'polynomial_visibility': polynomial_result,
        'dimensional_consistency': dimensional_result,
        'recommendations': recommendations,
        'summary': {
            'total_normalization_problems': len(normalization_result['problems']),
            'total_polynomial_problems': len(polynomial_result['problems']),
            'total_dimensional_problems': len(dimensional_result['problems']),
            'total_recommendations': len(recommendations)
        }
    }
    
    print("\n" + "="*60)
    print("📊 RESUMEN DEL ANÁLISIS AVANZADO")
    print("="*60)
    print(f"🔍 Problemas de normalización: {report['summary']['total_normalization_problems']}")
    print(f"🔢 Problemas de visibilidad polinómica: {report['summary']['total_polynomial_problems']}")
    print(f"📐 Problemas dimensionales: {report['summary']['total_dimensional_problems']}")
    print(f"💡 Recomendaciones: {report['summary']['total_recommendations']}")
    print("="*60)
    
    return report


if __name__ == "__main__":
    print("Módulo equation_validator_simple cargado correctamente")
