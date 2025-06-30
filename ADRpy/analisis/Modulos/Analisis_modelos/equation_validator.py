"""
Módulo avanzado para validar la racionalidad de normalizaciones, 
compatibilidad de ejes y visibilidad de modelos polinómicos.
"""

import json
import numpy as np
import pandas as pd
import re
from typing import Dict, List, Any, Tuple, Optional
import warnings


class EquationValidator:
    """Validador avanzado de ecuaciones y modelos."""
    
    def __init__(self, modelos_data: Dict[str, Any]):
        self.modelos_data = modelos_data
        self.report = {
            'normalization_issues': {},
            'polynomial_visibility': {},
            'equation_data_consistency': {},
            'dimensional_analysis': {},
            'recommendations': []
        }
    
    def analyze_all(self) -> Dict[str, Any]:
        """Ejecuta todos los análisis."""
        print("🔍 Iniciando análisis avanzado de validación...")
        
        # Análisis de normalización
        self.analyze_normalization()
        
        # Análisis de visibilidad polinómica
        self.analyze_polynomial_visibility()
        
        # Análisis de consistencia ecuación-datos
        self.analyze_equation_data_consistency()
        
        # Análisis dimensional
        self.analyze_dimensional_consistency()
        
        # Generar recomendaciones
        self.generate_recommendations()
        
        return self.report
    
    def analyze_normalization(self):
        """Analiza la racionalidad de las normalizaciones."""
        print("\n📏 Analizando normalizaciones...")
        
        normalization_stats = {}
        
        for celda_key, modelos_list in self.modelos_data.items():
            # Los datos están directamente como lista de modelos
            modelos = modelos_list if isinstance(modelos_list, list) else []
            
            for modelo in modelos:
                tipo_modelo = modelo.get('tipo', 'desconocido')
                predictores = modelo.get('predictores', [])
                parametro = modelo.get('Parámetro', 'desconocido')
                
                # Analizar datos de entrenamiento si están disponibles
                datos_entrenamiento = modelo.get('datos_entrenamiento', {})
                
                for predictor in predictores:
                    if predictor in datos_entrenamiento:
                        data = np.array(datos_entrenamiento[predictor])
                        
                        if len(data) > 0:
                            stats = {
                                'min': float(np.min(data)),
                                'max': float(np.max(data)),
                                'mean': float(np.mean(data)),
                                'std': float(np.std(data)),
                                'range': float(np.max(data) - np.min(data))
                            }
                            
                            # Detectar posibles problemas de normalización
                            issues = []
                            
                            # Rango muy amplio (puede indicar falta de normalización)
                            if stats['range'] > 1000:
                                issues.append('rango_muy_amplio')
                            
                            # Valores muy pequeños (puede indicar sobrenormalización)
                            if abs(stats['max']) < 0.001 and abs(stats['min']) < 0.001:
                                issues.append('valores_muy_pequeños')
                            
                            # Asimetría extrema
                            if abs(stats['mean']) > 10 * stats['std']:
                                issues.append('asimetria_extrema')
                            
                            key = f"{tipo_modelo}_{predictor}_{parametro}"
                            normalization_stats[key] = {
                                'celda': celda_key,
                                'stats': stats,
                                'issues': issues,
                                'tipo_modelo': tipo_modelo,
                                'predictor': predictor,
                                'parametro': parametro
                            }
        
        # Agregar al reporte
        problematic_normalizations = {k: v for k, v in normalization_stats.items() if v['issues']}
        self.report['normalization_issues'] = problematic_normalizations
        
        print(f"   ✓ Analizados {len(normalization_stats)} casos de normalización")
        print(f"   ⚠️  Encontrados {len(problematic_normalizations)} casos problemáticos")
    
    def analyze_polynomial_visibility(self):
        """Analiza la visibilidad de modelos polinómicos."""
        print("\n🔢 Analizando visibilidad de modelos polinómicos...")
        
        polynomial_analysis = {}
        
        for celda_key, modelos_list in self.modelos_data.items():
            modelos = modelos_list if isinstance(modelos_list, list) else []
            
            for modelo in modelos:
                tipo_modelo = modelo.get('tipo', '')
                
                if 'polynomial' in tipo_modelo.lower() or 'polinomial' in tipo_modelo.lower():
                    ecuacion = modelo.get('ecuacion_string', '')
                    coeficientes = modelo.get('coeficientes_originales', [])
                    
                    analysis = {
                        'celda': celda_key,
                        'tipo_modelo': tipo_modelo,
                        'ecuacion': ecuacion,
                        'n_coeficientes': len(coeficientes),
                        'visibility_issues': []
                    }
                    
                    # Analizar coeficientes
                    if coeficientes:
                        coef_array = np.array(coeficientes)
                        
                        # Coeficientes muy pequeños pueden ser invisibles
                        small_coefs = np.abs(coef_array) < 1e-6
                        if np.any(small_coefs):
                            analysis['visibility_issues'].append('coeficientes_muy_pequeños')
                        
                        # Diferencias de orden de magnitud extremas
                        if len(coef_array) > 1:
                            max_coef = np.max(np.abs(coef_array))
                            min_coef = np.min(np.abs(coef_array[coef_array != 0]))
                            if min_coef > 0 and max_coef / min_coef > 1e6:
                                analysis['visibility_issues'].append('ordenes_magnitud_extremos')
                    
                    # Analizar ecuación textual
                    if ecuacion:
                        # Buscar términos de alto grado
                        high_degree_terms = re.findall(r'\*\*\s*(\d+)', ecuacion)
                        if high_degree_terms:
                            max_degree = max(int(d) for d in high_degree_terms)
                            if max_degree > 4:
                                analysis['visibility_issues'].append('grado_muy_alto')
                    
                    key = f"{tipo_modelo}_{celda_key}"
                    polynomial_analysis[key] = analysis
        
        self.report['polynomial_visibility'] = polynomial_analysis
        
        print(f"   ✓ Analizados {len(polynomial_analysis)} modelos polinómicos")
        problematic_poly = {k: v for k, v in polynomial_analysis.items() if v['visibility_issues']}
        print(f"   ⚠️  Encontrados {len(problematic_poly)} con problemas de visibilidad")
    
    def analyze_equation_data_consistency(self):
        """Analiza la consistencia entre ecuaciones y datos de entrenamiento."""
        print("\n🔗 Analizando consistencia ecuación-datos...")
        
        consistency_analysis = {}
        
        for celda_key, modelos_list in self.modelos_data.items():
            modelos = modelos_list if isinstance(modelos_list, list) else []
            
            for modelo in modelos:
                ecuacion = modelo.get('ecuacion_string', '')
                datos_entrenamiento = modelo.get('datos_entrenamiento', {})
                coeficientes = modelo.get('coeficientes_originales', [])
                parametro = modelo.get('Parámetro', 'desconocido')
                
                if ecuacion and datos_entrenamiento:
                    analysis = {
                        'celda': celda_key,
                        'parametro': parametro,
                        'ecuacion': ecuacion,
                        'consistency_issues': []
                    }
                    
                    # Verificar que las variables en la ecuación existan en los datos
                    variables_ecuacion = self._extract_variables_from_equation(ecuacion)
                    variables_datos = set(datos_entrenamiento.keys())
                    
                    missing_vars = variables_ecuacion - variables_datos
                    if missing_vars:
                        analysis['consistency_issues'].append({
                            'tipo': 'variables_faltantes',
                            'variables': list(missing_vars)
                        })
                    
                    # Verificar rangos de aplicabilidad
                    if not missing_vars:  # Solo si tenemos todas las variables
                        try:
                            # Evaluar ecuación en algunos puntos de los datos
                            data_length = len(list(datos_entrenamiento.values())[0])
                            sample_size = min(10, data_length)
                            sample_indices = np.random.choice(data_length, sample_size, replace=False)
                            
                            for idx in sample_indices:
                                values = {var: datos_entrenamiento[var][idx] for var in variables_datos}
                                
                                try:
                                    result = self._evaluate_equation_safely(ecuacion, values)
                                    if np.isnan(result) or np.isinf(result):
                                        analysis['consistency_issues'].append({
                                            'tipo': 'resultado_invalido',
                                            'indice': int(idx),
                                            'valores': values
                                        })
                                        break
                                except Exception:
                                    analysis['consistency_issues'].append({
                                        'tipo': 'error_evaluacion',
                                        'indice': int(idx),
                                        'valores': values
                                    })
                                    break
                        
                        except Exception as e:
                            analysis['consistency_issues'].append({
                                'tipo': 'error_muestreo',
                                'error': str(e)
                            })
                    
                    key = f"{parametro}_{celda_key}"
                    consistency_analysis[key] = analysis
        
        self.report['equation_data_consistency'] = consistency_analysis
        
        print(f"   ✓ Analizados {len(consistency_analysis)} casos de consistencia")
        problematic_consistency = {k: v for k, v in consistency_analysis.items() if v['consistency_issues']}
        print(f"   ⚠️  Encontrados {len(problematic_consistency)} con problemas de consistencia")
    
    def analyze_dimensional_consistency(self):
        """Analiza la consistencia dimensional."""
        print("\n📐 Analizando consistencia dimensional...")
        
        # Mapeo básico de unidades conocidas
        unit_patterns = {
            'masa': ['kg', 'mass', 'peso', 'weight'],
            'longitud': ['m', 'length', 'span', 'chord', 'longitud', 'envergadura'],
            'velocidad': ['m/s', 'velocity', 'speed', 'velocidad'],
            'area': ['m2', 'area', 'superficie'],
            'potencia': ['W', 'power', 'potencia', 'hp'],
            'adimensional': ['ratio', 'coefficient', 'factor', 'coeficiente']
        }
        
        dimensional_analysis = {}
        
        for celda_key, modelos_list in self.modelos_data.items():
            modelos = modelos_list if isinstance(modelos_list, list) else []
            
            for modelo in modelos:
                predictores = modelo.get('predictores', [])
                parametro = modelo.get('Parámetro', 'desconocido')
                ecuacion = modelo.get('ecuacion_string', '')
                
                analysis = {
                    'celda': celda_key,
                    'parametro': parametro,
                    'predictores': predictores,
                    'dimensional_issues': []
                }
                
                # Inferir dimensionalidad de predictores y parámetros
                predictor_dims = {}
                for pred in predictores:
                    pred_lower = pred.lower()
                    for dim_type, patterns in unit_patterns.items():
                        if any(pattern in pred_lower for pattern in patterns):
                            predictor_dims[pred] = dim_type
                            break
                    else:
                        predictor_dims[pred] = 'desconocida'
                
                param_lower = parametro.lower()
                param_dim = 'desconocida'
                for dim_type, patterns in unit_patterns.items():
                    if any(pattern in param_lower for pattern in patterns):
                        param_dim = dim_type
                        break
                
                # Verificar compatibilidad dimensional básica
                if len(predictores) == 1 and ecuacion:
                    pred_dim = predictor_dims[predictores[0]]
                    
                    # Si es un modelo lineal simple, las dimensiones deberían ser compatibles
                    if 'linear' in modelo.get('tipo', '').lower():
                        if pred_dim != param_dim and pred_dim != 'desconocida' and param_dim != 'desconocida':
                            analysis['dimensional_issues'].append({
                                'tipo': 'incompatibilidad_dimensional',
                                'predictor_dim': pred_dim,
                                'parametro_dim': param_dim
                            })
                
                # Verificar normalización dimensional
                datos_entrenamiento = modelo.get('datos_entrenamiento', {})
                for pred in predictores:
                    if pred in datos_entrenamiento:
                        data = np.array(datos_entrenamiento[pred])
                        if len(data) > 0:
                            data_range = np.max(data) - np.min(data)
                            
                            # Si los datos están normalizados pero la variable tiene dimensión física
                            if abs(data_range) < 10 and predictor_dims[pred] not in ['adimensional', 'desconocida']:
                                analysis['dimensional_issues'].append({
                                    'tipo': 'posible_sobrenormalizacion',
                                    'predictor': pred,
                                    'rango': float(data_range),
                                    'dimension': predictor_dims[pred]
                                })
                
                key = f"{parametro}_{celda_key}"
                dimensional_analysis[key] = analysis
        
        self.report['dimensional_analysis'] = dimensional_analysis
        
        print(f"   ✓ Analizados {len(dimensional_analysis)} casos dimensionales")
        problematic_dimensional = {k: v for k, v in dimensional_analysis.items() if v['dimensional_issues']}
        print(f"   ⚠️  Encontrados {len(problematic_dimensional)} con problemas dimensionales")
    
    def generate_recommendations(self):
        """Genera recomendaciones basadas en el análisis."""
        print("\n💡 Generando recomendaciones...")
        
        recommendations = []
        
        # Recomendaciones para normalización
        if self.report['normalization_issues']:
            recommendations.append({
                'categoria': 'normalización',
                'prioridad': 'alta',
                'descripcion': 'Se detectaron problemas de normalización que pueden afectar la visualización',
                'acciones': [
                    'Verificar que los predictores estén adecuadamente escalados',
                    'Considerar normalización min-max o z-score según corresponda',
                    'Añadir advertencias en la visualización sobre escalas'
                ]
            })
        
        # Recomendaciones para visibilidad polinómica
        if self.report['polynomial_visibility']:
            recommendations.append({
                'categoria': 'visibilidad_polinómica',
                'prioridad': 'media',
                'descripcion': 'Algunos modelos polinómicos pueden tener problemas de visibilidad',
                'acciones': [
                    'Revisar coeficientes de modelos polinómicos',
                    'Considerar reescalado de términos polinómicos',
                    'Implementar visualización adaptativa para diferentes órdenes de magnitud'
                ]
            })
        
        # Recomendaciones para consistencia
        if self.report['equation_data_consistency']:
            recommendations.append({
                'categoria': 'consistencia',
                'prioridad': 'alta',
                'descripcion': 'Se encontraron inconsistencias entre ecuaciones y datos',
                'acciones': [
                    'Validar que las ecuaciones correspondan a los datos de entrenamiento',
                    'Verificar que todas las variables estén disponibles',
                    'Implementar validación robusta de entrada en la visualización'
                ]
            })
        
        # Recomendaciones dimensionales
        if self.report['dimensional_analysis']:
            recommendations.append({
                'categoria': 'dimensional',
                'prioridad': 'media',
                'descripcion': 'Se detectaron posibles problemas dimensionales',
                'acciones': [
                    'Verificar unidades y dimensionalidad de variables',
                    'Añadir etiquetas de unidades en visualizaciones',
                    'Considerar indicadores de normalización en la interfaz'
                ]
            })
        
        # Recomendaciones generales
        recommendations.append({
            'categoria': 'general',
            'prioridad': 'media',
            'descripcion': 'Mejoras generales para robustez',
            'acciones': [
                'Implementar validación de rangos en tiempo de visualización',
                'Añadir indicadores de calidad/confianza para cada modelo',
                'Proporcionar información contextual sobre normalización en la interfaz'
            ]
        })
        
        self.report['recommendations'] = recommendations
        
        print(f"   ✓ Generadas {len(recommendations)} categorías de recomendaciones")
    
    def _extract_variables_from_equation(self, equation: str) -> set:
        """Extrae variables de una ecuación."""
        # Buscar patrones de variables (letras seguidas de letras/números/guiones bajos)
        variables = set(re.findall(r'\b[a-zA-Z][a-zA-Z0-9_]*\b', equation))
        
        # Remover funciones matemáticas conocidas
        math_functions = {'sin', 'cos', 'tan', 'log', 'exp', 'sqrt', 'abs', 'pow', 'np', 'math'}
        variables -= math_functions
        
        return variables
    
    def _evaluate_equation_safely(self, equation: str, values: Dict[str, float]) -> float:
        """Evalúa una ecuación de forma segura."""
        # Crear un entorno seguro con las variables y funciones matemáticas
        safe_dict = {
            '__builtins__': {},
            'np': np,
            'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
            'log': np.log, 'exp': np.exp, 'sqrt': np.sqrt,
            'abs': abs, 'pow': pow
        }
        safe_dict.update(values)
        
        try:
            result = eval(equation, safe_dict)
            return float(result)
        except Exception:
            return np.nan
    
    def print_summary(self):
        """Imprime un resumen del análisis."""
        print("\n" + "="*60)
        print("📊 RESUMEN DEL ANÁLISIS AVANZADO")
        print("="*60)
        
        total_issues = (
            len(self.report['normalization_issues']) +
            len(self.report['polynomial_visibility']) +
            len(self.report['equation_data_consistency']) +
            len(self.report['dimensional_analysis'])
        )
        
        print(f"\n🔍 Total de problemas detectados: {total_issues}")
        
        for category, issues in self.report.items():
            if category != 'recommendations' and issues:
                print(f"\n📌 {category.replace('_', ' ').title()}:")
                if isinstance(issues, dict):
                    for key, issue in list(issues.items())[:3]:  # Mostrar solo los primeros 3
                        print(f"   • {key}: {len(issue.get('issues', issue.get('visibility_issues', issue.get('consistency_issues', issue.get('dimensional_issues', [])))))} problemas")
                    if len(issues) > 3:
                        print(f"   ... y {len(issues) - 3} más")
        
        print(f"\n💡 Recomendaciones generadas: {len(self.report['recommendations'])}")
        for rec in self.report['recommendations']:
            print(f"   • {rec['categoria']} (prioridad: {rec['prioridad']})")
        
        print("\n" + "="*60)


def main(modelos_por_celda: Dict[str, Any]) -> Dict[str, Any]:
    """Función principal del validador."""
    validator = EquationValidator(modelos_por_celda)
    report = validator.analyze_all()
    validator.print_summary()
    return report


if __name__ == "__main__":
    # Para testing independiente
    print("Módulo equation_validator cargado correctamente")
