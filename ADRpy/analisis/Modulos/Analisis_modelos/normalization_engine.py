"""
normalization_engine.py

Script unificado para manejo de normalización de datos de modelos.
Centraliza toda la lógica de normalización para visualización 2D y 3D.

FUNCIONES PRINCIPALES:
---------------------
1. normalize_model_data(): Normaliza datos de un modelo usando sus coeficientes originales
2. denormalize_predictions(): Desnormaliza predicciones a escala original
3. get_normalized_range_for_model(): Calcula rango normalizado para un modelo específico
4. normalize_x_values(): Normaliza valores X usando los rangos del modelo
5. normalize_y_values(): Normaliza valores Y usando los rangos del modelo

PRINCIPIOS DE NORMALIZACIÓN:
---------------------------
- Usamos SIEMPRE los coeficientes_originales e intercepto_original del JSON
- La normalización se hace por modelo individual (no global)
- Para 1 predictor: normalización Min-Max a [0,1] usando los datos de entrenamiento del modelo
- Para 2 predictores: normalización por predictor usando datos originales
- Los rangos se calculan a partir de datos_entrenamiento.X_original/y_original del JSON

COMPATIBILIDAD:
--------------
- Compatible con todos los tipos de modelo: linear, poly, log, exp, power
- Funciona para modelos de 1 y 2 predictores
- Mantiene consistencia entre gráficos 2D y 3D
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class ModelNormalizationEngine:
    """
    Motor de normalización centralizado para modelos de aeronaves.
    """
    
    def __init__(self):
        self.cache = {}  # Cache para evitar recálculos
    
    def get_model_data_ranges(self, modelo: Dict[str, Any]) -> Tuple[Optional[List[Tuple[float, float]]], Optional[Tuple[float, float]]]:
        """
        Extrae los rangos de datos originales de entrenamiento del modelo.
        
        Parameters:
        -----------
        modelo : Dict[str, Any]
            Diccionario del modelo con datos_entrenamiento
            
        Returns:
        --------
        Tuple[Optional[List[Tuple[float, float]]], Optional[Tuple[float, float]]]
            (rangos_x, rango_y) donde:
            - rangos_x: Lista de tuplas (min, max) para cada predictor
            - rango_y: Tupla (min, max) para la variable objetivo
        """
        try:
            datos_entrenamiento = modelo.get('datos_entrenamiento', {})
            X_original = datos_entrenamiento.get('X_original', [])
            y_original = datos_entrenamiento.get('y_original', [])
            
            if not X_original or not y_original:
                logger.warning(f"Modelo sin datos de entrenamiento originales")
                return None, None
            
            # Convertir a arrays numpy para facilidad de cálculo
            X_array = np.array(X_original)
            y_array = np.array(y_original)
            
            # Calcular rangos para cada predictor
            if X_array.ndim == 1:
                # Modelo de 1 predictor
                x_min, x_max = X_array.min(), X_array.max()
                rangos_x = [(float(x_min), float(x_max))]
            else:
                # Modelo de múltiples predictores
                rangos_x = []
                for i in range(X_array.shape[1]):
                    x_min, x_max = X_array[:, i].min(), X_array[:, i].max()
                    rangos_x.append((float(x_min), float(x_max)))
            
            # Calcular rango para variable objetivo
            y_min, y_max = y_array.min(), y_array.max()
            rango_y = (float(y_min), float(y_max))
            
            return rangos_x, rango_y
            
        except Exception as e:
            logger.error(f"Error calculando rangos del modelo: {e}")
            return None, None
    
    def normalize_x_values(self, x_values: Union[List, np.ndarray], rangos_x: List[Tuple[float, float]], 
                          predictor_index: int = 0) -> np.ndarray:
        """
        Normaliza valores X usando Min-Max scaling a [0,1].
        
        Parameters:
        -----------
        x_values : Union[List, np.ndarray]
            Valores X a normalizar
        rangos_x : List[Tuple[float, float]]
            Rangos (min, max) para cada predictor
        predictor_index : int
            Índice del predictor (0 para primer predictor, 1 para segundo, etc.)
            
        Returns:
        --------
        np.ndarray
            Valores X normalizados a [0,1]
        """
        try:
            if predictor_index >= len(rangos_x):
                logger.error(f"Índice de predictor {predictor_index} fuera de rango")
                return np.array(x_values)
            
            x_min, x_max = rangos_x[predictor_index]
            
            if x_max == x_min:
                logger.warning(f"Rango constante para predictor {predictor_index}, retornando 0.5")
                return np.full_like(x_values, 0.5)
            
            x_array = np.array(x_values)
            x_normalized = (x_array - x_min) / (x_max - x_min)
            
            return x_normalized
            
        except Exception as e:
            logger.error(f"Error normalizando valores X: {e}")
            return np.array(x_values)
    
    def normalize_y_values(self, y_values: Union[List, np.ndarray], rango_y: Tuple[float, float]) -> np.ndarray:
        """
        Normaliza valores Y usando Min-Max scaling a [0,1].
        
        Parameters:
        -----------
        y_values : Union[List, np.ndarray]
            Valores Y a normalizar
        rango_y : Tuple[float, float]
            Rango (min, max) para la variable objetivo
            
        Returns:
        --------
        np.ndarray
            Valores Y normalizados a [0,1]
        """
        try:
            y_min, y_max = rango_y
            
            if y_max == y_min:
                logger.warning(f"Rango Y constante, retornando 0.5")
                return np.full_like(y_values, 0.5)
            
            y_array = np.array(y_values)
            y_normalized = (y_array - y_min) / (y_max - y_min)
            
            return y_normalized
            
        except Exception as e:
            logger.error(f"Error normalizando valores Y: {e}")
            return np.array(y_values)
    
    def denormalize_predictions(self, predictions_normalized: np.ndarray, rango_y: Tuple[float, float]) -> np.ndarray:
        """
        Desnormaliza predicciones de [0,1] a escala original.
        
        Parameters:
        -----------
        predictions_normalized : np.ndarray
            Predicciones normalizadas en [0,1]
        rango_y : Tuple[float, float]
            Rango original (min, max) de la variable objetivo
            
        Returns:
        --------
        np.ndarray
            Predicciones en escala original
        """
        try:
            y_min, y_max = rango_y
            
            if y_max == y_min:
                return np.full_like(predictions_normalized, y_min)
            
            predictions_original = predictions_normalized * (y_max - y_min) + y_min
            
            return predictions_original
            
        except Exception as e:
            logger.error(f"Error desnormalizando predicciones: {e}")
            return predictions_normalized
    
    def generate_normalized_curve_from_model(self, modelo: Dict[str, Any], x_range_normalized: Optional[np.ndarray] = None, 
                                           resolution: int = 100) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
        """
        Genera curva normalizada usando directamente los coeficientes del modelo.
        
        Parameters:
        -----------
        modelo : Dict[str, Any]
            Diccionario del modelo con coeficientes_originales, intercepto_original
        x_range_normalized : np.ndarray, optional
            Rango X normalizado [0,1]. Si None, se genera automáticamente
        resolution : int
            Resolución de la curva (número de puntos)
            
        Returns:
        --------
        Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]
            (x_normalized, y_normalized, metadata) donde metadata contiene información del proceso
        """
        try:
            # Validar modelo
            n_predictores = modelo.get('n_predictores', 0)
            if n_predictores != 1:
                return None, None, {"error": "Solo modelos de 1 predictor soportados para curvas 2D"}
            
            # Obtener coeficientes y tipo
            coeficientes = modelo.get('coeficientes_originales', [])
            intercepto = modelo.get('intercepto_original', 0)
            tipo = modelo.get('tipo', '').lower()
            
            if not coeficientes:
                return None, None, {"error": "Modelo sin coeficientes válidos"}
            
            # Obtener rangos del modelo
            rangos_x, rango_y = self.get_model_data_ranges(modelo)
            if not rangos_x or not rango_y:
                # Generar rango sintético si no hay datos de entrenamiento
                logger.warning("Generando rango sintético para visualización")
                rangos_x = [(0, 100)]  # Rango sintético
                rango_y = (0, 100)     # Rango sintético
                metadata: Dict[str, Any] = {"synthetic_range": True, "warning": "Rango sintético utilizado"}
            else:
                metadata: Dict[str, Any] = {"synthetic_range": False}
            
            # Generar X normalizado si no se proporciona
            if x_range_normalized is None:
                x_range_normalized = np.linspace(0, 1, resolution)
            
            # Desnormalizar X para cálculos del modelo
            x_min, x_max = rangos_x[0]
            if x_max == x_min:
                x_original = np.full_like(x_range_normalized, x_min)
            else:
                x_original = x_range_normalized * (x_max - x_min) + x_min
            
            # Calcular predicciones según el tipo de modelo
            if tipo.startswith('linear'):
                # Modelo lineal: y = intercepto + c1*x1 + c2*x2 + ...
                if len(coeficientes) == 1:
                    y_original = intercepto + coeficientes[0] * x_original
                else:
                    # Múltiples predictores: solo podemos graficar el primer predictor
                    y_original = intercepto + coeficientes[0] * x_original
                    metadata["multi_predictor_warning"] = f"Modelo con {len(coeficientes)} predictores, graficando solo el primero"
            elif tipo.startswith('log'):
                # y = a + b*log(x), evitar log(0)
                x_original_safe = np.maximum(x_original, 1e-8)
                y_original = intercepto + coeficientes[0] * np.log(x_original_safe)
            elif tipo.startswith('exp'):
                # y = a*exp(b*x)
                y_original = intercepto * np.exp(coeficientes[0] * x_original)
            elif tipo.startswith('pot') or tipo.startswith('power'):
                # y = a*x^b, evitar x^0 con x=0
                x_original_safe = np.maximum(x_original, 1e-8)
                y_original = intercepto * np.power(x_original_safe, coeficientes[0])
            elif tipo.startswith('poly'):
                # Modelo polinómico: reconstruir usando PolynomialFeatures
                try:
                    from sklearn.preprocessing import PolynomialFeatures
                    # Para modelos poly-1: [x, x^2] -> coefs = [c1, c2]
                    # Para modelos poly-2: [x1, x2, x1^2, x1*x2, x2^2] -> coefs = [c1, c2, c3, c4, c5]
                    n_predictores = int(tipo.split('-')[1]) if '-' in tipo else 1
                    
                    if n_predictores == 1:
                        # Polinómico de 1 predictor: y = intercepto + c1*x + c2*x^2
                        if len(coeficientes) >= 2:
                            y_original = intercepto + coeficientes[0] * x_original + coeficientes[1] * (x_original ** 2)
                        else:
                            # Fallback a lineal si no hay suficientes coeficientes
                            y_original = intercepto + coeficientes[0] * x_original
                            metadata["poly_fallback_warning"] = "Insuficientes coeficientes para polinómico, usando aproximación lineal"
                    else:
                        # Múltiples predictores: solo graficar el primer predictor con sus términos
                        # Para poly-2: términos son [x1, x2, x1^2, x1*x2, x2^2]
                        # Graficamos: y = intercepto + c1*x1 + c3*x1^2 (asumiendo x2 = valor_medio)
                        if len(coeficientes) >= 3:
                            # Usar los coeficientes correspondientes al primer predictor
                            y_original = intercepto + coeficientes[0] * x_original + coeficientes[2] * (x_original ** 2)
                        else:
                            # Fallback a lineal
                            y_original = intercepto + coeficientes[0] * x_original
                            metadata["poly_fallback_warning"] = "Insuficientes coeficientes para polinómico completo, usando aproximación lineal"
                        metadata["multi_predictor_poly_warning"] = f"Modelo polinómico con {n_predictores} predictores, graficando curva del primer predictor"
                except ImportError:
                    # Si no está disponible sklearn, usar aproximación lineal
                    y_original = intercepto + coeficientes[0] * x_original
                    metadata["sklearn_missing_warning"] = "sklearn no disponible, usando aproximación lineal para modelo polinómico"
            else:
                # Por defecto, asumir lineal
                y_original = intercepto + coeficientes[0] * x_original
                metadata.update({"unknown_type_warning": f"Tipo de modelo desconocido '{tipo}', usando aproximación lineal"})
            
            # Normalizar Y usando el rango del modelo
            y_normalized = self.normalize_y_values(y_original, rango_y)
            
            # Validar resultados
            if np.any(np.isnan(y_normalized)) or np.any(np.isinf(y_normalized)):
                logger.warning("Predicciones contienen NaN o Inf, aplicando limpieza")
                y_normalized = np.nan_to_num(y_normalized, nan=0.5, posinf=1.0, neginf=0.0)
                metadata["cleaned"] = True
            
            metadata.update({
                "tipo_modelo": tipo,
                "n_points": len(x_range_normalized),
                "y_range_original": rango_y,
                "x_range_original": rangos_x[0]
            })
            
            return x_range_normalized, y_normalized, metadata
            
        except Exception as e:
            logger.error(f"Error generando curva normalizada: {e}")
            return None, None, {"error": str(e)}
    
    def normalize_training_points_from_model(self, modelo: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
        """
        Normaliza puntos de entrenamiento del modelo para visualización.
        
        Parameters:
        -----------
        modelo : Dict[str, Any]
            Diccionario del modelo con datos_entrenamiento
            
        Returns:
        --------
        Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]
            (x_normalized, y_normalized, metadata)
        """
        try:
            # Obtener datos de entrenamiento
            datos_entrenamiento = modelo.get('datos_entrenamiento', {})
            X_original = datos_entrenamiento.get('X_original', [])
            y_original = datos_entrenamiento.get('y_original', [])
            
            if not X_original or not y_original:
                return None, None, {"error": "Sin datos de entrenamiento"}
            
            # Obtener rangos
            rangos_x, rango_y = self.get_model_data_ranges(modelo)
            if not rangos_x or not rango_y:
                return None, None, {"error": "No se pudieron calcular rangos"}
            
            # Normalizar para modelo de 1 predictor
            n_predictores = modelo.get('n_predictores', 0)
            if n_predictores == 1:
                # Extraer valores X (pueden ser lista de listas o lista simple)
                if isinstance(X_original[0], list):
                    x_values = [x[0] for x in X_original]
                else:
                    x_values = X_original
                
                x_normalized = self.normalize_x_values(x_values, rangos_x, 0)
                y_normalized = self.normalize_y_values(y_original, rango_y)
                
                metadata = {
                    "n_predictores": 1,
                    "n_points": len(x_values),
                    "ranges_used": {"x": rangos_x[0], "y": rango_y}
                }
                
                return x_normalized, y_normalized, metadata
            
            else:
                # Para múltiples predictores, retornar información
                metadata = {
                    "n_predictores": n_predictores,
                    "n_points": len(X_original),
                    "ranges_used": {"x": rangos_x, "y": rango_y},
                    "note": "Múltiples predictores requieren procesamiento especial"
                }
                
                return None, None, metadata
                
        except Exception as e:
            logger.error(f"Error normalizando puntos de entrenamiento: {e}")
            return None, None, {"error": str(e)}
    
    def get_model_visualization_data(self, modelo: Dict[str, Any], curve_resolution: int = 100) -> Dict[str, Any]:
        """
        Función principal que retorna todos los datos normalizados necesarios para visualización.
        
        Parameters:
        -----------
        modelo : Dict[str, Any]
            Diccionario del modelo
        curve_resolution : int
            Resolución para la curva del modelo
            
        Returns:
        --------
        Dict[str, Any]
            Diccionario con todos los datos de visualización normalizados
        """
        try:
            result = {
                "modelo_valido": True,
                "tipo": modelo.get('tipo', 'unknown'),
                "n_predictores": modelo.get('n_predictores', 0),
                "predictor_names": modelo.get('predictores', []),
                "ecuacion_string": modelo.get('ecuacion_string', ''),
                "metrica_r2": modelo.get('r2', None),
                "metrica_mape": modelo.get('mape', None),
                "confianza": modelo.get('Confianza', None),
            }
            
            # Obtener rangos
            rangos_x, rango_y = self.get_model_data_ranges(modelo)
            result["rangos_originales"] = {"x": rangos_x, "y": rango_y}
            
            # Generar curva normalizada
            if result["n_predictores"] == 1:
                x_curve, y_curve, curve_meta = self.generate_normalized_curve_from_model(modelo, resolution=curve_resolution)
                result["curva"] = {
                    "x_normalized": x_curve.tolist() if x_curve is not None else None,
                    "y_normalized": y_curve.tolist() if y_curve is not None else None,
                    "metadata": curve_meta
                }
            else:
                result["curva"] = {"note": "Curvas solo para modelos de 1 predictor"}
            
            # Normalizar puntos de entrenamiento
            x_train, y_train, train_meta = self.normalize_training_points_from_model(modelo)
            result["puntos_entrenamiento"] = {
                "x_normalized": x_train.tolist() if x_train is not None else None,
                "y_normalized": y_train.tolist() if y_train is not None else None,
                "metadata": train_meta
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error obteniendo datos de visualización: {e}")
            return {
                "modelo_valido": False,
                "error": str(e),
                "tipo": modelo.get('tipo', 'unknown')
            }


# Instancia global del motor de normalización
normalization_engine = ModelNormalizationEngine()


# Funciones de conveniencia para compatibilidad con código existente
def get_normalized_model_data(modelo: Dict[str, Any], curve_resolution: int = 100) -> Dict[str, Any]:
    """
    Función de conveniencia para obtener datos normalizados de un modelo.
    
    Parameters:
    -----------
    modelo : Dict[str, Any]
        Diccionario del modelo
    curve_resolution : int
        Resolución para curvas
        
    Returns:
    --------
    Dict[str, Any]
        Datos normalizados listos para visualización
    """
    return normalization_engine.get_model_visualization_data(modelo, curve_resolution)


def normalize_models_for_visualization(modelos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normaliza una lista de modelos para visualización.
    
    Parameters:
    -----------
    modelos : List[Dict[str, Any]]
        Lista de modelos
        
    Returns:
    --------
    List[Dict[str, Any]]
        Lista de datos normalizados por modelo
    """
    results = []
    
    for i, modelo in enumerate(modelos):
        try:
            data = get_normalized_model_data(modelo)
            data["modelo_index"] = i
            results.append(data)
        except Exception as e:
            logger.error(f"Error procesando modelo {i}: {e}")
            results.append({
                "modelo_valido": False,
                "modelo_index": i,
                "error": str(e)
            })
    
    return results
