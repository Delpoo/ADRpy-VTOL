"""
plot_3d.py

Funciones para generar visualizaciones 3D de modelos con 2 predictores.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import logging
from .plot_config import COLORS, SYMBOLS
from .plot_data_access import get_model_original_data, get_model_training_data

logger = logging.getLogger(__name__)


def test_poly2_models_quick(modelos_por_celda: Dict) -> None:
    """
    Función de testeo rápido para modelos poly-2.
    Imprime todos los strings de ecuación y valida uno con datos dummy.
    """
    print(f"\n🧪 TESTEO RÁPIDO DE MODELOS POLY-2")
    print(f"=" * 60)
    
    poly2_models = []
    
    # Recopilar todos los modelos poly-2
    for celda_key, modelos in modelos_por_celda.items():
        if isinstance(modelos, list):
            for modelo in modelos:
                if isinstance(modelo, dict) and modelo.get('tipo') == 'poly-2':
                    modelo_copia = modelo.copy()
                    modelo_copia['celda'] = celda_key
                    poly2_models.append(modelo_copia)
    
    print(f"📊 Total modelos poly-2: {len(poly2_models)}")
    
    if not poly2_models:
        print(f"❌ No se encontraron modelos poly-2")
        return
    
    # Mostrar ecuaciones
    print(f"\n📋 ECUACIONES DE MODELOS POLY-2:")
    for i, modelo in enumerate(poly2_models):
        celda = modelo.get('celda', 'N/A')
        ecuacion = modelo.get('ecuacion_string', '')
        predictores = modelo.get('predictores', [])
        coefs = modelo.get('coeficientes', [])
        
        print(f"  {i+1}. {celda}:")
        print(f"     Predictores: {predictores}")
        print(f"     Ecuación: {ecuacion}")
        print(f"     Coeficientes: {len(coefs)} valores")
        
        # Detectar problemas
        problemas = []
        if not ecuacion:
            problemas.append("Sin ecuación")
        if len(coefs) != 6:
            problemas.append(f"Coefs: {len(coefs)}/6")
        if 'x0' not in ecuacion and 'x1' not in ecuacion:
            problemas.append("Variables no estándar")
        if any(c is None or (isinstance(c, float) and np.isnan(c)) for c in coefs):
            problemas.append("Coefs con NaN")
        
        if problemas:
            print(f"     ⚠️ Problemas: {', '.join(problemas)}")
        else:
            print(f"     ✅ OK")
    
    # Probar uno con datos dummy
    if poly2_models:
        print(f"\n🧪 PRUEBA CON DATOS DUMMY:")
        modelo_test = poly2_models[0]
        
        try:
            from .equation_analyzer import extract_equation_components, create_equation_function
            
            components = extract_equation_components(modelo_test)
            if components.get('es_valido', False):
                func = create_equation_function(components)
                if func is not None:
                    # Probar con valores dummy
                    test_vals = [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]
                    print(f"   Modelo: {modelo_test.get('celda')}")
                    print(f"   Función creada: ✅")
                    
                    for x1, x2 in test_vals:
                        try:
                            result = func(x1, x2)
                            print(f"   f({x1}, {x2}) = {result:.6f}")
                        except Exception as e:
                            print(f"   f({x1}, {x2}) = ERROR: {e}")
                else:
                    print(f"   ❌ No se pudo crear función")
            else:
                print(f"   ❌ Componentes no válidos")
                
        except Exception as e:
            print(f"   ❌ Error en prueba: {e}")
    
    print(f"=" * 60)


def create_3d_plot(
    modelos_filtrados: Dict,
    aeronave: str,
    parametro: str,
    show_training_points: bool = True,
    show_model_surfaces: bool = True,
    highlight_model_idx: Optional[int] = None,
    detalles_por_celda: Optional[Dict] = None,
    selected_imputation_methods: Optional[List[str]] = None
) -> go.Figure:
    """
    Crea un gráfico 3D interactivo para visualizar modelos con 2 predictores.
    
    Parameters:
    -----------
    modelos_filtrados : Dict
        Modelos filtrados por celda
    aeronave : str
        Nombre de la aeronave
    parametro : str
        Parámetro objetivo
    show_training_points : bool
        Si mostrar puntos de entrenamiento
    show_model_surfaces : bool
        Si mostrar superficies de los modelos
    highlight_model_idx : Optional[int]
        Índice del modelo a resaltar
    detalles_por_celda : Optional[Dict]
        Detalles de imputación por celda
    selected_imputation_methods : Optional[List[str]]
        Métodos de imputación seleccionados
        
    Returns:
    --------
    go.Figure
        Figura 3D de Plotly
    """
    # Filtrar solo modelos con 2 predictores
    celda_key = f"{aeronave}|{parametro}"
    modelos = modelos_filtrados.get(celda_key, [])
    modelos_2pred = [m for m in modelos if isinstance(m, dict) and m.get('n_predictores', 0) == 2]
    
    if not modelos_2pred:
        fig = go.Figure()
        fig.add_annotation(
            text="No hay modelos con 2 predictores disponibles para esta combinación",
            x=0.5, y=0.5, z=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            title=f"Vista 3D - {aeronave} - {parametro}",
            scene=dict(
                xaxis_title="Predictor 1",
                yaxis_title="Predictor 2", 
                zaxis_title=parametro
            )
        )
        return fig
    
    fig = go.Figure()
    
    # Obtener datos originales para normalización
    original_data = get_model_original_data(modelos_2pred[0])
    if original_data is None or (hasattr(original_data, 'empty') and original_data.empty) or len(original_data) < 2:
        # Sin datos originales suficientes
        fig.add_annotation(
            text="Datos insuficientes para visualización 3D",
            x=0.5, y=0.5, z=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig
    
    # Obtener nombres de predictores del primer modelo como referencia
    predictores = modelos_2pred[0].get('predictores', ['Predictor 1', 'Predictor 2'])
    predictor1_name = predictores[0]
    predictor2_name = predictores[1]
    
    # Calcular rangos de normalización basados en datos originales
    pred1_data = original_data.get(predictor1_name, [])
    pred2_data = original_data.get(predictor2_name, [])
    target_data = original_data.get(parametro, [])
    
    if len(pred1_data) == 0 or len(pred2_data) == 0 or len(target_data) == 0:
        fig.add_annotation(
            text="Datos de predictores insuficientes",
            x=0.5, y=0.5, z=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig
    
    # Rangos para normalización
    pred1_range = (min(pred1_data), max(pred1_data))
    pred2_range = (min(pred2_data), max(pred2_data))
    
    # Función de normalización
    def normalize_values(values, value_range):
        min_val, max_val = value_range
        if max_val == min_val:
            return [0.5] * len(values)
        return [(v - min_val) / (max_val - min_val) for v in values]
    
    # Agregar puntos de entrenamiento si se solicita
    if show_training_points:
        training_data = get_model_training_data(modelos_2pred[0])
        if training_data is not None and len(training_data) >= 3:
            train_pred1 = training_data.get(predictor1_name, [])
            train_pred2 = training_data.get(predictor2_name, [])
            train_target = training_data.get(parametro, [])
            
            if len(train_pred1) > 0 and len(train_pred2) > 0 and len(train_target) > 0:
                # Normalizar puntos de entrenamiento
                train_pred1_norm = normalize_values(train_pred1, pred1_range)
                train_pred2_norm = normalize_values(train_pred2, pred2_range)
                
                fig.add_trace(go.Scatter3d(
                    x=train_pred1_norm,
                    y=train_pred2_norm,
                    z=train_target,
                    mode='markers',
                    marker=dict(
                        size=6,
                        color='blue',
                        symbol='circle',
                        opacity=0.8
                    ),
                    name='Puntos de Entrenamiento',
                    hovertemplate=f'<b>Punto de Entrenamiento</b><br>' +
                                f'{predictor1_name}: %{{customdata[0]:.3f}}<br>' +
                                f'{predictor2_name}: %{{customdata[1]:.3f}}<br>' +
                                f'{parametro}: %{{z:.3f}}<extra></extra>',
                    customdata=list(zip(train_pred1, train_pred2))
                ))
    
    # Agregar superficies de modelos si se solicita
    if show_model_surfaces:
        for i, modelo in enumerate(modelos_2pred):
            try:
                surface_data = _create_model_surface_3d(
                    modelo, pred1_range, pred2_range, 
                    predictor1_name, predictor2_name, parametro
                )
                
                if surface_data is not None:
                    x_surf, y_surf, z_surf = surface_data
                    
                    # Determinar opacidad y color
                    if highlight_model_idx is not None:
                        opacity = 0.8 if i == highlight_model_idx else 0.3
                        color = COLORS['model_lines'][i % len(COLORS['model_lines'])]
                    else:
                        opacity = 0.6
                        color = COLORS['model_lines'][i % len(COLORS['model_lines'])]
                    
                    tipo_modelo = modelo.get('tipo', 'Desconocido')
                    mape = modelo.get('mape', 0)
                    r2 = modelo.get('r2', 0)
                    
                    fig.add_trace(go.Surface(
                        x=x_surf,
                        y=y_surf,
                        z=z_surf,
                        name=f'{tipo_modelo} (MAPE: {mape:.2f}%)',
                        opacity=opacity,
                        colorscale=[[0, color], [1, color]],
                        showscale=False,
                        hovertemplate=f'<b>{tipo_modelo}</b><br>' +
                                    f'MAPE: {mape:.2f}%<br>' +
                                    f'R²: {r2:.3f}<br>' +
                                    f'{predictor1_name}: %{{x:.3f}}<br>' +
                                    f'{predictor2_name}: %{{y:.3f}}<br>' +
                                    f'{parametro}: %{{z:.3f}}<extra></extra>'
                    ))
                    
            except Exception as e:
                logger.warning(f"Error creando superficie para modelo {i}: {e}")
                continue
    
    # Agregar puntos de imputación si están disponibles
    if detalles_por_celda and selected_imputation_methods:
        _add_imputation_points_3d(
            fig, detalles_por_celda, celda_key, modelos_2pred,
            selected_imputation_methods, pred1_range, pred2_range,
            predictor1_name, predictor2_name, parametro
        )
    
    # Configurar layout 3D
    fig.update_layout(
        title=f"Vista 3D - {aeronave} - {parametro}",
        scene=dict(
            xaxis_title=f"{predictor1_name} (normalizado)",
            yaxis_title=f"{predictor2_name} (normalizado)",
            zaxis_title=parametro,
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig


def _validate_poly2_model_before_surface(modelo: Dict) -> Tuple[bool, str]:
    """
    Valida un modelo poly-2 antes de intentar crear superficie 3D.
    
    Returns:
    --------
    Tuple[bool, str]: (es_valido, mensaje_error)
    """
    tipo = modelo.get('tipo', '')
    if tipo != 'poly-2':
        return False, f"Tipo {tipo} no es poly-2"
    
    # Verificar ecuación
    ecuacion_string = modelo.get('ecuacion_string', '')
    if not ecuacion_string:
        return False, "Sin ecuacion_string"
    
    # Verificar coeficientes
    coeficientes = modelo.get('coeficientes', [])
    if not coeficientes:
        return False, "Sin coeficientes"
    
    if len(coeficientes) != 6:
        return False, f"Coeficientes incorrectos: tiene {len(coeficientes)}, necesita 6"
    
    if any(c is None or (isinstance(c, float) and np.isnan(c)) for c in coeficientes):
        return False, "Coeficientes contienen NaN/None"
    
    # Verificar predictores
    predictores = modelo.get('predictores', [])
    if len(predictores) != 2:
        return False, f"Predictores incorrectos: tiene {len(predictores)}, necesita 2"
    
    # Verificar nombres de variables en ecuación
    if 'x0' not in ecuacion_string or 'x1' not in ecuacion_string:
        if any(pred in ecuacion_string for pred in predictores):
            return False, f"Ecuación usa nombres de variables ({predictores}) en lugar de x0, x1"
        else:
            return False, "Ecuación no contiene variables x0, x1"
    
    return True, "Válido"


def _preprocess_poly2_equation(modelo: Dict) -> Dict:
    """
    Preprocesa la ecuación de un modelo poly-2 para estandarizar variables.
    
    Returns:
    --------
    Dict: Modelo con ecuación corregida
    """
    modelo_copia = modelo.copy()
    ecuacion_string = modelo_copia.get('ecuacion_string', '')
    predictores = modelo_copia.get('predictores', [])
    
    if len(predictores) >= 2 and ecuacion_string:
        # Reemplazar nombres de variables por x0, x1
        ecuacion_corregida = ecuacion_string
        if predictores[0] in ecuacion_string:
            ecuacion_corregida = ecuacion_corregida.replace(predictores[0], 'x0')
        if predictores[1] in ecuacion_string:
            ecuacion_corregida = ecuacion_corregida.replace(predictores[1], 'x1')
        
        # Solo actualizar si se hicieron cambios
        if ecuacion_corregida != ecuacion_string:
            modelo_copia['ecuacion_string'] = ecuacion_corregida
            logger.info(f"Ecuación corregida para {predictores}: {ecuacion_string} -> {ecuacion_corregida}")
    
    return modelo_copia


def _create_model_surface_3d(
    modelo: Dict,
    pred1_range: Tuple[float, float],
    pred2_range: Tuple[float, float],
    predictor1_name: str,
    predictor2_name: str,
    parametro: str,
    resolution: int = 20
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Crea una superficie 3D para un modelo con 2 predictores.
    CON LOGGING DETALLADO PARA DEBUGGING.
    
    Parameters:
    -----------
    modelo : Dict
        Datos del modelo
    pred1_range : Tuple[float, float]
        Rango (min, max) del primer predictor
    pred2_range : Tuple[float, float]
        Rango (min, max) del segundo predictor
    predictor1_name : str
        Nombre del primer predictor
    predictor2_name : str
        Nombre del segundo predictor
    parametro : str
        Nombre del parámetro objetivo
    resolution : int
        Resolución de la grilla (número de puntos por dimensión)
        
    Returns:
    --------
    Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]
        Arrays X, Y, Z para la superficie, o None si no se puede crear
    """
    # LOGGING DETALLADO - INFORMACIÓN DEL MODELO
    print(f"\n🔧 CREANDO SUPERFICIE 3D:")
    print(f"   Tipo: {modelo.get('tipo', 'N/A')}")
    print(f"   Celda: {modelo.get('celda', 'N/A')}")
    print(f"   Parámetro: {parametro}")
    print(f"   Predictores: {modelo.get('predictores', [])}")
    print(f"   Predictor1: {predictor1_name} rango: {pred1_range}")
    print(f"   Predictor2: {predictor2_name} rango: {pred2_range}")
    print(f"   Resolución: {resolution}")
    
    # LOGGING DETALLADO - ECUACIÓN Y COEFICIENTES
    ecuacion_string = modelo.get('ecuacion_string', '')
    coeficientes = modelo.get('coeficientes', [])
    print(f"\n📊 ECUACIÓN Y COEFICIENTES:")
    print(f"   Ecuación string: {ecuacion_string}")
    print(f"   Coeficientes ({len(coeficientes)}): {coeficientes}")
    
    # Validación específica para poly-2
    if modelo.get('tipo') == 'poly-2':
        es_valido, mensaje = _validate_poly2_model_before_surface(modelo)
        print(f"   Validación poly-2: {es_valido} - {mensaje}")
        
        if not es_valido:
            print(f"   ❌ MODELO POLY-2 NO VÁLIDO: {mensaje}")
            return None
        
        # Preprocesar ecuación si es necesario
        modelo = _preprocess_poly2_equation(modelo)
        if modelo.get('ecuacion_string') != ecuacion_string:
            print(f"   🔧 Ecuación preprocesada: {modelo.get('ecuacion_string')}")
    
    try:
        # LOGGING DETALLADO - OBTENER FUNCIÓN DE PREDICCIÓN
        print(f"\n⚙️ OBTENIENDO FUNCIÓN DE PREDICCIÓN:")
        
        # Obtener función de predicción del modelo
        modelo_func = modelo.get('modelo_entrenado')
        print(f"   modelo_entrenado disponible: {modelo_func is not None}")
        
        if modelo_func is None:
            print(f"   Creando función desde ecuación...")
            
            # Crear función desde ecuación y coeficientes
            tipo = modelo.get('tipo', '')
            coeficientes = modelo.get('coeficientes_originales', [])
            intercepto = modelo.get('intercepto_original', 0)
            n_predictores = modelo.get('n_predictores', 0)
            
            print(f"   Tipo modelo: {tipo}")
            print(f"   Coeficientes: {coeficientes}")
            print(f"   Intercepto: {intercepto}")
            print(f"   N predictores: {n_predictores}")
            
            if tipo.startswith('linear') and len(coeficientes) >= n_predictores:
                # Modelo lineal: y = intercept + coef1*x1 + coef2*x2
                def modelo_func(x1, x2):
                    if n_predictores >= 2:
                        return intercepto + coeficientes[0]*x1 + coeficientes[1]*x2
                    else:
                        return intercepto + coeficientes[0]*x1
                print(f"   ✅ Función lineal creada")
                
            elif tipo.startswith('poly') and len(coeficientes) >= 5:
                # Modelo polinómico: usar ecuación string si está disponible
                ecuacion_string = modelo.get('ecuacion_string', '')
                print(f"   Ecuación string: {ecuacion_string}")
                
                if ecuacion_string and '=' in ecuacion_string:
                    # Extraer lado derecho de la ecuación
                    expression = ecuacion_string.split('=')[1].strip()
                    
                    def modelo_func(x1, x2):
                        try:
                            # Variables para la evaluación
                            variables = {
                                'x0': x1, 'x1': x2, 'x2': x1*x1, 'x3': x1*x2, 'x4': x2*x2
                            }
                            # Evaluar expresión de manera segura
                            safe_dict = {"__builtins__": {}, "pow": pow}
                            safe_dict.update(variables)
                            return eval(expression, safe_dict)
                        except Exception as e:
                            print(f"Error evaluando polinomio: {e}")
                            return np.nan
                    print(f"   ✅ Función polinómica creada desde ecuación")
                else:
                    # Usar coeficientes directamente para poly-2
                    def modelo_func(x1, x2):
                        if len(coeficientes) >= 5:
                            # Forma: intercept + c0*x1 + c1*x2 + c2*x1^2 + c3*x1*x2 + c4*x2^2
                            return (intercepto + coeficientes[0]*x1 + coeficientes[1]*x2 + 
                                   coeficientes[2]*x1*x1 + coeficientes[3]*x1*x2 + coeficientes[4]*x2*x2)
                        else:
                            return np.nan
                    print(f"   ✅ Función polinómica creada desde coeficientes")
            else:
                print(f"   ❌ Tipo de modelo no soportado: {tipo}")
                return None
            
            if modelo_func is None:
                print(f"   ❌ ERROR: No se pudo crear función para modelo tipo {tipo}")
                logger.warning(f"No se pudo crear función para modelo tipo {tipo}")
                return None
        
        # LOGGING DETALLADO - CREAR GRILLA
        print(f"\n📊 CREANDO GRILLA DE VALORES:")
        
        # Crear grilla de valores normalizados
        x_norm = np.linspace(0, 1, resolution)
        y_norm = np.linspace(0, 1, resolution)
        X_norm, Y_norm = np.meshgrid(x_norm, y_norm)
        print(f"   Grilla normalizada: {X_norm.shape}")
        
        # Convertir a valores reales para predicción
        pred1_min, pred1_max = pred1_range
        pred2_min, pred2_max = pred2_range
        
        X_real = X_norm * (pred1_max - pred1_min) + pred1_min
        Y_real = Y_norm * (pred2_max - pred2_min) + pred2_min
        print(f"   Rango real X: [{X_real.min():.3f}, {X_real.max():.3f}]")
        print(f"   Rango real Y: [{Y_real.min():.3f}, {Y_real.max():.3f}]")
        
        # Verificar que los arrays no estén vacíos ni sean constantes
        if X_real.size == 0 or Y_real.size == 0:
            print(f"   ❌ ERROR: Arrays X o Y están vacíos")
            return None
        
        if np.all(X_real == X_real.flat[0]) or np.all(Y_real == Y_real.flat[0]):
            print(f"   ❌ ERROR: Arrays X o Y son constantes")
            return None
        
        # LOGGING DETALLADO - GENERAR PREDICCIONES
        print(f"\n🎯 GENERANDO PREDICCIONES:")
        
        # Generar predicciones
        Z = np.zeros_like(X_real)
        error_count = 0
        
        for i in range(resolution):
            for j in range(resolution):
                try:
                    # Hacer predicción según el tipo de función
                    if hasattr(modelo_func, 'predict'):
                        # Modelo sklearn con método predict
                        point = {
                            predictor1_name: X_real[i, j],
                            predictor2_name: Y_real[i, j]
                        }
                        prediction = modelo_func.predict(pd.DataFrame([point]))[0]  # type: ignore
                    else:
                        # Función simple creada desde ecuación
                        prediction = modelo_func(X_real[i, j], Y_real[i, j])
                    
                    Z[i, j] = prediction
                    
                except Exception as e:
                    # Si hay error en la predicción, usar NaN
                    Z[i, j] = np.nan
                    error_count += 1
                    
                    # Logging detallado del primer error
                    if error_count == 1:
                        print(f"   ❌ PRIMER ERROR EN PREDICCIÓN:")
                        print(f"      Posición: ({i}, {j})")
                        print(f"      Valores entrada: x={X_real[i, j]:.6f}, y={Y_real[i, j]:.6f}")
                        print(f"      Error: {e}")
                        print(f"      Tipo función: {type(modelo_func)}")
                        print(f"      Ecuación: {modelo.get('ecuacion_string', 'N/A')}")
                        print(f"      Coeficientes: {modelo.get('coeficientes', [])}")
                        
                        # Stack trace completo para el primer error
                        import traceback
                        print(f"      Stack trace:")
                        traceback.print_exc()
        
        print(f"   Total errores: {error_count}/{resolution*resolution}")
        print(f"   Valores válidos: {(~np.isnan(Z)).sum()}/{Z.size}")
        print(f"   Rango Z: [{np.nanmin(Z):.6f}, {np.nanmax(Z):.6f}]")
        
        # Verificar que tengamos suficientes valores válidos
        valid_ratio = (~np.isnan(Z)).sum() / Z.size
        print(f"   Ratio válidos: {valid_ratio:.3f}")
        
        if valid_ratio < 0.5:
            print(f"   ❌ ERROR: Demasiados valores NaN ({valid_ratio:.3f} < 0.5)")
            return None
        
        print(f"   ✅ SUPERFICIE CREADA EXITOSAMENTE")
        return X_norm, Y_norm, Z
        
    except Exception as e:
        # LOGGING DETALLADO - ERROR GENERAL
        print(f"\n💥 ERROR GENERAL CREANDO SUPERFICIE:")
        print(f"   Error: {e}")
        print(f"   Tipo: {type(e).__name__}")
        print(f"   Modelo tipo: {modelo.get('tipo', 'N/A')}")
        print(f"   Modelo celda: {modelo.get('celda', 'N/A')}")
        print(f"   Ecuación: {modelo.get('ecuacion_string', 'N/A')}")
        print(f"   Coeficientes: {modelo.get('coeficientes', [])}")
        print(f"   Predictores: {modelo.get('predictores', [])}")
        
        # Stack trace completo
        import traceback
        print(f"   Stack trace completo:")
        traceback.print_exc()
        
        logger.warning(f"Error creando superficie del modelo: {e}")
        return None


def _add_imputation_points_3d(
    fig: go.Figure,
    detalles_por_celda: Dict,
    celda_key: str,
    modelos_2pred: List[Dict],
    selected_methods: List[str],
    pred1_range: Tuple[float, float],
    pred2_range: Tuple[float, float],
    predictor1_name: str,
    predictor2_name: str,
    parametro: str
) -> None:
    """
    Agrega puntos de imputación 3D a la figura.
    
    Parameters:
    -----------
    fig : go.Figure
        Figura a la que agregar los puntos
    detalles_por_celda : Dict
        Detalles de imputación por celda
    celda_key : str
        Clave de la celda actual
    modelos_2pred : List[Dict]
        Lista de modelos con 2 predictores
    selected_methods : List[str]
        Métodos de imputación seleccionados
    pred1_range : Tuple[float, float]
        Rango del primer predictor
    pred2_range : Tuple[float, float]
        Rango del segundo predictor
    predictor1_name : str
        Nombre del primer predictor
    predictor2_name : str
        Nombre del segundo predictor
    parametro : str
        Nombre del parámetro objetivo
    """
    detalles = detalles_por_celda.get(celda_key, {})
    
    # Función de normalización
    def normalize_value(value, value_range):
        min_val, max_val = value_range
        if max_val == min_val:
            return 0.5
        return (value - min_val) / (max_val - min_val)
    
    # Colores para métodos de imputación
    method_colors = {
        'final': 'red',
        'similitud': 'orange', 
        'correlacion': 'purple'
    }
    
    method_symbols = {
        'final': 'diamond',
        'similitud': 'square',
        'correlacion': 'cross'
    }
    
    for method in selected_methods:
        if method in detalles:
            method_data = detalles[method]
            
            # Verificar que tenemos datos para ambos predictores
            if predictor1_name in method_data and predictor2_name in method_data:
                pred1_val = method_data[predictor1_name]
                pred2_val = method_data[predictor2_name]
                target_val = method_data.get(parametro)
                
                if target_val is not None:
                    # Normalizar coordenadas
                    pred1_norm = normalize_value(pred1_val, pred1_range)
                    pred2_norm = normalize_value(pred2_val, pred2_range)
                    
                    fig.add_trace(go.Scatter3d(
                        x=[pred1_norm],
                        y=[pred2_norm],
                        z=[target_val],
                        mode='markers',
                        marker=dict(
                            size=10,
                            color=method_colors.get(method, 'gray'),
                            symbol=method_symbols.get(method, 'circle'),
                            opacity=0.9,
                            line=dict(width=2, color='black')
                        ),
                        name=f'Imputación ({method})',
                        hovertemplate=f'<b>Imputación - {method}</b><br>' +
                                    f'{predictor1_name}: {pred1_val:.3f}<br>' +
                                    f'{predictor2_name}: {pred2_val:.3f}<br>' +
                                    f'{parametro}: {target_val:.3f}<extra></extra>'
                    ))


def get_models_with_2_predictors(modelos_por_celda: Dict, aeronave: str, parametro: str) -> List[Dict]:
    """
    Obtiene modelos que tienen exactamente 2 predictores para una combinación aeronave-parámetro.
    
    Parameters:
    -----------
    modelos_por_celda : Dict
        Diccionario de modelos por celda
    aeronave : str
        Nombre de la aeronave
    parametro : str
        Nombre del parámetro
        
    Returns:
    --------
    List[Dict]
        Lista de modelos con 2 predictores
    """
    celda_key = f"{aeronave}|{parametro}"
    modelos = modelos_por_celda.get(celda_key, [])
    return [m for m in modelos if isinstance(m, dict) and m.get('n_predictores', 0) == 2]


def has_3d_models(modelos_por_celda: Dict, aeronave: str, parametro: str) -> bool:
    """
    Verifica si hay modelos con 2 predictores disponibles para mostrar en 3D.
    
    Parameters:
    -----------
    modelos_por_celda : Dict
        Diccionario de modelos por celda
    aeronave : str
        Nombre de la aeronave
    parametro : str
        Nombre del parámetro
        
    Returns:
    --------
    bool
        True si hay modelos 3D disponibles
    """
    return len(get_models_with_2_predictors(modelos_por_celda, aeronave, parametro)) > 0
