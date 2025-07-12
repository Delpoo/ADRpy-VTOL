"""
plot_model_curves.py

Funciones para curvas de modelos usando el motor de normalización unificado.
"""

from typing import Dict, List, Optional, Tuple
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import logging
try:
    from .plot_config import COLORS, SYMBOLS, _ensure_list
    from .normalization_engine import normalization_engine, get_normalized_model_data
    from .json_data_helpers import get_full_dataframe_from_celda
except ImportError:
    from plot_config import COLORS, SYMBOLS, _ensure_list
    from normalization_engine import normalization_engine, get_normalized_model_data
    from json_data_helpers import get_full_dataframe_from_celda

logger = logging.getLogger(__name__)

# Funciones auxiliares para normalización (lógica original restaurada)
def _confianza_promedio(modelo):
    """
    Calcula la confianza promedio de un modelo (entrenamiento + validación).
    Usa 'Confianza' y 'Confianza_LOOCV'.
    Si solo tiene 'Confianza', usa ese valor.
    Si ambos faltan, retorna -1.
    """
    conf_train = modelo.get('Confianza')
    conf_val = modelo.get('Confianza_LOOCV')
    if conf_val is not None and conf_train is not None:
        return (conf_train + conf_val) / 2
    elif conf_train is not None:
        return conf_train
    elif conf_val is not None:
        return conf_val
    return -1

def normalize_imputation_point(point, rango_x, rango_y, rango_z, n_predictores=1):
    """
    Normaliza un punto de imputación según la dimensión del gráfico (2D o 3D).
    
    Parameters:
    -----------
    point : Dict
        Punto con coordenadas originales. Soporta múltiples formatos:
        - 'x_original' o 'variable_independiente_1' para X
        - 'variable_independiente_2' o 'y_original' para Y
        - 'valor_imputado' o 'variable_dependiente' para Z
    rango_x, rango_y, rango_z : List[float]
        Rangos [min, max] para cada dimensión
    n_predictores : int
        1 para gráficos 2D, 2 para gráficos 3D
    
    Returns:
    --------
    Tuple:
        - Para 2D: (x_norm, y_original, None)
        - Para 3D: (x_norm, y_norm, z_original)
    """
    # 🔍 DEBUG COMPLETO DEL PUNTO
    if DEBUG_NORMALIZATION:
        print(f"\n🔎 [DEBUG COMPLETO] normalize_imputation_point llamada:")
        print(f"   n_predictores: {n_predictores}")
        print(f"   point completo: {point}")
        print(f"   rangos recibidos: X={rango_x}, Y={rango_y}, Z={rango_z}")
    
    # 🔧 BÚSQUEDA FLEXIBLE DE VALORES X, Y, Z
    # Buscar valor de X: múltiples nombres posibles
    x_orig = None
    x_fields = ['variable_independiente_1', 'x_original', 'X', 'x']
    for field in x_fields:
        if field in point and point[field] is not None:
            x_orig = point[field]
            if DEBUG_NORMALIZATION:
                print(f"   🔍 X: encontrado en campo '{field}': {x_orig}")
            break
    
    if x_orig is None:
        x_orig = 0
        if DEBUG_NORMALIZATION:
            print(f"   ⚠️ X: no encontrado en ningún campo, usando 0")
    
    # Buscar valor de Y (solo para 3D): múltiples nombres posibles
    y_orig = None
    if n_predictores == 2:  # Solo buscar Y para gráficos 3D
        y_fields = ['variable_independiente_2', 'y_original', 'Y', 'y']
        for field in y_fields:
            if field in point and point[field] is not None:
                y_orig = point[field]
                if DEBUG_NORMALIZATION:
                    print(f"   🔍 Y: encontrado en campo '{field}': {y_orig}")
                break
        
        if y_orig is None and DEBUG_NORMALIZATION:
            print(f"   ⚠️ Y: no encontrado en ningún campo para gráfico 3D")
    
    # Buscar valor de Z: múltiples nombres posibles
    z_orig = None
    z_fields = ['variable_dependiente', 'valor_imputado', 'Valor imputado', 'z_original', 'Z', 'z']
    for field in z_fields:
        if field in point and point[field] is not None:
            z_orig = point[field]
            if DEBUG_NORMALIZATION:
                print(f"   🔍 Z: encontrado en campo '{field}': {z_orig}")
            break
    
    if z_orig is None:
        z_orig = 0
        if DEBUG_NORMALIZATION:
            print(f"   ⚠️ Z: no encontrado en ningún campo, usando 0")

    # Normalizar X (siempre normalizada como variable independiente)
    print(f"\n🔎 [NORMALIZACIÓN X] Normalizando variable_independiente_1 (X):")
    print(f"   X valor usado: {x_orig}")
    print(f"   rango_x: {rango_x}")
    
    # 🔍 VALIDACIÓN CRÍTICA: Verificar coherencia del rango X
    if DEBUG_NORMALIZATION:
        print(f"   🔎 VALIDACIÓN X: ¿{x_orig} está en rango [{rango_x[0]}, {rango_x[1]}]?")
        if x_orig < rango_x[0] or x_orig > rango_x[1]:
            print(f"   ❌ PROBLEMA DETECTADO: X={x_orig} está FUERA del rango {rango_x}")
            print(f"   🔧 Esto causará normalización fuera de [0,1]")
        else:
            print(f"   ✅ OK: X={x_orig} está dentro del rango {rango_x}")
    
    # 🔧 CORRECCIÓN AUTOMÁTICA: Si el valor está fuera del rango, expandir el rango
    if x_orig < rango_x[0] or x_orig > rango_x[1]:
        rango_x_original = rango_x.copy()
        rango_x = [min(rango_x[0], x_orig * 0.95), max(rango_x[1], x_orig * 1.05)]
        if DEBUG_NORMALIZATION:
            print(f"   🔧 CORRECCIÓN APLICADA: Rango X expandido de {rango_x_original} a {rango_x}")
    
    if rango_x[1] != rango_x[0]:
        x_norm = (x_orig - rango_x[0]) / (rango_x[1] - rango_x[0])
        print(f"   x_norm calculado: {x_norm}")
        
        # 🔍 VALIDACIÓN RESULTADO
        if DEBUG_NORMALIZATION:
            if x_norm < 0 or x_norm > 1:
                print(f"   ❌ RESULTADO PROBLEMÁTICO: x_norm={x_norm} está fuera de [0,1]")
            else:
                print(f"   ✅ RESULTADO OK: x_norm={x_norm} está en [0,1]")
    else:
        x_norm = 0.5
        print("   ⚠️ Rango X constante, x_norm=0.5")

    if n_predictores == 1:
        # 2D: Y es variable dependiente, NO normalizar
        print(f"\n🔎 [2D] Y es variable dependiente - NO normalizar:")
        print(f"   Y valor original: {z_orig}")
        return x_norm, z_orig, None
    else:
        # 3D: Y es variable independiente, normalizar (y_orig ya fue buscado arriba)
        print(f"\n🔎 [3D] Y es variable independiente - normalizar:")
        print(f"   Y valor usado: {y_orig}")
        print(f"   rango_y: {rango_y}")
        
        # 🔍 VALIDACIÓN CRÍTICA: Verificar coherencia del rango Y
        if DEBUG_NORMALIZATION and y_orig is not None:
            print(f"   🔎 VALIDACIÓN Y: ¿{y_orig} está en rango [{rango_y[0]}, {rango_y[1]}]?")
            if y_orig < rango_y[0] or y_orig > rango_y[1]:
                print(f"   ❌ PROBLEMA DETECTADO: Y={y_orig} está FUERA del rango {rango_y}")
                print(f"   🔧 Esto causará normalización fuera de [0,1]")
            else:
                print(f"   ✅ OK: Y={y_orig} está dentro del rango {rango_y}")
        
        # 🔧 CORRECCIÓN AUTOMÁTICA: Si el valor está fuera del rango, expandir el rango
        if y_orig is not None and (y_orig < rango_y[0] or y_orig > rango_y[1]):
            rango_y_original = rango_y.copy()
            rango_y = [min(rango_y[0], y_orig * 0.95), max(rango_y[1], y_orig * 1.05)]
            if DEBUG_NORMALIZATION:
                print(f"   🔧 CORRECCIÓN APLICADA: Rango Y expandido de {rango_y_original} a {rango_y}")
        
        if y_orig is not None and rango_y[1] != rango_y[0]:
            y_norm = (y_orig - rango_y[0]) / (rango_y[1] - rango_y[0])
            print(f"   y_norm calculado: {y_norm}")
            
            # 🔍 VALIDACIÓN RESULTADO
            if DEBUG_NORMALIZATION:
                if y_norm < 0 or y_norm > 1:
                    print(f"   ❌ RESULTADO PROBLEMÁTICO: y_norm={y_norm} está fuera de [0,1]")
                else:
                    print(f"   ✅ RESULTADO OK: y_norm={y_norm} está en [0,1]")
        else:
            y_norm = 0.5
            print("   ⚠️ Rango Y constante o y_orig None, y_norm=0.5")
        
        print(f"\n🔎 [3D] Z es variable dependiente - NO normalizar:")
        print(f"   Z valor original: {z_orig}")
        
        return x_norm, y_norm, z_orig

# Control de debug global
DEBUG_NORMALIZATION = True  # ✅ ACTIVADO para debug detallado

def get_best_model_ranges(modelos_por_celda, celda_key, n_predictores_filter=None):
    """
    Obtiene los rangos de normalización del mejor modelo de una celda.
    LÓGICA ORIGINAL RESTAURADA CON FIX: Selecciona el mejor modelo por confianza promedio,
    y extrae los rangos de X_original y y_original de los datos de entrenamiento.
    
    FIXED: Maneja correctamente cuando se pasa una lista directamente como valor de celda.
    NUEVO: Filtra modelos por número de predictores para evitar confusión entre 2D/3D.
    
    Parameters:
    -----------
    modelos_por_celda : Dict
        Diccionario completo de modelos
    celda_key : str
        Clave de la celda
    n_predictores_filter : Optional[int]
        Filtro por número de predictores (1 para 2D, 2 para 3D, None para todos)
    
    Returns:
    --------
    Tuple[List, List, List, List]
        Rangos X, Y, Z para normalización y lista de nombres de predictores
    """
    # Rangos por defecto
    rango_x = [0, 1]
    rango_y = [0, 1]  
    rango_z = [0, 1]
    predictor_names = ["Predictor 1", "Predictor 2"]  # Nombres por defecto
    

    if DEBUG_NORMALIZATION:
        print(f"[DEBUG] get_best_model_ranges: celda_key buscada: '{celda_key}'")
        print(f"[DEBUG] n_predictores_filter aplicado: {n_predictores_filter}")
        print(f"[DEBUG] Claves disponibles en modelos_por_celda: {list(modelos_por_celda.keys())[:10]}")
    
    if not modelos_por_celda or celda_key not in modelos_por_celda:
        if DEBUG_NORMALIZATION:
            print(f"🔴 DEBUG: No hay modelos para celda {celda_key}")
        return rango_x, rango_y, rango_z, predictor_names

    data_celda = modelos_por_celda[celda_key]
    if DEBUG_NORMALIZATION:
        print(f"[DEBUG] type(data_celda): {type(data_celda)}")
        print(f"[DEBUG] data_celda (repr, truncado): {repr(data_celda)[:500]}")
    
    # 🔧 FIX: Detectar y advertir sobre estructura incorrecta
    if isinstance(data_celda, list):
        if DEBUG_NORMALIZATION:
            print(f"⚠️  [WARNING] Se detectó una lista como valor de celda. Esto puede causar problemas de normalización.")
            print(f"⚠️  [WARNING] Recomendación: Usar la estructura completa con 'informacion_modelos_celda'")
            print(f"[DEBUG] data_celda es una lista con {len(data_celda)} elementos")
        # Si data_celda es directamente una lista de modelos, úsala directamente
        modelos = data_celda
        if DEBUG_NORMALIZATION:
            print(f"🟢 DEBUG: Usando lista directamente como modelos, encontrados {len(modelos)} modelos para celda {celda_key}")
            for i, elem in enumerate(modelos[:2]):
                print(f"  [DEBUG] modelos[{i}] type: {type(elem)} | keys: {list(elem.keys()) if isinstance(elem, dict) else 'N/A'}")
                if isinstance(elem, dict) and 'tipo' in elem:
                    print(f"    [DEBUG] modelos[{i}] tipo: {elem.get('tipo')}")
                if isinstance(elem, dict) and 'datos_entrenamiento' in elem:
                    print(f"    [DEBUG] modelos[{i}] tiene datos_entrenamiento: {bool(elem.get('datos_entrenamiento'))}")
                if isinstance(elem, dict) and 'Confianza' in elem:
                    print(f"    [DEBUG] modelos[{i}] Confianza: {elem.get('Confianza')}")
                if isinstance(elem, dict) and 'Confianza_LOOCV' in elem:
                    print(f"    [DEBUG] modelos[{i}] Confianza_LOOCV: {elem.get('Confianza_LOOCV')}")
    elif isinstance(data_celda, dict):
        if DEBUG_NORMALIZATION:
            print(f"[DEBUG] data_celda keys: {list(data_celda.keys())}")
        if 'informacion_modelos_celda' not in data_celda:
            # Verificar si data_celda es directamente un modelo individual
            if 'datos_entrenamiento' in data_celda and 'tipo' in data_celda:
                if DEBUG_NORMALIZATION:
                    print(f"🔵 DEBUG: data_celda es un modelo individual, procesando directamente")
                modelos = [data_celda]  # Tratar el diccionario como un modelo único
            else:
                if DEBUG_NORMALIZATION:
                    print(f"🔴 DEBUG: No hay informacion_modelos_celda para celda {celda_key}")
                    print(f"[DEBUG] data_celda completo: {data_celda}")
                return rango_x, rango_y, rango_z, predictor_names
        else:
            modelos = data_celda['informacion_modelos_celda'].get('modelos', [])
        if DEBUG_NORMALIZATION:
            print(f"🔵 DEBUG: Encontrados {len(modelos)} modelos para celda {celda_key}")
    else:
        if DEBUG_NORMALIZATION:
            print(f"[DEBUG] data_celda es de tipo inesperado: {type(data_celda)}")
            print(f"[DEBUG] data_celda valor: {repr(data_celda)[:500]}")
        return rango_x, rango_y, rango_z, predictor_names
      # Buscar mejor modelo por confianza promedio (lógica original restaurada)
    mejor_modelo = None
    mejor_confianza = -1
    modelo_2_predictores = None  # Mejor modelo de 2 predictores
    mejor_confianza_2pred = -1
    
    for i, modelo in enumerate(modelos):
        if isinstance(modelo, dict):
            confianza = _confianza_promedio(modelo)
            n_pred = modelo.get('n_predictores', 1)
            
            if DEBUG_NORMALIZATION:
                print(f"🔵 DEBUG: Modelo {i+1} - Confianza: {confianza:.3f}, n_predictores: {n_pred}")
            
            # 🔧 APLICAR FILTRO: Solo considerar modelos que coincidan con el filtro
            if n_predictores_filter is not None and n_pred != n_predictores_filter:
                if DEBUG_NORMALIZATION:
                    print(f"   ⚪ Modelo {i+1} EXCLUIDO por filtro (n_pred={n_pred} != filter={n_predictores_filter})")
                continue
            
            # Buscar mejor modelo (ya filtrado)
            if confianza > mejor_confianza:
                mejor_confianza = confianza
                mejor_modelo = modelo
                if DEBUG_NORMALIZATION:
                    print(f"🟢 DEBUG: Nuevo mejor modelo (confianza: {confianza:.3f}, n_pred: {n_pred})")
            
            # 🔧 FIX: Buscar específicamente el mejor modelo de 2 predictores
            if n_pred == 2 and confianza > mejor_confianza_2pred:
                mejor_confianza_2pred = confianza
                modelo_2_predictores = modelo
                if DEBUG_NORMALIZATION:
                    print(f"� FIX: Nuevo mejor modelo de 2 predictores (confianza: {confianza:.3f})")
    
    if not mejor_modelo:
        if DEBUG_NORMALIZATION:
            print(f"🔴 DEBUG: No se encontró mejor modelo para celda {celda_key}")
        return rango_x, rango_y, rango_z, predictor_names

    if DEBUG_NORMALIZATION:
        print(f"🟢 DEBUG: Mejor modelo seleccionado con confianza: {mejor_confianza:.3f}")
        if modelo_2_predictores:
            print(f"🔧 DEBUG: Mejor modelo de 2 predictores con confianza: {mejor_confianza_2pred:.3f}")
    

    
    # Extraer rangos de los datos de entrenamiento del mejor modelo filtrado
    datos_ent = mejor_modelo.get('datos_entrenamiento', {})
    X_original = datos_ent.get('X_original', [])
    y_original = datos_ent.get('y_original', [])
    
    if not X_original or not y_original:
        if DEBUG_NORMALIZATION:
            print(f"🔴 DEBUG: No hay datos de entrenamiento originales")
            print(f"   X_original: {bool(X_original)}, y_original: {bool(y_original)}")
        return rango_x, rango_y, rango_z, predictor_names
    
    # Convertir a arrays numpy para facilidad de cálculo
    import numpy as np
    X_array = np.array(X_original)
    y_array = np.array(y_original)
    
    if DEBUG_NORMALIZATION:
        print(f"🔍 DEBUG: X_array.shape = {X_array.shape}")
        print(f"🔍 DEBUG: X_array.ndim = {X_array.ndim}")
    
    # Calcular rango X
    if X_array.ndim == 1:
        # Modelo de 1 predictor (array 1D)
        x_min, x_max = X_array.min(), X_array.max()
        rango_x = [float(x_min), float(x_max)]
        if DEBUG_NORMALIZATION:
            print(f"🟢 DEBUG: Rango X (1 predictor): [{x_min:.3f}, {x_max:.3f}]")
    elif X_array.ndim == 2:
        # Modelo de 2 predictores (array 2D) - tomar primera columna para X
        x_min, x_max = X_array[:, 0].min(), X_array[:, 0].max()
        rango_x = [float(x_min), float(x_max)]
        if DEBUG_NORMALIZATION:
            print(f"🟢 DEBUG: Rango X (desde 2 predictores): [{x_min:.3f}, {x_max:.3f}]")
        
        # Calcular rango Y solo para modelos de 2 predictores
        if X_array.shape[1] >= 2:
            y_min, y_max = X_array[:, 1].min(), X_array[:, 1].max()
            rango_y = [float(y_min), float(y_max)]
            if DEBUG_NORMALIZATION:
                print(f"🟢 DEBUG: Rango Y (2 predictores): [{y_min:.3f}, {y_max:.3f}]")
        else:
            if DEBUG_NORMALIZATION:
                print(f"⚠️ DEBUG: Modelo de 2 predictores no tiene datos 2D válidos")
    
    # Rango Z del mejor modelo
    z_min, z_max = y_array.min(), y_array.max()
    rango_z = [float(z_min), float(z_max)]
    if DEBUG_NORMALIZATION:
        print(f"🟢 DEBUG: Rango Z (objetivo): [{z_min:.3f}, {z_max:.3f}]")
    
    # 🔧 OBTENER NOMBRES DE PREDICTORES
    modelo_para_nombres = mejor_modelo
    
    try:
        # Intentar obtener nombres de predictores del modelo
        if modelo_para_nombres.get('predictores'):
            predictor_names = modelo_para_nombres['predictores']
            if DEBUG_NORMALIZATION:
                print(f"🟢 DEBUG: Nombres de predictores encontrados: {predictor_names}")
        elif modelo_para_nombres.get('features'):
            predictor_names = modelo_para_nombres['features']
            if DEBUG_NORMALIZATION:
                print(f"🟢 DEBUG: Nombres de features encontrados: {predictor_names}")
        else:
            # Usar nombres por defecto basados en el número de predictores
            n_predictores = modelo_para_nombres.get('n_predictores', 1)
            if n_predictores == 1:
                predictor_names = ["Variable Independiente"]
            else:
                predictor_names = ["Variable Independiente 1", "Variable Independiente 2"]
            if DEBUG_NORMALIZATION:
                print(f"🟢 DEBUG: Usando nombres por defecto: {predictor_names}")
    except Exception as e:
        if DEBUG_NORMALIZATION:
            print(f"🔴 DEBUG: Error obteniendo nombres de predictores: {e}")
        predictor_names = ["Variable Independiente 1", "Variable Independiente 2"]
    # 🔧 VALIDACIÓN FINAL: Asegurar que todos los rangos sean válidos
    if rango_x[0] == rango_x[1]:
        rango_x = [rango_x[0] - 0.1, rango_x[1] + 0.1]
    if rango_y[0] == rango_y[1]:
        rango_y = [rango_y[0] - 0.1, rango_y[1] + 0.1]
    if rango_z[0] == rango_z[1]:
        rango_z = [rango_z[0] - 0.1, rango_z[1] + 0.1]
    
    if DEBUG_NORMALIZATION:
        print(f"🔄 RANGOS FINALES:")
        print(f"   - X: {rango_x}")
        print(f"   - Y: {rango_y}")
        print(f"   - Z: {rango_z}")
        print(f"   - Predictores: {predictor_names}")
    
    return rango_x, rango_y, rango_z, predictor_names


def add_normalized_model_curves(fig: go.Figure, 
                               modelos: List[Dict], 
                               parametro: str,
                               show_synthetic_curves: bool = True,
                               show_only_real_curves: bool = False,
                               highlight_model_idx: Optional[int] = None) -> None:
    """
    Añade curvas de modelos normalizadas al gráfico usando el motor de normalización unificado.
    Solo procesa modelos de 1 predictor para gráficos 2D.
    
    Parameters:
    -----------
    fig : go.Figure
        Figura de Plotly donde añadir las curvas
    modelos : List[Dict]
        Lista de modelos (solo se procesan los de 1 predictor)
    parametro : str
        Nombre del parámetro objetivo
    show_synthetic_curves : bool
        Si mostrar curvas generadas con rangos sintéticos (líneas punteadas)
    show_only_real_curves : bool
        Si mostrar solo curvas con datos reales (omitir sintéticas)
    highlight_model_idx : int
        Índice del modelo a resaltar (opacidad y grosor de línea)
    """
    
    curves_added = 0
    warnings_added = []
    synthetic_ranges_used = 0
    omitted_synthetic = 0
    
    for i, modelo in enumerate(modelos):
        try:
            # Filtrar solo modelos de 1 predictor
            if not isinstance(modelo, dict) or modelo.get('n_predictores', 0) != 1:
                continue
                
            predictor = modelo.get('predictores', [None])[0]
            if not predictor:
                continue
            
            # Usar el motor de normalización para obtener datos
            vis_data = get_normalized_model_data(modelo, curve_resolution=100)
            
            if not vis_data.get('modelo_valido', False):
                warnings_added.append(f"Modelo {i+1} inválido: {vis_data.get('error', 'Error desconocido')}")
                continue
            
            # Obtener datos de la curva
            curva_data = vis_data.get('curva', {})
            x_normalized = curva_data.get('x_normalized')
            y_normalized = curva_data.get('y_normalized')
            curve_metadata = curva_data.get('metadata', {})
            
            if x_normalized is None or y_normalized is None:
                warnings_added.append(f"Modelo {i+1}: No se pudo generar curva")
                continue
            
            # Verificar si es rango sintético
            is_synthetic = curve_metadata.get('synthetic_range', False)
            
            if is_synthetic:
                synthetic_ranges_used += 1
                
            # Aplicar filtros de visualización
            if is_synthetic and not show_synthetic_curves:
                omitted_synthetic += 1
                continue
                
            if show_only_real_curves and is_synthetic:
                omitted_synthetic += 1
                continue
            
            # Configurar estilo de línea
            line_style = 'dash' if is_synthetic else 'solid'
            
            # Configurar colores y opacidad
            color = COLORS['model_lines'][i % len(COLORS['model_lines'])]
            opacity = 1.0
            line_width = 2
            
            if highlight_model_idx is not None:
                if i == highlight_model_idx:
                    opacity = 1.0
                    line_width = 3
                else:
                    opacity = 0.3
                    line_width = 1
            
            # Crear información de hover
            tipo_modelo = vis_data.get('tipo', 'unknown')
            ecuacion = vis_data.get('ecuacion_string', 'N/A')
            r2 = vis_data.get('metrica_r2')
            mape = vis_data.get('metrica_mape')
            confianza = vis_data.get('confianza')
            
            hover_parts = [
                f"<b>Modelo {i+1}:</b> {tipo_modelo}",
                f"<b>Predictor:</b> {predictor}",
                f"<b>Ecuación:</b> {ecuacion}"
            ]
            
            if r2 is not None:
                hover_parts.append(f"<b>R²:</b> {r2:.3f}")
            if mape is not None:
                hover_parts.append(f"<b>MAPE:</b> {mape:.1f}%")
            if confianza is not None:
                hover_parts.append(f"<b>Confianza:</b> {confianza:.3f}")
                
            if is_synthetic:
                hover_parts.append("<b>⚠️ Rango sintético</b>")
            
            hovertemplate = "<br>".join(hover_parts) + "<extra></extra>"
            
            # Añadir curva al gráfico
            fig.add_trace(go.Scatter(
                x=x_normalized,
                y=y_normalized,
                mode='lines',
                name=f"{tipo_modelo} - {predictor}" + (" (sintético)" if is_synthetic else ""),
                line=dict(
                    color=color,
                    width=line_width,
                    dash=line_style
                ),
                opacity=opacity,
                hovertemplate=hovertemplate,
                legendgroup=f"modelo_{i}",
                showlegend=True
            ))
            
            curves_added += 1
            
        except Exception as e:
            logger.error(f"Error procesando modelo {i}: {e}")
            warnings_added.append(f"Error en modelo {i+1}: {str(e)}")
    
    # Añadir anotaciones informativas si es necesario
    if warnings_added:
        warning_text = "⚠️ Advertencias: " + "; ".join(warnings_added[:3])
        if len(warnings_added) > 3:
            warning_text += f" (y {len(warnings_added)-3} más)"
        
        fig.add_annotation(
            text=warning_text,
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            font=dict(size=10, color="orange"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="orange",
            borderwidth=1
        )
    
    if synthetic_ranges_used > 0 and show_synthetic_curves:
        fig.add_annotation(
            text=f"📊 {synthetic_ranges_used} curva(s) con rango sintético (líneas punteadas)",
            xref="paper", yref="paper",
            x=0.02, y=0.02,
            showarrow=False,
            font=dict(size=9, color="blue"),
            bgcolor="rgba(255,255,255,0.8)"
        )
    
    if omitted_synthetic > 0:
        fig.add_annotation(
            text=f"🚫 {omitted_synthetic} curva(s) sintética(s) oculta(s)",
            xref="paper", yref="paper",
            x=0.98, y=0.02,
            showarrow=False,
            font=dict(size=9, color="gray"),
            bgcolor="rgba(255,255,255,0.8)",
            xanchor="right"
        )
    
    logger.info(f"Añadidas {curves_added} curvas normalizadas para parámetro {parametro}")
    logger.info(f"Estadísticas: {synthetic_ranges_used} sintéticas, {omitted_synthetic} omitidas")


def filter_single_predictor_models(modelos: List[Dict]) -> List[Dict]:
    """
    Filtra modelos para retornar solo aquellos con 1 predictor (para gráficos 2D).
    
    Parameters:
    -----------
    modelos : List[Dict]
        Lista de modelos
        
    Returns:
    --------
    List[Dict]
        Lista de modelos con 1 predictor únicamente
    """
    filtered_models = []
    
    for modelo in modelos:
        n_predictores = modelo.get('n_predictores', 0)
        if n_predictores == 1:
            filtered_models.append(modelo)
    
    logger.info(f"Filtrados {len(filtered_models)} modelos de 1 predictor de {len(modelos)} totales")
    return filtered_models





def create_model_hover_info(modelo: Dict) -> str:
    """
    Crea información de hover para un modelo.
    
    Parameters:
    -----------
    modelo : Dict
        Diccionario con información del modelo
        
    Returns:
    --------
    str
        String formateado para hover
    """
    try:
        tipo = modelo.get('tipo', 'unknown')
        ecuacion = modelo.get('ecuacion_string', 'N/A')
        r2 = modelo.get('r2')
        mape = modelo.get('mape')
        confianza = modelo.get('Confianza')
        
        hover_parts = [
            f"<b>Tipo:</b> {tipo}",
            f"<b>Ecuación:</b> {ecuacion}"
        ]
        
        if r2 is not None:
            hover_parts.append(f"<b>R²:</b> {r2:.3f}")
        if mape is not None:
            hover_parts.append(f"<b>MAPE:</b> {mape:.1f}%")
        if confianza is not None:
            hover_parts.append(f"<b>Confianza:</b> {confianza:.3f}")
        
        return "<br>".join(hover_parts)
        
    except Exception as e:
        logger.error(f"Error creando hover info: {e}")
        return "Error en información del modelo"





def extract_theoretical_imputation_points(modelos: List[Dict], 
                                         celda_key: str,
                                         n_predictores_filter: Optional[int] = None,
                                         modelos_por_celda: Optional[Dict] = None) -> List[Dict]:
    """
    Extrae puntos de imputación teóricos calculados usando los valores de 
    variable_independiente_1 y variable_independiente_2 de cada modelo
    y la ecuación del modelo. Usa los mismos rangos que las curvas para coherencia.
    
    Parameters:
    -----------
    modelos : List[Dict]
        Lista de modelos (1 o 2 predictores)
    celda_key : str
        Clave de la celda para logging
    n_predictores_filter : Optional[int]
        Si se especifica, filtra solo modelos con este número de predictores
    modelos_por_celda : Optional[Dict]
        Diccionario completo de modelos por celda (no se usa, se mantiene por compatibilidad)
        
    Returns:
    --------
    List[Dict]
        Lista de puntos teóricos con sus coordenadas normalizadas usando rangos del modelo
    """
    logger = logging.getLogger(__name__)
    theoretical_points = []
    
    for i, modelo in enumerate(modelos):
        try:
            # Obtener valores de las variables independientes
            var_indep_1 = modelo.get('variable_independiente_1')
            var_indep_2 = modelo.get('variable_independiente_2')
            n_predictores = modelo.get('n_predictores', 0)
            
            # Aplicar filtro de número de predictores si se especifica
            if n_predictores_filter is not None and n_predictores != n_predictores_filter:
                continue
                
            # Variable independiente 1 siempre debe existir
            if var_indep_1 is None:
                logger.warning(f"Modelo {i+1}: variable_independiente_1 es None")
                continue
                
            # 🔧 VALIDACIÓN DE RANGOS DE VARIABLES INDEPENDIENTES
            # Verificar si las variables independientes están dentro de rangos razonables
            rangos_x_temp, _ = normalization_engine.get_model_data_ranges(modelo)
            if rangos_x_temp and len(rangos_x_temp) > 0:
                x1_min, x1_max = rangos_x_temp[0]
                x1_span = x1_max - x1_min
                extrapolacion_factor = 3  # Factor más conservador para variables independientes
                
                x1_limite_inf = x1_min - extrapolacion_factor * x1_span
                x1_limite_sup = x1_max + extrapolacion_factor * x1_span
                
                if var_indep_1 < x1_limite_inf or var_indep_1 > x1_limite_sup:
                    logger.warning(f"Modelo {i+1}: variable_independiente_1 ({var_indep_1:.3f}) muy fuera del rango de entrenamiento [{x1_min:.3f}, {x1_max:.3f}]")
                
                # Verificar segunda variable si existe
                if var_indep_2 is not None and len(rangos_x_temp) > 1:
                    x2_min, x2_max = rangos_x_temp[1]
                    x2_span = x2_max - x2_min
                    
                    x2_limite_inf = x2_min - extrapolacion_factor * x2_span
                    x2_limite_sup = x2_max + extrapolacion_factor * x2_span
                    
                    if var_indep_2 < x2_limite_inf or var_indep_2 > x2_limite_sup:
                        logger.warning(f"Modelo {i+1}: variable_independiente_2 ({var_indep_2:.3f}) muy fuera del rango de entrenamiento [{x2_min:.3f}, {x2_max:.3f}]")
                
            # Obtener coeficientes y intercepto del modelo
            coeficientes = modelo.get('coeficientes_originales', [])
            intercepto = modelo.get('intercepto_original', 0)
            
            if not coeficientes or len(coeficientes) == 0:
                logger.warning(f"Modelo {i+1}: sin coeficientes válidos")
                continue
                
            # Calcular valor teórico usando la ecuación del modelo según su tipo
            valor_y_teorico = None
            tipo_modelo = modelo.get('tipo', '').lower()
            
            if n_predictores == 1:
                # MODELOS DE 1 PREDICTOR
                if tipo_modelo == 'linear-1':
                    # Modelo lineal: y = intercepto + coef[0] * x1
                    valor_y_teorico = intercepto + coeficientes[0] * var_indep_1
                    
                elif tipo_modelo == 'log-1':
                    # Modelo logarítmico: y = intercepto + coef[0] * log(x1)
                    if var_indep_1 > 0:  # log requiere valores positivos
                        import math
                        valor_y_teorico = intercepto + coeficientes[0] * math.log(var_indep_1)
                    else:
                        logger.warning(f"Modelo {i+1}: log-1 requiere variable_independiente_1 > 0, obtenido: {var_indep_1}")
                        continue
                        
                elif tipo_modelo == 'pot-1':
                    # Modelo potencial: y = intercepto * x1^coef[0]
                    if var_indep_1 > 0:  # potencia requiere valores positivos para exponentes fraccionarios
                        valor_y_teorico = intercepto * (var_indep_1 ** coeficientes[0])
                    else:
                        logger.warning(f"Modelo {i+1}: pot-1 requiere variable_independiente_1 > 0, obtenido: {var_indep_1}")
                        continue
                        
                elif tipo_modelo == 'exp-1':
                    # Modelo exponencial: y = intercepto * exp(coef[0] * x1)
                    import math
                    try:
                        valor_y_teorico = intercepto * math.exp(coeficientes[0] * var_indep_1)
                    except OverflowError:
                        logger.warning(f"Modelo {i+1}: exp-1 overflow con variable_independiente_1: {var_indep_1}")
                        continue
                        
                elif tipo_modelo == 'poly-1':
                    # Modelo polinómico: y = intercepto + coef[0]*x + coef[1]*x^2 + ...
                    valor_y_teorico = intercepto
                    for grado, coef in enumerate(coeficientes):
                        valor_y_teorico += coef * (var_indep_1 ** (grado + 1))
                        
                else:
                    logger.warning(f"Modelo {i+1}: tipo de modelo 1D no reconocido: {tipo_modelo}")
                    continue
                
            elif n_predictores == 2:
                # Modelo de 2 predictores
                if var_indep_2 is None:
                    logger.warning(f"Modelo {i+1}: modelo de 2 predictores pero variable_independiente_2 es None")
                    continue
                    
                tipo_modelo = modelo.get('tipo', '')
                
                if tipo_modelo.startswith('linear-2'):
                    # Modelo lineal: y = intercepto + coef[0] * x1 + coef[1] * x2
                    if len(coeficientes) >= 2:
                        valor_y_teorico = intercepto + coeficientes[0] * var_indep_1 + coeficientes[1] * var_indep_2
                    else:
                        logger.warning(f"Modelo {i+1}: modelo linear-2 sin suficientes coeficientes")
                        continue
                        
                elif tipo_modelo.startswith('poly-2'):
                    # Modelo polinómico de 2 predictores: y = intercepto + c1*x1 + c2*x2 + c3*x1² + c4*x1*x2 + c5*x2²
                    # ORDEN CORRECTO según PolynomialFeatures(degree=2, include_bias=False):
                    # Features: ['x1', 'x2', 'x1^2', 'x1 x2', 'x2^2']
                    # Powers:   [[1,0], [0,1], [2,0], [1,1], [0,2]]
                    # Por lo tanto: coeficientes = [c1, c2, c3, c4, c5] donde:
                    # c1 = x1, c2 = x2, c3 = x1², c4 = x1*x2, c5 = x2²
                    if len(coeficientes) >= 5:
                        valor_y_teorico = (intercepto + 
                                         coeficientes[0] * var_indep_1 +           # c1*x1
                                         coeficientes[1] * var_indep_2 +           # c2*x2  
                                         coeficientes[2] * (var_indep_1 ** 2) +    # c3*x1²
                                         coeficientes[3] * var_indep_1 * var_indep_2 +  # c4*x1*x2 (CORRECTO)
                                         coeficientes[4] * (var_indep_2 ** 2))     # c5*x2² (CORRECTO)
                        
                        logger.debug(f"Modelo poly-2 {i+1}: intercepto={intercepto:.6f}")
                        logger.debug(f"  x1={var_indep_1:.6f}, x2={var_indep_2:.6f}")
                        logger.debug(f"  coef[0]*x1={coeficientes[0]:.6f}*{var_indep_1:.6f}={coeficientes[0]*var_indep_1:.6f}")
                        logger.debug(f"  coef[1]*x2={coeficientes[1]:.6f}*{var_indep_2:.6f}={coeficientes[1]*var_indep_2:.6f}")
                        logger.debug(f"  coef[2]*x1²={coeficientes[2]:.6f}*{var_indep_1**2:.6f}={coeficientes[2]*var_indep_1**2:.6f}")
                        logger.debug(f"  coef[3]*x1*x2={coeficientes[3]:.6f}*{var_indep_1*var_indep_2:.6f}={coeficientes[3]*var_indep_1*var_indep_2:.6f}")
                        logger.debug(f"  coef[4]*x2²={coeficientes[4]:.6f}*{var_indep_2**2:.6f}={coeficientes[4]*var_indep_2**2:.6f}")
                        logger.debug(f"  valor_y_teorico={valor_y_teorico:.6f}")
                    else:
                        logger.warning(f"Modelo {i+1}: modelo poly-2 sin suficientes coeficientes (necesita 5, tiene {len(coeficientes)})")
                        continue
                else:
                    logger.warning(f"Modelo {i+1}: tipo de modelo 2D no reconocido: {tipo_modelo}")
                    continue
            else:
                logger.warning(f"Modelo {i+1}: número de predictores no soportado: {n_predictores}")
                continue
                
            if valor_y_teorico is None:
                continue
                
            # 🔧 VALIDACIÓN DE EXTRAPOLACIÓN EXTREMA
            # Verificar si el valor teórico está muy fuera del rango de entrenamiento
            rangos_x, rango_y = normalization_engine.get_model_data_ranges(modelo)
            if rangos_x and rango_y:
                y_min, y_max = rango_y
                rango_y_span = y_max - y_min
                
                # Considerar extrapolación extrema si está más de 10 veces fuera del rango
                extrapolacion_factor = 10
                limite_inferior = y_min - extrapolacion_factor * rango_y_span
                limite_superior = y_max + extrapolacion_factor * rango_y_span
                
                if valor_y_teorico < limite_inferior or valor_y_teorico > limite_superior:
                    logger.warning(f"Modelo {i+1}: valor teórico extremo {valor_y_teorico:.3f} fuera del rango de entrenamiento [{y_min:.3f}, {y_max:.3f}]")
                    logger.warning(f"  Variables: x1={var_indep_1}, x2={var_indep_2 if var_indep_2 is not None else 'N/A'}")
                    logger.warning(f"  Ecuación: {modelo.get('ecuacion_string', 'N/A')}")
                    # Opcionalmente, saltar este punto si es demasiado extremo
                    # continue  # Descomenta esta línea para omitir puntos extremos
                
            # 🔧 OBTENER RANGOS PARA NORMALIZACIÓN (USAR RANGOS DEL MODELO INDIVIDUAL PARA COHERENCIA)
            # IMPORTANTE: Usar los mismos rangos que se usan para las curvas del modelo
            rangos_x, rango_y = normalization_engine.get_model_data_ranges(modelo)
            logger.debug(f"Modelo {i+1}: usando rangos del modelo (coherencia con curvas) - rangos_x={rangos_x}, rango_y={rango_y}")
            
            if not rangos_x or not rango_y:
                logger.warning(f"Modelo {i+1}: no se pudieron obtener rangos para normalización")
                continue
                
            logger.debug(f"Modelo {i+1}: var_indep_1={var_indep_1}, var_indep_2={var_indep_2}, valor_y_teorico={valor_y_teorico}")
            
            # 🔧 NORMALIZAR USANDO LOS MISMOS RANGOS QUE LAS CURVAS DEL MODELO
            try:
                if n_predictores == 1:
                    # MODELO 2D: 1 variable independiente + 1 parámetro objetivo
                    # Usar índice 0 para el primer (y único) predictor
                    x_normalized = normalization_engine.normalize_x_values([var_indep_1], rangos_x, 0)[0]
                    # 🔧 CAMBIO: NO normalizar Y para mostrar valores originales de la variable dependiente
                    y_original = valor_y_teorico  # Mantener valor original de la variable dependiente
                    
                    predictor_name = modelo.get('predictores', [None])[0] or f"predictor_{i+1}"
                    logger.debug(f"Modelo {i+1} 2D: predictor={predictor_name}")
                    logger.debug(f"Modelo {i+1} 2D: x_original={var_indep_1} -> x_norm={x_normalized}")
                    logger.debug(f"Modelo {i+1} 2D: y_original={valor_y_teorico} -> y_original={y_original} (SIN NORMALIZAR)")
                    
                    # Crear punto teórico para gráfico 2D
                    tipo_modelo = modelo.get('tipo', 'unknown')
                    
                    punto_teorico = {
                        'x': x_normalized,
                        'y': y_original,  # 🔧 CAMBIO: usar valor original en lugar de normalizado
                        'x_original': var_indep_1,
                        'y_original': valor_y_teorico,
                        'modelo_idx': i,
                        'predictor': predictor_name,
                        'tipo_modelo': tipo_modelo,
                        'symbol': 'diamond',
                        'size': 8,
                        'color': 'purple',
                        'metodo': 'teorico',
                        'ecuacion': modelo.get('ecuacion_string', 'N/A'),
                        'r2': modelo.get('r2'),
                        'mape': modelo.get('mape'),
                        'confianza': modelo.get('Confianza'),
                        'n_predictores': n_predictores
                    }
                    
                    theoretical_points.append(punto_teorico)
                    logger.debug(f"Punto teórico 2D añadido: Modelo {i+1} -> x={var_indep_1}, y={valor_y_teorico}")
                    
                elif n_predictores == 2:
                    # MODELO 3D: 2 variables independientes + 1 parámetro objetivo
                    # Usar índices 0 y 1 para los dos predictores
                    x1_normalized = normalization_engine.normalize_x_values([var_indep_1], rangos_x, 0)[0]
                    x2_normalized = normalization_engine.normalize_x_values([var_indep_2], rangos_x, 1)[0] if var_indep_2 is not None and len(rangos_x) > 1 else 0
                    z_original = valor_y_teorico  # 🔧 CAMBIO: Mantener valor original de la variable dependiente
                    
                    predictores = modelo.get('predictores', [])
                    predictor_1_name = predictores[0] if len(predictores) > 0 else f"predictor_1_{i+1}"
                    predictor_2_name = predictores[1] if len(predictores) > 1 else f"predictor_2_{i+1}"
                    
                    logger.debug(f"Modelo {i+1} 3D: predictor_1={predictor_1_name}, predictor_2={predictor_2_name}")
                    logger.debug(f"Modelo {i+1} 3D: x1_original={var_indep_1} -> x1_norm={x1_normalized}")
                    logger.debug(f"Modelo {i+1} 3D: x2_original={var_indep_2} -> x2_norm={x2_normalized}")
                    logger.debug(f"Modelo {i+1} 3D: z_original={valor_y_teorico} -> z_original={z_original} (SIN NORMALIZAR)")
                    
                    # Crear punto teórico para gráfico 3D
                    tipo_modelo = modelo.get('tipo', 'unknown')
                    
                    punto_teorico = {
                        'x': x1_normalized,        # Variable independiente 1 normalizada
                        'y': x2_normalized,        # Variable independiente 2 normalizada  
                        'z': z_original,           # 🔧 CAMBIO: Parámetro objetivo en valor original (sin normalizar)
                        'x_original': var_indep_1, # Variable independiente 1 original
                        'y_original': var_indep_2, # Variable independiente 2 original
                        'z_original': valor_y_teorico, # Parámetro objetivo calculado
                        'modelo_idx': i,
                        'predictor_1': predictor_1_name,
                        'predictor_2': predictor_2_name,
                        'tipo_modelo': tipo_modelo,
                        'symbol': 'diamond',
                        'size': 8,
                        'color': 'purple',
                        'metodo': 'teorico',
                        'ecuacion': modelo.get('ecuacion_string', 'N/A'),
                        'r2': modelo.get('r2'),
                        'mape': modelo.get('mape'),
                        'confianza': modelo.get('Confianza'),
                        'n_predictores': n_predictores
                    }
                    
                    theoretical_points.append(punto_teorico)
                    logger.debug(f"Punto teórico 3D añadido: Modelo {i+1} -> x1={var_indep_1}, x2={var_indep_2}, z={valor_y_teorico}")
                    
            except Exception as e:
                logger.warning(f"Modelo {i+1}: error en normalización: {e}")
                continue
            
        except Exception as e:
            logger.error(f"Error calculando punto teórico para modelo {i+1}: {e}")
            continue
    
    logger.info(f"Extraídos {len(theoretical_points)} puntos teóricos para celda {celda_key}")
    return theoretical_points





def add_theoretical_imputation_points_to_plot(fig: go.Figure, 
                                             theoretical_points: List[Dict],
                                             show_theoretical_points: bool = True) -> None:
    """
    Añade puntos de imputación teóricos al gráfico de visualización 2D o 3D.
    Detecta automáticamente si los puntos son 2D o 3D basándose en n_predictores.
    
    Parameters:
    -----------
    fig : go.Figure
        Figura de Plotly donde añadir los puntos
    theoretical_points : List[Dict]
        Lista de puntos teóricos calculados
    show_theoretical_points : bool
        Si mostrar los puntos teóricos
    """
    if not show_theoretical_points or not theoretical_points:
        logger.info(f"Puntos teóricos no mostrados: show_theoretical_points={show_theoretical_points}, len(theoretical_points)={len(theoretical_points) if theoretical_points else 0}")
        return
        
    logger.info(f"Añadiendo {len(theoretical_points)} puntos teóricos al gráfico")
    
    # Separar puntos 2D y 3D
    points_2d = [p for p in theoretical_points if p.get('n_predictores', 1) == 1]
    points_3d = [p for p in theoretical_points if p.get('n_predictores', 1) == 2]
    
    # Añadir puntos 2D (para gráficos 2D)
    if points_2d:
        x_coords = [p['x'] for p in points_2d]  # Variable independiente normalizada
        y_coords = [p['y'] for p in points_2d]  # 🔧 CAMBIO: Parámetro objetivo en escala original
        
        # Crear texto de hover
        hover_texts = []
        for p in points_2d:
            hover_parts = [
                f"<b>Punto Teórico - Modelo {p['modelo_idx']+1}</b>",
                f"<b>Predictor:</b> {p['predictor']}",
                f"<b>Tipo:</b> {p['tipo_modelo']}",
                f"<b>X original:</b> {p['x_original']:.3f}",
                f"<b>Y calculado (escala original):</b> {p['y_original']:.3f}",
                f"<b>Ecuación:</b> {p['ecuacion']}"
            ]
            
            if p.get('r2') is not None:
                hover_parts.append(f"<b>R²:</b> {p['r2']:.3f}")
            if p.get('mape') is not None:
                hover_parts.append(f"<b>MAPE:</b> {p['mape']:.1f}%")
            if p.get('confianza') is not None:
                hover_parts.append(f"<b>Confianza:</b> {p['confianza']:.3f}")
                
            hover_texts.append("<br>".join(hover_parts))
        
        # Añadir al gráfico 2D
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers',
            name='Puntos Teóricos 2D',
            marker=dict(
                symbol='diamond',
                size=12,
                color='purple',
                line=dict(width=2, color='darkviolet')
            ),
            hovertemplate='%{hovertext}<extra></extra>',
            hovertext=hover_texts,
            legendgroup='theoretical',
            showlegend=True
        ))
        
        logger.info(f"Añadidos {len(points_2d)} puntos teóricos 2D al gráfico")
    
    # Añadir puntos 3D (para gráficos 3D)
    if points_3d:
        x_coords = [p['x'] for p in points_3d]  # Variable independiente 1 normalizada
        y_coords = [p['y'] for p in points_3d]  # Variable independiente 2 normalizada
        z_coords = [p['z'] for p in points_3d]  # 🔧 CAMBIO: Parámetro objetivo en escala original
        
        # Crear texto de hover
        hover_texts = []
        for p in points_3d:
            hover_parts = [
                f"<b>Punto Teórico - Modelo {p['modelo_idx']+1}</b>",
                f"<b>Predictor 1:</b> {p['predictor_1']}",
                f"<b>Predictor 2:</b> {p['predictor_2']}",
                f"<b>Tipo:</b> {p['tipo_modelo']}",
                f"<b>X1 original:</b> {p['x_original']:.3f}",
                f"<b>X2 original:</b> {p['y_original']:.3f}",
                f"<b>Z calculado (escala original):</b> {p['z_original']:.3f}",
                f"<b>Ecuación:</b> {p['ecuacion']}"
            ]
            
            if p.get('r2') is not None:
                hover_parts.append(f"<b>R²:</b> {p['r2']:.3f}")
            if p.get('mape') is not None:
                hover_parts.append(f"<b>MAPE:</b> {p['mape']:.1f}%")
            if p.get('confianza') is not None:
                hover_parts.append(f"<b>Confianza:</b> {p['confianza']:.3f}")
                
            hover_texts.append("<br>".join(hover_parts))
        
        # Añadir al gráfico 3D
        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers',
            name='Puntos Teóricos 3D',
            marker=dict(
                symbol='diamond',
                size=12,
                color='purple',
                line=dict(width=2, color='darkviolet')
            ),
            hovertemplate='%{hovertext}<extra></extra>',
            hovertext=hover_texts,
            legendgroup='theoretical',
            showlegend=True
        ))
        
        logger.info(f"Añadidos {len(points_3d)} puntos teóricos 3D al gráfico")


def extract_imputation_points(detalles_por_celda, 
                             celda_key: str,
                             n_predictores_filter: Optional[int] = None) -> List[Dict]:
    """
    Extrae puntos de imputación de los subdiccionarios 'similitud', 'correlacion' y 'final' 
    de detalles_por_celda usando la nueva estructura del JSON (lista de diccionarios).
    
    Parameters:
    -----------
    detalles_por_celda : List[Dict] o Dict
        Lista de diccionarios con estructura: [{'celda_key': str, 'imputacion': dict}, ...]
        O diccionario con estructura antigua
    celda_key : str
        Clave de la celda (ej: "A7|Payload")
    n_predictores_filter : Optional[int]
        Si se especifica, filtra solo puntos para modelos con este número de predictores
        
    Returns:
    --------
    List[Dict]
        Lista de puntos de imputación con coordenadas originales y metadatos
    """
    logger = logging.getLogger(__name__)
    imputation_points = []
    
    # 🚨 DEBUG: Verificar estructura real de datos
    print(f"🚨 DEBUG ESTRUCTURA: celda_key={celda_key}")
    print(f"🚨 DEBUG ESTRUCTURA: tipo detalles_por_celda={type(detalles_por_celda)}")
    
    try:
        # 🎯 NUEVA ESTRUCTURA: Lista de diccionarios
        if isinstance(detalles_por_celda, list):
            print(f"🔧 USANDO NUEVA ESTRUCTURA: Lista con {len(detalles_por_celda)} elementos")
            
            # Buscar la celda en la lista
            celda_detalles = None
            for elemento in detalles_por_celda:
                if isinstance(elemento, dict) and elemento.get('celda_key') == celda_key:
                    celda_detalles = elemento.get('imputacion', {})
                    print(f"✅ Celda {celda_key} encontrada en la lista")
                    break
            
            if celda_detalles is None:
                print(f"❌ Celda {celda_key} no encontrada en la lista")
                logger.warning(f"Celda {celda_key} no encontrada en detalles_por_celda")
                return imputation_points
                
        # 🎯 ESTRUCTURA ANTIGUA: Diccionario
        elif isinstance(detalles_por_celda, dict):
            print(f"🔧 USANDO ESTRUCTURA ANTIGUA: Diccionario")
            
            # Obtener detalles de la celda
            if celda_key not in detalles_por_celda:
                logger.warning(f"Celda {celda_key} no encontrada en detalles_por_celda")
                return imputation_points
                
            celda_detalles = detalles_por_celda[celda_key]
        
        else:
            print(f"❌ Estructura desconocida: {type(detalles_por_celda)}")
            return imputation_points
        
        # 🚨 DEBUG: Verificar estructura real de la celda
        print(f"🚨 DEBUG ESTRUCTURA: celda_detalles.keys()={list(celda_detalles.keys())}")
        print(f"🚨 DEBUG ESTRUCTURA: Contenido completo={celda_detalles}")
        
        # 🎯 CORRECCIÓN: Los puntos están DIRECTAMENTE en las claves, no en 'informacion_generica_celda'
        # La estructura real es: celda_detalles['final'], NO celda_detalles['informacion_generica_celda']['final']
        
        # Subdiccionarios a procesar (están directamente en la celda)
        subdiccionarios = ['similitud', 'correlacion', 'final']
        
        print(f"🔧 USANDO ESTRUCTURA REAL: Accediendo directamente a {subdiccionarios}")
        
        for subdic_name in subdiccionarios:
            subdic_data = celda_detalles.get(subdic_name, {})
            
            # Verificar si el subdiccionario tiene datos
            if not subdic_data or not isinstance(subdic_data, dict) or len(subdic_data) == 0:
                print(f"🔍 Subdiccionario '{subdic_name}' vacío o inválido en {celda_key}")
                continue
                
            # Extraer campos necesarios
            valor_imputado = subdic_data.get('Valor imputado')
            var_indep_1 = subdic_data.get('variable_independiente_1')
            var_indep_2 = subdic_data.get('variable_independiente_2')
            confianza = subdic_data.get('Confianza')
            metodo_predictivo = subdic_data.get('Método predictivo')
            iteracion = subdic_data.get('Iteración imputación')
            advertencia = subdic_data.get('Advertencia')
            
            # Validar campos obligatorios
            if valor_imputado is None or var_indep_1 is None:
                print(f"⚠️  Campos obligatorios faltantes en {celda_key}.{subdic_name}")
                logger.warning(f"Campos obligatorios faltantes en {celda_key}.{subdic_name}")
                continue
                
            print(f"✅ Punto {subdic_name} válido: valor={valor_imputado}, var1={var_indep_1}, var2={var_indep_2}")
                
            # Determinar número de predictores basado en variable_independiente_2
            n_predictores = 2 if var_indep_2 is not None else 1
            
            # Aplicar filtro de número de predictores si se especifica
            if n_predictores_filter is not None and n_predictores != n_predictores_filter:
                print(f"🔍 Punto {subdic_name} omitido por filtro de predictores: {n_predictores} != {n_predictores_filter}")
                logger.debug(f"Punto {subdic_name} omitido por filtro de predictores: {n_predictores} != {n_predictores_filter}")
                continue
            
            # Crear punto de imputación
            punto_imputacion = {
                'x_original': var_indep_1,
                'y_original': var_indep_2 if n_predictores == 2 else None,
                'valor_imputado': valor_imputado,
                'confianza': confianza,
                'metodo_predictivo': metodo_predictivo,
                'iteracion': iteracion,
                'advertencia': advertencia,
                'n_predictores': n_predictores,
                'celda_key': celda_key,
                'subdic_name': subdic_name,
                'metodo': 'imputacion',
                'symbol': 'diamond' if subdic_name == 'final' else ('circle' if subdic_name == 'similitud' else 'square'),
                'size': 10 if subdic_name == 'final' else 8,
                'color': 'red' if subdic_name == 'final' else ('blue' if subdic_name == 'similitud' else 'green')
            }
            
            imputation_points.append(punto_imputacion)
            print(f"🎯 Añadiendo punto {subdic_name}: ({var_indep_1}, {var_indep_2}, {valor_imputado})")
            # Mejoramos la visibilidad de los logs para facilitar la depuración
            logger.info(f"Punto de imputación añadido: {celda_key}.{subdic_name} -> var_indep_1={var_indep_1}, var_indep_2={var_indep_2}, valor_imputado={valor_imputado}")
    
    except Exception as e:
        logger.error(f"Error extrayendo puntos de imputación para {celda_key}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
    print(f"🎯 RESULTADO: {len(imputation_points)} puntos extraídos para {celda_key}")
    logger.info(f"Extraídos {len(imputation_points)} puntos de imputación para {celda_key}")
    return imputation_points


def add_imputation_points_to_plot(fig: go.Figure, 
                                 imputation_points: List[Dict],
                                 show_imputation_points: bool = True) -> None:
    """
    Añade puntos de imputación al gráfico de visualización 2D o 3D.
    Detecta automáticamente si los puntos son 2D o 3D basándose en n_predictores.
    
    Parameters:
    -----------
    fig : go.Figure
        Figura de Plotly donde añadir los puntos
    imputation_points : List[Dict]
        Lista de puntos de imputación extraídos
    show_imputation_points : bool
        Si mostrar los puntos de imputación
    """
    if not show_imputation_points or not imputation_points:
        logger.info(f"Puntos de imputación no mostrados: show_imputation_points={show_imputation_points}, len(imputation_points)={len(imputation_points) if imputation_points else 0}")
        return
        
    logger.info(f"Añadiendo {len(imputation_points)} puntos de imputación al gráfico")
    
    # Separar puntos 2D y 3D
    points_2d = [p for p in imputation_points if p.get('n_predictores', 1) == 1]
    points_3d = [p for p in imputation_points if p.get('n_predictores', 1) == 2]
    
    # Configuración de colores y símbolos por método
    method_config = {
        'final': {'color': 'red', 'symbol': 'star', 'size': 12, 'name': 'Final'},
        'similitud': {'color': 'blue', 'symbol': 'circle', 'size': 10, 'name': 'Similitud'},
        'correlacion': {'color': 'green', 'symbol': 'square', 'size': 10, 'name': 'Correlación'}
    }
    
    # Añadir puntos 2D (para gráficos 2D)
    if points_2d:
        # Agrupar por subdiccionario para crear trazas separadas
        for subdic_name, config in method_config.items():
            subdic_points = [p for p in points_2d if p.get('subdic_name') == subdic_name]
            
            if not subdic_points:
                continue
                
            x_coords = [p['x_original'] for p in subdic_points]  # Variable independiente original
            y_coords = [p['y_original'] for p in subdic_points]  # Valor imputado
            
            # Crear texto de hover
            hover_texts = []
            for p in subdic_points:
                hover_parts = [
                    f"<b>Punto de Imputación - {config['name']}</b>",
                    f"<b>Celda:</b> {p['celda_key']}",
                    f"<b>Método:</b> {p['metodo_predictivo']}",
                    f"<b>X original:</b> {p['x_original']:.3f}",
                    f"<b>Valor imputado:</b> {p['valor_imputado']:.3f}",
                ]
                
                if p.get('confianza') is not None:
                    hover_parts.append(f"<b>Confianza:</b> {p['confianza']:.3f}")
                if p.get('iteracion') is not None:
                    hover_parts.append(f"<b>Iteración:</b> {p['iteracion']}")
                if p.get('advertencia'):
                    hover_parts.append(f"<b>⚠️ Advertencia:</b> {p['advertencia']}")
                    
                hover_texts.append("<br>".join(hover_parts))
            
            # Añadir al gráfico 2D
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='markers',
                name=f'Imputación {config["name"]} 2D',
                marker=dict(
                    symbol=config['symbol'],
                    size=config['size'],
                    color=config['color'],
                    line=dict(width=2, color='black')
                ),
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=hover_texts,
                legendgroup=f'imputation_{subdic_name}',
                showlegend=True
            ))
            
            logger.info(f"Añadidos {len(subdic_points)} puntos de imputación 2D del tipo {config['name']}")
    
    # Añadir puntos 3D (para gráficos 3D)
    if points_3d:
        # Agrupar por subdiccionario para crear trazas separadas
        for subdic_name, config in method_config.items():
            subdic_points = [p for p in points_3d if p.get('subdic_name') == subdic_name]
            
            if not subdic_points:
                continue
                
            x_coords = [p['x_original'] for p in subdic_points]  # Variable independiente 1 original
            y_coords = [p['y_original'] for p in subdic_points]  # Variable independiente 2 original
            z_coords = [p['z_original'] for p in subdic_points]  # Valor imputado
            
            # Crear texto de hover
            hover_texts = []
            for p in subdic_points:
                hover_parts = [
                    f"<b>Punto de Imputación - {config['name']}</b>",
                    f"<b>Celda:</b> {p['celda_key']}",
                    f"<b>Método:</b> {p['metodo_predictivo']}",
                    f"<b>Variable Independiente 1:</b> {p['x_original']:.3f}",
                    f"<b>Variable Independiente 2:</b> {p['y_original']:.3f}",
                    f"<b>Valor imputado:</b> {p['valor_imputado']:.3f}",
                ]
                
                if p.get('confianza') is not None:
                    hover_parts.append(f"<b>Confianza:</b> {p['confianza']:.3f}")
                if p.get('iteracion') is not None:
                    hover_parts.append(f"<b>Iteración:</b> {p['iteracion']}")
                if p.get('advertencia'):
                    hover_parts.append(f"<b>⚠️ Advertencia:</b> {p['advertencia']}")
                    
                hover_texts.append("<br>".join(hover_parts))
            
            # Añadir al gráfico 3D
            fig.add_trace(go.Scatter3d(
                x=x_coords,
                y=y_coords,
                z=z_coords,
                mode='markers',
                name=f'Imputación {config["name"]} 3D',
                marker=dict(
                    symbol=config['symbol'],
                    size=config['size'],
                    color=config['color'],
                    line=dict(width=2, color='black')
                ),
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=hover_texts,
                legendgroup=f'imputation_{subdic_name}',
                showlegend=True
            ))
            
            logger.info(f"Añadidos {len(subdic_points)} puntos de imputación 3D del tipo {config['name']}")

def add_original_imputation_points(fig: go.Figure, 
                                   modelos_por_celda: Dict,
                                   celda_key: str,
                                   show_imputation_points: bool = True,
                                   n_predictores_filter: Optional[int] = None) -> None:
    """
    Añade puntos de imputación al gráfico usando sus coordenadas originales (sin normalización).
    
    Parameters:
    -----------
    fig : go.Figure
        Figura de Plotly donde añadir los puntos
    modelos_por_celda : Dict
        Diccionario completo de modelos por celda
    celda_key : str
        Clave de la celda
    show_imputation_points : bool
        Si mostrar los puntos de imputación
    n_predictores_filter : Optional[int]
        Filtro por número de predictores
    """
    if not show_imputation_points:
        logger.info("Puntos de imputación deshabilitados")
        return
    
    try:
        # Extraer puntos de imputación
        imputation_points = extract_imputation_points(
            modelos_por_celda, 
            celda_key, 
            n_predictores_filter=n_predictores_filter
        )
        
        if not imputation_points:
            logger.info(f"No se encontraron puntos de imputación para {celda_key}")
            return
        
        # Configuración de colores y símbolos por método
        method_config = {
            'final': {'color': 'red', 'symbol': 'diamond', 'size': 12, 'name': 'Final'},
            'similitud': {'color': 'blue', 'symbol': 'circle', 'size': 10, 'name': 'Similitud'},
            'correlacion': {'color': 'green', 'symbol': 'square', 'size': 10, 'name': 'Correlación'}
        }
        
        # Separar puntos por dimensión
        points_2d = [p for p in imputation_points if p.get('n_predictores', 1) == 1]
        points_3d = [p for p in imputation_points if p.get('n_predictores', 1) == 2]
        
        # Añadir puntos 2D (coordenadas originales)
        if points_2d:
            for subdic_name, config in method_config.items():
                subdic_points = [p for p in points_2d if p.get('subdic_name') == subdic_name]
                
                if not subdic_points:
                    continue
                
                # Usar coordenadas originales directamente
                x_coords = [p['x_original'] for p in subdic_points]
                y_coords = [p['y_original'] for p in subdic_points]
                
                # Crear texto de hover
                hover_texts = []
                for p in subdic_points:
                    hover_parts = [
                        f"<b>Punto de Imputación - {config['name']}</b>",
                        f"<b>Celda:</b> {p['celda_key']}",
                        f"<b>Método:</b> {p['metodo_predictivo']}",
                        f"<b>Variable Independiente:</b> {p['x_original']:.3f}",
                        f"<b>Valor imputado:</b> {p['valor_imputado']:.3f}",
                    ]
                    
                    if p.get('confianza') is not None:
                        hover_parts.append(f"<b>Confianza:</b> {p['confianza']:.3f}")
                    if p.get('iteracion') is not None:
                        hover_parts.append(f"<b>Iteración:</b> {p['iteracion']}")
                    if p.get('advertencia'):
                        hover_parts.append(f"<b>⚠️ Advertencia:</b> {p['advertencia']}")
                        
                    hover_texts.append("<br>".join(hover_parts))
                
                # Añadir al gráfico 2D
                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='markers',
                    name=f'Imputación {config["name"]}',
                    marker=dict(
                        symbol=config['symbol'],
                        size=config['size'],
                        color=config['color'],
                        line=dict(width=2, color='black')
                    ),
                    hovertemplate='%{hovertext}<extra></extra>',
                    hovertext=hover_texts,
                    legendgroup=f'imputation_{subdic_name}',
                    showlegend=True
                ))
                
                logger.info(f"Añadidos {len(subdic_points)} puntos de imputación 2D originales del tipo {config['name']}")
        
        # Añadir puntos 3D (coordenadas originales)
        if points_3d:
            for subdic_name, config in method_config.items():
                subdic_points = [p for p in points_3d if p.get('subdic_name') == subdic_name]
                
                if not subdic_points:
                    continue
                
                # Usar coordenadas originales directamente
                x_coords = [p['x_original'] for p in subdic_points]
                y_coords = [p['y_original'] for p in subdic_points]
                z_coords = [p['z_original'] for p in subdic_points]
                
                # Crear texto de hover
                hover_texts = []
                for p in subdic_points:
                    hover_parts = [
                        f"<b>Punto de Imputación - {config['name']}</b>",
                        f"<b>Celda:</b> {p['celda_key']}",
                        f"<b>Método:</b> {p['metodo_predictivo']}",
                        f"<b>Variable Independiente 1:</b> {p['x_original']:.3f}",
                        f"<b>Variable Independiente 2:</b> {p['y_original']:.3f}",
                        f"<b>Valor imputado:</b> {p['valor_imputado']:.3f}",
                    ]
                    
                    if p.get('confianza') is not None:
                        hover_parts.append(f"<b>Confianza:</b> {p['confianza']:.3f}")
                    if p.get('iteracion') is not None:
                        hover_parts.append(f"<b>Iteración:</b> {p['iteracion']}")
                    if p.get('advertencia'):
                        hover_parts.append(f"<b>⚠️ Advertencia:</b> {p['advertencia']}")
                        
                    hover_texts.append("<br>".join(hover_parts))
                
                # Añadir al gráfico 3D
                fig.add_trace(go.Scatter3d(
                    x=x_coords,
                    y=y_coords,
                    z=z_coords,
                    mode='markers',
                    name=f'Imputación {config["name"]}',
                    marker=dict(
                        symbol=config['symbol'],
                        size=config['size'],
                        color=config['color'],
                        line=dict(width=2, color='black')
                    ),
                    hovertemplate='%{hovertext}<extra></extra>',
                    hovertext=hover_texts,
                    legendgroup=f'imputation_{subdic_name}',
                    showlegend=True
                ))
                
                logger.info(f"Añadidos {len(subdic_points)} puntos de imputación 3D originales del tipo {config['name']}")
        
        logger.info(f"Procesamiento de puntos de imputación completado para {celda_key}")
        
    except Exception as e:
        logger.error(f"Error añadiendo puntos de imputación originales: {e}")
        import traceback
        traceback.print_exc()

def add_normalized_imputation_points(fig: go.Figure, 
                                     detalles_por_celda: Dict,
                                     celda_key: str,
                                     show_imputation_points: bool = True,
                                     n_predictores_filter: Optional[int] = None,
                                     global_ranges: Optional[Dict] = None,
                                     modelos_por_celda: Optional[Dict] = None) -> None:
    """
    Añade puntos de imputación al gráfico usando normalización para que coincidan 
    con el espacio de coordenadas del gráfico, usando los rangos del mejor modelo.
    
    La normalización se realiza de la siguiente manera:
    - Para puntos 2D: 
      * x = variable_independiente_1 (normalizada con rangos X_original[:, 0])
      * y = valor_imputado (normalizada con rangos y_original)
    
    - Para puntos 3D:
      * x = variable_independiente_1 (normalizada con rangos X_original[:, 0])
      * y = variable_independiente_2 (normalizada con rangos X_original[:, 1])
      * z = valor_imputado (normalizada con rangos y_original)
    
    Los rangos de normalización se toman del mejor modelo (basado en confianza)
    para asegurar consistencia con la visualización de las curvas del modelo.
    
    Parameters:
    -----------
    fig : go.Figure
        Figura de Plotly donde añadir los puntos
    detalles_por_celda : Dict
        Diccionario de detalles por celda con la nueva estructura JSON
    celda_key : str
        Clave de la celda
    show_imputation_points : bool
        Si mostrar los puntos de imputación
    n_predictores_filter : Optional[int]
        Filtro por número de predictores
    global_ranges : Optional[Dict]
        [OBSOLETO] Rangos globales para normalización (ya no se usa)
    modelos_por_celda : Optional[Dict]
        [OBSOLETO] Diccionario con los modelos por celda (ya no se usa)
    """
    # Simple logging instead of excessive debug output
    logger.info(f"Añadiendo puntos de imputación normalizados para celda {celda_key}")
    
    if not show_imputation_points:
        logger.info("Puntos de imputación deshabilitados")
        return
    
    if n_predictores_filter:
        logger.info(f"Filtrando por número de predictores: {n_predictores_filter}")
        logger.info(f"🔍 DEBUG CRÍTICO: Filtrando por número de predictores: {n_predictores_filter}")
    
    try:
        # Extraer puntos de imputación usando la función actualizada
        logger.info(f"🔍 DEBUG CRÍTICO: Llamando a extract_imputation_points...")
        imputation_points = extract_imputation_points(
            detalles_por_celda, 
            celda_key, 
            n_predictores_filter=n_predictores_filter
        )
        
        logger.info(f"Puntos de imputación extraídos: {len(imputation_points)}")
        
        if not imputation_points:
            logger.warning(f"No se encontraron puntos de imputación para {celda_key} con filtro {n_predictores_filter}")
            # Intentar sin filtro para diagnóstico
            puntos_sin_filtro = extract_imputation_points(detalles_por_celda, celda_key, n_predictores_filter=None)
            logger.warning(f"Puntos sin filtro: {len(puntos_sin_filtro)}")
            
            if puntos_sin_filtro:
                ejemplo = puntos_sin_filtro[0]
                logger.warning(f"Ejemplo de punto: n_predictores={ejemplo.get('n_predictores')}, filtro_requerido={n_predictores_filter}")
            
            return

        # Continuar con el resto de la función original...
        logger.info(f"Procediendo a normalizar {len(imputation_points)} puntos")
          # � TEMPORALMENTE SIN NORMALIZACIÓN - SALTAMOS LA BÚSQUEDA DEL MEJOR MODELO
        logger.info(f"� DEBUG: RESTAURANDO lógica original de normalización")
        
        # Configuración de colores y símbolos por método
        method_config = {
            'final': {'color': 'red', 'symbol': 'diamond', 'size': 12, 'name': 'Final'},
            'similitud': {'color': 'blue', 'symbol': 'circle', 'size': 10, 'name': 'Similitud'},
            'correlacion': {'color': 'green', 'symbol': 'square', 'size': 10, 'name': 'Correlación'}
        }
        
        # 🔧 RESTAURAR NORMALIZACIÓN - OBTENER RANGOS DEL MEJOR MODELO
        # Asegurarse de que modelos_por_celda tiene la estructura correcta
        if modelos_por_celda is not None and isinstance(modelos_por_celda, dict):
            # 🔧 FIX CRÍTICO: Si el valor es una lista, mantener la estructura actual
            # porque get_best_model_ranges ya maneja listas correctamente
            # NO necesitamos modificar la estructura aquí
            pass
        else:
            # Si no tenemos modelos_por_celda, usar una estructura vacía
            modelos_por_celda = {}
        
        rango_x, rango_y, rango_z, predictor_names = get_best_model_ranges(
            modelos_por_celda, celda_key, n_predictores_filter=n_predictores_filter
        )
        logger.info(f"🔧 DEBUG: Rangos de normalización obtenidos: X={rango_x}, Y={rango_y}, Z={rango_z}")
        logger.info(f"🟢 DEBUG: Nombres de predictores: {predictor_names}")
        print(f"🟢 LÓGICA ORIGINAL: Rangos X={rango_x}, Y={rango_y}, Z={rango_z}")
        print(f"🟢 LÓGICA ORIGINAL: Predictores: {predictor_names}")
        
        # 🔧 VALIDACIÓN: Advertir si Y range es problemático
        if rango_y == [0, 1]:
            print(f"⚠️ WARNING: Y range es [0, 1] para celda '{celda_key}' - esto puede indicar un problema de estructura de datos")
            if modelos_por_celda and celda_key in modelos_por_celda:
                data_type = type(modelos_por_celda[celda_key])
                print(f"⚠️ DEBUG: Tipo de datos para '{celda_key}': {data_type}")
                if isinstance(modelos_por_celda[celda_key], list):
                    print(f"⚠️ DEBUG: Es una lista con {len(modelos_por_celda[celda_key])} elementos")
                    # Contar modelos de 2 predictores
                    modelos_2pred = [m for m in modelos_por_celda[celda_key] 
                                   if isinstance(m, dict) and m.get('n_predictores') == 2]
                    print(f"⚠️ DEBUG: Modelos de 2 predictores en la lista: {len(modelos_2pred)}")
        else:
            print(f"✅ Y range correcto: {rango_y}")
        
        # Separar puntos por dimensión según el filtro
        if n_predictores_filter == 1:
            # Solo procesar puntos 2D
            logger.info(f"� DEBUG: Procesando puntos 2D CON LÓGICA ORIGINAL (n_predictores=1)")
            
            # Procesar cada punto 2D - COORDENADAS NORMALIZADAS
            for point in imputation_points:
                try:
                    subdic_name = point.get('subdic_name', 'unknown')
                    config = method_config.get(subdic_name, method_config['final'])
                    
                    # APLICAR NORMALIZACIÓN (X normalizada, Y original para 2D)
                    x_norm, y_val, _ = normalize_imputation_point(
                        point, rango_x, rango_y, rango_z, n_predictores=1
                    )
                    
                    # Para gráficos 2D: X normalizada, Y original
                    x_val = x_norm
                    # y_val ya está asignado por normalize_imputation_point
                    
                    logger.info(f"🔍 DEBUG: Punto 2D {subdic_name}: ORIGINAL ({point.get('x_original')}, {point.get('valor_imputado')}) → X_NORM={x_val:.3f}, Z_ORIGINAL={y_val}")
                    
                    print(f"🔍 AÑADIENDO TRAZA 2D: {subdic_name} en ({x_val:.3f}, {y_val})")
                    
                    # 🔧 MEJORA: Obtener información detallada para tooltip
                    # Obtener nombres reales de predictores
                    if len(predictor_names) > 0:
                        predictor_1_name = predictor_names[0]
                    else:
                        predictor_1_name = "Variable Independiente"
                    
                    # Obtener valores originales según el formato del campo
                    valor_original_x = point.get('variable_independiente_1')
                    if valor_original_x is None:
                        valor_original_x = point.get('x_original', 'N/A')
                    
                    valor_original_z = point.get('variable_dependiente')
                    if valor_original_z is None:
                        valor_original_z = point.get('valor_imputado', 'N/A')
                    
                    # Añadir punto con información detallada en tooltip
                    fig.add_trace(go.Scatter(
                        x=[x_val],
                        y=[y_val],
                        mode='markers',
                        marker=dict(
                            color=config['color'],
                            symbol=config['symbol'],
                            size=config['size'],
                            line=dict(width=2, color='black')
                        ),
                        name=f"Imputación {config['name']}",
                        hovertemplate=f"<b>Imputación {config['name']}</b><br>" +
                                      f"<b>{predictor_1_name}:</b><br>" +
                                      f"  Valor original: {valor_original_x}<br>" +
                                      f"  Valor normalizado: {x_val:.4f}<br>" +
                                      f"  Rango usado: [{rango_x[0]:.3f}, {rango_x[1]:.3f}]<br>" +
                                      f"<b>Valor Imputado:</b><br>" +
                                      f"  Valor original: {valor_original_z}<br>" +
                                      f"  ✅ Mostrado en escala original (no normalizado)<br>" +
                                      "<extra></extra>",
                        showlegend=True
                    ))
                    print(f"� TRAZA 2D AÑADIDA EXITOSAMENTE: {subdic_name}")
                    logger.info(f"� DEBUG: Trace 2D con lógica original añadido para {subdic_name}")
                    
                except Exception as e:
                    subdic_name = point.get('subdic_name', 'unknown')
                    logger.error(f"🔍 DEBUG CRÍTICO: Error procesando punto 2D {subdic_name}: {str(e)}")
                    
        elif n_predictores_filter == 2:
        # 🔧 DEBUG CRÍTICO: ENTRADA A SECCIÓN 3D
        # print("🚨🚨🚨 ENTRADA A PROCESAMIENTO 3D - n_predictores_filter == 2 🚨🚨🚨")
        # print(f"🚨 Número de puntos de imputación a procesar: {len(imputation_points)}")
        # print(f"🚨 Rangos disponibles: X={rango_x}, Y={rango_y}, Z={rango_z}")
        # print(f"🚨 Nombres de predictores: {predictor_names}")
        # print("🚨🚨🚨 COMENZANDO LOOP DE PUNTOS 3D 🚨🚨🚨")

            # Solo procesar puntos 3D
            logger.info(f"� DEBUG: Procesando puntos 3D CON LÓGICA ORIGINAL (n_predictores=2)")
            
            # Procesar cada punto 3D - COORDENADAS NORMALIZADAS
            for point in imputation_points:
                try:
                    subdic_name = point.get('subdic_name', 'unknown')
                    config = method_config.get(subdic_name, method_config['final'])
                    
                    # APLICAR NORMALIZACIÓN (X, Y normalizadas, Z original para 3D)
                    x_norm, y_norm, z_val = normalize_imputation_point(
                        point, rango_x, rango_y, rango_z, n_predictores=2
                    )
                    
                    # Para gráficos 3D: X, Y normalizadas, Z original
                    x_val = x_norm
                    y_val = y_norm if y_norm is not None else 0.5
                    # z_val ya está asignado por normalize_imputation_point
                    
                    # 🔧 DEBUG ADICIONAL: Verificar si Y no se normaliza
                    if y_norm is None:
                        print(f"   ❌ WARNING: y_norm es None para {subdic_name}")
                        print(f"   y_original: {point.get('y_original')}")
                        print(f"   rango_y: {rango_y}")
                    elif point.get('y_original') is not None and (y_norm < 0 or y_norm > 1):
                        print(f"   ❌ WARNING: y_norm={y_norm} está fuera de [0,1] para {subdic_name}")
                        print(f"   y_original: {point.get('y_original')}")
                        print(f"   rango_y: {rango_y}")
                        print(f"   ¿y_original está fuera del rango del modelo?")
                    # 🔧 DEBUG ADICIONAL: Verificar si Y no se normaliza
                    if y_norm is None:
                        print(f"   ❌ WARNING: y_norm es None para {subdic_name}")
                        print(f"   y_original: {point.get('y_original')}")
                        print(f"   rango_y: {rango_y}")
                        print(f"   ¿y_original está fuera del rango del modelo?")
                    
                    logger.info(f"🔍 DEBUG: Punto 3D {subdic_name}: ORIGINAL ({point.get('x_original')}, {point.get('y_original')}, {point.get('valor_imputado')}) → X_NORM={x_val:.3f}, Y_NORM={y_val:.3f}, Z_ORIGINAL={z_val}")
                    print(f"🔍 AÑADIENDO TRAZA 3D: {subdic_name} en ({x_val:.3f}, {y_val:.3f}, {z_val})")
                    
                    # 🔧 MEJORA: Obtener información detallada para tooltip 3D
                    # Obtener nombres reales de predictores
                    if len(predictor_names) > 0:
                        predictor_1_name = predictor_names[0]
                    else:
                        predictor_1_name = "Variable Independiente 1"
                    
                    if len(predictor_names) > 1:
                        predictor_2_name = predictor_names[1]
                    else:
                        predictor_2_name = "Variable Independiente 2"
                    
                    # Obtener valores originales según el formato del campo
                    valor_original_x = point.get('variable_independiente_1')
                    if valor_original_x is None:
                        valor_original_x = point.get('x_original', 'N/A')
                    
                    valor_original_y = point.get('variable_independiente_2')
                    if valor_original_y is None:
                        valor_original_y = point.get('y_original', 'N/A')
                    
                    valor_original_z = point.get('variable_dependiente')
                    if valor_original_z is None:
                        valor_original_z = point.get('valor_imputado', 'N/A')
                    
                    # Añadir punto con información detallada en tooltip
                    fig.add_trace(go.Scatter3d(
                        x=[x_val],
                        y=[y_val],
                        z=[z_val],
                        mode='markers',
                        marker=dict(
                            color=config['color'],
                            symbol=config['symbol'],
                            size=config['size'],
                            line=dict(width=2, color='black')
                        ),
                        name=f"Imputación {config['name']}",
                        hovertemplate=f"<b>Imputación {config['name']}</b><br>" +
                                      f"<b>{predictor_1_name} (X):</b><br>" +
                                      f"  Valor original: {valor_original_x}<br>" +
                                      f"  Valor normalizado: {x_val:.4f}<br>" +
                                      f"  Rango usado: [{rango_x[0]:.3f}, {rango_x[1]:.3f}]<br>" +
                                      f"<b>{predictor_2_name} (Y):</b><br>" +
                                      f"  Valor original: {valor_original_y}<br>" +
                                      f"  Valor normalizado: {y_val:.4f}<br>" +
                                      f"  Rango usado: [{rango_y[0]:.3f}, {rango_y[1]:.3f}]<br>" +
                                      f"<b>Valor Imputado (Z):</b><br>" +
                                      f"  Valor original: {valor_original_z}<br>" +
                                      f"  ✅ Mostrado en escala original (no normalizado)<br>" +
                                      "<extra></extra>",
                        showlegend=True
                    ))
                    print(f"🔍 TRAZA 3D AÑADIDA EXITOSAMENTE: {subdic_name}")
                    logger.info(f"🔍 DEBUG: Trace 3D con lógica original añadido para {subdic_name}")
                    
                except Exception as e:
                    subdic_name = point.get('subdic_name', 'unknown')
                    logger.error(f"🔍 DEBUG CRÍTICO: Error procesando punto 3D {subdic_name}: {str(e)}")
        
        logger.info(f"� DEBUG: Función add_normalized_imputation_points completada exitosamente")
        print(f"��� LÓGICA ORIGINAL RESTAURADA: Función completada exitosamente ���")
        
    except Exception as e:
        logger.error(f"🔍 DEBUG CRÍTICO: Error en add_normalized_imputation_points: {str(e)}")
        import traceback
        logger.error(f"🔍 DEBUG CRÍTICO: Traceback: {traceback.format_exc()}")
        raise
