# === NUEVAS FUNCIONES PARA ACCESO CENTRALIZADO Y ROBUSTO ===
def get_modelos_por_celda(json_data):
    """
    Devuelve un dict {celda_key: [modelos]} usando la nueva estructura.
    Retorna también advertencias si faltan claves.
    """
    modelos_por_celda = {}
    warnings = []
    for celda_key, celda_data in json_data.items():
        modelos = celda_data.get('informacion_modelos_celda', {}).get('modelos', None)
        if modelos is None:
            warnings.append(f"Celda {celda_key}: falta 'informacion_modelos_celda/modelos'")
            modelos = []
        modelos_por_celda[celda_key] = modelos
    return modelos_por_celda, warnings

def get_detalles_por_celda(json_data):
    """
    Devuelve un dict {celda_key: detalles} usando la nueva estructura.
    Retorna también advertencias si faltan claves.
    """
    detalles_por_celda = {}
    warnings = []
    for celda_key, celda_data in json_data.items():
        detalles = celda_data.get('informacion_generica_celda', None)
        if detalles is None:
            warnings.append(f"Celda {celda_key}: falta 'informacion_generica_celda'")
            detalles = {}
        detalles_por_celda[celda_key] = detalles
    return detalles_por_celda, warnings

def extract_unique_values(json_data):
    """
    Extrae valores únicos de aeronaves, tipos_modelo, parámetros, etc. de toda la estructura.
    """
    aeronaves = set()
    tipos_modelo = set()
    parametros = set()
    warnings = []
    for celda_key, celda_data in json_data.items():
        # Asumimos que el formato de celda_key es 'Aeronave|Parametro'
        if '|' in celda_key:
            aeronave, parametro = celda_key.split('|', 1)
            aeronaves.add(aeronave)
            parametros.add(parametro)
        modelos = celda_data.get('informacion_modelos_celda', {}).get('modelos', [])
        for modelo in modelos:
            tipo = modelo.get('tipo')
            if tipo:
                tipos_modelo.add(tipo)
    return {
        'aeronaves': sorted(aeronaves),
        'tipos_modelo': sorted(tipos_modelo),
        'parametros': sorted(parametros)
    }, warnings

def get_parametro_objetivo(json_data, celda_key):
    """
    Devuelve el nombre del parámetro objetivo para una celda.
    """
    warnings = []
    celda_data = json_data.get(celda_key, {})
    info_generica = celda_data.get('informacion_generica_celda', {})
    parametro = info_generica.get('parametro_objetivo')
    if parametro is None:
        warnings.append(f"Celda {celda_key}: falta 'parametro_objetivo' en 'informacion_generica_celda'")
    return parametro, warnings

def get_df_original(json_data, celda_key):
    """
    Devuelve el DataFrame original para una celda, usando la función robusta existente.
    """
    import pandas as pd
    warnings = []
    celda_data = json_data.get(celda_key, {})
    df, w = get_full_dataframe_from_celda(celda_data)
    warnings.extend(w)
    return df, warnings

def get_metadata(json_data, celda_key):
    """
    Devuelve metadatos útiles para UI/plotting de una celda.
    Retorna un diccionario con información relevante y advertencias si faltan claves.
    """
    warnings = []
    celda_data = json_data.get(celda_key, {})
    info_generica = celda_data.get('informacion_generica_celda', {})
    info_modelos = celda_data.get('informacion_modelos_celda', {})
    meta = {
        'parametro_objetivo': info_generica.get('parametro_objetivo'),
        'df_original': info_generica.get('df_original'),
        'n_modelos': len(info_modelos.get('modelos', [])),
        'otros': {k: v for k, v in info_generica.items() if k not in ['parametro_objetivo', 'df_original']}
    }
    if not info_generica:
        warnings.append(f"Celda {celda_key}: falta 'informacion_generica_celda'")
    if 'parametro_objetivo' not in info_generica:
        warnings.append(f"Celda {celda_key}: falta 'parametro_objetivo' en 'informacion_generica_celda'")
    if 'df_original' not in info_generica:
        warnings.append(f"Celda {celda_key}: falta 'df_original' en 'informacion_generica_celda'")
    return meta, warnings
"""
Helper functions para acceder al DataFrame completo desde la nueva estructura JSON
"""

def get_full_dataframe_from_celda(celda_data):
    """
    Obtiene el DataFrame completo desde informacion_generica_celda.df_original
    o construye uno desde datos_entrenamiento de los modelos.
    
    Parameters:
    -----------
    celda_data : dict
        Datos de la celda desde el JSON
    Returns:
    --------
    tuple: (pandas.DataFrame or None, list)
        DataFrame completo con todas las columnas disponibles y lista de advertencias
    """
    warnings = []
    try:
        import pandas as pd
        
        # Primero intentar con df_original
        info_generica = celda_data.get('informacion_generica_celda', {})
        df_original_dict = info_generica.get('df_original', {}) if info_generica else {}
        
        # Si existe df_original y es válido, úsalo
        if isinstance(df_original_dict, dict) and len(df_original_dict) > 0:
            try:
                df_completo = pd.DataFrame(df_original_dict)
                if not df_completo.empty:
                    return df_completo, warnings
            except Exception as e:
                warnings.append(f"Error convirtiendo 'df_original' a DataFrame: {e}")
        
        # Si no hay df_original válido, construir desde datos_entrenamiento de algún modelo
        info_modelos = celda_data.get('informacion_modelos_celda', {})
        modelos = info_modelos.get('modelos', [])
        
        # Intentar con cada modelo hasta encontrar datos válidos
        for i, modelo in enumerate(modelos):
            datos_ent = modelo.get('datos_entrenamiento', {})
            X = datos_ent.get('X_original')
            y = datos_ent.get('y_original')
            columnas_pred = datos_ent.get('columnas_predictores')
            
            if X is not None and columnas_pred is not None and y is not None:
                try:
                    # Validar estructura de datos
                    if not isinstance(X, list) or not isinstance(columnas_pred, list) or not isinstance(y, list):
                        continue
                    if len(X) == 0 or len(columnas_pred) == 0 or len(y) == 0:
                        continue
                    if len(X) != len(y):
                        continue
                    
                    # Construir DataFrame
                    df_X = pd.DataFrame(X, columns=columnas_pred)
                    
                    # Buscar el nombre del parámetro objetivo
                    parametro_objetivo = info_generica.get('parametro_objetivo')
                    if not parametro_objetivo:
                        # Intentar extraer del nombre de la celda si es posible
                        # o usar un nombre por defecto
                        parametro_objetivo = 'y'
                    
                    df_X[parametro_objetivo] = y
                    return df_X, warnings
                    
                except Exception as e:
                    warnings.append(f"Error construyendo DataFrame desde modelo {i}: {e}")
                    continue
        
        # Si llegamos aquí, no se pudo construir ningún DataFrame
        if not modelos:
            warnings.append("No hay modelos disponibles en la celda.")
        else:
            warnings.append("No se pudo construir DataFrame: ningún modelo tiene datos_entrenamiento válidos.")
        
        return None, warnings
        
    except Exception as e:
        warnings.append(f"Error obteniendo DataFrame completo: {e}")
        return None, warnings


def get_model_specific_data(celda_data, modelo):
    """
    Extrae datos específicos del modelo desde el DataFrame completo
    
    Parameters:
    -----------
    celda_data : dict
        Datos de la celda desde el JSON
    modelo : dict
        Diccionario del modelo específico
        
    Returns:
    --------
    tuple : (X_data, y_data, df_filtrado, warnings)
        X_data: datos de predictores para este modelo
        y_data: datos de la variable objetivo
        df_filtrado: DataFrame filtrado para este modelo
        warnings: lista de advertencias
    """
    warnings = []
    try:
        import pandas as pd
        # Obtener DataFrame completo
        df_completo, df_warnings = get_full_dataframe_from_celda(celda_data)
        warnings.extend(df_warnings)
        if df_completo is None:
            warnings.append("No se pudo obtener el DataFrame completo para la celda.")
            return None, None, None, warnings
        # Obtener predictores y variable objetivo del modelo
        predictores = modelo.get('predictores', [])
        if not predictores:
            warnings.append("El modelo no tiene 'predictores'.")
            return None, None, None, warnings
        info_generica = celda_data.get('informacion_generica_celda', {})
        if not info_generica:
            warnings.append("Falta 'informacion_generica_celda' en la celda.")
            return None, None, None, warnings
        parametro_objetivo = info_generica.get('parametro_objetivo')
        if not parametro_objetivo:
            warnings.append("Falta 'parametro_objetivo' en 'informacion_generica_celda'.")
            return None, None, None, warnings
        columnas_necesarias = predictores + [parametro_objetivo]
        columnas_disponibles = df_completo.columns.tolist()
        columnas_faltantes = [col for col in columnas_necesarias if col not in columnas_disponibles]
        if columnas_faltantes:
            warnings.append(f"Columnas faltantes en DataFrame: {columnas_faltantes}")
            return None, None, None, warnings
        try:
            df_filtrado = df_completo[columnas_necesarias].dropna()
        except Exception as e:
            warnings.append(f"Error filtrando DataFrame: {e}")
            return None, None, None, warnings
        X_data = df_filtrado[predictores].values.tolist()
        y_data = df_filtrado[parametro_objetivo].values.tolist()
        return X_data, y_data, df_filtrado, warnings
    except Exception as e:
        warnings.append(f"Error extrayendo datos específicos del modelo: {e}")
        return None, None, None, warnings


def get_training_data_safe(datos_entrenamiento):
    """
    Función de seguridad para obtener datos de entrenamiento.
    Usa X_original/y_original como fuente principal de datos.
    
    Parameters:
    -----------
    datos_entrenamiento : dict
        Diccionario con datos de entrenamiento del modelo
        
    Returns:
    --------
    tuple : (X_data, y_data, warnings)
        Datos de entrenamiento seguros y lista de advertencias
    """
    warnings = []
    X_data = datos_entrenamiento.get('X_original')
    if X_data is None:
        X_data = datos_entrenamiento.get('X_train', [])
        warnings.append("X_original no encontrado, usando X_train como fallback.")
    y_data = datos_entrenamiento.get('y_original')
    if y_data is None:
        y_data = datos_entrenamiento.get('y_train', [])
        warnings.append("y_original no encontrado, usando y_train como fallback.")
    return X_data, y_data, warnings


def validate_new_json_structure(json_data):
    """
    Valida que el JSON tenga la nueva estructura esperada
    
    Parameters:
    -----------
    json_data : dict
        Datos del JSON completo
        
    Returns:
    --------
    dict : Reporte de validación
    """
    report = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "celdas_procesadas": 0,
        "df_completos_encontrados": 0,
        "modelos_totales": 0
    }
    
    for celda_key, celda_data in json_data.items():
        report["celdas_procesadas"] += 1
        
        # Verificar estructura básica
        if 'informacion_generica_celda' not in celda_data:
            report["errors"].append(f"Celda {celda_key}: falta informacion_generica_celda")
            report["valid"] = False
            continue
            
        if 'informacion_modelos_celda' not in celda_data:
            report["errors"].append(f"Celda {celda_key}: falta informacion_modelos_celda")
            report["valid"] = False
            continue
        
        # Verificar DataFrame completo
        info_generica = celda_data['informacion_generica_celda']
        if 'df_original' in info_generica and info_generica['df_original']:
            report["df_completos_encontrados"] += 1
            df_dict = info_generica['df_original']
            
            # Verificar que tiene más de 2 columnas (no solo predictor + objetivo)
            num_columnas = len(df_dict.keys()) if isinstance(df_dict, dict) else 0
            if num_columnas <= 2:
                report["warnings"].append(f"Celda {celda_key}: df_original solo tiene {num_columnas} columnas")
        else:
            report["warnings"].append(f"Celda {celda_key}: df_original faltante o vacío")
        
        # Contar modelos
        modelos = celda_data.get('informacion_modelos_celda', {}).get('modelos', [])
        report["modelos_totales"] += len(modelos)
        
        # Verificar estructura de modelos
        for i, modelo in enumerate(modelos):
            if 'datos_entrenamiento' not in modelo:
                report["errors"].append(f"Celda {celda_key}, modelo {i}: falta datos_entrenamiento")
                report["valid"] = False
            else:
                datos_ent = modelo['datos_entrenamiento']
                if 'X_original' not in datos_ent:
                    report["warnings"].append(f"Celda {celda_key}, modelo {i}: falta X_original")
                if 'y_original' not in datos_ent:
                    report["warnings"].append(f"Celda {celda_key}, modelo {i}: falta y_original")
    
    return report


if __name__ == "__main__":
    print("🧪 Funciones helper para nueva estructura JSON creadas")
    print("📋 Funciones disponibles:")
    print("   - get_modelos_por_celda(json_data)")
    print("   - get_detalles_por_celda(json_data)")
    print("   - extract_unique_values(json_data)")
    print("   - get_parametro_objetivo(json_data, celda_key)")
    print("   - get_df_original(json_data, celda_key)")
    print("   - get_metadata(json_data, celda_key)")
    print("   - get_full_dataframe_from_celda(celda_data)")
    print("   - get_model_specific_data(celda_data, modelo)")
    print("   - get_training_data_safe(datos_entrenamiento)")
    print("   - validate_new_json_structure(json_data)")
    print("\nEjemplo de uso:")
    print("from json_data_helpers import get_metadata, get_full_dataframe_from_celda")
    print("meta, warnings = get_metadata(json_data, 'Aeronave|Parametro')")
    print("df, warnings = get_full_dataframe_from_celda(json_data['Aeronave|Parametro']))")
