def test_extract_imputation_points(modelos_por_celda: Dict, 
                                   celda_key: str = None,
                                   n_predictores_filter: Optional[int] = None) -> Dict:
    """
    Función de prueba para extraer y mostrar puntos de imputación.
    
    Parameters:
    -----------
    modelos_por_celda : Dict
        Diccionario completo de modelos por celda
    celda_key : str
        Clave específica de celda a probar (si None, prueba todas)
    n_predictores_filter : Optional[int]
        Filtro por número de predictores
        
    Returns:
    --------
    Dict
        Resumen de puntos extraídos por celda
    """
    logger = logging.getLogger(__name__)
    resultados = {}
    
    # Determinar qué celdas procesar
    if celda_key:
        celdas_a_procesar = [celda_key] if celda_key in modelos_por_celda else []
    else:
        celdas_a_procesar = list(modelos_por_celda.keys())[:5]  # Primeras 5 celdas para prueba
    
    logger.info(f"Probando extracción de puntos de imputación en {len(celdas_a_procesar)} celdas")
    
    for celda in celdas_a_procesar:
        puntos = extract_imputation_points(
            modelos_por_celda, 
            celda, 
            n_predictores_filter=n_predictores_filter
        )
        
        resultados[celda] = {
            'total_puntos': len(puntos),
            'puntos_2d': len([p for p in puntos if p.get('n_predictores', 1) == 1]),
            'puntos_3d': len([p for p in puntos if p.get('n_predictores', 1) == 2]),
            'por_metodo': {},
            'ejemplos': puntos[:2]  # Primeros 2 puntos como ejemplos
        }
        
        # Contar por método
        for punto in puntos:
            metodo = punto.get('subdic_name', 'unknown')
            if metodo not in resultados[celda]['por_metodo']:
                resultados[celda]['por_metodo'][metodo] = 0
            resultados[celda]['por_metodo'][metodo] += 1
        
        logger.info(f"Celda {celda}: {len(puntos)} puntos extraídos")
    
    return resultados
