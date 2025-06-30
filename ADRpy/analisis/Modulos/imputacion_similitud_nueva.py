"""
imputacion_similitud_nueva.py
-----------------------------
Nueva implementación de imputación por similitud basada en filtrado directo por familias características.

Diferencias con imputacion_similitud_flexible.py:
- No usa filtrado progresivo por capas F0-F2
- Filtra directamente por diferencias ≤20% en parámetros válidos 
- Requiere al menos un parámetro válido en cada familia (física, geométrica, prestacional)
- Coeficiente de similitud basado en promedio simple de diferencias relativas
- Misma metodología de cálculo de confianza final

Uso:
    resultado = imputar_por_similitud_nueva(
        df_filtrado=df,
        aeronave_objetivo="Aeronave X", 
        parametro_objetivo="Parámetro Y",
        verbose=True
    )
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

# ------------------------ HELPERS (del archivo original) ------------------------

def imprimir(msg, bold=False):
    """Imprime mensaje con formato opcional en negrita."""
    prefix = "\033[1m" if bold else ""
    suffix = "\033[0m" if bold else ""
    print(f"{prefix}{msg}{suffix}")

def is_missing(val):
    """
    Returns True if the value is considered missing (NaN, empty string, special codes, etc.).
    """
    if pd.isna(val):
        return True
    if isinstance(val, str) and val.strip().lower() in ["", "nan", "nan " "-", "#n/d", "n/d", "#¡valor!"]:
        return True
    return False

def penalizacion_por_k(k):
    """
    Calcula la penalización por cantidad de vecinos (k) usando una ecuación polinomial.
    Para k > 10, la penalización se fija en 1.0 (Confianza máxima).
    """
    if k > 10:
        return 1.0
    return max(0, min(1, 0.00002281 * k**5 - 0.00024 * k**4 - 0.0036 * k**3 + 0.046 * k**2 + 0.0095 * k + 0.024))

# ------------------------ CONFIGURACIÓN DE FAMILIAS ------------------------

def configurar_familias_caracteristicas():
    """
    Define las familias de características para la clasificación de parámetros.
    
    Returns:
        dict: Diccionario con tres familias (fisica, geometrica, prestacional)
    """
    familias = {
        "fisica": [
            "Peso máximo al despegue (MTOW)",
            "Empty weight",
            "Payload",
            "Cantidad de motores"
        ],
        "geometrica": [
            "Área del ala",
            "Envergadura",
            "Longitud del fuselaje",
            "Ancho del fuselaje",
            "Relación de aspecto del ala"
        ],
        "prestacional": [
            "Potencia específica (P/W)",
            "Autonomía de la aeronave",
            "Alcance de la aeronave", 
            "Velocidad a la que se realiza el crucero (KTAS)",
            "Velocidad máxima (KIAS)",
            "Rango de comunicación",
            "Potencia HP"
        ]
    }
    return familias

def clasificar_parametro(parametro: str, familias: dict) -> str:
    """
    Clasifica un parámetro en una de las tres familias características.
    
    Args:
        parametro: Nombre del parámetro
        familias: Diccionario de familias características
        
    Returns:
        str: Nombre de la familia ('fisica', 'geometrica', 'prestacional') o 'desconocida'
    """
    for familia_nombre, parametros in familias.items():
        if parametro in parametros:
            return familia_nombre
    return 'desconocida'

# ------------------------ FUNCIÓN PRINCIPAL ------------------------

def imputar_por_similitud_nueva(
    df_filtrado: pd.DataFrame,
    aeronave_objetivo: str,
    parametro_objetivo: str,
    df_procesado_base: pd.DataFrame = None,  # NUEVO argumento
    verbose: bool = True,
    debug: bool = True
) -> Optional[Dict]:
    """
    Nueva implementación de imputación por similitud basada en filtrado directo por familias.
    
    Args:
        df_filtrado: DataFrame donde filas=aeronaves, columnas=parámetros preestablecidos
        aeronave_objetivo: Nombre (índice) de la aeronave objetivo
        parametro_objetivo: Nombre de la columna/parámetro a imputar
        df_procesado_base: DataFrame adicional con datos procesados base para vecinos predictores
        verbose: Si imprimir detalles del proceso
        debug: Si imprimir información de depuración adicional
        
    Returns:
        dict: Resultado con valor imputado, confianza y detalles, o None si falla
    """
    
    if verbose:
        imprimir(f"\n=== Nueva Imputación por Similitud: {aeronave_objetivo} - {parametro_objetivo} ===", True)
    
    # Validaciones iniciales
    if parametro_objetivo not in df_filtrado.columns:
        if verbose:
            imprimir(f"⚠️ Parámetro '{parametro_objetivo}' no encontrado en df_filtrado.", True)
        return None
        
    if aeronave_objetivo not in df_filtrado.index:
        if verbose:
            imprimir(f"⚠️ Aeronave '{aeronave_objetivo}' no encontrada en df_filtrado.", True)
        return None
    
    # Verificar que el parámetro objetivo esté vacío en la aeronave objetivo
    if not is_missing(df_filtrado.loc[aeronave_objetivo, parametro_objetivo]):
        if verbose:
            imprimir(f"⚠️ El parámetro '{parametro_objetivo}' ya tiene valor en '{aeronave_objetivo}'.", True)
        return None
    
    # Configurar familias características
    familias = configurar_familias_caracteristicas()
    
    # Obtener valores de la aeronave objetivo
    valores_objetivo = df_filtrado.loc[aeronave_objetivo]
    
    if debug:
        imprimir(f"🔍 Aeronave objetivo: {aeronave_objetivo}")
        imprimir(f"🔍 Parámetros disponibles en df_filtrado: {len(df_filtrado.columns)}")
        imprimir(f"🔍 Aeronaves candidatas: {len(df_filtrado.index) - 1}")
    
    # Lista para almacenar aeronaves similares válidas
    aeronaves_similares = []
    
    # Evaluar cada aeronave candidata
    for aeronave_candidata in df_filtrado.index:
        if aeronave_candidata == aeronave_objetivo:
            continue
            
        # Verificar que la aeronave candidata tenga valor para el parámetro objetivo
        if is_missing(df_filtrado.loc[aeronave_candidata, parametro_objetivo]):
            continue
            
        if debug:
            imprimir(f"\n🔍 Evaluando candidata: {aeronave_candidata}")
        
        # Evaluar parámetros válidos y calcular diferencias
        parametros_validos = []
        diferencias_relativas = []
        familias_cumplidas = {"fisica": False, "geometrica": False, "prestacional": False}
        
        for parametro in df_filtrado.columns:
            if parametro == parametro_objetivo:
                continue
                
            valor_objetivo = valores_objetivo[parametro]
            valor_candidata = df_filtrado.loc[aeronave_candidata, parametro]
            
            # Verificar que ambos valores sean válidos
            if is_missing(valor_objetivo) or is_missing(valor_candidata):
                continue            # Convertir a float para cálculos de manera robusta
            try:
                # Usar str() para manejar cualquier tipo y luego convertir
                val_obj = float(str(valor_objetivo).strip())
                val_cand = float(str(valor_candidata).strip())
            except (ValueError, TypeError, AttributeError):
                continue
                
            # Evitar división por cero
            if val_obj == 0:
                if val_cand == 0:
                    diferencia_pct = 0.0
                else:
                    # Si objetivo es 0 y candidata no, usar diferencia absoluta grande
                    diferencia_pct = 100.0
            else:
                diferencia_pct = abs(val_cand - val_obj) / abs(val_obj) * 100
            
            # Si la diferencia es mayor al 20%, descartar la aeronave candidata
            if diferencia_pct > 20.0:
                if debug:
                    imprimir(f"  ❌ {parametro}: {diferencia_pct:.1f}% > 20% - Candidata descartada")
                break
                
            # Parámetro válido dentro del 20%
            parametros_validos.append(parametro)
            diferencias_relativas.append(diferencia_pct)
            
            # Marcar familia como cumplida
            familia = clasificar_parametro(parametro, familias)
            if familia in familias_cumplidas:
                familias_cumplidas[familia] = True
                
            if debug:
                imprimir(f"  ✅ {parametro}: {diferencia_pct:.1f}% (familia: {familia})")
        
        else:  # Este else se ejecuta si el bucle for NO fue interrumpido por break
            # Verificar criterios de aceptación
            familias_cumplidas_count = sum(familias_cumplidas.values())
            
            # Criterio principal: al menos un parámetro en cada familia
            if familias_cumplidas_count == 3:
                acepta_candidata = True
                razon = "Cumple las 3 familias"
            # Criterio de excepción: 2 familias + al menos 6 parámetros válidos
            elif familias_cumplidas_count == 2 and len(parametros_validos) >= 6:
                acepta_candidata = True
                razon = f"Cumple 2 familias + {len(parametros_validos)} parámetros"
            else:
                acepta_candidata = False
                razon = f"Solo {familias_cumplidas_count} familias, {len(parametros_validos)} parámetros"
            
            if acepta_candidata:
                # Calcular coeficiente de similitud global
                coeficientes_individuales = []
                for diff_pct in diferencias_relativas:
                    # Polinómica: sim(x) = -0.002x² - 0.01x + 1, para x en [0,20]
                    if diff_pct <= 20.0:
                        coef = max(0.0, min(1.0, -0.002 * diff_pct**2 - 0.01 * diff_pct + 1))
                    else:
                        coef = 0.0
                    coeficientes_individuales.append(coef)
                
                coeficiente_similitud = np.mean(coeficientes_individuales)
                valor_parametro_objetivo = df_filtrado.loc[aeronave_candidata, parametro_objetivo]
                
                aeronaves_similares.append({
                    'aeronave': aeronave_candidata,
                    'coeficiente_similitud': coeficiente_similitud,
                    'valor_parametro': float(str(valor_parametro_objetivo)),
                    'parametros_validos': len(parametros_validos),
                    'familias_cumplidas': familias_cumplidas_count,
                    'diferencias_promedio': np.mean(diferencias_relativas),
                    'razon_aceptacion': razon
                })
                
                if verbose:
                    imprimir(f"  ✅ {aeronave_candidata}: similitud={coeficiente_similitud:.3f}, valor={valor_parametro_objetivo:.3f} ({razon})")
            else:
                if debug:
                    imprimir(f"  ❌ {aeronave_candidata}: {razon}")
    
    # Verificar si tenemos aeronaves similares suficientes
    if len(aeronaves_similares) == 0:
        if verbose:
            imprimir("❌ No se encontraron aeronaves similares válidas.", True)
        return None
    
    if verbose:
        imprimir(f"\n📊 Se encontraron {len(aeronaves_similares)} aeronaves similares válidas")
    
    # Calcular valor imputado usando promedio ponderado
    coeficientes = np.array([a['coeficiente_similitud'] for a in aeronaves_similares])
    valores_objetivo = np.array([a['valor_parametro'] for a in aeronaves_similares])
    
    # Normalizar pesos
    pesos = coeficientes / coeficientes.sum()
    valor_imputado = np.dot(pesos, valores_objetivo)
    
    # Calcular confianza similitud (promedio ponderado de coeficientes)
    confianza_similitud = np.dot(pesos, coeficientes)
    
    # Calcular penalización por cantidad de vecinos
    k = len(aeronaves_similares)
    penalizacion_k = penalizacion_por_k(k)
    
    # Calcular penalización por dispersión (CV)
    if k > 1:
        media_valores = np.mean(valores_objetivo)
        std_valores = np.std(valores_objetivo, ddof=0)
        cv = std_valores / media_valores if media_valores != 0 else 1.0
        confianza_cv = max(0.0, float(1.0 - (cv / 0.5)))
    else:
        cv = 1.0  # Penalización máxima para k=1
        confianza_cv = 0.0
    
    # Confianza en datos (combinación lineal)
    confianza_datos = 0.7 * penalizacion_k + 0.3 * confianza_cv
    
    # Confianza final
    confianza_final = confianza_similitud * confianza_datos
    
    # Determinar advertencias
    advertencias = []
    if k < 3:
        advertencias.append("k<3")
    if confianza_final < 0.5:
        advertencias.append("confianza_baja")
    if cv > 0.5:
        advertencias.append("alta_variabilidad")
    
    advertencia_texto = ", ".join(advertencias)
    
    # Mostrar detalles del cálculo si verbose
    if verbose:
        imprimir("\n📈 Detalles del cálculo de confianza:")
        imprimir(f"  Confianza similitud (prom. ponderado coeficientes): {confianza_similitud:.3f}")
        imprimir(f"  Penalización por k vecinos: {penalizacion_k:.3f}")
        imprimir(f"  Confianza por CV: {confianza_cv:.3f} (CV: {cv:.3f})")
        imprimir(f"  Confianza datos: {confianza_datos:.3f}")
        imprimir(f"  Confianza final: {confianza_final:.3f}")
        
        imprimir(f"\n✅ Valor imputado: {valor_imputado:.3f} (conf: {confianza_final:.3f}, k: {k})")
        if advertencia_texto:
            imprimir(f"⚠️ Advertencias: {advertencia_texto}")
    
    # Preparar lista de vecinos para el reporte
    lista_vecinos = [a['aeronave'] for a in aeronaves_similares]
    coeficientes_vecinos = [a['coeficiente_similitud'] for a in aeronaves_similares]
    valores_vecinos = [a['valor_parametro'] for a in aeronaves_similares]
    # Para compatibilidad con pipeline: sim_vals = coeficientes_vecinos
    sim_vals = np.array(coeficientes_vecinos)
    vecinos_predictores = {}
    df_pred = df_procesado_base if df_procesado_base is not None else df_filtrado
    for parametro in df_filtrado.columns:
        if parametro == parametro_objetivo:
            continue
        valores_parametro = []
        for aeronave_detalle in aeronaves_similares:
            aeronave_nombre = aeronave_detalle['aeronave']
            if not is_missing(df_pred.loc[aeronave_nombre, parametro]):
                try:
                    valor = float(str(df_pred.loc[aeronave_nombre, parametro]))
                    valores_parametro.append(valor)
                except (ValueError, TypeError):
                    valores_parametro.append(np.nan)
            else:
                valores_parametro.append(np.nan)
        if len([v for v in valores_parametro if not pd.isna(v)]) > 0:
            vecinos_predictores[parametro] = valores_parametro

    # Retornar resultado completo
    return {
        "valor": valor_imputado,
        "Confianza": confianza_final,
        "num_vecinos": k,
        "Vecinos": lista_vecinos,
        "sim_vals": sim_vals.tolist(),  # <-- compatibilidad con pipeline y versión flexible
        "vecinos_predictores": vecinos_predictores,
        "coeficientes_similitud": coeficientes_vecinos,
        "valores_vecinos": valores_vecinos,
        "aeronaves_similares_detalle": aeronaves_similares,
        "k": k,
        "penalizacion_k": penalizacion_k,
        "confianza_similitud": confianza_similitud,
        "confianza_datos": confianza_datos,
        "confianza_cv": confianza_cv,
        "coef_variacion": cv,
        "warning": advertencia_texto,
        "Método predictivo": "Similitud Nueva",
        "familia": "Filtrado directo",
        "dispersion": np.std(valores_objetivo, ddof=0) if k > 1 else 0,
    }

# ------------------------ FUNCIÓN DE IMPUTACIÓN GLOBAL ------------------------

def imputacion_por_similitud(
    df_filtrado: pd.DataFrame,
    df_procesado_base: pd.DataFrame = None,  # NUEVO argumento
    verbose: bool = True,
    debug: bool = True
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Realiza imputaciones por similitud nueva para todos los parámetros y aeronaves faltantes en df_filtrado.
    
    Args:
        df_filtrado: DataFrame donde filas=aeronaves, columnas=parámetros 
        df_procesado_base: DataFrame adicional con datos procesados base para vecinos predictores
        verbose: Si imprimir detalles del proceso
        debug: Si imprimir información de depuración adicional
        
    Returns:
        tuple: (df_resultado con valores imputados, lista de reportes detallados)
    """
    
    df_resultado = df_filtrado.copy()
    reporte_similitud = []
    
    if verbose:
        imprimir(f"\n🔄 Iniciando imputación global para {len(df_filtrado.columns)} parámetros", True)
    
    total_imputaciones = 0
    total_intentos = 0
    
    for parametro in df_filtrado.columns:
        for aeronave in df_filtrado.index:
            if is_missing(df_resultado.at[aeronave, parametro]):
                total_intentos += 1
                
                resultado = imputar_por_similitud_nueva(
                    df_filtrado=df_filtrado,
                    aeronave_objetivo=aeronave,
                    parametro_objetivo=parametro,
                    df_procesado_base=df_procesado_base,  # NUEVO: pasar el DataFrame completo
                    verbose=debug,  # Solo verbose en debug mode para evitar spam
                    debug=debug
                )
                
                if resultado is not None:
                    # Actualizar valor en el DataFrame resultado
                    df_resultado.at[aeronave, parametro] = resultado["valor"]
                    total_imputaciones += 1
                    
                    # Agregar al reporte
                    resultado_reporte = {
                        "Aeronave": aeronave,
                        "Parámetro": parametro,
                        "Valor imputado": resultado["valor"],
                        "Confianza": resultado["Confianza"],
                        "Vecinos entrenamiento": resultado["Vecinos"],
                        "k": resultado["k"],
                        "Penalizacion_k": resultado["penalizacion_k"],
                        "Confianza Similitud": resultado["confianza_similitud"],
                        "Confianza Datos": resultado["confianza_datos"],
                        "Confianza CV": resultado["confianza_cv"],
                        "CV": resultado["coef_variacion"],
                        "Dispersión": resultado["dispersion"],
                        "Advertencia": resultado["warning"],
                        "Método predictivo": "Similitud",
                        "Familia": resultado["familia"],
                        # Campos adicionales para análisis
                        "coeficientes_similitud": resultado["coeficientes_similitud"],
                        "valores_vecinos": resultado["valores_vecinos"],
                        "aeronaves_similares_detalle": resultado["aeronaves_similares_detalle"],
                        # Compatibilidad con pipeline: sim_vals y vecinos_predictores
                        "sim_vals": resultado["sim_vals"],
                        "vecinos_predictores": resultado["vecinos_predictores"]
                    }
                    reporte_similitud.append(resultado_reporte)

                    if verbose and not debug:
                        imprimir(f"✅ {aeronave} | {parametro}: {resultado['valor']:.3f} (conf: {resultado['Confianza']:.3f})")
    
    if verbose:
        imprimir(f"\n📊 Resumen de imputación global:", True)
        imprimir(f"   Total intentos: {total_intentos}")
        imprimir(f"   Imputaciones exitosas: {total_imputaciones}")
        imprimir(f"   Tasa de éxito: {(total_imputaciones/total_intentos*100):.1f}%" if total_intentos > 0 else "   Tasa de éxito: N/A")
    
    return df_resultado, reporte_similitud

# ------------------------ FUNCIÓN DE TESTING ------------------------

def test_imputacion_similitud_nueva():
    """
    Función de testing básica para verificar el funcionamiento.
    """
    # Crear DataFrame de prueba
    data = {
        'Peso máximo al despegue (MTOW)': [1000, 1200, 800, 1100, np.nan],
        'Área del ala': [15, 18, 12, 16, 14],
        'Potencia específica (P/W)': [0.2, 0.25, 0.18, 0.22, 0.21],
        'Velocidad crucero': [120, 130, 110, 125, np.nan],
        'Parámetro objetivo': [50, 55, 45, 52, np.nan]
    }
    
    df_test = pd.DataFrame(data, index=['Aeronave_A', 'Aeronave_B', 'Aeronave_C', 'Aeronave_D', 'Aeronave_E'])
    
    print("=== TEST DE IMPUTACIÓN POR SIMILITUD NUEVA ===")
    print("\nDataFrame de prueba:")
    print(df_test)
    
    # Test 1: Imputación individual
    print("\n=== TEST 1: Imputación individual ===")
    resultado = imputar_por_similitud_nueva(
        df_filtrado=df_test,
        aeronave_objetivo='Aeronave_E',
        parametro_objetivo='Parámetro objetivo',
        verbose=True,
        debug=True
    )
    
    if resultado:
        print(f"\nResultado: {resultado}")
    
    # Test 2: Imputación global
    print("\n=== TEST 2: Imputación global ===")
    df_resultado, reporte = imputacion_por_similitud(
        df_filtrado=df_test,
        verbose=True,
        debug=False
    )
    
    print("\nDataFrame resultado:")
    print(df_resultado)
    print(f"\nReporte generado: {len(reporte)} registros")

if __name__ == "__main__":
    # Ejecutar test si se llama directamente
    test_imputacion_similitud_nueva()
