import pandas as pd
from .imputacion_similitud_flexible import *
from .html_utils import convertir_a_html
from .data_processing import generar_resumen_faltantes
from .imputacion_correlacion import imputaciones_correlacion

def is_missing(val):
    """
    Returns True if the value is considered missing (NaN, empty string, special codes, etc.).
    """
    if pd.isna(val):
        return True
    if isinstance(val, str) and val.strip().lower() in ["", "nan", "nan ", "-", "#n/d", "n/d", "#¡valor!"]:
        return True
    return False

def bucle_imputacion_similitud_correlacion(
    df_parametros,
    df_atributos,
    parametros_preseleccionados,
    bloques_rasgos,
    capas_familia,
    df_procesado,
    max_iteraciones=3,
    debug_mode=False
):

    """
    Realiza un bucle alternando imputaciones por similitud y correlación, consolidando los resultados.
    Ahora se evita actualizar los DataFrames inmediatamente, y se eligen las imputaciones finales
    al final de cada iteración.

    Retorna:
        df_procesado_base (pd.DataFrame): DataFrame con imputaciones realizadas.
        df_resumen (pd.DataFrame): Detalle consolidado de imputaciones realizadas.
    """

    df_procesado_base = df_procesado.copy() 
     # Copia base del DataFrame original
    convertir_a_html(
        df_procesado_base,
        titulo="df_procesado_base",
        ancho="100%",
        alto="400px",
        mostrar=True,
    )

    resumen_imputaciones = (
        []
    )  # Lista para consolidar detalles de todas las imputaciones finales

    # Inicializar acumuladores fuera del bucle principal
    detalles_para_excel = []
    imputaciones_finales = []  # Inicializar para evitar variable posiblemente no definida
    modelos_por_celda = {}  # Nuevo: diccionario global para modelos de correlación por celda

    iteracion = 0  # Inicializar iteracion antes del bucle
    for iteracion in range(1, max_iteraciones + 1):
        imputaciones_iteracion = []  # Inicializar la lista para cada iteración
        print("\n" + "=" * 80)
        print(f"\033[1m=== INICIO DE ITERACIÓN {iteracion} ===\033[0m")
        print("=" * 80)

        print(f"\n=== Iteración {iteracion}: Resumen antes de imputaciones ===")
        resumen_antes, total_faltantes_antes = generar_resumen_faltantes(
            df_procesado_base,
            titulo=f"Resumen de Valores Faltantes Antes de Iteración {iteracion}",
        )

        # Crear copias independientes para cada método
        df_similitud = df_procesado_base.copy()
        df_correlacion = df_procesado_base.copy()

        # Imputación por similitud (no actualiza todavía)
        print("\n" + "-" * 80)
        print(f"\033[1m*** IMPUTACIÓN POR SIMILITUD - ITERACIÓN {iteracion} ***\033[0m")
        print("-" * 80)

        df_similitud_resultado, reporte_similitud = imputacion_por_similitud_general(
            df_parametros=df_parametros,
            df_atributos=df_atributos,
            parametros_preseleccionados=parametros_preseleccionados,
            bloques_rasgos=bloques_rasgos,
            capas_familia=capas_familia,
            df_base=df_similitud
        )

        if reporte_similitud and len(reporte_similitud) > 0:
            validos_similitud = [r for r in reporte_similitud if not is_missing(r.get("Valor imputado", None))]
            print(f"\033[1m>>> Se realizaron imputaciones por similitud (Cantidad válida={len(validos_similitud)})\033[0m")
        else:
            print("\033[1mNo se realizaron imputaciones por similitud en esta iteración.\033[0m")

        # Imputación por correlación (no actualiza todavía)
        print("\n" + "-" * 80)
        print(
            f"\033[1m*** IMPUTACIÓN POR CORRELACIÓN - ITERACIÓN {iteracion} ***\033[0m"
        )
        print("-" * 80)
        # Cambia aquí: obtener también modelos_info
        df_correlacion_resultado, reporte_correlacion, modelos_info_correlacion = imputaciones_correlacion(df_correlacion)

        # Guardar modelos_info_correlacion por cada celda (idx, objetivo)
        if modelos_info_correlacion:
            for modelo in modelos_info_correlacion:
                idx = modelo["Aeronave"]
                objetivo = modelo["Parámetro"]
                key = f"{idx}|{objetivo}"
                if key not in modelos_por_celda:
                    modelos_por_celda[key] = []
                modelos_por_celda[key].append(modelo)

        if reporte_correlacion is not None and len(reporte_correlacion) > 0:
            validos_correlacion = [r for r in reporte_correlacion if not is_missing(r.get("Valor imputado", None))]
            print(
                f"\033[1m>>> Se realizaron imputacion por correlacion (Cantidad válida={len(validos_correlacion)})\033[0m"
                )
        else:
            print(
                "\033[1mNo se realizaron imputaciones por correlación en esta iteración.\033[0m"
            )

        # Combinar las imputaciones de similitud y correlación
        imputaciones_candidatas = {}

        def registrar_imputacion(regs):
            for reg in regs:
                parametro = reg["Parámetro"]  
                aeronave = reg["Aeronave"] 
                key = (parametro, aeronave)
                if key not in imputaciones_candidatas:
                    imputaciones_candidatas[key] = []
                imputaciones_candidatas[key].append(reg)

        if reporte_similitud and len(reporte_similitud) > 0:
            registrar_imputacion(reporte_similitud)
        if reporte_correlacion is not None and len(reporte_correlacion) > 0:
            registrar_imputacion(reporte_correlacion)

        # Seleccionar las mejores imputaciones por celda (promedio ponderado o método único)
        detalles_iteracion = []  # Para exportar todos los detalles relevantes por celda en esta iteración
        for key, candidatos in imputaciones_candidatas.items():
            parametro, aeronave = key
            if not is_missing(df_procesado_base.at[aeronave, parametro]):
                continue

            dict_similitud = next((c for c in candidatos if c.get("Método predictivo", "").lower().startswith("similitud")), None)
            dict_correlacion = next((c for c in candidatos if c.get("Método predictivo", "").lower().startswith("correlacion")), None)

            candidatos_validos = [c for c in candidatos if not is_missing(c["Valor imputado"])]

            # Añadir número de imputación a los diccionarios originales
            if dict_similitud is not None:
                dict_similitud = dict(dict_similitud)
                dict_similitud["Iteración imputación"] = iteracion
            if dict_correlacion is not None:
                dict_correlacion = dict(dict_correlacion)
                dict_correlacion["Iteración imputación"] = iteracion

            if len(candidatos_validos) == 1:
                unico = dict(candidatos_validos[0])
                unico["Iteración imputación"] = iteracion
                valor = unico["Valor imputado"]
                confianza = unico["Confianza"]
                metodo_pred = unico.get("Método predictivo", "Desconocido")
                imp = {
                    "Aeronave": aeronave,
                    "Parámetro": parametro,
                    "Valor imputado": valor,
                    "Confianza": confianza,
                    "Método predictivo": metodo_pred,
                    "Iteración imputación": iteracion,
                    "Detalle imputación": [unico]  # Detalle como lista para consistencia
                }
                detalles_iteracion.append({
                    "Aeronave": aeronave,
                    "Parámetro": parametro,
                    "final": imp,
                    "similitud": dict_similitud,
                    "correlacion": dict_correlacion
                })
                imputaciones_iteracion.append(imp)
            elif len(candidatos_validos) > 1:
                suma_valores = 0
                suma_pesos = 0
                metodos = set()
                detalles_candidatos = []
                for candidato in candidatos_validos:
                    candidato = dict(candidato)
                    candidato["Iteración imputación"] = iteracion
                    detalles_candidatos.append(candidato)
                    valor = candidato["Valor imputado"]
                    confianza = candidato["Confianza"]
                    metodo = candidato.get("Método predictivo", "Desconocido")
                    suma_valores += valor * confianza
                    suma_pesos += confianza
                    metodos.add(metodo)
                valor_promedio = suma_valores / suma_pesos if suma_pesos > 0 else None
                suma_confianza_ponderada = 0
                for candidato in candidatos_validos:
                    confianza = candidato["Confianza"]
                    peso = (confianza / suma_pesos) if suma_pesos > 0 else 0
                    suma_confianza_ponderada += confianza * peso
                confianza_promedio = suma_confianza_ponderada
                if len(metodos) == 1:
                    metodo_predictivo = list(metodos)[0]
                else:
                    metodo_predictivo = "Similitud y Correlación"
                imp = {
                    "Aeronave": aeronave,
                    "Parámetro": parametro,
                    "Valor imputado": valor_promedio,
                    "Confianza": confianza_promedio,
                    "Método predictivo": metodo_predictivo,
                    "Iteración imputación": iteracion,
                    "Detalle imputación": detalles_candidatos  # Lista de detalles de cada método
                }
                detalles_iteracion.append({
                    "Aeronave": aeronave,
                    "Parámetro": parametro,
                    "final": imp,
                    "similitud": dict_similitud,
                    "correlacion": dict_correlacion
                })
                imputaciones_iteracion.append(imp)
            else:
                detalles_iteracion.append({
                    "Aeronave": aeronave,
                    "Parámetro": parametro,
                    "final": None,
                    "similitud": dict_similitud,
                    "correlacion": dict_correlacion
                })
        # Aplicar las imputaciones finales al DataFrame base
        for imp in imputaciones_iteracion:
            parametro = imp["Parámetro"]
            aeronave = imp["Aeronave"]
            valor = imp["Valor imputado"]
            metodo = imp["Método predictivo"]
            df_procesado_base.at[aeronave, parametro] = valor
            resumen_imputaciones.append(imp)
            print(
                f"Imputación final aplicada: {parametro} - {aeronave} = {valor} ({metodo})"
            )

        # Al final de la iteración, acumula los resultados
        imputaciones_finales.extend(imputaciones_iteracion)
        detalles_para_excel.extend(detalles_iteracion)

        print(f"\n=== Iteración {iteracion}: Resumen después de imputaciones ===")
        resumen_despues, total_faltantes_despues = generar_resumen_faltantes(
            df_procesado_base,
            titulo=f"Resumen de Valores Faltantes Después de Iteración {iteracion}",
        )

        # Verificar condición de salida
        def tiene_valores_validos(reporte):
            if not reporte:
                return False
            for reg in reporte:
                if not is_missing(reg.get("Valor imputado", None)):
                    return True
            return False

        no_similitud = reporte_similitud is None or len(reporte_similitud) == 0 or not tiene_valores_validos(reporte_similitud)
        no_correlacion = reporte_correlacion is None or len(reporte_correlacion) == 0 or not tiene_valores_validos(reporte_correlacion)
        if no_similitud and no_correlacion:
            print("\033[1mNo se realizaron nuevas imputaciones válidas. Finalizando...\033[0m")
            # Asegurar que las variables existen antes de retornar
            if 'imputaciones_finales' not in locals():
                imputaciones_finales = []
            if 'detalles_para_excel' not in locals():
                detalles_para_excel = []
            break
       
        print("\n" + "=" * 80)
        print(f"\033[1m=== FIN DE ITERACIÓN {iteracion} ===\033[0m")
        print("=" * 80)

    print("\n" + "=" * 80)
    print("\033[1m=== RESUMEN FINAL ===\033[0m")
    print("=" * 80)

    print(f"\033[1mTotal de iteraciones realizadas: {iteracion}\033[0m")
    # Contar solo imputaciones válidas (no missing)
    imputaciones_validas = [imp for imp in resumen_imputaciones if not is_missing(imp.get("Valor imputado", None))]
    print(f"\033[1mTotal de valores imputados: {len(imputaciones_validas)}\033[0m")

    # Al final del bucle, exportar modelos_por_celda a JSON
    import json
    output_path = "ADRpy/analisis/Results/modelos_completos_por_celda.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(modelos_por_celda, f, ensure_ascii=False, indent=2)

    return df_procesado_base, pd.DataFrame(resumen_imputaciones), imputaciones_finales, detalles_para_excel, modelos_por_celda
