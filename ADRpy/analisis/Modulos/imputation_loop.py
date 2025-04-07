import pandas as pd
from .similarity_imputation import imputacion_similitud_con_rango
from .correlation_imputation import Imputacion_por_correlacion
from .html_utils import convertir_a_html
from .data_processing import generar_resumen_faltantes




def bucle_imputacion_similitud_correlacion(df_procesado, parametros_preseleccionados, tabla_completa, reduccion_confianza=0.05, max_iteraciones=7):
    """
    Realiza un bucle alternando imputaciones por similitud y correlación, consolidando los resultados.
    Ahora se evita actualizar los DataFrames inmediatamente, y se eligen las imputaciones finales
    al final de cada iteración.
    
    Retorna:
        df_procesado_base (pd.DataFrame): DataFrame con imputaciones realizadas.
        df_resumen (pd.DataFrame): Detalle consolidado de imputaciones realizadas.
    """

    df_procesado_base = df_procesado.copy()  # Copia base del DataFrame original
    df_filtrado_base = df_filtrado.copy()    # Copia base del DataFrame original

    convertir_a_html(df_procesado_base, titulo="df_procesado_base", ancho="100%", alto="400px", mostrar=True)
    convertir_a_html(df_filtrado_base, titulo="df_filtrado_base", ancho="100%", alto="400px", mostrar=True)
    resumen_imputaciones = []  # Lista para consolidar detalles de todas las imputaciones finales

    # Configuración inicial para imputaciones por similitud
    print("\n=== Configuración Inicial ===")
    try:
        rango_min = float(input("Ingrese el rango mínimo de MTOW (1-200, predeterminado 85): ") or 85) / 100
        rango_max = float(input("Ingrese el rango máximo de MTOW (1-200, predeterminado 115): ") or 115) / 100
        nivel_confianza_min = float(input("Ingrese el nivel mínimo de confianza (0-1, predeterminado 0.5): ") or 0.5)

        if not (0.01 <= rango_min <= 2.00 and 0.01 <= rango_max <= 2.00):
            raise ValueError("Los rangos deben estar entre 1% y 200%.")
        if rango_min >= rango_max:
            raise ValueError("El rango mínimo no puede ser mayor o igual al rango máximo.")
        if not (0 <= nivel_confianza_min <= 1):
            raise ValueError("El nivel de confianza debe estar entre 0 y 1.")
    except ValueError as e:
        print(f"Error: {e}. Usando valores predeterminados (85% mínimo, 115% máximo, 0.5 confianza mínima).")
        rango_min, rango_max, nivel_confianza_min = 0.85, 1.15, 0.5

    print(f"\nValores configurados: Rango MTOW [{rango_min*100:.0f}% - {rango_max*100:.0f}%], Confianza Mínima: {nivel_confianza_min:.2f}")

    # Configuración inicial para imputaciones por correlación
    try:
        umbral_correlacion = float(input("Ingrese el umbral mínimo de correlación (0-1, predeterminado 0.7): ") or 0.7)
        nivel_confianza_min_correlacion = float(input("Ingrese el nivel mínimo de confianza para correlación (0-1, predeterminado 0.5): ") or 0.5)

        if not (0 <= umbral_correlacion <= 1):
            raise ValueError("El umbral de correlación debe estar entre 0 y 1.")
        if not (0 <= nivel_confianza_min_correlacion <= 1):
            raise ValueError("El nivel de confianza debe estar entre 0 y 1.")
    except ValueError as e:
        print(f"Error: {e}. Usando valores predeterminados (umbral = 0.7, confianza mínima = 0.5).")
        umbral_correlacion, nivel_confianza_min_correlacion = 0.7, 0.5

    # Definir valores predeterminados para correlación
    min_datos_validos = 5  # Cantidad mínima de datos requeridos por parámetro
    umbral_correlacion = 0.7
    nivel_confianza_min_correlacion = 0.5
    reduccion_confianza = 0.05
    max_lineas_consola = 40000000

    for iteracion in range(1, max_iteraciones + 1):
        print("\n" + "="*80)
        print(f"\033[1m=== INICIO DE ITERACIÓN {iteracion} ===\033[0m")
        print("="*80)

        print(f"\n=== Iteración {iteracion}: Resumen antes de imputaciones ===")
        resumen_antes, total_faltantes_antes = generar_resumen_faltantes(
            df_procesado_base, titulo=f"Resumen de Valores Faltantes Antes de Iteración {iteracion}"
        )

        # Crear copias independientes para cada método
        df_similitud = df_filtrado_base.copy()
        df_correlacion = df_procesado_base.copy()

        # Imputación por similitud (no actualiza todavía)
        print("\n" + "-"*80)
        print(f"\033[1m*** IMPUTACIÓN POR SIMILITUD - ITERACIÓN {iteracion} ***\033[0m")
        print("-"*80)
        df_resultado_final, reporte_similitud = imputacion_similitud_con_rango(
            df_filtrado=df_similitud,
            df_procesado=df_procesado_base,
            rango_min=rango_min,
            rango_max=rango_max,
            nivel_confianza_min=nivel_confianza_min
        )

        if reporte_similitud and len(reporte_similitud) > 0:
            print("\033[1m>>> RESULTADOS DE IMPUTACIÓN POR SIMILITUD\033[0m")
            # Se guardan las imputaciones de similitud, pero NO se actualiza el DataFrame aún.
            # Se agregan la iteración y método aquí.
            for registro in reporte_similitud:
                registro["Iteración"] = iteracion
                registro["Método"] = "Similitud"
                registro["Nivel de Confianza"] *= (1 - reduccion_confianza) ** (iteracion - 1)
        else:
            print("\033[1mNo se realizaron imputaciones por similitud en esta iteración.\033[0m")

        # Imputación por correlación (no actualiza todavía)
        print("\n" + "-"*80)
        print(f"\033[1m*** IMPUTACIÓN POR CORRELACIÓN - ITERACIÓN {iteracion} ***\033[0m")
        print("-"*80)
        df_correlacion_final, reporte_correlacion = Imputacion_por_correlacion(
            df_correlacion,
            parametros_preseleccionados,
            tabla_completa,
            min_datos_validos=min_datos_validos,
            max_lineas_consola=max_lineas_consola,
            umbral_correlacion=umbral_correlacion,
            nivel_confianza_min_correlacion=nivel_confianza_min_correlacion,
            reduccion_confianza=reduccion_confianza
        )

        if reporte_correlacion and len(reporte_correlacion) > 0:
            print("\033[1m>>> RESULTADOS DE IMPUTACIÓN POR CORRELACIÓN\033[0m")
            for registro in reporte_correlacion:
                registro["Iteración"] = iteracion
                registro["Método"] = "Correlación"
                registro["Nivel de Confianza"] *= (1 - reduccion_confianza) ** (iteracion - 1)
        else:
            print("\033[1mNo se realizaron imputaciones por correlación en esta iteración.\033[0m")

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
        if reporte_correlacion and len(reporte_correlacion) > 0:
            registrar_imputacion(reporte_correlacion)

        # Seleccionar las mejores imputaciones por celda (la de mayor confianza)
        imputaciones_finales = []
        for key, candidatos in imputaciones_candidatas.items():
            parametro, aeronave = key
            # Si ya hay un valor en df_procesado_base, no imputar.
            if not pd.isna(df_procesado_base.loc[parametro, aeronave]):
                continue
            # Escoger la de mayor confianza
            mejor = max(candidatos, key=lambda x: x["Nivel de Confianza"])
            imputaciones_finales.append(mejor)

        # Ahora sí, aplicar las imputaciones finales al DataFrame base
        for imp in imputaciones_finales:
            parametro = imp["Parámetro"]
            aeronave = imp["Aeronave"]
            valor = imp["Valor Imputado"]
            metodo = imp["Método"]
            df_procesado_base.loc[parametro, aeronave] = valor
            df_filtrado_base.loc[parametro, aeronave] = valor
            resumen_imputaciones.append(imp)
            print(f"Imputación final aplicada: {parametro} - {aeronave} = {valor} ({metodo})")

        print(f"\n=== Iteración {iteracion}: Resumen después de imputaciones ===")
        resumen_despues, total_faltantes_despues = generar_resumen_faltantes(
            df_filtrado_base, titulo=f"Resumen de Valores Faltantes Después de Iteración {iteracion}"
        )

        # Verificar condición de salida
        no_similitud = (reporte_similitud is None or len(reporte_similitud) == 0)
        no_correlacion = (reporte_correlacion is None or len(reporte_correlacion) == 0)
        if no_similitud and no_correlacion:
            print("\033[1mNo se realizaron nuevas imputaciones. Finalizando...\033[0m")
            # Retornar resultados actuales antes de salir
            return df_procesado_base, pd.DataFrame(resumen_imputaciones)

        print("\n" + "="*80)
        print(f"\033[1m=== FIN DE ITERACIÓN {iteracion} ===\033[0m")
        print("="*80)

    # Si se terminan las iteraciones sin break:
    df_resumen = pd.DataFrame(resumen_imputaciones)
    print("\n" + "="*80)
    print("\033[1m=== RESUMEN FINAL ===\033[0m")
    print("="*80)

    convertir_a_html(
        df_procesado_base,
        titulo="DataFrame Procesado Final (df_procesado_base)"
    )
    convertir_a_html(
        df_resumen,
        titulo="Resumen Final de Imputaciones (df_resumen)"
    )

    print(f"\033[1mTotal de iteraciones realizadas: {iteracion}\033[0m")
    print(f"\033[1mTotal de valores imputados: {len(resumen_imputaciones)}\033[0m")

    return df_procesado_base, df_resumen