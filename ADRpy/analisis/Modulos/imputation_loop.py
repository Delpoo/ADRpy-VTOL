import pandas as pd
import numpy as np
import json
import os
import sys
import io
from datetime import datetime

# from .imputacion_similitud_flexible import *  # COMENTADO: Reemplazado por nueva implementación
from .imputacion_similitud_nueva import imputacion_por_similitud
from .html_utils import convertir_a_html
from .data_processing import generar_resumen_faltantes
from .imputacion_correlacion import imputaciones_correlacion

# Config integration
from .config_and_loading import (
    get_config,
    load_config_file,
    save_config_snapshot,
    apply_env_display_limits,
)

# Intentar importar el panel (opcional)
try:
    from .ux_config_panel import open_config_panel  # type: ignore
except Exception:
    open_config_panel = None  # type: ignore


def is_missing(val):
    """
    Returns True if the value is considered missing (NaN, empty string, special codes, etc.).
    """
    if pd.isna(val):
        return True
    if isinstance(val, str) and val.strip().lower() in [
        "",
        "nan",
        "nan ",
        "-",
        "#n/d",
        "n/d",
        "#¡valor!",
    ]:
        return True
    return False


def _cargar_configuracion() -> dict:
    """Carga configuración centralizada con posibles overrides y panel opcional."""
    overrides = None
    overrides_path = os.path.join(os.getcwd(), "config_overrides.json")
    if os.path.exists(overrides_path):
        try:
            overrides = load_config_file(overrides_path)
        except Exception:
            overrides = None

    # Flag local: abrir panel en runtime (puede enlazarse a CLI en el futuro)
    USE_PANEL = False
    if USE_PANEL and open_config_panel is not None:
        try:
            overrides = open_config_panel(overrides)
        except Exception:
            pass

    cfg = get_config(overrides if isinstance(overrides, dict) else None)
    apply_env_display_limits(cfg)
    return cfg


def bucle_imputacion_similitud_correlacion(
    df_filtrado,
    parametros_preseleccionados,
    #    bloques_rasgos,
    capas_familia,
    df_procesado,
    max_iteraciones=3,
    debug_mode=False,
    permitir_sin_filtro=False,
):
    """Bucle de imputaciones alternando similitud y correlación con mensajes de progreso."""

    # --- Configuración central ---
    cfg = _cargar_configuracion()

    # Orquestación
    max_iteraciones = cfg["orquestacion"]["max_iteraciones"]
    ejecutar_sim = cfg["orquestacion"]["ejecutar_similitud"]
    ejecutar_corr = cfg["orquestacion"]["ejecutar_correlacion"]
    permitir_sin_filtro = cfg["orquestacion"]["permitir_sin_filtro"]
    min_datos_validos = cfg["orquestacion"]["min_datos_validos"]
    mostrar_consola = cfg["orquestacion"]["mostrar_consola"]

    # Parámetros similitud
    sim_kwargs = {
        "umbral_pct_diferencia": cfg["similitud"]["umbral_pct_diferencia"],
        "min_familias": cfg["similitud"]["min_familias"],
        "excepcion_min_familias": cfg["similitud"]["excepcion_min_familias"],
        "k_min": cfg["similitud"]["k_min"],
        "k_max": cfg["similitud"]["k_max"],
        "peso_confianza_similitud": cfg["similitud"]["peso_confianza_similitud"],
        "peso_confianza_cv": cfg["similitud"]["peso_confianza_cv"],
        "verbosidad": cfg["similitud"]["verbosidad"],
    }

    # Parámetros de correlación/outliers + modelos
    out_kwargs = {
        "manejar_outliers": cfg["correlacion_outliers"]["manejar_outliers"],
        "umbral_z_suave": cfg["correlacion_outliers"]["umbral_z_suave"],
        "umbral_z_duro": cfg["correlacion_outliers"]["umbral_z_duro"],
        "alpha_pesos": cfg["correlacion_outliers"]["alpha_pesos"],
        "w_min": cfg["correlacion_outliers"]["w_min"],
        "remover_duro": cfg["correlacion_outliers"]["remover_duro"],
        # Política de extrapolación permanece deshabilitada salvo cambios internos explícitos
        "permitir_extrapolacion": cfg["correlacion_outliers"]["permitir_extrapolacion"],
        "tolerancia_fuera_rango": cfg["correlacion_outliers"]["tolerancia_fuera_rango"],
    }
    modelos_cfg = cfg["modelos"]
    # Interpretación de umbral MAPE: si viene como proporción (<=1), convertir a %
    _umbral_mape_cfg = modelos_cfg.get("umbral_mape_max", 7.5)
    umbral_mape_max_pct = (
        _umbral_mape_cfg * 100.0
        if isinstance(_umbral_mape_cfg, (int, float)) and _umbral_mape_cfg <= 1.0
        else _umbral_mape_cfg
    )

    # --- Loop/Combinación/Export desde config ---
    loop_cfg = cfg.get("loop", {})
    comb_cfg = loop_cfg.get(
        "combinacion", {"metodo": "promedio_ponderado", "conf_min": 0.0}
    )
    stop_cfg = loop_cfg.get(
        "stop", {"min_nuevas_por_iter": 1, "sin_mejora_consecutivas": 1}
    )
    export_cfg = loop_cfg.get(
        "export",
        {
            "json": {
                "enabled": True,
                "dir": "Results",
                "fname": "modelos_completos_por_celda.json",
            },
            "html_df_base": {"mostrar": True},
        },
    )
    orden = loop_cfg.get("orden", ["similitud", "correlacion"])

    # Redirección global de consola si se solicita silencio
    _restore_streams = False
    _old_stdout = None
    _old_stderr = None
    if not mostrar_consola:
        try:
            _old_stdout, _old_stderr = sys.stdout, sys.stderr
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            _restore_streams = True
        except Exception:
            _restore_streams = False  # si falla, no silenciar

    print("[INFO] Preparando los datos de trabajo. Esto puede demorar unos segundos...")
    df_procesado_base = df_procesado.copy()
    if export_cfg.get("html_df_base", {}).get("mostrar", True):
        convertir_a_html(
            df_procesado_base,
            titulo="df_procesado_base",
            ancho=f"{cfg['html']['ancho_px']}px",
            alto=f"{cfg['html']['alto_px']}px",
            mostrar=True,
        )

    print("\n=== Estado inicial del proceso ===")
    try:
        total_celdas = df_procesado_base.size
        faltantes = df_procesado_base.isna().sum().sum()
        print(f"[INFO] Celdas totales: {total_celdas}. Celdas vacias: {faltantes}.")
    except Exception:
        print("[INFO] Tabla cargada correctamente.")

    resumen_imputaciones = (
        []
    )  # Lista para consolidar detalles de todas las imputaciones finales

    # Inicializar acumuladores fuera del bucle principal
    detalles_para_excel = []
    imputaciones_finales = (
        []
    )  # Inicializar para evitar variable posiblemente no definida
    modelos_por_celda = (
        {}
    )  # Nuevo: diccionario global para modelos de correlación por celda

    iteracion = 0  # Inicializar iteracion antes del bucle
    _sin_mejora_consec = 0  # contador de iteraciones sin mejora
    print(
        f"\n[INFO] Se iniciara el ciclo de imputaciones (maximo {max_iteraciones} iteraciones)."
    )
    for iteracion in range(1, max_iteraciones + 1):
        imputaciones_iteracion = []  # Inicializar la lista para cada iteración
        print("\n" + "=" * 80)
        print(
            f"\033[1m=== Iteracion {iteracion}: el sistema esta trabajando ===\033[0m"
        )
        print("=" * 80)

        print(f"\n-- Resumen previo a las imputaciones (iteracion {iteracion}) --")
        resumen_antes, total_faltantes_antes = generar_resumen_faltantes(
            df_procesado_base,
            titulo=f"Resumen de Valores Faltantes Antes de Iteración {iteracion}",
        )

        # Crear copias independientes para cada método
        df_similitud = df_filtrado.copy()
        df_correlacion = df_procesado_base.copy()

        reporte_similitud = []
        reporte_correlacion = []
        modelos_info_correlacion = []

        for metodo in orden:
            if metodo == "similitud" and ejecutar_sim:
                print("\n" + "-" * 80)
                print(
                    f"\033[1m*** Etapa de similitud (iteracion {iteracion}) ***\033[0m"
                )
                print("-" * 80)
                print(
                    "  [INFO] Buscando aeronaves similares y calculando propuestas..."
                )
                df_similitud_resultado, reporte_similitud = imputacion_por_similitud(
                    df_filtrado=df_similitud,
                    df_procesado_base=df_procesado_base,
                    verbose=mostrar_consola,
                    debug=mostrar_consola,
                    **{
                        k: v
                        for k, v in sim_kwargs.items()
                        if k
                        in (
                            "umbral_pct_diferencia",
                            "min_familias",
                            "excepcion_min_familias",
                            "k_min",
                            "k_max",
                            "peso_confianza_similitud",
                            "peso_confianza_cv",
                            "verbosidad",
                        )
                    },
                )
            if metodo == "correlacion" and ejecutar_corr:
                print("\n" + "-" * 80)
                print(
                    f"\033[1m*** Etapa de correlacion (iteracion {iteracion}) ***\033[0m"
                )
                print("-" * 80)
                print(
                    "  [INFO] Probando relaciones matematicas entre parametros para completar valores..."
                )
                (
                    df_correlacion_resultado,
                    reporte_correlacion,
                    modelos_info_correlacion,
                ) = imputaciones_correlacion(
                    df_correlacion,
                    permitir_sin_filtro=permitir_sin_filtro,
                    min_datos_validos=min_datos_validos,
                    modelos_habilitados=modelos_cfg.get("habilitados", {}),
                    umbral_mape_max=umbral_mape_max_pct,
                    usar_loocv=modelos_cfg.get("usar_loocv", True),
                    loocv_usa_pesos=modelos_cfg.get("loocv_usa_pesos", False),
                    verbose=mostrar_consola,
                    manejar_outliers=out_kwargs.get("manejar_outliers", True),
                    umbral_z_suave=out_kwargs.get("umbral_z_suave", 3.0),
                    umbral_z_duro=out_kwargs.get("umbral_z_duro", 6.0),
                    alpha_pesos=out_kwargs.get("alpha_pesos", 0.5),
                    w_min=out_kwargs.get("w_min", 0.2),
                    remover_duro=out_kwargs.get("remover_duro", False),
                )

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
            validos_correlacion = [
                r
                for r in reporte_correlacion
                if not is_missing(r.get("Valor imputado", None))
            ]
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
        detalles_iteracion = (
            []
        )  # Para exportar todos los detalles relevantes por celda en esta iteración
        for key, candidatos in imputaciones_candidatas.items():
            parametro, aeronave = key
            if not is_missing(df_procesado_base.at[aeronave, parametro]):
                continue

            dict_similitud = next(
                (
                    c
                    for c in candidatos
                    if c.get("Método predictivo", "").lower().startswith("similitud")
                ),
                None,
            )
            dict_correlacion = next(
                (
                    c
                    for c in candidatos
                    if c.get("Método predictivo", "").lower().startswith("correlacion")
                ),
                None,
            )

            candidatos_validos = [
                c for c in candidatos if not is_missing(c["Valor imputado"])
            ]

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
                    "Detalle imputación": [
                        unico
                    ],  # Detalle como lista para consistencia
                }
                detalles_iteracion.append(
                    {
                        "Aeronave": aeronave,
                        "Parámetro": parametro,
                        "final": imp,
                        "similitud": dict_similitud,
                        "correlacion": dict_correlacion,
                    }
                )
                imputaciones_iteracion.append(imp)
            elif len(candidatos_validos) > 1:
                metodo_comb = comb_cfg.get("metodo", "promedio_ponderado")
                conf_min = float(comb_cfg.get("conf_min", 0.0))
                imp = None
                if metodo_comb == "mejor_confianza":
                    ganador = max(
                        candidatos_validos, key=lambda c: c.get("Confianza", 0)
                    )
                    if ganador.get("Confianza", 0) >= conf_min:
                        imp = {**ganador, "Aeronave": aeronave, "Parámetro": parametro}
                elif metodo_comb == "prioridad_correlacion":
                    prefer = next(
                        (
                            c
                            for c in candidatos_validos
                            if str(c.get("Método predictivo", ""))
                            .lower()
                            .startswith("correlacion")
                        ),
                        None,
                    )
                    candidato = prefer or max(
                        candidatos_validos, key=lambda c: c.get("Confianza", 0)
                    )
                    if candidato.get("Confianza", 0) >= conf_min:
                        imp = {
                            **candidato,
                            "Aeronave": aeronave,
                            "Parámetro": parametro,
                        }
                elif metodo_comb == "prioridad_similitud":
                    prefer = next(
                        (
                            c
                            for c in candidatos_validos
                            if str(c.get("Método predictivo", ""))
                            .lower()
                            .startswith("similitud")
                        ),
                        None,
                    )
                    candidato = prefer or max(
                        candidatos_validos, key=lambda c: c.get("Confianza", 0)
                    )
                    if candidato.get("Confianza", 0) >= conf_min:
                        imp = {
                            **candidato,
                            "Aeronave": aeronave,
                            "Parámetro": parametro,
                        }
                else:
                    # promedio_ponderado (comportamiento actual)
                    suma_valores = 0
                    suma_pesos = 0
                    metodos = set()
                    detalles_candidatos = []
                    for c in candidatos_validos:
                        cc = dict(c)
                        cc["Iteración imputación"] = iteracion
                        detalles_candidatos.append(cc)
                        suma_valores += cc["Valor imputado"] * cc["Confianza"]
                        suma_pesos += cc["Confianza"]
                        metodos.add(cc.get("Método predictivo", "Desconocido"))
                    if suma_pesos > 0:
                        valor_promedio = suma_valores / suma_pesos
                        confianza_promedio = sum(
                            cc["Confianza"] * (cc["Confianza"] / suma_pesos)
                            for cc in candidatos_validos
                        )
                        if confianza_promedio >= conf_min:
                            metodo_predictivo = (
                                "Similitud y Correlación"
                                if len(metodos) > 1
                                else list(metodos)[0]
                            )
                            imp = {
                                "Aeronave": aeronave,
                                "Parámetro": parametro,
                                "Valor imputado": valor_promedio,
                                "Confianza": confianza_promedio,
                                "Método predictivo": metodo_predictivo,
                                "Iteración imputación": iteracion,
                                "Detalle imputación": detalles_candidatos,
                            }
                if imp is None:
                    detalles_iteracion.append(
                        {
                            "Aeronave": aeronave,
                            "Parámetro": parametro,
                            "final": None,
                            "similitud": dict_similitud,
                            "correlacion": dict_correlacion,
                        }
                    )
                    continue
                detalles_iteracion.append(
                    {
                        "Aeronave": aeronave,
                        "Parámetro": parametro,
                        "final": imp,
                        "similitud": dict_similitud,
                        "correlacion": dict_correlacion,
                    }
                )
                imputaciones_iteracion.append(imp)
            else:
                detalles_iteracion.append(
                    {
                        "Aeronave": aeronave,
                        "Parámetro": parametro,
                        "final": None,
                        "similitud": dict_similitud,
                        "correlacion": dict_correlacion,
                    }
                )
            # === NUEVO: Cálculo de variable_independiente_1 y variable_independiente_2 para modelos 3D ===
            # Determinar predictores del mejor modelo de correlación
            predictor_1 = None
            predictor_2 = None
            if dict_correlacion is not None:
                predictores_corr = dict_correlacion.get("Predictores")
                if predictores_corr:
                    if isinstance(predictores_corr, str):
                        predictores_list = [
                            p.strip() for p in predictores_corr.split(",")
                        ]
                    elif isinstance(predictores_corr, (list, tuple)):
                        predictores_list = list(predictores_corr)
                    else:
                        predictores_list = []
                    if len(predictores_list) >= 1:
                        predictor_1 = predictores_list[0]
                    if len(predictores_list) >= 2:
                        predictor_2 = predictores_list[1]

            # --- SIMILITUD ---
            if dict_similitud is not None and predictor_1:
                if (
                    "sim_vals" in dict_similitud
                    and "vecinos_predictores" in dict_similitud
                ):
                    sim_vals = np.array(dict_similitud["sim_vals"])
                    vecinos_predictores = dict_similitud["vecinos_predictores"]
                    # variable_independiente_1 (primer predictor)
                    valores_x = np.array(
                        vecinos_predictores.get(predictor_1, []), dtype=float
                    )
                    if len(valores_x) == len(sim_vals) and sim_vals.sum() > 0:
                        x_similitud = float(
                            np.dot(sim_vals, valores_x) / sim_vals.sum()
                        )
                        dict_similitud["variable_independiente_1"] = x_similitud
                    else:
                        dict_similitud["variable_independiente_1"] = None
                    # variable_independiente_2 (segundo predictor, solo si modelo 3D)
                    if predictor_2:
                        valores_y = np.array(
                            vecinos_predictores.get(predictor_2, []), dtype=float
                        )
                        if len(valores_y) == len(sim_vals) and sim_vals.sum() > 0:
                            y_similitud = float(
                                np.dot(sim_vals, valores_y) / sim_vals.sum()
                            )
                            dict_similitud["variable_independiente_2"] = y_similitud
                        else:
                            dict_similitud["variable_independiente_2"] = None

            # --- CORRELACION ---
            if dict_correlacion is not None and predictor_1:
                try:
                    # variable_independiente_1: valor del predictor 1 de la aeronave objetivo
                    x_correlacion = float(df_procesado_base.at[aeronave, predictor_1])
                except Exception:
                    x_correlacion = None
                dict_correlacion["variable_independiente_1"] = x_correlacion
                # variable_independiente_2: valor del predictor 2 de la aeronave objetivo (solo si modelo 3D)
                if predictor_2:
                    try:
                        y_correlacion = float(
                            df_procesado_base.at[aeronave, predictor_2]
                        )
                    except Exception:
                        y_correlacion = None
                    dict_correlacion["variable_independiente_2"] = y_correlacion

            # --- FINAL (combinación por confianza) ---
            x_s = (
                dict_similitud.get("variable_independiente_1")
                if dict_similitud is not None
                else None
            )
            y_s = (
                dict_similitud.get("variable_independiente_2")
                if dict_similitud is not None
                else None
            )
            x_c = (
                dict_correlacion.get("variable_independiente_1")
                if dict_correlacion is not None
                else None
            )
            y_c = (
                dict_correlacion.get("variable_independiente_2")
                if dict_correlacion is not None
                else None
            )
            conf_s = (
                dict_similitud.get("Confianza", 0) if dict_similitud is not None else 0
            )
            conf_c = (
                dict_correlacion.get("Confianza", 0)
                if dict_correlacion is not None
                else 0
            )

            # Solo procesar variable_independiente_1/2 si hay elementos en detalles_iteracion
            if (
                len(detalles_iteracion) > 0
                and detalles_iteracion[-1].get("final") is not None
            ):
                # variable_independiente_1 (igual que antes)
                if x_s is not None and x_c is not None and (conf_s + conf_c) > 0:
                    x_final = float((x_s * conf_s + x_c * conf_c) / (conf_s + conf_c))
                    detalles_iteracion[-1]["final"][
                        "variable_independiente_1"
                    ] = x_final
                elif x_s is not None:
                    detalles_iteracion[-1]["final"]["variable_independiente_1"] = x_s
                elif x_c is not None:
                    detalles_iteracion[-1]["final"]["variable_independiente_1"] = x_c
                else:
                    detalles_iteracion[-1]["final"]["variable_independiente_1"] = None
                # variable_independiente_2 (solo si modelo 3D)
                if predictor_2:
                    if y_s is not None and y_c is not None and (conf_s + conf_c) > 0:
                        y_final = float(
                            (y_s * conf_s + y_c * conf_c) / (conf_s + conf_c)
                        )
                        detalles_iteracion[-1]["final"][
                            "variable_independiente_2"
                        ] = y_final
                    elif y_s is not None:
                        detalles_iteracion[-1]["final"][
                            "variable_independiente_2"
                        ] = y_s
                    elif y_c is not None:
                        detalles_iteracion[-1]["final"][
                            "variable_independiente_2"
                        ] = y_c
                    else:
                        detalles_iteracion[-1]["final"][
                            "variable_independiente_2"
                        ] = None
        # Aplicar las imputaciones finales al DataFrame base
        for imp in imputaciones_iteracion:
            parametro = imp["Parámetro"]
            aeronave = imp["Aeronave"]
            valor = imp["Valor imputado"]
            metodo = imp["Método predictivo"]
            df_procesado_base.at[aeronave, parametro] = valor
            # Solo imputar en df_filtrado si existen la fila y columna
            if (aeronave in df_filtrado.index) and (parametro in df_filtrado.columns):
                df_filtrado.at[aeronave, parametro] = valor
            resumen_imputaciones.append(imp)
            print(
                f"Imputación final aplicada: {parametro} - {aeronave} = {valor} ({metodo})"
            )

        # Al final de la iteración, acumula los resultados
        imputaciones_finales.extend(imputaciones_iteracion)
        detalles_para_excel.extend(detalles_iteracion)

        print(f"\n-- Resumen posterior a las imputaciones (iteracion {iteracion}) --")
        resumen_despues, total_faltantes_despues = generar_resumen_faltantes(
            df_procesado_base,
            titulo=f"Resumen de Valores Faltantes Después de Iteración {iteracion}",
        )

        # Criterio de parada configurable
        nuevas_validas = len(
            [
                imp
                for imp in imputaciones_iteracion
                if not is_missing(imp.get("Valor imputado", None))
            ]
        )
        if nuevas_validas < int(stop_cfg.get("min_nuevas_por_iter", 1)):
            _sin_mejora_consec = _sin_mejora_consec + 1
        else:
            _sin_mejora_consec = 0

        if _sin_mejora_consec >= int(stop_cfg.get("sin_mejora_consecutivas", 1)):
            print("\033[1mCriterio de parada: sin mejora suficiente.\033[0m")
            break

        print("\n" + "=" * 80)
        print(f"\033[1m=== Fin de la iteracion {iteracion} ===\033[0m")
        print("=" * 80)

    print("\n" + "=" * 80)
    print("\033[1m=== Resumen final del proceso ===\033[0m")
    print("=" * 80)

    print(f"\033[1mTotal de iteraciones realizadas: {iteracion}\033[0m")
    # Contar solo imputaciones válidas (no missing)
    imputaciones_validas = [
        imp
        for imp in resumen_imputaciones
        if not is_missing(imp.get("Valor imputado", None))
    ]
    print(f"\033[1mTotal de valores imputados: {len(imputaciones_validas)}\033[0m")
    print(
        "[INFO] El proceso termino. Si aun quedan celdas vacias, no se encontraron datos confiables en esta ejecucion."
    )

    # === EXPORTAR JSON OPTIMIZADO (estructura unificada por celda) ===
    import json

    try:
        import os

        exp_json = export_cfg.get(
            "json",
            {
                "enabled": True,
                "dir": "Results",
                "fname": "modelos_completos_por_celda.json",
            },
        )
        if exp_json.get("enabled", True):
            results_dir = os.path.join(
                os.path.dirname(__file__), "..", exp_json.get("dir", "Results")
            )
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

        # Estructura unificada: una sola entrada por celda con toda la información
        datos_unificados_por_celda = {}

        # Procesar información de imputación primero
        informacion_imputacion_por_celda = {}
        for detalle in detalles_para_excel:
            aeronave = detalle.get("Aeronave")
            parametro = detalle.get("Parámetro")
            if not aeronave or not parametro:
                continue

            key = f"{aeronave}|{parametro}"

            # Extraer información de imputación directamente
            campos_clave = [
                "Valor imputado",
                "Confianza",
                "Iteración imputación",
                "Método predictivo",
                "variable_independiente_1",
                "variable_independiente_2",
                "Advertencia",
            ]

            def extraer_info_imputacion(dic):
                if not dic:
                    return {}
                info = {}
                for campo in campos_clave:
                    if campo in dic:
                        info[campo] = dic[campo]
                return info

            informacion_imputacion_por_celda[key] = {
                "final": extraer_info_imputacion(detalle.get("final")),
                "similitud": extraer_info_imputacion(detalle.get("similitud")),
                "correlacion": extraer_info_imputacion(detalle.get("correlacion")),
            }

        # Procesar modelos y unificar con información de imputación
        for key, modelos in modelos_por_celda.items():
            if not modelos:
                continue

            idx, parametro = key.split("|")
            primer_modelo = modelos[0]

            # Crear estructura unificada para esta celda
            celda_unificada = {
                "informacion_generica_celda": {
                    "parametro_objetivo": parametro,
                    "idx_objetivo": idx,
                }
            }

            # Agregar df_original e información de normalización global
            if "df_original" in primer_modelo:
                celda_unificada["informacion_generica_celda"]["df_original"] = (
                    primer_modelo["df_original"]
                )

                # Incluir metadatos globales de normalización si existen
                if "datos_entrenamiento" in primer_modelo:
                    dt = primer_modelo["datos_entrenamiento"]
                    # Buscar información de normalización en cualquier parte del primer modelo
                    info_normalizacion = {}
                    for campo in ["x_min", "x_max", "y_min", "y_max"]:
                        if campo in dt:
                            info_normalizacion[campo] = dt[campo]

                    if info_normalizacion:
                        celda_unificada["informacion_generica_celda"][
                            "info_normalizacion_global"
                        ] = info_normalizacion

            # Agregar información de imputación a la información genérica
            if key in informacion_imputacion_por_celda:
                imputacion_info = informacion_imputacion_por_celda[key]
                celda_unificada["informacion_generica_celda"]["final"] = (
                    imputacion_info["final"]
                )
                celda_unificada["informacion_generica_celda"]["similitud"] = (
                    imputacion_info["similitud"]
                )
                celda_unificada["informacion_generica_celda"]["correlacion"] = (
                    imputacion_info["correlacion"]
                )

            # Procesar todos los modelos de esta celda
            modelos_celda = []
            for modelo in modelos:
                # Extraer información completa de datos_entrenamiento
                datos_ent = modelo.get("datos_entrenamiento", {})

                modelo_optimizado = {
                    # Información básica del modelo
                    "tipo": modelo.get("tipo", ""),
                    "predictores": modelo.get("predictores", []),
                    "n_predictores": modelo.get("n_predictores", 0),
                    "n_muestras_entrenamiento": modelo.get(
                        "n_muestras_entrenamiento", 0
                    ),
                    # Coeficientes y ecuaciones en escala original (AMBOS NECESARIOS)
                    "coeficientes_originales": modelo.get(
                        "coeficientes_originales", []
                    ),
                    "intercepto_original": modelo.get("intercepto_original", 0),
                    "ecuacion_string": modelo.get("ecuacion_string", ""),
                    # Métricas de evaluación (sin LOOCV que es redundante)
                    "mape": modelo.get("mape", 0),
                    "r2": modelo.get("r2", 0),
                    "corr": modelo.get("corr", 0),
                    "Confianza": modelo.get("Confianza", 0),
                    "Confianza_LOOCV": modelo.get("Confianza_LOOCV", 0),
                    "Corr_LOOCV": modelo.get("Corr_LOOCV", 0),
                    "MAPE_LOOCV": modelo.get("MAPE_LOOCV", 0),
                    "R2_LOOCV": modelo.get("R2_LOOCV", 0),
                    "Advertencia": modelo.get("Advertencia", None),
                    # Peso de predictores
                    "Peso de predictores": modelo.get("Peso de predictores", []),
                    # Transformación y método de imputación
                    "transformacion": modelo.get("tipo_transformacion", ""),
                    "metodo_imputacion": "correlacion",  # Especificar método usado
                    # Datos de entrenamiento COMPLETOS (CORREGIDOS)
                    "datos_entrenamiento": datos_ent,
                }

                # Copiar campos de variables para cálculo futuro (creados en imputacion_correlacion.py)
                # Orden coincide con orden de predictores: variable_independiente_1 = predictores[0], variable_independiente_2 = predictores[1]
                if "variable_independiente_1" in modelo:
                    modelo_optimizado["variable_independiente_1"] = modelo[
                        "variable_independiente_1"
                    ]
                if "variable_independiente_2" in modelo:
                    modelo_optimizado["variable_independiente_2"] = modelo[
                        "variable_independiente_2"
                    ]

                # Los valores de variables para cálculo futuro se agregan en imputacion_correlacion.py
                # donde se tienen los datos correctos del entrenamiento

                modelos_celda.append(modelo_optimizado)

            # Agregar modelos a la estructura unificada como lista
            celda_unificada["informacion_modelos_celda"] = {"modelos": modelos_celda}

            # Guardar en estructura final
            datos_unificados_por_celda[key] = celda_unificada

        # Exportar JSON con estructura unificada
        export_dict_unificado = datos_unificados_por_celda
        if exp_json.get("enabled", True):
            output_path = os.path.join(
                results_dir, exp_json.get("fname", "modelos_completos_por_celda.json")
            )
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(
                    export_dict_unificado,
                    f,
                    ensure_ascii=False,
                    indent=2,
                    separators=(",", ": "),
                )
            print(f"📝 [DEBUG] Archivo JSON unificado exportado a: {output_path}")
            print(
                f"📊 [DEBUG] Estructura: {len(datos_unificados_por_celda)} celdas con información completa"
            )

    except Exception as e:
        print(f"[WARNING] No se pudo exportar el archivo JSON: {e}")

    # Exportación Excel/HTML: pasar configuración de excel si las funciones lo soportan en el flujo superior.
    # Guardar snapshot de configuración junto al resultado final
    try:
        out_dir = os.path.join(os.path.dirname(__file__), "..", "Results")
        os.makedirs(out_dir, exist_ok=True)
        # Cargar cfg utilizado desde ámbito superior si existe, o volver a generarlo localmente
        cfg_local = locals().get("cfg")
        cfg_final: dict = (
            cfg_local if isinstance(cfg_local, dict) else _cargar_configuracion()
        )
        save_config_snapshot(cfg_final, out_dir, prefix="config_usada")
    except Exception:
        pass

    # Restaurar consola si fue redirigida
    if _restore_streams:
        try:
            sys.stdout = _old_stdout  # type: ignore[assignment]
            sys.stderr = _old_stderr  # type: ignore[assignment]
        except Exception:
            pass

    return (
        df_procesado_base,
        pd.DataFrame(resumen_imputaciones),
        imputaciones_finales,
        detalles_para_excel,
        modelos_por_celda,
    )


# === API pública para el controlador/UX ===
def ejecutar_pipeline(cfg: dict | None = None) -> None:
    """
    Punto de entrada único del pipeline para el controlador/UX notebook.
    - Si cfg es None, carga config efectiva (defaults + overrides).
    - Ejecuta el bucle usando SIEMPRE la config (nada por CLI).
    """
    # Importar helpers de config de controller (prioritario) para respetar overrides/snapshots
    try:
        from Modulos.controller import load_effective_config, get_config  # type: ignore
    except Exception:
        try:
            from controller import load_effective_config, get_config  # type: ignore
        except Exception:
            # Fallback mínimo al config local (compat)
            from .config_and_loading import get_config as get_config  # type: ignore

            load_effective_config = None  # type: ignore

    # Cargar cfg efectiva
    if cfg is None:
        if "load_effective_config" in locals() and callable(
            locals().get("load_effective_config")
        ):
            cfg = load_effective_config()  # type: ignore[operator]
        else:
            # Fallback: usar config local por defecto
            from .config_and_loading import default_config

            cfg = default_config()
    cfg = get_config(cfg)

    # Preparar datos como en main.py pero sin interacción CLI
    try:
        # Preferir las utilidades existentes
        from .config_and_loading import configurar_entorno, cargar_datos
        from .data_processing import procesar_datos_y_manejar_duplicados
        from .derivados import completar_campos_derivados
    except Exception as _e:
        print(f"[WARN] No se pudieron importar utilidades de carga/procesamiento: {_e}")
        return

    try:
        # Display y entorno
        configurar_entorno(
            max_rows=cfg.get("entorno", {}).get("max_rows", 200),
            max_columns=cfg.get("entorno", {}).get("max_columns", 120),
        )

        # Cargar Excel desde cfg (evitar prompt)
        ruta_excel = cfg.get("entorno", {}).get("ruta_excel")
        df_inicial, ruta_archivo = cargar_datos(ruta_archivo=ruta_excel)
        print(f"✅ Datos cargados desde: {ruta_archivo}")

        # Procesamiento base + derivados (con defaults seguros)
        df_procesado = procesar_datos_y_manejar_duplicados(df_inicial)
    except Exception as e:
        print(f"[ERROR] Falló la carga/procesamiento: {e}")
        return

    try:
        df_procesado, _ = completar_campos_derivados(
            df=df_procesado,
            solo_completar_vacios=True,
            usar_IAS_para_alcance=True,
        )
    except Exception as e:
        print(f"[WARN] completar_campos_derivados falló o no está disponible: {e}")

    # Selección de parámetros basada en cfg (familias) o fallback a intersección/numéricos
    try:
        familias_cfg = cfg.get("similitud", {}).get("familias", {}) or {}
        usadas = cfg.get("similitud", {}).get(
            "familias_usadas", list(familias_cfg.keys())
        )
        candidatos = []
        for fam in usadas:
            lst = familias_cfg.get(fam, [])
            if isinstance(lst, (list, tuple)):
                candidatos.extend(list(lst))
        columnas = list(df_procesado.columns)
        parametros_preseleccionados = [c for c in candidatos if c in columnas]
        if not parametros_preseleccionados:
            # Fallback: usar columnas numéricas o todas si no hay numéricas
            try:
                import pandas as _pd  # local

                parametros_preseleccionados = _pd.Index(columnas)[
                    _pd.Series(columnas).map(
                        lambda c: _pd.api.types.is_numeric_dtype(df_procesado[c])
                    )
                ].tolist()
            except Exception:
                parametros_preseleccionados = columnas
        df_filtrado = df_procesado[parametros_preseleccionados].copy()
    except Exception as e:
        print(f"[WARN] No se pudo construir df_filtrado desde familias: {e}")
        df_filtrado = df_procesado.copy()
        parametros_preseleccionados = list(df_procesado.columns)

    # Capas de familia (no críticas en el loop actual)
    capas_familia = [
        ["Misión", "Despegue", "Propulsión vertical", "Propulsión horizontal"],
        ["Misión", "Despegue"],
        ["Misión"],
    ]

    # Ejecutar el bucle principal con config actual (el propio bucle lee cfg internamente)
    try:
        # Firma moderna (solo cfg): si existiera en tu repo, descomenta y usa
        bucle_imputacion_similitud_correlacion(cfg)  # type: ignore[arg-type]
    except TypeError:
        # Firma actual: requiere dataframes y otros parámetros
        try:
            bucle_imputacion_similitud_correlacion(
                df_filtrado=df_filtrado,
                parametros_preseleccionados=parametros_preseleccionados,
                capas_familia=capas_familia,
                df_procesado=df_procesado,
                debug_mode=bool(cfg.get("entorno", {}).get("debug_mode", False)),
                permitir_sin_filtro=bool(
                    cfg.get("orquestacion", {}).get("permitir_sin_filtro", False)
                ),
            )
        except Exception as e:
            print(f"[ERROR] Falló el bucle de imputación: {e}")
