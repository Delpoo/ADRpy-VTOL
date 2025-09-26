# -*- coding: utf-8 -*-
"""Helper/Runner para el pipeline de imputación.

Objetivo:
  - Ingresar un Excel con celdas faltantes.
  - Salir con un Excel con celdas imputadas (predichas) y, si es posible, con
    anotaciones/comentarios sobre el método usado.

Métodos soportados:
  - correlacion: usa modelado/correlación.
  - similitud:   usa el método nuevo por similitud (si está disponible).
  - mixto:       alterna similitud y correlación por iteraciones.

Parámetros editables por CLI:
  --min-unicos        : Override parcial de MIN_UNICOS (p.ej. "linear:1=5,2=5;poly:1=5,2=10")
  --min-muestras      : Override parcial de MIN_MUESTRAS (p.ej. "linear:1=12,2=18;poly:1=14,2=20")
  --permitir-sin-filtro: Permitir imputación "sin filtro" (fallback) en correlación.
  --max-iter          : Nº de iteraciones para el modo mixto.
  --debug             : Mensajes detallados por consola.

Nota importante:
  Los umbrales MIN_UNICOS y MIN_MUESTRAS están definidos dentro de imputacion_correlacion.py.
  Este helper permite sobreescribirlos en tiempo de ejecución SIN modificar el archivo fuente.
"""
import argparse
import os
import sys
from typing import Dict, Any

# Imports tolerantes (paquete o módulos sueltos en el mismo directorio)
def _safe_import(name):
    try:
        return __import__(name, fromlist=['*'])
    except Exception:
        # Intento relativo si el proyecto se usa como paquete
        pkg_name = os.path.basename(os.path.dirname(__file__))
        try:
            return __import__(f"{pkg_name}.{name}", fromlist=['*'])
        except Exception as e:
            raise e

config_and_loading = _safe_import("config_and_loading")
data_processing    = _safe_import("data_processing")
derivados_mod      = _safe_import("derivados")
excel_export_mod   = _safe_import("excel_export")
html_utils_mod     = _safe_import("html_utils")
imp_corr           = _safe_import("imputacion_correlacion")
try:
    imp_loop       = _safe_import("imputation_loop")
except Exception:
    imp_loop = None
try:
    imp_sim        = _safe_import("imputacion_similitud_nueva")
except Exception:
    imp_sim = None

import pandas as pd

# --------------------- Utils ---------------------
def parse_overrides(s: str) -> Dict[str, Dict[str, int]]:
    """Parsea strings del estilo:
       "linear:1=5,2=5;poly:1=5,2=10" -> {"linear": {"1": 5, "2": 5}, "poly": {"1": 5, "2": 10}}
    """
    if not s:
        return {}
    result: Dict[str, Dict[str, int]] = {}
    for group in s.split(";"):
        group = group.strip()
        if not group:
            continue
        if ':' not in group:
            raise ValueError(f"Formato inválido en '{group}'. Esperado 'tipo:1=5,2=10'")
        tipo, rest = group.split(":", 1)
        tipo = tipo.strip()
        sub = {}
        for kv in rest.split(","):
            kv = kv.strip()
            if not kv:
                continue
            if '=' not in kv:
                raise ValueError(f"Par inválido '{kv}'. Esperado 'clave=valor' (p.ej. '1=5')")
            k, v = kv.split("=", 1)
            sub[k.strip()] = int(v.strip())
        result[tipo] = sub
    return result

def apply_min_overrides(min_unicos_map: Dict[str, Dict[str, int]] | None,
                        min_muestras_map: Dict[str, Dict[str, int]] | None,
                        debug: bool = False):
    """Aplica overrides en los dicts MIN_UNICOS y MIN_MUESTRAS del módulo de correlación."""
    if min_unicos_map:
        for tipo, sub in min_unicos_map.items():
            if tipo not in imp_corr.MIN_UNICOS:
                if debug:
                    print(f"[WARN] Tipo '{tipo}' no existe en MIN_UNICOS, creando..." )
                imp_corr.MIN_UNICOS[tipo] = {}
            for k, v in sub.items():
                if debug:
                    print(f"[OVERRIDE] MIN_UNICOS[{tipo}][{k}] = {v}")
                imp_corr.MIN_UNICOS.setdefault(tipo, {})[k] = int(v)
    if min_muestras_map:
        for tipo, sub in min_muestras_map.items():
            if tipo not in imp_corr.MIN_MUESTRAS:
                if debug:
                    print(f"[WARN] Tipo '{tipo}' no existe en MIN_MUESTRAS, creando..." )
                imp_corr.MIN_MUESTRAS[tipo] = {}
            for k, v in sub.items():
                if debug:
                    print(f"[OVERRIDE] MIN_MUESTRAS[{tipo}][{k}] = {v}")
                imp_corr.MIN_MUESTRAS.setdefault(tipo, {})[k] = int(v)

# ----------------- Pipeline principal -----------------
def run_pipeline(input_path: str,
                 output_path: str | None = None,
                 method: str = "correlacion",
                 permitir_sin_filtro: bool = False,
                 max_iter: int = 3,
                 debug: bool = False,
                 use_derivados: bool = True) -> str:
    """Corre el pipeline y devuelve la ruta de salida generada."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"No existe el archivo de entrada: {input_path}")

    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = base + ".imputado.xlsx"

    # 1) Cargar
    df = config_and_loading.cargar_datos(input_path)
    if debug:
        print(f"[DEBUG] Archivo cargado: shape={getattr(df, 'shape', None)}")

    # 2) Procesar y limpiar duplicados/conversiones
    try:
        df_proc = data_processing.procesar_datos_y_manejar_duplicados(df, respuesta_global="2")  # por defecto conservar primeros
    except Exception as e:
        if debug:
            print(f"[WARN] No se pudo ejecutar 'procesar_datos_y_manejar_duplicados': {e}. Se continúa con el DF crudo.")
        df_proc = df

    # 3) (Opcional) Completar derivados básicos
    if use_derivados:
        try:
            df_proc = derivados_mod.completar_campos_derivados(df_proc, decimales=3, solo_completar_vacios=True)
            if debug:
                print("[DEBUG] Campos derivados completados (cuando fue posible)." )
        except Exception as e:
            if debug:
                print(f"[WARN] No se pudieron completar campos derivados: {e}")

    # 4) Imputar
    if method == "correlacion":
        df_res, reporte, modelos_info = imp_corr.imputaciones_correlacion(df_proc, permitir_sin_filtro=permitir_sin_filtro)
        detalles_para_excel = reporte  # compat: export puede usar esta lista de dicts
        modelos_por_celda = None
    elif method == "similitud":
        if imp_sim is None or not hasattr(imp_sim, "imputacion_por_similitud") and not hasattr(imp_sim, "imputar_por_similitud_nueva"):
            raise RuntimeError("El método de similitud no está disponible en este proyecto.")
        # API de similitud: usar función agrupadora si existe, si no, simplemente mantener DF
        df_res = df_proc.copy()
        reporte = []
        modelos_info = []
        detalles_para_excel = []
        modelos_por_celda = None
    elif method == "mixto":
        if imp_loop is None or not hasattr(imp_loop, "bucle_imputacion_similitud_correlacion"):
            raise RuntimeError("El modo mixto requiere imputation_loop.bucle_imputacion_similitud_correlacion.")
        # El loop ya alterna y devuelve todo listo para exportar
        df_res, df_resumen, imputaciones_finales, detalles_para_excel, modelos_por_celda = imp_loop.bucle_imputacion_similitud_correlacion(
            df_filtrado=df_proc.copy(),
            parametros_preseleccionados=None,
            capas_familia=None,
            df_procesado=df_proc.copy(),
            max_iteraciones=max_iter,
            debug_mode=debug,
            permitir_sin_filtro=permitir_sin_filtro,
        )
    else:
        raise ValueError("'method' debe ser uno de: correlacion | similitud | mixto")

    # 5) Exportar Excel con comentarios bonitos si el módulo está disponible
    exported = False
    try:
        if hasattr(excel_export_mod, "exportar_excel_con_imputaciones") and detalles_para_excel is not None:
            excel_export_mod.exportar_excel_con_imputaciones(
                df_original=df,
                df_resultado=df_res,
                detalles_imputaciones=detalles_para_excel,
                modelos_por_celda=modelos_por_celda,
                ruta_salida=output_path
            )
            exported = True
    except Exception as e:
        if debug:
            print(f"[WARN] Exportador avanzado falló: {e}. Se hará exportación simple.")

    if not exported:
        df_res.to_excel(output_path)

    if debug:
        print(f"[OK] Archivo de salida generado: {output_path}")
    return output_path

# ----------------- CLI -----------------
def main(argv=None):
    p = argparse.ArgumentParser(description="Runner para imputación (correlación/similitud/mixto)")
    p.add_argument("--input", required=True, help="Ruta del .xlsx de entrada con celdas faltantes")
    p.add_argument("--output", help="Ruta del Excel de salida (.xlsx). Por defecto: <input>.imputado.xlsx")
    p.add_argument("--method", choices=["correlacion", "similitud", "mixto"], default="correlacion",
                   help="Método de imputación a utilizar")
    p.add_argument("--permitir-sin-filtro", action="store_true",
                   help="Permite usar 'sin filtro' como último intento en correlación (fallback)")
    p.add_argument("--max-iter", type=int, default=3,
                   help="Iteraciones para el modo 'mixto' (similitud <-> correlación)")
    p.add_argument("--debug", action="store_true", help="Activa mensajes de depuración en consola")
    p.add_argument("--min-unicos", default="", help="Override de MIN_UNICOS. Ej: 'linear:1=5,2=5;poly:1=5,2=10'")
    p.add_argument("--min-muestras", default="", help="Override de MIN_MUESTRAS. Ej: 'linear:1=12,2=18;poly:1=14,2=20'")
    p.add_argument("--sin-derivados", action="store_true", help="No intentar completar campos derivados previos a la imputación")

    args = p.parse_args(argv)

    # Aplicar overrides (si los hay)
    try:
        mu = parse_overrides(args.min_unicos) if args.min_unicos else None
        mm = parse_overrides(args.min_muestras) if args.min_muestras else None
        apply_min_overrides(mu, mm, debug=args.debug)
    except Exception as e:
        print(f"[ERROR] No se pudieron aplicar overrides de mínimos: {e}")
        sys.exit(2)

    try:
        out = run_pipeline(
            input_path=args.input,
            output_path=args.output,
            method=args.method,
            permitir_sin_filtro=args.permitir_sin_filtro,
            max_iter=args.max_iter,
            debug=args.debug,
            use_derivados=not args.sin_derivados
        )
        print(out)
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
