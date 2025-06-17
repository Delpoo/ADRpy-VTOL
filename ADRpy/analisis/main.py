"""
Main script for the ADRpy imputation and export workflow.
All comments and print statements are in English for clarity.
"""

#para depurar con Test frame

#para depurar con original dataframe
#cambio en nombre de parametros preseleccionados (main.py)
#cambio en definicion de bloques_rasgos (imputacion_similitud_flexible.py)

# TODO: Esto es algo que tengo que hacer más adelante
# FIXME: Esto está fallando o tiene errores
# ! Esto es importante
# ? Esto es una duda o algo que quiero revisar
# * Esto es algo que quiero hacer
# Normal comment: Este es un comentario normal

import argparse

# ===================== #
#   ARGUMENT PARSER    #
# ===================== #
# Parse command-line arguments for flexible script execution
parser = argparse.ArgumentParser(description="Run the ADRpy script with optional parameters")
parser.add_argument("--ruta_archivo", type=str, help="Path to the original Excel file", default=None)
parser.add_argument("--archivo_destino", type=str, help="Path to save the exported Excel file", default=None)
parser.add_argument("--debug_mode", action="store_true")
parser.add_argument("--parametros", type=str, help="Selected parameters as a list of indices (e.g. '4,5,6')", default=None)
parser.add_argument("--columna", type=str, help="Name of the column to analyze missing cells", default=None)

args = parser.parse_args()


# ===================== #
#      IMPORTS          #
# ===================== #
# Standard and external library imports for data processing, visualization, and Excel handling
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.comments import Comment
from tkinter import simpledialog, messagebox
import tkinter as tk
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from IPython.display import display, HTML

# ===================== #
#   MODULE IMPORTS      #
# ===================== #
# Import custom modules for each step of the workflow
from Modulos.config_and_loading import configurar_entorno, cargar_datos, normalizar_encabezados
from Modulos.data_processing import procesar_datos_y_manejar_duplicados
from Modulos.user_interaction import seleccionar_parametros_por_indices, solicitar_umbral
from Modulos.imputation_loop import bucle_imputacion_similitud_correlacion
from Modulos.excel_export import exportar_excel_con_imputaciones
from Modulos.html_utils import convertir_a_html
from Modulos.data_processing import mostrar_celdas_faltantes_con_seleccion, generar_resumen_faltantes
from Modulos.imputacion_similitud_flexible  import configurar_similitud 
from Modulos.imputacion_similitud_flexible import imputar_por_similitud

# Step 1: Configure environment for pandas display (limits rows/columns in output)
configurar_entorno(max_rows=20, max_columns=10)

# Step 2: Load data from Excel file (path can be provided as argument)
try:
    df_inicial, ruta_archivo = cargar_datos(ruta_archivo=args.ruta_archivo)
    print(f"Data loaded successfully from: {ruta_archivo}")
except ValueError as e:
    print(f"Error loading data: {e}")
    exit(1)  # Detiene el programa si hay un error

# Step 3: Validate loaded data to ensure it is not empty
print("\n=== Validating loaded data ===")
if df_inicial.empty:
    print("The loaded file contains no data. Please check the file and try again.")
    exit(1)

print("\n=== Proceeding with data processing ===")
print("\nInitial headers loaded:")
print(df_inicial.columns.tolist())

# Step 4: Show initial data in HTML for visual inspection
print("\n=== Displaying initial data in HTML format ===")
convertir_a_html(df_inicial, titulo="Initial Data", mostrar=True)

# Step 5: Process data (handle duplicates, clean up, etc.)
print("\n=== Processing data ===")
df_procesado = procesar_datos_y_manejar_duplicados(df_inicial)
df_original_para_analisis = df_inicial.copy()  # ← NUEVO: Guardar para análisis visual

# Compare headers before and after processing to ensure consistency
if df_inicial.columns.tolist() == df_procesado.columns.tolist():
    print("\n✅ Headers were preserved correctly.")
else:
    print("\n❌ Headers were modified during processing.")

# Step 6: Show processed data in HTML for review
print("\n=== Displaying processed data in HTML format ===")
convertir_a_html(df_procesado, titulo="Processed Data", mostrar=True)

# Step 7: Parameter selection (choose which columns to use for imputation)
parameters_available = df_procesado.columns.tolist()
print("Parameters available in df_procesado before selection:")
print(parameters_available)

# Preselect parameters of interest (can be customized)
parametros_preseleccionados = [
    "Velocidad a la que se realiza el crucero (KTAS)",
    "Techo de servicio máximo",
    "Área del ala",
    "Relación de aspecto del ala",
    "Longitud del fuselaje",
    "Peso máximo al despegue (MTOW)",
    "Alcance de la aeronave",
    "Autonomía de la aeronave",
    "Velocidad máxima (KIAS)",
    "Velocidad de pérdida (KCAS)",
    "envergadura",
    "Cuerda",
    "payload",
    "Empty weight"
]
"""
parametros_preseleccionados = [
    "MTOW",
    "Payload",
    "Potencia",
    "Envergadura",
    "Alcance",
    "Velocidad crucero",
    "Cantidad de motores",
    "Ancho de fuselaje",
    "Rango de comunicación",
]
"""
# Filter preselected parameters to keep only those present in the data
parametros_preseleccionados = [p for p in parametros_preseleccionados if p in parameters_available]

# Allow user to select parameters via command-line or GUI
parametros_seleccionados = seleccionar_parametros_por_indices(parameters_available, parametros_preseleccionados, args.parametros)
print("Parameters selected after filtering:")
print(parametros_seleccionados)

# Filter DataFrame by selected parameters
try:
    df_filtrado = df_procesado[parametros_seleccionados]
except KeyError as e:
    print(f"Error filtering df_procesado: {e}")
    print(f"Invalid selected parameters: {set(parametros_seleccionados) - set(df_procesado.index.tolist())}")
    raise

# Show filtered table in HTML with 3 significant digits (no scientific notation)
convertir_a_html(df_filtrado, titulo="Data Filtered by Parameters (df_filtrado)", mostrar=True)

# Step 8: Show missing cells with column selection (visualize missing data)
df_celdas_faltantes = mostrar_celdas_faltantes_con_seleccion(
    df_filtrado,
    fila_seleccionada=args.columna,
    debug_mode=args.debug_mode
)

# If there are missing cells, display them in HTML
if df_celdas_faltantes.empty:
    print("No missing values found in the selected column.")
else:
    convertir_a_html(df_celdas_faltantes, titulo="Missing Cells Identified in df_filtrado (df_celdas_faltantes)", mostrar=True)

# Step 9: Generate summary of missing values by column
print("\n=== Generating summary of missing values by column ===")
generar_resumen_faltantes(df_filtrado, titulo="Summary of Missing Values in df_filtrado")

# Step 10: Load similarity configuration (for flexible imputation)
bloques_rasgos, filas_familia, capas_familia = configurar_similitud()

# Select columns for df_atributos and df_parametros
df_atributos = df_procesado[filas_familia]
df_parametros = df_procesado.drop(columns=filas_familia)

# Step 11: Main imputation loop (alternates similarity and correlation methods)
# Returns processed DataFrame, summary, final imputations, and details for Excel export

# Usar bucle con generación de diccionarios para análisis visual
try:
    from Modulos.imputation_loop_con_diccionarios import bucle_imputacion_similitud_correlacion_con_diccionarios
    print("🔧 Usando bucle con generación de diccionarios...")
    
    resultado_bucle = bucle_imputacion_similitud_correlacion_con_diccionarios(
        df_parametros=df_parametros,
        df_atributos=df_atributos,
        parametros_preseleccionados=parametros_preseleccionados,
        bloques_rasgos=bloques_rasgos,
        capas_familia=capas_familia,
        df_procesado=df_procesado,
        debug_mode=args.debug_mode,
        generar_diccionarios=True  # ← Activar generación de diccionarios
    )
    
    if len(resultado_bucle) == 5:
        # Con diccionarios
        df_procesado_actualizado, resumen_imputaciones, imputaciones_finales, detalles_para_excel, diccionarios_modelos_globales = resultado_bucle
        print(f"✅ Diccionarios generados: {len(diccionarios_modelos_globales)} celdas")
    else:
        # Sin diccionarios (fallback)
        df_procesado_actualizado, resumen_imputaciones, imputaciones_finales, detalles_para_excel = resultado_bucle
        diccionarios_modelos_globales = {}
        print("⚠️ No se generaron diccionarios - usando modo fallback")
        
except ImportError as e:
    print(f"⚠️ Módulo con diccionarios no disponible ({e}) - usando versión original")
    from Modulos.imputation_loop import bucle_imputacion_similitud_correlacion
    
    df_procesado_actualizado, resumen_imputaciones, imputaciones_finales, detalles_para_excel = bucle_imputacion_similitud_correlacion(
        df_parametros=df_parametros,
        df_atributos=df_atributos,
        parametros_preseleccionados=parametros_preseleccionados,
        bloques_rasgos=bloques_rasgos,
        capas_familia=capas_familia,
        df_procesado=df_procesado,
        debug_mode=args.debug_mode
    )
    diccionarios_modelos_globales = {}

# ═══════════════════════════════════════════════════════════════════════════════
# ANÁLISIS VISUAL DE MODELOS (NUEVO)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("🎯 ANÁLISIS VISUAL DE MODELOS DE CORRELACIÓN")
print("=" * 60)

try:
    if 'diccionarios_modelos_globales' in locals() and diccionarios_modelos_globales:
        print(f"✅ Diccionarios disponibles: {len(diccionarios_modelos_globales)} celdas")
        
        # Guardar en namespace global para el notebook
        import __main__
        __main__.df_original_main = df_original_para_analisis
        __main__.diccionarios_modelos_main = diccionarios_modelos_globales  
        __main__.df_resultado_main = df_procesado_actualizado
        __main__.detalles_excel_main = detalles_para_excel
        
        print("📊 Variables guardadas para análisis visual:")
        print("  - df_original_main")
        print("  - diccionarios_modelos_main")
        print("  - df_resultado_main") 
        print("  - detalles_excel_main")
        
        print("\n🎮 INSTRUCCIONES PARA ANÁLISIS VISUAL:")
        print("=" * 45)
        print("1. Abra Jupyter: analisis_modelos_imputacion.ipynb")
        print("2. Ejecute las celdas de importación y definición de clases")
        print("3. Ejecute esta celda para cargar los datos:")
        print()
        print("```python")
        print("# Cargar datos del flujo principal")
        print("datos_cargados = analizador.cargar_desde_bucle_imputacion(")
        print("    df_original_main,")
        print("    detalles_excel_main,")
        print("    diccionarios_modelos_main")
        print(")")
        print()
        print("if datos_cargados is not None:")
        print("    print('✅ Datos cargados desde main.py')")
        print("    # Crear y mostrar interfaz")
        print("    interfaz = InterfazInteractiva(analizador)")
        print("    interfaz.mostrar_interfaz_completa()")
        print("else:")
        print("    print('❌ Error cargando datos')")
        print("```")
        print()
        print("4. Use la interfaz interactiva para explorar modelos")
        
    else:
        print("⚠️ No hay diccionarios de modelos disponibles")
        print("💡 El análisis visual requiere diccionarios del bucle de imputación")
        
except Exception as e:
    print(f"❌ Error en análisis visual: {e}")
    print("💡 Continuando con el flujo normal...")

print("\n" + "=" * 60)

# Step 12: Export results to Excel with color and comments for each imputed cell
archivo_destino = args.archivo_destino
if not archivo_destino:
    archivo_destino = input("Enter the path where you want to save the file with imputations (include .xlsx): ")
if not archivo_destino:
    archivo_destino = r"C:\Users\delpi\OneDrive\Tesis\ADRpy-VTOL\ADRpy\analisis\Results\Datos_imputados.xlsx"

# Call the export function with the correct parameter names
exportar_excel_con_imputaciones(
    source_file=ruta_archivo,
    df_processed=df_procesado_actualizado,
    details_for_excel=detalles_para_excel,
    output_file=archivo_destino
)

print("\n=== Workflow completed. Please check the generated file. ===")
print("✅ Script finished.")

