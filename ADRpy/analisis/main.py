# ===================== #
#    IMPORTACIONES      #
# ===================== #

# Librerías estándar
import os

# Librerías externas
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
#    IMPORTAR MÓDULOS   #
# ===================== #

from ADRpy.analisis.Modulos.config_and_loading import configurar_entorno, cargar_datos
from ADRpy.analisis.Modulos.data_processing import procesar_datos_y_manejar_duplicados
from ADRpy.analisis.Modulos.user_interaction import seleccionar_parametros_por_indices, solicitar_umbral
from ADRpy.analisis.Modulos.correlation_analysis import calcular_correlaciones_y_generar_heatmap_con_resumen
from ADRpy.analisis.Modulos.similarity_imputation import imputacion_similitud_con_rango, imprimir_detalles_imputacion
from ADRpy.analisis.Modulos.correlation_imputation import Imputacion_por_correlacion
from ADRpy.analisis.Modulos.imputation_loop import bucle_imputacion_similitud_correlacion
from ADRpy.analisis.Modulos.excel_export import exportar_excel_con_imputaciones
from ADRpy.analisis.Modulos.html_utils import convertir_a_html


# Solicitar la ruta del archivo al usuario
archivo_origen = input("Ingrese la ruta del archivo Excel original: ")

# Paso 1: Configurar entorno
configurar_entorno(max_rows=20, max_columns=10)

# Paso 2: Cargar datos
try:
    df_inicial, ruta_archivo = cargar_datos(archivo_origen)  # Aquí se valida la entrada
    print(f"Datos cargados correctamente desde: {ruta_archivo}")
except ValueError as e:
    print(f"Error al cargar datos: {e}")
    exit(1)  # Detiene el programa si hay un error

# Validar que los datos se hayan cargado correctamente
print("\n=== Validando datos cargados ===")
if df_inicial.empty:
    print("El archivo cargado no contiene datos. Verifica el archivo y vuelve a intentarlo.")
    exit(1)

# Continuar con el siguiente paso solo si los datos son válidos
print("\n=== Continuando con el procesamiento de datos ===")

# Validar encabezados iniciales
print("\nEncabezados iniciales cargados:")
print(df_inicial.columns.tolist())

# Paso 3: Mostrar datos iniciales en HTML
print("\n=== Mostrando datos iniciales en formato HTML ===")
convertir_a_html(df_inicial, titulo="Datos Iniciales", mostrar=True)

# Paso 4: Procesar datos
print("\n=== Procesando los datos ===")
df_procesado = procesar_datos_y_manejar_duplicados(df_inicial)

# Validar encabezados después del procesamiento
#print("\nEncabezados después del procesamiento:")
#print(df_procesado.columns.tolist())

# Comparar encabezados antes y después del procesamiento
if df_inicial.columns.tolist() == df_procesado.columns.tolist():
    print("\n✅ Los encabezados se preservaron correctamente.")
else:
    print("\n❌ Los encabezados fueron modificados durante el procesamiento.")

# Paso 5: Mostr en HTML
print("\n=== Mostrando datos procesados en formato HTML ===")
convertir_a_html(df_procesado, titulo="Datos Procesados", mostrar=True)

# Paso 6: Selección de parámetros

# Parámetros disponibles en el índice del DataFrame
parametros_disponibles = df_procesado.index.tolist()
print("Parámetros disponibles en df_procesado antes de seleccionar:")
print(parametros_disponibles)

# Parámetros preseleccionados de interés
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
# Filtrar preseleccionados para mantener solo los parámetros válidos
parametros_preseleccionados = [p for p in parametros_preseleccionados if p in parametros_disponibles]

# Imprimir parámetros preseleccionados válidos
#print("Parámetros preseleccionados válidos:")
#print(parametros_preseleccionados)

parametros_seleccionados = seleccionar_parametros_por_indices(parametros_disponibles, parametros_preseleccionados)
# Imprimir parámetros seleccionados después de filtrar
print("Parámetros seleccionados después de filtrar:")
print(parametros_seleccionados)

# Filtrar el DataFrame por los parámetros seleccionados
try:
    df_filtrado = df_procesado.loc[parametros_seleccionados]
except KeyError as e:
    print(f"Error al filtrar df_procesado: {e}")
    print(f"Parámetros seleccionados inválidos: {set(parametros_seleccionados) - set(df_procesado.index.tolist())}")
    raise

# Mostrar la tabla en formato HTML con 3 cifras significativas (sin notación científica)
convertir_a_html(df_filtrado, titulo="Datos Filtrados por Parámetros (df_filtrado)", mostrar=True)

# Paso 7: Mostrar celdas faltantes con selección de columna

# Analizar celdas faltantes en la columna seleccionada
df_celdas_faltantes = mostrar_celdas_faltantes_con_seleccion(df_filtrado)

# Verificar si hay celdas faltantes
if df_celdas_faltantes.empty:
    print("No se encontraron valores faltantes en la columna seleccionada.")
else:
    # Mostrar resultados en formato HTML
    convertir_a_html(df_celdas_faltantes, titulo="Celdas Faltantes Identificadas en df_filtrado (df_celdas_faltantes)", mostrar=True)
    
# Paso 8: Resumen de valores faltantes por columna
print("\n=== Generando resumen de valores faltantes por columna ===")
resumen_faltantes = generar_resumen_faltantes(df_filtrado, titulo="Resumen de Valores Faltantes de df_filtrado")

# Paso 9: Calculando correlaciones y generando heatmap
print("\n=== Calculando correlaciones y generando heatmap ===")
tabla_completa = calcular_correlaciones_y_generar_heatmap_con_resumen(df_procesado, parametros_seleccionados)
    
# Paso 10: Ajustar rango e imputar valores faltantes
#print("\n=== Paso 8: Imputación con ajuste de rango ===")
#imputacion_similitud_con_rango(df_filtrado, df_procesado)
 #Paso 11: Ajustar rango e imputar valores faltantes por correlación
#Imputacion_por_correlacion(df_procesado, parametros_preseleccionados, tabla_completa, umbral_correlacion=0.7, min_datos_validos=5, max_lineas_consola=250)

# Paso 10: Llamar a la función principal
df_procesado_actualizado, resumen_imputaciones = bucle_imputacion_similitud_correlacion(df_procesado, parametros_preseleccionados, tabla_completa, reduccion_confianza=0.05, max_iteraciones=7)

# Paso 11: Exportar resultados a Excel
archivo_destino = input("Ingrese la ruta donde desea guardar el archivo con las imputaciones (incluya .xlsx): ")
exportar_excel_con_imputaciones(
    archivo_origen=ruta_archivo,
    df_procesado=df_procesado_actualizado,
    resumen_imputaciones=resumen_imputaciones
)
print("\n=== Flujo completado. Verifique el archivo generado. ===")

