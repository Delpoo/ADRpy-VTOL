import pandas as pd
import tkinter as tk
from tkinter import simpledialog, messagebox
import sys


def configurar_entorno(max_rows=20, max_columns=10):
    """
    Configura el entorno para mostrar más datos en la consola.
    :param max_rows: Número máximo de filas para mostrar en consola.
    :param max_columns: Número máximo de columnas para mostrar en consola.
    """
    pd.set_option("display.max_rows", max_rows)
    pd.set_option("display.max_columns", max_columns)
            

def cargar_datos(ruta_archivo=None):
    """
    Carga los datos desde un archivo Excel y realiza validaciones.
    Devuelve el DataFrame cargado y la ruta utilizada.
    """
# Detectar si se está ejecutando en modo debug
    modo_debug = "--debug_mode" in sys.argv

    if not ruta_archivo:
        if modo_debug:
            ruta_archivo = r"C:\Users\delpi\OneDrive\Tesis\ADRpy-VTOL\ADRpy\analisis\Data\Datos_aeronaves.xlsx"
            print(f"DEBUG MODE ACTIVADO: usando ruta predeterminada: {ruta_archivo}")
        else:
            ruta_archivo = input(
                r"Ingrese la ruta del archivo Excel original (o presione Enter para usar la predeterminada): "
            ).strip() or r"C:\Users\delpi\OneDrive\Tesis\ADRpy-VTOL\ADRpy\analisis\Data\Datos_aeronaves.xlsx"

    print(f"DEBUG: ruta_archivo antes de validar: '{ruta_archivo}'")

    # Validar el formato del archivo
    if not ruta_archivo.endswith((".xlsx", ".xlsm")):
        raise ValueError(
            "El archivo debe estar en formato .xlsx o .xlsm compatible con openpyxl."
        )

    # Mostrar mensaje de carga
    print(f"=== Cargando datos desde el archivo: {ruta_archivo} ===")

    try:
        # Cargar el archivo con encabezado e índice configurados
        df = pd.read_excel(ruta_archivo, header=0, index_col=0)

        # Validaciones adicionales
        if df.empty:
            raise ValueError("El archivo cargado está vacío.")

        # Manejar índices nulos
        if df.index.isnull().any():
            print(
                "Advertencia: Índices nulos encontrados. Reemplazando por 'indice_desconocido'."
            )
            df.index = df.index.fillna("indice_desconocido")

        # Manejar columnas nulas
        if df.columns.isnull().any():
            print(
                "Advertencia: Columnas nulas encontradas. Reemplazando por 'columna_desconocida'."
            )
            df.columns = df.columns.fillna("columna_desconocida")

        # Mostrar información básica del DataFrame cargado
        print("\n=== Resumen inicial del DataFrame cargado ===")
        print(df.info())
        # print("\n=== Vista previa de índices y columnas ===")
        # print(f"Primeros índices: {df.index.tolist()[:10]}")
        # print(f"Primeras columnas: {df.columns.tolist()[:10]}")

        return df, ruta_archivo
    except FileNotFoundError:
        raise ValueError("Error: Archivo no encontrado.")
    except Exception as e:
        raise ValueError(f"Error al cargar el archivo: {e}")
