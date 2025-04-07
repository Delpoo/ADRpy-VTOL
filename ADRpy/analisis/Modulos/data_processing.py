import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import simpledialog



def procesar_datos_y_manejar_duplicados(df):
    """
    Limpia un DataFrame preservando la estructura original y maneja duplicados en índices y columnas.
    Incluye interacción para gestionar duplicados según las elecciones del usuario.
    :param df: DataFrame a procesar.
    :return: DataFrame limpio y procesado.
    """
    import tkinter as tk
    from tkinter import simpledialog

    try:
        print("=== Inicio del procesamiento de datos ===")
        
        # Paso 1: Limpieza inicial de encabezados
        df.columns = df.columns.str.strip().str.replace('\xa0', ' ', regex=True)
        df.index = df.index.astype(str).str.strip().str.replace('\xa0', ' ', regex=True)

        # Paso 2: Eliminar filas y columnas completamente vacías
        df.dropna(how='all', inplace=True)  # Filas vacías
        df.dropna(how='all', axis=1, inplace=True)  # Columnas vacías

        # Paso 3: Manejo de duplicados
        print("\n=== Comprobación de duplicados ===")
        duplicados_fila = df.index[df.index.duplicated()].tolist()
        duplicados_columna = df.columns[df.columns.duplicated()].tolist()

        if not duplicados_fila and not duplicados_columna:
            print("No se encontraron duplicados en índices o columnas.")
        else:
            print(f"Índices duplicados: {duplicados_fila}")
            print(f"Columnas duplicadas: {duplicados_columna}")

            # Crear ventana emergente para interacción
            root = tk.Tk()
            root.withdraw()

            # Preguntar manejo global de duplicados
            respuesta_global = simpledialog.askstring(
                "Manejo global de duplicados",
                "Se encontraron duplicados. ¿Deseas aplicar una acción global a todos?\n"
                "[1] Eliminar todos los duplicados\n"
                "[2] Conservar el primero en todos\n"
                "[3] Conservar el último en todos\n"
                "[4] Procesar duplicados uno por uno"
            )

            # Aplicar acción global si corresponde
            if respuesta_global in ['1', '2', '3']:
                if respuesta_global == '1':
                    print("Eliminando todos los duplicados...")
                    if duplicados_fila:
                        df = df.loc[~df.index.duplicated(keep=False)]
                    if duplicados_columna:
                        df = df.loc[:, ~df.columns.duplicated(keep=False)]

                elif respuesta_global == '2':
                    print("Conservando el primero en todos los duplicados...")
                    if duplicados_fila:
                        df = df.loc[~df.index.duplicated(keep='first')]
                    if duplicados_columna:
                        df = df.loc[:, ~df.columns.duplicated(keep='first')]

                elif respuesta_global == '3':
                    print("Conservando el último en todos los duplicados...")
                    if duplicados_fila:
                        df = df.loc[~df.index.duplicated(keep='last')]
                    if duplicados_columna:
                        df = df.loc[:, ~df.columns.duplicated(keep='last')]
            else:
                # Procesar duplicados uno por uno si respuesta_global es '4'
                for duplicado in duplicados_fila + duplicados_columna:
                    tipo = "Índice" if duplicado in duplicados_fila else "Columna"
                    respuesta = simpledialog.askstring(
                        "Duplicado encontrado",
                        f"{tipo} duplicado '{duplicado}' encontrado. Opciones:\n"
                        "[1] Eliminar\n"
                        "[2] Conservar el primero\n"
                        "[3] Conservar el último"
                    )
                    # Realizar la acción según la elección del usuario
                    if respuesta == '1':
                        if tipo == "Índice":
                            df = df[df.index != duplicado]
                        else:
                            df = df.loc[:, df.columns != duplicado]
                    elif respuesta == '2':
                        if tipo == "Índice":
                            df = df.loc[~df.index.duplicated(keep='first')]
                        else:
                            df = df.loc[:, ~df.columns.duplicated(keep='first')]
                    elif respuesta == '3':
                        if tipo == "Índice":
                            df = df.loc[~df.index.duplicated(keep='last')]
                        else:
                            df = df.loc[:, ~df.columns.duplicated(keep='last')]

        # Paso 4: Convertir valores internos a numéricos
        print("\n=== Convirtiendo valores a numéricos donde sea posible ===")
        for col in df.columns:
            try:
               df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                print(f"Advertencia: No se pudo convertir la columna '{col}' a numérico. Error: {e}")

        print("=== Procesamiento completado ===")
        return df

    except Exception as e:
        raise ValueError(f"Error durante el procesamiento y manejo de duplicados: {e}")
    

def mostrar_celdas_faltantes_con_seleccion(df):
    """
    Permite al usuario seleccionar una columna para analizar y muestra las celdas faltantes.
    Si el usuario no selecciona ninguna columna, utiliza una columna predeterminada.
    Maneja columnas con más de 26 posiciones generando etiquetas en formato Excel.
    :param df: DataFrame procesado.
    :return: DataFrame con los detalles de las celdas faltantes (si las hay).
    """
    def seleccionar_columna(df):
        """
        Permite al usuario seleccionar una columna específica para validar.
        Si no selecciona ninguna, retorna la primera columna como predeterminada.
        """
        # Crear un diccionario para asociar números con las columnas
        columnas_dict = {i + 1: col for i, col in enumerate(df.columns)}
        opciones_texto = "\n".join([f"{num}: {col}" for num, col in columnas_dict.items()])

        try:
            # Solicitar al usuario seleccionar una columna
            columna_numero = simpledialog.askstring(
                "Selección de columna",
                f"Selecciona el número correspondiente a la columna que deseas validar:\n\n{opciones_texto}"
            )
            if not columna_numero:  # Si no se selecciona nada, usar la primera columna
                print("No se seleccionó ninguna columna. Usando la primera columna como predeterminada.")
                return df.columns[0]

            columna_numero = int(columna_numero)

            if columna_numero not in columnas_dict:
                raise ValueError("Número ingresado fuera del rango válido.")

            return columnas_dict[columna_numero]

        except ValueError as e:
            print(f"Error: {e}. Finalizando la ejecución.")
            exit()

    def indice_a_columna_excel(indice):
        """
        Convierte un índice numérico de columna en una etiqueta al estilo Excel (A, B, ..., Z, AA, AB, ...).
        :param indice: Índice numérico de la columna (0 para A, 1 para B, ..., 25 para Z, 26 para AA, etc.).
        :return: Etiqueta de columna en formato Excel.
        """
        etiqueta = ""
        while indice >= 0:
            etiqueta = chr(indice % 26 + ord('A')) + etiqueta
            indice = indice // 26 - 1
        return etiqueta

    try:
        # Selección de columna
        columna_prueba = seleccionar_columna(df)

        # Identificar celdas faltantes en la columna seleccionada
        print(f"\n=== Analizando celdas faltantes en la columna: '{columna_prueba}' ===")
        missing_indices = df[df[columna_prueba].isna()].index.tolist()

        if not missing_indices:
            print(f"No se encontraron valores faltantes en la columna '{columna_prueba}'.")
            return pd.DataFrame()  # Devuelve un DataFrame vacío si no hay faltantes

        # Crear un DataFrame para almacenar los resultados
        resultados = []

        for idx in missing_indices:
            fila_excel = df.index.get_loc(idx) + 2  # +2 para ajustarse al formato Excel (encabezado en fila 1)
            columna_excel = indice_a_columna_excel(df.columns.get_loc(columna_prueba))
            celda_excel = f"{columna_excel}{fila_excel}"
            resultados.append({
                "Índice": idx,
                "Celda": celda_excel,
                "Columna": columna_prueba,
                "Valor Actual": "NaN"
            })

        # Convertir resultados a DataFrame
        df_resultados = pd.DataFrame(resultados)

        return df_resultados

    except Exception as e:
        print(f"Error al analizar celdas faltantes: {e}")
        raise


def generar_resumen_faltantes(df, titulo="Resumen de Valores Faltantes por Columna", ancho="50%", alto="300px"):
    """
    Genera un resumen de los valores faltantes por columna en un DataFrame.
    También genera una tabla HTML con la sumatoria total de los valores faltantes de todas las columnas.
    
    :param df: DataFrame a analizar.
    :param titulo: Título opcional para mostrar en la tabla HTML.
    :param ancho: Ancho del contenedor HTML.
    :param alto: Alto del contenedor HTML.
    :return: Tuple con dos DataFrames: resumen de valores faltantes por columna y sumatoria total.
    """
    # Calcular la cantidad de valores faltantes por columna
    faltantes_por_columna = df.isnull().sum()

    # Crear un DataFrame con el resumen por columna
    resumen_faltantes = faltantes_por_columna.reset_index()
    resumen_faltantes.columns = ["Columna", "Valores Faltantes"]

    # Calcular la sumatoria total de los valores faltantes
    total_faltantes = faltantes_por_columna.sum()
    resumen_total = pd.DataFrame({"Resumen": ["Total de Valores Faltantes"], "Cantidad": [total_faltantes]})

    # Mostrar el resumen por columna como una tabla HTML
    convertir_a_html(resumen_faltantes, titulo=titulo, ancho=ancho, alto=alto, mostrar=True)

    # Mostrar la sumatoria total como una tabla HTML
    convertir_a_html(resumen_total, titulo="Sumatoria Total de Valores Faltantes", ancho=ancho, alto="100px", mostrar=True)

    # Retornar ambos DataFrames para su posible uso posterior
    return resumen_faltantes, resumen_total