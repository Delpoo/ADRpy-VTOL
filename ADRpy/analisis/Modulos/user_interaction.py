from tkinter import simpledialog



def seleccionar_parametros_por_indices(elementos, predeterminados):
    """
    Permite seleccionar parámetros desde los índices usando números en lugar de nombres.
    :param elementos: Lista de nombres de parámetros disponibles (índices).
    :param predeterminados: Lista de parámetros preseleccionados por defecto.
    :return: Lista de parámetros seleccionados válidos.
    """
    print("\n=== Selección de Parámetros ===")
    print("Parámetros disponibles:")
    for i, elem in enumerate(elementos, 1):
        print(f"{i}. {elem}")

    # Mostrar preseleccionados
    preseleccion_indices = [elementos.index(param) + 1 for param in predeterminados if param in elementos]
    print("\nPreseleccionados: ", ", ".join([f"{i}" for i in preseleccion_indices]))

    # Entrada del usuario
    seleccion = input("\nIngresa los números separados por coma (o presiona Enter para usar los preseleccionados): ")

    # Manejar casos según la entrada del usuario
    if seleccion.strip():  # Si el usuario ingresó algo
        try:
            indices_seleccionados = [int(num.strip()) - 1 for num in seleccion.split(",")]
        except ValueError:
            print("⚠️ Entrada inválida. Usando parámetros preseleccionados.")
            indices_seleccionados = [i - 1 for i in preseleccion_indices]
    else:  # Si el usuario presiona Enter sin ingresar nada
        indices_seleccionados = [i - 1 for i in preseleccion_indices]

    # Construir la lista de seleccionados a partir de los índices válidos
    seleccionados = [elementos[i] for i in indices_seleccionados if 0 <= i < len(elementos)]

    # Validar parámetros seleccionados contra elementos disponibles
    seleccionados_validos = [p for p in seleccionados if p in elementos]
    if len(seleccionados) > len(seleccionados_validos):
        print(f"⚠️ Algunos parámetros seleccionados no son válidos y fueron eliminados: {set(seleccionados) - set(seleccionados_validos)}")

    # Retornar solo los parámetros válidos
    return seleccionados_validos

def solicitar_umbral(valor_por_defecto=0.7):
    """
    Solicita al usuario ingresar un umbral para las correlaciones significativas.
    Si el usuario no proporciona un valor válido, se usa el valor por defecto.
    :param valor_por_defecto: Valor predeterminado del umbral si el usuario no ingresa ninguno.
    :return: Umbral de correlación como flotante.
    """
    try:
        umbral = float(input(f"Ingrese el umbral mínimo de correlación significativa (por defecto {valor_por_defecto}): ") or valor_por_defecto)
        if not (0 < umbral < 1):
            raise ValueError("El umbral debe estar entre 0 y 1.")
        return umbral
    except ValueError as e:
        print(f"Valor inválido: {e}. Se usará el umbral por defecto de {valor_por_defecto}.")
        return valor_por_defecto