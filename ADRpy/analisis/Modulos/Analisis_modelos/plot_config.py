import plotly.express as px
import numpy as np

COLORS = {
    'original_data': '#B0BEC5',  # Gris claro para datos originales
    'training_data': '#FF6B6B',  # Rojo para datos de entrenamiento
    'model_lines': px.colors.qualitative.Set1,  # Colores para líneas de modelos
    'selected_model': '#2E8B57'  # Verde para modelo seleccionado
}

# Configuración de símbolos para diferentes tipos de modelo
SYMBOLS = {
    'linear': 'circle',
    'poly': 'square',
    'log': 'diamond',
    'exp': 'triangle-up',
    'pot': 'star'
}


def _ensure_list(x):
    # Convierte Series, ndarray, lista o escalar en lista
    if hasattr(x, 'tolist'):
        return x.tolist()
    elif isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    elif x is None:
        return []
    else:
        return [x]
