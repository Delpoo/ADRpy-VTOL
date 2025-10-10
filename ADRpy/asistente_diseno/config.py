"""
Configuración central del módulo.

Ruta de datos por defecto (Excel).

Selección de hoja: por diseño usamos la PRIMERA hoja (sheet_name=0).

Umbrales y constantes globales (sin lógica aún).

Nota: estos valores se pueden sobreescribir desde el notebook en tiempo de ejecución.
"""

from pathlib import Path

# Ruta base del proyecto (ajustada al entorno del usuario)
BASE_DIR = Path(r"C:\Users\delpi\OneDrive\Tesis\ADRpy-VTOL\ADRpy")

# Ruta del Excel de datos (no modificarlo desde el código)
DATA_XLSX = BASE_DIR / r"analisis\Results\Datos_imputados.xlsx"

# Siempre la primera hoja activa
EXCEL_SHEET = 0

# Umbrales de verificación
DELTA_REL_OK = 0.02  # < 2% → OK
DELTA_REL_AJUSTE = 0.10  # 2–10% → Ajuste sugerido; >10% → No confiable

# Densidad relativa fija (5000 ft) para verificación IAS↔TAS
SIGMA_5000_FT = 0.8620

# Parámetros para tendencias y sugerencias (valores por defecto, sin lógica aquí)
K_VECINOS_DEFAULT = 10
N_MIN_RECTA = 5

# ---------------- Filtros por selección (tolerancias) ----------------
# Banda relativa para 'fijo' u 'objetivo' (porcentaje del rango observado)
TOL_REL_FIJO = 0.01  # 1%
# Banda absoluta opcional (si se define, prevalece sobre la relativa para ese parámetro)
TOL_ABS_FIJO = None  # p.ej., 2.0 -> ±2 unidades; deja None para ignorar

# ---- NOMBRES DE COLUMNAS (EDITA ESTOS STRINGS PARA QUE COINCIDAN CON TU EXCEL) ----

COL_MTOW = "Peso máximo al despegue (MTOW)"
COL_W0 = (
    "Peso Vacio (MTOW - payload)"  # si en tu hoja tiene tilde o cambia, ponelo exacto
)
COL_PAYLOAD = "Payload"

COL_AR = "Relación de aspecto del ala"
COL_B = "Envergadura"
COL_C = "Cuerda"

# Estas dos son las más importantes para tu error: CAMBIALAS a lo que viste en el print
COL_IAS = (
    "Velocidad crucero (m/s IAS)"  # <--- PONÉ AQUÍ EL NOMBRE REAL DE TU COLUMNA IAS
)
COL_TAS = "Velocidad a la que se realiza el crucero (m/s TAS)"  # <--- y aquí la de TAS

COL_R = "Alcance de la aeronave (km)"
COL_H = "Autonomía de la aeronave (h)"

# V que se usa en la verificación de alcance: usamos TAS de crucero
COL_V_TAS = COL_TAS

# Nombre de la columna de segmentación principal (misión)
SEGMENT_COL = "Misión"  # <-- ajusta al header real de tu Excel si difiere

# Etiquetas legibles para los códigos de misión
# Si tu columna ya trae strings legibles, este mapping puede dejarse vacío.
SEGMENT_LABELS = {
    1: "Vigilancia / fotogrametría y carga",
    2: "Alcance",
    3: "Deportivo",
    4: "Recreativo",
    5: "Carga",
    6: "Kamikaze",
}

# --- NUEVO: defaults para la detección y UI dinámica ---
# mínimo de valores no nulos para considerar una columna útil
MIN_VALID_NUMERIC = 5
# columnas a excluir de la detección (ajusta a tus nombres reales)
EXCLUDE_COLS = {"aeronave", "Aeronav e", "Nombre", SEGMENT_COL}
# orden preferido al inicio del panel (si existen)
PREFERRED_ORDER = [
    "Peso máximo al despegue (MTOW)",
    "Payload",
    "Velocidad a la que se realiza el crucero (m/s TAS)",
    "Autonomía de la aeronave (h)",
]
# Map opcional de etiquetas legibles por columna; si no está, se usa el nombre tal cual
DISPLAY_LABELS = {
    # "col_df": "Etiqueta legible"
}
# Defaults por parámetro (si no hay estado previo)
PARAM_DEFAULTS = {
    "active": False,
    "mode": "ignorar",  # ignorar | minimo | maximo | fijo
    "weight": 1.0,
}
