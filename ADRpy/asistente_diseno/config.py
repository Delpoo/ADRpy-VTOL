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
