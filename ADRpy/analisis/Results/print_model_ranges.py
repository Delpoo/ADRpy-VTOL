import json
from ADRpy.analisis.Modulos.Analisis_modelos.plot_model_curves import get_best_model_ranges

json_path = "c:/Users/delpi/OneDrive/Tesis/ADRpy-VTOL/ADRpy/analisis/Results/modelos_completos_por_celda.json"

with open(json_path, "r", encoding="utf-8") as f:
    modelos_por_celda = json.load(f)

for celda_key in modelos_por_celda.keys():
    print(f"\n===== Celda: {celda_key} =====")
    rango_x, rango_y, rango_z, predictor_names = get_best_model_ranges(modelos_por_celda, celda_key)
    print(f"Rango X: {rango_x}")
    print(f"Rango Y: {rango_y}")
    print(f"Rango Z: {rango_z}")
    print(f"Nombres de predictores: {predictor_names}")
