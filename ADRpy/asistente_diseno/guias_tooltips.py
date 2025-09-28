from __future__ import annotations
from typing import Mapping
import ipywidgets as w

# Textos cortos y claros para tooltips de controles.
HELP: dict[str, str] = {
    # Panel izquierdo (ranking / similitud)
    "modo_param": (
        "Cómo usar cada parámetro en la comparación: • ignorar: no participa • mínimo/máximo: actúa como restricción blanda • "
        "fijo: busca cercanía al valor indicado."
    ),
    "valor_param": "Valor objetivo del parámetro (se usa si Modo=fijo/máximo/mínimo).",
    "peso_param": "Peso relativo del parámetro al combinar similitudes (1=neutral).",
    "alpha_sim": "α (agregación): α=1 promedio; α>1 penaliza desvíos grandes; α<1 suaviza.",
    "penalizar_nan": "Si hay NaN en un parámetro, agrega penalidad para no favorecer filas incompletas.",
    "penalidad_nan": "Cuánto sumar/restar a la distancia cuando hay NaN (sólo si Penalizar NaN está activo).",
    "segmentar_por": "Corte del dataset para trabajar por segmentos (p.ej. misión). (ninguno)=global.",
    "modo_global_familia": "Global: un único set. Por misión: calcula por cada segmento con n>=min_n.",
    "top_k": "Cantidad de vecinos más parecidos a considerar (lista Top-K).",
    "factor_prefer": "Multiplicador opcional para priorizar un segmento (si corresponde).",
    "iqr_on": "Quitar atípicos IQR antes de calcular ajustes/métricas.",
    "iqr_factor": "Factor IQR (1.5 típico). Mayor=recorta menos.",
    "min_n": "Mínimo de muestras válidas para ajustar una curva/métrica en cada segmento.",
    "auto": "Si está ON, recalcula automáticamente al cambiar un control.",
    # Ranking y tabla de resultados
    "ranking_sim": "Similitud: 1 es idéntico al objetivo, valores mayores indican peor ajuste (según agregación α y pesos).",
    "ranking_dist": "Distancia agregada entre parámetros (antes de normalización a similitud). Útil para diagnóstico.",
    "ranking_alerta": "Alerta en la fila 'Objetivo (usuario)': indica si el objetivo queda fuera del IQR de los Top‑K para ese parámetro (LOW/HIGH).",
    "segmentar_valor": "Valor específico del segmento cuando 'Segmentar por' está activo. Se muestran etiquetas legibles si hay mapeo.",
    "info_button": "Botón 'i': abre un panel con distribución, outliers, objetivo y valor sugerido para ese parámetro.",
    "report_button": "Generar informe: crea informe_diseno.md/html con resumen de objetivos, sugeridos, cobertura IQR y ranking.",
    # Tendencias (X–Y)
    "t_x": "Variable X (independiente) para la nube y la curva tendencia.",
    "t_y": "Variable Y (dependiente) para la nube y la curva tendencia.",
    "t_logx": "Escala logarítmica en el eje X (útil si X tiene varias órdenes de magnitud).",
    "t_obj_line": "Línea vertical con el valor objetivo de X (si está definido).",
    "r2_adj": (
        "R² ajustado: penaliza la complejidad del modelo (n vs parámetros). "
        "Se calcula como 1 - (1-R²)*(n-1)/(n-p-1); más alto es mejor."
    ),
    "dispersion_indicator": (
        "Indicador de dispersión: diagnóstico rápido de variabilidad relativa en la nube/vecinos. "
        "Útil para interpretar la confiabilidad de tendencias o sugerencias."
    ),
    # Sugerencias Top-K (histograma & box)
    "suger_box": (
        "Box overlay: rango intercuartil (Q1–Q3). Bigotes: hasta 1.5×IQR. Puntos fuera: atípicos."
    ),
    "suger_obj_line": "Línea vertical del valor objetivo actual del parámetro.",
    "suger_low_high": "LOW/HIGH (IQR): límites internos del box (Q1 y Q3). La mediana se indica como línea central.",
    "suger_w_mediana": "w_mediana: mediana ponderada por similitud (Top‑K más parecidos pesan más).",
    "suger_n_efectivo": (
        "n_efectivo: suma de pesos normalizados (0..1) de los vecinos usados; "
        "se interpreta como 'cantidad equivalente' de vecinos útiles."
    ),
    "suger_pesos": (
        "Pesos: w_dist proviene de un kernel acotado en [0,1] (p.ej., 1/(1+d) o exp(-γ·d)); "
        "w_conf en [0,1]; w_total=(w_dist^βdist)*(w_conf^βconf)."
    ),
    # Outliers
    "out_iqr_explain": "IQR=Q3−Q1; se marcan atípicos fuera de [Q1−k·IQR, Q3+k·IQR].",
    # Panel de detalle e informe
    "panel_info": "Panel de detalle del parámetro: muestra distribución, outliers y valores objetivo/sugeridos con acciones de navegación.",
    "narrativa": "Informe narrativo: resumen en Markdown/HTML con tablas y explicaciones listo para compartir.",
    # Glosario por tablas (encabezados y columnas)
    "col_dv": (
        "dv_*: aporte de la columna a la distancia total (ya normalizado por escala robusta e incluido el peso)."
    ),
    "col_viol": (
        "viol_*: indicador de violación de la restricción (True si está fuera de la regla definida para el parámetro)."
    ),
    "ranking_cols": (
        "Ranking: distancia/similitud (y sus medias) resumen la concordancia global con el objetivo; 'alerta' marca objetivos fuera de LOW/HIGH(IQR)."
    ),
    "neighbors_gloss": (
        "Top‑K vecinos: w_total combina w_dist (kernel de distancia 0..1) y w_conf (0..1). Se ordenan por w_total; 'valor' es el del parámetro."
    ),
    # Tendencias: guías orientativas
    "r2_adj_ranges": (
        "R²_ajustado (orientativo): ≥0.8 alto, 0.5–0.8 medio, <0.5 bajo. Interpretar junto con n y dispersión."
    ),
    "mape_guide": (
        "MAPE (orientativo): <10% bueno, 10–20% aceptable, >20% alto (posible ruido o no linealidad)."
    ),
}


def apply_tooltip(widget: w.Widget, key: str) -> None:
    """Aplica tooltip al 'description' o al propio widget según corresponda."""
    txt = HELP.get(key)
    if not txt:
        return
    # Widgets con 'description_tooltip' (Dropdown, Checkbox, FloatText, Sliders, etc.)
    if hasattr(widget, "description_tooltip"):
        try:
            setattr(widget, "description_tooltip", txt)
            return
        except Exception:
            pass
    # Botones y otros widgets
    if hasattr(widget, "tooltip"):
        try:
            setattr(widget, "tooltip", txt)
        except Exception:
            pass
