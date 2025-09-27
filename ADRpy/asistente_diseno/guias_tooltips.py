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
    # Sugerencias Top-K (histograma & box)
    "suger_box": (
        "Box overlay: rango intercuartil (Q1–Q3). Bigotes: hasta 1.5×IQR. Puntos fuera: atípicos."
    ),
    "suger_obj_line": "Línea vertical del valor objetivo actual del parámetro.",
    "suger_low_high": "LOW/HIGH (IQR): límites internos del box (Q1 y Q3). La mediana se indica como línea central.",
    "suger_w_mediana": "w_mediana: mediana ponderada por similitud (Top‑K más parecidos pesan más).",
    # Outliers
    "out_iqr_explain": "IQR=Q3−Q1; se marcan atípicos fuera de [Q1−k·IQR, Q3+k·IQR].",
    # Panel de detalle e informe
    "panel_info": "Panel de detalle del parámetro: muestra distribución, outliers y valores objetivo/sugeridos con acciones de navegación.",
    "narrativa": "Informe narrativo: resumen en Markdown/HTML con tablas y explicaciones listo para compartir.",
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
