"""
Tipos y contratos de datos usados por el asistente.
No contienen lógica de negocio, sólo estructuras reutilizables.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class MensajeUI:
    """
    Mensaje listo para mostrar en UI/Notebook.
    nivel: "info" | "warning" | "error".
    """

    titulo: str
    cuerpo: str
    nivel: str = "info"


@dataclass
class VerificationCard:
    """
    Tarjeta de verificación de identidades físicas (NO modifica el DataFrame).

    Campos
    ------
    nombre   : título corto de la verificación (p. ej. "MTOW = W0 + Payload").
    ecuacion : ecuación legible usada para comparar izquierda vs derecha.
    entradas : dict con los valores usados (medianas si modo agregado).
    delta_rel: |A-B| / max(|A|,|B|,1e-12); None si no se pudo evaluar.
    dictamen : "OK" (<2%), "Ajuste sugerido" (2–10%), "No confiable" (>10%), "no_evaluable".
    notas    : aclaraciones ("medianas", "σ fija 5000 ft", etc.).
    """

    nombre: str
    ecuacion: str
    entradas: Dict[str, float]
    delta_rel: Optional[float]
    dictamen: str
    notas: str
