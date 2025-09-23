"""
Paquete 'asistente_diseno'.

Rol:

Proveer utilidades puras (sin UI) para verificación física, ranking de similitud,
sugerencias robustas y tendencias X–Y.

No modifica el DataFrame original ni imputa valores en esta etapa.

Los módulos aún NO tienen lógica implementada; se completarán paso a paso.
"""

from . import (
    config,
    tipos,
    datos,
    verificacion,
    outliers,
    modelos_univariados,
    tendencias,
    similitud,
    narrativa,
)

__all__ = [
    "config",
    "tipos",
    "datos",
    "verificacion",
    "outliers",
    "modelos_univariados",
    "tendencias",
    "similitud",
    "narrativa",
]
