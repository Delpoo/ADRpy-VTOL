"""
Compatibilidad: re-exporta HELP y apply_tooltip desde guias.py
para no romper imports antiguos.
"""

from .guias import HELP, apply_tooltip  # noqa: F401

__all__ = ["HELP", "apply_tooltip"]
