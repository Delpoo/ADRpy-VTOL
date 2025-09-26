"""
Utilidades centralizadas para manejar matplotlib en entornos Jupyter/Styler.

Objetivos:
- Evitar errores de backend al importar matplotlib en notebooks.
- Proveer importación perezosa y segura de pyplot.
- Asegurar submódulos necesarios para pandas Styler (colormaps, colors).
"""

from __future__ import annotations

import os
import sys
from typing import Tuple


def fix_matplotlib_backend() -> None:
    """
    Selecciona un backend seguro para notebooks si no hay uno configurado aún.
    No rompe si matplotlib no está instalado.
    """
    # 1) Sanitizar variable de entorno si apunta a un backend no válido
    mpl_env = os.environ.get("MPLBACKEND", "")
    if mpl_env.startswith("module://"):
        # Algunas versiones no aceptan 'module://matplotlib_inline.backend_inline'
        # Forzamos un backend seguro para generación no interactiva
        os.environ["MPLBACKEND"] = "Agg"

    # 2) Intentar importar y, si falla por backend inválido, limpiar y reintentar
    try:
        import matplotlib  # type: ignore
    except Exception:
        # Elimina MPLBACKEND y reintenta
        os.environ.pop("MPLBACKEND", None)
        try:
            import matplotlib  # type: ignore
        except Exception:
            return

    # 3) Intentar activar inline en notebooks si se puede (opcional)
    try:
        if "JPY_PARENT_PID" in os.environ:
            import importlib

            importlib.import_module("matplotlib_inline.backend_inline")
    except Exception:
        pass

    # 4) Validar backend; si no hay uno usable, usar Agg
    try:
        matplotlib.get_backend()
    except Exception:
        try:
            matplotlib.use("Agg", force=True)
        except Exception:
            pass


def import_matplotlib() -> Tuple[object | None, bool]:
    """
    Importa perezosamente matplotlib.pyplot de forma segura.
    Devuelve (plt, is_available).
    """
    try:
        fix_matplotlib_backend()
        import matplotlib.pyplot as plt  # type: ignore

        return plt, True
    except Exception:
        return None, False


def ensure_matplotlib_for_styler() -> None:
    """
    Asegura que los submódulos usados por pandas Styler estén importados,
    evitando errores como 'module has no attribute colors' al aplicar gradientes.
    """
    try:
        # Importar base + submódulos usados por Styler
        import matplotlib  # noqa: F401
        import matplotlib.colors  # noqa: F401
        import matplotlib.cm  # noqa: F401

        # No es necesario mantener referencias; con import basta para registrar los módulos
    except Exception:
        # Si matplotlib no está, simplemente no hacemos nada; Styler puede funcionar sin gradientes
        return
