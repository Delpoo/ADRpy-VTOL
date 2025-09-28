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
from typing import Tuple, Iterable, Any

import pandas as pd

# Plotly import is optional; code guards on availability
try:
    import plotly.graph_objects as _go  # type: ignore
except Exception:  # pragma: no cover - optional
    _go = None


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


# ================================
# Formatting helpers (.2f)
# ================================


def f2(x: Any, sep_miles: bool = False) -> str:
    """Devuelve el número con 2 decimales fijos. Si sep_miles=True, usa separador de miles.

    - Evita notación científica.
    - Si no es convertible a float, retorna "-".
    """
    try:
        v = float(x)
    except Exception:
        return "-"
    return f"{v:,.2f}" if sep_miles else f"{v:.2f}"


def style_df_2dec(
    df: pd.DataFrame, cols: Iterable[str] | None = None
) -> "pd.io.formats.style.Styler":
    """Return a Styler applying fixed 2-decimal formatting via f2() to numeric columns.

    - Uses f2(v, sep_miles=False) for all numeric columns (or the provided subset).
    - Sets na_rep to '—'.
    """
    sty = df.style
    try:
        # Detect numeric columns to target
        if cols is None:
            target_cols = [
                c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
            ]
        else:
            target_cols = [c for c in cols if c in df.columns]

        if target_cols:
            try:
                # Preferred: simple format string with subset
                sty = sty.format("{:.2f}", subset=target_cols)
            except Exception:
                # Fallback: callable formatter
                sty = sty.format(lambda v: f2(v, False), subset=target_cols)
        # Set NA representation via format's na_rep where supported
        try:
            sty = sty.format(na_rep="—")
        except Exception:
            pass
    except Exception:
        # If anything fails, return the default styler without formatting
        return sty
    return sty


def apply_tickformat_2dec(fig: Any) -> Any:
    """Apply ,.2f tick formats (with thousands separator) to Plotly axes (noop if not Plotly)."""
    try:
        if _go is not None and hasattr(fig, "update_xaxes"):
            fig.update_xaxes(tickformat=",.2f")
            fig.update_yaxes(tickformat=",.2f")
    except Exception:
        pass
    return fig
