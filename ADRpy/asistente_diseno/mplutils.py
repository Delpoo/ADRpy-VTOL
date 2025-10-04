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
from typing import Tuple, Iterable, Any, ContextManager

import numpy as np  # optional, useful for type checks or future helpers
import pandas as pd
from IPython.display import display, clear_output
import ipywidgets as w

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


# ================================
# New common helpers (Sprint: Ámbito + Anti-duplicados + Formato 2 decimales)
# ================================

# Fixed-format string with thousands separator and 2 decimals
FMT2 = "{:,.2f}"


def fmt2(x: Any) -> str:
    """Return number as fixed two-decimals string without scientific notation.

    - Uses thousands separator.
    - Returns "—" for None/NaN/non-convertible values.
    """
    try:
        v = float(x)
        if not (v == v):  # NaN check without numpy
            return "—"
    except Exception:
        return "—"
    try:
        return FMT2.format(v)
    except Exception:
        return f"{v:.2f}"


def numeric_2dec_styler(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """Apply fixed 2-decimal formatting to all numeric columns with na_rep="—".

    This complements style_df_2dec; it always targets all numeric columns.
    """
    sty = df.style
    try:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if num_cols:
            try:
                sty = sty.format(FMT2, subset=num_cols)
            except Exception:
                sty = sty.format(lambda v: fmt2(v), subset=num_cols)
        try:
            sty = sty.format(na_rep="—")
        except Exception:
            pass
        return sty
    except Exception:
        return sty


def plotly_apply_2dec(fig: Any) -> Any:
    """Enforce .2f tick and hover formatting across a Plotly figure.

    - Axis ticks: tickformat=".2f", exponentformat="none".
    - Traces: if x/y are numeric, set hovertemplate with :.2f and <extra></extra>.
    """
    try:
        if _go is not None and hasattr(fig, "update_xaxes"):
            fig.update_xaxes(tickformat=".2f", exponentformat="none")
            fig.update_yaxes(tickformat=".2f", exponentformat="none")
        # Update traces' hovertemplate when applicable
        if hasattr(fig, "data"):
            for tr in list(getattr(fig, "data", [])):
                try:
                    x_is_num = (
                        hasattr(tr, "x")
                        and tr.x is not None
                        and len(tr.x) > 0
                        and all(
                            isinstance(v, (int, float)) for v in tr.x if v is not None
                        )
                    )
                except Exception:
                    x_is_num = False
                try:
                    y_is_num = (
                        hasattr(tr, "y")
                        and tr.y is not None
                        and len(tr.y) > 0
                        and all(
                            isinstance(v, (int, float)) for v in tr.y if v is not None
                        )
                    )
                except Exception:
                    y_is_num = False

                if x_is_num and y_is_num:
                    # Preserve customdata usage if already present by not overwriting when set
                    if not getattr(tr, "hovertemplate", None):
                        tr.hovertemplate = "%{x:.2f} → %{y:.2f}<extra></extra>"
        return fig
    except Exception:
        return fig


# Context manager to clear an ipywidgets.Output before rendering
class with_cleared(ContextManager[Any]):
    """Usage: with with_cleared(output): display(...)

    Calls output.clear_output(wait=True) on __enter__, no-op on __exit__.
    """

    def __init__(self, output_widget: Any):
        self._out = output_widget

    def __enter__(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        try:
            if hasattr(self._out, "clear_output"):
                self._out.clear_output(wait=True)
        except Exception:
            pass
        return self._out

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        return None


# Additional simple helpers requested


def format_df_2dec(df: pd.DataFrame) -> pd.DataFrame:
    """Round numeric columns to 2 decimals in-place and return df."""
    try:
        num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if num:
            df[num] = df[num].round(2)
    except Exception:
        pass
    return df


def format_2dec_df(df: pd.DataFrame, cols: list[str] | None = None):
    """Return a Styler formatting numeric columns to 2 decimals.

    - If cols is None, target all numeric columns.
    - Uses Styler.format with a single format string and subset to avoid typing issues.
    - Falls back to the raw DataFrame if Styler isn't available.
    """
    try:
        if cols is None:
            cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not cols:
            return df.style
        try:
            return df.style.format("{:.2f}", subset=cols)
        except Exception:
            # Fallback: try callable
            return df.style.format(lambda v: f"{v:.2f}", subset=cols)
    except Exception:
        return df


def render_figure_once(fig: Any, out=None) -> w.Output:
    """Render a Plotly/Matplotlib figure inside a single Output, clearing previous content."""
    out = out or w.Output()
    try:
        with out:
            clear_output(wait=True)
            display(fig)
    except Exception:
        # best-effort fallback
        pass
    return out
