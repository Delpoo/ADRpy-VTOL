"""
Utilidades HTML para la UI del asistente.

Reglas clave:
- No hacer autodisplay: `convertir_a_html(..., mostrar=False)` por defecto.
- Redondeo homogéneo a 2 decimales para valores numéricos.
"""

from __future__ import annotations

from IPython.display import HTML, display
import numpy as np
import pandas as pd


def convertir_a_html(
    datos_procesados,
    titulo: str = "",
    ancho: str = "100%",
    alto: str = "400px",
    mostrar: bool = False,
):
    """
    Convierte un DataFrame/Series a una tabla HTML, redondeando números a 2 decimales.

    Parámetros
    ----------
    datos_procesados : pandas.DataFrame | pandas.Series | any
        Datos a renderizar. Si es Series se convierte a DataFrame con índice visible.
    titulo : str
        Título opcional para mostrar arriba de la tabla.
    ancho : str
        Ancho máximo del contenedor con scroll (CSS), p.ej. "100%" o "800px".
    alto : str
        Alto máximo del contenedor con scroll (CSS), p.ej. "400px".
    mostrar : bool
        Si True, hace display(HTML(...)). Por defecto False: sólo devuelve el HTML.

    Returns
    -------
    str
        HTML generado (si `mostrar` es False). Si `mostrar` es True, también lo devuelve.
    """

    # Asegurar DataFrame
    if isinstance(datos_procesados, pd.Series):
        datos_procesados = datos_procesados.to_frame(name="Valores")
        if datos_procesados.index.name is None:
            datos_procesados.index.name = "Índice"

    # Redondeo homogéneo a 2 decimales (sin tocar no-numéricos)
    def _fmt_2dec(x):
        try:
            # Tratar NaN/inf
            if isinstance(x, (int, float, np.floating, np.integer)):
                xf = float(x)
                if not np.isfinite(xf) or pd.isna(xf):
                    return "nan"
                return f"{xf:.2f}"
            return x
        except Exception:
            return x

    try:
        if isinstance(datos_procesados, pd.DataFrame):
            df_fmt = datos_procesados.copy()
            num_cols = df_fmt.select_dtypes(include=[np.number]).columns
            for c in num_cols:
                try:
                    df_fmt[c] = df_fmt[c].map(_fmt_2dec)
                except Exception:
                    # fallback por columna
                    df_fmt[c] = df_fmt[c].astype(object).map(_fmt_2dec)
            datos_procesados = df_fmt
    except Exception:
        # En caso de estructuras no pandas o errores, se continúa sin formatear
        pass

    estilo_scroll = f"""
    <style>
        .scroll-table {{
            overflow-x: auto;
            overflow-y: auto;
            max-height: {alto};
            max-width: {ancho};
            display: block;
            border: 1px solid #ccc;
            margin-bottom: 12px;
            font-size: 12px;
        }}
        table {{
            border-collapse: collapse;
            width: auto;
            table-layout: auto;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 4px 6px;
            white-space: nowrap;
            text-align: center;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: bold;
        }}
        td {{
            word-wrap: break-word;
            max-width: 220px;
        }}
        h3 {{
            margin: 6px 0;
        }}
    </style>
    """

    try:
        tabla_html = (
            estilo_scroll
            + (f"<h3>{titulo}</h3>" if titulo else "")
            + f"<div class='scroll-table'>{getattr(datos_procesados, 'to_html', lambda: str(datos_procesados))()}</div>"
        )
    except Exception:
        # Fallback robusto
        tabla_html = (
            estilo_scroll
            + (f"<h3>{titulo}</h3>" if titulo else "")
            + f"<div class='scroll-table'><pre>{str(datos_procesados)}</pre></div>"
        )

    if mostrar:
        display(HTML(tabla_html))
    return tabla_html
