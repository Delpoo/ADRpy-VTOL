# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Optional
import io, datetime as dt
import numpy as np
import pandas as pd
from .guias import compute_param_stats, fmt
from .mplutils import f2


def narrativa_param(
    df: pd.DataFrame,
    col: str,
    objetivo: Optional[float],
    sugerido: Optional[float],
    *,
    iqr_factor: float = 1.5,
) -> str:
    st = compute_param_stats(df, col, factor=iqr_factor, max_list=5)
    line1 = (
        f"**{col}** — n={st.n_val}/{st.n_total}, NaN={fmt(st.pct_nan,1)}%, "
        f"IQR=[{fmt(st.low)},{fmt(st.high)}], mediana={fmt(st.median)}."
    )
    avisos: list[str] = []
    if objetivo is not None and np.isfinite(objetivo):
        if st.n_val >= 5 and (
            (st.low is not None and st.high is not None)
            and (objetivo < st.low or objetivo > st.high)
        ):
            avisos.append("objetivo fuera del IQR")
    if sugerido is not None and np.isfinite(sugerido):
        line1 += f" Sugerencia Top-K: **{fmt(sugerido)}**."
    if avisos:
        line1 += " _Avisos_: " + ", ".join(avisos) + "."
    return "- " + line1


def _escape_md_cell(val: object) -> str:
    s = (
        ""
        if val is None or (isinstance(val, float) and not np.isfinite(val))
        else str(val)
    )
    # escape pipes and newlines for simple Markdown table
    s = s.replace("|", "\\|")
    s = s.replace("\n", " ")
    return s


def _df_to_markdown_simple(df: pd.DataFrame, max_rows: int | None = None) -> str:
    """Render a DataFrame as a simple GitHub-flavored Markdown table without external deps."""
    if df is None or df.empty:
        return "(sin datos)"
    if max_rows is not None and max_rows > 0:
        df = df.head(max_rows)
    cols = list(df.columns)
    header = "| " + " | ".join(_escape_md_cell(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row.get(c)
            try:
                if isinstance(v, (int, np.integer)):
                    cells.append(str(int(v)))
                elif isinstance(v, (float, np.floating)):
                    cells.append(f2(v))
                else:
                    cells.append(_escape_md_cell(v))
            except Exception:
                cells.append(_escape_md_cell(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)


def narrativa_informe(
    df: pd.DataFrame,
    objetivo_map: Dict[str, Optional[float]],
    sugerido_map: Dict[str, Optional[float]],
    *,
    df_rank: Optional[pd.DataFrame] = None,
    top_k: int = 10,
    titulo: str = "Resumen de diseño (asistente)",
    iqr_factor: float = 1.5,
) -> str:
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    md = io.StringIO()
    md.write(f"# {titulo}\n\n")
    md.write(f"_Generado: {ts}_\n\n")
    # sección objetivo
    md.write("## Parámetros del objetivo\n")
    if not objetivo_map:
        md.write("_Aún no hay parámetros fijados por el usuario._\n\n")
    else:
        for col, obj in objetivo_map.items():
            sug = sugerido_map.get(col)
            md.write(narrativa_param(df, col, obj, sug, iqr_factor=iqr_factor) + "\n")
        md.write("\n")
    # sección similares
    if df_rank is not None and not df_rank.empty:
        md.write("## Similares (Top-K)\n")
        md.write(
            f"Se consideraron hasta **K={top_k}** aeronaves más próximas al objetivo.\n\n"
        )
        cols = [
            c
            for c in df_rank.columns
            if c.lower()
            in {
                "aeronave",
                "nombre",
                "uav",
                "modelo",
                "name",
                "similitud",
                "distancia",
                "segmento",
            }
        ]
        show = df_rank[cols].copy() if cols else df_rank.copy()
        # tabla markdown simple (sin depender de 'tabulate')
        md.write(_df_to_markdown_simple(show, max_rows=top_k) + "\n\n")
    # cierre
    md.write("## Notas\n")
    md.write(
        "- Los rangos IQR se calcularon sobre el conjunto vigente (sin garantía de representatividad de todo el mercado).\n"
    )
    md.write(
        "- Las sugerencias Top-K son una guía: validar con ingeniería y restricciones de misión.\n"
    )
    return md.getvalue()


def export_markdown(md_text: str, path: str) -> str:
    with open(path, "w", encoding="utf-8") as f:
        f.write(md_text)
    return path


def export_html(md_text: str, path: str) -> str:
    # Conversión mínima: envolvemos como <pre> para evitar dependencias extra.
    html = "<html><head><meta charset='utf-8'><title>Informe</title></head><body><pre style='font-family:ui-monospace,Consolas,monospace'>"
    html += md_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html += "</pre></body></html>"
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path
