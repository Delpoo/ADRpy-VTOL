# -*- coding: utf-8 -*-
import pandas as pd
from asistente_diseno.mplutils import f2
from openpyxl import load_workbook
from openpyxl.styles import PatternFill
from openpyxl.comments import Comment
from openpyxl.styles import Font
from openpyxl.cell.cell import MergedCell


def create_large_comment(text, author="System"):
    """Create a comment with enlarged size for better visibility"""
    comment = Comment(text, author)
    # Increase comment size significantly (12x height, 4x width)
    comment.width = 500  # Default is around 100, making it 4x
    comment.height = 1000  # Default is around 50, making it 12x
    return comment


# Helper function to check if a value is considered missing
MISSING_VALUES = ["", "nan", "nan ", "-", "#n/d", "n/d", "#¡valor!"]


def is_missing(val):
    if val is None:
        return True
    if isinstance(val, float):
        return pd.isna(val)
    return str(val).strip().lower() in MISSING_VALUES


# Helper function to format cell comments with bold titles and italic values
# Each field is shown on a new line, numeric values with 3 significant digits, section titles in bold
# openpyxl comments do not support rich text, so we use Markdown-like formatting for clarity


def format_comment(dictionary, title=None, indent=0, max_indent=2):
    import numbers

    if not dictionary:
        return ""
    lines = []
    prefix = "    " * indent
    # Separador visual para la sección
    if title and indent == 0:
        lines.append(f"=== {title.upper()} ===")
    for k, v in dictionary.items():
        if k == "Detalle imputación":
            continue  # Skip this field
        # Si es un subdiccionario y no estamos en la última anidación
        if isinstance(v, dict) and indent < max_indent:
            lines.append(f"{prefix}{k}:")
            sub_comment = format_comment(
                v, None, indent=indent + 1, max_indent=max_indent
            )
            if sub_comment:
                lines.append(sub_comment)
        # Si es un subdiccionario en la última anidación, mostrar como lista en una sola línea
        elif isinstance(v, dict) and indent >= max_indent:
            sub_items = []
            for subk, subv in v.items():
                if isinstance(subv, float):
                    value = f2(subv)
                elif isinstance(subv, numbers.Number):
                    value = f2(subv)
                elif hasattr(subv, "item") and callable(getattr(subv, "item", None)):
                    try:
                        val = subv.item()
                        value = f2(val)
                    except Exception:
                        value = str(subv)
                else:
                    value = str(subv)
                sub_items.append(f"{subk}: {value}")
            lines.append(f"{prefix}{k}: [{'; '.join(sub_items)}]")
        # Si es una lista de dicts, imprimir cada uno en nueva línea
        elif isinstance(v, list) and v and all(isinstance(i, dict) for i in v):
            lines.append(f"{prefix}{k}:")
            for i, subdict in enumerate(v):
                lines.append(f"{prefix}  - Item {i+1}:")
                sub_comment = format_comment(
                    subdict, None, indent=indent + 2, max_indent=max_indent
                )
                if sub_comment:
                    lines.append(sub_comment)
        # Si es una lista de valores simples, imprimir todos en una sola línea
        elif isinstance(v, list):
            value_list = []
            for item in v:
                if isinstance(item, float):
                    value = f2(item)
                elif isinstance(item, numbers.Number):
                    value = f2(item)
                elif hasattr(item, "item") and callable(getattr(item, "item", None)):
                    try:
                        val = item.item()
                        value = f2(val)
                    except Exception:
                        value = str(item)
                else:
                    value = str(item)
                value_list.append(value)
            lines.append(f"{prefix}{k}: [{', '.join(value_list)}]")
        # Si es un valor numérico, formatear a 3 cifras significativas y evitar notación científica
        elif isinstance(v, float):
            value = f2(v)
            lines.append(f"{prefix}{k}:   {value}")
        elif isinstance(v, numbers.Number):
            value = f2(v)
            lines.append(f"{prefix}{k}:   {value}")
        elif (
            (not isinstance(v, dict))
            and hasattr(v, "item")
            and callable(getattr(v, "item", None))
        ):
            try:
                val = v.item()
                value = f2(val)
            except Exception:
                value = str(v)
            lines.append(f"{prefix}{k}:   {value}")
        else:
            lines.append(f"{prefix}{k}:   {str(v)}")
    if indent == 0:
        lines.append("")
    return "\n".join(lines)


def exportar_excel_con_imputaciones(
    source_file,
    df_processed,
    details_for_excel,
    output_file=r"C:\Users\delpi\OneDrive\Tesis\ADRpy-VTOL\ADRpy\analisis\Results\Datos_imputados.xlsx",
    origen_por_celda=None,
):
    """
    Exports the processed DataFrame to an Excel file, preserving the original format.
    Adds colors and comments to cells imputed by similarity, correlation, or both, including full details for each method used.

    :param source_file: Path to the original Excel file.
    :param output_file: Path to the output Excel file.
    :param df_processed: DataFrame with the imputed values.
    :param details_for_excel: List of dicts with details for each imputation (final, similarity, correlation).
    """
    import os
    import re

    def _norm_label(x):
        if x is None:
            return None
        # openpyxl puede devolver tipos no-str (números, fechas). Igualamos a str.
        try:
            s = x if isinstance(x, str) else str(x)
        except Exception:
            return x
        return s.replace("\xa0", " ").strip()

    def _detect_header_row_and_index_col(
        ws, df_processed, max_header_rows: int = 5, max_index_cols: int = 3
    ):
        """Detecta (de forma conservadora) dónde están los encabezados de parámetros y el índice de aeronaves.

        Problema típico: Excel con filas de título/agrupación (encabezado real no está en fila 1).
        Elegimos la fila/col que maximiza coincidencias con df_processed.
        """
        # Precalcular sets normalizados
        df_cols_norm = {_norm_label(c) for c in getattr(df_processed, "columns", [])}
        df_idx_norm = {_norm_label(i) for i in getattr(df_processed, "index", [])}

        # Heurística: encabezados suelen estar desde col 2 (o más), y no vacíos
        best_row = 1
        best_score = -1
        max_col = ws.max_column or 1
        for r in range(1, min(max_header_rows, ws.max_row or 1) + 1):
            vals = []
            for c in range(2, min(max_col, 200) + 1):
                v = ws.cell(row=r, column=c).value
                if v is None:
                    continue
                s = _norm_label(v)
                if s:
                    vals.append(s)
            if not vals:
                continue
            score = sum(1 for v in vals if v in df_cols_norm)
            if score > best_score:
                best_score = score
                best_row = r

        best_col = 1
        best_score = -1
        max_row = ws.max_row or 1
        for c in range(1, min(max_index_cols, max_col) + 1):
            vals = []
            for r in range(best_row + 1, min(max_row, best_row + 1 + 500) + 1):
                v = ws.cell(row=r, column=c).value
                if v is None:
                    continue
                s = _norm_label(v)
                if s:
                    vals.append(s)
            if not vals:
                continue
            score = sum(1 for v in vals if v in df_idx_norm)
            if score > best_score:
                best_score = score
                best_col = c

        return best_row, best_col

    try:

        # --- Buscar nombre de archivo disponible para no sobrescribir ---
        base, ext = os.path.splitext(output_file)
        base_no_num = re.sub(r"\s*\(\d+\)$", "", base)
        candidate = output_file
        i = 1
        while os.path.exists(candidate):
            candidate = f"{base_no_num}({i}){ext}"
            i += 1
        if candidate != output_file:
            print(
                f"ℹ️ El archivo '{output_file}' ya existe. Guardando como '{candidate}' para evitar sobrescribir."
            )
        output_file = candidate

        print(f"📤 === Exporting data to file: {output_file} ===")
        wb = load_workbook(source_file)
        # Robust check for active sheet
        if wb.sheetnames:
            ws = wb.active
            if ws is None:
                print("❌ Error: Could not get the active sheet from the Excel file.")
                return
        else:
            print(f"❌ Error: The file '{source_file}' contains no sheets.")
            return

        # Detectar encabezados/índice (para no perder formato cuando el Excel tiene filas de título)
        header_row, index_col = 1, 1
        try:
            header_row, index_col = _detect_header_row_and_index_col(ws, df_processed)
        except Exception:
            header_row, index_col = 1, 1

        # Freeze panes (mantener visible encabezado e índice)
        try:
            ws.freeze_panes = ws.cell(row=header_row + 1, column=index_col + 1)
        except Exception:
            pass

        # Define fill colors for each imputation method
        color_similarity = PatternFill(
            start_color="FFFF00", end_color="FFFF00", fill_type="solid"
        )  # Yellow
        color_correlation = PatternFill(
            start_color="00FF00", end_color="00FF00", fill_type="solid"
        )  # Green
        color_weighted = PatternFill(
            start_color="00B0F0", end_color="00B0F0", fill_type="solid"
        )  # Blue
        color_orange = PatternFill(
            start_color="FFA500", end_color="FFA500", fill_type="solid"
        )  # Orange
        color_error = PatternFill(
            start_color="000000", end_color="000000", fill_type="solid"
        )  # Black

        # Build a quick-access dictionary by cell (can be empty)
        details_dict = (
            {(d["Parámetro"], d["Aeronave"]): d for d in details_for_excel}
            if details_for_excel
            else {}
        )

        # Mapas para tolerar variaciones de encabezados (espacios, NBSP, unidades, mojibake)
        try:
            from .column_aliases import resolve_name_in_columns, canonicalize_parametro
        except Exception:
            resolve_name_in_columns = None  # type: ignore
            canonicalize_parametro = None  # type: ignore

        df_index_map = {_norm_label(i): i for i in getattr(df_processed, "index", [])}
        df_col_map = {_norm_label(c): c for c in getattr(df_processed, "columns", [])}

        if details_for_excel:
            # Añadir llaves normalizadas para no perder comentarios/colores por pequeñas diferencias
            for d in details_for_excel:
                try:
                    p_raw = d.get("Parámetro")
                    a_raw = d.get("Aeronave")
                except Exception:
                    continue
                if p_raw is None or a_raw is None:
                    continue
                p_norm = _norm_label(p_raw)
                a_norm = _norm_label(a_raw)
                # Mantener la primera ocurrencia (consistente con el comportamiento actual)
                if p_norm is not None and a_norm is not None:
                    details_dict.setdefault((p_norm, a_norm), d)
                if canonicalize_parametro:
                    try:
                        p_can = canonicalize_parametro(p_raw)
                        if p_can:
                            details_dict.setdefault((p_can, a_norm), d)
                    except Exception:
                        pass

        for row in ws.iter_rows(min_row=header_row + 1, min_col=index_col + 1):
            for cell in row:
                if (
                    ws is None
                    or cell is None
                    or cell.column is None
                    or cell.row is None
                ):
                    continue
                # Skip non-top-left cells of merged ranges
                if isinstance(cell, MergedCell):
                    continue
                parameter_raw = ws.cell(row=header_row, column=cell.column).value
                aircraft_raw = ws.cell(row=cell.row, column=index_col).value
                parameter = _norm_label(parameter_raw)
                aircraft = _norm_label(aircraft_raw)

                # Resolver a nombres reales en df_processed (sin renombrar el Excel)
                df_parameter = df_col_map.get(parameter)
                if (
                    df_parameter is None
                    and resolve_name_in_columns
                    and parameter is not None
                ):
                    try:
                        df_parameter = resolve_name_in_columns(
                            parameter, list(df_processed.columns)
                        )
                    except Exception:
                        df_parameter = None
                df_aircraft = df_index_map.get(aircraft, aircraft_raw)

                # Construir llaves candidatas para detalles (orden conservador)
                candidate_keys = []
                candidate_keys.append((parameter_raw, aircraft_raw))
                candidate_keys.append((parameter, aircraft))
                candidate_keys.append((df_parameter, df_aircraft))
                if canonicalize_parametro and parameter is not None:
                    try:
                        candidate_keys.append(
                            (canonicalize_parametro(parameter), aircraft)
                        )
                    except Exception:
                        pass
                # Do not overwrite non-empty cells
                if not is_missing(cell.value):
                    continue
                wrote_something = False

                detail = None
                for k in candidate_keys:
                    if k in details_dict:
                        detail = details_dict[k]
                        break

                if detail is not None:
                    # Only write if there's a value in df_processed
                    try:
                        if df_aircraft is not None and df_parameter is not None:
                            imputed_value = df_processed.at[df_aircraft, df_parameter]
                        else:
                            imputed_value = None
                    except Exception:
                        imputed_value = None
                    if imputed_value is not None and not (
                        isinstance(imputed_value, float) and pd.isna(imputed_value)
                    ):
                        cell.value = imputed_value
                        wrote_something = True
                    # Validity check for imputed value
                    valid_sim = detail["similitud"] and not is_missing(
                        detail["similitud"].get("Valor imputado", None)
                    )
                    valid_corr = detail["correlacion"] and not is_missing(
                        detail["correlacion"].get("Valor imputado", None)
                    )
                    valid_weighted = (
                        detail["final"]
                        and not is_missing(detail["final"].get("Valor imputado", None))
                        and valid_sim
                        and valid_corr
                    )

                    # Error detection (si el motor registró un fallo por celda)
                    advert_sim = ""
                    advert_corr = ""
                    try:
                        advert_sim = str(
                            (detail.get("similitud") or {}).get("Advertencia") or ""
                        )
                        advert_corr = str(
                            (detail.get("correlacion") or {}).get("Advertencia") or ""
                        )
                    except Exception:
                        advert_sim = ""
                        advert_corr = ""
                    has_error = ("ERROR:" in advert_sim.upper()) or (
                        "ERROR:" in advert_corr.upper()
                    )
                    # Color logic
                    if has_error:
                        cell.fill = color_error
                    elif valid_weighted:
                        cell.fill = color_weighted
                    elif valid_sim:
                        cell.fill = color_similarity
                    elif valid_corr:
                        cell.fill = color_correlation
                    elif detail["similitud"] or detail["correlacion"]:
                        # Evaluated but no valid value
                        cell.fill = color_orange
                    # Build clean, ordered comment
                    comment = ""
                    if detail["final"]:
                        comment += format_comment(detail["final"], "IMPUTED VALUE")
                    if detail["similitud"]:
                        sim_comment = format_comment(
                            detail["similitud"], "SIMILARITY DETAILS"
                        )
                        if sim_comment:
                            comment += "\n" + sim_comment
                    if detail["correlacion"]:
                        corr_comment = format_comment(
                            detail["correlacion"], "CORRELATION DETAILS"
                        )
                        if corr_comment:
                            comment += "\n" + corr_comment
                    if comment:
                        # Append to existing comment instead of replacing
                        try:
                            if cell.comment and cell.comment.text:
                                new_text = (cell.comment.text or "") + "\n" + comment
                                author = cell.comment.author or "System"
                                cell.comment = create_large_comment(new_text, author)
                            else:
                                cell.comment = create_large_comment(comment, "System")
                        except Exception:
                            pass
                else:
                    # No details_for_excel: try writing from df_processed if available
                    try:
                        if df_aircraft is not None and df_parameter is not None:
                            value_df = df_processed.at[df_aircraft, df_parameter]
                        else:
                            value_df = None
                    except Exception:
                        value_df = None
                    if value_df is not None and not (
                        isinstance(value_df, float) and pd.isna(value_df)
                    ):
                        cell.value = value_df
                        wrote_something = True

                # Mark calculated cells (bold+italic) without changing fill
                try:
                    # Intentar el match con llave exacta y con llave normalizada
                    meta = None
                    if origen_por_celda:
                        if (df_aircraft, df_parameter) in origen_por_celda:
                            meta = origen_por_celda.get((df_aircraft, df_parameter), {})
                        elif (aircraft_raw, parameter_raw) in origen_por_celda:
                            meta = origen_por_celda.get(
                                (aircraft_raw, parameter_raw), {}
                            )
                        elif (aircraft, parameter) in origen_por_celda:
                            meta = origen_por_celda.get((aircraft, parameter), {})

                    if meta is not None:
                        est = str(meta.get("Estado") or "").upper()
                        fuente = str(meta.get("Fuente") or "").lower()
                        is_calc = ("CALCUL" in est) or any(
                            k in fuente for k in ("deriv", "cálcul", "calcu")
                        )
                        if is_calc:
                            current_font = cell.font or Font()
                            cell.font = Font(
                                name=current_font.name,
                                size=current_font.size,
                                bold=True,
                                italic=True,
                                vertAlign=current_font.vertAlign,
                                underline=current_font.underline,
                                strike=current_font.strike,
                                color=current_font.color,
                            )
                            # Append calculation comment
                            calc_payload = {
                                "formula": meta.get("formula"),
                                "inputs": meta.get("inputs"),
                            }
                            calc_comment = format_comment(
                                calc_payload, "CÁLCULO APLICADO"
                            )
                            if calc_comment:
                                if cell.comment and cell.comment.text:
                                    new_text = (
                                        (cell.comment.text or "") + "\n" + calc_comment
                                    )
                                    author = cell.comment.author or "System"
                                    cell.comment = create_large_comment(
                                        new_text, author
                                    )
                                else:
                                    cell.comment = create_large_comment(
                                        calc_comment, "System"
                                    )
                except Exception:
                    pass

        wb.save(output_file)
        print(f"✅ Export completed. File saved as '{output_file}'.")
    except FileNotFoundError:
        print(f"❌ Error: File '{source_file}' or {output_file} not found.")
    except Exception as e:
        print(f"❌ Error processing the file: {e}")
