import pandas as pd
import numpy as np
import logging
from itertools import combinations
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from asistente_diseno.mplutils import f2

# ---- CONFIG: mínimos de diversidad (valores únicos) y de muestras por tipo de modelo ----

MIN_UNICOS = {
    "exp-1": 5,
    "log-1": 5,
    "pot-1": 5,
    "linear-1": 5,
    "poly-1": 7,
    "linear-2": 8,
    "poly-2": 10,
}
MIN_MUESTRAS = {
    "exp-1": 6,
    "log-1": 6,
    "pot-1": 6,
    "linear-1": 6,
    "poly-1": 10,
    "linear-2": 10,
    "poly-2": 12,
}
"""
MIN_UNICOS = {
    "exp-1": 3, "log-1": 3, "pot-1": 3, "linear-1": 3,
    "poly-1": 3, "linear-2": 3, "poly-2": 3,
}
MIN_MUESTRAS = {
    "exp-1": 3, "log-1": 3, "pot-1": 3, "linear-1": 3,
    "poly-1": 3, "linear-2": 3, "poly-2": 3,
}
"""
"""
Notas:
- MIN_UNICOS = diversidad mínima: cuántos valores únicos exigimos en y y en cada predictor (tras todos los filtros). 
Evita modelos entrenados con valores muy repetidos (poca información).
- MIN_MUESTRAS = tamaño muestral mínimo: cuántas filas efectivas exigimos para poder entrenar ese tipo de modelo.
- Estas estructuras son la ÚNICA fuente de verdad. Queda prohibido redefinirlas en funciones.
"""

# ---- CONFIG: Chequeos "early-stop" para modelos de 2 predictores ----
CHECKS_2D = {
    # Colinealidad / inestabilidad
    "pearson_abs_r_max": 0.90,
    "vif_max": 10.0,
    "pc2_ratio_min": 0.03,
    "rank_min": 2,
    "cond_max": 1e5,
    # Cobertura / diversidad espacial
    "unique_pair_ratio_min": 0.60,
    "hull_ratio_min": 0.15,
    "ellipse_ratio_min": 0.10,
    # Tamaño efectivo vs parámetros
    "n_per_param_min_linear2": 8,
    "n_per_param_min_poly2": 10,
    # Modo agresivo
    "agresivo": True,
}
"""
Notas (2D checks):
- Umbrales conservadores; para ajustar, modificar solo aquí.
- area(elipse_1σ) = π*sqrt(λ1)*sqrt(λ2) con λ autovalores de la covarianza.
- area(hull) via algoritmo monotone chain (sin libs externas).
"""

logger = logging.getLogger(__name__)


class ModeloDescartado(Exception):
    """Modelo descartado por problemas numéricos."""

    def __init__(self, motivo: str):
        super().__init__(motivo)
        self.motivo = motivo


def is_missing(val):
    """
    Returns True if the value is considered missing (NaN, empty string, special codes, etc.).
    """
    if pd.isna(val):
        return True
    if isinstance(val, str) and val.strip().lower() in [
        "",
        "nan",
        "nan ",
        "-",
        "#n/d",
        "n/d",
        "#¡valor!",
    ]:
        return True
    return False


def _to_float_safe(val: Any) -> float:
    """Best-effort conversion to float; returns NaN on failure."""
    try:
        f = float(val)  # type: ignore[arg-type]
        return f
    except Exception:
        try:
            arr = np.asarray(val, dtype=float)
            return float(arr)  # handles numpy scalars
        except Exception:
            return float("nan")


def _early_checks_2d(X_raw: np.ndarray, tipo: str) -> tuple[bool, dict, list[str]]:
    """Chequeos previos agresivos para modelos de 2 predictores en espacio crudo (X1,X2).
    Devuelve (ok, metrics, reasons). No lanza excepciones.
    """
    cfg = CHECKS_2D
    X_raw = np.asarray(X_raw, dtype=float)
    n, k = X_raw.shape if X_raw.ndim == 2 else (0, 0)
    reasons: list[str] = []
    metrics: dict[str, float] = {}
    if n < 2 or k < 2:
        reasons.append("Datos insuficientes para chequeos 2D")
        return False, metrics, reasons
    x1 = X_raw[:, 0]
    x2 = X_raw[:, 1]
    # 1) Pearson y VIF
    try:
        r = float(np.corrcoef(x1, x2)[0, 1])
    except Exception:
        r = np.nan
    metrics["pearson_r"] = r
    if np.isfinite(r):
        if abs(r) >= cfg["pearson_abs_r_max"]:
            reasons.append(f"|r|={abs(r):.4f} >= {cfg['pearson_abs_r_max']}")
        vif = 1.0 / max(1e-12, (1.0 - r * r))
        metrics["vif"] = vif
        if vif >= cfg["vif_max"]:
            reasons.append(f"VIF={vif:.2f} >= {cfg['vif_max']}")
    # 2) PCA / rank / condición
    X_cent = np.column_stack([x1 - x1.mean(), x2 - x2.mean()])
    try:
        rank = int(np.linalg.matrix_rank(X_cent))
    except Exception:
        rank = 0
    metrics["rank"] = rank
    if rank < cfg["rank_min"]:
        reasons.append(f"rank={rank} < {cfg['rank_min']}")
    try:
        u, s, vh = np.linalg.svd(X_cent, full_matrices=False)
        var_total = float((s**2).sum())
        var_pc2 = float((s.min() ** 2)) if s.size == 2 else 0.0
        pc2_ratio = (var_pc2 / var_total) if var_total > 0 else 0.0
    except Exception:
        pc2_ratio = 0.0
    metrics["pc2_ratio"] = pc2_ratio
    if pc2_ratio < cfg["pc2_ratio_min"]:
        reasons.append(f"PC2_ratio={pc2_ratio:.4f} < {cfg['pc2_ratio_min']}")
    try:
        cond_num = float(np.linalg.cond(X_cent))
    except Exception:
        cond_num = np.inf
    metrics["cond"] = cond_num
    if cond_num > cfg["cond_max"]:
        reasons.append(f"cond={cond_num:.2e} > {cfg['cond_max']:.1e}")
    # 3) Cobertura
    pairs_unique = len({(float(a), float(b)) for a, b in X_raw})
    unique_pair_ratio = pairs_unique / max(1, n)
    metrics["unique_pair_ratio"] = unique_pair_ratio
    if unique_pair_ratio < cfg["unique_pair_ratio_min"]:
        reasons.append(
            f"unique_pair_ratio={unique_pair_ratio:.2f} < {cfg['unique_pair_ratio_min']}"
        )
    x1min, x1max = float(np.min(x1)), float(np.max(x1))
    x2min, x2max = float(np.min(x2)), float(np.max(x2))
    bbox_area = max(0.0, (x1max - x1min)) * max(0.0, (x2max - x2min))
    metrics["bbox_area"] = bbox_area

    def _hull_area(points: np.ndarray) -> float:
        pts = sorted(set(map(tuple, points)))
        if len(pts) <= 2:
            return 0.0

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        hull = lower[:-1] + upper[:-1]
        area = 0.0
        for i in range(len(hull)):
            x_i, y_i = hull[i]
            x_j, y_j = hull[(i + 1) % len(hull)]
            area += x_i * y_j - x_j * y_i
        return abs(area) * 0.5

    hull_area = _hull_area(X_raw[:, :2])
    metrics["hull_area"] = hull_area
    hull_ratio = (hull_area / bbox_area) if bbox_area > 0 else 0.0
    metrics["hull_ratio"] = hull_ratio
    if hull_ratio < cfg["hull_ratio_min"]:
        reasons.append(f"hull_ratio={hull_ratio:.3f} < {cfg['hull_ratio_min']}")
    # elipse 1σ
    try:
        cov = np.cov(np.vstack((x1, x2)))
        evals, _ = np.linalg.eigh(cov)
        evals = np.clip(evals, 0.0, None)
        ellipse_area = float(np.pi * np.sqrt(evals.max()) * np.sqrt(evals.min()))
    except Exception:
        ellipse_area = 0.0
    metrics["ellipse_area"] = ellipse_area
    ellipse_ratio = (ellipse_area / bbox_area) if bbox_area > 0 else 0.0
    metrics["ellipse_ratio"] = ellipse_ratio
    if ellipse_ratio < cfg["ellipse_ratio_min"]:
        reasons.append(
            f"ellipse_ratio={ellipse_ratio:.3f} < {cfg['ellipse_ratio_min']}"
        )
    # 4) n/p
    if tipo == "linear-2":
        p = 3
        n_per_param_min = cfg["n_per_param_min_linear2"]
    else:
        p = 6
        n_per_param_min = cfg["n_per_param_min_poly2"]
    n_per_p = n / float(p) if p > 0 else 0.0
    metrics["n_per_param"] = n_per_p
    if n_per_p < n_per_param_min:
        reasons.append(f"n/p={n_per_p:.2f} < {n_per_param_min} (p={p})")
    # agresivo extra
    if (
        cfg.get("agresivo", False)
        and np.isfinite(r)
        and abs(r) >= 0.95
        and not any("|r|=" in x for x in reasons)
    ):
        reasons.append(f"agresivo: |r|={abs(r):.4f} >= 0.95")
    return (len(reasons) == 0), metrics, reasons


###Funcion util para valida la imputacion por correlacion por si sola sin colocarla en el flujo general###
def cargar_y_validar_datos(path: str) -> pd.DataFrame:
    """Load Excel data from the given path using sheet 'data_frame_prueba'."""
    try:
        df = pd.read_excel(path, sheet_name="data_frame_prueba")
    except FileNotFoundError:
        raise FileNotFoundError(f"Archivo no encontrado: {path}")
    except ValueError:
        raise ValueError(
            "No se pudo leer la hoja 'data_frame_prueba'. Verifique el archivo"
        )
    df = df.rename(columns=lambda c: str(c).strip())

    # Reemplazar valores inválidos por np.nan
    df.replace("", np.nan, inplace=True)

    return df


def penalizacion_por_k(k: int) -> float:
    """Return penalization factor based on sample size."""
    if k > 20:
        return 1.0
    return max(
        0,
        min(
            1,
            0.00002281 * (k / 2) ** 5
            - 0.00024 * (k / 2) ** 4
            - 0.0036 * (k / 2) ** 3
            + 0.046 * (k / 2) ** 2
            + 0.0095 * (k / 2)
            + 0.024,
        ),
    )


def seleccionar_predictores_validos(
    df: pd.DataFrame,
    objetivo: str,
    idx_objetivo: int,
    nivel_familia: Optional[int] = None,
) -> Tuple[pd.DataFrame, str, bool]:
    """
    Devuelve un DF para imputar la fila idx_objetivo y la familia utilizada.
    • Mantiene idx_objetivo (objetivo = NaN) + todas las filas cuyo objetivo NO sea NaN.
    • NO elimina filas con NaNs en otros predictores.
    • Elimina columnas que en idx_objetivo valgan NaN.
    • Aplica filtrado progresivo de familia (F0, F1, F2, sin filtro) cuando nivel_familia es None.
    • Si nivel_familia es 0,1,2 intenta SOLO esa familia; si no reúne criterio (>=5 válidos) retorna DF vacío y familia_usada="".
    • El filtrado por rango es estricto (0% tolerancia): se elimina cualquier predictor cuyo valor en la aeronave objetivo esté fuera de [min, max] del entrenamiento.
    """
    # 1) Conservar idx_objetivo + filas con objetivo conocido
    df = df[(df.index == idx_objetivo) | df[objetivo].notna()].copy()

    # 2) Quitar columnas con NaN en idx_objetivo
    columnas_validas = [
        c for c in df.columns if c == objetivo or pd.notna(df.at[idx_objetivo, c])
    ]
    df = df[columnas_validas]

    # 3) Filtrado de familia (orden: F0 -> F1 -> F2) o familia específica
    familia_usada = "sin filtro"
    filtro_aplicado = False
    capas_familia = [
        [
            "Misión",
            "Despegue",
            "Propulsión vertical",
            "Propulsión horizontal",
            "Cantidad de motores propulsión vertical",
            "Cantidad de motores propulsión horizontal",
        ],  # F0 (más restrictiva)
        ["Misión", "Despegue", "Propulsión vertical", "Propulsión horizontal"],  # F1
        ["Misión", "Despegue"],  # F2 (menos restrictiva antes de sin filtro)
    ]

    if nivel_familia in (0, 1, 2):
        # Intentar solo esa familia
        capa = capas_familia[nivel_familia]
        if all(attr in df.columns for attr in capa):
            valores_obj = [df.at[idx_objetivo, attr] for attr in capa]
            mask = np.ones(df.shape[0], dtype=bool)
            for attr, val in zip(capa, valores_obj):
                mask &= (df[attr] == val).values
            df_fam = df[mask]
            n_validos = df_fam[objetivo].notna().sum()
            if n_validos >= 5:
                # Nota: Umbral fijo (>=5) para asegurar base mínima por familia.
                # Es independiente de MIN_MUESTRAS (que se valida luego por tipo de modelo).
                df = df_fam
                familia_usada = f"F{nivel_familia}"
                filtro_aplicado = True
            else:
                # Falló criterio: devolver DF vacío y familia_usada=""
                return pd.DataFrame(), "", False
        else:
            return pd.DataFrame(), "", False
    else:
        # Modo progresivo automático
        for i, capa in enumerate(capas_familia):
            if all(attr in df.columns for attr in capa):
                valores_obj = [df.at[idx_objetivo, attr] for attr in capa]
                mask = np.ones(df.shape[0], dtype=bool)
                for attr, val in zip(capa, valores_obj):
                    mask &= (df[attr] == val).values
                df_fam = df[mask]
                n_validos = df_fam[objetivo].notna().sum()
                if n_validos >= 5:
                    # Igual criterio (>=5) aquí; mantiene consistencia de filtrado previo al modelado.
                    df = df_fam
                    familia_usada = f"F{i}"
                    filtro_aplicado = True
                    break
    # Eliminar columnas de familia si existen
    for col in [
        "Misión",
        "Despegue",
        "Propulsión vertical",
        "Propulsión horizontal",
        "Cantidad de motores propulsión vertical",
        "Cantidad de motores propulsión horizontal",
    ]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # 4) Validación estricta de rango: eliminar predictores si el valor objetivo está fuera del rango de entrenamiento
    columnas_a_eliminar = []
    for col in df.columns:
        if col == df.columns[0] or col == objetivo:
            continue
        valores = pd.to_numeric(
            df[col].drop(index=idx_objetivo).dropna(), errors="coerce"
        )
        if valores.empty or valores.isna().all():
            columnas_a_eliminar.append(col)
            continue

        # Usar rango estricto del entrenamiento (sin tolerancia)
        mn, mx = valores.min(), valores.max()
        valor_objetivo = pd.to_numeric(df.at[idx_objetivo, col], errors="coerce")

        # CAMBIO CRÍTICO: Si el valor está fuera del rango de entrenamiento, eliminar el predictor completamente
        if not (pd.notna(valor_objetivo) and mn <= valor_objetivo <= mx):
            columnas_a_eliminar.append(col)
            logger.debug(
                f"Eliminando predictor '{col}': valor_objetivo={valor_objetivo} fuera del rango [{mn:.3f}, {mx:.3f}]"
            )

    df = df.drop(columns=columnas_a_eliminar)
    return df, familia_usada, filtro_aplicado


def generar_combinaciones(predictores: List[str]) -> List[Tuple[str, ...]]:
    combos = []
    for r in (1, 2):
        combos.extend(list(combinations(predictores, r)))
    return combos


def entrenar_modelo(
    df_filtrado: pd.DataFrame,
    objetivo: str,
    predictores: Tuple[str, ...],
    poly: bool,
    idx: int,
    modelo_extra: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Train linear, polynomial, log, power, or exponential model and compute metrics.

    Todos los modelos devuelven datos coherentes en unidades originales:
    - Coeficientes e interceptos desnormalizados/revertidos
    - Datos de entrenamiento (X, y) en escala original
    - Tipo de transformación aplicada
    - Métricas calculadas en escala original
    """
    df_train = df_filtrado.dropna(subset=[objetivo, *predictores])

    # Guardar datos originales SIEMPRE (antes de cualquier transformación)
    X_df_original = df_train[list(predictores)]
    y_original = df_train[objetivo].values

    # Almacenar datos originales para exportación
    datos_originales = {
        "X_original": X_df_original.values.tolist(),  # Lista de listas para JSON
        "y_original": y_original.tolist(),  # Lista para JSON
        "columnas_predictores": list(predictores),  # Nombres de columnas
    }
    # Inicializar variables comunes
    modelo = None
    coef_original = []
    intercepto_original = 0.0
    pred_original = np.array([])
    y_original_metrics = np.array([])
    ecuacion_desnormalizada = None
    tipo = "unknown"
    tipo_transformacion = "unknown"

    # --- NUEVO: Si y es constante, no ajustar modelo, imputar valor constante ---
    if len(df_train) > 0 and pd.Series(y_original).nunique() == 1:
        valor_constante = y_original[0]
        return {
            "descartado": False,
            "Aeronave": idx,
            "Parámetro": objetivo,
            "predictores": predictores,
            "tipo": "constante",
            "tipo_transformacion": "constante",
            "n": len(df_train),
            "n_predictores": len(predictores),
            "datos_originales": datos_originales,
            "datos_entrenamiento": datos_originales,
            "indices_entrenamiento": df_train.index.tolist(),
            "X_entrenamiento_original": X_df_original.values.tolist(),
            "coeficientes_originales": [],
            "intercepto_original": valor_constante,
            "Peso de predictores": [],
            "variable_independiente_1": None,
            "variable_independiente_2": None,
            "ecuacion_string": None,
            "mape": 0.0,
            "r2": 1.0,
            "corr": 1.0,
            "Confianza": 1.0 * penalizacion_por_k(len(df_train)),
            "modelo": None,
            "pf": None,
            "scaler_X": None,
            "scaler_y": None,
            "Advertencia": "Imputación directa: variable objetivo constante en entrenamiento. No se ajustó modelo predictivo.",
        }

    # --- NUEVO: Chequeo de diversidad mínima de valores únicos en y y predictores ---
    # Solo para modelos que no sean "constante"
    # Determinar tipo preliminar para chequeo de diversidad
    tipo_prelim = None
    if modelo_extra == "log":
        tipo_prelim = "log-1"
    elif modelo_extra == "potencia":
        tipo_prelim = "pot-1"
    elif modelo_extra == "exp":
        tipo_prelim = "exp-1"
    elif poly:
        tipo_prelim = f"poly-{len(predictores)}"
    else:
        tipo_prelim = f"linear-{len(predictores)}"

    # Usar tipo_prelim para chequeo de diversidad
    X_raw = df_train[list(predictores)].values if len(df_train) > 0 else np.array([])
    y_raw = (
        np.array(df_train[objetivo].values, dtype=float)
        if len(df_train) > 0
        else np.array([])
    )
    n_unique_y = len(np.unique(y_raw))
    n_samples = len(y_raw)
    # diversidad_requerida y minimos_muestras_requeridas eliminados; usar MIN_UNICOS / MIN_MUESTRAS
    min_unique = MIN_UNICOS.get(tipo_prelim, 5)
    # Chequeo en y
    if n_unique_y < min_unique:
        raise ModeloDescartado(
            f"Insuficiente diversidad en y: {n_unique_y} < {min_unique} para '{tipo_prelim}'"
        )
    # Chequeo en cada predictor
    for i, col in enumerate(predictores):
        n_unique_col = len(np.unique(X_raw[:, i])) if X_raw.shape[0] > 0 else 0
        if n_unique_col < min_unique:
            raise ModeloDescartado(
                f"Insuficiente diversidad en predictor '{col}': {n_unique_col} < {min_unique} para '{tipo_prelim}'"
            )
    # Alerta de información efectiva
    proporcion_info_efectiva = n_samples / n_unique_y if n_unique_y else float("inf")
    warning_msg = None
    if proporcion_info_efectiva > 2.5:
        warning_msg = f"Alerta: proporción n/n_unique_y = {proporcion_info_efectiva:.2f} (>2.5). Riesgo de sobreajuste a valores repetidos."

    try:
        # Determinar tipo de modelo y aplicar transformaciones si es necesario
        ecuacion_string = None
        # --- Inicializar advertencia para ratio MAPE ---
        warning_ratio = None
        # --- Chequeos tempranos agresivos para 2 predictores (linear-2 / poly-2) ---
        if len(predictores) == 2 and tipo_prelim in {"linear-2", "poly-2"}:
            ok2d, geom_metrics, reasons = _early_checks_2d(X_raw, tipo_prelim)
            if not ok2d:
                raise ModeloDescartado("Descartado 2D por: " + " | ".join(reasons))
            # opcional: podríamos adjuntar geom_metrics más adelante si se necesita para trazabilidad
        ratio_MAPE_val_vs_train = None
        if modelo_extra == "log":
            # Logarítmico: y = a + b*log(x)
            if len(predictores) != 1:
                return None
            X_df = df_train[list(predictores)]
            if (X_df <= 0).any().any():
                return None
            X_transformed = np.log(np.array(X_df.values, dtype=float))
            y_transformed = np.array(df_train[objetivo].values, dtype=float)
            min_required = MIN_MUESTRAS.get("log-1", 5)
            tipo = "log-1"
            tipo_transformacion = "logarítmica"
            pf = None
            scaler_X = None
            scaler_y = None
            if len(df_train) < min_required:
                return None
            # Entrenar modelo
            modelo = LinearRegression().fit(X_transformed.reshape(-1, 1), y_transformed)
            pred_transformed = modelo.predict(X_transformed.reshape(-1, 1))

            # Coeficientes ya están en unidades originales
            coef_original = modelo.coef_.tolist()
            intercepto_original = float(modelo.intercept_)
            pred_original = pred_transformed
            y_original_metrics = y_transformed
            # Ecuación string
            var = str(predictores[0])
            ecuacion_string = (
                f"y = {f2(intercepto_original)} + {f2(coef_original[0])}*log({var})"
            )
        elif modelo_extra == "potencia":
            # Potencia: y = a*x^b  <=> log(y) = log(a) + b*log(x)
            if len(predictores) != 1:
                return None
            X_df = df_train[list(predictores)]
            y_df = df_train[objetivo]
            if (X_df <= 0).any().any() or (y_df <= 0).any():
                return None
            X_transformed = np.log(np.array(X_df.values, dtype=float))
            y_transformed = np.log(np.array(y_df.values, dtype=float))
            min_required = MIN_MUESTRAS.get("pot-1", 5)
            tipo = "pot-1"
            tipo_transformacion = "potencia"
            pf = None
            scaler_X = None
            scaler_y = None
            if len(df_train) < min_required:
                return None

            # Entrenar modelo
            modelo = LinearRegression().fit(X_transformed.reshape(-1, 1), y_transformed)
            pred_transformed = modelo.predict(X_transformed.reshape(-1, 1))

            # Revertir transformación: y = a*x^b
            coef_original = modelo.coef_.tolist()  # b (exponente)
            intercepto_original = float(np.exp(modelo.intercept_))  # a (coeficiente)
            pred_original = np.exp(pred_transformed)
            y_original_metrics = np.exp(y_transformed)
            # Ecuación string
            var = str(predictores[0])
            ecuacion_string = (
                f"y = {f2(intercepto_original)}*{var}**{f2(coef_original[0])}"
            )
        elif modelo_extra == "exp":
            # Exponencial: y = a*exp(b*x) <=> log(y) = log(a) + b*x
            if len(predictores) != 1:
                return None
            X_df = df_train[list(predictores)]
            y_df = df_train[objetivo]
            if (y_df <= 0).any():
                return None
            X_transformed = np.array(X_df.values, dtype=float)
            y_transformed = np.log(np.array(y_df.values, dtype=float))
            min_required = MIN_MUESTRAS.get("exp-1", 5)
            tipo = "exp-1"
            tipo_transformacion = "exponencial"
            pf = None
            scaler_X = None
            scaler_y = None
            if len(df_train) < min_required:
                return None

            # Entrenar modelo
            modelo = LinearRegression().fit(X_transformed.reshape(-1, 1), y_transformed)
            pred_transformed = modelo.predict(X_transformed.reshape(-1, 1))

            # Revertir transformación: y = a*exp(b*x)
            coef_original = modelo.coef_.tolist()  # b
            intercepto_original = float(np.exp(modelo.intercept_))  # a
            pred_original = np.exp(pred_transformed)
            y_original_metrics = np.exp(y_transformed)
            # Ecuación string
            var = str(predictores[0])
            ecuacion_string = (
                f"y = {f2(intercepto_original)}*exp({f2(coef_original[0])}*{var})"
            )
        else:
            # Modelos lineales y polinómicos
            if poly:
                tipo_poly = f"poly-{len(predictores)}"
                min_required = MIN_MUESTRAS.get(tipo_poly, 5)
            else:
                tipo_lin = f"linear-{len(predictores)}"
                min_required = MIN_MUESTRAS.get(tipo_lin, 5)
            if len(df_train) < min_required:
                return None
            X_df = df_train[list(predictores)]
            X_raw = np.array(X_df.values, dtype=float)
            y_raw = np.array(df_train[objetivo].values, dtype=float)

            # Normalización
            if poly:
                pf = PolynomialFeatures(degree=2, include_bias=False)
                X_poly = pf.fit_transform(X_raw)
                scaler_X = StandardScaler()
                X_trans = scaler_X.fit_transform(X_poly)
                tipo_transformacion = "polinómica+normalización"

                # DEBUG: Imprimir información sobre las features polinómicas
                if hasattr(pf, "powers_"):
                    logger.debug(
                        f"Modelo poly-{len(predictores)}: PolynomialFeatures powers_: {pf.powers_}"
                    )
                    logger.debug(
                        f"  Orden de features: {pf.get_feature_names_out(['x0', 'x1'][:len(predictores)])}"
                    )
                else:
                    logger.debug(
                        f"Modelo poly-{len(predictores)}: PolynomialFeatures sin powers_"
                    )

            else:
                scaler_X = StandardScaler()
                X_trans = scaler_X.fit_transform(X_raw)
                pf = None
                tipo_transformacion = "normalización"
            scaler_y = StandardScaler()
            y_transformed = scaler_y.fit_transform(y_raw.reshape(-1, 1)).flatten()
            tipo = ("poly" if poly else "linear") + f"-{len(predictores)}"
            # Validaciones numéricas
            const_cols = [
                i for i in range(X_trans.shape[1]) if np.isclose(X_trans[:, i].var(), 0)
            ]
            if const_cols:
                raise ModeloDescartado(f"Varianza cero en columnas {const_cols}")

            if np.linalg.matrix_rank(X_trans) < X_trans.shape[1]:
                raise ModeloDescartado(
                    "Matriz de diseño singular (colinealidad perfecta)"
                )

            if np.linalg.cond(X_trans) > 1e12:
                raise ModeloDescartado(
                    "Condición numérica > 1e12 (riesgo de inestabilidad)"
                )

            if np.isclose(y_transformed.var(), 0):
                raise ModeloDescartado("Variable objetivo y es constante")

            # Entrenar modelo
            modelo = LinearRegression().fit(X_trans, y_transformed)
            coeficientes = modelo.coef_
            intercepto = modelo.intercept_

            # Desnormalizar coeficientes e intercepto (método estándar)
            if (
                scaler_X.scale_ is not None
                and scaler_X.mean_ is not None
                and scaler_y.scale_ is not None
                and scaler_y.mean_ is not None
            ):
                escalas_ajustadas = scaler_X.scale_
                medias_ajustadas = scaler_X.mean_
                coef_original = (
                    coeficientes * scaler_y.scale_[0] / escalas_ajustadas
                ).tolist()
                b_shift = sum(
                    [
                        coeficientes[j] * medias_ajustadas[j] / escalas_ajustadas[j]
                        for j in range(len(coeficientes))
                    ]
                )
                intercepto_original = float(
                    scaler_y.scale_[0] * (intercepto - b_shift) + scaler_y.mean_[0]
                )
            else:
                coef_original = coeficientes.tolist()
                intercepto_original = float(intercepto)

            ecuacion_desnormalizada = f"y = {intercepto_original} + " + " + ".join(
                f"{coef}*x{i}" for i, coef in enumerate(coef_original)
            )
            # Calcular predicciones y métricas en escala original
            pred_transformed = modelo.predict(X_trans)
            pred_original = scaler_y.inverse_transform(
                pred_transformed.reshape(-1, 1)
            ).flatten()
            y_original_metrics = scaler_y.inverse_transform(
                y_transformed.reshape(-1, 1)
            ).flatten()
            # Ecuación string y LaTeX (desnormalizada)
            ecuacion_string = ecuacion_desnormalizada
        # Calcular métricas siempre en escala original
        mape = float(
            mean_absolute_percentage_error(y_original_metrics, pred_original) * 100
        )
        r2 = r2_score(y_original_metrics, pred_original)
        corr = 0.5 * r2 + 0.5 * (1 - mape / 15)
        confianza = max(0, float(corr * penalizacion_por_k(len(df_train))))
        # Calcular pesos de predictores originales para polinómicos y lineales (normalizados)
        pesos_predictores = []
        if poly and pf is not None and hasattr(pf, "powers_"):
            powers = pf.powers_  # shape: (n_terms, n_predictores)
            coef_arr = np.array(coef_original)
            for i, pred in enumerate(predictores):
                mask = powers[:, i] > 0
                peso = float(np.sum(np.abs(coef_arr[mask])))
                pesos_predictores.append(peso)
        else:
            pesos_predictores = [float(abs(c)) for c in coef_original]
        suma_pesos = sum(pesos_predictores)
        if suma_pesos > 0:
            pesos_predictores = [p / suma_pesos for p in pesos_predictores]

        # Agregar valores de variables independientes para la aeronave objetivo
        # Los valores siguen el mismo orden que los predictores
        variable_independiente_1 = None
        variable_independiente_2 = None
        if len(predictores) >= 1:
            try:
                variable_independiente_1 = _to_float_safe(
                    df_filtrado.at[idx, predictores[0]]
                )
            except Exception:
                variable_independiente_1 = None
        if len(predictores) >= 2:
            try:
                variable_independiente_2 = _to_float_safe(
                    df_filtrado.at[idx, predictores[1]]
                )
            except Exception:
                variable_independiente_2 = None

        # --- Calcular ratio MAPE_LOOCV / mape y advertencia ---
        # NOTA: El campo 'MAPE_LOOCV' se agrega después en validar_con_loocv, pero aquí solo inicializamos en None
        ratio_MAPE_val_vs_train = None
        # El warning se agregará después de LOOCV, pero aquí lo inicializamos
        # Construir diccionario de retorno unificado
        resultado = {
            "descartado": False,
            "Aeronave": idx,
            "Parámetro": objetivo,
            "predictores": predictores,
            "tipo": tipo,
            "tipo_transformacion": tipo_transformacion,
            "n": len(df_train),
            "n_predictores": len(predictores),
            "datos_originales": datos_originales,
            "datos_entrenamiento": datos_originales,
            "indices_entrenamiento": df_train.index.tolist(),
            "X_entrenamiento_original": X_df_original.values.tolist(),
            "coeficientes_originales": coef_original,
            "intercepto_original": intercepto_original,
            "Peso de predictores": pesos_predictores,
            "variable_independiente_1": variable_independiente_1,
            "variable_independiente_2": variable_independiente_2,
            "ecuacion_string": ecuacion_string,
            "mape": mape,
            "r2": r2,
            "corr": corr,
            "Confianza": confianza,
            "modelo": modelo,
            "pf": pf if "pf" in locals() else None,
            "scaler_X": scaler_X if "scaler_X" in locals() else None,
            "scaler_y": scaler_y if "scaler_y" in locals() else None,
            "ratio_MAPE_val_vs_train": ratio_MAPE_val_vs_train,
        }
        # Agregar advertencia de información efectiva si corresponde
        if warning_msg is not None:
            resultado["Advertencia"] = warning_msg
        return resultado

    except ModeloDescartado as e:
        return {
            "Aeronave": idx,
            "Parámetro": objetivo,
            "descartado": True,
            "motivo": e.motivo,
            "predictores": predictores,
            "tipo": tipo,
            "tipo_transformacion": tipo_transformacion,
        }


#! en ecuaciones polinomiales de 1 predictor solo no me salta el error de matriz singular, me da una ecuacion valida ver tabla de word
def filtrar_mejores_modelos(
    modelos: List[Dict[str, Any]], top: int = 2
) -> List[Dict[str, Any]]:
    """Return top models per type based on Confianza."""
    # Nueva lógica de robustez y descarte
    modelos_filtrados = []
    for m in modelos:
        if m is None:
            continue
        mape = m.get("mape", np.inf)
        r2 = m.get("r2", -np.inf)
        # Descartar modelos inválidos
        if mape > 18 or r2 < 0.4:
            continue
        # Etiquetar como no robusto si está en zona intermedia
        motivo = ""
        if (7.5 < mape <= 18) or (0.4 < r2 < 0.6):
            motivo = "Modelo no robusto: "
            if 7.5 < mape <= 18:
                motivo += f"MAPE fuera de rango (>{7.5}%, <=18%) "
            if 0.4 < r2 < 0.6:
                motivo += f"R2 fuera de rango (>{0.4}, <0.6)"
            m["motivo"] = motivo.strip()
            m["no_robusto"] = True
        else:
            m["motivo"] = ""
            m["no_robusto"] = False
        modelos_filtrados.append(m)
    modelos = modelos_filtrados
    grupos: defaultdict[str, list] = defaultdict(list)
    for m in modelos:
        grupos[m["tipo"]].append(m)
    mejores = []
    for lst in grupos.values():
        lst.sort(key=lambda x: x["Confianza"], reverse=True)
        mejores.extend(lst[:top])
    return mejores


def validar_con_loocv(
    df: pd.DataFrame, objetivo: str, info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calcula MAPE_LOOCV, R2_LOOCV, Corr_LOOCV y Confianza_LOOCV usando Leave-One-Out,
        reproduciendo exactamente el pipeline de entrenamiento (sin leakage):
            • Si es polinómico: PolynomialFeatures -> StandardScaler (X) -> StandardScaler (y) -> Regresión
            • Si no es polinómico: StandardScaler (X) -> StandardScaler (y) -> Regresión
            • Siempre se hace fit SOLO con el fold de entrenamiento (no se usa información del test).
            • La predicción se des-escalada (inverse_transform) antes de calcular errores.
    """
    df_train = df.dropna(subset=[objetivo, *info["predictores"]])
    n_LOOCV = len(df_train)
    if n_LOOCV < 2:
        # No se puede hacer LOOCV con menos de 2 muestras
        advertencia = "Advertencia: No se puede realizar LOOCV con menos de 2 muestras."
        return {
            "n_LOOCV": n_LOOCV,
            "MAPE_LOOCV": np.inf,
            "R2_LOOCV": -np.inf,
            "Corr_LOOCV": -np.inf,
            "Confianza_LOOCV": 0,
            "Advertencia": advertencia,
        }

    X_full = np.asarray(df_train[list(info["predictores"])].values)
    y_full = np.asarray(df_train[objetivo].values, dtype=float)
    preds = np.zeros(n_LOOCV)
    errors = np.zeros(n_LOOCV)

    loo = LeaveOneOut()
    for i, (tr, te) in enumerate(loo.split(X_full)):
        # Obtener datos crudos del fold
        X_tr_raw = X_full[tr]
        X_te_raw = X_full[te]
        y_tr_raw = y_full[tr]
        y_te_raw = y_full[te]

        # Determinar si es modelo polinómico
        es_polinomico = info["pf"] is not None or info.get("tipo", "").startswith(
            "poly"
        )

        if es_polinomico:
            # 1) PolynomialFeatures sobre X "crudo" (fold de train)
            degree = 2  # grado por defecto
            if info.get("pf") is not None:
                degree = getattr(
                    info["pf"], "degree", 2
                )  # obtener grado del objeto pf si existe
            pf = PolynomialFeatures(degree=degree, include_bias=False)
            X_tr_poly = pf.fit_transform(X_tr_raw)  # fit SOLO con train fold
            X_te_poly = pf.transform(X_te_raw)

            # 2) StandardScaler sobre features polinómicos
            scaler_X = StandardScaler().fit(X_tr_poly)  # fit SOLO con train fold
            X_tr = scaler_X.transform(X_tr_poly)
            X_te = scaler_X.transform(X_te_poly)
        else:
            # Modelo no polinómico: StandardScaler sobre X crudo
            scaler_X = StandardScaler().fit(X_tr_raw)  # fit SOLO con train fold
            X_tr = scaler_X.transform(X_tr_raw)
            X_te = scaler_X.transform(X_te_raw)

        # y escalada como en entrenamiento (fit SOLO con train fold)
        y_tr_arr = np.asarray(y_tr_raw, dtype=float).reshape(-1, 1)
        scaler_y = StandardScaler().fit(y_tr_arr)
        y_tr = scaler_y.transform(y_tr_arr).ravel()

        # Entrenar y predecir con desescalado
        reg = LinearRegression().fit(X_tr, y_tr)
        y_hat_scaled = reg.predict(X_te)[0]
        y_hat = scaler_y.inverse_transform([[y_hat_scaled]])[0, 0]

        preds[i] = y_hat
        # evitar división por cero
        denom = y_te_raw[0] if y_te_raw[0] != 0 else 1e-9
        errors[i] = abs((y_te_raw[0] - y_hat) / denom)

    MAPE_LOOCV = errors.mean() * 100
    R2_LOOCV = r2_score(np.array(y_full), np.array(preds))
    Corr_LOOCV = 0.5 * R2_LOOCV + 0.5 * (1 - MAPE_LOOCV / 15)
    Conf_cv = max(0, Corr_LOOCV * penalizacion_por_k(n_LOOCV))

    return {
        "n_LOOCV": n_LOOCV,
        "MAPE_LOOCV": MAPE_LOOCV,
        "R2_LOOCV": R2_LOOCV,
        "Corr_LOOCV": Corr_LOOCV,
        "Confianza_LOOCV": Conf_cv,
    }


def imputar_valores_celda(
    df_resultado: pd.DataFrame,
    df_filtrado: pd.DataFrame,
    objetivo: str,
    info: Dict[str, Any],
    idx: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Imputar el valor de una celda específica utilizando el modelo desnormalizado."""
    if idx not in df_filtrado.index:
        raise KeyError(f"Index {idx} is not present in the DataFrame.")

    # Si el modelo es "constante", imputar directamente el valor constante
    if str(info.get("tipo", "")).lower() == "constante":
        valor = info["intercepto_original"]
    else:
        # Obtener predictores de la fila (escala original)
        X_pred_df = df_filtrado.loc[[idx], list(info["predictores"])]
        X_pred = X_pred_df.values
        # Inicializar valor
        valor = np.nan
        tipo_modelo = str(info.get("tipo", "")).lower()
        if (
            tipo_modelo.startswith("log")
            or tipo_modelo.startswith("pot")
            or tipo_modelo.startswith("exp")
        ):
            # Modelos especiales: predecir directamente sin escalado
            if tipo_modelo.startswith("log"):
                X_pred_trans = np.log(np.array(X_pred, dtype=float))
                valor = info["modelo"].predict(X_pred_trans.reshape(-1, 1))[0]
            elif tipo_modelo.startswith("pot"):
                X_pred_trans = np.log(np.array(X_pred, dtype=float))
                pred_log = info["modelo"].predict(X_pred_trans.reshape(-1, 1))[0]
                valor = np.exp(pred_log)
            elif tipo_modelo.startswith("exp"):
                X_pred_trans = np.array(X_pred, dtype=float)
                pred_log = info["modelo"].predict(X_pred_trans.reshape(-1, 1))[0]
                valor = np.exp(pred_log)
        else:
            # Modelos lineales y polinómicos: usar escalado y polinomio si corresponde
            if info["pf"] is not None:
                # Primero expandir a polinómicos
                X_pred_poly = info["pf"].transform(X_pred)
                if info["scaler_X"] is not None:
                    X_scaled = info["scaler_X"].transform(X_pred_poly)
                else:
                    X_scaled = X_pred_poly
            else:
                if info["scaler_X"] is not None:
                    X_scaled = info["scaler_X"].transform(X_pred)
                else:
                    X_scaled = X_pred
            y_norm = info["modelo"].predict(X_scaled)[0]
            if info["scaler_y"] is not None:
                valor = info["scaler_y"].inverse_transform([[y_norm]])[0, 0]
            else:
                valor = y_norm

    # ── advertencia de extrapolación (0% tolerancia) ─────────
    advert_extrap = ""
    df_train = df_filtrado.dropna(subset=[objetivo, *info["predictores"]])
    for col in info["predictores"]:
        rango_min = df_train[col].min()
        rango_max = df_train[col].max()
        v = df_filtrado.at[idx, col]
        if pd.isna(v) or not (rango_min <= v <= rango_max):
            advert_extrap = "Extrapolacion"
            break
    # Unificación de advertencias: solo se usa la clave 'Advertencia'
    advert_prev = info.get("Advertencia", "")
    if advert_prev and advert_extrap:
        advertencia_final = f"{advert_prev}, {advert_extrap}"
    elif advert_prev:
        advertencia_final = advert_prev
    else:
        advertencia_final = advert_extrap

    # 5. imputar el valor en el DataFrame de resultado
    df_resultado.at[idx, objetivo] = valor

    imputacion = {
        "Aeronave": idx,
        "Parámetro": objetivo,
        "Valor imputado": valor,
        "Confianza": info["Confianza"],
        "Tipo Modelo": info["tipo"],
        "Predictores": ",".join(info["predictores"]),
        "Coeficientes": list(info.get("coeficientes_originales", [])),
        "Peso de predictores": list(info.get("Peso de predictores", [])),
        "Aeronaves entrenamiento": list(df_train.index),
        "k": info["n"],
        "Penalizacion_k": penalizacion_por_k(info["n"]),
        "Corr": info["corr"],
        "MAPE": info["mape"],
        "R2": info["r2"],
        "Confianza_LOOCV": info.get("Confianza_LOOCV", np.nan),
        "k_LOOCV": info.get("n_LOOCV", np.nan),
        "Corr_LOOCV": info.get("Corr_LOOCV", np.nan),
        "MAPE_LOOCV": info.get("MAPE_LOOCV", np.nan),
        "R2_LOOCV": info.get("R2_LOOCV", np.nan),
        "Método predictivo": "Correlacion",
        "Advertencia": advertencia_final,
    }

    return df_resultado, imputacion


def imputaciones_correlacion(
    df: pd.DataFrame | str,
    exportar_modelos: bool = False,
    ruta_export: Optional[str] = None,
    permitir_sin_filtro: bool = False,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Imputa valores faltantes por correlación/modelado.

    Política de selección de familia (estricta y ordenada): F0 -> F1 -> F2 -> (sin filtro solo si permitir_sin_filtro=True).
    No se intenta 'sin filtro' cuando permitir_sin_filtro=False (por defecto).
    Mantiene contrato de retorno: (df_resultado, reporte, modelos_info).
    Advertencias unificadas bajo clave 'Advertencia'. Rango y extrapolación con 0% de tolerancia.
    """
    if isinstance(df, str):
        df = pd.read_excel(df)
    df = df.rename(columns=lambda c: str(c).strip())
    # Reemplazar valores inválidos por np.nan
    df.replace("", np.nan, inplace=True)
    df_original = df.copy()  # <- Copia del DF original antes de filtrar
    df_completo = df.copy()  # NUEVO: DataFrame completo para exportar
    df_resultado = df_original.copy()
    reporte = []
    modelos_info = []  # Lista de modelos completos (solo descartado=False)

    for objetivo in [c for c in df_original.columns if df_original[c].isna().any()]:
        faltantes = df_original[df_original[objetivo].isna()].index
        for idx in faltantes:
            modelos_validos = []
            familia_usada = ""
            filtro_aplicado = False
            modelos_info_familia = []
            df_filtrado_usado = (
                None  # Track the df_filtrado used for the selected model
            )
            # 1. Intentar con filtros de familia en orden F0, F1, F2 (explícito)
            for capa in (0, 1, 2):
                df_filtrado, fam, filtro = seleccionar_predictores_validos(
                    df_original, objetivo, idx, nivel_familia=capa
                )
                if fam != f"F{capa}" or df_filtrado is None or df_filtrado.empty:
                    continue
                predictores = [
                    col
                    for col in df_filtrado.columns
                    if col != df_filtrado.columns[0] and col != objetivo
                ]
                if not predictores:
                    continue
                modelos = []
                for combo in generar_combinaciones(predictores):
                    for poly in (False, True):
                        try:
                            modelos.append(
                                entrenar_modelo(df_filtrado, objetivo, combo, poly, idx)
                            )
                        except ModeloDescartado as e:
                            modelos.append(
                                {
                                    "Aeronave": idx,
                                    "Parámetro": objetivo,
                                    "descartado": True,
                                    "motivo": str(e),
                                    "predictores": combo,
                                    "tipo": f"{'poly' if poly else 'linear'}-{len(combo)}",
                                    "tipo_transformacion": None,
                                }
                            )
                    if len(combo) == 1:
                        for modelo_extra in ("log", "potencia", "exp"):
                            try:
                                modelos.append(
                                    entrenar_modelo(
                                        df_filtrado,
                                        objetivo,
                                        combo,
                                        False,
                                        idx,
                                        modelo_extra=modelo_extra,
                                    )
                                )
                            except ModeloDescartado as e:
                                modelos.append(
                                    {
                                        "Aeronave": idx,
                                        "Parámetro": objetivo,
                                        "descartado": True,
                                        "motivo": str(e),
                                        "predictores": combo,
                                        "tipo": f"{modelo_extra}-1",
                                        "tipo_transformacion": None,
                                    }
                                )
                constantes = [m for m in modelos if m and m.get("tipo") == "constante"]
                predictivos = [
                    m
                    for m in modelos
                    if m
                    and m.get("tipo") != "constante"
                    and not m.get("descartado", False)
                ]
                validos = [
                    m for m in predictivos if m["mape"] <= 7.5 and m["r2"] >= 0.6
                ]
                if validos:
                    modelos_validos = validos
                    familia_usada = fam
                    filtro_aplicado = filtro
                    modelos_info_familia = modelos
                    df_filtrado_usado = df_filtrado
                    break
                if constantes:
                    constante = constantes[0]
                    total_validos = df_original[objetivo].notna().sum()
                    respaldo_constante = constante["n"]
                    if (
                        respaldo_constante >= 8
                        and respaldo_constante >= 0.5 * total_validos
                    ):
                        modelos_validos = [constante]
                        familia_usada = fam
                        filtro_aplicado = filtro
                        modelos_info_familia = modelos
                        df_filtrado_usado = df_filtrado
                        break
            if not modelos_validos:
                if permitir_sin_filtro:
                    # Intentar recién ahora modo progresivo que puede terminar en 'sin filtro'
                    df_filtrado, fam, filtro = seleccionar_predictores_validos(
                        df_original, objetivo, idx, nivel_familia=None
                    )
                    if (
                        fam == "sin filtro"
                        and df_filtrado is not None
                        and not df_filtrado.empty
                    ):
                        predictores = [
                            col
                            for col in df_filtrado.columns
                            if col != df_filtrado.columns[0] and col != objetivo
                        ]
                        if predictores:
                            modelos = []
                            for combo in generar_combinaciones(predictores):
                                for poly in (False, True):
                                    try:
                                        modelos.append(
                                            entrenar_modelo(
                                                df_filtrado, objetivo, combo, poly, idx
                                            )
                                        )
                                    except ModeloDescartado as e:
                                        modelos.append(
                                            {
                                                "Aeronave": idx,
                                                "Parámetro": objetivo,
                                                "descartado": True,
                                                "motivo": str(e),
                                                "predictores": combo,
                                                "tipo": f"{'poly' if poly else 'linear'}-{len(combo)}",
                                                "tipo_transformacion": None,
                                            }
                                        )
                                if len(combo) == 1:
                                    for modelo_extra in ("log", "potencia", "exp"):
                                        try:
                                            modelos.append(
                                                entrenar_modelo(
                                                    df_filtrado,
                                                    objetivo,
                                                    combo,
                                                    False,
                                                    idx,
                                                    modelo_extra=modelo_extra,
                                                )
                                            )
                                        except ModeloDescartado as e:
                                            modelos.append(
                                                {
                                                    "Aeronave": idx,
                                                    "Parámetro": objetivo,
                                                    "descartado": True,
                                                    "motivo": str(e),
                                                    "predictores": combo,
                                                    "tipo": f"{modelo_extra}-1",
                                                    "tipo_transformacion": None,
                                                }
                                            )
                            constantes = [
                                m for m in modelos if m and m.get("tipo") == "constante"
                            ]
                            predictivos = [
                                m
                                for m in modelos
                                if m
                                and m.get("tipo") != "constante"
                                and not m.get("descartado", False)
                            ]
                            validos = [
                                m
                                for m in predictivos
                                if m["mape"] <= 7.5 and m["r2"] >= 0.6
                            ]
                            if validos:
                                modelos_validos = validos
                                familia_usada = fam
                                filtro_aplicado = filtro
                                modelos_info_familia = modelos
                                df_filtrado_usado = df_filtrado
                            elif constantes:
                                constante = constantes[0]
                                total_validos = df_original[objetivo].notna().sum()
                                respaldo_constante = constante["n"]
                                if (
                                    respaldo_constante >= 8
                                    and respaldo_constante >= 0.5 * total_validos
                                ):
                                    modelos_validos = [constante]
                                    familia_usada = fam
                                    filtro_aplicado = filtro
                                    modelos_info_familia = modelos
                                    df_filtrado_usado = df_filtrado
                else:
                    reporte.append(
                        {
                            "Aeronave": idx,
                            "Parámetro": objetivo,
                            "Valor imputado": np.nan,
                            "Confianza": 0.0,
                            "Corr": 0.0,
                            "k": 0,
                            "Tipo Modelo": "n/a",
                            "Predictores": "",
                            "Penalizacion_k": 0.0,
                            "Familia": "sin filtro",
                            "Método predictivo": "Correlacion",
                            "Advertencia": "❌ No se permite imputación sin filtro para esta celda.",
                        }
                    )
                    continue
            if modelos_validos and df_filtrado_usado is not None:
                if modelos_validos[0].get("tipo") != "constante":
                    for m in modelos_validos:
                        m.update(validar_con_loocv(df_filtrado_usado, objetivo, m))
                        m["Confianza_promedio"] = (
                            m["Confianza"] + m["Confianza_LOOCV"]
                        ) / 2
                        mape = m.get("mape", None)
                        mape_loocv = m.get("MAPE_LOOCV", None)
                        if mape is not None and mape_loocv is not None:
                            if mape == 0:
                                ratio = np.inf
                            else:
                                ratio = mape_loocv / mape
                            m["ratio_MAPE_val_vs_train"] = ratio
                            if ratio > 5:
                                advert_msg = (
                                    f"Advertencia: El MAPE de validación es más de 5 veces mayor que el de entrenamiento "
                                    f"({mape_loocv:.2f}% vs {mape:.2f}%). Posible sobreajuste."
                                )
                                if "Advertencia" in m and m["Advertencia"]:
                                    m["Advertencia"] += "; " + advert_msg
                                else:
                                    m["Advertencia"] = advert_msg
                        else:
                            m["ratio_MAPE_val_vs_train"] = None
                    for m in modelos_validos:
                        if m is not None and not m.get("descartado", False):
                            columnas_grafico = list(m["predictores"]) + [objetivo]
                            df_filtrado_graf = (
                                df_filtrado_usado[columnas_grafico]
                                .dropna()
                                .to_dict(orient="list")
                            )
                            modelos_info.append(
                                {
                                    "Aeronave": idx,
                                    "Parámetro": objetivo,
                                    "Familia": familia_usada,
                                    "Filtro_aplicado": filtro_aplicado,
                                    "predictores": list(m["predictores"]),
                                    "n_predictores": len(m["predictores"]),
                                    "n_muestras_entrenamiento": m["n"],
                                    "tipo": m["tipo"],
                                    "tipo_transformacion": m["tipo_transformacion"],
                                    "coeficientes_originales": m[
                                        "coeficientes_originales"
                                    ],
                                    "Peso de predictores": m.get(
                                        "Peso de predictores", []
                                    ),
                                    "intercepto_original": m["intercepto_original"],
                                    "ecuacion_string": m.get("ecuacion_string"),
                                    "variable_independiente_1": m.get(
                                        "variable_independiente_1"
                                    ),
                                    "variable_independiente_2": m.get(
                                        "variable_independiente_2"
                                    ),
                                    "mape": m["mape"],
                                    "r2": m["r2"],
                                    "corr": m["corr"],
                                    "Confianza": m["Confianza"],
                                    "Confianza_LOOCV": m.get("Confianza_LOOCV"),
                                    "k_LOOCV": m.get("n_LOOCV"),
                                    "Corr_LOOCV": m.get("Corr_LOOCV"),
                                    "MAPE_LOOCV": m.get("MAPE_LOOCV"),
                                    "R2_LOOCV": m.get("R2_LOOCV"),
                                    "Advertencia": m.get("Advertencia", None),
                                    "datos_entrenamiento": m["datos_originales"],
                                    "indices_entrenamiento": m.get(
                                        "indices_entrenamiento", []
                                    ),
                                    "X_entrenamiento_original": m.get(
                                        "X_entrenamiento_original",
                                        m["datos_originales"].get("X_original", []),
                                    ),
                                    "df_filtrado_shape": df_filtrado_usado.shape,
                                    "df_filtrado_columns": list(
                                        df_filtrado_usado.columns
                                    ),
                                    "df_original_shape": df_completo.shape,
                                    "df_original_columns": list(df_completo.columns),
                                    "df_original": df_completo.to_dict(orient="list"),
                                    "df_filtrado": df_filtrado_graf,
                                }
                            )
                    robustos = [
                        m
                        for m in modelos_validos
                        if m["MAPE_LOOCV"] <= 7.5 and m["R2_LOOCV"] >= 0.6
                    ]
                    no_robustos = [
                        m
                        for m in modelos_validos
                        if (m["MAPE_LOOCV"] <= 12.5 and m["R2_LOOCV"] >= 0.45)
                        and not (m["MAPE_LOOCV"] <= 7.5 and m["R2_LOOCV"] >= 0.6)
                    ]
                    if robustos:
                        mejor = max(robustos, key=lambda x: x["Confianza_promedio"])
                        warning_text = "🟢 Modelo robusto"
                    elif no_robustos:
                        mejor = max(no_robustos, key=lambda x: x["Confianza_promedio"])
                        warning_text = "🟡 Modelo no robusto"
                    else:
                        reporte.append(
                            {
                                "Aeronave": idx,
                                "Parámetro": objetivo,
                                "Valor imputado": np.nan,
                                "Confianza": 0.0,
                                "Corr": 0.0,
                                "k": 0,
                                "Tipo Modelo": "n/a",
                                "Predictores": "",
                                "Penalizacion_k": 0.0,
                                "Familia": familia_usada,
                                "Método predictivo": "Correlacion",
                                "Advertencia": "❌ Todos los modelos descartados por LOOCV (MAPE > 12.5% o R2 <= 0.45)",
                            }
                        )
                        continue
                    # Añadir o concatenar etiqueta de robustez en 'Advertencia'
                    if "Advertencia" in mejor and mejor["Advertencia"]:
                        if warning_text not in mejor["Advertencia"]:
                            mejor["Advertencia"] += "; " + warning_text
                    else:
                        mejor["Advertencia"] = warning_text
                    mejor["Familia"] = familia_usada
                    if not pd.isna(df_resultado.at[idx, objetivo]):
                        print(
                            f"⚠️ [ADVERTENCIA] Ya se imputó {objetivo} en fila {idx}. No debería ocurrir."
                        )
                    df_resultado, imputacion = imputar_valores_celda(
                        df_resultado, df_filtrado_usado, objetivo, mejor, idx
                    )
                    imputacion["Familia"] = familia_usada
                    advertencia_final = imputacion.get("Advertencia", "")
                    if warning_text not in advertencia_final:
                        if advertencia_final:
                            advertencia_final += "; " + warning_text
                        else:
                            advertencia_final = warning_text
                        imputacion["Advertencia"] = advertencia_final
                    reporte.append(imputacion)
                    continue
                else:
                    constante = modelos_validos[0]
                    total_validos = df_original[objetivo].notna().sum()
                    respaldo_constante = constante["n"]
                    porcentaje = (
                        100 * respaldo_constante / total_validos if total_validos else 0
                    )
                    advertencia = (
                        f"Imputación por valor constante respaldada por {respaldo_constante} de {total_validos} valores "
                        f"({porcentaje:.1f}% de la muestra original)."
                    )
                    if "Advertencia" in constante and constante["Advertencia"]:
                        if advertencia not in constante["Advertencia"]:
                            constante["Advertencia"] += "; " + advertencia
                    else:
                        constante["Advertencia"] = advertencia
                    constante["Familia"] = familia_usada
                    df_resultado, imputacion = imputar_valores_celda(
                        df_resultado, df_filtrado_usado, objetivo, constante, idx
                    )
                    imputacion["Familia"] = familia_usada
                    imputacion["Advertencia"] = advertencia
                    reporte.append(imputacion)
                    continue
            # 4. Si no se pudo imputar nada
            reporte.append(
                {
                    "Aeronave": idx,
                    "Parámetro": objetivo,
                    "Valor imputado": np.nan,
                    "Confianza": 0.0,
                    "Corr": 0.0,
                    "k": 0,
                    "Tipo Modelo": "n/a",
                    "Predictores": "",
                    "Penalizacion_k": 0.0,
                    "Familia": familia_usada if familia_usada else "n/a",
                    "Método predictivo": "Correlacion",
                    "Advertencia": "❌ No se pudo imputar valor por ningún método.",
                }
            )
    return df_resultado, reporte, modelos_info


"""
def test_imputacion_correlacion_basica():
    df_final, reporte = imputaciones_correlacion('ADRpy/analisis/Data/Datos_aeronaves.xlsx')
    assert not df_final.isna().any().any(), "Deberia imputar todos los valores faltantes"
    print("listo")

# Test removido para limpieza del código
"""
