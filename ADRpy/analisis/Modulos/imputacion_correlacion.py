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
from .outlier_utils import calcular_pesos_outliers

# === Config helpers ===
try:
    from .controller import load_effective_config
except Exception:
    from controller import (
        load_effective_config,
    )  # fallback cuando se ejecuta fuera de paquete


def _CFG() -> dict:
    try:
        return load_effective_config()
    except Exception:
        return {}


def _COR() -> dict:
    return _CFG().get("correlacion", {})


def _OUTLIERS() -> dict:
    return _CFG().get("correlacion_outliers", {})


def _MODELOS() -> dict:
    return _CFG().get("modelos", {})


# ---- CONFIG: mínimos de diversidad (valores únicos) y de muestras por tipo de modelo ----

# Reglas internas (FALLBACK). La UX puede sobreescribir estos valores.
MIN_UNICOS_FALLBACK = {
    "exp-1": 5,
    "log-1": 5,
    "pot-1": 5,
    "linear-1": 5,
    "poly-1": 7,
    "linear-2": 8,
    "poly-2": 10,
}
MIN_MUESTRAS_FALLBACK = {
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
- La UI/Config domina; estos valores son solo fallback.
"""


# Helpers: toman primero la config, si no existe usan fallback.
def _min_unicos(tipo_modelo: str) -> int:
    div = _COR().get("diversidad_minima", {})
    mu = div.get("min_unicos", {}) or {}
    return int(mu.get(tipo_modelo, MIN_UNICOS_FALLBACK.get(tipo_modelo, 5)))


def _min_muestras(tipo_modelo: str) -> int:
    div = _COR().get("diversidad_minima", {})
    mm = div.get("min_muestras", {}) or {}
    return int(mm.get(tipo_modelo, MIN_MUESTRAS_FALLBACK.get(tipo_modelo, 6)))


# ---- CONFIG: Chequeos "early-stop" para modelos de 2 predictores ----
CHECKS_2D_DEFAULT = {
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
    # Mezclar defaults con overrides desde config
    cfg_all = _COR().get("checks_2d", {})
    if not cfg_all or not bool(cfg_all.get("enabled", True)):
        return True, {}, []

    def on(key: str) -> bool:
        v = cfg_all.get(key, {})
        if isinstance(v, dict):
            return bool(v.get("enabled", True))
        return bool(v) if key != "enabled" else True

    def getv(key: str, sub: str, default):
        v = cfg_all.get(key, {})
        if isinstance(v, dict):
            return v.get(sub, default)
        return default

    cfg = CHECKS_2D_DEFAULT
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
        abs_r = abs(r)
        if on("pearson") and abs_r >= float(getv("pearson", "abs_r_max", 0.90)):
            reasons.append(
                f"|r|={abs_r:.4f} >= {float(getv('pearson','abs_r_max',0.90))}"
            )
        vif = 1.0 / max(1e-12, (1.0 - r * r))
        metrics["vif"] = vif
        if on("vif") and vif >= float(getv("vif", "max", 10.0)):
            reasons.append(f"VIF={vif:.2f} >= {float(getv('vif','max',10.0))}")
    # 2) PCA / rank / condición
    X_cent = np.column_stack([x1 - x1.mean(), x2 - x2.mean()])
    try:
        rank = int(np.linalg.matrix_rank(X_cent))
    except Exception:
        rank = 0
    metrics["rank"] = rank
    if on("rank") and rank < int(getv("rank", "min", 2)):
        reasons.append(f"rank={rank} < {int(getv('rank','min',2))}")
    try:
        u, s, vh = np.linalg.svd(X_cent, full_matrices=False)
        var_total = float((s**2).sum())
        var_pc2 = float((s.min() ** 2)) if s.size == 2 else 0.0
        pc2_ratio = (var_pc2 / var_total) if var_total > 0 else 0.0
    except Exception:
        pc2_ratio = 0.0
    metrics["pc2_ratio"] = pc2_ratio
    if on("pc2") and pc2_ratio < float(getv("pc2", "ratio_min", 0.03)):
        reasons.append(
            f"PC2_ratio={pc2_ratio:.4f} < {float(getv('pc2','ratio_min',0.03))}"
        )
    try:
        cond_num = float(np.linalg.cond(X_cent))
    except Exception:
        cond_num = np.inf
    metrics["cond"] = cond_num
    if on("cond") and cond_num > float(getv("cond", "max", 1e5)):
        reasons.append(f"cond={cond_num:.2e} > {float(getv('cond','max',1e5)):.1e}")
    # 3) Cobertura
    pairs_unique = len({(float(a), float(b)) for a, b in X_raw})
    unique_pair_ratio = pairs_unique / max(1, n)
    metrics["unique_pair_ratio"] = unique_pair_ratio
    if on("coverage_unique_pair") and unique_pair_ratio < float(
        getv("coverage_unique_pair", "ratio_min", 0.60)
    ):
        reasons.append(
            f"unique_pair_ratio={unique_pair_ratio:.2f} < {float(getv('coverage_unique_pair','ratio_min',0.60))}"
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
    if on("coverage_hull") and hull_ratio < float(
        getv("coverage_hull", "ratio_min", 0.15)
    ):
        reasons.append(
            f"hull_ratio={hull_ratio:.3f} < {float(getv('coverage_hull','ratio_min',0.15))}"
        )
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
    if on("coverage_ellipse") and ellipse_ratio < float(
        getv("coverage_ellipse", "ratio_min", 0.10)
    ):
        reasons.append(
            f"ellipse_ratio={ellipse_ratio:.3f} < {float(getv('coverage_ellipse','ratio_min',0.10))}"
        )
    # 4) n/p
    if tipo == "linear-2":
        p = 3
        n_per_param_min = float(getv("n_per_param", "linear2_min", 8))
    else:
        p = 6
        n_per_param_min = float(getv("n_per_param", "poly2_min", 10))
    n_per_p = n / float(p) if p > 0 else 0.0
    metrics["n_per_param"] = n_per_p
    if on("n_per_param") and n < p * n_per_param_min:
        reasons.append(f"n={n} < p*n_per_param_min={p*n_per_param_min}")
    # agresivo extra
    if on("agresivo") and np.isfinite(r):
        abs_min = float(getv("agresivo", "abs_r_min", 0.95))
        if abs_r >= abs_min and not any("|r|=" in x for x in reasons):
            reasons.append(f"agresivo: |r|={abs_r:.4f} >= {abs_min}")
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
    """Penalización configurable por tamaño muestral."""
    c = _COR().get("confianza", {}).get("penalizacion_k", {})
    if c.get("tipo", "polinomica") == "polinomica":
        p = c.get("params", {})
        a5 = p.get("a5", 0.00002281)
        a4 = p.get("a4", -0.00024)
        a3 = p.get("a3", -0.0036)
        a2 = p.get("a2", 0.046)
        a1 = p.get("a1", 0.0095)
        a0 = p.get("a0", 0.024)
        x = k / 2
        val = a5 * x**5 + a4 * x**4 + a3 * x**3 + a2 * x**2 + a1 * x + a0
        return max(0.0, min(1.0, float(val if k <= 20 else 1.0)))
    return 1.0


# === NUEVO: Cálculo de confianza configurable (entrena + métricas 2D + LOOCV) ===
def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _poly_eval(x: float, coefs: dict) -> float:
    """
    Evalúa polinomios con llaves tipo a5..a0, b3..b0 o c2..c0.
    Usa el dígito de la llave como exponente.
    """
    val = 0.0
    for k, v in (coefs or {}).items():
        try:
            p = int(k[1:])
        except Exception:
            continue
        val += float(v) * (x**p)
    return float(val)


def _penalizacion_n(n: int, cfg: dict | None = None) -> float:
    # Compatibilidad: disponible si alguien lo usa directo
    c = (cfg or _COR()).get("confianza", {}).get("penalizacion_n", {})
    if c.get("tipo", "polinomica") == "polinomica":
        p = c.get("params", {})
        return _clamp01(_poly_eval(float(n), p))
    return 1.0


def _metric_value_from_stats(metric_key: str, stats: dict) -> float | None:
    """Mapea claves de penalizaciones_metricas -> valores en stats."""
    try:
        if metric_key == "pearson_abs":
            r = stats.get("pearson_r")
            if r is None:
                r = stats.get("pearson_abs")
            return abs(float(r)) if r is not None else None
        if metric_key == "vif":
            _v = stats.get("vif", None)
            return float(_v) if _v is not None else None
        if metric_key == "cond":
            _c = stats.get("cond", None)
            return float(_c) if _c is not None else None
        if metric_key == "pc2_ratio":
            _p = stats.get("pc2_ratio", None)
            return float(_p) if _p is not None else None
        if metric_key == "coverage_unique_pair":
            _u = stats.get("unique_pair_ratio", None)
            return float(_u) if _u is not None else None
        if metric_key == "coverage_hull":
            _h = stats.get("hull_ratio", None)
            return float(_h) if _h is not None else None
        if metric_key == "coverage_ellipse":
            _e = stats.get("ellipse_ratio", None)
            return float(_e) if _e is not None else None
    except Exception:
        return None
    return None


def _clasificar_loocv(mape: float, r2: float, cfg: dict | None = None) -> str:
    """Clasifica LOOCV en robusto | no_robusto | rechazado según criterios de config."""
    crit = (cfg or _COR()).get("loocv", {}).get("criterios", {})
    rob = crit.get("robusto", {"mape_max": 7.5, "r2_min": 0.6})
    nor = crit.get("no_robusto", {"mape_max": 12.5, "r2_min": 0.45})
    try:
        if mape <= float(rob.get("mape_max", 7.5)) and r2 >= float(
            rob.get("r2_min", 0.6)
        ):
            return "robusto"
        if mape <= float(nor.get("mape_max", 12.5)) and r2 >= float(
            nor.get("r2_min", 0.45)
        ):
            return "no_robusto"
    except Exception:
        pass
    return "rechazado"


def calcular_confianza_modelo(stats: dict, cfg: dict | None = None) -> float:
    """
    Calcula la confianza final del modelo combinando:
      - base por r2 y mape,
      - penalización por k (histórico),
      - penalización por N (nuevo),
      - penalizaciones métricas 2D (opcional, producto),
      - y aporte LOOCV por clase (mezcla convexa).
    Campos esperados en stats (opcionales):
      r2, mape, k, n, pearson_abs, vif, cond, pc2_ratio,
      coverage_unique_pair, coverage_hull, coverage_ellipse, loocv_class
    """
    if cfg is None:
        cfg = _CFG()
    conf_cfg = cfg.get("correlacion", {}).get("confianza", {})

    # 1) base por r2 y mape
    w_r2 = float(conf_cfg.get("w_r2", 0.5))
    w_mp = float(conf_cfg.get("w_mape", 0.5))
    divm = max(1e-6, float(conf_cfg.get("mape_divisor", 15.0)))
    r2_term = _clamp01(float(stats.get("r2", 0.0)))
    mape_term = _clamp01(1.0 - float(stats.get("mape", 1e9)) / divm)
    base = _clamp01(w_r2 * r2_term + w_mp * mape_term)

    # 2) penalización por k (histórico)
    pk = conf_cfg.get("penalizacion_k", {}).get("params", {})
    k = float(stats.get("k", 1.0))
    # Compatibilidad: evaluar polinomio en k/2 (mismo escalado que penalizacion_por_k)
    pen_k = _clamp01(_poly_eval(k / 2.0, pk)) if pk else 1.0

    # 3) penalización por N (nuevo)
    pn = conf_cfg.get("penalizacion_n", {}).get("params", {})
    n = float(stats.get("n", k))
    pen_n = _clamp01(_poly_eval(n, pn)) if pn else 1.0

    # 4) penalizaciones métricas 2D (producto)
    pm = conf_cfg.get("penalizaciones_metricas", {}) or {}
    prod = 1.0
    aliases = {
        "pearson_abs": "pearson_abs",
        "vif": "vif",
        "cond": "cond",
        "pc2_ratio": "pc2_ratio",
        "coverage_unique_pair": "coverage_unique_pair",
        "coverage_hull": "coverage_hull",
        "coverage_ellipse": "coverage_ellipse",
    }
    for name, spec in pm.items():
        if not spec or not bool(spec.get("usar", True)):
            continue
        x = float(stats.get(aliases.get(name, name), 0.0))
        prod *= _clamp01(_poly_eval(x, spec.get("params", {})))

    conf_base = _clamp01(base * pen_k * pen_n * prod)

    # 5) aporte LOOCV (mezcla convexa por clase)
    la = conf_cfg.get(
        "loocv_aporte",
        {
            "w": 0.2,
            "factor_por_clase": {"robusto": 1.0, "no_robusto": 0.85, "rechazado": 0.6},
        },
    )
    w_lo = _clamp01(float(la.get("w", 0.2)))
    fpc = la.get("factor_por_clase", {})
    clase = str(stats.get("loocv_class", "robusto"))
    fac = float(fpc.get(clase, 1.0))
    conf_final = _clamp01(conf_base * ((1.0 - w_lo) + w_lo * fac))
    return conf_final


def seleccionar_predictores_validos(
    df: pd.DataFrame,
    objetivo: str,
    idx_objetivo: int,
    nivel_familia: Optional[int] = None,
    min_datos_validos: int = 5,
) -> Tuple[pd.DataFrame, str, bool]:
    """
    Devuelve un DF para imputar la fila idx_objetivo y la familia utilizada.
    • Mantiene idx_objetivo (objetivo = NaN) + todas las filas cuyo objetivo NO sea NaN.
    • NO elimina filas con NaNs en otros predictores.
    • Elimina columnas que en idx_objetivo valgan NaN.
    • Aplica filtrado progresivo de familia (F0, F1, F2, sin filtro) cuando nivel_familia es None.
    • Si nivel_familia es 0,1,2 intenta SOLO esa familia; si no reúne criterio (>=min_datos_validos válidos) retorna DF vacío y familia_usada="".
        • Filtrado por rango configurable: por defecto estricto (0% tolerancia). Si correlacion.extrapolacion.modo_predictores == 'permitir_con_tolerancia',
            se permite una tolerancia relativa al span del entrenamiento (tolerancia_pct).
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
            if n_validos >= min_datos_validos:
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
                if n_validos >= min_datos_validos:
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

    # 4) Validación de rango: eliminar predictores según política de extrapolación
    ex_cfg = _COR().get("extrapolacion", {})
    modo_pred = ex_cfg.get("modo_predictores", "eliminar")
    tol_pct = (
        float(ex_cfg.get("tolerancia_pct", 0.0))
        if modo_pred == "permitir_con_tolerancia"
        else 0.0
    )
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

        # Usar rango del entrenamiento con tolerancia opcional
        mn, mx = valores.min(), valores.max()
        valor_objetivo = pd.to_numeric(df.at[idx_objetivo, col], errors="coerce")

        # Tolerancia relativa al span
        span = float(mx - mn) if pd.notna(mx) and pd.notna(mn) else 0.0
        pad = span * tol_pct

        # Si el valor está fuera del rango (con pad), eliminar el predictor completamente
        if not (
            pd.notna(valor_objetivo) and (mn - pad) <= valor_objetivo <= (mx + pad)
        ):
            columnas_a_eliminar.append(col)
            logger.debug(
                f"Eliminando predictor '{col}': valor_objetivo={valor_objetivo} fuera del rango [{mn - pad:.3f}, {mx + pad:.3f}] (pad={pad:.3f})"
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
    # Opcionales de manejo de outliers (por defecto activado y sin remoción dura)
    manejar_outliers: bool = True,
    umbral_z_suave: float = 3.0,
    umbral_z_duro: float = 6.0,
    alpha_pesos: float = 0.5,
    w_min: float = 0.2,
    remover_duro: bool = False,
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
    # Datos de entrenamiento efectivos (post outliers duros si aplica)
    datos_entrenamiento = datos_originales
    indices_entrenamiento = df_train.index.tolist()
    X_entrenamiento_original_list = X_df_original.values.tolist()
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
    min_unique = _min_unicos(tipo_prelim)
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
            X_vals = np.array(X_df.values, dtype=float).reshape(-1, 1)
            y_vals = np.array(df_train[objetivo].values, dtype=float)
            # Pesos robustos (sobre espacio original)
            w_fit = None
            mask_keep = np.ones(X_vals.shape[0], dtype=bool)
            if manejar_outliers:
                w, mask, info_out = calcular_pesos_outliers(
                    X_vals.ravel(),
                    y_vals,
                    umbral_z_suave=umbral_z_suave,
                    umbral_z_duro=umbral_z_duro,
                    alpha_pesos=alpha_pesos,
                    w_min=w_min,
                    remover_duro=remover_duro,
                )
                mask_keep = mask
                # Aplicar exclusión dura solo si mantiene el mínimo
                min_req = _min_muestras("log-1")
                if remover_duro and np.count_nonzero(mask_keep) >= min_req:
                    X_vals = X_vals[mask_keep]
                    y_vals = y_vals[mask_keep]
                    w_fit = w[mask_keep]
                    indices_entrenamiento = df_train.index[mask_keep].tolist()
                else:
                    w_fit = w
                    indices_entrenamiento = df_train.index.tolist()
                # Actualizar datos de entrenamiento efectivos
                datos_entrenamiento = {
                    "X_original": X_vals.tolist(),
                    "y_original": y_vals.tolist(),
                    "columnas_predictores": list(predictores),
                }
                X_entrenamiento_original_list = X_vals.tolist()
            # Transformaciones específicas
            X_transformed = np.log(X_vals)
            y_transformed = y_vals
            min_required = _min_muestras("log-1")
            tipo = "log-1"
            tipo_transformacion = "logarítmica"
            pf = None
            scaler_X = None
            scaler_y = None
            if len(df_train) < min_required:
                return None
            # Entrenar modelo
            if manejar_outliers and w_fit is not None:
                modelo = LinearRegression().fit(
                    X_transformed.reshape(-1, 1), y_transformed, sample_weight=w_fit
                )
            else:
                modelo = LinearRegression().fit(
                    X_transformed.reshape(-1, 1), y_transformed
                )
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
            X_vals = np.array(X_df.values, dtype=float).reshape(-1, 1)
            y_vals = np.array(y_df.values, dtype=float)
            # Pesos robustos
            w_fit = None
            mask_keep = np.ones(X_vals.shape[0], dtype=bool)
            if manejar_outliers:
                w, mask, info_out = calcular_pesos_outliers(
                    X_vals.ravel(),
                    y_vals,
                    umbral_z_suave=umbral_z_suave,
                    umbral_z_duro=umbral_z_duro,
                    alpha_pesos=alpha_pesos,
                    w_min=w_min,
                    remover_duro=remover_duro,
                )
                mask_keep = mask
                min_req = _min_muestras("pot-1")
                if remover_duro and np.count_nonzero(mask_keep) >= min_req:
                    X_vals = X_vals[mask_keep]
                    y_vals = y_vals[mask_keep]
                    w_fit = w[mask_keep]
                    indices_entrenamiento = df_train.index[mask_keep].tolist()
                else:
                    w_fit = w
                    indices_entrenamiento = df_train.index.tolist()
                datos_entrenamiento = {
                    "X_original": X_vals.tolist(),
                    "y_original": y_vals.tolist(),
                    "columnas_predictores": list(predictores),
                }
                X_entrenamiento_original_list = X_vals.tolist()
            X_transformed = np.log(X_vals)
            y_transformed = np.log(y_vals)
            min_required = _min_muestras("pot-1")
            tipo = "pot-1"
            tipo_transformacion = "potencia"
            pf = None
            scaler_X = None
            scaler_y = None
            if len(df_train) < min_required:
                return None

            # Entrenar modelo
            if manejar_outliers and w_fit is not None:
                modelo = LinearRegression().fit(
                    X_transformed.reshape(-1, 1), y_transformed, sample_weight=w_fit
                )
            else:
                modelo = LinearRegression().fit(
                    X_transformed.reshape(-1, 1), y_transformed
                )
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
            X_vals = np.array(X_df.values, dtype=float).reshape(-1, 1)
            y_vals = np.array(y_df.values, dtype=float)
            # Pesos robustos
            w_fit = None
            mask_keep = np.ones(X_vals.shape[0], dtype=bool)
            if manejar_outliers:
                w, mask, info_out = calcular_pesos_outliers(
                    X_vals.ravel(),
                    y_vals,
                    umbral_z_suave=umbral_z_suave,
                    umbral_z_duro=umbral_z_duro,
                    alpha_pesos=alpha_pesos,
                    w_min=w_min,
                    remover_duro=remover_duro,
                )
                mask_keep = mask
                min_req = _min_muestras("exp-1")
                if remover_duro and np.count_nonzero(mask_keep) >= min_req:
                    X_vals = X_vals[mask_keep]
                    y_vals = y_vals[mask_keep]
                    w_fit = w[mask_keep]
                    indices_entrenamiento = df_train.index[mask_keep].tolist()
                else:
                    w_fit = w
                    indices_entrenamiento = df_train.index.tolist()
                datos_entrenamiento = {
                    "X_original": X_vals.tolist(),
                    "y_original": y_vals.tolist(),
                    "columnas_predictores": list(predictores),
                }
                X_entrenamiento_original_list = X_vals.tolist()
            X_transformed = X_vals
            y_transformed = np.log(y_vals)
            min_required = _min_muestras("exp-1")
            tipo = "exp-1"
            tipo_transformacion = "exponencial"
            pf = None
            scaler_X = None
            scaler_y = None
            if len(df_train) < min_required:
                return None

            # Entrenar modelo
            if manejar_outliers and w_fit is not None:
                modelo = LinearRegression().fit(
                    X_transformed.reshape(-1, 1), y_transformed, sample_weight=w_fit
                )
            else:
                modelo = LinearRegression().fit(
                    X_transformed.reshape(-1, 1), y_transformed
                )
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
                min_required = _min_muestras(tipo_poly)
            else:
                tipo_lin = f"linear-{len(predictores)}"
                min_required = _min_muestras(tipo_lin)
            if len(df_train) < min_required:
                return None
            X_df = df_train[list(predictores)]
            X_raw = np.array(X_df.values, dtype=float)
            y_raw = np.array(df_train[objetivo].values, dtype=float)

            # Pesos robustos en espacio original (1D: ravel, 2D: norma fila)
            w_fit = None
            mask_keep = np.ones(X_raw.shape[0], dtype=bool)
            if manejar_outliers:
                if X_raw.shape[1] == 1:
                    x_for_out = X_raw.ravel()
                else:
                    x_for_out = np.linalg.norm(X_raw, axis=1)
                w, mask, info_out = calcular_pesos_outliers(
                    x_for_out,
                    y_raw,
                    umbral_z_suave=umbral_z_suave,
                    umbral_z_duro=umbral_z_duro,
                    alpha_pesos=alpha_pesos,
                    w_min=w_min,
                    remover_duro=remover_duro,
                )
                mask_keep = mask
                # Decidir tipo prelim para mínimo de muestras
                tipo_min = (
                    f"poly-{len(predictores)}" if poly else f"linear-{len(predictores)}"
                )
                min_req = _min_muestras(tipo_min)
                if remover_duro and np.count_nonzero(mask_keep) >= min_req:
                    X_raw = X_raw[mask_keep]
                    y_raw = y_raw[mask_keep]
                    w_fit = w[mask_keep]
                    indices_entrenamiento = df_train.index[mask_keep].tolist()
                else:
                    w_fit = w
                    indices_entrenamiento = df_train.index.tolist()
                datos_entrenamiento = {
                    "X_original": X_raw.tolist(),
                    "y_original": y_raw.tolist(),
                    "columnas_predictores": list(predictores),
                }
                X_entrenamiento_original_list = X_raw.tolist()

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
            if manejar_outliers and w_fit is not None:
                modelo = LinearRegression().fit(
                    X_trans, y_transformed, sample_weight=w_fit
                )
            else:
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
        # Preparar stats para confianza avanzada
        try:
            n_eff = int(len(y_vals)) if "y_vals" in locals() else int(len(y_raw))
        except Exception:
            n_eff = len(df_train)
        stats_conf: dict[str, float] = {
            "r2": float(r2),
            "mape": float(mape),
            "k": int(n_eff),
            "n": int(n_eff),
        }
        # Si 2D, adjuntar métricas geométricas/calculadas del early check
        if len(predictores) == 2:
            # Recalcular o usar geom_metrics si estaba disponible
            try:
                X_tmp = np.array(df_train[list(predictores)].values, dtype=float)
                ok2d, geom_metrics, reasons = _early_checks_2d(X_tmp, tipo_prelim)
                # Guardar métricas aunque ok2d sea False (no descartar aquí)
                stats_conf.update(
                    {
                        "pearson_r": float(geom_metrics.get("pearson_r", np.nan)),
                        "vif": float(geom_metrics.get("vif", np.nan)),
                        "cond": float(geom_metrics.get("cond", np.nan)),
                        "pc2_ratio": float(geom_metrics.get("pc2_ratio", np.nan)),
                        "unique_pair_ratio": float(
                            geom_metrics.get("unique_pair_ratio", np.nan)
                        ),
                        "hull_ratio": float(geom_metrics.get("hull_ratio", np.nan)),
                        "ellipse_ratio": float(
                            geom_metrics.get("ellipse_ratio", np.nan)
                        ),
                    }
                )
                # Añadir alias esperados por penalizaciones (pearson_abs y coverage_*)
                try:
                    if "pearson_r" in stats_conf and np.isfinite(
                        stats_conf["pearson_r"]
                    ):
                        stats_conf["pearson_abs"] = abs(float(stats_conf["pearson_r"]))
                except Exception:
                    pass
                if "unique_pair_ratio" in stats_conf:
                    stats_conf["coverage_unique_pair"] = stats_conf["unique_pair_ratio"]
                if "hull_ratio" in stats_conf:
                    stats_conf["coverage_hull"] = stats_conf["hull_ratio"]
                if "ellipse_ratio" in stats_conf:
                    stats_conf["coverage_ellipse"] = stats_conf["ellipse_ratio"]
            except Exception:
                pass
        # Calcular correlación base para reporte y confianza avanzada para ranking
        conf_cfg = _COR().get("confianza", {})
        w_r2 = float(conf_cfg.get("w_r2", 0.5))
        w_mp = float(conf_cfg.get("w_mape", 0.5))
        div_m = float(conf_cfg.get("mape_divisor", 15.0))
        corr = w_r2 * r2 + w_mp * (1 - min(mape / max(div_m, 1e-9), 1.0))
        confianza = calcular_confianza_modelo(stats_conf)
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
            "n": n_eff,
            "n_predictores": len(predictores),
            "datos_originales": datos_originales,
            "datos_entrenamiento": datos_entrenamiento,
            "indices_entrenamiento": indices_entrenamiento,
            "X_entrenamiento_original": X_entrenamiento_original_list,
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
        # Propagar métricas 2D y n/k al resultado para uso posterior (LOOCV/penalizaciones)
        try:
            resultado["k"] = int(stats_conf.get("k", len(df_train)))
            resultado["n"] = int(stats_conf.get("n", len(df_train)))
            for kmet in (
                "pearson_r",
                "vif",
                "cond",
                "pc2_ratio",
                "unique_pair_ratio",
                "hull_ratio",
                "ellipse_ratio",
            ):
                if kmet in stats_conf:
                    resultado[kmet] = stats_conf[kmet]
        except Exception:
            pass
        # Info de outliers (si se calcularon)
        if manejar_outliers:
            try:
                # info_out puede no existir si no se ejecutó el bloque (p.ej., sin remover)
                resultado["Outliers_duros_removidos"] = int(
                    np.nansum(~mask_keep) if remover_duro else int(0)
                )
                # Suaves: estimación aproximada (no duros) marcados por soft en utilitario; si no está disponible, usar 0
                # Nota: Para simplicidad, registramos la cantidad total de pesos < 1 como 'suaves'.
                if "w_fit" in locals() and w_fit is not None:
                    resultado["Outliers_suaves_pesados"] = int(np.sum((w_fit < 0.9999)))
                else:
                    resultado["Outliers_suaves_pesados"] = 0
            except Exception:
                resultado["Outliers_duros_removidos"] = 0
                resultado["Outliers_suaves_pesados"] = 0
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
    sel_cfg = _COR().get("seleccion_modelos", {})
    pre = sel_cfg.get("pre_filtro", {"mape_max": 18.0, "r2_min": 0.4})
    train_thr = sel_cfg.get("train", {"mape_max": 7.5, "r2_min": 0.6})
    for m in modelos:
        if m is None:
            continue
        mape = m.get("mape", np.inf)
        r2 = m.get("r2", -np.inf)
        # Descartar modelos inválidos
        if mape > float(pre.get("mape_max", 18.0)) or r2 < float(
            pre.get("r2_min", 0.4)
        ):
            continue
        # Etiquetar como no robusto si está en zona intermedia
        motivo = ""
        if (
            float(train_thr.get("mape_max", 7.5))
            < mape
            <= float(pre.get("mape_max", 18.0))
        ) or (float(pre.get("r2_min", 0.4)) < r2 < float(train_thr.get("r2_min", 0.6))):
            motivo = "Modelo no robusto: "
            if mape > float(train_thr.get("mape_max", 7.5)):
                motivo += f"MAPE fuera de rango (>{float(train_thr.get('mape_max',7.5))}%, <= {float(pre.get('mape_max',18.0))}%) "
            if r2 < float(train_thr.get("r2_min", 0.6)):
                motivo += f"R2 fuera de rango (>{float(pre.get('r2_min',0.4))}, < {float(train_thr.get('r2_min',0.6))})"
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
        # Pesos de outliers opcionales en LOOCV
        usar_pesos = bool(
            _MODELOS().get("loocv_usa_pesos", False)
            or _COR().get("loocv", {}).get("usar_pesos_outliers", False)
        )
        sample_weight = None
        if usar_pesos:
            # Pesar en espacio original: x como norma/ravel de X_tr_raw, y como y_tr_raw
            try:
                if X_tr_raw.shape[1] == 1:
                    x_for_out = X_tr_raw.ravel()
                else:
                    x_for_out = np.linalg.norm(X_tr_raw, axis=1)
            except Exception:
                x_for_out = np.arange(y_tr.shape[0])  # fallback estable
            w, mask_keep, _ = calcular_pesos_outliers(
                x_for_out,
                y_tr_raw,
                umbral_z_suave=_OUTLIERS().get("umbral_z_suave", 3.0),
                umbral_z_duro=_OUTLIERS().get("umbral_z_duro", 6.0),
                alpha_pesos=_OUTLIERS().get("alpha_pesos", 0.5),
                w_min=_OUTLIERS().get("w_min", 0.2),
                remover_duro=_OUTLIERS().get("remover_duro", False),
            )
            # aplicar máscara si corresponde
            try:
                if (
                    mask_keep is not None
                    and mask_keep.shape[0] == y_tr.shape[0]
                    and np.any(~mask_keep)
                ):
                    X_tr = X_tr[mask_keep]
                    y_tr = y_tr[mask_keep]
                    sample_weight = w[mask_keep]
                else:
                    sample_weight = w
            except Exception:
                sample_weight = w
        reg = LinearRegression().fit(X_tr, y_tr, sample_weight=sample_weight)
        y_hat_scaled = reg.predict(X_te)[0]
        y_hat = scaler_y.inverse_transform([[y_hat_scaled]])[0, 0]

        preds[i] = y_hat
        # evitar división por cero
        denom = y_te_raw[0] if y_te_raw[0] != 0 else 1e-9
        errors[i] = abs((y_te_raw[0] - y_hat) / denom)

    MAPE_LOOCV = errors.mean() * 100
    R2_LOOCV = r2_score(np.array(y_full), np.array(preds))
    conf_cfg = _COR().get("confianza", {})
    w_r2 = float(conf_cfg.get("w_r2", 0.5))
    w_mp = float(conf_cfg.get("w_mape", 0.5))
    div_m = float(conf_cfg.get("mape_divisor", 15.0))
    Corr_LOOCV = w_r2 * R2_LOOCV + w_mp * (1 - min(MAPE_LOOCV / max(div_m, 1e-9), 1.0))
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

    # ── Extrapolación configurable ───────────────────────────
    advert_extrap = ""
    df_train = df_filtrado.dropna(subset=[objetivo, *info["predictores"]])
    ex_cfg = _COR().get("extrapolacion", {})
    modo_pred = ex_cfg.get("modo_predictores", "eliminar")
    tol_pct = float(ex_cfg.get("tolerancia_pct", 0.0))
    modo_2d = ex_cfg.get("modo_2d", "marginal")
    hull_pad = float(ex_cfg.get("tolerancia_hull_pad", 0.0))

    # Marginal: permite tolerancia sobre el rango [min,max]
    def _marginal_ok(col, v):
        rmin = df_train[col].min()
        rmax = df_train[col].max()
        span = float(rmax - rmin) if pd.notna(rmax) and pd.notna(rmin) else 0.0
        pad = span * tol_pct
        return (pd.notna(v)) and ((rmin - pad) <= v <= (rmax + pad))

    # Convex hull (2D): punto dentro del hull (con padding aproximado)
    def _in_convex_hull(
        xy_train: np.ndarray, pt: tuple[float, float], pad: float = 0.0
    ) -> bool:
        pts = sorted(set(map(tuple, xy_train.tolist())))
        if len(pts) < 3:
            return True  # sin hull útil => no consideramos extrapolación

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
        x, y = pt
        inside = False
        for i in range(len(hull)):
            x1, y1 = hull[i]
            x2, y2 = hull[(i + 1) % len(hull)]
            if ((y1 > y) != (y2 > y)) and (
                x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1
            ):
                inside = not inside
        return inside

    preds = info["predictores"]
    if len(preds) == 2 and modo_2d == "convex_hull":
        xy = df_train[preds].dropna().values
        vx = _to_float_safe(df_filtrado.at[idx, preds[0]])
        vy = _to_float_safe(df_filtrado.at[idx, preds[1]])
        try:
            if np.isfinite(vx) and np.isfinite(vy):
                if not _in_convex_hull(xy, (vx, vy), pad=hull_pad):
                    advert_extrap = "Extrapolacion"
        except Exception:
            pass
    else:
        for col in preds:
            v = df_filtrado.at[idx, col]
            if not _marginal_ok(col, v):
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
        "Outliers_duros_removidos": info.get("Outliers_duros_removidos", 0),
        "Outliers_suaves_pesados": info.get("Outliers_suaves_pesados", 0),
    }

    return df_resultado, imputacion


def imputaciones_correlacion(
    df: pd.DataFrame | str,
    exportar_modelos: bool = False,
    ruta_export: Optional[str] = None,
    permitir_sin_filtro: bool = False,
    # Config knobs (opcionales, con defaults que replican el comportamiento actual)
    min_datos_validos: int = 5,
    modelos_habilitados: Optional[Dict[str, bool]] = None,
    umbral_mape_max: float = 7.5,
    usar_loocv: bool = True,
    loocv_usa_pesos: bool = False,
    # Control de verbosidad
    verbose: bool = True,
    # Outliers/robustez
    manejar_outliers: bool = True,
    umbral_z_suave: float = 3.0,
    umbral_z_duro: float = 6.0,
    alpha_pesos: float = 0.5,
    w_min: float = 0.2,
    remover_duro: bool = False,
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

    # Logger interno controlado por 'verbose'
    def _log(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

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
                    df_original,
                    objetivo,
                    idx,
                    nivel_familia=capa,
                    min_datos_validos=min_datos_validos,
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
                        # Respetar configuraciones de modelos habilitados
                        if poly:
                            if (
                                modelos_habilitados is not None
                                and not modelos_habilitados.get("polinomico2", True)
                            ):
                                continue
                        else:
                            if (
                                modelos_habilitados is not None
                                and not modelos_habilitados.get("lineal", True)
                            ):
                                continue
                        try:
                            modelos.append(
                                entrenar_modelo(
                                    df_filtrado,
                                    objetivo,
                                    combo,
                                    poly,
                                    idx,
                                    # Outliers/robustez
                                    manejar_outliers=manejar_outliers,
                                    umbral_z_suave=umbral_z_suave,
                                    umbral_z_duro=umbral_z_duro,
                                    alpha_pesos=alpha_pesos,
                                    w_min=w_min,
                                    remover_duro=remover_duro,
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
                            if modelos_habilitados is not None:
                                if (
                                    modelo_extra == "log"
                                    and not modelos_habilitados.get("log", True)
                                ):
                                    continue
                                if (
                                    modelo_extra == "potencia"
                                    and not modelos_habilitados.get("potencia", True)
                                ):
                                    continue
                                if (
                                    modelo_extra == "exponencial"
                                    and not modelos_habilitados.get("exponencial", True)
                                ):
                                    continue
                            try:
                                modelos.append(
                                    entrenar_modelo(
                                        df_filtrado,
                                        objetivo,
                                        combo,
                                        False,
                                        idx,
                                        modelo_extra=modelo_extra,
                                        # Outliers/robustez
                                        manejar_outliers=manejar_outliers,
                                        umbral_z_suave=umbral_z_suave,
                                        umbral_z_duro=umbral_z_duro,
                                        alpha_pesos=alpha_pesos,
                                        w_min=w_min,
                                        remover_duro=remover_duro,
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
                train_cfg = _COR().get("seleccion_modelos", {}).get("train", {})
                train_mape_max = float(train_cfg.get("mape_max", umbral_mape_max))
                train_r2_min = float(train_cfg.get("r2_min", 0.6))
                validos = [
                    m
                    for m in predictivos
                    if m["mape"] <= train_mape_max and m["r2"] >= train_r2_min
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
                        df_original,
                        objetivo,
                        idx,
                        nivel_familia=None,
                        min_datos_validos=min_datos_validos,
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
                                    if poly:
                                        if (
                                            modelos_habilitados is not None
                                            and not modelos_habilitados.get(
                                                "polinomico2", True
                                            )
                                        ):
                                            continue
                                    else:
                                        if (
                                            modelos_habilitados is not None
                                            and not modelos_habilitados.get(
                                                "lineal", True
                                            )
                                        ):
                                            continue
                                    try:
                                        modelos.append(
                                            entrenar_modelo(
                                                df_filtrado,
                                                objetivo,
                                                combo,
                                                poly,
                                                idx,
                                                manejar_outliers=manejar_outliers,
                                                umbral_z_suave=umbral_z_suave,
                                                umbral_z_duro=umbral_z_duro,
                                                alpha_pesos=alpha_pesos,
                                                w_min=w_min,
                                                remover_duro=remover_duro,
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
                                        if modelos_habilitados is not None:
                                            if (
                                                modelo_extra == "log"
                                                and not modelos_habilitados.get(
                                                    "log", True
                                                )
                                            ):
                                                continue
                                            if (
                                                modelo_extra == "potencia"
                                                and not modelos_habilitados.get(
                                                    "potencia", True
                                                )
                                            ):
                                                continue
                                            if (
                                                modelo_extra == "exponencial"
                                                and not modelos_habilitados.get(
                                                    "exponencial", True
                                                )
                                            ):
                                                continue
                                        try:
                                            modelos.append(
                                                entrenar_modelo(
                                                    df_filtrado,
                                                    objetivo,
                                                    combo,
                                                    False,
                                                    idx,
                                                    modelo_extra=modelo_extra,
                                                    manejar_outliers=manejar_outliers,
                                                    umbral_z_suave=umbral_z_suave,
                                                    umbral_z_duro=umbral_z_duro,
                                                    alpha_pesos=alpha_pesos,
                                                    w_min=w_min,
                                                    remover_duro=remover_duro,
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
                            train_cfg = (
                                _COR().get("seleccion_modelos", {}).get("train", {})
                            )
                            train_mape_max = float(
                                train_cfg.get("mape_max", umbral_mape_max)
                            )
                            train_r2_min = float(train_cfg.get("r2_min", 0.6))
                            validos = [
                                m
                                for m in predictivos
                                if m["mape"] <= train_mape_max
                                and m["r2"] >= train_r2_min
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
                    if usar_loocv:
                        for m in modelos_validos:
                            lo = validar_con_loocv(df_filtrado_usado, objetivo, m)
                            m.update(lo)
                            # Clasificar LOOCV (opcional, para trazabilidad/penalizaciones externas)
                            lo_class = _clasificar_loocv(
                                m.get("MAPE_LOOCV", np.inf), m.get("R2_LOOCV", -np.inf)
                            )
                            m["loocv_class"] = lo_class
                            # Mezcla final de confianza: base vs LOOCV segun peso w
                            la = _COR().get("confianza", {}).get("loocv_aporte", {})
                            w = float(la.get("w", 0.2))
                            w = max(0.0, min(1.0, w))
                            conf_base = float(m.get("Confianza", 0.0))
                            conf_loocv = float(m.get("Confianza_LOOCV", 0.0))
                            conf_mix = (1.0 - w) * conf_base + w * conf_loocv
                            m["Confianza_promedio"] = conf_mix
                            m["Confianza"] = conf_mix
                            # Ratio de validación vs entrenamiento por modelo
                            mape_tr = m.get("mape", None)
                            mape_val = m.get("MAPE_LOOCV", None)
                            if mape_tr is not None and mape_val is not None:
                                ratio = np.inf if mape_tr == 0 else mape_val / mape_tr
                                m["ratio_MAPE_val_vs_train"] = ratio
                                limite_ratio = float(
                                    _COR()
                                    .get("loocv", {})
                                    .get("ratio_val_train_alerta", 5.0)
                                )
                                if ratio > limite_ratio:
                                    advert_msg = (
                                        f"Advertencia: El MAPE de validación es más de {limite_ratio:.0f} veces mayor que el de entrenamiento "
                                        f"({mape_val:.2f}% vs {mape_tr:.2f}%). Posible sobreajuste."
                                    )
                                    if "Advertencia" in m and m["Advertencia"]:
                                        m["Advertencia"] += "; " + advert_msg
                                    else:
                                        m["Advertencia"] = advert_msg
                            else:
                                m["ratio_MAPE_val_vs_train"] = None
                    else:
                        for m in modelos_validos:
                            m["n_LOOCV"] = m.get("n", 0)
                            m["MAPE_LOOCV"] = m["mape"]
                            m["R2_LOOCV"] = m["r2"]
                            m["Corr_LOOCV"] = m["corr"]
                            m["Confianza_LOOCV"] = m["Confianza"]
                            m["Confianza_promedio"] = m["Confianza"]
                            # En modo sin LOOCV, el ratio es 1 por construcción
                            m["ratio_MAPE_val_vs_train"] = 1.0
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
                                    "datos_entrenamiento": m.get(
                                        "datos_entrenamiento", m["datos_originales"]
                                    ),
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
                    crit = _COR().get("loocv", {}).get("criterios", {})
                    rob = crit.get("robusto", {"mape_max": 7.5, "r2_min": 0.6})
                    nor = crit.get("no_robusto", {"mape_max": 12.5, "r2_min": 0.45})
                    robustos = [
                        m
                        for m in modelos_validos
                        if m["MAPE_LOOCV"] <= float(rob.get("mape_max", 7.5))
                        and m["R2_LOOCV"] >= float(rob.get("r2_min", 0.6))
                    ]
                    no_robustos = [
                        m
                        for m in modelos_validos
                        if (
                            m["MAPE_LOOCV"] <= float(nor.get("mape_max", 12.5))
                            and m["R2_LOOCV"] >= float(nor.get("r2_min", 0.45))
                        )
                        and not (
                            m["MAPE_LOOCV"] <= float(rob.get("mape_max", 7.5))
                            and m["R2_LOOCV"] >= float(rob.get("r2_min", 0.6))
                        )
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
                        if verbose:
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
