import numpy as np

_EPS = 1e-12


def _robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    denom = 1.4826 * mad if mad > 0 else (np.nanstd(x) + _EPS)
    return np.abs(x - med) / denom


def _iqr_flags(x: np.ndarray, k: float = 1.5) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    q1, q3 = np.nanpercentile(x, [25, 75])
    iqr = q3 - q1
    low = q1 - k * iqr
    high = q3 + k * iqr
    return (x < low) | (x > high)


def calcular_pesos_outliers(
    x: np.ndarray,
    y: np.ndarray,
    umbral_z_suave: float = 3.0,
    umbral_z_duro: float = 6.0,
    alpha_pesos: float = 0.5,
    w_min: float = 0.2,
    remover_duro: bool = False,
):
    """
    Devuelve (weights, mask_keep, info_dict).
    - weights: pesos en [w_min, 1], 1 si no hay outliers.
    - mask_keep: máscara booleana de puntos a conservar (True=se queda).
    - info_dict: {'n_duros': int, 'n_suaves': int}
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = min(x.size, y.size)
    x, y = x[:n], y[:n]

    zx = _robust_z(x)
    zy = _robust_z(y)
    z = np.maximum(zx, zy)

    # IQR adicional (híbrido): marca como suaves si cae fuera por IQR aunque z < suave
    iqrx = _iqr_flags(x, 1.5)
    iqry = _iqr_flags(y, 1.5)
    soft_by_iqr = np.logical_or(iqrx, iqry)

    soft = (z >= umbral_z_suave) | soft_by_iqr
    hard = z >= umbral_z_duro

    # Pesos: primero por z, luego clip
    w = 1.0 / (1.0 + alpha_pesos * z)
    w = np.clip(w, w_min, 1.0)

    # Si remover_duro: excluye hard (pero deja soft con peso bajo)
    mask_keep = ~hard if remover_duro else np.ones_like(hard, dtype=bool)
    info = {
        "n_duros": int(np.nansum(hard)),
        "n_suaves": int(np.nansum(soft & ~hard)),
    }
    return w, mask_keep, info
