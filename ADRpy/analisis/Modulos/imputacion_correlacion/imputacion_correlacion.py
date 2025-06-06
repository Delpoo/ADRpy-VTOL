import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_percentage_error, r2_score


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
    df.replace("nan", np.nan, inplace=True)
    return df


def seleccionar_predictores_validos(df: pd.DataFrame, objetivo: str) -> list:
    """Return numeric predictors with at least 5 non-null values."""
    numericas = df.select_dtypes(include=[np.number]).columns
    return [
        col
        for col in numericas
        if col != objetivo and df[col].notna().sum() >= 5
    ]


def generar_combinaciones(predictores: list) -> list:
    combos = []
    for r in (1, 2):
        combos.extend(list(combinations(predictores, r)))
    return combos


def entrenar_modelo(df: pd.DataFrame, objetivo: str, predictores: tuple):
    df_train = df.dropna(subset=[objetivo, *predictores])
    if len(df_train) < len(predictores) + 1:
        return None
    X = df_train[list(predictores)]
    y = df_train[objetivo]
    poly = len(predictores) == 2
    if poly:
        pf = PolynomialFeatures(degree=2, include_bias=False)
        X_trans = pf.fit_transform(X)
    else:
        pf = None
        X_trans = X
    modelo = LinearRegression().fit(X_trans, y)
    pred = modelo.predict(X_trans)
    mape = mean_absolute_percentage_error(y, pred) * 100
    r2 = r2_score(y, pred)
    corr = 0.6 * (r2 / 0.7) + 0.4 * (1 - mape / 15)
    return {
        "predictores": predictores,
        "modelo": modelo,
        "pf": pf,
        "mape": mape,
        "r2": r2,
        "corr": corr,
    }


def filtrar_mejores_modelos(modelos: list, top: int = 2) -> list:
    modelos = [m for m in modelos if m is not None]
    modelos.sort(key=lambda m: m["corr"], reverse=True)
    return modelos[:top]


def validar_con_loocv(df: pd.DataFrame, objetivo: str, info: dict) -> float:
    df_train = df.dropna(subset=[objetivo, *info["predictores"]])
    if df_train.empty:
        return np.inf
    X = df_train[list(info["predictores"])]
    y = df_train[objetivo]
    if info["pf"] is not None:
        X = info["pf"].fit_transform(X)
    loo = LeaveOneOut()
    errores = []
    for train_idx, test_idx in loo.split(X):
        m = LinearRegression().fit(X[train_idx], y.iloc[train_idx])
        pred = m.predict(X[test_idx])
        errores.append(abs(y.iloc[test_idx].values[0] - pred[0]))
    return float(np.mean(errores))


def imputar_valores(df: pd.DataFrame, objetivo: str, info: dict):
    df_res = df.copy()
    faltantes = df_res[df_res[objetivo].isna()].index
    imputaciones = []
    if not len(faltantes):
        return df_res, imputaciones
    X_pred = df_res.loc[faltantes, list(info["predictores"])]
    if info["pf"] is not None:
        X_pred = info["pf"].transform(X_pred)
    df_res.loc[faltantes, objetivo] = info["modelo"].predict(X_pred)
    for idx in faltantes:
        imputaciones.append({"Fila": idx, "Parametro": objetivo, "Valor": df_res.at[idx, objetivo]})
    return df_res, imputaciones


def generar_reporte_final(registros: list) -> pd.DataFrame:
    return pd.DataFrame(registros)


def imputacion_correlacion(path: str = "ADRpy/analisis/Data/Datos_aeronaves.xlsx"):
    df = cargar_y_validar_datos(path)
    reporte = []
    for objetivo in [c for c in df.columns if df[c].isna().any()]:
        predictores = seleccionar_predictores_validos(df, objetivo)
        if not predictores:
            continue
        modelos = []
        for combo in generar_combinaciones(predictores):
            modelos.append(entrenar_modelo(df, objetivo, combo))
        mejores = filtrar_mejores_modelos(modelos)
        if not mejores:
            continue
        mejor = min(
            (
                (m, validar_con_loocv(df, objetivo, m))
                for m in mejores
            ),
            key=lambda t: t[1],
        )[0]
        df, imps = imputar_valores(df, objetivo, mejor)
        reporte.extend(imps)
    return df, generar_reporte_final(reporte)
