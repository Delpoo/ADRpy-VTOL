import pandas as pd
import numpy as np
from itertools import combinations
from collections import defaultdict
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_percentage_error, r2_score

class ModeloDescartado(Exception):
    """Modelo descartado por problemas numéricos."""
    def __init__(self, motivo: str):
        super().__init__(motivo)
        self.motivo = motivo

def penalizacion_por_k(k: int) -> float:
    """Return penalization factor based on sample size."""
    if k > 20:
        return 1.0
    return max(
        0,
        min(
            1,
            0.00002281 * (k/2) ** 5
            - 0.00024 * (k/2) ** 4
            - 0.0036 * (k/2) ** 3
            + 0.046 * (k/2) ** 2
            + 0.0095 * (k/2)
            + 0.024,
        ),
    )


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


def seleccionar_predictores_validos(
        df: pd.DataFrame,
        objetivo: str,
        idx_objetivo: int,
        rango: float = 0.15
) -> pd.DataFrame:
    """
    Devuelve un DF para imputar la fila idx_objetivo.
    • Mantiene idx_objetivo (objetivo = NaN)  +  todas las filas cuyo
      objetivo NO sea NaN.
    • NO elimina filas con NaNs en otros predictores.
    • Elimina columnas que en idx_objetivo valgan NaN.
    • Aplica tu filtrado por 'Misión' y rango ±15 %.
    """
    # 1) Conservar idx_objetivo + filas con objetivo conocido
    df = df[(df.index == idx_objetivo) | df[objetivo].notna()].copy()

    # 2) Quitar columnas con NaN en idx_objetivo
    columnas_validas = [
        c for c in df.columns
        if c == objetivo or pd.notna(df.at[idx_objetivo, c])
    ]
    df = df[columnas_validas]

    # 3) Filtrado por 'Misión'  (idéntico al tuyo)
    if 'Misión' in df.columns:
        mision_objetivo = df.at[idx_objetivo, 'Misión']
        df_mision = df[df['Misión'] == mision_objetivo]
        if len(df_mision) >= 5:
            df = df_mision
        df = df.drop(columns=['Misión'])

    # 4) Validación ±15 %  (idéntica a tu lógica)
    columnas_a_eliminar = []
    for col in df.columns:
        if col == df.columns[0] or col == objetivo:
            continue
        valores = pd.to_numeric(
            df[col].drop(index=idx_objetivo).dropna(), errors='coerce')
        if valores.empty or valores.isna().all():
            columnas_a_eliminar.append(col)
            continue
        mn, mx = valores.min(), valores.max()  # Obtener valores mínimo y máximo correctamente
        rango_min, rango_max = mn * (1 - rango), mx * (1 + rango)
        valor_objetivo = pd.to_numeric(df.at[idx_objetivo, col], errors='coerce')  # Asegurar conversión numérica
        if not (pd.notna(valor_objetivo) and rango_min <= valor_objetivo <= rango_max):
            columnas_a_eliminar.append(col)

    df = df.drop(columns=columnas_a_eliminar)
    return df


def generar_combinaciones(predictores: list) -> list:
    combos = []
    for r in (1, 2):
        combos.extend(list(combinations(predictores, r)))
    return combos
#! cuando se entrena un modelo de dos predictores se debe verificar que los datos observados correspondan a la misma aeronave, o sea puede ser que tenga 6 valores en cada predictor pero que correspondan a aeroanves diferentes, entonces no se puede colocar valores que no tengan su par
def entrenar_modelo(
    df_filtrado: pd.DataFrame, objetivo: str, predictores: tuple, poly: bool
) -> dict | None:
    """Train linear or polynomial model and compute metrics."""
    # Filtrar filas con valores válidos para el combo actual
    df_train = df_filtrado.dropna(subset=[objetivo, *predictores])

    # ----------------------------------------------------------
    # Calcular cuántos parámetros se ajustarán (incluye intercepto)
    if poly:
        pf_template = PolynomialFeatures(degree=2, include_bias=False)
        num_features = pf_template.fit(
            np.zeros((1, len(predictores)))
        ).n_output_features_
        num_coeficientes = num_features + 1  # +1 por el intercepto
    else:
        # Para el caso lineal: un coeficiente por predictor + intercepto
        num_coeficientes = len(predictores) + 1

    # Ajustar num_coeficientes para LOOCV
    num_coeficientes += 1  # Se agrega un valor extra para garantizar LOOCV
    # ----------------------------------------------------------

    # Validar que haya suficientes filas para entrenar el modelo
    if len(df_train) < num_coeficientes:
        print(f"Advertencia: Predictores {predictores} tienen valores insuficientes. Modelo descartado.")
        return None

    X_df = df_train[list(predictores)]
    X = X_df.values
    y = df_train[objetivo]

    # Normalización
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(np.array(y).reshape(-1, 1)).flatten()

    if poly:
        pf = PolynomialFeatures(degree=2, include_bias=False)
        X_trans = pf.fit_transform(X)
    else:
        pf = None
        X_trans = X
    try:
        # -------------  VALIDACIONES NUMÉRICAS  --------------------
        # 1) predictores constantes
        const_cols = [i for i in range(X.shape[1]) if np.isclose(X[:, i].var(), 0)]
        if const_cols:
            raise ModeloDescartado(f"Varianza cero en columnas {const_cols}")

        # 2) colinealidad perfecta
        if np.linalg.matrix_rank(X_trans) < X_trans.shape[1]:
            raise ModeloDescartado("Matriz de diseño singular (colinealidad perfecta)")

        # 3) condición numérica extrema
        if np.linalg.cond(X_trans) > 1e12:
            raise ModeloDescartado("Condición numérica > 1e12 (riesgo de inestabilidad)")

        # 4) objetivo constante
        if np.isclose(y.var(), 0):
            raise ModeloDescartado("Variable objetivo y es constante")
        # -----------------------------------------------------------

        modelo = LinearRegression().fit(X_trans, y)
        coeficientes = modelo.coef_
        intercepto = modelo.intercept_

        # Crear ecuación normalizada
        ecuacion_normalizada = f"y = {intercepto} + " + " + ".join([
            f"{coef}*x{i}" for i, coef in enumerate(coeficientes)
        ])

        # Desnormalizar coeficientes e intercepto
        if scaler_y.scale_ is not None and scaler_X.scale_ is not None:
            # --- DESNORMALIZACIÓN ROBUSTA ------------------------------
            if scaler_X is None or scaler_y is None:
                raise ValueError("Scaler objects are not initialized.")

            if scaler_X.scale_ is None or scaler_X.mean_ is None or scaler_y.scale_ is None or scaler_y.mean_ is None:
                raise ValueError("Scaler attributes are not properly initialized.")

            if poly:
                if pf is None or not hasattr(pf, 'powers_'):
                    raise ValueError("PolynomialFeatures object is not initialized or does not have 'powers_' attribute.")

                powers = pf.powers_   # (n_terms, n_predictores)
                escalas_ajustadas = np.prod(
                    np.power(scaler_X.scale_, powers), axis=1
                )
                mean_terms = np.prod(
                    np.power(scaler_X.mean_, powers), axis=1
                )
            else:
                escalas_ajustadas = scaler_X.scale_
                mean_terms         = scaler_X.mean_

            coef_desnormalizado = (
                coeficientes * scaler_y.scale_[0] / escalas_ajustadas
            )

            intercepto_desnormalizado = (
                scaler_y.mean_[0] - np.dot(coef_desnormalizado, mean_terms)
            )

            ecuacion_desnormalizada = (
                f"y = {intercepto_desnormalizado} + " +
                " + ".join(f"{coef}*x{i}"
                           for i, coef in enumerate(coef_desnormalizado))
            )
            # ------------------------------------------------------------
        else:
            coef_desnormalizado = coeficientes
            intercepto_desnormalizado = intercepto
            ecuacion_desnormalizada = ecuacion_normalizada

    
        pred = modelo.predict(X_trans)
        pred_desnormalizado = scaler_y.inverse_transform(pred.reshape(-1, 1)).flatten()
        y_desnormalizado = scaler_y.inverse_transform(y.reshape(-1, 1)).flatten()

        # Calcular el MAPE con los valores desnormalizados
        mape = float(mean_absolute_percentage_error(y_desnormalizado, pred_desnormalizado) * 100)

        r2 = r2_score(y, pred)
        corr = 0.5 * r2 + 0.5 * (1 - mape / 15)
        confianza = max(0, float(corr * penalizacion_por_k(len(df_train))))
        tipo = ("poly" if poly else "linear") + f"-{len(predictores)}"
        n = len(df_train)

        return {
            "descartado": False,
            "predictores": predictores,
            "modelo": modelo,
            "pf": pf,
            "mape": mape,
            "r2": r2,
            "corr": corr,
            "confianza": confianza,
            "tipo": tipo,
            "n": n,
            "ecuacion_normalizada": ecuacion_normalizada,
            "ecuacion_desnormalizada": ecuacion_desnormalizada,
            "scaler_X": scaler_X,
            "scaler_y": scaler_y,
        }
    except ModeloDescartado as e:
        # Devuelve solo la info necesaria para reportar el descarte
        return {
            "descartado": True,
            "motivo": e.motivo,
            "predictores": predictores,
            "tipo": ("poly" if poly else "linear") + f"-{len(predictores)}",
        }
#! en ecuaciones polinomiales de 1 predictor solo no me salta el error de matriz singular, me da una ecuacion valida ver tabla de word
def filtrar_mejores_modelos(modelos: list, top: int = 2) -> list:
    """Return top models per type based on confianza."""
    modelos = [m for m in modelos if m is not None and m["mape"] <= 7.5 and m["r2"] >= 0.6]
    grupos: defaultdict[str, list] = defaultdict(list)
    for m in modelos:
        grupos[m["tipo"]].append(m)
    mejores = []
    for lst in grupos.values():
        lst.sort(key=lambda x: x["confianza"], reverse=True)
        mejores.extend(lst[:top])
    return mejores


def validar_con_loocv(df: pd.DataFrame, objetivo: str, info: dict) -> dict:
    """
    Calcula MAPE_cv, R2_cv, Corr_cv y Confianza_cv usando Leave-One-Out,
    reproduciendo el pipeline real:
      • escalar X con StandardScaler
      • aplicar PolynomialFeatures si corresponde
      • escalar y
      • des-escalar la predicción antes de calcular el error
    """
    df_train = df.dropna(subset=[objetivo, *info["predictores"]])
    n = len(df_train)
    if n == 0:
        return {"MAPE_cv": np.inf, "R2_cv": -np.inf,
                "Corr_cv": -np.inf, "Confianza_cv": 0}

    X_full = df_train[list(info["predictores"])].values
    y_full = df_train[objetivo].values
    preds  = np.zeros(n)
    errors = np.zeros(n)

    loo = LeaveOneOut()
    for i, (tr, te) in enumerate(loo.split(X_full)):
        # ── entrenar escaladores en el subset de training ──
        scaler_X = StandardScaler().fit(X_full[tr])
        X_tr = scaler_X.transform(X_full[tr])
        X_te = scaler_X.transform(X_full[te])

        # polynomial si corresponde
        if info["pf"] is not None:
            pf = PolynomialFeatures(degree=2, include_bias=False)
            X_tr = pf.fit_transform(X_tr)
            X_te = pf.transform(X_te)

        # escalar y
        scaler_y = StandardScaler().fit(y_full[tr].reshape(-1, 1))
        y_tr = scaler_y.transform(y_full[tr].reshape(-1, 1)).flatten()

        # entrenar y predecir
        m = LinearRegression().fit(X_tr, y_tr)
        y_hat_scaled = m.predict(X_te)[0]
        # des-escalar la predicción
        y_hat = scaler_y.inverse_transform([[y_hat_scaled]])[0, 0]

        preds[i]  = y_hat
        # evitar división por cero
        denom = y_full[te][0] if y_full[te][0] != 0 else 1e-9
        errors[i] = abs((y_full[te][0] - y_hat) / denom)

    MAPE_cv = errors.mean() * 100
    R2_cv   = r2_score(np.array(y_full), np.array(preds))
    Corr_cv = 0.5 * R2_cv + 0.5 * (1 - MAPE_cv / 15)
    Conf_cv = max(0, Corr_cv * penalizacion_por_k(n))

    return {"MAPE_cv": MAPE_cv,
            "R2_cv": R2_cv,
            "Corr_cv": Corr_cv,
            "Confianza_cv": Conf_cv}

def generar_reporte_final(registros: list) -> pd.DataFrame:
    return pd.DataFrame(registros)


def imputacion_correlacion(df, path: str = "ADRpy/analisis/Data/Datos_aeronaves.xlsx"):
    df_original = cargar_y_validar_datos(path)  # Mantener el DataFrame original
    df_resultado = df_original.copy()  # DataFrame para almacenar imputaciones
    reporte = []

    for objetivo in [c for c in df_original.columns if df_original[c].isna().any()]:
        faltantes = df_original[df_original[objetivo].isna()].index
        for idx in faltantes:
            # Seleccionar predictores válidos para la celda actual
            df_filtrado = seleccionar_predictores_validos(df_original, objetivo, idx)
            if df_filtrado.empty:
                continue

            # Excluir la primera columna explícitamente
            predictores = [col for col in df_filtrado.columns if col != df_filtrado.columns[0] and col != objetivo]
            if not predictores:
                # Agregar advertencia al reporte
                reporte.append({
                    "Fila": idx,
                    "Parametro": objetivo,
                    "Valor imputado": "NAN",
                    "Confianza": 0,
                    "Corr": 0,
                    "k": 0,
                    "Tipo Modelo": "N/A",
                    "Predictores": "N/A",
                    "Penalizacion_k": 0,
                    "Advertencia": "No se pudo imputar por falta de parámetros válidos."
                })
                continue
            modelos = []
            for combo in generar_combinaciones(predictores):
                for poly in (False, True):
                    modelos.append(entrenar_modelo(df_filtrado, objetivo, combo, poly))

            validos     = [m for m in modelos if m is not None and not m["descartado"]]
            descartados = [m for m in modelos if m is not None and m["descartado"]]

            if not validos:
                # registra advertencias y NO imputa
                for m in descartados or [{"motivo": "Sin predictores válidos"}]:
                    for idx in faltantes:
                        reporte.append({
                            "Fila": idx,
                            "Parametro": objetivo,
                            "Valor imputado": np.nan,
                            "Confianza": 0.0,
                            "Corr": 0.0,
                            "k": 0,
                            "Tipo Modelo": m.get("tipo", "n/a"),
                            "Predictores": ",".join(m.get("predictores", [])),
                            "Penalizacion_k": 0.0,
                            "Advertencia": f"Modelo descartado: {m['motivo']}",
                        })
                continue

            mejores = filtrar_mejores_modelos(validos)
            for m in mejores:
                m.update(validar_con_loocv(df_filtrado, objetivo, m))

            robustos = [m for m in mejores if m["MAPE_cv"] <= 10 and m["R2_cv"] >= 0.6]
            candidatos = robustos or mejores
            warning_text = "Modelo robusto" if robustos else "Modelo no robusto"
            candidatos.sort(key=lambda x: (-x["Confianza_cv"], x["MAPE_cv"]))
            mejor = candidatos[0]
            mejor["warning"] = warning_text

            # Imputar el valor de la celda actual
            df_resultado, imputacion = imputar_valores_celda(df_resultado, df_filtrado, objetivo, mejor, idx)
            reporte.append(imputacion)

    return df_resultado, generar_reporte_final(reporte)


def imputar_valores_celda(df_resultado, df_filtrado, objetivo, info, idx):
    """Imputar el valor de una celda específica utilizando el modelo desnormalizado."""
    if idx not in df_filtrado.index:
        raise KeyError(f"Index {idx} is not present in the DataFrame.")

    # Obtener predictores de la fila (escala original)
    X_pred_df = df_filtrado.loc[[idx], list(info["predictores"])]
    X_pred    = X_pred_df.values
    # 1. Normalizar con el mismo scaler_X
    X_scaled = info["scaler_X"].transform(X_pred)

    # 2. Aplicar PolynomialFeatures si corresponde
    if info["pf"] is not None:
        X_scaled = info["pf"].transform(X_scaled)

    # 3. Predecir en la escala normalizada de y
    y_norm = info["modelo"].predict(X_scaled)[0]

    # 4. Des-normalizar la predicción
    valor = info["scaler_y"].inverse_transform([[y_norm]])[0, 0]
    
    # ── advertencia de extrapolación ─────────────────────────
    advert_extrap = ""
    df_train = df_filtrado.dropna(subset=[objetivo, *info["predictores"]])
    for col in info["predictores"]:
        rango_min = df_train[col].min() * 0.85
        rango_max = df_train[col].max() * 1.15
        v = df_filtrado.at[idx, col]
        if pd.isna(v) or not (rango_min <= v <= rango_max):
            advert_extrap = "Extrapolacion"
            break
    # Si el modelo ya traía un warning (“Modelo no robusto”), combina:
    advert_final = ", ".join(filter(None, [info.get("warning", ""), advert_extrap]))

    #5. imputar el valor en el DataFrame de resultado
    df_resultado.at[idx, objetivo] = valor

    imputacion = {
        "Fila": idx,
        "Parametro": objetivo,
        "Valor imputado": valor,
        "Confianza_train": info["confianza"],
        "Corr_train": info["corr"],
        "MAPE_train": info["mape"],
        "R2_train": info["r2"],
        "MAPE_cv": info["MAPE_cv"],
        "R2_cv": info["R2_cv"],
        "Corr_cv": info["Corr_cv"],
        "Confianza_cv": info["Confianza_cv"],
        "k": info["n"],
        "Tipo Modelo": info["tipo"],
        "Predictores": ",".join(info["predictores"]),
        "Penalizacion_k": penalizacion_por_k(info["n"]),
        "Advertencia": advert_final,
    }

    return df_resultado, imputacion

def test_imputacion_correlacion_basica():
    df_final, reporte = imputacion_correlacion('ADRpy/analisis/Data/Datos_aeronaves.xlsx')
    assert not df_final.isna().any().any(), "Deberia imputar todos los valores faltantes"
    print("listo")
test_imputacion_correlacion_basica()
