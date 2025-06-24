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

def is_missing(val):
    """
    Returns True if the value is considered missing (NaN, empty string, special codes, etc.).
    """
    if pd.isna(val):
        return True
    if isinstance(val, str) and val.strip().lower() in ["", "nan", "nan " "-", "#n/d", "n/d", "#¡valor!"]:
        return True
    return False



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
            0.00002281 * (k/2) ** 5
            - 0.00024 * (k/2) ** 4
            - 0.0036 * (k/2) ** 3
            + 0.046 * (k/2) ** 2
            + 0.0095 * (k/2)
            + 0.024,
        ),
    )


def seleccionar_predictores_validos(
        df: pd.DataFrame,
        objetivo: str,
        idx_objetivo: int,
        rango: float = 0.15
) -> tuple[pd.DataFrame, str, bool]:
    """
    Devuelve un DF para imputar la fila idx_objetivo y la familia utilizada.
    • Mantiene idx_objetivo (objetivo = NaN)  +  todas las filas cuyo
      objetivo NO sea NaN.
    • NO elimina filas con NaNs en otros predictores.
    • Elimina columnas que en idx_objetivo valgan NaN.
    • Aplica filtrado progresivo de familia (F0, F1, F2, sin filtro).
    """
    # 1) Conservar idx_objetivo + filas con objetivo conocido
    df = df[(df.index == idx_objetivo) | df[objetivo].notna()].copy()

    # 2) Quitar columnas con NaN en idx_objetivo
    columnas_validas = [
        c for c in df.columns
        if c == objetivo or pd.notna(df.at[idx_objetivo, c])
    ]
    df = df[columnas_validas]

    # 3) Filtrado progresivo de familia
    familia_usada = "sin filtro"
    filtro_aplicado = False
    capas_familia = [
        ["Misión", "Despegue", "Propulsión vertical", "Propulsión horizontal", "Cantidad de motores propulsión vertical", "Cantidad de motores propulsión horizontal"],
        ["Misión", "Despegue", "Propulsión vertical", "Propulsión horizontal"],
        ["Misión", "Despegue"]
    ]
    for i, capa in enumerate(capas_familia):
        if all(attr in df.columns for attr in capa):
            valores_obj = [df.at[idx_objetivo, attr] for attr in capa]
            mask = np.ones(df.shape[0], dtype=bool)
            for attr, val in zip(capa, valores_obj):
                mask &= (df[attr] == val).values
            df_fam = df[mask]
            # Solo contar filas con objetivo conocido (excluyendo idx_objetivo si está vacío)
            n_validos = df_fam[objetivo].notna().sum()
            if n_validos >= 5:
                df = df_fam
                familia_usada = f"F{i}"
                filtro_aplicado = True
                break
    # Eliminar columnas de familia si existen
    for col in ["Misión", "Despegue", "Propulsión vertical", "Propulsión horizontal", "Cantidad de motores propulsión vertical", "Cantidad de motores propulsión horizontal"]:
        if col in df.columns:
            df = df.drop(columns=[col])

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
    return df, familia_usada, filtro_aplicado


def generar_combinaciones(predictores: list) -> list:
    combos = []
    for r in (1, 2):
        combos.extend(list(combinations(predictores, r)))
    return combos

def entrenar_modelo(
    df_filtrado: pd.DataFrame, objetivo: str, predictores: tuple, poly: bool, idx: int, modelo_extra: str | None = None
) -> dict | None:
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
        "y_original": y_original.tolist(),           # Lista para JSON
        "columnas_predictores": list(predictores)     # Nombres de columnas
    }    # Inicializar variables comunes
    modelo = None
    coef_original = []
    intercepto_original = 0.0
    pred_original = np.array([])
    y_original_metrics = np.array([])
    ecuacion_normalizada = None
    ecuacion_desnormalizada = None
    tipo = "unknown"
    tipo_transformacion = "unknown"

    try:
        # Determinar tipo de modelo y aplicar transformaciones si es necesario
        ecuacion_string = None
        ecuacion_latex = None
        if modelo_extra == "log":
            # Logarítmico: y = a + b*log(x)
            if len(predictores) != 1:
                return None
            X_df = df_train[list(predictores)]
            if (X_df <= 0).any().any():
                return None
            X_transformed = np.log(np.array(X_df.values, dtype=float))
            y_transformed = np.array(df_train[objetivo].values, dtype=float)
            min_required = 5
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
            # Ecuación string y LaTeX
            var = str(predictores[0])
            ecuacion_string = f"y = {intercepto_original:.6g} + {coef_original[0]:.6g}*log({var})"
            ecuacion_latex = f"y = {intercepto_original:.6g} + {coef_original[0]:.6g} \\cdot \\log({var})"
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
            min_required = 5
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
            # Ecuación string y LaTeX
            var = str(predictores[0])
            ecuacion_string = f"y = {intercepto_original:.6g}*{var}**{coef_original[0]:.6g}"
            ecuacion_latex = f"y = {intercepto_original:.6g} \\cdot {var}^{{{coef_original[0]:.6g}}}"
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
            min_required = 5
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
            # Ecuación string y LaTeX
            var = str(predictores[0])
            ecuacion_string = f"y = {intercepto_original:.6g}*exp({coef_original[0]:.6g}*{var})"
            ecuacion_latex = f"y = {intercepto_original:.6g} \\cdot e^{{{coef_original[0]:.6g} \\cdot {var}}}"
        else:
            # Modelos lineales y polinómicos
            if poly:
                pf_template = PolynomialFeatures(degree=2, include_bias=False)
                num_features = pf_template.fit(
                    np.zeros((1, len(predictores)))
                ).n_output_features_
                num_coeficientes = num_features + 1  # +1 por el intercepto
                min_required = max(8, num_coeficientes)
            else:
                num_coeficientes = len(predictores) + 1
                min_required = max(5, num_coeficientes)
            num_coeficientes += 1  # Se agrega un valor extra para garantizar LOOCV
            if len(df_train) < min_required:
                return None
            X_df = df_train[list(predictores)]
            X_raw = np.array(X_df.values, dtype=float)
            y_raw = np.array(df_train[objetivo].values, dtype=float)
            
            # Normalización
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            X_transformed = scaler_X.fit_transform(X_raw)
            y_transformed = scaler_y.fit_transform(y_raw.reshape(-1, 1)).flatten()
            if poly:
                pf = PolynomialFeatures(degree=2, include_bias=False)
                X_trans = pf.fit_transform(X_transformed)
                tipo_transformacion = "polinómica+normalización"
            else:
                pf = None
                X_trans = X_transformed
                tipo_transformacion = "normalización"
            tipo = ("poly" if poly else "linear") + f"-{len(predictores)}"
            # Validaciones numéricas
            const_cols = [i for i in range(X_transformed.shape[1]) if np.isclose(X_transformed[:, i].var(), 0)]
            if const_cols:
                raise ModeloDescartado(f"Varianza cero en columnas {const_cols}")

            if np.linalg.matrix_rank(X_trans) < X_trans.shape[1]:
                raise ModeloDescartado("Matriz de diseño singular (colinealidad perfecta)")

            if np.linalg.cond(X_trans) > 1e12:
                raise ModeloDescartado("Condición numérica > 1e12 (riesgo de inestabilidad)")

            if np.isclose(y_transformed.var(), 0):
                raise ModeloDescartado("Variable objetivo y es constante")

            # Entrenar modelo
            modelo = LinearRegression().fit(X_trans, y_transformed)
            coeficientes = modelo.coef_
            intercepto = modelo.intercept_

            # Crear ecuación normalizada (string y LaTeX)
            ecuacion_normalizada = f"y = {intercepto} + " + " + ".join([
                f"{coef}*x{i}" for i, coef in enumerate(coeficientes)
            ])
            ecuacion_normalizada_latex = (
                "y = " + f"{intercepto:.6g}" + " + " + " + ".join([
                    f"{coef:.6g} x_{{{i}}}" for i, coef in enumerate(coeficientes)
                ])
            )
            # Desnormalizar coeficientes e intercepto
            if poly and pf is not None and hasattr(pf, 'powers_'):
                powers = pf.powers_
                if scaler_X.scale_ is not None and scaler_X.mean_ is not None:
                    escalas_ajustadas = np.prod(np.power(scaler_X.scale_, powers), axis=1)
                    mean_terms = np.prod(np.power(scaler_X.mean_, powers), axis=1)
                else:
                    # Fallback si los escaladores no están bien definidos
                    escalas_ajustadas = np.ones(len(coeficientes))
                    mean_terms = np.zeros(len(coeficientes))
            else:
                if scaler_X.scale_ is not None and scaler_X.mean_ is not None:
                    escalas_ajustadas = scaler_X.scale_
                    mean_terms = scaler_X.mean_
                else:
                    escalas_ajustadas = np.ones(len(coeficientes))
                    mean_terms = np.zeros(len(coeficientes))

            if scaler_y.scale_ is not None and scaler_y.mean_ is not None:
                coef_original = (coeficientes * scaler_y.scale_[0] / escalas_ajustadas).tolist()
                intercepto_original = float(scaler_y.mean_[0] - np.dot(coef_original, mean_terms))
            else:
                # Fallback si el escalador y no está bien definido
                coef_original = coeficientes.tolist()
                intercepto_original = float(intercepto)

            ecuacion_desnormalizada = (
                f"y = {intercepto_original} + " +
                " + ".join(f"{coef}*x{i}" for i, coef in enumerate(coef_original))
            )
            ecuacion_desnormalizada_latex = (
                "y = " + f"{intercepto_original:.6g}" + " + " + " + ".join([
                    f"{coef:.6g} x_{{{i}}}" for i, coef in enumerate(coef_original)
                ])
            )
            # Calcular predicciones y métricas en escala original
            pred_transformed = modelo.predict(X_trans)
            pred_original = scaler_y.inverse_transform(pred_transformed.reshape(-1, 1)).flatten()
            y_original_metrics = scaler_y.inverse_transform(y_transformed.reshape(-1, 1)).flatten()
            # Ecuación string y LaTeX (desnormalizada)
            ecuacion_string = ecuacion_desnormalizada
            ecuacion_latex = ecuacion_desnormalizada_latex
        # Calcular métricas siempre en escala original
        mape = float(mean_absolute_percentage_error(y_original_metrics, pred_original) * 100)
        r2 = r2_score(y_original_metrics, pred_original)
        corr = 0.5 * r2 + 0.5 * (1 - mape / 15)
        confianza = max(0, float(corr * penalizacion_por_k(len(df_train))))
        # Construir diccionario de retorno unificado
        resultado = {
            "descartado": False,
            "Aeronave": idx,
            "Parámetro": objetivo,
            "predictores": predictores,
            "tipo": tipo,
            "tipo_transformacion": tipo_transformacion,
            "n": len(df_train),
            # Datos en unidades originales
            "datos_originales": datos_originales,
            "coeficientes_originales": coef_original,
            "intercepto_original": intercepto_original,
            # Ecuaciones
            "ecuacion_string": ecuacion_string,
            "ecuacion_latex": ecuacion_latex,
            "ecuacion_normalizada": ecuacion_normalizada,
            "ecuacion_normalizada_latex": ecuacion_normalizada_latex if 'ecuacion_normalizada_latex' in locals() else None,
            # Métricas (calculadas en escala original)
            "mape": mape,
            "r2": r2,
            "corr": corr,
            "Confianza": confianza,
            # Objetos del modelo (para predicción posterior)
            "modelo": modelo,
            "pf": pf,
            "scaler_X": scaler_X,
            "scaler_y": scaler_y,
        }
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
def filtrar_mejores_modelos(modelos: list, top: int = 2) -> list:
    """Return top models per type based on Confianza."""
    modelos = [m for m in modelos if m is not None and m["mape"] <= 7.5 and m["r2"] >= 0.6]
    grupos: defaultdict[str, list] = defaultdict(list)
    for m in modelos:
        grupos[m["tipo"]].append(m)
    mejores = []
    for lst in grupos.values():
        lst.sort(key=lambda x: x["Confianza"], reverse=True)
        mejores.extend(lst[:top])
    return mejores


def validar_con_loocv(df: pd.DataFrame, objetivo: str, info: dict) -> dict:
    """
    Calcula MAPE_LOOCV, R2_LOOCV, Corr_LOOCV y Confianza_LOOCV usando Leave-One-Out,
    reproduciendo el pipeline real:
      • escalar X con StandardScaler
      • aplicar PolynomialFeatures si corresponde
      • escalar y
      • des-escalar la predicción antes de calcular el error
    """
    df_train = df.dropna(subset=[objetivo, *info["predictores"]])
    n_LOOCV = len(df_train)
    if n_LOOCV == 0:
        return {"MAPE_LOOCV": np.inf, "R2_LOOCV": -np.inf,
                "Corr_LOOCV": -np.inf, "Confianza_LOOCV": 0}

    X_full = df_train[list(info["predictores"])].values
    y_full = df_train[objetivo].values
    preds  = np.zeros(n_LOOCV)
    errors = np.zeros(n_LOOCV)

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

    MAPE_LOOCV = errors.mean() * 100
    R2_LOOCV   = r2_score(np.array(y_full), np.array(preds))
    Corr_LOOCV = 0.5 * R2_LOOCV + 0.5 * (1 - MAPE_LOOCV / 15)
    Conf_cv = max(0, Corr_LOOCV * penalizacion_por_k(n_LOOCV))

    return {"n_LOOCV": n_LOOCV,
            "MAPE_LOOCV": MAPE_LOOCV,
            "R2_LOOCV": R2_LOOCV,
            "Corr_LOOCV": Corr_LOOCV,
            "Confianza_LOOCV": Conf_cv}

def imputar_valores_celda(df_resultado, df_filtrado, objetivo, info, idx):
    """Imputar el valor de una celda específica utilizando el modelo desnormalizado."""
    if idx not in df_filtrado.index:
        raise KeyError(f"Index {idx} is not present in the DataFrame.")
    
    # Obtener predictores de la fila (escala original)
    X_pred_df = df_filtrado.loc[[idx], list(info["predictores"])]
    X_pred    = X_pred_df.values
    
    # Inicializar valor
    valor = np.nan# Detectar tipo de modelo especial
    tipo_modelo = str(info.get("tipo", "")).lower()
    if tipo_modelo.startswith("log") or tipo_modelo.startswith("pot") or tipo_modelo.startswith("exp"):
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
        if info["scaler_X"] is not None:
            X_scaled = info["scaler_X"].transform(X_pred)
        else:
            X_scaled = X_pred
            
        if info["pf"] is not None:
            X_scaled = info["pf"].transform(X_scaled)
            
        y_norm = info["modelo"].predict(X_scaled)[0]
        
        if info["scaler_y"] is not None:
            valor = info["scaler_y"].inverse_transform([[y_norm]])[0, 0]
        else:
            valor = y_norm

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
    advertencia_final = ", ".join(filter(None, [info.get("warning", ""), advert_extrap]))

    #5. imputar el valor en el DataFrame de resultado
    df_resultado.at[idx, objetivo] = valor

    imputacion = {
        "Aeronave" : idx,
        "Parámetro": objetivo,
        "Valor imputado": valor,
        "Confianza": info["Confianza"],
        "Tipo Modelo": info["tipo"],
        "Predictores": ",".join(info["predictores"]),
        "Aeronaves entrenamiento": list(df_train.index),
        "k": info["n"],
        "Penalizacion_k": penalizacion_por_k(info["n"]),
        "Corr": info["corr"],
        "MAPE": info["mape"],
        "R2": info["r2"],
        "Confianza_LOOCV": info["Confianza_LOOCV"],
        "k_LOOCV": info["n_LOOCV"],
        "Corr_LOOCV": info["Corr_LOOCV"],
        "MAPE_LOOCV": info["MAPE_LOOCV"],
        "R2_LOOCV": info["R2_LOOCV"],
        "Método predictivo": "Correlacion",
        "Advertencia": advertencia_final,
    }

    return df_resultado, imputacion

def imputaciones_correlacion(df, exportar_modelos: bool = False, ruta_export: str = None):
    if isinstance(df, str):
        df = pd.read_excel(df)
    df = df.rename(columns=lambda c: str(c).strip())
    # Reemplazar valores inválidos por np.nan
    df.replace("", np.nan, inplace=True)
    df_original = df.copy()  # <- Copia del DF original antes de filtrar
    df_resultado = df_original.copy()
    reporte = []
    modelos_info = []  # Lista de modelos completos (solo descartado=False)

    for objetivo in [c for c in df_original.columns if df_original[c].isna().any()]:
        faltantes = df_original[df_original[objetivo].isna()].index
        for idx in faltantes:
            # Seleccionar predictores válidos para la celda actual y familia usada
            df_filtrado, familia_usada, filtro_aplicado = seleccionar_predictores_validos(df_original, objetivo, idx)
            if df_filtrado.empty:
                continue

            # Excluir la primera columna explícitamente
            predictores = [col for col in df_filtrado.columns if col != df_filtrado.columns[0] and col != objetivo]
            if not predictores:
                # Agregar advertencia al reporte
                reporte.append({
                    "Aeronave": idx,
                    "Parámetro": objetivo,
                    "Valor imputado": "NAN",
                    "Confianza": 0,
                    "Tipo Modelo": "N/A",
                    "Predictores": "N/A",
                    "k": 0,
                    "Penalizacion_k": 0,
                    "Corr": 0,
                    "Familia": familia_usada,
                    "Método predictivo": "Correlacion",
                    "Advertencia": "No se pudo imputar por falta de parámetros válidos." + ("; modelo sin filtrado por familia" if not filtro_aplicado else ""),
                })
                continue
            modelos = []
            for combo in generar_combinaciones(predictores):
                for poly in (False, True):
                    modelos.append(entrenar_modelo(df_filtrado, objetivo, combo, poly, idx))
                # Modelos especiales solo para 1 predictor
                if len(combo) == 1:
                    modelos.append(entrenar_modelo(df_filtrado, objetivo, combo, False, idx, modelo_extra="log"))
                    modelos.append(entrenar_modelo(df_filtrado, objetivo, combo, False, idx, modelo_extra="potencia"))
                    modelos.append(entrenar_modelo(df_filtrado, objetivo, combo, False, idx, modelo_extra="exp"))

            # Guardar todos los modelos NO descartados (sin filtrar por mape/r2)
            # (Este bloque ha sido eliminado para evitar duplicados antes de LOOCV)

            # Solo pasar a LOOCV los modelos con MAPE <= 7.5% y R2 >= 0.6
            validos = [m for m in modelos if m is not None and not m["descartado"] and m["mape"] <= 7.5 and m["r2"] >= 0.6]
            descartados = [m for m in modelos if m is not None and m["descartado"]]
            if not validos:
                # Clasificar motivos de descarte para cada modelo no válido
                motivos_descartes = []
                for m in modelos:
                    if m is None:
                        continue
                    if m.get("descartado", False):
                        motivo = m.get("motivo", "Problema numérico, modelo descartado en entrenamiento")
                    elif m["mape"] > 7.5 and m["r2"] < 0.6:
                        motivo = "MAPE fuera de rango (>7.5%) y R2 fuera de rango (<0.6)"
                    elif m["mape"] > 7.5:
                        motivo = "MAPE fuera de rango (>7.5%)"
                    elif m["r2"] < 0.6:
                        motivo = "R2 fuera de rango (<0.6)"
                    else:
                        motivo = "Sin predictores válidos"
                    motivos_descartes.append((m, motivo))

                if not validos:
                    for m, motivo in motivos_descartes or [({"motivo": "Sin predictores válidos"}, "Sin predictores válidos")]:
                        reporte.append({
                            "Aeronave" : idx,
                            "Parámetro": objetivo,
                            "Valor imputado": np.nan,
                            "Confianza": 0.0,
                            "Corr": 0.0,
                            "k": 0,
                            "Tipo Modelo": m.get("tipo", "n/a"),
                            "Predictores": ",".join(m.get("predictores", [])),
                            "Penalizacion_k": 0.0,
                            "Familia": familia_usada,
                            "Método predictivo": "Correlacion",
                            "Advertencia": f"Modelo descartado: {motivo}" + ("; modelo sin filtrado por familia" if not filtro_aplicado else ""),
                        })
                    continue
            # Validar con LOOCV y calcular confianza promedio
            for m in validos:
                m.update(validar_con_loocv(df_filtrado, objetivo, m))
                m["Confianza_promedio"] = (m["Confianza"] + m["Confianza_LOOCV"]) / 2
            # Guardar todos los modelos NO descartados (sin filtrar por mape/r2), ahora con LOOCV
            for m in modelos:
                if m is not None and not m.get("descartado", False):
                    columnas_grafico = list(m["predictores"]) + [objetivo]
                    df_original_graf = df_original[columnas_grafico].dropna().to_dict(orient="list")
                    df_filtrado_graf = df_filtrado[columnas_grafico].dropna().to_dict(orient="list")
                    modelos_info.append({
                        "Aeronave": idx,
                        "Parámetro": objetivo,
                        "Familia": familia_usada,
                        "Filtro_aplicado": filtro_aplicado,
                        "predictores": list(m["predictores"]),
                        "n_predictores": len(m["predictores"]),
                        "n_muestras_entrenamiento": m["n"],
                        "tipo": m["tipo"],
                        "tipo_transformacion": m["tipo_transformacion"],
                        "coeficientes_originales": m["coeficientes_originales"],
                        "intercepto_original": m["intercepto_original"],
                        "ecuacion_string": m.get("ecuacion_string"),
                        "ecuacion_latex": m.get("ecuacion_latex"),
                        "ecuacion_normalizada": m.get("ecuacion_normalizada"),
                        "ecuacion_normalizada_latex": m.get("ecuacion_normalizada_latex"),
                        "mape": m["mape"],
                        "r2": m["r2"],
                        "corr": m["corr"],
                        "Confianza": m["Confianza"],
                        "Confianza_LOOCV": m.get("Confianza_LOOCV"),
                        "k_LOOCV": m.get("n_LOOCV"),
                        "Corr_LOOCV": m.get("Corr_LOOCV"),
                        "MAPE_LOOCV": m.get("MAPE_LOOCV"),
                        "R2_LOOCV": m.get("R2_LOOCV"),
                        "Advertencia": m.get("warning", None),
                        "datos_entrenamiento": m["datos_originales"],
                        "indices_entrenamiento": m["datos_originales"]["X_original"],
                        "df_filtrado_shape": df_filtrado.shape,
                        "df_filtrado_columns": list(df_filtrado.columns),
                        "df_original_shape": df_original.shape,
                        "df_original_columns": list(df_original.columns),
                        "df_original": df_original_graf,
                        "df_filtrado": df_filtrado_graf,
                    })
            # Filtrar modelos robustos por LOOCV
            robustos = [m for m in validos if m["MAPE_LOOCV"] <= 15 and m["R2_LOOCV"] >= 0.6]
            if robustos:
                mejor = max(robustos, key=lambda x: x["Confianza_promedio"])
                warning_text = "Modelo robusto"
            else:
                mejor = max(validos, key=lambda x: x["Confianza_promedio"])
                warning_text = "Modelo no robusto"
            if not filtro_aplicado:
                warning_text += "; modelo sin filtrado por familia"
            mejor["warning"] = warning_text
            mejor["Familia"] = familia_usada
            # Verificar si ya imputamos algo en esta fila/parámetro
            if not pd.isna(df_resultado.at[idx, objetivo]):
                print(f"⚠️ Ya se imputó {objetivo} en fila {idx}. No debería ocurrir.")
            # Imputar el valor de la celda actual
            df_resultado, imputacion = imputar_valores_celda(df_resultado, df_filtrado, objetivo, mejor, idx)
            imputacion["Familia"] = familia_usada
            # Solo agregar la advertencia si no está ya incluida
            advertencia_final = imputacion.get("Advertencia", "")
            if warning_text not in advertencia_final:
                if advertencia_final:
                    advertencia_final += "; " + warning_text
                else:
                    advertencia_final = warning_text
                imputacion["Advertencia"] = advertencia_final
            reporte.append(imputacion)


    return df_resultado, reporte, modelos_info
"""
def test_imputacion_correlacion_basica():
    df_final, reporte = imputaciones_correlacion('ADRpy/analisis/Data/Datos_aeronaves.xlsx')
    assert not df_final.isna().any().any(), "Deberia imputar todos los valores faltantes"
    print("listo")

test_imputacion_correlacion_basica()
"""