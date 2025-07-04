import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Tus datos
x = np.array([600, 700, 790, 870, 910, 960, 800]).reshape(-1, 1)
y = np.array([20, 32, 40, 52, 61, 70, 55])

# --- Ajuste directo (GeoGebra/np.polyfit) ---
coefs_direct = np.polyfit(X.flatten(), y, 2)
print('Ajuste directo:')
print(f'y = {coefs_direct[0]:.10f}*x^2 + {coefs_direct[1]:.10f}*x + {coefs_direct[2]:.10f}\n')

# --- Ajuste sklearn con normalización ---
# Normalización
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_norm = scaler_X.fit_transform(X)
y_norm = scaler_y.fit_transform(y.reshape(-1,1)).flatten()

print("X normalizados:", np.round(X_norm, 4))
print("y normalizados:", np.round(y_norm, 4))

# PolynomialFeatures grado 2
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly_norm = poly.fit_transform(X_norm)

# Ajuste en espacio normalizado
reg = LinearRegression(fit_intercept=True)
reg.fit(X_poly_norm, y_norm)

coefs_norm = reg.coef_
intercept_norm = reg.intercept_

print('Ajuste sklearn (espacio normalizado):')
print('intercepto:', intercept_norm)
print('coeficientes:', coefs_norm)

# --- "Desnormalización" de los coeficientes ---
# Para polinomios, no es trivial: la transformación inversa debe aplicarse término a término.
# Sin embargo, para evaluar la bondad de ajuste, puedes comparar predicciones desnormalizadas vs reales:

# Predicción sobre los datos originales usando el pipeline
X_poly_norm_pred = poly.transform(scaler_X.transform(X))
y_pred_norm = reg.predict(X_poly_norm_pred)
y_pred = scaler_y.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()

print('\nPredicción (pipeline sklearn, desnormalizada):')
print('Predicciones:', np.round(y_pred, 2))
print('Valores reales:', y)

# --- Para comparar: error cuadrático medio ---
rmse = np.sqrt(np.mean((y_pred - y)**2))
print('\nRMSE (pipeline sklearn vs real):', rmse)

# --- Desnormalización algebraica de la ecuación polinómica ---
mean_x = scaler_X.mean_
std_x = scaler_X.scale_
# Para x^2:
x2 = X.flatten()**2
mean_x2 = np.mean(x2)
std_x2 = np.std(x2)
mean_y = scaler_y.mean_
std_y = scaler_y.scale_

# Coeficientes normalizados
# coefs_norm[0] corresponde a x, coefs_norm[1] a x^2
# PolynomialFeatures pone primero x, luego x^2

a1_norm = coefs_norm[0]
a2_norm = coefs_norm[1]
b_norm = intercept_norm

# Desnormalización
# y = std_y * (a2_norm * (x^2 - mean_x2)/std_x2 + a1_norm * (x - mean_x)/std_x + b_norm) + mean_y
# = (a2_norm*std_y/std_x2)*x^2 + (a1_norm*std_y/std_x)*x + std_y*(b_norm - a2_norm*mean_x2/std_x2 - a1_norm*mean_x/std_x) + mean_y

a2_des = (a2_norm * std_y / std_x2).item()
a1_des = (a1_norm * std_y / std_x).item()
b_des = (std_y * (b_norm - a2_norm * mean_x2 / std_x2 - a1_norm * mean_x / std_x) + mean_y).item()

print('\nEcuación desnormalizada (algebraica):')
print(f'y = {a2_des:.10f}*x^2 + {a1_des:.10f}*x + {b_des:.10f}')

# Nota: la ecuación explícita en escala original no se puede escribir simplemente como
# y = a*x^2 + b*x + c
# cuando normalizas y luego desnormalizas, porque la relación es más compleja
