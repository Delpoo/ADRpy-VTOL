import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures

# Tus datos (1 predictor)

#x = np.array([600, 700, 790, 870, 910, 960, 800]).reshape(-1, 1)
#y = np.array([20, 32, 40, 52, 61, 70, 55])

x = np.array([0.25, 0.21, 0.23, 0.26, 0.24, 0.31, 0.28, 0.28]).reshape(-1, 1)
y = np.array([20, 32, 32, 40, 52, 61, 70, 55])

# Mostrar los valores de entrada para cada método
print('Valores originales:')
print('x:', x.flatten())
print('y:', y)

# MinMaxScaler (solo para impresión)
scaler_X_mm_print = MinMaxScaler()
scaler_y_mm_print = MinMaxScaler()
x_mm_print = scaler_X_mm_print.fit_transform(x)
y_mm_print = scaler_y_mm_print.fit_transform(y.reshape(-1,1)).flatten()
print('\nValores normalizados MinMaxScaler:')
print('x_mm:', np.round(x_mm_print.flatten(), 4))
print('y_mm:', np.round(y_mm_print, 4))

# StandardScaler (solo para impresión)
scaler_X_std_print = StandardScaler()
scaler_y_std_print = StandardScaler()
x_std_print = scaler_X_std_print.fit_transform(x)
y_std_print = scaler_y_std_print.fit_transform(y.reshape(-1,1)).flatten()
print('\nValores normalizados StandardScaler:')
print('x_std:', np.round(x_std_print.flatten(), 4))
print('y_std:', np.round(y_std_print, 4))

# --- LINEAL ---
print("\n==============================")
print("Regresión Lineal")
print("==============================")
# 1. Sin normalizar
model_orig = LinearRegression()
model_orig.fit(x, y)
a_orig = model_orig.coef_[0]
b_orig = model_orig.intercept_
print("\n[Sin normalizar]")
print(f"Ecuación: y = {a_orig:.6f}*x + {b_orig:.6f}")

# 2. MinMaxScaler (regresión)
scaler_X_mm = MinMaxScaler()
scaler_y_mm = MinMaxScaler()
x_mm = scaler_X_mm.fit_transform(x)
y_mm = scaler_y_mm.fit_transform(y.reshape(-1,1)).flatten()
model_mm = LinearRegression()
model_mm.fit(x_mm, y_mm)
a_mm = model_mm.coef_[0]
b_mm = model_mm.intercept_
print("\n[Normalizado MinMaxScaler]")
print(f"Ecuación (normalizada): y_norm = {a_mm:.6f}*x_norm + {b_mm:.6f}")
# Desnormalizar
x_min, x_max = scaler_X_mm.data_min_[0], scaler_X_mm.data_max_[0]
y_min, y_max = scaler_y_mm.data_min_[0], scaler_y_mm.data_max_[0]
a_mm_des = a_mm * (y_max - y_min) / (x_max - x_min)
b_mm_des = y_min + b_mm * (y_max - y_min) - a_mm_des * x_min
print(f"Ecuación (desnormalizada): y = {a_mm_des:.6f}*x + {b_mm_des:.6f}")

# 3. StandardScaler (lineal)
scaler_X_std_lin = StandardScaler()
scaler_y_std_lin = StandardScaler()
scaler_X_std_lin.fit(x)
scaler_y_std_lin.fit(y.reshape(-1,1))
mean_x = scaler_X_std_lin.mean_.item()
std_x = scaler_X_std_lin.scale_.item()
mean_y = scaler_y_std_lin.mean_.item()
std_y = scaler_y_std_lin.scale_.item()
x_std = scaler_X_std_lin.transform(x)
y_std = scaler_y_std_lin.transform(y.reshape(-1,1)).flatten()
model_std = LinearRegression()
model_std.fit(x_std, y_std)
a_std = model_std.coef_[0]
b_std = model_std.intercept_
print("\n[Normalizado StandardScaler]")
print(f"Ecuación (normalizada): y_norm = {a_std:.6f}*x_norm + {b_std:.6f}")
a_std_des = a_std * std_y / std_x
a_std_des = a_std_des.item() if hasattr(a_std_des, 'item') else a_std_des
b_std_des = mean_y + b_std * std_y - a_std_des * mean_x
b_std_des = b_std_des.item() if hasattr(b_std_des, 'item') else b_std_des
print(f"Ecuación (desnormalizada): y = {a_std_des:.6f}*x + {b_std_des:.6f}")

# --- POLINÓMICA GRADO 2 ---
print("\n==============================")
print("Regresión Polinómica (grado 2)")
print("==============================")
poly = PolynomialFeatures(degree=2, include_bias=False)
x_poly = poly.fit_transform(x)
print('\nx_poly shape:', x_poly.shape)

# 0. Ajuste directo (np.polyfit)
coefs_direct = np.polyfit(x.flatten(), y, 2)
print("\n[Ajuste directo np.polyfit]")
print(f"Ecuación: y = {coefs_direct[0]:.10f}*x^2 + {coefs_direct[1]:.10f}*x + {coefs_direct[2]:.15f}")

# 1. Sin normalizar
model_poly_orig = LinearRegression()
model_poly_orig.fit(x_poly, y)
a2_orig = model_poly_orig.coef_[1]
a1_orig = model_poly_orig.coef_[0]
b_orig = model_poly_orig.intercept_
print("\n[Sin normalizar]")
print(f"Ecuación: y = {a2_orig:.10f}*x^2 + {a1_orig:.10f}*x + {b_orig:.15f}")

# 2. MinMaxScaler (polinómica)
scaler_X_mm_poly = MinMaxScaler()
scaler_y_mm_poly = MinMaxScaler()
x_poly_mm = scaler_X_mm_poly.fit_transform(x_poly)
y_mm_poly = scaler_y_mm_poly.fit_transform(y.reshape(-1,1)).flatten()
model_poly_mm = LinearRegression()
model_poly_mm.fit(x_poly_mm, y_mm_poly)
a1_mm = model_poly_mm.coef_[0]
a2_mm = model_poly_mm.coef_[1]
b_mm = model_poly_mm.intercept_
x_min, x_max = scaler_X_mm_poly.data_min_[0], scaler_X_mm_poly.data_max_[0]
x2_min, x2_max = scaler_X_mm_poly.data_min_[1], scaler_X_mm_poly.data_max_[1]
y_min, y_max = scaler_y_mm_poly.data_min_[0], scaler_y_mm_poly.data_max_[0]
a2_mm_des = a2_mm/(x2_max-x2_min)*(y_max-y_min)
a1_mm_des = a1_mm/(x_max-x_min)*(y_max-y_min)
b_mm_des = y_min - a2_mm*x2_min/(x2_max-x2_min)*(y_max-y_min) - a1_mm*x_min/(x_max-x2_min)*(y_max-y_min) + b_mm*(y_max-y_min)
print("\n[Normalizado MinMaxScaler]")
print(f"Ecuación (normalizada): y_norm = {a2_mm:.10f}*x2_norm + {a1_mm:.10f}*x_norm + {b_mm:.15f}")
print(f"Ecuación (desnormalizada): y = {a2_mm_des:.10f}*x^2 + {a1_mm_des:.10f}*x + {b_mm_des:.15f}")

# 3. StandardScaler (polinómica)
scaler_X_std_poly = StandardScaler()
scaler_y_std_poly = StandardScaler()
scaler_X_std_poly.fit(x_poly)
scaler_y_std_poly.fit(y.reshape(-1,1))
print('\nx_poly shape:', x_poly.shape)
print('scaler_X_std_poly.mean_:', scaler_X_std_poly.mean_)
print('scaler_X_std_poly.scale_:', scaler_X_std_poly.scale_)
if scaler_X_std_poly.mean_ is None or scaler_X_std_poly.scale_ is None:
    print('ERROR: scaler_X_std_poly no fue ajustado correctamente.')
else:
    mean_x = scaler_X_std_poly.mean_[0].item()
    std_x = scaler_X_std_poly.scale_[0].item()
    mean_x2 = scaler_X_std_poly.mean_[1].item()
    std_x2 = scaler_X_std_poly.scale_[1].item()
    mean_y = scaler_y_std_poly.mean_.item()
    std_y = scaler_y_std_poly.scale_.item()
    x_poly_std = scaler_X_std_poly.transform(x_poly)
    y_std_poly = scaler_y_std_poly.transform(y.reshape(-1,1)).flatten()
    model_poly_std = LinearRegression()
    model_poly_std.fit(x_poly_std, y_std_poly)
    a1_std = model_poly_std.coef_[0]
    a2_std = model_poly_std.coef_[1]
    b_std = model_poly_std.intercept_
    b_std = model_poly_std.intercept_
    print('scaler_X_std_poly.mean_:', scaler_X_std_poly.mean_)
    print('scaler_X_std_poly.scale_:', scaler_X_std_poly.scale_)
    a2_std_des = a2_std*std_y/std_x2
    a1_std_des = a1_std*std_y/std_x
    a2_std_des = a2_std_des.item() if hasattr(a2_std_des, 'item') else a2_std_des
    a1_std_des = a1_std_des.item() if hasattr(a1_std_des, 'item') else a1_std_des
    b_std_des = std_y*(-a2_std*mean_x2/std_x2 - a1_std*mean_x/std_x + b_std) + mean_y
    b_std_des = b_std_des.item() if hasattr(b_std_des, 'item') else b_std_des
    print("\n[Normalizado StandardScaler]")
    print(f"Ecuación (normalizada): y_norm = {a2_std:.10f}*x2_norm + {a1_std:.10f}*x_norm + {b_std:.15f}")
    print(f"Ecuación (desnormalizada): y = {a2_std_des:.10f}*x^2 + {a1_std_des:.10f}*x + {b_std_des:.15f}")
