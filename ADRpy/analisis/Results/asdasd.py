import sympy as sp

# Variables simbólicas
x_star, y_star = sp.symbols('x y')

# Rango original
x_min, x_max = 0.21, 0.31
y_min, y_max = 20, 2500

# Transformación a [0,1]
x = x_min + x_star * (x_max - x_min)
y = y_min + y_star * (y_max - y_min)

# Ecuación original (pon aquí tus coeficientes)
z = (424.97446489749285
     + 1258.8660568945338*x
     + 0.056496780199067606*y
        )

# Expande y simplifica en términos de x_star, y_star
z_norm = sp.expand(z)

print(z_norm)