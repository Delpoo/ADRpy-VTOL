# Asistente de Diseño (ADRpy) — Módulo de verificación, similitud y tendencias

Este módulo NO modifica el dataset. Brinda:
- Verificaciones físicas básicas con umbrales (Δrel <2% OK; 2–10% ajuste sugerido; >10% no confiable).
- Ranking por similitud (vecinos) que respeta restricciones del usuario (exacto, objetivo±tol, max, min).
- Sugerencias robustas (min, max, mediana, media) con opción de excluir atípicos (IQR) y ponderar por confianza.
- Tendencias X–Y (nube + recta global y por Misión) con métricas (n, MAPE, R²) y semáforo de calidad.
- Insights por correlación (Spearman) para orientar, no para predecir.

**Datos de entrada**  
Ruta por defecto (configurable en `asistente_diseno/config.py`):  
`C:\Users\delpi\OneDrive\Tesis\ADRpy-VTOL\ADRpy\analisis\Results\Datos_imputados.xlsx`  
Se usa la **primera hoja** (sheet_name=0).

## Puesta en marcha (usando tu entorno ADRpy)
1. Activa tu entorno ADRpy (conda/venv).
   - Conda: `conda activate ADRpy`
   - venv: abre terminal con el entorno ADRpy activo
2. Instala dependencias en ese entorno:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. (Opcional) Registra/actualiza el kernel Jupyter para este Python:
   ```powershell
   python -m ipykernel install --user --name "ADRpy" --display-name "Python (ADRpy)"
   ```
4. En VS Code:
   - “Python: Select Interpreter” → elige el intérprete de tu entorno ADRpy.
   - En notebooks, selecciona el kernel **Python (ADRpy)** si lo registraste.
