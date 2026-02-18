"""
help_texts.py
=============
Textos de ayuda para el panel de configuración de correlación avanzada.
Cada entrada contiene el HTML formateado que se muestra en los paneles expandibles.
"""

# ============================================================================
# CHECKS 2D - Verificaciones de calidad para modelos 2D
# ============================================================================

CHECKS_2D = {
    "pearson": """
    <b>Correlación de Pearson entre predictores (|r| máximo permitido)</b><br><br>
    
    <b>🎯 ¿Para qué sirve?</b><br>
    Evita que uses dos predictores que miden básicamente lo mismo (ej: peso en kg y peso en lb).<br>
    Cuando dos predictores están muy correlacionados, el modelo se confunde sobre cuál es realmente importante.<br><br>
    
    <b>📊 ¿Qué es y cómo se mide?</b><br>
    La correlación de Pearson (r) mide si dos variables se mueven juntas:<br>
    • r = +1: Perfectamente correlacionadas (suben juntas)<br>
    • r = 0: No hay relación<br>
    • r = -1: Perfectamente anticorrelacionadas (cuando una sube, la otra baja)<br>
    Usamos |r| (valor absoluto) porque nos importa la fuerza, no la dirección.<br><br>
    
    <b>⚙️ ¿Cómo configuro este valor?</b><br>
    Este es el MÁXIMO |r| permitido entre tus predictores x1 y x2.<br>
    • Valor TÍPICO recomendado: 0.90 (equilibrado)<br>
    • Valor CONSERVADOR: 0.85 (más estricto, rechaza más modelos)<br>
    • Valor PERMISIVO: 0.95 (menos estricto, acepta más correlación)<br><br>
    
    <b>🔴 ¿Mayor o menor es mejor?</b><br>
    • MAYOR (0.95): Más permisivo, acepta predictores más correlacionados<br>
    • MENOR (0.85): Más estricto, exige predictores más independientes<br><br>
    
    <b>⚠️ ¿Qué pasa si lo configuro mal?</b><br>
    • Muy bajo (0.7): Rechazarás modelos válidos porque casi siempre hay algo de correlación<br>
    • Muy alto (0.99): Aceptarás modelos con predictores redundantes que dan coeficientes inestables<br><br>
    
    <b>💡 Ejemplo práctico:</b><br>
    Si tienes Envergadura y Longitud como predictores, probablemente |r|≈0.85.<br>
    Con umbral=0.90 → ✅ se acepta<br>
    Con umbral=0.80 → ❌ se rechaza (demasiado estricto para este caso real)<br><br>
    
    <b>🎓 Principio:</b> Predictores muy correlacionados → coeficientes inestables y difíciles de interpretar.
    """,
    "vif": """
    <b>VIF - Factor de Inflación de Varianza (máximo permitido)</b><br><br>
    
    <b>🎯 ¿Para qué sirve?</b><br>
    Detecta cuánto aumenta la incertidumbre de tus coeficientes cuando los predictores están correlacionados.<br>
    Es otra forma de medir el mismo problema que |r|, pero más sensible.<br><br>
    
    <b>📊 ¿Qué es y cómo se calcula?</b><br>
    VIF = 1 / (1 - r²)<br>
    Mide cuánto se "infla" la varianza (incertidumbre) de cada coeficiente por la correlación.<br><br>
    
    <b>📈 Escala de valores:</b><br>
    • VIF = 1: Perfecto, sin correlación<br>
    • VIF = 2: La varianza se duplica (r≈0.71)<br>
    • VIF = 5: La varianza es 5× mayor (r≈0.89) - empieza a ser preocupante<br>
    • VIF = 10: La varianza es 10× mayor (r≈0.95) - UMBRAL TÍPICO<br>
    • VIF = 20: La varianza es 20× mayor (r≈0.97) - problema serio<br>
    • VIF > 100: Muy mal condicionado<br><br>
    
    <b>⚙️ ¿Cómo configuro este valor?</b><br>
    Este es el MÁXIMO VIF permitido.<br>
    • Valor TÍPICO recomendado: 10 (estándar en la literatura)<br>
    • Valor CONSERVADOR: 5 (más estricto)<br>
    • Valor PERMISIVO: 15-20 (solo si tienes pocos datos y necesitas relajar)<br><br>
    
    <b>🔴 ¿Mayor o menor es mejor?</b><br>
    • MENOR (5): Más estricto, exige predictores más independientes<br>
    • MAYOR (20): Más permisivo, tolera más correlación (arriesgado)<br><br>
    
    <b>⚠️ ¿Qué pasa si lo configuro mal?</b><br>
    • Muy bajo (2-3): Rechazarás casi todo, solo aceptarás predictores casi independientes<br>
    • Muy alto (50+): Aceptarás modelos donde los coeficientes tienen intervalos de confianza enormes<br><br>
    
    <b>💡 Ejemplo práctico:</b><br>
    Si r=0.95 entre tus predictores → VIF = 1/(1-0.95²) ≈ 10.3<br>
    Con umbral=10 → ❌ se rechaza (apenas por encima)<br>
    Con umbral=15 → ✅ se acepta<br><br>
    
    <b>🎓 Principio:</b> VIF alto → coeficientes poco confiables (aunque el modelo prediga bien).
    """,
    "pc2": """
    <b>PC2 Ratio - Segundo Componente Principal (mínimo requerido)</b><br><br>
    
    <b>🎯 ¿Para qué sirve?</b><br>
    Detecta si tus dos predictores están casi "alineados" (uno crece proporcionalmente al otro).<br>
    Si PC2 es muy pequeño, significa que casi toda la variación está en una sola dirección.<br><br>
    
    <b>📊 ¿Qué es?</b><br>
    El análisis de componentes principales (PCA) descompone la variabilidad de tus datos en direcciones ortogonales:<br>
    • PC1 = dirección de máxima variabilidad<br>
    • PC2 = segunda dirección (perpendicular a PC1)<br>
    PC2_ratio = varianza(PC2) / varianza_total<br><br>
    
    <b>📈 Escala de valores:</b><br>
    • Ratio = 0.50: Ideal, ambas direcciones aportan igual (predictores independientes)<br>
    • Ratio = 0.20: Aceptable, todavía hay variación en segunda dirección<br>
    • Ratio = 0.03: UMBRAL TÍPICO - límite de seguridad<br>
    • Ratio < 0.03: Problemático, predictores casi redundantes (PC1 captura >97% varianza)<br>
    • Ratio ≈ 0.00: Muy malo, los predictores son casi linealmente dependientes<br><br>
    
    <b>⚙️ ¿Cómo configuro este valor?</b><br>
    Este es el MÍNIMO ratio requerido para aceptar el modelo.<br>
    • Valor TÍPICO recomendado: 0.03 (3%)<br>
    • Valor CONSERVADOR: 0.05 (5% - más estricto)<br>
    • Valor PERMISIVO: 0.01 (1% - arriesgado)<br><br>
    
    <b>🔴 ¿Mayor o menor es mejor?</b><br>
    • MAYOR (0.05): Más estricto, exige más independencia<br>
    • MENOR (0.01): Más permisivo, tolera predictores casi alineados<br><br>
    
    <b>⚠️ ¿Qué pasa si lo configuro mal?</b><br>
    • Muy alto (0.10+): Rechazarás modelos donde hay cierta correlación natural<br>
    • Muy bajo (0.001): Aceptarás modelos donde los predictores son casi el mismo dato<br><br>
    
    <b>💡 Ejemplo práctico:</b><br>
    Si tienes Peso y Masa (que son casi lo mismo, r≈0.99):<br>
    PC1 captura 99.5% varianza, PC2 solo 0.5% → ratio=0.005<br>
    Con umbral=0.03 → ❌ se rechaza (correctamente, son redundantes)<br><br>
    
    <b>🎓 Principio:</b> PC2 pequeño → predictores aportan información similar → modelo inestable.
    """,
    "rank": """
    <b>Rango de la Matriz (mínimo requerido)</b><br><br>
    
    <b>🎯 ¿Para qué sirve?</b><br>
    Verifica que tus dos predictores sean verdaderamente independientes.<br>
    Si el rango < 2, significa que uno es una copia exacta o transformación lineal del otro.<br><br>
    
    <b>📊 ¿Qué es?</b><br>
    El rango es el número de columnas (predictores) linealmente independientes en tu matriz de datos.<br>
    • Rango = 2: ✅ Los dos predictores son independientes (necesario para modelo 2D)<br>
    • Rango = 1: ❌ Un predictor es combinación lineal del otro (ej: x2 = 3×x1 + 5)<br>
    • Rango = 0: ❌❌ Todos los valores son iguales (sin variación)<br><br>
    
    <b>📈 Valores posibles:</b><br>
    • Rango = 2: ÚNICO VALOR ACEPTABLE para modelos 2D<br>
    • Rango < 2: Inaceptable, matriz singular (no invertible)<br><br>
    
    <b>⚙️ ¿Cómo configuro este valor?</b><br>
    Para modelos 2D, SIEMPRE debe ser 2 (no tocar este valor).<br>
    • Valor FIJO: 2 (no cambiar)<br><br>
    
    <b>🔴 ¿Mayor o menor es mejor?</b><br>
    No aplica - debe ser exactamente 2 para modelos con 2 predictores.<br><br>
    
    <b>⚠️ ¿Qué pasa si el rango < 2?</b><br>
    El modelo no se puede entrenar porque la matriz no es invertible.<br>
    Es como intentar resolver un sistema de ecuaciones donde una ecuación es copia de otra.<br><br>
    
    <b>💡 Ejemplos prácticos:</b><br>
    ❌ <b>Rango = 1:</b> x1=Velocidad(km/h), x2=Velocidad(m/s) → x2 = x1/3.6<br>
    ❌ <b>Rango = 1:</b> x1=Diámetro(m), x2=Radio(m) → x2 = x1/2<br>
    ✅ <b>Rango = 2:</b> x1=Envergadura, x2=Peso → independientes<br><br>
    
    <b>🎓 Principio:</b> Sin independencia lineal → sin solución única → modelo imposible.
    """,
    "cond": """
    <b>Número de Condición (máximo permitido)</b><br><br>
    
    <b>🎯 ¿Para qué sirve?</b><br>
    Mide qué tan "sensible" es tu modelo a pequeños errores en los datos.<br>
    Un número alto significa que un error tiny en medición puede causar un error ENORME en los coeficientes.<br><br>
    
    <b>📊 ¿Qué es y cómo se calcula?</b><br>
    Número de condición (κ) = σ_máx / σ_mín<br>
    Es el ratio entre el valor singular más grande y más pequeño de la matriz.<br><br>
    
    <b>📈 Escala de valores:</b><br>
    • κ < 10: ✅✅ Excelente - sistema muy estable<br>
    • κ ≈ 100: ✅ Bueno - errores controlados<br>
    • κ ≈ 1,000: ⚠️ Aceptable - empiezan problemas numéricos leves<br>
    • κ ≈ 10,000: ⚠️ Marginal - errores de redondeo notorios<br>
    • κ ≈ 100,000 (1e5): 🔴 UMBRAL TÍPICO - límite de seguridad<br>
    • κ > 1,000,000 (1e6): ❌ Muy mal - resultados poco confiables<br><br>
    
    <b>⚙️ ¿Cómo configuro este valor?</b><br>
    Este es el MÁXIMO número de condición permitido.<br>
    • Valor TÍPICO recomendado: 100,000 (1e5)<br>
    • Valor CONSERVADOR: 10,000 (1e4 - más estricto)<br>
    • Valor PERMISIVO: 1,000,000 (1e6 - solo si es inevitable)<br><br>
    
    <b>🔴 ¿Mayor o menor es mejor?</b><br>
    • MENOR (1e4): Más estricto, solo acepta sistemas bien condicionados<br>
    • MAYOR (1e6): Más permisivo, tolera más inestabilidad numérica (arriesgado)<br><br>
    
    <b>⚠️ ¿Qué pasa si lo configuro mal?</b><br>
    • Muy bajo (100): Rechazarás modelos que son numéricamente aceptables<br>
    • Muy alto (1e8): Aceptarás modelos donde pequeños cambios dan resultados totalmente diferentes<br><br>
    
    <b>💡 Ejemplo práctico:</b><br>
    Si κ=100,000 y trabajas con precisión de 15 dígitos (típico en computadoras):<br>
    Puedes perder hasta 5 dígitos de precisión → resultados confiables a ~10 dígitos<br><br>
    
    <b>🔗 Relación con correlación:</b><br>
    Si |r|=0.95 entre predictores → κ puede ser ~100<br>
    Si |r|=0.995 → κ puede ser ~10,000<br>
    Si |r|=0.9995 → κ puede ser ~100,000<br><br>
    
    <b>🎓 Principio:</b> Alto número de condición → pequeños errores de medición se amplifican enormemente.
    """,
    "coverage_unique_pair": """
    <b>Cobertura de Pares Únicos (mínimo requerido)</b><br><br>
    
    <b>🎯 ¿Para qué sirve?</b><br>
    Detecta si tienes muchos puntos repetidos (mismo x1, mismo x2).<br>
    Puntos repetidos no aportan información nueva - son como tener menos datos de los que piensas.<br><br>
    
    <b>📊 ¿Qué es y cómo se calcula?</b><br>
    Ratio = número_de_pares_únicos(x1,x2) / número_total_de_puntos<br><br>
    
    <b>📈 Escala de valores:</b><br>
    • Ratio = 1.00 (100%): ✅✅ Perfecto, todos los puntos son únicos<br>
    • Ratio = 0.80 (80%): ✅ Muy bueno, solo 20% repetidos<br>
    • Ratio = 0.60 (60%): ⚠️ UMBRAL TÍPICO - aceptable<br>
    • Ratio = 0.40 (40%): 🔴 Problemático, 60% repetidos<br>
    • Ratio < 0.30: ❌ Muy malo, más de 70% repetidos<br><br>
    
    <b>⚙️ ¿Cómo configuro este valor?</b><br>
    Este es el MÍNIMO ratio requerido (mínimo % de puntos únicos).<br>
    • Valor TÍPICO recomendado: 0.60 (60% únicos)<br>
    • Valor CONSERVADOR: 0.70 (70% únicos - más estricto)<br>
    • Valor PERMISIVO: 0.50 (50% únicos - solo si tienes pocos datos)<br><br>
    
    <b>🔴 ¿Mayor o menor es mejor?</b><br>
    • MAYOR (0.80): Más estricto, exige más variedad de datos<br>
    • MENOR (0.40): Más permisivo, tolera más repeticiones<br><br>
    
    <b>⚠️ ¿Qué pasa si lo configuro mal?</b><br>
    • Muy alto (0.95): Rechazarás datos legítimos que tienen algunas mediciones repetidas<br>
    • Muy bajo (0.20): Aceptarás datasets donde el 80% son copias (casi no hay información)<br><br>
    
    <b>💡 Ejemplo práctico:</b><br>
    Tienes n=100 puntos, pero solo 55 pares (x1,x2) distintos → ratio=0.55<br>
    • 45 puntos son repeticiones exactas de otros<br>
    • Tu muestra "efectiva" es realmente ~55, no 100<br>
    Con umbral=0.60 → ❌ se rechaza<br>
    Con umbral=0.50 → ✅ se acepta<br><br>
    
    <b>🔍 ¿Cuándo pasa esto?</b><br>
    • Mediciones en grilla/cuadrícula regular<br>
    • Valores redondeados (ej: altura siempre múltiplo de 0.5m)<br>
    • Datos categóricos o discretos<br>
    • Errores en entrada de datos (copiar-pegar)<br><br>
    
    <b>🎓 Principio:</b> Puntos repetidos → menos información real → sobreestimas la calidad del ajuste.
    """,
    "coverage_hull": """
    <b>Cobertura del Convex Hull (mínimo requerido)</b><br><br>
    
    <b>🎯 ¿Para qué sirve?</b><br>
    Detecta si tus puntos llenan uniformemente el espacio o están solo en esquinas/bordes.<br>
    Si están concentrados en pocas zonas, el modelo interpola mal en el resto del espacio.<br><br>
    
    <b>📊 ¿Qué es y cómo se calcula?</b><br>
    • <b>Convex hull:</b> El polígono convexo mínimo que contiene todos tus puntos<br>
    • <b>Bounding box:</b> El rectángulo mínimo que los contiene<br>
    • Ratio = área(convex_hull) / área(bounding_box)<br><br>
    
    <b>📈 Escala de valores:</b><br>
    • Ratio ≈ 1.00: ✅✅ Ideal, puntos llenan todo el rectángulo uniformemente<br>
    • Ratio ≈ 0.50: ✅ Bueno, distribución triangular o diagonal<br>
    • Ratio = 0.15: ⚠️ UMBRAL TÍPICO - mínimo aceptable<br>
    • Ratio < 0.10: 🔴 Problemático, puntos muy concentrados<br>
    • Ratio ≈ 0.00: ❌ Muy malo, todos en esquinas o una línea<br><br>
    
    <b>⚙️ ¿Cómo configuro este valor?</b><br>
    Este es el MÍNIMO ratio requerido.<br>
    • Valor TÍPICO recomendado: 0.15 (15%)<br>
    • Valor CONSERVADOR: 0.25 (25% - más estricto)<br>
    • Valor PERMISIVO: 0.10 (10% - solo si es inevitable)<br><br>
    
    <b>🔴 ¿Mayor o menor es mejor?</b><br>
    • MAYOR (0.30): Más estricto, exige mejor cobertura del espacio<br>
    • MENOR (0.08): Más permisivo, tolera puntos concentrados<br><br>
    
    <b>⚠️ ¿Qué pasa si lo configuro mal?</b><br>
    • Muy alto (0.60): Rechazarás datos legítimos con distribución natural<br>
    • Muy bajo (0.03): Aceptarás datos donde casi todo el espacio está vacío<br><br>
    
    <b>💡 Ejemplos visuales:</b><br>
    📊 <b>Ratio ≈ 1.0:</b> Puntos distribuidos uniformemente en rectángulo<br>
    🔺 <b>Ratio ≈ 0.5:</b> Puntos forman triángulo o diagonal<br>
    📍 <b>Ratio ≈ 0.15:</b> Puntos en forma de L o T<br>
    ⭐ <b>Ratio < 0.05:</b> Solo 4 puntos en esquinas, nada en medio<br><br>
    
    <b>🔍 Ejemplo numérico:</b><br>
    Puntos en (0,0), (10,0), (0,10), (10,10) - solo esquinas:<br>
    • Bounding box: 10×10 = 100<br>
    • Convex hull: también un cuadrado = 100<br>
    • Ratio = 100/100 = 1.0 ← ¡Engañoso! El interior está vacío<br>
    Pero si agregas un quinto punto en (5,5):<br>
    • El ratio se mantiene alto aunque la interpolación mejore<br><br>
    
    <b>🔧 Limitación del check:</b><br>
    Este ratio NO detecta "huecos internos". Es complementario a coverage_ellipse.<br><br>
    
    <b>🎓 Principio:</b> Ratio bajo → datos concentrados → interpolación poco confiable en zonas sin datos.
    """,
    "coverage_ellipse": """
    <b>Cobertura de la Elipse de Covarianza (mínimo requerido)</b><br><br>
    
    <b>🎯 ¿Para qué sirve?</b><br>
    Detecta si tus datos tienen outliers extremos o están muy "estirados" en una dirección.<br>
    Es un check de calidad de la distribución de tus puntos en el espacio 2D.<br><br>
    
    <b>📊 ¿Qué es y cómo se calcula?</b><br>
    • La <b>elipse 1σ</b> se construye con la matriz de covarianza de (x1, x2)<br>
    • Sus ejes vienen de los autovectores, y tamaños de los autovalores<br>
    • Contiene ~39% de los puntos si siguen distribución normal bivariada<br>
    • Ratio = área(elipse_1σ) / área(bounding_box)<br><br>
    
    <b>📈 Escala de valores:</b><br>
    • Ratio ≈ 0.40: ✅✅ Excelente, distribución bien comportada<br>
    • Ratio ≈ 0.20: ✅ Bueno, distribución razonable<br>
    • Ratio = 0.10: ⚠️ UMBRAL TÍPICO - límite aceptable<br>
    • Ratio < 0.05: 🔴 Problemático, ver causas abajo<br>
    • Ratio ≈ 0.00: ❌ Muy malo, distribución patológica<br><br>
    
    <b>⚙️ ¿Cómo configuro este valor?</b><br>
    Este es el MÍNIMO ratio requerido.<br>
    • Valor TÍPICO recomendado: 0.10 (10%)<br>
    • Valor CONSERVADOR: 0.15 (15% - más estricto)<br>
    • Valor PERMISIVO: 0.05 (5% - solo con precaución)<br><br>
    
    <b>🔴 ¿Mayor o menor es mejor?</b><br>
    • MAYOR (0.20): Más estricto, exige distribución más "normal"<br>
    • MENOR (0.03): Más permisivo, tolera distribuciones raras<br><br>
    
    <b>⚠️ ¿Qué pasa si lo configuro mal?</b><br>
    • Muy alto (0.30): Rechazarás datos reales que tienen cierta elongación natural<br>
    • Muy bajo (0.01): Aceptarás datos con outliers brutales o muy mal distribuidos<br><br>
    
    <b>🔍 ¿Qué causa un ratio bajo?</b><br>
    <b>Causa 1 - Outliers extremos:</b><br>
    Unos pocos puntos muy alejados inflan el bounding box, pero la elipse (basada en covarianza) ignora extremos.<br>
    Ejemplo: 95 puntos en [0,10]×[0,10], y 5 puntos en [100,100]<br>
    • Bbox: ~100×100 = 10,000<br>
    • Elipse: ~10×10 = 100<br>
    • Ratio: 100/10,000 = 0.01 ← ¡Alerta de outliers!<br><br>
    
    <b>Causa 2 - Distribución muy elongada:</b><br>
    Datos en forma de "puro" o "cigarro" (un autovalor >> otro).<br>
    Ejemplo: puntos en línea y=2x + ruido pequeño<br>
    • Elipse es muy delgada (λ₁≫λ₂)<br>
    • Área pequeña comparada con bbox<br><br>
    
    <b>Causa 3 - Estructura no elíptica:</b><br>
    Datos en L, T, U, o distribuciones multimodales.<br>
    La elipse gaussiana no captura bien la forma real.<br><br>
    
    <b>💡 Ejemplo práctico:</b><br>
    Dataset: 90 puntos en [0,1]×[0,1] + 10 puntos en [5,5]×[6,6]<br>
    • Bbox: 6×6 = 36<br>
    • Elipse 1σ: captura solo cluster principal ≈ 1×1 = 1<br>
    • Ratio: 1/36 ≈ 0.03<br>
    Con umbral=0.10 → ❌ se rechaza (correctamente, hay dos clusters)<br><br>
    
    <b>✅ ¿Cuándo es útil este check?</b><br>
    • Detectar outliers que otros checks no ven<br>
    • Identificar distribuciones no gaussianas<br>
    • Encontrar clusters separados o estructuras raras<br><br>
    
    <b>🎓 Principio:</b> Ratio muy bajo → distribución anómala → modelo puede no ser apropiado.
    """,
    "n_per_param_linear2": """
    <b>Puntos por Parámetro (n/p) para Modelos linear-2</b><br><br>
    
    <b>🎯 ¿Para qué sirve?</b><br>
    Evita entrenar modelos con muy pocos datos para el número de parámetros que necesitan estimar.<br>
    Un n/p bajo = modelo "memoriza" los puntos de entrenamiento en lugar de aprender el patrón real.<br><br>
    
    <b>📊 ¿Qué es n/p y cómo se calcula?</b><br>
    <b>Modelos linear-2 tienen p=3 parámetros:</b><br>
    y = b₀ + b₁·x1 + b₂·x2<br>
    • b₀ = intercepto<br>
    • b₁ = coeficiente de x1<br>
    • b₂ = coeficiente de x2<br><br>
    
    <b>n/p = número_de_puntos / 3</b><br><br>
    
    <b>📈 Escala de valores (para p=3):</b><br>
    • n=6 → n/p=2: ❌ Extremadamente bajo, modelo inútil<br>
    • n=9 → n/p=3: 🔴 Muy bajo, sobreajuste severo<br>
    • n=12 → n/p=4: ⚠️ Bajo, sobreajuste probable<br>
    • n=18 → n/p=6: ✅ UMBRAL TÍPICO - mínimo aceptable<br>
    • n=24 → n/p=8: ✅✅ Bueno, modelo confiable (valor DEFAULT actual)<br>
    • n=30 → n/p=10: ✅✅ Muy bueno, modelo confiable<br>
    • n=60 → n/p=20: ✅✅✅ Excelente<br>
    • n≥90 → n/p≥30: ✅✅✅ Óptimo<br><br>
    
    <b>⚙️ ¿Cómo configuro este valor?</b><br>
    Este es el MÍNIMO n/p requerido para aceptar el modelo.<br>
    • Valor ACTUAL por defecto: 8 (necesitas n≥24 puntos)<br>
    • Valor TÍPICO recomendado: 6 (necesitas n≥18 puntos - más permisivo)<br>
    • Valor CONSERVADOR: 10 (necesitas n≥30 puntos - más estricto)<br>
    • Valor PERMISIVO: 4 (necesitas n≥12 puntos - solo si es inevitable)<br><br>
    
    <b>🔴 ¿Mayor o menor es mejor?</b><br>
    • MAYOR (10): Más estricto, exige más datos → modelos más confiables<br>
    • MENOR (4): Más permisivo, acepta menos datos → mayor riesgo de sobreajuste<br><br>
    
    <b>⚠️ ¿Qué pasa si lo configuro mal?</b><br>
    • Muy alto (20): Rechazarás muchos modelos, especialmente si tienes dataset pequeño<br>
    • Muy bajo (2): Aceptarás modelos que "dibujan líneas" entre puntos sin captar el patrón real<br><br>
    
    <b>💡 Ejemplos prácticos:</b><br>
    <b>Caso 1:</b> Tienes n=20 puntos<br>
    • n/p = 20/3 ≈ 6.7<br>
    • Con umbral=8 → 6.7<8 → ❌ rechazado (valor default)<br>
    • Con umbral=6 → 6.7≥6 → ✅ aceptado<br><br>
    
    <b>Caso 2:</b> Tienes n=25 puntos<br>
    • n/p = 25/3 ≈ 8.3<br>
    • Con umbral=8 → 8.3≥8 → ✅ aceptado (justo pasa)<br>
    • Con umbral=10 → 8.3<10 → ❌ rechazado<br><br>
    
    <b>🔬 ¿Por qué es importante?</b><br>
    <b>Problema del sobreajuste:</b><br>
    Con solo n=6 puntos y p=3 parámetros (n/p=2):<br>
    • El modelo puede pasar "casi exactamente" por esos 6 puntos<br>
    • Parece perfecto en entrenamiento (R²≈1)<br>
    • Pero predice MUY MAL en datos nuevos<br>
    • No aprendió el patrón, solo "memorizó" posiciones<br><br>
    
    Con n=30 puntos y p=3 parámetros (n/p=10):<br>
    • El modelo debe encontrar el patrón general<br>
    • No puede "memorizar" cada punto<br>
    • Generaliza mejor a datos nuevos<br><br>
    
    <b>📐 Regla empírica general:</b><br>
    Para modelos lineales, necesitas al menos 5-10 observaciones por parámetro.<br>
    El valor default de 8 es un buen compromiso entre flexibilidad y rigor.<br><br>
    
    <b>🎓 Principio:</b> Menos datos que parámetros → modelo sobreajustado → no generaliza.
    """,
    "n_per_param_poly2": """
    <b>Puntos por Parámetro (n/p) para Modelos poly-2</b><br><br>
    
    <b>🎯 ¿Para qué sirve?</b><br>
    Igual que linear-2, pero MÁS ESTRICTO porque poly-2 tiene más parámetros y más flexibilidad.<br>
    Mayor flexibilidad = mayor riesgo de sobreajuste = necesitas MÁS datos para entrenar confiablemente.<br><br>
    
    <b>📊 ¿Qué es n/p y cómo se calcula?</b><br>
    <b>Modelos poly-2 tienen p=6 parámetros (el DOBLE que linear-2):</b><br>
    y = b₀ + b₁·x1 + b₂·x2 + b₃·x1² + b₄·x2² + b₅·x1·x2<br>
    • b₀ = intercepto<br>
    • b₁, b₂ = términos lineales<br>
    • b₃, b₄ = términos cuadráticos (capturan curvatura)<br>
    • b₅ = término de interacción (captura efectos combinados)<br><br>
    
    <b>n/p = número_de_puntos / 6</b><br><br>
    
    <b>📈 Escala de valores (para p=6):</b><br>
    • n=12 → n/p=2: ❌ Extremadamente bajo, inútil<br>
    • n=18 → n/p=3: 🔴 Muy bajo, sobreajuste brutal<br>
    • n=30 → n/p=5: ⚠️ Bajo, sobreajuste muy probable<br>
    • n=42 → n/p=7: ⚠️ Marginal, arriesgado<br>
    • n=60 → n/p=10: ✅ UMBRAL TÍPICO - mínimo aceptable (valor DEFAULT actual)<br>
    • n=90 → n/p=15: ✅✅ Bueno, modelo confiable<br>
    • n=120 → n/p=20: ✅✅✅ Muy bueno, alta confiabilidad<br>
    • n≥180 → n/p≥30: ✅✅✅ Excelente<br><br>
    
    <b>⚙️ ¿Cómo configuro este valor?</b><br>
    Este es el MÍNIMO n/p requerido para aceptar el modelo.<br>
    • Valor ACTUAL por defecto: 10 (necesitas n≥60 puntos)<br>
    • Valor CONSERVADOR: 15 (necesitas n≥90 puntos - muy estricto, recomendado)<br>
    • Valor PERMISIVO: 7 (necesitas n≥42 puntos - solo si dataset pequeño y no hay alternativa)<br><br>
    
    <b>🔴 ¿Mayor o menor es mejor?</b><br>
    • MAYOR (15): Más estricto, exige más datos → modelos mucho más confiables<br>
    • MENOR (7): Más permisivo, acepta menos datos → alto riesgo de sobreajuste<br><br>
    
    <b>⚠️ ¿Qué pasa si lo configuro mal?</b><br>
    • Muy alto (25): Rechazarás casi todos los modelos poly-2 (necesitas n≥150)<br>
    • Muy bajo (5): Aceptarás modelos que se ajustan al RUIDO, no a la señal real<br><br>
    
    <b>💡 Ejemplos prácticos:</b><br>
    <b>Caso 1:</b> Tienes n=50 puntos<br>
    • n/p = 50/6 ≈ 8.3<br>
    • Con umbral=10 → 8.3<10 → ❌ rechazado (correcto, muy pocos datos)<br>
    • Con umbral=7 → 8.3≥7 → ✅ aceptado (arriesgado)<br><br>
    
    <b>Caso 2:</b> Tienes n=100 puntos<br>
    • n/p = 100/6 ≈ 16.7<br>
    • Con umbral=10 → 16.7≥10 → ✅ aceptado (bien)<br>
    • Con umbral=15 → 16.7≥15 → ✅ aceptado (muy bien)<br><br>
    
    <b>🔬 ¿Por qué más estricto que linear-2?</b><br>
    <b>Mayor flexibilidad = mayor capacidad de sobreajuste:</b><br>
    • Términos cuadráticos (x1², x2²) pueden crear "jorobas" artificiales<br>
    • Término de interacción (x1·x2) puede capturar correlaciones espurias<br>
    • Con 6 grados de libertad, puedes ajustar MUY bien solo con ruido<br><br>
    
    <b>Ejemplo visual:</b><br>
    Con n=20 puntos y poly-2 (p=6, n/p≈3):<br>
    • El modelo crea curvas que pasan casi exactamente por ruido<br>
    • R² en entrenamiento ≈ 0.98 (parece perfecto)<br>
    • R² en validación ≈ 0.3 (desastre)<br>
    • No generaliza NADA<br><br>
    
    Con n=100 puntos y poly-2 (p=6, n/p≈17):<br>
    • El modelo encuentra curvatura real en los datos<br>
    • R² en entrenamiento ≈ 0.85<br>
    • R² en validación ≈ 0.82<br>
    • Generaliza bien<br><br>
    
    <b>📐 Comparación linear-2 vs poly-2:</b><br>
    • linear-2: n/p=8 típico → necesitas n≥24<br>
    • poly-2: n/p=10 típico → necesitas n≥60<br>
    → Poly-2 necesita 2.5× MÁS datos para la misma confianza<br><br>
    
    <b>🎓 Principio:</b> Mayor complejidad del modelo → necesitas exponencialmente más datos para evitar sobreajuste.
    """,
    "agresivo": """
    <b>Modo Agresivo - Correlación Mínima para Activar</b><br><br>
    
    <b>🎯 ¿Para qué sirve?</b><br>
    Permite aceptar modelos que fallan algunos checks numéricos SI y SOLO SI hay evidencia<br>
    de una correlación MUY FUERTE entre la variable objetivo (y) y al menos uno de los predictores.<br><br>
    
    <b>📊 ¿Qué es y cómo funciona?</b><br>
    El sistema calcula |r| entre y y cada predictor (x1, x2).<br>
    Si max(|r(y,x1)|, |r(y,x2)|) ≥ umbral_modo_agresivo:<br>
    → Se RELAJAN algunos checks de calidad numérica<br>
    → Se aceptan modelos que normalmente serían rechazados<br><br>
    
    <b>🤔 Justificación del "modo agresivo":</b><br>
    Si y tiene |r|=0.98 con x1:<br>
    • La relación es TAN fuerte que x1 prácticamente "predice" y<br>
    • Incluso con problemas numéricos (VIF alto, etc.), el modelo funcionará razonablemente<br>
    • Vale la pena asumir cierto riesgo numérico por capturar esa relación tan potente<br><br>
    
    <b>⚙️ ¿Cómo configuro este valor?</b><br>
    Este es el MÍNIMO |r(y, x_i)| necesario para activar el modo agresivo.<br>
    • Valor TÍPICO recomendado: 0.95 (muy selectivo, solo correlaciones excepcionales)<br>
    • Valor CONSERVADOR: 0.97 (extremadamente selectivo)<br>
    • Valor PERMISIVO: 0.90 (menos selectivo, activa más seguido - NO recomendado)<br><br>
    
    <b>🔴 ¿Mayor o menor es mejor?</b><br>
    • MAYOR (0.97): Más estricto, solo relaja con correlaciones excepcionalmente fuertes<br>
    • MENOR (0.90): Más permisivo, relaja con correlaciones moderadamente fuertes (arriesgado)<br><br>
    
    <b>⚠️ Checks que SE RELAJAN cuando se activa:</b><br>
    • <b>VIF:</b> Se toleran valores más altos (multicolinealidad)<br>
    • <b>Número de condición:</b> Se tolera peor condicionamiento numérico<br>
    • <b>PC2 ratio:</b> Se tolera menor varianza en segundo componente<br><br>
    
    <b>🛡️ Checks que NO se relajan (siempre se verifican):</b><br>
    • Rango de la matriz (rank): Debe ser 2 siempre<br>
    • n/p (puntos por parámetro): No se relaja<br>
    • Cobertura (hull, ellipse, unique pairs): No se relaja<br><br>
    
    <b>💡 Ejemplos prácticos:</b><br>
    <b>Caso 1 - Se activa modo agresivo:</b><br>
    • |r(y, x1)| = 0.96 (muy alta correlación)<br>
    • |r(y, x2)| = 0.45<br>
    • |r(x1, x2)| = 0.92 (predictores correlacionados, VIF alto)<br>
    • Con umbral=0.95 → max(0.96, 0.45)=0.96 ≥ 0.95 → ✅ modo agresivo ON<br>
    • El modelo se acepta AUNQUE VIF sea alto (x1 es tan buen predictor que justifica el riesgo)<br><br>
    
    <b>Caso 2 - NO se activa:</b><br>
    • |r(y, x1)| = 0.85<br>
    • |r(y, x2)| = 0.75<br>
    • Con umbral=0.95 → max(0.85, 0.75)=0.85 < 0.95 → ❌ modo agresivo OFF<br>
    • El modelo debe pasar TODOS los checks estrictos<br><br>
    
    <b>⚠️ PELIGROS del modo agresivo:</b><br>
    <b>1. Correlación espuria:</b> Alta correlación en muestra NO garantiza relación causal.<br>
    Ejemplo: Ventas de helado vs ahogamientos (ambos suben en verano, r≈0.95, pero sin relación real)<br><br>
    
    <b>2. Overfitting:</b> Alta correlación en datos de entrenamiento puede ser coincidencia.<br>
    Si solo tienes n=30 puntos, incluso r=0.95 podría ser casualidad.<br><br>
    
    <b>3. Extrapolación peligrosa:</b> Modelos numéricamente inestables predicen MAL fuera del rango de entrenamiento,<br>
    aunque tengan buena correlación en el rango entrenado.<br><br>
    
    <b>✅ Cuándo SÍ usar modo agresivo (relajar a 0.90-0.93):</b><br>
    • Conoces la física detrás de la relación (ej: ley de Newton, termodinámica)<br>
    • Tienes dataset grande (n>100) que valida la correlación<br>
    • Solo necesitas interpolar (no extrapolar)<br>
    • La correlación es consistente en validación cruzada<br><br>
    
    <b>❌ Cuándo NO usar (mantener 0.95+):</b><br>
    • Dataset pequeño (n<50)<br>
    • No entiendes por qué la correlación existe<br>
    • Necesitas extrapolar mucho fuera del rango de entrenamiento<br>
    • La correlación varía mucho entre train/val/test<br><br>
    
    <b>🎓 Principio:</b> Correlación muy alta justifica asumir riesgo numérico, pero SOLO si la relación es real y robusta.
    """,
}


# ============================================================================
# DIVERSIDAD - Requisitos mínimos por tipo de modelo
# ============================================================================

DIVERSIDAD = {
    "exp-1": """
    <div style="background:#FFF8E1;border-left:4px solid #F57C00;padding:12px;margin-bottom:8px;">
    <b style="color:#E65100;font-size:14px;">📊 Modelo exponencial 1D — y = β₀·e^(β₁·x)</b>
    </div>
    
    <table style="width:100%;border-collapse:collapse;margin:10px 0;font-size:13px;">
    <tr style="background:#E3F2FD;">
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Parámetro</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Valor</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Significado</th>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>Coeficientes (p)</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><code>2</code> (β₀, β₁)</td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Cantidad de parámetros a estimar</td>
    </tr>
    <tr style="background:#F5F5F5;">
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>Mín. únicos</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="color:#1976D2;">Configurable</span></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Valores distintos mínimos en y y en x</td>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>Muestras/coef.</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="color:#1976D2;">Configurable</span></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Cuántas observaciones por parámetro</td>
    </tr>
    <tr style="background:#F5F5F5;">
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>n mínimo</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><code>ceil(2 × muestras/coef.)</code></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Tamaño muestral mínimo requerido</td>
    </tr>
    </table>
    
    <div style="background:#E8F5E9;padding:10px;border-radius:4px;margin:10px 0;">
    <b style="color:#2E7D32;">✅ Criterios de aceptación (TODOS deben cumplirse):</b><br>
    1️⃣ <b>Diversidad:</b> valores únicos en y ≥ mín. únicos<br>
    2️⃣ <b>Diversidad:</b> valores únicos en x ≥ mín. únicos<br>
    3️⃣ <b>Tamaño muestral:</b> n (datos disponibles) ≥ n mínimo
    </div>
    
    <b>🔢 ¿Cómo se calcula n mínimo?</b><br>
    <code>n_min = ceil(p × muestras/coef.) = ceil(2 × muestras/coef.)</code><br><br>
    
    <b>💡 Ejemplos de configuración:</b><br>
    • <b>Conservador:</b> mín. únicos=5, muestras/coef.=5 → n_min=10<br>
    • <b>Equilibrado:</b> mín. únicos=4, muestras/coef.=3 → n_min=6<br>
    • <b>Permisivo:</b> mín. únicos=3, muestras/coef.=2 → n_min=4<br><br>
    
    <b>⚙️ ¿Qué debo poner?</b><br>
    <b>Mín. únicos:</b> Típicamente 4-5 (menos de 3 es arriesgado)<br>
    <b>Muestras/coef.:</b> Típicamente 3-5 (menos de 2 puede sobreajustar)<br><br>
    
    <b>⚠️ Cuidado con valores bajos:</b><br>
    • mín. únicos < 3: Modelo puede ajustar ruido en vez de tendencias reales<br>
    • muestras/coef. < 2: Alto riesgo de sobreajuste y coeficientes inestables
    """,
    "log-1": """
    <div style="background:#FFF8E1;border-left:4px solid #F57C00;padding:12px;margin-bottom:8px;">
    <b style="color:#E65100;font-size:14px;">📊 Modelo logarítmico 1D — y = β₀ + β₁·ln(x)</b>
    </div>
    
    <table style="width:100%;border-collapse:collapse;margin:10px 0;font-size:13px;">
    <tr style="background:#E3F2FD;">
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Parámetro</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Valor</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Significado</th>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>Coeficientes (p)</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><code>2</code> (β₀, β₁)</td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Cantidad de parámetros a estimar</td>
    </tr>
    <tr style="background:#F5F5F5;">
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>Mín. únicos</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="color:#1976D2;">Configurable</span></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Valores distintos mínimos en y y en x</td>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>Muestras/coef.</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="color:#1976D2;">Configurable</span></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Cuántas observaciones por parámetro</td>
    </tr>
    <tr style="background:#F5F5F5;">
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>n mínimo</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><code>ceil(2 × muestras/coef.)</code></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Tamaño muestral mínimo requerido</td>
    </tr>
    </table>
    
    <div style="background:#E8F5E9;padding:10px;border-radius:4px;margin:10px 0;">
    <b style="color:#2E7D32;">✅ Criterios de aceptación (TODOS deben cumplirse):</b><br>
    1️⃣ <b>Diversidad:</b> valores únicos en y ≥ mín. únicos<br>
    2️⃣ <b>Diversidad:</b> valores únicos en x ≥ mín. únicos<br>
    3️⃣ <b>Tamaño muestral:</b> n (datos disponibles) ≥ n mínimo
    </div>
    
    <b>🔢 ¿Cómo se calcula n mínimo?</b><br>
    <code>n_min = ceil(2 × muestras/coef.)</code><br><br>
    
    <b>💡 Guía rápida:</b> Similar al exponencial, usa valores típicos de 4-5 únicos y 3-5 muestras/coef.
    """,
    "pot-1": """
    <div style="background:#FFF8E1;border-left:4px solid #F57C00;padding:12px;margin-bottom:8px;">
    <b style="color:#E65100;font-size:14px;">📊 Modelo potencia 1D — y = β₀·x^β₁</b>
    </div>
    
    <table style="width:100%;border-collapse:collapse;margin:10px 0;font-size:13px;">
    <tr style="background:#E3F2FD;">
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Parámetro</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Valor</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Significado</th>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>Coeficientes (p)</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><code>2</code> (β₀, β₁)</td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Cantidad de parámetros a estimar</td>
    </tr>
    <tr style="background:#F5F5F5;">
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>Mín. únicos</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="color:#1976D2;">Configurable</span></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Valores distintos mínimos en y y en x</td>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>Muestras/coef.</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="color:#1976D2;">Configurable</span></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Cuántas observaciones por parámetro</td>
    </tr>
    <tr style="background:#F5F5F5;">
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>n mínimo</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><code>ceil(2 × muestras/coef.)</code></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Tamaño muestral mínimo requerido</td>
    </tr>
    </table>
    
    <div style="background:#E8F5E9;padding:10px;border-radius:4px;margin:10px 0;">
    <b style="color:#2E7D32;">✅ Criterios de aceptación (TODOS deben cumplirse):</b><br>
    1️⃣ <b>Diversidad:</b> valores únicos en y ≥ mín. únicos<br>
    2️⃣ <b>Diversidad:</b> valores únicos en x ≥ mín. únicos<br>
    3️⃣ <b>Tamaño muestral:</b> n (datos disponibles) ≥ n mínimo
    </div>
    
    <b>🔢 ¿Cómo se calcula n mínimo?</b><br>
    <code>n_min = ceil(2 × muestras/coef.)</code><br><br>
    
    <b>💡 Guía rápida:</b> Similar a los otros modelos 1D de 2 parámetros, usa 4-5 únicos y 3-5 muestras/coef.
    """,
    "linear-1": """
    <div style="background:#FFF8E1;border-left:4px solid #F57C00;padding:12px;margin-bottom:8px;">
    <b style="color:#E65100;font-size:14px;">📊 Modelo lineal 1D — y = β₀ + β₁·x</b>
    </div>
    
    <table style="width:100%;border-collapse:collapse;margin:10px 0;font-size:13px;">
    <tr style="background:#E3F2FD;">
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Parámetro</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Valor</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Significado</th>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>Coeficientes (p)</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><code>2</code> (β₀, β₁)</td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Cantidad de parámetros a estimar</td>
    </tr>
    <tr style="background:#F5F5F5;">
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>Mín. únicos</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="color:#1976D2;">Configurable</span></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Valores distintos mínimos en y y en x</td>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>Muestras/coef.</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="color:#1976D2;">Configurable</span></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Cuántas observaciones por parámetro</td>
    </tr>
    <tr style="background:#F5F5F5;">
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>n mínimo</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><code>ceil(2 × muestras/coef.)</code></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Tamaño muestral mínimo requerido</td>
    </tr>
    </table>
    
    <div style="background:#E8F5E9;padding:10px;border-radius:4px;margin:10px 0;">
    <b style="color:#2E7D32;">✅ Criterios de aceptación (TODOS deben cumplirse):</b><br>
    1️⃣ <b>Diversidad:</b> valores únicos en y ≥ mín. únicos<br>
    2️⃣ <b>Diversidad:</b> valores únicos en x ≥ mín. únicos<br>
    3️⃣ <b>Tamaño muestral:</b> n (datos disponibles) ≥ n mínimo
    </div>
    
    <b>🔢 ¿Cómo se calcula n mínimo?</b><br>
    <code>n_min = ceil(2 × muestras/coef.)</code><br><br>
    
    <b>💡 Guía rápida:</b> El modelo más simple y robusto. Valores típicos: 4-5 únicos, 3-5 muestras/coef.
    """,
    "poly-1": """
    <div style="background:#FFF8E1;border-left:4px solid #F57C00;padding:12px;margin-bottom:8px;">
    <b style="color:#E65100;font-size:14px;">📊 Modelo polinómico 1D (grado 2) — y = β₀ + β₁·x + β₂·x²</b>
    </div>
    
    <table style="width:100%;border-collapse:collapse;margin:10px 0;font-size:13px;">
    <tr style="background:#E3F2FD;">
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Parámetro</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Valor</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Significado</th>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>Coeficientes (p)</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><code>3</code> (β₀, β₁, β₂)</td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Cantidad de parámetros a estimar</td>
    </tr>
    <tr style="background:#F5F5F5;">
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>Mín. únicos</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="color:#1976D2;">Configurable</span></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Valores distintos mínimos en y y en x</td>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>Muestras/coef.</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="color:#1976D2;">Configurable</span></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Cuántas observaciones por parámetro</td>
    </tr>
    <tr style="background:#F5F5F5;">
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>n mínimo</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><code>ceil(3 × muestras/coef.)</code></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Tamaño muestral mínimo requerido</td>
    </tr>
    </table>
    
    <div style="background:#E8F5E9;padding:10px;border-radius:4px;margin:10px 0;">
    <b style="color:#2E7D32;">✅ Criterios de aceptación (TODOS deben cumplirse):</b><br>
    1️⃣ <b>Diversidad:</b> valores únicos en y ≥ mín. únicos<br>
    2️⃣ <b>Diversidad:</b> valores únicos en x ≥ mín. únicos<br>
    3️⃣ <b>Tamaño muestral:</b> n (datos disponibles) ≥ n mínimo
    </div>
    
    <b>🔢 ¿Cómo se calcula n mínimo?</b><br>
    <code>n_min = ceil(3 × muestras/coef.)</code><br><br>
    
    <b>💡 Ejemplos de configuración:</b><br>
    • <b>Conservador:</b> mín. únicos=5, muestras/coef.=5 → n_min=15<br>
    • <b>Equilibrado:</b> mín. únicos=5, muestras/coef.=3 → n_min=9<br>
    • <b>Permisivo:</b> mín. únicos=4, muestras/coef.=2 → n_min=6<br><br>
    
    <b>⚠️ IMPORTANTE:</b> Modelos cuadráticos necesitan MÁS datos que lineales.<br>
    Recomendado: mín. únicos ≥ 5 y muestras/coef. ≥ 3 para evitar sobreajuste.
    """,
    "linear-2": """
    <div style="background:#FFF8E1;border-left:4px solid #F57C00;padding:12px;margin-bottom:8px;">
    <b style="color:#E65100;font-size:14px;">📊 Modelo lineal 2D — y = β₀ + β₁·x₁ + β₂·x₂</b>
    </div>
    
    <table style="width:100%;border-collapse:collapse;margin:10px 0;font-size:13px;">
    <tr style="background:#E3F2FD;">
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Parámetro</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Valor</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Significado</th>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>Coeficientes (p)</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><code>3</code> (β₀, β₁, β₂)</td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Cantidad de parámetros a estimar</td>
    </tr>
    <tr style="background:#F5F5F5;">
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>Mín. únicos</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="color:#1976D2;">Configurable</span></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Valores distintos mínimos en y, x₁ y x₂</td>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>Muestras/coef.</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="color:#1976D2;">Configurable</span></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Cuántas observaciones por parámetro</td>
    </tr>
    <tr style="background:#F5F5F5;">
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>n mínimo</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><code>ceil(3 × muestras/coef.)</code></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Tamaño muestral mínimo requerido</td>
    </tr>
    </table>
    
    <div style="background:#E8F5E9;padding:10px;border-radius:4px;margin:10px 0;">
    <b style="color:#2E7D32;">✅ Criterios de aceptación (TODOS deben cumplirse):</b><br>
    1️⃣ <b>Diversidad:</b> valores únicos en y ≥ mín. únicos<br>
    2️⃣ <b>Diversidad:</b> valores únicos en x₁ ≥ mín. únicos<br>
    3️⃣ <b>Diversidad:</b> valores únicos en x₂ ≥ mín. únicos<br>
    4️⃣ <b>Tamaño muestral:</b> n (datos disponibles) ≥ n mínimo
    </div>
    
    <b>🔢 ¿Cómo se calcula n mínimo?</b><br>
    <code>n_min = ceil(3 × muestras/coef.)</code><br><br>
    
    <b>💡 Guía rápida:</b> Modelos 2D necesitan más variedad. Usa 5 únicos mínimo y 3-5 muestras/coef.
    """,
    "poly-2": """
    <div style="background:#FFF8E1;border-left:4px solid #F57C00;padding:12px;margin-bottom:8px;">
    <b style="color:#E65100;font-size:14px;">📊 Modelo polinómico 2D — y = β₀ + β₁·x₁ + β₂·x₂ + β₃·x₁² + β₄·x₂² + β₅·x₁·x₂</b>
    </div>
    
    <table style="width:100%;border-collapse:collapse;margin:10px 0;font-size:13px;">
    <tr style="background:#E3F2FD;">
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Parámetro</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Valor</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Significado</th>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>Coeficientes (p)</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><code>6</code> (β₀ a β₅)</td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Cantidad de parámetros a estimar</td>
    </tr>
    <tr style="background:#F5F5F5;">
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>Mín. únicos</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="color:#1976D2;">Configurable</span></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Valores distintos mínimos en y, x₁ y x₂</td>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>Muestras/coef.</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="color:#1976D2;">Configurable</span></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Cuántas observaciones por parámetro</td>
    </tr>
    <tr style="background:#F5F5F5;">
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>n mínimo</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><code>ceil(6 × muestras/coef.)</code></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Tamaño muestral mínimo requerido</td>
    </tr>
    </table>
    
    <div style="background:#E8F5E9;padding:10px;border-radius:4px;margin:10px 0;">
    <b style="color:#2E7D32;">✅ Criterios de aceptación (TODOS deben cumplirse):</b><br>
    1️⃣ <b>Diversidad:</b> valores únicos en y ≥ mín. únicos<br>
    2️⃣ <b>Diversidad:</b> valores únicos en x₁ ≥ mín. únicos<br>
    3️⃣ <b>Diversidad:</b> valores únicos en x₂ ≥ mín. únicos<br>
    4️⃣ <b>Tamaño muestral:</b> n (datos disponibles) ≥ n mínimo
    </div>
    
    <b>🔢 ¿Cómo se calcula n mínimo?</b><br>
    <code>n_min = ceil(6 × muestras/coef.)</code><br><br>
    
    <b>💡 Ejemplos de configuración:</b><br>
    • <b>Conservador:</b> mín. únicos=6, muestras/coef.=5 → n_min=30<br>
    • <b>Equilibrado:</b> mín. únicos=5, muestras/coef.=3 → n_min=18<br>
    • <b>Permisivo:</b> mín. únicos=5, muestras/coef.=2 → n_min=12<br><br>
    
    <b>⚠️ CRÍTICO:</b> Este es el modelo MÁS COMPLEJO y necesita MUCHOS datos.<br>
    • <b>Mínimo absoluto recomendado:</b> mín. únicos ≥ 5, muestras/coef. ≥ 3<br>
    • <b>Ideal:</b> mín. únicos ≥ 6, muestras/coef. ≥ 5<br>
    • <b>Con menos datos:</b> Alto riesgo de sobreajuste y predicciones inestables<br><br>
    
    <b>🎯 Regla de oro:</b> Si no tienes al menos n=18-20 puntos con buena diversidad,<br>
    considera usar un modelo más simple (linear-2 o poly-1).
    """,
}


# ============================================================================
# EXTRAPOLACIÓN - Control de predicciones fuera del rango de entrenamiento
# ============================================================================

EXTRAPOLACION = {
    "modo_predictores": """
    <div style="background:#FFF3E0;border-left:4px solid #F57C00;padding:12px;margin-bottom:8px;">
    <b style="color:#E65100;font-size:14px;">🚦 Modo de manejo de extrapolación en predictores</b>
    </div>
    
    <b>🎯 ¿Para qué sirve?</b><br>
    Decide si el modelo puede predecir cuando los valores de entrada (x) están FUERA del rango que vio durante el entrenamiento.<br>
    Es como preguntarle al modelo algo que nunca estudió - puede responder, pero ¿qué tan confiable será?<br><br>
    
    <table style="width:100%;border-collapse:collapse;margin:10px 0;font-size:13px;">
    <tr style="background:#E3F2FD;">
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Modo</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Comportamiento</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Nivel de riesgo</th>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>eliminar</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">❌ Rechaza predicciones fuera del rango de entrenamiento</td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="background:#4CAF50;color:white;padding:2px 8px;border-radius:4px;">SEGURO</span></td>
    </tr>
    <tr style="background:#F5F5F5;">
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>permitir_con_tolerancia</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">✅ Acepta extrapolación controlada por tolerancia_pct</td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="background:#FF9800;color:white;padding:2px 8px;border-radius:4px;">MODERADO</span></td>
    </tr>
    </table>
    
    <div style="background:#E8F5E9;padding:10px;border-radius:4px;margin:10px 0;">
    <b style="color:#2E7D32;">📊 Modo: eliminar (RECOMENDADO por defecto)</b><br>
    • Solo predice dentro del rango conocido<br>
    • Máxima seguridad: no asume nada fuera de lo observado<br>
    • Ideal cuando no conoces el comportamiento fuera del rango<br>
    • <b>Usa este si tienes dudas</b>
    </div>
    
    <div style="background:#FFF3E0;padding:10px;border-radius:4px;margin:10px 0;">
    <b style="color:#E65100;">⚠️ Modo: permitir_con_tolerancia</b><br>
    • Acepta pequeñas extrapolaciones (controladas por tolerancia_pct)<br>
    • Útil cuando esperas variaciones naturales cerca del borde<br>
    • <b>ASUME</b> que la relación se mantiene fuera del rango<br>
    • Requiere entender el fenómeno físico/matemático
    </div>
    
    <b>💡 Ejemplo práctico:</b><br>
    <b>Entrenamiento:</b> Velocidad ∈ [50, 200] km/h → Consumo<br><br>
    
    <b>Predicción 1:</b> Velocidad = 180 km/h<br>
    • <code>eliminar</code>: ✅ Acepta (dentro de [50,200])<br>
    • <code>permitir_con_tolerancia</code>: ✅ Acepta<br><br>
    
    <b>Predicción 2:</b> Velocidad = 220 km/h<br>
    • <code>eliminar</code>: ❌ Rechaza (fuera de [50,200])<br>
    • <code>permitir_con_tolerancia</code> (tol=10%): ✅ Acepta si 220 ≤ 200 + 0.1×(200-50) = 215... ❌ Rechaza (220>215)<br>
    • <code>permitir_con_tolerancia</code> (tol=15%): ✅ Acepta si 220 ≤ 200 + 0.15×150 = 222.5... ✅ Acepta<br><br>
    
    <b>⚠️ Peligros de la extrapolación:</b><br>
    1. <b>Relaciones no lineales:</b> y=x² funciona en [0,10], pero en [100,200] el comportamiento puede cambiar<br>
    2. <b>Límites físicos:</b> Consumo de combustible puede ser lineal hasta 200 km/h, pero después hay efectos aerodinámicos nuevos<br>
    3. <b>Datos faltantes sistemáticos:</b> Si no hay datos arriba de 200 km/h, puede haber una razón (límite del vehículo, seguridad, etc.)<br><br>
    
    <b>✅ Cuándo SÍ permitir extrapolación:</b><br>
    • Conoces la física/matemática del fenómeno<br>
    • La relación es lineal o conocida<br>
    • Solo necesitas extrapolar un poco (5-10%)<br>
    • Has validado el modelo en ese rango extendido<br><br>
    
    <b>❌ Cuándo NO permitir (usar eliminar):</b><br>
    • No entiendes por qué existe la relación<br>
    • Sospechas cambios de comportamiento fuera del rango<br>
    • Modelos polinómicos o no lineales (muy peligrosos para extrapolar)<br>
    • Datos críticos para seguridad o diseño<br><br>
    
    <b>🎓 Principio:</b> La extrapolación es una APUESTA de que el patrón continúa. Úsala solo si puedes justificarla.
    """,
    "tolerancia_pct": """
    <div style="background:#FFF3E0;border-left:4px solid #F57C00;padding:12px;margin-bottom:8px;">
    <b style="color:#E65100;font-size:14px;">📏 Tolerancia de extrapolación (%)</b>
    </div>
    
    <b>🎯 ¿Para qué sirve?</b><br>
    Define QUÉ TAN LEJOS del rango de entrenamiento puedes predecir.<br>
    Es el "margen de seguridad" que le das al modelo para explorar más allá de lo conocido.<br><br>
    
    <b>📊 ¿Cómo se calcula el rango ampliado?</b><br>
    <code>rango_entrenamiento = [x_min, x_max]</code><br>
    <code>amplitud = x_max - x_min</code><br>
    <code>margen = tolerancia_pct × amplitud</code><br>
    <code>rango_ampliado = [x_min - margen, x_max + margen]</code><br><br>
    
    <table style="width:100%;border-collapse:collapse;margin:10px 0;font-size:13px;">
    <tr style="background:#E3F2FD;">
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Tolerancia</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Descripción</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Riesgo</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Uso recomendado</th>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>0%</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Sin extrapolación</td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="background:#4CAF50;color:white;padding:2px 6px;border-radius:3px;">Mínimo</span></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Cuando usas modo "eliminar"</td>
    </tr>
    <tr style="background:#F5F5F5;">
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>3-5%</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Extrapolación mínima</td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="background:#8BC34A;color:white;padding:2px 6px;border-radius:3px;">Bajo</span></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Errores de medición, redondeos</td>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>10%</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Extrapolación moderada (TÍPICO)</td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="background:#FF9800;color:white;padding:2px 6px;border-radius:3px;">Moderado</span></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Relaciones bien entendidas</td>
    </tr>
    <tr style="background:#F5F5F5;">
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>20%</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Extrapolación agresiva</td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="background:#FF5722;color:white;padding:2px 6px;border-radius:3px;">Alto</span></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Solo con justificación física</td>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>>30%</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Extrapolación extrema</td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="background:#D32F2F;color:white;padding:2px 6px;border-radius:3px;">Muy alto</span></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">NO recomendado</td>
    </tr>
    </table>
    
    <b>💡 Ejemplos numéricos:</b><br><br>
    
    <b>Caso 1:</b> Rango = [10, 100], Tolerancia = 0%<br>
    • Amplitud = 100 - 10 = 90<br>
    • Margen = 0% × 90 = 0<br>
    • <b>Rango ampliado = [10, 100]</b> (sin cambio)<br><br>
    
    <b>Caso 2:</b> Rango = [10, 100], Tolerancia = 5%<br>
    • Amplitud = 90<br>
    • Margen = 0.05 × 90 = 4.5<br>
    • <b>Rango ampliado = [5.5, 104.5]</b><br>
    • Acepta valores 5.5 unidades más abajo y arriba<br><br>
    
    <b>Caso 3:</b> Rango = [10, 100], Tolerancia = 10%<br>
    • Amplitud = 90<br>
    • Margen = 0.10 × 90 = 9<br>
    • <b>Rango ampliado = [1, 109]</b><br>
    • Acepta hasta 9 unidades fuera del rango original<br><br>
    
    <b>Caso 4:</b> Rango = [50, 200], Tolerancia = 10%<br>
    • Amplitud = 200 - 50 = 150<br>
    • Margen = 0.10 × 150 = 15<br>
    • <b>Rango ampliado = [35, 215]</b><br>
    • Acepta hasta 15 unidades fuera<br><br>
    
    <b>⚙️ ¿Qué valor debo usar?</b><br>
    • <b>Si usas modo "eliminar":</b> Pon 0% (no importa el valor, no se usa)<br>
    • <b>Si usas modo "permitir_con_tolerancia":</b><br>
      - <b>3-5%:</b> Para compensar errores de redondeo/medición<br>
      - <b>10%:</b> Valor típico equilibrado (RECOMENDADO)<br>
      - <b>15-20%:</b> Solo si conoces bien la relación física<br><br>
    
    <b>⚠️ Cuidado con valores altos:</b><br>
    • Tolerancia alta = mayor rango = más predicciones aceptadas<br>
    • Pero predicciones muy lejos del rango de entrenamiento son POCO CONFIABLES<br>
    • Modelos polinómicos divergen rápidamente fuera del rango<br><br>
    
    <b>🔍 Ejemplo real de peligro:</b><br>
    Modelo cuadrático: y = 1 + 2x - 0.1x²<br>
    Entrenado en x ∈ [0, 10] → funciona bien<br>
    Con tolerancia=50%, acepta hasta x≈15:<br>
    • x=12: y = 1 + 24 - 14.4 = 10.6 (razonable)<br>
    • x=15: y = 1 + 30 - 22.5 = 8.5 (ya empieza a bajar!)<br>
    • x=20: y = 1 + 40 - 40 = 1 (¡colapso total!)<br>
    El modelo predice, pero las predicciones son INCORRECTAS porque el comportamiento cambió.<br><br>
    
    <b>🎓 Principio:</b> Tolerancia alta = más "libertad" para el modelo = menos seguridad en las predicciones.
    """,
    "modo_2d": """
    <div style="background:#FFF3E0;border-left:4px solid #F57C00;padding:12px;margin-bottom:8px;">
    <b style="color:#E65100;font-size:14px;">🗺️ Estrategia de extrapolación para modelos 2D</b>
    </div>
    
    <b>🎯 ¿Para qué sirve?</b><br>
    En modelos con 2 predictores (x₁, x₂), define cómo verificar si un punto nuevo está "dentro" del espacio de entrenamiento.<br>
    No es solo verificar x₁ y x₂ por separado - también importa si esa COMBINACIÓN de ambos fue observada.<br><br>
    
    <table style="width:100%;border-collapse:collapse;margin:10px 0;font-size:13px;">
    <tr style="background:#E3F2FD;">
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Modo</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Método de verificación</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Restrictividad</th>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>marginal</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Verifica x₁ y x₂ <b>independientemente</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="background:#FF9800;color:white;padding:2px 8px;border-radius:4px;">PERMISIVO</span></td>
    </tr>
    <tr style="background:#F5F5F5;">
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>convex_hull</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Verifica que (x₁,x₂) esté dentro del polígono convexo</td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="background:#4CAF50;color:white;padding:2px 8px;border-radius:4px;">ESTRICTO</span></td>
    </tr>
    </table>
    
    <div style="background:#E8F5E9;padding:10px;border-radius:4px;margin:10px 0;">
    <b style="color:#2E7D32;">📊 Modo: marginal (más simple)</b><br>
    <b>Regla:</b> Acepta si x₁ ∈ [x₁_min, x₁_max] Y x₂ ∈ [x₂_min, x₂_max]<br><br>
    <b>Ventajas:</b><br>
    • Simple y rápido de calcular<br>
    • Bueno si x₁ y x₂ son independientes<br>
    • Acepta todo el "rectángulo" de posibilidades<br><br>
    <b>Desventajas:</b><br>
    • Puede aceptar esquinas que NUNCA fueron observadas<br>
    • No considera restricciones físicas en combinaciones
    </div>
    
    <div style="background:#FFF3E0;padding:10px;border-radius:4px;margin:10px 0;">
    <b style="color:#E65100;">🔷 Modo: convex_hull (más riguroso)</b><br>
    <b>Regla:</b> Acepta solo si (x₁,x₂) está dentro del polígono que envuelve los puntos de entrenamiento<br><br>
    <b>Ventajas:</b><br>
    • Más conservador y seguro<br>
    • No acepta combinaciones que nunca existieron<br>
    • Respeta la "forma real" del espacio de datos<br><br>
    <b>Desventajas:</b><br>
    • Más complejo de calcular<br>
    • Puede rechazar combinaciones válidas si los datos tienen "huecos"
    </div>
    
    <b>💡 Ejemplo visual (entrenamiento en forma de L):</b><br><br>
    
    <b>Puntos de entrenamiento:</b><br>
    • Grupo A: x₁ ∈ [0,5], x₂ ∈ [0,5] (esquina inferior izquierda)<br>
    • Grupo B: x₁ ∈ [5,10], x₂ ∈ [5,10] (esquina superior derecha)<br>
    Forman una "L" en el espacio 2D.<br><br>
    
    <b>Predicción 1:</b> (x₁=3, x₂=3)<br>
    • Grupo A → ✅ Observado<br>
    • <code>marginal</code>: ✅ Acepta (x₁ en [0,10], x₂ en [0,10])<br>
    • <code>convex_hull</code>: ✅ Acepta (dentro del polígono L)<br><br>
    
    <b>Predicción 2:</b> (x₁=8, x₂=8)<br>
    • Grupo B → ✅ Observado<br>
    • <code>marginal</code>: ✅ Acepta<br>
    • <code>convex_hull</code>: ✅ Acepta<br><br>
    
    <b>Predicción 3:</b> (x₁=2, x₂=8) ← ¡Esquina superior izquierda!<br>
    • NO observado (está en el "hueco" de la L)<br>
    • <code>marginal</code>: ✅ Acepta (x₁ en [0,10] ✓, x₂ en [0,10] ✓)<br>
    • <code>convex_hull</code>: ❌ Rechaza (fuera del polígono L)<br><br>
    
    <b>Predicción 4:</b> (x₁=8, x₂=2) ← ¡Esquina inferior derecha!<br>
    • NO observado (otra esquina del "hueco")<br>
    • <code>marginal</code>: ✅ Acepta<br>
    • <code>convex_hull</code>: ❌ Rechaza<br><br>
    
    <b>⚙️ ¿Cuál debo usar?</b><br><br>
    
    <b>Usa "marginal" cuando:</b><br>
    • x₁ y x₂ son verdaderamente independientes<br>
    • No hay restricciones físicas en las combinaciones<br>
    • Quieres mayor cobertura de predicción<br>
    • Ejemplo: Temperatura y Presión en experimentos (se pueden combinar libremente)<br><br>
    
    <b>Usa "convex_hull" cuando:</b><br>
    • Hay restricciones físicas/mecánicas en las combinaciones<br>
    • No todas las combinaciones de x₁ y x₂ son posibles<br>
    • Quieres máxima seguridad (solo predecir en zona conocida)<br>
    • Ejemplos:<br>
      - Envergadura y Peso en aviones (no cualquier combinación es viable)<br>
      - RPM y Torque en motores (hay límites acoplados)<br>
      - Velocidad y Ángulo de ataque (existe envelope de vuelo)<br><br>
    
    <b>⚠️ Caso especial - Datos en forma de línea:</b><br>
    Si tus puntos están en una línea diagonal (ej: y=x), el convex hull es una línea delgada.<br>
    • <code>marginal</code>: Acepta todo el rectángulo<br>
    • <code>convex_hull</code>: Solo acepta cerca de la línea<br>
    En este caso, marginal puede ser demasiado permisivo y convex_hull demasiado restrictivo.<br>
    Solución: Usa <code>hull_pad</code> para agregar margen al hull.<br><br>
    
    <b>🎓 Principio:</b> Marginal asume independencia; convex_hull respeta la geometría real de los datos.
    """,
    "hull_pad": """
    <div style="background:#FFF3E0;border-left:4px solid #F57C00;padding:12px;margin-bottom:8px;">
    <b style="color:#E65100;font-size:14px;">📐 Padding del Convex Hull</b>
    </div>
    
    <b>🎯 ¿Para qué sirve?</b><br>
    Agranda el polígono convexo (hull) en todas direcciones para permitir pequeñas extrapolaciones.<br>
    Es como dibujar un "borde de seguridad" alrededor del área conocida.<br><br>
    
    <b>⚠️ IMPORTANTE:</b> Solo se usa si <code>modo_2d = "convex_hull"</code>. Con "marginal" no tiene efecto.<br><br>
    
    <b>📊 ¿Cómo funciona?</b><br>
    1. Se calcula el convex hull de los puntos de entrenamiento<br>
    2. Se "infla" el polígono agregando <code>hull_pad</code> unidades en todas direcciones<br>
    3. Puntos dentro del hull inflado son aceptados<br><br>
    
    <table style="width:100%;border-collapse:collapse;margin:10px 0;font-size:13px;">
    <tr style="background:#E3F2FD;">
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">hull_pad</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Descripción</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Uso recomendado</th>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>0</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Sin padding (hull estricto)</td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Máxima seguridad, no extrapolar</td>
    </tr>
    <tr style="background:#F5F5F5;">
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>1-5% del rango</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Padding pequeño</td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Compensar errores de medición (TÍPICO)</td>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>10% del rango</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Padding moderado</td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Permitir extrapolación controlada</td>
    </tr>
    <tr style="background:#F5F5F5;">
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>>20% del rango</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Padding grande</td>
        <td style="padding:8px;border:1px solid #E0E0E0;">NO recomendado (usa "marginal" mejor)</td>
    </tr>
    </table>
    
    <b>💡 Ejemplo numérico:</b><br><br>
    
    <b>Datos de entrenamiento:</b><br>
    Puntos: (0,0), (10,0), (10,10), (0,10) → Cuadrado de lado 10<br>
    Hull original: Cuadrado con vértices en esos 4 puntos<br><br>
    
    <b>Caso 1:</b> hull_pad = 0<br>
    • Hull inflado = Hull original<br>
    • Punto (10.5, 5): ❌ Fuera del hull → Rechazado<br>
    • Punto (9.5, 5): ✅ Dentro del hull → Aceptado<br><br>
    
    <b>Caso 2:</b> hull_pad = 1<br>
    • Hull inflado: Cuadrado de vértices (-1,-1), (11,-1), (11,11), (-1,11)<br>
    • Se agregó 1 unidad en cada dirección<br>
    • Punto (10.5, 5): ✅ Ahora está dentro → Aceptado<br>
    • Punto (11.5, 5): ❌ Todavía fuera → Rechazado<br><br>
    
    <b>Caso 3:</b> hull_pad = 2<br>
    • Hull inflado: Cuadrado más grande<br>
    • Punto (11.5, 5): ✅ Aceptado<br>
    • Punto (12.5, 5): ❌ Rechazado<br><br>
    
    <b>⚙️ ¿Cómo elegir el valor?</b><br><br>
    
    <b>Paso 1:</b> Calcula el rango típico de tus predictores<br>
    • Si x₁ ∈ [0, 100] → rango₁ = 100<br>
    • Si x₂ ∈ [0, 50] → rango₂ = 50<br>
    • rango_promedio = (100 + 50)/2 = 75<br><br>
    
    <b>Paso 2:</b> Decide el porcentaje de padding<br>
    • 3-5% del rango promedio: Padding conservador<br>
    • 10% del rango promedio: Padding moderado<br><br>
    
    <b>Paso 3:</b> Calcula hull_pad<br>
    • Padding 5%: hull_pad = 0.05 × 75 = 3.75<br>
    • Padding 10%: hull_pad = 0.10 × 75 = 7.5<br><br>
    
    <b>🔍 Ejemplo práctico - Aviones:</b><br>
    • x₁ = Envergadura [5, 15] metros → rango = 10m<br>
    • x₂ = Peso [500, 2000] kg → rango = 1500kg<br>
    • Rango promedio = (10 + 1500)/2 ≈ 755 (¡unidades incompatibles!)<br><br>
    
    <b>⚠️ PROBLEMA:</b> Las unidades son diferentes (metros vs kilogramos).<br>
    <b>Solución:</b> Normaliza los datos O usa padding en términos del rango de cada variable.<br><br>
    
    <b>Mejor enfoque:</b><br>
    • hull_pad = 5% del rango mínimo<br>
    • min(10, 1500) = 10<br>
    • hull_pad = 0.05 × 10 = 0.5<br>
    • Pero esto solo afecta a x₁... mejor normalizar los datos primero.<br><br>
    
    <b>✅ Recomendación práctica:</b><br>
    • Si tus datos están normalizados (escalados): hull_pad = 0.05 a 0.10<br>
    • Si tus datos están en unidades originales: Usa valores pequeños relativos al rango menor<br>
    • Valor típico seguro: <b>0</b> (sin padding) y confía en el hull estricto<br>
    • Si ves muchos rechazos justos en el borde: Prueba con 3-5% del rango<br><br>
    
    <b>⚠️ No uses padding grande:</b><br>
    Si necesitas hull_pad >20% del rango, probablemente deberías usar <code>modo_2d="marginal"</code> en vez de convex_hull.<br><br>
    
    <b>🎓 Principio:</b> Padding pequeño = tolerar errores de medición; padding grande = estás extrapolando demasiado.
    """,
}


# ============================================================================
# MODELOS - Configuración de tipos y parámetros de modelos
# ============================================================================

MODELOS = {
    "ponderaciones_seleccion": """
    <div style="background:#E3F2FD;border-left:4px solid #1976D2;padding:12px;margin-bottom:8px;">
    <b style="color:#0D47A1;font-size:14px;">🏆 Ponderaciones para selección de modelos</b>
    </div>
    
    <b>🎯 ¿Para qué sirve?</b><br>
    Cuando hay múltiples modelos candidatos (lineal, poly2, log, pot, exp en 1D o 2D),<br>
    este score combina múltiples métricas en un valor único para elegir el MEJOR modelo.<br><br>
    
    <b>📊 Fórmula del score de selección:</b><br>
    <code>score = w_mape × (1 - MAPE_norm) + w_r2 × R² + w_corr × |corr| + w_confianza × confianza</code><br><br>
    
    <b>Componentes explicados:</b><br><br>
    
    <table style="width:100%;border-collapse:collapse;margin:10px 0;font-size:13px;">
    <tr style="background:#E3F2FD;">
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Métrica</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Peso default</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Qué mide</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Cuándo aumentar</th>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>MAPE</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>0.5</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Error absoluto porcentual</td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Prioridad = PRECISIÓN de predicciones</td>
    </tr>
    <tr style="background:#F5F5F5;">
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>R²</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>0.2</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Varianza explicada</td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Prioridad = AJUSTE GLOBAL del patrón</td>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>Correlación</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>0.2</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Correlación de residuos</td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Prioridad = NORMALIDAD y validez estadística</td>
    </tr>
    <tr style="background:#F5F5F5;">
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>Confianza</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>0.1</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Penalizaciones (n, k, checks)</td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Prioridad = ESTABILIDAD y robustez</td>
    </tr>
    </table>
    
    <b>💡 Detalles de cada componente:</b><br><br>
    
    <b>1. MAPE (w_mape = 0.5 default):</b><br>
    • <code>(1 - MAPE_norm)</code> donde MAPE_norm = MAPE / umbral_mape_max<br>
    • Si MAPE=5% y umbral=35%, contribuye: 0.5 × (1 - 5/35) = 0.5 × 0.857 = 0.43<br>
    • Menor MAPE → Mayor contribución al score<br>
    • <b>Mayor peso:</b> Prioriza modelos con errores pequeños<br><br>
    
    <b>2. R² (w_r2 = 0.2 default):</b><br>
    • Directo: score += 0.2 × R²<br>
    • Si R²=0.8, contribuye: 0.2 × 0.8 = 0.16<br>
    • Mide cuánto de la variabilidad explica el modelo<br>
    • <b>Mayor peso:</b> Prioriza modelos que capturan el patrón global<br><br>
    
    <b>3. Correlación (w_corr = 0.2 default):</b><br>
    • Usa correlación absoluta de residuos: score += 0.2 × |corr|<br>
    • Modelos con residuos correlacionados son menos confiables<br>
    • <b>Mayor peso:</b> Prioriza modelos con residuos bien distribuidos<br><br>
    
    <b>4. Confianza (w_confianza = 0.1 default):</b><br>
    • Incorpora penalizaciones por n pequeño, k pequeño, checks 2D fallidos<br>
    • Score de confianza ya calculado (incluye LOOCV, métricas 2D, etc.)<br>
    • <b>Mayor peso:</b> Prioriza modelos con condiciones favorables (datos suficientes, no outliers, etc.)<br><br>
    
    <b>⚠️ Restricción crítica:</b><br>
    <b>w_mape + w_r2 + w_corr + w_confianza DEBE ser igual a 1.0</b><br>
    Si no, los scores no son comparables entre diferentes configuraciones.<br><br>
    
    <b>⚙️ Ejemplo de selección:</b><br><br>
    
    <b>Modelos candidatos para Peso_vacío:</b><br><br>
    
    <b>Modelo A (lineal-1D):</b><br>
    • MAPE=8%, R²=0.75, |corr|=0.65, confianza=0.70<br>
    • Score = 0.5×(1-8/35) + 0.2×0.75 + 0.2×0.65 + 0.1×0.70<br>
    • Score = 0.5×0.771 + 0.15 + 0.13 + 0.07 = <b>0.736</b><br><br>
    
    <b>Modelo B (poly2-1D):</b><br>
    • MAPE=5%, R²=0.85, |corr|=0.55, confianza=0.60 (penalizado por k=3)<br>
    • Score = 0.5×(1-5/35) + 0.2×0.85 + 0.2×0.55 + 0.1×0.60<br>
    • Score = 0.5×0.857 + 0.17 + 0.11 + 0.06 = <b>0.769</b> 🏆<br><br>
    
    <b>Modelo C (pot-1D):</b><br>
    • MAPE=10%, R²=0.80, |corr|=0.70, confianza=0.75<br>
    • Score = 0.5×(1-10/35) + 0.2×0.80 + 0.2×0.70 + 0.1×0.75<br>
    • Score = 0.5×0.714 + 0.16 + 0.14 + 0.075 = <b>0.732</b><br><br>
    
    <b>Resultado:</b> Se selecciona Modelo B (poly2-1D) porque tiene el score más alto.<br>
    Aunque tiene k=3 (penalizado), su MAPE bajo y R² alto compensan.<br><br>
    
    <div style="background:#E8F5E9;padding:10px;border-radius:4px;margin:10px 0;">
    <b style="color:#2E7D32;">💡 Configuraciones recomendadas:</b><br><br>
    
    <b>Balanceada (DEFAULT):</b><br>
    • mape=0.5, r2=0.2, corr=0.2, confianza=0.1<br>
    • Prioriza precisión pero considera otros factores<br>
    • Apropiada para mayoría de casos<br><br>
    
    <b>Priorizar precisión absoluta:</b><br>
    • mape=0.7, r2=0.1, corr=0.1, confianza=0.1<br>
    • Casi solo importa el error de predicción<br>
    • Útil para aplicaciones críticas de seguridad<br><br>
    
    <b>Priorizar ajuste global:</b><br>
    • mape=0.3, r2=0.4, corr=0.2, confianza=0.1<br>
    • Busca modelos que capturen bien el patrón<br>
    • Útil para análisis exploratorio<br><br>
    
    <b>Priorizar robustez:</b><br>
    • mape=0.4, r2=0.2, corr=0.2, confianza=0.2<br>
    • Da peso importante a condiciones favorables<br>
    • Útil con datos limitados o ruidosos
    </div>
    
    <b>🔍 Casos especiales:</b><br><br>
    
    <b>¿Qué pasa si todos los modelos tienen scores similares?</b><br>
    • El sistema usa "confianza" como desempate<br>
    • Si aún empatan, se prefiere el modelo más simple (menor k)<br><br>
    
    <b>¿Qué pasa si ningún modelo pasa los umbrales?</b><br>
    • No se selecciona ningún modelo<br>
    • La variable objetivo no se imputa (queda con NaNs)<br><br>
    
    <b>🎓 Principio:</b> Score de selección unifica múltiples criterios de calidad para elegir objetivamente el mejor modelo entre candidatos.
    """,
    "permitir_1d": """
    <div style="background:#E8F5E9;border-left:4px solid #4CAF50;padding:12px;margin-bottom:8px;">
    <b style="color:#2E7D32;font-size:14px;">📊 Modelos 1D (Univariados) — y = f(x)</b>
    </div>
    
    <b>🎯 ¿Para qué sirve?</b><br>
    Permite que el sistema use modelos con UN SOLO predictor para imputar valores faltantes.<br>
    Son los modelos más simples y robustos cuando una variable depende principalmente de otra.<br><br>
    
    <table style="width:100%;border-collapse:collapse;margin:10px 0;font-size:13px;">
    <tr style="background:#E3F2FD;">
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Aspecto</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Modelos 1D</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Comparación</th>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>Estructura</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">y = f(x₁)</td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Solo 1 predictor</td>
    </tr>
    <tr style="background:#F5F5F5;">
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>Datos necesarios</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Mínimo: n ≥ 2p (típ. 4-10 puntos)</td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="background:#4CAF50;color:white;padding:2px 6px;border-radius:3px;">POCOS</span></td>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>Complejidad</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Simple, fácil de interpretar</td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="background:#4CAF50;color:white;padding:2px 6px;border-radius:3px;">BAJA</span></td>
    </tr>
    <tr style="background:#F5F5F5;">
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>Robustez</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Alta, menor riesgo de sobreajuste</td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="background:#4CAF50;color:white;padding:2px 6px;border-radius:3px;">ALTA</span></td>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>Precisión potencial</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Buena si relación es univariada</td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="background:#FF9800;color:white;padding:2px 6px;border-radius:3px;">MEDIA</span></td>
    </tr>
    </table>
    
    <b>💡 Ejemplos prácticos:</b><br><br>
    
    <b>✅ Buenos casos para 1D:</b><br>
    • <b>Peso_vacío ~ Envergadura:</b> El peso tiende a crecer con la envergadura<br>
    • <b>Consumo ~ Velocidad:</b> Relación cuadrática típica<br>
    • <b>Carga_alar ~ Peso:</b> Relación directa para diseños similares<br>
    • <b>Empuje ~ Peso:</b> Proporcionalidad en motores de misma familia<br><br>
    
    <b>❌ Malos casos para 1D (mejor usar 2D):</b><br>
    • <b>Alcance ~ ?:</b> Depende de combustible Y eficiencia aerodinámica<br>
    • <b>Techo_servicio ~ ?:</b> Depende de empuje Y peso<br>
    • <b>Velocidad_pérdida ~ ?:</b> Depende de peso Y superficie alar<br><br>
    
    <b>⚙️ ¿Cuándo activar/desactivar?</b><br><br>
    
    <div style="background:#E8F5E9;padding:10px;border-radius:4px;margin:10px 0;">
    <b style="color:#2E7D32;">✅ ACTIVAR (recomendado por defecto):</b><br>
    • Dataset pequeño (n < 30 puntos por variable)<br>
    • Quieres máxima robustez<br>
    • Relaciones simples y directas<br>
    • Fase exploratoria (ver qué funciona)<br>
    • Prioridad: cobertura sobre precisión extrema
    </div>
    
    <div style="background:#FFEBEE;padding:10px;border-radius:4px;margin:10px 0;">
    <b style="color:#C62828;">❌ DESACTIVAR (casos específicos):</b><br>
    • Solo quieres modelos 2D (multivariados)<br>
    • Estás seguro de que todas las relaciones son multivariadas<br>
    • Dataset grande (n > 100) con muchas variables correlacionadas<br>
    • Ya probaste 1D y no funcionaron
    </div>
    
    <b>🎓 Principio:</b> Modelos 1D son el punto de partida ideal - simples, robustos, fáciles de validar.
    """,
    "permitir_2d": """
    <div style="background:#E8F5E9;border-left:4px solid #4CAF50;padding:12px;margin-bottom:8px;">
    <b style="color:#2E7D32;font-size:14px;">📊 Modelos 2D (Bivariados) — y = f(x₁, x₂)</b>
    </div>
    
    <b>🎯 ¿Para qué sirve?</b><br>
    Permite que el sistema use modelos con DOS predictores simultáneamente.<br>
    Capturan interacciones entre variables y relaciones más complejas que los modelos 1D.<br><br>
    
    <table style="width:100%;border-collapse:collapse;margin:10px 0;font-size:13px;">
    <tr style="background:#E3F2FD;">
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Aspecto</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Modelos 2D</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Comparación</th>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>Estructura</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">y = f(x₁, x₂)</td>
        <td style="padding:8px;border:1px solid #E0E0E0;">2 predictores + interacción</td>
    </tr>
    <tr style="background:#F5F5F5;">
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>Datos necesarios</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Mínimo: n ≥ 3p (típ. 15-30 puntos)</td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="background:#FF9800;color:white;padding:2px 6px;border-radius:3px;">MEDIOS</span></td>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>Complejidad</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Media, requiere interpretación</td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="background:#FF9800;color:white;padding:2px 6px;border-radius:3px;">MEDIA</span></td>
    </tr>
    <tr style="background:#F5F5F5;">
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>Robustez</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Media, riesgo de sobreajuste</td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="background:#FF9800;color:white;padding:2px 6px;border-radius:3px;">MEDIA</span></td>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>Precisión potencial</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Alta si hay interacción real</td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="background:#4CAF50;color:white;padding:2px 6px;border-radius:3px;">ALTA</span></td>
    </tr>
    </table>
    
    <b>💡 Ejemplos prácticos:</b><br><br>
    
    <b>✅ Buenos casos para 2D:</b><br>
    • <b>Velocidad_pérdida ~ (Peso, Superficie_alar):</b> V_stall = √(2·W/(ρ·S·C_L))<br>
    • <b>Alcance ~ (Combustible, Eficiencia_aero):</b> Interacción clara<br>
    • <b>Techo_servicio ~ (Empuje, Peso):</b> Relación de potencia/peso<br>
    • <b>Consumo_específico ~ (RPM, Temperatura):</b> Efectos combinados<br><br>
    
    <b>🔍 Ventaja clave - Captura de interacciones:</b><br>
    Modelo 1D: Peso_vacío = a + b·Envergadura<br>
    Modelo 2D: Peso_vacío = a + b₁·Envergadura + b₂·Área_alar + b₃·Envergadura·Área_alar<br>
    El término b₃ captura cómo el efecto de la envergadura CAMBIA según el área alar.<br><br>
    
    <b>⚠️ Desventajas:</b><br>
    • Necesita MÁS datos (al menos 3× más que 1D)<br>
    • Mayor riesgo de sobreajuste si n es pequeño<br>
    • Más difícil de interpretar (superficies vs líneas)<br>
    • Puede fallar si x₁ y x₂ están muy correlacionados (multicolinealidad)<br><br>
    
    <b>⚙️ ¿Cuándo activar/desactivar?</b><br><br>
    
    <div style="background:#E8F5E9;padding:10px;border-radius:4px;margin:10px 0;">
    <b style="color:#2E7D32;">✅ ACTIVAR (recomendado por defecto):</b><br>
    • Dataset mediano/grande (n > 20-30 puntos)<br>
    • Sabes que hay interacciones entre variables<br>
    • Modelos 1D no alcanzan precisión suficiente<br>
    • Tienes suficiente diversidad en los datos<br>
    • Quieres máxima precisión (aceptando complejidad)
    </div>
    
    <div style="background:#FFEBEE;padding:10px;border-radius:4px;margin:10px 0;">
    <b style="color:#C62828;">❌ DESACTIVAR:</b><br>
    • Dataset muy pequeño (n < 15)<br>
    • Variables altamente correlacionadas (riesgo de multicolinealidad)<br>
    • Prioridad es simplicidad sobre precisión<br>
    • Fase inicial exploratoria (empieza con 1D primero)
    </div>
    
    <b>📊 Comparación directa:</b><br>
    Modelo 1D con 2 parámetros necesita n ≥ 6 puntos (típico)<br>
    Modelo 2D con 3 parámetros necesita n ≥ 18 puntos (típico)<br>
    Modelo 2D polinómico con 6 parámetros necesita n ≥ 36 puntos<br><br>
    
    <b>🎓 Principio:</b> Usa 2D cuando hay interacciones reales y tienes suficientes datos para estimarlas confiablemente.
    """,
    "poly_grado": """
    <div style="background:#FFF8E1;border-left:4px solid #F57C00;padding:12px;margin-bottom:8px;">
    <b style="color:#E65100;font-size:14px;">🔢 Grado máximo del polinomio</b>
    </div>
    
    <b>🎯 ¿Para qué sirve?</b><br>
    Define la máxima potencia permitida en los términos polinómicos.<br>
    Mayor grado = mayor flexibilidad = mayor capacidad de ajustar curvas complejas.<br>
    Pero también = mayor riesgo de sobreajuste.<br><br>
    
    <table style="width:100%;border-collapse:collapse;margin:10px 0;font-size:13px;">
    <tr style="background:#E3F2FD;">
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Grado</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Forma (1D)</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Parámetros (2D)</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Uso</th>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>1</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">y = β₀ + β₁x</td>
        <td style="padding:8px;border:1px solid #E0E0E0;">3 (β₀, β₁, β₂)</td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Relaciones lineales</td>
    </tr>
    <tr style="background:#F5F5F5;">
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>2</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">y = β₀ + β₁x + β₂x²</td>
        <td style="padding:8px;border:1px solid #E0E0E0;">6 (+ x₁², x₂², x₁x₂)</td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="background:#4CAF50;color:white;padding:2px 6px;border-radius:3px;">ÓPTIMO</span></td>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>3</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">+ β₃x³</td>
        <td style="padding:8px;border:1px solid #E0E0E0;">10 (muchos términos)</td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="background:#FF5722;color:white;padding:2px 6px;border-radius:3px;">RARO</span></td>
    </tr>
    </table>
    
    <b>📊 Desglose por grado:</b><br><br>
    
    <div style="background:#E8F5E9;padding:10px;border-radius:4px;margin:10px 0;">
    <b style="color:#2E7D32;">Grado 1 (Lineal)</b><br>
    <b>1D:</b> y = β₀ + β₁x → 2 parámetros<br>
    <b>2D:</b> y = β₀ + β₁x₁ + β₂x₂ → 3 parámetros<br><br>
    <b>Características:</b><br>
    • Líneas rectas (1D) o planos (2D)<br>
    • Muy robusto, difícil de sobreajustar<br>
    • Solo captura relaciones proporcionales<br>
    • Necesita: n ≥ 2p (típ. 4-6 puntos en 1D, 9-12 en 2D)
    </div>
    
    <div style="background:#E3F2FD;padding:10px;border-radius:4px;margin:10px 0;">
    <b style="color:#1565C0;">Grado 2 (Cuadrático) — RECOMENDADO</b><br>
    <b>1D:</b> y = β₀ + β₁x + β₂x² → 3 parámetros<br>
    <b>2D:</b> y = β₀ + β₁x₁ + β₂x₂ + β₃x₁² + β₄x₂² + β₅x₁x₂ → 6 parámetros<br><br>
    <b>Características:</b><br>
    • Parábolas (1D) o superficies curvas (2D)<br>
    • Captura efectos cuadráticos (drag ∝ V², lift ∝ α²...)<br>
    • Captura interacciones (término x₁x₂ en 2D)<br>
    • Balance ideal entre flexibilidad y robustez<br>
    • Necesita: n ≥ 3p (típ. 9-15 puntos en 1D, 18-30 en 2D)<br><br>
    <b>✅ Perfecto para:</b> Fenómenos aerodinámicos, termodinámicos, mecánicos
    </div>
    
    <div style="background:#FFEBEE;padding:10px;border-radius:4px;margin:10px 0;">
    <b style="color:#C62828;">Grado 3+ (Cúbico o mayor) — NO RECOMENDADO</b><br>
    <b>1D:</b> y = β₀ + β₁x + β₂x² + β₃x³ → 4 parámetros<br>
    <b>2D:</b> Hasta 10 parámetros (incluye x₁³, x₂³, x₁²x₂, x₁x₂², etc.)<br><br>
    <b>Problemas:</b><br>
    • Necesita MUCHOS datos (n ≥ 30-40 para 2D)<br>
    • Alto riesgo de sobreajuste<br>
    • Oscilaciones artificiales (efecto Runge)<br>
    • Difícil de justificar físicamente<br>
    • Extrapolación muy peligrosa<br><br>
    <b>❌ Solo usar si:</b> Tienes n > 100, fenómeno altamente no lineal, validación cruzada robusta
    </div>
    
    <b>💡 Ejemplo visual - Efecto del grado:</b><br>
    Datos reales: y = 10 + 2x - 0.1x² + ruido<br><br>
    
    <b>Grado 1:</b> Ajusta y = 10 + 1.8x<br>
    • No captura la curvatura → error sistemático<br>
    • Pero muy estable, no sobreajusta<br><br>
    
    <b>Grado 2:</b> Ajusta y = 10.2 + 1.95x - 0.09x²<br>
    • Captura la curvatura correctamente ✅<br>
    • Error mínimo, generaliza bien<br><br>
    
    <b>Grado 3:</b> Ajusta y = 10 + 2x - 0.08x² - 0.001x³<br>
    • Ajusta ligeramente mejor en entrenamiento<br>
    • Pero el término x³ es ruido, no señal<br>
    • Extrapolación diverge rápidamente ❌<br><br>
    
    <b>⚙️ ¿Qué valor usar?</b><br>
    • <b>Grado 1:</b> Solo si sabes que la relación es lineal<br>
    • <b>Grado 2:</b> <b>VALOR PREDETERMINADO</b> - úsalo siempre como punto de partida<br>
    • <b>Grado 3+:</b> Solo en casos excepcionales con validación rigurosa<br><br>
    
    <b>🎓 Principio:</b> Grado 2 captura la mayoría de relaciones físicas reales sin sobreajustar.
    """,
    "tipos_modelos": """
    <div style="background:#E3F2FD;border-left:4px solid #1976D2;padding:12px;margin-bottom:8px;">
    <b style="color:#0D47A1;font-size:14px;">🔧 Catálogo de tipos de modelos disponibles</b>
    </div>
    
    <b>🎯 ¿Para qué sirve?</b><br>
    Activa/desactiva familias de modelos específicas según el tipo de relación que esperas.<br>
    Cada familia se ajusta mejor a ciertos fenómenos físicos/matemáticos.<br><br>
    
    <table style="width:100%;border-collapse:collapse;margin:10px 0;font-size:13px;">
    <tr style="background:#E3F2FD;">
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Tipo</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Ecuación (1D)</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Restricciones</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Uso típico</th>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>lineal</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">y = β₀ + β₁x</td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Ninguna</td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Relaciones proporcionales</td>
    </tr>
    <tr style="background:#F5F5F5;">
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>poly2</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">y = β₀ + β₁x + β₂x²</td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Ninguna</td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Curvaturas, efectos cuadráticos</td>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>log</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">y = β₀ + β₁·ln(x)</td>
        <td style="padding:8px;border:1px solid #E0E0E0;">x > 0</td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Crec. decreciente (saturación)</td>
    </tr>
    <tr style="background:#F5F5F5;">
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>pot</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">y = β₀·x^β₁</td>
        <td style="padding:8px;border:1px solid #E0E0E0;">x > 0, y > 0</td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Leyes de escala, alometría</td>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>exp</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">y = β₀·e^(β₁x)</td>
        <td style="padding:8px;border:1px solid #E0E0E0;">y > 0 (para ajuste linealizado)</td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Crec./decaim. exponencial</td>
    </tr>
    </table>
    
    <b>📊 Descripción detallada:</b><br><br>
    
    <div style="background:#E8F5E9;padding:10px;border-radius:4px;margin:10px 0;">
    <b style="color:#2E7D32;">✅ lineal (ACTIVAR SIEMPRE)</b><br>
    <b>Forma:</b> y = β₀ + β₁x₁ (1D) o y = β₀ + β₁x₁ + β₂x₂ (2D)<br>
    <b>Características:</b><br>
    • El más simple y robusto<br>
    • No requiere transformaciones<br>
    • Funciona para cualquier rango de datos<br>
    • Coeficientes fáciles de interpretar<br><br>
    <b>Ejemplos de uso:</b><br>
    • Peso ~ Envergadura (proporcionalidad aproximada)<br>
    • Costo ~ Tamaño (relación casi lineal)<br>
    • Empuje ~ Área_motor (dentro de misma familia)<br><br>
    <b>Cuándo falla:</b> Relaciones con curvatura pronunciada o saturación
    </div>
    
    <div style="background:#E3F2FD;padding:10px;border-radius:4px;margin:10px 0;">
    <b style="color:#1565C0;">✅ poly2 (ACTIVAR SIEMPRE)</b><br>
    <b>Forma 1D:</b> y = β₀ + β₁x + β₂x²<br>
    <b>Forma 2D:</b> y = β₀ + β₁x₁ + β₂x₂ + β₃x₁² + β₄x₂² + β₅x₁x₂<br>
    <b>Características:</b><br>
    • Captura curvaturas y efectos no lineales<br>
    • Ideal para fenómenos físicos cuadráticos<br>
    • En 2D captura interacciones (x₁x₂)<br>
    • Necesita más datos que lineal<br><br>
    <b>Ejemplos de uso:</b><br>
    • Drag ~ Velocidad² (física fundamental)<br>
    • Consumo ~ RPM² (efectos mecánicos)<br>
    • Sustentación ~ Ángulo_ataque (con pérdida)<br>
    • Alcance ~ (Combustible, Eficiencia) con interacción<br><br>
    <b>Ventaja:</b> Muchos fenómenos aerodinámicos son inherentemente cuadráticos
    </div>
    
    <div style="background:#FFF3E0;padding:10px;border-radius:4px;margin:10px 0;">
    <b style="color:#E65100;">⚠️ log (activar selectivamente)</b><br>
    <b>Forma:</b> y = β₀ + β₁·ln(x)<br>
    <b>Restricción:</b> <b>x debe ser > 0</b> (logaritmo indefinido en x≤0)<br>
    <b>Características:</b><br>
    • Crecimiento rápido inicial, luego desacelera<br>
    • Captura saturación/rendimientos decrecientes<br>
    • Muy usado en economía y escalado<br><br>
    <b>Ejemplos de uso:</b><br>
    • Eficiencia ~ Tamaño (rendimientos decrecientes a escala)<br>
    • Costo_unitario ~ Cantidad (economías de escala)<br>
    • Percepción ~ Estímulo (ley de Weber-Fechner)<br><br>
    <b>Cuándo desactivar:</b><br>
    • Datos con x ≤ 0 o cerca de cero<br>
    • Relación no muestra saturación<br>
    • Preferís mantenerlo simple (lineal/poly2 suficiente)
    </div>
    
    <div style="background:#FFF3E0;padding:10px;border-radius:4px;margin:10px 0;">
    <b style="color:#E65100;">⚠️ pot (potencia - activar selectivamente)</b><br>
    <b>Forma:</b> y = β₀·x^β₁<br>
    <b>Restricciones:</b> <b>x > 0 Y y > 0</b> (se ajusta en espacio log-log)<br>
    <b>Características:</b><br>
    • Captura leyes de escala (alometría)<br>
    • Exponente β₁ indica tipo de escalado<br>
    • Se linealiza: ln(y) = ln(β₀) + β₁·ln(x)<br><br>
    <b>Ejemplos de uso:</b><br>
    • Área ~ Radio² (β₁≈2, escalado geométrico)<br>
    • Volumen ~ Radio³ (β₁≈3)<br>
    • Metabolismo ~ Masa^(3/4) (ley de Kleiber en biología)<br>
    • Resistencia_estructural ~ Diámetro^β (leyes de material)<br><br>
    <b>Cuándo activar:</b><br>
    • Conoces ley de escala física (área, volumen, etc.)<br>
    • Datos en rango amplio de magnitudes<br>
    • Gráfico log-log muestra línea recta<br><br>
    <b>Cuándo desactivar:</b><br>
    • Datos con valores ≤ 0<br>
    • No hay justificación física para ley de potencia<br>
    • Rangos estrechos (lineal funciona igual)
    </div>
    
    <div style="background:#FFF3E0;padding:10px;border-radius:4px;margin:10px 0;">
    <b style="color:#E65100;">⚠️ exp (exponencial - activar con precaución)</b><br>
    <b>Forma:</b> y = β₀·e^(β₁x)<br>
    <b>Restricción:</b> <b>y > 0</b> para ajuste linealizado (ln(y) = ln(β₀) + β₁x)<br>
    <b>Características:</b><br>
    • Crecimiento/decaimiento exponencial<br>
    • Puede diverger rápidamente fuera de rango<br>
    • MUY sensible a extrapolación<br>
    • β₁ > 0: crecimiento, β₁ < 0: decaimiento<br><br>
    <b>Ejemplos de uso:</b><br>
    • Temperatura ~ Tiempo (enfriamiento de Newton)<br>
    • Presión ~ Altitud (atmósfera exponencial)<br>
    • Crecimiento bacteriano ~ Tiempo<br>
    • Decaimiento radiactivo ~ Tiempo<br><br>
    <b>⚠️ PELIGROS:</b><br>
    • Extrapola a infinito muy rápido<br>
    • Datos con y ≤ 0 causan errores<br>
    • Fácil sobreajustar ruido como "exponencial"<br><br>
    <b>Cuándo activar:</b><br>
    • Fenómeno físico ES exponencial (temperatura, presión...)<br>
    • No vas a extrapolar mucho<br>
    • Datos claramente exponenciales en gráfico semilog<br><br>
    <b>Cuándo desactivar (RECOMENDADO):</b><br>
    • Datos con y ≤ 0<br>
    • No hay razón física para esperar exponencial<br>
    • Prefieres seguridad (poly2 es más robusto)
    </div>
    
    <b>⚙️ Configuración recomendada:</b><br><br>
    
    <b>🟢 Configuración ESTÁNDAR (máxima cobertura):</b><br>
    ✅ lineal<br>
    ✅ poly2<br>
    ✅ log<br>
    ✅ pot<br>
    ✅ exp<br>
    → Deja que el sistema pruebe todo y elija el mejor<br><br>
    
    <b>🟡 Configuración CONSERVADORA (solo robustos):</b><br>
    ✅ lineal<br>
    ✅ poly2<br>
    ❌ log, pot, exp<br>
    → Solo modelos que funcionan en cualquier rango<br><br>
    
    <b>🔴 Configuración MÍNIMA (exploración rápida):</b><br>
    ✅ lineal<br>
    ❌ poly2, log, pot, exp<br>
    → Solo relaciones simples, muy rápido<br><br>
    
    <b>🎓 Principio:</b> Activa todo por defecto; el sistema descartará automáticamente modelos inadecuados (datos fuera de rango, checks fallidos).
    """,
    "umbral_mape": """
    <div style="background:#FFEBEE;border-left:4px solid #D32F2F;padding:12px;margin-bottom:8px;">
    <b style="color:#C62828;font-size:14px;">📉 Umbral MAPE máximo — Filtro de calidad</b>
    </div>
    
    <b>🎯 ¿Para qué sirve?</b><br>
    Descarta automáticamente modelos cuyo error porcentual promedio supera este umbral.<br>
    Es un filtro de calidad mínima - si el modelo se equivoca demasiado, no vale la pena usarlo.<br><br>
    
    <b>📊 ¿Qué es MAPE?</b><br>
    <b>MAPE = Mean Absolute Percentage Error</b><br>
    <code>MAPE = (1/n) × Σ |y_real - y_pred| / |y_real| × 100%</code><br><br>
    
    <b>Interpretación:</b> Promedio del error relativo en cada predicción.<br>
    • MAPE = 10% significa que en promedio te equivocas un 10% del valor real<br>
    • Es independiente de la escala (puedes comparar modelos de diferentes variables)<br><br>
    
    <table style="width:100%;border-collapse:collapse;margin:10px 0;font-size:13px;">
    <tr style="background:#E3F2FD;">
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">MAPE</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Calidad</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Descripción</th>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>< 5%</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="background:#4CAF50;color:white;padding:2px 8px;border-radius:4px;">EXCELENTE</span></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Modelo muy preciso, alta confianza</td>
    </tr>
    <tr style="background:#F5F5F5;">
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>5-15%</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="background:#8BC34A;color:white;padding:2px 8px;border-radius:4px;">BUENO</span></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Precisión aceptable para la mayoría de usos</td>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>15-25%</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="background:#FF9800;color:white;padding:2px 8px;border-radius:4px;">MODERADO</span></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Útil para estimaciones preliminares</td>
    </tr>
    <tr style="background:#F5F5F5;">
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>25-35%</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="background:#FF5722;color:white;padding:2px 8px;border-radius:4px;">POBRE</span></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Baja confianza, usar con precaución</td>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>> 35%</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="background:#D32F2F;color:white;padding:2px 8px;border-radius:4px;">INACEPTABLE</span></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Modelo inútil, descartar</td>
    </tr>
    </table>
    
    <b>💡 Ejemplo numérico:</b><br>
    <b>Predicción de Peso_vacío (kg):</b><br><br>
    
    Observación 1: Real=1000kg, Predicho=950kg → Error=50kg → %Error=5%<br>
    Observación 2: Real=500kg, Predicho=550kg → Error=50kg → %Error=10%<br>
    Observación 3: Real=2000kg, Predicho=1900kg → Error=100kg → %Error=5%<br><br>
    
    <b>MAPE = (5% + 10% + 5%) / 3 = 6.7%</b><br>
    → Clasificación: BUENO ✅<br><br>
    
    <b>⚙️ ¿Qué valor usar como umbral?</b><br><br>
    
    <div style="background:#E8F5E9;padding:10px;border-radius:4px;margin:10px 0;">
    <b style="color:#2E7D32;">Configuraciones recomendadas:</b><br><br>
    <b>Umbral = 0.35 (35%) - VALOR PREDETERMINADO:</b><br>
    • Balance entre cobertura y calidad<br>
    • Descarta solo modelos muy malos<br>
    • Acepta modelos "pobres" si no hay alternativa mejor<br>
    • <b>Usa este como punto de partida</b><br><br>
    
    <b>Umbral = 0.20 (20%) - MODERADO:</b><br>
    • Más exigente, rechaza modelos "pobres"<br>
    • Solo acepta modelos razonables<br>
    • Puede reducir cobertura si datos son difíciles<br><br>
    
    <b>Umbral = 0.15 (15%) - ESTRICTO:</b><br>
    • Solo acepta modelos "buenos" o mejores<br>
    • Alta calidad garantizada<br>
    • Puede dejar muchas imputaciones sin hacer<br><br>
    
    <b>Umbral = 0.50 (50%) - PERMISIVO:</b><br>
    • Acepta casi cualquier modelo<br>
    • Máxima cobertura, calidad variable<br>
    • Solo usar si prioridad absoluta es completar datos
    </div>
    
    <b>⚠️ Problema conocido - Sensibilidad a valores pequeños:</b><br>
    Si y_real es muy pequeño (cercano a 0), el error porcentual explota:<br><br>
    
    <b>Ejemplo:</b><br>
    Real=0.1kg, Predicho=0.2kg → Error=0.1kg → %Error=100%!<br>
    Real=10kg, Predicho=11kg → Error=1kg → %Error=10%<br><br>
    
    Ambos tienen 1kg de error absoluto, pero el porcentaje es MUY diferente.<br>
    MAPE penaliza mucho los errores en valores pequeños.<br><br>
    
    <b>Solución:</b> Si trabajas con variables que tienen valores cerca de cero,<br>
    considera usar un umbral más permisivo o validar con otras métricas (RMSE, R²).<br><br>
    
    <b>🔍 Comparación con otras métricas:</b><br>
    • <b>MAPE:</b> Error relativo (%), independiente de escala<br>
    • <b>RMSE:</b> Error absoluto, en unidades originales<br>
    • <b>R²:</b> Proporción de varianza explicada (0-1)<br><br>
    
    MAPE es mejor cuando:<br>
    • Quieres comparar modelos de variables diferentes<br>
    • El error relativo importa más que el absoluto<br>
    • Trabajas con rangos amplios de valores<br><br>
    
    <b>🎓 Principio:</b> MAPE es un filtro de calidad mínima - modelos con MAPE alto son poco confiables y se descartan.
    """,
    "usar_loocv": """
    <div style="background:#E3F2FD;border-left:4px solid #1976D2;padding:12px;margin-bottom:8px;">
    <b style="color:#0D47A1;font-size:14px;">🔄 Leave-One-Out Cross-Validation (LOOCV)</b>
    </div>
    
    <b>🎯 ¿Para qué sirve?</b><br>
    Valida el modelo de forma robusta sin necesitar conjunto de prueba separado.<br>
    Especialmente útil cuando tienes POCOS datos (n < 50).<br><br>
    
    <b>📊 ¿Cómo funciona?</b><br>
    Para cada observación i de las n totales:<br>
    1. <b>Entrena</b> el modelo con todas las observaciones EXCEPTO la i<br>
    2. <b>Predice</b> el valor de la observación i (que el modelo nunca vio)<br>
    3. <b>Calcula</b> el error: error_i = y_i_real - y_i_predicho<br>
    4. <b>Repite</b> para todas las n observaciones<br>
    5. <b>MAPE_LOOCV</b> = promedio de |error_i / y_i_real| × 100%<br><br>
    
    <b>💡 Ejemplo visual con n=5 puntos:</b><br><br>
    
    <b>Iteración 1:</b> Entrena con {2,3,4,5}, predice 1 → error₁<br>
    <b>Iteración 2:</b> Entrena con {1,3,4,5}, predice 2 → error₂<br>
    <b>Iteración 3:</b> Entrena con {1,2,4,5}, predice 3 → error₃<br>
    <b>Iteración 4:</b> Entrena con {1,2,3,5}, predice 4 → error₄<br>
    <b>Iteración 5:</b> Entrena con {1,2,3,4}, predice 5 → error₅<br><br>
    
    <b>Resultado:</b> Tienes 5 predicciones, una para cada punto, donde cada predicción<br>
    se hizo SIN haber visto ese punto en el entrenamiento.<br><br>
    
    <table style="width:100%;border-collapse:collapse;margin:10px 0;font-size:13px;">
    <tr style="background:#E3F2FD;">
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Aspecto</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">LOOCV</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Entrenamiento simple</th>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>Validación</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Cada punto se predice sin haberlo visto</td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Evalúa en los mismos datos</td>
    </tr>
    <tr style="background:#F5F5F5;">
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>Robustez</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="background:#4CAF50;color:white;padding:2px 6px;border-radius:3px;">ALTA</span></td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="background:#FF5722;color:white;padding:2px 6px;border-radius:3px;">BAJA</span></td>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>Detecta sobreajuste</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">✅ Sí</td>
        <td style="padding:8px;border:1px solid #E0E0E0;">❌ No</td>
    </tr>
    <tr style="background:#F5F5F5;">
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>Costo computacional</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Alto (n entrenamientos)</td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Bajo (1 entrenamiento)</td>
    </tr>
    </table>
    
    <b>✅ Ventajas:</b><br>
    • <b>Detección de sobreajuste:</b> Si MAPE_LOOCV >> MAPE_entrenamiento, hay sobreajuste<br>
    • <b>Uso eficiente de datos:</b> Entrena con n-1 puntos (casi todos) en cada iteración<br>
    • <b>Estimación confiable:</b> Error refleja capacidad de generalización real<br>
    • <b>No necesita split:</b> No pierdes datos para validación<br>
    • <b>Ideal para n pequeño:</b> Cuando no puedes separar train/test<br><br>
    
    <b>❌ Desventajas:</b><br>
    • <b>Costo computacional:</b> Requiere entrenar n modelos (puede ser lento)<br>
    • <b>Varianza alta:</b> Con n muy pequeño (n<10), los resultados pueden variar mucho<br>
    • <b>Optimismo:</b> Con n muy grande, puede ser optimista (pero más eficiente usar split)<br><br>
    
    <b>⚙️ ¿Cuándo activar LOOCV?</b><br><br>
    
    <div style="background:#E8F5E9;padding:10px;border-radius:4px;margin:10px 0;">
    <b style="color:#2E7D32;">✅ ACTIVAR (RECOMENDADO) cuando:</b><br>
    • Dataset pequeño: <b>n < 50 puntos</b><br>
    • Quieres validación robusta sin perder datos<br>
    • Te preocupa el sobreajuste<br>
    • Modelos complejos (poly-2, 2D con interacciones)<br>
    • Datos de ingeniería/experimentales (costosos de obtener)<br>
    • <b>SIEMPRE</b> en diseño aeronáutico con datos limitados
    </div>
    
    <div style="background:#FFEBEE;padding:10px;border-radius:4px;margin:10px 0;">
    <b style="color:#C62828;">❌ DESACTIVAR cuando:</b><br>
    • Dataset muy grande: n > 200 (innecesario, usa split clásico)<br>
    • Tiempo de cómputo es crítico<br>
    • Modelos muy lentos de entrenar<br>
    • Confiabilidad temporal no es importante
    </div>
    
    <b>💡 Ejemplo real - Detección de sobreajuste:</b><br><br>
    
    <b>Modelo A (lineal):</b><br>
    • MAPE_entrenamiento = 12%<br>
    • MAPE_LOOCV = 13%<br>
    • Diferencia = 1% → ✅ Generaliza bien<br><br>
    
    <b>Modelo B (poly-2 con n=15):</b><br>
    • MAPE_entrenamiento = 3%<br>
    • MAPE_LOOCV = 18%<br>
    • Diferencia = 15% → ❌ SOBREAJUSTE severo<br>
    El modelo "memoriza" los datos pero no generaliza.<br><br>
    
    <b>Decisión:</b> Elegir modelo A aunque tenga peor error en entrenamiento,<br>
    porque LOOCV muestra que generalizará mejor.<br><br>
    
    <b>🔍 Comparación con otras técnicas:</b><br>
    • <b>LOOCV:</b> n iteraciones, usa n-1 puntos cada vez<br>
    • <b>K-fold CV:</b> k iteraciones (típ. k=5-10), usa (k-1)/k puntos<br>
    • <b>Hold-out:</b> 1 iteración, usa típ. 70-80% puntos<br><br>
    
    LOOCV es el MÁS exhaustivo pero también el MÁS costoso.<br><br>
    
    <b>🎓 Principio:</b> LOOCV estima el error de generalización de forma casi insesgada, esencial con pocos datos.
    """,
    "loocv_usa_pesos": """
    <div style="background:#E3F2FD;border-left:4px solid #1976D2;padding:12px;margin-bottom:8px;">
    <b style="color:#0D47A1;font-size:14px;">⚖️ LOOCV con pesos de outliers</b>
    </div>
    
    <b>🎯 ¿Para qué sirve?</b><br>
    Reduce la influencia de outliers (valores atípicos) al calcular las métricas de LOOCV.<br>
    Los puntos marcados como outliers contribuyen menos al error promedio.<br><br>
    
    <b>📊 ¿Cómo funciona?</b><br>
    Cada observación tiene un peso w_i entre 0 y 1:<br>
    • w_i = 1: Observación normal, peso completo<br>
    • w_i = 0.5: Outlier moderado, peso reducido<br>
    • w_i = 0: Outlier severo, excluido<br><br>
    
    <b>Sin pesos (default):</b><br>
    <code>MAPE_LOOCV = (1/n) × Σ |error_i / y_i|</code><br><br>
    
    <b>Con pesos activado:</b><br>
    <code>MAPE_LOOCV = Σ (w_i × |error_i / y_i|) / Σ w_i</code><br><br>
    
    <b>💡 Ejemplo numérico:</b><br>
    5 puntos con errores porcentuales: [5%, 8%, 50%, 6%, 7%]<br>
    Punto 3 es outlier identificado con peso w₃=0.2<br>
    Otros puntos tienen peso w=1<br><br>
    
    <b>Sin pesos:</b><br>
    MAPE = (5 + 8 + 50 + 6 + 7) / 5 = <b>15.2%</b><br>
    → Outlier infla mucho el error promedio<br><br>
    
    <b>Con pesos:</b><br>
    Numerador = 1×5 + 1×8 + 0.2×50 + 1×6 + 1×7 = 5 + 8 + 10 + 6 + 7 = 36<br>
    Denominador = 1 + 1 + 0.2 + 1 + 1 = 4.2<br>
    MAPE = 36 / 4.2 = <b>8.6%</b><br>
    → Refleja mejor el error en puntos "normales"<br><br>
    
    <table style="width:100%;border-collapse:collapse;margin:10px 0;font-size:13px;">
    <tr style="background:#E3F2FD;">
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Configuración</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Sensibilidad a outliers</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Uso</th>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>Sin pesos</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="background:#FF5722;color:white;padding:2px 6px;border-radius:3px;">ALTA</span></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Quieres que outliers afecten la métrica</td>
    </tr>
    <tr style="background:#F5F5F5;">
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>Con pesos</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="background:#4CAF50;color:white;padding:2px 6px;border-radius:3px;">BAJA</span></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Validar modelo en datos "típicos"</td>
    </tr>
    </table>
    
    <b>⚙️ ¿Cuándo activar?</b><br><br>
    
    <div style="background:#E8F5E9;padding:10px;border-radius:4px;margin:10px 0;">
    <b style="color:#2E7D32;">✅ ACTIVAR cuando:</b><br>
    • Tienes outliers identificados (sistema de detección activo)<br>
    • Quieres evaluar el modelo en la "población normal"<br>
    • Los outliers son errores de medición o casos excepcionales<br>
    • Te interesa el desempeño típico, no el peor caso<br>
    • Comparar modelos de forma justa (sin que outliers dominen)<br><br>
    <b>Ejemplo:</b> Datos de aviones donde un prototipo experimental es outlier.<br>
    El LOOCV con pesos evalúa cómo funciona en aviones "normales".
    </div>
    
    <div style="background:#FFF3E0;padding:10px;border-radius:4px;margin:10px 0;">
    <b style="color:#E65100;">❌ DESACTIVAR (default) cuando:</b><br>
    • No tienes sistema de detección de outliers<br>
    • Quieres que el modelo se evalúe en TODOS los datos por igual<br>
    • Los "outliers" son casos legítimos que deben influir<br>
    • Prefieres una métrica conservadora (peor caso)<br>
    • No estás seguro de si hay outliers reales<br><br>
    <b>Ejemplo:</b> Todos los puntos son mediciones válidas sin casos especiales.
    </div>
    
    <b>🔍 Relación con detección de outliers:</b><br>
    Esta opción solo tiene efecto si:<br>
    1. Tienes un sistema de detección de outliers activo<br>
    2. Algunos puntos fueron marcados como outliers (w < 1)<br><br>
    
    Si todos los puntos tienen w=1, no hay diferencia entre activar o no.<br><br>
    
    <b>⚠️ Advertencia - Uso incorrecto:</b><br>
    Si "marcas" puntos difíciles como outliers solo para mejorar métricas,<br>
    estás engañándote. Los outliers deben ser casos reales excepcionales,<br>
    no solo puntos donde el modelo falla.<br><br>
    
    <b>✅ Outliers legítimos:</b><br>
    • Errores de medición confirmados<br>
    • Prototipos o diseños experimentales únicos<br>
    • Configuraciones extremas fuera de operación normal<br>
    • Datos de otra población/generación<br><br>
    
    <b>❌ NO son outliers:</b><br>
    • Puntos donde el modelo simplemente predice mal<br>
    • Datos en regiones con alta incertidumbre natural<br>
    • Casos legítimos que no te gustan<br><br>
    
    <b>🎓 Principio:</b> Pesos permiten evaluar modelos de forma robusta cuando hay casos excepcionales identificados, sin que dominen la métrica.
    """,
}

# =====================================================================
# CONFIANZA - Cálculo del score de confianza del modelo
# =====================================================================
CONFIANZA = {
    "ponderaciones": """
    <div style="background:#E3F2FD;border-left:4px solid #1976D2;padding:12px;margin-bottom:8px;">
    <b style="color:#0D47A1;font-size:14px;">⚖️ Ponderaciones de confianza — Score unificado</b>
    </div>
    
    <b>🎯 ¿Para qué sirve?</b><br>
    Combina múltiples métricas (R², MAPE) en un score de confianza unificado que refleja<br>
    qué tan confiable es el modelo para hacer predicciones.<br><br>
    
    <b>📊 Fórmula del score base:</b><br>
    <code>confianza = w_r2 × R² + w_mape × (1/(1 + MAPE/divisor))</code><br><br>
    
    <b>Donde:</b><br>
    • <b>R²:</b> Coeficiente de determinación (0-1), mide varianza explicada<br>
    • <b>MAPE:</b> Mean Absolute Percentage Error (%), mide error promedio<br>
    • <b>w_r2, w_mape:</b> Pesos de cada componente (<b>deben sumar 1.0</b>)<br>
    • <b>divisor:</b> Controla sensibilidad al MAPE<br><br>
    
    <b>💡 Interpretación del divisor:</b><br>
    El divisor define el punto de "neutralidad" del MAPE:<br><br>
    
    <b>Si MAPE = divisor:</b><br>
    • Componente MAPE = 1/(1+1) = 0.5<br>
    • El MAPE contribuye con la mitad de su potencial máximo<br><br>
    
    <b>Si MAPE < divisor:</b><br>
    • Componente MAPE > 0.5<br>
    • Error bajo → contribución positiva a la confianza<br><br>
    
    <b>Si MAPE > divisor:</b><br>
    • Componente MAPE < 0.5<br>
    • Error alto → penaliza la confianza<br><br>
    
    <table style="width:100%;border-collapse:collapse;margin:10px 0;font-size:13px;">
    <tr style="background:#E3F2FD;">
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Divisor</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Efecto</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">MAPE=10% contrib.</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">MAPE=20% contrib.</th>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>5</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Muy exigente</td>
        <td style="padding:8px;border:1px solid #E0E0E0;">0.33</td>
        <td style="padding:8px;border:1px solid #E0E0E0;">0.20</td>
    </tr>
    <tr style="background:#F5F5F5;">
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>10</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Moderado-estricto</td>
        <td style="padding:8px;border:1px solid #E0E0E0;">0.50</td>
        <td style="padding:8px;border:1px solid #E0E0E0;">0.33</td>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>15 (DEFAULT)</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Balanceado</td>
        <td style="padding:8px;border:1px solid #E0E0E0;">0.60</td>
        <td style="padding:8px;border:1px solid #E0E0E0;">0.43</td>
    </tr>
    <tr style="background:#F5F5F5;">
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>20</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Tolerante</td>
        <td style="padding:8px;border:1px solid #E0E0E0;">0.67</td>
        <td style="padding:8px;border:1px solid #E0E0E0;">0.50</td>
    </tr>
    </table>
    
    <b>⚙️ Ejemplo numérico completo:</b><br><br>
    
    <b>Configuración:</b> w_r2=0.5, w_mape=0.5, divisor=15<br>
    <b>Modelo:</b> R²=0.80, MAPE=10%<br><br>
    
    <b>Cálculo:</b><br>
    • Componente R²: 0.5 × 0.80 = 0.40<br>
    • Componente MAPE: 0.5 × (1/(1+10/15)) = 0.5 × (1/1.667) = 0.5 × 0.60 = 0.30<br>
    • <b>Confianza total = 0.40 + 0.30 = 0.70</b> (70%)<br><br>
    
    <div style="background:#E8F5E9;padding:10px;border-radius:4px;margin:10px 0;">
    <b style="color:#2E7D32;">💡 Configuraciones recomendadas:</b><br><br>
    
    <b>Balanceada (DEFAULT):</b><br>
    • w_r2 = 0.5, w_mape = 0.5, divisor = 15<br>
    • Equilibrio entre ajuste global y precisión<br>
    • Apropiada para mayoría de casos<br><br>
    
    <b>Priorizar ajuste global:</b><br>
    • w_r2 = 0.7, w_mape = 0.3, divisor = 15<br>
    • Útil cuando buscas modelos que capturen patrones generales<br>
    • Menos sensible a errores puntuales<br><br>
    
    <b>Priorizar precisión:</b><br>
    • w_r2 = 0.3, w_mape = 0.7, divisor = 10<br>
    • Útil cuando errores de predicción son críticos<br>
    • Penaliza fuertemente imprecisiones
    </div>
    
    <b>⚠️ Restricción importante:</b><br>
    <b>w_r2 + w_mape DEBE ser igual a 1.0</b><br>
    Si no, el score no está normalizado y puede dar resultados inconsistentes.<br><br>
    
    <b>🔍 Relación entre componentes:</b><br><br>
    
    <b>R² alto, MAPE bajo:</b> ✅ Modelo excelente → confianza alta<br>
    <b>R² alto, MAPE alto:</b> ⚠️ Ajusta bien pero con errores → confianza moderada<br>
    <b>R² bajo, MAPE bajo:</b> ⚠️ Precisión sin capturar varianza → confianza baja-moderada<br>
    <b>R² bajo, MAPE alto:</b> ❌ Modelo pobre en ambos aspectos → confianza muy baja<br><br>
    
    <b>🎓 Principio:</b> El score de confianza unifica múltiples aspectos de calidad del modelo en una métrica interpretable (0-1).
    """,
    "loocv_aporte": """
    <div style="background:#E3F2FD;border-left:4px solid #1976D2;padding:12px;margin-bottom:8px;">
    <b style="color:#0D47A1;font-size:14px;">🔄 Aporte de LOOCV a la confianza</b>
    </div>
    
    <b>🎯 ¿Para qué sirve?</b><br>
    Ajusta el score de confianza según qué tan bien generaliza el modelo (evaluado con LOOCV).<br>
    Los modelos que pasan LOOCV con buenos resultados reciben un boost de confianza.<br><br>
    
    <b>📊 Sistema de clasificación LOOCV:</b><br>
    Cada modelo se clasifica en una de tres categorías según su desempeño en LOOCV:<br><br>
    
    <table style="width:100%;border-collapse:collapse;margin:10px 0;font-size:13px;">
    <tr style="background:#E3F2FD;">
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Clasificación</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Criterio (default)</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Factor</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Efecto</th>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="background:#4CAF50;color:white;padding:2px 8px;border-radius:4px;">ROBUSTO</span></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">MAPE ≤ 7.5% y R² ≥ 0.6</td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>1.0</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Sin penalización (excelente)</td>
    </tr>
    <tr style="background:#F5F5F5;">
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="background:#FF9800;color:white;padding:2px 8px;border-radius:4px;">NO ROBUSTO</span></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">MAPE ≤ 12.5% y R² ≥ 0.45</td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>0.85</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Penalización leve</td>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="background:#D32F2F;color:white;padding:2px 8px;border-radius:4px;">RECHAZADO</span></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">No cumple criterios</td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>0.6</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Penalización fuerte</td>
    </tr>
    </table>
    
    <b>📐 Fórmula de ajuste:</b><br>
    <code>conf_final = conf_base + w_aporte × (factor_clase - 0.85)</code><br><br>
    
    <b>Donde:</b><br>
    • <b>conf_base:</b> Confianza calculada con R² y MAPE (sin LOOCV)<br>
    • <b>w_aporte:</b> Peso del ajuste LOOCV (0-1)<br>
    • <b>factor_clase:</b> Factor según clasificación (robusto=1.0, no_robusto=0.85, rechazado=0.6)<br>
    • <b>0.85:</b> Punto de referencia (no_robusto no modifica la confianza)<br><br>
    
    <b>💡 Ejemplo numérico:</b><br><br>
    
    <b>Modelo A — ROBUSTO:</b><br>
    • conf_base = 0.70<br>
    • w_aporte = 0.20<br>
    • factor_clase = 1.0<br>
    • Ajuste = 0.20 × (1.0 - 0.85) = 0.20 × 0.15 = <b>+0.03</b><br>
    • <b>conf_final = 0.70 + 0.03 = 0.73</b> (mejora!)<br><br>
    
    <b>Modelo B — NO ROBUSTO:</b><br>
    • conf_base = 0.70<br>
    • w_aporte = 0.20<br>
    • factor_clase = 0.85<br>
    • Ajuste = 0.20 × (0.85 - 0.85) = <b>0.00</b><br>
    • <b>conf_final = 0.70</b> (sin cambio)<br><br>
    
    <b>Modelo C — RECHAZADO:</b><br>
    • conf_base = 0.70<br>
    • w_aporte = 0.20<br>
    • factor_clase = 0.6<br>
    • Ajuste = 0.20 × (0.6 - 0.85) = 0.20 × (-0.25) = <b>-0.05</b><br>
    • <b>conf_final = 0.70 - 0.05 = 0.65</b> (penalización!)<br><br>
    
    <div style="background:#E8F5E9;padding:10px;border-radius:4px;margin:10px 0;">
    <b style="color:#2E7D32;">⚙️ Configuración del peso (w_aporte):</b><br><br>
    
    <b>w = 0.0:</b> LOOCV no afecta la confianza (solo usa R² y MAPE)<br>
    <b>w = 0.1-0.2 (RECOMENDADO):</b> Influencia moderada, premia robustez sin dominar<br>
    <b>w = 0.3-0.5:</b> Influencia alta, LOOCV tiene peso significativo<br>
    <b>w ≥ 0.5:</b> LOOCV dominante (puede ser demasiado agresivo)
    </div>
    
    <b>⚠️ Consideración importante:</b><br>
    Este ajuste solo tiene sentido si LOOCV está activado.<br>
    Si "usar_loocv" está desactivado, este parámetro no hace nada.<br><br>
    
    <b>🎓 Principio:</b> Premia modelos que generalizan bien (LOOCV) y penaliza los que solo funcionan en entrenamiento (overfitting).
    """,
    "penalizacion_k": """
    <div style="background:#FFF3E0;border-left:4px solid #F57C00;padding:12px;margin-bottom:8px;">
    <b style="color:#E65100;font-size:14px;">📉 Penalización por tamaño muestral (k)</b>
    </div>
    
    <b>🎯 ¿Para qué sirve?</b><br>
    Reduce la confianza de modelos entrenados con POCOS datos.<br>
    k es el número de aeronaves (muestras) usadas para entrenar el modelo.<br><br>
    
    <b>📊 Funcionamiento:</b><br>
    Se aplica un factor multiplicativo que decrece cuando k es pequeño:<br>
    <code>Confianza_ajustada = Confianza_base × factor_penalización(k)</code><br><br>
    
    <b>Polinomio de penalización:</b><br>
    <code>factor = a5×(k/2)⁵ + a4×(k/2)⁴ + a3×(k/2)³ + a2×(k/2)² + a1×(k/2) + a0</code><br><br>
    
    <b>💡 Ejemplo:</b><br>
    • k=5 aeronaves → factor ≈ 0.4 → Confianza reducida al 40%<br>
    • k=10 aeronaves → factor ≈ 0.7 → Confianza reducida al 70%<br>
    • k=20+ aeronaves → factor ≈ 1.0 → Sin penalización<br><br>
    
    <b>⚠️ Advertencia:</b> Los valores por defecto están calibrados. Solo modificar si tienes experiencia con la curva de penalización.<br><br>
    
    <b>🔧 Parámetros (a5 a a0):</b> Coeficientes del polinomio que definen la forma de la curva de penalización.
    """,
    "penalizacion_n": """
    <div style="background:#FFF3E0;border-left:4px solid #F57C00;padding:12px;margin-bottom:8px;">
    <b style="color:#E65100;font-size:14px;">📉 Penalización por predictores (N)</b>
    </div>
    
    <b>🎯 ¿Para qué sirve?</b><br>
    Reduce la confianza de modelos con MUCHOS predictores.<br>
    N es el número de variables independientes (1 para 1D, 2 para 2D).<br><br>
    
    <b>📊 Funcionamiento:</b><br>
    Se aplica un factor que penaliza modelos más complejos:<br>
    <code>Confianza_ajustada = Confianza_base × factor_penalización(N)</code><br><br>
    
    <b>Polinomio de penalización:</b><br>
    <code>factor = b3×N³ + b2×N² + b1×N + b0</code><br><br>
    
    <b>💡 Ejemplo:</b><br>
    • N=1 (modelo 1D) → factor ≈ 1.1 → Leve bonificación<br>
    • N=2 (modelo 2D) → factor ≈ 1.0 → Sin penalización<br>
    • N=3+ (muy raro) → factor < 1.0 → Penalización<br><br>
    
    <b>⚠️ Advertencia:</b> Los valores por defecto están calibrados. Solo modificar si tienes experiencia.<br><br>
    
    <b>🔧 Parámetros (b3 a b0):</b> Coeficientes del polinomio cúbico de penalización.
    """,
    "penalizaciones_metricas": """
    <div style="background:#E8F5E9;border-left:4px solid #2E7D32;padding:12px;margin-bottom:8px;">
    <b style="color:#1B5E20;font-size:14px;">📊 Penalizaciones por métricas 2D</b>
    </div>
    
    <b>🎯 ¿Para qué sirve?</b><br>
    Ajusta la confianza según métricas específicas de modelos 2D:<br>
    • Correlación entre predictores (pearson, VIF)<br>
    • Condición de la matriz (cond, pc2_ratio)<br>
    • Cobertura del espacio (coverage_unique_pair, coverage_hull, coverage_ellipse)<br><br>
    
    <b>📊 Funcionamiento:</b><br>
    Cada métrica aporta un factor de ajuste (cuadrático):<br>
    <code>factor_métrica = c2×métrica² + c1×métrica + c0</code><br><br>
    
    <b>💡 Ejemplo - pearson_abs:</b><br>
    Si |correlación| = 0.8 entre x1 y x2:<br>
    <code>factor = -1.2×(0.8)² + 1.2×(0.8) + 0.2 = -0.768 + 0.96 + 0.2 = 0.392</code><br>
    → Penalización por correlación moderada<br><br>
    
    <b>✅ Checkbox "usar":</b><br>
    • Activado: La métrica afecta la confianza<br>
    • Desactivado: La métrica se ignora<br><br>
    
    <b>🔧 Parámetros c2, c1, c0:</b><br>
    Definen cómo la métrica afecta la confianza. Los valores por defecto están calibrados.<br><br>
    
    <b>⚠️ Advertencia:</b> Esta es configuración avanzada. Los defaults funcionan bien en la mayoría de casos.
    """,
}

# =====================================================================
# SELECCIÓN - Criterios de filtrado y selección de modelos
# =====================================================================
SELECCION = {
    "umbrales_train_prefiltro": """
    <div style="background:#FFF3E0;border-left:4px solid #F57C00;padding:12px;margin-bottom:8px;">
    <b style="color:#E65100;font-size:14px;">🎯 Umbrales de selección — Filtros de calidad</b>
    </div>
    
    <b>🎯 ¿Para qué sirve?</b><br>
    Sistema de dos barreras para filtrar modelos de baja calidad ANTES de evaluaciones costosas.<br>
    Ahorra tiempo computacional descartando modelos malos tempranamente.<br><br>
    
    <b>📊 Sistema de dos barreras:</b><br><br>
    
    <table style="width:100%;border-collapse:collapse;margin:10px 0;font-size:13px;">
    <tr style="background:#E3F2FD;">
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Barrera</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Criterio (default)</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Cuándo se aplica</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Si falla</th>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>1. TRAIN</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">MAPE ≤ 7.5%<br>R² ≥ 0.6</td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Inmediatamente después de entrenar</td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="background:#D32F2F;color:white;padding:2px 6px;border-radius:3px;">DESCARTADO</span></td>
    </tr>
    <tr style="background:#F5F5F5;">
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>2. PRE-FILTRO</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">MAPE ≤ 18%<br>R² ≥ 0.4</td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Antes de ejecutar LOOCV</td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="background:#FF5722;color:white;padding:2px 6px;border-radius:3px;">NO EVALÚA LOOCV</span></td>
    </tr>
    </table>
    
    <b>🔍 ¿Por qué dos barreras?</b><br><br>
    
    <b>Barrera 1 — TRAIN (estricta):</b><br>
    • Primera línea de defensa<br>
    • Estándares altos de calidad mínima<br>
    • Si el modelo no puede ajustar bien sus propios datos, es inútil<br>
    • <b>Efecto:</b> Descarta modelos claramente malos SIN ejecutar LOOCV<br><br>
    
    <b>Barrera 2 — PRE-FILTRO (relajada):</b><br>
    • Segunda oportunidad para modelos que no pasaron train<br>
    • Estándares más permisivos<br>
    • Evita ejecutar LOOCV (costoso) en modelos que ya sabemos son malos<br>
    • <b>Efecto:</b> Permite evaluar modelos "regulares" con LOOCV pero sin perder tiempo en pésimos<br><br>
    
    <b>💡 Flujo de decisión:</b><br><br>
    
    <div style="background:#ECEFF1;padding:10px;border-left:3px solid #607D8B;margin:10px 0;font-family:monospace;font-size:12px;">
    1. Entrenar modelo<br>
    ↓<br>
    2. ¿Pasa TRAIN (MAPE≤7.5%, R²≥0.6)?<br>
       ├─ ✅ SÍ → Modelo ACEPTADO → Continuar a LOOCV (si está activado)<br>
       └─ ❌ NO → Ir a paso 3<br>
    ↓<br>
    3. ¿Pasa PRE-FILTRO (MAPE≤18%, R²≥0.4)?<br>
       ├─ ✅ SÍ → Modelo CANDIDATO → Ejecutar LOOCV para decidir<br>
       └─ ❌ NO → <b>DESCARTADO</b> (no vale la pena evaluar con LOOCV)
    </div>
    
    <b>⚙️ Ejemplo numérico:</b><br><br>
    
    <b>Modelo A:</b> MAPE_train=6%, R²_train=0.75<br>
    • ✅ Pasa TRAIN → ACEPTADO → Se evalúa con LOOCV<br><br>
    
    <b>Modelo B:</b> MAPE_train=12%, R²_train=0.50<br>
    • ❌ No pasa TRAIN (MAPE>7.5%)<br>
    • ✅ Pasa PRE-FILTRO (MAPE≤18%, R²≥0.4) → CANDIDATO → Se evalúa con LOOCV<br>
    • (LOOCV decidirá si realmente sirve)<br><br>
    
    <b>Modelo C:</b> MAPE_train=25%, R²_train=0.30<br>
    • ❌ No pasa TRAIN<br>
    • ❌ No pasa PRE-FILTRO → <b>DESCARTADO</b> (no se ejecuta LOOCV)<br><br>
    
    <div style="background:#E8F5E9;padding:10px;border-radius:4px;margin:10px 0;">
    <b style="color:#2E7D32;">💡 Configuraciones recomendadas:</b><br><br>
    
    <b>Estricto (DEFAULT):</b><br>
    • TRAIN: MAPE ≤ 7.5%, R² ≥ 0.6<br>
    • PRE-FILTRO: MAPE ≤ 18%, R² ≥ 0.4<br>
    • Balance entre calidad y cobertura<br><br>
    
    <b>Muy exigente:</b><br>
    • TRAIN: MAPE ≤ 5%, R² ≥ 0.7<br>
    • PRE-FILTRO: MAPE ≤ 12%, R² ≥ 0.5<br>
    • Solo acepta modelos muy buenos<br><br>
    
    <b>Permisivo:</b><br>
    • TRAIN: MAPE ≤ 10%, R² ≥ 0.5<br>
    • PRE-FILTRO: MAPE ≤ 25%, R² ≥ 0.3<br>
    • Da más oportunidades a modelos mediocres
    </div>
    
    <b>⚠️ Costo computacional:</b><br>
    Cada modelo que pasa PRE-FILTRO ejecuta LOOCV, que requiere entrenar <b>n modelos</b><br>
    (uno por cada observación). Si n=30, LOOCV entrena 30 veces.<br>
    → PRE-FILTRO muy permisivo puede hacer el proceso LENTO.<br><br>
    
    <b>🎓 Principio:</b> Dos barreras progresivas ahorran tiempo descartando temprano modelos malos sin perder oportunidades.
    """,
    "criterios_loocv": """
    <div style="background:#E3F2FD;border-left:4px solid #1976D2;padding:12px;margin-bottom:8px;">
    <b style="color:#0D47A1;font-size:14px;">🔄 Criterios LOOCV — Clasificación de robustez</b>
    </div>
    
    <b>🎯 ¿Para qué sirve?</b><br>
    Clasifica modelos según su capacidad de generalización (evaluada con LOOCV) en tres categorías:<br>
    <b>ROBUSTO</b>, <b>NO ROBUSTO</b>, o <b>RECHAZADO</b>.<br><br>
    
    <b>📊 Sistema de clasificación:</b><br><br>
    
    <table style="width:100%;border-collapse:collapse;margin:10px 0;font-size:13px;">
    <tr style="background:#E3F2FD;">
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Clase</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Criterio LOOCV (default)</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Significado</th>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="background:#4CAF50;color:white;padding:2px 8px;border-radius:4px;">ROBUSTO</span></td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>MAPE ≤ 7.5%</b><br><b>R² ≥ 0.6</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Generalización excelente<br>Predice bien en datos no vistos</td>
    </tr>
    <tr style="background:#F5F5F5;">
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="background:#FF9800;color:white;padding:2px 8px;border-radius:4px;">NO ROBUSTO</span></td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>MAPE ≤ 12.5%</b><br><b>R² ≥ 0.45</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Generalización aceptable<br>Útil pero con reservas</td>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="background:#D32F2F;color:white;padding:2px 8px;border-radius:4px;">RECHAZADO</span></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">No cumple criterios</td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Mala generalización<br>Probablemente sobreajuste</td>
    </tr>
    </table>
    
    <b>🔍 ¿Por qué importa la clasificación LOOCV?</b><br><br>
    
    LOOCV evalúa cómo predice el modelo en observaciones que NO vio durante entrenamiento.<br>
    Es la prueba definitiva de si el modelo aprendió patrones reales o solo "memorizó" los datos.<br><br>
    
    <b>Caso típico - Overfitting:</b><br>
    • MAPE_train = 3% (¡excelente!)<br>
    • MAPE_LOOCV = 18% (¡terrible!)<br>
    • Diagnóstico: Modelo sobreajustado → RECHAZADO<br><br>
    
    <b>Caso ideal:</b><br>
    • MAPE_train = 6%<br>
    • MAPE_LOOCV = 7%<br>
    • Diagnóstico: Generaliza muy bien → ROBUSTO<br><br>
    
    <b>⚙️ Ratio val/train (detección de overfitting):</b><br><br>
    
    <code>ratio = MAPE_LOOCV / MAPE_train</code><br><br>
    
    <b>Interpretación:</b><br>
    • <b>ratio ≈ 1.0-1.5:</b> Generalización saludable ✅<br>
    • <b>ratio = 2-5:</b> Algo de overfitting ⚠️<br>
    • <b>ratio > 5:</b> <b>ALERTA</b> — Overfitting severo 🚨<br><br>
    
    <b>Umbral de alerta (default = 5.0):</b><br>
    Si ratio > 5, se genera una advertencia aunque cumpla otros criterios.<br><br>
    
    <b>💡 Ejemplo completo:</b><br><br>
    
    <b>Modelo A:</b><br>
    • MAPE_train = 5%, R²_train = 0.75<br>
    • MAPE_LOOCV = 6%, R²_LOOCV = 0.72<br>
    • Ratio = 6/5 = 1.2<br>
    • <b>Clasificación: ROBUSTO</b> ✅<br>
    • Explicación: Métricas excelentes, generaliza perfectamente<br><br>
    
    <b>Modelo B:</b><br>
    • MAPE_train = 8%, R²_train = 0.65<br>
    • MAPE_LOOCV = 11%, R²_LOOCV = 0.52<br>
    • Ratio = 11/8 = 1.375<br>
    • <b>Clasificación: NO ROBUSTO</b> ⚠️<br>
    • Explicación: Cumple criterios relajados, generalización aceptable<br><br>
    
    <b>Modelo C:</b><br>
    • MAPE_train = 3%, R²_train = 0.85<br>
    • MAPE_LOOCV = 22%, R²_LOOCV = 0.35<br>
    • Ratio = 22/3 = 7.33 > 5 🚨<br>
    • <b>Clasificación: RECHAZADO</b> ❌<br>
    • Explicación: Overfitting severo, no cumple criterios LOOCV<br><br>
    
    <div style="background:#E8F5E9;padding:10px;border-radius:4px;margin:10px 0;">
    <b style="color:#2E7D32;">⚙️ Configuraciones recomendadas:</b><br><br>
    
    <b>Estricto (DEFAULT):</b><br>
    • ROBUSTO: MAPE ≤ 7.5%, R² ≥ 0.6<br>
    • NO ROBUSTO: MAPE ≤ 12.5%, R² ≥ 0.45<br>
    • Ratio alerta: 5.0<br><br>
    
    <b>Muy exigente:</b><br>
    • ROBUSTO: MAPE ≤ 5%, R² ≥ 0.7<br>
    • NO ROBUSTO: MAPE ≤ 10%, R² ≥ 0.5<br>
    • Ratio alerta: 3.0<br><br>
    
    <b>Permisivo:</b><br>
    • ROBUSTO: MAPE ≤ 10%, R² ≥ 0.5<br>
    • NO ROBUSTO: MAPE ≤ 18%, R² ≥ 0.4<br>
    • Ratio alerta: 7.0
    </div>
    
    <b>🎓 Principio:</b> LOOCV clasifica modelos según su capacidad real de predicción en datos nuevos, identificando sobreajuste.
    """,
}

# =====================================================================
# OUTLIERS - Detección y manejo de valores atípicos
# =====================================================================
OUTLIERS = {
    "deteccion_pesos": """
    <div style="background:#FFEBEE;border-left:4px solid #D32F2F;padding:12px;margin-bottom:8px;">
    <b style="color:#C62828;font-size:14px;">🚨 Outliers — Detección y ponderación</b>
    </div>
    
    <b>🎯 ¿Para qué sirve?</b><br>
    Identifica y reduce la influencia de valores atípicos (outliers) sin eliminarlos completamente.<br>
    Los outliers reciben pesos reducidos en los análisis de correlación y regresión.<br><br>
    
    <b>📊 Método: Z-score estandarizado</b><br><br>
    
    <b>Z-score:</b> Mide cuántas desviaciones estándar se aleja un valor de la media:<br>
    <code>z = (x - μ) / σ</code><br><br>
    
    <b>Donde:</b><br>
    • x = valor observado<br>
    • μ = media de la variable<br>
    • σ = desviación estándar<br><br>
    
    <b>Interpretación:</b><br>
    • |z| = 1 → 1 desviación estándar del promedio (típico)<br>
    • |z| = 2 → 2 desviaciones (~95% de datos están dentro)<br>
    • |z| = 3 → 3 desviaciones (~99.7% dentro) → outlier suave<br>
    • |z| > 6 → Outlier extremo<br><br>
    
    <b>🎚️ Sistema de tres zonas:</b><br><br>
    
    <table style="width:100%;border-collapse:collapse;margin:10px 0;font-size:13px;">
    <tr style="background:#E3F2FD;">
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Z-score</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Clasificación</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Peso</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Acción</th>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>|z| ≤ z_suave (3)</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="background:#4CAF50;color:white;padding:2px 6px;border-radius:3px;">NORMAL</span></td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>w = 1.0</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Sin penalización</td>
    </tr>
    <tr style="background:#F5F5F5;">
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>z_suave < |z| ≤ z_duro (6)</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="background:#FF9800;color:white;padding:2px 6px;border-radius:3px;">OUTLIER SUAVE</span></td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>w reducido</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Peso calculado con fórmula</td>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>|z| > z_duro (6)</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><span style="background:#D32F2F;color:white;padding:2px 6px;border-radius:3px;">OUTLIER EXTREMO</span></td>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>w = w_min o 0</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Peso mínimo o eliminado</td>
    </tr>
    </table>
    
    <b>📐 Fórmula de pesos (zona intermedia):</b><br><br>
    
    Para outliers suaves (z_suave < |z| ≤ z_duro):<br>
    <code>w = w_min + (1 - w_min) × exp(-α × (|z| - z_suave)²)</code><br><br>
    
    <b>Parámetros:</b><br>
    • <b>α (alpha):</b> Controla agresividad de penalización<br>
      - α bajo (0.1-0.3): Penalización gradual<br>
      - α alto (0.5-1.0): Penalización agresiva<br>
    • <b>w_min:</b> Peso mínimo asignado (típ. 0.2 = 20%)<br>
    • <b>|z| - z_suave:</b> Distancia desde el umbral suave<br><br>
    
    <b>💡 Ejemplo numérico:</b><br><br>
    
    <b>Configuración:</b> z_suave=3, z_duro=6, α=0.5, w_min=0.2<br><br>
    
    <b>Punto A:</b> z = 2.0<br>
    • |z| ≤ 3 → Zona NORMAL<br>
    • <b>Peso = 1.0</b> (sin penalización)<br><br>
    
    <b>Punto B:</b> z = 4.5<br>
    • 3 < |z| ≤ 6 → Zona OUTLIER SUAVE<br>
    • Distancia = 4.5 - 3 = 1.5<br>
    • w = 0.2 + (1-0.2) × exp(-0.5 × 1.5²)<br>
    • w = 0.2 + 0.8 × exp(-1.125)<br>
    • w = 0.2 + 0.8 × 0.325 = 0.2 + 0.26 = <b>0.46</b><br><br>
    
    <b>Punto C:</b> z = 8.0<br>
    • |z| > 6 → Zona OUTLIER EXTREMO<br>
    • <b>Peso = 0.2</b> (w_min)<br>
    • Si remover_duro=True → <b>Peso = 0</b> (eliminado)<br><br>
    
    <div style="background:#E8F5E9;padding:10px;border-radius:4px;margin:10px 0;">
    <b style="color:#2E7D32;">⚙️ Configuraciones recomendadas:</b><br><br>
    
    <b>Moderado (DEFAULT):</b><br>
    • z_suave = 3.0, z_duro = 6.0<br>
    • alpha = 0.5, w_min = 0.2<br>
    • remover_duro = False<br>
    • Balance entre robustez e inclusión<br><br>
    
    <b>Conservador (más tolerante):</b><br>
    • z_suave = 4.0, z_duro = 8.0<br>
    • alpha = 0.3, w_min = 0.5<br>
    • Solo penaliza outliers muy extremos<br><br>
    
    <b>Agresivo (más estricto):</b><br>
    • z_suave = 2.5, z_duro = 5.0<br>
    • alpha = 1.0, w_min = 0.1<br>
    • remover_duro = True<br>
    • Penaliza fuertemente desviaciones moderadas
    </div>
    
    <b>⚙️ Opción "remover_duro":</b><br><br>
    
    <b>remover_duro = False (DEFAULT):</b><br>
    • Outliers extremos reciben peso w_min (típ. 0.2)<br>
    • Siguen contribuyendo mínimamente al análisis<br>
    • Más robusto a errores de clasificación<br><br>
    
    <b>remover_duro = True:</b><br>
    • Outliers extremos reciben peso 0 (eliminados efectivamente)<br>
    • Más agresivo, puede perder información valiosa<br>
    • Solo usar si estás SEGURO de que son errores de medición<br><br>
    
    <b>⚠️ Advertencia:</b><br>
    No todos los outliers son errores. Algunos son casos legítimos excepcionales:<br>
    • Prototipos experimentales<br>
    • Configuraciones extremas válidas<br>
    • Innovaciones disruptivas<br><br>
    
    Penalizar outliers legítimos puede sesgar los modelos.<br><br>
    
    <b>🎓 Principio:</b> Sistema de pesos reduce influencia de outliers sin eliminarlos, balanceando robustez y preservación de información.
    """,
}


# =====================================================================
# SIMILITUD - Configuración del motor de imputación por similitud
# =====================================================================
SIMILITUD = {
    "umbral_pct_diferencia": """
    <div style="background:#E8F5E9;border-left:4px solid #2E7D32;padding:12px;margin-bottom:8px;">
    <b style="color:#1B5E20;font-size:14px;">📏 Umbral de diferencia porcentual</b>
    </div>
    
    <b>🎯 ¿Para qué sirve?</b><br>
    Define cuánto puede diferir un vecino (aeronave similar) de la aeronave objetivo en cada parámetro
    para ser considerado "similar".<br><br>
    
    <b>📊 Funcionamiento:</b><br>
    Para cada parámetro, se calcula la diferencia porcentual:<br>
    <code>dif_pct = |valor_vecino - valor_objetivo| / |valor_objetivo|</code><br>
    Si <code>dif_pct ≤ umbral</code>, el vecino se acepta para ese parámetro.<br><br>
    
    <b>💡 Ejemplo:</b><br>
    Umbral = 0.20 (20%).<br>
    Tu aeronave tiene MTOW = 10,000 kg.<br>
    • Vecino A: MTOW = 11,500 kg → dif = 15% → ✅ Acepta<br>
    • Vecino B: MTOW = 13,000 kg → dif = 30% → ❌ Rechaza<br><br>
    
    <table style="width:100%;border-collapse:collapse;margin:10px 0;font-size:13px;">
    <tr style="background:#E3F2FD;">
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Umbral</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Efecto</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Uso</th>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>0.10 (10%)</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Muy estricto, pocos vecinos</td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Datos abundantes, alta precisión</td>
    </tr>
    <tr style="background:#F5F5F5;">
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>0.20 (20%) DEFAULT</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Balanceado</td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Mayoría de casos</td>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>0.30 (30%)</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Permisivo, más vecinos</td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Datos escasos</td>
    </tr>
    </table>
    
    <b>⚠️ Rango:</b> 0–1 (fracción, no porcentaje).<br>
    <b>🎓 Principio:</b> Umbral bajo = más selectivo = menos vecinos pero más parecidos.
    """,
    "min_familias": """
    <div style="background:#E3F2FD;border-left:4px solid #1976D2;padding:12px;margin-bottom:8px;">
    <b style="color:#0D47A1;font-size:14px;">👪 Mínimo de familias requeridas</b>
    </div>
    
    <b>🎯 ¿Para qué sirve?</b><br>
    Para considerar una aeronave como "vecino válido", debe ser similar en un mínimo de familias
    de parámetros (física, geométrica, prestacional, etc.).<br><br>
    
    <b>📊 ¿Qué es una familia?</b><br>
    Grupo de parámetros relacionados. Ejemplo:<br>
    • <b>Física:</b> MTOW, peso vacío, carga útil<br>
    • <b>Geométrica:</b> envergadura, superficie alar, alargamiento<br>
    • <b>Prestacional:</b> velocidad crucero, alcance, techo<br><br>
    
    <b>💡 Ejemplo:</b><br>
    min_familias = 3. Un vecino es válido si es similar en al menos 3 familias.<br>
    • Vecino A: similar en física ✅, geométrica ✅, prestacional ✅ → ✅ Acepta (3/3)<br>
    • Vecino B: similar en física ✅, geométrica ✅, prestacional ❌ → ❌ Rechaza (2/3)<br><br>
    
    <b>⚙️ Valores recomendados:</b><br>
    • <b>2:</b> Mínimo viable (si tienes pocas familias definidas)<br>
    • <b>3 (DEFAULT):</b> Balanceado, exige similitud multi-dimensional<br>
    • <b>4+:</b> Muy estricto, solo para datasets grandes<br><br>
    
    <b>🎓 Principio:</b> Exigir múltiples familias evita "vecinos falsos" que solo coinciden en un aspecto.
    """,
    "excepcion_min_familias": """
    <div style="background:#FFF3E0;border-left:4px solid #F57C00;padding:12px;margin-bottom:8px;">
    <b style="color:#E65100;font-size:14px;">🔓 Excepción al mínimo de familias</b>
    </div>
    
    <b>🎯 ¿Para qué sirve?</b><br>
    Relaja el requisito de familias mínimas si un vecino coincide en muchos parámetros individuales,
    aunque no cubra suficientes familias.<br><br>
    
    <b>📊 Lógica:</b><br>
    Si un vecino NO cumple <code>min_familias</code>, se acepta igualmente si:<br>
    <code>familias_coincidentes ≥ exc_min_familias AND parámetros_coincidentes ≥ exc_min_parametros</code><br><br>
    
    <b>💡 Ejemplo:</b><br>
    min_familias=3, excepción: min_familias=2, min_parametros=6.<br>
    • Vecino C: coincide en 2 familias y 8 parámetros → ✅ Acepta por excepción<br>
    • Vecino D: coincide en 2 familias y 4 parámetros → ❌ Rechaza<br><br>
    
    <b>⚙️ Valores por defecto:</b><br>
    • Excepción min familias: <b>2</b><br>
    • Excepción min parámetros: <b>6</b><br><br>
    
    <b>🎓 Principio:</b> Si un vecino es similar en muchos parámetros aunque dispersos entre pocas familias,
    sigue siendo un buen candidato.
    """,
    "k_min": """
    <div style="background:#E3F2FD;border-left:4px solid #1976D2;padding:12px;margin-bottom:8px;">
    <b style="color:#0D47A1;font-size:14px;">📊 Mínimo de vecinos (k_min)</b>
    </div>
    
    <b>🎯 ¿Para qué sirve?</b><br>
    Número mínimo de vecinos similares requeridos para hacer una imputación.<br>
    Si no se encuentran suficientes vecinos, NO se imputa (evita estimaciones con poca evidencia).<br><br>
    
    <b>💡 Ejemplo:</b><br>
    k_min = 3. Para imputar el alcance de la aeronave X, necesito al menos 3 aeronaves similares
    que tengan dato de alcance.<br>
    • Se encuentran 5 vecinos → ✅ Se imputa (promedio ponderado de 5)<br>
    • Se encuentran 2 vecinos → ❌ No se imputa (insuficiente evidencia)<br><br>
    
    <b>⚙️ Valores recomendados:</b><br>
    • <b>2:</b> Mínimo absoluto (acepta con pocos datos, riesgo alto)<br>
    • <b>3 (DEFAULT):</b> Razonable para la mayoría de datasets<br>
    • <b>5:</b> Más robusto, requiere más datos disponibles<br><br>
    
    <b>🎓 Principio:</b> k_min bajo = más imputaciones pero menos confiables; k_min alto = menos imputaciones pero más robustas.
    """,
    "k_max": """
    <div style="background:#E3F2FD;border-left:4px solid #1976D2;padding:12px;margin-bottom:8px;">
    <b style="color:#0D47A1;font-size:14px;">📊 Máximo de vecinos (k_max)</b>
    </div>
    
    <b>🎯 ¿Para qué sirve?</b><br>
    Limita cuántos vecinos se usan para promediar. Evita incluir vecinos "lejanos" que diluyan la estimación.<br><br>
    
    <b>💡 Ejemplo:</b><br>
    k_max = 10. Se encuentran 25 vecinos similares.<br>
    Solo se usan los 10 más cercanos (mayor similitud) para promediar.<br><br>
    
    <b>⚙️ Valores recomendados:</b><br>
    • <b>5:</b> Solo los más cercanos (menos dilución, más sensible a outliers)<br>
    • <b>10 (DEFAULT):</b> Buen balance<br>
    • <b>20+:</b> Más suavizado, bueno para datasets grandes<br><br>
    
    <b>🎓 Principio:</b> Limitar vecinos mejora la precisión si los extras son menos relevantes.
    """,
    "peso_confianza_similitud": """
    <div style="background:#E8F5E9;border-left:4px solid #2E7D32;padding:12px;margin-bottom:8px;">
    <b style="color:#1B5E20;font-size:14px;">⚖️ Peso confianza similitud</b>
    </div>
    
    <b>🎯 ¿Para qué sirve?</b><br>
    Determina cuánto pesa la "similitud promedio" de los vecinos en la confianza final de la imputación.<br><br>
    
    <b>📊 Fórmula de confianza:</b><br>
    <code>confianza = peso_sim × score_similitud + peso_cv × (1 - CV)</code><br>
    Donde:<br>
    • <b>score_similitud:</b> Qué tan parecidos son los vecinos al objetivo (0-1)<br>
    • <b>CV:</b> Coeficiente de variación de los valores de los vecinos<br>
    • <b>peso_sim + peso_cv = 1.0</b><br><br>
    
    <b>💡 Interpretación:</b><br>
    • peso_sim = 0.7 → Confías más en que los vecinos sean "parecidos"<br>
    • peso_sim = 0.3 → Confías más en que los vecinos "coincidan" entre sí<br><br>
    
    <b>⚙️ Valor por defecto:</b> 0.7 (priorizar similitud sobre dispersión).<br>
    <b>🎓 Principio:</b> Si los vecinos son muy parecidos al objetivo, la imputación es buena aunque difieran algo entre sí.
    """,
    "peso_confianza_cv": """
    <div style="background:#E8F5E9;border-left:4px solid #2E7D32;padding:12px;margin-bottom:8px;">
    <b style="color:#1B5E20;font-size:14px;">⚖️ Peso confianza CV (variabilidad)</b>
    </div>
    
    <b>🎯 ¿Para qué sirve?</b><br>
    Determina cuánto pesa la "concordancia" entre los vecinos (baja variabilidad = alta confianza).<br><br>
    
    <b>📊 ¿Qué es CV?</b><br>
    <b>CV = Coeficiente de Variación = std / mean</b><br>
    Mide qué tan dispersos están los valores de los vecinos.<br>
    • CV = 0.05 (5%): Vecinos muy concordantes → alta confianza<br>
    • CV = 0.30 (30%): Vecinos dispersos → baja confianza<br><br>
    
    <b>💡 Ejemplo:</b><br>
    3 vecinos con alcance: [2000, 2100, 2050] km → CV=0.025 → alta concordancia<br>
    3 vecinos con alcance: [1500, 2500, 3000] km → CV=0.32 → baja concordancia<br><br>
    
    <b>⚙️ Valor por defecto:</b> 0.3 (complemento del peso de similitud: 1.0 - 0.7 = 0.3).<br>
    <b>⚠️ Restricción:</b> peso_sim + peso_cv debe sumar 1.0.<br>
    <b>🎓 Principio:</b> Si los vecinos dan valores similares entre sí, es más creíble la imputación.
    """,
    "verbosidad": """
    <div style="background:#F3E5F5;border-left:4px solid #7B1FA2;padding:12px;margin-bottom:8px;">
    <b style="color:#4A148C;font-size:14px;">🔊 Verbosidad (nivel de detalle)</b>
    </div>
    
    <b>🎯 ¿Para qué sirve?</b><br>
    Controla cuánta información se imprime en la consola durante la ejecución.<br><br>
    
    <table style="width:100%;border-collapse:collapse;margin:10px 0;font-size:13px;">
    <tr style="background:#E3F2FD;">
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Nivel</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Detalle</th>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>0 (DEFAULT)</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Silencioso: solo errores</td>
    </tr>
    <tr style="background:#F5F5F5;">
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>1</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Resumen: progreso general</td>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>2</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Detallado: vecinos encontrados, scores</td>
    </tr>
    <tr style="background:#F5F5F5;">
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>3</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Debug completo: cálculos intermedios</td>
    </tr>
    </table>
    
    <b>💡 Recomendación:</b> Usar 0 en producción, 2 para diagnosticar problemas.
    """,
    "familias": """
    <div style="background:#FFF3E0;border-left:4px solid #F57C00;padding:12px;margin-bottom:8px;">
    <b style="color:#E65100;font-size:14px;">👪 Familias y características</b>
    </div>
    
    <b>🎯 ¿Para qué sirve?</b><br>
    Define grupos de parámetros (familias) y qué columnas pertenecen a cada una.<br>
    La similitud se evalúa por familia: un vecino es "similar en familia X" si coincide en
    suficientes parámetros de esa familia.<br><br>
    
    <b>📊 Estructura:</b><br>
    • <b>Nombre de familia:</b> Identificador (ej: "fisica")<br>
    • <b>Características (CSV):</b> Columnas del Excel separadas por coma<br><br>
    
    <b>💡 Ejemplo:</b><br>
    <code>fisica: MTOW, peso_vacio, carga_util, empuje_max</code><br>
    <code>geometrica: envergadura, superficie_alar, alargamiento</code><br>
    <code>prestacional: velocidad_crucero, alcance, techo_servicio</code><br><br>
    
    <b>⚙️ familias_usadas:</b><br>
    De todas las familias definidas, selecciona cuáles se usan activamente para buscar vecinos.
    Esto permite tener familias definidas pero no usarlas temporalmente.<br><br>
    
    <b>⚠️ Importante:</b><br>
    • Los nombres de las características deben coincidir EXACTAMENTE con las columnas del Excel<br>
    • Si una columna no existe, se ignora (no da error)<br>
    • Familias vacías se omiten automáticamente<br><br>
    
    <b>🎓 Principio:</b> Agrupar parámetros en familias permite evaluar similitud multi-dimensional.
    """,
    "funcion_similitud": """
    <div style="background:#E3F2FD;border-left:4px solid #1976D2;padding:12px;margin-bottom:8px;">
    <b style="color:#0D47A1;font-size:14px;">📈 Función de similitud (polinomio)</b>
    </div>
    
    <b>🎯 ¿Para qué sirve?</b><br>
    Convierte la diferencia porcentual entre vecino y objetivo en un "score de similitud" (0-1).<br>
    Diferencia 0% → score 1.0 (idéntico). Diferencia alta → score bajo (poco similar).<br><br>
    
    <b>📊 Fórmula:</b><br>
    <code>score = a2×x² + a1×x + a0</code><br>
    Donde x = diferencia porcentual (0 ≤ x ≤ dominio_max_pct).<br><br>
    
    <b>💡 Ejemplo con defaults (a2=-0.002, a1=-0.01, a0=1.0):</b><br>
    • x=0% (idéntico) → score = 1.0<br>
    • x=5% → score = -0.002×25 + (-0.01)×5 + 1.0 = -0.05 - 0.05 + 1.0 = 0.90<br>
    • x=10% → score = -0.002×100 + (-0.01)×10 + 1.0 = -0.2 - 0.1 + 1.0 = 0.70<br>
    • x=20% (dominio) → score ≈ 0.0 (límite)<br><br>
    
    <b>⚙️ Parámetros:</b><br>
    • <b>a2:</b> Curvatura (negativo = decae más rápido al final)<br>
    • <b>a1:</b> Pendiente lineal (negativo = penaliza diferencias)<br>
    • <b>a0:</b> Score cuando x=0 (normalmente 1.0)<br>
    • <b>dominio_max_pct:</b> Más allá de este %, el score es 0 (máxima diferencia aceptable)<br><br>
    
    <b>⚠️ Advertencia:</b> Modificar coeficientes cambia la forma de la curva. Los defaults están calibrados.
    """,
    "vecinos": """
    <div style="background:#E8F5E9;border-left:4px solid #2E7D32;padding:12px;margin-bottom:8px;">
    <b style="color:#1B5E20;font-size:14px;">🎯 Selección de vecinos</b>
    </div>
    
    <b>🎯 ¿Para qué sirve?</b><br>
    Controla cómo se seleccionan los vecinos una vez identificados como "similares".<br><br>
    
    <table style="width:100%;border-collapse:collapse;margin:10px 0;font-size:13px;">
    <tr style="background:#E3F2FD;">
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Modo</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Descripción</th>
        <th style="padding:8px;text-align:left;border:1px solid #90CAF9;">Uso</th>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>todos</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Usa todos los vecinos que pasen los filtros</td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Datasets pequeños</td>
    </tr>
    <tr style="background:#F5F5F5;">
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>top_k</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Solo los k más similares</td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Datasets grandes</td>
    </tr>
    <tr>
        <td style="padding:8px;border:1px solid #E0E0E0;"><b>k_en_rango</b></td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Los k mejores dentro del rango de similitud</td>
        <td style="padding:8px;border:1px solid #E0E0E0;">Balance calidad/cantidad</td>
    </tr>
    </table>
    
    <b>⚙️ Parámetros adicionales:</b><br>
    • <b>top_k:</b> Cantidad máxima de vecinos cuando modo = "top_k"<br>
    • <b>enforce_k_min:</b> Si se activa, rechaza la imputación si hay menos de k_min vecinos<br><br>
    
    <b>🎓 Principio:</b> "todos" maximiza información; "top_k" prioriza calidad sobre cantidad.
    """,
    "confianza_sim": """
    <div style="background:#E3F2FD;border-left:4px solid #1976D2;padding:12px;margin-bottom:8px;">
    <b style="color:#0D47A1;font-size:14px;">📊 Confianza en similitud</b>
    </div>
    
    <b>🎯 ¿Para qué sirve?</b><br>
    Configura cómo se calcula la confianza de una imputación por similitud.<br><br>
    
    <b>📊 cv_ref (coeficiente de variación de referencia):</b><br>
    Normaliza la dispersión de los valores de los vecinos.<br>
    <code>componente_cv = max(0, 1 - CV / cv_ref)</code><br><br>
    
    • CV < cv_ref → componente positivo (vecinos concordantes)<br>
    • CV = cv_ref → componente = 0 (punto neutro)<br>
    • CV > cv_ref → se clampea a 0<br><br>
    
    <b>💡 Ejemplo:</b><br>
    cv_ref = 0.5. Vecinos con valores [100, 110, 105] → CV ≈ 0.05<br>
    componente_cv = 1 - 0.05/0.5 = 0.90 → Muy concordantes, alta confianza.<br><br>
    
    <b>📉 Penalización por k:</b><br>
    Misma lógica que en correlación: reduce confianza cuando hay pocos vecinos.<br>
    <code>factor = a5×(k/2)⁵ + ... + a0</code><br><br>
    
    <b>⚙️ Valor por defecto cv_ref:</b> 0.5<br>
    <b>🎓 Principio:</b> Una imputación es confiable si los vecinos son similares al objetivo Y concordantes entre sí.
    """,
    "umbral_por_familia": """
    <div style="background:#FFF3E0;border-left:4px solid #F57C00;padding:12px;margin-bottom:8px;">
    <b style="color:#E65100;font-size:14px;">🎚️ Umbral por familia</b>
    </div>
    
    <b>🎯 ¿Para qué sirve?</b><br>
    Permite definir un umbral de diferencia diferente para cada familia de parámetros,
    en lugar de usar el umbral global.<br><br>
    
    <b>📊 Valores posibles:</b><br>
    • <b>Vacío / None:</b> Usa el umbral global (umbral_pct_diferencia)<br>
    • <b>0.10:</b> 10% de tolerancia para esta familia<br>
    • <b>0.30:</b> 30% de tolerancia para esta familia<br><br>
    
    <b>💡 Ejemplo:</b><br>
    Umbral global = 0.20.<br>
    • fisica: 0.15 → Más estricto para peso, empuje<br>
    • geometrica: None → Usa el global (0.20)<br>
    • prestacional: 0.25 → Más permisivo para velocidad, alcance<br><br>
    
    <b>⚠️ Formato:</b> Usar fracción (0-1), no porcentaje. Vacío = usa global.<br>
    <b>🎓 Principio:</b> Permite ser más estricto en familias críticas y más permisivo en familias secundarias.
    """,
    "outliers_sim": """
    <div style="background:#FFEBEE;border-left:4px solid #D32F2F;padding:12px;margin-bottom:8px;">
    <b style="color:#C62828;font-size:14px;">🔍 Outliers en vecinos de similitud</b>
    </div>
    
    <b>🎯 ¿Para qué sirve?</b><br>
    Detecta y maneja vecinos cuyos valores son atípicos respecto al grupo de vecinos encontrados.<br>
    Si un vecino tiene un valor muy diferente al resto, se le reduce el peso o se elimina.<br><br>
    
    <b>📊 Sistema de z-scores:</b><br>
    Para cada vecino, calcula cuántas desviaciones estándar se aleja de la media del grupo:<br>
    <code>z = |valor - media_vecinos| / std_vecinos</code><br><br>
    
    <b>⚙️ Parámetros:</b><br>
    • <b>Usar outliers:</b> Activar/desactivar detección<br>
    • <b>z_suave:</b> Umbral para penalizar (default 3.0) → peso reducido<br>
    • <b>z_duro:</b> Umbral para eliminar (default 6.0) → peso = 0<br>
    • <b>alpha:</b> Intensidad de la penalización suave (0-1)<br>
    • <b>w_min:</b> Peso mínimo asignado a outliers suaves<br>
    • <b>remover_duro:</b> Si TRUE, elimina completamente outliers duros<br><br>
    
    <b>💡 Ejemplo:</b><br>
    5 vecinos con alcance: [2000, 2100, 2050, 1950, 5000] km<br>
    Media ≈ 2620, std ≈ 1250. El vecino con 5000 tiene z ≈ 1.9.<br>
    Con z_suave=3.0 no se penaliza. Si fuera 7000 km (z > 3), se penalizaría.<br><br>
    
    <b>⚠️ Recomendación:</b> Desactivar (default) a menos que sepas que hay datos erróneos en el dataset.<br>
    <b>🎓 Principio:</b> Protege la imputación de vecinos con valores extremos que distorsionarían el promedio.
    """,
}


# Función para obtener un texto de ayuda
def get_help_text(category, key):
    """
    Obtiene el texto de ayuda para una categoría y key dados.

    Args:
        category: str - Categoría del texto (ej: "CHECKS_2D", "DIVERSIDAD")
        key: str - Key del texto dentro de la categoría

    Returns:
        str - Texto HTML de ayuda, o None si no existe
    """
    category_dict = globals().get(category)
    if category_dict is None:
        return None
    return category_dict.get(key)
