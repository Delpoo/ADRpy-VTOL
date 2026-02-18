"""
Script para refactorizar ux_notebook_panel.py y extraer los help_html a help_texts.py
"""

import re
from pathlib import Path

# Ruta del archivo
file_path = Path(
    r"c:\Users\delpi\OneDrive\Tesis\ADRpy-VTOL\ADRpy\analisis\Modulos\ux_notebook_panel.py"
)

# Leer contenido
content = file_path.read_text(encoding="utf-8")

# Mapeo de reemplazos para Checks 2D
replacements = {
    # Pearson
    (
        r'row_check\("pearson".*?help_html=""".*?"""',
        'row_check("pearson", "abs_r_max", "|r| máx", 0.90, "float",\n                 tooltip="Correlación de Pearson máxima entre predictores",\n                 help_html=HT.CHECKS_2D["pearson"]',
    ),
    # VIF
    (
        r'row_check\("vif".*?help_html=""".*?"""',
        'row_check("vif", "max", "VIF máx", 10.0, "float",\n                 tooltip="Variance Inflation Factor máximo",\n                 help_html=HT.CHECKS_2D["vif"]',
    ),
    # PC2
    (
        r'row_check\("pc2".*?help_html=""".*?"""',
        'row_check("pc2", "ratio_min", "PC2 ratio mín", 0.03, "float",\n                 tooltip="Ratio mínimo de varianza del segundo componente principal",\n                 help_html=HT.CHECKS_2D["pc2"]',
    ),
    # Rank
    (
        r'row_check\("rank".*?help_html=""".*?"""',
        'row_check("rank", "min", "Rango mínimo", 2, "int",\n                 tooltip="Rango mínimo de la matriz de predictores",\n                 help_html=HT.CHECKS_2D["rank"]',
    ),
    # Cond
    (
        r'row_check\("cond".*?help_html=""".*?"""',
        'row_check("cond", "max", "Condición máx", 1e5, "float",\n                 tooltip="Número de condición máximo de la matriz",\n                 help_html=HT.CHECKS_2D["cond"]',
    ),
    # Coverage unique pair
    (
        r'row_check\(\s*"coverage_unique_pair".*?help_html=""".*?"""\s*\)',
        'row_check(\n            "coverage_unique_pair",\n            "ratio_min",\n            "Cobertura pares únicos mín",\n            0.60,\n            "float",\n            tooltip="Ratio mínimo de pares (x1,x2) únicos",\n            help_html=HT.CHECKS_2D["coverage_unique_pair"]\n        )',
    ),
    # Coverage hull
    (
        r'row_check\(\s*"coverage_hull".*?help_html=""".*?"""\s*\)',
        'row_check(\n            "coverage_hull", "ratio_min", "Cobertura hull mín", 0.15, "float",\n            tooltip="Ratio mínimo entre área del convex hull y bounding box",\n            help_html=HT.CHECKS_2D["coverage_hull"]\n        )',
    ),
    # Coverage ellipse
    (
        r'row_check\(\s*"coverage_ellipse".*?help_html=""".*?"""\s*\)',
        'row_check(\n            "coverage_ellipse", "ratio_min", "Cobertura elipse mín", 0.10, "float",\n            tooltip="Ratio entre área de elipse 1σ y bounding box",\n            help_html=HT.CHECKS_2D["coverage_ellipse"]\n        )',
    ),
    # N per param linear2
    (
        r'row_check_n_per_param\("n_per_param", "linear2_min".*?help_html=""".*?"""\s*\)',
        'row_check_n_per_param("n_per_param", "linear2_min", "n/param (linear-2)", 8, 3, "linear-2 (β0, β1, β2)",\n            help_html=HT.CHECKS_2D["n_per_param_linear2"]\n        )',
    ),
    # N per param poly2
    (
        r'row_check_n_per_param\("n_per_param", "poly2_min".*?help_html=""".*?"""\s*\)',
        'row_check_n_per_param("n_per_param", "poly2_min", "n/param (poly-2)", 10, 6, "poly-2 (β0 + 5 términos)",\n            help_html=HT.CHECKS_2D["n_per_param_poly2"]\n        )',
    ),
    # Agresivo
    (
        r'row_check\("agresivo".*?help_html=""".*?"""',
        'row_check("agresivo", "abs_r_min", "Modo agresivo |r| mín", 0.95, "float",\n                 tooltip="Correlación mínima para activar modo agresivo",\n                 help_html=HT.CHECKS_2D["agresivo"]',
    ),
}

# Aplicar reemplazos usando regex con DOTALL para capturar multilínea
for pattern, replacement in replacements:
    content = re.sub(pattern, replacement, content, flags=re.DOTALL)

# Guardar archivo modificado
file_path.write_text(content, encoding="utf-8")

print("✅ Refactorización completada")
print(f"📝 Archivo modificado: {file_path}")
print("\n⚠️ Ahora necesitas reiniciar el kernel del notebook para cargar los cambios")
