import sys
import os

# Ensure repository root is on the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def test_mejor_copy():
    candidatos = [
        {"Confianza_cv": 0.8, "MAPE_cv": 3},
        {"Confianza_cv": 0.7, "MAPE_cv": 2},
    ]
    candidatos.sort(key=lambda x: (-x["Confianza_cv"], x["MAPE_cv"]))
    mejor = candidatos[0].copy()
    mejor["warning"] = "changed"
    assert "warning" not in candidatos[0]
