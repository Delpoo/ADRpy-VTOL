import glob
import json
import os
import sys

# Ensure we can import as the notebook does: `from Modulos...`
ANALISIS_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(ANALISIS_DIR, ".."))

# Order matters: project root first (for `asistente_diseno`, etc), then analisis/ (for `Modulos`).
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, ANALISIS_DIR)

from Modulos.imputation_loop import (
    _cargar_configuracion,
    ejecutar_pipeline,
)  # noqa: E402


def main() -> None:
    cfg = _cargar_configuracion()
    cfg.setdefault("orquestacion", {})["mostrar_consola"] = False

    ejecutar_pipeline(cfg)

    results_dir = os.path.abspath(os.path.join(ANALISIS_DIR, "Results"))
    audits = sorted(glob.glob(os.path.join(results_dir, "audit_evaluacion_*.json")))
    print("results_dir=", results_dir)
    print("audits_total=", len(audits))

    if not audits:
        print("last_audit=None")
        return

    last = audits[-1]
    print("last_audit=", os.path.basename(last))

    with open(last, "r", encoding="utf-8") as f:
        payload = json.load(f)

    print("initial.missing_cells=", payload.get("initial", {}).get("missing_cells"))

    iters = payload.get("iterations", [])
    if iters:
        counts = (iters[0].get("corr_debug", {}) or {}).get("counts")
        print("iter0.corr_counts=", counts)
        print("iter0.reporte_similitud=", iters[0].get("reporte_similitud"))
        print("iter0.reporte_correlacion=", iters[0].get("reporte_correlacion"))

    print("final=", payload.get("final"))


if __name__ == "__main__":
    main()
