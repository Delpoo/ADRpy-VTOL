# ADRpy package and final-degree project additions

This directory contains the original **ADRpy framework by Andras Sobester and contributors**, alongside **Ezequiel Delpino's ADRpy-VTOL final-degree project additions**.

Start with the [project overview](../README.md) or the [technical guide](analisis/README.md).

| Area | Purpose |
| --- | --- |
| [`analisis/`](analisis/) | UAV data preparation, predictor / imputation engine, and model-analysis dashboard. |
| [`asistente_diseno/`](asistente_diseno/) | Interactive conceptual design assistance: constraints, aircraft ranking, trends, suggestions and reports. |
| [`notebooks/`](notebooks/) | Notebook entry points for the three project modules. |
| [`constraintanalysis.py`](constraintanalysis.py), [`atmospheres.py`](atmospheres.py) and other original library modules | ADRpy conceptual-design and performance-analysis framework. |

The assistant supports parameter selection for subsequent ADRpy use. It does not automatically launch ADRpy or certify the physical validity of suggested values. Standalone verification utilities are separate from the integrated assistant interface.

For the original framework, see the [upstream repository](https://github.com/sobester/ADRpy), [ADRpy documentation](https://adrpy.readthedocs.io/en/latest/), [preserved original README](../docs/upstream/README-ADRpy.md) and unchanged [license](../LICENSE.md).
