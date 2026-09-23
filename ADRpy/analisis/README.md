# ADRpy-VTOL technical guide

This guide covers Ezequiel Delpino's project-specific analysis and design-assistance code. For the problem, authorship and ADRpy relationship, start with the [repository overview](../../README.md). The implementation, rather than the broader thesis architecture, defines the capabilities described here.

## Repository structure

```text
ADRpy/
  analisis/
    Data/                       UAV market workbook
    Modulos/                    Data preparation and imputation
      Analisis_modelos/         Model-analysis dashboard
    Results/                    Saved workbooks, model JSON and audit records
    exports/                    Historical visualization exports
    salidas/                    Configuration snapshots
    config_overrides.json       Predictor configuration overrides
    run_pipeline_once.py        Script entry point for the predictor
    launch_app.py               Dashboard launcher
    main.py                     Older interactive predictor entry point
  asistente_diseno/             Constraints, ranking, trends and suggestions
  notebooks/                   Three project interface notebooks
docs/
  portfolio/                   Simplified architecture diagram and source
  thesis/                      Academic documents and context
  upstream/                    Preserved original ADRpy README
  ADRpy/notebooks/              ADRpy examples and separate thesis case study
```

Existing Spanish file names and dataset columns are preserved. The original ADRpy library shares the `ADRpy/` package directory; it is not part of the project-specific imputation implementation.

## Three modules and entry points

| Module | Entry points | Responsibilities |
| --- | --- | --- |
| **1. Predictor / imputation** | [`1- motor_predictor.ipynb`](../notebooks/1-%20motor_predictor.ipynb), [`run_pipeline_once.py`](run_pipeline_once.py) | Load and prepare tabular data; propose estimates through similarity and regression; consolidate estimates over iterations; export results. [`controller.py`](Modulos/controller.py) manages configuration and execution; [`imputation_loop.py`](Modulos/imputation_loop.py) orchestrates the loop. |
| **2. Model analysis and visualization** | [`2- analisis_modelos_motor_predictor.ipynb`](../notebooks/2-%20analisis_modelos_motor_predictor.ipynb), [`launch_app.py`](launch_app.py) | Inspect saved models through Dash / Plotly plots and metrics. Main implementation: [`main_visualizacion_modelos.py`](Modulos/Analisis_modelos/main_visualizacion_modelos.py). |
| **3. Conceptual design assistant** | [`3- asistente_diseno.ipynb`](../notebooks/3-%20asistente_diseno.ipynb), [`asistente_diseno/main.py`](../asistente_diseno/main.py) | Use an ipywidgets interface to define target values and constraints, rank comparable aircraft, explore trends, obtain suggestions and save reports. |

These are separate tools. Module 2 reads model JSON; module 3 reads an Excel dataset, rather than consuming a validated result from module 2. Inspection informs the user's choices. The older [`main.py`](main.py) is retained as a separate workflow; it should not be assumed equivalent to the controller-based entry point.

## Inputs, outputs and traceability

| Stage | Input | Output / inspection material |
| --- | --- | --- |
| Predictor | [`Data/Datos_aeronaves.xlsx`](Data/Datos_aeronaves.xlsx), configuration defaults and [`config_overrides.json`](config_overrides.json) | `Results/Datos_imputados.xlsx`, cell comments and reports, `modelos_completos_por_celda*.json`, timestamped `audit_evaluacion_*.json`, and configuration snapshots. Missing cells can remain. |
| Model dashboard | A selected model JSON file | Interactive fitted curves or surfaces, training samples, metrics and warnings. The launcher defaults to `Results/modelos_completos_por_celda.json`, not necessarily the latest numbered output. |
| Design assistant | An Excel dataset, user constraints, selected parameters and weights | Aircraft rankings, parameter suggestions, session configuration in JSON / Excel, and `informe_diseno.md`, `.html` and `.xlsx` reports with supporting tables. |
| Separate ADRpy case study | Manually populated design brief, aircraft definition, performance assumptions and atmosphere | Conceptual constraint analysis in [`ADRpy_Tesis_Delpino.ipynb`](../../docs/ADRpy/notebooks/ADRpy_Tesis_Delpino.ipynb). |

Traceability is available at the level of individual estimates and models, with different detail in Excel and JSON. Model records can contain coefficients, transformations, training data, metrics and warnings; Excel comments also include imputation context. The main model JSON is built from correlation-model records and does not cover all similarity-only estimates or retain every intermediate diagnostic. There is no single manifest tying every saved artifact to one run, nor complete confidence propagation into design-assistant suggestions.

Tracked results and notebook outputs are **historical examples**, not evidence of a fresh execution or a single consistent experiment. A dashboard coverage percentage refers to its loaded JSON population, not necessarily all missing cells in the source workbook.

## Statistical methods

| Method | Implementation and scope |
| --- | --- |
| Similarity-based imputation | [`imputacion_similitud_nueva.py`](Modulos/imputacion_similitud_nueva.py): compares shared mass, geometry and performance parameters through percentage differences, selects neighbors and computes weighted estimates. This is distinct from the assistant's distance-based aircraft ranking. |
| One-predictor regression | [`imputacion_correlacion.py`](Modulos/imputacion_correlacion.py): linear, quadratic, logarithmic, exponential and power-law fits, with the corresponding data-domain restrictions. |
| Two-predictor regression | The same module implements linear and quadratic fits. Predictor combinations are limited to one or two variables. |
| Metrics and validation | R² and MAPE; leave-one-out cross-validation routines, sample-size penalties, model-quality filters and heuristic ranking. The limitations below affect interpretation. |
| Outlier handling | [`outlier_utils.py`](Modulos/outlier_utils.py): median / MAD-based scoring, IQR flags, weights and optional exclusion. The assistant also has its own [`outliers.py`](../asistente_diseno/outliers.py). |
| Two-predictor quality checks | Pairwise correlation and VIF, matrix rank and condition number, SVD-based second-component variance, unique sample pairs, convex-hull / bounding-box and covariance-ellipse measures, and sample-to-parameter checks. These assess sample geometry, not aerodynamic validity. |
| Design assistance | [`similitud.py`](../asistente_diseno/similitud.py), [`sugerencias.py`](../asistente_diseno/sugerencias.py) and [`tendencias.py`](../asistente_diseno/tendencias.py): constrained aircraft ranking, neighbor-based suggestions and exploratory trend fits. Trend fits are separate from the predictor's stored models and LOOCV. |

## Setup and execution status

The repository does not yet provide a verified, reproducible environment for all three project modules. Code inspection and historical outputs establish implementation evidence; selected isolated numerical checks were performed during the audit, but the full predictor, dashboard and assistant were not run end to end during this documentation phase.

For a local execution review:

1. Use a separate checkout or copy of the data and saved outputs. Predictor execution writes workbooks, model records and configuration snapshots; assistant report exports also write files.
2. Review the selected notebook and its working-directory / import-path setup. The three project notebooks and [`asistente_diseno/config.py`](../asistente_diseno/config.py) contain author-specific absolute Windows paths that need local adjustment.
3. Prepare the dependencies required by the chosen entry point. The code uses NumPy, pandas, SciPy, scikit-learn, statsmodels and openpyxl, plus Matplotlib, Plotly / Dash and IPython / ipywidgets for interfaces; some utilities also use seaborn and psutil. This is an orientation list, not a tested environment specification.
4. Select the input workbook and intended model JSON explicitly, and review configuration overrides before running a module. Check units and column definitions rather than assuming all historical artifacts are interchangeable.

The root `requirements.txt` primarily serves the original ADRpy repository and is incomplete for these additions. The analysis `requirements.txt` contains uncommented section headings that prevent direct installation as a requirements file; the dashboard's own requirements cover only part of the system. Existing tests also contain an obsolete imputation import path. Dependency, path and test repairs are deferred to the engineering phase. Installing the published `ADRpy` package alone does not install these project additions.

## Known limitations

- **LOOCV is not uniform across model families.** Linear and quadratic models have refitting routines, but the current LOOCV path does not reproduce the logarithmic, exponential and power-law transformations. When LOOCV is disabled, training metrics can be copied into LOOCV-labelled fields. Those fields must not be treated as equivalent independent validation.
- **Confidence is an operational heuristic, not a calibrated probability.** Engine and dashboard aggregation differ. With the current penalties, missing two-dimensional diagnostics can reduce a one-predictor model's confidence to zero. Model ranking and displayed confidence need engineering review.
- **Similarity and geometric checks have known edge cases.** Neighbor acceptance thresholds and scoring domains can disagree; zero total neighbor weight can produce a zero estimate. Two-dimensional SVD / condition diagnostics are sensitive to parameter units and scaling.
- **Extrapolation control is incomplete.** Predictor-range filters and final warnings exist, but some paths warn without rejecting the estimate. Convex-hull checks do not establish physical feasibility.
- **Scope is limited.** There is no implemented MAE metric or regression with three or more predictors. Several exposed configuration options are not consistently consumed by the active engine. Imputation does not guarantee a complete dataset.
- **Inspection is not automatic acceptance.** Being valid for plotting does not mean a model is statistically or physically acceptable. Standalone [`verificacion.py`](../asistente_diseno/verificacion.py) utilities are not an automatic physical-validation stage in the integrated assistant.
- **Outputs need context.** Saved JSON may contain non-standard `NaN` values and schemas vary between historical exports. Artifact names alone do not identify a coherent run. Traceability and confidence are not carried completely across every stage.
- **ADRpy use remains a separate step.** Assistant parameter definitions are not an automatically generated `brief` / `design` / `performance` input vector. The user must map values, reconcile units, supply assumptions and execute ADRpy separately. The case-study notebook uses manually defined inputs and does not model the complete VTOL hover / transition envelope.

These issues are documented, not corrected, in the portfolio cleanup. The [thesis and summary](../../docs/thesis/README.md) describe the academic work and some broader intended architecture; they should not override the implementation scope above.
