# Portfolio architecture visual

- [`architecture.svg`](architecture.svg): the English overview embedded in the root README.
- [`architecture.drawio`](architecture.drawio): editable diagrams.net source for the same overview.

The primary reference is *Diagrama integral — Tesis ADRpy-VTOL v4.drawio*, reviewed in `docs/Diagramas/`; a tracked copy is preserved under `ADRpy/Diagramas/`. The original thesis diagrams are unchanged. Existing untracked diagrams remain reference material and are not required to render the portfolio overview.

The simplified view follows the implemented three-module structure and user workflow. It deliberately places subsequent ADRpy execution outside the ADRpy-VTOL software boundary. Arrows describe the working process, not an automatic software connection: model inspection informs user decisions, while the design assistant reads an Excel dataset. The final handoff requires user selection, input mapping, unit checks and assumptions.

Broader thesis claims about automatic ADRpy execution, fully validated confidence, MAE and complete traceability are not reproduced. Unrelated import / logistics diagrams are excluded from portfolio navigation. See the [technical guide](../../ADRpy/analisis/README.md) for the implementation limits behind these choices.
