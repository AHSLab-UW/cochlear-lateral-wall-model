# Cochlear lateral-wall effective-circuit model

This repository contains the cochlear lateral-wall model implementation,
calibration and diagnostic scripts, selected digitized evidence, reproducibility
outputs, and the associated figure and table folders. Manuscript source, the
compiled manuscript PDF, and submission highlights are not distributed here.
Standalone fit and plotting workflows outside the current analysis pipeline are
excluded.

## Reproducibility

The validated environment uses Python 3.11.8. Create it from the pinned
environment specification in [`environment.yml`](environment.yml) with:

```bash
conda env create -f environment.yml
conda activate cochlear-lateral-wall-model
```

The environment specification pins Python, the runtime packages, and the test
runner; platform-specific transitive dependencies are resolved by Conda. The
pip runtime dependencies are also pinned in
[`requirements.txt`](requirements.txt). From the repository root, run the
focused tests with:

```bash
python -m pytest -q
```

Recompute the limited weight/grid sensitivity diagnostics with:

```bash
python -m src.sensitivity_diagnostics
```

Rebuild the sampled-domain audit after placing the two source PDFs named below
in the ignored local `refs/target_source/` directory:

```bash
python -m src.domain_audit
```

The fitting, domain-audit, sensitivity, and figure-building entry points are
[`src/fit_parameters.py`](src/fit_parameters.py),
[`src/domain_audit.py`](src/domain_audit.py),
[`src/sensitivity_diagnostics.py`](src/sensitivity_diagnostics.py), and the scripts in
[`manuscript/current_model_manuscript/figures/`](manuscript/current_model_manuscript/figures/).
Run Python modules from the repository root so package imports and relative
data paths resolve correctly. Rebuild the manuscript figures from the repository
root with:

```bash
python manuscript/current_model_manuscript/figures/rebuild_all_figures.py
```

The sensitivity diagnostic refits after doubling the acoustic residual weight
relative to the pharmacological components and compares endpoint residuals and
refitted parameters on a 0.025-min output grid. These are limited local
sensitivity checks, not uncertainty estimates. The domain audit reports
per-call initial/final samples and sampled ranges for each state and derived
observable, together with the simulation module, call order, time range, and
solver and forcing profile. The CSV also identifies the study, protocol, and
solver/grid variant, separating the Zhang and Hou benchmark runs. Its flat output is
[`results/protocol_domain_summary.csv`](results/protocol_domain_summary.csv).
Neither audit includes every adaptive solver step.

The Zhang et al. (2021) retrospective benchmark and full domain audit also
require the Zhang and Hou source PDFs at
`refs/target_source/zhang_2021_j_physiol.pdf` and
`refs/target_source/hou_2020_am_j_pathol.pdf`. Source publications are not
redistributed here; provide lawfully accessible copies locally before running
the audit.

Digitized Salt and Konishi (1979) observations and source-page hash metadata
are included for reproducibility. Rasterized source pages and figure crops are
not redistributed. When those images are available locally, the code verifies
their hashes; the publication package validates the recorded hashes against
the digitization metadata. The anatomical overview's full-size vector version
is available as
[`anatomical-infographic.svg`](manuscript/current_model_manuscript/figures/anatomical-infographic.svg)
for enlarged inspection.

The fit is outcome-informed and is not independent validation. The reported
domain audit evaluates requested solver output samples, not every internal
adaptive step or a continuous-time invariance proof. The corresponding
manuscript and audit output describe these scope and limitations.

## Release status

The public GitHub repository is available under AHSLab-UW. The
`r11-revision` tag identifies the model and analysis source snapshot for the
R11 revision; manuscript text, PDF, and highlights are not included. GitHub is
the public source repository, not a DOI-backed archive; no archival DOI has
been minted, and no software license has been assigned.
Until a license is added, reuse is not granted. Funding and acknowledgement
declarations also require author confirmation before manuscript submission.
The source-publication PDFs and raster page images are not included; users must
obtain source publications lawfully. Digitized observations are provided with
citations and source-page hash metadata.
