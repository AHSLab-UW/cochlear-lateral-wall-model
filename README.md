# Cochlear lateral-wall effective-circuit model

This repository accompanies the *Hearing Research* manuscript in
[`manuscript/current_model_manuscript/`](manuscript/current_model_manuscript/).
It contains the model implementation, calibration and diagnostic scripts,
selected digitized evidence, reproducibility outputs, manuscript source, and figures.

## Reproducibility

Use Python 3 with the dependencies in [`requirements.txt`](requirements.txt).
From the repository root, run the focused tests with:

```bash
python -m pytest -q
```

The fitting, domain-audit, and figure-building entry points are
[`src/fit_parameters.py`](src/fit_parameters.py),
[`src/domain_audit.py`](src/domain_audit.py), and the scripts in
[`manuscript/current_model_manuscript/figures/`](manuscript/current_model_manuscript/figures/).
Run Python modules from the repository root so package imports and relative
data paths resolve correctly. Compile the manuscript with Tectonic from its directory:

```bash
cd manuscript/current_model_manuscript
tectonic -k --keep-logs current_model_manuscript.tex
```

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
the digitization metadata.

The fit is outcome-informed and is not independent validation. The reported
domain audit evaluates requested solver output samples, not every internal
adaptive step or a continuous-time invariance proof. See the manuscript and
the audit output for scope and limitations.

## Release status

The public GitHub repository is being established under AHSLab-UW. No
archival DOI or software license has been configured. The source-publication
PDFs and raster page images are not included; users must obtain source
publications lawfully. The digitized observation data are provided with
citations and source-page hash metadata.
