#!/usr/bin/env python3
"""Rebuild manuscript Figures 1 and 2 (and Graphical Abstract) from frozen fit and diagnostic data."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(OUT))
sys.path.insert(0, str(ROOT))

from rebuild_all_figures import (
    BLUE, GREEN, INK, LIGHT, ORANGE, configure_style,
    build_figure1, build_figure2, build_graphical_abstract,
)


def main():
    configure_style()
    # If run from the full repository root with src/ and results/ accessible, refresh JSON caches first
    try:
        from src.parameter_freeze import FIT, FREEZE, load_freeze
        from src.plot_calibration import build_final_data
        stability_path = ROOT / "results/six_study_candidate_finite_stability_20261116.json"
        forecast_path = ROOT / "results/six_study_model_only_forecast_20261116.json"
        if FIT.exists() and FREEZE.exists() and stability_path.exists() and forecast_path.exists():
            fit_bytes = FIT.read_bytes()
            fit_report = json.loads(fit_bytes)
            parameters, freeze = load_freeze()
            fit_data = build_final_data(fit_report, parameters, freeze)
            stability_bytes = stability_path.read_bytes()
            forecast_bytes = forecast_path.read_bytes()
            stability = json.loads(stability_bytes)
            forecast = json.loads(forecast_bytes)
            (OUT / "tier1-figure-data.json").write_text(
                json.dumps(fit_data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            (OUT / "diagnostics-figure-data.json").write_text(
                json.dumps({"stability": stability, "forecast": forecast}, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            source_hashes = {
                "fit_report_sha256": hashlib.sha256(fit_bytes).hexdigest(),
                "candidate_freeze_sha256": hashlib.sha256(FREEZE.read_bytes()).hexdigest(),
                "model_source_sha256": freeze["model_sha256"],
                "stability_report_sha256": hashlib.sha256(stability_bytes).hexdigest(),
                "forecast_report_sha256": hashlib.sha256(forecast_bytes).hexdigest(),
            }
            (OUT / "figure-source-hashes.json").write_text(
                json.dumps(source_hashes, indent=2) + "\n", encoding="utf-8"
            )
    except Exception:
        pass

    build_figure1()
    build_figure2()
    build_graphical_abstract()
    print("Built Figure 1, Figure 2, and Graphical Abstract in figures/.")


if __name__ == "__main__":
    main()
