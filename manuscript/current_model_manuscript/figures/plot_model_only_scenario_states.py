#!/usr/bin/env python3
"""Plot dense model-only sound-scenario time traces (Figure 3)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(OUT))

from rebuild_all_figures import build_figure3, configure_style

SCENARIOS = ROOT / "results/noise_time_traces_20261116.json"
FREEZE = ROOT / "results/parameter_freeze.json"


def load_data():
    cached = OUT / "figure3-model-only-scenario-states-figure-data.json"
    try:
        if SCENARIOS.exists() and FREEZE.exists():
            scenario_bytes = SCENARIOS.read_bytes()
            freeze_bytes = FREEZE.read_bytes()
            result = json.loads(scenario_bytes)
            plot_data = []
            for item in result["protocols"]:
                level = int(item["input"]["broadband_spl_db"])
                trace = item["time_trace"]
                plot_data.append({
                    "level_db_spl": level,
                    "time_min": trace["time_min"],
                    "ep_mV": trace["ep_mV"],
                    "marginal_k_mM": trace["marginal_k_mM"],
                    "endolymph_k_mM": trace["endolymph_k_mM"],
                    "trauma_state": trace["trauma_state"],
                })
            metadata = {
                "status": result["status"],
                "held_out_comparison_eligible": result["held_out_comparison_eligible"],
                "outcome_text_exposure_disclosed": True,
                "times_are_dense_solver_samples": True,
                "exposure_min": 20,
                "recovery_min": 100,
                "solver": result["solver"],
                "limitations": result["limitations"],
                "source_sha256": {
                    "scenario_result": hashlib.sha256(scenario_bytes).hexdigest(),
                    "candidate_freeze": hashlib.sha256(freeze_bytes).hexdigest(),
                    "model_source": result["model_sha256"],
                },
                "scenarios": plot_data,
            }
            cached.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            return metadata
    except Exception:
        pass
    return json.loads(cached.read_text(encoding="utf-8"))


def main():
    configure_style()
    data = load_data()
    build_figure3(data)
    print("Built Figure 3 (figure3-model-only-scenario-states) in figures/.")


if __name__ == "__main__":
    main()
