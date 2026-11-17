"""Re-evaluate historical fits at modeled EP nadir rather than an assumed 6 min."""

import hashlib
import json
from pathlib import Path

import numpy as np

from src.assumption_comparison import Circuit
from src.cochlear_model import Inputs, Parameters
from src.joint_calibration_screen import RESISTANCE_RATIO, partition_resistance_ratio
from src.joint_mechanism_comparison import make_simulator
from src.low_dose_assumption_sensitivity import GEOMETRY
from src.transport_assumption_calibration import inhibition


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "exploratory"
INPUT = RESULTS / "low_dose_assumption_sensitivity.json"
TARGET = RESULTS / "asakuma_nadir_timing_audit.json"


def evaluate_case(case, method, step, *, rtol=1e-8, atol=1e-10):
    if not np.isfinite(step) or step <= 0 or not np.isclose(120 / step, round(120 / step)):
        raise ValueError("grid step must be positive, finite, and divide 120 minutes")
    p = case["parameters"]
    params = Parameters(potassium_flux=p["potassium_flux"], drug_sink=0,
                        endolymph_volume_coupling=0, water_rate=case["water_rate"],
                        **GEOMETRY)
    circuit = Circuit(True, True, True, p["met_conductance"], p["drug_loss_fraction"])
    forcing = lambda t: Inputs(drug=inhibition(t, 60, p["effective_dose_scale_mg_kg"],
                                               p["exposure_decay_min"], case["hill"]))
    simulator = make_simulator(exchange=case["exchange"], volume_gate=False)
    times = np.linspace(0, 120, int(round(120 / step)) + 1)
    sol = simulator(times, forcing, params, circuit, "saturating",
                    method=method, rtol=rtol, atol=atol)
    if not sol.success or sol.y.shape[1] != len(times) or not np.isfinite(sol.y).all():
        raise RuntimeError(f"{method} failed or returned incomplete nonfinite output")
    nadir = int(np.argmin(sol.y[10]))
    i6 = int(np.argmin(abs(times - 6)))
    if abs(times[i6] - 6) > 1e-10:
        raise ValueError("the grid must include 6 minutes")
    ratios = [partition_resistance_ratio(sol.y[:, i], forcing(times[i]), params,
                                          circuit, series_fraction=0.5)
              for i in (nadir, i6)]
    return {
        "solver": method, "grid_step_min": step,
        "sampled_nadir_time_min": float(times[nadir]),
        "sampled_nadir_ep_mV": float(sol.y[10, nadir]),
        "resistance_ratio_at_sampled_nadir": float(ratios[0]),
        "resistance_ratio_at_assumed_6_min": float(ratios[1]),
        "nadir_at_window_boundary": nadir in (0, len(times) - 1),
    }


def main():
    raw = INPUT.read_bytes()
    data = json.loads(raw)
    cases = []
    for index, case in enumerate(data["cases"]):
        if case["status"] != "converged":
            raise ValueError(f"source case {index} did not converge")
        runs = [evaluate_case(case, method, step)
                for method in ("Radau", "BDF") for step in (0.25, 0.05)]
        cases.append({"source_case_index": index, "hill": case["hill"],
                      "water_rate": case["water_rate"], "exchange": case["exchange"],
                      "parameters": case["parameters"], "runs": runs})
    report = {
        "status": "fixed historical parameter re-evaluation; not a corrected refit",
        "source_file": str(INPUT.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "observed_ratio_of_group_means": RESISTANCE_RATIO,
        "dose_mg_kg": 60,
        "cases": cases,
        "limitations": [
            "The original parameters were fitted using an incorrectly sourced 6-minute resistance mapping and were not refitted.",
            "Source reports resistance at the table-cohort greatest EP decrease, not a numerically specified time.",
            "A single modeled trajectory's nadir-associated resistance need not equal a ratio of group means at individual nadirs.",
            "Baseline series fraction, shared exposure mapping and extrema window are assumptions, not measured quantities.",
            "The arbitrary six-minute comparison is retained only to show impact of the previous mapping.",
            "This is not independent validation, statistical model selection or proof of ion, charge or water closure.",
        ],
    }
    TARGET.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {TARGET}")


if __name__ == "__main__":
    main()
