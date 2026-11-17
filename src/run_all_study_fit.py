"""Fit all six studies with the documented Salt figure trace residuals."""

import json

import numpy as np

from src.all_study_fit import (
    PARAMETERS, ROOT, SolverSettings, evaluate_all, fit_joint,
)
from src.salt_trace_fit_residuals import evaluate_salt_traces
from src.wang_noise_traces import load_noise
from src.pike_trace_diagnostic import load_anchors
from src.salt_1979_digitize import load_evidence


OUTPUT = ROOT / "results/all_six_studies_joint_fit.json"


def main():
    evidence = load_evidence()
    report = fit_joint(salt_trace_callback=evaluate_salt_traces, salt_trace_data=evidence)
    if not report["optimizer"]["success"]:
        raise RuntimeError(f"joint optimizer did not converge: {report['optimizer']['message']}")
    values = np.asarray([report["parameters"][name] for name in PARAMETERS])
    checks = []
    for settings in (
            SolverSettings(rtol=1e-8, atol=1e-10),
            SolverSettings(method="BDF", rtol=1e-8, atol=1e-10),
            SolverSettings(step=.025, rtol=1e-8, atol=1e-10)):
        residual, check = evaluate_all(
            values, load_anchors(), load_noise(), salt_trace_callback=evaluate_salt_traces,
            salt_trace_data=evidence, solver=settings,
        )
        checks.append({
            "solver": check["solver"], "weighted_sum_squares": float(residual @ residual),
            "components": check["components"],
        })
    report["numerical_checks"] = checks
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {OUTPUT}; score {report['optimizer']['final_optimizer_weighted_sum_squares']:.6f}")


if __name__ == "__main__":
    main()
