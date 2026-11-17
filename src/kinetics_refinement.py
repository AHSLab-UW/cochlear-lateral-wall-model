"""Refine reference Tier 1 drug calibration on a finer observation grid."""

import hashlib
import json

import numpy as np
from scipy.optimize import least_squares

from src.drug_calibration import LOWER, NAMES, OUTPUT as PREVIOUS, UPPER, evaluate
from src.pike_trace_diagnostic import ROOT, load_anchors


OUTPUT = PREVIOUS.parent / "kinetics_refinement.json"


def main():
    raw = PREVIOUS.read_bytes()
    previous = json.loads(raw)
    if previous["status"] != "completed exploratory tier 1 drug calibration":
        raise ValueError("completed coarse-grid fit required")
    attempts = [a for a in previous["attempts"] if a["success"]]
    if len(attempts) < 2:
        raise ValueError("at least two converged coarse-grid starts required")
    ordered = sorted(attempts, key=lambda a: a["optimization_grid_score"])
    starts = [np.array([a["parameters"][name] for name in NAMES]) for a in (ordered[0], ordered[-1])]
    records = load_anchors()
    report = {
        "status": "in progress",
        "source": str(PREVIOUS.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "optimization_grid_min": .1,
        "limitations": previous["limitations"] + [
            "The 0.1-min grid is still discrete; the nadir objective is nonsmooth and optimization is local.",
            "Both starts come from the earlier outcome-informed coarse-grid fits and need not explore other basins.",
            "Sampling refinements are post-fit reevaluations, not verification of the continuous optimum.",
        ],
        "attempts": [],
    }
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    completed = []
    for start in starts:
        try:
            residual = lambda x: evaluate(x, records, step=.1)[0]
            initial = residual(start)
            opt = least_squares(residual, start, bounds=(LOWER, UPPER), max_nfev=80,
                                diff_step=1e-3, ftol=1e-6, xtol=1e-6, gtol=1e-6)
            attempt = {
                "start": dict(zip(NAMES, map(float, start))), "success": bool(opt.success),
                "message": str(opt.message), "evaluations": int(opt.nfev),
                "initial_score": float(initial @ initial), "optimization_score": float(opt.fun @ opt.fun),
                "parameters": dict(zip(NAMES, map(float, opt.x))),
                "active_bounds": dict(zip(NAMES, map(int, opt.active_mask))),
            }
            if opt.success:
                completed.append(opt)
        except (ValueError, RuntimeError, FloatingPointError, OverflowError) as error:
            attempt = {"start": dict(zip(NAMES, map(float, start))), "success": False,
                       "error": f"{type(error).__name__}: {error}"}
        report["attempts"].append(attempt)
        OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"Finished refinement: {attempt}", flush=True)
    if not completed:
        report["status"] = "no converged refinement"
    else:
        best = min(completed, key=lambda x: x.cost)
        report["best_parameters"] = dict(zip(NAMES, map(float, best.x)))
        report["checks"] = []
        for method, step in (("Radau", .1), ("BDF", .1), ("Radau", .05), ("Radau", .025)):
            r, pred = evaluate(best.x, records, step=step, method=method, rtol=1e-8, atol=1e-10)
            report["checks"].append({"method": method, "step_min": step,
                                     "score": float(r @ r), "predictions": pred})
        report["status"] = "completed exploratory fine-grid tier 1 drug calibration"
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
