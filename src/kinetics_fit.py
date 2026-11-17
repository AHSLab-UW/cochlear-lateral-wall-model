"""Explore whether flexible drug-proxy decay improves the reference Tier 1 fit."""

from dataclasses import replace
import hashlib
import json

import numpy as np
from scipy.optimize import least_squares

from src.model_equations import Parameters
from src.drug_calibration import LOWER, NAMES, UPPER, evaluate
from src.kinetics_refinement import OUTPUT as PREVIOUS
from src.pike_trace_diagnostic import ROOT, load_anchors


OUTPUT = PREVIOUS.parent / "kinetics_fit.json"
ALL_NAMES = (*NAMES, "drug_slow_fraction", "drug_slow_min")
BOUNDS_LOW = np.r_[LOWER, .05, 10.]
BOUNDS_HIGH = np.r_[UPPER, .9, 200.]


def residual(values, records, *, step=.1, method="Radau", rtol=2e-6, atol=2e-8):
    x = np.asarray(values, dtype=float)
    if x.shape != (6,) or not np.isfinite(x).all():
        raise ValueError("six finite parameters are required")
    if np.any(x < BOUNDS_LOW) or np.any(x > BOUNDS_HIGH):
        raise ValueError("parameters outside declared numerical bounds")
    p = replace(Parameters(), drug_slow_fraction=float(x[4]), drug_slow=float(x[5]))
    return evaluate(x[:4], records, step=step, method=method, rtol=rtol, atol=atol,
                    base_parameters=p)


def main():
    raw = PREVIOUS.read_bytes()
    previous = json.loads(raw)
    if previous["status"] != "completed exploratory fine-grid tier 1 drug calibration":
        raise ValueError("completed refined fit required")
    x0 = [previous["best_parameters"][name] for name in NAMES]
    starts = [np.array([*x0, .27, 48.]), np.array([*x0, .5, 90.]),
              np.array([*x0, .15, 150.])]
    records = load_anchors()
    report = {
        "status": "in progress",
        "source": str(PREVIOUS.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "parameter_bounds": dict(zip(ALL_NAMES, [[float(a), float(b)]
                                                  for a, b in zip(BOUNDS_LOW, BOUNDS_HIGH)])),
        "optimization_grid_min": .1,
        "limitations": previous["limitations"] + [
            "The two drug decay parameters are outcome-informed proxy-shape parameters, not measured pharmacokinetics.",
            "New bounds are illustrative numerical limits, not biological confidence bounds.",
            "Three starts share the previously fitted four parameters and do not establish a global optimum.",
            "The objective is unchanged and remains a weighted diagnostic, not a likelihood.",
        ],
        "attempts": [],
    }

    def save():
        OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    complete = []
    for start in starts:
        try:
            opt = least_squares(lambda x: residual(x, records)[0], start,
                                bounds=(BOUNDS_LOW, BOUNDS_HIGH), max_nfev=100,
                                diff_step=1e-3, ftol=1e-6, xtol=1e-6, gtol=1e-6)
            attempt = {"start": dict(zip(ALL_NAMES, map(float, start))),
                       "success": bool(opt.success), "message": str(opt.message),
                       "evaluations": int(opt.nfev),
                       "score": float(opt.fun @ opt.fun),
                       "parameters": dict(zip(ALL_NAMES, map(float, opt.x))),
                       "active_bounds": dict(zip(ALL_NAMES, map(int, opt.active_mask)))}
            if opt.success:
                complete.append(opt)
        except (ValueError, RuntimeError, FloatingPointError, OverflowError) as error:
            attempt = {"start": dict(zip(ALL_NAMES, map(float, start))),
                       "success": False, "error": f"{type(error).__name__}: {error}"}
        report["attempts"].append(attempt)
        save()
        print(f"Finished {attempt}", flush=True)
    if complete:
        best = min(complete, key=lambda x: x.cost)
        report["best_parameters"] = dict(zip(ALL_NAMES, map(float, best.x)))
        report["checks"] = []
        for method, step in (("Radau", .1), ("BDF", .1), ("Radau", .05), ("Radau", .025)):
            r, pred = residual(best.x, records, step=step, method=method, rtol=1e-8, atol=1e-10)
            report["checks"].append({"method": method, "step_min": step,
                                     "score": float(r @ r), "predictions": pred})
        report["status"] = "completed exploratory six-parameter tier 1 drug calibration"
    else:
        report["status"] = "no converged start"
    save()
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
