"""Joint exploratory drug and noise fit using the existing nine parameters."""

import hashlib
import json

import numpy as np
from scipy.optimize import least_squares

from src.drug_calibration import evaluate
from src.kinetics_fit import ALL_NAMES, BOUNDS_HIGH, BOUNDS_LOW
from src.wang_noise_traces import (
    LOWER, NAMES, OUTPUT as SEQUENTIAL, PAPER, UPPER, evaluate_noise, load_noise,
    make_parameters,
)
from src.pike_trace_diagnostic import ROOT, SOURCE, load_anchors


OUTPUT = SEQUENTIAL.parent / "parameter_schema.json"
PARAMETERS = (*ALL_NAMES, *NAMES)
LOW = np.r_[BOUNDS_LOW, LOWER]
HIGH = np.r_[BOUNDS_HIGH, UPPER]


def evaluate_joint(values, drug_records, noise_records, *, step=.1, method="Radau",
                   rtol=2e-6, atol=2e-8):
    x = np.asarray(values, dtype=float)
    if x.shape != (9,) or not np.isfinite(x).all() or np.any(x < LOW) or np.any(x > HIGH):
        raise ValueError("nine finite parameters within declared bounds are required")
    drug, sound = x[:6], x[6:]
    p = make_parameters(sound, drug)
    dr, dp = evaluate(drug[:4], drug_records, step=step, method=method,
                      rtol=rtol, atol=atol, base_parameters=p)
    nr, npred = evaluate_noise(sound, drug, noise_records, method=method,
                               rtol=rtol, atol=atol)
    return np.concatenate((dr, nr)), {
        "drug_score": float(dr @ dr), "noise_score": float(nr @ nr),
        "drug_predictions": dp, "noise_predictions": npred,
    }


def main():
    raw = SEQUENTIAL.read_bytes()
    previous = json.loads(raw)
    if previous["status"] != "completed exploratory sequential noise calibration":
        raise ValueError("completed sequential fit required")
    x0 = np.array([previous["drug_parameters_frozen"][n] for n in ALL_NAMES]
                  + [previous["sound_parameters"][n] for n in NAMES])
    drug_records, noise_records = load_anchors(), load_noise()
    starts = (("sequential", x0),
              ("interior_perturbation", np.clip(x0 * np.array(
                  [1.03, .97, .99, 1.03, .97, 1.03, 1.03, .97, 1.10]), LOW, HIGH)))
    attempts, completed = [], []
    for label, start in starts:
        initial, _ = evaluate_joint(start, drug_records, noise_records)
        print(f"Starting {label} joint score {initial @ initial:.6f}", flush=True)
        opt = least_squares(lambda x: evaluate_joint(x, drug_records, noise_records)[0], start,
                            bounds=(LOW, HIGH), method="dogbox", x_scale="jac",
                            max_nfev=100, diff_step=1e-3,
                            ftol=1e-6, xtol=1e-6, gtol=1e-6)
        attempts.append({"start": label, "success": bool(opt.success),
                         "message": str(opt.message), "evaluations": int(opt.nfev),
                         "initial_score": float(initial @ initial), "score": float(opt.fun @ opt.fun),
                         "optimality": float(opt.optimality),
                         "parameters": dict(zip(PARAMETERS, map(float, opt.x)))})
        print(f"Finished {label}: {attempts[-1]}", flush=True)
        if opt.success:
            completed.append(opt)
    if not completed:
        raise RuntimeError(f"No joint optimization converged: {attempts}")
    opt = min(completed, key=lambda fit: fit.cost)
    report = {
        "status": "exploratory joint optimization attempt; optimality not established",
        "input_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (SEQUENTIAL, SOURCE, PAPER)},
        "parameters": dict(zip(PARAMETERS, map(float, opt.x))),
        "bounds": dict(zip(PARAMETERS, [[float(a), float(b)] for a, b in zip(LOW, HIGH)])),
        "active_bounds": dict(zip(PARAMETERS, map(int, opt.active_mask))),
        "attempts": attempts,
        "limitations": [
            "The objective adds the previously specified drug and noise squared residuals without additional rescaling; their relative weights are choices, not a likelihood or evidence of precision.",
            "Fits only the previously implemented Tier 1 drug and Wang 1992 noise targets; other Tier 1 observations are not fitted.",
            "Two nearby starting points and numerical parameter bounds cannot establish a global optimum or identify parameters; local termination alone is not evidence of an optimum.",
            "The best attempt stopped by step tolerance without a meaningful reduction from the sequential start; a nonzero reported optimality and finite-difference/grid effects leave even local optimality unestablished.",
            "The nine parameters are already present in prior fits; this does not add a new mechanism or exposure parameter.",
            "The source traces are correlated, digitized group means; species and exposure mappings, phenomenological K boundary and hybrid closure are unverified assumptions.",
            "The 125 dB noise mechanism represents severe injury conditionally, not a validated injury model.",
            "Extrema are sampled on a finite grid; state-range checks are not a proof of invariance, physiological validity, or stability.",
        ],
        "checks": [],
    }
    for label, x, method, step in (
            ("sequential_initial", x0, "Radau", .025),
            ("joint", opt.x, "Radau", .1),
            ("joint", opt.x, "BDF", .1),
            ("joint", opt.x, "Radau", .025)):
        r, predictions = evaluate_joint(x, drug_records, noise_records, step=step,
                                        method=method, rtol=1e-8, atol=1e-10)
        report["checks"].append({"candidate": label, "method": method, "step_min": step,
                                 "score": float(r @ r), **predictions})
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Finished {report['status']}: {opt.message}; wrote {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
