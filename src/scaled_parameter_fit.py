"""Reassess the joint fit with scaled coordinates and controlled differences."""

import hashlib
import json

import numpy as np
from scipy.optimize import least_squares

from src.parameter_schema import HIGH, LOW, OUTPUT as PREVIOUS, PARAMETERS, evaluate_joint
from src.wang_noise_traces import OUTPUT as SEQUENTIAL, load_noise
from src.pike_trace_diagnostic import ROOT, SOURCE, load_anchors


OUTPUT = PREVIOUS.parent / "scaled_parameter_fit.json"


def bounded_jacobian(function, point, step=1e-3):
    """Finite differences in unit-box coordinates, one-sided at a boundary."""
    x = np.asarray(point, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all() or np.any(x < 0) or np.any(x > 1):
        raise ValueError("point must be a finite vector in the unit box")
    if not np.isfinite(step) or not 0 < step < 1:
        raise ValueError("step must be finite and between zero and one")
    columns = []
    for i in range(len(x)):
        plus, minus = x.copy(), x.copy()
        plus[i] = min(1., x[i] + step)
        minus[i] = max(0., x[i] - step)
        columns.append((function(plus) - function(minus)) / (plus[i] - minus[i]))
    return np.column_stack(columns)


def from_unit(point):
    x = np.asarray(point, dtype=float)
    if x.shape != LOW.shape or not np.isfinite(x).all() or np.any(x < 0) or np.any(x > 1):
        raise ValueError("nine finite unit-box coordinates are required")
    return LOW + x * (HIGH - LOW)


def main():
    raw = PREVIOUS.read_bytes()
    previous = json.loads(raw)
    if previous["status"] != "exploratory joint optimization attempt; optimality not established":
        raise ValueError("expected prior joint optimization attempt")
    x0 = np.array([previous["parameters"][name] for name in PARAMETERS])
    u0 = np.clip((x0 - LOW) / (HIGH - LOW), 1e-5, 1. - 1e-5)
    drug_records, noise_records = load_anchors(), load_noise()

    def residual(u):
        return evaluate_joint(from_unit(u), drug_records, noise_records,
                              step=.1, rtol=2e-7, atol=2e-9)[0]

    initial = residual(u0)
    print(f"Starting scaled fit: score {initial @ initial:.6f}", flush=True)
    opt = least_squares(residual, u0, jac=lambda u: bounded_jacobian(residual, u),
                        bounds=(np.zeros(9), np.ones(9)), method="trf", max_nfev=70,
                        ftol=1e-6, xtol=1e-6, gtol=1e-6)
    x = from_unit(opt.x)
    report = {
        "status": "exploratory scaled joint optimization; optimality and full Tier 1 coverage not established",
        "input_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (PREVIOUS, SEQUENTIAL, SOURCE)},
        "optimizer": {"method": "scipy least_squares trf in unit-box coordinates",
                      "jacobian_step_unit_box": .001, "max_nfev": 70,
                      "success": bool(opt.success), "message": str(opt.message),
                      "evaluations": int(opt.nfev), "jacobian_evaluations": int(opt.njev),
                      "initial_score": float(initial @ initial), "score": float(opt.fun @ opt.fun),
                      "optimality": float(opt.optimality),
                      "active_bounds": dict(zip(PARAMETERS, map(int, opt.active_mask)))},
        "parameters": dict(zip(PARAMETERS, map(float, x))),
        "limitations": [
            "This is a numerical diagnostic for the existing nine-parameter drug and Wang 1992 noise objective, not full Tier 1 calibration.",
            "A single outcome-informed start, chosen bounds and local termination do not establish global optimality, parameter identifiability or physiological validity.",
            "Unit-box finite differences and a sampled-extrema objective can mask nonsmoothness; optimizer optimality is conditional on numerical choices.",
            "Drug/noise weights remain illustrative; correlated group means, model discrepancy and cross-study covariance are not represented by a likelihood.",
            "Species/exposure mappings, 125 dB injury mechanism, hybrid closure and treated K boundary remain assumptions.",
            "State ranges are sampled, not enforced or proven invariant. Solver agreement does not establish model stability.",
        ],
        "checks": [],
    }
    for label, values, method, step in (
            ("sequential_initial", x0, "Radau", .025),
            ("scaled_candidate", x, "Radau", .1),
            ("scaled_candidate", x, "BDF", .1),
            ("scaled_candidate", x, "Radau", .025)):
        r, details = evaluate_joint(values, drug_records, noise_records, method=method,
                                    step=step, rtol=1e-8, atol=1e-10)
        report["checks"].append({"candidate": label, "method": method, "step_min": step,
                                 "score": float(r @ r), **details})
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Finished scaled fit: {opt.message}; score {opt.fun @ opt.fun:.6f}; wrote {OUTPUT}",
          flush=True)
    if not opt.success:
        raise RuntimeError(f"Scaled optimization did not converge: {opt.message}")


if __name__ == "__main__":
    main()
