"""Fit existing sound parameters to two Wang noise studies, conditionally."""

import hashlib
import json

import numpy as np
from scipy.optimize import least_squares

from src.manuscript_audit import sampled_state_ranges
from src.model_equations import Inputs, broadband_sound_multiplier, related_code_closure, simulate
from src.drug_calibration import evaluate as evaluate_drug
from src.scaled_parameter_fit import bounded_jacobian
from src.kinetics_fit import ALL_NAMES
from src.wang_noise_traces import (
    LOWER, NAMES, OUTPUT as PREVIOUS, PAPER as WANG1992, UPPER,
    evaluate_noise, load_noise, make_parameters,
)
from src.pike_trace_diagnostic import ROOT, SOURCE, load_anchors
from src.plot_initial_fit import WANG, WANG_BASELINE, WANG_MEANS


OUTPUT = PREVIOUS.parent / "wang_noise_cohorts.json"


def wang1990(values, drug, *, delay=45., method="Radau", rtol=2e-6, atol=2e-8,
             audit_grid=False):
    """Compare text cohort differences at an assumed fixed post-exposure delay."""
    if not np.isfinite(delay) or not 30. <= delay <= 60.:
        raise ValueError("the assumed post-exposure delay must be 30 to 60 minutes")
    p = make_parameters(values, drug)
    closure = related_code_closure()
    errors, predictions = [], []
    for duration, mean in WANG_MEANS:
        if audit_grid:
            times = np.unique(np.r_[np.arange(0., duration + delay, .1),
                                    float(duration), duration + delay])
        else:
            times = np.array([0., float(duration), duration + delay])
        forcing = lambda t: Inputs(sound_multiplier=broadband_sound_multiplier(125.)
                                   if t < duration else 1.)
        y = simulate(times, forcing, closure, p, boundaries=(float(duration),),
                     method=method, rtol=rtol, atol=atol)
        observed = mean - WANG_BASELINE
        modeled = float(y[8, -1] - y[8, 0])
        errors.append((modeled - observed) / (5. * np.sqrt(len(WANG_MEANS))))
        row = {"exposure_min": duration, "assumed_delay_min": float(delay),
               "observed_difference_mV": observed, "modeled_difference_mV": modeled}
        if audit_grid:
            row["sampled_state_ranges"] = sampled_state_ranges(times, y)
            if delay == 60.:
                window = (times >= duration + 30. - 1e-9) & (times <= duration + 60. + 1e-9)
                changes = y[8, window] - y[8, 0]
                row["model_range_30_60_min_mV"] = [float(np.min(changes)), float(np.max(changes))]
        predictions.append(row)
    return np.array(errors), predictions


def combined(values, drug, records, *, delay=45., method="Radau", rtol=2e-6, atol=2e-8):
    noise_r, noise_p = evaluate_noise(values, drug, records, method=method, rtol=rtol, atol=atol)
    cohort_r, cohort_p = wang1990(values, drug, delay=delay, method=method, rtol=rtol, atol=atol)
    return np.r_[noise_r, cohort_r], {
        "wang1992_score": float(noise_r @ noise_r), "wang1990_score": float(cohort_r @ cohort_r),
        "wang1992": noise_p, "wang1990": cohort_p,
    }


def main():
    raw = PREVIOUS.read_bytes()
    previous = json.loads(raw)
    if previous["status"] != "completed exploratory sequential noise calibration":
        raise ValueError("completed sequential noise fit required")
    drug = np.array([previous["drug_parameters_frozen"][key] for key in ALL_NAMES])
    start = np.array([previous["sound_parameters"][key] for key in NAMES])
    unit_start = np.clip((start - LOWER) / (UPPER - LOWER), 1e-5, 1. - 1e-5)
    records = load_noise()

    def residual(unit):
        return combined(LOWER + unit * (UPPER - LOWER), drug, records)[0]

    initial = residual(unit_start)
    print(f"Starting two-study sound fit: score {initial @ initial:.6f}", flush=True)
    opt = least_squares(residual, unit_start, jac=lambda x: bounded_jacobian(residual, x),
                        bounds=(np.zeros(3), np.ones(3)), max_nfev=70,
                        ftol=1e-6, xtol=1e-6, gtol=1e-6)
    values = LOWER + opt.x * (UPPER - LOWER)
    report = {
        "status": "conditional exploratory two-study noise optimization",
        "input_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (PREVIOUS, SOURCE, WANG, WANG1992)},
        "drug_parameters_frozen": dict(zip(ALL_NAMES, map(float, drug))),
        "sound_parameters": dict(zip(NAMES, map(float, values))),
        "sound_parameter_bounds": dict(zip(NAMES, [[float(a), float(b)]
                                                    for a, b in zip(LOWER, UPPER)])),
        "optimizer": {"success": bool(opt.success), "message": str(opt.message),
                      "evaluations": int(opt.nfev), "initial_score": float(initial @ initial),
                      "score": float(opt.fun @ opt.fun), "optimality": float(opt.optimality),
                      "active_bounds": dict(zip(NAMES, map(int, opt.active_mask)))},
        "limitations": [
            "Wang 1990 compared exposed and separate control cohorts, not paired individual changes; text and figure group counts and means disagree.",
            "All 1990 cohort observations are conditionally assigned one 45-minute measurement delay; the paper reports approximately 30–60 minutes for initial results, not each animal's exact delay.",
            "Wang 1990 exposed awake restrained animals; Wang 1992 recorded under anesthesia. The same sound-to-model drive is assumed, not established.",
            "Wang 1990 cohort differences receive equal weighting at an illustrative 5 mV scale; Wang 1992 uses the existing 3 mV curve scale. The combined score is not a likelihood.",
            "No new model or fitted delay parameters are introduced; only three existing sound parameters change, with all six drug parameters frozen.",
            "One outcome-informed starting point, numerical bounds and conditional observation timing do not establish a global optimum or identify sound mechanisms.",
            "The trauma model for 125 dB is an assumption; sampled state-range excursions are not physiological data or a stability proof.",
        ],
        "checks": [],
    }
    for label, x, method, delay in (
            ("sequential_initial", start, "Radau", 45.),
            ("two_study_candidate", values, "Radau", 45.),
            ("two_study_candidate", values, "BDF", 45.),
            ("two_study_candidate", values, "Radau", 30.),
            ("two_study_candidate", values, "Radau", 60.)):
        r, details = combined(x, drug, records, delay=delay, method=method, rtol=1e-8, atol=1e-10)
        report["checks"].append({"candidate": label, "method": method, "delay_min": delay,
                                 "score": float(r @ r), **details})
    _, ranges = wang1990(values, drug, delay=60., method="Radau", rtol=1e-8, atol=1e-10,
                         audit_grid=True)
    report["wang1990_sampled_ranges_0_1_min"] = ranges
    drug_residual, drug_predictions = evaluate_drug(
        drug[:4], load_anchors(), step=.025, method="Radau", rtol=1e-8, atol=1e-10,
        base_parameters=make_parameters(values, drug))
    report["drug_recheck"] = {"score": float(drug_residual @ drug_residual),
                              "predictions": drug_predictions}
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Finished two-study fit: {opt.message}; score {opt.fun @ opt.fun:.6f}; wrote {OUTPUT}",
          flush=True)
    if not opt.success:
        raise RuntimeError(f"Two-study optimization did not converge: {opt.message}")


if __name__ == "__main__":
    main()
