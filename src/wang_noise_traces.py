"""Sequential fit of existing reference sound parameters to Wang 1992 EP anchors."""

from dataclasses import replace
import hashlib
import json

import numpy as np
from scipy.optimize import least_squares

from src.manuscript_audit import sampled_state_ranges
from src.model_equations import (Inputs, Parameters, broadband_sound_multiplier,
                                         related_code_closure, simulate)
from src.drug_calibration import evaluate
from src.kinetics_fit import ALL_NAMES, OUTPUT as DRUG_FIT
from src.pike_trace_diagnostic import ROOT, SOURCE, load_anchors


OUTPUT = DRUG_FIT.parent / "wang_noise_traces.json"
PAPER = ROOT / "refs" / "target_source" / "wang_1992_hear_res.pdf"
LEVELS = (105, 115, 125)
NAMES = ("adapt_rate", "adapt_evoked", "trauma_vasoconstriction")
LOWER = np.array([.03, .2, .2])
UPPER = np.array([.5, 1., .99])
STARTS = ([.145, .68, .76], [.08, .9, .4], [.3, .4, .9])


def load_noise(path=SOURCE):
    source = json.loads(path.read_text())
    result = []
    for level in LEVELS:
        key = f"07_fig1_acute_ep_during_and_after_noise_{level}_db_spl"
        matches = [e for e in source["experiments"] if e.get("id") == key]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one {key}")
        rows = matches[0]["observations"]
        t = np.array([r["time_min"] for r in rows], dtype=float)
        ep = np.array([r["ep_mV"] for r in rows], dtype=float)
        if len(t) < 3 or t[0] != 0 or t[-1] <= 20 or np.any(np.diff(t) <= 0):
            raise ValueError(f"invalid time series for {key}")
        if not np.isfinite(t).all() or not np.isfinite(ep).all():
            raise ValueError(f"nonfinite data in {key}")
        result.append((level, t, ep))
    return result


def make_parameters(values, drug_values):
    x = np.asarray(values, dtype=float)
    if x.shape != (3,) or not np.isfinite(x).all() or np.any(x < LOWER) or np.any(x > UPPER):
        raise ValueError("invalid bounded sound parameters")
    return replace(Parameters(), adapt_rate=float(x[0]), adapt_evoked=float(x[1]),
                   trauma_vasoconstriction=float(x[2]),
                   drug_conductance_loss=float(drug_values[2]), drug_k_loss=float(drug_values[3]),
                   drug_slow_fraction=float(drug_values[4]), drug_slow=float(drug_values[5]))


def evaluate_noise(values, drug_values, records, *, method="Radau", rtol=2e-6, atol=2e-8):
    p = make_parameters(values, drug_values)
    closure = related_code_closure()
    errors, predictions = [], []
    for level, times, observed in records:
        forcing = lambda t: Inputs(sound_multiplier=broadband_sound_multiplier(level) if t < 20. else 1.)
        y = simulate(times, forcing, closure, p, boundaries=(20.,),
                     method=method, rtol=rtol, atol=atol)
        observed_change = observed - observed[0]
        modeled_change = y[8] - y[8, 0]
        difference = modeled_change - observed_change
        errors.extend(difference / (3. * np.sqrt(len(difference))))
        predictions.append({"level_db_spl": level, "time_min": times.tolist(),
                            "observed_change_mV": observed_change.tolist(),
                            "model_change_mV": modeled_change.tolist(),
                            "rmse_mV": float(np.sqrt(np.mean(difference ** 2))),
                            "sampled_state_ranges": sampled_state_ranges(times, y)})
    return np.asarray(errors), predictions


def main():
    raw = DRUG_FIT.read_bytes()
    previous = json.loads(raw)
    if previous["status"] != "completed exploratory six-parameter tier 1 drug calibration":
        raise ValueError("completed six-parameter drug fit required")
    drug = np.array([previous["best_parameters"][name] for name in ALL_NAMES])
    records = load_noise()
    report = {
        "status": "in progress",
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (DRUG_FIT, SOURCE, PAPER)},
        "drug_parameters_frozen": dict(zip(ALL_NAMES, map(float, drug))),
        "sound_parameter_bounds": dict(zip(NAMES, [[float(a), float(b)]
                                                   for a, b in zip(LOWER, UPPER)])),
        "limitations": [
            "Sequential noise-only optimization of three existing parameters with six drug parameters frozen; this is not a joint optimization over all parameters or all Tier 1 evidence.",
            "Wang 1992 Fig. 1 provides approximate correlated group-mean observations; within-animal covariance is unavailable.",
            "Each level receives equal total weight, with illustrative 3 mV scale; objective is not a likelihood.",
            "The 125 dB condition includes severe acoustic trauma; the reference trauma model and broadband SPL-to-drive transform are unverified assumptions.",
            "Digitized times close to stimulus onset and offset have finite resolution, and plotted group means cannot reveal individual trajectories.",
            "Bounds are numerical choices, not measured physiological ranges; three starts do not establish a global optimum.",
            "Drug K boundary, species-exposure multiplier and hybrid workload closure remain conditional.",
            "No new model parameters are introduced; changing existing sound parameters may alter the drug fit, so it is separately reevaluated.",
        ],
        "attempts": [],
    }

    def save():
        OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    completed = []
    for start in STARTS:
        try:
            initial, _ = evaluate_noise(start, drug, records)
            opt = least_squares(lambda x: evaluate_noise(x, drug, records)[0], start,
                                bounds=(LOWER, UPPER), max_nfev=100, diff_step=1e-3,
                                ftol=1e-6, xtol=1e-6, gtol=1e-6)
            attempt = {"start": list(start), "success": bool(opt.success),
                       "message": str(opt.message), "evaluations": int(opt.nfev),
                       "initial_score": float(initial @ initial), "score": float(opt.fun @ opt.fun),
                       "parameters": dict(zip(NAMES, map(float, opt.x))),
                       "active_bounds": dict(zip(NAMES, map(int, opt.active_mask)))}
            if opt.success:
                completed.append(opt)
        except (ValueError, RuntimeError, FloatingPointError, OverflowError) as error:
            attempt = {"start": list(start), "success": False, "error": f"{type(error).__name__}: {error}"}
        report["attempts"].append(attempt)
        save()
        print(f"Finished noise fit: {attempt}", flush=True)
    if not completed:
        report["status"] = "no converged noise fit"
    else:
        best = min(completed, key=lambda opt: opt.cost)
        report["sound_parameters"] = dict(zip(NAMES, map(float, best.x)))
        report["noise_checks"] = []
        for method in ("Radau", "BDF"):
            r, pred = evaluate_noise(best.x, drug, records, method=method, rtol=1e-8, atol=1e-10)
            report["noise_checks"].append({"method": method, "score": float(r @ r), "predictions": pred})
        drug_r, drug_pred = evaluate(drug[:4], load_anchors(), step=.025, method="Radau",
                                     rtol=1e-8, atol=1e-10,
                                     base_parameters=make_parameters(best.x, drug))
        report["drug_recheck"] = {"score": float(drug_r @ drug_r), "predictions": drug_pred}
        report["status"] = "completed exploratory sequential noise calibration"
    save()
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
