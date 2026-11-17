"""Exploratory Tier 1 drug anchoring of the reference manuscript reconstruction."""

from dataclasses import replace
import hashlib
import json

import numpy as np
from scipy.optimize import least_squares

from src.asakuma_ep_endpoint_sensitivity import EVIDENCE, load_endpoint
from src.joint_calibration_screen import RESISTANCE_RATIO, RYBAK
from src.manuscript_audit import MANUSCRIPT, sampled_state_ranges
from src.model_equations import (
    Inputs, Parameters, STATE_NAMES, drug_concentration, observables,
    related_code_closure, simulate,
)
from src.pike_trace_diagnostic import IDS, ROOT, SOURCE, load_anchors


OUTPUT = ROOT / "results" / "exploratory" / "drug_calibration.json"
NAMES = ("guinea_pig_proxy_per_mg_kg", "chinchilla_exposure_multiplier",
         "drug_conductance_loss", "drug_k_loss")
LOWER = np.array([.01, .5, .05, .001])
UPPER = np.array([.2, 2., .98, .08])
STARTS = (np.array([.0525, 1., .44, .0165]),
          np.array([.1, 1.5, .8, .03]),
          np.array([.03, .7, .65, .05]))


def evaluate(values, records, *, step=1., method="Radau", rtol=2e-6, atol=2e-8,
             base_parameters=None):
    """Return a curve-balanced residual vector and endpoint predictions."""
    x = np.asarray(values, dtype=float)
    if x.shape != (len(NAMES),) or not np.isfinite(x).all():
        raise ValueError("four finite calibration parameters are required")
    if np.any(x < LOWER) or np.any(x > UPPER):
        raise ValueError("parameters outside declared numerical bounds")
    if not np.isfinite(step) or step <= 0 or not np.isclose(120 / step, round(120 / step)):
        raise ValueError("grid step must divide 120 minutes")
    scale, species, loss, k_loss = x
    p = replace(Parameters() if base_parameters is None else base_parameters,
                drug_conductance_loss=float(loss), drug_k_loss=float(k_loss))
    closure = related_code_closure()
    residuals, predictions = [], {}
    grid = np.linspace(0., 120., int(round(120. / step)) + 1)

    def run(dose, multiplier, duration, times):
        level = float(dose * scale * multiplier)
        forcing = lambda t: Inputs(drug=drug_concentration(t, 0., duration, level, p),
                                   furosemide_protocol=True)
        y = simulate(times, forcing, closure, p, boundaries=(duration,),
                     method=method, rtol=rtol, atol=atol)
        return y, forcing

    for key in IDS:
        rows = [r for r in records if r["curve"] == key]
        if len(rows) < 2 or rows[0]["time_min"] != 0:
            raise ValueError(f"{key} must have ordered anchors starting at zero")
        t = np.array([r["time_min"] for r in rows])
        if np.any(np.diff(t) <= 0) or t[-1] <= .542:
            raise ValueError(f"invalid anchor times for {key}")
        model, _ = run(rows[0]["dose_mg_kg"], 1., .542, t)
        observed = np.array([r["ep_mV"] for r in rows])
        errors = (model[8] - model[8, 0]) - (observed - observed[0])
        residuals.extend(errors / (10. * np.sqrt(len(errors))))
        predictions[key] = {"change_rmse_mV": float(np.sqrt(np.mean(errors ** 2))),
                            "modeled_baseline_mV": float(model[8, 0]),
                            "observed_digitized_baseline_mV": float(observed[0])}

    model, forcing = run(60., 1., .5, grid)
    index = int(np.argmin(model[8]))
    if index == 0 or index == len(grid) - 1:
        raise ValueError("Asakuma sampled EP nadir at window boundary")
    ratio = observables(model[:, index], forcing(float(grid[index])), closure, p)["resistance_ratio"]
    mean, _, _ = load_endpoint()
    predicted_drop = float(model[8, 0] - model[8, index])
    observed_drop = 85. - mean
    residuals.extend(((ratio - RESISTANCE_RATIO) / .2, (predicted_drop - observed_drop) / 10.))
    predictions["asakuma_60"] = {
        "ep_reduction_mV": predicted_drop, "ep_nadir_mV_model_reference": float(model[8, index]),
        "resistance_ratio_at_ep_nadir": float(ratio), "ep_nadir_time_min": float(grid[index]),
        "observed_baseline_minus_nadir_mean_mV": observed_drop,
    }

    for dose, ep_observed, ep_se, k_observed, k_se, n in RYBAK:
        model, _ = run(dose, species, .5, grid)
        volume = 1. + p.strial_volume_base - model[4]
        ke = (model[10] - p.sync_volume * model[5] - model[4] * model[6]
              - p.marginal_volume * model[7]) / volume
        if np.any(ke <= 0) or not np.isfinite(ke).all():
            raise ValueError("nonpositive or nonfinite sampled endolymph K")
        ep_i, k_i = int(np.argmin(model[8])), int(np.argmin(ke))
        if ep_i in (0, len(grid) - 1) or k_i in (0, len(grid) - 1):
            raise ValueError("Rybak sampled minimum at window boundary")
        ep_drop, k_drop = float(model[8, 0] - model[8, ep_i]), float(ke[0] - ke[k_i])
        residuals.extend(((ep_drop - ep_observed) / ep_se, (k_drop - k_observed) / k_se))
        predictions[f"rybak_{dose}"] = {
            "ep_reduction_mV": ep_drop, "k_reduction_mM": k_drop, "n": n,
            "ep_nadir_time_min": float(grid[ep_i]), "k_nadir_time_min": float(grid[k_i]),
            "declared_ranges": sampled_state_ranges(grid, model),
        }
    return np.asarray(residuals), predictions


def main():
    records = load_anchors()
    evidence = json.loads(EVIDENCE.read_text())
    study = next(s for s in evidence["studies"] if s["id"] == "asakuma_snow_1980")
    if study["observation"]["baseline_ep_mean_mV"] != 85.:
        raise ValueError("Asakuma baseline transcription changed; review observation operator")
    report = {
        "status": "in progress",
        "input_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (EVIDENCE, SOURCE, MANUSCRIPT)},
        "parameter_bounds": dict(zip(NAMES, [[float(a), float(b)] for a, b in zip(LOWER, UPPER)])),
        "limitations": [
            "Exploratory fit to a subset of Tier 1 drug evidence; no other Tier 1 endpoints were fitted.",
            "The 13-state draft is incomplete; the named hybrid closure is borrowed from the related 19-state code.",
            "Furosemide exposure is an assumed linear dose-to-effective-proxy mapping, not measured pharmacokinetics or NKCC occupancy.",
            "Pike and Asakuma are guinea pig, Rybak chinchilla; a single species multiplier is an unverified assumption.",
            "Infusion durations 0.542 min Pike, 0.5 min Asakuma and assumed 0.5 min Rybak; treated K boundary enabled for entire simulation.",
            "Pike manual group curve errors are baseline-relative, equal-curve-weighted and divided by an illustrative 10 mV, not independent animals or a likelihood.",
            "Asakuma uses difference of reported EP means and ratio of rounded resistance means at modeled EP nadir, weighted by illustrative 10 mV and 0.2; individual correlations and model discrepancy are unknown.",
            "Rybak group mean EP and K reductions use reported SE as scales, but covariance, baseline differences and model discrepancy are not included.",
            "One representative trajectory's minimum need not equal a group mean of individual minima.",
            "Bounds and grid are numerical choices, not physiological limits; starts are outcome-informed and do not establish a global optimum.",
            "Other reference parameters remain frozen and rounded; the reference state is not exactly stationary, and no ion/current/water closure is established.",
        ],
        "optimization_grid_min": 1.,
        "attempts": [],
    }
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    completed = []
    for start in STARTS:
        try:
            initial, _ = evaluate(start, records)
            print(f"Starting {start.tolist()}, score {initial @ initial:.4f}", flush=True)
            opt = least_squares(lambda x: evaluate(x, records)[0], start,
                                bounds=(LOWER, UPPER), max_nfev=60, diff_step=1e-3,
                                ftol=1e-5, xtol=1e-5, gtol=1e-5)
        except (ValueError, RuntimeError, FloatingPointError, OverflowError) as error:
            report["attempts"].append({"start": start.tolist(), "success": False,
                                       "error": f"{type(error).__name__}: {error}"})
            OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            continue
        report["attempts"].append({
            "start": start.tolist(), "success": bool(opt.success), "message": str(opt.message),
            "evaluations": int(opt.nfev), "initial_score": float(initial @ initial),
            "optimization_grid_score": float(opt.fun @ opt.fun),
            "parameters": dict(zip(NAMES, map(float, opt.x))),
            "active_bounds": dict(zip(NAMES, map(int, opt.active_mask))),
        })
        if opt.success:
            completed.append(opt)
        OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"Finished: {opt.message}", flush=True)
    if not completed:
        report["status"] = "no converged start"
    else:
        best = min(completed, key=lambda x: x.cost)
        report["best_parameters"] = dict(zip(NAMES, map(float, best.x)))
        report["checks"] = []
        for method, step in (("Radau", .25), ("BDF", .25), ("Radau", .05)):
            r, pred = evaluate(best.x, records, step=step, method=method, rtol=1e-8, atol=1e-10)
            report["checks"].append({"method": method, "step_min": step,
                                     "score": float(r @ r), "predictions": pred})
        report["status"] = "completed exploratory tier 1 drug calibration"
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
