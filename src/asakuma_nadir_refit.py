"""Outcome-informed refits using resistance at a sampled modeled EP nadir."""

import hashlib
import json

import numpy as np
from scipy.optimize import least_squares

from src.asakuma_nadir_timing_audit import INPUT, RESULTS, evaluate_case
from src.joint_calibration_screen import RESISTANCE_RATIO, configuration, parameter_space
from src.joint_mechanism_comparison import make_simulator
from src.joint_timing_sensitivity import temporal_residuals
from src.low_dose_assumption_sensitivity import GEOMETRY
from src.pike_trace_diagnostic import load_anchors


OUTPUT = RESULTS / "asakuma_nadir_refit.json"
SCENARIOS = ((2.6, (4, 2)), (1.0, (5, 3)))
WATER_RATE = .00025


def corrected_residuals(records, names, values, exchange, hill, *, step=1.,
                        method="Radau", rtol=1e-6, atol=1e-8):
    """Replace only the historical six-minute Asakuma residual."""
    simulator = make_simulator(exchange=exchange, volume_gate=False)
    residuals, predictions = temporal_residuals(
        records, values, names, series_fraction=.5, resistance_time=6,
        grid_step=step, method=method, rtol=rtol, atol=atol,
        simulator=simulator, hill=hill, water_rate=WATER_RATE, **GEOMETRY)
    case = {"parameters": dict(zip(names, map(float, values))), "exchange": float(exchange),
            "hill": hill, "water_rate": WATER_RATE}
    nadir = evaluate_case(case, method, step, rtol=rtol, atol=atol)
    if nadir["nadir_at_window_boundary"]:
        raise ValueError("modeled Asakuma EP nadir occurs at observation window boundary")
    residuals[len(records)] = (nadir["resistance_ratio_at_sampled_nadir"]
                               - RESISTANCE_RATIO) / .2
    predictions.pop("conditional_resistance_ratio")
    predictions.pop("conditional_resistance_time_min")
    predictions["asakuma_sampled_nadir"] = nadir
    return residuals, predictions


def fit_scenario(records, names, lower, upper, source, hill, start_indices):
    bounds = (np.append(np.log(lower), 0.), np.append(np.log(upper), 10.))
    row = {"hill": hill, "water_rate": WATER_RATE, "attempts": [],
           "status": "no converged start"}
    completed = []
    for index in start_indices:
        prior = source["cases"][index]
        if prior["status"] != "converged":
            raise ValueError(f"source start {index} did not converge")
        x0 = np.append(np.log([prior["parameters"][name] for name in names]), prior["exchange"])
        attempt = {"source_case_index": index, "starting_parameters": prior["parameters"],
                   "starting_exchange": prior["exchange"]}
        try:
            def residual(x):
                return corrected_residuals(records, names, np.exp(x[:-1]), x[-1], hill)[0]

            initial = residual(x0)
            result = least_squares(residual, x0, bounds=bounds, max_nfev=100,
                                   diff_step=1e-3, ftol=1e-6, xtol=1e-6, gtol=1e-6)
            attempt.update({"initial_weighted_sum_squares": float(initial @ initial),
                            "success": bool(result.success), "message": str(result.message),
                            "evaluations": int(result.nfev),
                            "weighted_sum_squares_optimization_grid": float(result.fun @ result.fun),
                            "parameters": dict(zip(names, map(float, np.exp(result.x[:-1])))),
                            "exchange": float(result.x[-1]),
                            "active_bounds": dict(zip([*names, "exchange"],
                                                      map(int, result.active_mask)))})
            if result.success:
                completed.append(result)
        except (ValueError, RuntimeError, FloatingPointError, OverflowError) as error:
            attempt.update({"success": False, "error": f"{type(error).__name__}: {error}"})
        row["attempts"].append(attempt)
    if not completed:
        return row
    best = min(completed, key=lambda x: x.cost)
    values, exchange = np.exp(best.x[:-1]), float(best.x[-1])
    row.update({"parameters": dict(zip(names, map(float, values))), "exchange": exchange,
                "active_bounds": dict(zip([*names, "exchange"], map(int, best.active_mask)))})
    row["evaluations"] = []
    for method, step in (("Radau", .25), ("BDF", .25), ("Radau", .05)):
        try:
            residuals, predictions = corrected_residuals(
                records, names, values, exchange, hill, step=step,
                method=method, rtol=1e-8, atol=1e-10)
            n = len(records)
            row["evaluations"].append({
                "status": "ok", "method": method, "grid_step_min": step,
                "weighted_sum_squares": float(residuals @ residuals),
                "objective_components": {
                    "pike_ep": float(residuals[:n] @ residuals[:n]),
                    "asakuma_resistance": float(residuals[n] ** 2),
                    "rybak_ep": float(np.sum(residuals[n+1:n+7:2] ** 2)),
                    "rybak_k": float(np.sum(residuals[n+2:n+7:2] ** 2)),
                    "rybak_timing": float(residuals[-2:] @ residuals[-2:]),
                },
                "predictions": predictions,
            })
        except (ValueError, RuntimeError, FloatingPointError, OverflowError) as error:
            row["evaluations"].append({"status": "failed", "method": method,
                                       "grid_step_min": step,
                                       "error": f"{type(error).__name__}: {error}"})
    row["status"] = ("converged" if all(r["status"] == "ok" for r in row["evaluations"])
                     else "post-fit evaluation failed")
    return row


def main():
    raw = INPUT.read_bytes()
    source = json.loads(raw)
    names, lower, upper, _ = parameter_space(configuration(True, 0))
    upper[names.index("potassium_flux")] = 2.
    upper[names.index("drug_loss_fraction")] = .98
    records = load_anchors()
    report = {
        "status": "in progress",
        "source": str(INPUT.relative_to(RESULTS.parent.parent)),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "bounds": {**dict(zip(names, [[float(lo), float(hi)] for lo, hi in zip(lower, upper)])),
                   "exchange": [0., 10.]},
        "geometry": GEOMETRY, "asakuma_nadir_window_min": [0, 120],
        "limitations": [
            "Outcome-informed exploratory refits, not validation or formal model selection.",
            "The representative modeled nadir is compared to a ratio of group means at individual nadirs; covariance is unavailable.",
            "The 0–120 minute nadir window, series fraction 0.5, parameter bounds and exposure proxy are assumptions.",
            "Asakuma EP nadir is not an objective endpoint; Rybak timing penalties, scales and remaining endpoints are unchanged.",
            "Optimization uses a 1-minute sampled nadir, leading to a nonsmooth objective; finer-grid and solver comparisons are post-fit only.",
            "Both starts and all scenarios are outcome-informed; local convergence does not establish a global optimum, identifiability or physiological closure.",
        ],
        "cases": [],
    }
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for hill, indices in SCENARIOS:
        print(f"Starting Hill {hill:g}", flush=True)
        report["cases"].append(fit_scenario(records, names, lower, upper, source, hill, indices))
        OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"Completed Hill {hill:g}: {report['cases'][-1]['status']}", flush=True)
    report["status"] = ("completed exploratory refits" if
                        all(c["status"] == "converged" for c in report["cases"])
                        else "completed with failures; inspect cases")
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
