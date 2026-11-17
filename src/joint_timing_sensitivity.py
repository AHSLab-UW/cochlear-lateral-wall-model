"""Outcome-informed sensitivity to Rybak timing and resistance observations."""

import json
from itertools import product
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from src.assumption_comparison import Circuit
from src.cochlear_model import Inputs, Parameters, baseline
from src.joint_calibration_screen import (
    RESISTANCE_RATIO, RYBAK, configuration, evaluate, parameter_space,
    partition_resistance_ratio, sampled_domain,
)
from src.pike_trace_diagnostic import load_anchors
from src.transport_assumption_calibration import inhibition, simulate_transport


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "results" / "exploratory" / "joint_timing_sensitivity.json"


def temporal_residuals(records, values, names, *, series_fraction, resistance_time,
                       grid_step=1.0, method="Radau", rtol=1e-6, atol=1e-8,
                       simulator=simulate_transport, strial_reference_volume=0.1,
                       syncytial_volume=0.2, marginal_volume=0.3, hill=2.6,
                       water_rate=0.005):
    """Replace conditional observations and penalize a later, lower EP minimum."""
    if not np.isfinite(resistance_time) or resistance_time <= 0:
        raise ValueError("resistance observation time must be positive and finite")
    if not np.isfinite(grid_step) or not 0 < grid_step <= 1:
        raise ValueError("grid step must be positive and at most one minute")
    config = configuration(True, 0.0)
    residuals, predictions = evaluate(records, config, values, names, grid_step=grid_step,
                                       method=method, rtol=rtol, atol=atol,
                                       simulator=simulator,
                                       strial_reference_volume=strial_reference_volume,
                                       syncytial_volume=syncytial_volume,
                                       marginal_volume=marginal_volume, hill=hill,
                                       water_rate=water_rate)
    v = dict(zip(names, values))
    params = Parameters(potassium_flux=float(v["potassium_flux"]), drug_sink=0,
                        endolymph_volume_coupling=0,
                        strial_reference_volume=strial_reference_volume,
                        syncytial_volume=syncytial_volume, marginal_volume=marginal_volume,
                        water_rate=water_rate)
    circuit = Circuit(True, True, True, v["met_conductance"], v["drug_loss_fraction"],
                      v.get("met_reversal_mV", -70.0))

    def forcing(dose, multiplier=1.0):
        return lambda t: Inputs(drug=inhibition(t, dose * multiplier,
                                                 v["effective_dose_scale_mg_kg"],
                                                 v["exposure_decay_min"], hill))

    resistance_forcing = forcing(60)
    sol = simulator([0, resistance_time], resistance_forcing, params, circuit,
                    "saturating", method=method, rtol=rtol, atol=atol)
    ratio = partition_resistance_ratio(sol.y[:, -1], resistance_forcing(resistance_time),
                                       params, circuit, series_fraction=series_fraction)
    residuals[len(records)] = (ratio - RESISTANCE_RATIO) / 0.2
    predictions["conditional_resistance_ratio"] = float(ratio)
    predictions["conditional_resistance_time_min"] = resistance_time
    predictions["conditional_series_fraction"] = series_fraction

    # Include exactly 3 and 120 even when the chosen grid interval does not
    # divide either interval; the source reports a nadir before 3 minutes.
    grid = np.unique(np.concatenate((np.arange(0, 120, grid_step), [3., 120.])))
    timing_penalties = []
    for index, (dose, ep_observed, ep_se, _, _, _) in enumerate(RYBAK):
        if dose not in (50, 100):
            continue
        sol = simulator(grid, forcing(dose, v["rybak_exposure_multiplier"]),
                        params, circuit, "saturating",
                        method=method, rtol=rtol, atol=atol)
        early = float(np.min(sol.y[10, grid <= 3]))
        overall = float(np.min(sol.y[10]))
        early_reduction = float(baseline(params)[10] - early)
        residuals[len(records) + 1 + 2 * index] = (early_reduction - ep_observed) / ep_se
        timing_penalties.append(max(0.0, early - overall) / 10.0)
        predictions[f"rybak_{dose}_early_ep_reduction_mV"] = early_reduction
        predictions[f"rybak_{dose}_late_lower_ep_gap_mV"] = early - overall
        predictions[f"rybak_{dose}_early_minimum_time_min"] = float(
            grid[grid <= 3][np.argmin(sol.y[10, grid <= 3])])
    return np.concatenate((residuals, timing_penalties)), predictions


def fit(records, series_fraction, resistance_time, initial):
    config = configuration(True, 0.0)
    names, lower, upper, starts = parameter_space(config)
    starts = [np.asarray(initial), starts[1]]
    attempts, completed = [], []
    for start in starts:
        try:
            result = least_squares(
                lambda x: temporal_residuals(records, np.exp(x), names,
                                             series_fraction=series_fraction,
                                             resistance_time=resistance_time)[0],
                np.log(start), bounds=(np.log(lower), np.log(upper)),
                max_nfev=100, diff_step=1e-3, ftol=1e-6, xtol=1e-6, gtol=1e-6,
            )
        except (ValueError, RuntimeError, FloatingPointError, OverflowError) as error:
            attempts.append({"start": start.tolist(), "success": False,
                             "error": f"{type(error).__name__}: {error}"})
            continue
        attempts.append({"start": start.tolist(), "success": bool(result.success),
                         "message": str(result.message), "evaluations": int(result.nfev),
                         "cost": float(result.cost),
                         "parameters": dict(zip(names, map(float, np.exp(result.x)))),
                         "active_bounds": dict(zip(names, map(int, result.active_mask)))})
        if result.success:
            completed.append(result)
    report = {"series_fraction": series_fraction, "resistance_time_min": resistance_time,
              "bounds": dict(zip(names, [[float(lo), float(hi)] for lo, hi in zip(lower, upper)])),
              "attempts": attempts, "status": "no converged start" if not completed else "converged"}
    if completed:
        best = min(completed, key=lambda x: x.cost)
        values = np.exp(best.x)
        report["parameters"] = dict(zip(names, map(float, values)))
        report["active_bounds"] = dict(zip(names, map(int, best.active_mask)))
        try:
            residuals, predictions = temporal_residuals(
                records, values, names, series_fraction=series_fraction,
                resistance_time=resistance_time, grid_step=0.25, rtol=1e-8, atol=1e-10)
            report["weighted_sum_squares"] = float(np.dot(residuals, residuals))
            report["predictions"] = predictions
            report["sampled_domain"] = {}
            for method in ("Radau", "BDF"):
                try:
                    report["sampled_domain"][method] = sampled_domain(
                        config, values, names, method=method)
                except (ValueError, RuntimeError, FloatingPointError, OverflowError) as error:
                    report["sampled_domain"][method] = {
                        "status": "failed", "error": f"{type(error).__name__}: {error}"}
        except (ValueError, RuntimeError, FloatingPointError, OverflowError) as error:
            report["status"] = "post-fit evaluation failed"
            report["evaluation_error"] = f"{type(error).__name__}: {error}"
    return report


def main():
    source = json.loads((OUTPUT.parent / "joint_calibration_screen.json").read_text())
    matches = [c for c in source["candidates"] if c["configuration"] == configuration(True, 0.0)
               and c["status"] == "converged"]
    if len(matches) != 1:
        raise ValueError("expected one converged closed-boundary combined-circuit joint fit")
    names, _, _, _ = parameter_space(configuration(True, 0.0))
    initial = [matches[0]["parameters"][name] for name in names]
    report = {
        "status": "outcome-informed timing and resistance sensitivity, not model selection",
        "model": "combined 14-state circuit, closed K boundary, fixed endolymph volume",
        "data": "same Pike, Asakuma and Rybak endpoints as joint_calibration_screen; no Tier 2 data",
        "observation_changes": [
            "For 50 and 100 mg/kg Rybak EP, compare group mean reduction to modeled minimum in 0-3 min, scaled by reported SE, instead of modeled 0-120 min minimum.",
            "Add one-sided penalty for each 50 and 100 mg/kg trajectory whose 0-120 min EP minimum is lower than its 0-3 min minimum; scale 10 mV is illustrative.",
            "Compare hypothetical resistance times of 2 or 6 min and series fraction 0 or 0.5. Source resistance is at the table-cohort EP nadir of unspecified time; 2 min describes a different figure cohort, and 6 min is the saline time.",
        ],
        "limitations": [
            "Source reports EP nadir under 3 min after 50 and 100 mg/kg; interval endpoint inclusion and mean-trajectory mapping are approximations.",
            "The 0-120 min observation window is assumed, not a source-reported common follow-up for all animals.",
            "Time-grid minima are approximations; optimization uses 1-min grid and reporting uses 0.25-min grid.",
            "The two resistance times and series fractions are sensitivity hypotheses, not direct measurements of separate circuit parameters.",
            "Reported SE scaling and illustrative penalties do not account for covariance or model discrepancy; objective is not a likelihood.",
            "One candidate family with two starting points per condition cannot establish global optimization or generalize to other model structures.",
            "Model, observation assumptions, bounds and timing constraints are outcome-informed. Numerical bounds are not physiological limits.",
            "Sampled Radau/BDF domain checks do not prove global stability, continuous positivity or physical completeness.",
        ],
        "scenarios": [],
    }
    records = load_anchors()
    for fraction, time in product((0.0, 0.5), (2.0, 6.0)):
        report["scenarios"].append(fit(records, fraction, time, initial))
        OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"Completed {len(report['scenarios'])}/4: {report['scenarios'][-1]['status']}",
              flush=True)


if __name__ == "__main__":
    main()
