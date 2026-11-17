"""Outcome-informed common-protocol calibration of structural alternatives."""

import json
from itertools import product
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from src.assumption_comparison import Circuit, simulate_alternative
from src.cochlear_model import Inputs, Parameters
from src.pike_ode_fit import drug_proxy
from src.pike_trace_diagnostic import IDS, load_anchors


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "results" / "exploratory" / "assumption_calibration.json"
SHARED_NAMES = ("potassium_flux", "effective_dose_scale_mg_kg", "exposure_decay_min")
SHARED_LOWER = (0.05, 30.0, 3.0)
SHARED_UPPER = (0.5, 400.0, 80.0)
SHARED_STARTS = ((0.2, 60.0, 40.0), (0.45, 35.0, 20.0), (0.3, 150.0, 70.0))


def configurations():
    """Fixed factorial candidates; names and bounds are not evidence of plausibility."""
    for apical, met, loss, coupling, sink in product(
            (False, True), (False, True), (False, True), (0.0, 1.0), (0.0, 0.02)):
        yield {
            "apical_battery": apical, "met_load": met,
            "drug_conductance_loss": loss, "volume_coupling": coupling,
            "drug_sink": sink,
        }


def parameter_space(config):
    names = list(SHARED_NAMES)
    lower = list(SHARED_LOWER)
    upper = list(SHARED_UPPER)
    starts = [list(start) for start in SHARED_STARTS]
    if config["met_load"]:
        names.append("met_conductance")
        lower.append(0.01)
        upper.append(0.6)
        for start, value in zip(starts, (0.3, 0.1, 0.5)):
            start.append(value)
    if config["drug_conductance_loss"]:
        names.append("drug_loss_fraction")
        lower.append(0.01)
        upper.append(0.8)
        for start, value in zip(starts, (0.44, 0.15, 0.7)):
            start.append(value)
    return names, np.array(lower), np.array(upper), np.asarray(starts)


def evaluate(records, config, values, names, *, rtol=1e-6, atol=1e-8):
    """Residuals at observed points and unweighted, curve-balanced residual vector."""
    values_by_name = dict(zip(names, values))
    params = Parameters(potassium_flux=float(values_by_name["potassium_flux"]),
                        endolymph_volume_coupling=config["volume_coupling"],
                        drug_sink=config["drug_sink"])
    circuit = Circuit(
        config["apical_battery"], config["met_load"], config["drug_conductance_loss"],
        float(values_by_name.get("met_conductance", 0.3)),
        float(values_by_name.get("drug_loss_fraction", 0.44)),
    )
    groups = []
    for identifier in IDS:
        group = [r for r in records if r["curve"] == identifier]
        if len(group) < 2 or group[0]["time_min"] != 0:
            raise ValueError(f"{identifier} needs observations starting at zero")
        dose = float(group[0]["dose_mg_kg"])
        if any(float(row["dose_mg_kg"]) != dose for row in group):
            raise ValueError(f"inconsistent dose for {identifier}")
        forcing = lambda t: Inputs(drug=drug_proxy(
            t, dose, values_by_name["effective_dose_scale_mg_kg"],
            values_by_name["exposure_decay_min"]))
        result = simulate_alternative([r["time_min"] for r in group], forcing, params,
                                      circuit, rtol=rtol, atol=atol)
        residual = result.y[10] - np.array([r["ep_mV"] for r in group])
        groups.append((identifier, residual, result.y[10]))
    balanced = np.concatenate([residual / np.sqrt(len(residual))
                               for _, residual, _ in groups])
    return balanced, groups


def fit_configuration(records, config):
    names, lower, upper, starts = parameter_space(config)
    attempts = []
    converged = []

    def residual(log_values):
        return evaluate(records, config, np.exp(log_values), names)[0]

    for start in starts:
        try:
            result = least_squares(
                residual, np.log(start), bounds=(np.log(lower), np.log(upper)),
                max_nfev=100, diff_step=1e-3, ftol=1e-6, xtol=1e-6, gtol=1e-6,
            )
        except (ValueError, RuntimeError, FloatingPointError) as error:
            attempts.append({"start": start.tolist(), "success": False,
                             "error": f"{type(error).__name__}: {error}"})
            continue
        attempts.append({
            "start": start.tolist(), "success": bool(result.success),
            "message": str(result.message), "evaluations": int(result.nfev),
            "cost": float(result.cost), "active_bounds": result.active_mask.tolist(),
            "parameters": dict(zip(names, map(float, np.exp(result.x)))),
        })
        if result.success:
            converged.append(result)
    report = {
        "configuration": config,
        "numerical_bounds": dict(zip(names, [[float(lo), float(hi)]
                                              for lo, hi in zip(lower, upper)])),
        "attempts": attempts,
        "status": "no converged start" if not converged else "converged",
    }
    if converged:
        best = min(converged, key=lambda result: result.cost)
        values = np.exp(best.x)
        try:
            balanced, groups = evaluate(records, config, values, names, rtol=1e-8, atol=1e-10)
        except (ValueError, RuntimeError, FloatingPointError) as error:
            report["status"] = "tight-tolerance evaluation failed"
            report["evaluation_error"] = f"{type(error).__name__}: {error}"
            return report
        report["best_parameters"] = dict(zip(names, map(float, values)))
        report["best_active_bounds"] = best.active_mask.tolist()
        report["balanced_sum_squared_error_mV2"] = float(np.dot(balanced, balanced))
        report["rmse_mV_by_curve"] = {
            identifier: float(np.sqrt(np.mean(residual ** 2)))
            for identifier, residual, _ in groups
        }
        report["minimum_ep_at_anchor_times_mV_by_curve"] = {
            identifier: float(np.min(ep)) for identifier, _, ep in groups
        }
    return report


def main():
    records = load_anchors()
    report = {
        "status": "outcome-informed conditional calibrations; not independent validation",
        "data": "Pike and Bosher 1980 manual EP anchors, all three dose curves",
        "objective": "equal total squared-error weight per dose; not inverse variance",
        "protocol": "three starts per candidate with the same shared-parameter bounds and solver settings",
        "limitations": [
            "Candidates and bounds were designed after outcomes were known, not preregistered.",
            "Bounds are numerical exploration bounds, not established physiological bounds.",
            "MET conductance and drug loss are separately fitted only when present; both have positive numerical lower bounds.",
            "MET reversal, volume couplings and external sink values are fixed assumptions, not estimated.",
            "All non-EP subsystems and the dose proxy remain provisional; no other datasets were fitted.",
            "Point correlations, digitization uncertainty, biological variance and parameter uncertainty are not estimated.",
            "Multistart convergence does not establish a global optimum, identifiability or continuous-time admissibility.",
            "Complexity and independent-data performance are not quantified; do not rank candidates by training RMSE alone.",
            "Any failed start or tight-tolerance evaluation is explicitly recorded.",
        ],
        "candidates": [],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for config in configurations():
        report["candidates"].append(fit_configuration(records, config))
        OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"Completed {len(report['candidates'])}/32: {report['candidates'][-1]['status']}",
              flush=True)


if __name__ == "__main__":
    main()
