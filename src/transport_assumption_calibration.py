"""Exploratory transport and exposure alternatives fitted to Pike EP anchors."""

import json
import math
from itertools import product
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from src.assumption_calibration import parameter_space
from src.assumption_comparison import Circuit, alternative_rhs
from src.cochlear_model import Inputs, Parameters, baseline, potassium_state
from src.pike_ode_fit import drug_proxy
from src.pike_trace_diagnostic import IDS, load_anchors


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "results" / "exploratory" / "transport_assumption_calibration.json"


def inhibition(time, dose, dose_scale, decay, hill):
    """Apply a Hill exponent to the odds of the shared effective exposure."""
    if not math.isfinite(hill) or hill <= 0:
        raise ValueError("Hill exponent must be positive and finite")
    base = drug_proxy(time, dose, dose_scale, decay)
    if base == 0:
        return 0.0
    odds = base / (1.0 - base)
    powered = odds ** hill
    return powered / (1.0 + powered)


def transport_rhs(time, state, inputs, params, circuit, uptake):
    """Replace S-to-M flux with a saturating law, preserving amount balance."""
    if uptake not in ("linear", "saturating"):
        raise ValueError("unknown uptake law")
    rates = alternative_rhs(time, state, inputs, params, circuit)
    if uptake == "linear":
        return rates
    ks, am, volume = state[7], state[4], state[5]
    original_factor = ks / 5.0
    saturating_factor = 2.0 * ks / (5.0 + ks)
    change = params.potassium_flux * am * (1.0 - inputs.drug) * (
        saturating_factor - original_factor
    )
    rates[7] -= change / volume
    rates[8] += change / params.marginal_volume
    return rates


def simulate_transport(times, forcing, params, circuit, uptake, *,
                       method="Radau", rtol=1e-6, atol=1e-8):
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("times must be finite and strictly increasing")
    result = solve_ivp(
        lambda t, state: transport_rhs(t, state, forcing(t), params, circuit, uptake),
        (times[0], times[-1]), baseline(params), t_eval=times, method=method, rtol=rtol, atol=atol,
    )
    if not result.success:
        raise RuntimeError(f"ODE integration failed: {result.message}")
    return result


def configurations():
    for uptake, hill, circuit in product(("linear", "saturating"), (1.0, 2.6),
                                          ("original", "combined")):
        yield {
            "uptake": uptake, "hill": hill, "circuit": circuit,
            "apical_battery": circuit == "combined", "met_load": circuit == "combined",
            "drug_conductance_loss": circuit == "combined",
            "volume_coupling": 1.0, "drug_sink": 0.0,
        }


def evaluate(records, config, values, names, *, method="Radau", rtol=1e-6, atol=1e-8):
    v = dict(zip(names, values))
    params = Parameters(potassium_flux=float(v["potassium_flux"]), drug_sink=0,
                        endolymph_volume_coupling=1)
    circuit = Circuit(config["apical_battery"], config["met_load"],
                      config["drug_conductance_loss"],
                      float(v.get("met_conductance", 0.3)),
                      float(v.get("drug_loss_fraction", 0.44)))
    groups = []
    for identifier in IDS:
        group = [r for r in records if r["curve"] == identifier]
        if len(group) < 2 or group[0]["time_min"] != 0:
            raise ValueError(f"{identifier} needs observations starting at zero")
        dose = float(group[0]["dose_mg_kg"])
        if any(float(row["dose_mg_kg"]) != dose for row in group):
            raise ValueError(f"inconsistent dose for {identifier}")
        forcing = lambda t: Inputs(drug=inhibition(t, dose, v["effective_dose_scale_mg_kg"],
                                                   v["exposure_decay_min"], config["hill"]))
        result = simulate_transport([r["time_min"] for r in group], forcing, params, circuit,
                                    config["uptake"], method=method, rtol=rtol, atol=atol)
        errors = result.y[10] - np.array([r["ep_mV"] for r in group])
        groups.append((identifier, errors, result))
    return np.concatenate([error / np.sqrt(len(error)) for _, error, _ in groups]), groups


def fit(records, config):
    names, lower, upper, starts = parameter_space(config)
    attempts, completed = [], []
    for start in starts:
        try:
            result = least_squares(
                lambda x: evaluate(records, config, np.exp(x), names)[0],
                np.log(start), bounds=(np.log(lower), np.log(upper)), max_nfev=100,
                diff_step=1e-3, ftol=1e-6, xtol=1e-6, gtol=1e-6,
            )
        except (ValueError, RuntimeError, FloatingPointError, OverflowError) as error:
            attempts.append({"start": start.tolist(), "success": False,
                             "error": f"{type(error).__name__}: {error}"})
            continue
        attempts.append({"start": start.tolist(), "success": bool(result.success),
                         "message": str(result.message), "evaluations": int(result.nfev),
                         "cost": float(result.cost), "active_bounds": result.active_mask.tolist(),
                         "parameters": dict(zip(names, map(float, np.exp(result.x))))})
        if result.success:
            completed.append(result)
    output = {"configuration": config,
              "bounds": dict(zip(names, [[float(lo), float(hi)] for lo, hi in zip(lower, upper)])),
              "attempts": attempts, "status": "no converged start" if not completed else "converged"}
    if not completed:
        return output
    best = min(completed, key=lambda x: x.cost)
    values = np.exp(best.x)
    output["parameters"] = dict(zip(names, map(float, values)))
    output["active_bounds"] = best.active_mask.tolist()
    try:
        residual, groups = evaluate(records, config, values, names, rtol=1e-8, atol=1e-10)
        _, checks = evaluate(records, config, values, names, method="BDF", rtol=1e-9, atol=1e-11)
    except (ValueError, RuntimeError, FloatingPointError, OverflowError) as error:
        output["status"] = "post-fit verification failed"
        output["verification_error"] = f"{type(error).__name__}: {error}"
        return output
    output["balanced_sum_squared_error_mV2"] = float(np.dot(residual, residual))
    output["curves"] = []
    for (identifier, error, solution), (_, _, check) in zip(groups, checks):
        params = Parameters(potassium_flux=float(values[0]), drug_sink=0)
        output["curves"].append({
            "curve": identifier, "rmse_mV": float(np.sqrt(np.mean(error ** 2))),
            "minimum_ep_at_anchors_mV": float(np.min(solution.y[10])),
            "minimum_endolymph_k_at_anchors_mM": float(min(
                potassium_state(state, params).endolymph.concentration for state in solution.y.T
            )),
            "max_ep_solver_difference_at_anchors_mV": float(
                np.max(np.abs(solution.y[10] - check.y[10]))),
            "max_total_k_drift": float(np.max(np.abs(solution.y[9] - solution.y[9, 0]))),
        })
    return output


def main():
    records = load_anchors()
    report = {
        "status": "outcome-informed restricted sensitivity calibration; not validated or unbiased selection",
        "design": "2 uptake laws x 2 fixed Hill exponents x 2 circuit bundles; 3 starts",
        "limitations": [
            "Saturating uptake is normalized at baseline with half-scale 5 mM; this is assumed, not measured.",
            "Hill 2.6 is a reference-derived scenario, not an independently measured local NKCC1 dose law.",
            "Circuit bundles confound three electrical changes; fluid coupling and sink are fixed.",
            "Pike EP anchors alone are fitted; drug exposure, additional endpoints and species differences remain unresolved.",
            "Bounds are numerical, not physiological; optimization is local and can fail on inadmissible trial states.",
            "Anchor-time numerical checks do not prove continuous-time admissibility or stability.",
        ],
        "candidates": [],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for config in configurations():
        report["candidates"].append(fit(records, config))
        OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"{len(report['candidates'])}/8 {report['candidates'][-1]['status']}", flush=True)


if __name__ == "__main__":
    main()
