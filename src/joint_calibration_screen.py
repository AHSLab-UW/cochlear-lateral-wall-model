"""Exploratory joint EP, resistance and K calibration of provisional circuits.

Not a physiologically admissible model certification or independent validation.
"""

import json
from itertools import product
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from src.assumption_comparison import Circuit
from src.cochlear_model import Inputs, Parameters, baseline, potassium_state, relative_circuit_conductance
from src.pike_trace_diagnostic import IDS, load_anchors
from src.transport_assumption_calibration import inhibition, simulate_transport


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "results" / "exploratory" / "joint_calibration_screen.json"
RYBAK = ((25, 22.5, 5.3, 11.3, 3.3, 3),
         (50, 97.4, 10.6, 15.7, 5.4, 5),
         (100, 110.5, 12.5, 32.5, 8.3, 7))
RESISTANCE_RATIO = 5.41 / 2.54
SERIES_FRACTION = 0.5
NAMES = ("potassium_flux", "effective_dose_scale_mg_kg", "exposure_decay_min",
         "rybak_exposure_multiplier")
LOWER = np.array([0.05, 30.0, 3.0, 0.5])
UPPER = np.array([0.5, 400.0, 80.0, 2.0])
STARTS = ([0.2, 60., 40., 1.], [0.45, 35., 20., 1.5], [0.3, 150., 70., 0.7])


def partition_resistance_ratio(state, inputs, params, circuit, *, series_fraction=SERIES_FRACTION):
    """Conditional series-plus-parallel small-signal resistance observation model."""
    if not 0 <= series_fraction < 1:
        raise ValueError("series fraction must be in [0,1)")
    # Derivative of max(0,V-E_MET) is undefined at the reversal.
    if circuit.met_load and state[10] == circuit.met_reversal_mV:
        raise ValueError("slope resistance is undefined at the load reversal")
    load = circuit.met_conductance if circuit.met_load and state[10] > circuit.met_reversal_mV else 0
    loss = circuit.drug_loss_fraction * inputs.drug if circuit.drug_conductance_loss else 0
    slope = relative_circuit_conductance(state, inputs, params) - loss + load
    baseline_slope = 1 + (circuit.met_conductance if circuit.met_load else 0)
    if slope <= 0:
        raise ValueError("parallel slope conductance must be positive")
    return series_fraction + (1 - series_fraction) * baseline_slope / slope


def configuration(combined, sink):
    return {"combined_circuit": combined, "drug_sink": sink}


def parameter_space(config):
    names = list(NAMES)
    lower, upper = LOWER.tolist(), UPPER.tolist()
    starts = [list(start) for start in STARTS]
    if config["combined_circuit"]:
        names += ["met_conductance", "drug_loss_fraction"]
        lower += [0.01, 0.01]
        upper += [0.6, 0.8]
        for start, met, loss in zip(starts, (0.3, 0.1, 0.5), (0.44, 0.15, 0.7)):
            start.extend((met, loss))
    return names, np.array(lower), np.array(upper), np.asarray(starts)


def evaluate(records, config, values, names, *, rtol=1e-6, atol=1e-8, grid_step=1.0,
             method="Radau", simulator=simulate_transport, strial_reference_volume=0.1,
             syncytial_volume=0.2, marginal_volume=0.3, hill=2.6, water_rate=0.005):
    """Return weighted residuals and study-specific predictions."""
    v = dict(zip(names, values))
    params = Parameters(potassium_flux=float(v["potassium_flux"]),
                        drug_sink=config["drug_sink"], endolymph_volume_coupling=0,
                        strial_reference_volume=strial_reference_volume,
                        syncytial_volume=syncytial_volume, marginal_volume=marginal_volume,
                        water_rate=water_rate)
    combined = config["combined_circuit"]
    circuit = Circuit(combined, combined, combined, v.get("met_conductance", 0.3),
                      v.get("drug_loss_fraction", 0.44), v.get("met_reversal_mV", -70.0))
    dose_scale, decay = v["effective_dose_scale_mg_kg"], v["exposure_decay_min"]

    def forcing(dose, multiplier=1.0):
        return lambda time: Inputs(drug=inhibition(time, dose * multiplier, dose_scale, decay, hill))

    residuals, predictions = [], {}
    for key in IDS:
        group = [r for r in records if r["curve"] == key]
        if len(group) < 2 or group[0]["time_min"] != 0:
            raise ValueError(f"{key} must start at zero and contain observations")
        dose = float(group[0]["dose_mg_kg"])
        if any(float(row["dose_mg_kg"]) != dose for row in group):
            raise ValueError(f"inconsistent dose in {key}")
        sol = simulator([r["time_min"] for r in group], forcing(dose), params,
                        circuit, "saturating", method=method, rtol=rtol, atol=atol)
        error = sol.y[10] - np.array([r["ep_mV"] for r in group])
        residuals.extend(error / (10.0 * np.sqrt(len(group))))
        predictions[key] = {"rmse_mV": float(np.sqrt(np.mean(error ** 2))),
                            "minimum_at_anchor_times_mV": float(np.min(sol.y[10]))}

    # Historical 6-min assumption; source places this resistance at the
    # table-cohort EP nadir, whose time was not reported numerically.
    asakuma_forcing = forcing(60)
    sol = simulator([0, 6], asakuma_forcing, params, circuit, "saturating",
                    method=method, rtol=rtol, atol=atol)
    ratio = partition_resistance_ratio(sol.y[:, -1], asakuma_forcing(6), params, circuit)
    residuals.append((ratio - RESISTANCE_RATIO) / 0.2)
    predictions["asakuma_60_resistance_ratio_at_6min"] = float(ratio)

    if not np.isfinite(grid_step) or not 0 < grid_step <= 5:
        raise ValueError("grid step must be positive and no greater than five minutes")
    grid = np.append(np.arange(0, 120, grid_step), 120.)
    initial_ke = potassium_state(baseline(params), params).endolymph.concentration
    for dose, ep_observed, ep_se, observed, se, n in RYBAK:
        sol = simulator(grid, forcing(dose, v["rybak_exposure_multiplier"]),
                        params, circuit, "saturating", method=method, rtol=rtol, atol=atol)
        ke = np.array([potassium_state(state, params).endolymph.concentration
                       for state in sol.y.T])
        reduction = float(initial_ke - np.min(ke))
        ep_reduction = float(baseline(params)[10] - np.min(sol.y[10]))
        # Scale uses reported SE for each group; correlations and model error
        # are not included. Time-grid extrema approximate reported reductions.
        residuals.extend(((ep_reduction - ep_observed) / ep_se,
                          (reduction - observed) / se))
        predictions[f"rybak_{dose}_sampled_ep_reduction_mV"] = ep_reduction
        predictions[f"rybak_{dose}_sampled_k_reduction_mM"] = reduction
        predictions[f"rybak_{dose}_sampled_ep_minimum_time_min"] = float(grid[np.argmin(sol.y[10])])
        predictions[f"rybak_{dose}_sampled_k_minimum_time_min"] = float(grid[np.argmin(ke)])
        predictions[f"rybak_{dose}_n"] = n
    return np.asarray(residuals), predictions


def sampled_domain(config, values, names, *, method, simulator=simulate_transport,
                   strial_reference_volume=0.1, syncytial_volume=0.2, marginal_volume=0.3,
                   hill=2.6, water_rate=0.005):
    """Inspect all study doses through 120 min; sampled checks are not a proof."""
    v = dict(zip(names, values))
    params = Parameters(potassium_flux=float(v["potassium_flux"]),
                        drug_sink=config["drug_sink"], endolymph_volume_coupling=0,
                        strial_reference_volume=strial_reference_volume,
                        syncytial_volume=syncytial_volume, marginal_volume=marginal_volume,
                        water_rate=water_rate)
    combined = config["combined_circuit"]
    circuit = Circuit(combined, combined, combined, v.get("met_conductance", 0.3),
                      v.get("drug_loss_fraction", 0.44), v.get("met_reversal_mV", -70.0))
    grid = np.linspace(0, 120, 481)
    summaries = []
    for study, dose, multiplier in (
            ("guinea_pig", 40, 1), ("guinea_pig", 60, 1), ("guinea_pig", 80, 1),
            ("chinchilla", 25, v["rybak_exposure_multiplier"]),
            ("chinchilla", 50, v["rybak_exposure_multiplier"]),
            ("chinchilla", 100, v["rybak_exposure_multiplier"])):
        forcing = lambda t: Inputs(drug=inhibition(t, dose * multiplier,
                                                   v["effective_dose_scale_mg_kg"],
                                                   v["exposure_decay_min"], hill))
        sol = simulator(grid, forcing, params, circuit, "saturating",
                        method=method, rtol=1e-8, atol=1e-10)
        y = sol.y
        ke = np.array([potassium_state(state, params).endolymph.concentration for state in y.T])
        slope = [partition_resistance_ratio(state, forcing(t), params, circuit)
                 for t, state in zip(grid, y.T)]
        valid = (np.isfinite(y).all() and np.min(y[:9]) > 0 and np.min(ke) > 0
                 and np.min(y[11]) >= 0 and np.min(y[12:14]) >= 0
                 and np.max(y[12:14]) <= 1 and np.isfinite(slope).all())
        summaries.append({"study": study, "dose_mg_kg": dose, "method": method,
                          "sampled_domain_pass": bool(valid),
                          "minimum_strial_volume": float(np.min(y[5])),
                          "minimum_endolymph_k_mM": float(np.min(ke)),
                          "minimum_ep_mV": float(np.min(y[10]))})
    return summaries


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
    out = {"configuration": config,
           "bounds": dict(zip(names, [[float(a), float(b)] for a, b in zip(lower, upper)])),
           "attempts": attempts, "status": "no converged start" if not completed else "converged"}
    if completed:
        best = min(completed, key=lambda x: x.cost)
        out["parameters"] = dict(zip(names, map(float, np.exp(best.x))))
        out["active_bounds"] = dict(zip(names, map(int, best.active_mask)))
        try:
            residual, predictions = evaluate(records, config, np.exp(best.x), names,
                                             rtol=1e-8, atol=1e-10, grid_step=0.25)
        except (ValueError, RuntimeError, FloatingPointError, OverflowError) as error:
            out["status"] = "post-fit evaluation failed"
            out["evaluation_error"] = f"{type(error).__name__}: {error}"
            return out
        out["weighted_sum_squares"] = float(np.dot(residual, residual))
        out["predictions"] = predictions
        out["sampled_domain"] = {}
        for method in ("Radau", "BDF"):
            try:
                out["sampled_domain"][method] = sampled_domain(
                    config, np.exp(best.x), names, method=method)
            except (ValueError, RuntimeError, FloatingPointError, OverflowError) as error:
                out["sampled_domain"][method] = {
                    "status": "failed", "error": f"{type(error).__name__}: {error}"}
    return out


def main():
    report = {
        "status": "joint endpoint screening only; not validated or physically certified",
        "data": [
            "Pike and Bosher 1980 three manual mean EP traces, guinea pig.",
            "Asakuma and Snow 1980 60 mg/kg guinea pig effective partition resistance, ratio of group means 5.41/2.54.",
            "Rybak and Morizono 1982 Table III chinchilla EP and endolymph K reductions, 25, 50 and 100 mg/kg, n=3, 5 and 7.",
        ],
        "objective": [
            "Each Pike curve gets weight 1/100 mV^-2, normalized by number of anchors; points are correlated.",
            "Asakuma group-mean ratio residual uses illustrative scale 0.2, not a measured SE.",
            "Each Rybak EP and K reduction uses its reported SE as a scale; paired correlation, model discrepancy and uncertainty in timing are ignored; objective is not a likelihood.",
        ],
        "assumptions_and_limits": [
            "Asakuma ratio evaluated at 6 min is a historical hypothetical timing assumption, not the table-cohort time; source associates it with the EP nadir.",
            "Fixed series resistance fraction 0.5 assumes equal baseline series and parallel resistance; it is illustrative and not measured by this model.",
            "Rybak reductions use minima over 0-120 min, sampled every minute during fitting and every 0.25 min for reporting; source group nadir times vary, and the mean of individual minima need not equal the minimum of a group mean.",
            "Pike and Asakuma share dose proxy; Rybak has one estimated exposure multiplier, not measured local PK.",
            "Species share all other coefficients; volume coupling zero keeps endolymph volume fixed without identifying the water source.",
            "Closed or hypothetical drug-dependent K sink; neither equals an experimentally measured external K flux.",
            "Parameter bounds are numerical, not physiological; optimization is local; failed starts remain reported.",
            "Post-fit sample-grid domain checks through 120 min with Radau and BDF do not establish nonnegative states over the continuum, global stability, identifiable parameters, or independent predictive performance.",
            "18-state ion alternative excluded: charge, pump and water boundaries not physically closed.",
        ],
        "candidates": [],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    records = load_anchors()
    for combined, sink in product((False, True), (0.0, 0.02)):
        report["candidates"].append(fit(records, configuration(combined, sink)))
        OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"Completed {len(report['candidates'])}/4: {report['candidates'][-1]['status']}",
              flush=True)


if __name__ == "__main__":
    main()
