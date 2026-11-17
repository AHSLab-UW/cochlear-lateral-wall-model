"""Outcome-informed geometry and K-exchange refits of the joint objective."""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from src.assumption_comparison import Circuit
from src.cochlear_model import Inputs, Parameters, baseline, potassium_state
from src.joint_calibration_screen import configuration, parameter_space, sampled_domain
from src.joint_fit_stability import branch_matrix, recovery_cases
from src.joint_mechanism_comparison import make_simulator, mechanism_rhs
from src.joint_timing_sensitivity import temporal_residuals
from src.pike_trace_diagnostic import load_anchors


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "exploratory"
OUTPUT = RESULTS / "reference_geometry_comparison.json"


def fit_case(records, names, lower, upper, starts, reference, include_exchange, *,
             syncytial_volume=.2, marginal_volume=.3, hill=2.6, water_rate=.005):
    """Use identical starts, bounds and objective for each geometry/mechanism pair."""
    bounds = (np.log(lower), np.log(upper))
    if include_exchange:
        bounds = (np.append(bounds[0], 0.), np.append(bounds[1], 10.))

    def unpack(x):
        return np.exp(x[:len(names)]), float(x[-1]) if include_exchange else 0.

    def evaluate(x, **kwargs):
        values, exchange = unpack(x)
        return temporal_residuals(
            records, values, names, series_fraction=.5, resistance_time=6.,
            strial_reference_volume=reference,
            syncytial_volume=syncytial_volume, marginal_volume=marginal_volume,
            hill=hill, water_rate=water_rate,
            simulator=make_simulator(exchange=exchange, volume_gate=False), **kwargs)

    row = {"strial_reference_volume": reference, "syncytial_volume": syncytial_volume,
           "marginal_volume": marginal_volume, "hill": hill, "water_rate": water_rate,
           "include_exchange": include_exchange,
           "status": "no converged start", "attempts": []}
    complete = []
    for label, values, exchange in starts:
        x0 = np.log(values)
        if include_exchange:
            x0 = np.append(x0, exchange)
        start_values, start_exchange = unpack(x0)
        attempt = {"start_label": label, "start_parameters": dict(zip(names, map(float, start_values))),
                   "start_exchange": start_exchange}
        try:
            initial_residual, _ = evaluate(x0)
            attempt["initial_weighted_sum_squares"] = float(initial_residual @ initial_residual)
            result = least_squares(lambda x: evaluate(x)[0], x0, bounds=bounds,
                                   max_nfev=100, diff_step=1e-3,
                                   ftol=1e-6, xtol=1e-6, gtol=1e-6)
            fitted, exchange = unpack(result.x)
            attempt.update({"success": bool(result.success), "message": str(result.message),
                            "evaluations": int(result.nfev), "cost": float(result.cost),
                            "parameters": dict(zip(names, map(float, fitted))),
                            "exchange": exchange,
                            "active_bounds": dict(zip([*names, *(["exchange"] if include_exchange else [])],
                                                      map(int, result.active_mask)))})
            if result.success:
                complete.append(result)
        except (ValueError, RuntimeError, FloatingPointError, OverflowError) as error:
            attempt.update({"success": False, "error": f"{type(error).__name__}: {error}"})
        row["attempts"].append(attempt)
    if not complete:
        return row
    best = min(complete, key=lambda x: x.cost)
    values, exchange = unpack(best.x)
    row.update({"parameters": dict(zip(names, map(float, values))), "exchange": exchange,
                "active_bounds": dict(zip([*names, *(["exchange"] if include_exchange else [])],
                                          map(int, best.active_mask)))})
    try:
        residual, predictions = evaluate(best.x, grid_step=.25, rtol=1e-8, atol=1e-10)
        row["weighted_sum_squares"] = float(residual @ residual)
        row["predictions"] = predictions
        row["status"] = "converged"
    except (ValueError, RuntimeError, FloatingPointError, OverflowError) as error:
        row.update({"status": "post-fit evaluation failed",
                    "evaluation_error": f"{type(error).__name__}: {error}"})
        return row
    simulator = make_simulator(exchange=exchange, volume_gate=False)
    row["sampled_domain"] = {}
    for method in ("Radau", "BDF"):
        try:
            row["sampled_domain"][method] = sampled_domain(
                configuration(True, 0), values, names, method=method, simulator=simulator,
                strial_reference_volume=reference, syncytial_volume=syncytial_volume,
                marginal_volume=marginal_volume, hill=hill, water_rate=water_rate)
        except (ValueError, RuntimeError, FloatingPointError, OverflowError) as error:
            row["sampled_domain"][method] = {"status": "failed",
                                               "error": f"{type(error).__name__}: {error}"}

    params = Parameters(potassium_flux=float(values[names.index("potassium_flux")]),
                        drug_sink=0, endolymph_volume_coupling=0,
                        strial_reference_volume=reference, syncytial_volume=syncytial_volume,
                        marginal_volume=marginal_volume, water_rate=water_rate)
    v = dict(zip(names, values))
    circuit = Circuit(True, True, True, v["met_conductance"], v["drug_loss_fraction"])

    def derivative(t, state, inputs, params, circuit, uptake):
        if uptake != "saturating":
            raise ValueError("only saturating uptake is supported")
        return mechanism_rhs(t, state, inputs, params, circuit,
                             exchange=exchange, volume_gate=False)

    try:
        row["stability"] = {
            "reference_max_abs_rate": float(np.max(np.abs(
                derivative(0, baseline(params), Inputs(), params, circuit, "saturating")))),
            "branch_max_real_eigenvalues_per_minute": {
                branch: float(np.max(np.linalg.eigvals(
                    branch_matrix(params, circuit, branch, 1e-6, derivative=derivative)).real))
                for branch in ("glucose", "oxygen")},
            "recovery_cases": recovery_cases(params, circuit, derivative=derivative),
        }
    except (ValueError, RuntimeError, FloatingPointError, OverflowError) as error:
        row["stability"] = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
    row["baseline_endolymph_k_mM"] = potassium_state(baseline(params), params).endolymph.concentration
    return row


def main():
    source = json.loads((RESULTS / "joint_combined_sensitivity.json").read_text())
    if source["status"] != "converged" or source["control"]["status"] != "converged":
        raise ValueError("both prior starts must be converged")
    names, lower, upper, _ = parameter_space(configuration(True, 0))
    upper[names.index("drug_loss_fraction")] = .98
    starts = [
        ("combined", np.array([source["parameters"][name] for name in names]),
         source["parameters"]["exchange"]),
        ("no_exchange", np.array([source["control"]["parameters"][name] for name in names]), 0.),
    ]
    records = load_anchors()
    report = {
        "status": "in progress; outcome-informed geometry sensitivity",
        "source": "results/joint_combined_sensitivity.json",
        "bounds": {**dict(zip(names, [[float(a), float(b)] for a, b in zip(lower, upper)])),
                   "exchange": [0., 10.]},
        "limitations": [
            "0.02 and 0.04 satisfy only the strial-space-alone upper bound, not the combined S and M bound; they are illustrative, not measured. 0.1 is the original effective-volume control.",
            "Only the S reference changes: C and M normalized volumes, basal concentrations, amount fluxes, absolute normalized water coefficient, and all other parameter bounds are held fixed; alternate rescalings remain untested.",
            "The same reference is shared across guinea-pig and chinchilla by assumption; the whole-stria geometric bound is from chinchilla group means and does not anchor guinea-pig anatomy.",
            "E volume is fixed and its water source unspecified; the ungated water law is phenomenological, and positive sampled states do not establish boundary invariance.",
            "The objective is not a likelihood, the model and prior starts are outcome-informed, and two local starts do not establish global optima or unbiased model preference.",
            "Branch spectra and finite perturbations are unforced fixed-inventory local checks, not global or nonsmooth stability proofs.",
            "The Santi local swelling endpoint is not fitted; no bumetanide exposure model or independent prediction is provided.",
            "K exchange is not electrochemical or charge-complete; other physical closure and observation-model limitations remain.",
        ],
        "cases": [],
    }
    for reference in (.1, .04, .02):
        for exchange in (False, True):
            report["cases"].append(fit_case(records, names, lower, upper, starts, reference, exchange))
            OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(f"reference={reference:g}, exchange={exchange}: {report['cases'][-1]['status']}", flush=True)
    report["status"] = "completed outcome-informed geometry sensitivity; not independent validation or model selection"
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
