"""Outcome-informed alternatives for potassium exchange and volume dynamics."""

import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from src.cochlear_model import baseline, potassium_state
from src.joint_calibration_screen import configuration, parameter_space, sampled_domain
from src.joint_timing_sensitivity import temporal_residuals
from src.pike_trace_diagnostic import load_anchors
from src.transport_assumption_calibration import transport_rhs


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "exploratory"
OUTPUT = RESULTS / "joint_mechanism_comparison.json"


def mechanism_rhs(time, state, inputs, params, circuit, *, exchange, volume_gate):
    """Add a balanced M/E exchange and optionally gate the S volume derivative."""
    if not np.isfinite(exchange) or exchange < 0:
        raise ValueError("exchange coefficient must be finite and nonnegative")
    rates = transport_rhs(time, state, inputs, params, circuit, "saturating")
    ke = potassium_state(state, params).endolymph.concentration
    j_me = exchange * (state[8] / 100.0 - ke / 150.0)
    rates[8] -= j_me / params.marginal_volume
    if volume_gate:
        volume_rate = rates[5] * state[5] / params.strial_reference_volume
        rates[7] -= state[7] * (volume_rate - rates[5]) / state[5]
        rates[5] = volume_rate
    return rates


def make_simulator(*, exchange, volume_gate):
    """Match the existing transport simulator's public interface."""
    if not np.isfinite(exchange) or exchange < 0:
        raise ValueError("exchange coefficient must be finite and nonnegative")

    def simulate(times, forcing, params, circuit, uptake, *, method="Radau",
                 rtol=1e-6, atol=1e-8):
        if uptake != "saturating":
            raise ValueError("only the saturating uptake model is supported")
        t = np.asarray(times, dtype=float)
        if (t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all()
                or np.any(np.diff(t) <= 0)):
            raise ValueError("times must be finite and strictly increasing")
        sol = solve_ivp(
            lambda x, y: mechanism_rhs(x, y, forcing(x), params, circuit,
                                       exchange=exchange, volume_gate=volume_gate),
            (t[0], t[-1]), baseline(params), t_eval=t, method=method, rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(f"ODE integration failed: {sol.message}")
        return sol

    return simulate


def fit(records, *, include_exchange, volume_gate, control):
    """Refit the same six shared parameters and optional exchange coefficient."""
    config = configuration(True, 0.0)
    names, lower, upper, starts = parameter_space(config)
    prior = np.array([control["parameters"][name] for name in names])
    start_vectors = [prior, starts[1]]
    bounds = (np.log(lower).tolist(), np.log(upper).tolist())
    if include_exchange:
        bounds[0].append(0.)
        bounds[1].append(10.)
        start_vectors = [np.append(np.log(start), q) for start, q in zip(start_vectors, (.25, 2.))]
    else:
        start_vectors = [np.log(start) for start in start_vectors]

    def unpack(x):
        return np.exp(x[:len(names)]), float(x[-1]) if include_exchange else 0.

    def evaluate(x, *, grid_step=1., rtol=1e-6, atol=1e-8):
        values, exchange = unpack(x)
        simulator = make_simulator(exchange=exchange, volume_gate=volume_gate)
        return temporal_residuals(records, values, names, series_fraction=.5,
                                  resistance_time=6., grid_step=grid_step, rtol=rtol,
                                  atol=atol, simulator=simulator)

    attempts, completed = [], []
    for start in start_vectors:
        try:
            result = least_squares(lambda x: evaluate(x)[0], start, bounds=bounds,
                                   max_nfev=100, diff_step=1e-3,
                                   ftol=1e-6, xtol=1e-6, gtol=1e-6)
            values, exchange = unpack(result.x)
            attempts.append({"success": bool(result.success), "message": str(result.message),
                             "evaluations": int(result.nfev), "cost": float(result.cost),
                             "parameters": dict(zip(names, map(float, values))),
                             "exchange": exchange,
                             "active_bounds": result.active_mask.tolist()})
            if result.success:
                completed.append(result)
        except (ValueError, RuntimeError, FloatingPointError, OverflowError) as error:
            attempts.append({"success": False, "error": f"{type(error).__name__}: {error}"})
    output = {"include_exchange": include_exchange, "volume_gate": volume_gate,
              "bounds": dict(zip(names, [[float(a), float(b)] for a, b in zip(lower, upper)])),
              "exchange_bounds": [0., 10.] if include_exchange else [0., 0.],
              "attempts": attempts, "status": "converged" if completed else "no converged start"}
    if not completed:
        return output
    best = min(completed, key=lambda x: x.cost)
    values, exchange = unpack(best.x)
    output.update({"parameters": dict(zip(names, map(float, values))), "exchange": exchange,
                   "active_bounds": best.active_mask.tolist()})
    try:
        residual, predictions = evaluate(best.x, grid_step=.25, rtol=1e-8, atol=1e-10)
        output["weighted_sum_squares"] = float(np.dot(residual, residual))
        output["predictions"] = predictions
        output["sampled_domain"] = {}
        simulator = make_simulator(exchange=exchange, volume_gate=volume_gate)
        for method in ("Radau", "BDF"):
            try:
                output["sampled_domain"][method] = sampled_domain(
                    config, values, names, method=method, simulator=simulator)
            except (ValueError, RuntimeError, FloatingPointError, OverflowError) as error:
                output["sampled_domain"][method] = {
                    "status": "failed", "error": f"{type(error).__name__}: {error}"}
    except (ValueError, RuntimeError, FloatingPointError, OverflowError) as error:
        output["status"] = "post-fit evaluation failed"
        output["evaluation_error"] = f"{type(error).__name__}: {error}"
    return output


def main():
    source = json.loads((RESULTS / "joint_timing_sensitivity.json").read_text())
    matches = [row for row in source["scenarios"] if row["series_fraction"] == .5
               and row["resistance_time_min"] == 6 and row["status"] == "converged"]
    if len(matches) != 1:
        raise ValueError("expected one converged 0.5-series, 6-min control")
    report = {
        "status": "outcome-informed exploratory alternatives; not independent model selection",
        "control_source": "results/joint_timing_sensitivity.json",
        "control": matches[0],
        "methods": [
            "All candidates use the same timing-constrained objective, fixed 0.5 series fraction and six-minute resistance observation.",
            "The extra reversible M-to-E K amount flux is exchange*(K_M/100 - K_E/150), with opposite amount change in M and E; exchange has amount/time units.",
            "The optional volume gate multiplies the pre-existing S volume rate by S volume/0.1, so it leaves the reference equilibrium and its local volume derivative unchanged.",
            "The exchange has zero basal net flux and bounded optimization range 0-10; it is a phenomenological concentration-driven exchange, not a measured channel law or electrochemical flux.",
            "The gate is a phenomenological positive-volume hypothesis, not a measured water or pressure law; multiplying the entire rate changes swelling and relaxation away from baseline.",
            "Control is the previously fitted no-exchange ungated candidate; three alternatives receive two starts with identical shared bounds and tolerances.",
        ],
        "limitations": [
            "All structures and bounds are outcome-informed and numerical; objective is not a likelihood and cost cannot establish unbiased model preference.",
            "Neither new mechanism identifies an anatomical pathway, other charge-carrying ions, electrodiffusion, physical water boundary, or measured parameter.",
            "Volume gating alone does not prove positivity of all coupled concentrations, bounded coefficients, global solutions, or physiological volume.",
            "No new electrical conductance model is introduced; direct drug-dependent conductance loss remains phenomenological.",
            "Cross-species parameter sharing, timing mapping, arbitrary weight scales, incomplete covariance and sampled extrema carry forward unchanged.",
            "Failures and bound hits remain reportable; sparse multistart convergence and sampled two-solver checks do not prove global optimization or admissibility.",
        ],
        "candidates": [],
    }
    records = load_anchors()
    for exchange, gate in ((True, False), (False, True), (True, True)):
        report["candidates"].append(fit(records, include_exchange=exchange,
                                        volume_gate=gate, control=matches[0]))
        OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"Completed {len(report['candidates'])}/3: {report['candidates'][-1]['status']}",
              flush=True)


if __name__ == "__main__":
    main()
