"""Local and finite-perturbation checks for joint-screen fitted candidates."""

import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from src.assumption_comparison import Circuit
from src.cochlear_model import Inputs, baseline, Parameters, potassium_state
from src.transport_assumption_calibration import transport_rhs


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "exploratory"
CASES = (
    ("flow_low", 0, .9), ("oxygen_low", 1, .9), ("glucose_low", 2, .9),
    ("energy_sync_low", 3, .9), ("energy_marginal_low", 4, .9),
    ("volume_high", 5, 1.05), ("sync_k_high", 6, 1.01),
    ("intrastrial_k_high", 7, 1.05), ("marginal_k_high", 8, 1.01),
    ("voltage_low", 10, .9), ("atp_high", 11, 1.1),
)


def model(candidate):
    if candidate["status"] != "converged":
        raise ValueError("candidate must have a converged best fit")
    config, fitted = candidate["configuration"], candidate["parameters"]
    params = Parameters(potassium_flux=fitted["potassium_flux"],
                        drug_sink=config["drug_sink"], endolymph_volume_coupling=0)
    combined = config["combined_circuit"]
    circuit = Circuit(combined, combined, combined, fitted.get("met_conductance", .3),
                      fitted.get("drug_loss_fraction", .44), fitted.get("met_reversal_mV", -70.))
    return params, circuit


def branch_matrix(params, circuit, branch, step, *, derivative=transport_rhs):
    """Finite difference the appropriate smooth side of the substrate switch."""
    if branch not in ("glucose", "oxygen") or not 0 < step < 1e-2:
        raise ValueError("invalid branch or differentiation step")
    point = baseline(params)
    point[2 if branch == "glucose" else 1] -= 20 * step
    inputs = Inputs()
    base = derivative(0, point, inputs, params, circuit, "saturating")
    matrix = np.column_stack([
        (derivative(0, point + np.eye(14)[i] * step, inputs, params, circuit,
                    "saturating") - base) / step
        for i in range(14)
    ])
    # K inventory is conserved in the unforced closed-boundary system. A
    # zero eigenvalue for that coordinate is not asymptotic stability.
    return np.delete(np.delete(matrix, 9, axis=0), 9, axis=1)


def recovery_cases(params, circuit, *, duration=600, method="Radau", derivative=transport_rhs):
    """Test a fixed inventory and a finite set of single-state perturbations."""
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be positive and finite")
    reference = baseline(params)
    scales = np.maximum(np.abs(reference), 1.0)
    times = np.linspace(0, duration, 301)
    rows = []
    for label, index, multiplier in CASES:
        initial = reference.copy()
        initial[index] *= multiplier
        try:
            solution = solve_ivp(
                lambda t, y: derivative(t, y, Inputs(), params, circuit, "saturating"),
                (0, duration), initial, t_eval=times, method=method, rtol=1e-8, atol=1e-10)
            if not solution.success:
                raise RuntimeError(solution.message)
            distance = np.max(np.abs((solution.y.T - reference) / scales), axis=1)
            ke = [potassium_state(state, params).endolymph.concentration for state in solution.y.T]
            rows.append({"case": label, "status": "completed",
                         "initial_scaled_distance": float(distance[0]),
                         "final_scaled_distance": float(distance[-1]),
                         "maximum_scaled_distance": float(np.max(distance)),
                         "minimum_sampled_endolymph_k_mM": float(np.min(ke)),
                         "maximum_absolute_k_inventory_change": float(
                             np.max(np.abs(solution.y[9] - initial[9])))})
        except (ValueError, RuntimeError, FloatingPointError, OverflowError) as error:
            rows.append({"case": label, "status": "failed",
                         "error": f"{type(error).__name__}: {error}"})
    return rows


def inspect_candidate(candidate):
    params, circuit = model(candidate)
    reference_rates = transport_rhs(0, baseline(params), Inputs(), params, circuit, "saturating")
    spectra = {}
    for branch in ("glucose", "oxygen"):
        for step in (1e-5, 1e-6, 1e-7):
            spectra[f"{branch}_{step:g}"] = float(
                np.max(np.linalg.eigvals(branch_matrix(params, circuit, branch, step)).real))
    return {"configuration": candidate["configuration"],
            "parameters": candidate["parameters"],
            "reference_max_abs_rate": float(np.max(np.abs(reference_rates))),
            "branch_max_real_eigenvalues_per_minute": spectra,
            "recovery_cases": recovery_cases(params, circuit)}


def main():
    source = json.loads((RESULTS / "joint_calibration_screen.json").read_text())
    report = {
        "status": "limited fitted-candidate stability diagnostics, not a proof",
        "limitations": [
            "Unforced dynamics only; not stability of the time-varying drug response.",
            "The equilibrium is on a nonsmooth minimum-substrate switch; branch spectra do not prove switched-system stability.",
            "The total-K coordinate is removed for fixed-inventory spectra; full-space asymptotic stability cannot be inferred.",
            "Finite perturbations, 600-minute horizon and sample grid are not a basin-of-attraction or global positivity proof.",
            "The best fits are bound-limited, outcome-informed and scientifically provisional.",
        ],
        "candidates": [],
    }
    for candidate in source["candidates"]:
        if candidate["status"] == "converged":
            report["candidates"].append(inspect_candidate(candidate))
        else:
            report["candidates"].append({"configuration": candidate["configuration"],
                                         "status": "not analyzed: fit did not converge"})
    target = RESULTS / "joint_fit_stability.json"
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
