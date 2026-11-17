"""Outcome-informed factorial sensitivity analysis, not model selection."""

import csv
from dataclasses import dataclass
from itertools import product
import json
import math
from pathlib import Path
from collections.abc import Callable, Sequence

import numpy as np
from scipy.integrate import solve_ivp

from src.cochlear_model import Inputs, Parameters, baseline, potassium_state, rhs
from src.pike_ode_fit import drug_proxy
from src.pike_trace_diagnostic import IDS, load_anchors
from src.two_diffusion_diagnostic import FARADAY, GAS_CONSTANT, TEMPERATURE_K


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "results" / "exploratory"
reference_LOAD = 0.30
reference_REVERSAL_MV = -70.0
reference_DRUG_LOSS = 0.44


@dataclass(frozen=True)
class Circuit:
    """Illustrative switches, not independently established circuit parameters."""

    apical_battery: bool = False
    met_load: bool = False
    drug_conductance_loss: bool = False
    met_conductance: float = reference_LOAD
    drug_loss_fraction: float = reference_DRUG_LOSS
    met_reversal_mV: float = reference_REVERSAL_MV

    def __post_init__(self) -> None:
        if not math.isfinite(self.met_conductance) or self.met_conductance < 0:
            raise ValueError("MET conductance must be finite and nonnegative")
        if not math.isfinite(self.drug_loss_fraction) or not 0 <= self.drug_loss_fraction < 1:
            raise ValueError("drug conductance loss fraction must be in [0, 1)")
        if not math.isfinite(self.met_reversal_mV) or self.met_reversal_mV >= 80.0:
            raise ValueError("MET reversal must be finite and below the 80 mV reference")


def alternative_rhs(time: float, state: Sequence[float], inputs: Inputs,
                    parameters: Parameters, circuit: Circuit) -> np.ndarray:
    """Replace only the EP current balance; all other prototype states are shared."""
    rates = rhs(time, state, inputs, parameters)
    if not (circuit.apical_battery or circuit.met_load or circuit.drug_conductance_loss):
        return rates

    y = np.asarray(state)
    kc, ks, km, ep, am = y[6], y[7], y[8], y[10], y[4]
    source = parameters.baseline_source + 20.0 * math.log((kc / ks) / 20.0)
    if circuit.apical_battery:
        ke = potassium_state(y, parameters).endolymph.concentration
        scale = 1000 * GAS_CONSTANT * TEMPERATURE_K / FARADAY
        source += scale * math.log((km / ke) / (100.0 / 150.0))
    load = circuit.met_conductance if circuit.met_load else 0.0
    source += load * (parameters.baseline_source - circuit.met_reversal_mV)
    # Remove and replace only the original unit strial conductance; retain
    # the prototype's ATP/sound-dependent shunts.
    loss = circuit.drug_loss_fraction * inputs.drug if circuit.drug_conductance_loss else 0.0
    rates[10] += (
        loss * (ep - source * am)
        + (source - (parameters.baseline_source + 20.0 * math.log((kc / ks) / 20.0))) * am
        - load * max(0.0, ep - circuit.met_reversal_mV)
    ) / parameters.voltage_tau
    return rates


def simulate_alternative(times: Sequence[float], forcing: Callable[[float], Inputs],
                         parameters: Parameters, circuit: Circuit, *,
                         method: str = "Radau", rtol: float = 1e-7,
                         atol: float = 1e-9):
    """Integrate the alternative circuit with the shared baseline and forcing."""
    t = np.asarray(times, dtype=float)
    if t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("times must be finite and strictly increasing")
    solution = solve_ivp(
        lambda time, y: alternative_rhs(time, y, forcing(time), parameters, circuit),
        (t[0], t[-1]), baseline(parameters), t_eval=t, method=method, rtol=rtol, atol=atol,
    )
    if not solution.success:
        raise RuntimeError(f"ODE integration failed: {solution.message}")
    return solution


def compare(records: list[dict], fit_parameters: dict, *, method: str = "Radau",
            rtol: float = 1e-7, atol: float = 1e-9) -> list[dict]:
    """Evaluate every factorial combination at the same retained observations."""
    groups = {}
    for identifier in IDS:
        group = [row for row in records if row["curve"] == identifier]
        if len(group) < 2 or group[0]["time_min"] != 0:
            raise ValueError(f"{identifier} needs observations starting at zero")
        if any(row["dose_mg_kg"] != group[0]["dose_mg_kg"] for row in group):
            raise ValueError(f"inconsistent dose for {identifier}")
        groups[identifier] = group

    flux = float(fit_parameters["potassium_flux"])
    dose_scale = float(fit_parameters["effective_dose_scale_mg_kg"])
    decay = float(fit_parameters["exposure_decay_min"])
    rows = []
    for apical, met, loss, coupling, sink in product(
            (False, True), (False, True), (False, True), (0.0, 1.0), (0.0, 0.02)):
        circuit = Circuit(apical, met, loss)
        parameters = Parameters(potassium_flux=flux, drug_sink=sink,
                                endolymph_volume_coupling=coupling)
        for identifier, group in groups.items():
            dose = float(group[0]["dose_mg_kg"])
            forcing = lambda t: Inputs(drug=drug_proxy(t, dose, dose_scale, decay))
            solution = simulate_alternative([r["time_min"] for r in group], forcing,
                                            parameters, circuit, method=method,
                                            rtol=rtol, atol=atol)
            predicted = solution.y[10]
            residual = predicted - np.array([r["ep_mV"] for r in group])
            rows.append({
                "apical_battery": apical, "met_load": met,
                "drug_conductance_loss": loss, "volume_coupling": coupling,
                "drug_sink": sink, "curve": identifier, "dose_mg_kg": dose,
                "anchor_count": len(group),
                "rmse_mV": float(np.sqrt(np.mean(residual ** 2))),
                "minimum_at_anchor_times_mV": float(np.min(predicted)),
                "minimum_endolymph_k_at_anchor_times_mM": float(min(
                    potassium_state(y, parameters).endolymph.concentration
                    for y in solution.y.T
                )),
            })
    return rows


def main() -> None:
    fit = json.loads((OUTPUT / "pike_ode_fit.json").read_text())
    records = load_anchors()
    rows = compare(records, fit["parameters"])
    solver_rows = compare(records, fit["parameters"], method="BDF",
                          rtol=1e-9, atol=1e-11)
    if len(rows) != len(solver_rows):
        raise RuntimeError("solver comparisons produced different numbers of cases")
    max_rmse_difference = max(abs(a["rmse_mV"] - b["rmse_mV"])
                              for a, b in zip(rows, solver_rows))
    max_minimum_difference = max(
        abs(a["minimum_at_anchor_times_mV"] - b["minimum_at_anchor_times_mV"])
        for a, b in zip(rows, solver_rows)
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "assumption_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "status": "outcome-informed fixed-parameter sensitivity; not model selection or validation",
        "design": "2x2x2 circuit switches x 2 fluid closures x 2 K-boundary assumptions x 3 doses",
        "shared_parameters_from": "pike_ode_fit.json, fitted only to original circuit",
        "shared_parameters": fit["parameters"],
        "conditional_coefficients": {
            "met_load_relative_conductance": reference_LOAD,
            "met_reversal_mV": reference_REVERSAL_MV,
            "maximum_drug_conductance_loss_fraction": reference_DRUG_LOSS,
        },
        "solver_comparison": {
            "radau_tolerances": {"rtol": 1e-7, "atol": 1e-9},
            "bdf_tolerances": {"rtol": 1e-9, "atol": 1e-11},
            "max_rmse_difference_mV": max_rmse_difference,
            "max_anchor_minimum_difference_mV": max_minimum_difference,
        },
        "limitations": [
            "These coefficients are illustrative reference-derived assumptions, not measured here.",
            "The original circuit alone was previously fitted; alternatives were not refitted.",
            "RMSEs are descriptive and cannot fairly rank or select the alternatives.",
            "Apical term uses concentrations as activity proxies; syncytial term is unchanged.",
            "The circuit is an adaptation, not a reconstruction of the reference model.",
            "Fixed-endolymph closure leaves the source of strial water unspecified.",
            "The drug sink is a hypothetical external K loss, not an observed flux.",
            "Minima and positivity are checked at observation times, not over continuous paths.",
            "All outcomes were already known; no held-out or blinded evaluation is performed.",
            "No uncertainty, identifiability or stability analysis is established.",
        ],
        "rows": len(rows),
    }
    (OUTPUT / "assumption_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
