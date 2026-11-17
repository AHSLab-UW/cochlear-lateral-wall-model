"""Explore Nin's two-diffusion-potential equation on existing ODE trajectories.

This is a postprocessed hypothetical readout, not feedback into the ODE,
nor a replacement for experimental ion-activity measurements.
"""

import csv
import json
import math
from pathlib import Path

import numpy as np

from src.cochlear_model import Inputs, Parameters, baseline, potassium_state, simulate
from src.pike_ode_fit import drug_proxy
from src.pike_trace_diagnostic import IDS, load_anchors


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "results" / "exploratory"
GAS_CONSTANT = 8.31446261815324
FARADAY = 96485.33212
TEMPERATURE_K = 310.15


def two_potential_ep(k_sync: float, k_strial: float, k_marginal: float,
                     k_endolymph: float, offset_mv: float,
                     temperature_k: float = TEMPERATURE_K) -> float:
    """Approximate EP in mV, using concentrations as activity proxies."""
    values = (k_sync, k_strial, k_marginal, k_endolymph, temperature_k)
    if not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError("potassium concentrations and temperature must be positive and finite")
    if not math.isfinite(offset_mv):
        raise ValueError("offset must be finite")
    scale = 1000 * GAS_CONSTANT * temperature_k / FARADAY
    return offset_mv + scale * (
        math.log(k_sync / k_strial) + math.log(k_marginal / k_endolymph)
    )


def run_diagnostic(fitted: dict) -> list[dict]:
    """Evaluate the alternative readout on the previously fitted ODE paths."""
    flux = fitted["potassium_flux"]
    dose_scale = fitted["effective_dose_scale_mg_kg"]
    decay = fitted["exposure_decay_min"]
    parameters = Parameters(potassium_flux=flux, drug_sink=0)
    reference = baseline()
    initial_ke = potassium_state(reference, parameters).endolymph.concentration
    raw_reference = two_potential_ep(reference[6], reference[7], reference[8],
                                     initial_ke, 0)
    offset = reference[10] - raw_reference
    records = load_anchors()
    rows = []
    for identifier in IDS:
        group = [record for record in records if record["curve"] == identifier]
        if len(group) < 2:
            raise ValueError(f"missing observations for {identifier}")
        dose = group[0]["dose_mg_kg"]
        solution = simulate(
            [row["time_min"] for row in group],
            lambda t: Inputs(drug=drug_proxy(t, dose, dose_scale, decay)),
            parameters=parameters, rtol=1e-8, atol=1e-10,
        )
        for record, state in zip(group, solution.y.T):
            ke = potassium_state(state, parameters).endolymph.concentration
            alternative = two_potential_ep(state[6], state[7], state[8], ke, offset)
            rows.append({**record, "original_ep_mV": float(state[10]),
                         "two_diffusion_ep_mV": alternative,
                         "marginal_k_mM": float(state[8]), "endolymph_k_mM": ke})
    return rows


def main() -> None:
    fit = json.loads((OUTPUT / "pike_ode_fit.json").read_text())
    rows = run_diagnostic(fit["parameters"])
    with (OUTPUT / "two_diffusion_diagnostic.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "post-hoc alternate readout; not a new calibrated ODE or prediction",
        "source": "Nin et al. 2008, PNAS 105:1751-1756, doi:10.1073/pnas.0711463105, p.1755",
        "temperature_K": TEMPERATURE_K,
        "assumptions": [
            "concentrations used as activity proxies; syncytial reference and offsets held constant",
            "baseline additive offset selected to reproduce the prototype's 80 mV reference",
            "readout uses trajectories from the original fitted model and does not feed back",
            "ATP and sound-dependent shunts are not represented in this alternate readout",
            "drug exposure model and underlying fitted ODE parameters remain hypothetical and bound-limited",
        ],
        "minimum_original_ep_mV_by_curve": {
            key: min(row["original_ep_mV"] for row in rows if row["curve"] == key)
            for key in IDS
        },
        "minimum_alternate_ep_mV_by_curve": {
            key: min(row["two_diffusion_ep_mV"] for row in rows if row["curve"] == key)
            for key in IDS
        },
        "limitation": "does not establish the true circuit, measured ion activities, or solve the drug fit",
    }
    (OUTPUT / "two_diffusion_diagnostic.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
