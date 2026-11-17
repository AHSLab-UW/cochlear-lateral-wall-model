"""One-at-a-time sensitivity checks for selected fixed structural parameters."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np

from src.model_equations import (
    Inputs,
    baseline,
    drug_concentration,
    related_code_closure,
    simulate,
)
from src.noise_scenarios import run_scenario
from src.parameter_freeze import FREEZE, load_freeze
from src.pike_trace_diagnostic import ROOT
from src.tier2_zhang_hou_benchmarks import run_zhang
from src.fit_parameters import OUTPUT as FIT_OUTPUT


OUTPUT = ROOT / "results" / "structural_sensitivity.json"
RELATIVE_CHANGES = (-0.20, 0.20)
STRUCTURAL_PARAMETERS = {
    "phi_C": "sync_volume",
    "phi_M": "marginal_volume",
    "v_S_0": "strial_volume_base",
    "q_K": "k_flux",
    "c_a": "atp_cleft_factor",
}
NOISE_LEVELS_DB_SPL = (105.0, 115.0, 125.0)
NOISE_SAMPLE_INTERVAL_MIN = 0.25
DRUG_DOSE_MG_KG = 80.0
DRUG_INFUSION_MIN = 0.542
DRUG_TOTAL_MIN = 240.0
DRUG_SAMPLE_INTERVAL_MIN = 0.25
SOLVER = {"method": "Radau", "rtol": 1e-8, "atol": 1e-10}
ZHANG_TONE_LEVELS_DB_SPL = (90,)
ZHANG_TONE_DRIVES = (1.076,)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _summarize_zhang(parameters):
    result = run_zhang(
        parameters,
        ZHANG_TONE_LEVELS_DB_SPL,
        ZHANG_TONE_DRIVES,
        method=SOLVER["method"],
        step_seconds=0.1,
    )
    trace = result["averages"][0]
    return {
        "tone_level_db_spl": ZHANG_TONE_LEVELS_DB_SPL[0],
        "outcome_informed_drive": ZHANG_TONE_DRIVES[0],
        "plateau_isp_mV": trace["plateau_isp_mV"],
        "plateau_ep_mV": trace["plateau_ep_mV"],
        "maximum_tone_evoked_delta_k_is_mM": trace["max_k_is_mM"],
    }


def _summarize_noise(parameters, initial_state):
    result = {}
    for level in NOISE_LEVELS_DB_SPL:
        scenario = run_scenario(
            level,
            parameters,
            initial_state,
            sample_interval_min=NOISE_SAMPLE_INTERVAL_MIN,
            include_time_trace=True,
        )
        trace = scenario["time_trace"]
        time = np.asarray(trace["time_min"])
        ep = np.asarray(trace["ep_mV"])
        marginal_k = np.asarray(trace["marginal_k_mM"])
        exposure = time <= scenario["input"]["exposure_stop_min"]
        recovery = time >= scenario["input"]["exposure_stop_min"]
        rebound_index = int(np.flatnonzero(recovery)[np.argmax(ep[recovery])])
        result[str(int(level))] = {
            "maximum_k_m_during_exposure_mM": float(np.max(marginal_k[exposure])),
            "minimum_ep_during_exposure_mV": scenario["metrics"]["minimum_ep_mV"],
            "post_offset_ep_peak_mV": float(ep[rebound_index]),
            "post_offset_ep_peak_time_min": float(time[rebound_index]),
            "ep_at_recovery_end_mV": scenario["metrics"]["ep_at_recovery_end_mV"],
        }
    return result


def _summarize_drug(parameters, dose_proxy):
    times = np.arange(
        0.0,
        DRUG_TOTAL_MIN + DRUG_SAMPLE_INTERVAL_MIN / 2,
        DRUG_SAMPLE_INTERVAL_MIN,
    )
    initial = baseline(parameters)

    def forcing(time):
        return Inputs(
            drug=drug_concentration(
                time, 0.0, DRUG_INFUSION_MIN, dose_proxy, parameters
            ),
            furosemide_protocol=True,
        )

    states = simulate(
        times,
        forcing,
        related_code_closure(),
        parameters,
        initial=initial,
        boundaries=(DRUG_INFUSION_MIN,),
        **SOLVER,
    )
    marginal_k = states[7]
    availability = np.asarray([
        1.0 / (1.0 + forcing(float(time)).drug ** parameters.drug_hill)
        for time in times
    ])
    return {
        "dose_mg_kg": DRUG_DOSE_MG_KG,
        "furosemide_exposure_proxy_peak": float(dose_proxy),
        "minimum_k_m_during_protocol_mM": float(np.min(marginal_k)),
        "time_of_minimum_k_m_min": float(times[np.argmin(marginal_k)]),
        "time_of_minimum_nkcc_availability_min": float(times[np.argmin(availability)]),
        "minimum_nkcc_availability": float(np.min(availability)),
    }


def _measurements(parameters, dose_proxy):
    initial = baseline(parameters)
    return {
        "zhang_90_db_tone": _summarize_zhang(parameters),
        "generic_broadband_noise": _summarize_noise(parameters, initial),
        "80_mg_kg_furosemide": _summarize_drug(parameters, dose_proxy),
    }


def build_report():
    parameters, freeze = load_freeze()
    fit = json.loads(FIT_OUTPUT.read_text())
    dose_proxy = (
        DRUG_DOSE_MG_KG
        * fit["drug_exposure_proxy"]["guinea_pig_proxy_per_mg_kg"]
    )
    reference = _measurements(parameters, dose_proxy)
    variants = []

    for symbol, field in STRUCTURAL_PARAMETERS.items():
        base_value = getattr(parameters, field)
        for relative_change in RELATIVE_CHANGES:
            changed_value = base_value * (1.0 + relative_change)
            changed_parameters = replace(parameters, **{field: changed_value})
            variants.append({
                "symbol": symbol,
                "parameter_field": field,
                "relative_change": relative_change,
                "reference_value": base_value,
                "changed_value": changed_value,
                "measurements": _measurements(changed_parameters, dose_proxy),
            })

    return {
        "status": "one-at-a-time fixed-structure sensitivity at the frozen fit",
        "reference_parameters": asdict(parameters),
        "fixed_parameters_varied": {
            symbol: {
                "implementation_field": field,
                "reference_value": getattr(parameters, field),
                "relative_changes": list(RELATIVE_CHANGES),
            }
            for symbol, field in STRUCTURAL_PARAMETERS.items()
        },
        "fit_parameter_treatment": (
            "All nine fitted parameters remain at their reference values; no refits "
            "are performed after changing a fixed structural parameter."
        ),
        "reference_measurements": reference,
        "variants": variants,
        "protocols": {
            "zhang_90_db_tone": {
                "description": (
                    "Four 10-s pulses at 90 dB SPL, with the outcome-informed "
                    "drive used for the separate Figure 4 cohort."
                ),
                "tone_drive": ZHANG_TONE_DRIVES[0],
                "requested_output_step_seconds": 0.1,
                "solver": SOLVER,
            },
            "generic_broadband_noise": {
                "description": (
                    "20-min model-only exposures followed by 100-min recovery; "
                    "not study-matched or independently validated."
                ),
                "levels_db_spl": list(NOISE_LEVELS_DB_SPL),
                "requested_output_step_min": NOISE_SAMPLE_INTERVAL_MIN,
                "solver": SOLVER,
            },
            "80_mg_kg_furosemide": {
                "description": (
                    "80-mg/kg guinea-pig dose proxy with a 0.542-min infusion and "
                    "240-min simulation; the boundary-exchange protocol flag remains on."
                ),
                "requested_output_step_min": DRUG_SAMPLE_INTERVAL_MIN,
                "solver": SOLVER,
            },
        },
        "interpretation": (
            "The perturbation magnitudes are analyst-selected numerical stress tests, "
            "not empirical confidence ranges or physiological bounds. These frozen-fit "
            "comparisons assess only the listed outputs and do not establish global "
            "robustness, parameter identifiability, uncertainty, or physiological validity."
        ),
        "source_sha256": {
            "freeze": _sha256(FREEZE),
            "fit": _sha256(FIT_OUTPUT),
            "model_equations": _sha256(ROOT / "src" / "model_equations.py"),
            "analysis_code": _sha256(Path(__file__)),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run one-at-a-time fixed-structure sensitivity checks."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT,
        help=f"new JSON output path (default: {OUTPUT})",
    )
    output = parser.parse_args().output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {output}")
    report = build_report()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        file.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(output, flush=True)


if __name__ == "__main__":
    main()
