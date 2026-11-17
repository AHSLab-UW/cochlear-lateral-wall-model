"""Model-only sound-input scenarios using the verified six-study freeze."""

import hashlib
import json
from pathlib import Path

import numpy as np

from src.model_equations import (
    STATE_NAMES,
    Inputs,
    broadband_sound_multiplier,
    endolymph,
    related_code_closure,
    simulate,
)
from src.parameter_freeze import FREEZE, load_freeze
from src.pike_trace_diagnostic import ROOT


OUTPUT = ROOT / "results/noise_scenarios.json"
SOUND_LEVELS_DB_SPL = (105.0, 115.0, 125.0)
EXPOSURE_MIN = 20.0
RECOVERY_MIN = 100.0
TOTAL_MIN = EXPOSURE_MIN + RECOVERY_MIN
SAMPLE_INTERVAL_MIN = 0.25
SOLVER = {"method": "Radau", "rtol": 1e-8, "atol": 1e-10}
PROTOCOL_SOURCE = "data/study_protocol.md"
PROTOCOL_STATUS = (
    "Generic model-only sound-input scenarios carried forward from the documented "
    "forecast grid; not a study-matched protocol or empirical Tier 2 test."
)


def _rounded(value):
    return float(np.round(float(value), 9))


def run_scenario(level_db_spl, parameters, initial_state,
                 sample_interval_min=SAMPLE_INTERVAL_MIN,
                 include_time_trace=False):
    """Run one fixed sound level followed by recovery; no outcomes are read."""
    if level_db_spl not in SOUND_LEVELS_DB_SPL:
        raise ValueError("sound level must be one of the prespecified scenario levels")
    if sample_interval_min <= 0:
        raise ValueError("sample interval must be positive")
    sample_count = round(TOTAL_MIN / sample_interval_min)
    if not np.isclose(sample_count * sample_interval_min, TOTAL_MIN):
        raise ValueError("sample interval must divide total duration")

    times = np.linspace(0.0, TOTAL_MIN, sample_count + 1)
    sound_drive = broadband_sound_multiplier(level_db_spl)

    def forcing(time):
        return Inputs(sound_multiplier=sound_drive if time < EXPOSURE_MIN else 1.0)

    trajectory = simulate(
        times,
        forcing,
        related_code_closure(),
        parameters,
        initial=np.asarray(initial_state, dtype=float),
        boundaries=(EXPOSURE_MIN,),
        **SOLVER,
    )
    index_at = {minute: int(np.flatnonzero(np.isclose(times, minute))[0])
                for minute in (0.0, EXPOSURE_MIN, TOTAL_MIN)}
    states = {
        f"t_{int(minute)}_min": {
            name: _rounded(trajectory[index, index_at[minute]])
            for index, name in enumerate(STATE_NAMES)
        }
        for minute in (0.0, EXPOSURE_MIN, TOTAL_MIN)
    }
    ep = trajectory[8]
    minimum_index = int(np.argmin(ep))
    minimum_state = trajectory[:, minimum_index]
    _, k_at_exposure_end = endolymph(trajectory[:, index_at[EXPOSURE_MIN]], parameters)
    _, k_at_recovery_end = endolymph(trajectory[:, index_at[TOTAL_MIN]], parameters)
    metrics = {
        "minimum_ep_mV": _rounded(ep[minimum_index]),
        "time_of_minimum_ep_min": _rounded(times[minimum_index]),
        "ep_at_exposure_end_mV": _rounded(trajectory[8, index_at[EXPOSURE_MIN]]),
        "ep_at_recovery_end_mV": _rounded(trajectory[8, index_at[TOTAL_MIN]]),
        "endolymph_k_at_exposure_end_mM": _rounded(k_at_exposure_end),
        "endolymph_k_at_recovery_end_mM": _rounded(k_at_recovery_end),
        "minimum_flow": _rounded(np.min(trajectory[0])),
        "maximum_trauma_state": _rounded(np.max(trajectory[12])),
        "trauma_state_at_minimum_ep": _rounded(minimum_state[12]),
    }
    result = {
        "scenario_id": f"sound_{int(level_db_spl)}_dB_SPL",
        "input": {
            "broadband_spl_db": level_db_spl,
            "model_sound_multiplier": _rounded(sound_drive),
            "exposure_start_min": 0.0,
            "exposure_stop_min": EXPOSURE_MIN,
            "recovery_min": RECOVERY_MIN,
            "total_simulation_min": TOTAL_MIN,
        },
        "output_states": states,
        "metrics": metrics,
    }
    if include_time_trace:
        endolymph_k = [
            _rounded(endolymph(trajectory[:, index], parameters)[1])
            for index in range(len(times))
        ]
        result["time_trace"] = {
            "time_min": [_rounded(time) for time in times],
            "ep_mV": [_rounded(value) for value in trajectory[8]],
            "marginal_k_mM": [_rounded(value) for value in trajectory[7]],
            "endolymph_k_mM": endolymph_k,
            "trauma_state": [_rounded(value) for value in trajectory[12]],
        }
    return result


def build_result(sample_interval_min=SAMPLE_INTERVAL_MIN, include_time_traces=False):
    parameters, freeze = load_freeze()
    initial_state = np.asarray(freeze["initial_state"], dtype=float)
    scenarios = [
        run_scenario(
            level, parameters, initial_state, sample_interval_min,
            include_time_trace=include_time_traces,
        )
        for level in SOUND_LEVELS_DB_SPL
    ]
    return {
        "status": (
            "model-only, non-matched input-scenario simulation with time traces"
            if include_time_traces else
            "model-only, non-matched input-scenario simulation"
        ),
        "candidate": freeze["candidate"],
        "freeze_path": str(FREEZE.relative_to(ROOT)),
        "freeze_sha256": hashlib.sha256(FREEZE.read_bytes()).hexdigest(),
        "fit_sha256": freeze["fit_sha256"],
        "model_path": freeze["model_source"],
        "model_sha256": freeze["model_sha256"],
        "candidate_freeze_verified_by": "src.parameter_freeze.load_freeze()",
        "protocol_source": PROTOCOL_SOURCE,
        "protocol_status": PROTOCOL_STATUS,
        "time_trace_included": include_time_traces,
        "protocols": scenarios,
        "solver": {
            **SOLVER,
            "sample_interval_min": sample_interval_min,
            "forcing_boundary_min": [EXPOSURE_MIN],
        },
        "held_out_comparison_eligible": False,
        "limitations": [
            "The generic scenarios were not selected or tuned using Tier 2 outcomes. However, an initial broad view of a historical benchmark-protocol document inadvertently exposed outcome text during preparation, so the strict no-inspection condition was not fully met.",
            "No observed Tier 2 content informed the model-only simulation inputs or parameter values.",
            "No protocol is certified as an untouched Tier 2 study; this generic scenario grid is not a real held-out comparison.",
            "Sound levels and timing follow the documented model-only forecast grid, not a separable outcome-blind empirical protocol.",
            "The frozen nominal initial state is used without outcome-based normalization; parameters are loaded unchanged and not refit.",
            "These simulations are computational outputs, not independent validation or evidence of predictive success.",
        ],
    }


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {OUTPUT}")
    result = build_result()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8") as file:
        file.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
