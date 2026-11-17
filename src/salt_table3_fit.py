"""Conditionally add Salt & Konishi Table III to the frozen reference noise fit."""

from dataclasses import asdict
import hashlib
import json

import numpy as np
from scipy.optimize import least_squares

from src.candidate_freeze import FREEZE, load_freeze
from src.model_equations import (
    Inputs,
    broadband_sound_multiplier,
    related_code_closure,
    simulate,
)
from src.drug_calibration import evaluate as evaluate_drug
from src.scaled_parameter_fit import bounded_jacobian
from src.kinetics_fit import ALL_NAMES
from src.wang_noise_cohorts import (
    OUTPUT as WANG_FIT,
    combined as evaluate_wang,
)
from src.wang_noise_traces import LOWER, NAMES, UPPER, load_noise, make_parameters
from src.pike_trace_diagnostic import ROOT, SOURCE, load_anchors


OUTPUT = ROOT / "results" / "exploratory" / "salt_table3_fit.json"
SALT_PAPER = ROOT / "refs" / "target_source" / "salt_1979_hear_res.pdf"
WANG1990_PAPER = ROOT / "refs" / "target_source" / "wang_1990_hear_res.pdf"
WANG1992_PAPER = ROOT / "refs" / "target_source" / "wang_1992_hear_res.pdf"

EXPOSURE_MIN = 20.0
QUIET_INTERVAL_MIN = 5.0
MEASUREMENT_START_AFTER_MIN = 1.0
MEASUREMENT_DURATION_MIN = 2.0
EXPOSURES = 3
SALT_WEIGHT_SCALE_MV = 3.0

# These are the basal-turn ΔEPo entries only.  The Table III n column is
# described by the authors as a number of observations, not independent animals.
SALT_TARGETS = (
    {"noise_dba": 95, "exposure_number": 1, "mean_mV": 0.1, "sd_mV": 1.1, "n_observations": 3},
    {"noise_dba": 95, "exposure_number": 2, "mean_mV": -0.4, "sd_mV": 0.8, "n_observations": 3},
    {"noise_dba": 95, "exposure_number": 3, "mean_mV": -0.8, "sd_mV": 0.8, "n_observations": 3},
    {"noise_dba": 105, "exposure_number": 1, "mean_mV": 0.4, "sd_mV": 2.3, "n_observations": 3},
    {"noise_dba": 105, "exposure_number": 2, "mean_mV": -0.5, "sd_mV": 0.5, "n_observations": 3},
    {"noise_dba": 105, "exposure_number": 3, "mean_mV": -1.4, "sd_mV": 1.3, "n_observations": 3},
    {"noise_dba": 115, "exposure_number": 1, "mean_mV": 5.1, "sd_mV": 2.9, "n_observations": 9},
    {"noise_dba": 115, "exposure_number": 2, "mean_mV": -0.5, "sd_mV": 1.0, "n_observations": 9},
    {"noise_dba": 115, "exposure_number": 3, "mean_mV": -0.4, "sd_mV": 1.1, "n_observations": 7},
)

MAPPING_SCENARIOS = (
    {
        "name": "primary_same_numeric_level",
        "illustrative_offset_db": 0,
        "interpretation": (
            "Conditionally assign each dBA number to the same numerical dB SPL "
            "value. This is an illustrative fixed input assumption, not a conversion."
        ),
    },
    {
        "name": "alternate_minus_10_db",
        "illustrative_offset_db": -10,
        "interpretation": (
            "Illustrative fixed -10 dB offset from the same-number mapping; not a "
            "justified conversion, correction, or uncertainty bound."
        ),
    },
    {
        "name": "alternate_plus_10_db",
        "illustrative_offset_db": 10,
        "interpretation": (
            "Illustrative fixed +10 dB offset from the same-number mapping; not a "
            "justified conversion, correction, or uncertainty bound."
        ),
    },
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frozen_drug_values(parameters, freeze):
    proxy = freeze["drug_exposure_proxy"]
    values = [
        proxy["guinea_pig_proxy_per_mg_kg"],
        proxy["chinchilla_exposure_multiplier"],
        parameters.drug_conductance_loss,
        parameters.drug_k_loss,
        parameters.drug_slow_fraction,
        parameters.drug_slow,
    ]
    if len(ALL_NAMES) != len(values):
        raise ValueError("frozen drug vector does not match the existing fit schema")
    return np.asarray(values, dtype=float)


def input_scenarios():
    """Return fixed acoustic-input assumptions, none of which is fitted."""
    scenarios = []
    for spec in MAPPING_SCENARIOS:
        offset = spec["illustrative_offset_db"]
        scenarios.append({
            **spec,
            "levels": [
                {
                    "noise_dba": level,
                    "conditional_model_level_db_spl": level + offset,
                    "sound_multiplier": float(broadband_sound_multiplier(level + offset)),
                }
                for level in (95, 105, 115)
            ],
        })
    return scenarios


def simulate_salt_level(sound_values, drug_values, initial_state, level_dba, *,
                        offset_db=0, method="Radau", rtol=2e-6, atol=2e-8):
    """Return per-episode basal EP changes using a repeated-exposure operator.

    For each 20-minute exposure, the reported model change is the trapezoidal
    mean EP over minutes 1-3 after sound offset minus EP immediately before
    that episode. The state is carried through two 5-minute quiet intervals.
    """
    level_spl = float(level_dba + offset_db)
    if not np.isfinite(level_spl):
        raise ValueError("conditional model sound level must be finite")
    p = make_parameters(sound_values, drug_values)
    closure = related_code_closure()
    starts = [index * (EXPOSURE_MIN + QUIET_INTERVAL_MIN) for index in range(EXPOSURES)]
    ends = [start + EXPOSURE_MIN for start in starts]
    samples = [0.0]
    for end in ends:
        samples.extend((end, end + 1.0, end + 2.0, end + 3.0))
    samples.extend(starts[1:])
    times = np.asarray(sorted(set(samples)), dtype=float)
    boundaries = sorted(set(ends[:-1] + starts[1:] + [ends[-1]]))
    multiplier = broadband_sound_multiplier(level_spl)

    def forcing(time):
        active = any(start <= time < end for start, end in zip(starts, ends))
        return Inputs(sound_multiplier=multiplier if active else 1.0)

    states = simulate(
        times,
        forcing,
        closure,
        p,
        initial=np.asarray(initial_state, dtype=float),
        boundaries=boundaries,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    observations = []
    for number, (start, end) in enumerate(zip(starts, ends), start=1):
        window = np.asarray([end + 1.0, end + 2.0, end + 3.0])
        indices = [int(np.flatnonzero(np.isclose(times, t))[0]) for t in window]
        start_index = int(np.flatnonzero(np.isclose(times, start))[0])
        pre_ep = float(states[8, start_index])
        window_ep = states[8, indices]
        window_mean = float(np.trapz(window_ep, window) / MEASUREMENT_DURATION_MIN)
        observations.append({
            "noise_dba": int(level_dba),
            "conditional_model_level_db_spl": float(level_spl),
            "sound_multiplier": float(multiplier),
            "exposure_number": number,
            "exposure_start_min": float(start),
            "exposure_end_min": float(end),
            "measurement_window_min": [float(window[0]), float(window[-1])],
            "pre_exposure_ep_mV": pre_ep,
            "measurement_window_mean_ep_mV": window_mean,
            "predicted_overall_ep_change_mV": window_mean - pre_ep,
            "method": method,
        })
    return observations


def salt_predictions(sound_values, drug_values, initial_state, *, offset_db=0,
                     method="Radau", rtol=2e-6, atol=2e-8):
    rows = []
    for level in (95, 105, 115):
        rows.extend(simulate_salt_level(
            sound_values,
            drug_values,
            initial_state,
            level,
            offset_db=offset_db,
            method=method,
            rtol=rtol,
            atol=atol,
        ))
    return rows


def salt_residuals(predictions):
    """Use equal fixed total weight per sound level, not n-based precision."""
    by_key = {
        (row["noise_dba"], row["exposure_number"]): row
        for row in predictions
    }
    if len(by_key) != len(SALT_TARGETS):
        raise ValueError("exactly one prediction is required for every Salt target")
    divisor = SALT_WEIGHT_SCALE_MV * np.sqrt(EXPOSURES)
    return np.asarray([
        (
            by_key[(target["noise_dba"], target["exposure_number"])]
            ["predicted_overall_ep_change_mV"] - target["mean_mV"]
        ) / divisor
        for target in SALT_TARGETS
    ])


def _joint_residual(sound_values, drug_values, initial_state, wang_records, *, offset_db=0,
                    method="Radau", rtol=2e-6, atol=2e-8):
    wang_residual, _ = evaluate_wang(
        sound_values, drug_values, wang_records, delay=45.0,
        method=method, rtol=rtol, atol=atol,
    )
    salt_rows = salt_predictions(
        sound_values, drug_values, initial_state, offset_db=offset_db,
        method=method, rtol=rtol, atol=atol,
    )
    return np.r_[wang_residual, salt_residuals(salt_rows)]


def _salt_report_rows(predictions):
    lookup = {
        (row["noise_dba"], row["exposure_number"]): row
        for row in predictions
    }
    rows = []
    for target in SALT_TARGETS:
        prediction = lookup[(target["noise_dba"], target["exposure_number"])]
        rows.append({
            **target,
            **prediction,
            "residual_model_minus_target_mV": (
                prediction["predicted_overall_ep_change_mV"] - target["mean_mV"]
            ),
        })
    return rows


def _fit_primary(frozen_sound, drug_values, initial_state, wang_records):
    start = np.clip((frozen_sound - LOWER) / (UPPER - LOWER), 1e-5, 1.0 - 1e-5)

    def residual(unit):
        values = LOWER + unit * (UPPER - LOWER)
        return _joint_residual(values, drug_values, initial_state, wang_records)

    initial_residual = residual(start)
    opt = least_squares(
        residual,
        start,
        jac=lambda unit: bounded_jacobian(residual, unit),
        bounds=(np.zeros(3), np.ones(3)),
        max_nfev=70,
        ftol=1e-6,
        xtol=1e-6,
        gtol=1e-6,
    )
    values = LOWER + opt.x * (UPPER - LOWER)
    return values, opt, {
        "initial_weighted_sum_squares": float(initial_residual @ initial_residual),
        "final_optimizer_weighted_sum_squares": float(opt.fun @ opt.fun),
    }


def evaluate_joint_fit():
    """Fit only the three existing sound parameters under a fixed central map."""
    frozen_parameters, freeze = load_freeze()
    drug_values = _frozen_drug_values(frozen_parameters, freeze)
    frozen_sound = np.asarray([getattr(frozen_parameters, name) for name in NAMES])
    initial_state = np.asarray(freeze["initial_state"], dtype=float)
    wang_records = load_noise()

    sound_values, opt, score_summary = _fit_primary(
        frozen_sound, drug_values, initial_state, wang_records
    )
    if not opt.success:
        raise RuntimeError(f"joint sound fit did not converge: {opt.message}")

    check_rows = []
    check_payloads = {}
    for method in ("Radau", "BDF"):
        wang_residual, wang_detail = evaluate_wang(
            sound_values,
            drug_values,
            wang_records,
            delay=45.0,
            method=method,
            rtol=1e-8,
            atol=1e-10,
        )
        salt_rows = salt_predictions(
            sound_values,
            drug_values,
            initial_state,
            offset_db=0,
            method=method,
            rtol=1e-8,
            atol=1e-10,
        )
        salt_residual = salt_residuals(salt_rows)
        joint_residual = np.r_[wang_residual, salt_residual]
        check_rows.append({
            "method": method,
            "weighted_sum_squares": float(joint_residual @ joint_residual),
            "wang_1992_weighted_sum_squares": float(wang_residual[:45] @ wang_residual[:45]),
            "wang_1990_weighted_sum_squares": float(wang_residual[45:] @ wang_residual[45:]),
            "salt_table_iii_weighted_sum_squares": float(salt_residual @ salt_residual),
        })
        check_payloads[method] = {"wang": wang_detail, "salt": salt_rows}

    primary_salt_rows = _salt_report_rows(check_payloads["Radau"]["salt"])
    alternate_scenarios = []
    for scenario in input_scenarios()[1:]:
        offset = scenario["illustrative_offset_db"]
        rows = salt_predictions(
            sound_values,
            drug_values,
            initial_state,
            offset_db=offset,
            method="Radau",
            rtol=1e-8,
            atol=1e-10,
        )
        residual = salt_residuals(rows)
        alternate_scenarios.append({
            **scenario,
            "status": "fixed-parameter sensitivity; no alternate refit",
            "salt_weighted_sum_squares": float(residual @ residual),
            "salt_targets_and_predictions": _salt_report_rows(rows),
        })

    final_parameters = make_parameters(sound_values, drug_values)
    drug_records = load_anchors()
    drug_residual, drug_predictions = evaluate_drug(
        drug_values[:4],
        drug_records,
        step=0.025,
        method="Radau",
        rtol=1e-8,
        atol=1e-10,
        base_parameters=final_parameters,
    )
    frozen_drug_residual, frozen_drug_predictions = evaluate_drug(
        drug_values[:4],
        drug_records,
        step=0.025,
        method="Radau",
        rtol=1e-8,
        atol=1e-10,
        base_parameters=make_parameters(frozen_sound, drug_values),
    )

    wang_final = check_payloads["Radau"]["wang"]
    return {
        "status": "completed conditional joint Wang and Salt Table III fit",
        "interpretation": (
            "Exploratory, nonblinded conditional calibration. Salt outcomes were "
            "previously inspected; this is not independent validation or a matched "
            "acoustic prediction."
        ),
        "provenance": {
            "salt_citation": (
                "Salt AN, Konishi T. Effects of noise on cochlear potentials and "
                "endolymph potassium concentration recorded with potassium-selective "
                "electrodes. Hearing Research. 1979;1:343-363. "
                "doi:10.1016/0378-5955(79)90005-4."
            ),
            "salt_primary_source": str(SALT_PAPER.relative_to(ROOT)),
            "salt_source_page": "PDF page 10 (printed page 352), Table III",
            "salt_table_definition": (
                "ΔEPo is the difference between pre-exposure EP and EP measured "
                "after exposure under quiet conditions; Table III values are mean ± "
                "standard deviation, with n identified as observations."
            ),
            "wang_sources": [
                str(WANG1990_PAPER.relative_to(ROOT)),
                str(WANG1992_PAPER.relative_to(ROOT)),
            ],
            "wang_digitized_data_source": str(SOURCE.relative_to(ROOT)),
            "frozen_candidate_source": str(FREEZE.relative_to(ROOT)),
            "source_sha256": {
                str(path.relative_to(ROOT)): _sha256(path)
                for path in (
                    SALT_PAPER, WANG1990_PAPER, WANG1992_PAPER,
                    SOURCE, FREEZE, WANG_FIT,
                )
            },
        },
        "candidate": {
            "frozen_status": freeze["status"],
            "freeze_date": freeze["freeze_date"],
            "frozen_model_sha256": freeze["model_sha256"],
            "frozen_fit_sha256": freeze["fit_sha256"],
            "hybrid_closure": freeze["closure"],
            "drug_parameter_values": dict(zip(ALL_NAMES, map(float, drug_values))),
            "all_model_parameter_values": asdict(final_parameters),
            "only_fitted_parameters": list(NAMES),
            "drug_source": (
                "All six existing drug-proxy/calibration values loaded from the "
                "validated freeze; not varied in this fit."
            ),
        },
        "salt_source_data": {
            "included_rows": [dict(row) for row in SALT_TARGETS],
            "unit": "mV",
            "recording_site": "basal turn",
            "excluded_outcomes": (
                "Only ΔEPo overall per-exposure changes were fitted. ΔEPc, "
                "ΔEPon-off, ΔEPq, CM/SP/AP, potassium concentrations, and separate "
                "second-turn rows were not included. The separate second-turn "
                "115-dBA group is not pooled with the basal-turn target because the "
                "spatially lumped model has no turn-specific observation operator; "
                "correlated Table III components were not counted as additional targets."
            ),
            "repeat_measures": (
                "Exposure numbers 1-3 are retained as repeated outcomes. The paper "
                "reports n observations per row (3, 3, 3 at 95/105 dBA; 9, 9, 7 "
                "at 115 dBA), but provides no within-subject covariance. Repeated "
                "rows are therefore not treated as independent likelihood samples."
            ),
        },
        "conditional_acoustic_input": {
            "mapping_is_fitted": False,
            "mapping_is_supported_conversion": False,
            "central_scenario": input_scenarios()[0],
            "alternate_scenarios": input_scenarios()[1:],
            "interpretation": (
                "The existing model expects a dB SPL-like input to its empirical "
                "broadband transform. No unique mapping from the paper's dBA "
                "open-bulla stimulus to dB SPL or model drive was identified. The "
                "central same-number mapping and ±10 dB offsets are fixed illustrative "
                "scenarios, not conversions, justified corrections, or uncertainty "
                "bounds. Alternates are predictions with the central fitted parameters."
            ),
        },
        "observation_protocol": {
            "source_protocol": (
                "NIH-strain guinea pigs (350-550 g) were anesthetized throughout "
                "recording and exposure (pentobarbital 30-35 mg/kg, gallamine "
                "immobilization, artificial respiration). Filtered broadband white "
                "noise (approximately 400 Hz-8 kHz) was delivered to the open bulla "
                "for repeated 20-minute periods, at levels including 95, 105, and "
                "115 dBA. EP was recorded from basal-turn scala media. Measurements "
                "of auditory-response input-output functions started 1 minute after "
                "noise offset and took approximately 2 minutes; this does not specify "
                "the timing of Table III EP summaries. Quiet intervals were usually "
                "5 minutes, sometimes 10."
            ),
            "model_schedule": {
                "exposure_duration_min": EXPOSURE_MIN,
                "quiet_interval_min": QUIET_INTERVAL_MIN,
                "exposure_starts_min": [0.0, 25.0, 50.0],
                "measurement_window_after_each_offset_min": [1.0, 3.0],
                "temporal_operator": (
                    "For each exposure, trapezoidal mean of model EP at 1, 2, and "
                    "3 min after offset, minus model EP immediately before that "
                    "exposure. This is a hypothetical EP observation operator, not "
                    "a source-established Table III measurement window."
                ),
                "episode_change_is_local": True,
            },
            "model_observation": "reference state y[8], EP in mV.",
        },
        "weighting": {
            "objective": (
                "Existing Wang 1992 and Wang 1990 residual definitions/scales are "
                "unchanged. Salt residuals are (predicted ΔEPo - Table III mean) / "
                "(3 mV × sqrt(3 episodes)); sum of squares is therefore equally "
                "weighted per Salt level, with the three repeated episodes sharing "
                "one level's total weight."
            ),
            "salt_scale_mV": SALT_WEIGHT_SCALE_MV,
            "salt_episode_count_per_level": EXPOSURES,
            "uses_n_or_sd_as_independent_precision": False,
            "objective_is_likelihood": False,
        },
        "fit": {
            "scenario": "primary_same_numeric_level",
            "optimizer": {
                "success": bool(opt.success),
                "message": str(opt.message),
                "function_evaluations": int(opt.nfev),
                "optimality": float(opt.optimality),
                "active_bounds": dict(zip(NAMES, map(int, opt.active_mask))),
                "bounds": dict(zip(
                    NAMES,
                    [[float(low), float(high)] for low, high in zip(LOWER, UPPER)],
                )),
                "initial_sound_parameters": dict(zip(NAMES, map(float, frozen_sound))),
                "fitted_sound_parameters": dict(zip(NAMES, map(float, sound_values))),
                **score_summary,
            },
            "solver_checks": check_rows,
            "wang1992_predictions": wang_final["wang1992"],
            "wang1990_predictions": wang_final["wang1990"],
            "salt_targets_and_predictions": primary_salt_rows,
            "alternate_fixed_parameter_predictions": alternate_scenarios,
        },
        "drug_recheck": {
            "frozen_candidate_weighted_sum_squares": float(
                frozen_drug_residual @ frozen_drug_residual
            ),
            "joint_candidate_weighted_sum_squares": float(drug_residual @ drug_residual),
            "frozen_candidate_predictions": frozen_drug_predictions,
            "joint_candidate_predictions": drug_predictions,
            "interpretation": (
                "Existing drug objective reevaluated after changing only the three "
                "sound parameters; six frozen drug parameters remain fixed."
            ),
        },
        "limitations": [
            "The table gives group mean ± SD and observation counts but no within-animal or cross-episode covariance; repeated rows and Table III components are correlated. The deterministic equal-level weighting is not a likelihood and does not infer a covariance.",
            "The central dBA-to-dB-SPL numerical mapping and ±10 dB alternatives are unsupported illustrative assumptions; they are not acoustic conversions, corrections, or uncertainty bounds.",
            "The primary 95-dBA-to-95-dB-SPL assumption lies below the 105-125 dB SPL Wang calibration levels; its model transform input is not directly constrained by the fitted Wang range.",
            "The model is a spatially lumped guinea-pig cochlear reconstruction, not a matched basal-turn electrode/acoustic-field model; species/preparation and output measurement processes remain simplified.",
            "The 5-min quiet gap is the reported usual interval; some experiments used 10 min. The 1–3-min auditory-response measurement window does not establish when Table III EP was measured. Averaging EP over that window is a hypothetical, unverified observation operator.",
            "The table's n is number of observations, not necessarily independent animals; n and SD are preserved but not used to imply independent residual precision.",
            "Only basal-turn ΔEPo was fit; the separate second-turn 115-dBA group was excluded because the lumped model cannot resolve turn-specific EP, and correlated EP components were excluded.",
            "The frozen candidate was outcome-informed and Salt was previously inspected, so this is nonblinded recalibration, not independent validation. The candidate freeze file, model source, previous fitted result, and figure are not modified.",
            "The model uses a related-code hybrid closure absent from the manuscript; sound parameters have numerical bounds and a local optimizer does not establish a global optimum or identify mechanism.",
            "The nominal frozen initial state is not exactly stationary, though Salt per-episode changes use the immediate pre-exposure EP and therefore report episode-local change.",
        ],
    }


def main():
    report = evaluate_joint_fit()
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {OUTPUT}")
    fit = report["fit"]["optimizer"]
    print(
        f"Converged={fit['success']} message={fit['message']} "
        f"score={fit['final_optimizer_weighted_sum_squares']:.6f}"
    )
    if not fit["success"]:
        raise RuntimeError(f"joint sound fit did not converge: {fit['message']}")


if __name__ == "__main__":
    main()
