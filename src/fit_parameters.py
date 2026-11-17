"""Fit the existing nine parameters to six reference studies, excluding Salt Fig. 5."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.optimize import least_squares

from src import (
    asakuma_ep_endpoint_sensitivity,
    joint_calibration_screen,
    manuscript_audit,
    model_equations,
    salt_table3_fit as salt_table,
    salt_1979_digitize,
    drug_calibration,
    parameter_schema as drug_noise,
    kinetics_fit,
    wang_noise_cohorts as wang1990,
    wang_noise_traces as wang1992,
    pike_trace_diagnostic,
)
from src.model_equations import baseline, related_code_closure
from src.salt_trace_residuals import evaluate_fig3
from src.scaled_parameter_fit import bounded_jacobian, from_unit
from src.pike_trace_diagnostic import ROOT, load_anchors


START = ROOT / "results/salt_table3_fit.json"
OUTPUT = ROOT / "results/fit_parameters.json"
PARAMETERS = drug_noise.PARAMETERS
LOW = drug_noise.LOW
HIGH = drug_noise.HIGH
MAX_NFEV = 15
UNIT_FD_STEP = 1e-3
OPTIMIZER_FTOL = 1e-5
OPTIMIZER_XTOL = 1e-5
OPTIMIZER_GTOL = 5e-3


@dataclass(frozen=True)
class SolverSettings:
    step: float = 0.1
    method: str = "Radau"
    rtol: float = 2e-7
    atol: float = 2e-9

    def __post_init__(self):
        if self.method not in ("Radau", "BDF"):
            raise ValueError("solver must be Radau or BDF")
        if (not np.isfinite([self.step, self.rtol, self.atol]).all()
                or min(self.step, self.rtol, self.atol) <= 0
                or not np.isclose(120 / self.step, round(120 / self.step))):
            raise ValueError("positive solver tolerances and a grid dividing 120 are required")


@dataclass(frozen=True)
class Fig3Context:
    model_parameters: object
    closure: object
    initial_state: tuple
    solver: SolverSettings


def load_start(path=START):
    """Load only the pre-Figure-5 joint-Wang/Salt-Table-III parameter values."""
    data = json.loads(Path(path).read_text())
    if data.get("status") != "completed conditional joint Wang and Salt Table III fit":
        raise ValueError("pre-Figure-5 Salt Table III starting result is required")
    drug = data["candidate"]["drug_parameter_values"]
    sound = data["fit"]["optimizer"]["fitted_sound_parameters"]
    values = np.asarray([drug[name] for name in PARAMETERS[:6]]
                        + [sound[name] for name in PARAMETERS[6:]], dtype=float)
    if (values.shape != (9,) or not np.isfinite(values).all()
            or np.any(values < LOW) or np.any(values > HIGH)):
        raise ValueError("pre-Figure-5 starting values are invalid or outside bounds")
    return values


def _add(parts, components, name, residual, predictions, weighting):
    values = np.asarray(residual, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError(f"{name} must return a nonempty finite residual vector")
    start = sum(len(part) for part in parts)
    parts.append(values)
    components[name] = {
        "residual_slice": [start, start + len(values)],
        "residual_count": int(len(values)),
        "weighted_sum_squares": float(values @ values),
        "weighting": weighting,
        "predictions": predictions,
    }


def evaluate_all(values, drug_records, noise_records, fig3_data, *, solver=SolverSettings()):
    """Evaluate the five existing studies plus Salt Fig. 3 and Table III."""
    x = np.asarray(values, dtype=float)
    if (x.shape != (9,) or not np.isfinite(x).all()
            or np.any(x < LOW) or np.any(x > HIGH)):
        raise ValueError("nine finite existing parameters within bounds are required")
    drug, sound = x[:6], x[6:]
    parameters = wang1992.make_parameters(sound, drug)
    closure = related_code_closure()
    initial_state = baseline(parameters)
    kwargs = {"method": solver.method, "rtol": solver.rtol, "atol": solver.atol}
    drug_residual, drug_predictions = drug_calibration.evaluate(
        drug[:4], drug_records, step=solver.step, base_parameters=parameters, **kwargs)
    noise_residual, noise_predictions = wang1992.evaluate_noise(
        sound, drug, noise_records, **kwargs)
    long_noise_residual, long_noise_predictions = wang1990.wang1990(
        sound, drug, delay=45.0, **kwargs)
    table_rows = salt_table.salt_predictions(
        sound, drug, initial_state, offset_db=0, **kwargs)
    table_residual = salt_table.salt_residuals(table_rows)
    trace_residual, trace_detail = evaluate_fig3(
        Fig3Context(parameters, closure, tuple(initial_state), solver), fig3_data)

    parts, components = [], {}
    pike_count = sum(
        sum(row["curve"] == key for row in drug_records)
        for key in pike_trace_diagnostic.IDS
    )
    expected_drug_count = pike_count + 2 + 2 * len(joint_calibration_screen.RYBAK)
    if len(drug_residual) != expected_drug_count:
        raise ValueError("drug evaluator residual layout changed")
    _add(components=components, parts=parts, name="pike_bosher",
         residual=drug_residual[:pike_count],
         predictions={key: drug_predictions[key] for key in pike_trace_diagnostic.IDS},
         weighting="Baseline-relative EP error / (10 mV * sqrt(anchors per dose curve)).")
    _add(components=components, parts=parts, name="asakuma",
         residual=drug_residual[pike_count:pike_count + 2],
         predictions=drug_predictions["asakuma_60"],
         weighting="Resistance ratio error / 0.2; baseline-minus-nadir EP error / 10 mV.")
    _add(components=components, parts=parts, name="rybak",
         residual=drug_residual[pike_count + 2:],
         predictions={key: value for key, value in drug_predictions.items()
                      if key.startswith("rybak_")},
         weighting="EP and K group-mean reduction errors / reported SE; covariance unknown.")
    _add(components=components, parts=parts, name="wang_1992",
         residual=noise_residual, predictions=noise_predictions,
         weighting="Baseline-relative EP error / (3 mV * sqrt(anchors per sound level)).")
    _add(components=components, parts=parts, name="wang_1990",
         residual=long_noise_residual, predictions=long_noise_predictions,
         weighting="Separate-cohort EP difference error / (5 mV * sqrt(number of cohorts)).")
    _add(components=components, parts=parts, name="salt_table_iii",
         residual=table_residual, predictions=salt_table._salt_report_rows(table_rows),
         weighting=("Local-episode delta EP / (3 mV * sqrt(3 episodes)); "
                    "no SD/n precision; timing window is hypothetical."))
    _add(components=components, parts=parts, name="salt_figure_3",
         residual=trace_residual, predictions=trace_detail["predictions"],
         weighting=trace_detail["weighting"])
    residuals = np.concatenate(parts)
    return residuals, {
        "score": float(residuals @ residuals),
        "residual_count": int(len(residuals)),
        "components": components,
        "parameters": dict(zip(PARAMETERS, map(float, x))),
        "bounds": dict(zip(PARAMETERS, [[float(lo), float(hi)] for lo, hi in zip(LOW, HIGH)])),
        "active_bounds": {
            name: int(np.isclose(value, LOW[index], rtol=0, atol=1e-8)
                      or np.isclose(value, HIGH[index], rtol=0, atol=1e-8))
            for index, (name, value) in enumerate(zip(PARAMETERS, x))
        },
        "solver": asdict(solver),
        "all_model_parameter_values": asdict(parameters),
        "initial_state": initial_state.tolist(),
        "closure": closure.description,
        "fig3_data_sha256": trace_detail["fig3_data_sha256"],
        "fig3_source_image_sha256": trace_detail["fig3_source_image_sha256"],
        "fig3_source_path": trace_detail["source_path"],
        "salt_fig3": {
            key: trace_detail[key] for key in (
                "figure_schedule_min", "conditional_input", "omissions")
        },
    }


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _unit_starts(x0):
    unit = np.clip((x0 - LOW) / (HIGH - LOW), 0.0, 1.0)
    signs = np.asarray([1, -1, 1, -1, 1, -1, -1, -1, 1], dtype=float)
    second = np.asarray([-1, 1, -1, 1, -1, 1, 1, 1, -1], dtype=float)
    return [
        ("pre_fig5_table3_fit", unit),
        ("seed_perturbation_a", np.clip(unit + .035 * signs, .01, .99)),
    ]


def _fit():
    from src.salt_1979_digitize import load_evidence
    from src.wang_noise_traces import load_noise
    from src.pike_trace_diagnostic import load_anchors

    start_bytes = START.read_bytes()
    x0 = load_start()
    drug_records, noise_records = load_anchors(), load_noise()
    fig3_data = load_evidence()

    def residual(unit):
        return evaluate_all(from_unit(unit), drug_records, noise_records, fig3_data)[0]

    attempts, completed = [], []
    initial_residual = residual(np.clip((x0 - LOW) / (HIGH - LOW), 0.0, 1.0))
    for label, unit_start in _unit_starts(x0):
        start_residual = residual(unit_start)
        print(f"Starting {label}: score {start_residual @ start_residual:.8g}", flush=True)
        result = least_squares(
            residual, unit_start, jac=lambda point: bounded_jacobian(
                residual, point, step=UNIT_FD_STEP),
            bounds=(np.zeros(9), np.ones(9)), method="trf", max_nfev=MAX_NFEV,
            ftol=OPTIMIZER_FTOL, xtol=OPTIMIZER_XTOL, gtol=OPTIMIZER_GTOL,
        )
        attempt = {
            "start": label,
            "initial_parameters": dict(zip(PARAMETERS, map(float, from_unit(unit_start)))),
            "success": bool(result.success),
            "message": str(result.message),
            "function_evaluations": int(result.nfev),
            "jacobian_evaluations": int(result.njev or 0),
            "initial_score": float(start_residual @ start_residual),
            "final_score": float(result.fun @ result.fun),
            "optimality": float(result.optimality),
            "unit_box_fd_step": UNIT_FD_STEP,
            "active_bounds": dict(zip(PARAMETERS, map(int, result.active_mask))),
            "parameters": dict(zip(PARAMETERS, map(float, from_unit(result.x)))),
        }
        attempts.append(attempt)
        print(f"Finished {label}: success={result.success}, "
              f"score={attempt['final_score']:.8g}, optimality={result.optimality:.5g}",
              flush=True)
        if (result.success and np.isfinite(result.fun).all()
                and np.isfinite(result.optimality) and result.fun @ result.fun <= start_residual @ start_residual):
            completed.append((result, label))
    if not completed:
        raise RuntimeError(f"No optimizer start passed convergence checks: {attempts}")
    best, best_label = min(completed, key=lambda row: row[0].fun @ row[0].fun)
    fitted_values = from_unit(best.x)
    final_residual, final_audit = evaluate_all(
        fitted_values, drug_records, noise_records, fig3_data)
    if not np.allclose(final_residual, best.fun, rtol=2e-5, atol=2e-6):
        raise RuntimeError("recomputed optimizer residuals do not match reported optimum")

    initial_candidate_residual, initial_audit = evaluate_all(
        x0, drug_records, noise_records, fig3_data)
    checks = []
    for method, step, rtol, atol in (
            ("Radau", .1, 1e-8, 1e-10),
            ("BDF", .1, 1e-8, 1e-10),
            ("Radau", .025, 1e-8, 1e-10)):
        residual_check, audit_check = evaluate_all(
            fitted_values, drug_records, noise_records, fig3_data,
            solver=SolverSettings(step=step, method=method, rtol=rtol, atol=atol))
        checks.append({
            "solver": audit_check["solver"],
            "weighted_sum_squares": float(residual_check @ residual_check),
            "components": audit_check["components"],
        })

    code_modules = (
        model_equations, manuscript_audit, drug_calibration,
        kinetics_fit, drug_noise, wang1992, wang1990, salt_table,
        pike_trace_diagnostic, asakuma_ep_endpoint_sensitivity,
        joint_calibration_screen, salt_1979_digitize,
    )
    code_hashes = {
        str(Path(module.__file__).resolve().relative_to(ROOT)): _sha256(module.__file__)
        for module in code_modules
    }
    code_hashes["src/fit_parameters.py"] = _sha256(__file__)
    trace_module = Path(__file__).with_name("salt_trace_residuals.py")
    code_hashes["src/salt_trace_residuals.py"] = _sha256(trace_module)
    result = {
        "status": "completed six-study nine-parameter local joint fit; Salt Figure 5 excluded",
        "interpretation": (
            "Outcome-informed exploratory calibration, not independent validation. "
            "The selected optimizer result is local; two deterministic seed-derived "
            "starts do not establish global optimality."
        ),
        "study_inclusions": [
            "Pike/Bosher EP traces", "Asakuma EP and effective resistance",
            "Rybak EP and endolymph-K reductions", "Wang 1992 EP time traces",
            "Wang 1990 post-noise EP cohorts", "Salt & Konishi Table III",
            "Salt & Konishi Figure 3 EP/K paired trace",
        ],
        "salt_fig5_exclusion": {
            "used_for_initialization": False,
            "used_for_start_selection": False,
            "used_in_objective": False,
            "data_or_image_hash_in_active_provenance": False,
            "note": (
                "The joint extraction file contains both figures; the objective selects "
                "only figures.fig3 and hashes only Figure 3 data and source images."
            ),
        },
        "starting_candidate": {
            "path": str(START.relative_to(ROOT)),
            "sha256": hashlib.sha256(start_bytes).hexdigest(),
            "description": (
                "Existing pre-Figure-5 conditional Wang/Salt Table III result; "
                "the nine values are read from its six drug proxy values and three "
                "sound parameter estimates only."
            ),
            "parameters": dict(zip(PARAMETERS, map(float, x0))),
            "initial_weighted_sum_squares": float(initial_candidate_residual @ initial_candidate_residual),
            "components": initial_audit["components"],
        },
        "optimizer": {
            "method": "scipy.optimize.least_squares(method='trf') on unit-box coordinates",
            "success": bool(best.success),
            "message": str(best.message),
            "function_evaluations": int(best.nfev),
            "jacobian_evaluations": int(best.njev or 0),
            "optimality_unit_box": float(best.optimality),
            "best_start": best_label,
            "max_nfev_per_start": MAX_NFEV,
            "unit_box_finite_difference_step": UNIT_FD_STEP,
            "finite_difference_units": "fraction of each declared parameter range",
            "termination_tolerances": {
                "ftol": OPTIMIZER_FTOL, "xtol": OPTIMIZER_XTOL, "gtol": OPTIMIZER_GTOL,
            },
            "initial_seed_score": float(initial_residual @ initial_residual),
            "final_optimizer_score": float(final_residual @ final_residual),
            "initial_candidate_score": float(initial_candidate_residual @ initial_candidate_residual),
            "final_solver_check_score": checks[0]["weighted_sum_squares"],
            "starts": attempts,
        },
        "parameters": dict(zip(PARAMETERS, map(float, fitted_values))),
        "bounds": final_audit["bounds"],
        "active_bounds": final_audit["active_bounds"],
        "initial_state": final_audit["initial_state"],
        "all_model_parameter_values": final_audit["all_model_parameter_values"],
        "drug_exposure_proxy": {
            "guinea_pig_proxy_per_mg_kg": float(fitted_values[0]),
            "chinchilla_exposure_multiplier": float(fitted_values[1]),
        },
        "closure": final_audit["closure"],
        "components": final_audit["components"],
        "fig3_provenance": {
            key: final_audit[key] for key in (
                "fig3_data_sha256", "fig3_source_image_sha256", "fig3_source_path",
                "salt_fig3")
        },
        "numerical_checks": checks,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "provenance": {
            "code_sha256": code_hashes,
            "sources": {
                "selected_observation_sha256": {
                    "pike_bosher": _json_hash(drug_records),
                    "wang_1992": _json_hash([
                        {"level_db_spl": level, "time_min": time.tolist(),
                         "ep_mV": ep.tolist()}
                        for level, time, ep in noise_records
                    ]),
                    "wang_1990": _json_hash(wang1990.WANG_MEANS),
                    "salt_table_iii": _json_hash(salt_table.SALT_TARGETS),
                    "salt_figure_3": final_audit["fig3_data_sha256"],
                },
                "pike_anchor_file": str(pike_trace_diagnostic.SOURCE.relative_to(ROOT)),
                "wang_1992_paper": str(wang1992.PAPER.relative_to(ROOT)),
                "wang_1990_paper": str(wang1990.WANG.relative_to(ROOT)),
                "salt_table_iii_paper_page": "refs/target_source/salt_1979_hear_res.pdf, printed p. 352, Table III (citation only; file hash omitted to avoid Figure 5 source content)",
                "salt_fig3_source_data": final_audit["fig3_source_path"],
            },
            "excluded_fig5_hashes": True,
        },
        "limitations": [
            "The fixed residual scales are illustrative deterministic weights, not a likelihood; correlated observations and unknown cross-study covariance are not modeled.",
            "The Salt dBA-to-model input mapping is same-numeric-level and conditional, not a supported acoustic conversion.",
            "Salt Table III uses an inherited hypothetical 1–3 minute EP observation window; the paper does not establish that EP timing.",
            "Wang 1990 uses an assumed fixed 45-minute measurement delay; Wang 1992 group digitization and temporal points are correlated.",
            "The six-study candidate is outcome-informed and nonblinded; it is not independent validation.",
            "The reconstructed model retains its related-code hybrid closure, species/exposure proxy assumptions, and treated potassium boundary.",
            "A successful bounded local least-squares termination and three seed-derived starts do not establish a global optimum, identifiability, physiological validity, or stability.",
        ],
    }
    return result


def main():
    report = _fit()
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        f"Wrote {OUTPUT}; score {report['optimizer']['final_optimizer_score']:.8g} "
        f"(seed {report['optimizer']['initial_candidate_score']:.8g}); "
        f"best start {report['optimizer']['best_start']}"
    )


if __name__ == "__main__":
    main()
