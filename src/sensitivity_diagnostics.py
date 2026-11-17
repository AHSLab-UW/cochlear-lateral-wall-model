"""Limited weight and endpoint-grid sensitivity checks for the frozen fit."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from src import fit_parameters
from src.pike_trace_diagnostic import ROOT, load_anchors
from src.salt_1979_digitize import DEFAULT_EVIDENCE, load_evidence
from src.scaled_parameter_fit import bounded_jacobian, from_unit
from src.wang_noise_traces import load_noise


OUTPUT = ROOT / "results" / "sensitivity_diagnostics.json"
ACOUSTIC_COMPONENTS = ("wang_1992", "wang_1990", "salt_table_iii", "salt_figure_3")
LOW = fit_parameters.LOW
HIGH = fit_parameters.HIGH
PARAMETERS = fit_parameters.PARAMETERS


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rescale_components(residual, audit, multiplier):
    weighted = np.asarray(residual, dtype=float).copy()
    components = audit["components"]
    missing = set(ACOUSTIC_COMPONENTS) - set(components)
    if missing:
        raise ValueError(f"missing acoustic residual components: {sorted(missing)}")
    for name in ACOUSTIC_COMPONENTS:
        start, stop = components[name]["residual_slice"]
        weighted[start:stop] *= multiplier
    return weighted


def _component_grid_changes(coarse, fine, coarse_audit, fine_audit):
    report = {}
    for name, metadata in coarse_audit["components"].items():
        start, stop = metadata["residual_slice"]
        delta = fine[start:stop] - coarse[start:stop]
        report[name] = {
            "coarse_weighted_sum_squares": metadata["weighted_sum_squares"],
            "fine_weighted_sum_squares": fine_audit["components"][name]["weighted_sum_squares"],
            "maximum_absolute_residual_change": float(np.max(np.abs(delta))),
            "rms_residual_change": float(np.sqrt(np.mean(delta ** 2))),
        }
    return report


def _fit_variant(start_values, records, *, solver, acoustic_multiplier=1.0):
    unit_start = np.clip((start_values - LOW) / (HIGH - LOW), 0.0, 1.0)

    def residual(unit):
        values = from_unit(unit)
        vector, audit = fit_parameters.evaluate_all(values, *records, solver=solver)
        return _rescale_components(vector, audit, acoustic_multiplier)

    result = least_squares(
        residual,
        unit_start,
        jac=lambda point: bounded_jacobian(residual, point, step=1e-3),
        bounds=(np.zeros(9), np.ones(9)),
        method="trf",
        max_nfev=15,
        ftol=1e-5,
        xtol=1e-5,
        gtol=5e-3,
    )
    return result, from_unit(result.x)


def main():
    fit = json.loads(fit_parameters.OUTPUT.read_text())
    values = np.asarray([fit["parameters"][name] for name in PARAMETERS], dtype=float)
    records = (load_anchors(), load_noise(), load_evidence())
    tight_coarse = fit_parameters.SolverSettings(
        step=0.1, method="Radau", rtol=1e-8, atol=1e-10
    )
    fine = fit_parameters.SolverSettings(
        step=0.025, method="Radau", rtol=1e-8, atol=1e-10
    )

    coarse_residual, coarse_audit = fit_parameters.evaluate_all(
        values, *records, solver=tight_coarse
    )
    fine_residual, fine_audit = fit_parameters.evaluate_all(
        values, *records, solver=fine
    )
    grid_report = {
        "coarse_solver": coarse_audit["solver"],
        "fine_solver": fine_audit["solver"],
        "coarse_weighted_sum_squares": float(coarse_residual @ coarse_residual),
        "fine_weighted_sum_squares": float(fine_residual @ fine_residual),
        "component_changes": _component_grid_changes(
            coarse_residual, fine_residual, coarse_audit, fine_audit
        ),
        "peak_endpoint_predictions": {
            name: {
                "coarse": coarse_audit["components"][name]["predictions"],
                "fine": fine_audit["components"][name]["predictions"],
            }
            for name in ("asakuma", "rybak")
        },
        "interpretation": (
            "This compares requested-output-grid extrema at the same frozen parameter "
            "vector; it does not itself refit the objective or prove continuous-time "
            "extrema."
        ),
    }

    acoustic_factor = np.sqrt(2.0)
    alternative_result, alternative_values = _fit_variant(
        values, records, solver=fit_parameters.SolverSettings(),
        acoustic_multiplier=acoustic_factor,
    )
    alternative_residual, alternative_audit = fit_parameters.evaluate_all(
        alternative_values, *records
    )
    alternative_report = {
        "weight_change": (
            "Multiply all acoustic residual components by sqrt(2), doubling their "
            "combined squared-residual weight relative to pharmacological components."
        ),
        "success": bool(alternative_result.success),
        "termination": str(alternative_result.message),
        "function_evaluations": int(alternative_result.nfev),
        "optimality_unit_box": float(alternative_result.optimality),
        "alternative_weighted_sum_squares": float(alternative_result.fun @ alternative_result.fun),
        "unmodified_objective_at_alternative_fit": float(alternative_residual @ alternative_residual),
        "parameter_values": dict(zip(PARAMETERS, map(float, alternative_values))),
        "parameter_shift_physical": dict(zip(
            PARAMETERS, map(float, alternative_values - values)
        )),
        "parameter_shift_unit_box": dict(zip(
            PARAMETERS, map(float, (alternative_values - values) / (HIGH - LOW))
        )),
        "components_under_unmodified_objective": alternative_audit["components"],
        "interpretation": (
            "A single, deliberately simple weight perturbation; it is not a likelihood, "
            "confidence interval, or comprehensive robustness analysis."
        ),
    }

    fine_result, fine_values = _fit_variant(
        values, records, solver=fine, acoustic_multiplier=1.0
    )
    fine_refit_residual, fine_refit_audit = fit_parameters.evaluate_all(
        fine_values, *records, solver=fine
    )
    fine_refit_coarse_residual, _ = fit_parameters.evaluate_all(
        fine_values, *records, solver=tight_coarse
    )
    fine_refit_report = {
        "solver": asdict(fine),
        "success": bool(fine_result.success),
        "termination": str(fine_result.message),
        "function_evaluations": int(fine_result.nfev),
        "optimality_unit_box": float(fine_result.optimality),
        "fine_grid_weighted_sum_squares": float(fine_refit_residual @ fine_refit_residual),
        "coarse_grid_weighted_sum_squares_at_fine_refit": float(
            fine_refit_coarse_residual @ fine_refit_coarse_residual
        ),
        "parameter_values": dict(zip(PARAMETERS, map(float, fine_values))),
        "parameter_shift_physical": dict(zip(PARAMETERS, map(float, fine_values - values))),
        "parameter_shift_unit_box": dict(zip(
            PARAMETERS, map(float, (fine_values - values) / (HIGH - LOW))
        )),
        "components": fine_refit_audit["components"],
        "interpretation": (
            "The 0.025-min refit tests endpoint-grid sensitivity locally from the frozen "
            "candidate; it is not an event-based extremum calculation or a new primary fit."
        ),
    }

    report = {
        "status": "post-review limited sensitivity diagnostics",
        "reference_parameter_order": list(PARAMETERS),
        "reference_parameters": dict(zip(PARAMETERS, map(float, values))),
        "reference_optimality_unit_box": fit["optimizer"]["optimality_unit_box"],
        "study_inclusion_sensitivity": {
            "omitted_study": "Wang et al. (1990)",
            "retained_score_at_reference": 19.906813,
            "retained_score_after_refit": 19.906702,
            "rho_W_at_lower_bound_before_and_after": 0.2,
            "scope": "One existing exclusion refit; other study exclusions remain untested.",
        },
        "requested_output_grid_sensitivity_at_frozen_fit": grid_report,
        "twofold_acoustic_weight_refit": alternative_report,
        "fine_grid_refit": fine_refit_report,
        "source_sha256": {
            "fit": _sha256(fit_parameters.OUTPUT),
            "fit_code": _sha256(ROOT / "src" / "fit_parameters.py"),
            "diagnostic_code": _sha256(Path(__file__)),
            "evidence": _sha256(DEFAULT_EVIDENCE),
        },
        "limitations": [
            "Historical traces are digitized and serially correlated; target count is not independent information.",
            "The alternative weight is one analyst-selected sensitivity perturbation, not a probability model.",
            "The endpoint-grid comparisons use requested output samples, not continuous-time events.",
            "Neither refit establishes parameter identifiability, uncertainty, global optimality, or predictive validity.",
        ],
    }
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
