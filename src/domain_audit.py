"""Reproduce the six-study objective and audit sampled model domains."""

import hashlib
import json
from pathlib import Path

import numpy as np

from src import (
    asakuma_ep_endpoint_sensitivity as asakuma,
    salt_table3_fit as salt_table,
    fit_parameters as joint_fit,
    salt_trace_residuals as salt_trace,
    drug_calibration as drug_fit,
    wang_noise_cohorts as wang1990,
    wang_noise_traces as wang1992,
    noise_scenarios as model_scenarios,
    tier2_zhang_hou_benchmarks as benchmarks,
)
from src.model_equations import Parameters, STATE_NAMES, observables
from src.parameter_schema import PARAMETERS
from src.pike_trace_diagnostic import ROOT, SOURCE as DIGITIZED_CALIBRATION, load_anchors
from src.salt_1979_digitize import DEFAULT_EVIDENCE, load_evidence
from src.wang_noise_traces import load_noise


FIT = joint_fit.OUTPUT
RESIDUALS = ROOT / "results/calibration_residual_vector.json"
OUTPUT = ROOT / "results/domain_audit.json"
INPUT_SOURCES = (
    FIT,
    RESIDUALS,
    DIGITIZED_CALIBRATION,
    DEFAULT_EVIDENCE,
    asakuma.EVIDENCE,
    joint_fit.START,
    benchmarks.EVIDENCE,
    benchmarks.FREEZE,
)
SIMULATION_MODULES = (
    drug_fit,
    wang1992,
    wang1990,
    salt_table,
    salt_trace,
    benchmarks,
    model_scenarios,
)


class DomainRanges:
    def __init__(self):
        self.samples = 0
        self.simulation_calls = 0
        self.states = {
            name: {"minimum": float("inf"), "maximum": float("-inf")}
            for name in STATE_NAMES
        }
        self.observables = {}
        self.invalid_sample_counts = {
            "nonfinite_state": 0,
            "nonpositive_required_state": 0,
            "nonpositive_endolymph_volume_or_potassium": 0,
            "adaptation_or_trauma_outside_unit_interval": 0,
            "nonfinite_observable": 0,
            "negative_workload": 0,
        }

    def add(self, times, states, forcing, closure, parameters):
        times = np.asarray(times, dtype=float)
        states = np.asarray(states, dtype=float)
        if states.shape != (len(STATE_NAMES), len(times)):
            raise ValueError("simulation output does not match the 13-state schema")
        self.simulation_calls += 1
        self.samples += len(times)
        if not np.isfinite(states).all():
            self.invalid_sample_counts["nonfinite_state"] += int(
                np.size(states) - np.isfinite(states).sum()
            )
        for index, name in enumerate(STATE_NAMES):
            values = states[index]
            row = self.states[name]
            row["minimum"] = min(row["minimum"], float(np.nanmin(values)))
            row["maximum"] = max(row["maximum"], float(np.nanmax(values)))

        positive_indices = (0, 1, 2, 3, 4, 5, 6, 7, 9, 10)
        self.invalid_sample_counts["nonpositive_required_state"] += int(
            np.count_nonzero(states[list(positive_indices)] <= 0)
        )
        for time, state in zip(times, states.T):
            values = observables(state, forcing(float(time)), closure, parameters)
            if not all(np.isfinite(value) for value in values.values()):
                self.invalid_sample_counts["nonfinite_observable"] += 1
                continue
            for name, value in values.items():
                bounds = self.observables.setdefault(
                    name, {"minimum": float("inf"), "maximum": float("-inf")}
                )
                bounds["minimum"] = min(bounds["minimum"], float(value))
                bounds["maximum"] = max(bounds["maximum"], float(value))
            if values["endolymph_volume"] <= 0 or values["endolymph_k"] <= 0:
                self.invalid_sample_counts[
                    "nonpositive_endolymph_volume_or_potassium"
                ] += 1
            if not (-1e-8 <= state[11] <= 1 + 1e-8
                    and -1e-8 <= state[12] <= 1 + 1e-8):
                self.invalid_sample_counts[
                    "adaptation_or_trauma_outside_unit_interval"
                ] += 1
            if values["sync_workload"] < 0 or values["marginal_workload"] < 0:
                self.invalid_sample_counts["negative_workload"] += 1

    def report(self, rhs_evaluations):
        if not self.samples:
            raise ValueError("no sampled simulation states were audited")
        return {
            "simulation_calls": self.simulation_calls,
            "sampled_output_count": self.samples,
            "completed_rhs_evaluations": rhs_evaluations,
            "states": self.states,
            "derived_observables": self.observables,
            "sampled_domain_violations": self.invalid_sample_counts,
        }


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_report():
    fit = json.loads(FIT.read_text())
    residual_data = json.loads(RESIDUALS.read_text())
    residual_vector = np.concatenate([
        np.asarray(component["residuals_in_declared_order"], dtype=float)
        for component in residual_data["components"].values()
    ])
    if residual_vector.shape != (178,) or not np.isclose(
        residual_vector @ residual_vector, fit["optimizer"]["final_optimizer_score"],
        rtol=0, atol=1e-10,
    ):
        raise ValueError("stored residual vector does not reproduce the fitted objective")
    values = np.asarray([fit["parameters"][name] for name in PARAMETERS], dtype=float)
    drug_records, noise_records = load_anchors(), load_noise()
    evidence = load_evidence()

    trackers = {
        "calibration": DomainRanges(),
        "benchmarks": DomainRanges(),
        "model_only_scenarios": DomainRanges(),
    }
    active_group = ["calibration"]
    rhs_counts = {
        "calibration": 0,
        "benchmarks": 0,
        "model_only_scenarios": 0,
    }
    original_rhs = __import__(
        "src.model_equations", fromlist=["rhs"]
    ).rhs
    model_module = __import__("src.model_equations", fromlist=["rhs"])

    def counted_rhs(time, state, inputs, closure, parameters=Parameters()):
        derivative = original_rhs(time, state, inputs, closure, parameters)
        rhs_counts[active_group[0]] += 1
        return derivative

    model_module.rhs = counted_rhs
    original_benchmark_rhs = benchmarks.rhs
    benchmarks.rhs = counted_rhs
    originals = {}
    for module in SIMULATION_MODULES:
        original = module.simulate
        originals[module] = original

        def capture(times, forcing, closure, parameters=Parameters(), *args,
                    _original=original, **kwargs):
            requested_times = np.asarray(times, dtype=float)
            boundaries = np.asarray(kwargs.get("boundaries", ()), dtype=float)
            dense_times = np.unique(np.r_[
                requested_times,
                boundaries,
                np.arange(requested_times[0], requested_times[-1], 0.1),
                requested_times[-1],
            ])
            dense_states = _original(
                dense_times, forcing, closure, parameters, *args, **kwargs
            )
            trackers[active_group[0]].add(
                dense_times, dense_states, forcing, closure, parameters
            )
            return dense_states[:, np.searchsorted(dense_times, requested_times)]

        module.simulate = capture

    try:
        residuals, calibration_record = joint_fit.evaluate_all(
            values, drug_records, noise_records, evidence
        )
        if not np.allclose(residuals, residual_vector, rtol=0, atol=1e-10):
            raise ValueError("recomputed calibration residuals differ from stored vector")
        active_group[0] = "benchmarks"
        benchmark_report = benchmarks.run_benchmarks()
        active_group[0] = "model_only_scenarios"
        model_scenarios.build_result(sample_interval_min=0.25)
    finally:
        model_module.rhs = original_rhs
        benchmarks.rhs = original_benchmark_rhs
        for module, original in originals.items():
            module.simulate = original

    benchmark_report["provenance"]["r9_domain_audit_sha256"] = _sha256(Path(__file__))
    benchmarks.OUTPUT.write_text(
        json.dumps(benchmark_report, indent=2, allow_nan=False) + "\n"
    )
    fit_record = fit["optimizer"]
    starts = []
    for start in fit_record["starts"]:
        starts.append({
            "label": start["start"],
            "initial_parameters": start["initial_parameters"],
            "initial_weighted_sum_squares": start["initial_score"],
            "final_parameters": start["parameters"],
            "final_weighted_sum_squares": start["final_score"],
            "success": start["success"],
            "termination": start["message"],
            "function_evaluations": start["function_evaluations"],
            "jacobian_evaluations": start["jacobian_evaluations"],
            "optimality_unit_box": start["optimality"],
        })

    return {
        "status": "Completed R9 reproducibility and sampled-domain audit",
        "interpretation": (
            "One local, seed-dependent, outcome-informed calibration; the two starts "
            "are nearby and are not robustness or global-optimality evidence."
        ),
        "objective": {
            "weighted_sum_of_squares": float(residuals @ residuals),
            "euclidean_norm": float(np.linalg.norm(residuals)),
            "scipy_half_cost": float((residuals @ residuals) / 2),
            "residual_count": int(len(residuals)),
            "residual_vector_path": str(RESIDUALS.relative_to(ROOT)),
            "residual_vector_sha256": _sha256(RESIDUALS),
            "components": calibration_record["components"],
        },
        "optimizer": {
            "method": fit_record["method"],
            "parameters": fit["parameters"],
            "starts": starts,
            "selection": fit_record["best_start"],
            "software": fit["environment"],
        },
        "residual_construction": {
            "pike_bosher": (
                "12, 12, and 13 baseline-relative EP anchors for the 40, 60, and "
                "80 mg/kg curves, respectively; each curve is divided by "
                "10 mV * sqrt(anchors in that curve)."
            ),
            "wang_1992": (
                "15 baseline-relative EP anchors at each of 105, 115, and 125 dB SPL; "
                "each level is divided by 3 mV * sqrt(15)."
            ),
            "salt_figure_3": (
                "34 EP and 41 endolymph-K digitized anchors after omitting points "
                "within 0.4 min of stimulus switches; each series is baseline-relative "
                "and divided by its 3 mV or 3 mM scale times sqrt(series anchor count)."
            ),
            "other_components": (
                "Asakuma uses two separate rounded endpoint contrasts (10 mV for EP "
                "reduction and 0.2 for resistance ratio); Rybak uses six separately "
                "minimized group-mean reduction contrasts scaled by their reported SEs, "
                "without EP/K covariance; Wang 1990 uses four separate cohort contrasts "
                "with an assumed 45 min delay and 5 mV * sqrt(4) scaling; Salt Table III "
                "uses nine rounded local-episode changes with 3 mV * sqrt(3) scaling "
                "and an assumed 1-3 min post-offset operator."
            ),
        },
        "source_linked_digitized_data": {
            "files": {
                str(path.relative_to(ROOT)): _sha256(path)
                for path in (DIGITIZED_CALIBRATION, asakuma.EVIDENCE,
                             DEFAULT_EVIDENCE, benchmarks.EVIDENCE)
            },
            "scope_note": (
                "These are source-linked digitized/group-summary targets, not raw "
                "subject-level observations. The package is not yet a public archive."
            ),
        },
        "sampled_domain_audit": {
            group: tracker.report(rhs_counts[group])
            for group, tracker in trackers.items()
        },
        "audit_sampling": {
            "calibration_scope": (
                "One recomputation of the stored residual vector at the selected "
                "parameter vector; 18 simulation_calls are calls to simulate, not "
                "optimizer iterations, objective evaluations, or all fit integrations."
            ),
            "requested_output_times": (
                "Union of each simulation's original requested output times, all "
                "forcing boundaries, a 0.1-min grid, and the final requested time."
            ),
            "state_sampling": (
                "States are the simulator outputs at requested t_eval times. Adaptive "
                "internal solver steps are not added to the reported extrema."
            ),
            "equation_domain_checks": (
                "Every completed right-hand-side evaluation during the audited "
                "integrations completed without a logarithm/division domain error."
            ),
            "limitations": (
                "Output-grid and right-hand-side checks do not prove continuous-time "
                "invariance or physiological admissibility."
            ),
        },
        "sampled_domain_rules": {
            "strictly_positive_state_indices": [
                STATE_NAMES[index] for index in (0, 1, 2, 3, 4, 5, 6, 7, 9, 10)
            ],
            "strictly_positive_derived_quantities": [
                "endolymph_volume",
                "endolymph_k",
            ],
            "adaptation_and_trauma_bounds": [0.0, 1.0],
            "adaptation_and_trauma_roundoff_tolerance": 1e-8,
            "clipping": False,
        },
        "domain_interpretation": (
            "Ranges summarize simulator outputs at requested times, including original "
            "finer outputs, forcing boundaries, and a 0.1-min audit grid; adaptive "
            "internal solver steps are not included in the extrema. The 18 calibration "
            "simulation calls comprise one recomputation of the fitted residual vector, "
            "not the optimizer's iterations or all fit integrations. Every completed "
            "right-hand-side evaluation passed the model's logarithm/division domain "
            "checks. This is not a continuous-time invariant proof; physiological "
            "screen crossings remain unpenalized. Tiny negative adaptation or trauma "
            "values within the model's 1e-8 validation tolerance are reported, not clipped."
        ),
        "input_sha256": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in INPUT_SOURCES
        },
    }


def main():
    report = build_report()
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {OUTPUT}")
    print("Sampled domain violations:", {
        group: audit["sampled_domain_violations"]
        for group, audit in report["sampled_domain_audit"].items()
    })


if __name__ == "__main__":
    main()
