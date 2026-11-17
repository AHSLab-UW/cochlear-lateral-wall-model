"""Joint nine-parameter Tier 1 objective; Salt Figures 3/5 are injected explicitly.

``python -m src.all_study_fit`` only checks the starting candidate
with Radau/BDF and a finer extrema grid; it neither optimizes nor writes results.
The incomplete baseline requires ``allow_missing_traces=True`` at API level.

For integration, pass ``salt_trace_callback(context, data)`` and
``salt_trace_data`` to evaluate_all, numerical_baseline_check, or fit_joint.
Return SaltTraceContribution with both figure residual vectors already weighted,
their predictions, a weighting description, and the actual extraction/source
paths. No callback-specific parameters are fitted. The context supplies the
current nine values, model, closure, baseline and solver settings on every call.
"""

from dataclasses import asdict, dataclass
import hashlib
import inspect
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
    salt_table3_fit as salt,
    drug_calibration,
    parameter_schema,
    scaled_parameter_fit,
    kinetics_fit,
    wang_noise_cohorts as wang,
    wang_noise_traces,
    pike_trace_diagnostic,
    plot_initial_fit,
)
from src.model_equations import baseline, related_code_closure
from src.parameter_schema import HIGH, LOW, PARAMETERS
from src.scaled_parameter_fit import bounded_jacobian, from_unit
from src.pike_trace_diagnostic import IDS, ROOT, SOURCE, load_anchors


START = salt.OUTPUT
TRACE_COMPONENTS = ("salt_figure_3", "salt_figure_5")
LIMITATIONS = (
    "Illustrative numerical weights, not a likelihood or independent-likelihood "
    "contributions; no confidence intervals or independent validation are implied.",
    "Salt Table III and Figures 3/5 may share animals; within-trace, repeated-episode, "
    "EP/K and table/trace correlations are unknown and not modeled.",
    "Salt's 1–3 min timing describes auditory input-output measurements, not direct "
    "EP measurements. The inherited Table III 1–3 min EP averaging window is an "
    "explicit timing hypothesis, not a source-established EP observation time.",
    "Salt's same-number dBA-to-model-SPL mapping and five-minute quiet intervals "
    "are fixed conditional assumptions, not fitted conversions or exact protocols.",
    "Wang 1990 uses separate cohorts and an assumed fixed 45-minute delay; its "
    "30–60 minute timing uncertainty is not fitted. Wang 1992 uses digitized means.",
    "Drug exposure/species proxies, treated K boundary and related-code hybrid "
    "closure remain unchanged assumptions; physical closure is not established.",
    "Local termination, bounds and sampled extrema do not establish global or "
    "local optimality, identifiability, physiological validity or model stability.",
    "All outcomes and the starting candidate are outcome-informed; inclusion of "
    "residuals does not establish that all targets are fitted acceptably.",
)


@dataclass(frozen=True)
class SolverSettings:
    step: float = .1
    method: str = "Radau"
    rtol: float = 2e-7
    atol: float = 2e-9

    def __post_init__(self):
        if self.method not in ("Radau", "BDF"):
            raise ValueError("solver method must be Radau or BDF")
        if (not np.isfinite([self.step, self.rtol, self.atol]).all()
                or min(self.step, self.rtol, self.atol) <= 0
                or not np.isclose(120 / self.step, round(120 / self.step))):
            raise ValueError("positive solver tolerances and a grid dividing 120 are required")


@dataclass(frozen=True)
class SaltTraceContext:
    values: tuple
    drug_values: tuple
    sound_values: tuple
    model_parameters: object
    closure: object
    initial_state: tuple
    solver: SolverSettings
    offset_db: float = 0.


@dataclass(frozen=True)
class SaltTraceContribution:
    """Both keys in TRACE_COMPONENTS are mandatory, with nonempty finite vectors."""

    residuals: dict
    predictions: dict
    weighting: str
    source_paths: tuple


def _validated_values(values):
    x = np.asarray(values, dtype=float)
    if (x.shape != LOW.shape or not np.isfinite(x).all()
            or np.any(x < LOW) or np.any(x > HIGH)):
        raise ValueError("nine finite existing parameters within declared bounds are required")
    return x


def load_start(path=START):
    """Read the six drug and three sound values from the conditional Salt fit."""
    data = json.loads(Path(path).read_text())
    if data["status"] != "completed conditional joint Wang and Salt Table III fit":
        raise ValueError("expected the completed conditional Salt Table III starting candidate")
    drug = data["candidate"]["drug_parameter_values"]
    sound = data["fit"]["optimizer"]["fitted_sound_parameters"]
    return _validated_values([drug[name] for name in PARAMETERS[:6]]
                             + [sound[name] for name in PARAMETERS[6:]])


def _component(residuals, predictions, weighting, start):
    r = np.asarray(residuals, dtype=float)
    if r.ndim != 1 or not len(r) or not np.isfinite(r).all():
        raise ValueError("each component requires a nonempty finite residual vector")
    if not isinstance(weighting, str) or not weighting.strip():
        raise ValueError("each component requires an explicit weighting description")
    return r, {
        "residual_slice": [start, start + len(r)],
        "residual_count": len(r),
        "weighted_sum_squares": float(r @ r),
        "weighted_residuals": r.tolist(),
        "weighting": weighting,
        "predictions": predictions,
    }


def evaluate_all(values, drug_records, noise_records, *, salt_trace_callback=None,
                 salt_trace_data=None, allow_missing_traces=False,
                 solver=SolverSettings()):
    """Evaluate every study at the same nine values, returning residuals and audit.

    Missing Salt traces raise by default. Explicit incomplete mode is for baseline
    diagnostics only; fit_joint never permits an absent or empty trace component.
    """
    x = _validated_values(values)
    if salt_trace_callback is None:
        if salt_trace_data is not None:
            raise ValueError("Salt trace data require a residual callback")
        if not allow_missing_traces:
            raise ValueError("Salt Figures 3/5 callback and data are required")
    elif salt_trace_data is None:
        raise ValueError("Salt trace callback requires explicit trace data")
    drug, sound = x[:6], x[6:]
    p = wang_noise_traces.make_parameters(sound, drug)
    closure = related_code_closure()
    initial = baseline(p)
    kwargs = {"method": solver.method, "rtol": solver.rtol, "atol": solver.atol}
    dr, dp = drug_calibration.evaluate(
        drug[:4], drug_records, step=solver.step, base_parameters=p, **kwargs)
    nr, npred = wang_noise_traces.evaluate_noise(
        sound, drug, noise_records, **kwargs)
    wr, wp = wang.wang1990(sound, drug, delay=45., **kwargs)
    sp = salt.salt_predictions(sound, drug, initial, offset_db=0, **kwargs)
    sr = salt.salt_residuals(sp)

    pieces, components = [], {}

    def add(name, residuals, predictions, weighting):
        r, detail = _component(residuals, predictions, weighting, sum(map(len, pieces)))
        pieces.append(r)
        components[name] = detail

    pike_count = sum(sum(row["curve"] == key for row in drug_records) for key in IDS)
    expected = pike_count + 2 + 2 * len(joint_calibration_screen.RYBAK)
    if len(dr) != expected:
        raise ValueError("drug evaluator residual layout changed; review study attribution")
    add("pike_bosher", dr[:pike_count], {key: dp[key] for key in IDS},
        "Baseline-relative EP error / (10 mV * sqrt(anchors per dose curve)).")
    add("asakuma", dr[pike_count:pike_count + 2], dp["asakuma_60"],
        "Resistance ratio error / 0.2; baseline-minus-nadir EP error / 10 mV.")
    add("rybak", dr[pike_count + 2:],
        {key: value for key, value in dp.items() if key.startswith("rybak_")},
        "EP and K group-mean reduction errors / reported SE; covariance unknown.")
    add("wang_1992", nr, npred,
        "Baseline-relative EP error / (3 mV * sqrt(anchors per sound level)).")
    add("wang_1990", wr, wp,
        "Separate-cohort EP difference error / (5 mV * sqrt(number of cohorts)).")
    add("salt_table_iii", sr, salt._salt_report_rows(sp),
        "Local-episode delta EP error / (3 mV * sqrt(3 episodes)); no SD/n precision.")
    trace_sources = []
    trace_integrated = salt_trace_callback is not None
    if trace_integrated:
        context = SaltTraceContext(
            tuple(x), tuple(drug), tuple(sound), p, closure, tuple(initial), solver)
        contribution = salt_trace_callback(context, salt_trace_data)
        if not isinstance(contribution, SaltTraceContribution):
            raise ValueError("callback must return SaltTraceContribution")
        if (set(contribution.residuals) != set(TRACE_COMPONENTS)
                or set(contribution.predictions) != set(TRACE_COMPONENTS)):
            raise ValueError("both Salt Figure 3 and Figure 5 components are required")
        if not contribution.source_paths:
            raise ValueError("trace extraction/source paths must be recorded")
        trace_sources = [str(Path(path).resolve(strict=True))
                         for path in contribution.source_paths]
        for key in TRACE_COMPONENTS:
            add(key, contribution.residuals[key], contribution.predictions[key],
                contribution.weighting)
    residuals = np.concatenate(pieces)
    return residuals, {
        "status": ("exploratory all-six-study objective evaluated; target fit not established"
                   if trace_integrated else
                   "incomplete baseline diagnostic: Salt Figures 3/5 are missing"),
        "trace_integrated": trace_integrated,
        "missing_components": [] if trace_integrated else list(TRACE_COMPONENTS),
        "all_targets_fit_established": False,
        "residual_count": len(residuals),
        "weighted_sum_squares": float(residuals @ residuals),
        "components": components,
        "trace_source_paths": trace_sources,
        "parameters": dict(zip(PARAMETERS, map(float, x))),
        "bounds": dict(zip(PARAMETERS, [[float(a), float(b)] for a, b in zip(LOW, HIGH)])),
        "at_lower_bound": [name for name, value, low in zip(PARAMETERS, x, LOW)
                           if np.isclose(value, low, rtol=0, atol=1e-8)],
        "at_upper_bound": [name for name, value, high in zip(PARAMETERS, x, HIGH)
                           if np.isclose(value, high, rtol=0, atol=1e-8)],
        "solver": asdict(solver),
        "all_model_parameter_values": asdict(p),
        "closure": closure.description,
        "initial_state": initial.tolist(),
        "salt_table_iii_timing": "hypothetical EP window 1–3 min after offset, not direct source EP timing",
        "salt_input_mapping": salt.input_scenarios()[0],
        "wang_1990_assumed_delay_min": 45.,
        "objective_is_likelihood": False,
        "correlations": "unknown within and between Salt table/traces; animals may overlap",
        "limitations": list(LIMITATIONS),
    }


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError("trace data must be JSON serializable (numpy arrays are supported)")


def _data_hash(data):
    raw = json.dumps(data, default=_jsonable, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def provenance(drug_records, noise_records, *, start_path=START,
               salt_trace_callback=None, salt_trace_data=None, trace_source_paths=()):
    """Hash actual inputs/code, including injected data and callback source."""
    modules = (
        model_equations, manuscript_audit, drug_calibration,
        kinetics_fit, parameter_schema, scaled_parameter_fit,
        wang_noise_traces, wang, salt, pike_trace_diagnostic,
        asakuma_ep_endpoint_sensitivity, joint_calibration_screen, plot_initial_fit,
    )
    paths = {Path(inspect.getfile(module)).resolve() for module in modules}
    paths.update((Path(__file__).resolve(), Path(start_path).resolve(), SOURCE,
                  asakuma_ep_endpoint_sensitivity.EVIDENCE,
                  manuscript_audit.MANUSCRIPT, salt.SALT_PAPER,
                  salt.WANG1990_PAPER, salt.WANG1992_PAPER))
    paths.update(ROOT / "refs" / "target_source" / name for name in (
        "pike_1980_hear_res.pdf", "bosher_1979_j_physiol.pdf",
        "asakuma_1980_otolaryngol_head_neck_surg.pdf", "rybak_1982_hear_res.pdf"))
    paths.update(Path(path).resolve() for path in trace_source_paths)
    callback_source = None
    if salt_trace_callback is not None:
        try:
            callback_source = inspect.getsourcefile(salt_trace_callback)
        except TypeError:
            pass
        if callback_source is not None and Path(callback_source).is_file():
            paths.add(Path(callback_source).resolve())
        else:
            callback_source = None
    noise_data = [(level, times.tolist(), ep.tolist()) for level, times, ep in noise_records]
    return {
        "source_sha256": {
            str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path):
                hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)
        },
        "drug_records_sha256": _data_hash(drug_records),
        "noise_records_sha256": _data_hash(noise_data),
        "salt_trace_data_sha256": (None if salt_trace_data is None
                                   else _data_hash(salt_trace_data)),
        "salt_trace_callback": (None if salt_trace_callback is None else
                                getattr(salt_trace_callback, "__qualname__",
                                        type(salt_trace_callback).__qualname__)),
        "callback_source_file": callback_source,
        "callback_source_unavailable": salt_trace_callback is not None and callback_source is None,
        "environment": {"python": platform.python_version(),
                        "numpy": np.__version__, "scipy": scipy.__version__},
    }


def numerical_baseline_check(*, start_path=START, salt_trace_callback=None,
                             salt_trace_data=None, allow_missing_traces=False):
    """No optimization or file output; compare solvers and sampled-extrema grids."""
    values, drug_records = load_start(start_path), load_anchors()
    noise_records = wang_noise_traces.load_noise()
    checks = []
    reference = None
    for settings in (SolverSettings(rtol=1e-8, atol=1e-10),
                     SolverSettings(method="BDF", rtol=1e-8, atol=1e-10),
                     SolverSettings(step=.025, rtol=1e-8, atol=1e-10)):
        r, detail = evaluate_all(
            values, drug_records, noise_records, solver=settings,
            salt_trace_callback=salt_trace_callback, salt_trace_data=salt_trace_data,
            allow_missing_traces=allow_missing_traces)
        if reference is None:
            reference = r
        if r.shape != reference.shape:
            raise ValueError("residual dimension changed across solver/grid checks")
        checks.append({**detail, "maximum_residual_difference_from_first":
                       float(np.max(np.abs(r - reference)))})
    return {
        "status": checks[0]["status"],
        "optimizer": {"performed": False, "reason": "baseline numerical check only"},
        "checks": checks,
        "provenance": provenance(
            drug_records, noise_records, start_path=start_path,
            salt_trace_callback=salt_trace_callback, salt_trace_data=salt_trace_data,
            trace_source_paths=checks[0]["trace_source_paths"]),
    }


def fit_joint(*, salt_trace_callback, salt_trace_data, start_path=START,
              solver=SolverSettings(), max_nfev=70, jacobian_step=.001):
    """Explicit local optimization API, disabled until both Salt traces contribute.

    Returns an audit, never writes a final result. Optimizer success is reported
    separately from scientific fit adequacy. No final optimization is run by CLI.
    """
    if salt_trace_callback is None or salt_trace_data is None:
        raise ValueError("joint fitting is blocked pending Salt Figures 3/5 callback and data")
    if not isinstance(max_nfev, int) or max_nfev < 1:
        raise ValueError("max_nfev must be a positive integer")
    if not np.isfinite(jacobian_step) or not 0 < jacobian_step < 1:
        raise ValueError("jacobian_step must be between zero and one")
    x0, drug_records = load_start(start_path), load_anchors()
    noise_records = wang_noise_traces.load_noise()
    u0 = (x0 - LOW) / (HIGH - LOW)
    initial, initial_report = evaluate_all(
        x0, drug_records, noise_records, solver=solver,
        salt_trace_callback=salt_trace_callback, salt_trace_data=salt_trace_data)
    input_provenance = provenance(
        drug_records, noise_records, start_path=start_path,
        salt_trace_callback=salt_trace_callback, salt_trace_data=salt_trace_data,
        trace_source_paths=initial_report["trace_source_paths"])
    layout = {key: row["residual_count"] for key, row in initial_report["components"].items()}
    residual_calls = 0

    def residual(unit):
        nonlocal residual_calls
        residual_calls += 1
        r, detail = evaluate_all(
            from_unit(unit), drug_records, noise_records, solver=solver,
            salt_trace_callback=salt_trace_callback, salt_trace_data=salt_trace_data)
        if {key: row["residual_count"] for key, row in detail["components"].items()} != layout:
            raise ValueError("component residual dimensions changed during optimization")
        if detail["trace_source_paths"] != initial_report["trace_source_paths"]:
            raise ValueError("trace sources changed during optimization")
        return r

    # Installed SciPy stubs incorrectly restrict jac to str; runtime accepts callables.
    opt = least_squares(
        residual, u0,
        jac=lambda u: bounded_jacobian(residual, u, step=jacobian_step),  # pyright: ignore[reportArgumentType]
        bounds=(np.zeros(9), np.ones(9)), method="trf", max_nfev=max_nfev,
        ftol=1e-6, xtol=1e-6, gtol=1e-6)
    _, report = evaluate_all(
        from_unit(opt.x), drug_records, noise_records, solver=solver,
        salt_trace_callback=salt_trace_callback, salt_trace_data=salt_trace_data)
    report["optimizer"] = {
        "performed": True, "method": "scipy least_squares trf, unit-box coordinates",
        "loss": "linear", "jacobian": "bounded finite differences",
        "jacobian_step_unit_box": jacobian_step, "max_nfev": max_nfev,
        "ftol": 1e-6, "xtol": 1e-6, "gtol": 1e-6,
        "success": bool(opt.success), "status": int(opt.status), "message": str(opt.message),
        "function_evaluations": int(opt.nfev),
        "residual_calls_including_finite_differences": residual_calls,
        "jacobian_evaluations": None if opt.njev is None else int(opt.njev),
        "optimality": float(opt.optimality),
        "active_bounds": dict(zip(PARAMETERS, map(int, opt.active_mask))),
        "initial_parameters": dict(zip(PARAMETERS, map(float, x0))),
        "initial_weighted_sum_squares": float(initial @ initial),
        "final_optimizer_weighted_sum_squares": float(opt.fun @ opt.fun),
        "cost": float(opt.cost),
        "optimality_established": False,
    }
    report["provenance"] = input_provenance
    return report


def main():
    print(json.dumps(numerical_baseline_check(allow_missing_traces=True),
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
