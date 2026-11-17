"""Conditional residuals for digitized Salt and Konishi Figures 3 and 5."""

import numpy as np

from src.all_study_fit import SaltTraceContribution
from src.model_equations import Inputs, broadband_sound_multiplier, endolymph, simulate
from src.salt_1979_digitize import DEFAULT_EVIDENCE, calibrated_points, verify_source_hashes


FIG3_INTERVALS = ((0., 20.), (30., 50.), (60., 80.))
EVENT_MARGIN_MIN = .4
EP_SCALE_MV = 3.
K_SCALE_MM = 3.


def _selected(points, events):
    """Keep calibrated post-onset points away from uncertain switching times."""
    return [
        point for point in points
        if point["time_min"] is not None
        and point["time_min"] > EVENT_MARGIN_MIN
        and all(abs(point["time_min"] - event) > EVENT_MARGIN_MIN for event in events)
    ]


def _run(context, times, level, intervals):
    evaluation_times = np.unique(np.r_[0., times])
    final = float(evaluation_times[-1])
    events = sorted({time for interval in intervals for time in interval if 0. < time < final})
    multiplier = broadband_sound_multiplier(level + context.offset_db)

    def forcing(time):
        return Inputs(sound_multiplier=multiplier if any(
            start <= time < end for start, end in intervals) else 1.)

    return simulate(
        evaluation_times, forcing, context.closure, context.model_parameters,
        initial=context.initial_state, boundaries=events,
        method=context.solver.method, rtol=context.solver.rtol, atol=context.solver.atol,
    ), evaluation_times


def _series(points, predicted, baseline, scale):
    observed = np.asarray([point["value"] - baseline for point in points])
    predicted = np.asarray(predicted)
    residual = (predicted - observed) / (scale * np.sqrt(len(points)))
    return residual, {
        "time_min": [point["time_min"] for point in points],
        "source_pixel_xy": [[point["x_pixel"], point["y_pixel"]] for point in points],
        "observed_change": observed.tolist(),
        "modeled_change": predicted.tolist(),
        "observed_baseline": float(baseline),
        "scale": scale,
        "rmse": float(np.sqrt(np.mean((predicted - observed) ** 2))),
    }


def evaluate_salt_traces(context, evidence):
    """Use baseline-relative, per-trace balanced errors under fixed sound mapping."""
    mismatches = verify_source_hashes(evidence)
    if mismatches:
        raise ValueError(f"Salt source or rendered-image hashes changed: {mismatches}")
    events = [edge for interval in FIG3_INTERVALS for edge in interval]
    fig3_points = {
        name: _selected(calibrated_points(evidence, "fig3", name), events)
        for name in ("ep", "endolymph_k")
    }
    if any(len(points) < 2 for points in fig3_points.values()):
        raise ValueError("insufficient calibrated Figure 3 points")
    times = [point["time_min"] for points in fig3_points.values() for point in points]
    states, sampled_times = _run(context, times, 115., FIG3_INTERVALS)
    k = np.asarray([endolymph(state, context.model_parameters)[1] for state in states.T])
    fig3_residuals, fig3_predictions = [], {}
    for name, state, scale in (("ep", states[8], EP_SCALE_MV),
                               ("endolymph_k", k, K_SCALE_MM)):
        points = fig3_points[name]
        initial_points = [
            point["value"] for point in calibrated_points(evidence, "fig3", name)
            if point["time_min"] < -EVENT_MARGIN_MIN
        ]
        if not initial_points:
            raise ValueError("Figure 3 requires a pre-exposure baseline")
        baseline = float(np.mean(initial_points))
        indices = np.searchsorted(sampled_times, [point["time_min"] for point in points])
        residual, prediction = _series(points, state[indices] - state[0], baseline, scale)
        prediction["unit"] = "mV" if name == "ep" else "mM"
        fig3_residuals.extend(residual)
        fig3_predictions[name] = prediction

    fig5_residuals, fig5_predictions = [], {}
    for level in (95, 105, 115):
        points = _selected(calibrated_points(evidence, "fig5", f"{level}_dba"), (0., 20.))
        if len(points) < 2:
            raise ValueError(f"insufficient calibrated Figure 5 points at {level} dBA")
        states, sampled_times = _run(context, [point["time_min"] for point in points],
                                     level, ((0., 20.),))
        indices = np.searchsorted(sampled_times, [point["time_min"] for point in points])
        residual, prediction = _series(points, states[8, indices] - states[8, 0],
                                       0., EP_SCALE_MV)
        prediction["unit"] = "mV"
        fig5_residuals.extend(residual)
        fig5_predictions[str(level)] = prediction
    sources = [DEFAULT_EVIDENCE, DEFAULT_EVIDENCE.parent.parent.parent / evidence["source"]["file"]]
    sources.extend(DEFAULT_EVIDENCE.parent.parent.parent / row["file"]
                   for row in evidence["source"]["rendering"]["images"].values())
    return SaltTraceContribution(
        residuals={"salt_figure_3": np.asarray(fig3_residuals),
                   "salt_figure_5": np.asarray(fig5_residuals)},
        predictions={"salt_figure_3": fig3_predictions, "salt_figure_5": fig5_predictions},
        weighting=(
            "Each individual trace uses baseline-relative errors divided by sqrt(point count) "
            "and an illustrative scale of 3 mV (EP) or 3 mM (K); not a likelihood. "
            "Figure 3 is one paired EP/K animal; Figure 5 has one animal per level. "
            "Points within 0.4 min of stimulus switches and Fig. 5 unscaled post-offset "
            "points are omitted; temporal localization is uncertain. Fig. 3 uses "
            "0–20, 30–50, 60–80 min exposures and 10-min quiet intervals."
        ),
        source_paths=tuple(sources),
    )
