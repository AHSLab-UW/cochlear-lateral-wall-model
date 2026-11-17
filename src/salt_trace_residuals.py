"""Salt Figure 3-only residuals for the six-study joint fit."""

import hashlib
import json
from pathlib import Path

import numpy as np

from src.model_equations import Inputs, broadband_sound_multiplier, endolymph, simulate
from src.salt_1979_digitize import DEFAULT_EVIDENCE, calibrated_points


ROOT = DEFAULT_EVIDENCE.parents[2]
FIG3_INTERVALS = ((0.0, 20.0), (30.0, 50.0), (60.0, 80.0))
EVENT_MARGIN_MIN = 0.4
EP_SCALE_MV = 3.0
K_SCALE_MM = 3.0
FIG3_IMAGE_KEYS = ("fig3_page", "fig3_crop")


def _selected(points):
    events = [edge for interval in FIG3_INTERVALS for edge in interval]
    return [
        point for point in points
        if point["time_min"] is not None
        and point["time_min"] > EVENT_MARGIN_MIN
        and all(abs(point["time_min"] - event) > EVENT_MARGIN_MIN for event in events)
    ]


def _fig3_source_hashes(evidence):
    images = evidence["source"]["rendering"]["images"]
    hashes = {}
    for key in FIG3_IMAGE_KEYS:
        image = images[key]
        path = ROOT / image["file"]
        expected = image["sha256"]
        if path.exists():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != expected:
                raise ValueError(
                    f"Salt Figure 3 source image hash mismatch: {image['file']}"
                )
        hashes[image["file"]] = expected
    return hashes


def fig3_data_hash(evidence):
    """Hash only the extracted Figure 3 content, excluding all Figure 5 fields."""
    data = {
        "figure": evidence["figures"]["fig3"],
        "source_images": {
            key: evidence["source"]["rendering"]["images"][key]
            for key in FIG3_IMAGE_KEYS
        },
    }
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def evaluate_fig3(context, evidence):
    """Evaluate Figure 3 EP and K only; Figure 5 is never read by this objective."""
    if "fig3" not in evidence.get("figures", {}):
        raise ValueError("Salt Figure 3 extraction is required")
    image_hashes = _fig3_source_hashes(evidence)
    figure = evidence["figures"]["fig3"]
    points_by_series = {
        name: _selected(calibrated_points(evidence, "fig3", name))
        for name in ("ep", "endolymph_k")
    }
    if any(len(points) < 2 for points in points_by_series.values()):
        raise ValueError("insufficient calibrated Figure 3 EP/K points")
    all_times = [point["time_min"] for points in points_by_series.values() for point in points]
    times = np.unique(np.r_[0.0, all_times])
    boundaries = sorted({
        event for interval in FIG3_INTERVALS for event in interval
        if 0.0 < event < float(times[-1])
    })
    multiplier = float(broadband_sound_multiplier(115.0))

    def forcing(time):
        active = any(start <= time < end for start, end in FIG3_INTERVALS)
        return Inputs(sound_multiplier=multiplier if active else 1.0)

    states = simulate(
        times, forcing, context.closure, context.model_parameters,
        initial=np.asarray(context.initial_state, dtype=float), boundaries=boundaries,
        method=context.solver.method, rtol=context.solver.rtol, atol=context.solver.atol,
    )
    k_values = np.asarray([
        endolymph(state, context.model_parameters)[1] for state in states.T
    ])
    residuals, predictions = [], {}
    for name, state, scale, unit in (
            ("ep", states[8], EP_SCALE_MV, "mV"),
            ("endolymph_k", k_values, K_SCALE_MM, "mM")):
        points = points_by_series[name]
        baseline_points = [
            point["value"] for point in calibrated_points(evidence, "fig3", name)
            if point["time_min"] < -EVENT_MARGIN_MIN
        ]
        if not baseline_points:
            raise ValueError(f"Salt Figure 3 {name} series has no pre-exposure baseline")
        observed_baseline = float(np.mean(baseline_points))
        sampled = np.searchsorted(times, [point["time_min"] for point in points])
        observed = np.asarray([point["value"] for point in points]) - observed_baseline
        predicted = state[sampled] - state[0]
        residual = (predicted - observed) / (scale * np.sqrt(len(points)))
        residuals.extend(residual)
        predictions[name] = {
            "unit": unit,
            "time_min": [point["time_min"] for point in points],
            "source_pixel_xy": [[point["x_pixel"], point["y_pixel"]] for point in points],
            "observed_change": observed.tolist(),
            "modeled_change": predicted.tolist(),
            "observed_baseline": observed_baseline,
            "scale": scale,
            "rmse": float(np.sqrt(np.mean((predicted - observed) ** 2))),
        }
    return np.asarray(residuals), {
        "predictions": predictions,
        "weighting": (
            "Each Figure 3 series uses baseline-relative errors divided by "
            "3 mV or 3 mM times sqrt(points in that series); illustrative deterministic "
            "weights, not a likelihood. Fig. 3 is one paired EP/K animal."
        ),
        "fig3_data_sha256": fig3_data_hash(evidence),
        "fig3_source_image_sha256": image_hashes,
        "source_path": str(DEFAULT_EVIDENCE.relative_to(ROOT)),
        "figure_schedule_min": [list(interval) for interval in FIG3_INTERVALS],
        "conditional_input": {
            "noise_dba": 115,
            "model_level_db_spl": 115,
            "interpretation": "same-numeric-level scenario, not an acoustic conversion",
            "sound_multiplier": multiplier,
        },
        "omissions": (
            "Points within 0.4 min of stimulus switches are omitted; temporal localization "
            "is uncertain. Figure 5 is excluded from data selection, objective, and hashes."
        ),
    }
