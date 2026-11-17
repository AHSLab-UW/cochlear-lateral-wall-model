"""Calibrate manually digitized Salt & Konishi (1979) figure coordinates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE = ROOT / "refs/research_evidence/salt_1979_fig3_fig5_digitized.json"


def load_evidence(path: Path = DEFAULT_EVIDENCE) -> dict[str, Any]:
    """Load the source-linked raw coordinates and their axis calibrations."""
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _linear_fit(anchors: list[dict[str, float]]) -> tuple[float, float]:
    """Return slope and intercept for least-squares value = slope*pixel + intercept."""
    if len(anchors) < 2:
        raise ValueError("At least two calibration anchors are required")
    pixels = [float(anchor["pixel"]) for anchor in anchors]
    values = [float(anchor["value"]) for anchor in anchors]
    mean_pixel = sum(pixels) / len(pixels)
    mean_value = sum(values) / len(values)
    denominator = sum((pixel - mean_pixel) ** 2 for pixel in pixels)
    if denominator == 0:
        raise ValueError("Calibration anchors must have distinct pixel coordinates")
    slope = sum(
        (pixel - mean_pixel) * (value - mean_value)
        for pixel, value in zip(pixels, values)
    ) / denominator
    return slope, mean_value - slope * mean_pixel


def _map_pixel(pixel: float, anchors: list[dict[str, float]]) -> float:
    slope, intercept = _linear_fit(anchors)
    return slope * pixel + intercept


def calibrated_points(
    evidence: dict[str, Any], figure_id: str, series_id: str
) -> list[dict[str, Any]]:
    """Return raw crop pixels with calibrated time and measurement values.

    Fig. 5 times are only returned during its printed 20-min bar. The onset
    event is anchored to t=0; pre-exposure and post-offset horizontal spacing
    is intentionally left uncalibrated.
    """
    figure = evidence["figures"][figure_id]
    series = figure["series"][series_id]
    output: list[dict[str, Any]] = []

    for index, raw_point in enumerate(series["points"]):
        x_pixel, y_pixel = raw_point[:2]
        point: dict[str, Any] = {
            "x_pixel": x_pixel,
            "y_pixel": y_pixel,
            "break_before": index
            in series.get("break_before_point_indices_zero_based", []),
        }

        if figure_id == "fig3":
            point["time_min"] = _map_pixel(
                x_pixel, figure["horizontal_axis"]["anchors"]
            )
            point["value"] = _map_pixel(y_pixel, series["vertical_axis"]["anchors"])
            point["unit"] = series["unit"]
            point["time_uncertainty_min"] = 0.4
            point["value_uncertainty"] = (
                0.3 if series_id == "ep" else 0.5
            )
            point["phase"] = _fig3_phase(point["time_min"], figure["noise_intervals"])
        elif figure_id == "fig5":
            phase = raw_point[2]
            anchors = figure["horizontal_axis"]["anchors"]
            lower = min(anchor["pixel"] for anchor in anchors)
            upper = max(anchor["pixel"] for anchor in anchors)
            if phase == "onset":
                point["time_min"] = 0.0
            elif lower <= x_pixel <= upper:
                point["time_min"] = _map_pixel(x_pixel, anchors)
            else:
                point["time_min"] = None
            scale = figure["vertical_scale"]
            baseline_y = series["baseline_y_pixel"]
            scale_pixels = (
                scale["scale_bar_bottom_pixel"] - scale["scale_bar_top_pixel"]
            )
            point["value"] = (
                baseline_y - y_pixel
            ) * scale["scale_bar_value"] / scale_pixels
            point["unit"] = series["unit"]
            point["value_uncertainty"] = 0.25
            point["time_uncertainty_min"] = (
                0.3 if point["time_min"] is not None else None
            )
            point["phase"] = phase
        else:
            raise ValueError(f"Unsupported figure: {figure_id}")

        output.append(point)
    return output


def _fig3_phase(time_min: float, intervals: list[dict[str, float]]) -> str:
    for interval in intervals:
        if interval["start_min"] <= time_min <= interval["end_min"]:
            return "noise"
    return "quiet_or_pre_post"


def verify_source_hashes(evidence: dict[str, Any]) -> list[str]:
    """Return relative source-image paths whose recorded SHA-256 does not match."""
    source = evidence["source"]
    checks = [
        (source["file"], source["sha256"]),
        *(
            (image["file"], image["sha256"])
            for image in source["rendering"]["images"].values()
        ),
    ]
    mismatches = []
    for relative_path, expected in checks:
        path = ROOT / relative_path
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            mismatches.append(relative_path)
    return mismatches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--points",
        action="store_true",
        help="print every calibrated point with its source crop pixel coordinates",
    )
    args = parser.parse_args()
    evidence = load_evidence()
    if args.points:
        result = {
            figure_id: {
                series_id: calibrated_points(evidence, figure_id, series_id)
                for series_id in figure["series"]
            }
            for figure_id, figure in evidence["figures"].items()
        }
    else:
        result = {
            figure_id: {
                series_id: {
                    "point_count": len(series["points"]),
                    "unit": series["unit"],
                    "sample_unit": series["sample_unit"],
                }
                for series_id, series in figure["series"].items()
            }
            for figure_id, figure in evidence["figures"].items()
        }
    result["source_hash_mismatches"] = verify_source_hashes(evidence)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
