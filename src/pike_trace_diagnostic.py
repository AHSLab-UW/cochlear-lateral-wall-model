"""Fit an empirical response shape to retained Pike figure anchors.

This does not calibrate the cochlear ODE or identify drug pharmacokinetics.
"""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "refs" / "research_evidence" / "research_evidence.json"
OUTPUT = ROOT / "results" / "exploratory"
IDS = ("pike_40", "pike_60", "pike_80")


def load_anchors(path: Path = SOURCE) -> list[dict]:
    """Load retained manual anchors, not dense points generated from the curves."""
    data = json.loads(path.read_text())
    experiments = data["experiments"]
    records = []
    for identifier in IDS:
        matches = [item for item in experiments if item.get("id") == identifier]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one {identifier} record")
        study = matches[0]
        points = study["manual_observations"]
        if len(points) < 3:
            raise ValueError(f"too few manual anchors for {identifier}")
        previous_time = -1.0
        for point in points:
            time = float(point["time_min"])
            ep = float(point["ep_mV"])
            if not np.isfinite(time) or not np.isfinite(ep) or time <= previous_time:
                raise ValueError(f"invalid or unordered anchor in {identifier}")
            previous_time = time
            records.append({
                "curve": identifier,
                "dose_mg_kg": int(study["dose_mg_kg"]),
                "animals": int(study["n"]),
                "time_min": time,
                "ep_mV": ep,
            })
    return records


def predict(times: np.ndarray, curve_indices: np.ndarray,
            parameters: np.ndarray) -> np.ndarray:
    """Common time constants with separately fitted dose-specific amplitudes."""
    baseline, a40, a60, a80, rise, decay_gap = parameters
    decay = rise + decay_gap
    amplitudes = np.array([a40, a60, a80])
    return baseline - amplitudes[curve_indices] * (
        np.exp(-times / decay) - np.exp(-times / rise)
    )


def fit_anchors(records: list[dict]) -> dict:
    """Unweighted curve-balanced descriptive fit with explicit numerical bounds."""
    if not records:
        raise ValueError("no anchors")
    indices = np.array([IDS.index(row["curve"]) for row in records], dtype=int)
    times = np.array([row["time_min"] for row in records], dtype=float)
    values = np.array([row["ep_mV"] for row in records], dtype=float)
    counts = np.bincount(indices, minlength=len(IDS))
    if np.any(counts == 0):
        raise ValueError("all three curves must be present")
    weights = 1 / np.sqrt(counts[indices])
    lower = np.array([40, 0, 0, 0, 0.05, 0.05])
    upper = np.array([130, 500, 500, 500, 20, 500])
    starts = (
        [82, 100, 170, 190, 1, 45],
        [80, 90, 150, 200, 3, 80],
        [85, 120, 200, 230, 0.5, 120],
    )

    def residual(parameters: np.ndarray) -> np.ndarray:
        return (predict(times, indices, parameters) - values) * weights

    results = [
        least_squares(residual, start, bounds=(lower, upper), max_nfev=5000,
                      ftol=1e-11, xtol=1e-11, gtol=1e-11)
        for start in starts
    ]
    if not all(result.success for result in results):
        raise RuntimeError(f"fit failed: {[r.message for r in results if not r.success]}")
    best = min(results, key=lambda result: result.cost)
    predicted = predict(times, indices, best.x)
    errors = predicted - values
    names = ("baseline_mV", "amplitude_40_mV", "amplitude_60_mV",
             "amplitude_80_mV", "rise_min", "decay_minus_rise_min")
    return {
        "parameters": dict(zip(names, map(float, best.x))),
        "decay_min": float(best.x[4] + best.x[5]),
        "rmse_mV_by_curve": {
            identifier: float(np.sqrt(np.mean(errors[indices == index] ** 2)))
            for index, identifier in enumerate(IDS)
        },
        "cost_by_start": [float(result.cost) for result in results],
        "near_numerical_bound": [
            name for name, value, low, high in zip(names, best.x, lower, upper)
            if min(value - low, high - value) < 0.01 * (high - low)
        ],
        "points": [
            {**record, "fitted_ep_mV": float(pred), "residual_mV": float(err)}
            for record, pred, err in zip(records, predicted, errors)
        ],
    }


def main() -> None:
    records = load_anchors()
    result = fit_anchors(records)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "pike_manual_anchor_diagnostic.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["points"][0]))
        writer.writeheader()
        writer.writerows(result["points"])
    result.pop("points")
    report = {
        "status": "empirical descriptive fit, not cochlear model calibration",
        "source": "Pike and Bosher 1980, Fig. 1, DOI 10.1016/0378-5955(80)90009-x",
        "data": "previously retained manual digitization anchors; overall shape compared visually with figure, individual coordinates not independently redigitized",
        "weights": "each curve receives equal total squared-error weight; not inverse variance",
        "limitations": [
            "group mean traces, n=2,3,3; not independent individual recordings",
            "anchor values and the shared baseline are approximate and correlated",
            "no reported pointwise biological uncertainty; no confidence intervals",
            "simple two-exponential shape and common time constants are assumptions",
            "amplitudes are independent by dose, not an identified dose-response curve",
            "numerical bounds are not evidence-based physiological bounds",
            "time constants are empirical, not measured drug exposure or transport parameters",
        ],
        **result,
    }
    (OUTPUT / "pike_manual_anchor_diagnostic.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
