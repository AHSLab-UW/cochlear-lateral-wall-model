"""Conditional multidosage calibration of a restricted exploratory ODE model."""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from src.cochlear_model import Inputs, Parameters, simulate
from src.pike_trace_diagnostic import IDS, load_anchors


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "results" / "exploratory"
NAMES = ("potassium_flux", "effective_dose_scale_mg_kg", "exposure_decay_min")
LOWER = np.array([0.05, 30.0, 3.0])
UPPER = np.array([0.5, 400.0, 80.0])
STARTS = ([0.2, 60.0, 40.0], [0.45, 35.0, 20.0], [0.3, 150.0, 70.0])


def drug_proxy(time: float, dose: float, dose_scale: float, decay: float) -> float:
    """Dimensionless effective inhibition; local concentration is not measured."""
    if not all(np.isfinite(x) for x in (time, dose, dose_scale, decay)):
        raise ValueError("input values must be finite")
    if time < 0 or dose < 0 or dose_scale <= 0 or decay <= 0:
        raise ValueError("invalid input to dose-to-proxy model")
    exposure = dose * (-np.expm1(-time / 0.3)) * np.exp(-time / decay)
    return float(exposure / (dose_scale + exposure))


def simulate_groups(records: list[dict], parameters: np.ndarray,
                    *, rtol: float = 1e-6, atol: float = 1e-8) -> list[dict]:
    """Evaluate every group at its manual-anchor times, without interpolation."""
    flux, dose_scale, decay = parameters
    result = []
    for identifier in IDS:
        group = [row for row in records if row["curve"] == identifier]
        if len(group) < 2 or group[0]["time_min"] != 0:
            raise ValueError(f"{identifier} must have ordered observations beginning at zero")
        dose = float(group[0]["dose_mg_kg"])
        if any(row["dose_mg_kg"] != dose for row in group):
            raise ValueError(f"inconsistent dose for {identifier}")
        times = np.array([row["time_min"] for row in group])
        solution = simulate(
            times,
            lambda t: Inputs(drug=drug_proxy(t, dose, dose_scale, decay)),
            parameters=Parameters(potassium_flux=float(flux), drug_sink=0),
            rtol=rtol, atol=atol,
        )
        for row, ep in zip(group, solution.y[10]):
            result.append({**row, "modeled_ep_mV": float(ep),
                           "residual_mV": float(ep - row["ep_mV"])})
    return result


def fit(records: list[dict]) -> dict:
    """Fit flux, effective dose scale and decay, balancing the three curves."""
    counts = {key: sum(row["curve"] == key for row in records) for key in IDS}
    if any(count < 2 for count in counts.values()):
        raise ValueError("all three dose curves are required")

    def residual(log_parameters: np.ndarray) -> np.ndarray:
        rows = simulate_groups(records, np.exp(log_parameters))
        return np.array([row["residual_mV"] / np.sqrt(counts[row["curve"]])
                         for row in rows])

    results = [
        least_squares(residual, np.log(start),
                      bounds=(np.log(LOWER), np.log(UPPER)),
                      max_nfev=100, diff_step=1e-3, ftol=1e-6, xtol=1e-6, gtol=1e-6)
        for start in STARTS
    ]
    if not all(result.success for result in results):
        raise RuntimeError(f"optimization did not converge: {[r.message for r in results]}")
    best = min(results, key=lambda result: result.cost)
    parameters = np.exp(best.x)
    rows = simulate_groups(records, parameters, rtol=1e-8, atol=1e-10)
    return {
        "parameters": dict(zip(NAMES, map(float, parameters))),
        "numerical_bounds": dict(zip(NAMES, [[float(l), float(u)]
                                             for l, u in zip(LOWER, UPPER)])),
        "active_bounds_by_start": [r.active_mask.tolist() for r in results],
        "cost_by_start": [float(r.cost) for r in results],
        "evaluations_by_start": [int(r.nfev) for r in results],
        "rmse_mV_by_curve": {
            key: float(np.sqrt(np.mean([row["residual_mV"] ** 2 for row in rows
                                       if row["curve"] == key])))
            for key in IDS
        },
        "points": rows,
    }


def main() -> None:
    result = fit(load_anchors())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "pike_ode_fit.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["points"][0]))
        writer.writeheader()
        writer.writerows(result["points"])
    result.pop("points")
    report = {
        "status": "conditional partial 14-state fit; not a full calibration or validation",
        "data": "Pike and Bosher 1980 Fig. 1 manual anchors; n=2,3,3 group means",
        "objective": "equal total squared-error weight per dose; not inverse variance",
        "assumptions": [
            "shared effective dose scale and exposure decay; fixed 0.3 min rise",
            "zero external potassium sink, default endolymph-volume closure, other parameters fixed",
            "relative 80 mV baseline; no fitted baseline, species scaling or observation operator",
            "bounds are numerical exploration bounds, not established physiological bounds",
            "digitized points are correlated; uncertainty and covariance are not estimated",
        ],
        "limitations": [
            "fit cannot separately identify local exposure and transporter affinity",
            "only three parameters vary; no joint multi-subsystem fit or uncertainty analysis",
            "no held-out prediction and no proof of global optimum",
            "dose-input function is hypothetical and local drug exposure is unmeasured",
            "search bounds were narrowed after an invalid residual endolymph potassium amount occurred",
        ],
        **result,
    }
    (OUTPUT / "pike_ode_fit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
