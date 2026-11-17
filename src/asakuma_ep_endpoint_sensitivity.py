"""Outcome-informed sensitivity to adding the Asakuma EP nadir endpoint."""

import hashlib
import json

import numpy as np
from scipy.optimize import least_squares

from src.asakuma_nadir_refit import OUTPUT as PREVIOUS, corrected_residuals
from src.asakuma_nadir_timing_audit import INPUT as HISTORICAL, RESULTS
from src.joint_calibration_screen import configuration, parameter_space
from src.pike_trace_diagnostic import ROOT, load_anchors


EVIDENCE = ROOT / "refs" / "research_evidence" / "joint_calibration_evidence.json"
OUTPUT = RESULTS / "asakuma_ep_endpoint_sensitivity.json"


def load_endpoint():
    data = json.loads(EVIDENCE.read_text())
    matches = [s for s in data["studies"] if s["id"] == "asakuma_snow_1980"]
    if len(matches) != 1:
        raise ValueError("expected exactly one Asakuma source")
    study = matches[0]
    obs = study["observation"]
    mean, sd, n = float(obs["nadir_ep_mean_mV"]), float(obs["nadir_ep_sd_mV"]), study["group_size"]
    if not np.isfinite(mean) or not np.isfinite(sd) or sd <= 0 or n <= 0:
        raise ValueError("invalid Asakuma EP endpoint")
    return mean, sd, n


def endpoint_residuals(records, names, values, exchange, hill, scale, *,
                       step=1., method="Radau", rtol=1e-6, atol=1e-8):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("EP endpoint scale must be finite and positive")
    residuals, predictions = corrected_residuals(
        records, names, values, exchange, hill, step=step, method=method, rtol=rtol, atol=atol)
    mean, _, _ = load_endpoint()
    ep = predictions["asakuma_sampled_nadir"]["sampled_nadir_ep_mV"]
    return np.append(residuals, (ep - mean) / scale), predictions


def fit(records, names, lower, upper, starts, hill, scale):
    bounds = (np.append(np.log(lower), 0.), np.append(np.log(upper), 10.))
    result_row = {"hill": hill, "ep_scale_mV": float(scale), "attempts": [],
                  "status": "no converged start"}
    completed = []
    for label, case in starts:
        x0 = np.append(np.log([case["parameters"][name] for name in names]), case["exchange"])
        attempt = {"start": label, "initial_parameters": case["parameters"],
                   "initial_exchange": case["exchange"]}
        try:
            def residual(x):
                return endpoint_residuals(records, names, np.exp(x[:-1]), x[-1], hill, scale)[0]

            initial = residual(x0)
            opt = least_squares(residual, x0, bounds=bounds, max_nfev=100,
                                diff_step=1e-3, ftol=1e-6, xtol=1e-6, gtol=1e-6)
            attempt.update({"success": bool(opt.success), "message": str(opt.message),
                            "evaluations": int(opt.nfev),
                            "initial_sum_squares": float(initial @ initial),
                            "sum_squares_optimization_grid": float(opt.fun @ opt.fun),
                            "parameters": dict(zip(names, map(float, np.exp(opt.x[:-1])))),
                            "exchange": float(opt.x[-1]),
                            "active_bounds": dict(zip([*names, "exchange"],
                                                      map(int, opt.active_mask)))})
            if opt.success:
                completed.append(opt)
        except (ValueError, RuntimeError, FloatingPointError, OverflowError) as error:
            attempt.update({"success": False, "error": f"{type(error).__name__}: {error}"})
        result_row["attempts"].append(attempt)
    if not completed:
        return result_row
    best = min(completed, key=lambda x: x.cost)
    values, exchange = np.exp(best.x[:-1]), float(best.x[-1])
    result_row.update({"parameters": dict(zip(names, map(float, values))), "exchange": exchange,
                       "active_bounds": dict(zip([*names, "exchange"], map(int, best.active_mask)))})
    result_row["checks"] = []
    for method, step in (("Radau", .25), ("BDF", .25), ("Radau", .05)):
        try:
            r, predictions = endpoint_residuals(records, names, values, exchange, hill, scale,
                                                step=step, method=method, rtol=1e-8, atol=1e-10)
            n = len(records)
            components = {
                "pike_ep": float(r[:n] @ r[:n]),
                "asakuma_resistance": float(r[n] ** 2),
                "rybak_ep": float(np.sum(r[n+1:n+7:2] ** 2)),
                "rybak_k": float(np.sum(r[n+2:n+7:2] ** 2)),
                "rybak_timing": float(r[-3:-1] @ r[-3:-1]),
                "asakuma_ep": float(r[-1] ** 2),
            }
            result_row["checks"].append({"status": "ok", "method": method,
                                         "grid_step_min": step,
                                         "sum_squares": float(r @ r),
                                         "components": components, "predictions": predictions})
        except (ValueError, RuntimeError, FloatingPointError, OverflowError) as error:
            result_row["checks"].append({"status": "failed", "method": method,
                                         "grid_step_min": step,
                                         "error": f"{type(error).__name__}: {error}"})
    result_row["status"] = ("converged" if all(c["status"] == "ok" for c in result_row["checks"])
                            else "post-fit evaluation failed")
    return result_row


def main():
    previous_raw, historical_raw, evidence_raw = (p.read_bytes() for p in (PREVIOUS, HISTORICAL, EVIDENCE))
    previous, historical = json.loads(previous_raw), json.loads(historical_raw)
    if previous["status"] != "completed exploratory refits":
        raise ValueError("completed resistance-only refits required")
    matches = [c for c in previous["cases"] if c["hill"] == 2.6 and c["status"] == "converged"]
    if len(matches) != 1 or historical["cases"][4]["status"] != "converged":
        raise ValueError("expected two converged Hill 2.6 starts")
    starts = [("resistance_only_refit", matches[0]), ("historical_fixed_time", historical["cases"][4])]
    mean, sd, n = load_endpoint()
    names, lower, upper, _ = parameter_space(configuration(True, 0))
    upper[names.index("potassium_flux")] = 2.
    upper[names.index("drug_loss_fraction")] = .98
    report = {
        "status": "in progress",
        "inputs_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(raw).hexdigest()
                          for p, raw in ((PREVIOUS, previous_raw), (HISTORICAL, historical_raw),
                                         (EVIDENCE, evidence_raw))},
        "asakuma_ep": {"mean_mV": mean, "sd_mV": sd, "n": n,
                       "reported_sd_over_sqrt_n_mV": float(sd / np.sqrt(n))},
        "bounds": {**dict(zip(names, [[float(lo), float(hi)] for lo, hi in zip(lower, upper)])),
                   "exchange": [0., 10.]},
        "limitations": [
            "Exploratory outcome-informed sensitivity, not independent evaluation or model selection.",
            "The endpoint scales are an illustrative 10 mV and reported SD/sqrt(n); neither includes model discrepancy or EP-resistance covariance.",
            "A modeled trajectory minimum is not generally the mean of individual minima; both EP and resistance are correlated group-level endpoints.",
            "Absolute EP is compared despite differing baseline means and shared cross-study parameter/exposure assumptions.",
            "Parameter bounds, series fraction, 0–120 minute window and 1-minute optimization grid are assumed; finer grids are post-fit only.",
            "Two outcome-informed local starts do not establish global optimality, parameter identifiability or physical closure.",
        ],
        "cases": [],
    }
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    records = load_anchors()
    for scale in (10., sd / np.sqrt(n)):
        print(f"Starting EP scale {scale:.4f} mV", flush=True)
        report["cases"].append(fit(records, names, lower, upper, starts, 2.6, scale))
        OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"Completed EP scale {scale:.4f}: {report['cases'][-1]['status']}", flush=True)
    report["status"] = ("completed exploratory endpoint sensitivity" if
                        all(c["status"] == "converged" for c in report["cases"])
                        else "completed with failures; inspect cases")
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
