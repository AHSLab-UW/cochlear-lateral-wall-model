"""Explore low-dose K mismatch under exposure, flux and water assumptions."""

import json

import numpy as np

from src.joint_calibration_screen import configuration, parameter_space
from src.joint_mechanism_comparison import make_simulator
from src.joint_timing_sensitivity import temporal_residuals
from src.pike_trace_diagnostic import load_anchors
from src.reference_geometry_comparison import RESULTS, fit_case
from src.transport_assumption_calibration import inhibition


OUTPUT = RESULTS / "low_dose_assumption_sensitivity.json"
GEOMETRY = {"strial_reference_volume": .005, "syncytial_volume": .005,
            "marginal_volume": .02}
SCENARIOS = (
    (2.6, .5, .005), (1., .5, .005),
    (2.6, 2., .005), (1., 2., .005),
    (2.6, 2., .00025), (1., 2., .00025),
)


def summarize(case, records, names):
    """Re-evaluate and partition the actual fine-grid objective and input proxy."""
    if case["status"] != "converged":
        raise ValueError("case must have a successful post-fit evaluation")
    values = np.array([case["parameters"][name] for name in names])
    residual, _ = temporal_residuals(
        records, values, names, series_fraction=.5, resistance_time=6,
        grid_step=.25, rtol=1e-8, atol=1e-10,
        simulator=make_simulator(exchange=case["exchange"], volume_gate=False),
        hill=case["hill"], water_rate=case["water_rate"], **GEOMETRY)
    n = len(records)
    if len(residual) != n + 9:
        raise ValueError("unexpected joint objective residual shape")
    components = {
        "pike_ep_curves": float(residual[:n] @ residual[:n]),
        "asakuma_resistance": float(residual[n] ** 2),
        "rybak_ep": float(np.sum(residual[n + 1:n + 7:2] ** 2)),
        "rybak_k": float(np.sum(residual[n + 2:n + 7:2] ** 2)),
        "ep_timing_penalties": float(residual[-2:] @ residual[-2:]),
    }
    if not np.isclose(sum(components.values()), case["weighted_sum_squares"],
                      rtol=1e-8, atol=1e-8):
        raise ValueError("objective decomposition does not match stored fine-grid score")
    p = case["parameters"]
    times = np.linspace(0, 120, 4801)
    exposure = {}
    for dose in (25, 50, 100):
        drug = np.array([inhibition(t, dose * p["rybak_exposure_multiplier"],
                                    p["effective_dose_scale_mg_kg"],
                                    p["exposure_decay_min"], case["hill"]) for t in times])
        exposure[str(dose)] = {
            "maximum_inhibition": float(np.max(drug)),
            "time_of_maximum_min": float(times[np.argmax(drug)]),
            "area_inhibition_min": float(np.trapz(drug, times)),
        }
    return {"objective_components": components, "chinchilla_exposure_proxy": exposure}


def main():
    source = json.loads((RESULTS / "nested_geometry_comparison.json").read_text())
    if not source["status"].startswith("completed"):
        raise ValueError("completed nested geometry comparison is required")
    matching = [c for c in source["cases"]
                if all(c[key] == value for key, value in GEOMETRY.items())]
    exchange_case = next(c for c in matching if c["include_exchange"])
    no_exchange_case = next(c for c in matching if not c["include_exchange"])
    if any(c["status"] != "converged" for c in (exchange_case, no_exchange_case)):
        raise ValueError("both prior starts must be converged")
    names, lower, original_upper, _ = parameter_space(configuration(True, 0))
    records = load_anchors()
    starts = [(label, np.array([case["parameters"][name] for name in names]), case["exchange"])
              for label, case in (("nested_exchange", exchange_case),
                                  ("nested_no_exchange", no_exchange_case))]
    report = {
        "status": "in progress; outcome-informed sensitivity",
        "source": "results/nested_geometry_comparison.json",
        "geometry": GEOMETRY,
        "scenarios": [{"hill": h, "potassium_flux_upper": f, "water_rate": w}
                      for h, f, w in SCENARIOS],
        "limitations": [
            "Hill 1 and 2.6 are fixed illustrative dose-to-inhibition shapes, not measured pharmacokinetics or NKCC dose-response constants.",
            "Flux upper bounds 0.5 and 2 are numerical exploration bounds, not physiological limits; both fits retain the same lower bound and baseline normalized amount-flux law.",
            "Water coefficient 0.00025 scales 0.005 by the ratio of assumed S reference volumes 0.005/0.1; 0.005 holds the normalized absolute coefficient fixed. Neither scaling is measured or a demonstrated hydraulic law.",
            "The same fixed E volume and water source assumption applies to all cases; morphology and bumetanide exposures are not fitted.",
            "The 6 scenarios are not a full factorial; two starts are outcome-informed and convergence does not prove a global optimum.",
            "Objective is not a likelihood; data correlations, model error and measurement covariance are omitted. Cost differences are not evidence of model selection.",
            "Peak inhibition and its area describe only the prescribed model proxy, not measured tissue exposure.",
            "These comparisons do not close ion charge balance, establish physiological admissibility, or test independent predictions.",
        ],
        "cases": [],
    }
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for hill, flux_upper, water in SCENARIOS:
        upper = original_upper.copy()
        upper[names.index("potassium_flux")] = flux_upper
        upper[names.index("drug_loss_fraction")] = .98
        print(f"Starting hill={hill:g}, flux upper={flux_upper:g}, water={water:g}", flush=True)
        case = fit_case(records, names, lower, upper, starts,
                        GEOMETRY["strial_reference_volume"], True,
                        syncytial_volume=GEOMETRY["syncytial_volume"],
                        marginal_volume=GEOMETRY["marginal_volume"],
                        hill=hill, water_rate=water)
        case["potassium_flux_upper"] = flux_upper
        if case["status"] == "converged":
            try:
                case["diagnostic"] = summarize(case, records, names)
            except (ValueError, RuntimeError, FloatingPointError, OverflowError) as error:
                case["diagnostic"] = {"status": "failed",
                                      "error": f"{type(error).__name__}: {error}"}
        report["cases"].append(case)
        OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"Completed {len(report['cases'])}/{len(SCENARIOS)}: {case['status']}", flush=True)
    report["status"] = "completed outcome-informed sensitivity; not independent validation or model selection"
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
