"""Plot the frozen six-parameter reference candidate against its Tier 1 drug targets."""

from dataclasses import replace
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from src.joint_calibration_screen import RESISTANCE_RATIO, RYBAK
from src.model_equations import (Inputs, Parameters, broadband_sound_multiplier,
                                         drug_concentration, related_code_closure, simulate)
from src.kinetics_fit import ALL_NAMES, OUTPUT as FIT, residual
from src.pike_trace_diagnostic import IDS, ROOT, SOURCE, load_anchors
from src.asakuma_ep_endpoint_sensitivity import EVIDENCE, load_endpoint


OUTPUT = FIT.parent / "tier1_fit"
OBSERVED = "#111111"
MODEL = "#0072B2"
WANG = ROOT / "refs" / "target_source" / "wang_1990_hear_res.pdf"
# Wang et al. (1990), results text, journal p. 144; separate control/exposure cohorts.
WANG_BASELINE = 83.8
WANG_MEANS = ((20, 76.3), (40, 70.3), (60, 63.0), (80, 60.0))


def noise_comparison(p):
    """Compare reported cohort differences with a 30–60 min post-noise model window."""
    closure = related_code_closure()
    result = []
    for duration, mean_ep in WANG_MEANS:
        times = np.linspace(0., duration + 60., (duration + 60) * 10 + 1)
        forcing = lambda t: Inputs(sound_multiplier=broadband_sound_multiplier(125.)
                                   if t < duration else 1.)
        y = simulate(times, forcing, closure, p, boundaries=(float(duration),),
                     rtol=1e-8, atol=1e-10)
        changes = y[8, (duration + 30) * 10:] - y[8, 0]
        result.append({"exposure_min": duration, "observed_difference_mV": mean_ep - WANG_BASELINE,
                       "model_min_mV": float(np.min(changes)),
                       "model_max_mV": float(np.max(changes))})
    return result


def build_data(fit, records):
    """Recalculate the plotted data using the existing fitted parameters."""
    if fit.get("status") != "completed exploratory six-parameter tier 1 drug calibration":
        raise ValueError("the completed six-parameter fit is required")
    values = np.array([fit["best_parameters"][name] for name in ALL_NAMES])
    _, predictions = residual(values, records, step=.025, method="Radau", rtol=1e-8, atol=1e-10)
    checks = [check for check in fit["checks"]
              if check["method"] == "Radau" and check["step_min"] == .025]
    if len(checks) != 1:
        raise ValueError("expected exactly one 0.025-minute Radau fit check")
    reference = checks[0]["predictions"]
    for key in ("asakuma_60", "rybak_25", "rybak_50", "rybak_100"):
        for field in ("ep_reduction_mV",):
            if not np.isclose(predictions[key][field], reference[key][field], rtol=1e-6, atol=1e-5):
                raise ValueError(f"{key} {field} differs from stored fit check")
    for dose in (25, 50, 100):
        key = f"rybak_{dose}"
        if not np.isclose(predictions[key]["k_reduction_mM"],
                          reference[key]["k_reduction_mM"], rtol=1e-6, atol=1e-5):
            raise ValueError(f"{key} K reduction differs from stored fit check")
    if not np.isclose(predictions["asakuma_60"]["resistance_ratio_at_ep_nadir"],
                      reference["asakuma_60"]["resistance_ratio_at_ep_nadir"], rtol=1e-6, atol=1e-5):
        raise ValueError("Asakuma resistance ratio differs from stored fit check")
    p = replace(Parameters(), drug_conductance_loss=float(values[2]), drug_k_loss=float(values[3]),
                drug_slow_fraction=float(values[4]), drug_slow=float(values[5]))
    curves = {}
    for key in IDS:
        rows = [r for r in records if r["curve"] == key]
        if len(rows) < 2 or rows[0]["time_min"] != 0:
            raise ValueError(f"{key} has invalid observation anchors")
        observed_time = np.array([r["time_min"] for r in rows])
        observed_ep = np.array([r["ep_mV"] for r in rows])
        if np.any(np.diff(observed_time) <= 0):
            raise ValueError(f"{key} has unordered anchors")
        times = np.unique(np.r_[np.linspace(0, observed_time[-1], 1201), observed_time])
        level = rows[0]["dose_mg_kg"] * values[0]
        forcing = lambda t: Inputs(drug=drug_concentration(t, 0., .542, level, p),
                                   furosemide_protocol=True)
        states = simulate(times, forcing, related_code_closure(), p, boundaries=(.542,),
                          rtol=1e-8, atol=1e-10)
        model_change = states[8] - states[8, 0]
        modeled_at_anchors = model_change[np.searchsorted(times, observed_time)]
        observed_change = observed_ep - observed_ep[0]
        rmse = float(np.sqrt(np.mean((modeled_at_anchors - observed_change) ** 2)))
        if not np.isclose(rmse, reference[key]["change_rmse_mV"], rtol=1e-6, atol=1e-5):
            raise ValueError(f"{key} RMSE differs from stored fit check")
        curves[key] = {"dose": rows[0]["dose_mg_kg"], "n": rows[0]["animals"],
                       "time": times.tolist(), "model_change": model_change.tolist(),
                       "observed_time": observed_time.tolist(), "observed_change": observed_change.tolist(),
                       "model_at_observed": modeled_at_anchors.tolist(), "rmse": rmse}
    observed_nadir, _, _ = load_endpoint()
    return {
        "parameters": dict(zip(ALL_NAMES, map(float, values))),
        "pike": curves,
        "rybak": [{"dose": dose, "n": n, "observed_ep": ep, "ep_se": ep_se,
                   "observed_k": k, "k_se": k_se,
                   "model_ep": predictions[f"rybak_{dose}"]["ep_reduction_mV"],
                   "model_k": predictions[f"rybak_{dose}"]["k_reduction_mM"]}
                  for dose, ep, ep_se, k, k_se, n in RYBAK],
        "asakuma": {"observed_ep_drop": 85. - observed_nadir,
                    "model_ep_drop": predictions["asakuma_60"]["ep_reduction_mV"],
                    "observed_resistance_ratio": RESISTANCE_RATIO,
                    "model_resistance_ratio": predictions["asakuma_60"]["resistance_ratio_at_ep_nadir"]},
        "wang_noise": noise_comparison(p),
    }


def create_figure(data):
    """Draw drug-fit panels and a separately labeled no-refit noise comparison."""
    with plt.rc_context({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
                         "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}):
        fig = plt.figure(figsize=(7.5, 8.))
        grid = fig.add_gridspec(3, 4, width_ratios=(1, 1, .47, .47),
                                left=.09, right=.98, top=.95, bottom=.11,
                                hspace=.65, wspace=.7)
        pike_axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
                     fig.add_subplot(grid[0, 2:4])]
        for ax, key, label in zip(pike_axes, IDS, "ABC"):
            curve = data["pike"][key]
            ax.axhline(0, color="0.8", linewidth=.6)
            ax.plot(curve["time"], curve["model_change"], color=MODEL, linewidth=1.6)
            ax.scatter(curve["observed_time"], curve["observed_change"], s=18, marker="o",
                       facecolor="white", edgecolor=OBSERVED, linewidth=.85, zorder=3)
            ax.set(xlim=(-2, 122), ylim=(-140, 10), xticks=[0, 30, 60, 90, 120],
                   xlabel="Time after injection (min)",
                   title=f"{label}  Pike: {curve['dose']} mg/kg (n={curve['n']})")
            ax.set_ylabel("EP change (mV)")
        ep_ax = fig.add_subplot(grid[1, 0])
        k_ax = fig.add_subplot(grid[1, 1])
        for ax, observed, error, modeled, title, ylabel in (
                (ep_ax, "observed_ep", "ep_se", "model_ep", "D  Rybak: EP", "EP reduction (mV)"),
                (k_ax, "observed_k", "k_se", "model_k", "E  Rybak: endolymph K", "K reduction (mM)")):
            entries = data["rybak"]
            doses = np.array([row["dose"] for row in entries])
            ax.errorbar(doses - 2, [row[observed] for row in entries],
                        yerr=[row[error] for row in entries], fmt="o", markersize=4,
                        color=OBSERVED, markerfacecolor="white", capsize=2, linewidth=.8)
            ax.scatter(doses + 2, [row[modeled] for row in entries], marker="s",
                       s=24, color=MODEL, zorder=3)
            ax.set(xticks=doses, xlim=(12, 113), xlabel="Furosemide (mg/kg)",
                   ylabel=ylabel, title=title, ylim=(0, None))
        asakuma_ep = fig.add_subplot(grid[1, 2])
        asakuma_resistance = fig.add_subplot(grid[1, 3])
        a = data["asakuma"]
        for ax, observed, modeled, label in (
                (asakuma_ep, a["observed_ep_drop"], a["model_ep_drop"], "EP drop\n(mV)"),
                (asakuma_resistance, a["observed_resistance_ratio"],
                 a["model_resistance_ratio"], "Resistance\nratio")):
            ax.scatter([0], [observed], marker="o", s=22, facecolor="white",
                       edgecolor=OBSERVED, linewidth=.85, zorder=3)
            ax.scatter([1], [modeled], marker="s", s=24, color=MODEL, zorder=3)
            ax.set(xlim=(-.5, 1.5), ylim=(0, max(observed, modeled) * 1.32),
                   xticks=[0, 1], xticklabels=["Obs.", "Model"], ylabel=label)
        asakuma_ep.set_title("F  Asakuma: 60 mg/kg", loc="left")
        noise_ax = fig.add_subplot(grid[2, :])
        noise = data["wang_noise"]
        duration = np.array([row["exposure_min"] for row in noise])
        lower = np.array([row["model_min_mV"] for row in noise])
        upper = np.array([row["model_max_mV"] for row in noise])
        midpoint = (lower + upper) / 2
        noise_ax.scatter(duration - 1, [row["observed_difference_mV"] for row in noise],
                         marker="o", s=22, facecolor="white", edgecolor=OBSERVED, zorder=3)
        noise_ax.errorbar(duration + 1, midpoint, yerr=(upper - lower) / 2, fmt="s",
                          color=MODEL, markersize=4, capsize=3, linewidth=1)
        noise_ax.axhline(0, color="0.8", linewidth=.6)
        noise_ax.set(xlim=(5, 95), xticks=[20, 40, 60, 80],
                     xlabel="125 dB SPL white-noise exposure duration (min)",
                     ylabel="EP change (mV)",
                     title="G  Wang: post-noise EP comparison (not fitted)")
        noise_ax.text(.02, .06, "Model bars: 30–60 min after noise; not uncertainty",
                      transform=noise_ax.transAxes, fontsize=7, va="bottom")
        for ax in [*pike_axes, ep_ax, k_ax, asakuma_ep, asakuma_resistance, noise_ax]:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(direction="out", length=3)
        fig.legend(handles=[
            Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor=OBSERVED,
                   linestyle="none", markersize=5, label="Observed"),
            Line2D([], [], color=MODEL, marker="s", markersize=4, linewidth=1.3, label="Model"),
        ], loc="lower center", bbox_to_anchor=(.5, .015), ncol=2, frameon=False, fontsize=8)
    return fig


def main():
    raw = FIT.read_bytes()
    data = build_data(json.loads(raw), load_anchors())
    fig = create_figure(data)
    with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}):
        for extension in ("pdf", "svg", "png"):
            fig.savefig(OUTPUT.with_suffix(f".{extension}"), dpi=300)
    plt.close(fig)
    report = {
        "status": "exploratory in-sample fit visualization; not independent validation",
        "sources_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in (FIT, SOURCE, EVIDENCE, WANG)},
        "model": "reference 13-state hybrid, six-parameter flexible drug-proxy fit; no refit",
        "notes": [
            "Pike markers are approximate, correlated manually digitized group-mean anchors; lines are model EP changes relative to the modeled initial state.",
            "Rybak error bars are reported group SEM; model points are 0.025-min sampled trajectory minima, not means of individual minima.",
            "Asakuma EP is the difference of reported group means; resistance is a ratio of rounded group means at modeled EP nadir, not an individual paired ratio.",
            "Pike and Asakuma guinea pig, Rybak chinchilla; a fitted exposure multiplier and unmeasured proxy pharmacokinetics are assumed.",
            "Other Tier 1 evidence is not fitted in this figure; state-range excursions and remaining mismatches preclude physiological validation.",
            "Wang comparison uses text-reported EP means minus the separate control mean, not paired within-animal changes. The figure's sample sizes conflict with the results text.",
            "Wang model bars show the range of sampled model EP changes 30–60 min after exposure, not error bars; the source reports approximate measurement delays and does not give each cohort's exact delay.",
            "The 125 dB SPL broadband sound mapping is an unverified reference model assumption. Noise observations were not fitted; no additional parameter is estimated.",
        ],
        "data": data,
    }
    OUTPUT.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {OUTPUT}.[pdf, svg, png, json]")


if __name__ == "__main__":
    main()
