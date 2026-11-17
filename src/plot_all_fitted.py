"""Plot all currently fitted reference-candidate targets with one parameter set."""

import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from src.asakuma_ep_endpoint_sensitivity import EVIDENCE, load_endpoint
from src.joint_calibration_screen import RESISTANCE_RATIO, RYBAK
from src.model_equations import Inputs, drug_concentration, related_code_closure, simulate
from src.drug_calibration import evaluate
from src.kinetics_fit import ALL_NAMES
from src.wang_noise_cohorts import OUTPUT as FIT, WANG, WANG1992
from src.wang_noise_traces import NAMES, make_parameters
from src.pike_trace_diagnostic import IDS, ROOT, SOURCE, load_anchors
from src.plot_two_study_noise_fit import build_data as build_noise_data


OUTPUT = FIT.parent / "all_fitted_tier1"
MODEL = "#0072B2"
OBSERVED = "#111111"
DRUG_PAPERS = tuple(ROOT / "refs" / "target_source" / name for name in (
    "pike_1980_hear_res.pdf", "asakuma_1980_otolaryngol_head_neck_surg.pdf",
    "rybak_1982_hear_res.pdf"))


def build_data(report, records):
    """Recompute all plotted outputs using the sound-updated candidate."""
    if report.get("status") != "conditional exploratory two-study noise optimization":
        raise ValueError("conditional two-study fit required")
    drug = [report["drug_parameters_frozen"][key] for key in ALL_NAMES]
    sound = [report["sound_parameters"][key] for key in NAMES]
    p = make_parameters(sound, drug)
    _, predictions = evaluate(drug[:4], records, step=.025, method="Radau",
                              rtol=1e-8, atol=1e-10, base_parameters=p)
    reference = report["drug_recheck"]["predictions"]
    for key in ("asakuma_60", "rybak_25", "rybak_50", "rybak_100"):
        fields = ("ep_reduction_mV", "resistance_ratio_at_ep_nadir") if key == "asakuma_60" else (
            "ep_reduction_mV", "k_reduction_mM")
        for field in fields:
            if not np.isclose(predictions[key][field], reference[key][field], rtol=1e-6, atol=1e-5):
                raise ValueError(f"{key} {field} differs from stored drug recheck")
    curves = []
    for key in IDS:
        rows = [r for r in records if r["curve"] == key]
        if len(rows) < 2 or rows[0]["time_min"] != 0:
            raise ValueError(f"invalid Pike observations for {key}")
        observed_time = np.array([r["time_min"] for r in rows], dtype=float)
        if np.any(np.diff(observed_time) <= 0):
            raise ValueError(f"unordered Pike observations for {key}")
        observed_ep = np.array([r["ep_mV"] for r in rows], dtype=float)
        observed_change = observed_ep - observed_ep[0]
        times = np.unique(np.r_[np.linspace(0., observed_time[-1], 1201), observed_time])
        forcing = lambda t: Inputs(
            drug=drug_concentration(t, 0., .542, rows[0]["dose_mg_kg"] * drug[0], p),
            furosemide_protocol=True)
        states = simulate(times, forcing, related_code_closure(), p, boundaries=(.542,),
                          rtol=1e-8, atol=1e-10)
        changes = states[8] - states[8, 0]
        at_points = changes[np.searchsorted(times, observed_time)]
        rmse = float(np.sqrt(np.mean((at_points - observed_change) ** 2)))
        if not np.isclose(rmse, reference[key]["change_rmse_mV"], rtol=1e-6, atol=1e-5):
            raise ValueError(f"{key} differs from stored drug recheck")
        curves.append({"id": key, "dose_mg_kg": rows[0]["dose_mg_kg"], "n": rows[0]["animals"],
                       "time_min": times.tolist(), "model_change_mV": changes.tolist(),
                       "observed_time_min": observed_time.tolist(),
                       "observed_change_mV": observed_change.tolist(),
                       "model_at_observed_mV": at_points.tolist(), "rmse_mV": rmse})
    nadir, _, _ = load_endpoint()
    return {
        "drug_parameters": dict(zip(ALL_NAMES, map(float, drug))),
        "sound_parameters": dict(zip(NAMES, map(float, sound))),
        "pike": curves,
        "rybak": [{"dose": dose, "n": n, "observed_ep": ep, "ep_se": ep_se,
                   "observed_k": k, "k_se": k_se,
                   "model_ep": predictions[f"rybak_{dose}"]["ep_reduction_mV"],
                   "model_k": predictions[f"rybak_{dose}"]["k_reduction_mM"]}
                  for dose, ep, ep_se, k, k_se, n in RYBAK],
        "asakuma": {"observed_ep_drop": 85. - nadir,
                    "model_ep_drop": predictions["asakuma_60"]["ep_reduction_mV"],
                    "observed_resistance_ratio": RESISTANCE_RATIO,
                    "model_resistance_ratio": predictions["asakuma_60"]["resistance_ratio_at_ep_nadir"]},
        **build_noise_data(report),
    }


def create_figure(data):
    with plt.rc_context({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}):
        fig, axes = plt.subplots(4, 3, figsize=(7.5, 10.1))
        fig.subplots_adjust(left=.085, right=.985, bottom=.08, top=.965, hspace=.58, wspace=.48)
        for ax, curve, label in zip(axes[0], data["pike"], "ABC"):
            ax.axhline(0, color=".8", lw=.6)
            ax.plot(curve["time_min"], curve["model_change_mV"], color=MODEL, lw=1.5)
            ax.scatter(curve["observed_time_min"], curve["observed_change_mV"], s=17,
                       facecolor="white", edgecolor=OBSERVED, linewidth=.8, zorder=3)
            ax.set(xlim=(-2, 122), ylim=(-140, 10), xticks=[0, 40, 80, 120],
                   xlabel="Time after injection (min)", ylabel="EP change (mV)",
                   title=f"{label}  Pike: {curve['dose_mg_kg']} mg/kg (n={curve['n']})")
        for ax, curve, label in zip(axes[1], data["wang1992_curves"], "DEF"):
            ax.axvspan(0, 20, color=".92", zorder=0)
            ax.axhline(0, color=".8", lw=.6)
            ax.plot(curve["curve_time_min"], curve["curve_change_mV"], color=MODEL, lw=1.5)
            ax.scatter(curve["time_min"], curve["observed_change_mV"], s=17,
                       facecolor="white", edgecolor=OBSERVED, linewidth=.8, zorder=3)
            ax.set(xlim=(-.5, 30.6), ylim=(-25, 16), xticks=[0, 10, 20, 30],
                   xlabel="Time after noise onset (min)", ylabel="EP change (mV)",
                   title=f"{label}  Wang 1992: {curve['level_db_spl']} dB SPL")
        rows = data["rybak"]
        doses = np.array([row["dose"] for row in rows])
        for ax, obs, error, mod, title, ylabel in (
                (axes[2, 0], "observed_ep", "ep_se", "model_ep", "G  Rybak: EP", "EP reduction (mV)"),
                (axes[2, 1], "observed_k", "k_se", "model_k", "H  Rybak: endolymph K", "K reduction (mM)")):
            ax.errorbar(doses - 2, [row[obs] for row in rows], yerr=[row[error] for row in rows],
                        fmt="o", markersize=4, color=OBSERVED, markerfacecolor="white",
                        capsize=2, linewidth=.8)
            ax.scatter(doses + 2, [row[mod] for row in rows], marker="s", s=23, color=MODEL, zorder=3)
            ax.set(xlim=(12, 113), ylim=(0, None), xticks=doses, xlabel="Furosemide (mg/kg)",
                   ylabel=ylabel, title=title)
        a = data["asakuma"]
        for ax, observed, modeled, title, ylabel in (
                (axes[2, 2], a["observed_ep_drop"], a["model_ep_drop"],
                 "I  Asakuma: EP", "EP reduction (mV)"),
                (axes[3, 0], a["observed_resistance_ratio"], a["model_resistance_ratio"],
                 "J  Asakuma: resistance", "Resistance ratio")):
            ax.scatter([0], [observed], marker="o", s=23, facecolor="white", edgecolor=OBSERVED, zorder=3)
            ax.scatter([1], [modeled], marker="s", s=23, color=MODEL, zorder=3)
            ax.set(xlim=(-.5, 1.5), ylim=(0, max(observed, modeled) * 1.25),
                   xticks=[0, 1], xticklabels=["Data", "Model"], ylabel=ylabel, title=title)
        ax = axes[3, 1]
        rows = data["wang1990_cohorts"]
        duration = np.array([row["exposure_min"] for row in rows])
        lower = [row["model_range_30_60_min_mV"][0] for row in rows]
        upper = [row["model_range_30_60_min_mV"][1] for row in rows]
        ax.axhline(0, color=".8", lw=.6)
        ax.vlines(duration + 1, lower, upper, color=MODEL, lw=1.2)
        ax.plot(duration + 1, [row["modeled_difference_mV"] for row in rows],
                "s-", color=MODEL, markersize=4, lw=1.)
        ax.plot(duration - 1, [row["observed_difference_mV"] for row in rows],
                "o", markerfacecolor="white", markeredgecolor=OBSERVED, markersize=4)
        ax.set(xlim=(10, 90), ylim=(-30, 5), xticks=[20, 40, 60, 80],
               xlabel="125 dB exposure (min)", ylabel="EP difference (mV)",
               title="K  Wang 1990: separate cohorts")
        legend_ax = axes[3, 2]
        legend_ax.axis("off")
        legend_ax.legend(handles=[
            Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor=OBSERVED,
                   linestyle="none", markersize=5, label="Observed"),
            Line2D([], [], color=MODEL, marker="s", markersize=4, lw=1.3, label="Model"),
        ], loc="upper left", frameon=False, fontsize=8)
        legend_ax.text(0., .55, "D–F shading: noise exposure\nG–H bars: reported SEM\nK bars: 30–60 min timing range,\nnot uncertainty",
                       transform=legend_ax.transAxes, fontsize=7.5, va="top", linespacing=1.4)
        for ax in axes.flat[:-1]:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(direction="out", length=3)
        fig.text(.5, .025, "Exploratory in-sample fits; Wang 1990 assumes a 45-min observation time.",
                 ha="center", fontsize=7)
    return fig


def main():
    raw = FIT.read_bytes()
    fit = json.loads(raw)
    data = build_data(fit, load_anchors())
    fig = create_figure(data)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(OUTPUT.with_suffix(f".{extension}"), dpi=300)
    plt.close(fig)
    report = {
        "status": "exploratory in-sample visualization of currently fitted targets; not full Tier 1 or validation",
        "sources_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in (FIT, SOURCE, EVIDENCE, WANG, WANG1992, *DRUG_PAPERS)},
        "notes": [
            "One conditional reference hybrid candidate is used throughout: six drug parameters were fitted first, then three existing sound parameters were fitted to Wang 1992 and conditionally Wang 1990; not a joint optimum.",
            "Pike and Wang 1992 points are approximate correlated digitized group means, plotted as changes relative to each observed or modeled baseline.",
            "Rybak bars are reported SEM; modeled EP and K reductions use individual sampled trajectory minima, not means of individual minima.",
            "Asakuma EP is the difference of reported means and resistance is a ratio of rounded group means; model resistance is evaluated at model EP nadir.",
            "Wang 1990 points are differences from a separate control cohort; the source text and figure disagree on group counts and means.",
            "Wang 1990 model squares use a fixed 45-minute assumed delay; blue ranges show sampled 30–60-minute timing sensitivity, not uncertainty.",
            "The sound drive, drug exposure proxy, species multiplier, K boundary and hybrid closure are unverified assumptions; some model states exceed draft-stated ranges.",
            "Other Tier 1 evidence is not fitted and is not shown; this is neither full Tier 1 calibration nor independent validation.",
        ],
        "data": data,
    }
    OUTPUT.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {OUTPUT}.[pdf, svg, png, json]")


if __name__ == "__main__":
    main()
