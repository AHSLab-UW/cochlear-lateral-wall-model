"""Plot the conditional Wang 1990 and 1992 noise calibration together."""

import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.model_equations import Inputs, broadband_sound_multiplier, related_code_closure, simulate
from src.kinetics_fit import ALL_NAMES
from src.wang_noise_cohorts import OUTPUT as FIT
from src.wang_noise_traces import NAMES, make_parameters
from src.pike_trace_diagnostic import ROOT


OUTPUT = FIT.parent / "two_study_noise_fit"


def build_data(report):
    if report.get("status") != "conditional exploratory two-study noise optimization":
        raise ValueError("two-study conditional fit required")
    drug = [report["drug_parameters_frozen"][key] for key in ALL_NAMES]
    sound = [report["sound_parameters"][key] for key in NAMES]
    p = make_parameters(sound, drug)
    selected = [x for x in report["checks"] if x["candidate"] == "two_study_candidate"
                and x["method"] == "Radau"]
    checks = {x["delay_min"]: x for x in selected}
    if len(selected) != 3 or set(checks) != {30., 45., 60.}:
        raise ValueError("expected exactly one Radau check per observation delay")
    curves = []
    for prediction in checks[45.]["wang1992"]:
        level = prediction["level_db_spl"]
        observed_time = np.array(prediction["time_min"])
        times = np.unique(np.r_[np.arange(0., 30.001, .025), observed_time])
        forcing = lambda t: Inputs(sound_multiplier=broadband_sound_multiplier(level) if t < 20. else 1.)
        y = simulate(times, forcing, related_code_closure(), p, boundaries=(20.,),
                     rtol=1e-8, atol=1e-10)
        change = y[8] - y[8, 0]
        if not np.allclose(change[np.searchsorted(times, observed_time)],
                           prediction["model_change_mV"], rtol=1e-6, atol=1e-5):
            raise ValueError(f"{level} dB curve differs from stored fit")
        curves.append({**prediction, "curve_time_min": times.tolist(), "curve_change_mV": change.tolist()})
    cohorts = []
    for i, row in enumerate(checks[45.]["wang1990"]):
        delay_rows = [checks[delay]["wang1990"][i] for delay in (30., 45., 60.)]
        if any(other["exposure_min"] != row["exposure_min"] or
               other["observed_difference_mV"] != row["observed_difference_mV"]
               for other in delay_rows):
            raise ValueError("inconsistent Wang 1990 cohort comparisons")
        audit = report["wang1990_sampled_ranges_0_1_min"][i]
        if audit["exposure_min"] != row["exposure_min"]:
            raise ValueError("inconsistent Wang 1990 range audit")
        window = audit["model_range_30_60_min_mV"]
        if any(not window[0] - 1e-5 <= item["modeled_difference_mV"] <= window[1] + 1e-5
               for item in delay_rows):
            raise ValueError("timing check outside sampled model window")
        cohorts.append({**row, "model_range_30_60_min_mV": window})
    return {"wang1992_curves": curves, "wang1990_cohorts": cohorts}


def create_figure(data):
    with plt.rc_context({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
                         "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "pdf.fonttype": 42, "svg.fonttype": "none"}):
        fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.3))
        fig.subplots_adjust(left=.085, right=.985, bottom=.14, top=.94, hspace=.45, wspace=.28)
        for ax, curve, label in zip(axes.flat, data["wang1992_curves"], "ABC"):
            ax.axvspan(0, 20, color=".92", zorder=0)
            ax.axhline(0, color=".7", linewidth=.6)
            ax.plot(curve["curve_time_min"], curve["curve_change_mV"], color="#0072B2", lw=1.5)
            ax.scatter(curve["time_min"], curve["observed_change_mV"], s=18, marker="o",
                       facecolor="white", edgecolor="black", linewidth=.8, zorder=3)
            ax.set(xlim=(-.5, 30.6), xticks=[0, 10, 20, 30], ylim=(-25, 16),
                   yticks=[-20, -10, 0, 10], xlabel="Time after onset (min)",
                   ylabel="EP change (mV)", title=f"{label}  Wang 1992: {curve['level_db_spl']} dB SPL")
        ax = axes[1, 1]
        rows = data["wang1990_cohorts"]
        duration = np.array([row["exposure_min"] for row in rows])
        midpoint = np.array([row["modeled_difference_mV"] for row in rows])
        low = np.array([row["model_range_30_60_min_mV"][0] for row in rows])
        high = np.array([row["model_range_30_60_min_mV"][1] for row in rows])
        ax.axhline(0, color=".7", linewidth=.6)
        ax.vlines(duration + 1, low, high, color="#0072B2", linewidth=1.2)
        ax.plot(duration + 1, midpoint, "s-", color="#0072B2", markersize=4, linewidth=1.)
        ax.plot(duration - 1, [row["observed_difference_mV"] for row in rows],
                "o", markerfacecolor="white", markeredgecolor="black", markersize=4)
        ax.set(xlim=(10, 90), xticks=[20, 40, 60, 80], ylim=(-30, 5),
               xlabel="125 dB exposure duration (min)", ylabel="EP difference (mV)",
               title="D  Wang 1990: separate cohorts")
        for ax in axes.flat:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        fig.text(.5, .035, "Open circles: data  |  Blue: model  |  D bars: 30–60 min timing scenarios, not uncertainty",
                 ha="center", fontsize=7)
    return fig


def main():
    raw = FIT.read_bytes()
    report = json.loads(raw)
    data = build_data(report)
    fig = create_figure(data)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(OUTPUT.with_suffix(f".{extension}"), dpi=300)
    plt.close(fig)
    out = {"status": "in-sample conditional two-study noise fit; not validation",
           "fit_source": str(FIT.relative_to(ROOT)), "fit_sha256": hashlib.sha256(raw).hexdigest(),
           "limitations": report["limitations"], **data}
    OUTPUT.with_suffix(".json").write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {OUTPUT}.[pdf, svg, png, json]")


if __name__ == "__main__":
    main()
