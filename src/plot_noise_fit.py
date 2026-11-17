"""Plot Wang 1992 noise time-course fits for the sequential reference candidate."""

import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.model_equations import (Inputs, broadband_sound_multiplier,
                                         related_code_closure, simulate)
from src.kinetics_fit import ALL_NAMES
from src.wang_noise_traces import NAMES, OUTPUT as FIT, make_parameters
from src.pike_trace_diagnostic import ROOT


OUTPUT = FIT.parent / "wang_noise_traces"


def build_data(report):
    if report.get("status") != "completed exploratory sequential noise calibration":
        raise ValueError("completed noise fit required")
    drug = [report["drug_parameters_frozen"][key] for key in ALL_NAMES]
    sound = [report["sound_parameters"][key] for key in NAMES]
    p = make_parameters(sound, drug)
    checks = [c for c in report["noise_checks"] if c["method"] == "Radau"]
    if len(checks) != 1:
        raise ValueError("expected exactly one Radau check")
    curves = []
    for prediction in checks[0]["predictions"]:
        level = prediction["level_db_spl"]
        observed_time = np.array(prediction["time_min"])
        times = np.unique(np.r_[np.arange(0., 30.101, .025), observed_time])
        forcing = lambda t: Inputs(sound_multiplier=broadband_sound_multiplier(level) if t < 20. else 1.)
        y = simulate(times, forcing, related_code_closure(), p, boundaries=(20.,),
                     rtol=1e-8, atol=1e-10)
        change = y[8] - y[8, 0]
        at_points = change[np.searchsorted(times, observed_time)]
        if not np.allclose(at_points, prediction["model_change_mV"], rtol=1e-6, atol=1e-5):
            raise ValueError(f"{level} dB curve differs from stored noise fit")
        curves.append({**prediction, "curve_time_min": times.tolist(), "curve_change_mV": change.tolist()})
    return curves


def create_figure(curves):
    with plt.rc_context({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
                         "xtick.labelsize": 7, "ytick.labelsize": 7}):
        fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8), sharey=True)
        fig.subplots_adjust(left=.085, right=.985, bottom=.26, top=.86, wspace=.25)
        for ax, curve, label in zip(axes, curves, "ABC"):
            ax.axvspan(0, 20, color="0.92", zorder=0)
            ax.axhline(0, color="0.7", lw=.6)
            ax.plot(curve["curve_time_min"], curve["curve_change_mV"], color="#0072B2", lw=1.5,
                    label="Model")
            ax.scatter(curve["time_min"], curve["observed_change_mV"], s=18, marker="o",
                       facecolor="white", edgecolor="black", linewidth=.8, zorder=3,
                       label="Digitized group means")
            ax.set(xlim=(-.5, 30.6), xticks=[0, 10, 20, 30], ylim=(-25, 16),
                   yticks=[-20, -10, 0, 10], xlabel="Time after onset (min)",
                   title=f"{label}  {curve['level_db_spl']} dB SPL")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        axes[0].set_ylabel("EP change (mV)")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center",
                   bbox_to_anchor=(.5, .01), ncol=2, frameon=False, fontsize=8)
    return fig


def main():
    raw = FIT.read_bytes()
    report = json.loads(raw)
    curves = build_data(report)
    fig = create_figure(curves)
    with plt.rc_context({"pdf.fonttype": 42, "svg.fonttype": "none"}):
        for ext in ("pdf", "svg", "png"):
            fig.savefig(OUTPUT.with_name(OUTPUT.name + "_traces").with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)
    result = {"status": "in-sample exploratory sequential noise fit; not validation",
              "fit_source": str(FIT.relative_to(ROOT)), "fit_sha256": hashlib.sha256(raw).hexdigest(),
              "noise_window_min": [0, 20], "curves": curves,
              "limitations": report["limitations"]}
    out = OUTPUT.with_name(OUTPUT.name + "_traces").with_suffix(".json")
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {out.stem}.[pdf, svg, png, json]")


if __name__ == "__main__":
    main()
