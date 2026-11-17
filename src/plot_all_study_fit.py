"""Plot all joint-objective outcomes for the Salt-trace-inclusive candidate."""

import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.all_study_fit import PARAMETERS, SaltTraceContext, SolverSettings
from src.model_equations import endolymph, related_code_closure
from src.salt_trace_fit_residuals import FIG3_INTERVALS, _run
from src.drug_calibration import evaluate as evaluate_drug
from src.kinetics_fit import ALL_NAMES
from src.wang_noise_traces import NAMES, make_parameters
from src.pike_trace_diagnostic import ROOT, load_anchors
from src.plot_calibration_data import COLORS, LINESTYLES, MARKERS, build_common_data, create_figure
from src.run_all_study_fit import OUTPUT as FIT
from src.salt_1979_digitize import DEFAULT_EVIDENCE, calibrated_points, load_evidence, verify_source_hashes


OUTPUT = FIT.parent / "all_six_studies_joint_fit_figure"


def build_data(report):
    if not report.get("trace_integrated") or not report.get("optimizer", {}).get("success"):
        raise ValueError("converged joint fit with both Salt trace components required")
    values = report["parameters"]
    drug = [values[name] for name in ALL_NAMES]
    sound = [values[name] for name in NAMES]
    p = make_parameters(sound, drug)
    if any(getattr(p, name) != value for name, value in report["all_model_parameter_values"].items()):
        raise ValueError("candidate parameters inconsistent")
    _, drug_predictions = evaluate_drug(
        drug[:4], load_anchors(), step=.025, method="Radau", rtol=1e-8, atol=1e-10,
        base_parameters=p,
    )
    adapter = {"drug_parameters_frozen": dict(zip(ALL_NAMES, drug)),
               "sound_parameters": dict(zip(NAMES, sound)),
               "drug_recheck": {"predictions": drug_predictions}}
    data = build_common_data(adapter, load_anchors())
    check = report["numerical_checks"][0]
    table = check["components"]["salt_table_iii"]["predictions"]
    data["salt_conditions"] = [
        {"level_dba": level, "exposures": [
            {"exposure_number": row["exposure_number"], "observed_change_mV": row["mean_mV"],
             "observed_sd_mV": row["sd_mV"], "n_observations": row["n_observations"],
             "modeled_change_mV": row["predicted_overall_ep_change_mV"]}
            for row in table if row["noise_dba"] == level]}
        for level in (95, 105, 115)
    ]
    data["salt_figure_3"] = check["components"]["salt_figure_3"]["predictions"]
    data["salt_figure_5"] = check["components"]["salt_figure_5"]["predictions"]
    expected_hash = report["provenance"]["source_sha256"][str(DEFAULT_EVIDENCE.relative_to(ROOT))]
    if hashlib.sha256(DEFAULT_EVIDENCE.read_bytes()).hexdigest() != expected_hash:
        raise ValueError("Salt trace digitization differs from the joint fit input")
    evidence = load_evidence()
    mismatches = verify_source_hashes(evidence)
    if mismatches:
        raise ValueError(f"Salt source or rendered-image hashes changed: {mismatches}")
    data["salt_figure_5_unscaled"] = {}
    for level in (95, 105, 115):
        points = calibrated_points(evidence, "fig5", f"{level}_dba")
        unscaled = [point for point in points
                    if point["phase"] in ("pre_exposure", "post_offset_unscaled_time")]
        if (any(point["time_min"] is not None for point in unscaled)
                or not any(point["phase"] == "pre_exposure" for point in unscaled)
                or not any(point["phase"] == "post_offset_unscaled_time" for point in unscaled)):
            raise ValueError(f"invalid unscaled Figure 5 observations at {level} dBA")
        data["salt_figure_5_unscaled"][str(level)] = unscaled
    context = SaltTraceContext(
        tuple(values[name] for name in PARAMETERS), tuple(drug), tuple(sound), p,
        related_code_closure(), tuple(report["initial_state"]),
        SolverSettings(rtol=1e-8, atol=1e-10),
    )
    times3 = np.linspace(0., 90., 901)
    states, times3 = _run(context, times3, 115., FIG3_INTERVALS)
    potassium = np.array([endolymph(state, p)[1] for state in states.T])
    data["salt_figure_3_model"] = {
        "time_min": times3.tolist(), "ep": (states[8] - states[8, 0]).tolist(),
        "endolymph_k": (potassium - potassium[0]).tolist(),
    }
    data["salt_figure_5_model"] = {}
    for level in (95, 105, 115):
        states, times = _run(context, np.linspace(0., 20., 201), level, ((0., 20.),))
        data["salt_figure_5_model"][str(level)] = {
            "time_min": times.tolist(), "change": (states[8] - states[8, 0]).tolist()
        }
    return data


def create_joint_figure(data):
    fig = create_figure(data)
    fig.set_size_inches(7.5, 11.0)
    grid = fig.add_gridspec(4, 3, left=.085, right=.985, bottom=.075, top=.97,
                            hspace=.65, wspace=.48)
    # Keep the four Salt panels together: table and Fig. 3 above Fig. 5.
    positions = (0, 1, 3, 7, 8, 9, 10, 2, 11)
    titles = (
        "A  Pike and Bosher", "B  Wang 1992", "D  Salt: Table III",
        "H  Rybak: EP", "I  Rybak: endolymph K", "J  Asakuma: EP",
        "K  Asakuma: resistance", "C  Wang 1990",
    )
    for index, ax in enumerate(fig.axes[:9]):
        position = positions[index]
        ax.set_position(grid[position // 3, position % 3].get_position(fig))
        if index < len(titles):
            ax.set_title(titles[index])
    legend_ax = fig.axes[8]
    for text in legend_ax.texts:
        text.set_text("B, E–F shading: noise\nD bars: reported SD\n"
                      "H–I bars: reported SEM\nC bars: timing sensitivity,\n"
                      "not uncertainty\nG ×: unscaled, not fitted")
    for text in fig.texts:
        text.set_text("Joint in-sample fit; Salt acoustic mapping and Table III timing are assumed.")
    for column, variable, title, ylabel in (
            (1, "ep", "E  Salt Fig. 3: EP", "EP change (mV)"),
            (2, "endolymph_k", "F  Salt Fig. 3: K", "K change (mM)")):
        ax = fig.add_subplot(grid[1, column])
        for start, stop in FIG3_INTERVALS:
            ax.axvspan(start, stop, color=".93", zorder=0)
        ax.axhline(0, color=".8", lw=.6)
        model = data["salt_figure_3_model"]
        observed = data["salt_figure_3"][variable]
        ax.plot(model["time_min"], model[variable], color=COLORS[0], lw=1.3)
        ax.scatter(observed["time_min"], observed["observed_change"], s=14,
                   facecolor="white", edgecolor="black", linewidth=.7, zorder=3)
        ax.set(xlim=(0, 90), xticks=[0, 30, 60, 90], xlabel="Time after onset (min)",
               ylabel=ylabel, title=title)
    ax = fig.add_subplot(grid[2, 0])
    ax.axhline(0, color=".8", lw=.6)
    ax.axvspan(0, 20, color=".93", zorder=0)
    ax.axvline(0, color=".6", lw=.6, linestyle="--")
    ax.axvline(20, color=".6", lw=.6, linestyle="--")
    for level, color, marker, linestyle in zip(("95", "105", "115"), COLORS, MARKERS, LINESTYLES):
        observed = data["salt_figure_5"][level]
        model = data["salt_figure_5_model"][level]
        ax.plot(model["time_min"], model["change"], color=color, linestyle=linestyle,
                lw=1.3, label=f"{level} dBA")
        ax.scatter(observed["time_min"], observed["observed_change"], s=14, marker=marker,
                   facecolor="white", edgecolor=color, linewidth=.7, zorder=3)
        points = data["salt_figure_5_unscaled"][level]
        for phase, limits in (("pre_exposure", (-4.2, -.8)),
                              ("post_offset_unscaled_time", (20.8, 24.2))):
            selected = [point for point in points if point["phase"] == phase]
            ax.scatter(np.linspace(*limits, len(selected)), [point["value"] for point in selected],
                       marker="x", s=13, color=color, linewidth=.8, zorder=3)
    ax.set(xlim=(-5, 25), xticks=[0, 10, 20],
           xlabel="Time (min); pre/post unscaled",
           ylabel="EP change (mV)", title="G  Salt Fig. 5: EP")
    low, high = ax.get_ylim()
    ax.set_ylim(low, high + 1.)
    for x, label in ((-2.5, "pre"), (22.5, "post")):
        ax.text(x, .98, label, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=7)
    ax.legend(frameon=False, fontsize=6, loc="best")
    for ax in fig.axes[-3:]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=3, labelsize=7)
        ax.title.set_fontsize(8.5)
        ax.xaxis.label.set_size(8)
        ax.yaxis.label.set_size(8)
    return fig


def main():
    raw = FIT.read_bytes()
    report = json.loads(raw)
    data = build_data(report)
    fig = create_joint_figure(data)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(OUTPUT.with_suffix(f".{extension}"), dpi=300)
    plt.close(fig)
    payload = {
        "status": "in-sample conditional six-study joint fit, not independent validation",
        "fit_source": str(FIT.relative_to(ROOT)),
        "fit_sha256": hashlib.sha256(raw).hexdigest(),
        "notes": [
            "All panels use a single jointly optimized parameter set; Figure 3 EP and K are one animal, Figure 5 traces are separate animals.",
            "Salt Fig. 3 and 5 traces are baseline-relative approximations; transition-adjacent and un-timed post-offset points are not fitted.",
            "Figure 5 crosses show pre- and post-exposure observations with no calibrated times; horizontal positions show order only, not minutes, and were not fitted. No model predictions are drawn in those regions.",
            "Salt Table III error bars are reported SD; Rybak bars are SEM; Wang 1990 bars are timing sensitivity, not uncertainty.",
            "Salt equal-number dBA-to-SPL mapping and Table III EP observation timing are unverified assumptions; components may share animals.",
        ],
        "data": data,
    }
    OUTPUT.with_suffix(".json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {OUTPUT}.[pdf, svg, png, json]")


if __name__ == "__main__":
    main()
