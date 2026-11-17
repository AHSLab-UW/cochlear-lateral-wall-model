"""Publication-oriented layout for the Figure-5-excluded Tier 1 candidate."""

from dataclasses import asdict
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.model_equations import Inputs, broadband_sound_multiplier, endolymph, related_code_closure, simulate
from src.parameter_freeze import FREEZE, FIT, load_freeze
from src.salt_trace_residuals import FIG3_INTERVALS
from src.kinetics_fit import ALL_NAMES
from src.wang_noise_traces import NAMES, make_parameters
from src.pike_trace_diagnostic import ROOT, load_anchors
from src.plot_calibration_data import COLORS, build_common_data, create_figure


OUTPUT = FIT.parent / "six_study_final_tier1_figure"


def build_final_data(report, parameters, freeze):
    """Recompute the displayed predictions from the verified frozen candidate."""
    if report["salt_fig5_exclusion"]["used_in_objective"] is not False:
        raise ValueError("the final figure requires a Figure-5-free fit")
    if report["all_model_parameter_values"] != asdict(parameters):
        raise ValueError("fit and frozen model parameters differ")
    if report["parameters"] != freeze["fitted_parameter_values"]:
        raise ValueError("fit and frozen fitted parameters differ")
    values = report["parameters"]
    drug = [values[name] for name in ALL_NAMES]
    sound = [values[name] for name in NAMES]
    if asdict(make_parameters(sound, drug)) != asdict(parameters):
        raise ValueError("figure parameter reconstruction differs from freeze")
    check = report["numerical_checks"][2]
    if check["solver"]["step"] != .025 or check["solver"]["method"] != "Radau":
        raise ValueError("expected fine-grid Radau recheck")
    components = check["components"]
    reference = {
        **components["pike_bosher"]["predictions"],
        "asakuma_60": components["asakuma"]["predictions"],
        **components["rybak"]["predictions"],
    }
    data = build_common_data({
        "drug_parameters_frozen": dict(zip(ALL_NAMES, drug)),
        "sound_parameters": dict(zip(NAMES, sound)),
        "drug_recheck": {"predictions": reference},
    }, load_anchors())
    table = components["salt_table_iii"]["predictions"]
    if len(table) != 9:
        raise ValueError("expected nine Salt Table III observations")
    data["salt_conditions"] = [
        {"level_dba": level, "exposures": [
            {"exposure_number": row["exposure_number"], "observed_change_mV": row["mean_mV"],
             "observed_sd_mV": row["sd_mV"], "n_observations": row["n_observations"],
             "modeled_change_mV": row["predicted_overall_ep_change_mV"]}
            for row in table if row["noise_dba"] == level]}
        for level in (95, 105, 115)
    ]
    data["salt_figure_3"] = components["salt_figure_3"]["predictions"]
    times = np.unique(np.r_[np.linspace(0., 90., 901), *(
        data["salt_figure_3"][key]["time_min"] for key in ("ep", "endolymph_k"))])
    forcing = lambda t: Inputs(sound_multiplier=broadband_sound_multiplier(115.)
                               if any(start <= t < end for start, end in FIG3_INTERVALS)
                               else 1.)
    states = simulate(times, forcing, related_code_closure(), parameters,
                      initial=freeze["initial_state"],
                      boundaries=tuple(edge for interval in FIG3_INTERVALS for edge in interval
                                       if 0. < edge < 90.),
                      method="Radau", rtol=1e-8, atol=1e-10)
    potassium = np.array([endolymph(state, parameters)[1] for state in states.T])
    curves = {"ep": states[8] - states[8, 0],
              "endolymph_k": potassium - potassium[0]}
    for key, curve in curves.items():
        target = data["salt_figure_3"][key]
        if not np.allclose(curve[np.searchsorted(times, target["time_min"])],
                           target["modeled_change"], rtol=1e-6, atol=1e-5):
            raise ValueError(f"Figure 3 {key} curve differs from frozen fit")
    data["salt_figure_3_model"] = {"time_min": times.tolist(),
                                    **{key: curve.tolist() for key, curve in curves.items()}}
    return data


def create_final_figure(data):
    """Draw all fitted targets; caller supplies one verified frozen candidate."""
    fig = create_figure(data)
    fig.set_size_inches(7.5, 10.1)
    grid = fig.add_gridspec(4, 3, left=.085, right=.985, bottom=.075, top=.97,
                            hspace=.65, wspace=.48)
    positions = (0, 1, 3, 6, 7, 8, 9, 2, 10)
    titles = (
        "A  Pike and Bosher", "B  Wang 1992", "D  Salt: Table III",
        "G  Rybak: EP", "H  Rybak: endolymph K", "I  Asakuma: EP",
        "J  Asakuma: resistance", "C  Wang 1990",
    )
    for index, ax in enumerate(fig.axes[:9]):
        position = positions[index]
        ax.set_position(grid[position // 3, position % 3].get_position(fig))
        if index < len(titles):
            ax.set_title(titles[index])
    legend_ax = fig.axes[8]
    for text in legend_ax.texts:
        text.set_text("B, E–F shading: noise\nD bars: reported SD\n"
                      "G–H bars: reported SEM\nC bars: timing sensitivity,\n"
                      "not uncertainty")
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
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=3, labelsize=7)
        ax.title.set_fontsize(8.5)
        ax.xaxis.label.set_size(8)
        ax.yaxis.label.set_size(8)
    return fig


def main():
    parameters, freeze = load_freeze()
    report = json.loads(FIT.read_text())
    data = build_final_data(report, parameters, freeze)
    paths = [OUTPUT.with_suffix(f".{extension}") for extension in ("pdf", "svg", "png", "json")]
    if any(path.exists() for path in paths):
        raise FileExistsError("final figure output already exists; refusing to overwrite")
    fig = create_final_figure(data)
    for path in paths[:3]:
        fig.savefig(path, dpi=300)
    plt.close(fig)
    payload = {
        "status": "in-sample conditional six-study fit; not independent validation",
        "freeze_source": str(FREEZE.relative_to(ROOT)),
        "freeze_sha256": hashlib.sha256(FREEZE.read_bytes()).hexdigest(),
        "fit_source": str(FIT.relative_to(ROOT)),
        "fit_sha256": hashlib.sha256(FIT.read_bytes()).hexdigest(),
        "fit_residual_counts": {name: component["residual_count"]
                                for name, component in report["components"].items()},
        "limitations": report["limitations"],
        "data": data,
    }
    with paths[3].open("x", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, allow_nan=False)
        file.write("\n")
    print(f"Wrote {OUTPUT}.[pdf, svg, png, json]")


if __name__ == "__main__":
    main()
