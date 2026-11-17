"""Visualization of the revised in-sample Tier 1 calibration."""

import hashlib
import json
from dataclasses import asdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from src.kinetics_fit import ALL_NAMES
from src.salt_table3_fit import OUTPUT as FIT, salt_predictions
from src.candidate_freeze import FREEZE, load_freeze
from src.wang_noise_cohorts import combined, wang1990
from src.wang_noise_traces import NAMES, load_noise, make_parameters
from src.pike_trace_diagnostic import ROOT, SOURCE, load_anchors
from src.plot_all_fitted import build_data as build_previous_data


COLORS = ("#0072B2", "#D55E00", "#009E73")
MARKERS = ("o", "s", "^")
LINESTYLES = ("-", "--", ":")
MODEL = "#0072B2"
OBSERVED = "#111111"
OUTPUT = FIT.parent / "all_fitted_tier1_salt"


def build_common_data(fit, records):
    """Recompute all pre-existing targets with the revised sound parameters."""
    drug = [fit["drug_parameters_frozen"][key] for key in ALL_NAMES]
    sound = [fit["sound_parameters"][key] for key in NAMES]
    noise = load_noise()
    checks = []
    for delay in (30., 45., 60.):
        score_residual, details = combined(
            sound, drug, noise, delay=delay, method="Radau", rtol=1e-8, atol=1e-10
        )
        checks.append({
            "candidate": "two_study_candidate", "method": "Radau",
            "delay_min": delay, "score": float(score_residual @ score_residual),
            **details,
        })
    _, ranges = wang1990(sound, drug, delay=60., method="Radau", rtol=1e-8,
                         atol=1e-10, audit_grid=True)
    adapter = {
        "status": "conditional exploratory two-study noise optimization",
        "drug_parameters_frozen": fit["drug_parameters_frozen"],
        "sound_parameters": fit["sound_parameters"],
        "drug_recheck": fit["drug_recheck"],
        "checks": checks,
        "wang1990_sampled_ranges_0_1_min": ranges,
    }
    return build_previous_data(adapter, records)


def build_data(report, records):
    """Recompute every plotted prediction from the one Salt-updated candidate."""
    if report.get("status") != "completed conditional joint Wang and Salt Table III fit":
        raise ValueError("completed conditional Salt fit required")
    _, freeze = load_freeze()
    candidate = report["candidate"]
    sound = report["fit"]["optimizer"]["fitted_sound_parameters"]
    if any(candidate["all_model_parameter_values"][name] != sound[name] for name in NAMES):
        raise ValueError("sound parameter values differ within fit report")
    if report["provenance"]["source_sha256"][str(FREEZE.relative_to(ROOT))] != hashlib.sha256(FREEZE.read_bytes()).hexdigest():
        raise ValueError("fit does not match the current historical freeze")
    fit = {
        "drug_parameters_frozen": candidate["drug_parameter_values"],
        "sound_parameters": sound,
        "drug_recheck": {"predictions": report["drug_recheck"]["joint_candidate_predictions"]},
    }
    parameters = make_parameters(
        [sound[key] for key in NAMES],
        [fit["drug_parameters_frozen"][key] for key in ALL_NAMES],
    )
    if asdict(parameters) != candidate["all_model_parameter_values"]:
        raise ValueError("plotted parameters do not match complete fitted candidate")
    data = build_common_data(fit, records)
    recomputed = salt_predictions(
        [sound[key] for key in NAMES],
        [fit["drug_parameters_frozen"][key] for key in ALL_NAMES],
        freeze["initial_state"], method="Radau", rtol=1e-8, atol=1e-10,
    )
    saved = report["fit"]["salt_targets_and_predictions"]
    targets = report["salt_source_data"]["included_rows"]
    if len(recomputed) != 9 or len(saved) != 9 or len(targets) != 9:
        raise ValueError("expected nine Salt predictions and targets")
    conditions = []
    for level in (95, 105, 115):
        rows = []
        for actual, recorded, target in zip(recomputed, saved, targets):
            if actual["noise_dba"] != level:
                continue
            key = (actual["noise_dba"], actual["exposure_number"])
            if key != (target["noise_dba"], target["exposure_number"]) or key != (
                    recorded["noise_dba"], recorded["exposure_number"]):
                raise ValueError("Salt target and prediction order mismatch")
            if any(recorded[name] != target[name] for name in ("mean_mV", "sd_mV", "n_observations")):
                raise ValueError("Salt observed data differ within fit report")
            value = actual["predicted_overall_ep_change_mV"]
            if not np.isclose(value, recorded["predicted_overall_ep_change_mV"],
                              rtol=1e-6, atol=1e-5):
                raise ValueError(f"Salt {key} differs from stored fit")
            rows.append({
                "exposure_number": target["exposure_number"],
                "observed_change_mV": target["mean_mV"],
                "observed_sd_mV": target["sd_mV"],
                "n_observations": target["n_observations"],
                "modeled_change_mV": value,
            })
        conditions.append({"level_dba": level, "exposures": rows})
    data["salt_conditions"] = conditions
    return data


def create_figure(data):
    """Plot all quantitative targets, including the conditional Salt fit."""
    if len(data["pike"]) != 3 or len(data["wang1992_curves"]) != 3:
        raise ValueError("Pike and Wang plots require all three conditions")
    conditions = data["salt_conditions"]
    if (len(conditions) != 3 or
            any([row["exposure_number"] for row in condition["exposures"]] != [1, 2, 3]
                for condition in conditions)):
        raise ValueError("Salt plot requires three levels with three exposures each")
    with plt.rc_context({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}):
        fig, axes = plt.subplots(3, 3, figsize=(7.5, 8.4))
        fig.subplots_adjust(left=.085, right=.985, bottom=.08, top=.96,
                            hspace=.65, wspace=.48)

        ax = axes[0, 0]
        ax.axhline(0, color=".8", lw=.6)
        for curve, color, marker, linestyle in zip(data["pike"], COLORS, MARKERS, LINESTYLES):
            ax.plot(curve["time_min"], curve["model_change_mV"], color=color,
                    linestyle=linestyle, lw=1.3, label=f'{curve["dose_mg_kg"]} mg/kg')
            ax.scatter(curve["observed_time_min"], curve["observed_change_mV"],
                       marker=marker, s=17, facecolor="white", edgecolor=color,
                       linewidth=.8, zorder=3)
        ax.set(xlim=(-2, 122), xticks=[0, 40, 80, 120],
               xlabel="Time after injection (min)", ylabel="EP change (mV)",
               title="A  Pike and Bosher")
        ax.legend(frameon=False, fontsize=6, loc="lower right")

        ax = axes[0, 1]
        ax.axvspan(0, 20, color=".93", zorder=0)
        ax.axhline(0, color=".8", lw=.6)
        for curve, color, marker, linestyle in zip(data["wang1992_curves"], COLORS, MARKERS, LINESTYLES):
            ax.plot(curve["curve_time_min"], curve["curve_change_mV"], color=color,
                    linestyle=linestyle, lw=1.3, label=f'{curve["level_db_spl"]} dB SPL')
            ax.scatter(curve["time_min"], curve["observed_change_mV"], s=17,
                       marker=marker, facecolor="white", edgecolor=color,
                       linewidth=.8, zorder=3)
        ax.set(xlim=(-.5, 30.6), xticks=[0, 10, 20, 30],
               xlabel="Time after noise onset (min)", ylabel="EP change (mV)",
               title="B  Wang 1992")
        ax.legend(frameon=False, fontsize=6, loc="lower left")

        ax = axes[0, 2]
        ax.axhline(0, color=".8", lw=.6)
        for condition, color, marker, linestyle, offset in zip(
                conditions, COLORS, MARKERS, LINESTYLES, (-.16, 0., .16)):
            rows = condition["exposures"]
            x = np.array([row["exposure_number"] for row in rows], dtype=float) + offset
            ax.errorbar(x - .035, [row["observed_change_mV"] for row in rows],
                        yerr=[row["observed_sd_mV"] for row in rows], fmt=marker,
                        markersize=3.5, color=color, markerfacecolor="white",
                        capsize=2, linewidth=.7)
            ax.plot(x + .035, [row["modeled_change_mV"] for row in rows],
                    marker=marker, markersize=3.2, color=color, linestyle=linestyle,
                    lw=1., label=f'{condition["level_dba"]} dBA')
        ax.set(xticks=[1, 2, 3], xlim=(.65, 3.35), xlabel="Exposure number",
               ylabel="Post-noise EP change (mV)", title="C  Salt and Konishi")
        ax.legend(frameon=False, fontsize=6, loc="best")

        rows = data["rybak"]
        doses = np.array([row["dose"] for row in rows])
        for ax, obs, error, mod, title, ylabel in (
                (axes[1, 0], "observed_ep", "ep_se", "model_ep", "D  Rybak: EP", "EP reduction (mV)"),
                (axes[1, 1], "observed_k", "k_se", "model_k", "E  Rybak: endolymph K", "K reduction (mM)")):
            ax.errorbar(doses - 2, [row[obs] for row in rows], yerr=[row[error] for row in rows],
                        fmt="o", markersize=4, color=OBSERVED, markerfacecolor="white",
                        capsize=2, linewidth=.8)
            ax.scatter(doses + 2, [row[mod] for row in rows], marker="s",
                       s=23, color=MODEL, zorder=3)
            ax.set(xlim=(12, 113), ylim=(0, None), xticks=doses,
                   xlabel="Furosemide (mg/kg)", ylabel=ylabel, title=title)
        asakuma = data["asakuma"]
        for ax, observed, modeled, title, ylabel in (
                (axes[1, 2], asakuma["observed_ep_drop"], asakuma["model_ep_drop"],
                 "F  Asakuma: EP", "EP reduction (mV)"),
                (axes[2, 0], asakuma["observed_resistance_ratio"], asakuma["model_resistance_ratio"],
                 "G  Asakuma: resistance", "Resistance ratio")):
            ax.scatter([0], [observed], marker="o", s=23, facecolor="white",
                       edgecolor=OBSERVED, zorder=3)
            ax.scatter([1], [modeled], marker="s", s=23, color=MODEL, zorder=3)
            ax.set(xlim=(-.5, 1.5), ylim=(0, max(observed, modeled) * 1.25),
                   xticks=[0, 1], xticklabels=["Data", "Model"], ylabel=ylabel, title=title)

        ax = axes[2, 1]
        rows = data["wang1990_cohorts"]
        duration = np.array([row["exposure_min"] for row in rows])
        ax.axhline(0, color=".8", lw=.6)
        ax.vlines(duration + 1, [row["model_range_30_60_min_mV"][0] for row in rows],
                  [row["model_range_30_60_min_mV"][1] for row in rows], color=MODEL, lw=1.2)
        ax.plot(duration + 1, [row["modeled_difference_mV"] for row in rows],
                "s-", color=MODEL, markersize=4, lw=1.)
        ax.plot(duration - 1, [row["observed_difference_mV"] for row in rows],
                "o", markerfacecolor="white", markeredgecolor=OBSERVED, markersize=4)
        ax.set(xlim=(10, 90), xticks=[20, 40, 60, 80],
               xlabel="125 dB exposure (min)", ylabel="EP difference (mV)",
               title="H  Wang 1990: separate cohorts")

        ax = axes[2, 2]
        ax.axis("off")
        ax.legend(handles=[
            Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor=OBSERVED,
                   linestyle="none", markersize=5, label="Observed"),
            Line2D([], [], color=MODEL, marker="s", markersize=4, lw=1.3, label="Model"),
        ], loc="upper left", frameon=False, fontsize=8)
        ax.text(0., .55, "B shading: noise exposure\nC bars: reported SD\n"
                "D–E bars: reported SEM\nH bars: timing sensitivity,\nnot uncertainty",
                transform=ax.transAxes, fontsize=7.2, va="top", linespacing=1.4)
        for ax in axes.flat[:-1]:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(direction="out", length=3)
        fig.text(.5, .025, "In-sample, conditional fit; Salt acoustic mapping assumed, "
                 "Wang 1990 delay fixed at 45 min.", ha="center", fontsize=7)
    return fig


def main():
    raw = FIT.read_bytes()
    report = json.loads(raw)
    data = build_data(report, load_anchors())
    fig = create_figure(data)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(OUTPUT.with_suffix(f".{extension}"), dpi=300)
    plt.close(fig)
    payload = {
        "status": "conditional in-sample Tier 1 visualization; not independent validation",
        "fit_source": str(FIT.relative_to(ROOT)),
        "fit_sha256": hashlib.sha256(raw).hexdigest(),
        "source_data_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "limitations": [
            "All panels use the same Salt-updated model parameters, not the historical frozen parameter set.",
            "Salt Table III basal-turn EP overall changes only: fixed illustrative same-number dBA-to-SPL input, not a conversion; 1–3 minute EP average is an unverified observation assumption.",
            "Salt error bars are reported SD; repeated exposures are correlated and n counts observations.",
            "Pike and Wang 1992 markers are correlated approximate digitized means relative to each trace's own baseline.",
            "Rybak bars are reported SEM; Asakuma points are derived from group means.",
            "Wang 1990 model values assume a 45-minute delay; bars show sampled 30–60-minute timing sensitivity, not uncertainty.",
            "Other Tier 1 outcomes including Salt K, EP components and second-turn data are not fitted or plotted.",
        ],
        "data": data,
    }
    OUTPUT.with_suffix(".json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {OUTPUT}.[pdf, svg, png, json]")


if __name__ == "__main__":
    main()
