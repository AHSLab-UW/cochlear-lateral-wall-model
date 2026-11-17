#!/usr/bin/env python3
"""Master figure generator for all Hearing Research manuscript figures (Figures 1-5 and Graphical Abstract)."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import numpy as np

FIG_DIR = Path(__file__).resolve().parent

# Colorblind-safe Okabe-Ito & scientific journal palette
INK = "#1E293B"
SLATE = "#475569"
MUTED = "#64748B"
LIGHT = "#CBD5E1"
GRID_COLOR = "#E2E8F0"
EXPOSURE_BG = "#E8EEF5"
WARN_BG = "#FEF2F2"

BLUE = "#0072B2"       # Primary / Low dose / 105 dB / ISP
ORANGE = "#D55E00"     # Secondary / Mid dose / 115 dB / Data highlight
GREEN = "#009E73"      # Tertiary / High dose / 125 dB / EP
PURPLE = "#CC79A7"     # Accent / Counterfactual
AMBER = "#E69F00"      # Accent / Warning
CRIMSON = "#B91C1C"    # Threshold / Audit bound
LEVEL_COLORS = [BLUE, ORANGE, GREEN]


def configure_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "mathtext.fontset": "dejavusans",
        "font.size": 6.8,
        "axes.labelsize": 6.8,
        "axes.titlesize": 7.3,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.linewidth": 0.75,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 6.3,
        "ytick.labelsize": 6.3,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "legend.fontsize": 6.0,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.facecolor": "white",
        "legend.edgecolor": "none",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def save_all_formats(fig, stem: str, number: int | None = None):
    fig.canvas.draw()
    pdf_path = FIG_DIR / f"{stem}.pdf"
    svg_path = FIG_DIR / f"{stem}.svg"
    png_path = FIG_DIR / f"{stem}.png"
    tiff_path = FIG_DIR / f"{stem}.tiff"

    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(tiff_path, dpi=600, bbox_inches="tight", pad_inches=0.05)

    if number is not None:
        for ext, src in ((".pdf", pdf_path), (".svg", svg_path), (".png", png_path), (".tiff", tiff_path)):
            shutil.copy2(src, FIG_DIR / f"Figure_{number}{ext}")
    plt.close(fig)


def style_axis(ax, letter: str, title: str, *, pad: float = 4.5, grid: bool = True):
    ax.set_title(f"{letter}   {title}", loc="left", pad=pad, fontsize=7.2, fontweight="bold", color=INK)
    ax.tick_params(direction="out", pad=2.0)
    ax.xaxis.labelpad = 2.5
    ax.yaxis.labelpad = 2.5
    if grid:
        ax.grid(True, linestyle=":", linewidth=0.45, color=GRID_COLOR, zorder=0)


# ==============================================================================
# FIGURE 1: JOINT TIER 1 CALIBRATION (10 PANELS IN SPACIOUS 3-ROW LAYOUT)
# ==============================================================================
def build_figure1(data: dict | None = None):
    configure_style()
    if data is None:
        data = json.loads((FIG_DIR / "tier1-figure-data.json").read_text())
    fig = plt.figure(figsize=(7.6, 7.3))
    gs = GridSpec(
        3, 12, figure=fig,
        left=0.065, right=0.985, bottom=0.065, top=0.915,
        wspace=2.05, hspace=0.58,
        height_ratios=[1.05, 1.05, 0.94]
    )

    banner_handles = [
        Line2D([], [], color=BLUE, lw=1.5, marker="s", ms=3.8, label="Frozen 13-state model (solid lines / filled squares)"),
        Line2D([], [], color=ORANGE, lw=0, marker="o", ms=4.5, mfc="white", mec=ORANGE, mew=1.0, label="In vivo observations (open circles ± SD or SE)"),
        mpatches.Patch(facecolor=EXPOSURE_BG, edgecolor="#CBD5E1", lw=0.6, label="Acoustic exposure interval"),
    ]
    fig.legend(
        handles=banner_handles, loc="upper center", bbox_to_anchor=(0.52, 0.985),
        ncol=3, fontsize=6.6, frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1",
        columnspacing=1.5, handlelength=1.8, borderpad=0.45
    )

    # --- ROW 1: A (Pike & Bosher), B (Wang 1992), C (Wang 1990) ---
    axA = fig.add_subplot(gs[0, 0:4])
    style_axis(axA, "A", "Pike & Bosher (1980): IV furosemide")
    axA.axhline(0, color=LIGHT, lw=0.7, zorder=1)
    for curve, color, dose in zip(data["pike"], LEVEL_COLORS, (40, 60, 80)):
        axA.plot(curve["time_min"], curve["model_change_mV"], color=color, lw=1.35, zorder=3, label=f"{dose} mg/kg")
        axA.scatter(
            curve["observed_time_min"], curve["observed_change_mV"],
            s=14, marker="o", facecolor="white", edgecolor=color, linewidth=0.85, zorder=4
        )
    axA.set(xlim=(-4, 128), ylim=(-128, 10), xticks=[0, 30, 60, 90, 120],
            xlabel="Time after infusion (min)", ylabel="ΔEP from baseline (mV)")
    axA.legend(loc="lower right", fontsize=5.8, title="Furosemide dose", title_fontsize=5.8)

    axB = fig.add_subplot(gs[0, 4:8])
    style_axis(axB, "B", "Wang et al. (1992): broadband noise")
    axB.axvspan(0, 20, color=EXPOSURE_BG, zorder=0)
    axB.axvline(20, color=MUTED, lw=0.6, ls="--", zorder=1)
    axB.axhline(0, color=LIGHT, lw=0.7, zorder=1)
    for curve, color, lvl in zip(data["wang1992_curves"], LEVEL_COLORS, (105, 115, 125)):
        axB.plot(curve["curve_time_min"], curve["curve_change_mV"], color=color, lw=1.35, zorder=3, label=f"{lvl} dB SPL")
        axB.scatter(
            curve["time_min"], curve["observed_change_mV"],
            s=14, marker="o", facecolor="white", edgecolor=color, linewidth=0.85, zorder=4
        )
    axB.text(10, 9.2, "Noise (20 min)", ha="center", va="center", fontsize=5.6, color=SLATE, fontweight="bold")
    axB.set(xlim=(-2, 34), ylim=(-26, 12.5), xticks=[0, 10, 20, 30],
            xlabel="Time from noise onset (min)", ylabel="ΔEP from baseline (mV)")
    axB.legend(loc="lower left", fontsize=5.8, title="Broadband level", title_fontsize=5.8)

    axC = fig.add_subplot(gs[0, 8:12])
    style_axis(axC, "C", "Wang et al. (1990): duration cohorts*")
    cohorts = data["wang1990_cohorts"]
    durations = np.array([row["exposure_min"] for row in cohorts])
    for i, (x, row) in enumerate(zip(durations, cohorts)):
        low, high = row["model_range_30_60_min_mV"]
        axC.vlines(x, low, high, color=BLUE, lw=2.0, alpha=0.45, zorder=2,
                   label="30–60 min delay range" if i == 0 else None)
    axC.plot(durations, [r["modeled_difference_mV"] for r in cohorts],
             "s-", color=BLUE, ms=3.8, lw=1.25, zorder=3, label="Model (45-min delay)")
    axC.scatter(durations, [r["observed_difference_mV"] for r in cohorts],
                s=20, marker="o", facecolor="white", edgecolor=ORANGE, linewidth=1.0, zorder=4,
                label="Observed cohort ΔEP")
    axC.set(xlim=(10, 90), ylim=(-25.5, -6.0), xticks=[20, 40, 60, 80],
            xlabel="125 dB SPL exposure duration (min)", ylabel="Exposed − control EP (mV)")
    axC.legend(loc="upper right", fontsize=5.6)

    # --- ROW 2: D (Salt Table III), E (Salt Fig 3 EP), F (Salt Fig 3 K) ---
    axD = fig.add_subplot(gs[1, 0:4])
    style_axis(axD, "D", "Salt & Konishi (1979): repeated exposures")
    axD.axhline(0, color=LIGHT, lw=0.7, zorder=1)
    offsets = (-0.14, 0.0, 0.14)
    for condition, color, offset, lvl in zip(data["salt_conditions"], LEVEL_COLORS, offsets, (95, 105, 115)):
        rows = condition["exposures"]
        x = np.array([row["exposure_number"] for row in rows], dtype=float)
        axD.errorbar(
            x + offset, [r["observed_change_mV"] for r in rows],
            yerr=[r["observed_sd_mV"] for r in rows], fmt="o",
            markersize=3.6, color=color, markerfacecolor="white", markeredgewidth=0.85,
            capsize=2.0, linewidth=0.8, zorder=3
        )
        axD.plot(
            x + offset, [r["modeled_change_mV"] for r in rows],
            color=color, marker="s", markersize=3.2, lw=1.15, zorder=4, label=f"{lvl} dBA"
        )
    axD.set(xlim=(0.6, 3.4), ylim=(-3.2, 8.8), xticks=[1, 2, 3],
            xticklabels=["Exp. 1", "Exp. 2", "Exp. 3"],
            xlabel="Sequential 20-min exposure", ylabel="Early post-onset ΔEP (mV)")
    axD.legend(loc="upper right", fontsize=5.8, title="Noise level", title_fontsize=5.8)

    salt = data["salt_figure_3"]
    model = data["salt_figure_3_model"]
    axE = fig.add_subplot(gs[1, 4:8])
    style_axis(axE, "E", "Salt & Konishi (1979): paired EP trace")
    for s0, s1 in ((0, 20), (30, 50), (60, 80)):
        axE.axvspan(s0, s1, color=EXPOSURE_BG, zorder=0)
    axE.axhline(0, color=LIGHT, lw=0.7, zorder=1)
    axE.plot(model["time_min"], model["ep"], color=BLUE, lw=1.25, zorder=3, label="Model EP")
    axE.scatter(
        salt["ep"]["time_min"], salt["ep"]["observed_change"],
        s=13, marker="o", facecolor="white", edgecolor=ORANGE, linewidth=0.8, zorder=4, label="Observed EP"
    )
    axE.set(xlim=(-3, 98), ylim=(-7.8, 7.0), xticks=[0, 20, 40, 60, 80, 100],
            xlabel="Time (min; 3 × 20-min exposures at 115 dBA)", ylabel="ΔEP from baseline (mV)")
    axE.legend(loc="lower right", fontsize=5.6)

    axF = fig.add_subplot(gs[1, 8:12])
    style_axis(axF, "F", "Salt & Konishi (1979): paired K+ trace")
    for s0, s1 in ((0, 20), (30, 50), (60, 80)):
        axF.axvspan(s0, s1, color=EXPOSURE_BG, zorder=0)
    axF.axhline(0, color=LIGHT, lw=0.7, zorder=1)
    axF.plot(model["time_min"], model["endolymph_k"], color=BLUE, lw=1.25, zorder=3, label=r"Model $K_E$")
    axF.scatter(
        salt["endolymph_k"]["time_min"], salt["endolymph_k"]["observed_change"],
        s=13, marker="o", facecolor="white", edgecolor=ORANGE, linewidth=0.8, zorder=4, label=r"Observed $K_E$"
    )
    axF.set(xlim=(-3, 98), ylim=(-3.2, 10.2), xticks=[0, 20, 40, 60, 80, 100],
            xlabel="Time (min; same animal as Panel E)", ylabel=r"Endolymph $\Delta K_E$ (mM)")
    axF.legend(loc="lower right", fontsize=5.6)

    # --- ROW 3: G (Rybak EP), H (Rybak K), I (Asakuma EP), J (Asakuma R ratio) ---
    doses = np.array([r["dose"] for r in data["rybak"]], dtype=float)
    rows = data["rybak"]

    axG = fig.add_subplot(gs[2, 0:3])
    style_axis(axG, "G", "Rybak & Morizono (1982)")
    axG.errorbar(
        doses - 1.8, [r["observed_ep"] for r in rows], yerr=[r["ep_se"] for r in rows],
        fmt="o", markersize=4.0, color=ORANGE, markerfacecolor="white", markeredgewidth=0.9,
        capsize=2.0, linewidth=0.85, zorder=4, label="Observed ± SE"
    )
    axG.plot(
        doses + 1.8, [r["model_ep"] for r in rows], "s-",
        color=BLUE, markersize=3.5, linewidth=1.2, zorder=3, label="Model"
    )
    axG.set(xlim=(12, 112), ylim=(10, 132), xticks=[25, 50, 100],
            xlabel="Furosemide dose (mg/kg)", ylabel="Peak EP drop (mV)")
    axG.legend(loc="lower right", fontsize=5.4)

    axH = fig.add_subplot(gs[2, 3:6])
    style_axis(axH, "H", "Rybak & Morizono (1982)")
    axH.errorbar(
        doses - 1.8, [r["observed_k"] for r in rows], yerr=[r["k_se"] for r in rows],
        fmt="o", markersize=4.0, color=ORANGE, markerfacecolor="white", markeredgewidth=0.9,
        capsize=2.0, linewidth=0.85, zorder=4, label="Observed ± SE"
    )
    axH.plot(
        doses + 1.8, [r["model_k"] for r in rows], "s-",
        color=BLUE, markersize=3.5, linewidth=1.2, zorder=3, label="Model"
    )
    axH.set(xlim=(12, 112), ylim=(0, 44), xticks=[25, 50, 100],
            xlabel="Furosemide dose (mg/kg)", ylabel=r"Peak $K_E$ drop (mM)")
    axH.legend(loc="lower right", fontsize=5.4)

    axI = fig.add_subplot(gs[2, 6:9])
    style_axis(axI, "I", "Asakuma & Snow (1980)")
    obs_ep = data["asakuma"]["observed_ep_drop"]
    mod_ep = data["asakuma"]["model_ep_drop"]
    axI.bar([0, 1], [obs_ep, mod_ep], width=0.44, color=["#FFF7ED", "#EFF6FF"],
            edgecolor=[ORANGE, BLUE], linewidth=1.2, zorder=2)
    axI.scatter([0], [obs_ep], s=30, marker="o", facecolor="white", edgecolor=ORANGE, linewidth=1.2, zorder=4)
    axI.scatter([1], [mod_ep], s=30, marker="s", color=BLUE, zorder=4)
    axI.text(0, obs_ep + 5.5, f"{obs_ep:.1f} mV", ha="center", va="bottom", fontsize=5.9, fontweight="bold", color=ORANGE)
    axI.text(1, mod_ep + 5.5, f"{mod_ep:.1f} mV", ha="center", va="bottom", fontsize=5.9, fontweight="bold", color=BLUE)
    axI.set(xlim=(-0.6, 1.6), ylim=(0, 145), xticks=[0, 1], xticklabels=["Observed\n(n = 6)", "Frozen\nmodel"],
            xlabel="60 mg/kg IV furosemide", ylabel="Nadir EP drop (mV)")

    axJ = fig.add_subplot(gs[2, 9:12])
    style_axis(axJ, "J", "Asakuma & Snow (1980)")
    obs_r = data["asakuma"]["observed_resistance_ratio"]
    mod_r = data["asakuma"]["model_resistance_ratio"]
    axJ.axhline(1.0, color=MUTED, lw=0.75, ls="--", zorder=1)
    axJ.text(0.5, 1.05, "Rest (1.0×)", ha="center", va="bottom", fontsize=5.4, color=SLATE,
             bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.85))
    axJ.bar([0, 1], [obs_r, mod_r], width=0.44, color=["#FFF7ED", "#EFF6FF"],
            edgecolor=[ORANGE, BLUE], linewidth=1.2, zorder=2)
    axJ.scatter([0], [obs_r], s=30, marker="o", facecolor="white", edgecolor=ORANGE, linewidth=1.2, zorder=4)
    axJ.scatter([1], [mod_r], s=30, marker="s", color=BLUE, zorder=4)
    axJ.text(0, obs_r + 0.10, f"{obs_r:.2f}×", ha="center", va="bottom", fontsize=5.9, fontweight="bold", color=ORANGE)
    axJ.text(1, mod_r + 0.10, f"{mod_r:.2f}×", ha="center", va="bottom", fontsize=5.9, fontweight="bold", color=BLUE)
    axJ.set(xlim=(-0.6, 1.6), ylim=(0, 2.65), xticks=[0, 1], xticklabels=["Observed\n(n = 6)", "Frozen\nmodel"],
            xlabel="60 mg/kg IV furosemide", ylabel=r"Resistance ratio ($R/R_0$)")

    save_all_formats(fig, "figure1-tier1-joint-fit", number=1)


# ==============================================================================
# FIGURE 2: RIGOROUS STABILITY, PERTURBATION & FLUX-BUDGET DIAGNOSTICS
# ==============================================================================
def build_figure2(stab: dict | None = None):
    configure_style()
    if stab is None:
        diag = json.loads((FIG_DIR / "diagnostics-figure-data.json").read_text())
        stab = diag["stability"]

    fig = plt.figure(figsize=(7.5, 5.4))
    gs = GridSpec(
        2, 2, figure=fig,
        left=0.08, right=0.98, bottom=0.10, top=0.92,
        wspace=0.36, hspace=0.48
    )

    # Panel A: Full 12-mode eigenvalue decay spectrum + slowest mode across 6 branches
    axA = fig.add_subplot(gs[0, 0])
    style_axis(axA, "A", "Local Jacobian spectrum at constructed rest balance")
    branches = stab["branch_spectra"]
    for idx, b in enumerate(branches):
        eigs = sorted([ev[0] for ev in b["eigenvalues_per_min"]])
        neg_rates = [-val for val in eigs]
        axA.scatter(
            np.full(len(neg_rates), idx + 1), neg_rates,
            s=18, color=BLUE if b["atp_side"] < 0 else GREEN,
            alpha=0.78, edgecolors="white", linewidth=0.4, zorder=3,
            label=("ATP− branch" if idx == 0 else ("ATP+ branch" if idx == 3 else None))
        )
    axA.axhline(0.010526, color=ORANGE, lw=0.95, ls="--", zorder=2)
    axA.text(6.45, 0.0135, r"Slowest mode: $\mathrm{Re}(\lambda) = -0.0105\ \mathrm{min}^{-1}$" + "\n(all 12 modes strictly stable < 0)",
             ha="right", va="bottom", fontsize=5.8, color=ORANGE, fontweight="bold")
    axA.set_yscale("log")
    axA.set(
        xlim=(0.4, 6.6), ylim=(0.005, 110.0), xticks=range(1, 7),
        xticklabels=[
            r"B1 ($-,10^{-4}$)", r"B2 ($-,10^{-5}$)", r"B3 ($-,10^{-6}$)",
            r"B4 ($+,10^{-4}$)", r"B5 ($+,10^{-5}$)", r"B6 ($+,10^{-6}$)"
        ],
        xlabel="One-sided Jacobian branch (ATP side, relative step)",
        ylabel=r"Decay rate $-\mathrm{Re}(\lambda)$ ($\mathrm{min}^{-1}$, log scale)"
    )
    axA.tick_params(axis="x", labelsize=5.6)
    axA.legend(loc="center right", fontsize=5.6)

    # Panel B: State-resolved perturbation amplification across protocols (240 min)
    axB = fig.add_subplot(gs[0, 1])
    style_axis(axB, "B", "Finite-horizon (±1%) perturbation amplification (240 min)")
    protocols = [
        ("Rest", stab["finite_perturbations"], "#64748B"),
        ("105 dB", stab["forced_protocols"][0]["cases"], BLUE),
        ("125 dB", stab["forced_protocols"][1]["cases"], GREEN),
        ("Drug 80", stab["forced_protocols"][2]["cases"], ORANGE),
    ]
    state_labels = ["Trauma\n($W$)", "Syncytial\n($K_C$)", "IS volume\n($v_S$)", "Adaptation\n($\\alpha$)", "Remaining\n8 states"]
    x_pos = np.arange(len(state_labels))
    bar_w = 0.18
    for p_idx, (p_label, cases, col) in enumerate(protocols):
        by_st = {}
        for c in cases:
            s = c["state"]
            m = c["methods"]["Radau"]["distance_to_matched_control"]["maximum"]
            by_st[s] = max(by_st.get(s, 0.0), m)
        vals = [
            by_st["trauma"],
            by_st["k_sync"],
            by_st["strial_volume"],
            by_st["met_adaptation"],
            max(v for k, v in by_st.items() if k not in ("trauma", "k_sync", "strial_volume", "met_adaptation"))
        ]
        axB.bar(x_pos + (p_idx - 1.5) * bar_w, vals, width=bar_w, color=col, label=p_label, zorder=3)
    axB.axhline(0.01, color=INK, lw=0.75, ls=":", zorder=2)
    axB.text(4.45, 0.0115, "Initial 1% perturbation (0.01)", ha="right", va="bottom", fontsize=5.6, color=INK)
    axB.set_yscale("log")
    axB.set(xlim=(-0.55, 4.55), ylim=(0.006, 1.3), xticks=x_pos, xticklabels=state_labels,
            xlabel="Perturbed initial state (±1% perturbation)",
            ylabel="Max normalized trajectory distance")
    axB.legend(loc="upper right", fontsize=5.6, ncol=2, title="Forcing protocol", title_fontsize=5.6)

    # Panel C: Analyst-selected screen crossings vs. capacity-feedback counterfactual
    axC = fig.add_subplot(gs[1, 0])
    style_axis(axC, "C", "Marginal-state screen crossings & capacity-feedback test")
    categories = [
        "125 dB noise:\nPeak $K_M$\n(ceiling 180 mM)",
        "Drug 80 mg/kg:\nMin $K_M$\n(floor 60 mM)",
        "Drug 80 mg/kg:\nPeak $A_M$\n(ceiling 1.50)"
    ]
    y_pos = np.arange(3)
    ref_norm = [
        (195.039 - 60.0) / 120.0 * 100.0,
        (52.464 - 60.0) / 120.0 * 100.0,
        (1.8386 - 0.1) / 1.4 * 100.0,
    ]
    cap_norm = [
        (164.25 - 60.0) / 120.0 * 100.0,
        (85.37 - 60.0) / 120.0 * 100.0,
        (1.195 - 0.1) / 1.4 * 100.0,
    ]
    axC.axvspan(0, 100, color="#F0FDF4", alpha=0.85, zorder=0, label="Analyst-selected screen display [0%, 100%]")
    axC.axvline(0, color=CRIMSON, lw=0.9, ls="--", zorder=2)
    axC.axvline(100, color=CRIMSON, lw=0.9, ls="--", zorder=2)
    axC.barh(y_pos - 0.16, ref_norm, height=0.28, color=ORANGE, zorder=3, label="Frozen reference")
    axC.barh(y_pos + 0.16, cap_norm, height=0.28, color=BLUE, zorder=3, label=r"Capacity feedback ($f_K^*, f_A^*$)")

    axC.text(ref_norm[0] + 2.5, 0 - 0.16, "195.0 mM (+15.0)", va="center", ha="left", fontsize=5.6, fontweight="bold", color=ORANGE)
    axC.text(cap_norm[0] - 2.5, 0 + 0.16, "164.3 mM", va="center", ha="right", fontsize=5.6, fontweight="bold", color="white")

    axC.text(2.5, 1 - 0.16, "52.5 mM (−7.5 below floor)", va="center", ha="left", fontsize=5.6, fontweight="bold", color=ORANGE)
    axC.text(cap_norm[1] + 2.5, 1 + 0.16, "85.4 mM", va="center", ha="left", fontsize=5.6, fontweight="bold", color=BLUE)

    axC.text(ref_norm[2] + 2.5, 2 - 0.16, "1.84 (+0.34)", va="center", ha="left", fontsize=5.6, fontweight="bold", color=ORANGE)
    axC.text(cap_norm[2] - 2.5, 2 + 0.16, "1.20", va="center", ha="right", fontsize=5.6, fontweight="bold", color="white")

    axC.set(xlim=(-18, 178), ylim=(-0.55, 2.65), yticks=y_pos, yticklabels=categories,
            xlabel="Position within analyst-selected screening interval (0% = floor, 100% = ceiling)")
    axC.invert_yaxis()
    axC.legend(loc="center right", bbox_to_anchor=(1.0, 0.46), fontsize=5.3)

    # Panel D: Exact voltage partition of the -23.76 mV capacity-feedback EP shift at 125 dB
    axD = fig.add_subplot(gs[1, 1])
    style_axis(axD, "D", "20-min 125-dB EP partition: capacity feedback vs. reference")
    terms = [
        "Intrastrial\n$f_K \\Delta E_{CS}$\n($K_S$: 2.9→9.1)",
        "Apical\n$\\Delta E_{ME}$\n($K_M$: 195→164)",
        "ATP pump\n$V_{\\mathrm{pump}} \\Delta\\bar{A}$\n($A_M$: 0.90→0.96)",
        "Circuit load\n$-\\Delta I_{\\mathrm{circ}}/G$\n(current)",
        "Net dynamic\n$\\Delta$EP shift\n(60.7→36.9 mV)"
    ]
    contribs = [-21.790, -3.605, +1.642, -0.012, -23.762]
    colors_d = [ORANGE, AMBER, GREEN, MUTED, CRIMSON]
    x_d = np.arange(len(terms))
    axD.axhline(0, color=INK, lw=0.7, zorder=1)
    axD.bar(x_d, contribs, width=0.50, color=colors_d, zorder=3)
    for xi, val in zip(x_d, contribs):
        offset = -1.2 if val < 0 else 0.7
        va = "top" if val < 0 else "bottom"
        axD.text(xi, val + offset, f"{val:+.2f} mV", ha="center", va=va, fontsize=5.7, fontweight="bold", color=INK)
    axD.set(xlim=(-0.6, 4.6), ylim=(-28.5, 5.5), xticks=x_d, xticklabels=terms,
            ylabel="Contribution to ΔEP at 20 min (mV)")
    axD.tick_params(axis="x", labelsize=5.4)

    save_all_formats(fig, "figure2-stability-and-model-only-forecast", number=2)


# ==============================================================================
# FIGURE 3: DENSE MODEL-ONLY SOUND SCENARIO TRAJECTORIES (4 PANELS A-D)
# ==============================================================================
def build_figure3(data: dict | None = None):
    configure_style()
    if data is None:
        data = json.loads((FIG_DIR / "figure3-model-only-scenario-states-figure-data.json").read_text())
    fig = plt.figure(figsize=(7.2, 4.9))
    gs = GridSpec(
        2, 2, figure=fig,
        left=0.085, right=0.98, bottom=0.14, top=0.91,
        wspace=0.30, hspace=0.44
    )

    colors = {105: BLUE, 115: ORANGE, 125: GREEN}
    panels = [
        (0, 0, "A", "ep_mV", "Endocochlear potential ($V$)", "EP (mV)", 80.0),
        (0, 1, "B", "marginal_k_mM", "Marginal-cell potassium ($K_M$)", r"$K_M$ (mM)", 145.0),
        (1, 0, "C", "endolymph_k_mM", "Derived endolymph potassium ($K_E$)", r"$K_E$ (mM)", 150.0),
        (1, 1, "D", "trauma_state", "Acute acoustic-trauma proxy ($W$)", "Trauma state $W$ (dimensionless)", 0.0),
    ]

    for row, col, letter, key, title, ylabel, baseline in panels:
        ax = fig.add_subplot(gs[row, col])
        style_axis(ax, letter, title)
        ax.axvspan(0, 20, color=EXPOSURE_BG, zorder=0)
        ax.axvline(20, color=MUTED, lw=0.65, ls="--", zorder=1)
        ax.axhline(baseline, color=LIGHT, lw=0.75, ls=":", zorder=1)

        for scenario in data["scenarios"]:
            level = int(scenario["level_db_spl"])
            ax.plot(
                scenario["time_min"], scenario[key],
                color=colors[level], lw=1.4, label=f"{level} dB SPL", zorder=3
            )

        ax.set(xlim=(0, 122), xticks=[0, 20, 40, 60, 80, 100, 120],
               xlabel="Time from noise onset (min)", ylabel=ylabel)

        if key == "ep_mV":
            ax.axhline(64.0, color=CRIMSON, lw=0.75, ls="--", alpha=0.7, zorder=2)
            ax.text(119, 64.6, "80% recovery threshold (64 mV)", ha="right", va="bottom", fontsize=5.6, color=CRIMSON)
            ax.annotate("Post-offset overshoot\n(86.0 mV at 21.6 min)", xy=(21.6, 85.96), xytext=(46, 84.8),
                        arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.7),
                        fontsize=5.7, color=BLUE, fontweight="bold")
            ax.set_ylim(58.5, 88.0)
        elif key == "marginal_k_mM":
            ax.axhline(180.0, color=CRIMSON, lw=0.85, ls="--", zorder=2)
            ax.axhspan(180.0, 198.0, color=WARN_BG, alpha=0.7, zorder=0)
            ax.text(119, 181.2, "Analyst-selected upper screen (180 mM)", ha="right", va="bottom", fontsize=5.6, color=CRIMSON, fontweight="bold")
            ax.annotate("Peak 195.0 mM\n(crosses 180 mM at 8.15 min)", xy=(18.5, 195.0), xytext=(45, 188.5),
                        arrowprops=dict(arrowstyle="->", color=GREEN, lw=0.7),
                        fontsize=5.7, color=GREEN, fontweight="bold")
            ax.set_ylim(131.0, 198.0)
        elif key == "endolymph_k_mM":
            ax.text(10, 168.8, "Noise\n(0–20 min)", ha="center", va="top", fontsize=5.6, color=SLATE, fontweight="bold")
            ax.set_ylim(147.5, 172.0)
        elif key == "trauma_state":
            ax.annotate(r"Exponential repair ($\tau_R = 95\ \mathrm{min}$)", xy=(60, 0.243), xytext=(55, 0.31),
                        arrowprops=dict(arrowstyle="->", color=GREEN, lw=0.7),
                        fontsize=5.7, color=GREEN, fontweight="bold")
            ax.set_ylim(-0.015, 0.40)

    legend_handles = [
        Line2D([], [], color=BLUE, lw=1.6, label="105 dB SPL (sub-trauma)"),
        Line2D([], [], color=ORANGE, lw=1.6, label="115 dB SPL (moderate trauma)"),
        Line2D([], [], color=GREEN, lw=1.6, label="125 dB SPL (intense trauma)"),
        mpatches.Patch(facecolor=EXPOSURE_BG, edgecolor="#CBD5E1", lw=0.6, label="20-min broadband exposure"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", bbox_to_anchor=(0.52, 0.015),
        ncol=4, fontsize=6.4, frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1",
        columnspacing=1.4, handlelength=1.6, borderpad=0.4
    )

    save_all_formats(fig, "figure3-model-only-scenario-states", number=3)


# ==============================================================================
# FIGURE 4: ZHANG ET AL. (2021) RETROSPECTIVE BENCHMARK & VOLTAGE BUDGET
# ==============================================================================
def build_figure4(t2_zhang: dict | None = None):
    configure_style()
    if t2_zhang is None:
        t2_zhang = json.loads((FIG_DIR / "tier2-zhang-hou-figure-data.json").read_text())["zhang"]

    fig = plt.figure(figsize=(7.6, 5.9))
    gs = GridSpec(
        2, 12, figure=fig,
        left=0.07, right=0.985, bottom=0.10, top=0.93,
        wspace=1.95, hspace=0.46,
        height_ratios=[1.15, 1.0]
    )

    # Panel A: Full descending 100-60 dB sequence
    axA = fig.add_subplot(gs[0, :])
    style_axis(axA, "A", "Descending 1-kHz tone sequence (100 to 60 dB SPL, four 10-s pulses per level): state carryover & drift")
    seq = t2_zhang["fig5_sequence"]
    axA.axhline(0, color=LIGHT, lw=0.75, zorder=1)

    for lvl_idx, lvl in enumerate([100, 90, 80, 70, 60]):
        block_start_s = 4.0 + 160.0 * lvl_idx
        for p_idx in range(4):
            p0 = (block_start_s + 40.0 * p_idx) / 60.0
            p1 = p0 + 10.0 / 60.0
            axA.axvspan(p0, p1, color=EXPOSURE_BG, zorder=0)
        mid_min = (block_start_s + 75.0) / 60.0
        axA.text(mid_min, 3.55, f"{lvl} dB SPL", ha="center", va="center",
                 fontsize=6.4, fontweight="bold", color=INK,
                 bbox=dict(boxstyle="round,pad=0.18", facecolor="#F1F5F9", edgecolor="#CBD5E1", lw=0.5))

    axA.plot(seq["tone_isp"]["time_min"], seq["tone_isp"]["delta_mV"],
             color=BLUE, lw=1.15, label="ISP (outcome-calibrated tone sensitivity)", zorder=3)
    axA.plot(seq["tone_ep"]["time_min"], seq["tone_ep"]["delta_mV"],
             color=GREEN, lw=1.15, label="EP (outcome-calibrated tone sensitivity)", zorder=3)
    axA.plot(seq["broadband_isp"]["time_min"], seq["broadband_isp"]["delta_mV"],
             color=INK, ls=":", lw=1.1, label="ISP (unretuned frozen broadband map)", zorder=4)
    axA.set(xlim=(0, 13.4), ylim=(-6.4, 4.2),
            xlabel="Cumulative time across descending sequence (min; gray bands mark 10-s tone bursts)",
            ylabel="Voltage change from initial state (mV)")
    axA.legend(loc="lower right", fontsize=6.0, ncol=3)

    # Panel B: Level response (Peak ΔISP vs Sound Level)
    axB = fig.add_subplot(gs[1, 0:4])
    style_axis(axB, "B", "Level dependence: peak ΔISP")
    lvls_asc = [60, 70, 80, 90, 100]
    lvls_desc = [100, 90, 80, 70, 60]
    fl = t2_zhang["fig5_levels"]
    for idx, k in enumerate(("animal_1", "animal_2", "animal_3")):
        axB.plot(lvls_asc, fl[k], color=LIGHT, lw=0.85, marker="o", ms=2.6, zorder=2,
                 label="Individual animals" if idx == 0 else None)

    means_asc = np.array([-1.028, -1.733, -2.511, -3.686, -5.110])
    sds_asc = np.array([0.642, 0.843, 1.101, 1.616, 2.208])
    axB.errorbar(
        lvls_asc, means_asc, yerr=sds_asc,
        fmt="o", ms=4.2, mfc="white", mec=ORANGE, color=ORANGE, mew=1.0,
        capsize=2.2, lw=0.95, zorder=4, label="Measured mean ± SD (n = 3)"
    )
    axB.plot(lvls_desc, fl["model_broadband_peak_isp_mV"],
             color=INK, ls=":", lw=1.3, marker="^", ms=3.2, zorder=3, label="Frozen broadband map")
    axB.plot(lvls_desc, fl["model_tone_peak_isp_mV"],
             color=BLUE, ls="-", lw=1.35, marker="s", ms=3.4, zorder=5, label="Calibrated tone drive")
    axB.set(xlim=(104, 56), xticks=[100, 90, 80, 70, 60], ylim=(-8.2, 0.5),
            xlabel="Tone level (dB SPL)", ylabel="Peak ΔISP (mV)")
    axB.legend(loc="lower right", fontsize=5.5)

    # Panel C: 90-dB 10-s pulse representative traces
    axC = fig.add_subplot(gs[1, 4:8])
    style_axis(axC, "C", "90-dB pulse: ISP & EP dynamics")
    fp = t2_zhang["fig4_pulse"]
    axC.axvspan(0, 10, color=EXPOSURE_BG, zorder=0)
    axC.axhline(0, color=LIGHT, lw=0.7, zorder=1)
    axC.plot(fp["model_isp"]["time_s"], fp["model_isp"]["delta_mV"],
             color=BLUE, lw=1.35, zorder=3, label="ISP model")
    axC.scatter(fp["obs_isp"]["time_s"], fp["obs_isp"]["delta_mV"],
                s=13, marker="o", facecolor="white", edgecolor=BLUE, linewidth=0.8, zorder=4, label="ISP data")
    axC.plot(fp["model_ep"]["time_s"], fp["model_ep"]["delta_mV"],
             color=GREEN, lw=1.35, zorder=3, label="EP model")
    axC.scatter(fp["obs_ep"]["time_s"], fp["obs_ep"]["delta_mV"],
                s=13, marker="s", facecolor="white", edgecolor=GREEN, linewidth=0.8, zorder=4, label="EP data")
    axC.text(5.0, 0.62, "10-s tone (90 dB)", ha="center", va="center", fontsize=5.6, color=SLATE, fontweight="bold")
    axC.set(xlim=(-3, 14), ylim=(-3.65, 0.85), xticks=[0, 5, 10],
            xlabel="Time from tone onset (s)", ylabel="ΔVoltage (mV)")
    axC.legend(loc="lower right", fontsize=5.4, ncol=2, columnspacing=0.6, handletextpad=0.35)

    # Panel D: ISP-EP phase trajectory + inset box
    axD = fig.add_subplot(gs[1, 8:12])
    style_axis(axD, "D", "ISP–EP loop & plateau budget")
    lp = t2_zhang["fig4_loop"]
    axD.plot([-3.8, 0.3], [-3.8, 0.3], color=LIGHT, lw=0.85, ls="--", zorder=1, label="Equality (ΔEP = ΔISP)")
    axD.plot(lp["tone_loop"]["isp_mV"], lp["tone_loop"]["ep_mV"],
             color=BLUE, lw=1.15, zorder=3, label="Tone trajectory")
    axD.plot(lp["broadband_loop"]["isp_mV"], lp["broadband_loop"]["ep_mV"],
             color=INK, ls=":", lw=1.1, zorder=3, label="Broadband trajectory")
    axD.scatter([-2.584], [-2.55], color=BLUE, marker="s", s=26, zorder=5, label="Tone plateau (−2.58, −2.55)")
    axD.scatter([-0.51], [-0.51], color=INK, marker="s", s=24, zorder=5)
    axD.errorbar(
        [-2.7], [-2.7], xerr=[0.6], yerr=[0.5],
        fmt="o", color=ORANGE, mfc="white", mec=ORANGE, mew=1.1, ms=5.0, capsize=2.5, lw=1.0, zorder=6,
        label="Cohort ± SD (n = 3)"
    )
    axD.annotate(
        "Plateau ΔISP budget:\n"
        r"• Load $-R_C\Delta I_{\mathrm{circ}}$: −2.583 mV (99.96%)" + "\n"
        r"• Battery $f_K\Delta E_{CS}$: −0.001 mV (0.04%)" + "\n"
        r"• $\Delta K_S$: $7.6\times 10^{-4}$ vs 0.74 mM data",
        xy=(-2.584, -2.55), xytext=(-3.68, -1.25),
        arrowprops=dict(arrowstyle="->", color=SLATE, lw=0.65),
        fontsize=5.0, color=INK,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#FFFBEB", edgecolor="#F59E0B", lw=0.6)
    )
    axD.set(xlim=(-3.8, 0.75), ylim=(-3.8, 0.35), xlabel="ΔISP (mV)", ylabel="ΔEP (mV)")
    axD.legend(loc="lower right", fontsize=5.0, handlelength=1.2)

    save_all_formats(fig, "figure4-zhang-retrospective-benchmark", number=4)


# ==============================================================================
# FIGURE 5: HOU ET AL. (2020) CHRONIC OBSERVATIONS VS ACUTE SUSCEPTIBILITY GRID
# ==============================================================================
def build_figure5(t2_hou: dict | None = None):
    configure_style()
    if t2_hou is None:
        t2_hou = json.loads((FIG_DIR / "tier2-zhang-hou-figure-data.json").read_text())["hou"]
    rows_csv = []
    with (FIG_DIR / "tier2-hou-endpoints.csv").open() as f:
        for r in csv.DictReader(f):
            rows_csv.append(r)

    fig = plt.figure(figsize=(7.6, 5.9))
    gs = GridSpec(
        2, 13, figure=fig,
        left=0.075, right=0.96, bottom=0.10, top=0.92,
        wspace=1.2, hspace=0.48,
        height_ratios=[1.0, 1.1]
    )

    # Panel A: Chronic observed flow-EP association
    axA = fig.add_subplot(gs[0, :])
    style_axis(axA, "A", "Hou et al. (2020) chronic empirical observations (2–5 weeks post-noise; no acute model overlay)")
    obs = t2_hou["chronic_obs"]
    flow = np.array(obs["flow_norm"])
    ep = np.array(obs["ep_norm"])
    flow_sd = flow - np.array(obs["xerr_caps"])
    ep_sd = ep - np.array(obs["yerr_caps"])

    group_colors = [BLUE, ORANGE, PURPLE, GREEN]
    group_names = [
        "1. Healthy Control\n(Flow 100%, EP 101.0 mV)",
        "2. Noise 120 dB Untreated\n(Flow 35.5%, EP 72.7 mV)",
        "3. Noise + Control Pericytes\n(Flow 71.0%, EP 86.0 mV)",
        "4. Noise + Conditioned Pericytes\n(Flow 90.3%, EP 97.8 mV)"
    ]
    offsets = [(14, -22), (0, 26), (14, -24), (-14, 18)]
    ha_list = ["left", "center", "left", "right"]

    for i in range(4):
        axA.errorbar(
            [flow[i]], [ep[i]], xerr=[flow_sd[i]], yerr=[ep_sd[i]],
            fmt="o", ms=6.5, mfc="white", mec=group_colors[i], ecolor=group_colors[i],
            mew=1.5, capsize=3.0, lw=1.1, zorder=5
        )
        axA.annotate(
            group_names[i], xy=(flow[i], ep[i]), xytext=offsets[i],
            textcoords="offset points", ha=ha_list[i], va="center",
            fontsize=6.1, fontweight="bold", color=group_colors[i],
            arrowprops=dict(arrowstyle="-", color=group_colors[i], lw=0.6, alpha=0.6),
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=group_colors[i], alpha=0.95, lw=0.7)
        )

    axA.set(xlim=(0.15, 1.32), ylim=(0.62, 1.12),
            xlabel="Normalized cochlear capillary volumetric flow (relative to healthy control mean ± SD, n = 6)",
            ylabel="Normalized EP (relative to control ± SD, n = 3)")

    # Panel B: Acute second-day EP trajectories
    axB = fig.add_subplot(gs[1, 0:5])
    style_axis(axB, "B", "Acute Day-2 EP (120 dB, 180 min)")
    axB.axvspan(0, 180, color=EXPOSURE_BG, zorder=0)
    axB.axvline(180, color=MUTED, lw=0.65, ls="--", zorder=1)
    tr = t2_hou["acute_trajectories"]
    for key, col, lbl in (
        ("reference", BLUE, "Reference health (80.0 mV)"),
        ("moderate_deficit", ORANGE, "Moderate deficit (77.9 mV)"),
        ("larger_deficit", GREEN, "Larger deficit (72.3 mV)"),
    ):
        axB.plot(tr[key]["time_min"], tr[key]["ep_norm"], color=col, lw=1.45, label=lbl, zorder=3)
    axB.text(90, 1.01, "Noise (180 min)", ha="center", va="center", fontsize=5.8, color=SLATE, fontweight="bold")
    axB.text(242, 1.01, "Recovery", ha="center", va="center", fontsize=5.8, color=SLATE)
    axB.set(xlim=(0, 300), ylim=(0.48, 1.06),
            xlabel="Time from Day-2 onset (min)",
            ylabel="EP / healthy quiet EP (80 mV)")
    axB.legend(loc="lower right", fontsize=5.4)

    # Panel C: 3x3 Acute susceptibility grid
    axC = fig.add_subplot(gs[1, 7:13])
    style_axis(axC, "C", "3×3 acute EP loss grid: ΔEP vs. matched quiet (mV)", grid=False)
    health_order = ["Reference", "Moderate deficit", "Larger deficit"]
    susc_order = ["Lower susceptibility", "Reference susceptibility", "Higher susceptibility"]
    loss_grid = np.zeros((3, 3))
    abs_ep_grid = np.zeros((3, 3))

    for i, h in enumerate(health_order):
        for j, s in enumerate(susc_order):
            for r in rows_csv:
                if r["health"] == h and r["susceptibility"] == s and float(r["time_min"]) == 1620.0:
                    loss_grid[i, j] = float(r["noise_minus_quiet_ep_mV"])
                    abs_ep_grid[i, j] = float(r["ep_mV"])

    im = axC.imshow(-loss_grid, cmap="YlGnBu", vmin=14, vmax=37, aspect="auto")
    for i in range(3):
        for j in range(3):
            txt_col = "white" if -loss_grid[i, j] > 25.5 else INK
            axC.text(
                j, i, f"{loss_grid[i, j]:.1f} mV\n(EP: {abs_ep_grid[i, j]:.1f})",
                ha="center", va="center", color=txt_col, fontsize=6.1, fontweight="bold"
            )
    axC.set(
        xticks=[0, 1, 2],
        xticklabels=[
            r"Lower" + "\n" + r"($\rho_W=0.1, \tau_R=48\mathrm{m}$)",
            r"Reference" + "\n" + r"($\rho_W=0.2, \tau_R=95\mathrm{m}$)",
            r"Higher" + "\n" + r"($\rho_W=0.4, \tau_R=190\mathrm{m}$)"
        ],
        yticks=[0, 1, 2],
        yticklabels=[
            "Reference\n(80.0 mV quiet)",
            "Moderate deficit\n(77.9 mV quiet)",
            "Larger deficit\n(72.3 mV quiet)"
        ],
        xlabel="Assumed acute trauma susceptibility bundle",
        ylabel="Assumed baseline vascular health"
    )
    axC.tick_params(axis="both", labelsize=5.5)
    cbar = fig.colorbar(im, ax=axC, fraction=0.046, pad=0.035)
    cbar.set_label("|ΔEP| loss vs. matched quiet (mV)", fontsize=5.8)
    cbar.ax.tick_params(labelsize=5.5)

    save_all_formats(fig, "figure5-hou-vascular-trauma-exploration", number=5)


# ==============================================================================
# GRAPHICAL ABSTRACT / ARCHITECTURAL SCHEMATIC
# ==============================================================================
def build_graphical_abstract():
    from build_anatomical_infographic import build_anatomical_infographic
    build_anatomical_infographic()


def main():
    build_figure1()
    build_figure2()
    build_figure3()
    build_figure4()
    build_figure5()
    build_graphical_abstract()
    print("All figures (Figures 1-5 and Graphical Abstract) rebuilt in figures/!")


if __name__ == "__main__":
    main()
