#!/usr/bin/env python3
"""Build a compact overview of the model's compartments and potassium cycle."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


FIG_DIR = Path(__file__).resolve().parent
INK = "#172033"
MUTED = "#536174"
GREEN = "#00875A"
BLUE = "#0052CC"
PALE = "#F5F7FA"
PALE_BLUE = "#F2F7FC"


def add_box(ax, xy, width, height, title, detail, *, dashed=False):
    edge = BLUE if dashed else "#657286"
    face = PALE_BLUE if dashed else PALE
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.25,rounding_size=0.8",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.0,
        linestyle="--" if dashed else "-",
        zorder=2,
    )
    ax.add_patch(box)
    x, y = xy
    ax.text(
        x + width / 2,
        y + height * 0.72,
        title,
        ha="center",
        va="center",
        color=INK,
        fontsize=7.6,
        fontweight="bold",
        zorder=3,
    )
    ax.text(
        x + width / 2,
        y + height * 0.27,
        detail,
        ha="center",
        va="center",
        color=INK,
        fontsize=7.8,
        linespacing=1.12,
        zorder=3,
    )


def add_arrow(ax, start, end, *, color=GREEN, dashed=False, width=1.5, scale=8):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=scale,
            linewidth=width,
            linestyle="--" if dashed else "-",
            color=color,
            shrinkA=1.5,
            shrinkB=1.5,
            zorder=4,
        )
    )


def build_anatomical_overview():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "mathtext.fontset": "dejavusans",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")

    ax.text(
        50,
        59,
        "Integrated cochlear potassium-recycling framework",
        ha="center",
        va="center",
        color=INK,
        fontsize=9.0,
        fontweight="bold",
    )

    compartments = (
        ((2, 34), 14, "Endolymph", r"$K_E$; EP $V$"),
        ((19, 34), 14, "Hair cells", "MET / P2X2"),
        ((36, 34), 14, "Perilymph", r"$K^+$ return"),
        ((53, 34), 16, "Fibrocyte\nsyncytium", r"$K_C$; ATP $A_C$"),
        ((72, 34), 12, "Intrastrial\nspace", r"$K_S$; $v_S$"),
        ((87, 34), 11, "Marginal\ncells", r"$K_M$; ATP $A_M$"),
    )
    for (xy, width, title, detail) in compartments:
        add_box(ax, xy, width, 11, title, detail)

    centers = (9, 26, 43, 61, 78, 92.5)
    for left, right in zip(centers, centers[1:]):
        add_arrow(ax, (left + 7, 39.5), (right - 7, 39.5))

    for x, label in zip(
        (17.5, 34.5, 51.5, 70, 85.5),
        ("MET", r"$K^+$", r"$K^+$ uptake", r"$J_{\mathrm{Kir}}$", "NKCC1"),
    ):
        ax.text(x, 31.7, label, ha="center", va="center", color=MUTED, fontsize=7.8)

    ax.plot([9, 9], [45, 50], color=GREEN, linewidth=1.4, zorder=3)
    ax.plot([92.5, 92.5], [45, 50], color=GREEN, linewidth=1.4, zorder=3)
    add_arrow(ax, (92.5, 50), (9, 50), width=1.4, scale=7)
    ax.text(
        50,
        53,
        "Apical KCNQ1 secretion returns K$^+$ to endolymph",
        ha="center",
        va="center",
        color=GREEN,
        fontsize=7.8,
        fontweight="bold",
    )

    add_box(
        ax,
        (2, 8),
        29,
        13,
        "Microvascular energetics",
        r"Perfusion $F$ $\rightarrow$ oxygen $O$" "\n" r"$\rightarrow$ ATP pools $A_C$, $A_M$",
        dashed=True,
    )
    add_box(
        ax,
        (35.5, 8),
        29,
        13,
        "Acoustic loading",
        r"$q(L)$ loads MET / P2X2; ATP $a$ drives $\alpha$" "\n"
        r"Trauma $W$ alters $J_{\mathrm{para}}$ and flow",
        dashed=True,
    )
    add_box(
        ax,
        (69, 8),
        29,
        13,
        "Drug and boundary closures",
        r"$D(t)$ inhibits NKCC1; $J_{\mathrm{peri}}$ is gated" "\n"
        r"Reference blood K$^+$ boundary: $J_B=0$",
        dashed=True,
    )

    ax.plot([3, 8], [3.5, 3.5], color=GREEN, linewidth=1.6)
    add_arrow(ax, (7, 3.5), (8, 3.5), width=1.6, scale=7)
    ax.text(
        9,
        3.5,
        "Established potassium-recycling relationships",
        ha="left",
        va="center",
        color=INK,
        fontsize=7.8,
    )
    ax.add_patch(
        FancyBboxPatch(
            (54, 2.4),
            5,
            2.2,
            boxstyle="round,pad=0.12",
            facecolor=PALE_BLUE,
            edgecolor=BLUE,
            linewidth=0.9,
            linestyle="--",
        )
    )
    ax.text(
        61,
        3.5,
        "Dashed boxes: model inputs and assumptions",
        ha="left",
        va="center",
        color=INK,
        fontsize=7.8,
    )

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.02)
    fig.canvas.draw()
    fig.savefig(FIG_DIR / "anatomical-infographic.pdf", bbox_inches="tight", pad_inches=0.04)
    svg_path = FIG_DIR / "anatomical-infographic.svg"
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.04)
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8",
    )
    fig.savefig(
        FIG_DIR / "anatomical-infographic.png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    fig.savefig(
        FIG_DIR / "anatomical-infographic.tiff",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.04,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)
    print(f"Saved compact anatomical overview to {FIG_DIR / 'anatomical-infographic.pdf'}")


if __name__ == "__main__":
    build_anatomical_overview()
