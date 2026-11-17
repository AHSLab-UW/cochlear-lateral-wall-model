#!/usr/bin/env python3
"""Build a realistic, textbook-grade vector anatomical cross-section & biophysical
infographic of the mammalian cochlear duct, lateral wall, and organ of Corti.

Features:
  - True cochlear duct cross-section (no top title banner for clean journal/graphical use):
    Bony Otic Capsule (with osteon lamellae), Spiral Ligament (collagen matrix +
    Fibrocytes I-V), Reissner's Membrane, Scala Vestibuli, Scala Media, Spiral Limbus
    (with interdental cells), striated Tectorial Membrane (with embedded OHC stereocilia
    tips), Organ of Corti (IHC, arched Inner/Outer Pillar Cells / Tunnel of Corti,
    3 tilted OHCs in cup-shaped Deiters' cells with basolateral KCNQ4/BK efflux,
    Hensen's/Claudius' cells), Basilar Membrane, and Scala Tympani.
  - Detailed Stria Vascularis: 2-layer flattened Basal Cells (Claudin-11 & Cx26/30),
    dendritic Intermediate Cell melanocyte (Kir4.1 + melanin granules), Strial
    Capillary cross-section with RBCs, endothelium, basement membrane, pericyte,
    and PVM/M, and Marginal Cells with deeply folded basolateral membranes,
    cristae-filled mitochondria, NKCC1/Na-K-ATPase, and KCNQ1/KCNE1.
  - Explicit visual distinction (dashed arrows & dashed-border callouts + style key)
    between established anatomy/K+ transport and modeled boundary assumptions or
    proposed perturbation pathways (J_B = 0, [N3] Claudin-11 leak, [N4] vasoconstriction,
    [F4] perilymphatic sink).

Outputs (PDF, SVG, 600-dpi PNG & TIFF):
  - figures/anatomical-infographic.*
  - figures/graphical-abstract.*
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import (
    FancyBboxPatch,
    FancyArrowPatch,
    Polygon,
    Ellipse,
    Circle,
    PathPatch,
)
from matplotlib.path import Path as MplPath
import numpy as np

FIG_DIR = Path(__file__).resolve().parent

INK = "#0F172A"
SLATE = "#334155"
MUTED = "#64748B"

K_GREEN = "#00875A"
DRUG_RED = "#D55E00"
DRUG_LIGHT = "#FFF7ED"
NOISE_BLUE = "#0052CC"
TRAUMA_MAG = "#9D174D"

BONE_BG = "#E2E8F0"
LIGAMENT_BG = "#F3E8FF"
FIBRO_FILL = "#DDD6FE"
FIBRO_EDGE = "#6D28D9"
BC_FILL = "#E7D5BE"
BC_EDGE = "#78350F"
IC_FILL = "#D6B48C"
IC_EDGE = "#7C2D12"
IS_YELLOW = "#FEF9C3"
MC_PINK = "#FCE7F3"
MC_EDGE = "#9D174D"
ENDO_BLUE = "#E0F2FE"
SV_GRAY = "#F8FAFC"
PERI_GRAY = "#F1F5F9"
HC_LAV = "#E9D5FF"
HC_EDGE = "#581C87"
TEC_BG = "#DBEAFE"
PILLAR_FILL = "#E2E8F0"
PILLAR_EDGE = "#475569"


def configure_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "mathtext.fontset": "dejavusans",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def smooth_closed_blob(ax, pts, *, fc, ec, lw=1.0, zorder=4, alpha=1.0):
    """Draw a smooth organic cell contour through control points using Catmull-Rom-like Bezier."""
    pts = np.asarray(pts, dtype=float)
    n = len(pts)
    verts = []
    codes = []
    for i in range(n):
        p0 = pts[(i - 1) % n]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        p3 = pts[(i + 2) % n]
        c1 = p1 + (p2 - p0) / 6.0
        c2 = p2 - (p3 - p1) / 6.0
        if i == 0:
            verts.append(p1)
            codes.append(MplPath.MOVETO)
        verts.extend([c1, c2, p2])
        codes.extend([MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
    verts.append(pts[0])
    codes.append(MplPath.CLOSEPOLY)
    patch = PathPatch(MplPath(verts, codes), facecolor=fc, edgecolor=ec, lw=lw, zorder=zorder, alpha=alpha)
    ax.add_patch(patch)
    return patch


def draw_stellate_fibrocyte(ax, cx, cy, rx, ry, angle_deg, label=None, *, fc=FIBRO_FILL, ec=FIBRO_EDGE, zorder=3):
    """Draw a branched/spindle fibrocyte with realistic cytoplasmic processes."""
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    radii_scale = np.array([1.28, 0.52, 0.96, 0.48, 1.32, 0.52, 0.92, 0.48] * 2)
    th = np.deg2rad(angle_deg)
    pts = []
    for a, s in zip(angles, radii_scale):
        x0 = rx * s * np.cos(a)
        y0 = ry * s * np.sin(a)
        xr = cx + x0 * np.cos(th) - y0 * np.sin(th)
        yr = cy + x0 * np.sin(th) + y0 * np.cos(th)
        pts.append((xr, yr))
    smooth_closed_blob(ax, pts, fc=fc, ec=ec, lw=0.85, zorder=zorder)
    ax.add_patch(Ellipse((cx, cy), width=rx * 0.55, height=ry * 0.45, angle=angle_deg,
                         facecolor="#A78BFA", edgecolor=ec, lw=0.55, zorder=zorder + 0.1))
    if label:
        ax.text(cx, cy, label, ha="center", va="center", color=INK, fontsize=5.0, fontweight="bold", zorder=zorder + 0.2)


def draw_mitochondrion(ax, cx, cy, w=1.45, h=0.68, angle=0, zorder=6):
    ax.add_patch(Ellipse((cx, cy), width=w, height=h, angle=angle,
                         facecolor="#FCA5A5", edgecolor="#991B1B", lw=0.65, zorder=zorder))
    th = np.deg2rad(angle)
    xs = np.linspace(-w * 0.32, w * 0.32, 5)
    ys = np.array([0.0, h * 0.22, -h * 0.22, h * 0.22, 0.0])
    xr = cx + xs * np.cos(th) - ys * np.sin(th)
    yr = cy + xs * np.sin(th) + ys * np.cos(th)
    ax.plot(xr, yr, color="#991B1B", lw=0.55, zorder=zorder + 0.1)


def draw_tight_junction(ax, x, y, width=1.6, *, horiz=True, color="#1E3A8A", zorder=9):
    if horiz:
        for dx in np.linspace(-width / 2, width / 2, 5):
            ax.plot([x + dx, x + dx], [y - 0.38, y + 0.38], color=color, lw=1.25, zorder=zorder)
        ax.plot([x - width / 2 - 0.15, x + width / 2 + 0.15], [y, y], color=color, lw=0.85, zorder=zorder)
    else:
        for dy in np.linspace(-width / 2, width / 2, 5):
            ax.plot([x - 0.38, x + 0.38], [y + dy, y + dy], color=color, lw=1.25, zorder=zorder)
        ax.plot([x, x], [y - width / 2 - 0.15, y + width / 2 + 0.15], color=color, lw=0.85, zorder=zorder)


def draw_gap_junction(ax, x, y, zorder=9):
    for dy in (-0.32, 0.0, 0.32):
        ax.plot([x - 0.5, x + 0.5], [y + dy, y + dy], color="#0F766E", lw=1.3, zorder=zorder)


def draw_channel(ax, x, y, *, horiz=True, fc="#10B981", ec="#065F46", scale=1.0, zorder=10):
    if horiz:
        w, h = 1.8 * scale, 0.88 * scale
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.08",
                                    facecolor=fc, edgecolor=ec, lw=1.0, zorder=zorder))
        ax.plot([x - w * 0.58, x + w * 0.58], [y, y], color="white", lw=1.2 * scale, zorder=zorder + 0.2)
    else:
        w, h = 0.88 * scale, 1.7 * scale
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.08",
                                    facecolor=fc, edgecolor=ec, lw=1.0, zorder=zorder))
        ax.plot([x, x], [y - h * 0.58, y + h * 0.58], color="white", lw=1.2 * scale, zorder=zorder + 0.2)


def draw_arrow(ax, p1, p2, *, color=K_GREEN, lw=2.4, connectionstyle="arc3,rad=0.0",
               ls="-", head_w=0.40, head_l=0.52, mscale=12.5, zorder=13):
    """Draw a sleek, high-contrast vector arrow with a crisp white halo and well-proportioned arrowhead."""
    patch = FancyArrowPatch(
        p1, p2,
        connectionstyle=connectionstyle,
        arrowstyle=f"-|>,head_width={head_w},head_length={head_l}",
        mutation_scale=mscale,
        color=color,
        lw=lw,
        linestyle=ls,
        zorder=zorder,
    )
    patch.set_path_effects([
        pe.Stroke(linewidth=lw + 1.9, foreground="white", alpha=0.94),
        pe.Normal(),
    ])
    ax.add_patch(patch)
    return patch


def add_pill(ax, x, y, text, *, fg="white", bg=INK, border="none", ls="-", fontsize=5.6, pad=0.25, zorder=16, ha="center"):
    ax.text(
        x, y, text, ha=ha, va="center", color=fg, fontsize=fontsize, fontweight="bold", zorder=zorder,
        bbox=dict(boxstyle=f"round,pad={pad}", facecolor=bg, edgecolor=border,
                  linestyle=ls, lw=1.05 if border != "none" else 0.0)
    )


def build_anatomical_infographic():
    configure_style()
    fig, ax = plt.subplots(figsize=(13.8, 7.4))
    ax.set_xlim(0, 138)
    ax.set_ylim(3.5, 72.0)
    ax.axis("off")

    # ==========================================================================
    # 1. COCHLEAR DUCT CROSS-SECTIONAL REGIONS
    # ==========================================================================
    bone_pts = [(0.8, 7.5), (5.2, 7.5), (4.6, 28.0), (4.2, 48.0), (5.0, 70.5), (0.8, 70.5)]
    ax.add_patch(Polygon(bone_pts, closed=True, facecolor=BONE_BG, edgecolor="#64748B", lw=1.1, zorder=0))
    for oy in np.linspace(11.0, 67.0, 9):
        ox = 2.8 + 0.35 * np.sin(oy * 0.2)
        ax.add_patch(Circle((ox, oy), radius=1.25, facecolor="none", edgecolor="#94A3B8", lw=0.6, alpha=0.75, zorder=0.5))
        ax.add_patch(Circle((ox, oy), radius=0.55, facecolor="none", edgecolor="#94A3B8", lw=0.5, alpha=0.75, zorder=0.5))
        ax.add_patch(Circle((ox, oy), radius=0.18, facecolor="#64748B", alpha=0.8, zorder=0.6))
    ax.text(2.6, 39.0, "OTIC CAPSULE (BONE)", rotation=90, ha="center", va="center",
            color=SLATE, fontsize=6.9, fontweight="bold", zorder=2,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#94A3B8", lw=0.6, alpha=0.92))

    lig_pts = [(4.8, 7.5), (27.2, 7.5), (27.2, 62.5), (26.8, 70.5), (5.0, 70.5), (4.2, 48.0), (4.6, 28.0)]
    ax.add_patch(Polygon(lig_pts, closed=True, facecolor=LIGAMENT_BG, edgecolor="#94A3B8", lw=0.9, zorder=0))
    xs_w = np.linspace(5.5, 26.2, 24)
    for y_w in np.linspace(10.5, 67.0, 14):
        ax.plot(xs_w, y_w + 0.55 * np.sin(xs_w * 0.45 + y_w), color="#C4B5FD", lw=0.6, alpha=0.45, zorder=0.5)

    sv_pts = [(27.2, 61.5), (112.5, 68.5), (112.5, 70.5), (27.2, 70.5)]
    ax.add_patch(Polygon(sv_pts, closed=True, facecolor=SV_GRAY, edgecolor="#CBD5E1", lw=0.8, zorder=0))

    sm_pts = [(67.5, 16.5), (67.5, 64.5), (112.5, 68.5), (112.5, 16.5)]
    ax.add_patch(Polygon(sm_pts, closed=True, facecolor=ENDO_BLUE, edgecolor="#94A3B8", lw=0.9, zorder=0))

    ax.add_patch(FancyBboxPatch((32.2, 17.5), 18.0, 44.5, boxstyle="round,pad=0.15",
                                facecolor=IS_YELLOW, edgecolor="#CBD5E1", lw=0.8, zorder=0))
    ax.add_patch(FancyBboxPatch((50.2, 17.5), 17.3, 44.5, boxstyle="round,pad=0.15",
                                facecolor="#FFF5F7", edgecolor="#CBD5E1", lw=0.8, zorder=0))

    rm_x = np.array([27.2, 112.5])
    rm_y = np.array([61.8, 68.4])
    ax.fill_between(rm_x, rm_y, rm_y + 0.85, color="#E2E8F0", zorder=3)
    ax.plot(rm_x, rm_y, color="#0284C7", lw=2.0, zorder=4)
    ax.plot(rm_x, rm_y + 0.85, color="#64748B", lw=0.95, zorder=4)
    ax.text(72.0, 69.2, "Scala Vestibuli (Perilymph)", ha="center", va="center", color=SLATE, fontsize=7.0, fontweight="bold", zorder=5)
    ax.text(85.0, 65.2, "Reissner's Membrane (2-cell tight barrier)", rotation=4.4, ha="center", va="center",
            color="#0369A1", fontsize=5.9, fontweight="bold", zorder=5)

    ax.text(16.0, 68.2, "Spiral Ligament\n(Fibrocytes I–V)", ha="center", va="center", color=INK, fontsize=8.3, fontweight="bold", zorder=5)
    ax.text(41.2, 59.8, "Intrastrial Space (IS)", ha="center", va="center", color=INK, fontsize=7.2, fontweight="bold", zorder=5)
    ax.text(59.0, 60.8, "Marginal Cells (MC)", ha="center", va="center", color=INK, fontsize=7.3, fontweight="bold", zorder=5)

    # ==========================================================================
    # 2. SPIRAL LIGAMENT STELLATE FIBROCYTES (TYPES I-V) & SUBSYSTEMS I/II
    # ==========================================================================
    draw_stellate_fibrocyte(ax, 12.2, 61.5, 3.0, 1.9, -15, "V", zorder=3)
    draw_stellate_fibrocyte(ax, 19.2, 62.2, 3.0, 1.9, 10, "V", zorder=3)
    ax.text(15.7, 64.8, "Type V (nNOS)", ha="center", va="center", color=NOISE_BLUE, fontsize=5.4, fontweight="bold")

    draw_stellate_fibrocyte(ax, 6.8, 53.5, 1.8, 4.4, 4, "III", zorder=3)
    draw_stellate_fibrocyte(ax, 6.6, 39.5, 1.8, 4.6, -3, "III", zorder=3)
    draw_stellate_fibrocyte(ax, 6.8, 25.5, 1.8, 4.4, 5, "III", zorder=3)

    draw_stellate_fibrocyte(ax, 23.6, 52.5, 1.8, 4.0, 2, "I", zorder=3)
    draw_stellate_fibrocyte(ax, 23.6, 42.0, 1.8, 4.0, -4, "I", zorder=3)
    draw_stellate_fibrocyte(ax, 23.6, 31.5, 1.8, 4.0, 3, "I", zorder=3)

    draw_stellate_fibrocyte(ax, 14.5, 14.2, 3.3, 1.9, -12, "II", zorder=3)
    draw_stellate_fibrocyte(ax, 22.8, 13.8, 3.3, 1.9, 8, "IV", zorder=3)
    ax.text(18.6, 17.8, "Type II/IV Fibrocytes\n(Perilymph K+ Uptake)", ha="center", va="center",
            color=FIBRO_EDGE, fontsize=5.4, fontweight="bold")

    ax.add_patch(FancyBboxPatch((9.8, 43.2), 11.4, 12.2, boxstyle="round,pad=0.25",
                                facecolor="white", edgecolor=FIBRO_EDGE, lw=1.1, alpha=0.96, zorder=6))
    ax.text(15.5, 53.6, "Syncytium (Sub. II)", ha="center", va="center", color=FIBRO_EDGE, fontsize=6.2, fontweight="bold", zorder=7)
    ax.text(15.5, 48.2, r"$K_C = 140\ \mathrm{mM}$" + "\n" + r"ATP $A_C = 1.0$" + "\n" + r"$\phi_C = 0.40\ V_{E,0}$" + "\n" + r"Cx26/30 + $U_C$",
            ha="center", va="center", color=INK, fontsize=5.55, zorder=7)

    ax.add_patch(FancyBboxPatch((9.8, 25.8), 11.4, 13.5, boxstyle="round,pad=0.25",
                                facecolor="#EFF6FF", edgecolor=NOISE_BLUE, lw=1.0, alpha=0.96, zorder=6))
    ax.text(15.5, 37.6, "Vascular & O2 (Sub. I)", ha="center", va="center", color=NOISE_BLUE, fontsize=6.0, fontweight="bold", zorder=7)
    ax.text(15.5, 31.6, r"Flow $F(t)$ ($\tau_F=1.5\mathrm{m}$)" + "\n" + r"Oxygen $O(t)$ ($0.07\ \mathrm{mM}$)" + "\n" + r"Dilation $+\rho\Delta\bar{A}$" + "\n" + r"Constr. $-\rho_W W$",
            ha="center", va="center", color=INK, fontsize=5.4, zorder=7)
    # Modeled noise-induced vasoconstriction (dashed border + dashed arrow)
    add_pill(ax, 15.5, 22.8, r"[N4] $-\rho_W W \Rightarrow F\downarrow$", fg=TRAUMA_MAG, bg="#FDF2F8",
             border=TRAUMA_MAG, ls="--", fontsize=5.1)
    draw_arrow(ax, (15.5, 23.8), (15.5, 25.7), color=TRAUMA_MAG, lw=1.5, ls="--", head_w=0.34, head_l=0.45, mscale=10.5)

    # ==========================================================================
    # 3. STRIA VASCULARIS: 2-LAYER FLATTENED BASAL CELLS, DENDRITIC IC, CAPILLARY
    # ==========================================================================
    for y_b in np.linspace(22.5, 54.5, 6):
        pts1 = [(26.9, y_b - 3.0), (28.7, y_b - 2.6), (29.0, y_b), (28.7, y_b + 2.6), (26.9, y_b + 3.0), (26.6, y_b)]
        smooth_closed_blob(ax, pts1, fc=BC_FILL, ec=BC_EDGE, lw=0.95, zorder=4)
        ax.add_patch(Ellipse((27.8, y_b), width=0.8, height=2.2, facecolor="#A16207", alpha=0.55, zorder=4.2))
        draw_gap_junction(ax, 26.6, y_b, zorder=9)
    for y_b in np.linspace(25.5, 51.5, 5):
        pts2 = [(29.0, y_b - 3.0), (31.0, y_b - 2.5), (31.3, y_b), (31.0, y_b + 2.5), (29.0, y_b + 3.0), (28.7, y_b)]
        smooth_closed_blob(ax, pts2, fc="#DEC5A4", ec=BC_EDGE, lw=0.95, zorder=4.1)
        ax.add_patch(Ellipse((30.0, y_b), width=0.8, height=2.1, facecolor="#A16207", alpha=0.55, zorder=4.3))
        draw_tight_junction(ax, 29.0, y_b - 2.5, width=1.4, zorder=9)
        draw_tight_junction(ax, 30.8, y_b + 2.4, width=1.2, zorder=9)

    ax.text(29.0, 59.5, "Basal Cells\n(Claudin-11)", ha="center", va="center", color=BC_EDGE, fontsize=5.8, fontweight="bold", zorder=6)

    ic_pts = [
        (31.2, 39.2), (34.4, 40.3), (38.8, 43.0), (39.5, 41.7), (35.8, 38.5),
        (37.0, 35.6), (41.6, 35.6), (41.8, 34.0), (37.0, 33.2), (36.2, 29.2),
        (40.2, 26.8), (39.5, 25.4), (34.5, 27.2), (31.2, 26.4)
    ]
    smooth_closed_blob(ax, ic_pts, fc=IC_FILL, ec=IC_EDGE, lw=1.1, zorder=4)
    ax.add_patch(Ellipse((33.8, 33.2), width=2.4, height=3.8, facecolor="#78350F", alpha=0.45, zorder=4.2))
    for mx, my in ((33.0, 36.6), (34.8, 37.9), (33.2, 29.5), (35.0, 28.2), (36.2, 34.8)):
        ax.add_patch(Circle((mx, my), radius=0.3, facecolor="#451A03", zorder=4.3))
    ax.text(35.0, 44.6, "Intermediate Cell\n(Melanocyte)", ha="center", va="center", color=IC_EDGE, fontsize=5.4, fontweight="bold", zorder=6)

    draw_channel(ax, 39.1, 42.3, horiz=True, fc="#10B981", ec="#065F46", zorder=10)
    draw_channel(ax, 41.7, 34.8, horiz=True, fc="#10B981", ec="#065F46", zorder=10)
    ax.text(43.0, 36.8, "Kir4.1\n(Kcnj10)", ha="left", va="center", color="#065F46", fontsize=5.8, fontweight="bold", zorder=10)

    cx_c, cy_c = 44.2, 25.5
    ax.add_patch(Circle((cx_c, cy_c), radius=3.5, facecolor="#FDE68A", edgecolor="#92400E", lw=0.9, ls=":", zorder=4.5))
    ax.add_patch(Circle((cx_c, cy_c), radius=2.7, facecolor="#FCA5A5", edgecolor="#991B1B", lw=1.15, zorder=5))
    ax.add_patch(Circle((cx_c, cy_c), radius=1.85, facecolor="#FEE2E2", edgecolor="#DC2626", lw=0.8, zorder=5.2))
    ax.add_patch(Ellipse((cx_c - 0.6, cy_c + 0.3), width=1.1, height=0.65, angle=25, facecolor="#EF4444", edgecolor="#991B1B", lw=0.5, zorder=5.4))
    ax.add_patch(Ellipse((cx_c + 0.55, cy_c - 0.35), width=1.1, height=0.65, angle=-20, facecolor="#EF4444", edgecolor="#991B1B", lw=0.5, zorder=5.4))
    draw_tight_junction(ax, cx_c, cy_c + 2.3, width=0.9, horiz=False, color="#991B1B", zorder=6)
    draw_tight_junction(ax, cx_c, cy_c - 2.3, width=0.9, horiz=False, color="#991B1B", zorder=6)
    ax.add_patch(Ellipse((46.8, 27.7), width=2.8, height=1.3, angle=-42, facecolor="#93C5FD", edgecolor="#1D4ED8", lw=0.9, zorder=6))
    ax.text(47.6, 29.7, "Pericyte", ha="center", va="center", color="#1E3A8A", fontsize=5.2, fontweight="bold", zorder=7)
    ax.add_patch(Ellipse((41.0, 23.3), width=2.5, height=1.2, angle=-40, facecolor="#A16207", edgecolor="#451A03", lw=0.8, zorder=6))
    ax.text(37.8, 22.3, "PVM/M", ha="center", va="center", color="#451A03", fontsize=5.0, fontweight="bold", zorder=7)
    ax.text(44.2, 20.6, "Strial Capillary (BLB: Claudin-5, ZO-1)", ha="center", va="center", color="#991B1B", fontsize=5.05, fontweight="bold", zorder=7)
    # Explicit model boundary assumption tag (dashed border)
    add_pill(ax, 44.2, 18.7, r"Model assumption: $J_B = 0$", fg=SLATE, bg="white", border=SLATE, ls="--", fontsize=4.75, pad=0.2)

    ax.add_patch(FancyBboxPatch((36.3, 46.8), 12.8, 11.2, boxstyle="round,pad=0.25",
                                facecolor="white", edgecolor="#CA8A04", lw=1.05, alpha=0.96, zorder=6))
    ax.text(42.7, 56.2, "IS Microdomain", ha="center", va="center", color="#854D0E", fontsize=6.3, fontweight="bold", zorder=7)
    ax.text(42.7, 51.0, r"$K_S = 1.5\ \mathrm{mM}$" + "\n" + r"Vol $v_S = 0.035$" + "\n" + r"$V_{\mathrm{ISP}} = +80\ \mathrm{mV}$",
            ha="center", va="center", color=INK, fontsize=5.65, zorder=7)
    add_pill(ax, 43.2, 43.8, r"[F2] $v_S\uparrow$ (Edema)", fg=DRUG_RED, bg=DRUG_LIGHT, border=DRUG_RED, fontsize=5.05)

    ax.add_patch(FancyBboxPatch((32.8, 11.5), 16.6, 5.2, boxstyle="round,pad=0.22",
                                facecolor="white", edgecolor="#047857", lw=1.0, zorder=6))
    ax.text(41.1, 14.1, "Battery #1 (Kir4.1):\n" + r"$E_{CS} = V_T\ln(K_C/K_S) \approx +121.1\ \mathrm{mV}$",
            ha="center", va="center", color="#047857", fontsize=5.2, fontweight="bold", zorder=7)

    # ==========================================================================
    # 4. STRIAL MARGINAL CELLS WITH CONTINUOUS UNDULATING BASOLATERAL INFOLDINGS
    # ==========================================================================
    cell_ys = [(49.0, 59.0), (38.5, 48.5), (28.0, 38.0), (18.0, 27.5)]
    for idx, (y0, y1) in enumerate(cell_ys):
        pleat_ys = np.linspace(y0 + 0.6, y1 - 0.6, 9)
        mc_pts = [(53.2, y0), (66.7, y0), (66.7, y1), (53.2, y1)]
        for k, py in enumerate(reversed(pleat_ys)):
            px = 49.5 if k % 2 == 0 else 52.7
            mc_pts.append((px, py))
        smooth_closed_blob(ax, mc_pts, fc=MC_PINK, ec=MC_EDGE, lw=1.05, zorder=4)

        for ym in (y0 + 2.0, y1 - 2.0):
            draw_mitochondrion(ax, 54.0, ym, w=1.55, h=0.7, angle=12, zorder=5)
        for ym in np.linspace(y0 + 0.7, y1 - 0.7, 7):
            ax.plot([66.7, 68.1], [ym, ym], color=MC_EDGE, lw=0.85, zorder=4.2)
        if idx > 0:
            draw_tight_junction(ax, 66.7, y1 + 0.25, width=1.4, horiz=False, color=MC_EDGE, zorder=9)
        if idx in (1, 3):
            ax.add_patch(Ellipse((63.5, (y0 + y1) / 2), width=3.0, height=2.2,
                                 facecolor="#F9A8D4", edgecolor=MC_EDGE, lw=0.75, zorder=4.3))

    ax.text(59.8, 54.0, r"$K_M = 145\ \mathrm{mM}$" + "\n" + r"ATP $A_M = 1.0$" + "\n" + r"($\phi_M = 0.042$)",
            ha="center", va="center", color=INK, fontsize=5.65, fontweight="bold", zorder=6)
    ax.text(56.2, 49.8, "Mitochondria", ha="left", va="center", color="#991B1B", fontsize=4.8, fontweight="bold", zorder=6)

    draw_channel(ax, 49.8, 43.5, horiz=True, fc="#FB923C", ec=DRUG_RED, zorder=10)
    ax.text(52.8, 43.5, "NKCC1", ha="left", va="center", color=DRUG_RED, fontsize=6.0, fontweight="bold", zorder=10)
    add_pill(ax, 59.6, 46.3, r"[F1] $h_D\downarrow \Rightarrow J_U\downarrow$", fg=DRUG_RED, bg=DRUG_LIGHT, border=DRUG_RED, fontsize=5.05)

    draw_channel(ax, 49.8, 33.0, horiz=True, fc="#FBBF24", ec="#B45309", zorder=10)
    ax.text(52.8, 33.0, r"$\mathrm{Na}^+/\mathrm{K}^+$" + "\nATPase", ha="left", va="center", color="#92400E", fontsize=5.8, fontweight="bold", zorder=10)

    draw_channel(ax, 66.8, 43.5, horiz=True, fc="#10B981", ec="#065F46", zorder=10)
    ax.text(65.8, 46.2, "KCNQ1/\nKCNE1", ha="right", va="bottom", color="#065F46", fontsize=5.5, fontweight="bold", zorder=10)

    ax.add_patch(FancyBboxPatch((51.2, 11.5), 16.4, 5.2, boxstyle="round,pad=0.22",
                                facecolor="white", edgecolor=MC_EDGE, lw=0.95, zorder=6))
    ax.text(59.4, 14.1, "Battery #2 (Apical KCNQ1):\n" + r"$E_{ME} = V_T\ln(K_M/K_E) \approx -0.9\ \mathrm{mV}$",
            ha="center", va="center", color=INK, fontsize=5.15, fontweight="bold", zorder=7)
    add_pill(ax, 59.4, 20.0, r"[F3] $g_S\downarrow \Rightarrow R/R_0\approx 2.15$", fg=DRUG_RED, bg=DRUG_LIGHT, border=DRUG_RED, fontsize=5.05)

    # ==========================================================================
    # 5. SCALA MEDIA, SPIRAL LIMBUS, TECTORIAL MEMBRANE & ORGAN OF CORTI
    # ==========================================================================
    limbus_pts = [(104.2, 16.5), (112.5, 16.5), (112.5, 38.2), (107.8, 37.6), (105.0, 33.2), (104.2, 27.5)]
    smooth_closed_blob(ax, limbus_pts, fc="#E2E8F0", ec="#64748B", lw=1.0, zorder=3)
    for y_id in (33.5, 35.0, 36.5):
        ax.add_patch(Ellipse((106.5, y_id), width=1.4, height=0.8, angle=25, facecolor="#CBD5E1", edgecolor="#64748B", lw=0.6, zorder=3.5))
    ax.text(109.2, 24.5, "Spiral\nLimbus", ha="center", va="center", color=SLATE, fontsize=5.2, fontweight="bold", zorder=4)

    ax.add_patch(FancyBboxPatch((71.2, 47.0), 36.8, 12.8, boxstyle="round,pad=0.3",
                                facecolor="white", edgecolor="#0284C7", lw=1.25, alpha=0.96, zorder=5))
    ax.text(89.6, 57.8, "Scala Media Endolymph (Subsystems III & IV)", ha="center", va="center",
            color="#0369A1", fontsize=7.1, fontweight="bold", zorder=6)
    ax.text(
        89.6, 51.8,
        r"Endocochlear Potential: $\mathbf{EP\ (V) = +80\ mV}$ ($C_V \dot{V} = g_S(V_S - V) - I_{\mathrm{circ}}$)" + "\n"
        r"Derived $\mathrm{K}^+$: $\mathbf{K_E = 150\ mM}$,  Inventory: $\mathbf{N_K = 212.14\ mM\cdot V_{E,0}}$" + "\n"
        r"Bulk ATP: $\mathbf{a = 7.4\ nM}$ $\rightarrow$ Subtectorial Cleft ATP $a_r$ $\rightarrow$ P2X2 Shunt $g_P(a_r)$",
        ha="center", va="center", color=INK, fontsize=5.75, zorder=6
    )

    # Curved Tectorial Membrane (slightly translucent so embedded OHC stereocilia tips are clearly visible inside it)
    tec_pts = [(107.2, 36.8), (92.0, 36.2), (77.0, 33.0), (74.3, 30.6), (77.2, 29.3), (82.2, 29.6), (87.2, 30.0), (91.5, 30.7), (107.2, 33.4)]
    smooth_closed_blob(ax, tec_pts, fc=TEC_BG, ec="#3B82F6", lw=1.2, zorder=7, alpha=0.86)
    for xf in np.linspace(76.5, 105.0, 12):
        ax.plot([xf, xf + 1.2], [30.6 + 0.08 * (xf - 76), 33.4 + 0.1 * (xf - 76)], color="#93C5FD", lw=0.6, alpha=0.6, zorder=7.2)
    ax.text(88.0, 34.0, "Tectorial Membrane", ha="center", va="center", color="#1E40AF", fontsize=5.6, fontweight="bold", zorder=8)
    ax.text(88.0, 31.8, r"Subtectorial Cleft: $a_r = a_0 + c_a\Delta a$", ha="center", va="center", color="#1E3A8A", fontsize=5.0, fontweight="bold", zorder=8)

    # Purely anatomical Basilar Membrane & Scala Tympani label
    bm_x = np.linspace(68.0, 108.0, 60)
    bm_y = 14.8 + 1.05 * np.sin((bm_x - 68.0) / 40.0 * np.pi)
    ax.fill_between(bm_x, 7.5, bm_y, color=PERI_GRAY, zorder=1)
    ax.fill_between(bm_x, bm_y - 0.5, bm_y, color="#CBD5E1", zorder=3.5)
    ax.plot(bm_x, bm_y, color="#334155", lw=1.8, zorder=4)
    ax.text(88.5, 11.7, "Basilar Membrane  |  Scala Tympani (Perilymph)",
            ha="center", va="center", color=SLATE, fontsize=5.8, fontweight="bold", zorder=5)

    for xh, yh_top in [(69.4, 21.5), (71.6, 24.4), (73.8, 26.8)]:
        hc_pts = [(xh - 1.0, 15.6), (xh + 1.0, 15.6), (xh + 1.05, yh_top - 0.5), (xh, yh_top + 0.3), (xh - 1.05, yh_top - 0.5)]
        smooth_closed_blob(ax, hc_pts, fc="#E2E8F0", ec="#64748B", lw=0.85, zorder=4)

    # 3 Slightly Tilted Cylindrical Outer Hair Cells with stereocilia tips embedded into the Tectorial Membrane
    for x_ohc in (77.2, 82.2, 87.2):
        dc_pts = [(x_ohc - 1.7, 15.6), (x_ohc + 1.7, 15.6), (x_ohc + 1.6, 19.6), (x_ohc, 18.8), (x_ohc - 1.6, 19.6)]
        smooth_closed_blob(ax, dc_pts, fc="#CBD5E1", ec="#475569", lw=0.85, zorder=4)
        ohc_pts = [(x_ohc - 1.35, 19.2), (x_ohc + 1.35, 19.2), (x_ohc + 1.6, 27.0), (x_ohc - 1.1, 27.0)]
        smooth_closed_blob(ax, ohc_pts, fc=HC_LAV, ec=HC_EDGE, lw=1.05, zorder=5)
        ax.add_patch(Circle((x_ohc, 20.8), radius=0.75, facecolor="#A855F7", alpha=0.55, zorder=5.2))
        for dx_s in (-0.7, -0.35, 0.0, 0.35, 0.7):
            y_tip = 30.25 + 0.075 * (x_ohc - 77.2) + abs(dx_s) * 0.2
            ax.plot([x_ohc + dx_s + 0.2, x_ohc + dx_s + 0.25], [27.0, y_tip], color=HC_EDGE, lw=1.0, zorder=6.8)
        # Basolateral KCNQ4/BK channel glyph on each OHC basolateral membrane
        draw_channel(ax, x_ohc + 1.15, 22.0, horiz=True, fc="#10B981", ec="#065F46", scale=0.55, zorder=10)

    # Arched Inner & Outer Pillar Cells forming the Tunnel of Corti
    tunnel_outer = Polygon([[90.0, 15.6], [92.8, 27.0], [94.3, 27.0], [97.0, 15.6]], closed=True,
                           facecolor=PILLAR_FILL, edgecolor=PILLAR_EDGE, lw=1.0, zorder=4)
    ax.add_patch(tunnel_outer)
    tunnel_inner = Polygon([[91.3, 15.7], [93.6, 25.4], [95.7, 15.7]], closed=True,
                           facecolor="white", edgecolor=PILLAR_EDGE, lw=0.9, zorder=4.2)
    ax.add_patch(tunnel_inner)
    ax.text(93.5, 18.4, "Tunnel\nof\nCorti", ha="center", va="center", color=MUTED, fontsize=4.65, fontweight="bold", zorder=5)

    # Flask-shaped Inner Hair Cell (IHC — stereocilia remain free-standing in subtectorial fluid)
    ihc_pts = [(98.8, 17.5), (102.2, 17.5), (102.5, 22.0), (101.6, 27.0), (99.4, 27.0), (98.5, 22.0)]
    smooth_closed_blob(ax, ihc_pts, fc="#DDD6FE", ec=HC_EDGE, lw=1.1, zorder=5)
    ax.add_patch(Circle((100.5, 20.5), radius=0.9, facecolor="#7C3AED", alpha=0.5, zorder=5.2))
    for dx_s in (-0.8, -0.4, 0.0, 0.4, 0.8):
        ax.plot([100.5 + dx_s, 100.5 + dx_s], [27.0, 29.4 + abs(dx_s) * 0.3], color=HC_EDGE, lw=0.95, zorder=6)
    draw_channel(ax, 102.1, 21.5, horiz=True, fc="#10B981", ec="#065F46", scale=0.55, zorder=10)

    # Anchor KCNQ4 / BK efflux callout directly to the hair cells' basolateral membranes (fits cleanly left of P2X2 arrow at x=96.5)
    ax.annotate(
        "KCNQ4 / BK\n(K+ → perilymph)",
        xy=(88.8, 22.0), xytext=(89.3, 24.4),
        ha="left", va="center", color="#065F46", fontsize=4.65, fontweight="bold", zorder=12,
        bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="#10B981", lw=0.75, alpha=0.95),
        arrowprops=dict(arrowstyle="-|>,head_width=0.24,head_length=0.34", color="#065F46", lw=0.9)
    )

    ax.text(82.2, 13.7, "3 OHCs + Deiters' Cells", ha="center", va="center", color=HC_EDGE, fontsize=5.2, fontweight="bold", zorder=6)
    ax.text(100.5, 13.7, "IHC", ha="center", va="center", color=HC_EDGE, fontsize=5.4, fontweight="bold", zorder=6)

    ax.plot([73.5, 104.2], [27.1, 27.1], color="#0F172A", lw=1.6, zorder=7)
    ax.text(74.8, 28.0, "Reticular Lamina", ha="center", va="bottom", color=INK, fontsize=5.0, fontweight="bold", zorder=8)

    draw_channel(ax, 79.6, 27.1, horiz=False, fc="#10B981", ec="#065F46", zorder=10)
    add_pill(ax, 79.6, 40.4, "MET ($g_{\\mathrm{MET}}m,\\,J_{\\mathrm{MET}}$)", fg="#065F46", bg="white", border="#10B981", fontsize=5.4)
    add_pill(ax, 79.6, 43.8, r"[N1] $m\uparrow \Rightarrow I_{\mathrm{circ}}\uparrow$", fg=NOISE_BLUE, bg="#EFF6FF", border=NOISE_BLUE, fontsize=5.1)

    draw_channel(ax, 96.5, 27.1, horiz=False, fc="#F472B6", ec=TRAUMA_MAG, zorder=10)
    add_pill(ax, 96.5, 40.4, "P2X2 ($g_P(a_r),\\,J_{\\mathrm{P2X}}$)", fg=TRAUMA_MAG, bg="white", border=TRAUMA_MAG, fontsize=5.4)
    add_pill(ax, 96.5, 43.8, r"[N2] $a\uparrow \Rightarrow g_P\uparrow,\,\alpha\uparrow$", fg=NOISE_BLUE, bg="#EFF6FF", border=NOISE_BLUE, fontsize=5.1)

    # ==========================================================================
    # 6. HIGH-CONTRAST CLOCKWISE K+ RECYCLING ARROWS & MODELED PATHWAYS
    # ==========================================================================
    draw_arrow(ax, (35.2, 34.8), (41.4, 34.8), color=K_GREEN, lw=2.4, head_w=0.40, head_l=0.52, mscale=12.5)
    add_pill(ax, 38.4, 31.8, r"$J_{\mathrm{Kir}}$", bg=K_GREEN, fontsize=6.0)

    draw_arrow(ax, (46.6, 43.5), (52.4, 43.5), color=K_GREEN, lw=2.4, head_w=0.40, head_l=0.52, mscale=12.5)
    add_pill(ax, 46.8, 40.5, r"$J_U$", bg=K_GREEN, fontsize=6.0)

    draw_arrow(ax, (64.2, 43.5), (70.4, 43.5), color=K_GREEN, lw=2.4, head_w=0.40, head_l=0.52, mscale=12.5)
    add_pill(ax, 68.8, 46.2, r"$J_A$", bg=K_GREEN, fontsize=6.0)

    draw_arrow(ax, (20.5, 62.8), (34.5, 62.8), color=K_GREEN, lw=2.5, head_w=0.40, head_l=0.52, mscale=12.5)
    draw_arrow(ax, (68.2, 62.0), (109.6, 39.2), color=K_GREEN, lw=2.5,
               connectionstyle="angle,angleA=0,angleB=-90,rad=5", head_w=0.42, head_l=0.55, mscale=13.0)

    draw_arrow(ax, (79.6, 38.0), (79.6, 21.8), color=K_GREEN, lw=2.2, head_w=0.38, head_l=0.50, mscale=12.0, zorder=11)
    draw_arrow(ax, (96.5, 38.0), (96.5, 21.8), color=TRAUMA_MAG, lw=2.1, head_w=0.38, head_l=0.50, mscale=12.0, zorder=11)

    draw_arrow(ax, (105.5, 9.2), (11.2, 25.4), color=K_GREEN, lw=2.5,
               connectionstyle="angle,angleA=180,angleB=90,rad=6", head_w=0.42, head_l=0.55, mscale=13.0)
    add_pill(ax, 59.0, 9.2,
             r"Closed-Loop $\mathrm{K}^+$ Recycling: Endolymph $\rightarrow$ Hair Cells ($J_{\mathrm{MET}}+J_{\mathrm{P2X}}$) $\rightarrow$ Perilymph $\rightarrow$ Type II/IV Fibrocytes $\rightarrow$ Syncytium ($K_C$)",
             bg=K_GREEN, fontsize=5.85, pad=0.28)

    # Dashed-border modeled/proposed pathways ([N3], [F4])
    add_pill(ax, 38.2, 39.2, r"[N3] $J_{\mathrm{para}}(\ell_W W)$", fg=TRAUMA_MAG, bg="#FDF2F8",
             border=TRAUMA_MAG, ls="--", fontsize=5.05)
    draw_arrow(ax, (29.8, 39.2), (34.4, 39.2), color=TRAUMA_MAG, lw=1.8, ls="--", head_w=0.36, head_l=0.48, mscale=11.0)

    add_pill(ax, 74.3, 21.8, r"[F4] $J_{\mathrm{peri}}$ ($\mathbf{1}_{\mathrm{drug}}$)", fg=DRUG_RED, bg=DRUG_LIGHT,
             border=DRUG_RED, ls="--", fontsize=5.05)
    draw_arrow(ax, (71.5, 46.2), (69.6, 12.8), color=DRUG_RED, lw=1.8, ls="--",
               connectionstyle="arc3,rad=0.12", head_w=0.36, head_l=0.48, mscale=11.0)

    # ==========================================================================
    # 7. RIGHT-SIDE PERTURBATION PATHWAY CARDS (X = 114.0..136.5)
    # ==========================================================================
    ax.add_patch(FancyBboxPatch((114.0, 42.5), 22.5, 27.5, boxstyle="round,pad=0.35",
                                facecolor=DRUG_LIGHT, edgecolor=DRUG_RED, lw=1.6, zorder=7))
    add_pill(ax, 125.25, 67.6, "FUROSEMIDE PATHWAY", bg=DRUG_RED, fontsize=6.9, pad=0.32)
    furo_text = (
        r"Input: $D(t) \rightarrow h_D = (1+D^{n_D})^{-1}$" + "\n\n"
        r"[F1] Blocks NKCC1 ($h_D\downarrow \Rightarrow J_U\downarrow$)" + "\n"
        r"     $\rightarrow K_S\uparrow \Rightarrow E_{CS}\downarrow$ & $A_M\uparrow$" + "\n\n"
        r"[F2] Drives osmotic IS edema ($v_S\uparrow$)" + "\n\n"
        r"[F3] Collapses source $g_S$ ($\ell_D=0.949$)" + "\n"
        r"     $\rightarrow R/R_0\approx 2.15$, EP $\rightarrow -33\ \mathrm{mV}$" + "\n\n"
        r"[F4] Gates perilymphatic sink $J_{\mathrm{peri}}$" + "\n"
        r"     $\rightarrow K_E\downarrow$ by $16–38\ \mathrm{mM}$"
    )
    ax.text(115.0, 54.6, furo_text, ha="left", va="center", color=INK, fontsize=5.65, zorder=8)

    ax.add_patch(FancyBboxPatch((114.0, 8.8), 22.5, 31.8, boxstyle="round,pad=0.35",
                                facecolor="#FDF2F8", edgecolor=NOISE_BLUE, lw=1.6, zorder=7))
    add_pill(ax, 125.25, 38.4, "NOISE EXPOSURE PATHWAY", bg=NOISE_BLUE, fontsize=6.6, pad=0.32)
    noise_text = (
        r"Input: $q(L),\, e = \max(0, q-1)$" + "\n\n"
        r"[N1] Elevates MET load ($m\uparrow \Rightarrow I_{\mathrm{circ}}\uparrow$)" + "\n"
        r"     $\rightarrow$ resistive $\Delta V_{\mathrm{ISP}},\,\Delta\mathrm{EP}$" + "\n\n"
        r"[N2] Evokes ATP ($a\uparrow \Rightarrow a_r\uparrow$)" + "\n"
        r"     $\rightarrow$ protective P2X2 shunt ($g_P\uparrow$)" + "\n"
        r"     & transducer adaptation ($\alpha\uparrow$)" + "\n\n"
        r"[N3] Modeled trauma ($W\uparrow >115\ \mathrm{dB}$):" + "\n"
        r"     Claudin-11 leak ($J_{\mathrm{para}} = \ell_W W\Delta K$)" + "\n\n"
        r"[N4] Modeled vasoconstriction" + "\n"
        r"     ($-\rho_W W \Rightarrow F\downarrow,\, O\downarrow$)"
    )
    ax.text(115.0, 23.2, noise_text, ha="left", va="center", color=INK, fontsize=5.65, zorder=8)

    # ==========================================================================
    # 8. COMPACT BOTTOM STYLE KEY (DISTINGUISHES ANATOMY VS. MODEL ASSUMPTIONS)
    # ==========================================================================
    ax.add_patch(FancyBboxPatch((0.8, 4.3), 135.7, 2.2, boxstyle="round,pad=0.15",
                                facecolor="#F8FAFC", edgecolor="#CBD5E1", lw=0.85, zorder=5))
    ax.plot([3.0, 6.2], [5.4, 5.4], color=K_GREEN, lw=2.3, zorder=6)
    ax.text(7.0, 5.4, r"Solid green arrows: Physiological $\mathrm{K}^+$ recycling", ha="left", va="center", color=INK, fontsize=5.4, fontweight="bold", zorder=6)

    ax.plot([36.0, 39.2], [5.4, 5.4], color=TRAUMA_MAG, lw=1.9, ls="--", zorder=6)
    ax.add_patch(FancyBboxPatch((40.0, 4.8), 3.6, 1.2, boxstyle="round,pad=0.05",
                                facecolor="white", edgecolor=SLATE, ls="--", lw=0.95, zorder=6))
    ax.text(44.4, 5.4, r"Dashed arrows / dashed-border boxes: Model assumptions & proposed perturbation pathways ($J_B=0$, [N3], [N4], [F4])",
            ha="left", va="center", color=INK, fontsize=5.4, fontweight="bold", zorder=6)

    ax.text(135.2, 5.4, r"[F1–F4] Furosemide   |   [N1–N4] Noise", ha="right", va="center", color=SLATE, fontsize=5.3, fontweight="bold", zorder=6)

    # Save PDF, SVG, and 600-dpi PNG/TIFF
    fig.canvas.draw()
    for stem in ("anatomical-infographic", "graphical-abstract"):
        fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.06)
        fig.savefig(FIG_DIR / f"{stem}.svg", bbox_inches="tight", pad_inches=0.06)
        fig.savefig(FIG_DIR / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.06)
        fig.savefig(FIG_DIR / f"{stem}.tiff", dpi=600, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print("Saved realistic vector anatomical infographic to figures/anatomical-infographic.* and figures/graphical-abstract.*")


if __name__ == "__main__":
    build_anatomical_infographic()
