#!/usr/bin/env python3
"""Figures 4--5: Zhang et al. (2021) tone-level benchmark and Hou et al. (2020) vascular/trauma grid."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))

from rebuild_all_figures import build_figure4, build_figure5, configure_style


def main():
    parser = argparse.ArgumentParser(
        description="Build the manuscript's Tier 2 benchmark figures (Figures 4 and 5)."
    )
    parser.add_argument(
        "--figures",
        nargs="+",
        choices=("4", "5"),
        default=("4", "5"),
        help="Figures to export; defaults to both 4 and 5.",
    )
    selected_figures = set(parser.parse_args().figures)
    configure_style()
    if "4" in selected_figures:
        build_figure4()
    if "5" in selected_figures:
        build_figure5()
    print(f"Built Tier 2 figure(s) {sorted(selected_figures)} in figures/.")


if __name__ == "__main__":
    main()
