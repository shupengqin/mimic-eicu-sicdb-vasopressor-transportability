"""Redraw the main Figure 2 by combining study design and validation discrimination.

The figure uses only disclosure-reviewed aggregate CSV files. It does not load
patient-level data or alter any reported estimates.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve()
DEFAULT_PACKAGE = HERE.parents[1] / "submission_package_2026-08-26"
DEFAULT_DATA = HERE.parents[1] / "outputs"
DATA = DEFAULT_DATA
FIGURES = DEFAULT_PACKAGE / "figures"

BLUE = "#17639A"
BLUE_LIGHT = "#E5F0F7"
TEAL = "#3B8D8B"
TEAL_LIGHT = "#E6F2F0"
GREY = "#6B6F72"
GREY_LIGHT = "#F1F3F4"
RED = "#B44E4A"
TEXT = "#20252A"
GRID = "#D9DEE2"
DATASETS = ["mimic_temporal_test", "eicu_external", "sicdb_external"]
DATASET_LABELS = ["MIMIC-IV\n2020-2022", "eICU-CRD", "SICdb"]

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 8.0,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "legend.frameon": False,
    }
)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.10, 1.06, label, transform=ax.transAxes, fontsize=10, fontweight="bold", color=TEXT, va="bottom")


def box(ax: plt.Axes, xy: tuple[float, float], width: float, height: float, title: str, lines: list[str],
        fill: str, edge: str, title_color: str = TEXT) -> None:
    x, y = xy
    patch = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.012,rounding_size=0.018",
                           facecolor=fill, edgecolor=edge, linewidth=1.1)
    ax.add_patch(patch)
    ax.text(x + 0.018, y + height - 0.035, title, fontsize=8.4, fontweight="bold", color=title_color,
            va="top", ha="left")
    for i, line in enumerate(lines):
        ax.text(x + 0.018, y + height - 0.085 - 0.052 * i, line, fontsize=7.1,
                color=GREY if i else TEXT, va="top", ha="left")


def draw_design(ax: plt.Axes, selection: pd.DataFrame, cohort: pd.DataFrame) -> None:
    ax.set_axis_off()
    # A compact left-to-right pipeline makes the estimand and freeze point explicit.
    box(ax, (0.01, 0.22), 0.19, 0.56, "Development", ["MIMIC-IV 2008-2016", "50,200 stays", "Model fitting"], BLUE_LIGHT, BLUE)
    box(ax, (0.25, 0.22), 0.22, 0.56, "Model selection", ["MIMIC-IV 2017-2019", "11,997 stays", "HGB: AUROC 0.841", "LR: AUROC 0.822"], TEAL_LIGHT, TEAL)
    box(ax, (0.52, 0.22), 0.19, 0.56, "Frozen models", ["Refit on 2008-2019", "HGB selected", "No validation data"], "white", BLUE)
    cohort = cohort.set_index("dataset")
    outputs = [(0.80, 0.62, "MIMIC-IV 2020-2022", f"{int(cohort.loc['mimic_temporal_test_2020_2022', 'n_stays']):,} stays", BLUE_LIGHT),
               (0.80, 0.42, "eICU-CRD", f"{int(cohort.loc['eicu_external', 'n_stays']):,} stays", "white"),
               (0.80, 0.22, "SICdb", f"{int(cohort.loc['sicdb_external', 'n_stays']):,} stays", "white")]
    for x, y, title, detail, fill in outputs:
        box(ax, (x, y), 0.18, 0.13, title, [detail], fill, BLUE)
    for start, end in [(0.20, 0.25), (0.47, 0.52)]:
        ax.add_patch(FancyArrowPatch((start, 0.50), (end, 0.50), arrowstyle="-|>", mutation_scale=11,
                                     linewidth=1.0, color=GREY))
    for y in (0.685, 0.485, 0.285):
        ax.add_patch(FancyArrowPatch((0.71, 0.50), (0.80, y), arrowstyle="-|>", mutation_scale=10,
                                     linewidth=0.9, color=BLUE))
    ax.text(0.50, 0.10, "Frozen models were scored without refitting in temporal or external cohorts.",
            fontsize=6.8, color=TEXT, ha="center", va="center")
    ax.text(0.50, 0.045, "Outcome: first documented continuous vasopressor initiation within 6 h.", fontsize=6.8,
            color=TEXT, ha="center", va="center")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 0.88)


def forest(ax: plt.Axes, data: pd.DataFrame, metric: str, xlim: tuple[float, float], xlabel: str, label: str) -> None:
    y = np.arange(3)[::-1]
    offsets = {"logistic_regression": -0.105, "hist_gradient_boosting": 0.105}
    styles = {
        "logistic_regression": (GREY, "o", "Logistic regression"),
        "hist_gradient_boosting": (BLUE, "s", "HGB"),
    }
    for model, (color, marker, _) in styles.items():
        part = data[data.model.eq(model)].set_index("dataset").loc[DATASETS]
        estimate = part[metric].to_numpy(dtype=float)
        low = part[f"{metric}_ci_low"].to_numpy(dtype=float)
        high = part[f"{metric}_ci_high"].to_numpy(dtype=float)
        ax.errorbar(estimate, y + offsets[model], xerr=[estimate - low, high - estimate], fmt=marker,
                    color=color, ecolor=color, elinewidth=1.25, capsize=2.7, ms=4.8, markeredgewidth=0.6)
        for yi, value, upper in zip(y + offsets[model], estimate, high):
            fmt = f"{value:.3f}"
            delta = (xlim[1] - xlim[0]) * 0.018
            if upper + delta < xlim[1] - delta * 0.2:
                ax.text(upper + delta, yi, fmt, fontsize=6.6, color=color, va="center", ha="left")
            else:
                ax.text(value - delta, yi, fmt, fontsize=6.6, color=color, va="center", ha="right")
    ax.set_yticks(y)
    ax.set_yticklabels(DATASET_LABELS)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color=GRID, linewidth=0.55)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=3)
    panel_label(ax, label)


def save(fig: plt.Figure) -> None:
    for stem in ("Figure_2", "Figure_2_combined_design_discrimination"):
        target = FIGURES / stem
        fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
        fig.savefig(target.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
        fig.savefig(target.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
        fig.savefig(target.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")


def main() -> None:
    global DATA, FIGURES
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    DATA = args.data.resolve()
    FIGURES = (args.package.resolve() / "figures")
    FIGURES.mkdir(parents=True, exist_ok=True)
    selection = pd.read_csv(DATA / "corrected_model_selection_metrics.csv")
    ci = pd.read_csv(DATA / "corrected_clustered_confidence_intervals.csv")
    cohort = pd.read_csv(DATA / "corrected_table1_cohort_summary.csv")
    # The tight bounding box includes the panel labels and long y-axis labels.
    # A slightly narrower source canvas keeps the final exported width within a
    # standard double-column figure width after that bounding-box expansion.
    fig = plt.figure(figsize=(7.0, 5.52))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.08, 1.0], hspace=0.60, wspace=0.42)
    ax_design = fig.add_subplot(grid[0, :])
    draw_design(ax_design, selection, cohort)
    panel_label(ax_design, "a")
    ax_auroc = fig.add_subplot(grid[1, 0])
    forest(ax_auroc, ci, "auroc", (0.60, 0.88), "AUROC", "b")
    ax_auprc = fig.add_subplot(grid[1, 1])
    forest(ax_auprc, ci, "auprc", (0.00, 0.18), "AUPRC", "c")
    handles = [Line2D([0], [0], marker="o", color=GREY, markerfacecolor=GREY, linewidth=1.1, markersize=4.5,
                      label="Logistic regression"),
               Line2D([0], [0], marker="s", color=BLUE, markerfacecolor=BLUE, linewidth=1.1, markersize=4.5,
                      label="HGB")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=2, fontsize=7.3,
               handletextpad=0.45, columnspacing=1.4)
    fig.text(0.5, 0.037, "Points show estimates; bars show 95% CIs from 500 stay-clustered bootstrap replicates.",
             ha="center", va="bottom", fontsize=6.8, color=GREY)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.96, bottom=0.14)
    save(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()
