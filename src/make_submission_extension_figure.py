"""Create the submission-package supplementary robustness figure with Python."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "submission_package_2026-08-26" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size": 7.2,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
})

BLUE = "#0F4D92"
TEAL = "#42949E"
RED = "#B64342"
GREY = "#767676"
LIGHT = "#D8D8D8"
TEXT = "#272727"
DATASETS = ["mimic_temporal_test", "eicu_external", "sicdb_external"]
LABELS = ["MIMIC-IV\n2020-2022", "eICU-CRD", "SICdb"]


def label_panel(ax: plt.Axes, label: str) -> None:
    ax.text(-0.16, 1.03, label, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom", color=TEXT)


def save(fig: plt.Figure) -> None:
    target = FIGURES / "Figure_S3_methodological_extensions"
    fig.savefig(target.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(target.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(target.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def panel_action_gap(ax: plt.Axes) -> None:
    gap = pd.read_csv(OUTPUTS / "corrected_one_hour_gap_metrics.csv")
    primary = gap[gap.model.eq("primary_six_hour_hgb_on_gap_cohort")].set_index("dataset")
    trained = gap[gap.model.eq("one_hour_gap_trained_hgb")].set_index("dataset")
    x = np.arange(3)
    for i, dataset in enumerate(DATASETS):
        ax.plot([i - 0.12, i + 0.12], [primary.loc[dataset, "auroc"], trained.loc[dataset, "auroc"]],
                color=LIGHT, linewidth=1.1, zorder=1)
        ax.errorbar(i - 0.12, primary.loc[dataset, "auroc"],
                    yerr=[[primary.loc[dataset, "auroc"] - primary.loc[dataset, "auroc_ci_low"]],
                          [primary.loc[dataset, "auroc_ci_high"] - primary.loc[dataset, "auroc"]]],
                    fmt="s", color=BLUE, ecolor=BLUE, ms=4, capsize=2, linewidth=1.0, zorder=3)
        ax.errorbar(i + 0.12, trained.loc[dataset, "auroc"],
                    yerr=[[trained.loc[dataset, "auroc"] - trained.loc[dataset, "auroc_ci_low"]],
                          [trained.loc[dataset, "auroc_ci_high"] - trained.loc[dataset, "auroc"]]],
                    fmt="o", color=RED, ecolor=RED, ms=4, capsize=2, linewidth=1.0, zorder=3)
        ax.text(i + 0.19, trained.loc[dataset, "auroc"], f"{trained.loc[dataset, 'auroc']:.3f}",
                va="center", fontsize=6.5, color=RED)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_xlim(-0.42, 2.48)
    ax.set_ylim(0.62, 0.88)
    ax.set_ylabel("AUROC")
    ax.set_title("Actionability gap", fontsize=8, pad=5)
    ax.grid(axis="y", color=LIGHT, linewidth=0.45)
    ax.set_axisbelow(True)
    label_panel(ax, "a")
    handles = [
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=BLUE,
                   markeredgecolor=BLUE, markersize=4, label="Primary 6-h target"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=RED,
                   markeredgecolor=RED, markersize=4, label="1-h action gap"),
    ]
    ax.legend(handles=handles, fontsize=6.1, loc="lower left", bbox_to_anchor=(-0.02, 0.01),
              ncol=2, columnspacing=0.8, handletextpad=0.3)


def panel_estimands(ax: plt.Axes) -> None:
    primary = pd.read_csv(OUTPUTS / "corrected_frozen_validation_metrics.csv")
    primary = primary[primary.model.eq("hist_gradient_boosting")].set_index("dataset")
    balanced = pd.read_csv(OUTPUTS / "corrected_stay_balanced_metrics.csv").set_index("dataset")
    fixed = pd.read_csv(OUTPUTS / "corrected_fixed_hour_metrics.csv")
    fixed = fixed[fixed.index_hour.eq(6)].set_index("dataset")
    gap = pd.read_csv(OUTPUTS / "corrected_one_hour_gap_metrics.csv")
    gap = gap[gap.model.eq("one_hour_gap_trained_hgb")].set_index("dataset")
    estimates = [primary.auroc, balanced.auroc, fixed.auroc, gap.auroc]
    names = ["Primary", "Stay-balanced", "Hour 6", "1-h action gap"]
    colors = [BLUE, TEAL, GREY, RED]
    offsets = np.linspace(-0.24, 0.24, 4)
    for idx, dataset in enumerate(DATASETS):
        for values, name, color, offset in zip(estimates, names, colors, offsets):
            value = float(values.loc[dataset])
            ax.scatter(idx + offset, value, s=26, marker="o", color=color, zorder=3)
        ax.plot([idx - 0.24, idx + 0.24],
                [float(estimates[0].loc[dataset]), float(estimates[-1].loc[dataset])],
                color=LIGHT, linewidth=0.7, zorder=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels(LABELS)
    ax.set_ylim(0.62, 0.90)
    ax.set_ylabel("AUROC")
    ax.set_title("Alternative estimands", fontsize=8, pad=5)
    ax.grid(axis="y", color=LIGHT, linewidth=0.45)
    ax.set_axisbelow(True)
    label_panel(ax, "b")
    handles = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=c,
                          markeredgecolor="none", markersize=4.5, label=n)
               for c, n in zip(colors, names)]
    ax.legend(handles=handles, fontsize=6.1, loc="lower left", bbox_to_anchor=(-0.02, 0.01),
              ncol=2, columnspacing=0.8, handletextpad=0.3)


def panel_recalibration(ax: plt.Axes) -> None:
    data = pd.read_csv(OUTPUTS / "corrected_repeated_recalibration.csv")
    x = np.arange(3)
    for i, dataset in enumerate(DATASETS):
        vals = data.loc[data.dataset.eq(dataset), "brier_change_after_minus_before"].to_numpy()
        parts = ax.violinplot(vals, positions=[i], widths=0.55, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(TEAL)
            body.set_edgecolor(TEAL)
            body.set_alpha(0.38)
        ax.boxplot(vals, positions=[i], widths=0.19, showfliers=False, patch_artist=True,
                   boxprops={"facecolor": "white", "edgecolor": TEAL, "linewidth": 0.8},
                   medianprops={"color": TEAL, "linewidth": 1.0},
                   whiskerprops={"color": TEAL, "linewidth": 0.8},
                   capprops={"color": TEAL, "linewidth": 0.8})
    ax.axhline(0, color=TEXT, linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("Brier change\n(after − before)")
    ax.set_title("Repeated intercept recalibration", fontsize=8, pad=5)
    ax.grid(axis="y", color=LIGHT, linewidth=0.45)
    ax.set_axisbelow(True)
    label_panel(ax, "c")


def panel_hospital_calibration(ax: plt.Axes) -> None:
    data = pd.read_csv(OUTPUTS / "corrected_eicu_hospital_metrics.csv")
    data = data.sort_values("calibration_slope")
    ax.scatter(data.calibration_slope, data.calibration_in_the_large,
               s=18, color=BLUE, alpha=0.75, edgecolors="none")
    ax.axvline(1, color=GREY, linestyle="--", linewidth=0.8)
    ax.axhline(0, color=GREY, linestyle="--", linewidth=0.8)
    ax.set_xlabel("Hospital calibration slope")
    ax.set_ylabel("Hospital CITL")
    ax.set_title("eICU hospital calibration", fontsize=8, pad=5)
    ax.set_xlim(0.60, 1.75)
    ax.set_ylim(-1.10, 1.65)
    ax.grid(color=LIGHT, linewidth=0.45)
    ax.set_axisbelow(True)
    label_panel(ax, "d")
    ax.text(0.03, 0.96, "73 hospitals with adequate event counts", transform=ax.transAxes,
            va="top", fontsize=6.3, color=TEXT)


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.1))
    panel_action_gap(axes[0, 0])
    panel_estimands(axes[0, 1])
    panel_recalibration(axes[1, 0])
    panel_hospital_calibration(axes[1, 1])
    fig.subplots_adjust(left=0.10, right=0.98, top=0.94, bottom=0.12, wspace=0.38, hspace=0.48)
    save(fig)


if __name__ == "__main__":
    main()
