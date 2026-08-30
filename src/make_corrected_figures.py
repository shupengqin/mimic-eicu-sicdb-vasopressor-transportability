"""Create publication figures for the corrected transportability analysis."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd
from pathlib import Path


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})
plt.rcParams["font.size"] = 7.5
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["legend.frameon"] = False
plt.rcParams["xtick.major.width"] = 0.7
plt.rcParams["ytick.major.width"] = 0.7


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

COLORS = {
    "logistic": "#767676",
    "hgb": "#0F4D92",
    "hgb_light": "#8DB3D9",
    "recal": "#42949E",
    "uncal": "#767676",
    "accent": "#B64342",
    "grid": "#D8D8D8",
    "text": "#272727",
    "selection": "#DDEAF6",
    "development": "#E8E8E8",
    "test": "#F6CFCB",
}

DATASET_LABELS = {
    "mimic_temporal_test": "MIMIC-IV\n2020-2022",
    "eicu_external": "eICU-CRD",
    "sicdb_external": "SICdb",
}


def panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES / f"{name}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / f"{name}.tiff", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def study_design(ax: plt.Axes) -> None:
    ax.set_axis_off()
    cohort_path = OUTPUTS / "corrected_table1_cohort_summary.csv"
    cohort = pd.read_csv(cohort_path).set_index("dataset") if cohort_path.exists() else None
    eicu_stays = int(cohort.loc["eicu_external", "n_stays"]) if cohort is not None else 157765
    panel_label(ax, "a", x=-0.03, y=1.00)
    x_positions = [0.02, 0.29]
    widths = [0.23, 0.21]
    colors = [COLORS["development"], COLORS["selection"]]
    titles = ["Development", "Model selection"]
    years = ["MIMIC-IV 2008-2016", "MIMIC-IV 2017-2019"]
    details = ["50,200 stays", "11,997 stays"]
    for x, width, color, title, year, detail in zip(x_positions, widths, colors, titles, years, details):
        ax.add_patch(Rectangle((x, 0.42), width, 0.42, facecolor=color, edgecolor="#555555", linewidth=0.8))
        ax.text(x + 0.015, 0.76, title, fontsize=8, fontweight="bold", va="top")
        ax.text(x + 0.015, 0.62, year, fontsize=7.2, va="top")
        ax.text(x + 0.015, 0.49, detail, fontsize=7, color="#555555", va="top")
    for start, end in [(0.25, 0.29), (0.50, 0.57)]:
        ax.add_patch(FancyArrowPatch((start, 0.63), (end, 0.63), arrowstyle="-|>", mutation_scale=9, linewidth=0.9, color="#555555"))
    ax.add_patch(Rectangle((0.57, 0.46), 0.14, 0.34, facecolor="white", edgecolor=COLORS["hgb"], linewidth=1.1))
    ax.text(0.64, 0.70, "Frozen HGB", fontsize=8, fontweight="bold", ha="center")
    ax.text(0.64, 0.59, "Refit on\n2008-2019", fontsize=6.8, ha="center", va="center", color="#555555")
    for y, title, detail, color in [
        (0.75, "MIMIC-IV 2020-2022", "9,258 stays", COLORS["test"]),
        (0.49, "eICU-CRD", f"{eicu_stays:,} stays", "white"),
        (0.23, "SICdb", "3,769 stays", "white"),
    ]:
        ax.add_patch(Rectangle((0.79, y - 0.10), 0.19, 0.20, facecolor=color, edgecolor=COLORS["hgb"], linewidth=1.0))
        ax.text(0.885, y + 0.025, title, ha="center", va="center", fontsize=7.3, fontweight="bold")
        ax.text(0.885, y - 0.045, detail, ha="center", va="center", fontsize=6.8, color="#555555")
        ax.add_patch(FancyArrowPatch((0.71, 0.63), (0.79, y), arrowstyle="-|>", mutation_scale=8, linewidth=0.8, color=COLORS["hgb"]))
    ax.text(0.02, 0.18, "Algorithm choice used MIMIC 2017-2019 only; all three validation cohorts were scored after freezing.", fontsize=7.2, color=COLORS["text"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def metric_forest(ax: plt.Axes, ci: pd.DataFrame, metric: str, panel: str, xlim: tuple[float, float]) -> None:
    datasets = ["mimic_temporal_test", "eicu_external", "sicdb_external"]
    y_base = np.arange(len(datasets))[::-1]
    offsets = {"logistic_regression": -0.10, "hist_gradient_boosting": 0.10}
    styles = {
        "logistic_regression": (COLORS["logistic"], "Logistic regression", "o"),
        "hist_gradient_boosting": (COLORS["hgb"], "HistGradientBoosting", "s"),
    }
    for model, (color, label, marker) in styles.items():
        subset = ci[ci.model.eq(model)].set_index("dataset").loc[datasets]
        y = y_base + offsets[model]
        estimate = subset[metric].to_numpy()
        low = subset[f"{metric}_ci_low"].to_numpy()
        high = subset[f"{metric}_ci_high"].to_numpy()
        ax.errorbar(estimate, y, xerr=[estimate - low, high - estimate], fmt=marker, color=color, ecolor=color, elinewidth=1.2, capsize=2.5, ms=4.2, label=label)
    ax.set_yticks(y_base)
    ax.set_yticklabels([DATASET_LABELS[d] for d in datasets])
    ax.set_xlim(*xlim)
    ax.set_xlabel(metric.upper())
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.5)
    ax.set_axisbelow(True)
    panel_label(ax, panel)


def calibration_slope_panel(ax: plt.Axes, ci: pd.DataFrame) -> None:
    datasets = ["mimic_temporal_test", "eicu_external", "sicdb_external"]
    subset = ci[(ci.model.eq("hist_gradient_boosting"))].set_index("dataset").loc[datasets]
    y = np.arange(len(datasets))[::-1]
    estimate = subset.calibration_slope.to_numpy()
    low = subset.calibration_slope_ci_low.to_numpy()
    high = subset.calibration_slope_ci_high.to_numpy()
    ax.errorbar(estimate, y, xerr=[estimate - low, high - estimate], fmt="s", color=COLORS["hgb"], ecolor=COLORS["hgb"], elinewidth=1.2, capsize=2.5, ms=4.2)
    ax.axvline(1, color="#555555", linestyle="--", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([DATASET_LABELS[d] for d in datasets])
    ax.set_xlim(0.45, 1.12)
    ax.set_xlabel("Calibration slope")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.5)
    ax.set_axisbelow(True)
    panel_label(ax, "d")


def figure_1() -> None:
    ci = pd.read_csv(OUTPUTS / "corrected_clustered_confidence_intervals.csv")
    fig = plt.figure(figsize=(7.1, 5.3))
    grid = fig.add_gridspec(2, 3, height_ratios=[0.9, 1.25], hspace=0.48, wspace=0.62)
    study_design(fig.add_subplot(grid[0, :]))
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])
    ax_d = fig.add_subplot(grid[1, 2])
    metric_forest(ax_b, ci, "auroc", "b", (0.60, 0.86))
    metric_forest(ax_c, ci, "auprc", "c", (0.0, 0.17))
    calibration_slope_panel(ax_d, ci)
    handles, labels = ax_b.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.015), handletextpad=0.5, columnspacing=1.5)
    fig.subplots_adjust(bottom=0.12, left=0.11, right=0.98, top=0.98)
    save_figure(fig, "Figure_1_transportability")


def figure_2() -> None:
    curves = pd.read_csv(OUTPUTS / "corrected_calibration_curve.csv")
    dca = pd.read_csv(OUTPUTS / "corrected_recalibrated_decision_curve.csv")
    recal_metrics = pd.read_csv(OUTPUTS / "corrected_recalibration_metrics.csv")
    datasets = ["mimic_temporal_test", "eicu_external", "sicdb_external"]
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 5.4))
    for column, dataset in enumerate(datasets):
        ax = axes[0, column]
        subset = curves[curves.dataset.eq(dataset)]
        max_value = max(subset.predicted_mean.max(), subset.observed_fraction.max()) * 1.08
        max_value = max(max_value, 0.025)
        ax.plot([0, max_value], [0, max_value], linestyle="--", color="#999999", linewidth=0.8)
        for calibration, color, marker, label in [
            ("uncalibrated", COLORS["uncal"], "o", "Uncalibrated"),
            ("intercept_only", COLORS["recal"], "s", "Intercept recalibration"),
        ]:
            part = subset[subset.calibration.eq(calibration)].sort_values("predicted_mean")
            ax.plot(part.predicted_mean, part.observed_fraction, marker=marker, ms=3.4, linewidth=1.1, color=color, label=label)
        ax.set_xlim(0, max_value)
        ax.set_ylim(0, max_value)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(DATASET_LABELS[dataset].replace("\n", " "), fontsize=8, pad=4)
        ax.set_xlabel("Predicted risk")
        if column == 0:
            ax.set_ylabel("Observed fraction")
        ax.grid(color=COLORS["grid"], linewidth=0.4)
        panel_label(ax, chr(ord("a") + column))
        metric_part = recal_metrics[(recal_metrics.dataset.eq(dataset))].set_index("calibration")
        ax.text(0.03, 0.94, f"Brier {metric_part.loc['uncalibrated','brier']:.4f} → {metric_part.loc['intercept_only','brier']:.4f}", transform=ax.transAxes, va="top", fontsize=6.5)

        ax_dca = axes[1, column]
        part = dca[(dca.dataset.eq(dataset)) & (dca.calibration.eq("intercept_only"))].sort_values("threshold")
        ax_dca.plot(part.threshold, part.net_benefit_model, color=COLORS["hgb"], linewidth=1.5, label="Recalibrated HGB")
        ax_dca.plot(part.threshold, part.net_benefit_treat_all, color="#999999", linestyle="--", linewidth=0.9, label="Alert all")
        ax_dca.axhline(0, color="#333333", linewidth=0.7, label="Alert none")
        ax_dca.set_xlim(0, 0.20)
        upper = max(0.01, float(part.net_benefit_model.max()) * 1.15)
        ax_dca.set_ylim(-0.002, upper)
        ax_dca.set_xlabel("Risk threshold")
        if column == 0:
            ax_dca.set_ylabel("Net benefit")
        ax_dca.grid(color=COLORS["grid"], linewidth=0.4)
        panel_label(ax_dca, chr(ord("d") + column))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles_dca, labels_dca = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles + handles_dca, labels + labels_dca, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.005), fontsize=6.8, handlelength=2.1, columnspacing=1.2)
    fig.subplots_adjust(bottom=0.14, left=0.10, right=0.98, top=0.96, hspace=0.48, wspace=0.38)
    save_figure(fig, "Figure_2_recalibration_utility")


def supplementary_figure() -> None:
    hospitals = pd.read_csv(OUTPUTS / "corrected_eicu_hospital_metrics.csv").sort_values("auroc")
    units = pd.read_csv(OUTPUTS / "corrected_sicdb_unit_metrics.csv")
    all_units = pd.read_csv(OUTPUTS / "corrected_sicdb_all_units_metrics.csv").iloc[0]
    strict = pd.read_csv(OUTPUTS / "corrected_strict_future_metrics.csv")
    primary = pd.read_csv(OUTPUTS / "corrected_frozen_validation_metrics.csv")
    primary = primary[primary.model.eq("hist_gradient_boosting")]

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.7), gridspec_kw={"width_ratios": [1.35, 1, 1]})
    ax = axes[0]
    x = np.arange(len(hospitals))
    ax.scatter(x, hospitals.auroc, s=10, color=COLORS["hgb"], alpha=0.75, edgecolors="none")
    ax.axhline(hospitals.auroc.median(), color=COLORS["accent"], linewidth=1.0, label="Median")
    ax.fill_between(x, hospitals.auroc.quantile(.25), hospitals.auroc.quantile(.75), color=COLORS["hgb_light"], alpha=0.25, label="IQR")
    ax.set_xlabel("eICU hospitals ranked by AUROC")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.68, 0.96)
    ax.set_xticks([])
    panel_label(ax, "a")
    ax.legend(fontsize=6.5, loc="lower right")

    ax = axes[1]
    labels = list(units.unit_name) + ["All units"]
    auroc = list(units.auroc) + [all_units.auroc]
    auprc = list(units.auprc) + [all_units.auprc]
    y = np.arange(len(labels))[::-1]
    ax.scatter(auroc, y + 0.10, color=COLORS["hgb"], marker="s", s=22, label="AUROC")
    ax.scatter(auprc, y - 0.10, color=COLORS["recal"], marker="o", s=20, label="AUPRC")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 0.78)
    ax.set_xlabel("Performance")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.4)
    panel_label(ax, "b")
    ax.legend(fontsize=6.5)

    ax = axes[2]
    datasets = ["mimic_temporal_test", "eicu_external", "sicdb_external"]
    primary_values = primary.set_index("dataset").loc[datasets].auroc.to_numpy()
    strict_lookup = strict.copy()
    strict_lookup["base_dataset"] = strict_lookup.dataset.str.replace("_strict_future", "", regex=False)
    strict_values = strict_lookup.set_index("base_dataset").loc[datasets].auroc.to_numpy()
    y = np.arange(len(datasets))[::-1]
    ax.plot(primary_values, y + 0.08, "s", color=COLORS["hgb"], ms=4, label="Primary")
    ax.plot(strict_values, y - 0.08, "o", color=COLORS["accent"], ms=3.8, label="Strict future")
    for yi, first, second in zip(y, primary_values, strict_values):
        ax.plot([first, second], [yi, yi], color="#999999", linewidth=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([DATASET_LABELS[d] for d in datasets])
    ax.set_xlim(0.64, 0.85)
    ax.set_xlabel("AUROC")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.4)
    panel_label(ax, "c")
    ax.legend(fontsize=6.5, loc="lower right")

    fig.subplots_adjust(bottom=0.22, left=0.10, right=0.98, top=0.94, wspace=0.55)
    save_figure(fig, "Figure_S1_robustness")


def source_manifest() -> None:
    rows = [
        {"figure": "Figure 1", "panels": "b-d", "source": "corrected_clustered_confidence_intervals.csv"},
        {"figure": "Figure 2", "panels": "a-c", "source": "corrected_calibration_curve.csv"},
        {"figure": "Figure 2", "panels": "d-f", "source": "corrected_recalibrated_decision_curve.csv"},
        {"figure": "Figure S1", "panels": "a", "source": "corrected_eicu_hospital_metrics.csv"},
        {"figure": "Figure S1", "panels": "b", "source": "corrected_sicdb_unit_metrics.csv; corrected_sicdb_all_units_metrics.csv"},
        {"figure": "Figure S1", "panels": "c", "source": "corrected_frozen_validation_metrics.csv; corrected_strict_future_metrics.csv"},
        {"figure": "Figure S2", "panels": "all", "source": "cohort_flow.csv"},
    ]
    pd.DataFrame(rows).to_csv(OUTPUTS / "figure_source_manifest.csv", index=False)


def main() -> None:
    figure_1()
    figure_2()
    supplementary_figure()
    source_manifest()


if __name__ == "__main__":
    main()
