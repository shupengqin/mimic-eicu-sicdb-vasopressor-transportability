"""Create the five main figures and the two retained supplementary figures."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = ROOT / "submission_package_2026-08-26" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size": 7.5,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "legend.frameon": False,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
})

BLUE = "#175A9D"
TEAL = "#3D929C"
GREY = "#707070"
RED = "#B64B46"
LIGHT = "#D7D7D7"
TEXT = "#252525"
PALE_BLUE = "#DDE8F2"
PALE_TEAL = "#DDEBE8"
PALE_RED = "#F2D9D4"
DATASETS = ["mimic_temporal_test", "eicu_external", "sicdb_external"]
LABELS = ["MIMIC-IV\n2020-2022", "eICU-CRD", "SICdb"]


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.14, 1.04, label, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom", color=TEXT)


def save_figure(name: str, fig: plt.Figure) -> None:
    target = FIG / name
    fig.savefig(target.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(target.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(target.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def copy_figure(source_stem: str, target_stem: str) -> None:
    for suffix in (".svg", ".pdf", ".tiff", ".png"):
        source = FIG / f"{source_stem}{suffix}"
        if source.exists():
            shutil.copy2(source, FIG / f"{target_stem}{suffix}")


def cohort_flow(ax: plt.Axes) -> None:
    """Render the database-specific inclusion and exclusion flow."""
    ax.set_axis_off()
    columns = [
        {
            "title": "MIMIC-IV 2020-2022",
            "subtitle": "Temporal test",
            "boxes": [
                ("Source adult ICU stays", "n = 94,458"),
                ("Adult stays with LOS >= 12 h", "n = 90,467"),
                ("Eligible risk-set stays", "n = 71,455"),
                ("Hourly landmark rows", "n = 1,110,614\nPositive: 18,608"),
            ],
            "exclusions": ["Excluded: 3,991", "Excluded: 19,012"],
        },
        {
            "title": "eICU-CRD",
            "subtitle": "US multicenter external validation",
            "boxes": [
                ("Source adult unit stays", "n = 200,234"),
                ("Adult stays with LOS >= 12 h", "n = 172,392"),
                ("Eligible risk-set stays", "n = 157,765"),
                ("Hourly landmark rows", "n = 2,417,020\nPositive: 26,767"),
            ],
            "exclusions": ["Excluded: 27,842", "Excluded: 14,627"],
        },
        {
            "title": "SICdb",
            "subtitle": "CWIN and INBD; one Austrian hospital",
            "boxes": [
                ("Source adult cases", "n = 27,223"),
                ("After main-unit restriction", "n = 10,283"),
                ("Main-unit stays with LOS >= 12 h", "n = 9,696"),
                ("Eligible risk-set stays", "n = 3,769"),
                ("Hourly landmark rows", "n = 50,054\nPositive: 2,450"),
            ],
            "exclusions": ["Excluded: 16,940", "Excluded: 587", "Excluded: 5,927"],
        },
    ]
    x_positions = [0.02, 0.35, 0.68]
    widths = [0.29, 0.29, 0.29]
    box_h = 0.105
    for x, width, column in zip(x_positions, widths, columns):
        ax.text(x + width / 2, 0.985, column["title"], ha="center", va="top",
                fontsize=8.1, fontweight="bold", color=TEXT)
        ax.text(x + width / 2, 0.945, column["subtitle"], ha="center", va="top",
                fontsize=6.2, color=GREY)
        count = len(column["boxes"])
        ys = np.linspace(0.80, 0.18, count)
        for index, ((title, detail), y) in enumerate(zip(column["boxes"], ys)):
            fill = PALE_BLUE if index == 0 else (PALE_TEAL if index == count - 1 else "white")
            ax.add_patch(Rectangle((x, y), width, box_h, facecolor=fill,
                                   edgecolor=BLUE if index == count - 1 else GREY,
                                   linewidth=1.0))
            ax.text(x + width / 2, y + box_h * 0.64, title, ha="center", va="center",
                    fontsize=6.6, fontweight="bold" if index in (0, count - 1) else "normal",
                    color=TEXT)
            ax.text(x + width / 2, y + box_h * 0.27, detail, ha="center", va="center",
                    fontsize=6.5, color=GREY)
            if index < count - 1:
                next_y = ys[index + 1] + box_h
                ax.add_patch(FancyArrowPatch((x + width / 2, y),
                                             (x + width / 2, next_y),
                                             arrowstyle="-|>", mutation_scale=8,
                                             linewidth=0.8, color=GREY))
                if index < len(column["exclusions"]):
                    ax.text(x + width - 0.008, (y + next_y) / 2,
                            column["exclusions"][index], ha="right", va="center",
                            fontsize=5.9, color=RED)
        if column["title"] != "SICdb":
            ax.text(x + width / 2, 0.075,
                    "Eligible landmark: ICU hour 6-24\nwith a complete 6-hour horizon",
                    ha="center", va="top", fontsize=6.0, color=GREY)
        else:
            ax.text(x + width / 2, 0.075,
                    "Eligible landmark: ICU hour 6-24\nno prior target vasopressor",
                    ha="center", va="top", fontsize=6.0, color=GREY)
    ax.text(0.50, 0.015,
            "Landmarks after a prior target-vasopressor start were excluded; rows are repeated within stays.",
            ha="center", va="bottom", fontsize=6.6, color=TEXT)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.0)


def figure_1() -> None:
    fig, ax = plt.subplots(figsize=(7.1, 4.55))
    cohort_flow(ax)
    fig.subplots_adjust(left=0.02, right=0.99, top=0.98, bottom=0.02)
    save_figure("Figure_1_cohort_flow", fig)


def study_design(ax: plt.Axes) -> None:
    ax.set_axis_off()
    boxes = [
        (0.01, 0.35, 0.22, 0.42, PALE_BLUE, "Development", "MIMIC-IV 2008-2016", "50,200 stays"),
        (0.28, 0.35, 0.22, 0.42, PALE_TEAL, "Model selection", "MIMIC-IV 2017-2019", "11,997 stays"),
        (0.57, 0.36, 0.16, 0.40, "white", "Frozen HGB", "Refit on 2008-2019", ""),
    ]
    for x, y, w, h, fill, title, line, detail in boxes:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=GREY if fill != "white" else BLUE, linewidth=1.1))
        ax.text(x + 0.018, y + h - 0.08, title, fontsize=8.2, fontweight="bold", va="top")
        ax.text(x + 0.018, y + h - 0.22, line, fontsize=7.1, va="top")
        if detail:
            ax.text(x + 0.018, y + 0.08, detail, fontsize=6.9, color=GREY, va="bottom")
    for start, end in [(0.235, 0.28), (0.505, 0.57)]:
        ax.add_patch(FancyArrowPatch((start, 0.56), (end, 0.56), arrowstyle="-|>", mutation_scale=10, linewidth=0.9, color=GREY))
    outputs = [
        (0.80, 0.70, PALE_RED, "MIMIC-IV 2020-2022", "9,258 stays"),
        (0.80, 0.50, "white", "eICU-CRD", "157,765 stays"),
        (0.80, 0.30, "white", "SICdb", "3,769 stays"),
    ]
    for x, y, fill, title, detail in outputs:
        ax.add_patch(Rectangle((x, y - 0.075), 0.19, 0.15, facecolor=fill, edgecolor=BLUE, linewidth=1.1))
        ax.text(x + 0.095, y + 0.016, title, ha="center", va="center", fontsize=7.4, fontweight="bold")
        ax.text(x + 0.095, y - 0.035, detail, ha="center", va="center", fontsize=6.7, color=GREY)
        ax.add_patch(FancyArrowPatch((0.73, 0.56), (x, y), arrowstyle="-|>", mutation_scale=9, linewidth=0.8, color=BLUE))
    ax.text(0.01, 0.12, "Algorithm selection used MIMIC-IV 2017-2019 only; all validation cohorts were scored after freezing.", fontsize=7.0, color=TEXT)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 0.86)


def figure_2() -> None:
    fig, ax = plt.subplots(figsize=(7.1, 2.45))
    study_design(ax)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.98, bottom=0.04)
    save_figure("Figure_2_study_design", fig)


def metric_forest(ax: plt.Axes, data: pd.DataFrame, metric: str, xlim: tuple[float, float], label: str) -> None:
    offsets = {"logistic_regression": -0.11, "hist_gradient_boosting": 0.11}
    styles = {
        "logistic_regression": (GREY, "Logistic regression", "o"),
        "hist_gradient_boosting": (BLUE, "HGB", "s"),
    }
    y = np.arange(3)[::-1]
    for model, (color, name, marker) in styles.items():
        subset = data[data.model.eq(model)].set_index("dataset").loc[DATASETS]
        estimate = subset[metric].to_numpy()
        low = subset[f"{metric}_ci_low"].to_numpy()
        high = subset[f"{metric}_ci_high"].to_numpy()
        ax.errorbar(estimate, y + offsets[model], xerr=[estimate - low, high - estimate], fmt=marker,
                    color=color, ecolor=color, elinewidth=1.15, capsize=2.6, ms=4.2, label=name)
    ax.set_yticks(y)
    ax.set_yticklabels(LABELS)
    ax.set_xlim(*xlim)
    ax.set_xlabel(label)
    ax.grid(axis="x", color=LIGHT, linewidth=0.5)
    ax.set_axisbelow(True)


def figure_3() -> None:
    data = pd.read_csv(OUT / "corrected_clustered_confidence_intervals.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75), gridspec_kw={"wspace": 0.46})
    metric_forest(axes[0], data, "auroc", (0.60, 0.88), "AUROC")
    metric_forest(axes[1], data, "auprc", (0.00, 0.18), "AUPRC")
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04), handletextpad=0.5, columnspacing=1.4)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.96, bottom=0.22)
    save_figure("Figure_3_discrimination", fig)


def calibration_curve(ax: plt.Axes, curves: pd.DataFrame, metrics: pd.DataFrame, dataset: str, label: str, panel: str) -> None:
    part = curves[curves.dataset.eq(dataset)]
    max_value = max(part.predicted_mean.max(), part.observed_fraction.max()) * 1.08
    max_value = max(max_value, 0.025)
    ax.plot([0, max_value], [0, max_value], "--", color="#999999", linewidth=0.8)
    for calibration, color, marker, name in [("uncalibrated", GREY, "o", "Uncalibrated"), ("intercept_only", TEAL, "s", "Intercept recalibration")]:
        p = part[part.calibration.eq(calibration)].sort_values("predicted_mean")
        ax.plot(p.predicted_mean, p.observed_fraction, marker=marker, ms=3.3, linewidth=1.1, color=color, label=name)
    ax.set_xlim(0, max_value)
    ax.set_ylim(0, max_value)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(label, fontsize=8, pad=4)
    ax.set_xlabel("Predicted risk")
    ax.set_ylabel("Observed fraction")
    ax.grid(color=LIGHT, linewidth=0.4)
    panel_label(ax, panel)
    m = metrics[metrics.dataset.eq(dataset)].set_index("calibration")
    ax.text(0.03, 0.94, f"Brier {m.loc['uncalibrated', 'brier']:.4f} -> {m.loc['intercept_only', 'brier']:.4f}", transform=ax.transAxes, va="top", fontsize=6.4)


def stat_forest(ax: plt.Axes, data: pd.DataFrame, metric: str, xlabel: str, xlim: tuple[float, float], panel: str) -> None:
    subset = data[data.model.eq("hist_gradient_boosting")].set_index("dataset").loc[DATASETS]
    y = np.arange(3)[::-1]
    estimate = subset[metric].to_numpy()
    low = subset[f"{metric}_ci_low"].to_numpy()
    high = subset[f"{metric}_ci_high"].to_numpy()
    ax.errorbar(estimate, y, xerr=[estimate - low, high - estimate], fmt="s", color=BLUE, ecolor=BLUE,
                elinewidth=1.15, capsize=2.6, ms=4.2)
    if metric == "calibration_slope":
        ax.axvline(1, color=GREY, linestyle="--", linewidth=0.8)
    if metric == "calibration_in_the_large":
        ax.axvline(0, color=GREY, linestyle="--", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(LABELS)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color=LIGHT, linewidth=0.5)
    ax.set_axisbelow(True)
    panel_label(ax, panel)


def figure_4() -> None:
    curves = pd.read_csv(OUT / "corrected_calibration_curve.csv")
    recal = pd.read_csv(OUT / "corrected_recalibration_metrics.csv")
    clustered = pd.read_csv(OUT / "corrected_clustered_confidence_intervals.csv")
    fig = plt.figure(figsize=(7.1, 5.35))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.22, 0.88], hspace=0.56, wspace=0.47)
    for i, (dataset, label) in enumerate(zip(DATASETS, LABELS)):
        calibration_curve(fig.add_subplot(grid[0, i]), curves, recal, dataset, label.replace("\n", " "), chr(ord("a") + i))
    stat_forest(fig.add_subplot(grid[1, 0]), clustered, "calibration_in_the_large", "CITL", (-1.5, 0.75), "d")
    stat_forest(fig.add_subplot(grid[1, 1]), clustered, "calibration_slope", "Calibration slope", (0.4, 1.15), "e")
    ax = fig.add_subplot(grid[1, 2])
    for i, dataset in enumerate(DATASETS):
        m = recal[recal.dataset.eq(dataset)].set_index("calibration").loc[["uncalibrated", "intercept_only"]]
        before, after = m.brier.to_numpy()
        ax.plot([i - 0.13, i + 0.13], [before, after], color=LIGHT, linewidth=1.2, zorder=1)
        ax.scatter([i - 0.13, i + 0.13], [before, after], s=24, color=[GREY, TEAL], zorder=2)
    ax.set_xticks(range(3))
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("Brier score")
    ax.grid(axis="y", color=LIGHT, linewidth=0.5)
    ax.set_axisbelow(True)
    panel_label(ax, "f")
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=GREY, markeredgecolor=GREY, markersize=4, label="Uncalibrated"),
               Line2D([0], [0], marker="o", color="none", markerfacecolor=TEAL, markeredgecolor=TEAL, markersize=4, label="Intercept recalibration")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02), handletextpad=0.5, columnspacing=1.4)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.96, bottom=0.16)
    save_figure("Figure_4_calibration", fig)


def decision_curve(ax: plt.Axes, dca: pd.DataFrame, dataset: str, label: str, panel: str) -> None:
    part = dca[(dca.dataset.eq(dataset)) & (dca.calibration.eq("intercept_only"))].sort_values("threshold")
    ax.plot(part.threshold, part.net_benefit_model, color=BLUE, linewidth=1.5, label="Recalibrated HGB")
    ax.plot(part.threshold, part.net_benefit_treat_all, color=GREY, linestyle="--", linewidth=0.9, label="Alert all")
    ax.axhline(0, color=TEXT, linewidth=0.7, label="Alert none")
    ax.set_xlim(0, 0.20)
    upper = max(0.01, float(part.net_benefit_model.max()) * 1.15)
    ax.set_ylim(-0.002, upper)
    ax.set_title(label, fontsize=8, pad=4)
    ax.set_xlabel("Risk threshold")
    ax.set_ylabel("Net benefit")
    ax.grid(color=LIGHT, linewidth=0.4)
    ax.set_axisbelow(True)
    panel_label(ax, panel)


def operating_points(ax: plt.Axes, suppression: pd.DataFrame, dataset: str, label: str, panel: str) -> None:
    part = suppression[(suppression.dataset.eq(dataset)) & (suppression.threshold.eq(0.05)) & (suppression.suppression_hours.eq(6))]
    for calibration, color, marker, name in [("uncalibrated", GREY, "o", "Uncalibrated"), ("intercept_only", TEAL, "s", "Intercept recalibration")]:
        row = part[part.calibration.eq(calibration)].iloc[0]
        ax.scatter(row.event_stay_sensitivity, row.false_alert_episodes_per_100_patient_days, s=34, color=color, marker=marker, label=name, zorder=3)
    ax.set_title(label, fontsize=8, pad=4)
    ax.set_xlabel("Event-stay sensitivity")
    ax.set_ylabel("False episodes / 100 patient-days")
    ymax = float(part.false_alert_episodes_per_100_patient_days.max())
    ax.set_ylim(0, max(1.0, ymax * 1.16))
    ax.grid(color=LIGHT, linewidth=0.4)
    ax.set_axisbelow(True)
    panel_label(ax, panel)


def figure_5() -> None:
    dca = pd.read_csv(OUT / "corrected_recalibrated_decision_curve.csv")
    suppression = pd.read_csv(OUT / "corrected_alert_suppression_metrics.csv")
    fig = plt.figure(figsize=(7.1, 5.2))
    grid = fig.add_gridspec(2, 3, hspace=0.58, wspace=0.48)
    for i, (dataset, label) in enumerate(zip(DATASETS, LABELS)):
        decision_curve(fig.add_subplot(grid[0, i]), dca, dataset, label.replace("\n", " "), chr(ord("a") + i))
        operating_points(fig.add_subplot(grid[1, i]), suppression, dataset, label.replace("\n", " "), chr(ord("d") + i))
    handles = [Line2D([0], [0], color=BLUE, linewidth=1.5, label="Recalibrated HGB"),
               Line2D([0], [0], color=GREY, linestyle="--", linewidth=0.9, label="Alert all"),
               Line2D([0], [0], color=TEXT, linewidth=0.7, label="Alert none"),
               Line2D([0], [0], marker="o", color="none", markerfacecolor=GREY, markeredgecolor=GREY, markersize=4, label="Uncalibrated"),
               Line2D([0], [0], marker="s", color="none", markerfacecolor=TEAL, markeredgecolor=TEAL, markersize=4, label="Intercept recalibration")]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.04), handletextpad=0.5, columnspacing=1.1, fontsize=6.7)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.96, bottom=0.17)
    save_figure("Figure_5_clinical_utility", fig)


def main() -> None:
    figure_1()
    figure_2()
    figure_3()
    figure_4()
    figure_5()
    copy_figure("Figure_S1_robustness", "Supplementary_Figure_S1_robustness")
    copy_figure("Figure_S3_methodological_extensions", "Supplementary_Figure_S2_methodological_extensions")
    for index, stem in enumerate(
        [
            "Figure_1_cohort_flow",
            "Figure_2_study_design",
            "Figure_3_discrimination",
            "Figure_4_calibration",
            "Figure_5_clinical_utility",
        ],
        start=1,
    ):
        copy_figure(stem, f"Figure_{index}")
    manifest = pd.DataFrame([
        ["Figure 1", "all", "cohort_flow.csv"],
        ["Figure 2", "all", "corrected_model_selection_metrics.csv; corrected_model_selection_record.csv"],
        ["Figure 3", "a-b", "corrected_clustered_confidence_intervals.csv"],
        ["Figure 4", "a-c", "corrected_calibration_curve.csv; corrected_recalibration_metrics.csv"],
        ["Figure 4", "d-e", "corrected_clustered_confidence_intervals.csv"],
        ["Figure 4", "f", "corrected_recalibration_metrics.csv"],
        ["Figure 5", "a-c", "corrected_recalibrated_decision_curve.csv"],
        ["Figure 5", "d-f", "corrected_alert_suppression_metrics.csv"],
        ["Supplementary Figure S1", "all", "corrected_eicu_hospital_metrics.csv; corrected_sicdb_unit_metrics.csv; corrected_sicdb_all_units_metrics.csv; corrected_strict_future_metrics.csv"],
        ["Supplementary Figure S2", "all", "corrected_one_hour_gap_metrics.csv; corrected_stay_balanced_metrics.csv; corrected_fixed_hour_metrics.csv; corrected_repeated_recalibration.csv; corrected_eicu_hospital_metrics.csv"],
    ], columns=["figure", "panels", "source"])
    manifest.to_csv(OUT / "figure_source_manifest.csv", index=False)
    manifest.to_csv(ROOT / "submission_package_2026-08-26" / "figure_source_manifest.csv", index=False)
    source_files = sorted({source.strip() for value in manifest["source"] for source in value.split(";")})
    for source in source_files:
        source_path = OUT / source
        if source_path.exists():
            shutil.copy2(source_path, FIG / source)


if __name__ == "__main__":
    main()
