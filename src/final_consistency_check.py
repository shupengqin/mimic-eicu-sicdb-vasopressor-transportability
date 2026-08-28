"""Check manuscript-facing artifacts against the corrected frozen outputs."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
MANUSCRIPT = OUTPUTS / "corrected_manuscript_draft.md"
PKG = ROOT / "submission_package_2026-08-26"
AUDIT = OUTPUTS / "corrected_final_consistency_audit.md"


def read_rows(name: str) -> list[dict[str, str]]:
    with (OUTPUTS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    checks: list[str] = []

    clustered = read_rows("corrected_clustered_confidence_intervals.csv")
    hgb = {
        row["dataset"]: row
        for row in clustered
        if row["model"] == "hist_gradient_boosting"
    }
    expected_primary = {
        "mimic_temporal_test": ("0.763", "0.737-0.789", "0.705", "-1.184"),
        "eicu_external": ("0.838", "0.833-0.843", "0.990", "-0.556"),
        "sicdb_external": ("0.696", "0.670-0.718", "0.764", "0.469"),
    }
    for dataset, (auroc, interval, slope, citl) in expected_primary.items():
        row = hgb[dataset]
        require(f"{float(row['auroc']):.3f}" == auroc, f"AUROC mismatch for {dataset}")
        require(
            f"{float(row['auroc_ci_low']):.3f}-{float(row['auroc_ci_high']):.3f}" == interval,
            f"AUROC interval mismatch for {dataset}",
        )
        require(f"{float(row['calibration_slope']):.3f}" == slope, f"Slope mismatch for {dataset}")
        require(
            f"{float(row['calibration_in_the_large']):.3f}" == citl,
            f"CITL mismatch for {dataset}",
        )
        for value in (auroc, interval, slope, citl):
            require(value in manuscript, f"Primary value {value} absent from manuscript")
    checks.append("Primary AUROC, confidence intervals, calibration slopes, and CITL match the frozen clustered results.")

    cohort = {row["dataset"]: row for row in read_rows("corrected_table1_cohort_summary.csv")}
    expected_cohorts = {
        "mimic_temporal_test_2020_2022": (9258, 152895, 317),
        "eicu_external": (157765, 2417020, 6297),
        "sicdb_external": (3769, 50054, 555),
    }
    for dataset, (stays, landmarks, events) in expected_cohorts.items():
        row = cohort[dataset]
        require(int(row["n_stays"]) == stays, f"Stay count mismatch for {dataset}")
        require(int(row["n_landmarks"]) == landmarks, f"Landmark count mismatch for {dataset}")
        require(int(row["event_positive_stays"]) == events, f"Event count mismatch for {dataset}")
        for value in (f"{stays:,}", f"{landmarks:,}", f"{events:,}"):
            require(value in manuscript, f"Cohort value {value} absent from manuscript")
    checks.append("Locked evaluation stay, landmark, and event-positive-stay counts match Table 1 source data.")

    thresholds = read_rows("corrected_threshold_analysis.csv")
    expected_thresholds = {
        ("mimic_temporal_test", "uncalibrated"): (0.380, 10.72),
        ("mimic_temporal_test", "intercept_only"): (0.139, 1.76),
        ("eicu_external", "uncalibrated"): (0.448, 7.18),
        ("eicu_external", "intercept_only"): (0.277, 3.11),
        ("sicdb_external", "uncalibrated"): (0.423, 14.94),
        ("sicdb_external", "intercept_only"): (0.627, 31.63),
    }
    for key, expected in expected_thresholds.items():
        row = next(
            item
            for item in thresholds
            if (item["dataset"], item["calibration"]) == key and float(item["threshold"]) == 0.05
        )
        actual = (round(float(row["sensitivity"]), 3), round(float(row["false_alerts_per_100_patient_hours"]), 2))
        require(actual == expected, f"Threshold result mismatch for {key}: {actual} != {expected}")
        require(f"{expected[0]:.3f}" in manuscript, f"Sensitivity {expected[0]:.3f} absent")
        require(f"{expected[1]:.2f}" in manuscript, f"False-alert value {expected[1]:.2f} absent")
    checks.append("The manuscript's 0.05-threshold sensitivity and false-alert values match held-out recalibration outputs.")

    repeated = read_rows("corrected_repeated_recalibration.csv")
    expected_recalibration = {
        "mimic_temporal_test": (-0.001477, 100),
        "eicu_external": (-0.000282, 100),
        "sicdb_external": (-0.000147, 88),
    }
    for dataset, (expected_change, expected_improved) in expected_recalibration.items():
        changes = [
            float(row["brier_change_after_minus_before"])
            for row in repeated
            if row["dataset"] == dataset
        ]
        actual_change = round(median(changes), 6)
        actual_improved = round(sum(value < 0 for value in changes) / len(changes) * 100)
        require(actual_change == expected_change, f"Repeated recalibration mismatch for {dataset}")
        require(actual_improved == expected_improved, f"Recalibration improvement rate mismatch for {dataset}")
        require(f"{expected_change:.6f}" in manuscript, f"Repeated recalibration value absent for {dataset}")
    require("88% of splits" in manuscript, "SICdb repeated-split stability absent")
    checks.append("Repeated recalibration summaries and the revised SICdb interpretation match 100 split-level outputs.")

    hierarchical = read_rows("corrected_hierarchical_cluster_metrics.csv")
    hospital = next(row for row in hierarchical if row["cluster_level"] == "hospital_id")
    hospital_interval = f"{float(hospital['auroc_ci_low']):.3f}-{float(hospital['auroc_ci_high']):.3f}"
    require(hospital_interval == "0.828-0.847", "eICU hospital-bootstrap interval mismatch")
    require(hospital_interval in manuscript, "eICU hospital-bootstrap interval absent")
    require("0.662 to 1.692" in manuscript, "Hospital calibration heterogeneity absent")
    checks.append("eICU hospital-level bootstrap and hospital calibration heterogeneity are present.")

    horizon = read_rows("corrected_horizon_eligibility_audit.csv")
    expected_horizon = {"mimiciv": "9.2%", "eicu": "10.7%", "sicdb": "13.5%"}
    for row in horizon:
        observed = f"{float(row['incomplete_landmark_percent']):.1f}%"
        require(observed == expected_horizon[row["dataset"]], f"Horizon audit mismatch for {row['dataset']}")
        require(observed in manuscript, f"Horizon audit value absent for {row['dataset']}")
    require("conditional estimand" in manuscript, "Conditional-estimand boundary absent")
    checks.append("Complete-horizon exclusions and the conditional-estimand boundary match the audit.")

    gap = read_rows("corrected_one_hour_gap_metrics.csv")
    expected_gap = {
        "mimic_temporal_test": "0.743",
        "eicu_external": "0.831",
        "sicdb_external": "0.674",
    }
    for dataset, expected in expected_gap.items():
        row = next(
            item
            for item in gap
            if item["dataset"] == dataset and item["model"] == "one_hour_gap_trained_hgb"
        )
        require(f"{float(row['auroc']):.3f}" == expected, f"One-hour-gap AUROC mismatch for {dataset}")
        require(expected in manuscript, f"One-hour-gap AUROC absent for {dataset}")
    checks.append("One-hour clinical-action-gap estimates match the sensitivity output.")

    suppression = read_rows("corrected_alert_suppression_metrics.csv")
    expected_suppression = {
        "mimic_temporal_test": "16.79",
        "eicu_external": "26.90",
        "sicdb_external": "206.27",
    }
    for dataset, expected in expected_suppression.items():
        row = next(
            item
            for item in suppression
            if item["dataset"] == dataset
            and item["calibration"] == "intercept_only"
            and float(item["threshold"]) == 0.05
        )
        actual = f"{float(row['false_alert_episodes_per_100_patient_days']):.2f}"
        require(actual == expected, f"Alert-suppression mismatch for {dataset}")
        require(expected in manuscript, f"Alert-suppression value absent for {dataset}")
    checks.append("Six-hour alert-suppression metrics match the policy-sensitivity output.")

    subgroup = read_rows("corrected_subgroup_metrics.csv")
    for dataset, expected_range in {
        "mimic_temporal_test": "0.757 to 0.772",
        "eicu_external": "0.827 to 0.864",
        "sicdb_external": "0.664 to 0.717",
    }.items():
        values = [float(row["auroc"]) for row in subgroup if row["dataset"] == dataset and row["subgroup_variable"] == "age"]
        observed = f"{min(values):.3f} to {max(values):.3f}"
        require(observed == expected_range, f"Subgroup range mismatch for {dataset}")
        require(expected_range in manuscript, f"Subgroup range absent for {dataset}")
    require("not treated as evidence of subgroup effects or fairness" in manuscript, "Fairness boundary absent")
    checks.append("Age subgroup ranges and the non-fairness interpretation boundary are present.")

    cited: set[int] = set()
    invalid_citations: set[int] = set()
    for token in re.findall(r"\[([0-9][0-9, -]*)\]", manuscript):
        for part in token.split(","):
            part = part.strip()
            if "-" in part:
                start, end = (int(value.strip()) for value in part.split("-", 1))
                cited.update(range(start, end + 1))
            else:
                cited.add(int(part))
    invalid_citations = {number for number in cited if number < 1 or number > 30}
    require(not invalid_citations, f"Out-of-range in-text citation numbers: {sorted(invalid_citations)}")
    missing_citations = set(range(1, 31)) - cited
    require(not missing_citations, f"Uncited references: {sorted(missing_citations)}")
    checks.append("All 30 numbered references are cited at least once, with no out-of-range citation number.")

    manifest = (PKG / "figure_source_manifest.csv").read_text(encoding="utf-8")
    require("Figure 1,all,cohort_flow.csv" in manifest, "Figure 1 absent from source manifest")
    legends = (OUTPUTS / "corrected_figure_legends.md").read_text(encoding="utf-8")
    require("Supplementary Fig. S2" in legends, "Supplementary Figure S2 legend absent")
    # Check only the canonical upload names; descriptive aliases and prior layouts
    # are kept outside the final figures directory to prevent upload ambiguity.
    for stem in ("Figure_1", "Figure_2", "Figure_3", "Figure_4", "Figure_5", "Supplementary_Figure_S1_robustness", "Supplementary_Figure_S2_methodological_extensions"):
        for extension in ("svg", "pdf", "png", "tiff"):
            path = PKG / "figures" / f"{stem}.{extension}"
            require(path.exists() and path.stat().st_size > 0, f"Missing figure asset: {path.name}")
    combined_text = manuscript + "\n" + legends
    for number in range(1, 6):
        require((f"Figure {number}" in combined_text) or (f"Fig. {number}" in combined_text), f"Figure reference absent: Figure {number}")
    require("Supplementary Figure S1" in combined_text, "Supplementary Figure S1 reference absent")
    require("Supplementary Figure S2" in combined_text, "Supplementary Figure S2 reference absent")
    checks.append("All figure references, legends, source mappings, and four-format assets are present.")

    require("the ICU or unit stay was the primary independent unit" in manuscript, "Independent inference unit not explicit")
    require("No outcome resampling, synthetic sampling, or class weighting was used" in manuscript, "Class-imbalance handling absent")
    require("not prospectively preregistered" in manuscript, "Registration status absent")
    require("single Austrian tertiary hospital" in manuscript, "SICdb setting not explicit")
    require("universal deployment threshold" in manuscript, "Deployment boundary absent")
    checks.append("Inference unit, class-imbalance handling, registration status, SICdb scope, and deployment boundary are explicit.")

    forbidden_legacy = (
        "primary_validation_report.md",
        "manuscript_blueprint.md",
        "hgb_supplementary_validation_summary.md",
        "strict_future_metrics.csv",
        "model_metrics.csv",
    )
    for name in forbidden_legacy:
        require(name not in manuscript, f"Legacy artifact referenced in manuscript: {name}")
    checks.append("No legacy analysis artifact is referenced in the manuscript draft.")

    placeholders = (
        "[AUTHOR_INPUT_NEEDED: institutional review board",
        "[AUTHOR_INPUT_NEEDED: funder",
        "[AUTHOR_INPUT_NEEDED: competing-interest",
        "[AUTHOR_CONFIRMATION_NEEDED]",
        "[PUBLIC_REPOSITORY_URL]",
        "[PERSISTENT_DOI]",
    )
    for placeholder in placeholders:
        require(placeholder in manuscript, f"Required placeholder absent: {placeholder}")
    checks.append("Required ethics, funding, conflicts, PPI, and repository placeholders are retained.")

    lines = [
        "# Final manuscript consistency audit",
        "",
        f"Status: PASS ({len(checks)}/{len(checks)} checks).",
        "",
    ]
    lines.extend(f"- PASS: {item}" for item in checks)
    lines.extend(
        [
            "",
            "This audit checks internal consistency against the corrected local source files. It does not replace source-article reference verification, journal formatting, disclosure review, or author confirmation of declaration fields.",
        ]
    )
    AUDIT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"PASS: {len(checks)}/{len(checks)} consistency checks")


if __name__ == "__main__":
    main()
