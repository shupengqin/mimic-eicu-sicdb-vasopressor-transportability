"""Update disclosure-reviewed aggregate metrics to landmark-hour denominators.

Older output files called row-count rates patient-time rates. The underlying
calculation used one rate per eligible hourly landmark, so this migration
renames the fields and recomputes alert-episode rates with a 100-row scale.
It never reads patient-level data.
"""

from __future__ import annotations

from pathlib import Path
import argparse

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


def migrate_threshold_file(path: Path) -> None:
    if not path.exists():
        return
    frame = pd.read_csv(path)
    rename = {
        "alerts_per_100_patient_hours": "alerts_per_100_eligible_landmark_hours",
        "false_alerts_per_100_patient_hours": "false_alerts_per_100_eligible_landmark_hours",
    }
    frame = frame.rename(columns=rename)
    frame["analysis_unit"] = "eligible_landmark"
    frame["landmarks_repeated_within_stay"] = True
    sample_counts: dict[tuple[str, str], int] = {}
    recalibration_path = path.parent / "corrected_recalibration_metrics.csv"
    if recalibration_path.exists():
        recal = pd.read_csv(recalibration_path)
        sample_counts = {
            (str(row.dataset), str(row.calibration)): int(row.n_samples)
            for row in recal.itertuples(index=False)
        }
    if sample_counts:
        frame["n_eligible_landmark_hours"] = [
            sample_counts.get((str(row.dataset), str(row.calibration)), int(row.n_eligible_landmark_hours) if hasattr(row, "n_eligible_landmark_hours") else 0)
            for row in frame.itertuples(index=False)
        ]
    frame.to_csv(path, index=False)


def migrate_metric_columns(path: Path) -> None:
    if not path.exists():
        return
    frame = pd.read_csv(path)
    rename = {
        col: col.replace("false_alerts_per_100_patient_hours", "false_alerts_per_100_eligible_landmark_hours")
        for col in frame.columns
        if "false_alerts_per_100_patient_hours" in col
    }
    if rename:
        frame = frame.rename(columns=rename)
        frame.to_csv(path, index=False)


def migrate_suppression(path: Path) -> None:
    if not path.exists():
        return
    frame = pd.read_csv(path)
    if "n_landmarks" not in frame or "false_alert_episodes" not in frame:
        return
    frame["n_eligible_landmark_hours"] = frame["n_landmarks"].astype(int)
    frame["exposure_unit"] = "eligible_landmark_hours"
    frame["alert_episodes_per_100_eligible_landmark_hours"] = (
        frame["alert_episodes"] / frame["n_eligible_landmark_hours"] * 100
    )
    frame["false_alert_episodes_per_100_eligible_landmark_hours"] = (
        frame["false_alert_episodes"] / frame["n_eligible_landmark_hours"] * 100
    )
    frame = frame.drop(
        columns=[
            c
            for c in [
                "alert_episodes_per_100_patient_days",
                "false_alert_episodes_per_100_patient_days",
            ]
            if c in frame
        ]
    )
    frame.to_csv(path, index=False)


def main() -> None:
    global OUTPUTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, default=OUTPUTS)
    args = parser.parse_args()
    OUTPUTS = args.outputs
    migrate_suppression(OUTPUTS / "corrected_alert_suppression_metrics.csv")
    migrate_threshold_file(OUTPUTS / "corrected_threshold_analysis.csv")
    for name in [
        "corrected_frozen_validation_metrics.csv",
        "corrected_model_selection_metrics.csv",
        "corrected_recalibration_metrics.csv",
        "corrected_eicu_hospital_metrics.csv",
        "corrected_clinical_rule_metrics.csv",
        "corrected_reduced_model_metrics.csv",
    ]:
        migrate_metric_columns(OUTPUTS / name)
    print("Updated aggregate metric field names and landmark-hour rates.")


if __name__ == "__main__":
    main()
