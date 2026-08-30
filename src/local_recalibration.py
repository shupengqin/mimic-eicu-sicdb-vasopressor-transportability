"""Identifier-disjoint intercept-only local recalibration experiments."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import expit, logit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_validation as rv


CALIBRATION_FRACTION = 0.20
RANDOM_SEED = 20260824
THRESHOLDS = [0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20]


def group_split(
    group_ids: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict]:
    unique_groups, group_codes = np.unique(group_ids, return_inverse=True)
    group_event = np.zeros(len(unique_groups), dtype=np.int8)
    np.maximum.at(group_event, group_codes, y)
    calibration_group = np.zeros(len(unique_groups), dtype=bool)
    for event_value in [0, 1]:
        candidates = np.flatnonzero(group_event == event_value)
        rng.shuffle(candidates)
        n_selected = max(1, int(round(CALIBRATION_FRACTION * len(candidates))))
        calibration_group[candidates[:n_selected]] = True
    calibration_mask = calibration_group[group_codes]
    evaluation_mask = ~calibration_mask
    summary = {
        "n_split_groups": len(unique_groups),
        "n_calibration_groups": int(calibration_group.sum()),
        "n_evaluation_groups": int((~calibration_group).sum()),
        "n_calibration_rows": int(calibration_mask.sum()),
        "n_evaluation_rows": int(evaluation_mask.sum()),
        "n_calibration_positive_rows": int(y[calibration_mask].sum()),
        "n_evaluation_positive_rows": int(y[evaluation_mask].sum()),
    }
    return calibration_mask, evaluation_mask, summary


def calibration_curve_rows(
    dataset: str,
    calibration: str,
    y: np.ndarray,
    p: np.ndarray,
) -> list[dict]:
    bins = pd.qcut(pd.Series(p), q=10, labels=False, duplicates="drop").to_numpy()
    rows = []
    for bin_id in sorted(np.unique(bins)):
        mask = bins == bin_id
        rows.append(
            {
                "dataset": dataset,
                "calibration": calibration,
                "bin": int(bin_id) + 1,
                "n": int(mask.sum()),
                "predicted_mean": float(np.mean(p[mask])),
                "observed_fraction": float(np.mean(y[mask])),
            }
        )
    return rows


def threshold_rows(
    dataset: str,
    calibration: str,
    y: np.ndarray,
    p: np.ndarray,
) -> list[dict]:
    rows = []
    for threshold in THRESHOLDS:
        predicted = p >= threshold
        tp = int(np.sum(predicted & (y == 1)))
        fp = int(np.sum(predicted & (y == 0)))
        fn = int(np.sum(~predicted & (y == 1)))
        tn = int(np.sum(~predicted & (y == 0)))
        rows.append(
            {
                "dataset": dataset,
                "calibration": calibration,
                "threshold": threshold,
                "analysis_unit": "eligible_landmark",
                "landmarks_repeated_within_stay": True,
                "n_eligible_landmark_hours": len(y),
                "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
                "specificity": tn / (tn + fp) if tn + fp else np.nan,
                "ppv": tp / (tp + fp) if tp + fp else np.nan,
                "alerts_per_100_eligible_landmark_hours": (tp + fp) / len(y) * 100,
                "false_alerts_per_100_eligible_landmark_hours": fp / len(y) * 100,
            }
        )
    return rows


def dca_rows(
    dataset: str,
    calibration: str,
    y: np.ndarray,
    p: np.ndarray,
) -> list[dict]:
    prevalence = float(np.mean(y))
    rows = []
    n = len(y)
    for threshold in THRESHOLDS:
        predicted = p >= threshold
        tp = int(np.sum(predicted & (y == 1)))
        fp = int(np.sum(predicted & (y == 0)))
        rows.append(
            {
                "dataset": dataset,
                "calibration": calibration,
                "threshold": threshold,
                "analysis_unit": "eligible_landmark",
                "landmarks_repeated_within_stay": True,
                "n_eligible_landmark_hours": n,
                "net_benefit_model": tp / n - fp / n * threshold / (1 - threshold),
                "net_benefit_treat_all": prevalence - (1 - prevalence) * threshold / (1 - threshold),
                "net_benefit_treat_none": 0.0,
            }
        )
    return rows


def run_dataset(dataset: str, rng: np.random.Generator) -> tuple[list[dict], list[dict], list[dict], list[dict], dict]:
    data = np.load(rv.WORK / f"corrected_predictions_{dataset}.npz")
    y = data["y"].astype(np.int8)
    p = np.clip(data["p_hist_gradient_boosting"].astype(float), 1e-7, 1 - 1e-7)
    group_ids = data["patient_id"].astype(np.int64)
    record_ids = data["record_id"].astype(np.int64)
    lead_times = data["lead_time_hours"].astype(float)
    calibration_mask, evaluation_mask, split_summary = group_split(group_ids, y, rng)

    intercept_shift, _, _ = rv.calibration_fit(y[calibration_mask], p[calibration_mask])
    p_recalibrated = expit(intercept_shift + logit(p))

    metric_rows, curve_rows, threshold_output, dca_output = [], [], [], []
    evaluation_y = y[evaluation_mask]
    evaluation_record = record_ids[evaluation_mask]
    for calibration_name, probabilities in [
        ("uncalibrated", p[evaluation_mask]),
        ("intercept_only", p_recalibrated[evaluation_mask]),
    ]:
        metrics, _, _ = rv.summarize_predictions(
            dataset,
            evaluation_y,
            probabilities,
            evaluation_record,
            lead_times[evaluation_mask],
            {},
            len(evaluation_y),
        )
        metrics["calibration"] = calibration_name
        metrics["fitted_intercept_shift"] = intercept_shift if calibration_name == "intercept_only" else 0.0
        metric_rows.append(metrics)
        curve_rows.extend(calibration_curve_rows(dataset, calibration_name, evaluation_y, probabilities))
        threshold_output.extend(threshold_rows(dataset, calibration_name, evaluation_y, probabilities))
        dca_output.extend(dca_rows(dataset, calibration_name, evaluation_y, probabilities))

    split_summary.update(
        {
            "dataset": dataset,
            "split_unit": {
                "mimic_temporal_test": "subject_id",
                "eicu_external": "patienthealthsystemstayid",
                "sicdb_external": "PatientID",
            }[dataset],
            "calibration_fraction": CALIBRATION_FRACTION,
            "intercept_shift": intercept_shift,
            "split_group_overlap": 0,
        }
    )
    return metric_rows, curve_rows, threshold_output, dca_output, split_summary


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    metrics, curves, thresholds, dca, splits = [], [], [], [], []
    for dataset in ["mimic_temporal_test", "eicu_external", "sicdb_external"]:
        result = run_dataset(dataset, rng)
        metrics.extend(result[0])
        curves.extend(result[1])
        thresholds.extend(result[2])
        dca.extend(result[3])
        splits.append(result[4])
    pd.DataFrame(metrics).to_csv(rv.OUTPUTS / "corrected_recalibration_metrics.csv", index=False)
    pd.DataFrame(curves).to_csv(rv.OUTPUTS / "corrected_calibration_curve.csv", index=False)
    pd.DataFrame(thresholds).to_csv(rv.OUTPUTS / "corrected_threshold_analysis.csv", index=False)
    pd.DataFrame(dca).to_csv(rv.OUTPUTS / "corrected_recalibrated_decision_curve.csv", index=False)
    pd.DataFrame(splits).to_csv(rv.OUTPUTS / "corrected_recalibration_split.csv", index=False)


if __name__ == "__main__":
    main()
