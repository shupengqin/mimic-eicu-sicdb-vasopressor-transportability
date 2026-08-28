"""Descriptive alert-episode analysis with a six-hour suppression window."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import expit, logit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_recalibration as lr
import run_validation as rv


DATASETS = ["mimic_temporal_test", "eicu_external", "sicdb_external"]
THRESHOLDS = [0.02, 0.05, 0.10]
SUPPRESSION_HOURS = 6


def emitted_alerts(
    record_id: np.ndarray,
    index_hour: np.ndarray,
    predicted: np.ndarray,
) -> np.ndarray:
    emitted = np.zeros(len(predicted), dtype=bool)
    last_record = -1
    last_hour = -10_000
    for row in np.flatnonzero(predicted):
        current_record = int(record_id[row])
        current_hour = int(index_hour[row])
        if current_record != last_record or current_hour - last_hour >= SUPPRESSION_HOURS:
            emitted[row] = True
            last_record = current_record
            last_hour = current_hour
    return emitted


def summarize(
    dataset: str,
    calibration: str,
    threshold: float,
    y: np.ndarray,
    p: np.ndarray,
    record_id: np.ndarray,
    index_hour: np.ndarray,
) -> dict:
    order = np.lexsort((index_hour, record_id))
    y = y[order]
    p = p[order]
    record_id = record_id[order]
    index_hour = index_hour[order]
    predicted = p >= threshold
    emitted = emitted_alerts(record_id, index_hour, predicted)

    unique_records, record_codes = np.unique(record_id, return_inverse=True)
    event_stay = np.zeros(len(unique_records), dtype=np.int8)
    detected_stay = np.zeros(len(unique_records), dtype=np.int8)
    np.maximum.at(event_stay, record_codes, y)
    np.maximum.at(detected_stay, record_codes, (emitted & (y == 1)).astype(np.int8))

    true_alerts = int(np.sum(emitted & (y == 1)))
    false_alerts = int(np.sum(emitted & (y == 0)))
    event_stays = int(event_stay.sum())
    detected_stays = int(np.sum((event_stay == 1) & (detected_stay == 1)))
    return {
        "dataset": dataset,
        "calibration": calibration,
        "threshold": threshold,
        "suppression_hours": SUPPRESSION_HOURS,
        "n_landmarks": len(y),
        "n_stays": len(unique_records),
        "n_event_stays": event_stays,
        "alert_episodes": true_alerts + false_alerts,
        "true_alert_episodes": true_alerts,
        "false_alert_episodes": false_alerts,
        "event_stay_sensitivity": detected_stays / event_stays if event_stays else np.nan,
        "episode_ppv": true_alerts / (true_alerts + false_alerts)
        if true_alerts + false_alerts
        else np.nan,
        "alert_episodes_per_100_patient_days": (true_alerts + false_alerts)
        / len(y)
        * 2400,
        "false_alert_episodes_per_100_patient_days": false_alerts
        / len(y)
        * 2400,
    }


def main() -> None:
    rng = np.random.default_rng(lr.RANDOM_SEED)
    rows: list[dict] = []
    for dataset in DATASETS:
        with np.load(rv.WORK / f"corrected_predictions_{dataset}.npz") as data:
            y = data["y"].astype(np.int8)
            p = np.clip(
                data["p_hist_gradient_boosting"].astype(float), 1e-7, 1 - 1e-7
            )
            group_id = data["patient_id"].astype(np.int64)
            record_id = data["record_id"].astype(np.int64)
            index_hour = data["index_hour"].astype(np.int16)
        calibration_mask, evaluation_mask, _ = lr.group_split(group_id, y, rng)
        intercept_shift, _, _ = rv.calibration_fit(
            y[calibration_mask], p[calibration_mask]
        )
        probabilities = {
            "uncalibrated": p[evaluation_mask],
            "intercept_only": expit(intercept_shift + logit(p[evaluation_mask])),
        }
        for calibration, probability in probabilities.items():
            for threshold in THRESHOLDS:
                rows.append(
                    summarize(
                        dataset,
                        calibration,
                        threshold,
                        y[evaluation_mask],
                        probability,
                        record_id[evaluation_mask],
                        index_hour[evaluation_mask],
                    )
                )
        print(f"Completed alert-suppression sensitivity: {dataset}", flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(
        rv.OUTPUTS / "corrected_alert_suppression_metrics.csv", index=False
    )
    labels = {
        "mimic_temporal_test": "MIMIC-IV 2020-2022",
        "eicu_external": "eICU-CRD",
        "sicdb_external": "SICdb",
    }
    selected = results[results["threshold"].eq(0.05)]
    lines = [
        "# Six-hour alert-suppression sensitivity",
        "",
        "This descriptive post hoc analysis converted consecutive threshold crossings into alert episodes. After an emitted alert, further alerts in the same stay were suppressed for six hours. Results use the initial identifier-disjoint 80% recalibration evaluation subset and do not establish clinical utility.",
        "",
        "| Dataset | Calibration | Event-stay sensitivity | Episode PPV | False episodes per 100 patient-days |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in selected.itertuples(index=False):
        lines.append(
            f"| {labels[row.dataset]} | {row.calibration} | "
            f"{row.event_stay_sensitivity:.3f} | {row.episode_ppv:.3f} | "
            f"{row.false_alert_episodes_per_100_patient_days:.2f} |"
        )
    lines.extend(
        [
            "",
            "Episode-level metrics are policy-dependent. Prospective workflow evaluation is still required to determine alarm burden, clinician response, and net clinical benefit.",
        ]
    )
    (rv.OUTPUTS / "corrected_alert_suppression_sensitivity.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
