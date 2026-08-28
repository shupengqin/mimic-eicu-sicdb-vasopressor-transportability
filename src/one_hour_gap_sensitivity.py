"""Post hoc one-hour clinical-action-gap sensitivity analysis.

Landmarks followed by vasopressor initiation within one hour are excluded.
The resulting estimand is first initiation during hours (1, 6] after the
landmark. The sensitivity HGB is trained only in MIMIC-IV 2008-2019 with the
same fixed feature set and hyperparameters as the primary HGB.
"""

from __future__ import annotations

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clustered_inference as ci
import methodological_extensions as me
import model_benchmark as mb
import run_validation as rv


BOOTSTRAP_REPLICATES = 500
RANDOM_SEED = 20260828
GAP_HOURS = 1.0


def retain_after_gap(label: np.ndarray, lead_time: np.ndarray) -> np.ndarray:
    label = np.asarray(label, dtype=np.int8)
    lead_time = np.asarray(lead_time, dtype=float)
    invalid_positive = (label == 1) & ~np.isfinite(lead_time)
    if invalid_positive.any():
        raise RuntimeError("Positive landmarks with missing lead time")
    return ~((label == 1) & (lead_time <= GAP_HOURS))


def main() -> None:
    training = me.load_mimic_training(
        rv.FEATURE_COLS + ["record_id", "lead_time_hours"]
    )
    train_keep = retain_after_gap(
        training["label"].to_numpy(),
        training["lead_time_hours"].to_numpy(),
    )
    excluded_training = int((~train_keep).sum())
    training = training.loc[train_keep].reset_index(drop=True)

    model = mb.build_model()
    model.fit(
        rv.feature_frame(training),
        training["label"].astype(int).to_numpy(),
    )
    joblib.dump(
        {
            "analysis_role": "post_hoc_one_hour_clinical_action_gap_sensitivity",
            "model": model,
            "feature_cols": rv.FEATURE_COLS,
            "training_groups": "MIMIC-IV 2008-2019",
            "target": "first continuous vasopressor initiation during hours (1, 6]",
            "excluded_training_landmarks": excluded_training,
            "external_data_used_for_fitting_or_selection": False,
        },
        rv.OUTPUTS / "corrected_one_hour_gap_hgb.joblib",
    )
    del training

    rng = np.random.default_rng(RANDOM_SEED)
    metric_rows: list[dict] = []
    contrast_rows: list[dict] = []
    for dataset in me.DATASETS:
        columns = me.META_COLUMNS + ["lead_time_hours"] + rv.FEATURE_COLS
        frame = me.load_source(dataset, columns)
        with np.load(me.prediction_path(dataset)) as data:
            me.validate_alignment(dataset, frame, data)
            all_y = data["y"].astype(np.int8)
            all_lead = data["lead_time_hours"].astype(float)
            keep = retain_after_gap(all_y, all_lead)
            y = all_y[keep]
            record_ids = data["record_id"].astype(np.int64)[keep]
            primary_prediction = data["p_hist_gradient_boosting"].astype(float)[keep]

        gap_prediction = model.predict_proba(
            rv.feature_frame(frame.loc[keep])
        )[:, 1]
        predictions = {
            "primary_six_hour_hgb_on_gap_cohort": primary_prediction,
            "one_hour_gap_trained_hgb": gap_prediction,
        }

        _, cluster_codes = np.unique(record_ids, return_inverse=True)
        cluster_codes = cluster_codes.astype(np.int32)
        n_clusters = int(cluster_codes.max()) + 1
        evaluators = {
            name: ci.RankingEvaluator(y.astype(float), prediction, cluster_codes)
            for name, prediction in predictions.items()
        }
        point = {
            name: evaluator.evaluate(np.ones(n_clusters, dtype=np.int32))
            for name, evaluator in evaluators.items()
        }
        bootstrap = {
            name: np.empty((BOOTSTRAP_REPLICATES, 3), dtype=float)
            for name in evaluators
        }
        for replicate in range(BOOTSTRAP_REPLICATES):
            sampled = rng.integers(0, n_clusters, size=n_clusters)
            multiplicity = np.bincount(
                sampled, minlength=n_clusters
            ).astype(np.int32)
            for name, evaluator in evaluators.items():
                bootstrap[name][replicate] = evaluator.evaluate(multiplicity)

        for name, prediction in predictions.items():
            calibration = ci.cluster_robust_calibration(
                y, prediction, cluster_codes
            )
            row = {
                "dataset": dataset,
                "model": name,
                "gap_hours": GAP_HOURS,
                "n_landmarks": len(y),
                "n_stays": n_clusters,
                "n_positive_landmarks": int(y.sum()),
                "n_positive_stays": int(np.unique(record_ids[y == 1]).size),
                "excluded_imminent_positive_landmarks": int((~keep).sum()),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            }
            for position, metric in enumerate(["auroc", "auprc", "brier"]):
                low, high = ci.percentile_interval(
                    bootstrap[name][:, position]
                )
                row[metric] = point[name][position]
                row[f"{metric}_ci_low"] = low
                row[f"{metric}_ci_high"] = high
            row.update(calibration)
            metric_rows.append(row)

        for position, metric in enumerate(["auroc", "auprc", "brier"]):
            differences = (
                bootstrap["one_hour_gap_trained_hgb"][:, position]
                - bootstrap["primary_six_hour_hgb_on_gap_cohort"][:, position]
            )
            low, high = ci.percentile_interval(differences)
            contrast_rows.append(
                {
                    "dataset": dataset,
                    "contrast": "gap_trained_minus_primary_on_gap_cohort",
                    "metric": metric,
                    "difference": point["one_hour_gap_trained_hgb"][position]
                    - point["primary_six_hour_hgb_on_gap_cohort"][position],
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                }
            )
        print(f"Completed one-hour-gap sensitivity: {dataset}", flush=True)

    metrics = pd.DataFrame(metric_rows)
    contrasts = pd.DataFrame(contrast_rows)
    metrics.to_csv(
        rv.OUTPUTS / "corrected_one_hour_gap_metrics.csv", index=False
    )
    contrasts.to_csv(
        rv.OUTPUTS / "corrected_one_hour_gap_contrasts.csv", index=False
    )

    labels = {
        "mimic_temporal_test": "MIMIC-IV 2020-2022",
        "eicu_external": "eICU-CRD",
        "sicdb_external": "SICdb",
    }
    gap_rows = metrics[metrics["model"].eq("one_hour_gap_trained_hgb")]
    lines = [
        "# One-hour clinical-action-gap sensitivity",
        "",
        "This post hoc analysis excluded landmarks followed by target-vasopressor initiation within one hour and retrained HGB in MIMIC-IV 2008-2019 using the fixed primary feature set and hyperparameters. The target was first initiation during hours (1, 6] after the landmark. No temporal-test or external data were used for fitting or selection.",
        "",
        "| Dataset | Retained landmarks | Positive landmarks | Excluded imminent positives | AUROC (95% CI) | AUPRC (95% CI) | Calibration slope |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in gap_rows.itertuples(index=False):
        lines.append(
            f"| {labels[row.dataset]} | {int(row.n_landmarks):,} | "
            f"{int(row.n_positive_landmarks):,} | "
            f"{int(row.excluded_imminent_positive_landmarks):,} | "
            f"{row.auroc:.3f} ({row.auroc_ci_low:.3f}-{row.auroc_ci_high:.3f}) | "
            f"{row.auprc:.3f} ({row.auprc_ci_low:.3f}-{row.auprc_ci_high:.3f}) | "
            f"{row.calibration_slope:.3f} |"
        )
    lines.extend(
        [
            "",
            "This is a robustness analysis for temporal separation and clinical actionability, not a new primary model-selection exercise.",
        ]
    )
    (rv.OUTPUTS / "corrected_one_hour_gap_sensitivity.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
