"""Post hoc HGB sensitivity with equal total training weight per ICU stay."""

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
RANDOM_SEED = 20260827


def main() -> None:
    training = me.load_mimic_training(rv.FEATURE_COLS + ["record_id"])
    counts = training.groupby("record_id")["record_id"].transform("size").to_numpy(dtype=float)
    sample_weight = 1.0 / counts
    sample_weight *= len(sample_weight) / sample_weight.sum()
    model = mb.build_model()
    model.fit(
        rv.feature_frame(training),
        training["label"].astype(int).to_numpy(),
        sample_weight=sample_weight,
    )
    joblib.dump(
        {
            "analysis_role": "post_hoc_equal_total_stay_weight_sensitivity",
            "model": model,
            "feature_cols": rv.FEATURE_COLS,
            "training_groups": "MIMIC-IV 2008-2019",
            "weight_rule": "each ICU stay has equal total fitting weight",
            "external_data_used_for_fitting_or_selection": False,
        },
        rv.OUTPUTS / "corrected_stay_weighted_hgb.joblib",
    )
    del training, sample_weight

    rng = np.random.default_rng(RANDOM_SEED)
    metric_rows = []
    contrast_rows = []
    for dataset in me.DATASETS:
        frame = me.load_source(dataset, me.META_COLUMNS + rv.FEATURE_COLS)
        with np.load(me.prediction_path(dataset)) as data:
            me.validate_alignment(dataset, frame, data)
            y = data["y"].astype(np.int8)
            record_ids = data["record_id"].astype(np.int64)
            predictions = {
                "primary_landmark_weighted_hgb": data["p_hist_gradient_boosting"].astype(float),
                "stay_weighted_training_hgb": model.predict_proba(rv.feature_frame(frame))[:, 1],
            }
        _, cluster_codes = np.unique(record_ids, return_inverse=True)
        cluster_codes = cluster_codes.astype(np.int32)
        evaluators = {
            name: ci.RankingEvaluator(y.astype(float), prediction, cluster_codes)
            for name, prediction in predictions.items()
        }
        n_clusters = int(cluster_codes.max()) + 1
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
            multiplicity = np.bincount(sampled, minlength=n_clusters).astype(np.int32)
            for name, evaluator in evaluators.items():
                bootstrap[name][replicate] = evaluator.evaluate(multiplicity)
        for name, prediction in predictions.items():
            calibration = ci.cluster_robust_calibration(y, prediction, cluster_codes)
            row = {
                "dataset": dataset,
                "model": name,
                "n_landmarks": len(y),
                "n_stays": n_clusters,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            }
            for position, metric in enumerate(["auroc", "auprc", "brier"]):
                low, high = ci.percentile_interval(bootstrap[name][:, position])
                row[metric] = point[name][position]
                row[f"{metric}_ci_low"] = low
                row[f"{metric}_ci_high"] = high
            row.update(calibration)
            metric_rows.append(row)
        for position, metric in enumerate(["auroc", "auprc", "brier"]):
            differences = (
                bootstrap["stay_weighted_training_hgb"][:, position]
                - bootstrap["primary_landmark_weighted_hgb"][:, position]
            )
            low, high = ci.percentile_interval(differences)
            contrast_rows.append(
                {
                    "dataset": dataset,
                    "contrast": "stay_weighted_minus_primary_hgb",
                    "metric": metric,
                    "difference": point["stay_weighted_training_hgb"][position]
                    - point["primary_landmark_weighted_hgb"][position],
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                }
            )
        print(f"Completed stay-weighted training sensitivity: {dataset}", flush=True)

    metrics = pd.DataFrame(metric_rows)
    contrasts = pd.DataFrame(contrast_rows)
    metrics.to_csv(rv.OUTPUTS / "corrected_stay_weighted_model_metrics.csv", index=False)
    contrasts.to_csv(rv.OUTPUTS / "corrected_stay_weighted_model_contrasts.csv", index=False)

    labels = {
        "mimic_temporal_test": "MIMIC-IV 2020-2022",
        "eicu_external": "eICU-CRD",
        "sicdb_external": "SICdb",
    }
    lines = [
        "# Equal-stay-weight training sensitivity",
        "",
        "This post hoc sensitivity model used the same full predictor set and fixed HGB hyperparameters as the corrected primary model, but each MIMIC-IV 2008-2019 ICU stay contributed equal total fitting weight. No temporal-test or external data were used for fitting or selection.",
        "",
        "| Dataset | Model | AUROC (95% CI) | AUPRC (95% CI) | Brier | Calibration slope |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(
            f"| {labels[row.dataset]} | {row.model} | {row.auroc:.3f} ({row.auroc_ci_low:.3f}-{row.auroc_ci_high:.3f}) | "
            f"{row.auprc:.3f} ({row.auprc_ci_low:.3f}-{row.auprc_ci_high:.3f}) | {row.brier:.5f} | {row.calibration_slope:.3f} |"
        )
    lines.extend(
        [
            "",
            "The sensitivity model changes the training estimand from an average landmark to equal total contribution per stay. It is a robustness analysis rather than replacement model selection.",
        ]
    )
    (rv.OUTPUTS / "corrected_stay_weighted_training_sensitivity.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
