"""Patient- and hospital-clustered sensitivity inference for frozen HGB predictions."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clustered_inference as ci
import run_validation as rv


BOOTSTRAP_REPLICATES = 500
RANDOM_SEED = 20260826
DATASETS = ["mimic_temporal_test", "eicu_external", "sicdb_external"]


def bootstrap_by_cluster(
    dataset: str,
    cluster_level: str,
    y: np.ndarray,
    p: np.ndarray,
    cluster_ids: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    _, codes = np.unique(cluster_ids, return_inverse=True)
    codes = codes.astype(np.int32)
    n_clusters = int(codes.max()) + 1
    evaluator = ci.RankingEvaluator(y.astype(float), p.astype(float), codes)
    point = evaluator.evaluate(np.ones(n_clusters, dtype=np.int32))
    replicates = np.empty((BOOTSTRAP_REPLICATES, 3), dtype=float)
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled = rng.integers(0, n_clusters, size=n_clusters)
        multiplicity = np.bincount(sampled, minlength=n_clusters).astype(np.int32)
        replicates[replicate] = evaluator.evaluate(multiplicity)
    calibration = ci.cluster_robust_calibration(y, p, codes)
    row = {
        "dataset": dataset,
        "cluster_level": cluster_level,
        "n_landmarks": len(y),
        "n_clusters": n_clusters,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }
    for position, metric in enumerate(["auroc", "auprc", "brier"]):
        low, high = ci.percentile_interval(replicates[:, position])
        row[metric] = point[position]
        row[f"{metric}_ci_low"] = low
        row[f"{metric}_ci_high"] = high
    row.update(calibration)
    return row


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for dataset in DATASETS:
        with np.load(rv.WORK / f"corrected_predictions_{dataset}.npz") as data:
            y = data["y"].astype(np.int8)
            p = data["p_hist_gradient_boosting"].astype(float)
            patient_ids = data["patient_id"].astype(np.int64)
        rows.append(
            bootstrap_by_cluster(
                dataset,
                {
                    "mimic_temporal_test": "subject_id",
                    "eicu_external": "patienthealthsystemstayid",
                    "sicdb_external": "PatientID",
                }[dataset],
                y,
                p,
                patient_ids,
                rng,
            )
        )
        print(f"Completed patient-level clustering: {dataset}", flush=True)

    with np.load(rv.WORK / "corrected_predictions_eicu_external.npz") as data:
        y = data["y"].astype(np.int8)
        p = data["p_hist_gradient_boosting"].astype(float)
        record_ids = data["record_id"].astype(np.int64)
    site_map = pd.read_csv(rv.WORK / "eicu_site_map.csv", usecols=["record_id", "hospital_id"])
    lookup = site_map.drop_duplicates("record_id").set_index("record_id")["hospital_id"]
    hospital_ids = pd.Series(record_ids).map(lookup).to_numpy()
    if pd.isna(hospital_ids).any():
        raise RuntimeError("Missing hospital mapping for eICU predictions")
    rows.append(
        bootstrap_by_cluster(
            "eicu_external",
            "hospital_id",
            y,
            p,
            hospital_ids.astype(np.int64),
            rng,
        )
    )
    print("Completed hospital-level clustering: eICU", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(rv.OUTPUTS / "corrected_hierarchical_cluster_metrics.csv", index=False)

    hospitals = pd.read_csv(rv.OUTPUTS / "corrected_eicu_hospital_metrics.csv")
    summary_rows = []
    for metric in [
        "auroc",
        "auprc",
        "calibration_in_the_large",
        "calibration_slope",
        "record_event_prevalence",
    ]:
        values = hospitals[metric]
        summary_rows.append(
            {
                "metric": metric,
                "n_hospitals": len(values),
                "minimum": values.min(),
                "q1": values.quantile(0.25),
                "median": values.median(),
                "q3": values.quantile(0.75),
                "maximum": values.max(),
            }
        )
    hospital_summary = pd.DataFrame(summary_rows)
    hospital_summary.to_csv(
        rv.OUTPUTS / "corrected_eicu_hospital_calibration_summary.csv", index=False
    )

    labels = {
        "mimic_temporal_test": "MIMIC-IV 2020-2022",
        "eicu_external": "eICU-CRD",
        "sicdb_external": "SICdb",
    }
    lines = [
        "# Hierarchical inference sensitivity",
        "",
        "The frozen HGB predictions were re-evaluated with clusters defined at the patient or health-system-stay level. eICU-CRD was additionally bootstrapped by hospital as the highest observed sampling level. Point estimates are unchanged; uncertainty intervals and robust calibration intervals reflect the alternative clustering unit.",
        "",
        "| Dataset | Cluster level | Clusters | AUROC (95% CI) | AUPRC (95% CI) | Calibration slope (95% CI) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| {labels[row.dataset]} | {row.cluster_level} | {row.n_clusters:,} | "
            f"{row.auroc:.3f} ({row.auroc_ci_low:.3f}-{row.auroc_ci_high:.3f}) | "
            f"{row.auprc:.3f} ({row.auprc_ci_low:.3f}-{row.auprc_ci_high:.3f}) | "
            f"{row.calibration_slope:.3f} ({row.calibration_slope_ci_low:.3f}-{row.calibration_slope_ci_high:.3f}) |"
        )
    lines.extend(
        [
            "",
            "## Descriptive eICU hospital calibration heterogeneity",
            "",
            "| Metric | Hospitals | Median (IQR) | Range |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in hospital_summary.itertuples(index=False):
        lines.append(
            f"| {row.metric} | {row.n_hospitals} | {row.median:.3f} ({row.q1:.3f}-{row.q3:.3f}) | "
            f"{row.minimum:.3f}-{row.maximum:.3f} |"
        )
    lines.extend(
        [
            "",
            "Hospital-specific calibration estimates are descriptive and unshrunk. The hospital-level bootstrap addresses dependence at the highest available eICU sampling level but does not make the participating hospitals a random probability sample of all US hospitals.",
        ]
    )
    (rv.OUTPUTS / "corrected_hierarchical_inference.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
