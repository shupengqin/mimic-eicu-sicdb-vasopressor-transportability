"""Age- and sex-stratified performance for the corrected frozen HGB model."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clustered_inference as ci
import run_validation as rv


BOOTSTRAP_REPLICATES = 500
RANDOM_SEED = 20260824
DEMOGRAPHIC_COLUMNS = ["record_id", "age", "sex_male", "label"]


def read_demographics(dataset: str) -> pd.DataFrame:
    if dataset == "mimic_temporal_test":
        parts = []
        columns = DEMOGRAPHIC_COLUMNS + ["time_group"]
        for chunk in pd.read_csv(
            rv.WORK / "mimic_samples_anchor.csv",
            usecols=columns,
            chunksize=250_000,
            low_memory=False,
        ):
            chunk = chunk[chunk["time_group"].eq("2020 - 2022")]
            parts.append(chunk[DEMOGRAPHIC_COLUMNS])
        frame = pd.concat(parts, ignore_index=True)
    elif dataset == "eicu_external":
        parts = []
        for chunk in pd.read_csv(
            rv.EICU_SAMPLES,
            usecols=DEMOGRAPHIC_COLUMNS,
            chunksize=250_000,
            low_memory=False,
        ):
            parts.append(chunk)
        frame = pd.concat(parts, ignore_index=True)
    elif dataset == "sicdb_external":
        frame = pd.read_csv(
            rv.WORK / "sicdb_samples_main_units.csv",
            usecols=DEMOGRAPHIC_COLUMNS,
            low_memory=False,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    for column in DEMOGRAPHIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def age_groups(age: np.ndarray) -> np.ndarray:
    groups = np.full(len(age), "unknown", dtype=object)
    groups[(age >= 18) & (age <= 44)] = "18-44"
    groups[(age >= 45) & (age <= 64)] = "45-64"
    groups[(age >= 65) & (age <= 79)] = "65-79"
    groups[age >= 80] = "80+"
    return groups


def sex_groups(sex_male: np.ndarray) -> np.ndarray:
    groups = np.full(len(sex_male), "unknown", dtype=object)
    groups[np.isclose(sex_male, 0.0, equal_nan=False)] = "female"
    groups[np.isclose(sex_male, 1.0, equal_nan=False)] = "male"
    return groups


def subgroup_row(
    dataset: str,
    variable: str,
    subgroup: str,
    mask: np.ndarray,
    y_all: np.ndarray,
    p_all: np.ndarray,
    record_ids_all: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    y = y_all[mask].astype(np.int8)
    p = p_all[mask].astype(float)
    record_ids = record_ids_all[mask].astype(np.int64)
    unique_ids, cluster_codes = np.unique(record_ids, return_inverse=True)
    cluster_codes = cluster_codes.astype(np.int32)
    n_clusters = len(unique_ids)

    stay_event = np.zeros(n_clusters, dtype=np.int8)
    np.maximum.at(stay_event, cluster_codes, y)
    if y.sum() == 0 or y.sum() == len(y) or stay_event.sum() == 0 or stay_event.sum() == n_clusters:
        raise RuntimeError(f"Subgroup lacks both outcome classes: {dataset}/{variable}/{subgroup}")

    evaluator = ci.RankingEvaluator(y.astype(float), p, cluster_codes)
    point = {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }
    exact = evaluator.evaluate(np.ones(n_clusters, dtype=np.int32))
    if not np.allclose(exact, [point["auroc"], point["auprc"], point["brier"]], atol=1e-10):
        raise RuntimeError(f"Metric implementation mismatch: {dataset}/{variable}/{subgroup}")

    bootstrap = {metric: np.empty(BOOTSTRAP_REPLICATES) for metric in point}
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled = rng.integers(0, n_clusters, size=n_clusters)
        multiplicity = np.bincount(sampled, minlength=n_clusters).astype(np.int32)
        values = evaluator.evaluate(multiplicity)
        for metric, value in zip(["auroc", "auprc", "brier"], values):
            bootstrap[metric][replicate] = value

    calibration = ci.cluster_robust_calibration(y, p, cluster_codes)
    row = {
        "dataset": dataset,
        "subgroup_variable": variable,
        "subgroup": subgroup,
        "n_landmarks": len(y),
        "n_stays": n_clusters,
        "n_event_positive_stays": int(stay_event.sum()),
        "event_positive_stay_prevalence": float(stay_event.mean()),
        "n_positive_landmarks": int(y.sum()),
        "landmark_prevalence": float(y.mean()),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }
    for metric in ["auroc", "auprc", "brier"]:
        low, high = ci.percentile_interval(bootstrap[metric])
        row[metric] = point[metric]
        row[f"{metric}_ci_low"] = low
        row[f"{metric}_ci_high"] = high
    row.update(calibration)
    return row


def write_summary(results: pd.DataFrame, unknown: pd.DataFrame) -> None:
    lines = [
        "# Demographic subgroup performance",
        "",
        "The frozen histogram-based gradient-boosting model was evaluated by recorded age and sex. These analyses describe representation and performance heterogeneity; they do not establish algorithmic fairness. Each ICU or unit stay contributed to one subgroup per variable. Confidence intervals for AUROC, AUPRC and Brier score used 500 bootstrap replicates sampled at the stay level; calibration intervals used stay-clustered sandwich standard errors.",
        "",
        "## Results",
        "",
        "| Dataset | Variable | Subgroup | Stays | Event-positive stays | AUROC (95% CI) | AUPRC (95% CI) | Calibration slope (95% CI) | CITL (95% CI) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    dataset_names = {
        "mimic_temporal_test": "MIMIC-IV 2020-2022",
        "eicu_external": "eICU-CRD",
        "sicdb_external": "SICdb",
    }
    subgroup_order = {"18-44": 0, "45-64": 1, "65-79": 2, "80+": 3, "female": 4, "male": 5}
    ordered = results.assign(_order=results["subgroup"].map(subgroup_order)).sort_values(
        ["dataset", "subgroup_variable", "_order"]
    )
    for row in ordered.itertuples(index=False):
        lines.append(
            f"| {dataset_names[row.dataset]} | {row.subgroup_variable} | {row.subgroup} | "
            f"{row.n_stays:,} | {row.n_event_positive_stays:,} | "
            f"{row.auroc:.3f} ({row.auroc_ci_low:.3f}-{row.auroc_ci_high:.3f}) | "
            f"{row.auprc:.3f} ({row.auprc_ci_low:.3f}-{row.auprc_ci_high:.3f}) | "
            f"{row.calibration_slope:.3f} ({row.calibration_slope_ci_low:.3f} to {row.calibration_slope_ci_high:.3f}) | "
            f"{row.calibration_in_the_large:.3f} ({row.calibration_in_the_large_ci_low:.3f} to {row.calibration_in_the_large_ci_high:.3f}) |"
        )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Subgroup estimates were exploratory and no family of null-hypothesis tests was defined. Differences in point estimates or overlap of confidence intervals were not treated as formal evidence of between-group differences. Race and ethnicity could not be harmonized across all three databases and were not evaluated. Sex reflects the binary field available in each source database and should not be interpreted as gender identity.",
            "",
            "## Unclassified records",
            "",
            "Records with missing or non-binary demographic values were excluded only from the corresponding subgroup analysis, not from the primary analysis.",
            "",
            "| Dataset | Variable | Unclassified landmarks | Unclassified stays |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for row in unknown.itertuples(index=False):
        lines.append(
            f"| {dataset_names[row.dataset]} | {row.subgroup_variable} | "
            f"{row.n_landmarks:,} | {row.n_stays:,} |"
        )
    (rv.OUTPUTS / "corrected_subgroup_performance.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    unknown_rows = []
    for dataset in ["mimic_temporal_test", "eicu_external", "sicdb_external"]:
        demographics = read_demographics(dataset)
        prediction_path = rv.WORK / f"corrected_predictions_{dataset}.npz"
        with np.load(prediction_path) as prediction:
            record_ids = prediction["record_id"].astype(np.int64)
            y = prediction["y"].astype(np.int8)
            p = prediction["p_hist_gradient_boosting"].astype(float)

        source_record_ids = demographics["record_id"].to_numpy(dtype=np.int64)
        source_y = demographics["label"].to_numpy(dtype=np.int8)
        if len(demographics) != len(y):
            raise RuntimeError(f"Row-count mismatch for {dataset}: {len(demographics)} != {len(y)}")
        if not np.array_equal(source_record_ids, record_ids):
            raise RuntimeError(f"Record-order mismatch for {dataset}")
        if not np.array_equal(source_y, y):
            raise RuntimeError(f"Outcome-order mismatch for {dataset}")

        group_arrays = {
            "age": age_groups(demographics["age"].to_numpy(dtype=float)),
            "sex": sex_groups(demographics["sex_male"].to_numpy(dtype=float)),
        }
        group_levels = {
            "age": ["18-44", "45-64", "65-79", "80+"],
            "sex": ["female", "male"],
        }
        for variable, groups in group_arrays.items():
            unknown_mask = groups == "unknown"
            unknown_rows.append(
                {
                    "dataset": dataset,
                    "subgroup_variable": variable,
                    "n_landmarks": int(unknown_mask.sum()),
                    "n_stays": int(np.unique(record_ids[unknown_mask]).size),
                }
            )
            for subgroup in group_levels[variable]:
                print(f"Evaluating {dataset}/{variable}/{subgroup}", flush=True)
                rows.append(
                    subgroup_row(
                        dataset,
                        variable,
                        subgroup,
                        groups == subgroup,
                        y,
                        p,
                        record_ids,
                        rng,
                    )
                )

    results = pd.DataFrame(rows)
    unknown = pd.DataFrame(unknown_rows)
    results.to_csv(rv.OUTPUTS / "corrected_subgroup_metrics.csv", index=False)
    unknown.to_csv(rv.OUTPUTS / "corrected_subgroup_unclassified.csv", index=False)
    write_summary(results, unknown)


if __name__ == "__main__":
    main()
