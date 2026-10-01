"""Stay-clustered uncertainty for corrected frozen-model validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_validation as rv


BOOTSTRAP_REPLICATES = 500
RANDOM_SEED = 20260824


@dataclass
class RankingEvaluator:
    y: np.ndarray
    p: np.ndarray
    cluster_codes: np.ndarray

    def __post_init__(self) -> None:
        self.order = np.argsort(-self.p, kind="mergesort")
        sorted_p = self.p[self.order]
        self.sorted_y = self.y[self.order].astype(np.float64)
        self.sorted_clusters = self.cluster_codes[self.order]
        self.starts = np.r_[0, np.flatnonzero(sorted_p[1:] != sorted_p[:-1]) + 1]
        self.n_clusters = int(self.cluster_codes.max()) + 1
        self.cluster_n = np.bincount(self.cluster_codes, minlength=self.n_clusters).astype(float)
        self.cluster_sse = np.bincount(
            self.cluster_codes,
            weights=(self.y - self.p) ** 2,
            minlength=self.n_clusters,
        )

    def evaluate(self, cluster_multiplicity: np.ndarray) -> tuple[float, float, float]:
        row_weight = cluster_multiplicity[self.sorted_clusters]
        positive = np.add.reduceat(row_weight * self.sorted_y, self.starts)
        total = np.add.reduceat(row_weight, self.starts)
        negative = total - positive
        total_positive = float(positive.sum())
        total_negative = float(negative.sum())
        cumulative_positive = np.cumsum(positive)
        cumulative_negative = np.cumsum(negative)
        precision = np.divide(
            cumulative_positive,
            cumulative_positive + cumulative_negative,
            out=np.zeros_like(cumulative_positive),
            where=(cumulative_positive + cumulative_negative) > 0,
        )
        auprc = float(np.sum((positive / total_positive) * precision))
        lower_score_negative = total_negative - cumulative_negative
        auroc = float(
            np.sum(positive * (lower_score_negative + 0.5 * negative))
            / (total_positive * total_negative)
        )
        denominator = float(np.dot(cluster_multiplicity, self.cluster_n))
        brier = float(np.dot(cluster_multiplicity, self.cluster_sse) / denominator)
        return auroc, auprc, brier


def cluster_robust_calibration(
    y: np.ndarray,
    p: np.ndarray,
    cluster_codes: np.ndarray,
) -> dict[str, float]:
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    y = np.asarray(y, dtype=float)
    z = logit(p)
    citl, intercept, slope = rv.calibration_fit(y, p)
    n_clusters = int(cluster_codes.max()) + 1
    n = len(y)

    q0 = expit(citl + z)
    score0 = np.bincount(cluster_codes, weights=y - q0, minlength=n_clusters)
    h0 = float(np.sum(q0 * (1 - q0)))
    correction0 = n_clusters / (n_clusters - 1)
    se_citl = float(np.sqrt(correction0 * np.sum(score0**2) / (h0**2)))

    q = expit(intercept + slope * z)
    variance_weight = q * (1 - q)
    hessian = np.array(
        [
            [np.sum(variance_weight), np.sum(variance_weight * z)],
            [np.sum(variance_weight * z), np.sum(variance_weight * z * z)],
        ]
    )
    score_intercept = np.bincount(cluster_codes, weights=y - q, minlength=n_clusters)
    score_slope = np.bincount(cluster_codes, weights=(y - q) * z, minlength=n_clusters)
    scores = np.column_stack([score_intercept, score_slope])
    meat = scores.T @ scores
    correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - 2))
    inverse_hessian = np.linalg.inv(hessian)
    covariance = correction * inverse_hessian @ meat @ inverse_hessian
    se_intercept, se_slope = np.sqrt(np.diag(covariance))
    return {
        "calibration_in_the_large": citl,
        "calibration_in_the_large_ci_low": citl - 1.96 * se_citl,
        "calibration_in_the_large_ci_high": citl + 1.96 * se_citl,
        "calibration_intercept": intercept,
        "calibration_intercept_ci_low": intercept - 1.96 * se_intercept,
        "calibration_intercept_ci_high": intercept + 1.96 * se_intercept,
        "calibration_slope": slope,
        "calibration_slope_ci_low": slope - 1.96 * se_slope,
        "calibration_slope_ci_high": slope + 1.96 * se_slope,
    }


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def run_dataset(dataset: str, path: Path, rng: np.random.Generator) -> tuple[list[dict], list[dict]]:
    data = np.load(path)
    y = data["y"].astype(np.int8)
    record_ids = data["record_id"].astype(np.int64)
    _, cluster_codes = np.unique(record_ids, return_inverse=True)
    cluster_codes = cluster_codes.astype(np.int32)
    n_clusters = int(cluster_codes.max()) + 1
    model_names = ["logistic_regression", "hist_gradient_boosting"]
    evaluators = {
        name: RankingEvaluator(y.astype(float), data[f"p_{name}"].astype(float), cluster_codes)
        for name in model_names
    }

    point = {}
    for name, evaluator in evaluators.items():
        p = evaluator.p
        point[name] = {
            "auroc": float(roc_auc_score(y, p)),
            "auprc": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
        }
        exact = evaluator.evaluate(np.ones(n_clusters, dtype=np.int32))
        if not np.allclose(exact, [point[name]["auroc"], point[name]["auprc"], point[name]["brier"]], atol=1e-10):
            raise RuntimeError(f"Optimized metric implementation mismatch for {dataset}/{name}")

    bootstrap = {
        name: {metric: np.empty(BOOTSTRAP_REPLICATES) for metric in ["auroc", "auprc", "brier"]}
        for name in model_names
    }
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled = rng.integers(0, n_clusters, size=n_clusters)
        multiplicity = np.bincount(sampled, minlength=n_clusters).astype(np.int32)
        for name, evaluator in evaluators.items():
            values = evaluator.evaluate(multiplicity)
            for metric, value in zip(["auroc", "auprc", "brier"], values):
                bootstrap[name][metric][replicate] = value
        if (replicate + 1) % 50 == 0:
            print(f"{dataset}: {replicate + 1}/{BOOTSTRAP_REPLICATES} cluster bootstraps", flush=True)

    metric_rows = []
    for name in model_names:
        calibration = cluster_robust_calibration(y, evaluators[name].p, cluster_codes)
        row = {
            "dataset": dataset,
            "model": name,
            "n_samples": len(y),
            "n_clusters": n_clusters,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        }
        for metric in ["auroc", "auprc", "brier"]:
            low, high = percentile_interval(bootstrap[name][metric])
            row[metric] = point[name][metric]
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        row.update(calibration)
        metric_rows.append(row)

    contrast_rows = []
    for metric in ["auroc", "auprc", "brier"]:
        differences = bootstrap["hist_gradient_boosting"][metric] - bootstrap["logistic_regression"][metric]
        low, high = percentile_interval(differences)
        contrast_rows.append(
            {
                "dataset": dataset,
                "contrast": "hist_gradient_boosting_minus_logistic_regression",
                "metric": metric,
                "difference": point["hist_gradient_boosting"][metric] - point["logistic_regression"][metric],
                "ci_low": low,
                "ci_high": high,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            }
        )
    return metric_rows, contrast_rows


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    metric_rows, contrast_rows = [], []
    for dataset in ["mimic_temporal_test", "eicu_external", "sicdb_external"]:
        metrics, contrasts = run_dataset(
            dataset,
            rv.WORK / f"corrected_predictions_{dataset}.npz",
            rng,
        )
        metric_rows.extend(metrics)
        contrast_rows.extend(contrasts)
    pd.DataFrame(metric_rows).to_csv(rv.OUTPUTS / "corrected_clustered_confidence_intervals.csv", index=False)
    pd.DataFrame(contrast_rows).to_csv(rv.OUTPUTS / "corrected_clustered_model_contrasts.csv", index=False)


if __name__ == "__main__":
    main()
