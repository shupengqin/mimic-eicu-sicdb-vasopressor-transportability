"""High-value post hoc checks requested by a strict methodological review.

The corrected primary model remains frozen. New models are sensitivity models
trained only in MIMIC-IV 2008-2019 with the existing fixed HGB parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clustered_inference as ci
import model_benchmark as mb
import run_validation as rv


BOOTSTRAP_REPLICATES = 500
RECALIBRATION_REPEATS = 100
RANDOM_SEED = 20260825
DATASETS = ["mimic_temporal_test", "eicu_external", "sicdb_external"]
DATASET_LABELS = {
    "mimic_temporal_test": "MIMIC-IV 2020-2022",
    "eicu_external": "eICU-CRD",
    "sicdb_external": "SICdb",
}
SOURCE_PATHS = {
    "mimic_temporal_test": rv.WORK / "mimic_samples_anchor.csv",
    "eicu_external": rv.EICU_SAMPLES,
    "sicdb_external": rv.WORK / "sicdb_samples_main_units.csv",
}
META_COLUMNS = ["record_id", "patient_id", "index_hour", "label"]
VITAL_FEATURES = ["age", "sex_male", "index_hour"] + [
    f"{variable}_{suffix}"
    for variable in rv.VITAL_VARS
    for suffix in ["last", "mean6", "min6", "max6"]
]
NO_TEMPERATURE_FEATURES = [
    column for column in rv.FEATURE_COLS if not column.startswith("temp_")
]


@dataclass
class WeightedRankingEvaluator:
    y: np.ndarray
    p: np.ndarray
    cluster_codes: np.ndarray
    base_weight: np.ndarray

    def __post_init__(self) -> None:
        self.y = np.asarray(self.y, dtype=float)
        self.p = np.asarray(self.p, dtype=float)
        self.cluster_codes = np.asarray(self.cluster_codes, dtype=np.int32)
        self.base_weight = np.asarray(self.base_weight, dtype=float)
        self.order = np.argsort(-self.p, kind="mergesort")
        sorted_p = self.p[self.order]
        self.sorted_y = self.y[self.order]
        self.sorted_clusters = self.cluster_codes[self.order]
        self.sorted_base_weight = self.base_weight[self.order]
        self.starts = np.r_[0, np.flatnonzero(sorted_p[1:] != sorted_p[:-1]) + 1]
        self.n_clusters = int(self.cluster_codes.max()) + 1
        self.cluster_weight = np.bincount(
            self.cluster_codes,
            weights=self.base_weight,
            minlength=self.n_clusters,
        )
        self.cluster_sse = np.bincount(
            self.cluster_codes,
            weights=self.base_weight * (self.y - self.p) ** 2,
            minlength=self.n_clusters,
        )

    def evaluate(self, cluster_multiplicity: np.ndarray) -> tuple[float, float, float]:
        row_weight = (
            cluster_multiplicity[self.sorted_clusters] * self.sorted_base_weight
        )
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
        denominator = float(np.dot(cluster_multiplicity, self.cluster_weight))
        brier = float(np.dot(cluster_multiplicity, self.cluster_sse) / denominator)
        return auroc, auprc, brier


def prediction_path(dataset: str) -> Path:
    return rv.WORK / f"corrected_predictions_{dataset}.npz"


def load_source(dataset: str, columns: list[str]) -> pd.DataFrame:
    usecols = list(dict.fromkeys(columns + (["time_group"] if dataset == "mimic_temporal_test" else [])))
    parts = []
    chunksize = 250_000 if dataset != "sicdb_external" else None
    iterator = pd.read_csv(
        SOURCE_PATHS[dataset],
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False,
    )
    if chunksize is None:
        iterator = [iterator]
    for chunk in iterator:
        if dataset == "mimic_temporal_test":
            chunk = chunk[chunk["time_group"].eq("2020 - 2022")].drop(columns="time_group")
        parts.append(chunk)
    frame = pd.concat(parts, ignore_index=True)
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def load_mimic_training(columns: list[str]) -> pd.DataFrame:
    usecols = list(dict.fromkeys(columns + ["label", "time_group"]))
    parts = []
    for chunk in pd.read_csv(
        SOURCE_PATHS["mimic_temporal_test"],
        usecols=usecols,
        chunksize=250_000,
        low_memory=False,
    ):
        chunk = chunk[~chunk["time_group"].eq("2020 - 2022")].drop(columns="time_group")
        parts.append(chunk)
    frame = pd.concat(parts, ignore_index=True)
    for column in columns + ["label"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def validate_alignment(dataset: str, frame: pd.DataFrame, data: object) -> None:
    if len(frame) != len(data["y"]):
        raise RuntimeError(f"Row mismatch for {dataset}: {len(frame)} != {len(data['y'])}")
    if not np.array_equal(frame["record_id"].to_numpy(dtype=np.int64), data["record_id"]):
        raise RuntimeError(f"Record order mismatch for {dataset}")
    if not np.array_equal(frame["index_hour"].to_numpy(dtype=np.int16), data["index_hour"]):
        raise RuntimeError(f"Landmark-hour order mismatch for {dataset}")
    if not np.array_equal(frame["label"].to_numpy(dtype=np.int8), data["y"]):
        raise RuntimeError(f"Outcome order mismatch for {dataset}")


def weighted_calibration(y: np.ndarray, p: np.ndarray, weight: np.ndarray) -> tuple[float, float, float]:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    weight = np.asarray(weight, dtype=float)
    weight = weight / weight.sum()
    z = logit(p)
    prevalence = float(np.sum(weight * y))

    def offset_fn(intercept: float) -> float:
        return float(np.sum(weight * expit(intercept + z)) - prevalence)

    citl = float(brentq(offset_fn, -40, 40))

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = theta[0] + theta[1] * z
        probability = expit(eta)
        value = -float(np.sum(weight * (y * np.log(probability + 1e-12) + (1 - y) * np.log(1 - probability + 1e-12))))
        gradient = np.array(
            [
                -np.sum(weight * (y - probability)),
                -np.sum(weight * (y - probability) * z),
            ]
        )
        return value, gradient

    result = minimize(
        lambda theta: objective(theta),
        np.array([0.0, 1.0]),
        jac=True,
        method="L-BFGS-B",
    )
    if not result.success:
        raise RuntimeError("Weighted calibration fit failed")
    return citl, float(result.x[0]), float(result.x[1])


def bootstrap_evaluator(
    evaluator: WeightedRankingEvaluator,
    rng: np.random.Generator,
) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    multiplicity = np.ones(evaluator.n_clusters, dtype=np.int32)
    values = evaluator.evaluate(multiplicity)
    replicates = np.empty((BOOTSTRAP_REPLICATES, 3), dtype=float)
    for index in range(BOOTSTRAP_REPLICATES):
        sampled = rng.integers(0, evaluator.n_clusters, size=evaluator.n_clusters)
        cluster_multiplicity = np.bincount(
            sampled, minlength=evaluator.n_clusters
        ).astype(np.int32)
        replicates[index] = evaluator.evaluate(cluster_multiplicity)
    metrics = dict(zip(["auroc", "auprc", "brier"], values))
    intervals = {
        metric: ci.percentile_interval(replicates[:, position])
        for position, metric in enumerate(["auroc", "auprc", "brier"])
    }
    return metrics, intervals


def stay_balanced_and_fixed_hour(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    balanced_rows = []
    hour_rows = []
    for dataset in DATASETS:
        with np.load(prediction_path(dataset)) as data:
            y = data["y"].astype(np.int8)
            p = data["p_hist_gradient_boosting"].astype(float)
            record_ids = data["record_id"].astype(np.int64)
            hours = data["index_hour"].astype(np.int16)
        _, cluster_codes = np.unique(record_ids, return_inverse=True)
        cluster_codes = cluster_codes.astype(np.int32)
        counts = np.bincount(cluster_codes)
        weight = 1.0 / counts[cluster_codes]
        evaluator = WeightedRankingEvaluator(y, p, cluster_codes, weight)
        metrics, intervals = bootstrap_evaluator(evaluator, rng)
        citl, intercept, slope = weighted_calibration(y, p, weight)
        balanced_rows.append(
            {
                "dataset": dataset,
                "estimand": "average_stay_equal_total_weight",
                "n_landmarks": len(y),
                "n_stays": evaluator.n_clusters,
                **metrics,
                **{f"{metric}_ci_low": intervals[metric][0] for metric in intervals},
                **{f"{metric}_ci_high": intervals[metric][1] for metric in intervals},
                "calibration_in_the_large": citl,
                "calibration_intercept": intercept,
                "calibration_slope": slope,
            }
        )

        for hour in [6, 12, 18, 24]:
            mask = hours == hour
            y_hour = y[mask]
            p_hour = p[mask]
            records_hour = record_ids[mask]
            _, codes_hour = np.unique(records_hour, return_inverse=True)
            codes_hour = codes_hour.astype(np.int32)
            evaluator_hour = WeightedRankingEvaluator(
                y_hour,
                p_hour,
                codes_hour,
                np.ones(len(y_hour), dtype=float),
            )
            metrics_hour, intervals_hour = bootstrap_evaluator(evaluator_hour, rng)
            calibration = ci.cluster_robust_calibration(y_hour, p_hour, codes_hour)
            hour_rows.append(
                {
                    "dataset": dataset,
                    "index_hour": hour,
                    "n_landmarks": len(y_hour),
                    "n_positive_landmarks": int(y_hour.sum()),
                    **metrics_hour,
                    **{f"{metric}_ci_low": intervals_hour[metric][0] for metric in intervals_hour},
                    **{f"{metric}_ci_high": intervals_hour[metric][1] for metric in intervals_hour},
                    **calibration,
                }
            )
        print(f"Completed stay-balanced and fixed-hour analysis: {dataset}", flush=True)
    return pd.DataFrame(balanced_rows), pd.DataFrame(hour_rows)


def score_baseline(y: np.ndarray, score: np.ndarray, record_ids: np.ndarray, rng: np.random.Generator) -> dict:
    mask = np.isfinite(score)
    y = y[mask]
    score = score[mask]
    record_ids = record_ids[mask]
    _, codes = np.unique(record_ids, return_inverse=True)
    codes = codes.astype(np.int32)
    evaluator = WeightedRankingEvaluator(y, score, codes, np.ones(len(y), dtype=float))
    metrics, intervals = bootstrap_evaluator(evaluator, rng)
    return {
        "n_landmarks": len(y),
        "n_stays": evaluator.n_clusters,
        "n_positive_landmarks": int(y.sum()),
        "auroc": metrics["auroc"],
        "auroc_ci_low": intervals["auroc"][0],
        "auroc_ci_high": intervals["auroc"][1],
        "auprc": metrics["auprc"],
        "auprc_ci_low": intervals["auprc"][0],
        "auprc_ci_high": intervals["auprc"][1],
    }


def rule_metrics(y: np.ndarray, alert: np.ndarray) -> dict[str, float]:
    tp = int(np.sum(alert & (y == 1)))
    fp = int(np.sum(alert & (y == 0)))
    tn = int(np.sum(~alert & (y == 0)))
    fn = int(np.sum(~alert & (y == 1)))
    return {
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "ppv": tp / (tp + fp) if tp + fp else np.nan,
        "false_alerts_per_100_eligible_landmark_hours": fp / len(y) * 100,
    }


def clinical_baselines(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ranking_rows = []
    rule_rows = []
    temperature_rows = []
    source_columns = META_COLUMNS + ["map_last", "hr_last", "sbp_last", "temp_last"]
    for dataset in DATASETS:
        frame = load_source(dataset, source_columns)
        with np.load(prediction_path(dataset)) as data:
            validate_alignment(dataset, frame, data)
            y = data["y"].astype(np.int8)
            full_prediction = data["p_hist_gradient_boosting"].astype(float)
            record_ids = data["record_id"].astype(np.int64)
        map_last = frame["map_last"].where(frame["map_last"].between(*rv.PHYSIOLOGIC_RANGES["map"])).to_numpy(dtype=float)
        hr_last = frame["hr_last"].where(frame["hr_last"].between(*rv.PHYSIOLOGIC_RANGES["hr"])).to_numpy(dtype=float)
        sbp_last = frame["sbp_last"].where(frame["sbp_last"].between(*rv.PHYSIOLOGIC_RANGES["sbp"])).to_numpy(dtype=float)
        baseline_scores = {
            "map_alone": -map_last,
            "shock_index": hr_last / sbp_last,
            "modified_shock_index": hr_last / map_last,
        }
        scores = {"full_hgb_all_rows": full_prediction}
        for score_name, score in baseline_scores.items():
            scores[score_name] = score
            scores[f"full_hgb_on_{score_name}_available"] = np.where(
                np.isfinite(score), full_prediction, np.nan
            )
        for score_name, score in scores.items():
            row = score_baseline(y, score, record_ids, rng)
            ranking_rows.append({"dataset": dataset, "score": score_name, **row})

        rules = {
            "MAP<65 mmHg": frame["map_last"].to_numpy(dtype=float) < 65,
            "shock index>0.9": baseline_scores["shock_index"] > 0.9,
            "modified shock index>1.3": baseline_scores["modified_shock_index"] > 1.3,
        }
        for rule_name, alert in rules.items():
            finite = np.isfinite(
                scores[
                    {
                        "MAP<65 mmHg": "map_alone",
                        "shock index>0.9": "shock_index",
                        "modified shock index>1.3": "modified_shock_index",
                    }[rule_name]
                ]
            )
            rule_rows.append(
                {
                    "dataset": dataset,
                    "rule": rule_name,
                    "n_landmarks": int(finite.sum()),
                    **rule_metrics(y[finite], alert[finite]),
                }
            )

        temperature = frame["temp_last"].to_numpy(dtype=float)
        valid = temperature[np.isfinite(temperature)]
        temperature_rows.append(
            {
                "dataset": dataset,
                "n_landmarks": len(temperature),
                "n_temperature_available": len(valid),
                "temperature_median": float(np.median(valid)),
                "temperature_q01": float(np.quantile(valid, 0.01)),
                "temperature_q99": float(np.quantile(valid, 0.99)),
                "fraction_below_32": float(np.mean(valid < 32)),
                "fraction_below_34": float(np.mean(valid < 34)),
                "fraction_above_40": float(np.mean(valid > 40)),
            }
        )
        print(f"Completed clinical baselines: {dataset}", flush=True)
    return pd.DataFrame(ranking_rows), pd.DataFrame(rule_rows), pd.DataFrame(temperature_rows)


def clean_feature_subset(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    return rv.feature_frame(frame).reindex(columns=feature_columns)


def fit_sensitivity_model(training: pd.DataFrame, feature_columns: list[str]) -> object:
    model = mb.build_model()
    model.fit(
        clean_feature_subset(training, feature_columns),
        training["label"].astype(int).to_numpy(),
    )
    return model


def sensitivity_models(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    training = load_mimic_training(rv.FEATURE_COLS)
    specifications = {
        "vital_only_hgb": VITAL_FEATURES,
        "full_minus_temperature_hgb": NO_TEMPERATURE_FEATURES,
    }
    models = {}
    for model_name, feature_columns in specifications.items():
        print(f"Fitting post hoc sensitivity model: {model_name}", flush=True)
        models[model_name] = fit_sensitivity_model(training, feature_columns)
        joblib.dump(
            {
                "analysis_role": "post_hoc_methodological_sensitivity",
                "model": models[model_name],
                "feature_cols": feature_columns,
                "training_groups": "MIMIC-IV 2008-2019",
                "external_data_used_for_fitting_or_selection": False,
            },
            rv.OUTPUTS / f"corrected_{model_name}.joblib",
        )
    del training

    metric_rows = []
    contrast_rows = []
    for dataset in DATASETS:
        columns = META_COLUMNS + rv.FEATURE_COLS
        frame = load_source(dataset, columns)
        with np.load(prediction_path(dataset)) as data:
            validate_alignment(dataset, frame, data)
            y = data["y"].astype(np.int8)
            record_ids = data["record_id"].astype(np.int64)
            predictions = {
                "full_hgb": data["p_hist_gradient_boosting"].astype(float),
            }
        for model_name, model in models.items():
            predictions[model_name] = model.predict_proba(
                clean_feature_subset(frame, specifications[model_name])
            )[:, 1]

        _, cluster_codes = np.unique(record_ids, return_inverse=True)
        cluster_codes = cluster_codes.astype(np.int32)
        evaluators = {
            name: WeightedRankingEvaluator(
                y,
                prediction,
                cluster_codes,
                np.ones(len(y), dtype=float),
            )
            for name, prediction in predictions.items()
        }
        point = {name: evaluator.evaluate(np.ones(evaluator.n_clusters, dtype=np.int32)) for name, evaluator in evaluators.items()}
        bootstrap = {
            name: np.empty((BOOTSTRAP_REPLICATES, 3), dtype=float)
            for name in evaluators
        }
        n_clusters = next(iter(evaluators.values())).n_clusters
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
        for comparator in ["vital_only_hgb", "full_minus_temperature_hgb"]:
            for position, metric in enumerate(["auroc", "auprc", "brier"]):
                difference = bootstrap["full_hgb"][:, position] - bootstrap[comparator][:, position]
                low, high = ci.percentile_interval(difference)
                contrast_rows.append(
                    {
                        "dataset": dataset,
                        "contrast": f"full_hgb_minus_{comparator}",
                        "metric": metric,
                        "difference": point["full_hgb"][position] - point[comparator][position],
                        "ci_low": low,
                        "ci_high": high,
                        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    }
                )
        print(f"Completed post hoc sensitivity models: {dataset}", flush=True)
    return pd.DataFrame(metric_rows), pd.DataFrame(contrast_rows)


def stratified_group_split(group_codes: np.ndarray, group_event: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    calibration_group = np.zeros(len(group_event), dtype=bool)
    for event_value in [0, 1]:
        candidates = np.flatnonzero(group_event == event_value)
        rng.shuffle(candidates)
        n_selected = max(1, int(round(0.20 * len(candidates))))
        calibration_group[candidates[:n_selected]] = True
    return calibration_group[group_codes]


def repeated_recalibration(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        with np.load(prediction_path(dataset)) as data:
            y = data["y"].astype(np.int8)
            p = np.clip(data["p_hist_gradient_boosting"].astype(float), 1e-7, 1 - 1e-7)
            group_ids = data["patient_id"].astype(np.int64)
        _, group_codes = np.unique(group_ids, return_inverse=True)
        group_codes = group_codes.astype(np.int32)
        group_event = np.zeros(int(group_codes.max()) + 1, dtype=np.int8)
        np.maximum.at(group_event, group_codes, y)
        z = logit(p)
        for repeat in range(RECALIBRATION_REPEATS):
            calibration_mask = stratified_group_split(group_codes, group_event, rng)
            evaluation_mask = ~calibration_mask
            intercept_shift, _, _ = rv.calibration_fit(y[calibration_mask], p[calibration_mask])
            evaluation_y = y[evaluation_mask]
            before = p[evaluation_mask]
            after = expit(intercept_shift + z[evaluation_mask])
            metrics = {}
            for label, probability in [("before", before), ("after", after)]:
                alert = probability >= 0.05
                tp = int(np.sum(alert & (evaluation_y == 1)))
                fp = int(np.sum(alert & (evaluation_y == 0)))
                fn = int(np.sum(~alert & (evaluation_y == 1)))
                metrics[f"brier_{label}"] = float(brier_score_loss(evaluation_y, probability))
                metrics[f"sensitivity_at_0_05_{label}"] = tp / (tp + fn) if tp + fn else np.nan
                metrics[f"false_alerts_per_100_eligible_landmark_hours_at_0_05_{label}"] = fp / len(evaluation_y) * 100
            rows.append(
                {
                    "dataset": dataset,
                    "repeat": repeat + 1,
                    "n_calibration_groups": int(np.unique(group_codes[calibration_mask]).size),
                    "n_evaluation_groups": int(np.unique(group_codes[evaluation_mask]).size),
                    "intercept_shift": intercept_shift,
                    **metrics,
                    "brier_change_after_minus_before": metrics["brier_after"] - metrics["brier_before"],
                }
            )
        print(f"Completed repeated recalibration: {dataset}", flush=True)
    return pd.DataFrame(rows)


def summarize(
    balanced: pd.DataFrame,
    hourly: pd.DataFrame,
    clinical: pd.DataFrame,
    rules: pd.DataFrame,
    temperatures: pd.DataFrame,
    model_metrics: pd.DataFrame,
    model_contrasts: pd.DataFrame,
    repeated: pd.DataFrame,
) -> None:
    lines = [
        "# Methodological extension analyses",
        "",
        "All analyses in this document are post hoc robustness checks. The corrected primary HGB model remained frozen. New sensitivity models were trained only in MIMIC-IV 2008-2019 with the existing fixed HGB hyperparameters; external data were not used for fitting or selection.",
        "",
        "## Stay-balanced and fixed-hour estimands",
        "",
        "| Dataset | Stay-balanced AUROC (95% CI) | Stay-balanced AUPRC (95% CI) | Slope | Hour-6 AUROC (95% CI) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for dataset in DATASETS:
        row = balanced[balanced.dataset.eq(dataset)].iloc[0]
        hour6 = hourly[(hourly.dataset.eq(dataset)) & (hourly.index_hour.eq(6))].iloc[0]
        lines.append(
            f"| {DATASET_LABELS[dataset]} | {row.auroc:.3f} ({row.auroc_ci_low:.3f}-{row.auroc_ci_high:.3f}) | "
            f"{row.auprc:.3f} ({row.auprc_ci_low:.3f}-{row.auprc_ci_high:.3f}) | {row.calibration_slope:.3f} | "
            f"{hour6.auroc:.3f} ({hour6.auroc_ci_low:.3f}-{hour6.auroc_ci_high:.3f}) |"
        )

    lines.extend(
        [
            "",
            "## Clinical ranking baselines",
            "",
            "| Dataset | Score | AUROC (95% CI) | AUPRC (95% CI) | Landmarks |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in clinical.itertuples(index=False):
        lines.append(
            f"| {DATASET_LABELS[row.dataset]} | {row.score} | {row.auroc:.3f} ({row.auroc_ci_low:.3f}-{row.auroc_ci_high:.3f}) | "
            f"{row.auprc:.3f} ({row.auprc_ci_low:.3f}-{row.auprc_ci_high:.3f}) | {row.n_landmarks:,} |"
        )

    lines.extend(
        [
            "",
            "## Fixed clinical rules",
            "",
            "| Dataset | Rule | Sensitivity | Specificity | PPV | False alerts/100 eligible landmark-hours |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rules.itertuples(index=False):
        lines.append(
            f"| {DATASET_LABELS[row.dataset]} | {row.rule} | {row.sensitivity:.3f} | {row.specificity:.3f} | "
            f"{row.ppv:.3f} | {row.false_alerts_per_100_eligible_landmark_hours:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Reduced-feature sensitivity models",
            "",
            "| Dataset | Model | AUROC (95% CI) | AUPRC (95% CI) | Brier | Calibration slope |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in model_metrics.itertuples(index=False):
        lines.append(
            f"| {DATASET_LABELS[row.dataset]} | {row.model} | {row.auroc:.3f} ({row.auroc_ci_low:.3f}-{row.auroc_ci_high:.3f}) | "
            f"{row.auprc:.3f} ({row.auprc_ci_low:.3f}-{row.auprc_ci_high:.3f}) | {row.brier:.5f} | {row.calibration_slope:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Temperature audit",
            "",
            "| Dataset | Median | 1st-99th percentile | Below 32 C | Below 34 C |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in temperatures.itertuples(index=False):
        lines.append(
            f"| {DATASET_LABELS[row.dataset]} | {row.temperature_median:.2f} | {row.temperature_q01:.2f}-{row.temperature_q99:.2f} | "
            f"{100*row.fraction_below_32:.2f}% | {100*row.fraction_below_34:.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Repeated held-out intercept recalibration",
            "",
            "| Dataset | Intercept shift, median (IQR) | Brier change after-minus-before, median (IQR) | Splits with Brier improvement |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for dataset in DATASETS:
        subset = repeated[repeated.dataset.eq(dataset)]
        shift = subset.intercept_shift
        change = subset.brier_change_after_minus_before
        lines.append(
            f"| {DATASET_LABELS[dataset]} | {shift.median():.3f} ({shift.quantile(.25):.3f} to {shift.quantile(.75):.3f}) | "
            f"{change.median():.6f} ({change.quantile(.25):.6f} to {change.quantile(.75):.6f}) | {100*np.mean(change < 0):.0f}% |"
        )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Stay-balanced metrics target an average stay rather than an average eligible landmark. Fixed-hour analyses remove within-stay repetition at the selected hour but remain conditional on the original stay-duration eligibility. Clinical scores were evaluated as ranking scores on rows where their required measurements were available and are not calibrated probability models. Reduced-feature models and repeated recalibration were post hoc and should be reported as sensitivity analyses, not as new primary model selection.",
        ]
    )
    (rv.OUTPUTS / "corrected_methodological_extensions.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    balanced, hourly = stay_balanced_and_fixed_hour(rng)
    clinical, rules, temperatures = clinical_baselines(rng)
    model_metrics, model_contrasts = sensitivity_models(rng)
    repeated = repeated_recalibration(rng)

    balanced.to_csv(rv.OUTPUTS / "corrected_stay_balanced_metrics.csv", index=False)
    hourly.to_csv(rv.OUTPUTS / "corrected_fixed_hour_metrics.csv", index=False)
    clinical.to_csv(rv.OUTPUTS / "corrected_clinical_baseline_metrics.csv", index=False)
    rules.to_csv(rv.OUTPUTS / "corrected_clinical_rule_metrics.csv", index=False)
    temperatures.to_csv(rv.OUTPUTS / "corrected_temperature_audit.csv", index=False)
    model_metrics.to_csv(rv.OUTPUTS / "corrected_reduced_model_metrics.csv", index=False)
    model_contrasts.to_csv(rv.OUTPUTS / "corrected_reduced_model_contrasts.csv", index=False)
    repeated.to_csv(rv.OUTPUTS / "corrected_repeated_recalibration.csv", index=False)
    summarize(
        balanced,
        hourly,
        clinical,
        rules,
        temperatures,
        model_metrics,
        model_contrasts,
        repeated,
    )
    print("Methodological extensions completed.", flush=True)


if __name__ == "__main__":
    main()
