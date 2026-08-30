"""Post hoc analyses requested during the final methodological review.

Patient-level extracts and prediction arrays remain under ``work/``. Only
aggregate tables written to ``outputs/`` are suitable for public release.
None of these analyses changes the corrected primary model or its estimand.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alert_suppression_sensitivity as alert_analysis
import clustered_inference as ci
import model_benchmark as mb
import run_validation as rv


RANDOM_SEED = 20260830
BOOTSTRAP_REPLICATES = 500
POLICY_REPEATS = 100
SUPPRESSION_HOURS = 6
TARGET_ALERTS_PER_100 = 5.0
TARGET_SENSITIVITY = 0.80

DATASETS = ["mimic_temporal_test", "eicu_external", "sicdb_external"]
DATASET_LABELS = {
    "mimic_development": "MIMIC-IV 2008-2016 development",
    "mimic_selection": "MIMIC-IV 2017-2019 model selection",
    "mimic_temporal_test": "MIMIC-IV 2020-2022 temporal test",
    "eicu_external": "eICU-CRD",
    "sicdb_external": "SICdb",
}
PREDICTION_FILES = {
    dataset: rv.WORK / f"corrected_predictions_{dataset}.npz" for dataset in DATASETS
}
AGENT_FILES = {
    "mimic": rv.WORK / "reviewer_mimic_first_agent.csv",
    "eicu": rv.WORK / "reviewer_eicu_first_agent.csv",
    "sicdb": rv.WORK / "reviewer_sicdb_first_agent.csv",
}
HOUR6_FILES = {
    "mimic": rv.WORK / "reviewer_mimic_hour6_competing_exit.csv",
    "eicu": rv.WORK / "reviewer_eicu_hour6_competing_exit.csv",
    "sicdb": rv.WORK / "reviewer_sicdb_hour6_competing_exit.csv",
}

MIMIC_GROUP_TO_DATASET = {
    "2008 - 2010": "mimic_development",
    "2011 - 2013": "mimic_development",
    "2014 - 2016": "mimic_development",
    "2017 - 2019": "mimic_selection",
    "2020 - 2022": "mimic_temporal_test",
}

CLINICAL_FEATURES = [
    feature
    for feature in rv.FEATURE_COLS
    if feature not in {"age", "sex_male", "index_hour"}
]
AVAILABILITY_FEATURES = ["index_hour"] + [
    f"available_{feature}" for feature in CLINICAL_FEATURES
]
MISSINGNESS_FEATURES = [
    f"available_{feature}" for feature in CLINICAL_FEATURES
]


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError("Missing or empty required inputs:\n" + "\n".join(missing))


def load_agents(database: str) -> pd.DataFrame:
    path = AGENT_FILES[database]
    require_files([path])
    frame = pd.read_csv(path, low_memory=False)
    frame["record_id"] = pd.to_numeric(frame["record_id"], errors="raise").astype(np.int64)
    frame["n_first_agents"] = pd.to_numeric(
        frame["n_first_agents"], errors="coerce"
    ).fillna(0).astype(int)
    frame["norepinephrine_at_first"] = pd.to_numeric(
        frame["norepinephrine_at_first"], errors="coerce"
    ).fillna(0).astype(np.int8)
    return frame.drop_duplicates("record_id", keep="first")


def norepinephrine_map(database: str) -> dict[int, int]:
    agents = load_agents(database)
    return dict(
        zip(
            agents["record_id"].astype(int),
            agents["norepinephrine_at_first"].astype(int),
        )
    )


def norepinephrine_only_map(database: str) -> dict[int, int]:
    agents = load_agents(database)
    strict_only = (
        agents["n_first_agents"].eq(1)
        & agents["first_agents"].astype(str).str.strip().str.lower().eq("norepinephrine")
    )
    return dict(zip(agents["record_id"].astype(int), strict_only.astype(np.int8)))


def map_norepinephrine_labels(
    frame: pd.DataFrame, mapping: dict[int, int]
) -> np.ndarray:
    included = frame["record_id"].map(mapping).fillna(0).to_numpy(dtype=np.int8)
    return frame["label"].to_numpy(dtype=np.int8) * included


def availability_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = rv.feature_frame(frame)
    output = pd.DataFrame(index=frame.index)
    output["index_hour"] = cleaned["index_hour"].astype(float)
    for feature in CLINICAL_FEATURES:
        output[f"available_{feature}"] = cleaned[feature].notna().astype(np.uint8)
    return output.reindex(columns=AVAILABILITY_FEATURES)


def missingness_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return availability_frame(frame).reindex(columns=MISSINGNESS_FEATURES)


def grouped_event_records(record_id: np.ndarray, y: np.ndarray) -> np.ndarray:
    unique_records, record_codes = np.unique(record_id.astype(np.int64), return_inverse=True)
    event = np.zeros(len(unique_records), dtype=np.int8)
    np.maximum.at(event, record_codes, y.astype(np.int8))
    return unique_records[event == 1]


def metric_with_clustered_ci(
    dataset: str,
    analysis: str,
    y: np.ndarray,
    p: np.ndarray,
    record_id: np.ndarray,
    rng: np.random.Generator,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    y = np.asarray(y, dtype=np.int8)
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    record_id = np.asarray(record_id, dtype=np.int64)
    if len(y) != len(p) or len(y) != len(record_id):
        raise ValueError(f"Array length mismatch for {dataset}/{analysis}")
    if np.unique(y).size != 2:
        raise ValueError(f"Both outcome classes are required for {dataset}/{analysis}")

    unique_records, cluster_codes = np.unique(record_id, return_inverse=True)
    cluster_codes = cluster_codes.astype(np.int32)
    n_clusters = len(unique_records)
    evaluator = ci.RankingEvaluator(y.astype(float), p, cluster_codes)
    point = evaluator.evaluate(np.ones(n_clusters, dtype=np.int32))
    bootstrap = np.empty((BOOTSTRAP_REPLICATES, 3), dtype=float)
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled = rng.integers(0, n_clusters, size=n_clusters)
        multiplicity = np.bincount(sampled, minlength=n_clusters).astype(np.int32)
        bootstrap[replicate] = evaluator.evaluate(multiplicity)

    record_event = np.zeros(n_clusters, dtype=np.int8)
    np.maximum.at(record_event, cluster_codes, y)
    prevalence = float(np.mean(y))
    brier_null = float(brier_score_loss(y, np.full(len(y), prevalence)))
    row: dict[str, object] = {
        "dataset": dataset,
        "analysis": analysis,
        "n_landmarks": len(y),
        "n_stays": n_clusters,
        "n_event_stays": int(record_event.sum()),
        "n_positive_landmarks": int(y.sum()),
        "landmark_prevalence": prevalence,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "brier_null_evaluation_prevalence": brier_null,
        "brier_skill_evaluation_prevalence": 1.0 - point[2] / brier_null,
    }
    for position, metric in enumerate(["auroc", "auprc", "brier"]):
        low, high = ci.percentile_interval(bootstrap[:, position])
        row[f"{metric}_ci_low"] = low
        row[f"{metric}_ci_high"] = high
    row.update(ci.cluster_robust_calibration(y, p, cluster_codes))
    if extra:
        row.update(extra)
    return row


def composition_category(row: pd.Series) -> str:
    if int(row["n_first_agents"]) > 1:
        return "concurrent_multiple"
    agent = str(row["first_agents"]).strip().lower()
    if agent in {"norepinephrine", "phenylephrine", "epinephrine", "vasopressin", "dopamine"}:
        return f"{agent}_only"
    return "unmatched_or_unclassified"


def mimic_positive_record_sets() -> dict[str, set[int]]:
    result = {
        "mimic_development": set(),
        "mimic_selection": set(),
        "mimic_temporal_test": set(),
    }
    usecols = ["record_id", "time_group", "label"]
    for chunk in pd.read_csv(
        rv.WORK / "mimic_samples_anchor.csv",
        usecols=usecols,
        chunksize=250_000,
        low_memory=False,
    ):
        positive = chunk[pd.to_numeric(chunk["label"], errors="coerce").eq(1)]
        for time_group, group in positive.groupby("time_group", sort=False):
            dataset = MIMIC_GROUP_TO_DATASET.get(str(time_group).strip())
            if dataset:
                result[dataset].update(
                    pd.to_numeric(group["record_id"], errors="coerce")
                    .dropna()
                    .astype(np.int64)
                    .tolist()
                )
    return result


def run_first_agent_composition() -> None:
    require_files(
        [
            rv.WORK / "mimic_samples_anchor.csv",
            *PREDICTION_FILES.values(),
            *AGENT_FILES.values(),
        ]
    )
    positive_records = mimic_positive_record_sets()
    for dataset in ["eicu_external", "sicdb_external"]:
        with np.load(PREDICTION_FILES[dataset]) as data:
            positive_records[dataset] = set(
                grouped_event_records(data["record_id"], data["y"]).astype(int).tolist()
            )

    agent_tables = {
        "mimic_development": load_agents("mimic"),
        "mimic_selection": load_agents("mimic"),
        "mimic_temporal_test": load_agents("mimic"),
        "eicu_external": load_agents("eicu"),
        "sicdb_external": load_agents("sicdb"),
    }
    categories = [
        "norepinephrine_only",
        "phenylephrine_only",
        "epinephrine_only",
        "vasopressin_only",
        "dopamine_only",
        "concurrent_multiple",
        "unmatched_or_unclassified",
    ]
    rows: list[dict[str, object]] = []
    for dataset, records in positive_records.items():
        table = agent_tables[dataset]
        subset = table[table["record_id"].isin(records)].copy()
        subset["category"] = subset.apply(composition_category, axis=1)
        matched_ids = set(subset["record_id"].astype(int))
        unmatched = len(records - matched_ids)
        counts = subset["category"].value_counts().to_dict()
        counts["unmatched_or_unclassified"] = int(
            counts.get("unmatched_or_unclassified", 0) + unmatched
        )
        denominator = len(records)
        for category in categories:
            count = int(counts.get(category, 0))
            rows.append(
                {
                    "dataset": dataset,
                    "dataset_label": DATASET_LABELS[dataset],
                    "measure": category,
                    "mutually_exclusive": True,
                    "n_event_positive_stays": denominator,
                    "n": count,
                    "percent": 100.0 * count / denominator if denominator else np.nan,
                }
            )
        norepinephrine_included = int(
            subset["norepinephrine_at_first"].astype(int).sum()
        )
        rows.append(
            {
                "dataset": dataset,
                "dataset_label": DATASET_LABELS[dataset],
                "measure": "norepinephrine_included_at_first",
                "mutually_exclusive": False,
                "n_event_positive_stays": denominator,
                "n": norepinephrine_included,
                "percent": 100.0 * norepinephrine_included / denominator
                if denominator
                else np.nan,
            }
        )
    output = pd.DataFrame(rows)
    output.to_csv(rv.OUTPUTS / "reviewer_first_agent_composition.csv", index=False)
    print("Completed first-agent composition.", flush=True)


def run_hour6_competing_exit() -> None:
    require_files(list(HOUR6_FILES.values()))
    rng = np.random.default_rng(RANDOM_SEED + 1)
    mimic = rv.read_csv_samples(HOUR6_FILES["mimic"])
    development, selection, temporal_test = rv.split_mimic_eras(mimic)
    training = pd.concat([development, selection], ignore_index=True)
    model = mb.build_model()
    model.fit(rv.feature_frame(training), training["label"].astype(int).to_numpy())
    joblib.dump(
        {
            "analysis_role": "post_hoc_hour6_competing_exit_sensitivity",
            "model": model,
            "feature_cols": rv.FEATURE_COLS,
            "training_groups": "MIMIC-IV 2008-2019",
            "hyperparameters": model.get_params(),
            "eligibility": "alive/in ICU and untreated at ICU hour 6; no future-availability requirement",
            "outcome": "first target vasopressor before ICU exit or ICU hour 12",
            "competing_event": "ICU/unit exit before hour 12 without prior target vasopressor",
            "external_data_used_for_fitting_or_selection": False,
        },
        rv.OUTPUTS / "reviewer_hour6_competing_exit_hgb.joblib",
    )

    frames = {
        "mimic_temporal_test": temporal_test,
        "eicu_external": rv.read_csv_samples(HOUR6_FILES["eicu"]),
        "sicdb_external": rv.read_csv_samples(HOUR6_FILES["sicdb"]),
    }
    original_hour6 = pd.read_csv(rv.OUTPUTS / "corrected_fixed_hour_metrics.csv")
    rows = []
    for dataset, frame in frames.items():
        p = model.predict_proba(rv.feature_frame(frame))[:, 1]
        original = original_hour6[
            original_hour6["dataset"].eq(dataset)
            & pd.to_numeric(original_hour6["index_hour"], errors="coerce").eq(6)
        ].iloc[0]
        extra = {
            "n_competing_exits": int(frame["competing_exit"].astype(int).sum()),
            "competing_exit_percent": 100.0
            * float(frame["competing_exit"].astype(int).mean()),
            "mean_observed_horizon_hours": float(
                frame["observed_horizon_hours"].astype(float).mean()
            ),
            "complete_horizon_hour6_n": int(original["n_landmarks"]),
            "complete_horizon_hour6_auroc": float(original["auroc"]),
            "complete_horizon_hour6_auprc": float(original["auprc"]),
        }
        rows.append(
            metric_with_clustered_ci(
                dataset,
                "hour6_prediction_time_identifiable_competing_exit",
                frame["label"].to_numpy(dtype=np.int8),
                p,
                frame["record_id"].to_numpy(dtype=np.int64),
                rng,
                extra,
            )
        )
        print(f"Completed hour-6 competing-exit model: {dataset}", flush=True)
    pd.DataFrame(rows).to_csv(
        rv.OUTPUTS / "reviewer_hour6_competing_exit_metrics.csv", index=False
    )


def load_mimic_for_models() -> tuple[pd.DataFrame, pd.DataFrame]:
    mimic = rv.read_csv_samples(rv.WORK / "mimic_samples_anchor.csv")
    development, selection, temporal_test = rv.split_mimic_eras(mimic)
    return pd.concat([development, selection], ignore_index=True), temporal_test


def score_external_models(
    norepinephrine_model: object,
    norepinephrine_only_model: object,
    availability_model: object,
    missingness_model: object,
) -> dict[str, dict[str, np.ndarray]]:
    mappings = {
        "eicu_external": norepinephrine_map("eicu"),
        "sicdb_external": norepinephrine_map("sicdb"),
    }
    strict_mappings = {
        "eicu_external": norepinephrine_only_map("eicu"),
        "sicdb_external": norepinephrine_only_map("sicdb"),
    }
    result: dict[str, dict[str, np.ndarray]] = {}
    sicdb = rv.read_csv_samples(rv.WORK / "sicdb_samples_main_units.csv")
    result["sicdb_external"] = {
        "record_id": sicdb["record_id"].to_numpy(dtype=np.int64),
        "patient_id": sicdb["patient_id"].fillna(-1).to_numpy(dtype=np.int64),
        "index_hour": sicdb["index_hour"].to_numpy(dtype=np.int16),
        "y_any": sicdb["label"].to_numpy(dtype=np.int8),
        "y_norepinephrine": map_norepinephrine_labels(
            sicdb, mappings["sicdb_external"]
        ),
        "y_norepinephrine_only": map_norepinephrine_labels(
            sicdb, strict_mappings["sicdb_external"]
        ),
        "lead_time_hours": sicdb["lead_time_hours"].to_numpy(dtype=np.float32),
        "p_norepinephrine": norepinephrine_model.predict_proba(
            rv.feature_frame(sicdb)
        )[:, 1].astype(np.float32),
        "p_norepinephrine_only": norepinephrine_only_model.predict_proba(
            rv.feature_frame(sicdb)
        )[:, 1].astype(np.float32),
        "p_availability": availability_model.predict_proba(
            availability_frame(sicdb)
        )[:, 1].astype(np.float32),
        "p_missingness": missingness_model.predict_proba(
            missingness_frame(sicdb)
        )[:, 1].astype(np.float32),
    }
    del sicdb

    parts: dict[str, list[np.ndarray]] = defaultdict(list)
    mapping = mappings["eicu_external"]
    strict_mapping = strict_mappings["eicu_external"]
    n_rows = 0
    for chunk in rv.iter_csv_chunks(rv.EICU_SAMPLES):
        parts["record_id"].append(chunk["record_id"].to_numpy(dtype=np.int64))
        parts["patient_id"].append(
            chunk["patient_id"].fillna(-1).to_numpy(dtype=np.int64)
        )
        parts["index_hour"].append(chunk["index_hour"].to_numpy(dtype=np.int16))
        parts["y_any"].append(chunk["label"].to_numpy(dtype=np.int8))
        parts["y_norepinephrine"].append(
            map_norepinephrine_labels(chunk, mapping)
        )
        parts["y_norepinephrine_only"].append(
            map_norepinephrine_labels(chunk, strict_mapping)
        )
        parts["lead_time_hours"].append(
            chunk["lead_time_hours"].to_numpy(dtype=np.float32)
        )
        parts["p_norepinephrine"].append(
            norepinephrine_model.predict_proba(rv.feature_frame(chunk))[:, 1].astype(
                np.float32
            )
        )
        parts["p_norepinephrine_only"].append(
            norepinephrine_only_model.predict_proba(rv.feature_frame(chunk))[:, 1].astype(
                np.float32
            )
        )
        parts["p_availability"].append(
            availability_model.predict_proba(availability_frame(chunk))[:, 1].astype(
                np.float32
            )
        )
        parts["p_missingness"].append(
            missingness_model.predict_proba(missingness_frame(chunk))[:, 1].astype(
                np.float32
            )
        )
        n_rows += len(chunk)
        print(f"eICU post hoc model scoring: {n_rows:,} rows", flush=True)
    result["eicu_external"] = {
        name: np.concatenate(values) for name, values in parts.items()
    }
    return result


def run_sensitivity_models() -> None:
    require_files(
        [
            rv.WORK / "mimic_samples_anchor.csv",
            rv.EICU_SAMPLES,
            rv.WORK / "sicdb_samples_main_units.csv",
            *AGENT_FILES.values(),
            *PREDICTION_FILES.values(),
        ]
    )
    rng = np.random.default_rng(RANDOM_SEED + 2)
    training, temporal_test = load_mimic_for_models()
    mimic_norepinephrine = norepinephrine_map("mimic")
    mimic_norepinephrine_only = norepinephrine_only_map("mimic")
    y_training_norepinephrine = map_norepinephrine_labels(
        training, mimic_norepinephrine
    )
    y_training_norepinephrine_only = map_norepinephrine_labels(
        training, mimic_norepinephrine_only
    )

    norepinephrine_model = mb.build_model()
    norepinephrine_model.fit(
        rv.feature_frame(training), y_training_norepinephrine
    )
    norepinephrine_only_model = mb.build_model()
    norepinephrine_only_model.fit(
        rv.feature_frame(training), y_training_norepinephrine_only
    )
    availability_model = mb.build_model()
    availability_model.fit(
        availability_frame(training), training["label"].astype(int).to_numpy()
    )
    missingness_model = mb.build_model()
    missingness_model.fit(
        missingness_frame(training), training["label"].astype(int).to_numpy()
    )
    joblib.dump(
        {
            "analysis_role": "post_hoc_norepinephrine_first_sensitivity",
            "model": norepinephrine_model,
            "feature_cols": rv.FEATURE_COLS,
            "training_groups": "MIMIC-IV 2008-2019",
            "outcome": "norepinephrine included at the first target-pressor timestamp within 6 hours",
            "concurrent_first_agents": "included when norepinephrine was present",
            "external_data_used_for_fitting_or_selection": False,
        },
        rv.OUTPUTS / "reviewer_norepinephrine_first_hgb.joblib",
    )
    joblib.dump(
        {
            "analysis_role": "post_hoc_norepinephrine_only_sensitivity",
            "model": norepinephrine_only_model,
            "feature_cols": rv.FEATURE_COLS,
            "training_groups": "MIMIC-IV 2008-2019",
            "outcome": "norepinephrine as the sole agent at the first target-pressor timestamp within 6 hours",
            "concurrent_first_agents": "excluded from the positive class",
            "external_data_used_for_fitting_or_selection": False,
        },
        rv.OUTPUTS / "reviewer_norepinephrine_only_hgb.joblib",
    )
    joblib.dump(
        {
            "analysis_role": "post_hoc_availability_only_sensitivity",
            "model": availability_model,
            "feature_cols": AVAILABILITY_FEATURES,
            "training_groups": "MIMIC-IV 2008-2019",
            "predictors": "index hour plus 39 post-range-filter availability indicators",
            "external_data_used_for_fitting_or_selection": False,
        },
        rv.OUTPUTS / "reviewer_availability_only_hgb.joblib",
    )
    joblib.dump(
        {
            "analysis_role": "post_hoc_missingness_only_sensitivity",
            "model": missingness_model,
            "feature_cols": MISSINGNESS_FEATURES,
            "training_groups": "MIMIC-IV 2008-2019",
            "predictors": "39 post-range-filter availability indicators; no physiologic values or index hour",
            "external_data_used_for_fitting_or_selection": False,
        },
        rv.OUTPUTS / "reviewer_missingness_only_hgb.joblib",
    )

    scored: dict[str, dict[str, np.ndarray]] = {}
    scored["mimic_temporal_test"] = {
        "record_id": temporal_test["record_id"].to_numpy(dtype=np.int64),
        "patient_id": temporal_test["patient_id"].fillna(-1).to_numpy(dtype=np.int64),
        "index_hour": temporal_test["index_hour"].to_numpy(dtype=np.int16),
        "y_any": temporal_test["label"].to_numpy(dtype=np.int8),
        "y_norepinephrine": map_norepinephrine_labels(
            temporal_test, mimic_norepinephrine
        ),
        "y_norepinephrine_only": map_norepinephrine_labels(
            temporal_test, mimic_norepinephrine_only
        ),
        "lead_time_hours": temporal_test["lead_time_hours"].to_numpy(
            dtype=np.float32
        ),
        "p_norepinephrine": norepinephrine_model.predict_proba(
            rv.feature_frame(temporal_test)
        )[:, 1].astype(np.float32),
        "p_norepinephrine_only": norepinephrine_only_model.predict_proba(
            rv.feature_frame(temporal_test)
        )[:, 1].astype(np.float32),
        "p_availability": availability_model.predict_proba(
            availability_frame(temporal_test)
        )[:, 1].astype(np.float32),
        "p_missingness": missingness_model.predict_proba(
            missingness_frame(temporal_test)
        )[:, 1].astype(np.float32),
    }
    del training, temporal_test
    scored.update(
        score_external_models(
            norepinephrine_model,
            norepinephrine_only_model,
            availability_model,
            missingness_model,
        )
    )

    rows = []
    for dataset in DATASETS:
        arrays = scored[dataset]
        rows.append(
            metric_with_clustered_ci(
                dataset,
                "norepinephrine_first_hgb",
                arrays["y_norepinephrine"],
                arrays["p_norepinephrine"],
                arrays["record_id"],
                rng,
                {
                    "endpoint_note": "norepinephrine included at first target-pressor timestamp; concurrent first agents retained",
                },
            )
        )
        rows.append(
            metric_with_clustered_ci(
                dataset,
                "norepinephrine_only_hgb",
                arrays["y_norepinephrine_only"],
                arrays["p_norepinephrine_only"],
                arrays["record_id"],
                rng,
                {
                    "endpoint_note": "norepinephrine was the sole agent at the first target-pressor timestamp; concurrent first agents excluded from the positive class",
                },
            )
        )
        rows.append(
            metric_with_clustered_ci(
                dataset,
                "availability_only_hgb",
                arrays["y_any"],
                arrays["p_availability"],
                arrays["record_id"],
                rng,
                {
                    "endpoint_note": "any first target vasopressor; index hour plus 39 availability indicators",
                },
            )
        )
        rows.append(
            metric_with_clustered_ci(
                dataset,
                "missingness_only_hgb",
                arrays["y_any"],
                arrays["p_missingness"],
                arrays["record_id"],
                rng,
                {
                    "endpoint_note": "any first target vasopressor; 39 availability indicators without physiologic values or index hour",
                },
            )
        )
        print(f"Completed post hoc sensitivity models: {dataset}", flush=True)
    pd.DataFrame(rows).to_csv(
        rv.OUTPUTS / "reviewer_sensitivity_model_metrics.csv", index=False
    )


def stratified_group_calibration_mask(
    group_codes: np.ndarray,
    group_event: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    selected = np.zeros(len(group_event), dtype=bool)
    for event_value in [0, 1]:
        candidates = np.flatnonzero(group_event == event_value)
        rng.shuffle(candidates)
        n_selected = max(1, int(round(0.20 * len(candidates))))
        selected[candidates[:n_selected]] = True
    return selected[group_codes]


def select_alert_burden_threshold(p: np.ndarray) -> float:
    return float(
        np.quantile(
            np.asarray(p, dtype=float),
            1.0 - TARGET_ALERTS_PER_100 / 100.0,
            method="higher",
        )
    )


def select_sensitivity_threshold(y: np.ndarray, p: np.ndarray) -> float:
    positive = np.asarray(p, dtype=float)[np.asarray(y, dtype=np.int8) == 1]
    if positive.size == 0:
        raise ValueError("Calibration subset has no positive landmarks")
    return float(np.quantile(positive, 1.0 - TARGET_SENSITIVITY, method="lower"))


def policy_metrics(
    y: np.ndarray,
    p: np.ndarray,
    record_id: np.ndarray,
    index_hour: np.ndarray,
    lead_time_hours: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=float)
    record_id = np.asarray(record_id, dtype=np.int64)
    index_hour = np.asarray(index_hour, dtype=np.int16)
    lead_time_hours = np.asarray(lead_time_hours, dtype=float)
    predicted = p >= threshold
    tp = int(np.sum(predicted & (y == 1)))
    fp = int(np.sum(predicted & (y == 0)))
    fn = int(np.sum(~predicted & (y == 1)))

    emitted = alert_analysis.emitted_alerts(record_id, index_hour, predicted)
    unique_records, record_codes = np.unique(record_id, return_inverse=True)
    event_stay = np.zeros(len(unique_records), dtype=np.int8)
    detected_stay = np.zeros(len(unique_records), dtype=np.int8)
    np.maximum.at(event_stay, record_codes, y)
    np.maximum.at(
        detected_stay, record_codes, (emitted & (y == 1)).astype(np.int8)
    )
    emitted_true = emitted & (y == 1) & np.isfinite(lead_time_hours)
    emitted_true_positions = np.flatnonzero(emitted_true)
    if emitted_true_positions.size:
        _, first_positions = np.unique(
            record_id[emitted_true_positions], return_index=True
        )
        first_true_rows = emitted_true_positions[first_positions]
        detected_leads = lead_time_hours[first_true_rows]
    else:
        detected_leads = np.array([], dtype=float)

    true_episodes = int(np.sum(emitted & (y == 1)))
    false_episodes = int(np.sum(emitted & (y == 0)))
    event_stays = int(event_stay.sum())
    detected_stays = int(np.sum((event_stay == 1) & (detected_stay == 1)))
    n = len(y)
    return {
        "threshold": threshold,
        "n_evaluation_landmarks": n,
        "n_evaluation_stays": len(unique_records),
        "n_event_stays": event_stays,
        "landmark_sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "alerts_per_100_landmarks": 100.0 * (tp + fp) / n,
        "false_alerts_per_100_landmarks": 100.0 * fp / n,
        "landmark_ppv": tp / (tp + fp) if tp + fp else np.nan,
        "alert_episodes": true_episodes + false_episodes,
        "true_alert_episodes": true_episodes,
        "false_alert_episodes": false_episodes,
        "episodes_per_100_landmarks": 100.0
        * (true_episodes + false_episodes)
        / n,
        "false_episodes_per_100_landmarks": 100.0 * false_episodes / n,
        "event_stay_sensitivity": detected_stays / event_stays
        if event_stays
        else np.nan,
        "episode_ppv": true_episodes / (true_episodes + false_episodes)
        if true_episodes + false_episodes
        else np.nan,
        "n_detected_event_stays_with_lead_time": len(detected_leads),
        "policy_lead_time_median_hours": float(np.median(detected_leads))
        if detected_leads.size
        else np.nan,
        "policy_lead_time_q1_hours": float(np.quantile(detected_leads, 0.25))
        if detected_leads.size
        else np.nan,
        "policy_lead_time_q3_hours": float(np.quantile(detected_leads, 0.75))
        if detected_leads.size
        else np.nan,
    }


def summarize_repeats(
    frame: pd.DataFrame, group_columns: list[str], metric_columns: list[str]
) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(group_columns, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row["repeats"] = len(group)
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_q1"] = values.quantile(0.25)
            row[f"{metric}_median"] = values.median()
            row[f"{metric}_q3"] = values.quantile(0.75)
        rows.append(row)
    return pd.DataFrame(rows)


def run_brier_skill_and_fixed_policies() -> None:
    require_files(list(PREDICTION_FILES.values()))
    rng = np.random.default_rng(RANDOM_SEED + 3)
    brier_rows = []
    policy_rows = []
    full_rows = []
    for dataset in DATASETS:
        with np.load(PREDICTION_FILES[dataset]) as data:
            y = data["y"].astype(np.int8)
            p = np.clip(
                data["p_hist_gradient_boosting"].astype(float), 1e-7, 1 - 1e-7
            )
            record_id = data["record_id"].astype(np.int64)
            patient_id = data["patient_id"].astype(np.int64)
            index_hour = data["index_hour"].astype(np.int16)
            lead_time = data["lead_time_hours"].astype(float)

        order = np.lexsort((index_hour, record_id))
        y = y[order]
        p = p[order]
        record_id = record_id[order]
        patient_id = patient_id[order]
        index_hour = index_hour[order]
        lead_time = lead_time[order]

        prevalence = float(np.mean(y))
        brier_model = float(brier_score_loss(y, p))
        brier_null = float(
            brier_score_loss(y, np.full(len(y), prevalence, dtype=float))
        )
        full_rows.append(
            {
                "dataset": dataset,
                "n_landmarks": len(y),
                "landmark_prevalence": prevalence,
                "model_brier": brier_model,
                "null_brier_evaluation_prevalence": brier_null,
                "brier_skill_evaluation_prevalence": 1.0
                - brier_model / brier_null,
                "interpretation": "descriptive full-cohort reference; the repeated held-out reference is primary for this post hoc check",
            }
        )

        _, group_codes = np.unique(patient_id, return_inverse=True)
        group_codes = group_codes.astype(np.int32)
        group_event = np.zeros(int(group_codes.max()) + 1, dtype=np.int8)
        np.maximum.at(group_event, group_codes, y)
        z = logit(p)
        for repeat in range(1, POLICY_REPEATS + 1):
            calibration_mask = stratified_group_calibration_mask(
                group_codes, group_event, rng
            )
            evaluation_mask = ~calibration_mask
            calibration_y = y[calibration_mask]
            calibration_p = p[calibration_mask]
            evaluation_y = y[evaluation_mask]
            evaluation_p = p[evaluation_mask]

            calibration_prevalence = float(np.mean(calibration_y))
            null_probability = np.full(
                len(evaluation_y), calibration_prevalence, dtype=float
            )
            null_brier = float(brier_score_loss(evaluation_y, null_probability))
            model_brier = float(brier_score_loss(evaluation_y, evaluation_p))
            intercept_shift, _, _ = rv.calibration_fit(
                calibration_y, calibration_p
            )
            recalibrated = expit(intercept_shift + z[evaluation_mask])
            recalibrated_brier = float(
                brier_score_loss(evaluation_y, recalibrated)
            )
            brier_rows.append(
                {
                    "dataset": dataset,
                    "repeat": repeat,
                    "n_calibration_identifiers": int(
                        np.unique(group_codes[calibration_mask]).size
                    ),
                    "n_evaluation_identifiers": int(
                        np.unique(group_codes[evaluation_mask]).size
                    ),
                    "calibration_prevalence": calibration_prevalence,
                    "evaluation_prevalence": float(np.mean(evaluation_y)),
                    "intercept_shift": intercept_shift,
                    "brier_null_calibration_prevalence": null_brier,
                    "brier_model_uncalibrated": model_brier,
                    "brier_skill_uncalibrated": 1.0 - model_brier / null_brier,
                    "brier_model_intercept_recalibrated": recalibrated_brier,
                    "brier_skill_intercept_recalibrated": 1.0
                    - recalibrated_brier / null_brier,
                }
            )

            strategies = {
                "fixed_5_alerts_per_100_calibration_landmarks": select_alert_burden_threshold(
                    calibration_p
                ),
                "fixed_80_percent_calibration_sensitivity": select_sensitivity_threshold(
                    calibration_y, calibration_p
                ),
            }
            for strategy, threshold in strategies.items():
                calibration_predicted = calibration_p >= threshold
                calibration_tp = int(
                    np.sum(calibration_predicted & (calibration_y == 1))
                )
                calibration_fn = int(
                    np.sum(~calibration_predicted & (calibration_y == 1))
                )
                row = {
                    "dataset": dataset,
                    "repeat": repeat,
                    "strategy": strategy,
                    "suppression_hours": SUPPRESSION_HOURS,
                    "calibration_alerts_per_100_landmarks": 100.0
                    * float(np.mean(calibration_predicted)),
                    "calibration_landmark_sensitivity": calibration_tp
                    / (calibration_tp + calibration_fn)
                    if calibration_tp + calibration_fn
                    else np.nan,
                }
                row.update(
                    policy_metrics(
                        evaluation_y,
                        evaluation_p,
                        record_id[evaluation_mask],
                        index_hour[evaluation_mask],
                        lead_time[evaluation_mask],
                        threshold,
                    )
                )
                policy_rows.append(row)
        print(f"Completed Brier skill and fixed policies: {dataset}", flush=True)

    full = pd.DataFrame(full_rows)
    repeated = pd.DataFrame(brier_rows)
    policies = pd.DataFrame(policy_rows)
    full.to_csv(rv.OUTPUTS / "reviewer_brier_skill_full_cohort.csv", index=False)
    repeated.to_csv(
        rv.OUTPUTS / "reviewer_brier_skill_repeated_splits.csv", index=False
    )
    policies.to_csv(rv.OUTPUTS / "reviewer_fixed_policy_repeated_splits.csv", index=False)
    summarize_repeats(
        repeated,
        ["dataset"],
        [
            "brier_null_calibration_prevalence",
            "brier_model_uncalibrated",
            "brier_skill_uncalibrated",
            "brier_model_intercept_recalibrated",
            "brier_skill_intercept_recalibrated",
        ],
    ).to_csv(rv.OUTPUTS / "reviewer_brier_skill_summary.csv", index=False)
    summarize_repeats(
        policies,
        ["dataset", "strategy"],
        [
            "threshold",
            "calibration_alerts_per_100_landmarks",
            "calibration_landmark_sensitivity",
            "landmark_sensitivity",
            "alerts_per_100_landmarks",
            "false_alerts_per_100_landmarks",
            "landmark_ppv",
            "episodes_per_100_landmarks",
            "false_episodes_per_100_landmarks",
            "event_stay_sensitivity",
            "episode_ppv",
            "policy_lead_time_median_hours",
            "policy_lead_time_q1_hours",
            "policy_lead_time_q3_hours",
        ],
    ).to_csv(rv.OUTPUTS / "reviewer_fixed_policy_summary.csv", index=False)


def feature_base(feature: str) -> str | None:
    for variable in rv.PHYSIOLOGIC_RANGES:
        if feature == variable or feature.startswith(variable + "_"):
            return variable
    return None


def logical_range(feature: str) -> tuple[float, float] | None:
    if feature == "age":
        return 18.0, np.inf
    if feature == "sex_male":
        return 0.0, 1.0
    if feature == "index_hour":
        return 6.0, 24.0
    base = feature_base(feature)
    return rv.PHYSIOLOGIC_RANGES.get(base) if base else None


def update_qc_state(
    state: dict[str, dict[str, object]],
    frame: pd.DataFrame,
    sample_mask: np.ndarray,
) -> None:
    n_rows = len(frame)
    for feature in rv.FEATURE_COLS:
        values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)
        base = feature_base(feature)
        preprocessing_range = rv.PHYSIOLOGIC_RANGES.get(base) if base else None
        if preprocessing_range:
            low, high = preprocessing_range
            rejected = finite & ((values < low) | (values > high))
        else:
            rejected = np.zeros(n_rows, dtype=bool)
        accepted = finite & ~rejected
        check_range = logical_range(feature)
        if check_range:
            low, high = check_range
            violations = finite & ((values < low) | (values > high))
        else:
            violations = np.zeros(n_rows, dtype=bool)
        entry = state[feature]
        entry["n_rows"] = int(entry.get("n_rows", 0)) + n_rows
        entry["input_nonmissing_n"] = int(entry.get("input_nonmissing_n", 0)) + int(
            finite.sum()
        )
        entry["python_range_rejected_n"] = int(
            entry.get("python_range_rejected_n", 0)
        ) + int(rejected.sum())
        entry["accepted_nonmissing_n"] = int(
            entry.get("accepted_nonmissing_n", 0)
        ) + int(accepted.sum())
        entry["clean_range_violation_n"] = int(
            entry.get("clean_range_violation_n", 0)
        ) + int(violations.sum())
        if accepted.any():
            accepted_values = values[accepted]
            entry["accepted_minimum"] = min(
                float(entry.get("accepted_minimum", np.inf)),
                float(np.min(accepted_values)),
            )
            entry["accepted_maximum"] = max(
                float(entry.get("accepted_maximum", -np.inf)),
                float(np.max(accepted_values)),
            )
        sampled = values[accepted & sample_mask].astype(np.float32)
        if sampled.size:
            entry.setdefault("samples", []).append(sampled)


def qc_rows(
    cohort: str,
    state: dict[str, dict[str, object]],
    quantile_basis: str,
    upstream_filtering: str,
) -> list[dict[str, object]]:
    rows = []
    for feature in rv.FEATURE_COLS:
        entry = state[feature]
        sample_parts = entry.get("samples", [])
        sample = (
            np.concatenate(sample_parts).astype(float)
            if sample_parts
            else np.array([], dtype=float)
        )
        n_rows = int(entry["n_rows"])
        accepted = int(entry["accepted_nonmissing_n"])
        row: dict[str, object] = {
            "cohort": cohort,
            "cohort_label": DATASET_LABELS.get(cohort, cohort),
            "feature": feature,
            "n_rows": n_rows,
            "input_nonmissing_n": int(entry["input_nonmissing_n"]),
            "input_nonmissing_percent": 100.0
            * int(entry["input_nonmissing_n"])
            / n_rows,
            "python_range_rejected_n": int(entry["python_range_rejected_n"]),
            "accepted_nonmissing_n": accepted,
            "missing_after_preprocessing_n": n_rows - accepted,
            "missing_after_preprocessing_percent": 100.0
            * (n_rows - accepted)
            / n_rows,
            "clean_range_violation_n": int(entry["clean_range_violation_n"]),
            "quantile_basis": quantile_basis,
            "quantile_sample_n": len(sample),
            "upstream_filtering": upstream_filtering,
        }
        if sample.size:
            row.update(
                {
                    "minimum": float(entry["accepted_minimum"]),
                    "q01": float(np.quantile(sample, 0.01)),
                    "q25": float(np.quantile(sample, 0.25)),
                    "median": float(np.quantile(sample, 0.50)),
                    "q75": float(np.quantile(sample, 0.75)),
                    "q99": float(np.quantile(sample, 0.99)),
                    "maximum": float(entry["accepted_maximum"]),
                }
            )
        else:
            row.update(
                {
                    key: np.nan
                    for key in ["minimum", "q01", "q25", "median", "q75", "q99", "maximum"]
                }
            )
        rows.append(row)
    return rows


def run_predictor_qc() -> None:
    require_files(
        [
            rv.WORK / "mimic_samples_anchor.csv",
            rv.EICU_SAMPLES,
            rv.WORK / "sicdb_samples_main_units.csv",
        ]
    )
    rng = np.random.default_rng(RANDOM_SEED + 4)
    states: dict[str, dict[str, dict[str, object]]] = {
        cohort: {feature: {} for feature in rv.FEATURE_COLS}
        for cohort in [
            "mimic_development",
            "mimic_selection",
            "mimic_temporal_test",
            "eicu_external",
            "sicdb_external",
        ]
    }
    usecols = ["time_group", *rv.FEATURE_COLS]
    for chunk in pd.read_csv(
        rv.WORK / "mimic_samples_anchor.csv",
        usecols=usecols,
        chunksize=200_000,
        low_memory=False,
    ):
        groups = chunk["time_group"].astype(str).str.strip().map(MIMIC_GROUP_TO_DATASET)
        for cohort in [
            "mimic_development",
            "mimic_selection",
            "mimic_temporal_test",
        ]:
            subset = chunk[groups.eq(cohort)].reset_index(drop=True)
            if subset.empty:
                continue
            sample_mask = rng.random(len(subset)) < 0.05
            update_qc_state(states[cohort], subset, sample_mask)
    print("Predictor QC: MIMIC-IV complete", flush=True)

    eicu_rows = 0
    for chunk in pd.read_csv(
        rv.EICU_SAMPLES,
        usecols=rv.FEATURE_COLS,
        chunksize=200_000,
        low_memory=False,
    ):
        sample_mask = rng.random(len(chunk)) < 0.05
        update_qc_state(states["eicu_external"], chunk, sample_mask)
        eicu_rows += len(chunk)
        print(f"Predictor QC: eICU {eicu_rows:,} rows", flush=True)

    sicdb = pd.read_csv(
        rv.WORK / "sicdb_samples_main_units.csv",
        usecols=rv.FEATURE_COLS,
        low_memory=False,
    )
    update_qc_state(
        states["sicdb_external"], sicdb, np.ones(len(sicdb), dtype=bool)
    )
    del sicdb

    rows = []
    for cohort, state in states.items():
        if cohort == "eicu_external":
            upstream = (
                "eICU source-observation range filtering occurred in SQL before aggregation; "
                "counts here start at the harmonized model-input table"
            )
        else:
            upstream = (
                "counts start at the harmonized model-input table; common Python ranges were rechecked"
            )
        basis = (
            "exact accepted values for extrema and quantiles"
            if cohort == "sicdb_external"
            else "exact accepted-value extrema; deterministic seeded 5% row sample for quantiles"
        )
        rows.extend(qc_rows(cohort, state, basis, upstream))
    pd.DataFrame(rows).to_csv(
        rv.OUTPUTS / "reviewer_predictor_qc_42_features.csv", index=False
    )
    print("Completed 42-predictor QC.", flush=True)


def random_effects_reml(
    estimates: np.ndarray, variances: np.ndarray
) -> dict[str, float]:
    estimates = np.asarray(estimates, dtype=float)
    variances = np.asarray(variances, dtype=float)
    valid = np.isfinite(estimates) & np.isfinite(variances) & (variances > 0)
    estimates = estimates[valid]
    variances = variances[valid]
    k = len(estimates)
    if k < 3:
        raise ValueError("At least three valid hospitals are required")

    def objective(tau2: float) -> float:
        weights = 1.0 / (variances + tau2)
        pooled = float(np.sum(weights * estimates) / np.sum(weights))
        residual = float(np.sum(weights * (estimates - pooled) ** 2))
        return 0.5 * (
            float(np.sum(np.log(variances + tau2)))
            + float(np.log(np.sum(weights)))
            + residual
        )

    upper = max(float(np.var(estimates, ddof=1)) * 20.0, 1e-6)
    optimized = minimize_scalar(objective, bounds=(0.0, upper), method="bounded")
    tau2 = max(0.0, float(optimized.x))
    weights = 1.0 / (variances + tau2)
    pooled = float(np.sum(weights * estimates) / np.sum(weights))
    pooled_se = float(np.sqrt(1.0 / np.sum(weights)))

    fixed_weights = 1.0 / variances
    fixed = float(np.sum(fixed_weights * estimates) / np.sum(fixed_weights))
    q = float(np.sum(fixed_weights * (estimates - fixed) ** 2))
    df = k - 1
    i2 = max(0.0, (q - df) / q) * 100.0 if q > 0 else 0.0
    prediction_se = float(np.sqrt(tau2 + pooled_se**2))
    return {
        "n_hospitals": k,
        "pooled": pooled,
        "pooled_se": pooled_se,
        "ci_low": pooled - 1.96 * pooled_se,
        "ci_high": pooled + 1.96 * pooled_se,
        "tau_squared": tau2,
        "q": q,
        "q_df": df,
        "i_squared_percent": i2,
        "prediction_interval_low": pooled - 1.96 * prediction_se,
        "prediction_interval_high": pooled + 1.96 * prediction_se,
    }


def run_hospital_random_effects() -> None:
    require_files(
        [
            PREDICTION_FILES["eicu_external"],
            rv.WORK / "eicu_site_map.csv",
            rv.OUTPUTS / "corrected_eicu_hospital_metrics.csv",
        ]
    )
    eligible_hospitals = pd.read_csv(
        rv.OUTPUTS / "corrected_eicu_hospital_metrics.csv",
        usecols=["hospital_id"],
    )["hospital_id"].astype(np.int64)
    with np.load(PREDICTION_FILES["eicu_external"]) as data:
        y = data["y"].astype(np.int8)
        p = data["p_hist_gradient_boosting"].astype(float)
        record_id = data["record_id"].astype(np.int64)
        patient_id = data["patient_id"].astype(np.int64)
    site_map = pd.read_csv(
        rv.WORK / "eicu_site_map.csv", usecols=["record_id", "hospital_id"]
    ).drop_duplicates("record_id")
    hospital_lookup = site_map.set_index("record_id")["hospital_id"]
    hospital_id = pd.Series(record_id).map(hospital_lookup).to_numpy()
    if pd.isna(hospital_id).any():
        raise RuntimeError("Missing eICU hospital mapping")
    hospital_id = hospital_id.astype(np.int64)

    rows = []
    for hospital in sorted(eligible_hospitals.unique()):
        mask = hospital_id == hospital
        hospital_patients = patient_id[mask]
        _, codes = np.unique(hospital_patients, return_inverse=True)
        calibration = ci.cluster_robust_calibration(
            y[mask], p[mask], codes.astype(np.int32)
        )
        citl_se = (
            calibration["calibration_in_the_large_ci_high"]
            - calibration["calibration_in_the_large_ci_low"]
        ) / 3.92
        slope_se = (
            calibration["calibration_slope_ci_high"]
            - calibration["calibration_slope_ci_low"]
        ) / 3.92
        rows.append(
            {
                "hospital_id": int(hospital),
                "n_landmarks": int(mask.sum()),
                "n_patienthealthsystemstayid_clusters": int(np.unique(codes).size),
                **calibration,
                "calibration_in_the_large_se": citl_se,
                "calibration_slope_se": slope_se,
                "log_calibration_slope": float(
                    np.log(calibration["calibration_slope"])
                ),
                "log_calibration_slope_se": slope_se
                / calibration["calibration_slope"],
            }
        )
    hospitals = pd.DataFrame(rows)
    hospitals.to_csv(
        rv.OUTPUTS / "reviewer_eicu_hospital_calibration_cluster_robust.csv",
        index=False,
    )

    summaries = []
    citl = random_effects_reml(
        hospitals["calibration_in_the_large"].to_numpy(),
        hospitals["calibration_in_the_large_se"].to_numpy() ** 2,
    )
    summaries.append(
        {
            "metric": "calibration_in_the_large",
            "analysis_scale": "original",
            **citl,
        }
    )
    slope_log = random_effects_reml(
        hospitals["log_calibration_slope"].to_numpy(),
        hospitals["log_calibration_slope_se"].to_numpy() ** 2,
    )
    summaries.append(
        {
            "metric": "calibration_slope",
            "analysis_scale": "log; pooled values back-transformed",
            **{
                key: (np.exp(value) if key in {"pooled", "ci_low", "ci_high", "prediction_interval_low", "prediction_interval_high"} else value)
                for key, value in slope_log.items()
            },
        }
    )
    pd.DataFrame(summaries).to_csv(
        rv.OUTPUTS / "reviewer_eicu_hospital_random_effects_summary.csv",
        index=False,
    )
    print("Completed eICU hospital random-effects calibration analysis.", flush=True)


def fmt_interval(row: pd.Series, metric: str, digits: int = 3) -> str:
    return (
        f"{row[metric]:.{digits}f} "
        f"({row[f'{metric}_ci_low']:.{digits}f}-{row[f'{metric}_ci_high']:.{digits}f})"
    )


def run_summary() -> None:
    required = [
        rv.OUTPUTS / "reviewer_first_agent_composition.csv",
        rv.OUTPUTS / "reviewer_hour6_competing_exit_metrics.csv",
        rv.OUTPUTS / "reviewer_sensitivity_model_metrics.csv",
        rv.OUTPUTS / "reviewer_brier_skill_summary.csv",
        rv.OUTPUTS / "reviewer_fixed_policy_summary.csv",
        rv.OUTPUTS / "reviewer_predictor_qc_42_features.csv",
        rv.OUTPUTS / "reviewer_eicu_hospital_random_effects_summary.csv",
    ]
    require_files(required)
    hour6 = pd.read_csv(required[1])
    models = pd.read_csv(required[2])
    brier = pd.read_csv(required[3])
    policy = pd.read_csv(required[4])
    hospital = pd.read_csv(required[6])

    lines = [
        "# Reviewer-priority post hoc analyses",
        "",
        "All analyses below were conducted after the primary results were known. They do not replace the corrected primary model or convert the study into a preregistered analysis.",
        "",
        "## Prediction-time-identifiable hour-6 analysis",
        "",
        "Eligibility was determined at ICU hour 6 without requiring future ICU presence. Target-pressor initiation before hour 12 or ICU exit was the event; earlier ICU exit without initiation was retained as a competing outcome.",
        "",
        "| Dataset | Stays | Events | Competing exits | AUROC (95% CI) | AUPRC (95% CI) | Brier skill |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in hour6.itertuples(index=False):
        lines.append(
            f"| {DATASET_LABELS[row.dataset]} | {row.n_stays:,} | {row.n_event_stays:,} | "
            f"{row.n_competing_exits:,} ({row.competing_exit_percent:.1f}%) | "
            f"{row.auroc:.3f} ({row.auroc_ci_low:.3f}-{row.auroc_ci_high:.3f}) | "
            f"{row.auprc:.3f} ({row.auprc_ci_low:.3f}-{row.auprc_ci_high:.3f}) | "
            f"{row.brier_skill_evaluation_prevalence:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Outcome and measurement-process sensitivities",
            "",
            "| Dataset | Analysis | Event stays | AUROC (95% CI) | AUPRC (95% CI) | Brier |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in models.itertuples(index=False):
        lines.append(
            f"| {DATASET_LABELS[row.dataset]} | {row.analysis} | {row.n_event_stays:,} | "
            f"{row.auroc:.3f} ({row.auroc_ci_low:.3f}-{row.auroc_ci_high:.3f}) | "
            f"{row.auprc:.3f} ({row.auprc_ci_low:.3f}-{row.auprc_ci_high:.3f}) | {row.brier:.5f} |"
        )

    lines.extend(
        [
            "",
            "## Repeated held-out Brier skill",
            "",
            "The reference forecast was the event prevalence estimated in each identifier-disjoint 20% calibration subset and applied unchanged to its 80% evaluation subset.",
            "",
            "| Dataset | Uncalibrated Brier skill, median (IQR) | Intercept-recalibrated Brier skill, median (IQR) |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in brier.itertuples(index=False):
        lines.append(
            f"| {DATASET_LABELS[row.dataset]} | {row.brier_skill_uncalibrated_median:.3f} "
            f"({row.brier_skill_uncalibrated_q1:.3f} to {row.brier_skill_uncalibrated_q3:.3f}) | "
            f"{row.brier_skill_intercept_recalibrated_median:.3f} "
            f"({row.brier_skill_intercept_recalibrated_q1:.3f} to {row.brier_skill_intercept_recalibrated_q3:.3f}) |"
        )

    lines.extend(
        [
            "",
            "## Fixed operating policies",
            "",
            "Thresholds were selected in each calibration subset to target either five alerts per 100 eligible landmark rows or 80% landmark sensitivity, then applied unchanged to the evaluation subset. Policy lead time is the interval from the first emitted true-positive alert to the first target-pressor timestamp among detected event stays.",
            "",
            "| Dataset | Strategy | Sensitivity | Alerts/100 rows | False episodes/100 rows | Event-stay sensitivity | Lead time, h |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in policy.itertuples(index=False):
        strategy = (
            "5 alerts/100 rows"
            if row.strategy.startswith("fixed_5")
            else "80% calibration sensitivity"
        )
        lines.append(
            f"| {DATASET_LABELS[row.dataset]} | {strategy} | {row.landmark_sensitivity_median:.3f} | "
            f"{row.alerts_per_100_landmarks_median:.2f} | {row.false_episodes_per_100_landmarks_median:.2f} | "
            f"{row.event_stay_sensitivity_median:.3f} | {row.policy_lead_time_median_hours_median:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Hospital-level calibration model",
            "",
            "The 73 eligible eICU hospitals were synthesized with descriptive random-effects models using patienthealthsystemstayid-clustered within-hospital standard errors. CITL was analyzed on its original scale and calibration slope on the log scale. This is not prospective new-hospital recalibration.",
            "",
            "| Metric | Pooled estimate (95% CI) | Tau-squared | I-squared | 95% prediction interval |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in hospital.itertuples(index=False):
        lines.append(
            f"| {row.metric} | {row.pooled:.3f} ({row.ci_low:.3f}-{row.ci_high:.3f}) | "
            f"{row.tau_squared:.4f} | {row.i_squared_percent:.1f}% | "
            f"{row.prediction_interval_low:.3f}-{row.prediction_interval_high:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Predictor-QC boundary",
            "",
            "The 42-predictor long table reports model-input nonmissingness, post-Python-range missingness, range violations, and accepted-value distributions in five cohorts. Quantiles use an exact SICdb calculation and a seeded 5% row sample for the larger MIMIC-IV and eICU cohorts. eICU source-observation range checks occurred in SQL before aggregation, so the table cannot reconstruct counts of observations rejected upstream.",
            "",
            "## Interpretation boundary",
            "",
            "Equal-total-stay evaluation estimates performance for an average stay; the primary landmark-weighted analysis estimates performance for an average eligible prediction opportunity and gives more influence to stays contributing more landmarks. Availability-only performance can reveal use of measurement patterns but is not a full measurement-process model because recency and sampling frequency were not included.",
        ]
    )
    (rv.OUTPUTS / "reviewer_priority_extensions_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("Wrote reviewer-priority summary.", flush=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_public_release() -> None:
    names = [
        "reviewer_first_agent_composition.csv",
        "reviewer_hour6_competing_exit_metrics.csv",
        "reviewer_sensitivity_model_metrics.csv",
        "reviewer_brier_skill_full_cohort.csv",
        "reviewer_brier_skill_repeated_splits.csv",
        "reviewer_brier_skill_summary.csv",
        "reviewer_fixed_policy_repeated_splits.csv",
        "reviewer_fixed_policy_summary.csv",
        "reviewer_predictor_qc_42_features.csv",
        "reviewer_eicu_hospital_calibration_cluster_robust.csv",
        "reviewer_eicu_hospital_random_effects_summary.csv",
        "reviewer_priority_extensions_summary.md",
        "reviewer_input_manifest.json",
    ]
    sources = [rv.OUTPUTS / name for name in names]
    require_files(sources)
    destination = rv.ROOT / "results" / "reviewer_priority_2026-08-30"
    destination.mkdir(parents=True, exist_ok=True)
    for source in sources:
        shutil.copy2(source, destination / source.name)
    readme = "\n".join(
        [
            "# Reviewer-priority aggregate results",
            "",
            "These files contain disclosure-reviewed aggregate results for post hoc analyses dated 30 August 2026.",
            "",
            "The analyses do not replace the corrected primary model or constitute prospective preregistration. Patient-, stay-, landmark-, or row-level extracts, prediction arrays, and model objects are excluded. The input manifest records local file hashes only and does not redistribute its patient-level inputs.",
            "",
            "Hospital-level aggregates require final author disclosure review before journal submission or further redistribution.",
            "",
        ]
    )
    (destination / "README.md").write_text(readme, encoding="utf-8")
    public_files = sorted(path for path in destination.iterdir() if path.name != "manifest.json")
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_status": "post_hoc_reviewer_priority_aggregate_release",
        "contains_patient_level_data": False,
        "files": [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in public_files
        ],
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Wrote public aggregate release: {destination}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=[
            "all",
            "composition",
            "hour6",
            "models",
            "policy",
            "qc",
            "hospital",
            "summary",
            "public",
        ],
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    rv.ensure_dirs()
    stage = parse_args().stage
    functions = {
        "composition": run_first_agent_composition,
        "hour6": run_hour6_competing_exit,
        "models": run_sensitivity_models,
        "policy": run_brier_skill_and_fixed_policies,
        "qc": run_predictor_qc,
        "hospital": run_hospital_random_effects,
        "summary": run_summary,
        "public": run_public_release,
    }
    if stage == "all":
        for name in [
            "composition",
            "hour6",
            "models",
            "policy",
            "qc",
            "hospital",
            "summary",
            "public",
        ]:
            functions[name]()
    else:
        functions[stage]()


if __name__ == "__main__":
    main()
