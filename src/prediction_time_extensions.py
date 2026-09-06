"""Aggregate secondary analyses for the prediction-time risk set.

All outputs are aggregate and remain local. Patient/stay-level arrays are read
from the local prediction-time analysis directory and are never copied to the
public repository.
"""

from __future__ import annotations

from pathlib import Path
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import clustered_inference as ci
import model_benchmark as mb
import run_validation as rv

ROOT = Path(os.environ.get("PREDICTION_TIME_WORK_DIR", REPO / "work" / "prediction_time"))
ANALYSIS = Path(os.environ.get("PREDICTION_TIME_OUTPUT_DIR", ROOT / "analysis"))
OUT = ANALYSIS
RNG = np.random.default_rng(20260906)
REPEATS = 100
DATASETS = ["mimic_temporal_test", "eicu_external", "sicdb_external"]
LABELS = {"mimic_temporal_test": "MIMIC-IV 2020-2022", "eicu_external": "eICU-CRD", "sicdb_external": "SICdb"}
AGENT_DIR = Path(os.environ.get("PREDICTION_TIME_AGENT_DIR", ROOT))
AGENTS = {dataset: AGENT_DIR / f"reviewer_{suffix}_first_agent.csv" for dataset, suffix in {
    "mimic_temporal_test": "mimic", "eicu_external": "eicu", "sicdb_external": "sicdb"
}.items()}


def arrays(dataset: str) -> dict[str, np.ndarray]:
    paths = list(ANALYSIS.glob(f"prediction_time_{dataset}_hist_gradient_boosting.npz"))
    if not paths:
        raise FileNotFoundError(dataset)
    with np.load(paths[0]) as d:
        return {k: d[k].copy() for k in d.files}


def group_split(record: np.ndarray, y: np.ndarray, patient: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Patient identifiers are used where available; otherwise a stay is the
    # independent grouping unit.
    group = patient.astype(np.int64).copy()
    group[group < 0] = record[group < 0]
    unique, codes = np.unique(group, return_inverse=True)
    event = np.zeros(len(unique), dtype=np.int8)
    np.maximum.at(event, codes, y.astype(np.int8))
    calibration = np.zeros(len(unique), dtype=bool)
    for value in (0, 1):
        candidates = np.flatnonzero(event == value)
        RNG.shuffle(candidates)
        calibration[candidates[: max(1, int(round(0.20 * len(candidates))))]] = True
    mask = calibration[codes]
    return mask, codes.astype(np.int32)


def threshold_fixed_rate(p: np.ndarray, rate: float) -> float:
    return float(np.quantile(p, 1.0 - rate / 100.0, method="higher"))


def threshold_sensitivity(p: np.ndarray, y: np.ndarray, target: float) -> float:
    positives = p[y == 1]
    if positives.size == 0:
        return float("inf")
    return float(np.quantile(positives, 1.0 - target, method="lower"))


def suppress(record: np.ndarray, index_hour: np.ndarray, alert: np.ndarray, hours: int = 6) -> np.ndarray:
    order = np.lexsort((index_hour, record))
    kept = np.zeros(len(alert), dtype=bool)
    last_record = None
    last_hour = -1e9
    for pos in order:
        rec = int(record[pos])
        hour = float(index_hour[pos])
        if rec != last_record:
            last_record, last_hour = rec, -1e9
        if alert[pos] and hour - last_hour >= hours:
            kept[pos] = True
            last_hour = hour
    return kept


def policy_metrics(a: dict[str, np.ndarray], eval_mask: np.ndarray, threshold: float) -> dict[str, float]:
    y = a["y"].astype(np.int8)
    p = a["p"].astype(float)
    record = a["record_id"].astype(np.int64)
    index = a["index_hour"].astype(float)
    eval_y, eval_p = y[eval_mask], p[eval_mask]
    eval_record, eval_index = record[eval_mask], index[eval_mask]
    raw = eval_p >= threshold
    raw_tp = raw & (eval_y == 1)
    raw_fp = raw & (eval_y == 0)
    # Suppression is evaluated within the held-out subset.
    emitted = suppress(eval_record, eval_index, raw)
    tp = emitted & (eval_y == 1)
    fp = emitted & (eval_y == 0)
    event_rows = eval_y == 1
    event_sensitivity = float(tp.sum() / event_rows.sum()) if event_rows.sum() else np.nan
    record_event = pd.DataFrame({"record": eval_record, "y": eval_y}).groupby("record").y.max()
    record_alert = pd.DataFrame({"record": eval_record, "alert": emitted}).groupby("record").alert.max()
    event_stay = record_event.index[record_event.eq(1)]
    event_stay_sens = float(record_alert.reindex(event_stay, fill_value=False).mean()) if len(event_stay) else np.nan
    alerted_stays = record_alert[record_alert].index
    episodes = pd.Series(emitted).groupby(eval_record).sum()
    # The event time is recoverable from every positive row as index + lead time.
    event_time = pd.Series(np.where(eval_y == 1, eval_index + a["lead_time_hours"][eval_mask], np.nan)).groupby(eval_record).min()
    first_alert = pd.Series(np.where(emitted, eval_index, np.nan)).groupby(eval_record).min()
    detected = event_time.index.intersection(first_alert.dropna().index)
    detected = detected[event_time.reindex(detected).notna()]
    lead = (event_time.reindex(detected) - first_alert.reindex(detected)).astype(float)
    return {
        "threshold": threshold,
        "raw_landmark_sensitivity": float(raw_tp.sum() / event_rows.sum()) if event_rows.sum() else np.nan,
        "raw_landmark_ppv": float(raw_tp.sum() / raw.sum()) if raw.sum() else np.nan,
        "raw_alerts_per_100_rows": float(raw.mean() * 100),
        "raw_false_alerts_per_100_rows": float(raw_fp.sum() / len(raw) * 100),
        "landmark_sensitivity": float(tp.sum() / event_rows.sum()) if event_rows.sum() else np.nan,
        "landmark_ppv": float(tp.sum() / emitted.sum()) if emitted.sum() else np.nan,
        "alerts_per_100_rows": float(emitted.mean() * 100),
        "false_episodes_per_100_rows": float(fp.sum() / len(emitted) * 100),
        "event_stay_sensitivity": event_stay_sens,
        "stays_with_at_least_one_alert_percent": float(record_alert.mean() * 100),
        "stays_with_at_least_two_alerts_percent": float((episodes >= 2).mean() * 100),
        "median_episodes_per_alerted_stay": float(episodes.reindex(alerted_stays).median()) if len(alerted_stays) else np.nan,
        "median_policy_lead_time_hours": float(lead.median()) if len(lead) else np.nan,
        "q25_policy_lead_time_hours": float(lead.quantile(.25)) if len(lead) else np.nan,
        "q75_policy_lead_time_hours": float(lead.quantile(.75)) if len(lead) else np.nan,
    }


def run_calibration_and_policy() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    brier_rows: list[dict] = []
    recal_rows: list[dict] = []
    policy_rows: list[dict] = []
    for dataset in DATASETS:
        a = arrays(dataset)
        y = a["y"].astype(np.int8)
        p = np.clip(a["p"].astype(float), 1e-7, 1 - 1e-7)
        mask_group, codes = group_split(a["record_id"], y, a["patient_id"])
        group_event = np.zeros(int(codes.max()) + 1, dtype=np.int8)
        np.maximum.at(group_event, codes, y)
        z = logit(p)
        for repeat in range(REPEATS):
            # Re-draw a stratified split for every repeat.
            cal, _ = group_split(a["record_id"], y, a["patient_id"])
            ev = ~cal
            null = float(y[ev].mean())
            b_before = brier_score_loss(y[ev], p[ev])
            b_null = brier_score_loss(y[ev], np.full(ev.sum(), null))
            intercept, full_intercept, full_slope = rv.calibration_fit(y[cal], p[cal])
            p_int = expit(intercept + z[ev])
            p_full = expit(full_intercept + full_slope * z[ev])
            brier_rows.append({"dataset": dataset, "repeat": repeat + 1, "brier_skill_uncalibrated": 1 - b_before / b_null, "brier_skill_intercept_only": 1 - brier_score_loss(y[ev], p_int) / b_null, "brier_skill_full_logistic": 1 - brier_score_loss(y[ev], p_full) / b_null})
            recal_rows.append({"dataset": dataset, "repeat": repeat + 1, "intercept_shift": intercept, "fitted_intercept": full_intercept, "fitted_slope": full_slope, "brier_before": b_before, "brier_intercept_only": brier_score_loss(y[ev], p_int), "brier_full_logistic": brier_score_loss(y[ev], p_full)})
            for strategy, threshold, target in [
                ("fixed_5_alerts_per_100_rows", threshold_fixed_rate(p[cal], 5.0), 5.0),
                ("target_80_percent_calibration_sensitivity", threshold_sensitivity(p[cal], y[cal], .80), .80),
            ]:
                cal_p = p[cal]
                cal_y = y[cal]
                cal_raw = cal_p >= threshold
                cal_tp = cal_raw & (cal_y == 1)
                cal_fp = cal_raw & (cal_y == 0)
                row = {
                    "dataset": dataset,
                    "repeat": repeat + 1,
                    "strategy": strategy,
                    "target_alerts_per_100_rows": target if strategy.startswith("fixed") else np.nan,
                    "target_calibration_sensitivity": target if strategy.startswith("target") else np.nan,
                    "calibration_raw_sensitivity": float(cal_tp.sum() / (cal_y == 1).sum()) if (cal_y == 1).sum() else np.nan,
                    "calibration_raw_alerts_per_100_rows": float(cal_raw.mean() * 100),
                    "calibration_raw_false_alerts_per_100_rows": float(cal_fp.sum() / len(cal_raw) * 100),
                }
                row.update(policy_metrics(a, ev, threshold))
                policy_rows.append(row)
        print(f"Completed recalibration and policies: {dataset}", flush=True)
    brier = pd.DataFrame(brier_rows)
    recal = pd.DataFrame(recal_rows)
    policy = pd.DataFrame(policy_rows)
    brier.to_csv(OUT / "prediction_time_brier_skill_repeated.csv", index=False)
    recal.to_csv(OUT / "prediction_time_recalibration_repeated.csv", index=False)
    policy.to_csv(OUT / "prediction_time_policy_repeated.csv", index=False)
    return brier, recal, policy


def summarize_repeated(brier: pd.DataFrame, recal: pd.DataFrame, policy: pd.DataFrame) -> None:
    rows = []
    for dataset, group in brier.groupby("dataset"):
        row = {"dataset": dataset}
        for col in ["brier_skill_uncalibrated", "brier_skill_intercept_only", "brier_skill_full_logistic"]:
            row[f"{col}_median"] = group[col].median()
            row[f"{col}_q1"] = group[col].quantile(.25)
            row[f"{col}_q3"] = group[col].quantile(.75)
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "prediction_time_brier_skill_summary.csv", index=False)
    rows = []
    for (dataset, strategy), group in policy.groupby(["dataset", "strategy"]):
        row = {"dataset": dataset, "strategy": strategy}
        for col in ["threshold", "calibration_raw_sensitivity", "calibration_raw_alerts_per_100_rows", "calibration_raw_false_alerts_per_100_rows", "raw_landmark_sensitivity", "raw_landmark_ppv", "raw_alerts_per_100_rows", "raw_false_alerts_per_100_rows", "landmark_sensitivity", "landmark_ppv", "alerts_per_100_rows", "false_episodes_per_100_rows", "event_stay_sensitivity", "stays_with_at_least_one_alert_percent", "stays_with_at_least_two_alerts_percent", "median_episodes_per_alerted_stay", "median_policy_lead_time_hours", "q25_policy_lead_time_hours", "q75_policy_lead_time_hours"]:
            row[f"{col}_median"] = group[col].median()
            row[f"{col}_q1"] = group[col].quantile(.25)
            row[f"{col}_q3"] = group[col].quantile(.75)
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "prediction_time_policy_summary.csv", index=False)


def outcome_sensitivity() -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        a = arrays(dataset)
        agent = pd.read_csv(AGENTS[dataset], low_memory=False)
        agent["record_id"] = pd.to_numeric(agent["record_id"], errors="coerce").astype("Int64")
        agent["n_first_agents"] = pd.to_numeric(agent["n_first_agents"], errors="coerce").fillna(0)
        strict = agent["n_first_agents"].eq(1) & agent["first_agents"].astype(str).str.lower().eq("norepinephrine")
        at_first = agent["norepinephrine_at_first"].astype(int).eq(1)
        strict_map = dict(zip(agent.record_id.astype(int), strict.astype(int)))
        first_map = dict(zip(agent.record_id.astype(int), at_first.astype(int)))
        for label, mapping in [("strict_norepinephrine_only", strict_map), ("norepinephrine_at_first", first_map)]:
            included = pd.Series(a["record_id"]).map(mapping).fillna(0).to_numpy(np.int8)
            y = a["y"].astype(np.int8) * included
            # Keep all negative rows and replace positive rows by the outcome-specific label.
            p = a["p"].astype(float)
            rows.append({"dataset": dataset, "analysis": label, "n_landmarks": len(y), "n_event_landmarks": int(y.sum()), "auroc": roc_auc_score(y, p), "auprc": average_precision_score(y, p), "event_prevalence": float(y.mean())})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "prediction_time_outcome_sensitivity.csv", index=False)
    return out


def run_one_hour_gap() -> pd.DataFrame:
    # Train on MIMIC development rows whose event is not within one hour of
    # the landmark; retain all negatives. Evaluate on the same restricted rows.
    mimic_path = ROOT / "mimic_prediction_time.csv"
    mimic = pd.read_csv(mimic_path, low_memory=False)
    groups = mimic.time_group.astype(str).str.strip()
    train = mimic[groups.isin(rv.MIMIC_DEVELOPMENT_GROUPS)].copy()
    selection = mimic[groups.isin(rv.MIMIC_SELECTION_GROUPS)].copy()
    temporal = mimic[groups.isin(rv.MIMIC_TEMPORAL_TEST_GROUPS)].copy()
    keep = lambda f: (f.label.eq(0) | f.lead_time_hours.gt(1.0))
    train_gap = train[keep(train)].copy()
    hgb = mb.build_model().fit(rv.feature_frame(train_gap), train_gap.label.astype(int))
    selected = pd.concat([train_gap, selection[keep(selection)]], ignore_index=True)
    hgb = mb.build_model().fit(rv.feature_frame(selected), selected.label.astype(int))
    frames = {"mimic_temporal_test": temporal, "sicdb_external": pd.read_csv(ROOT / "sicdb_prediction_time.csv", low_memory=False)}
    # eICU is scored in chunks to avoid retaining a second huge frame.
    rows = []
    for dataset, frame in frames.items():
        sub = frame[keep(frame)].copy()
        p_gap = hgb.predict_proba(rv.feature_frame(sub))[:, 1]
        p_primary = arrays(dataset)["p"]
        primary = arrays(dataset)["y"]
        # Align the primary array to the restricted frame by row order: both
        # were written from the same sorted source table.
        if dataset == "mimic_temporal_test":
            primary = primary[keep(temporal).to_numpy()]
            p_primary = p_primary[keep(temporal).to_numpy()]
        else:
            primary = primary[keep(frame).to_numpy()]
            p_primary = p_primary[keep(frame).to_numpy()]
        for name, pred in [("primary_hgb_on_gap_cohort", p_primary), ("one_hour_gap_hgb", p_gap)]:
            rows.append({"dataset": dataset, "analysis": name, "n_landmarks": len(sub), "n_event_landmarks": int(sub.label.sum()), "auroc": roc_auc_score(primary if name.startswith("primary") else sub.label, pred), "auprc": average_precision_score(primary if name.startswith("primary") else sub.label, pred)})
    # Chunked eICU scoring and alignment by row order.
    eicu_path = ROOT / "eicu_prediction_time.csv"
    eicu_rows = []
    offset = 0
    for chunk in pd.read_csv(eicu_path, chunksize=200_000, low_memory=False):
        sub = chunk[keep(chunk)].copy()
        if len(sub):
            pred_gap = hgb.predict_proba(rv.feature_frame(sub))[:, 1]
            a = arrays("eicu_external")
            # The mask is reconstructed once below to avoid relying on chunk
            # offsets after filtering.
            eicu_rows.append((sub, pred_gap))
        offset += len(chunk)
    full = pd.concat([x[0] for x in eicu_rows], ignore_index=True)
    pred_gap = np.concatenate([x[1] for x in eicu_rows])
    a = arrays("eicu_external")
    # Source and prediction arrays share the SQL ORDER BY; recreate the mask.
    all_keep = []
    for chunk in pd.read_csv(eicu_path, usecols=["label", "lead_time_hours"], chunksize=400_000, low_memory=False):
        all_keep.append((pd.to_numeric(chunk.label, errors="coerce").eq(0) | pd.to_numeric(chunk.lead_time_hours, errors="coerce").gt(1)).to_numpy())
    mask = np.concatenate(all_keep)
    p_primary = a["p"][mask]
    y_primary = a["y"][mask]
    rows.extend([
        {"dataset": "eicu_external", "analysis": "primary_hgb_on_gap_cohort", "n_landmarks": len(full), "n_event_landmarks": int(full.label.sum()), "auroc": roc_auc_score(y_primary, p_primary), "auprc": average_precision_score(y_primary, p_primary)},
        {"dataset": "eicu_external", "analysis": "one_hour_gap_hgb", "n_landmarks": len(full), "n_event_landmarks": int(full.label.sum()), "auroc": roc_auc_score(full.label, pred_gap), "auprc": average_precision_score(full.label, pred_gap)},
    ])
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "prediction_time_one_hour_gap.csv", index=False)
    return out


def main() -> None:
    brier, recal, policy = run_calibration_and_policy()
    summarize_repeated(brier, recal, policy)
    outcome_sensitivity()
    gap = run_one_hour_gap()
    print(gap.to_string(index=False), flush=True)
    print("Prediction-time extensions complete.", flush=True)


if __name__ == "__main__":
    main()
