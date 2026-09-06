"""Run the prediction-time-identifiable hourly analysis.

This is a local-only analysis script. It uses the existing harmonization and
model definitions, but rebuilds the risk set so that future ICU/unit exit does
not determine landmark eligibility. The operational endpoint is documented
pressor initiation before six hours or before the observed ICU/unit exit,
whichever occurs first. An exit without initiation is retained as an
operational non-event and is summarized separately.
"""

from __future__ import annotations

from pathlib import Path
import json
import os
import sys
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import clustered_inference as ci
import model_benchmark as mb
import run_validation as rv


ROOT = Path(os.environ.get("PREDICTION_TIME_WORK_DIR", REPO / "work" / "prediction_time"))
MIMIC_CSV = Path(os.environ.get("PREDICTION_TIME_MIMIC_CSV", ROOT / "mimic_prediction_time.csv"))
EICU_CSV = Path(os.environ.get("PREDICTION_TIME_EICU_CSV", ROOT / "eicu_prediction_time.csv"))
SICDB = Path(os.environ.get("SICDB_PATH", REPO / "data" / "sicdb"))
OUT = Path(os.environ.get("PREDICTION_TIME_OUTPUT_DIR", ROOT / "analysis"))
OUT.mkdir(parents=True, exist_ok=True)
rv.SICDB = SICDB

SEED = 20260905
BOOTSTRAP_REPLICATES = 300


def read_numeric(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    for col in ["record_id", "patient_id", "time_year", "index_hour", "label", "exit_without_event"]:
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    for col in rv.FEATURE_COLS + ["lead_time_hours", "observed_horizon_hours"]:
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def require_columns(frame: pd.DataFrame, path: Path) -> None:
    required = {"record_id", "patient_id", "index_hour", "label", "exit_without_event", "observed_horizon_hours"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    missing_features = sorted(set(rv.FEATURE_COLS).difference(frame.columns))
    if missing_features:
        raise ValueError(f"{path} is missing model features: {', '.join(missing_features)}")


def read_sicdb_cases_prediction(units: Iterable[int]) -> pd.DataFrame:
    usecols = [
        "CaseID", "PatientID", "AdmissionYear", "TimeOfStay", "ICUOffset",
        "AgeOnAdmission", "HospitalUnit", "Sex",
    ]
    cases = pd.read_csv(SICDB / "cases.csv.gz", compression="gzip", usecols=usecols, low_memory=False)
    numeric_cols = ["CaseID", "PatientID", "AdmissionYear", "TimeOfStay", "ICUOffset", "AgeOnAdmission", "HospitalUnit"]
    for col in numeric_cols:
        cases[col] = rv.numeric(cases[col])
    cases["icu_los_s"] = cases["TimeOfStay"] - cases["ICUOffset"]
    cases = cases[
        cases["AgeOnAdmission"].ge(18)
        & cases["HospitalUnit"].isin(set(units))
        & cases["icu_los_s"].gt(6 * 3600)
        & cases["CaseID"].notna()
    ].copy()
    cases["record_id"] = cases["CaseID"].astype(np.int64)
    cases["patient_id"] = cases["PatientID"].astype("Int64")
    cases["age"] = cases["AgeOnAdmission"].astype(float)
    cases["time_year"] = cases["AdmissionYear"].astype("Int64")
    cases["sex_male"] = rv.decode_sicdb_sex(cases["Sex"], rv.decode_reference_map())
    cases["unit_name"] = cases["HospitalUnit"].map({2: "INIC", 3: "CWIN", 4: "INBD", 5: "INID"})
    return cases[["record_id", "patient_id", "time_year", "icu_los_s", "ICUOffset", "age", "sex_male", "unit_name"]].reset_index(drop=True)


def build_sicdb_prediction_grid(cases: pd.DataFrame, first: pd.Series) -> pd.DataFrame:
    cases = cases.copy()
    cases["first_start_s"] = cases["record_id"].map(first)
    rows: list[dict] = []
    for row in cases.itertuples(index=False):
        first_start = float(row.first_start_s) if pd.notna(row.first_start_s) else np.nan
        for hour in range(6, 25):
            index_s = hour * 3600.0
            if float(row.icu_los_s) <= index_s:
                continue
            if pd.notna(first_start) and first_start < index_s:
                continue
            observed_end = min(index_s + 6 * 3600.0, float(row.icu_los_s))
            event = bool(pd.notna(first_start) and first_start < observed_end)
            exit_without_event = bool(float(row.icu_los_s) < index_s + 6 * 3600.0 and not event)
            rows.append(
                {
                    "dataset": "sicdb",
                    "record_id": int(row.record_id),
                    "patient_id": row.patient_id,
                    "unit_name": row.unit_name,
                    "time_year": row.time_year,
                    "index_hour": hour,
                    "age": float(row.age),
                    "sex_male": float(row.sex_male) if pd.notna(row.sex_male) else np.nan,
                    "label": int(event),
                    "lead_time_hours": (first_start / 3600.0 - hour) if event else np.nan,
                    "observed_horizon_hours": (observed_end - index_s) / 3600.0,
                    "exit_without_event": int(exit_without_event),
                }
            )
    return pd.DataFrame(rows)


def build_sicdb_prediction() -> pd.DataFrame:
    cases = read_sicdb_cases_prediction(rv.SICDB_MAIN_UNITS)
    first = rv.read_sicdb_first_pressor(cases)
    samples = build_sicdb_prediction_grid(cases, first)
    if samples.empty:
        raise RuntimeError("SICdb produced no prediction-time landmarks")
    vitals = rv.read_sicdb_vitals(cases)
    labs = rv.read_sicdb_labs(cases)
    samples = rv.add_sicdb_features(samples, pd.concat([vitals, labs], ignore_index=True))
    return samples.sort_values(["record_id", "index_hour"]).reset_index(drop=True)


def fit_logistic(frame: pd.DataFrame) -> Pipeline:
    x = rv.feature_frame(frame)
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(C=0.5, solver="lbfgs", max_iter=300, random_state=SEED)),
        ]
    ).fit(x, frame["label"].astype(int).to_numpy())


def fit_hgb(frame: pd.DataFrame):
    model = mb.build_model()
    model.fit(rv.feature_frame(frame), frame["label"].astype(int).to_numpy())
    return model


def metrics_with_cluster_ci(dataset: str, frame: pd.DataFrame, p: np.ndarray, rng: np.random.Generator) -> dict:
    y = frame["label"].astype(np.int8).to_numpy()
    record = frame["record_id"].astype(np.int64).to_numpy()
    unique, codes = np.unique(record, return_inverse=True)
    codes = codes.astype(np.int32)
    evaluator = ci.RankingEvaluator(y.astype(float), p.astype(float), codes)
    point = evaluator.evaluate(np.ones(len(unique), dtype=np.int32))
    boot = np.empty((BOOTSTRAP_REPLICATES, 3), dtype=float)
    for i in range(BOOTSTRAP_REPLICATES):
        sampled = rng.integers(0, len(unique), size=len(unique))
        multiplicity = np.bincount(sampled, minlength=len(unique)).astype(np.int32)
        boot[i] = evaluator.evaluate(multiplicity)
    record_event = np.zeros(len(unique), dtype=np.int8)
    np.maximum.at(record_event, codes, y)
    brier_null = brier_score_loss(y, np.full(len(y), y.mean()))
    positive_lead = frame.loc[y == 1, "lead_time_hours"].astype(float)
    row = {
        "dataset": dataset,
        "n_landmarks": int(len(y)),
        "n_stays": int(len(unique)),
        "n_event_stays": int(record_event.sum()),
        "n_positive_landmarks": int(y.sum()),
        "landmark_prevalence": float(y.mean()),
        "exit_without_event_landmarks": int(frame["exit_without_event"].sum()),
        "exit_without_event_percent": float(frame["exit_without_event"].mean() * 100),
        "median_observed_horizon_hours": float(frame["observed_horizon_hours"].median()),
        "q25_observed_horizon_hours": float(frame["observed_horizon_hours"].quantile(.25)),
        "q75_observed_horizon_hours": float(frame["observed_horizon_hours"].quantile(.75)),
        "auroc": float(point[0]),
        "auprc": float(point[1]),
        "brier": float(point[2]),
        "brier_skill": float(1.0 - point[2] / brier_null) if brier_null else np.nan,
        "auroc_ci_low": float(np.quantile(boot[:, 0], .025)),
        "auroc_ci_high": float(np.quantile(boot[:, 0], .975)),
        "auprc_ci_low": float(np.quantile(boot[:, 1], .025)),
        "auprc_ci_high": float(np.quantile(boot[:, 1], .975)),
        "brier_ci_low": float(np.quantile(boot[:, 2], .025)),
        "brier_ci_high": float(np.quantile(boot[:, 2], .975)),
        "median_lead_time_hours": float(positive_lead.median()) if not positive_lead.empty else np.nan,
        "q25_lead_time_hours": float(positive_lead.quantile(.25)) if not positive_lead.empty else np.nan,
        "q75_lead_time_hours": float(positive_lead.quantile(.75)) if not positive_lead.empty else np.nan,
    }
    calibration = ci.cluster_robust_calibration(y, p, codes)
    row["calibration_in_the_large"] = calibration["calibration_in_the_large"]
    row["calibration_in_the_large_ci_low"] = calibration["calibration_in_the_large_ci_low"]
    row["calibration_in_the_large_ci_high"] = calibration["calibration_in_the_large_ci_high"]
    row["calibration_slope"] = calibration["calibration_slope"]
    row["calibration_slope_ci_low"] = calibration["calibration_slope_ci_low"]
    row["calibration_slope_ci_high"] = calibration["calibration_slope_ci_high"]
    return row


def main() -> None:
    if not MIMIC_CSV.exists() or not EICU_CSV.exists():
        raise FileNotFoundError("Prediction-time MIMIC/eICU extracts are missing")
    rng = np.random.default_rng(SEED)
    mimic = read_numeric(MIMIC_CSV)
    require_columns(mimic, MIMIC_CSV)
    sicdb_path = ROOT / "sicdb_prediction_time.csv"

    groups = mimic["time_group"].astype(str).str.strip()
    development = mimic[groups.isin(rv.MIMIC_DEVELOPMENT_GROUPS)].copy()
    selection = mimic[groups.isin(rv.MIMIC_SELECTION_GROUPS)].copy()
    temporal = mimic[groups.isin(rv.MIMIC_TEMPORAL_TEST_GROUPS)].copy()
    if any(part.empty for part in [development, selection, temporal]):
        raise RuntimeError("Empty MIMIC era partition")
    if set(development.patient_id.dropna().astype(int)) & set(selection.patient_id.dropna().astype(int)):
        raise RuntimeError("MIMIC patient leakage between development and selection")

    candidates = {"logistic_regression": fit_logistic(development), "hist_gradient_boosting": fit_hgb(development)}
    selection_rows = []
    for name, model in candidates.items():
        p = model.predict_proba(rv.feature_frame(selection))[:, 1]
        selection_rows.append({"model": name, "auroc": roc_auc_score(selection.label, p), "auprc": average_precision_score(selection.label, p), "brier": brier_score_loss(selection.label, p)})
    selection_table = pd.DataFrame(selection_rows).sort_values(["auroc", "auprc"], ascending=False)
    selection_table.to_csv(OUT / "prediction_time_model_selection.csv", index=False)
    selected = str(selection_table.iloc[0]["model"])
    print(f"Selected model: {selected}", flush=True)

    final_training = pd.concat([development, selection], ignore_index=True)
    final_models = {"logistic_regression": fit_logistic(final_training), "hist_gradient_boosting": fit_hgb(final_training)}
    # Keep only the temporal test in memory while external cohorts are read one
    # at a time; the eICU landmark extract is intentionally large.
    del development, selection, final_training, mimic
    if sicdb_path.exists() and sicdb_path.stat().st_size > 0:
        sicdb = read_numeric(sicdb_path)
        require_columns(sicdb, sicdb_path)
    else:
        sicdb = build_sicdb_prediction()
        sicdb.to_csv(sicdb_path, index=False)
    print(f"MIMIC temporal={len(temporal):,}; SICdb={len(sicdb):,}", flush=True)
    frames = {"mimic_temporal_test": temporal, "sicdb_external": sicdb}
    rows = []
    for dataset, frame in frames.items():
        for name, model in final_models.items():
            p = model.predict_proba(rv.feature_frame(frame))[:, 1]
            np.savez_compressed(OUT / f"prediction_time_{dataset}_{name}.npz", record_id=frame.record_id.to_numpy(np.int64), patient_id=frame.patient_id.fillna(-1).to_numpy(np.int64), index_hour=frame.index_hour.to_numpy(np.int16), y=frame.label.to_numpy(np.int8), p=p.astype(np.float32), lead_time_hours=frame.lead_time_hours.to_numpy(np.float32), observed_horizon_hours=frame.observed_horizon_hours.to_numpy(np.float32), exit_without_event=frame.exit_without_event.to_numpy(np.int8))
            row = metrics_with_cluster_ci(dataset, frame, p, rng)
            row["model"] = name
            rows.append(row)
            print(f"{dataset} {name}: AUROC={row['auroc']:.3f}; AUPRC={row['auprc']:.3f}; Brier skill={row['brier_skill']:.3f}", flush=True)
    del sicdb
    eicu = read_numeric(EICU_CSV)
    require_columns(eicu, EICU_CSV)
    print(f"eICU={len(eicu):,}", flush=True)
    for name, model in final_models.items():
        p = model.predict_proba(rv.feature_frame(eicu))[:, 1]
        np.savez_compressed(OUT / f"prediction_time_eicu_external_{name}.npz", record_id=eicu.record_id.to_numpy(np.int64), patient_id=eicu.patient_id.fillna(-1).to_numpy(np.int64), index_hour=eicu.index_hour.to_numpy(np.int16), y=eicu.label.to_numpy(np.int8), p=p.astype(np.float32), lead_time_hours=eicu.lead_time_hours.to_numpy(np.float32), observed_horizon_hours=eicu.observed_horizon_hours.to_numpy(np.float32), exit_without_event=eicu.exit_without_event.to_numpy(np.int8))
        row = metrics_with_cluster_ci("eicu_external", eicu, p, rng)
        row["model"] = name
        rows.append(row)
        print(f"eicu_external {name}: AUROC={row['auroc']:.3f}; AUPRC={row['auprc']:.3f}; Brier skill={row['brier_skill']:.3f}", flush=True)
    frames["eicu_external"] = eicu
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "prediction_time_metrics.csv", index=False)

    exit_rows = []
    for dataset, frame in frames.items():
        exit_rows.append({
            "dataset": dataset,
            "n_stays": int(frame.record_id.nunique()),
            "n_landmarks": int(len(frame)),
            "n_exit_truncated_landmarks": int(frame.exit_without_event.sum()),
            "percent_exit_truncated_landmarks": float(frame.exit_without_event.mean() * 100),
            "median_observed_horizon_hours": float(frame.observed_horizon_hours.median()),
            "q25_observed_horizon_hours": float(frame.observed_horizon_hours.quantile(.25)),
            "q75_observed_horizon_hours": float(frame.observed_horizon_hours.quantile(.75)),
            "n_exit_truncated_stays": int(frame.groupby("record_id").exit_without_event.max().sum()),
        })
    pd.DataFrame(exit_rows).to_csv(OUT / "prediction_time_exit_summary.csv", index=False)
    manifest = {
        "analysis": "prediction_time_identifiable_hourly_operational_endpoint",
        "endpoint": "documented continuous vasopressor initiation before min(index+6h, ICU/unit exit)",
        "exit_without_event": "retained as operational non-event; not treated as death",
        "selected_model": selected,
        "training": "MIMIC-IV 2008-2016 development; 2017-2019 selection; final refit 2008-2019",
        "external_data_used_for_selection": False,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "files": {p.name: p.stat().st_size for p in sorted(OUT.iterdir())},
    }
    (OUT / "prediction_time_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote outputs to {OUT}", flush=True)


if __name__ == "__main__":
    main()
