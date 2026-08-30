"""Reproducible MIMIC-IV/eICU/SICdb validation run.

The pipeline uses hourly landmark samples from ICU hour 6 through hour 24.
At each landmark, the label is first initiation of a continuous vasopressor
within the following six hours, provided that no target vasopressor was
started before the landmark. MIMIC-IV is used for development and temporal
validation; the fitted model is then applied unchanged to eICU and SICdb.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.special import expit, logit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("MIMIC_WORK_DIR", ROOT / "work"))
OUTPUTS = Path(os.environ.get("MIMIC_OUTPUT_DIR", ROOT / "outputs"))
SICDB = Path(os.environ.get("SICDB_PATH", ROOT / "data" / "sicdb"))
PSQL = Path(os.environ.get("PSQL_PATH", "psql"))
EICU_SAMPLES = Path(os.environ.get("EICU_SAMPLES_PATH", WORK / "eicu_samples_harmonized.csv"))

MIMIC_DB = os.environ.get("MIMIC_DB", "mimiciv31")
EICU_DB = os.environ.get("EICU_DB", "eicu")
PG_HOST = os.environ.get("PGHOST", "127.0.0.1")
PG_PORT = os.environ.get("PGPORT", "5442")
PG_USER = os.environ.get("PGUSER", "postgres")
PG_PASSWORD = os.environ.get("PGPASSWORD")

SICDB_MAIN_UNITS = {3, 4}  # CWIN and INBD: high-level ICU units.
SICDB_ALL_UNITS = {2, 3, 4, 5}
SICDB_PRESSOR_IDS = {1502, 1550, 1562, 1593, 1618}

VITAL_DATA_IDS = {
    701: "sbp",
    702: "dbp",
    703: "map",
    704: "sbp",
    705: "dbp",
    706: "map",
    707: "hr",
    708: "hr",
    709: "temp",
    710: "spo2",
    719: "rr",
    724: "hr",
    2274: "rr",
    2280: "rr",
}

MIMIC_DEVELOPMENT_GROUPS = {"2008 - 2010", "2011 - 2013", "2014 - 2016"}
MIMIC_SELECTION_GROUPS = {"2017 - 2019"}
MIMIC_TEMPORAL_TEST_GROUPS = {"2020 - 2022"}

LAB_ID_TO_VAR = {
    454: "lactate",
    465: "lactate",
    657: "lactate",
    217: "hematocrit",
    183: "hematocrit",
    682: "hematocrit",
    289: "hemoglobin",
    658: "hemoglobin",
    301: "wbc",
    367: "creatinine",
    368: "creatinine",
    355: "bun",
    333: "bilirubin",
    348: "glucose",
    656: "glucose",
    538: "ph",
    688: "ph",
    456: "bicarbonate",
    451: "bicarbonate",
    666: "bicarbonate",
    463: "potassium",
    453: "potassium",
    685: "potassium",
    469: "sodium",
    455: "sodium",
    686: "sodium",
}

VITAL_VARS = ["hr", "sbp", "dbp", "map", "rr", "temp", "spo2"]
LAB_VARS = [
    "creatinine",
    "sodium",
    "potassium",
    "bicarbonate",
    "glucose",
    "bun",
    "lactate",
    "ph",
    "hemoglobin",
    "hematocrit",
    "wbc",
]

FEATURE_COLS = ["age", "sex_male", "index_hour"]
for _v in VITAL_VARS:
    FEATURE_COLS.extend([f"{_v}_last", f"{_v}_mean6", f"{_v}_min6", f"{_v}_max6"])
FEATURE_COLS.extend([f"{_v}_last" for _v in LAB_VARS])

PHYSIOLOGIC_RANGES = {
    "hr": (20.0, 250.0),
    "sbp": (30.0, 300.0),
    "dbp": (10.0, 200.0),
    "map": (20.0, 250.0),
    "rr": (2.0, 100.0),
    "temp": (25.0, 45.0),
    "spo2": (50.0, 100.0),
    "creatinine": (0.05, 30.0),
    "sodium": (80.0, 200.0),
    "potassium": (1.0, 12.0),
    "bicarbonate": (5.0, 60.0),
    "glucose": (20.0, 1000.0),
    "bun": (1.0, 200.0),
    "lactate": (0.1, 30.0),
    "ph": (6.5, 8.0),
    "hemoglobin": (3.0, 25.0),
    "hematocrit": (10.0, 80.0),
    "wbc": (0.1, 300.0),
}


def log(message: str) -> None:
    print(message, flush=True)


def ensure_dirs() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)


def run_psql(database: str, sql_file: Path, output_file: Path) -> None:
    """Run a psql extraction; stdout is the CSV emitted by the copy command."""
    psql_executable = str(PSQL)
    if not PSQL.exists():
        resolved = shutil.which(psql_executable)
        if resolved is None:
            raise FileNotFoundError(
                f"psql not found: {PSQL}. Set PSQL_PATH or add psql to PATH."
            )
        psql_executable = resolved
    log(f"Running {database} extraction from {sql_file.name} ...")
    env = os.environ.copy()
    if PG_PASSWORD:
        env["PGPASSWORD"] = PG_PASSWORD
    log_file = output_file.with_suffix(output_file.suffix + ".log")
    with output_file.open("wb") as out, log_file.open("wb") as err:
        proc = subprocess.run(
            [
                psql_executable,
                "-h",
                PG_HOST,
                "-p",
                PG_PORT,
                "-U",
                PG_USER,
                "-d",
                database,
                "-q",
                "-f",
                str(sql_file),
            ],
            stdout=out,
            stderr=err,
            env=env,
            check=False,
        )
    if proc.returncode != 0:
        tail = log_file.read_text(encoding="utf-8", errors="replace")[-5000:]
        raise RuntimeError(f"psql extraction failed for {database}:\n{tail}")
    log(f"Finished {database}: {output_file}")


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def decode_reference_map() -> Dict[int, str]:
    ref = pd.read_csv(SICDB / "d_references.csv.gz", compression="gzip", low_memory=False)
    return dict(zip(numeric(ref["ReferenceGlobalID"]).astype("Int64"), ref["ReferenceValue"].astype(str)))


def decode_sicdb_sex(values: pd.Series, reference_map: Mapping[int, str]) -> pd.Series:
    def one(value: object) -> float:
        try:
            label = str(reference_map.get(int(float(value)), value)).lower()
        except (TypeError, ValueError):
            label = str(value).lower()
        if "female" in label or "frau" in label or label in {"f", "0"}:
            return 0.0
        if "male" in label or "mann" in label or label in {"m", "1"}:
            return 1.0
        return np.nan

    return values.map(one).astype(float)


def read_sicdb_cases(units: Sequence[int]) -> pd.DataFrame:
    usecols = [
        "CaseID",
        "PatientID",
        "AdmissionYear",
        "TimeOfStay",
        "ICUOffset",
        "AgeOnAdmission",
        "HospitalUnit",
        "Sex",
    ]
    cases = pd.read_csv(SICDB / "cases.csv.gz", compression="gzip", usecols=usecols, low_memory=False)
    for col in ["CaseID", "PatientID", "AdmissionYear", "TimeOfStay", "ICUOffset", "AgeOnAdmission", "HospitalUnit"]:
        cases[col] = numeric(cases[col])
    cases["icu_los_s"] = cases["TimeOfStay"] - cases["ICUOffset"]
    cases = cases[
        cases["AgeOnAdmission"].ge(18)
        & cases["HospitalUnit"].isin(set(units))
        & cases["icu_los_s"].ge(12 * 3600)
        & cases["CaseID"].notna()
    ].copy()
    cases["record_id"] = cases["CaseID"].astype(np.int64)
    cases["patient_id"] = cases["PatientID"].astype("Int64")
    cases["age"] = cases["AgeOnAdmission"].astype(float)
    cases["time_year"] = cases["AdmissionYear"].astype("Int64")
    reference_map = decode_reference_map()
    cases["sex_male"] = decode_sicdb_sex(cases["Sex"], reference_map)
    unit_names = {2: "INIC", 3: "CWIN", 4: "INBD", 5: "INID"}
    cases["unit_name"] = cases["HospitalUnit"].map(unit_names).fillna(cases["HospitalUnit"].astype(str))
    return cases[
        [
            "record_id",
            "patient_id",
            "time_year",
            "icu_los_s",
            "ICUOffset",
            "age",
            "sex_male",
            "unit_name",
        ]
    ].reset_index(drop=True)


def read_sicdb_first_pressor(cases: pd.DataFrame) -> pd.Series:
    ids = set(cases["record_id"].astype(int))
    offset_map = cases.set_index("record_id")["ICUOffset"]
    first: Dict[int, float] = {}
    usecols = ["CaseID", "DrugID", "Offset", "IsSingleDose"]
    total_rows = 0
    for chunk in pd.read_csv(
        SICDB / "medication.csv.gz",
        compression="gzip",
        usecols=usecols,
        chunksize=250_000,
        low_memory=False,
    ):
        total_rows += len(chunk)
        chunk["CaseID"] = numeric(chunk["CaseID"])
        chunk["DrugID"] = numeric(chunk["DrugID"])
        chunk["Offset"] = numeric(chunk["Offset"])
        chunk["IsSingleDose"] = numeric(chunk["IsSingleDose"])
        chunk = chunk[
            chunk["CaseID"].isin(ids)
            & chunk["DrugID"].isin(SICDB_PRESSOR_IDS)
            & chunk["IsSingleDose"].eq(0)
            & chunk["Offset"].notna()
        ].copy()
        if chunk.empty:
            continue
        chunk["record_id"] = chunk["CaseID"].astype(np.int64)
        chunk["rel_start_s"] = chunk["Offset"] - chunk["record_id"].map(offset_map)
        grouped = chunk.groupby("record_id")["rel_start_s"].min()
        for record_id, value in grouped.items():
            value = float(value)
            if record_id not in first or value < first[record_id]:
                first[record_id] = value
    log(f"SICdb medication scan complete: {total_rows:,} rows scanned; {len(first):,} main-unit cases with continuous pressor records")
    return pd.Series(first, dtype=float, name="first_start_s")


def build_sicdb_grid(cases: pd.DataFrame, first: pd.Series) -> pd.DataFrame:
    cases = cases.copy()
    cases["first_start_s"] = cases["record_id"].map(first)
    rows: List[dict] = []
    for row in cases.itertuples(index=False):
        first_start = float(row.first_start_s) if pd.notna(row.first_start_s) else np.nan
        for hour in range(6, 25):
            if float(row.icu_los_s) < (hour + 6) * 3600:
                continue
            if pd.notna(first_start) and first_start < hour * 3600:
                continue
            label = int(pd.notna(first_start) and first_start < (hour + 6) * 3600)
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
                    "label": label,
                    "lead_time_hours": (first_start / 3600.0 - hour) if label else np.nan,
                }
            )
    return pd.DataFrame(rows)


def read_sicdb_vitals(cases: pd.DataFrame) -> pd.DataFrame:
    ids = set(cases["record_id"].astype(int))
    offset_map = cases.set_index("record_id")["ICUOffset"]
    parts: List[pd.DataFrame] = []
    usecols = ["CaseID", "DataID", "Offset", "Val"]
    total_rows = 0
    selected_rows = 0
    for chunk in pd.read_csv(
        SICDB / "data_float_h.csv.gz",
        compression="gzip",
        usecols=usecols,
        chunksize=300_000,
        low_memory=False,
    ):
        total_rows += len(chunk)
        chunk["CaseID"] = numeric(chunk["CaseID"])
        chunk["DataID"] = numeric(chunk["DataID"])
        chunk["Offset"] = numeric(chunk["Offset"])
        chunk["Val"] = numeric(chunk["Val"])
        chunk = chunk[
            chunk["CaseID"].isin(ids)
            & chunk["DataID"].isin(VITAL_DATA_IDS)
            & chunk["Offset"].notna()
            & chunk["Val"].notna()
        ].copy()
        if chunk.empty:
            continue
        chunk["record_id"] = chunk["CaseID"].astype(np.int64)
        chunk["rel_hour"] = (chunk["Offset"] - chunk["record_id"].map(offset_map)) / 3600.0
        chunk = chunk[chunk["rel_hour"].gt(0) & chunk["rel_hour"].le(24)].copy()
        if chunk.empty:
            continue
        chunk["variable"] = chunk["DataID"].astype(int).map(VITAL_DATA_IDS)
        chunk["value"] = chunk["Val"].astype(float)
        parts.append(chunk[["record_id", "rel_hour", "variable", "value"]])
        selected_rows += len(chunk)
        if total_rows % 3_000_000 < 300_000:
            log(f"SICdb signal scan: {total_rows:,} rows read; {selected_rows:,} relevant rows retained")
    if not parts:
        return pd.DataFrame(columns=["record_id", "rel_hour", "variable", "value"])
    out = pd.concat(parts, ignore_index=True)
    out = out.groupby(["record_id", "rel_hour", "variable"], as_index=False)["value"].mean()
    log(f"SICdb signal scan complete: {total_rows:,} rows read; {len(out):,} aggregated observations")
    return out


def read_sicdb_labs(cases: pd.DataFrame) -> pd.DataFrame:
    ids = set(cases["record_id"].astype(int))
    offset_map = cases.set_index("record_id")["ICUOffset"]
    id_set = set(LAB_ID_TO_VAR)
    parts: List[pd.DataFrame] = []
    usecols = ["CaseID", "LaboratoryID", "Offset", "LaboratoryValue"]
    for chunk in pd.read_csv(
        SICDB / "laboratory.csv.gz",
        compression="gzip",
        usecols=usecols,
        chunksize=250_000,
        low_memory=False,
    ):
        chunk["CaseID"] = numeric(chunk["CaseID"])
        chunk["LaboratoryID"] = numeric(chunk["LaboratoryID"])
        chunk["Offset"] = numeric(chunk["Offset"])
        chunk["LaboratoryValue"] = numeric(chunk["LaboratoryValue"])
        chunk = chunk[
            chunk["CaseID"].isin(ids)
            & chunk["LaboratoryID"].isin(id_set)
            & chunk["Offset"].notna()
            & chunk["LaboratoryValue"].notna()
        ].copy()
        if chunk.empty:
            continue
        chunk["record_id"] = chunk["CaseID"].astype(np.int64)
        chunk["rel_hour"] = (chunk["Offset"] - chunk["record_id"].map(offset_map)) / 3600.0
        chunk = chunk[chunk["rel_hour"].gt(0) & chunk["rel_hour"].le(24)].copy()
        if chunk.empty:
            continue
        chunk["variable"] = chunk["LaboratoryID"].astype(int).map(LAB_ID_TO_VAR)
        chunk["value"] = chunk["LaboratoryValue"].astype(float)
        parts.append(chunk[["record_id", "rel_hour", "variable", "value"]])
    if not parts:
        return pd.DataFrame(columns=["record_id", "rel_hour", "variable", "value"])
    out = pd.concat(parts, ignore_index=True)
    out = out.groupby(["record_id", "rel_hour", "variable"], as_index=False)["value"].mean()
    log(f"SICdb laboratory scan complete: {len(out):,} aggregated observations")
    return out


def add_sicdb_features(samples: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    samples = samples.copy().reset_index(drop=True)
    positions: Dict[int, np.ndarray] = {
        int(record_id): np.asarray(indexes, dtype=int)
        for record_id, indexes in samples.groupby("record_id", sort=False).groups.items()
    }
    for variable in VITAL_VARS:
        for suffix in ["last", "mean6", "min6", "max6"]:
            samples[f"{variable}_{suffix}"] = np.nan
        obs_var = observations[observations["variable"].eq(variable)]
        for record_id, obs_group in obs_var.groupby("record_id", sort=False):
            idx = positions.get(int(record_id))
            if idx is None:
                continue
            obs_group = obs_group.sort_values("rel_hour")
            times = obs_group["rel_hour"].to_numpy(dtype=float)
            values = obs_group["value"].to_numpy(dtype=float)
            hours = samples.loc[idx, "index_hour"].to_numpy(dtype=float)
            for target_pos, hour in zip(idx, hours):
                end = int(np.searchsorted(times, hour, side="right"))
                start = int(np.searchsorted(times, hour - 6.0, side="right"))
                window = values[start:end]
                if window.size == 0:
                    continue
                samples.at[target_pos, f"{variable}_last"] = values[end - 1]
                samples.at[target_pos, f"{variable}_mean6"] = float(np.mean(window))
                samples.at[target_pos, f"{variable}_min6"] = float(np.min(window))
                samples.at[target_pos, f"{variable}_max6"] = float(np.max(window))
    for variable in LAB_VARS:
        samples[f"{variable}_last"] = np.nan
        obs_var = observations[observations["variable"].eq(variable)]
        for record_id, obs_group in obs_var.groupby("record_id", sort=False):
            idx = positions.get(int(record_id))
            if idx is None:
                continue
            obs_group = obs_group.sort_values("rel_hour")
            times = obs_group["rel_hour"].to_numpy(dtype=float)
            values = obs_group["value"].to_numpy(dtype=float)
            hours = samples.loc[idx, "index_hour"].to_numpy(dtype=float)
            for target_pos, hour in zip(idx, hours):
                end = int(np.searchsorted(times, hour, side="right"))
                if end:
                    samples.at[target_pos, f"{variable}_last"] = values[end - 1]
    return samples


def build_sicdb_samples(all_units: bool = False) -> pd.DataFrame:
    units = SICDB_ALL_UNITS if all_units else SICDB_MAIN_UNITS
    cases = read_sicdb_cases(sorted(units))
    first = read_sicdb_first_pressor(cases)
    samples = build_sicdb_grid(cases, first)
    if samples.empty:
        raise RuntimeError("SICdb produced no eligible landmark samples")
    vitals = read_sicdb_vitals(cases)
    labs = read_sicdb_labs(cases)
    observations = pd.concat([vitals, labs], ignore_index=True)
    samples = add_sicdb_features(samples, observations)
    return samples


def read_csv_samples(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, low_memory=False)
    for col in ["record_id", "patient_id", "time_year", "index_hour", "label"]:
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    for col in FEATURE_COLS + ["lead_time_hours"]:
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def split_mimic_eras(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "time_group" not in frame.columns:
        raise ValueError("Corrected MIMIC extraction requires time_group=anchor_year_group")
    groups = frame["time_group"].astype(str).str.strip()
    development = frame[groups.isin(MIMIC_DEVELOPMENT_GROUPS)].copy()
    selection = frame[groups.isin(MIMIC_SELECTION_GROUPS)].copy()
    temporal_test = frame[groups.isin(MIMIC_TEMPORAL_TEST_GROUPS)].copy()
    if development.empty or selection.empty or temporal_test.empty:
        raise RuntimeError("One or more corrected MIMIC era partitions are empty")
    patient_sets = [set(part["patient_id"].dropna().astype(int)) for part in [development, selection, temporal_test]]
    if patient_sets[0] & patient_sets[1] or patient_sets[0] & patient_sets[2] or patient_sets[1] & patient_sets[2]:
        raise RuntimeError("Patient leakage detected across MIMIC era partitions")
    return development, selection, temporal_test


def feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.reindex(columns=FEATURE_COLS).copy()
    for col in FEATURE_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    for variable, (lower, upper) in PHYSIOLOGIC_RANGES.items():
        for col in out.columns:
            if col == variable or col.startswith(variable + "_"):
                out.loc[~out[col].between(lower, upper), col] = np.nan
    return out


def fit_model(train: pd.DataFrame) -> Pipeline:
    x = feature_frame(train)
    y = train["label"].astype(int).to_numpy()
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=0.5,
                    solver="lbfgs",
                    max_iter=300,
                    random_state=20260823,
                ),
            ),
        ]
    )
    log(f"Fitting logistic regression on {len(train):,} samples and {len(FEATURE_COLS)} raw features ...")
    model.fit(x, y)
    return model


def iter_csv_chunks(path: Path, chunksize: int = 200_000) -> Iterator[pd.DataFrame]:
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        for col in FEATURE_COLS + ["label", "record_id", "lead_time_hours"]:
            if col in chunk:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        yield chunk


def calibration_fit(y: np.ndarray, p: np.ndarray) -> Tuple[float, float, float]:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    z = logit(p)
    prevalence = float(np.mean(y))

    def offset_fn(intercept: float) -> float:
        return float(np.mean(expit(intercept + z)) - prevalence)

    try:
        intercept_fixed_slope = float(brentq(offset_fn, -40, 40))
    except ValueError:
        intercept_fixed_slope = float(logit(np.clip(prevalence, 1e-7, 1 - 1e-7)) - np.mean(z))

    def objective(theta: np.ndarray) -> Tuple[float, np.ndarray]:
        eta = theta[0] + theta[1] * z
        prob = expit(eta)
        value = -float(np.sum(y * np.log(prob + 1e-12) + (1 - y) * np.log(1 - prob + 1e-12)))
        grad = np.array([-np.sum(y - prob), -np.sum((y - prob) * z)])
        return value, grad

    result = minimize(lambda t: objective(t), np.array([0.0, 1.0]), jac=True, method="L-BFGS-B")
    intercept, slope = (float(result.x[0]), float(result.x[1])) if result.success else (np.nan, np.nan)
    return intercept_fixed_slope, intercept, slope


def decision_curve(y: np.ndarray, p: np.ndarray, dataset: str) -> List[dict]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    n = len(y)
    prevalence = float(np.mean(y))
    rows = []
    for threshold in [0.02, 0.05, 0.10, 0.20, 0.30, 0.50]:
        predicted = p >= threshold
        tp = int(np.sum(predicted & (y == 1)))
        fp = int(np.sum(predicted & (y == 0)))
        model_nb = tp / n - fp / n * threshold / (1 - threshold)
        all_nb = prevalence - (1 - prevalence) * threshold / (1 - threshold)
        rows.append(
            {
                "dataset": dataset,
                "analysis_unit": "eligible_landmark",
                "landmarks_repeated_within_stay": True,
                "n_eligible_landmark_hours": n,
                "threshold": threshold,
                "net_benefit_model": model_nb,
                "net_benefit_treat_all": all_nb,
                "net_benefit_treat_none": 0.0,
                "n_alerts": int(np.sum(predicted)),
            }
        )
    return rows


def summarize_predictions(
    dataset: str,
    y: np.ndarray,
    p: np.ndarray,
    record_ids: np.ndarray,
    lead_times: np.ndarray,
    n_missing: Mapping[str, int],
    n_feature_rows: int,
) -> Tuple[dict, List[dict], dict]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    record_ids = np.asarray(record_ids)
    lead_times = np.asarray(lead_times, dtype=float)
    record_labels = pd.DataFrame({"record_id": record_ids, "label": y}).groupby("record_id")["label"].max()
    positive_leads = lead_times[y == 1]
    fixed_offset, calibration_intercept, calibration_slope = calibration_fit(y, p)
    metrics = {
        "dataset": dataset,
        "analysis_unit": "eligible_landmark",
        "landmarks_repeated_within_stay": True,
        "n_eligible_landmark_hours": int(len(y)),
        "n_samples": int(len(y)),
        "n_records": int(pd.Series(record_ids).nunique()),
        "n_positive_records": int(record_labels.sum()),
        "record_event_prevalence": float(record_labels.mean()),
        "n_positive_samples": int(np.sum(y)),
        "sample_prevalence": float(np.mean(y)),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "auprc": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "brier": float(brier_score_loss(y, p)),
        "calibration_in_the_large": fixed_offset,
        "calibration_intercept": calibration_intercept,
        "calibration_slope": calibration_slope,
        "median_lead_time_hours": float(np.nanmedian(positive_leads)) if positive_leads.size else np.nan,
        "p25_lead_time_hours": float(np.nanpercentile(positive_leads, 25)) if positive_leads.size else np.nan,
        "p75_lead_time_hours": float(np.nanpercentile(positive_leads, 75)) if positive_leads.size else np.nan,
    }
    for threshold in [0.10, 0.20, 0.30]:
        predicted = p >= threshold
        tp = int(np.sum(predicted & (y == 1)))
        tn = int(np.sum(~predicted & (y == 0)))
        fp = int(np.sum(predicted & (y == 0)))
        fn = int(np.sum(~predicted & (y == 1)))
        metrics[f"sensitivity_at_{threshold:.2f}"] = tp / (tp + fn) if tp + fn else np.nan
        metrics[f"specificity_at_{threshold:.2f}"] = tn / (tn + fp) if tn + fp else np.nan
        metrics[f"ppv_at_{threshold:.2f}"] = tp / (tp + fp) if tp + fp else np.nan
        metrics[f"false_alerts_per_100_eligible_landmark_hours_at_{threshold:.2f}"] = fp / len(y) * 100.0
    quality = {
        "dataset": dataset,
        "n_rows": int(n_feature_rows),
    }
    for col, count in n_missing.items():
        quality[f"missing_fraction_{col}"] = count / n_feature_rows if n_feature_rows else np.nan
    return metrics, decision_curve(y, p, dataset), quality


def evaluate_frame(dataset: str, frame: pd.DataFrame, model: Pipeline) -> Tuple[dict, List[dict], dict]:
    x = feature_frame(frame)
    p = model.predict_proba(x)[:, 1]
    missing = {col: int(x[col].isna().sum()) for col in FEATURE_COLS}
    return summarize_predictions(
        dataset,
        frame["label"].astype(int).to_numpy(),
        p,
        frame["record_id"].to_numpy(),
        frame["lead_time_hours"].to_numpy(dtype=float),
        missing,
        len(frame),
    )


def evaluate_csv(dataset: str, path: Path, model: Pipeline) -> Tuple[dict, List[dict], dict]:
    ys: List[np.ndarray] = []
    ps: List[np.ndarray] = []
    records: List[np.ndarray] = []
    leads: List[np.ndarray] = []
    missing = {col: 0 for col in FEATURE_COLS}
    n_rows = 0
    for chunk in iter_csv_chunks(path):
        x = feature_frame(chunk)
        p = model.predict_proba(x)[:, 1]
        ys.append(chunk["label"].astype(int).to_numpy())
        ps.append(p)
        records.append(chunk["record_id"].to_numpy())
        leads.append(chunk["lead_time_hours"].to_numpy(dtype=float))
        for col in FEATURE_COLS:
            missing[col] += int(x[col].isna().sum())
        n_rows += len(chunk)
        if n_rows % 1_000_000 < len(chunk):
            log(f"{dataset} validation: {n_rows:,} samples scored")
    return summarize_predictions(
        dataset,
        np.concatenate(ys),
        np.concatenate(ps),
        np.concatenate(records),
        np.concatenate(leads),
        missing,
        n_rows,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_report(metrics: pd.DataFrame, quality: pd.DataFrame, all_units: bool) -> None:
    columns = list(metrics.columns)
    table_lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in metrics.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append("" if np.isnan(value) else f"{value:.6g}")
            else:
                values.append(str(value))
        table_lines.append("| " + " | ".join(values) + " |")
    lines = [
        "# Cross-database validation run",
        "",
        "Outcome: first initiation of a continuous vasopressor within 6 hours after an hourly landmark.",
        "Landmarks: ICU hour 6 through hour 24; observations were restricted to the preceding 6 hours for vital summaries and to the current ICU stay for latest laboratory values.",
        "MIMIC-IV: anchor_year_group 2008-2016 for development, 2017-2019 for model selection, and 2020-2022 for locked temporal testing.",
        "eICU-CRD: adult unit stays; positive infusiondrug records with a positive drug rate define initiation.",
        f"SICdb: {'all four units' if all_units else 'CWIN and INBD (HospitalUnit 3 and 4)'}; continuous medication rows only (IsSingleDose=0); medication and signal offsets were corrected by ICUOffset.",
        "",
        "Before imputation, physiologically implausible values were set to missing using the same prespecified ranges in all datasets. The model was a regularized logistic regression with median imputation, missingness indicators, and standardization. The same fitted model was applied to all validation datasets without recalibration.",
        "",
        "## Metrics",
        "",
        "\n".join(table_lines),
        "",
        "## Data-quality note",
        "",
        "Missing fractions are calculated before imputation. SICdb is a single Austrian tertiary-hospital source rather than a multicenter European database; this is an external geographic validation, not a multicenter validation.",
        "",
    ]
    (OUTPUTS / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument("--sicdb-all-units", action="store_true")
    args = parser.parse_args()
    ensure_dirs()

    mimic_path = WORK / "mimic_samples_anchor.csv"
    eicu_path = EICU_SAMPLES
    sicdb_path = WORK / ("sicdb_samples_all_units.csv" if args.sicdb_all_units else "sicdb_samples_main_units.csv")

    if not args.skip_extraction:
        run_psql(MIMIC_DB, WORK / "mimic_extract.sql", mimic_path)
        run_psql(EICU_DB, WORK / "eicu_extract.sql", eicu_path)
        log("Building SICdb samples from compressed source files ...")
        sicdb = build_sicdb_samples(all_units=args.sicdb_all_units)
        sicdb.to_csv(sicdb_path, index=False)
        log(f"Finished SICdb: {sicdb_path} ({len(sicdb):,} samples)")
    else:
        sicdb = read_csv_samples(sicdb_path)

    mimic = read_csv_samples(mimic_path)
    eicu = None
    log(f"MIMIC samples: {len(mimic):,}")
    log(f"eICU sample file: {eicu_path}")
    log(f"SICdb samples: {len(sicdb):,}")

    train, selection, temporal = split_mimic_eras(mimic)
    model = fit_model(train)
    joblib.dump(
        {
            "model": model,
            "feature_cols": FEATURE_COLS,
            "outcome": "continuous vasopressor initiation within 6 hours",
            "index_hours": [6, 24],
            "mimic_development_groups": sorted(MIMIC_DEVELOPMENT_GROUPS),
            "mimic_selection_groups": sorted(MIMIC_SELECTION_GROUPS),
            "mimic_temporal_test_groups": sorted(MIMIC_TEMPORAL_TEST_GROUPS),
            "physiologic_ranges": PHYSIOLOGIC_RANGES,
        },
        OUTPUTS / "mimic_frozen_model.joblib",
    )

    metric_rows: List[dict] = []
    curve_rows: List[dict] = []
    quality_rows: List[dict] = []

    for name, frame in [
        ("mimic_development", train),
        ("mimic_model_selection", selection),
        ("mimic_temporal_test", temporal),
        ("sicdb_external", sicdb),
    ]:
        m, curve, q = evaluate_frame(name, frame, model)
        metric_rows.append(m)
        curve_rows.extend(curve)
        quality_rows.append(q)

    eicu_metrics, eicu_curve, eicu_quality = evaluate_csv("eicu_external", eicu_path, model)
    metric_rows.append(eicu_metrics)
    curve_rows.extend(eicu_curve)
    quality_rows.append(eicu_quality)

    metrics = pd.DataFrame(metric_rows)
    curves = pd.DataFrame(curve_rows)
    quality = pd.DataFrame(quality_rows)
    metrics.to_csv(OUTPUTS / "model_metrics.csv", index=False)
    curves.to_csv(OUTPUTS / "decision_curve.csv", index=False)
    quality.to_csv(OUTPUTS / "feature_missingness.csv", index=False)
    pd.DataFrame({"feature": FEATURE_COLS}).to_csv(OUTPUTS / "model_features.csv", index=False)
    manifest = {
        "mimic_extract_sha256": sha256(WORK / "mimic_extract.sql"),
        "eicu_extract_sha256": sha256(WORK / "eicu_extract.sql"),
        "run_script_sha256": sha256(Path(__file__)),
        "sicdb_version_file": str(SICDB / "Documentation.pdf"),
        "sicdb_main_units": sorted(SICDB_ALL_UNITS if args.sicdb_all_units else SICDB_MAIN_UNITS),
    }
    (OUTPUTS / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_report(metrics, quality, args.sicdb_all_units)
    log("Validation complete. Outputs written to outputs/.")


if __name__ == "__main__":
    main()
