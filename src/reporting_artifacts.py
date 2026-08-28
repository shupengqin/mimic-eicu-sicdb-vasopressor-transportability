"""Cohort, missingness, provenance, and outcome-harmonization artifacts."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import platform
import sys

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_validation as rv


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_stays(frame: pd.DataFrame, dataset: str) -> dict:
    stay = frame.groupby("record_id", sort=False).agg(
        patient_id=("patient_id", "first"),
        age=("age", "first"),
        sex_male=("sex_male", "first"),
        event=("label", "max"),
        landmarks=("label", "size"),
    )
    return {
        "dataset": dataset,
        "n_stays": len(stay),
        "n_patients": int(stay.patient_id.nunique()),
        "age_median": float(stay.age.median()),
        "age_q1": float(stay.age.quantile(0.25)),
        "age_q3": float(stay.age.quantile(0.75)),
        "male_percent": float(stay.sex_male.mean() * 100),
        "event_positive_stays": int(stay.event.sum()),
        "event_positive_stays_percent": float(stay.event.mean() * 100),
        "n_landmarks": int(stay.landmarks.sum()),
        "positive_landmarks": int(frame.label.sum()),
        "positive_landmarks_percent": float(frame.label.mean() * 100),
    }


def summarize_eicu_stays() -> dict:
    parts = []
    usecols = ["record_id", "patient_id", "age", "sex_male", "label"]
    for chunk in pd.read_csv(rv.EICU_SAMPLES, usecols=usecols, chunksize=200_000):
        for column in usecols:
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
        part = chunk.groupby("record_id", sort=False).agg(
            patient_id=("patient_id", "first"),
            age=("age", "first"),
            sex_male=("sex_male", "first"),
            event=("label", "max"),
            landmarks=("label", "size"),
            positive_landmarks=("label", "sum"),
        )
        parts.append(part.reset_index())
    combined = pd.concat(parts, ignore_index=True)
    stay = combined.groupby("record_id", sort=False).agg(
        patient_id=("patient_id", "first"),
        age=("age", "first"),
        sex_male=("sex_male", "first"),
        event=("event", "max"),
        landmarks=("landmarks", "sum"),
        positive_landmarks=("positive_landmarks", "sum"),
    )
    return {
        "dataset": "eicu_external",
        "n_stays": len(stay),
        "n_patients": int(stay.patient_id.nunique()),
        "age_median": float(stay.age.median()),
        "age_q1": float(stay.age.quantile(0.25)),
        "age_q3": float(stay.age.quantile(0.75)),
        "male_percent": float(stay.sex_male.mean() * 100),
        "event_positive_stays": int(stay.event.sum()),
        "event_positive_stays_percent": float(stay.event.mean() * 100),
        "n_landmarks": int(stay.landmarks.sum()),
        "positive_landmarks": int(stay.positive_landmarks.sum()),
        "positive_landmarks_percent": float(stay.positive_landmarks.sum() / stay.landmarks.sum() * 100),
    }


def missingness_row(frame: pd.DataFrame, dataset: str) -> dict:
    x = rv.feature_frame(frame)
    row = {"dataset": dataset, "n_landmarks": len(frame)}
    row.update({column: float(x[column].isna().mean()) for column in rv.FEATURE_COLS})
    return row


def eicu_missingness() -> dict:
    counts = {column: 0 for column in rv.FEATURE_COLS}
    n_rows = 0
    for chunk in rv.iter_csv_chunks(rv.EICU_SAMPLES):
        x = rv.feature_frame(chunk)
        for column in rv.FEATURE_COLS:
            counts[column] += int(x[column].isna().sum())
        n_rows += len(chunk)
    row = {"dataset": "eicu_external", "n_landmarks": n_rows}
    row.update({column: counts[column] / n_rows for column in rv.FEATURE_COLS})
    return row


def main() -> None:
    mimic = rv.read_csv_samples(rv.WORK / "mimic_samples_anchor.csv")
    development, selection, temporal = rv.split_mimic_eras(mimic)
    sicdb = rv.read_csv_samples(rv.WORK / "sicdb_samples_main_units_rr.csv")
    cohort_rows = [
        summarize_stays(development, "mimic_development_2008_2016"),
        summarize_stays(selection, "mimic_model_selection_2017_2019"),
        summarize_stays(temporal, "mimic_temporal_test_2020_2022"),
        summarize_eicu_stays(),
        summarize_stays(sicdb, "sicdb_external"),
    ]
    pd.DataFrame(cohort_rows).to_csv(rv.OUTPUTS / "corrected_table1_cohort_summary.csv", index=False)

    missing_rows = [
        missingness_row(development, "mimic_development_2008_2016"),
        missingness_row(selection, "mimic_model_selection_2017_2019"),
        missingness_row(temporal, "mimic_temporal_test_2020_2022"),
        eicu_missingness(),
        missingness_row(sicdb, "sicdb_external"),
    ]
    pd.DataFrame(missing_rows).to_csv(rv.OUTPUTS / "corrected_feature_missingness.csv", index=False)

    outcome_lines = [
        "# Outcome harmonization audit",
        "",
        "The target was the first documented start of a continuous norepinephrine, epinephrine, phenylephrine, vasopressin, or dopamine infusion during the ICU stay. Landmark rows after a prior target start were excluded.",
        "",
        "| Database | Source | Continuous-use rule | Start time | Drug mapping |",
        "| --- | --- | --- | --- | --- |",
        "| MIMIC-IV | mimiciv_derived.vasoactive_agent | At least one target agent rate greater than zero | Earliest starttime | Named columns for the five target agents |",
        "| eICU-CRD | infusiondrug | Numeric drugrate or infusionrate greater than zero | Earliest infusionoffset | Drug-name regex including generic and common synonym names |",
        "| SICdb | medication.csv.gz | IsSingleDose = 0 | Earliest medication Offset corrected by ICUOffset | DrugIDs 1502 epinephrine, 1550 vasopressin, 1562 norepinephrine, 1593 phenylephrine, 1618 dopamine |",
        "",
        "Database records establish documented treatment initiation, not physiologic shock onset or clinician intent. Differences in medication charting remain a potential source of outcome misclassification.",
    ]
    (rv.OUTPUTS / "corrected_outcome_harmonization.md").write_text("\n".join(outcome_lines) + "\n", encoding="utf-8")

    manifest = {
        "analysis_status": "corrected_primary_analysis",
        "source_datasets": {
            "mimic_iv": {
                "version": "3.1",
                "doi": "10.13026/kpb9-mt58",
            },
            "eicu_crd": {
                "version": "2.0",
                "doi": "10.13026/C2WM1R",
            },
            "sicdb": {
                "version": "1.0.8",
                "doi": "10.13026/8m72-6j83",
            },
        },
        "mimic_time_axis": "patients.anchor_year_group",
        "development_groups": sorted(rv.MIMIC_DEVELOPMENT_GROUPS),
        "model_selection_groups": sorted(rv.MIMIC_SELECTION_GROUPS),
        "locked_temporal_test_groups": sorted(rv.MIMIC_TEMPORAL_TEST_GROUPS),
        "selected_model": joblib.load(rv.OUTPUTS / "corrected_primary_model.joblib")["selected_model"],
        "bootstrap_unit": "ICU stay / unit stay",
        "bootstrap_replicates": 500,
        "mimic_extract_sql_sha256": sha256(rv.WORK / "mimic_extract.sql"),
        "eicu_extract_sql_sha256": sha256(rv.WORK / "eicu_extract.sql"),
        "corrected_modeling_sha256": sha256(rv.WORK / "corrected_modeling.py"),
        "clustered_inference_sha256": sha256(rv.WORK / "clustered_inference.py"),
        "local_recalibration_sha256": sha256(rv.WORK / "local_recalibration.py"),
        "corrected_supplementary_sha256": sha256(rv.WORK / "corrected_supplementary.py"),
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }
    (rv.OUTPUTS / "corrected_run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
