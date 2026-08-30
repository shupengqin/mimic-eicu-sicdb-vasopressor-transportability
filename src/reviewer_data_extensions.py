"""Build local inputs for the post hoc reviewer-priority analyses.

Patient-level extracts are written only under ``work/`` and are excluded from
the public repository. Public outputs contain aggregate results only.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_validation as rv


DRUG_NAMES = {
    1502: "epinephrine",
    1550: "vasopressin",
    1562: "norepinephrine",
    1593: "phenylephrine",
    1618: "dopamine",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_sicdb_cases_after_hour6() -> pd.DataFrame:
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
    cases = pd.read_csv(
        rv.SICDB / "cases.csv.gz",
        compression="gzip",
        usecols=usecols,
        low_memory=False,
    )
    numeric_cols = [
        "CaseID",
        "PatientID",
        "AdmissionYear",
        "TimeOfStay",
        "ICUOffset",
        "AgeOnAdmission",
        "HospitalUnit",
    ]
    for column in numeric_cols:
        cases[column] = rv.numeric(cases[column])
    cases["icu_los_s"] = cases["TimeOfStay"] - cases["ICUOffset"]
    cases = cases[
        cases["AgeOnAdmission"].ge(18)
        & cases["HospitalUnit"].isin(rv.SICDB_MAIN_UNITS)
        & cases["icu_los_s"].gt(6 * 3600)
        & cases["CaseID"].notna()
    ].copy()
    cases["record_id"] = cases["CaseID"].astype(np.int64)
    cases["patient_id"] = cases["PatientID"].astype("Int64")
    cases["age"] = cases["AgeOnAdmission"].astype(float)
    cases["time_year"] = cases["AdmissionYear"].astype("Int64")
    cases["sex_male"] = rv.decode_sicdb_sex(cases["Sex"], rv.decode_reference_map())
    cases["unit_name"] = cases["HospitalUnit"].map(
        {2: "INIC", 3: "CWIN", 4: "INBD", 5: "INID"}
    )
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


def read_sicdb_first_agents(cases: pd.DataFrame) -> pd.DataFrame:
    ids = set(cases["record_id"].astype(int))
    offset_map = cases.set_index("record_id")["ICUOffset"]
    parts: list[pd.DataFrame] = []
    usecols = ["CaseID", "DrugID", "Offset", "IsSingleDose"]
    for chunk in pd.read_csv(
        rv.SICDB / "medication.csv.gz",
        compression="gzip",
        usecols=usecols,
        chunksize=250_000,
        low_memory=False,
    ):
        for column in usecols:
            chunk[column] = rv.numeric(chunk[column])
        chunk = chunk[
            chunk["CaseID"].isin(ids)
            & chunk["DrugID"].isin(DRUG_NAMES)
            & chunk["IsSingleDose"].eq(0)
            & chunk["Offset"].notna()
        ].copy()
        if chunk.empty:
            continue
        chunk["record_id"] = chunk["CaseID"].astype(np.int64)
        chunk["first_start_s"] = chunk["Offset"] - chunk["record_id"].map(offset_map)
        chunk["agent"] = chunk["DrugID"].astype(int).map(DRUG_NAMES)
        parts.append(chunk[["record_id", "first_start_s", "agent"]])
    events = pd.concat(parts, ignore_index=True)
    events = events.dropna(subset=["first_start_s", "agent"]).drop_duplicates()
    first_time = events.groupby("record_id", as_index=False)["first_start_s"].min()
    first = events.merge(first_time, on=["record_id", "first_start_s"], how="inner")
    rows = []
    for record_id, group in first.groupby("record_id", sort=True):
        agents = sorted(group["agent"].unique())
        rows.append(
            {
                "record_id": int(record_id),
                "first_start_s": float(group["first_start_s"].iloc[0]),
                "first_start_hour": float(group["first_start_s"].iloc[0]) / 3600.0,
                "first_agents": "+".join(agents),
                "n_first_agents": len(agents),
                "norepinephrine_at_first": int("norepinephrine" in agents),
            }
        )
    return pd.DataFrame(rows)


def build_sicdb_hour6(cases: pd.DataFrame, agents: pd.DataFrame) -> pd.DataFrame:
    first_map = agents.set_index("record_id")["first_start_s"]
    cases = cases.copy()
    cases["first_start_s"] = cases["record_id"].map(first_map)
    cases = cases[
        cases["first_start_s"].isna() | cases["first_start_s"].ge(6 * 3600)
    ].copy()
    observed_end = np.minimum(12 * 3600, cases["icu_los_s"].to_numpy(dtype=float))
    first_start = cases["first_start_s"].to_numpy(dtype=float)
    event = np.isfinite(first_start) & (first_start < observed_end)
    competing_exit = (cases["icu_los_s"].to_numpy(dtype=float) < 12 * 3600) & ~event
    samples = pd.DataFrame(
        {
            "dataset": "sicdb",
            "record_id": cases["record_id"].to_numpy(dtype=np.int64),
            "patient_id": cases["patient_id"].to_numpy(),
            "unit_name": cases["unit_name"].to_numpy(),
            "time_year": cases["time_year"].to_numpy(),
            "index_hour": 6,
            "age": cases["age"].to_numpy(dtype=float),
            "sex_male": cases["sex_male"].to_numpy(dtype=float),
            "label": event.astype(np.int8),
            "lead_time_hours": np.where(event, first_start / 3600.0 - 6.0, np.nan),
            "observed_horizon_hours": np.minimum(
                6.0, cases["icu_los_s"].to_numpy(dtype=float) / 3600.0 - 6.0
            ),
            "competing_exit": competing_exit.astype(np.int8),
        }
    )
    vitals = rv.read_sicdb_vitals(cases)
    labs = rv.read_sicdb_labs(cases)
    samples = rv.add_sicdb_features(
        samples, pd.concat([vitals, labs], ignore_index=True)
    )
    return samples.sort_values(["record_id", "index_hour"]).reset_index(drop=True)


def run_sql_extracts() -> None:
    jobs = [
        (
            rv.MIMIC_DB,
            rv.ROOT / "sql" / "mimic_first_agent.sql",
            rv.WORK / "reviewer_mimic_first_agent.csv",
        ),
        (
            rv.EICU_DB,
            rv.ROOT / "sql" / "eicu_first_agent.sql",
            rv.WORK / "reviewer_eicu_first_agent.csv",
        ),
        (
            rv.MIMIC_DB,
            rv.ROOT / "sql" / "mimic_hour6_competing_exit.sql",
            rv.WORK / "reviewer_mimic_hour6_competing_exit.csv",
        ),
        (
            rv.EICU_DB,
            rv.ROOT / "sql" / "eicu_hour6_competing_exit.sql",
            rv.WORK / "reviewer_eicu_hour6_competing_exit.csv",
        ),
    ]
    for database, sql_path, output_path in jobs:
        if not output_path.exists() or output_path.stat().st_size == 0:
            rv.run_psql(database, sql_path, output_path)


def main() -> None:
    rv.ensure_dirs()
    run_sql_extracts()
    cases = read_sicdb_cases_after_hour6()
    agents = read_sicdb_first_agents(cases)
    agents.to_csv(rv.WORK / "reviewer_sicdb_first_agent.csv", index=False)
    hour6 = build_sicdb_hour6(cases, agents)
    hour6.to_csv(rv.WORK / "reviewer_sicdb_hour6_competing_exit.csv", index=False)

    local_outputs = [
        rv.WORK / "reviewer_mimic_first_agent.csv",
        rv.WORK / "reviewer_eicu_first_agent.csv",
        rv.WORK / "reviewer_sicdb_first_agent.csv",
        rv.WORK / "reviewer_mimic_hour6_competing_exit.csv",
        rv.WORK / "reviewer_eicu_hour6_competing_exit.csv",
        rv.WORK / "reviewer_sicdb_hour6_competing_exit.csv",
    ]
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_status": "post_hoc_reviewer_priority_inputs",
        "redistribution": "local_only_patient_level_inputs",
        "files": [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in local_outputs
        ],
    }
    (rv.OUTPUTS / "reviewer_input_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("Reviewer-priority local inputs completed.", flush=True)


if __name__ == "__main__":
    main()
