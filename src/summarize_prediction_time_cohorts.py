"""Create aggregate cohort descriptors for the prediction-time risk set."""
from pathlib import Path
import os
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("PREDICTION_TIME_WORK_DIR", REPO / "work" / "prediction_time"))
OUT = Path(os.environ.get("PREDICTION_TIME_OUTPUT_DIR", ROOT / "analysis"))

def summarize(frame: pd.DataFrame, dataset: str, label: str) -> dict:
    stay = frame.groupby("record_id", sort=False)
    first = frame.drop_duplicates("record_id")
    event = stay["label"].max()
    return {
        "dataset": dataset,
        "label": label,
        "n_stays": int(event.size),
        "n_landmarks": int(len(frame)),
        "event_positive_stays": int(event.sum()),
        "event_positive_stays_percent": float(event.mean() * 100),
        "positive_landmarks": int(frame["label"].sum()),
        "positive_landmarks_percent": float(frame["label"].mean() * 100),
        "exit_truncated_landmarks": int(frame["exit_without_event"].sum()),
        "exit_truncated_landmarks_percent": float(frame["exit_without_event"].mean() * 100),
        "median_age": float(first["age"].median()),
        "age_q1": float(first["age"].quantile(.25)),
        "age_q3": float(first["age"].quantile(.75)),
        "male_stays_percent": float(first["sex_male"].mean() * 100),
        "median_observed_horizon": float(frame["observed_horizon_hours"].median()),
    }

def main() -> None:
    rows = []
    mimic_path = ROOT / "mimic_prediction_time.csv"
    parts = {"development": [], "selection": [], "temporal_test": []}
    for chunk in pd.read_csv(mimic_path, low_memory=False, chunksize=250_000):
        group = chunk["time_group"].astype(str).str.strip()
        parts["development"].append(chunk[group.isin(["2008 - 2010", "2011 - 2013", "2014 - 2016"])])
        parts["selection"].append(chunk[group.eq("2017 - 2019")])
        parts["temporal_test"].append(chunk[group.eq("2020 - 2022")])
    for key, label in [("development", "MIMIC-IV development, 2008-2016"), ("selection", "MIMIC-IV model selection, 2017-2019"), ("temporal_test", "MIMIC-IV temporal test, 2020-2022")]:
        rows.append(summarize(pd.concat(parts[key], ignore_index=True), f"mimic_{key}", label))
    for dataset, label, path in [
        ("eicu_external", "eICU-CRD external cohort", ROOT / "eicu_prediction_time.csv"),
        ("sicdb_external", "SICdb primary Austrian cohort", ROOT / "sicdb_prediction_time.csv"),
    ]:
        chunks = [c for c in pd.read_csv(path, low_memory=False, chunksize=250_000)]
        rows.append(summarize(pd.concat(chunks, ignore_index=True), dataset, label))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "prediction_time_cohort_summary.csv", index=False)
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
