"""Strict-future-window evaluation for the frozen HGB model."""

from __future__ import annotations

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_artifacts as pa
import run_validation as rv


def evaluate() -> None:
    model = joblib.load(rv.OUTPUTS / "mimic_hgb_frozen_model.joblib")["model"]
    rows = []
    mimic = rv.read_csv_samples(rv.WORK / "mimic_samples.csv")
    for dataset, subset in [
        ("mimic_development", mimic[mimic.time_year.le(2177)]),
        ("mimic_temporal", mimic[mimic.time_year.ge(2178)]),
    ]:
        clean = pa.strict_frame(subset)
        metrics, _, _ = rv.evaluate_frame(dataset + "_strict_future", clean, model)
        rows.append(metrics)

    sicdb = rv.read_csv_samples(rv.WORK / "sicdb_samples_main_units.csv")
    clean = pa.strict_frame(sicdb)
    metrics, _, _ = rv.evaluate_frame("sicdb_external_strict_future", clean, model)
    rows.append(metrics)

    y_parts, p_parts, record_parts, lead_parts = [], [], [], []
    for chunk in rv.iter_csv_chunks(rv.WORK / "eicu_samples.csv"):
        clean = pa.strict_frame(chunk)
        y_parts.append(clean["label"].astype(int).to_numpy())
        p_parts.append(model.predict_proba(rv.feature_frame(clean))[:, 1])
        record_parts.append(clean["record_id"].to_numpy())
        lead_parts.append(clean["lead_time_hours"].to_numpy(dtype=float))
    metrics, _, _ = rv.summarize_predictions(
        "eicu_external_strict_future",
        np.concatenate(y_parts),
        np.concatenate(p_parts),
        np.concatenate(record_parts),
        np.concatenate(lead_parts),
        {},
        sum(len(part) for part in y_parts),
    )
    rows.append(metrics)
    output = pd.DataFrame(rows)
    output.insert(0, "model", "hist_gradient_boosting")
    output.to_csv(rv.OUTPUTS / "hgb_strict_future_metrics.csv", index=False)
    print(output[["dataset", "auroc", "auprc", "brier", "calibration_intercept", "calibration_slope"]].to_string(index=False))


if __name__ == "__main__":
    evaluate()
