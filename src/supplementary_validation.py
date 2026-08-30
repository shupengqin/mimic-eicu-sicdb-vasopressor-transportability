"""Supplementary frozen-model checks without re-running database extraction."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_validation as rv


def run_eicu_hospital_metrics(model: object, prefix: str = "") -> None:
    site_map = pd.read_csv(rv.WORK / "eicu_site_map.csv", low_memory=False)
    site_map["record_id"] = pd.to_numeric(site_map["record_id"], errors="coerce").astype("Int64")
    site_map = site_map.dropna(subset=["record_id"]).copy()
    site_lookup = site_map.set_index("record_id")["hospital_id"].to_dict()
    buckets = {}
    for chunk in rv.iter_csv_chunks(rv.EICU_SAMPLES):
        chunk = chunk.reset_index(drop=True)
        hospitals = chunk["record_id"].map(site_lookup).fillna(-1).astype(int)
        p = model.predict_proba(rv.feature_frame(chunk))[:, 1]
        for hospital_id, indices in hospitals.groupby(hospitals, sort=False).groups.items():
            idx = np.asarray(indices, dtype=int)
            bucket = buckets.setdefault(int(hospital_id), {"y": [], "p": [], "record": [], "lead": []})
            bucket["y"].append(chunk.iloc[idx]["label"].astype(int).to_numpy())
            bucket["p"].append(p[idx])
            bucket["record"].append(chunk.iloc[idx]["record_id"].to_numpy())
            bucket["lead"].append(chunk.iloc[idx]["lead_time_hours"].to_numpy(dtype=float))
    rows = []
    for hospital_id, bucket in sorted(buckets.items()):
        y = np.concatenate(bucket["y"])
        if len(y) < 1000 or len(np.unique(y)) < 2:
            continue
        p = np.concatenate(bucket["p"])
        record = np.concatenate(bucket["record"])
        lead = np.concatenate(bucket["lead"])
        metrics, _, _ = rv.summarize_predictions(
            f"eicu_hospital_{hospital_id}", y, p, record, lead, {}, len(y)
        )
        metrics["hospital_id"] = hospital_id
        rows.append(metrics)
    out = pd.DataFrame(rows).sort_values("hospital_id")
    out.to_csv(rv.OUTPUTS / f"{prefix}eicu_hospital_metrics.csv", index=False)
    print(f"eICU hospitals retained for stratified metrics: {len(out)}")


def run_sicdb_unit_metrics(model: object, prefix: str = "") -> None:
    frame = rv.read_csv_samples(rv.WORK / "sicdb_samples_main_units.csv")
    rows = []
    for unit, group in frame.groupby("unit_name", sort=True):
        metrics, _, _ = rv.evaluate_frame(f"sicdb_{unit}", group, model)
        metrics["unit_name"] = unit
        rows.append(metrics)
    pd.DataFrame(rows).to_csv(rv.OUTPUTS / f"{prefix}sicdb_unit_metrics.csv", index=False)


def run_all_unit_sensitivity(model: object, prefix: str = "") -> None:
    path = rv.WORK / "sicdb_samples_all_units.csv"
    if not path.exists():
        frame = rv.build_sicdb_samples(all_units=True)
        frame.to_csv(path, index=False)
    else:
        frame = rv.read_csv_samples(path)
    metrics, curve, quality = rv.evaluate_frame("sicdb_all_units_sensitivity", frame, model)
    pd.DataFrame([metrics]).to_csv(rv.OUTPUTS / f"{prefix}sicdb_all_units_metrics.csv", index=False)
    pd.DataFrame(curve).to_csv(rv.OUTPUTS / f"{prefix}sicdb_all_units_decision_curve.csv", index=False)
    pd.DataFrame([quality]).to_csv(rv.OUTPUTS / f"{prefix}sicdb_all_units_missingness.csv", index=False)
    print(f"SICdb all-unit sensitivity samples: {len(frame):,}")


def write_supplementary_summary(prefix: str = "") -> None:
    hospitals = pd.read_csv(rv.OUTPUTS / f"{prefix}eicu_hospital_metrics.csv")
    units = pd.read_csv(rv.OUTPUTS / f"{prefix}sicdb_unit_metrics.csv")
    all_units_path = rv.OUTPUTS / f"{prefix}sicdb_all_units_metrics.csv"
    all_units = pd.read_csv(all_units_path) if all_units_path.exists() else None
    lines = [
        "# Supplementary validation summary",
        "",
        f"- eICU hospital strata retained: {len(hospitals)} (at least 1,000 landmark samples and both outcome classes).",
        f"- eICU hospital AUROC: median {hospitals.auroc.median():.3f}, IQR {hospitals.auroc.quantile(.25):.3f}-{hospitals.auroc.quantile(.75):.3f}, range {hospitals.auroc.min():.3f}-{hospitals.auroc.max():.3f}.",
        "- SICdb unit-level results:",
    ]
    for row in units.itertuples(index=False):
        lines.append(f"  - {row.unit_name}: AUROC {row.auroc:.3f}, AUPRC {row.auprc:.3f}, calibration intercept {row.calibration_intercept:.3f}, n={int(row.n_samples):,} samples.")
    if all_units is not None:
        row = all_units.iloc[0]
        lines.append(f"- SICdb all-unit sensitivity: AUROC {row.auroc:.3f}, AUPRC {row.auprc:.3f}, n={int(row.n_samples):,} samples.")
    (rv.OUTPUTS / f"{prefix}supplementary_validation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-unit-sensitivity", action="store_true")
    parser.add_argument("--model", choices=["logistic", "hgb"], default="logistic")
    args = parser.parse_args()
    prefix = "" if args.model == "logistic" else "hgb_"
    model_file = "mimic_frozen_model.joblib" if args.model == "logistic" else "mimic_hgb_frozen_model.joblib"
    artifact = joblib.load(rv.OUTPUTS / model_file)
    model = artifact["model"]
    run_eicu_hospital_metrics(model, prefix)
    run_sicdb_unit_metrics(model, prefix)
    if args.all_unit_sensitivity:
        run_all_unit_sensitivity(model, prefix)
    write_supplementary_summary(prefix)


if __name__ == "__main__":
    main()
