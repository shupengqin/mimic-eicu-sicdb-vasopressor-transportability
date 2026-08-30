"""Supplementary analyses for the corrected frozen HGB model."""

from __future__ import annotations

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_validation as rv


def metrics_from_arrays(dataset: str, data: object, mask: np.ndarray) -> dict:
    metrics, _, _ = rv.summarize_predictions(
        dataset,
        data["y"][mask].astype(int),
        data["p_hist_gradient_boosting"][mask].astype(float),
        data["record_id"][mask],
        data["lead_time_hours"][mask].astype(float),
        {},
        int(mask.sum()),
    )
    return metrics


def strict_future() -> None:
    rows = []
    for dataset in ["mimic_temporal_test", "eicu_external", "sicdb_external"]:
        data = np.load(rv.WORK / f"corrected_predictions_{dataset}.npz")
        lead = data["lead_time_hours"].astype(float)
        mask = ~((data["y"] == 1) & np.isclose(lead, 0.0, atol=1e-6, equal_nan=False))
        rows.append(metrics_from_arrays(dataset + "_strict_future", data, mask))
    pd.DataFrame(rows).to_csv(rv.OUTPUTS / "corrected_strict_future_metrics.csv", index=False)


def eicu_hospitals() -> None:
    data = np.load(rv.WORK / "corrected_predictions_eicu_external.npz")
    site_map = pd.read_csv(rv.WORK / "eicu_site_map.csv", usecols=["record_id", "hospital_id"])
    site_map["record_id"] = pd.to_numeric(site_map["record_id"], errors="coerce").astype("Int64")
    site_lookup = site_map.dropna(subset=["record_id"]).set_index("record_id")["hospital_id"].to_dict()
    hospitals = pd.Series(data["record_id"]).map(site_lookup).fillna(-1).to_numpy(dtype=int)
    rows = []
    for hospital_id in np.unique(hospitals):
        mask = hospitals == hospital_id
        if mask.sum() < 1000:
            continue
        record_labels = (
            pd.DataFrame({"record_id": data["record_id"][mask], "label": data["y"][mask]})
            .groupby("record_id")["label"]
            .max()
        )
        positive_stays = int(record_labels.sum())
        negative_stays = int((record_labels == 0).sum())
        if positive_stays < 20 or negative_stays < 20:
            continue
        metrics = metrics_from_arrays(f"eicu_hospital_{hospital_id}", data, mask)
        metrics["hospital_id"] = hospital_id
        metrics["positive_stays"] = positive_stays
        metrics["negative_stays"] = negative_stays
        rows.append(metrics)
    pd.DataFrame(rows).sort_values("hospital_id").to_csv(
        rv.OUTPUTS / "corrected_eicu_hospital_metrics.csv", index=False
    )


def sicdb_units_and_all() -> None:
    model = joblib.load(rv.OUTPUTS / "corrected_primary_model.joblib")["model"]
    main = rv.read_csv_samples(rv.WORK / "sicdb_samples_main_units.csv")
    rows = []
    for unit, frame in main.groupby("unit_name", sort=True):
        metrics, _, _ = rv.evaluate_frame(f"sicdb_{unit}", frame, model)
        metrics["unit_name"] = unit
        rows.append(metrics)
    pd.DataFrame(rows).to_csv(rv.OUTPUTS / "corrected_sicdb_unit_metrics.csv", index=False)

    all_units = rv.read_csv_samples(rv.WORK / "sicdb_samples_all_units.csv")
    metrics, curve, quality = rv.evaluate_frame("sicdb_all_units_sensitivity", all_units, model)
    pd.DataFrame([metrics]).to_csv(rv.OUTPUTS / "corrected_sicdb_all_units_metrics.csv", index=False)
    pd.DataFrame(curve).to_csv(rv.OUTPUTS / "corrected_sicdb_all_units_decision_curve.csv", index=False)
    pd.DataFrame([quality]).to_csv(rv.OUTPUTS / "corrected_sicdb_all_units_missingness.csv", index=False)


def write_summary() -> None:
    strict = pd.read_csv(rv.OUTPUTS / "corrected_strict_future_metrics.csv").set_index("dataset")
    hospitals = pd.read_csv(rv.OUTPUTS / "corrected_eicu_hospital_metrics.csv")
    units = pd.read_csv(rv.OUTPUTS / "corrected_sicdb_unit_metrics.csv")
    all_units = pd.read_csv(rv.OUTPUTS / "corrected_sicdb_all_units_metrics.csv").iloc[0]
    lines = [
        "# Corrected supplementary validation",
        "",
        "## Strict future window",
        "",
    ]
    for dataset, row in strict.iterrows():
        lines.append(f"- {dataset}: AUROC {row.auroc:.3f}, AUPRC {row.auprc:.3f}, n={int(row.n_samples):,} landmarks.")
    lines.extend(
        [
            "",
            "## eICU hospital heterogeneity",
            "",
            f"{len(hospitals)} hospitals met the criteria of at least 1,000 landmarks, 20 event-positive stays, and 20 event-negative stays.",
            f"Hospital AUROC median {hospitals.auroc.median():.3f}, IQR {hospitals.auroc.quantile(.25):.3f}-{hospitals.auroc.quantile(.75):.3f}, range {hospitals.auroc.min():.3f}-{hospitals.auroc.max():.3f}.",
            "",
            "## SICdb sensitivity analyses",
            "",
        ]
    )
    for row in units.itertuples(index=False):
        lines.append(f"- {row.unit_name}: AUROC {row.auroc:.3f}, AUPRC {row.auprc:.3f}, n={int(row.n_samples):,} landmarks.")
    lines.append(
        f"- All four units: AUROC {all_units.auroc:.3f}, AUPRC {all_units.auprc:.3f}, n={int(all_units.n_samples):,} landmarks."
    )
    (rv.OUTPUTS / "corrected_supplementary_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    strict_future()
    eicu_hospitals()
    sicdb_units_and_all()
    write_summary()


if __name__ == "__main__":
    main()
