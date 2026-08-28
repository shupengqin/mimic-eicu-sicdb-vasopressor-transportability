"""Generate paper-ready tables and strict-future sensitivity results."""

from __future__ import annotations

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_validation as rv


def strict_frame(frame: pd.DataFrame) -> pd.DataFrame:
    lead = pd.to_numeric(frame["lead_time_hours"], errors="coerce")
    keep = ~(frame["label"].eq(1) & lead.eq(0))
    return frame.loc[keep].copy()


def run_strict_future(model: object) -> None:
    rows = []
    for path, name, split in [
        (rv.WORK / "mimic_samples.csv", "mimic", "mimic"),
        (rv.WORK / "sicdb_samples_main_units.csv", "sicdb", "sicdb"),
    ]:
        frame = rv.read_csv_samples(path)
        if name == "mimic":
            subsets = [("mimic_development", frame[frame.time_year.le(2177)]), ("mimic_temporal", frame[frame.time_year.ge(2178)])]
        else:
            subsets = [("sicdb_external", frame)]
        for dataset, subset in subsets:
            clean = strict_frame(subset)
            metrics, _, _ = rv.evaluate_frame(dataset + "_strict_future", clean, model)
            rows.append(metrics)

    y_parts = []
    p_parts = []
    rec_parts = []
    lead_parts = []
    for chunk in rv.iter_csv_chunks(rv.WORK / "eicu_samples.csv"):
        clean = strict_frame(chunk)
        p = model.predict_proba(rv.feature_frame(clean))[:, 1]
        y_parts.append(clean["label"].astype(int).to_numpy())
        p_parts.append(p)
        rec_parts.append(clean["record_id"].to_numpy())
        lead_parts.append(clean["lead_time_hours"].to_numpy(dtype=float))
    y = np.concatenate(y_parts)
    p = np.concatenate(p_parts)
    record = np.concatenate(rec_parts)
    lead = np.concatenate(lead_parts)
    metrics, _, _ = rv.summarize_predictions("eicu_external_strict_future", y, p, record, lead, {}, len(y))
    rows.append(metrics)
    pd.DataFrame(rows).to_csv(rv.OUTPUTS / "strict_future_metrics.csv", index=False)


def write_coefficients(model_artifact: dict) -> None:
    model = model_artifact["model"]
    imputer = model.named_steps["imputer"]
    classifier = model.named_steps["classifier"]
    names = imputer.get_feature_names_out(model_artifact["feature_cols"])
    coef = classifier.coef_[0]
    out = pd.DataFrame(
        {
            "feature": names,
            "coefficient": coef,
            "odds_ratio_per_scaled_unit": np.exp(coef),
            "absolute_coefficient": np.abs(coef),
        }
    ).sort_values("absolute_coefficient", ascending=False)
    out.to_csv(rv.OUTPUTS / "model_coefficients.csv", index=False)


def write_cohort_flow(metrics: pd.DataFrame) -> None:
    rows = [
        {"dataset": "mimiciv", "source_adult_stays": 94458, "adult_stays_los_ge_12h": 90467, "risk_set_records": 71455},
        {"dataset": "eicu", "source_adult_unit_stays": 200234, "adult_stays_los_ge_12h": 172392, "risk_set_records": 157765},
        {"dataset": "sicdb_main_units", "source_adult_cases": 27223, "adult_main_unit_cases": 10283, "adult_main_unit_los_ge_12h": 9696, "risk_set_records": 3769},
    ]
    flow = pd.DataFrame(rows)
    metric_map = metrics.set_index("dataset")
    metric_sets = {
        "mimiciv": ["mimic_development", "mimic_temporal"],
        "eicu": ["eicu_external"],
        "sicdb_main_units": ["sicdb_external"],
    }
    for source_name, metric_names in metric_sets.items():
        selected = metric_map.loc[metric_names]
        flow.loc[flow.dataset.eq(source_name), "landmark_samples"] = int(selected["n_samples"].sum())
        flow.loc[flow.dataset.eq(source_name), "positive_records"] = int(selected["n_positive_records"].sum())
        flow.loc[flow.dataset.eq(source_name), "positive_landmarks"] = int(selected["n_positive_samples"].sum())
    flow.to_csv(rv.OUTPUTS / "cohort_flow.csv", index=False)


def write_manuscript_blueprint(metrics: pd.DataFrame) -> None:
    lookup = metrics.set_index("dataset")
    lines = [
        "# Manuscript blueprint",
        "",
        "## Recommended title",
        "",
        "Cross-database transportability of an hourly model for impending continuous vasopressor initiation: development in MIMIC-IV with temporal, US multicenter, and Austrian external validation",
        "",
        "## Chinese working title",
        "",
        "连续升压药即将启动的小时级动态预警模型：MIMIC-IV开发、时间外验证及eICU-CRD和SICdb外部验证",
        "",
        "## Primary question",
        "",
        "Can a model trained on early MIMIC-IV data identify adult ICU patients who are not yet receiving a continuous vasopressor but will initiate one within the next six hours, and does its discrimination, calibration, and alert burden transport across time, US hospitals, and an Austrian ICU system?",
        "",
        "## Design",
        "",
        "Adult ICU stays were sampled hourly from ICU hour 6 through hour 24. The risk set excluded landmark times after a prior target vasopressor start. The primary endpoint was initiation of norepinephrine, epinephrine, phenylephrine, vasopressin, or dopamine within the subsequent six hours. MIMIC-IV years through deidentified year 2177 formed development; later MIMIC-IV years formed temporal validation. The fitted model was then frozen and applied to eICU-CRD and SICdb without recalibration.",
        "",
        "## Core results",
        "",
        f"MIMIC-IV temporal AUROC was {lookup.loc['mimic_temporal','auroc']:.3f} with calibration slope {lookup.loc['mimic_temporal','calibration_slope']:.3f}. eICU AUROC was {lookup.loc['eicu_external','auroc']:.3f}, while its calibration intercept was {lookup.loc['eicu_external','calibration_intercept']:.3f}. SICdb AUROC was {lookup.loc['sicdb_external','auroc']:.3f}, with calibration intercept {lookup.loc['sicdb_external','calibration_intercept']:.3f}.",
        "",
        "## Novelty claim",
        "",
        "The defensible novelty is not the prediction target alone. Prior studies have already modeled vasopressor or hemodynamic intervention use in MIMIC/eICU. The contribution is a prespecified transportability evaluation combining temporal drift, US multicenter external validation, geographic validation in SICdb, hospital-level heterogeneity, calibration drift, and alert burden under one landmark definition.",
        "",
        "## Duplicate-research safeguard",
        "",
        "The introduction should explicitly distinguish this study from prior MIMIC/eICU vasopressor-use prediction, sepsis intervention prediction, and recent MIMIC-IV vasopressor-initiation models. Avoid claiming first prediction of vasopressor initiation. Claim instead that the study evaluates whether a frozen, hourly model remains clinically usable across databases and quantifies when recalibration is required.",
        "",
        "## Main limitations to state",
        "",
        "SICdb is a single Austrian tertiary-hospital database rather than a European multicenter cohort. Treatment initiation is inferred from database medication/infusion records and may reflect local documentation practice. Hourly samples are repeated within stays, so confidence intervals should be clustered by ICU stay. External calibration should be performed before clinical deployment.",
    ]
    (rv.OUTPUTS / "manuscript_blueprint.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    artifact = joblib.load(rv.OUTPUTS / "mimic_frozen_model.joblib")
    metrics = pd.read_csv(rv.OUTPUTS / "model_metrics.csv")
    run_strict_future(artifact["model"])
    write_coefficients(artifact)
    write_cohort_flow(metrics)
    write_manuscript_blueprint(metrics)
    print("Paper artifacts generated")


if __name__ == "__main__":
    main()
