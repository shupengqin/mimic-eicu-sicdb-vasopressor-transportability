"""Corrected era-based model selection and frozen external validation.

MIMIC-IV anchor-year groups are used instead of shifted admission years:
2008-2016 development, 2017-2019 algorithm selection, and 2020-2022 locked
temporal testing. External datasets are scored only after algorithm selection
and final refitting on 2008-2019 MIMIC-IV data.
"""

from __future__ import annotations

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_benchmark as mb
import run_validation as rv


def fit_logistic(frame: pd.DataFrame) -> object:
    return rv.fit_model(frame)


def fit_hgb(frame: pd.DataFrame) -> object:
    model = mb.build_model()
    model.fit(rv.feature_frame(frame), frame["label"].astype(int).to_numpy())
    return model


def score_frame(frame: pd.DataFrame, models: dict[str, object]) -> dict[str, np.ndarray]:
    x = rv.feature_frame(frame)
    return {name: model.predict_proba(x)[:, 1] for name, model in models.items()}


def save_predictions(dataset: str, frame: pd.DataFrame, predictions: dict[str, np.ndarray]) -> None:
    payload = {
        "record_id": frame["record_id"].to_numpy(dtype=np.int64),
        "patient_id": frame["patient_id"].fillna(-1).to_numpy(dtype=np.int64),
        "index_hour": frame["index_hour"].to_numpy(dtype=np.int16),
        "y": frame["label"].to_numpy(dtype=np.int8),
        "lead_time_hours": frame["lead_time_hours"].to_numpy(dtype=np.float32),
    }
    for name, values in predictions.items():
        payload[f"p_{name}"] = values.astype(np.float32)
    np.savez_compressed(rv.WORK / f"corrected_predictions_{dataset}.npz", **payload)


def metric_row(
    dataset: str,
    model_name: str,
    frame: pd.DataFrame,
    p: np.ndarray,
) -> tuple[dict, list[dict]]:
    metrics, curve, _ = rv.summarize_predictions(
        dataset,
        frame["label"].astype(int).to_numpy(),
        p,
        frame["record_id"].to_numpy(),
        frame["lead_time_hours"].to_numpy(dtype=float),
        {},
        len(frame),
    )
    metrics["model"] = model_name
    for row in curve:
        row["model"] = model_name
    return metrics, curve


def score_eicu(models: dict[str, object]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    arrays: dict[str, list[np.ndarray]] = {
        "record_id": [],
        "patient_id": [],
        "index_hour": [],
        "y": [],
        "lead_time_hours": [],
    }
    prediction_parts = {name: [] for name in models}
    n_rows = 0
    for chunk in rv.iter_csv_chunks(rv.EICU_SAMPLES):
        x = rv.feature_frame(chunk)
        arrays["record_id"].append(chunk["record_id"].to_numpy(dtype=np.int64))
        arrays["patient_id"].append(chunk["patient_id"].fillna(-1).to_numpy(dtype=np.int64))
        arrays["index_hour"].append(chunk["index_hour"].to_numpy(dtype=np.int16))
        arrays["y"].append(chunk["label"].to_numpy(dtype=np.int8))
        arrays["lead_time_hours"].append(chunk["lead_time_hours"].to_numpy(dtype=np.float32))
        for name, model in models.items():
            prediction_parts[name].append(model.predict_proba(x)[:, 1].astype(np.float32))
        n_rows += len(chunk)
        print(f"eICU frozen scoring: {n_rows:,} rows", flush=True)
    combined = {name: np.concatenate(parts) for name, parts in arrays.items()}
    predictions = {name: np.concatenate(parts) for name, parts in prediction_parts.items()}
    np.savez_compressed(
        rv.WORK / "corrected_predictions_eicu_external.npz",
        **combined,
        **{f"p_{name}": value for name, value in predictions.items()},
    )
    return combined, predictions


def main() -> None:
    rv.ensure_dirs()
    mimic = rv.read_csv_samples(rv.WORK / "mimic_samples_anchor.csv")
    development, selection, temporal_test = rv.split_mimic_eras(mimic)

    development_models = {
        "logistic_regression": fit_logistic(development),
        "hist_gradient_boosting": fit_hgb(development),
    }
    selection_rows = []
    for model_name, model in development_models.items():
        p = model.predict_proba(rv.feature_frame(selection))[:, 1]
        metrics, _ = metric_row("mimic_model_selection", model_name, selection, p)
        selection_rows.append(metrics)
        print(
            f"Selection {model_name}: AUROC={metrics['auroc']:.4f}, AUPRC={metrics['auprc']:.4f}",
            flush=True,
        )
    selection_metrics = pd.DataFrame(selection_rows)
    selection_metrics.to_csv(rv.OUTPUTS / "corrected_model_selection_metrics.csv", index=False)
    selected_model_name = str(selection_metrics.sort_values(["auroc", "auprc"], ascending=False).iloc[0]["model"])
    print(f"Selected using MIMIC 2017-2019 only: {selected_model_name}", flush=True)

    final_training = pd.concat([development, selection], ignore_index=True)
    final_models = {
        "logistic_regression": fit_logistic(final_training),
        "hist_gradient_boosting": fit_hgb(final_training),
    }
    common_artifact = {
        "feature_cols": rv.FEATURE_COLS,
        "outcome": "first continuous vasopressor initiation within 6 hours",
        "index_hours": [6, 24],
        "development_groups": sorted(rv.MIMIC_DEVELOPMENT_GROUPS),
        "selection_groups": sorted(rv.MIMIC_SELECTION_GROUPS),
        "temporal_test_groups": sorted(rv.MIMIC_TEMPORAL_TEST_GROUPS),
        "selection_rule": "highest AUROC in MIMIC-IV 2017-2019; AUPRC used as tie-breaker",
        "selected_model": selected_model_name,
        "physiologic_ranges": rv.PHYSIOLOGIC_RANGES,
    }
    for model_name, model in final_models.items():
        artifact = {**common_artifact, "algorithm": model_name, "model": model}
        joblib.dump(artifact, rv.OUTPUTS / f"corrected_{model_name}_model.joblib")
    joblib.dump(
        {**common_artifact, "algorithm": selected_model_name, "model": final_models[selected_model_name]},
        rv.OUTPUTS / "corrected_primary_model.joblib",
    )

    sicdb = rv.read_csv_samples(rv.WORK / "sicdb_samples_main_units.csv")
    metric_rows: list[dict] = []
    curve_rows: list[dict] = []
    for dataset, frame in [("mimic_temporal_test", temporal_test), ("sicdb_external", sicdb)]:
        predictions = score_frame(frame, final_models)
        save_predictions(dataset, frame, predictions)
        for model_name, p in predictions.items():
            metrics, curve = metric_row(dataset, model_name, frame, p)
            metric_rows.append(metrics)
            curve_rows.extend(curve)
            print(
                f"Final {dataset} {model_name}: AUROC={metrics['auroc']:.4f}, AUPRC={metrics['auprc']:.4f}",
                flush=True,
            )

    eicu_arrays, eicu_predictions = score_eicu(final_models)
    eicu_frame = pd.DataFrame(
        {
            "record_id": eicu_arrays["record_id"],
            "patient_id": eicu_arrays["patient_id"],
            "index_hour": eicu_arrays["index_hour"],
            "label": eicu_arrays["y"],
            "lead_time_hours": eicu_arrays["lead_time_hours"],
        }
    )
    for model_name, p in eicu_predictions.items():
        metrics, curve = metric_row("eicu_external", model_name, eicu_frame, p)
        metric_rows.append(metrics)
        curve_rows.extend(curve)
        print(
            f"Final eicu_external {model_name}: AUROC={metrics['auroc']:.4f}, AUPRC={metrics['auprc']:.4f}",
            flush=True,
        )

    metrics = pd.DataFrame(metric_rows)
    metrics = metrics[["model"] + [column for column in metrics.columns if column != "model"]]
    metrics.to_csv(rv.OUTPUTS / "corrected_frozen_validation_metrics.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(rv.OUTPUTS / "corrected_decision_curve.csv", index=False)
    pd.DataFrame(
        [
            {
                "selected_model": selected_model_name,
                "selection_dataset": "MIMIC-IV anchor_year_group 2017-2019",
                "final_training_groups": "2008-2019",
                "locked_temporal_test_group": "2020-2022",
                "external_data_used_for_selection": False,
            }
        ]
    ).to_csv(rv.OUTPUTS / "corrected_model_selection_record.csv", index=False)


if __name__ == "__main__":
    main()
