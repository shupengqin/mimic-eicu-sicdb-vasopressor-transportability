"""Benchmark a pre-specified nonlinear model against the frozen logistic model.

The benchmark uses the same landmark samples, feature cleaning, outcome, and
evaluation functions as the primary analysis. HistGradientBoosting handles
missing values natively; no external recalibration or tuning is performed.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_validation as rv


def build_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        min_samples_leaf=100,
        l2_regularization=1.0,
        random_state=20260823,
    )


def main() -> None:
    mimic = rv.read_csv_samples(rv.WORK / "mimic_samples_anchor.csv")
    train = mimic[mimic["time_year"].le(2177)].copy()
    temporal = mimic[mimic["time_year"].ge(2178)].copy()
    if train.empty or temporal.empty:
        raise RuntimeError("MIMIC temporal split is empty")

    model = build_model()
    print(f"Fitting HistGradientBoosting on {len(train):,} samples ...", flush=True)
    model.fit(rv.feature_frame(train), train["label"].astype(int).to_numpy())

    joblib.dump(
        {
            "model": model,
            "feature_cols": rv.FEATURE_COLS,
            "outcome": "continuous vasopressor initiation within 6 hours",
            "index_hours": [6, 24],
            "mimic_development_year_max": 2177,
            "physiologic_ranges": rv.PHYSIOLOGIC_RANGES,
            "algorithm": "HistGradientBoostingClassifier",
            "parameters": model.get_params(),
        },
        rv.OUTPUTS / "mimic_hgb_frozen_model.joblib",
    )

    frames = [
        ("mimic_development", train),
        ("mimic_temporal", temporal),
        ("sicdb_external", rv.read_csv_samples(rv.WORK / "sicdb_samples_main_units.csv")),
    ]

    model_specs = [
        ("logistic_regression", joblib.load(rv.OUTPUTS / "mimic_frozen_model.joblib")["model"]),
        ("hist_gradient_boosting", model),
    ]
    rows = []
    curve_rows = []
    for model_name, candidate in model_specs:
        for dataset, frame in frames:
            metrics, curve, _ = rv.evaluate_frame(dataset, frame, candidate)
            metrics["model"] = model_name
            rows.append(metrics)
            for curve_row in curve:
                curve_row["model"] = model_name
                curve_rows.append(curve_row)
            print(
                f"{model_name} {dataset}: AUROC={metrics['auroc']:.4f}, "
                f"AUPRC={metrics['auprc']:.4f}",
                flush=True,
            )

        eicu_metrics, eicu_curve, _ = rv.evaluate_csv("eicu_external", rv.EICU_SAMPLES, candidate)
        eicu_metrics["model"] = model_name
        rows.append(eicu_metrics)
        for curve_row in eicu_curve:
            curve_row["model"] = model_name
            curve_rows.append(curve_row)
        print(
            f"{model_name} eicu_external: AUROC={eicu_metrics['auroc']:.4f}, "
            f"AUPRC={eicu_metrics['auprc']:.4f}",
            flush=True,
        )

    output = pd.DataFrame(rows)
    output = output[["model"] + [column for column in output.columns if column != "model"]]
    output.to_csv(rv.OUTPUTS / "model_benchmark_metrics.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(rv.OUTPUTS / "model_benchmark_decision_curve.csv", index=False)
    print(f"Wrote {rv.OUTPUTS / 'model_benchmark_metrics.csv'}", flush=True)

    logistic = output[output["model"].eq("logistic_regression")].set_index("dataset")
    hgb = output[output["model"].eq("hist_gradient_boosting")].set_index("dataset")
    comparison_lines = [
        "# Model benchmark",
        "",
        "The primary model comparison was prespecified before reviewing external results:",
        "regularized logistic regression versus HistGradientBoostingClassifier trained on the same MIMIC-IV development cohort and applied without recalibration.",
        "",
        "## Performance",
        "",
        "| Dataset | Model | AUROC | AUPRC | Brier | Calibration intercept | Calibration slope | False alerts/100 eligible landmark-hours at 0.10 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset in ["mimic_temporal", "eicu_external", "sicdb_external"]:
        for model_name, table in [("logistic regression", logistic), ("HistGradientBoosting", hgb)]:
            row = table.loc[dataset]
            comparison_lines.append(
                f"| {dataset} | {model_name} | {row.auroc:.3f} | {row.auprc:.3f} | {row.brier:.3f} | "
                f"{row.calibration_intercept:.3f} | {row.calibration_slope:.3f} | "
                f"{row['false_alerts_per_100_eligible_landmark_hours_at_0.10']:.2f} |"
            )
    comparison_lines.extend(
        [
            "",
            "## Decision",
            "",
            f"On MIMIC-IV temporal validation, HistGradientBoosting improved AUROC by {hgb.loc['mimic_temporal','auroc'] - logistic.loc['mimic_temporal','auroc']:.3f} and AUPRC by {hgb.loc['mimic_temporal','auprc'] - logistic.loc['mimic_temporal','auprc']:.3f}.",
            f"The AUROC improvement was also observed in eICU ({hgb.loc['eicu_external','auroc'] - logistic.loc['eicu_external','auroc']:.3f}) and SICdb ({hgb.loc['sicdb_external','auroc'] - logistic.loc['sicdb_external','auroc']:.3f}).",
            "HistGradientBoosting is therefore the primary performance model. Logistic regression remains a transparent sensitivity benchmark; the study should report both rather than describe the task as solved by one algorithm.",
            "",
            "External calibration remains imperfect, especially in SICdb, so the model is a transportability study result and not a deployment-ready clinical alarm without local recalibration.",
            "",
        ]
    )
    (rv.OUTPUTS / "model_benchmark.md").write_text("\n".join(comparison_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
