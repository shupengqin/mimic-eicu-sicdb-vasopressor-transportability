"""Post hoc full logistic recalibration of the frozen HGB probabilities.

The frozen model and all evaluation cohorts are unchanged.  For each of 100
identifier-disjoint splits within an evaluation cohort, the calibration subset
estimates an intercept and a slope in ``logit(y) ~ logit(p_frozen)``.  The
transformation is then applied once to the held-out evaluation subset.  This
script writes aggregate metrics only; patient- and row-level inputs remain in
the local ``work/`` directory.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_recalibration as lr
import run_validation as rv


DATASETS = ["mimic_temporal_test", "eicu_external", "sicdb_external"]
REPEATS = 100
SEED = 20260830 + 3


def stratified_group_mask(
    group_codes: np.ndarray,
    group_event: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    selected = np.zeros(len(group_event), dtype=bool)
    for event_value in (0, 1):
        candidates = np.flatnonzero(group_event == event_value)
        rng.shuffle(candidates)
        n_selected = max(1, int(round(0.20 * len(candidates))))
        selected[candidates[:n_selected]] = True
    return selected[group_codes]


def fit_full_recalibration(y: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    """Fit an unregularized logistic calibration model on the logit scale."""
    model = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=1000,
        fit_intercept=True,
        random_state=SEED,
    )
    model.fit(z.reshape(-1, 1), y.astype(np.int8))
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def run_dataset(dataset: str, rng: np.random.Generator) -> list[dict[str, object]]:
    path = rv.WORK / f"corrected_predictions_{dataset}.npz"
    with np.load(path) as data:
        y = data["y"].astype(np.int8)
        p = np.clip(data["p_hist_gradient_boosting"].astype(float), 1e-7, 1 - 1e-7)
        group_ids = data["patient_id"].astype(np.int64)
    _, group_codes = np.unique(group_ids, return_inverse=True)
    group_codes = group_codes.astype(np.int32)
    group_event = np.zeros(int(group_codes.max()) + 1, dtype=np.int8)
    np.maximum.at(group_event, group_codes, y)
    z = logit(p)
    rows: list[dict[str, object]] = []

    for repeat in range(1, REPEATS + 1):
        calibration_mask = stratified_group_mask(group_codes, group_event, rng)
        evaluation_mask = ~calibration_mask
        y_cal = y[calibration_mask]
        y_eval = y[evaluation_mask]
        p_eval = p[evaluation_mask]
        z_cal = z[calibration_mask]
        z_eval = z[evaluation_mask]

        intercept_only, _, _ = rv.calibration_fit(y_cal, p[calibration_mask])
        p_intercept = expit(intercept_only + z_eval)
        intercept_full, slope_full = fit_full_recalibration(y_cal, z_cal)
        p_full = expit(intercept_full + slope_full * z_eval)

        calibration_prevalence = float(np.mean(y_cal))
        null_brier = float(
            brier_score_loss(y_eval, np.full(len(y_eval), calibration_prevalence))
        )
        brier_uncalibrated = float(brier_score_loss(y_eval, p_eval))
        brier_intercept = float(brier_score_loss(y_eval, p_intercept))
        brier_full = float(brier_score_loss(y_eval, p_full))
        rows.extend(
            [
                {
                    "dataset": dataset,
                    "repeat": repeat,
                    "calibration_method": "uncalibrated",
                    "fitted_intercept": 0.0,
                    "fitted_slope": 1.0,
                    "calibration_prevalence": calibration_prevalence,
                    "brier": brier_uncalibrated,
                    "brier_skill": 1.0 - brier_uncalibrated / null_brier,
                },
                {
                    "dataset": dataset,
                    "repeat": repeat,
                    "calibration_method": "intercept_only",
                    "fitted_intercept": float(intercept_only),
                    "fitted_slope": 1.0,
                    "calibration_prevalence": calibration_prevalence,
                    "brier": brier_intercept,
                    "brier_skill": 1.0 - brier_intercept / null_brier,
                },
                {
                    "dataset": dataset,
                    "repeat": repeat,
                    "calibration_method": "intercept_and_slope",
                    "fitted_intercept": intercept_full,
                    "fitted_slope": slope_full,
                    "calibration_prevalence": calibration_prevalence,
                    "brier": brier_full,
                    "brier_skill": 1.0 - brier_full / null_brier,
                },
            ]
        )
    return rows


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    output: list[dict[str, object]] = []
    for (dataset, method), group in rows.groupby(
        ["dataset", "calibration_method"], sort=False
    ):
        summary: dict[str, object] = {
            "dataset": dataset,
            "calibration_method": method,
            "repeats": int(len(group)),
        }
        for column in ["fitted_intercept", "fitted_slope", "brier", "brier_skill"]:
            values = pd.to_numeric(group[column], errors="coerce")
            summary[f"{column}_q1"] = float(values.quantile(0.25))
            summary[f"{column}_median"] = float(values.median())
            summary[f"{column}_q3"] = float(values.quantile(0.75))
        output.append(summary)
    return pd.DataFrame(output)


def main() -> None:
    rv.ensure_dirs()
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        rows.extend(run_dataset(dataset, rng))
    detail = pd.DataFrame(rows)
    summary = summarize(detail)
    detail.to_csv(rv.OUTPUTS / "reviewer_full_logistic_recalibration_repeated.csv", index=False)
    summary.to_csv(rv.OUTPUTS / "reviewer_full_logistic_recalibration_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
