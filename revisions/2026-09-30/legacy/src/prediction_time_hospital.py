"""Recompute descriptive eICU hospital calibration for the prediction-time outcome."""
from __future__ import annotations
from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import clustered_inference as ci

ROOT = Path(os.environ.get("PREDICTION_TIME_WORK_DIR", REPO / "work" / "prediction_time"))
OUT = Path(os.environ.get("PREDICTION_TIME_OUTPUT_DIR", ROOT / "analysis"))
SITE_MAP = Path(os.environ.get("EICU_SITE_MAP_PATH", ROOT / "eicu_site_map.csv"))


def reml(est, var):
    est = np.asarray(est, float); var = np.asarray(var, float)
    valid = np.isfinite(est) & np.isfinite(var) & (var > 0)
    est, var = est[valid], var[valid]
    def obj(tau):
        w = 1 / (var + tau); pooled = np.sum(w * est) / np.sum(w)
        return .5 * (np.sum(np.log(var + tau)) + np.log(np.sum(w)) + np.sum(w * (est - pooled) ** 2))
    upper = max(float(np.var(est, ddof=1)) * 20, 1e-6)
    tau = max(0, float(minimize_scalar(obj, bounds=(0, upper), method="bounded").x))
    w = 1 / (var + tau); pooled = float(np.sum(w * est) / np.sum(w)); se = float(np.sqrt(1 / np.sum(w)))
    fw = 1 / var; fixed = np.sum(fw * est) / np.sum(fw); q = np.sum(fw * (est - fixed) ** 2); df = len(est) - 1
    i2 = max(0, (q - df) / q) * 100 if q > 0 else 0
    pred_se = np.sqrt(tau + se ** 2)
    return {"n_hospitals": len(est), "pooled": pooled, "ci_low": pooled - 1.96 * se, "ci_high": pooled + 1.96 * se, "tau_squared": tau, "q": q, "q_df": df, "i_squared_percent": i2, "prediction_interval_low": pooled - 1.96 * pred_se, "prediction_interval_high": pooled + 1.96 * pred_se}


def main():
    with np.load(OUT / "prediction_time_eicu_external_hist_gradient_boosting.npz") as d:
        y = d["y"].astype(np.int8); p = d["p"].astype(float); record = d["record_id"].astype(np.int64); patient = d["patient_id"].astype(np.int64)
    site = pd.read_csv(SITE_MAP, usecols=["record_id", "hospital_id"]).drop_duplicates("record_id")
    lookup = site.set_index("record_id").hospital_id
    hospital = pd.Series(record).map(lookup).to_numpy()
    if pd.isna(hospital).any(): raise RuntimeError("Missing hospital mapping")
    hospital = hospital.astype(np.int64)
    frame = pd.DataFrame({"hospital_id": hospital, "record_id": record, "patient_id": patient, "y": y})
    event_by_stay = frame.groupby(["hospital_id", "record_id"], sort=False).y.max().reset_index()
    counts = event_by_stay.groupby("hospital_id").y.agg(event_stays="sum", stays="size")
    counts["non_event_stays"] = counts["stays"] - counts["event_stays"]
    landmark_counts = frame.groupby("hospital_id").size().rename("landmarks")
    counts = counts.join(landmark_counts)
    eligible = counts.index[
        (counts["landmarks"] >= 1000)
        & (counts["event_stays"] >= 20)
        & (counts["non_event_stays"] >= 20)
    ].astype(np.int64).to_numpy()
    rows = []
    for h in sorted(eligible):
        mask = hospital == h
        _, codes = np.unique(patient[mask], return_inverse=True)
        cal = ci.cluster_robust_calibration(y[mask], p[mask], codes.astype(np.int32))
        citl_se = (cal["calibration_in_the_large_ci_high"] - cal["calibration_in_the_large_ci_low"]) / 3.92
        slope_se = (cal["calibration_slope_ci_high"] - cal["calibration_slope_ci_low"]) / 3.92
        rows.append({"hospital_id": int(h), "n_landmarks": int(mask.sum()), "n_clusters": int(np.unique(codes).size), **cal, "citl_se": citl_se, "slope_se": slope_se, "log_slope": np.log(cal["calibration_slope"]), "log_slope_se": slope_se / cal["calibration_slope"]})
    hospitals = pd.DataFrame(rows)
    hospitals.to_csv(OUT / "prediction_time_hospital_calibration.csv", index=False)
    summaries = []
    citl = reml(hospitals.calibration_in_the_large, hospitals.citl_se ** 2)
    summaries.append({"metric": "calibration_in_the_large", **citl})
    slope = reml(hospitals.log_slope, hospitals.log_slope_se ** 2)
    for key in ["pooled", "ci_low", "ci_high", "prediction_interval_low", "prediction_interval_high"]: slope[key] = float(np.exp(slope[key]))
    summaries.append({"metric": "calibration_slope", **slope})
    pd.DataFrame(summaries).to_csv(OUT / "prediction_time_hospital_random_effects_summary.csv", index=False)
    print(pd.DataFrame(summaries).to_string(index=False))

if __name__ == "__main__": main()
