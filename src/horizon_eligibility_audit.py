"""Quantify landmark exclusion caused by requiring six future ICU hours."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_validation as rv


def sicdb_summary() -> dict:
    cases = rv.read_sicdb_cases(sorted(rv.SICDB_MAIN_UNITS))
    first = rv.read_sicdb_first_pressor(cases)
    cases = cases.copy()
    cases["first_start_s"] = cases["record_id"].map(first)
    rows = []
    for row in cases.itertuples(index=False):
        first_start = float(row.first_start_s) if pd.notna(row.first_start_s) else np.nan
        for hour in range(6, 25):
            index_s = hour * 3600.0
            if float(row.icu_los_s) <= index_s:
                continue
            if pd.notna(first_start) and first_start < index_s:
                continue
            complete = float(row.icu_los_s) >= (hour + 6) * 3600.0
            observed_event = (
                not complete
                and pd.notna(first_start)
                and first_start < (hour + 6) * 3600.0
                and first_start <= float(row.icu_los_s)
            )
            rows.append(
                {
                    "record_id": int(row.record_id),
                    "complete_horizon": complete,
                    "incomplete_with_observed_event": observed_event,
                }
            )
    frame = pd.DataFrame(rows)
    return {
        "dataset": "sicdb",
        "at_risk_landmarks_present_at_index": len(frame),
        "complete_six_hour_landmarks": int(frame.complete_horizon.sum()),
        "incomplete_six_hour_landmarks": int((~frame.complete_horizon).sum()),
        "incomplete_with_observed_event": int(frame.incomplete_with_observed_event.sum()),
        "incomplete_censored_without_observed_event": int(
            ((~frame.complete_horizon) & (~frame.incomplete_with_observed_event)).sum()
        ),
        "at_risk_stays": int(frame.record_id.nunique()),
        "stays_with_complete_horizon_landmark": int(
            frame.loc[frame.complete_horizon, "record_id"].nunique()
        ),
        "stays_with_incomplete_horizon_landmark": int(
            frame.loc[~frame.complete_horizon, "record_id"].nunique()
        ),
    }


def main() -> None:
    mimic_path = rv.WORK / "mimic_horizon_audit.csv"
    eicu_path = rv.WORK / "eicu_horizon_audit.csv"
    rv.run_psql("mimiciv31", rv.WORK / "mimic_horizon_audit.sql", mimic_path)
    rv.run_psql("eicu", rv.WORK / "eicu_horizon_audit.sql", eicu_path)
    rows = [
        pd.read_csv(mimic_path).iloc[0].to_dict(),
        pd.read_csv(eicu_path).iloc[0].to_dict(),
        sicdb_summary(),
    ]
    frame = pd.DataFrame(rows)
    frame["incomplete_landmark_percent"] = (
        100 * frame.incomplete_six_hour_landmarks / frame.at_risk_landmarks_present_at_index
    )
    frame["incomplete_stay_percent"] = (
        100 * frame.stays_with_incomplete_horizon_landmark / frame.at_risk_stays
    )
    frame.to_csv(rv.OUTPUTS / "corrected_horizon_eligibility_audit.csv", index=False)

    labels = {"mimiciv": "MIMIC-IV", "eicu": "eICU-CRD", "sicdb": "SICdb"}
    lines = [
        "# Six-hour horizon eligibility audit",
        "",
        "This audit starts from hourly landmarks at which the patient was still in the ICU or unit and had not previously received a target vasopressor. It quantifies the landmarks removed because six subsequent ICU hours were unavailable.",
        "",
        "| Dataset | At-risk landmarks present at index | Complete 6-h horizon | Incomplete 6-h horizon | Incomplete, observed event before exit | Incomplete, censored without observed event | Incomplete landmarks, % |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| {labels[row.dataset]} | {int(row.at_risk_landmarks_present_at_index):,} | "
            f"{int(row.complete_six_hour_landmarks):,} | {int(row.incomplete_six_hour_landmarks):,} | "
            f"{int(row.incomplete_with_observed_event):,} | {int(row.incomplete_censored_without_observed_event):,} | "
            f"{row.incomplete_landmark_percent:.1f} |"
        )
    lines.extend(
        [
            "",
            "The primary binary estimand is conditional on remaining under ICU observation for the complete six-hour horizon. Discharge or transfer before six hours is informative censoring, not a known negative outcome. The fixed ICU-hour-6 sensitivity analysis avoids this exclusion because the study already requires at least 12 hours of ICU follow-up.",
        ]
    )
    (rv.OUTPUTS / "corrected_horizon_eligibility_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("Horizon eligibility audit completed.", flush=True)


if __name__ == "__main__":
    main()
