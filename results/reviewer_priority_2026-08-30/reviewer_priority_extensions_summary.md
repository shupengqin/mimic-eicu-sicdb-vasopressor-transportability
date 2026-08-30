# Reviewer-priority post hoc analyses

All analyses below were conducted after the primary results were known. They do not replace the corrected primary model or convert the study into a preregistered analysis.

## Prediction-time-identifiable hour-6 analysis

Eligibility was determined at ICU hour 6 without requiring future ICU presence. Target-pressor initiation before hour 12 or ICU exit was the event; earlier ICU exit without initiation was retained as a competing outcome.

| Dataset | Stays | Events | Competing exits | AUROC (95% CI) | AUPRC (95% CI) | Brier skill |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MIMIC-IV 2020-2022 temporal test | 9,535 | 167 | 277 (2.9%) | 0.786 (0.752-0.817) | 0.134 (0.095-0.200) | -0.275 |
| eICU-CRD | 168,645 | 3,362 | 10,823 (6.4%) | 0.826 (0.820-0.831) | 0.079 (0.075-0.084) | -0.029 |
| SICdb | 3,997 | 246 | 228 (5.7%) | 0.725 (0.696-0.755) | 0.155 (0.125-0.202) | 0.038 |

## Outcome and measurement-process sensitivities

| Dataset | Analysis | Event stays | AUROC (95% CI) | AUPRC (95% CI) | Brier |
| --- | --- | ---: | ---: | ---: | ---: |
| MIMIC-IV 2020-2022 temporal test | norepinephrine_first_hgb | 236 | 0.767 (0.738-0.794) | 0.021 (0.017-0.029) | 0.00686 |
| MIMIC-IV 2020-2022 temporal test | norepinephrine_only_hgb | 231 | 0.766 (0.736-0.796) | 0.020 (0.015-0.026) | 0.00673 |
| MIMIC-IV 2020-2022 temporal test | availability_only_hgb | 317 | 0.671 (0.636-0.702) | 0.038 (0.027-0.051) | 0.00863 |
| MIMIC-IV 2020-2022 temporal test | missingness_only_hgb | 317 | 0.627 (0.595-0.656) | 0.031 (0.022-0.044) | 0.00860 |
| eICU-CRD | norepinephrine_first_hgb | 4,484 | 0.858 (0.852-0.863) | 0.055 (0.052-0.058) | 0.00784 |
| eICU-CRD | norepinephrine_only_hgb | 4,063 | 0.855 (0.850-0.861) | 0.047 (0.044-0.050) | 0.00720 |
| eICU-CRD | availability_only_hgb | 6,297 | 0.691 (0.685-0.698) | 0.023 (0.022-0.024) | 0.01103 |
| eICU-CRD | missingness_only_hgb | 6,297 | 0.657 (0.650-0.664) | 0.019 (0.018-0.020) | 0.01105 |
| SICdb | norepinephrine_first_hgb | 540 | 0.701 (0.676-0.723) | 0.133 (0.111-0.159) | 0.04494 |
| SICdb | norepinephrine_only_hgb | 539 | 0.703 (0.678-0.725) | 0.127 (0.107-0.150) | 0.04515 |
| SICdb | availability_only_hgb | 555 | 0.538 (0.515-0.561) | 0.055 (0.049-0.061) | 0.04712 |
| SICdb | missingness_only_hgb | 555 | 0.510 (0.488-0.529) | 0.051 (0.047-0.057) | 0.04709 |

## Repeated held-out Brier skill

The reference forecast was the event prevalence estimated in each identifier-disjoint 20% calibration subset and applied unchanged to its 80% evaluation subset.

| Dataset | Uncalibrated Brier skill, median (IQR) | Intercept-recalibrated Brier skill, median (IQR) |
| --- | ---: | ---: |
| MIMIC-IV 2020-2022 temporal test | -0.180 (-0.185 to -0.174) | 0.003 (-0.000 to 0.007) |
| eICU-CRD | -0.013 (-0.014 to -0.012) | 0.024 (0.023 to 0.024) |
| SICdb | 0.029 (0.028 to 0.031) | 0.032 (0.030 to 0.035) |

## Fixed operating policies

Thresholds were selected in each calibration subset to target either five alerts per 100 eligible landmark rows or 80% landmark sensitivity, then applied unchanged to the evaluation subset. Policy lead time is the interval from the first emitted true-positive alert to the first target-pressor timestamp among detected event stays.

| Dataset | Strategy | Sensitivity | Alerts/100 rows | False episodes/100 rows | Event-stay sensitivity | Lead time, h |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MIMIC-IV 2020-2022 temporal test | 5 alerts/100 rows | 0.249 | 5.03 | 1.73 | 0.416 | 1.48 |
| MIMIC-IV 2020-2022 temporal test | 80% calibration sensitivity | 0.805 | 44.14 | 10.13 | 0.882 | 2.00 |
| eICU-CRD | 5 alerts/100 rows | 0.351 | 5.03 | 1.57 | 0.494 | 2.20 |
| eICU-CRD | 80% calibration sensitivity | 0.801 | 27.76 | 6.97 | 0.876 | 2.32 |
| SICdb | 5 alerts/100 rows | 0.198 | 4.94 | 1.43 | 0.318 | 2.13 |
| SICdb | 80% calibration sensitivity | 0.803 | 57.97 | 13.28 | 0.879 | 2.34 |

## Hospital-level calibration model

The 73 eligible eICU hospitals were synthesized with descriptive random-effects models using patienthealthsystemstayid-clustered within-hospital standard errors. CITL was analyzed on its original scale and calibration slope on the log scale. This is not prospective new-hospital recalibration.

| Metric | Pooled estimate (95% CI) | Tau-squared | I-squared | 95% prediction interval |
| --- | ---: | ---: | ---: | ---: |
| calibration_in_the_large | 0.003 (-0.105-0.112) | 0.1989 | 91.8% | -0.878-0.884 |
| calibration_slope | 1.170 (1.128-1.215) | 0.0180 | 72.0% | 0.898-1.526 |

## Predictor-QC boundary

The 42-predictor long table reports model-input nonmissingness, post-Python-range missingness, range violations, and accepted-value distributions in five cohorts. Quantiles use an exact SICdb calculation and a seeded 5% row sample for the larger MIMIC-IV and eICU cohorts. eICU source-observation range checks occurred in SQL before aggregation, so the table cannot reconstruct counts of observations rejected upstream.

## Interpretation boundary

Equal-total-stay evaluation estimates performance for an average stay; the primary landmark-weighted analysis estimates performance for an average eligible prediction opportunity and gives more influence to stays contributing more landmarks. Availability-only performance can reveal use of measurement patterns but is not a full measurement-process model because recency and sampling frequency were not included.
