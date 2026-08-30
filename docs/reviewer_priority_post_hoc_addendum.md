# Reviewer-priority post hoc analysis addendum

## Status

These analyses were specified after review of the corrected primary results. They are post hoc robustness, transportability, and policy analyses. They do not alter the corrected primary model, make the study prospectively preregistered, or establish clinical utility.

## Corrected-rerun lock

The corrected rerun used MIMIC-IV `anchor_year_group` for the 2008-2016 development, 2017-2019 model-selection, and 2020-2022 temporal-test partitions. The HGB hyperparameters and common preprocessing rules were fixed for the corrected rerun, and no temporal-test or external cohort was used for fitting or selection. Earlier exploratory project work had already inspected some external results. The term `locked` therefore describes the corrected rerun and is not a claim of prospective blinding throughout the project.

## Added analyses

1. Prediction-time-identifiable hour-6 estimand: eligibility is determined at ICU hour 6 without requiring future ICU presence. The event is first target-pressor initiation before ICU hour 12 or ICU exit. ICU exit before hour 12 without prior initiation is retained as a competing outcome.
2. First-agent composition and outcome sensitivity: norepinephrine, phenylephrine, epinephrine, vasopressin, dopamine, and concurrent first agents are reported as mutually exclusive categories. The strict norepinephrine-only endpoint requires norepinephrine to be the sole agent at the first target-pressor timestamp. A separate norepinephrine-at-first endpoint includes concurrent first timestamps containing norepinephrine but excludes norepinephrine started only after another target pressor.
3. Brier skill: the reference forecast is the event prevalence estimated in an identifier-disjoint 20% calibration subset and applied unchanged to the 80% evaluation subset. Results are summarized over 100 repetitions.
4. Fixed operating policies: thresholds are selected in the calibration subset to target five alerts per 100 eligible landmark rows or 80% landmark sensitivity. Thresholds are applied unchanged to the evaluation subset. A six-hour suppression rule converts repeated crossings into alert episodes.
5. Policy lead time: lead time is the interval from the first emitted true-positive alert to the first target-pressor timestamp among detected event stays. Positive landmark-window timing is not reported as alarm lead time.
6. Predictor QC: all 42 predictors are audited in five cohorts. Exact accepted-value extrema and missingness counts are reported; quantiles use all SICdb rows and a reproducible seeded 5% row sample for larger cohorts. eICU source-observation range filtering occurred before aggregation, so upstream rejection counts cannot be reconstructed from the model-input table.
7. Measurement-pattern models: the strict missingness-only HGB uses 39 post-range-filter availability indicators and excludes index hour and all physiologic values. The availability-only HGB adds index hour. These analyses test whether measurement presence carries predictive information; neither is a complete measurement-process model because recency and sampling frequency are omitted.
8. Hospital-level calibration: 73 eICU hospitals meeting the descriptive sample and event-count criteria receive `patienthealthsystemstayid`-clustered CITL and slope estimates. Descriptive random-effects synthesis uses CITL on its original scale and slope on the log scale.

## Redistribution boundary

The first-agent extracts, hour-6 row-level inputs, prediction arrays, and fitted models remain local and are excluded from the public repository. Public results contain only disclosure-reviewed aggregates. Hospital-level aggregates require final author disclosure review before journal submission.
