# Corrected study protocol and statistical analysis plan

## Status and transparency statement

This document records the corrected final retrospective analysis for reproducibility. It is not a preregistration. An earlier analysis used shifted ICU admission years and was invalid for temporal partitioning. The corrected run uses MIMIC-IV `patients.anchor_year_group`. Although external cohorts were not used by the corrected model-selection code, the overall project and selection procedure were not prospectively prespecified; the manuscript must describe them as corrected retrospective model selection.

## Design and objectives

This is a retrospective, multi-database prediction-model development and external-validation study using repeated hourly landmarks. The primary objective is to quantify whether a locked corrected-rerun model for impending continuous vasopressor initiation transports across time, US hospitals, and an Austrian ICU system. Secondary objectives are to compare nonlinear and transparent algorithms, characterize calibration drift and alert burden, assess hospital/unit heterogeneity, and evaluate intercept-only local recalibration.

## Data sources and partitions

| Role | Database and partition | Use |
| --- | --- | --- |
| Development | MIMIC-IV anchor-year groups 2008-2016 | Fit candidate algorithms |
| Model selection | MIMIC-IV anchor-year group 2017-2019 | Select candidate by AUROC, with AUPRC as tie-breaker |
| Final refit | MIMIC-IV anchor-year groups 2008-2019 | Refit both algorithms after selection |
| Locked temporal test | MIMIC-IV anchor-year group 2020-2022 | Temporal validation only |
| External validation | eICU-CRD | US multicenter validation only |
| External validation | SICdb CWIN and INBD | Austrian single-center geographic validation only |

Patient identifiers do not overlap across MIMIC partitions because `anchor_year_group` is patient-level. The final validation cohorts were not used for fitting or selection in the corrected run. Earlier exploratory project work had inspected some external results, so this statement describes the corrected rerun and must not be presented as prospective blinding to all external evidence.

## Population and landmarks

Eligible records are adult ICU or unit stays with at least 12 h of follow-up. Candidate landmarks occur hourly from ICU hour 6 through hour 24. A landmark is retained only if the ICU stay continues for the complete 6-h prediction horizon and no target vasopressor was documented before that landmark. Multiple landmarks from one stay are retained and treated as repeated observations. This defines a conditional estimand among patients remaining under ICU observation for six hours; the proportion of otherwise at-risk landmarks excluded for incomplete follow-up is audited by database. All timestamps are retrospective chart or measurement timestamps. We did not audit result-reporting delay, data-entry delay, backfilled documentation, or maximum measurement age, so a recorded value may not have been available in real time at the landmark.

## Outcome

The binary outcome at each landmark is the first documented start of continuous norepinephrine, epinephrine, phenylephrine, vasopressin, or dopamine within the next 6 h. Database-specific treatment records define documented initiation, not physiologic shock onset or clinician intent. A strict-future sensitivity analysis removes positive landmarks whose recorded initiation time equals the landmark time.

## Predictors and preprocessing

The 42 predictors comprise age, sex, landmark hour, the latest value plus 6-h mean/minimum/maximum for heart rate, systolic pressure, diastolic pressure, mean arterial pressure, respiratory rate, temperature, and oxygen saturation, and the latest available creatinine, sodium, potassium, bicarbonate, glucose, blood urea nitrogen, lactate, pH, hemoglobin, hematocrit, and white-cell count. Values outside common physiologic ranges are set to missing using the same rules in every database.

HGB handles missing values internally. Logistic regression uses median imputation, missingness indicators, standardization, and regularization. Missingness is reported before imputation for every predictor and cohort. No complete-case exclusion is used in the primary analysis.

## Candidate models and selection

The candidates are regularized logistic regression and `HistGradientBoostingClassifier` with learning rate 0.05, 200 iterations, 15 maximum leaf nodes, minimum leaf size 100, L2 regularization 1.0, and random seed 20260823. Both are fitted on MIMIC-IV 2008-2016 and evaluated in MIMIC-IV 2017-2019. The candidate with the higher AUROC is selected, with AUPRC as a tie-breaker. Both models are then refitted on MIMIC-IV 2008-2019; the selected HGB model is the primary performance model and logistic regression remains the transparent benchmark.

## Performance measures

Primary discrimination is AUROC. AUPRC is a key secondary discrimination measure because event prevalence differs substantially across cohorts. Probability accuracy is assessed by Brier score. Calibration is assessed by CITL, the jointly estimated calibration intercept and slope, calibration curves, and observed event fractions within prediction-risk deciles. Operational summaries include sensitivity, specificity, positive predictive value, and false alerts per 100 eligible landmark rows at stated thresholds. Each eligible landmark row represents one hourly prediction opportunity, and rows may repeat within a stay; these rates are not patient-time rates. Design-window lead times describe positive landmark timing and are not alarm lead times. In the fixed-policy analysis, lead time is instead defined from the first emitted true-positive alert to the first target-pressor timestamp among detected event stays.

## Inference

The independent resampling unit is the ICU or unit stay. AUROC, AUPRC, and Brier 95% confidence intervals use 500 stay-clustered percentile bootstrap replicates. HGB-minus-logistic contrasts use the same paired bootstrap samples. CITL and flexible calibration intercept/slope confidence intervals use stay-clustered sandwich standard errors. Landmark counts describe prediction opportunities and are not treated as independent patient counts. Patient-level clustering is a sensitivity analysis in MIMIC-IV and SICdb. eICU-CRD is additionally bootstrapped by hospital as the highest observed sampling level; hospital-specific estimates are descriptive and unshrunk.

No multiplicity-adjusted hypothesis-testing family is defined because the analysis emphasizes explicitly defined metrics, paired effect estimates, and 95% confidence intervals rather than dichotomous significance testing. Model contrasts across metrics and datasets should be interpreted jointly and conservatively.

All records meeting the eligibility and landmark rules are included. No formal a priori sample-size calculation is performed; precision is assessed using stay-clustered confidence intervals and reported event counts. No outcome resampling, synthetic sampling, or class weighting is used, and all probability-based metrics are evaluated at each cohort's observed event prevalence.

## Exploratory local recalibration and utility

Within each validation cohort, database-level grouping identifiers are stratified by whether any landmark is positive and randomly divided into 20% calibration and 80% evaluation subsets using seed 20260824. The grouping identifiers are MIMIC-IV `subject_id`, eICU-CRD `patienthealthsystemstayid`, and SICdb `PatientID`. An intercept shift with the prediction-logit slope fixed at 1 is fitted in the calibration subset and applied unchanged to the evaluation subset. No grouping identifier overlaps the subsets. Because eICU-CRD does not provide a permanent person identifier across separate hospitalizations, `patienthealthsystemstayid` prevents overlap within a health-system stay but cannot guarantee person-level separation across admissions. Calibration curves, Brier score, threshold performance, and decision-curve net benefit are evaluated only in the held-out 80% subset. Split stability is assessed post hoc over 100 repeated identifier-disjoint splits. This is an exploratory database-level recalibration analysis, not new-hospital external validation, and it does not represent prospective deployment or clinical benefit. Decision-curve net benefit is calculated at the landmark level and is interpreted only as a descriptive policy comparison because hourly rows overlap within stays.

## Sensitivity and heterogeneity analyses

- Strict-future outcome window excluding landmark-concurrent starts.
- eICU hospital-level analysis restricted to hospitals with at least 1,000 landmarks, 20 event-positive stays, and 20 event-negative stays.
- SICdb unit-level analyses in CWIN and INBD.
- SICdb all-four-unit sensitivity analysis.
- Cross-database predictor-missingness audit and outcome-harmonization audit.
- Exploratory age groups (18-44, 45-64, 65-79 and 80 years or older) and recorded binary-sex subgroups, using the same stay-clustered uncertainty methods as the primary analysis. No formal between-group tests or fairness constraints are defined; race and ethnicity are not analysed because they cannot be harmonized across all three databases.

## Post hoc methodological extensions

The following analyses were added after strict methodological review and must not be described as prospectively prespecified:

- Quantification of at-risk landmarks excluded for incomplete six-hour ICU follow-up.
- Stay-balanced evaluation and one-landmark-per-stay evaluation at ICU hour 6.
- Clinical ranking comparators using MAP, shock index, and modified shock index, plus fixed clinical rules.
- Vital-sign-only and full-minus-temperature HGB models trained in MIMIC-IV 2008-2019 with the fixed primary hyperparameters.
- Equal-total-stay-weight HGB training, with paired comparison against the primary landmark-weighted HGB.
- A one-hour clinical-action-gap HGB targeting first initiation during hours (1, 6], with starts during the first hour excluded from training and evaluation.
- Repeated identifier-disjoint intercept recalibration over 100 splits.
- Patient-level clustered inference and eICU hospital-level bootstrap inference.
- Six-hour suppression of repeated threshold crossings, summarized as event-stay sensitivity, episode PPV, and false alert episodes per 100 eligible landmark-hours. The denominator is the number of eligible landmark rows, not patient time.
- A prediction-time-identifiable ICU-hour-6 analysis requiring only that the patient remains in the ICU and has not received a target pressor at hour 6. Target-pressor initiation before hour 12 or ICU exit is the event; ICU exit before hour 12 without prior initiation is retained as a competing outcome.
- Mutually exclusive first-agent composition, with concurrent agents retained as a separate category. The strict norepinephrine-only HGB endpoint requires norepinephrine to be the sole agent at the first target-pressor timestamp. A second HGB endpoint permits concurrent first agents when norepinephrine is present. Norepinephrine added only after another target pressor does not qualify for either endpoint.
- Brier skill relative to a prevalence-only forecast estimated in each 20% identifier-disjoint calibration subset and applied to the corresponding 80% evaluation subset.
- Thresholds selected in the calibration subset to target either five alerts per 100 eligible landmark rows or 80% landmark sensitivity, followed by unchanged evaluation and six-hour suppression.
- Complete 42-predictor model-input QC across MIMIC-IV development, model-selection, and temporal-test cohorts, eICU-CRD, and SICdb. The table reports nonmissingness, range rejection, accepted-value distributions, and the boundary created by eICU range filtering before aggregation.
- A strict missingness-only HGB using 39 post-range-filter availability indicators and no physiologic values or index hour, plus an availability-only HGB that adds index hour. Neither is a complete measurement-process model because measurement recency and sampling frequency are omitted.
- Descriptive random-effects synthesis of CITL and calibration slope across 73 eligible eICU hospitals using within-hospital `patienthealthsystemstayid`-clustered standard errors. CITL is analyzed on its original scale and slope on the log scale.

No temporal-test or external data were used to fit or select any post hoc sensitivity model. External data were evaluated only after each post hoc model specification had been fixed. These analyses evaluate robustness and do not replace the corrected primary model-selection procedure.

## Software and reproducibility

The corrected run used Python 3.14.5, pandas 3.0.4, NumPy 2.5.0, SciPy 1.18.0, scikit-learn 1.9.0, matplotlib 3.11.0, joblib 1.5.3, and python-docx 1.2.0. Extraction and analysis file hashes are recorded in `corrected_run_manifest.json`. Frozen model artifacts and prediction arrays are retained with the corrected outputs but are not included in the unrestricted public repository.
