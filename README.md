# Transportability of hourly vasopressor-initiation prediction

Reproducible analysis code for the retrospective MIMIC-IV, eICU-CRD and SICdb study of temporal and geographic transportability of an hourly model for first documented continuous vasopressor initiation within six hours.

## Scope

This repository contains SQL extraction templates, Python analysis scripts, reporting helpers, methodological documentation, and disclosure-reviewed aggregate results. It does not contain raw database files, patient-level extracts, row-level predictions, credentials, or model objects.

## Analysis status and provenance

The study was not prospectively preregistered. The corrected rerun used MIMIC-IV `anchor_year_group` for temporal partitioning and locked the selected algorithm, hyperparameters, and preprocessing rules before the corrected temporal and external scoring pass. External data were not used for fitting, preprocessing estimation, hyperparameter selection, or algorithm selection in that corrected rerun. Earlier exploratory project work had already inspected some external results, however. In this repository, `locked` or `frozen` therefore means no external refitting or selection during the corrected rerun; it does not mean that the investigators were prospectively blinded to every external result.

The reviewer-priority analyses dated 30 August 2026 are explicitly post hoc. They test alternative estimands, first-agent composition, strict norepinephrine-only and norepinephrine-at-first outcomes, probability skill, operating policies, measurement availability, predictor quality, and hospital-level calibration heterogeneity. They do not replace the corrected primary analysis.

## Data access

Obtain MIMIC-IV, eICU-CRD and SICdb directly from PhysioNet under their applicable credentialing, training and data-use agreements. Configure local paths and PostgreSQL connection settings with environment variables; do not commit passwords or local database paths.

Important environment variables for `src/run_validation.py`:

```text
SICDB_PATH       Local SICdb directory
PSQL_PATH        psql executable, if not available on PATH
MIMIC_WORK_DIR   Working directory for local extracts
MIMIC_OUTPUT_DIR Output directory for local results
MIMIC_DB         MIMIC-IV database name
EICU_DB          eICU database name
PGHOST           PostgreSQL host
PGPORT           PostgreSQL port
PGUSER           PostgreSQL user
PGPASSWORD       PostgreSQL password, supplied only through the environment
```

## Reproduction outline

1. Create a local working directory outside this repository and obtain the three source databases through PhysioNet.
2. Review and adapt the SQL templates in `sql/` to the local schema and database versions.
3. Run `src/run_validation.py` to create the local landmark extracts and the
   baseline outputs. Run `src/corrected_modeling.py` to fit and freeze the
   corrected MIMIC-IV models, then run the clustered, recalibration,
   sensitivity, table, and figure scripts in `src/` in that order.
4. Use `src/generate_submission_tables.py` and
   `src/generate_submission_documents.py` only after the aggregate outputs
   have been reviewed.
5. Run `src/reviewer_data_extensions.py` locally to create the nonredistributable
   first-agent and prediction-time-identifiable hour-6 inputs. Then run
   `src/reviewer_analysis_extensions.py` by stage (`composition`, `hour6`,
   `models`, `policy`, `qc`, `hospital`, and `summary`).
6. Review the resulting aggregate outputs for disclosure compliance before
   sharing.

The primary manuscript analysis reports a six-hour complete-horizon conditional estimand. A post hoc ICU-hour-6 analysis determines eligibility using information available at hour 6, retains early ICU exit as a competing outcome, and predicts target-pressor initiation before hour 12 or ICU exit. Neither estimand should be interpreted as treatment need, clinical effectiveness, or deployment readiness. Decision curves and alert-policy analyses are exploratory landmark-level analyses. Their rates use eligible landmark rows as the denominator because rows are repeated hourly opportunities within stays; they are not patient-time rates.

Brier skill is reported relative to a prevalence-only forecast estimated in each identifier-disjoint calibration subset and applied unchanged to the corresponding evaluation subset. Fixed-policy thresholds are also selected only in the calibration subset. Policy lead time is the interval from the first emitted true-positive alert to the first target-pressor timestamp among detected event stays, not the distribution of all positive landmark windows.

The eICU extraction template explicitly prioritizes periodic over aperiodic
vital signs at the same timestamp, removes duplicate stay-time-variable rows,
uses token-boundary laboratory matching, and applies physiologic range checks
before window aggregation. The primary manuscript numbers were generated from
the disclosure-reviewed aggregate outputs in the accompanying package. If the
SQL templates are changed or rerun, regenerate all downstream outputs and
record the new hashes in the run manifest.

## Reproducibility and sharing

The source databases, patient/stay/landmark-level extracts, row-level predictions, and model objects must not be uploaded here. Public releases should contain only scrubbed code, environment metadata, and disclosure-reviewed aggregate outputs. Any model sharing requires approval from the relevant data custodians and a controlled-access route.

The public aggregate results for the final reviewer-priority analyses are under `results/reviewer_priority_2026-08-30/`. The local `work/` and `outputs/` directories remain ignored.

## Citation

When the manuscript is published, replace this section with the final citation and DOI. Until then, cite the corresponding manuscript and the official PhysioNet dataset records.
