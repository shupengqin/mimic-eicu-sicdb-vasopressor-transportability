> Historical documentation for the submitted analysis at commit 7844ca86089cca0a6b3196fc02007f338e728b01. Superseded by revisions/2026-09-30 for revised manuscript results.

# Prediction time vasopressor transportability

This repository contains the reproducible SQL, Python analysis code, protocol documentation, and disclosure reviewed aggregate outputs for the prediction time identifiable hourly analysis reported in the accompanying manuscript.

## Scope

The primary estimand is an hourly operational risk set. A landmark is eligible when an adult patient is still in the ICU or unit and no target continuous vasopressor has been documented before that landmark. The event is the first documented initiation of norepinephrine, epinephrine, phenylephrine, vasopressin, or dopamine before the earlier of six hours after the landmark or ICU or unit exit. A stay exit without initiation is retained as an operational non event. This endpoint describes documented treatment transitions; it is not a measure of physiologic shock onset, treatment need, or clinician intent.

The analysis uses MIMIC IV v3.1 for development, temporal model selection, and temporal testing; eICU CRD v2.0 for US multicenter evaluation; and SICdb v1.0.8 for single center Austrian evaluation. Source databases are restricted access and are not redistributed here.

## Repository layout

- `docs/` protocol, data access, and code availability documentation
- `sql/` extraction queries for MIMIC IV and eICU CRD
- `src/` analysis, extension, hospital heterogeneity, cohort summary, and figure scripts
- `results/prediction_time_2026-09-05/` disclosure reviewed aggregate source data for manuscript tables and figures

## Reproduction order

1. Obtain the three source databases directly from PhysioNet and comply with each database's credentialing, training, data use, and contributor review requirements.
2. Run the database specific SQL under the relevant data access controls and save the resulting extracts locally. Set `PREDICTION_TIME_WORK_DIR` to that protected working directory.
3. Run `src/prediction_time_analysis.py` to fit the fixed candidates and generate locked evaluation arrays and aggregate metrics.
4. Run `src/prediction_time_extensions.py`, `src/prediction_time_hospital.py`, `src/summarize_prediction_time_cohorts.py`, and `src/make_prediction_time_figures.py` in that order. Set `PREDICTION_TIME_OUTPUT_DIR` and `PREDICTION_TIME_FIGURE_DIR` when outputs are stored outside the repository.

The scripts check required columns before analysis. Random seeds, model hyperparameters, accepted value ranges, endpoint definitions, and uncertainty procedures are recorded in `docs/prediction_time_protocol.md`.

The analysis environment was Python 3.14.5 with NumPy 2.5.0, pandas 3.0.4, SciPy 1.18.0, scikit-learn 1.9.0, joblib 1.5.3, matplotlib 3.11.0, and python-docx 1.2.0. The corresponding lock file in the repository is the installation reference; package availability can vary by operating system.

## What is excluded

The public repository does not contain raw database files, patient-, stay-, or landmark-level extracts, row-level predictions, model objects, credentials, database connection details, or local machine paths. The aggregate files are intended to support the published tables and figures; they should not be reverse engineered into individual records.

## Data and code access

See `docs/prediction_time_data_code_availability.md` for source database citations, access requirements, and the data availability wording used in the manuscript. The repository is public, but access to the source data remains controlled by PhysioNet and the SICdb contributor review process.

## Citation

When using this code, cite the accompanying manuscript and the source database publications listed in `docs/prediction_time_data_code_availability.md`. A permanent archive DOI or software license is not asserted here because those author owned release details must be confirmed separately.
