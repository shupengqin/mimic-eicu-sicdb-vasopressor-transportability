# Revised vasopressor transportability analysis

This package accompanies the manuscript. The analytical run is dated 30 September 2026; release revision-2026-10-03 updates the input-audit reporting on 3 October 2026. The original reviewed implementation remains available at commit 7844ca86089cca0a6b3196fc02007f338e728b01. SHA256.csv binds this public revision; PUBLICATION_PROVENANCE.csv maps the originally supplied package to the current public files, and RELEASE_UPDATE_PROVENANCE.csv identifies changes since the preceding publication. Original databases, patient/stay/landmark extracts, individual predictions, fitted models, and credentials are excluded.

## Data and software

Use credentialed MIMIC-IV 3.1, eICU-CRD 2.0, and SICdb 1.0.8 access. Import MIMIC/eICU into PostgreSQL databases named mimiciv31 and eicu, including the MIMIC derived tables used by the SQL. Put SICdb cases.csv.gz, d_references.csv.gz, data_float_h.csv.gz, laboratory.csv.gz, and medication.csv.gz in SICDB_PATH. Check the exact reference filename in legacy/src/run_validation.py. Software versions are in results/software_versions.csv; Matplotlib 3.11.0 generated figures. Set PSQL_PATH, PGHOST, PGPORT, PGUSER and use PostgreSQL authentication outside the package. No password is stored.

Run commands from this directory. The supplied sql/*_revised.sql files are the actual executed queries; do not regenerate them merely to run the analysis. All DROP statements in revised extraction address session temporary tables only.

1. python run_extract.py mimic
2. python run_extract.py eicu
3. python prepare_sicdb.py
4. python prepare_arrays.py mimic; python prepare_arrays.py eicu; python prepare_arrays.py sicdb
5. python analyse.py fit
6. python analyse.py test; python analyse.py eicu; python analyse.py sicdb
7. python additional_audits.py development; python additional_audits.py selection; python additional_audits.py test; python additional_audits.py eicu; python additional_audits.py sicdb
8. python additional_audits.py hospitals
9. python test_revision_logic.py; python test_input_audit.py; python verify_numeric.py
10. python figures.py

Run the memory-intensive cohort stages sequentially. First-agent queries run during the corresponding audit. Individual-level outputs stay in private/. Use REVISION_PRIVATE, REVISION_RESULTS, REVISION_EXPORT to redirect them.

## Historical input audits

The delivered MIMIC/eICU before/after comparison was run against the locally retained submitted extracts. To reproduce those historical differences, set REVISION_SUBMITTED_INPUTS to a directory containing mimic_prediction_time.csv, eicu_prediction_time.csv, and sicdb_prediction_time.csv from the submitted implementation. These restricted extracts are not released. Without them, revised primary extraction, arrays, model fitting, and evaluation still run; historical MIMIC/eICU input-change outputs are not newly generated. SICdb reconstructs the submitted-style feature table from its source files if it is absent, then applies the timing correction. Historical reproducibility does not establish who viewed external outcomes or whether those views influenced earlier project choices.

execution_snapshot/ preserves the extraction/array scripts used in the completed run and a configuration snapshot with four machine-specific paths explicitly redacted. The source snapshot hash is retained in PUBLICATION_PROVENANCE.csv; the public snapshot is not byte-identical to the local original. Use the top-level config.py to run the package. Top-level configuration has portable path defaults and optional historical-input handling; calculation definitions are unchanged. The exported adapters were syntax checked and alert logic tested, but a second full database rerun of this portable package was not performed. Local database names remain explicit in the scripts.

legacy/ contains only required helper modules and source SQL for provenance. Do not use the legacy primary-analysis entry points to generate revised manuscript results. prepare_sql.py is a record of the query revision, not a required stage for the already supplied executed SQL.

## Definitions

An event requires first initiation strictly after the landmark and before both landmark+6h and unit exit. SICdb hourly aggregates enter at the end of their hour. Ranges apply before study-level aggregation. Correct emitted alerts must have a positive label; premature false alerts can suppress later opportunities. Brier null forecasts for held-out splits use calibration prevalence. The revision seed is 20260930; HGB retains 20260823. All revised outputs use current cohorts unless explicitly labeled otherwise.

## Public release checks

The release copies only files listed in the verified supplementary-code manifest. Python caches, raw extracts, individual records, predictions and model objects are excluded. The initial publication checked 109 supplied manifest entries and redacted four machine-specific configuration paths. The 3 October update recomputed the SICdb aligned input audit at the timing/filter audit precision; other existing aggregate results and executed SQL are unchanged. Historical execution snapshots are preserved. Release notes and provenance manifests distinguish the updates from the completed analytical run.

Install dependencies with `python -m pip install -r requirements.txt`. Python 3.14.5 was used for the saved analyses. The existing alert-window/suppression checks and Python syntax checks were rerun for publication. This was not another full extraction or model-training run. The source data and retained private arrays are required for full numeric verification.

The study uses patient reference-period groups in MIMIC, not verified strictly chronological admission separation. Earlier exposure to external results cannot be reconstructed from this code and remains disclosed in the manuscript. This release does not remove that historical limitation.

## SICdb audit synchronization on 3 October 2026

Both SICdb comparisons now use float64 feature values before model-array casting, with relative tolerance 1e-7, absolute tolerance 1e-8, and missing-to-missing equality. The first is the aligned baseline-to-final comparison; the second separates timing and subsequent range-filter ordering. Their final per-feature counts agree, totaling 1,116,906. Sequential timing and filter counts are not additive.

MIMIC-IV/eICU aligned audits retain their recorded float32, relative 1e-6 and absolute 1e-7 convention. The change does not alter float32 model inputs, risk sets, labels, scores or fitted models. The untouched execution snapshots document the convention used in the original run; use the top-level scripts for the synchronized audit.

With the retained submitted SICdb input and revised extract available in the configured locations, `python audit_sicdb_inputs.py` recomputes the audit without modifying private arrays or fitting models. It asserts agreement with every row of `results/sicdb_timing_and_filter_changes.csv` and writes the aligned CSV plus `results/sicdb_audit_precision_check.json`. That check reports the 44-cell discrepancy under the old convention. `results/sicdb_legacy_reconstruction_check.json` contains the aggregate results of the separate retained-input timing-convention check described in the response letter; it contains no individual records.

The two reviewer-facing check files are `test_revision_logic.py` and `results/verification.json`. The latter records the completed analytical run, not a new full-database rerun. The 3 October checks include the actual SICdb array-preparation path on retained inputs, identity of all saved arrays, all 42 audit counts, Python syntax and the alert/input-audit tests. See RELEASE_NOTES_2026-10-03.md.
