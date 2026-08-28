# Transportability of hourly vasopressor-initiation prediction

Reproducible analysis code for the retrospective MIMIC-IV, eICU-CRD and SICdb study of temporal and geographic transportability of an hourly model for first documented continuous vasopressor initiation within six hours.

## Scope

This repository contains SQL extraction templates, Python analysis scripts, reporting helpers and methodological documentation. It does not contain raw database files, patient-level extracts, row-level predictions, credentials or frozen model objects.

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
3. Run the extraction and modelling scripts in `src/` using local, credentialed data.
4. Run the reporting and figure scripts only on locally generated data.
5. Review the resulting aggregate outputs for disclosure compliance before sharing.

The manuscript reports a six-hour complete-horizon conditional estimand. It should not be interpreted as unconditional dynamic risk, clinical effectiveness or deployment readiness. Decision curves and alert-suppression analyses are exploratory.

## Reproducibility and sharing

The source databases, patient/stay/landmark-level extracts, row-level predictions and unapproved model objects must not be uploaded here. Public releases should contain only scrubbed code, environment metadata and disclosure-reviewed aggregate outputs. Any frozen-model sharing requires approval from the relevant data custodians and a controlled-access route.

## Citation

When the manuscript is published, replace this section with the final citation and DOI. Until then, cite the corresponding manuscript and the official PhysioNet dataset records.
