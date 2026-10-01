# Vasopressor prediction across MIMIC-IV, eICU-CRD and SICdb

## Current manuscript revision

The corrected analysis, executed extraction SQL and aggregate outputs are in **[revisions/2026-09-30](revisions/2026-09-30/README.md)**, published on 1 October 2026. Use that directory to reproduce the revised manuscript. It contains a runnable configuration, dependency pins, execution records and SHA-256 checksums.

The revision corrects SICdb hourly aggregate availability, source range-filter order, the eICU discharge-offset query, treatment-time risk-set boundaries, correct-alert detection and lead time, and held-out Brier reference forecasts. It adds paired model-difference intervals, current-cohort quality checks, laboratory measurement-age sensitivities, and exit-composition audits. MIMIC groups are described as patient reference-period groups rather than strictly chronological admission splits. See [the change record](docs/revision_2026-09-30.md).

## Reproduce the revised analysis

1. Obtain authorized access to MIMIC-IV 3.1, eICU-CRD 2.0 and SICdb 1.0.8, following each source's data-access requirements.
2. Enter `revisions/2026-09-30`, install its requirements and configure local source locations and database authentication as described in its README.
3. Follow that README's extraction, fitting, evaluation and verification order. Patient-level outputs must remain in the protected local workspace.

Only code and aggregate research outputs are shared. Original data, patient/stay/landmark extracts, individual predictions, fitted models and credentials are excluded. The public historical configuration snapshot redacts machine-specific paths; its changes are explicitly recorded. No new full-database run was performed solely for this GitHub publication.

## Historical submitted analysis

The reviewed implementation is preserved at [commit 7844ca86089cca0a6b3196fc02007f338e728b01](https://github.com/shupengqin/mimic-eicu-sicdb-vasopressor-transportability/tree/7844ca86089cca0a6b3196fc02007f338e728b01). Root-level `src/`, `sql/`, the older `docs/`, and `results/prediction_time_2026-09-05/` remain historical materials; they do not generate the corrected manuscript results. The [submitted README](docs/submitted_README_2026-09-05.md) is retained for traceability.

For source-database access and citations, see [the data-access documentation](docs/prediction_time_data_code_availability.md). Cite the applicable code version alongside the manuscript and source database publications. No archive DOI or new software license is asserted by this update.
