# Vasopressor prediction across MIMIC-IV, eICU-CRD and SICdb

## Download and reproduce

Use [release revision-2026-10-03](https://github.com/shupengqin/mimic-eicu-sicdb-vasopressor-transportability/releases/tag/revision-2026-10-03) and the [analysis package](revisions/2026-09-30/README.md). The directory date identifies the 30 September analytical run; the 3 October release synchronizes its input-audit code and aggregate results with Supplementary Table 6. It is the only supported package in the default branch.

Obtain authorized access to MIMIC-IV 3.1, eICU-CRD 2.0 and SICdb 1.0.8, then follow the package README. The release contains code, executed SQL, aggregate results, tests and SHA-256 manifests. Patient-level records, individual predictions, fitted models and credentials are not distributed.

## Audit correction

The aligned SICdb audit now uses the same float64 comparison and tolerance as its timing/filter-order audit. Both report 1,116,906 changed cells across 58,756 landmarks and 42 predictors. The earlier aligned audit used float32 and a looser tolerance, giving 1,116,862. The 44-cell difference affected 13 predictors and was reproduced from retained inputs. It was an audit-count discrepancy, not a model-data or prediction change. MIMIC-IV and eICU audit conventions and results are unchanged.

The updated SICdb array preparation was rerun on retained extracts and all saved model-array fields matched the existing arrays. Models were not refitted. [Release notes](revisions/2026-09-30/RELEASE_NOTES_2026-10-03.md) record the verification and its limits.

## Version history

Superseded root-level scripts, SQL, aggregate results and documents have been removed from the default branch to prevent accidental use. Required helper modules remain inside the analysis package. The previous release and its download assets are replaced by revision-2026-10-03.

The originally reviewed implementation remains traceable at [commit 7844ca86089cca0a6b3196fc02007f338e728b01](https://github.com/shupengqin/mimic-eicu-sicdb-vasopressor-transportability/tree/7844ca86089cca0a6b3196fc02007f338e728b01), and the preceding publication at [commit 088f8246caa341f9ccc9114ad22328ce5f22d4a2](https://github.com/shupengqin/mimic-eicu-sicdb-vasopressor-transportability/tree/088f8246caa341f9ccc9114ad22328ce5f22d4a2). Git history is preserved. No archive DOI or new software license is asserted.
