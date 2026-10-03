# Release revision-2026-10-03

The previous release reported 1,116,862 changed SICdb predictor cells in its aligned audit and 1,116,906 in its stepwise timing/filter audit. Retained source inputs reproduce the discrepancy: the aligned audit converted to float32 and used rtol=1e-6, atol=1e-7; the stepwise audit compared float64 feature values with rtol=1e-7, atol=1e-8. Thirteen predictor counts differed by 44 cells in total.

## Changes

- Compare SICdb audit cells at the stepwise audit precision before casting model arrays. Keep the original conventions for MIMIC-IV and eICU.
- Regenerate results/sicdb_input_changes.csv from retained baseline and final inputs, with 58,756 aligned landmarks and 42 predictors. Every feature count agrees with results/sicdb_timing_and_filter_changes.csv, totaling 1,116,906.
- Add a standalone audit script, precision-check results and input-audit tests. Include the aggregate retained-input timing-convention check referenced by the response letter.
- Refresh manifests and provenance. Preserve historical execution snapshots as records of the original run.
- Remove superseded root-level analysis files and replace the previous release/download assets. Required legacy helpers remain inside the runnable package; original commits remain traceable through Git history.

## Verification

The standalone audit and the actual prepare_arrays.py sicdb path independently returned the same 42 counts. All fields in the regenerated SICdb model arrays exactly matched the saved arrays, including missing values. Existing alert-logic tests and new tolerance/missingness tests passed. All Python sources parsed successfully. Existing result files other than sicdb_input_changes.csv, executed SQL and historical execution snapshots were unchanged. The package manifest and archive are checked before publication and downloaded again for byte-level verification.

No model was refitted, no prediction or patient record was published, and the three databases were not re-extracted end to end for this release. This update does not establish prospectively blinded external validation or alter other disclosed study limitations.
