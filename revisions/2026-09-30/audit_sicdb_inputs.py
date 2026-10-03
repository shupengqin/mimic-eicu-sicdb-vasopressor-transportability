"""Recompute the aligned SICdb audit from restricted retained input files.

Only aggregate counts are written to RESULTS. No model is refitted and no
patient-level arrays or predictions are modified.
"""
import json
import sys
import numpy as np
import pandas as pd
from config import OLD, PRIVATE, RESULTS, SOURCE
from input_audit import changed_cells

sys.path.insert(0, str(SOURCE / 'src'))
import run_validation as rv


def main():
    key = ['record_id', 'index_hour']
    baseline = pd.read_csv(OLD / 'sicdb_prediction_time.csv', low_memory=False)
    baseline = baseline.loc[~(baseline.label.eq(1) & baseline.lead_time_hours.le(0))]
    current = pd.read_csv(PRIVATE / 'sicdb_revised.csv.gz', low_memory=False)
    assert not baseline.duplicated(key).any() and not current.duplicated(key).any()
    baseline = baseline.set_index(key).reindex(pd.MultiIndex.from_frame(current[key])).reset_index()
    assert baseline.label.notna().all()
    np.testing.assert_array_equal(baseline.label.to_numpy(), current.label.to_numpy())
    old_x = rv.feature_frame(baseline).to_numpy(np.float64)
    new_x = rv.feature_frame(current).to_numpy(np.float64)
    changes = changed_cells(new_x, old_x, 'sicdb').sum(axis=0)
    old_changes = (~np.isclose(new_x.astype(np.float32), old_x.astype(np.float32),
                               equal_nan=True, rtol=1e-6, atol=1e-7)).sum(axis=0)
    stepwise = pd.read_csv(RESULTS / 'sicdb_timing_and_filter_changes.csv').set_index('feature').loc[rv.FEATURE_COLS]
    np.testing.assert_array_equal(stepwise.n_landmarks.to_numpy(), len(current))
    np.testing.assert_array_equal(changes, stepwise.total_changed.to_numpy())
    result = pd.DataFrame({'dataset': 'sicdb', 'feature': rv.FEATURE_COLS,
                           'current_landmarks': len(current),
                           'changed_from_submitted_input': changes,
                           'changed_percent': 100 * changes / len(current)})
    result.to_csv(RESULTS / 'sicdb_input_changes.csv', index=False)
    check = {'aligned_landmarks': len(current), 'predictors': len(rv.FEATURE_COLS),
             'previous_float32_rtol_1e_6_atol_1e_7_total': int(old_changes.sum()),
             'aligned_float64_rtol_1e_7_atol_1e_8_total': int(changes.sum()),
             'difference_cells': int(changes.sum() - old_changes.sum()),
             'features_with_different_counts': int((changes != old_changes).sum()),
             'all_feature_counts_match_stepwise_csv': True,
             'model_arrays_or_predictions_modified': False}
    (RESULTS / 'sicdb_audit_precision_check.json').write_text(json.dumps(check, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(check, indent=2))


if __name__ == '__main__':
    main()
