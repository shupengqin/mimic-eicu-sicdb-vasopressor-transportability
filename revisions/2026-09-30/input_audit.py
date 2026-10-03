"""Compare predictor cells without changing the arrays used by the models."""
import numpy as np


def changed_cells(current, baseline, dataset):
    """Use the stepwise SICdb audit convention; retain other cohorts' convention.

    SICdb comparisons occur before float32 model-array conversion. The argument
    order matches prepare_sicdb.py because isclose scales its relative tolerance
    by the second argument. NaN-to-NaN is unchanged; NaN-to-value is changed.
    """
    if dataset == 'sicdb':
        current = np.asarray(current, dtype=np.float64)
        baseline = np.asarray(baseline, dtype=np.float64)
        rtol, atol = 1e-7, 1e-8
        left, right = baseline, current
    elif dataset in ('mimic', 'eicu'):
        current = np.asarray(current, dtype=np.float32)
        baseline = np.asarray(baseline, dtype=np.float32)
        rtol, atol = 1e-6, 1e-7
        left, right = current, baseline
    else:
        raise ValueError(f'Unknown dataset: {dataset}')
    if current.shape != baseline.shape:
        raise ValueError('Predictor arrays must have the same shape and alignment')
    return ~np.isclose(left, right, equal_nan=True, rtol=rtol, atol=atol)
