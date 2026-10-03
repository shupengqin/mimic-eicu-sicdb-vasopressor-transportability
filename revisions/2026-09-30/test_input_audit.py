"""Check rounding-boundary and missingness behavior without patient data."""
import numpy as np
from input_audit import changed_cells

old = np.array([[100., np.nan, np.nan, 20.]])
new = np.array([[100.00005, np.nan, 1., np.nan]])
assert changed_cells(new, old, 'sicdb').tolist() == [[True, False, True, True]]
for dataset in ('mimic', 'eicu'):
    assert changed_cells(new, old, dataset).tolist() == [[False, False, True, True]]

# A float32 rounding boundary must not inflate the float64 SICdb change count.
old = np.array([[100.0000188]])
new = np.array([[100.0000269]])
assert not changed_cells(new, old, 'sicdb').item()
assert not np.array_equal(old.astype(np.float32), new.astype(np.float32))
try:
    changed_cells(np.ones((2, 1)), np.ones((1, 1)), 'sicdb')
except ValueError:
    pass
else:
    raise AssertionError('Mismatched alignment was accepted')
print('PASS: audit tolerance, pre-cast comparison, and missing-value transitions.')
