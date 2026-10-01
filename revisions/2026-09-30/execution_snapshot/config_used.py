# Public provenance snapshot: four machine-specific paths redacted.
# Use ../config.py for the portable runnable configuration.
from pathlib import Path

WORK = Path(__file__).resolve().parent
PRIVATE = WORK / 'private'
RESULTS = WORK / 'results'
SOURCE = Path('REDACTED_SOURCE_DIRECTORY')
OLD = SOURCE.parent / 'prediction_time_20260905'
BASE = Path('REDACTED_MANUSCRIPT_DIRECTORY')
ORIGINAL = BASE / 'MIMIC_eICU_SICdb_投稿检查包_2026-09-05_PREDICTION_TIME'
DELIVERY = BASE / '审稿修订_2026-09-30'
PSQL = Path('REDACTED_PSQL_EXECUTABLE')
SICDB = Path('REDACTED_SICDB_DIRECTORY')
SEED = 20260930
for p in (PRIVATE, RESULTS, WORK / 'sql', DELIVERY):
    p.mkdir(parents=True, exist_ok=True)

