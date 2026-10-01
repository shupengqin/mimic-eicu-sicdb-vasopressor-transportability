from pathlib import Path
import os
WORK=Path(__file__).resolve().parent
PRIVATE=Path(os.environ.get('REVISION_PRIVATE',WORK/'private'))
RESULTS=Path(os.environ.get('REVISION_RESULTS',WORK/'results'))
SOURCE=Path(os.environ.get('REVISION_LEGACY',WORK/'legacy'))
OLD=Path(os.environ.get('REVISION_SUBMITTED_INPUTS',WORK/'submitted_inputs'))
DELIVERY=Path(os.environ.get('REVISION_EXPORT',WORK/'exports'))
ORIGINAL=Path(os.environ.get('REVISION_ORIGINAL',WORK/'original'))
PSQL=Path(os.environ.get('PSQL_PATH','psql'))
SICDB=Path(os.environ.get('SICDB_PATH',WORK/'data'/'sicdb'))
SEED=20260930
for p in (PRIVATE,RESULTS,DELIVERY):p.mkdir(parents=True,exist_ok=True)
