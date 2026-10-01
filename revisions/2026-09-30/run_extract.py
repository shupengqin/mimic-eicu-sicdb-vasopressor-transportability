import os, sys, subprocess, gzip, time, hashlib, json
from config import *
dataset=sys.argv[1]
query=WORK/'sql'/f'{dataset}_revised.sql'
target=PRIVATE/f'{dataset}_revised.csv.gz'
start=time.time()
cmd=[str(PSQL),'-h',os.environ.get('PGHOST','127.0.0.1'),'-p',os.environ.get('PGPORT','5442'),'-U',os.environ.get('PGUSER','postgres'),'-d',{'mimic':'mimiciv31','eicu':'eicu'}[dataset],'-w','-X','-q','-v','ON_ERROR_STOP=1','-f',str(query)]
with (PRIVATE/f'{dataset}_extract.log').open('wb') as err, gzip.open(target,'wb',compresslevel=3) as out:
    proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=err,cwd=WORK)
    for block in iter(lambda:proc.stdout.read(1024*1024),b''):
        out.write(block)
    code=proc.wait()
if code: raise RuntimeError(f'{dataset} extraction failed; see protected log')
info={'dataset':dataset,'seconds':time.time()-start,'query_sha256':hashlib.sha256(query.read_bytes()).hexdigest(),'exit_code':code,'compressed_bytes':target.stat().st_size}
(RESULTS/f'{dataset}_execution.json').write_text(json.dumps(info,indent=2))
print(info,flush=True)
