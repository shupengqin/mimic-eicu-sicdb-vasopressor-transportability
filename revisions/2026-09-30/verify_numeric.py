import json,itertools
import numpy as np
import pandas as pd
from config import *
from analyse import load

report={}
patients={}
for n in ['development','selection','test','eicu','sicdb']:
    a=load(n);r=a['record_id'];h=a['index_hour'];y=a['y'];age=a['lab_age']
    assert len(np.unique(r*100+h))==len(r)
    assert a['x'].shape==(len(r),42)
    assert np.all((h>=6)&(h<=24))
    assert np.all((a['observed_horizon_hours']>0)&(a['observed_horizon_hours']<=6))
    assert np.all((a['lead_time_hours'][y==1]>0)&(a['lead_time_hours'][y==1]<6))
    assert np.all((age[np.isfinite(age)]>=0)&(age[np.isfinite(age)]<=24))
    assert np.all(y[a['exit_without_event']==1]==0)
    q=pd.read_csv(RESULTS/f'{n}_predictor_qc.csv');assert len(q)==42 and q.landmarks.eq(len(r)).all()
    np.testing.assert_array_equal(q.missing.to_numpy(),np.isnan(a['x']).sum(axis=0))
    if n in ['development','selection','test']:patients[n]=set(a['patient_id'].tolist())
    report[n]={'landmarks':len(r),'stays':len(np.unique(r)),'positive_rows':int(y.sum()),'unique_landmarks':True,'42_predictors':True,'positive_lead_strictly_under_6h':True,'lab_age_bounded_24h':True,'QC_matches_full_inputs':True}
    if n in ['test','eicu','sicdb']:
        p=pd.read_csv(RESULTS/f'{n}_policy_repeated.csv');b=pd.read_csv(RESULTS/f'{n}_brier_repeated.csv');sp=pd.read_csv(RESULTS/f'{n}_split_audit.csv')
        assert len(p)==200 and len(b)==300 and len(sp)==100
        assert p.lead_max.lt(6).all() and p.lead_q1.ge(0).all()
        assert p.event_sensitivity.between(0,1).all()
        np.testing.assert_allclose(p.event_sensitivity,p.n_detected_stays/p.n_event_stays)
        np.testing.assert_allclose(b.brier_skill,1-b.brier/b.brier_null)
        assert b.groupby('repeat').brier_null.nunique().eq(1).all()
        assert sp.group_assignment_sha256.nunique()==100
        report[n].update(repeated_splits=100,paired_bootstraps=300,all_correct_alert_leads_under_6h=True)
    del a
for a,b in itertools.combinations(patients,2):assert not patients[a].intersection(patients[b])
report['mimic_patient_disjoint']=True
# The range-check/timing sequence is tested by inspecting actual extraction guards.
sic=(WORK/'prepare_sicdb.py').read_text(encoding='utf-8')
assert 'vitals.rel_hour=vitals.rel_hour+1' in sic
for name in ['mimic','eicu']:
    execution=json.loads((RESULTS/f'{name}_execution.json').read_text())
    assert execution['exit_code']==0
    import hashlib
    assert execution['query_sha256']==hashlib.sha256((WORK/'sql'/f'{name}_revised.sql').read_bytes()).hexdigest()
report['SQL_hashes_match_executed_queries']=True
(RESULTS/'verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
