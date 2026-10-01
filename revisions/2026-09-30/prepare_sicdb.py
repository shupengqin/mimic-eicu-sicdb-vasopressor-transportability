"""SICdb hourly aggregates become available at the END of their source hour."""
import sys, json, time
import numpy as np
import pandas as pd
from config import *
sys.path.insert(0,str(SOURCE/'src'))
import run_validation as rv
rv.SICDB=SICDB

baseline=OLD/'sicdb_prediction_time.csv'
if baseline.exists():
    grid=pd.read_csv(baseline,low_memory=False)
else:
    import prediction_time_analysis as legacy_prediction
    legacy_prediction.SICDB=SICDB
    # Reconstruct submitted-style inputs only to establish the source risk set
    # and before/after timing audit. No legacy model evaluation is invoked.
    grid=legacy_prediction.build_sicdb_prediction()
zero=(grid.label.eq(1)&grid.lead_time_hours.le(0))
pd.DataFrame([{'dataset':'sicdb','removed_same_timestamp_landmarks':int(zero.sum())}]).to_csv(RESULTS/'sicdb_boundary_audit.csv',index=False)
grid=grid.loc[~zero].reset_index(drop=True)
cases=pd.read_csv(SICDB/'cases.csv.gz',low_memory=False).set_index('CaseID')
ids=set(grid.record_id.astype(int))
offset=cases.ICUOffset
qc=[]
parts=[]
start=time.time()
for kind,filename,idcol,valcol,mapping in [
    ('vitals','data_float_h.csv.gz','DataID','Val',rv.VITAL_DATA_IDS),
    ('labs','laboratory.csv.gz','LaboratoryID','LaboratoryValue',rv.LAB_ID_TO_VAR)]:
    count=0
    kept=[]
    rejected={}
    totals={}
    mapping={k:v for k,v in mapping.items() if v in rv.PHYSIOLOGIC_RANGES}
    cached=PRIVATE/f'sicdb_{kind}_observations.pkl'
    if cached.exists():
        obs=pd.read_pickle(cached)
        obs=obs[obs.variable.isin(rv.PHYSIOLOGIC_RANGES)]
        for v,g in obs.groupby('variable'):
            qc.append({'dataset':'sicdb','variable':v,'nonmissing_observations':int(g.value.notna().sum()),'rejected_observations':int((g.value.notna()&~g.value.between(*rv.PHYSIOLOGIC_RANGES[v])).sum()),'source_level':'published hourly aggregates' if kind=='vitals' else 'laboratory records'})
        del obs
        continue
    for ch in pd.read_csv(SICDB/filename,usecols=['CaseID',idcol,'Offset',valcol],chunksize=300000,low_memory=False):
        count+=len(ch)
        ch=ch[ch.CaseID.isin(ids)&ch[idcol].isin(mapping)].copy()
        if ch.empty:continue
        ch['rel_hour']=(ch.Offset-ch.CaseID.map(offset))/3600.
        ch=ch[ch.rel_hour.gt(0)&ch.rel_hour.le(24)].copy()
        ch['variable']=ch[idcol].map(mapping)
        ch['value']=pd.to_numeric(ch[valcol],errors='coerce')
        ch['record_id']=ch.CaseID.astype('int64')
        for v,g in ch.groupby('variable'):
            valid=g.value.between(*rv.PHYSIOLOGIC_RANGES[v])
            totals[v]=totals.get(v,0)+int(g.value.notna().sum())
            rejected[v]=rejected.get(v,0)+int((g.value.notna()&~valid).sum())
        ch=ch[['record_id','rel_hour','variable','value']]
        kept.append(ch)
        if count%3000000<300000:print(kind,count,'source rows',round(time.time()-start),flush=True)
    obs=pd.concat(kept,ignore_index=True)
    for v,n in totals.items():qc.append({'dataset':'sicdb','variable':v,'nonmissing_observations':n,'rejected_observations':rejected[v],'source_level':'published hourly aggregates' if kind=='vitals' else 'laboratory records'})
    obs.to_pickle(PRIVATE/f'sicdb_{kind}_observations.pkl')
    print(kind,'retained',len(obs),flush=True)
    del kept,obs
pd.DataFrame(qc).to_csv(RESULTS/'sicdb_source_qc.csv',index=False)

vitals=pd.read_pickle(PRIVATE/'sicdb_vitals_observations.pkl')
labs=pd.read_pickle(PRIVATE/'sicdb_labs_observations.pkl')
def clean(obs):
    obs=obs.copy()
    for v,(lo,hi) in rv.PHYSIOLOGIC_RANGES.items():
        m=obs.variable.eq(v)&~obs.value.between(lo,hi)
        obs.loc[m,'value']=np.nan
    return obs.dropna(subset=['value'])

# Source offset is the beginning of the aggregate hour. Even a partial-hour
# aggregate is conservatively withheld until offset + 3600 seconds.
vitals.rel_hour=vitals.rel_hour+1
timing=rv.add_sicdb_features(grid,pd.concat([
    vitals.groupby(['record_id','rel_hour','variable'],as_index=False).value.mean(),
    labs.groupby(['record_id','rel_hour','variable'],as_index=False).value.mean()]))
timing.to_csv(PRIVATE/'sicdb_timing_only.csv.gz',index=False,compression='gzip')
vitals=clean(vitals).groupby(['record_id','rel_hour','variable'],as_index=False).value.mean()
labs=clean(labs).groupby(['record_id','rel_hour','variable'],as_index=False).value.mean()
revised=rv.add_sicdb_features(grid,pd.concat([vitals,labs],ignore_index=True))
for v in rv.LAB_VARS:
    revised[v+'_age_hours']=np.nan
    subset=labs[labs.variable.eq(v)]
    grouped={k:g for k,g in subset.groupby('record_id')}
    for rec,g in revised.groupby('record_id',sort=False):
        if rec not in grouped:continue
        obs=grouped[rec].sort_values('rel_hour')
        ts=obs.rel_hour.to_numpy()
        h=g.index_hour.to_numpy()
        pos=np.searchsorted(ts,h,side='right')-1
        ok=pos>=0
        revised.loc[g.index[ok],v+'_age_hours']=h[ok]-ts[pos[ok]]
refs=rv.decode_reference_map()
for col in ['DischargeState','HospitalDischargeType','DischargeUnit']:
    revised[col]=revised.record_id.map(cases[col]).map(refs)
revised['death_before_or_at_case_exit']=(revised.record_id.map(cases.OffsetOfDeath).le(revised.record_id.map(cases.TimeOfStay))).astype(int)
revised.to_csv(PRIVATE/'sicdb_revised.csv.gz',index=False,compression='gzip')
changes=[]
for f in rv.FEATURE_COLS:
    a=rv.feature_frame(grid)[f].to_numpy(); b=rv.feature_frame(timing)[f].to_numpy(); c=rv.feature_frame(revised)[f].to_numpy()
    eq=lambda x,y:np.isclose(x,y,equal_nan=True,rtol=1e-7,atol=1e-8)
    changes.append({'feature':f,'n_landmarks':len(grid),'timing_changed':int((~eq(a,b)).sum()),'filter_order_changed_after_timing':int((~eq(b,c)).sum()),'total_changed':int((~eq(a,c)).sum())})
pd.DataFrame(changes).to_csv(RESULTS/'sicdb_timing_and_filter_changes.csv',index=False)

# First-agent composition at the exact first continuous-use timestamp.
med=[]
for ch in pd.read_csv(SICDB/'medication.csv.gz',usecols=['CaseID','DrugID','Offset','IsSingleDose'],chunksize=300000):
    ch=ch[ch.CaseID.isin(ids)&ch.DrugID.isin(rv.SICDB_PRESSOR_IDS)&ch.IsSingleDose.eq(0)]
    if len(ch):med.append(ch)
med=pd.concat(med)
med=med[med.Offset.eq(med.groupby('CaseID').Offset.transform('min'))]
rows=[]
for rec,g in med.groupby('CaseID'):
    names={1502:'epinephrine',1550:'vasopressin',1562:'norepinephrine',1593:'phenylephrine',1618:'dopamine'}
    agents=sorted(set(names[x] for x in g.DrugID))
    rows.append({'record_id':rec,'first_agents':'+'.join(agents),'n_first_agents':len(agents),'norepinephrine_at_first':int(1562 in set(g.DrugID)),'drug_ids':'+'.join(map(str,sorted(set(g.DrugID))))})
pd.DataFrame(rows).to_csv(PRIVATE/'sicdb_first_agents.csv',index=False)
print('Completed SICdb corrected extraction',len(revised),'landmarks',round(time.time()-start),'seconds',flush=True)
