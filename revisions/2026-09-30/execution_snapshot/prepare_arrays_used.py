"""Input audits on CURRENT primary landmarks, with protected compact arrays."""
import sys,json,gc,subprocess,io
import numpy as np
import pandas as pd
from config import *
sys.path.insert(0,str(SOURCE/'src'))
import run_validation as rv

def numeric_csv(path,cols):
    return pd.read_csv(path,usecols=cols,low_memory=False)

dataset=sys.argv[1]
path=PRIVATE/f'{dataset}_revised.csv.gz'
feature_cols=rv.FEATURE_COLS
key=['record_id','index_hour']
oldpath=OLD/f'{dataset}_prediction_time.csv'
old=pd.read_csv(oldpath,usecols=key+feature_cols[:2]+feature_cols[3:]+['label','lead_time_hours'],low_memory=False)
old=old[~(old.label.eq(1)&old.lead_time_hours.le(0))].set_index(key)
groups={}
allchanges=np.zeros(len(feature_cols),dtype=np.int64)
allcount=0
comparison=[]
for chunk in pd.read_csv(path,chunksize=100000,low_memory=False):
    idx=pd.MultiIndex.from_frame(chunk[key])
    baseline=old.reindex(idx).reset_index()
    assert len(baseline)==len(chunk) and baseline.label.notna().all()
    assert np.array_equal(baseline.label.to_numpy(),chunk.label.to_numpy())
    x=rv.feature_frame(chunk).to_numpy(np.float32)
    oldx=rv.feature_frame(baseline).to_numpy(np.float32)
    changed=~np.isclose(x,oldx,equal_nan=True,rtol=1e-6,atol=1e-7)
    allchanges+=changed.sum(axis=0);allcount+=len(chunk)
    if dataset=='mimic':
        split=chunk.time_group.map(lambda t:'development' if t in rv.MIMIC_DEVELOPMENT_GROUPS else ('selection' if t in rv.MIMIC_SELECTION_GROUPS else 'test'))
    else:split=pd.Series(dataset,index=chunk.index)
    for name,sub in chunk.groupby(split,sort=False):
        pos=chunk.index.get_indexer(sub.index)
        cols={'x':x[pos],'record_id':sub.record_id.to_numpy(np.int64),'patient_id':sub.patient_id.fillna(-1).to_numpy(np.int64),'y':sub.label.to_numpy(np.int8),
              'index_hour':sub.index_hour.to_numpy(np.int16),'lead_time_hours':sub.lead_time_hours.to_numpy(np.float32),'observed_horizon_hours':sub.observed_horizon_hours.to_numpy(np.float32),
              'exit_without_event':sub.exit_without_event.to_numpy(np.int8),'lab_age':sub[[v+'_age_hours' for v in rv.LAB_VARS]].to_numpy(np.float32)}
        if dataset=='mimic':
            cols['anchor_delta']=sub.anchor_year_delta.to_numpy(np.int16)
            cols['death_exit_proxy']=sub.death_on_exit_date.to_numpy(np.int8)
            cols['anchor_lower']=sub.time_group.str.slice(0,4).astype(int).to_numpy(np.int16)
        elif dataset=='eicu':
            cols['hospitalid']=sub.hospitalid.to_numpy(np.int16)
            cols['death_exit_proxy']=sub.unitdischargestatus.str.lower().eq('expired').to_numpy(np.int8)
            cols['hospice_proxy']=sub.unitdischargelocation.str.lower().str.contains('hospice',na=False).to_numpy(np.int8)
        else:
            cols['death_exit_proxy']=sub.death_before_or_at_case_exit.to_numpy(np.int8)
        groups.setdefault(name,[]).append(cols)
del old;gc.collect()
for name,parts in groups.items():
    arrays={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    np.savez_compressed(PRIVATE/f'{name}_inputs.npz',**arrays)
    del arrays
pd.DataFrame({'dataset':dataset,'feature':feature_cols,'current_landmarks':allcount,'changed_from_submitted_input':allchanges,'changed_percent':100*allchanges/allcount}).to_csv(RESULTS/f'{dataset}_input_changes.csv',index=False)
print(dataset,'input alignment verified;',allcount,'rows;',int(allchanges.sum()),'changed cells',flush=True)

