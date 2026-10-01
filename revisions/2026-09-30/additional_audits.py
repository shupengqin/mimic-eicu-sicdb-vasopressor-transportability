import os,sys,json,subprocess,io,gc
import numpy as np
import pandas as pd
from config import *
from analyse import load,basic
sys.path.insert(0,str(SOURCE/'src'))
import run_validation as rv
import clustered_inference as ci
from prediction_time_hospital import reml

def first_agent(dataset):
    target=PRIVATE/f'{dataset}_first_agents.csv'
    if dataset in ('mimic','eicu') and not target.exists():
        cmd=[str(PSQL),'-h',os.environ.get('PGHOST','127.0.0.1'),'-p',os.environ.get('PGPORT','5442'),'-U',os.environ.get('PGUSER','postgres'),'-w','-X','-q','-d','mimiciv31' if dataset=='mimic' else 'eicu','-f',str(SOURCE/'sql'/f'{dataset}_first_agent.sql')]
        with target.open('wb') as f:
            subprocess.run(cmd,stdout=f,check=True)
    agent=pd.read_csv(target)
    if dataset=='sicdb':
        names={1502:'epinephrine',1550:'vasopressin',1562:'norepinephrine',1593:'phenylephrine',1618:'dopamine'}
        # Verify normalized names against the released reference dictionary.
        agent['first_agents']=agent.drug_ids.astype(str).map(lambda s:'+'.join(sorted(names[int(float(x))] for x in s.split('+'))))
        agent['norepinephrine_at_first']=agent.drug_ids.astype(str).map(lambda s:int('1562' in s.split('+')))
        agent.to_csv(target,index=False)
    return agent.set_index('record_id')

def audit(name):
    a=load(name);x=a['x'];y=a['y'];r=a['record_id']
    predpath=PRIVATE/f'{name}_predictions.npz'
    pred=np.load(predpath)['HGB'] if predpath.exists() else None
    first=np.unique(r,return_index=True)[1];event=pd.Series(y).groupby(r).max()
    df=pd.DataFrame(x,columns=rv.FEATURE_COLS)
    q=[]
    for f in rv.FEATURE_COLS:
        v=df[f].dropna()
        q.append({'cohort':name,'feature':f,'landmarks':len(df),'missing':int(df[f].isna().sum()),'missing_percent':df[f].isna().mean()*100,'minimum':v.min(),'q01':v.quantile(.01),'q25':v.quantile(.25),'median':v.median(),'q75':v.quantile(.75),'q99':v.quantile(.99),'maximum':v.max()})
    pd.DataFrame(q).to_csv(RESULTS/f'{name}_predictor_qc.csv',index=False)
    lab=[]
    for j,v in enumerate(rv.LAB_VARS):
        age=a['lab_age'][:,j];observed=age[np.isfinite(age)]
        assert len(observed)==np.isfinite(df[v+'_last']).sum()
        assert np.all((observed>=0)&(observed<=24))
        lab.append({'cohort':name,'laboratory':v,'n_available':len(observed),'missing_percent':100*(1-len(observed)/len(y)),'age_q1':np.quantile(observed,.25),'age_median':np.median(observed),'age_q3':np.quantile(observed,.75),'age_p95':np.quantile(observed,.95),'age_max':np.max(observed),'over6_percent_available':100*np.mean(observed>6),'over12_percent_available':100*np.mean(observed>12)})
    pd.DataFrame(lab).to_csv(RESULTS/f'{name}_lab_freshness.csv',index=False)
    cohort={'cohort':name,'stays':len(first),'landmarks':len(y),'event_stays':int(event.sum()),'event_stay_percent':event.mean()*100,'positive_landmarks':int(y.sum()),'positive_percent':100*y.mean(),'exit_truncated_landmarks':int(a['exit_without_event'].sum()),'exit_percent':a['exit_without_event'].mean()*100,'age_q1':float(np.quantile(x[first,0],.25)),'age_median':float(np.median(x[first,0])),'age_q3':float(np.quantile(x[first,0],.75)),'male_percent':float(np.nanmean(x[first,1])*100)}
    pd.DataFrame([cohort]).to_csv(RESULTS/f'{name}_cohort.csv',index=False)
    exitrows=[]
    for label,mask in [('all',np.ones(len(y),bool)),('exit_truncated',a['exit_without_event']==1)]:
        ids,idx=np.unique(r[mask],return_index=True)
        death=a['death_exit_proxy'][mask][idx]
        exitrows.append({'cohort':name,'population':label,'landmarks':int(mask.sum()),'stays':len(ids),'death_proxy_stays':int(death.sum()),'death_proxy_percent':float(death.mean()*100),'hospice_stays':int(a['hospice_proxy'][mask][idx].sum()) if 'hospice_proxy' in a else np.nan,'death_definition':'same date as ICU exit; date-level proxy' if name in ('development','selection','test') else ('recorded expired at unit discharge' if name=='eicu' else 'death offset no later than case closure')})
    pd.DataFrame(exitrows).to_csv(RESULTS/f'{name}_exit_composition.csv',index=False)
    if 'anchor_delta' in a:
        lo=a['anchor_lower'][first]+a['anchor_delta'][first];hi=lo+2
        bounds={'development':(2008,2016),'selection':(2017,2019),'test':(2020,2022)}[name]
        pd.DataFrame([{'cohort':name,'stays':len(first),'anchor_delta_nonzero_stays':int((a['anchor_delta'][first]!=0).sum()),'possible_outside_claimed_period':int(((lo<bounds[0])|(hi>bounds[1])).sum()),'definitely_outside_claimed_period':int(((hi<bounds[0])|(lo>bounds[1])).sum()),'min_possible_admission_year':int(lo.min()),'max_possible_admission_year':int(hi.max()),'claimed_lower':bounds[0],'claimed_upper':bounds[1]}]).to_csv(RESULTS/f'{name}_admission_period_audit.csv',index=False)
    if pred is None:return
    sub=[]
    for label,mask in [('age18_44',(x[:,0]>=18)&(x[:,0]<45)),('age45_64',(x[:,0]>=45)&(x[:,0]<65)),('age65_79',(x[:,0]>=65)&(x[:,0]<80)),('age80plus',x[:,0]>=80),('recorded_female',x[:,1]==0),('recorded_male',x[:,1]==1)]:
        sub.append(dict(cohort=name,subgroup=label,n_landmarks=int(mask.sum()),n_stays=len(np.unique(r[mask])),event_stays=int(pd.Series(y[mask]).groupby(r[mask]).max().sum()),**basic(y[mask],pred[mask])))
    pd.DataFrame(sub).to_csv(RESULTS/f'{name}_subgroups.csv',index=False)
    source='mimic' if name=='test' else name
    agent=first_agent(source)
    event_ids=event[event==1].index
    selected=agent.reindex(event_ids)
    assert selected.first_agents.notna().all()
    comp=selected.first_agents.value_counts().rename_axis('first_agents').reset_index(name='event_stays')
    comp['cohort']=name;comp['percent']=comp.event_stays/len(event_ids)*100
    comp.to_csv(RESULTS/f'{name}_agent_composition.csv',index=False)
    endpoint=[]
    for label,keep in [('strict_norepinephrine',agent.first_agents.eq('norepinephrine')),('norepinephrine_at_first',agent.norepinephrine_at_first.eq(1))]:
        k=pd.Series(r).map(keep).fillna(False).to_numpy(bool);yy=y*k
        endpoint.append(dict(cohort=name,analysis=label,n_landmarks=len(y),n_positive_landmarks=int(yy.sum()),**basic(yy,pred)))
    pd.DataFrame(endpoint).to_csv(RESULTS/f'{name}_endpoint_sensitivity.csv',index=False)
    counts=pd.Series(r).value_counts();weights=1/pd.Series(r).map(counts).to_numpy()
    from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
    equal={'cohort':name,'analysis':'equal_total_stay_evaluation_weight','auroc':roc_auc_score(y,pred,sample_weight=weights),'auprc':average_precision_score(y,pred,sample_weight=weights),'brier':brier_score_loss(y,pred,sample_weight=weights)}
    pd.DataFrame([equal]).to_csv(RESULTS/f'{name}_stay_equal.csv',index=False)

def hospitals():
    a=load('eicu');p=np.load(PRIVATE/'eicu_predictions.npz')['HGB'];y=a['y'];hospital=a['hospitalid']
    unique,codes=np.unique(hospital,return_inverse=True);e=ci.RankingEvaluator(y.astype(float),p,codes)
    rng=np.random.default_rng(SEED);boot=[]
    for i in range(300):boot.append(e.evaluate(np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique))))
    point=e.evaluate(np.ones(len(unique)));row={'hospitals':len(unique)}
    for j,m in enumerate(['auroc','auprc','brier']):row[m]=point[j];row[m+'_low'],row[m+'_high']=np.quantile(np.asarray(boot)[:,j],[.025,.975])
    pd.DataFrame([row]).to_csv(RESULTS/'eicu_hospital_bootstrap.csv',index=False)
    rows=[]
    for h in unique:
        m=hospital==h;ev=pd.Series(y[m]).groupby(a['record_id'][m]).max()
        if m.sum()<1000 or ev.sum()<20 or (1-ev).sum()<20:continue
        _,c=np.unique(a['patient_id'][m],return_inverse=True)
        cal=ci.cluster_robust_calibration(y[m],p[m],c)
        rows.append(dict(hospital_id=int(h),landmarks=int(m.sum()),stays=len(ev),event_stays=int(ev.sum()),**cal))
    df=pd.DataFrame(rows);df.to_csv(RESULTS/'eicu_hospital_calibration.csv',index=False)
    out=[]
    for m in ['calibration_in_the_large','calibration_slope']:
        se=(df[m+'_ci_high']-df[m+'_ci_low'])/3.92
        if m=='calibration_slope':v=df[m]>0;res=reml(np.log(df.loc[v,m]),(se[v]/df.loc[v,m])**2)
        else:res=reml(df[m],se**2)
        if m=='calibration_slope':
            for k in ['pooled','ci_low','ci_high','prediction_interval_low','prediction_interval_high']:res[k]=np.exp(res[k])
        out.append(dict(metric=m,**res))
    pd.DataFrame(out).to_csv(RESULTS/'eicu_hospital_meta.csv',index=False)

if __name__=='__main__':
    if sys.argv[1]=='hospitals':hospitals()
    else:audit(sys.argv[1])
