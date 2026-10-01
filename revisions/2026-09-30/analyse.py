"""Revision analysis with fixed hyperparameters and independently checked metrics.

Patient/stay-level arrays remain in private/. Only aggregate results are exported.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import sys,gc,json,time,hashlib,platform
import numpy as np
import pandas as pd
import scipy,sklearn,joblib
from scipy.special import expit,logit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from config import *
sys.path.insert(0,str(SOURCE/'src'))
import run_validation as rv
import model_benchmark as mb
import clustered_inference as ci

def load(name):
    with np.load(PRIVATE/f'{name}_inputs.npz') as f:return {k:f[k] for k in f.files}

def fit_lr(x,y):
    return make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),LogisticRegression(C=.5,solver='lbfgs',max_iter=300,random_state=SEED)).fit(x,y)

def basic(y,p):
    p=np.asarray(p,dtype=np.float64)
    return dict(auroc=roc_auc_score(y,p),auprc=average_precision_score(y,p),brier=brier_score_loss(y,p))

def model_stage():
    dev=load('development');sel=load('selection')
    candidates={'LR':fit_lr(dev['x'],dev['y']),'HGB':mb.build_model().fit(dev['x'],dev['y'])}
    rows=[dict(model=n,**basic(sel['y'],m.predict_proba(sel['x'])[:,1])) for n,m in candidates.items()]
    pd.DataFrame(rows).to_csv(RESULTS/'model_selection.csv',index=False)
    assert sorted(rows,key=lambda r:(r['auroc'],r['auprc']),reverse=True)[0]['model']=='HGB','Model choice changed; review manuscript before proceeding'
    x=np.concatenate([dev['x'],sel['x']]);y=np.r_[dev['y'],sel['y']]
    lead=np.r_[dev['lead_time_hours'],sel['lead_time_hours']]
    models={'LR':fit_lr(x,y),'HGB':mb.build_model().fit(x,y)}
    keep=(y==0)|(lead>1)
    models['gap']=mb.build_model().fit(x[keep],y[keep])
    models['availability']=make_pipeline(StandardScaler(),LogisticRegression(C=.5,max_iter=300,random_state=SEED)).fit(np.isnan(x).astype(np.float32),y)
    joblib.dump(models,PRIVATE/'revised_models.joblib',compress=3)
    pd.DataFrame([{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'scikit_learn':sklearn.__version__,'joblib':joblib.__version__,'revision_seed':SEED,'HGB_seed':20260823}]).to_csv(RESULTS/'software_versions.csv',index=False)
    print('Models fitted on revised MIMIC inputs; selection',rows,flush=True)

def policy(a,mask,p,threshold):
    r=a['record_id'][mask];h=a['index_hour'][mask];y=a['y'][mask];lead=a['lead_time_hours'][mask]
    raw=p[mask]>=threshold
    # Inputs are checked for record/hour ordering; suppression uses elapsed hours.
    order=np.lexsort((h,r));emitted=np.zeros(len(r),bool);previous=None;last=-1e9
    for i in order:
        if r[i]!=previous:previous=r[i];last=-1e9
        if raw[i] and h[i]-last>=6:emitted[i]=True;last=h[i]
    tp=emitted&(y==1);fp=emitted&(y==0)
    unique,codes=np.unique(r,return_inverse=True)
    event=np.zeros(len(unique),bool);detected=np.zeros(len(unique),bool);alerts=np.zeros(len(unique),bool);false=np.zeros(len(unique),bool)
    np.logical_or.at(event,codes,y==1);np.logical_or.at(detected,codes,tp);np.logical_or.at(alerts,codes,emitted);np.logical_or.at(false,codes,fp)
    # Exactly one correct alert is possible under the open six-hour outcome window.
    assert tp.sum()==detected.sum()
    assert np.all((lead[tp]>0)&(lead[tp]<6+1e-5))
    counts=np.bincount(codes,weights=emitted,minlength=len(unique))
    falsecounts=np.bincount(codes,weights=fp,minlength=len(unique))
    valid_lead=lead[tp]
    return dict(raw_sensitivity=float((raw&(y==1)).sum()/y.sum()),raw_alerts_100=float(raw.mean()*100),
        emitted_alerts_100=float(emitted.mean()*100),emitted_landmark_sensitivity=float(tp.sum()/y.sum()),
        event_sensitivity=float(detected.sum()/event.sum()),ppv=float(tp.sum()/emitted.sum()) if emitted.any() else np.nan,
        false_alerts_100=float(fp.mean()*100),stays_alerted_percent=float(alerts.mean()*100),
        event_free_stays_false_alert_percent=float(false[~event].mean()*100),
        false_episodes_per_alerted_stay=float(falsecounts[alerts].mean()),
        episodes_per_alerted_stay_median=float(np.median(counts[alerts])),
        lead_median=float(np.median(valid_lead)) if len(valid_lead) else np.nan,
        lead_q1=float(np.quantile(valid_lead,.25)) if len(valid_lead) else np.nan,
        lead_q3=float(np.quantile(valid_lead,.75)) if len(valid_lead) else np.nan,
        lead_max=float(np.max(valid_lead)) if len(valid_lead) else np.nan,
        n_event_stays=int(event.sum()),n_detected_stays=int(detected.sum()))

def cal_policy(name,a,p):
    rng=np.random.default_rng(SEED)
    group=a['patient_id'].copy();group[group<0]=a['record_id'][group<0]
    unique,codes=np.unique(group,return_inverse=True)
    event=np.zeros(len(unique),np.int8);np.maximum.at(event,codes,a['y'])
    y=a['y'];z=logit(np.clip(p,1e-7,1-1e-7))
    bs=[];pol=[];audit=[]
    for rep in range(100):
        ids=[]
        for label in (0,1):
            members=np.flatnonzero(event==label);rng.shuffle(members);ids.extend(members[:max(1,round(.2*len(members)))])
        cgroup=np.zeros(len(unique),bool);cgroup[ids]=True;cal=cgroup[codes];ev=~cal
        audit.append({'dataset':name,'repeat':rep+1,'calibration_groups':len(ids),'evaluation_groups':int((~cgroup).sum()),'group_assignment_sha256':hashlib.sha256(cgroup.tobytes()).hexdigest()})
        prevalence=float(y[cal].mean());null=brier_score_loss(y[ev],np.full(ev.sum(),prevalence))
        intercept,fulli,slope=rv.calibration_fit(y[cal],p[cal])
        forecasts={'uncalibrated':p[ev],'intercept_only':expit(intercept+z[ev]),'full_logistic':expit(fulli+slope*z[ev])}
        for method,pred in forecasts.items():
            bs.append({'dataset':name,'repeat':rep+1,'method':method,'brier':brier_score_loss(y[ev],pred),'brier_null':null,'reference_prevalence':prevalence,'brier_skill':1-brier_score_loss(y[ev],pred)/null})
        for target,threshold in [('five_raw_alerts',np.quantile(p[cal],.95,method='higher')),('80pct_raw_sensitivity',np.quantile(p[cal&(y==1)],.2,method='lower'))]:
            pol.append(dict(dataset=name,repeat=rep+1,policy=target,threshold=float(threshold),**policy(a,ev,p,threshold)))
        if (rep+1)%20==0:print(name,'recalibration/policy',rep+1,flush=True)
    pd.DataFrame(bs).to_csv(RESULTS/f'{name}_brier_repeated.csv',index=False)
    pd.DataFrame(pol).to_csv(RESULTS/f'{name}_policy_repeated.csv',index=False)
    pd.DataFrame(audit).to_csv(RESULTS/f'{name}_split_audit.csv',index=False)

def evaluate(name):
    a=load(name);models=joblib.load(PRIVATE/'revised_models.joblib');y=a['y'];x=a['x']
    pred={n:models[n].predict_proba(x)[:,1].astype(np.float64) for n in ('LR','HGB')}
    unique,codes=np.unique(a['record_id'],return_inverse=True);n=len(unique)
    evaluators={n:ci.RankingEvaluator(y.astype(float),p,codes) for n,p in pred.items()}
    rng=np.random.default_rng(SEED);boots={n:[] for n in pred}
    for rep in range(300):
        mult=np.bincount(rng.integers(0,n,n),minlength=n)
        for model,e in evaluators.items():boots[model].append(e.evaluate(mult))
        if (rep+1)%100==0:print(name,'paired bootstrap',rep+1,flush=True)
    rows=[];diff=[]
    for model,p in pred.items():
        point=basic(y,p);np.testing.assert_allclose(list(point.values()),evaluators[model].evaluate(np.ones(n)),rtol=1e-8)
        row=dict(dataset=name,model=model,n_landmarks=len(y),n_stays=n,n_positive_landmarks=int(y.sum()),n_event_stays=int(pd.Series(y).groupby(a['record_id']).max().sum()),**point,**ci.cluster_robust_calibration(y,p,codes))
        for j,m in enumerate(('auroc','auprc','brier')):row[m+'_low'],row[m+'_high']=np.quantile(np.asarray(boots[model])[:,j],[.025,.975])
        row['retrospective_brier_skill']=1-point['brier']/(y.mean()*(1-y.mean()))
        rows.append(row)
    for j,m in enumerate(('auroc','auprc','brier')):
        vals=np.asarray(boots['HGB'])[:,j]-np.asarray(boots['LR'])[:,j]
        diff.append({'dataset':name,'metric':m,'HGB_minus_LR':rows[1][m]-rows[0][m],'low':np.quantile(vals,.025),'high':np.quantile(vals,.975)})
    pd.DataFrame(rows).to_csv(RESULTS/f'{name}_metrics.csv',index=False)
    pd.DataFrame(diff).to_csv(RESULTS/f'{name}_paired_differences.csv',index=False)
    np.savez_compressed(PRIVATE/f'{name}_predictions.npz',**pred)
    sens=[]
    for cap in (6,12):
        xc=x.copy()
        for j,v in enumerate(rv.LAB_VARS):xc[a['lab_age'][:,j]>cap,rv.FEATURE_COLS.index(v+'_last')]=np.nan
        pc=models['HGB'].predict_proba(xc)[:,1]
        citl,intercept,slope=rv.calibration_fit(y,pc)
        sens.append(dict(dataset=name,analysis=f'lab_age_le_{cap}h',n_landmarks=len(y),n_event_landmarks=int(y.sum()),**basic(y,pc),citl=citl,slope=slope,lab_missing_percent=float(np.isnan(xc[:,31:]).mean()*100)))
        del xc
    keep=(y==0)|(a['lead_time_hours']>1)
    pg=models['gap'].predict_proba(x[keep])[:,1]
    sens.append(dict(dataset=name,analysis='one_hour_gap',n_landmarks=int(keep.sum()),n_event_landmarks=int(y[keep].sum()),**basic(y[keep],pg)))
    for label,keep in [('complete_horizon',a['observed_horizon_hours']>=6),('single_hour6',a['index_hour']==6)]:
        sens.append(dict(dataset=name,analysis=label,n_landmarks=int(keep.sum()),n_event_landmarks=int(y[keep].sum()),**basic(y[keep],pred['HGB'][keep])))
    pm=models['availability'].predict_proba(np.isnan(x).astype(np.float32))[:,1]
    sens.append(dict(dataset=name,analysis='availability_only',n_landmarks=len(y),n_event_landmarks=int(y.sum()),**basic(y,pm)))
    pd.DataFrame(sens).to_csv(RESULTS/f'{name}_sensitivities.csv',index=False)
    # Decile curves are aggregate figure sources.
    curve=pd.DataFrame({'p':pred['HGB'],'y':y});curve['decile']=pd.qcut(curve.p,10,duplicates='drop')
    curve.groupby('decile',observed=True).agg(predicted=('p','mean'),observed=('y','mean'),n=('y','size')).reset_index(drop=True).to_csv(RESULTS/f'{name}_calibration_curve.csv',index=False)
    cal_policy(name,a,pred['HGB'])
    print(name,'evaluation complete',flush=True)

if __name__=='__main__':
    if sys.argv[1]=='fit':model_stage()
    else:evaluate(sys.argv[1])
