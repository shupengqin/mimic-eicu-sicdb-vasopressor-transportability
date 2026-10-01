"""Revised figures: current cohorts and recalculated policy metrics only.

Quantitative panels use all aggregate results. Error bars are bootstrap 95%
intervals or repeated-split IQR as explicitly noted in figure legends.
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from config import *
FIG=DELIVERY/'Figures';FIG.mkdir(exist_ok=True)
mpl.rcParams.update({'font.family':'Times New Roman','font.size':9,'axes.titlesize':10,'axes.labelsize':9,
    'legend.fontsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'svg.fonttype':'none','pdf.fonttype':42,
    'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'savefig.facecolor':'white'})
NAMES=['test','eicu','sicdb'];LABELS=['MIMIC-IV\nreference-period test','eICU-CRD','SICdb'];SHORT=['MIMIC-IV','eICU-CRD','SICdb']
COLORS=['#3B6C8E','#3F8B82','#AA7448']
metrics=pd.concat([pd.read_csv(RESULTS/f'{n}_metrics.csv') for n in NAMES])
coh=pd.concat([pd.read_csv(RESULTS/f'{n}_cohort.csv') for n in NAMES]).set_index('cohort')

def save(fig,name):
    fig.savefig(FIG/f'{name}.svg',bbox_inches='tight')
    fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(FIG/f'{name}.png',dpi=300,bbox_inches='tight')
    fig.savefig(FIG/f'{name}.tiff',dpi=600,bbox_inches='tight',pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)

def panel(ax,s):ax.text(-.13,1.07,s,transform=ax.transAxes,weight='bold',fontsize=11,va='top')

# Fig. 1: only cohort construction is a flow. Outcome and exit quantities are
# parallel attributes of the full risk set, never nested attrition stages.
fig,axs=plt.subplots(1,3,figsize=(7.2,5.2))
for i,(n,ax) in enumerate(zip(NAMES,axs)):
    ax.axis('off');c=coh.loc[n]
    entries=[(0.92,LABELS[i]),(.72,f"Eligible stays\n{int(c.stays):,}"),(.49,f"Hourly landmarks\n{int(c.landmarks):,}"),(.25,f"Positive landmarks\n{int(c.positive_landmarks):,} ({c.positive_percent:.2f}%)"),(.055,f"Exit-truncated non-events\n{int(c.exit_truncated_landmarks):,} ({c.exit_percent:.2f}%)")]
    for j,(y,t) in enumerate(entries):
        ax.text(.5,y,t,ha='center',va='center',transform=ax.transAxes,fontsize=9,
            bbox=None if j==0 else dict(boxstyle='round,pad=.5',fc='#F4F7F8',ec=COLORS[i],lw=.8))
    ax.annotate('',xy=(.5,.58),xytext=(.5,.64),xycoords='axes fraction',arrowprops=dict(arrowstyle='->',color='#555555'))
fig.text(.5,-.02,'Positive and exit-truncated landmarks are separate attributes of the hourly risk set.',ha='center',fontsize=9)
fig.subplots_adjust(wspace=.28)
save(fig,'Figure_1')

fig,axs=plt.subplots(1,2,figsize=(7.2,3.5),layout='constrained')
for j,metric in enumerate(['auroc','auprc']):
    ax=axs[j];panel(ax,chr(97+j))
    for k,(model,color,marker) in enumerate([('LR','#8A8F95','s'),('HGB','#366D92','o')]):
        vals=metrics[metrics.model.eq(model)].set_index('dataset').loc[NAMES]
        yy=vals[metric].to_numpy();lo=vals[metric+'_low'].to_numpy();hi=vals[metric+'_high'].to_numpy()
        ax.errorbar(np.arange(3)+(k-.5)*.13,yy,yerr=[yy-lo,hi-yy],fmt=marker,color=color,capsize=3,label=model)
    ax.set_xticks(range(3),LABELS);ax.set_ylabel(metric.upper());ax.grid(axis='y',alpha=.15)
    if metric=='auroc':ax.set_ylim(.5,1)
    else:ax.set_ylim(bottom=0)
    ax.legend(loc='upper left' if metric=='auprc' else 'lower left')
save(fig,'Figure_2')

fig,axs=plt.subplots(2,3,figsize=(7.2,6.4),layout='constrained')
for i,n in enumerate(NAMES):
    ax=axs[0,i];c=pd.read_csv(RESULTS/f'{n}_calibration_curve.csv');top=max(c.predicted.max(),c.observed.max())*1.08
    ax.plot([0,top],[0,top],ls='--',color='#888888',lw=.8)
    ax.plot(c.predicted,c.observed,'o-',color=COLORS[i],markersize=4);ax.set_title(SHORT[i]);ax.set_xlabel('Mean predicted risk');ax.set_ylabel('Observed event rate');panel(ax,chr(97+i))
for j,(metric,target,title) in enumerate([('calibration_in_the_large',0,'Calibration-in-the-large'),('calibration_slope',1,'Calibration slope')]):
    ax=axs[1,j];v=metrics[metrics.model.eq('HGB')].set_index('dataset').loc[NAMES];yy=v[metric].to_numpy()
    ax.axhline(target,color='#888888',lw=.8,ls='--');ax.errorbar(range(3),yy,yerr=[yy-v[metric+'_ci_low'].to_numpy(),v[metric+'_ci_high'].to_numpy()-yy],fmt='o',color='#366D92',capsize=3)
    ax.set_xticks(range(3),SHORT,rotation=25);ax.set_ylabel(title);panel(ax,chr(100+j))
ax=axs[1,2]
for j,(method,label,color) in enumerate([('uncalibrated','Uncalibrated','#8A8F95'),('intercept_only','Intercept only','#8AB3AA'),('full_logistic','Intercept + slope','#366D92')]):
    vals=[]
    for n in NAMES:
        b=pd.read_csv(RESULTS/f'{n}_brier_repeated.csv');vals.append(b.loc[b.method.eq(method),'brier_skill'].quantile([.25,.5,.75]).to_numpy())
    vals=np.asarray(vals)
    ax.errorbar(np.arange(3)+(j-1)*.16,vals[:,1],yerr=[vals[:,1]-vals[:,0],vals[:,2]-vals[:,1]],fmt='o',markersize=4,color=color,capsize=2,label=label)
ax.axhline(0,color='#888888',ls='--',lw=.8);ax.set_xticks(range(3),SHORT,rotation=25);ax.set_ylabel('Held-out Brier skill');panel(ax,'f')
ax.legend(loc='lower center',bbox_to_anchor=(.5,1.03),fontsize=7)
save(fig,'Figure_3')

fig,axs=plt.subplots(2,3,figsize=(7.2,6.0),layout='constrained')
policies=[('five_raw_alerts','Target: 5 raw alerts per 100 rows'),('80pct_raw_sensitivity','Target: 80% raw landmark sensitivity')]
for row,(policy,title) in enumerate(policies):
    for col,(metric,ylabel,factor) in enumerate([('event_sensitivity','Events detected (%)',100),('false_alerts_100','False emitted alerts / 100 rows',1),('event_free_stays_false_alert_percent','Event-free stays with ≥1 false alert (%)',1)]):
        ax=axs[row,col]
        for i,n in enumerate(NAMES):
            p=pd.read_csv(RESULTS/f'{n}_policy_repeated.csv');v=p.loc[p.policy.eq(policy),metric].quantile([.25,.5,.75]).to_numpy()*factor
            ax.errorbar(i,v[1],yerr=[[v[1]-v[0]],[v[2]-v[1]]],fmt='o',color=COLORS[i],capsize=3)
        ax.set_xticks(range(3),SHORT,rotation=25);ax.set_ylabel(ylabel);ax.set_ylim(bottom=0)
        if factor==100 or metric.endswith('percent'):ax.set_ylim(0,100)
        panel(ax,chr(97+row*3+col))
        if col==1:ax.set_title(title,pad=16)
save(fig,'Figure_4')

fig,axs=plt.subplots(1,3,figsize=(7.2,3.3),layout='constrained')
for j,(metric,ylabel) in enumerate([('auroc','AUROC'),('citl','Calibration-in-the-large'),('lab_missing_percent','Missing laboratory inputs (%)')]):
    ax=axs[j]
    for i,n in enumerate(NAMES):
        s=pd.read_csv(RESULTS/f'{n}_sensitivities.csv').set_index('analysis')
        vals=[s.loc[f'lab_age_le_{h}h',metric] for h in [6,12]]
        ax.plot([6,12],vals,'o-',color=COLORS[i],label=SHORT[i])
    ax.set_xticks([6,12]);ax.set_xlabel('Maximum laboratory age (h)');ax.set_ylabel(ylabel);panel(ax,chr(97+j))
    if j==0:ax.set_ylim(.5,1);ax.legend(loc='lower right',fontsize=7)
save(fig,'Figure_S1')

h=pd.read_csv(RESULTS/'eicu_hospital_calibration.csv')
fig,ax=plt.subplots(figsize=(5.0,4.0),layout='constrained');ax.scatter(h.calibration_in_the_large,h.calibration_slope,s=20,color='#366D92',alpha=.7)
ax.axvline(0,color='#888888',ls='--',lw=.8);ax.axhline(1,color='#888888',ls='--',lw=.8)
ax.set_xlabel('Hospital calibration-in-the-large');ax.set_ylabel('Hospital calibration slope');ax.set_title(f'{len(h)} eligible eICU hospitals')
save(fig,'Figure_S2')
print('Four main and two supplementary figures exported.')
