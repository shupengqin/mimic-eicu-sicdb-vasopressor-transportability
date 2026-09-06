"""Create updated publication figures for the prediction-time estimand."""
from __future__ import annotations
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("PREDICTION_TIME_OUTPUT_DIR", REPO / "work" / "prediction_time" / "analysis"))
FIG_ROOT = Path(os.environ.get("PREDICTION_TIME_FIGURE_DIR", REPO / "work" / "prediction_time" / "figures"))
OUT = FIG_ROOT / "main"
SUPP = FIG_ROOT / "supplementary"
SOURCE = FIG_ROOT / "source_data"
OUT.mkdir(parents=True, exist_ok=True); SUPP.mkdir(parents=True, exist_ok=True); SOURCE.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 7.5, "axes.linewidth": .8, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none", "pdf.fonttype": 42, "legend.frameon": False})
BLUE, TEAL, GREY, RED, LIGHT, TEXT = "#175A9D", "#3D929C", "#707070", "#B64B46", "#D7D7D7", "#252525"
PALE_BLUE, PALE_TEAL, PALE_RED = "#DDE8F2", "#DDEBE8", "#F2D9D4"
DS = ["mimic_temporal_test", "eicu_external", "sicdb_external"]
LABELS = ["MIMIC-IV\n2020-2022", "eICU-CRD", "SICdb"]

def panel(ax, label): ax.text(-.16, 1.04, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", color=TEXT)
def save(fig, name, folder=OUT):
    for ext, kw in [("svg", {}), ("pdf", {}), ("png", {"dpi": 300}), ("tiff", {"dpi": 300})]:
        fig.savefig(folder / f"{name}.{ext}", bbox_inches="tight", facecolor="white", **kw)
    plt.close(fig)

def figure1():
    c = pd.read_csv(WORK / "prediction_time_cohort_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.1))
    for ax, d, lab in zip(axes, DS, LABELS):
        r = c[c.dataset.eq(d)].iloc[0]; ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.text(.5, .96, lab.replace("\n", " "), ha="center", va="top", fontsize=8.5, fontweight="bold")
        ax.text(.5, .89, "prediction-time risk set", ha="center", va="top", fontsize=6.8, color=GREY)
        boxes = [("Eligible stays", f"n = {int(r.n_stays):,}"), ("Hourly landmarks", f"n = {int(r.n_landmarks):,}"), ("Positive landmarks", f"n = {int(r.positive_landmarks):,} ({r.positive_landmarks_percent:.2f}%)"), ("Exit-truncated landmarks", f"{r.exit_truncated_landmarks_percent:.1f}%")]
        ys = [.72, .53, .34, .15]
        for i, (title, detail) in enumerate(boxes):
            fill = PALE_BLUE if i == 0 else (PALE_TEAL if i == 2 else "white")
            ax.add_patch(Rectangle((.08, ys[i]), .84, .12, facecolor=fill, edgecolor=BLUE if i == 2 else GREY, linewidth=1))
            ax.text(.5, ys[i] + .075, title, ha="center", va="center", fontsize=7.0, fontweight="bold" if i in (0, 2) else "normal")
            ax.text(.5, ys[i] + .035, detail, ha="center", va="center", fontsize=6.8, color=GREY)
            if i < len(boxes)-1: ax.add_patch(FancyArrowPatch((.5, ys[i]), (.5, ys[i+1]+.12), arrowstyle="-|>", mutation_scale=8, linewidth=.7, color=GREY))
        ax.text(.5, .04, "Untreated at the landmark;\nexit without initiation retained as non-event", ha="center", va="bottom", fontsize=6.0, color=TEXT)
    fig.subplots_adjust(left=.02, right=.98, top=.96, bottom=.03, wspace=.16)
    save(fig, "Figure_1_prediction_time_risk_set")

def forest(ax, metrics, metric, xlim, xlabel):
    y = np.arange(3)[::-1]; offsets = {"logistic_regression": -.10, "hist_gradient_boosting": .10}
    for model, color, marker, name in [("logistic_regression", GREY, "o", "Logistic regression"), ("hist_gradient_boosting", BLUE, "s", "HGB")]:
        sub = metrics[metrics.model.eq(model)].set_index("dataset").loc[DS]
        est = sub[metric].to_numpy(); lo = sub[f"{metric}_ci_low"].to_numpy(); hi = sub[f"{metric}_ci_high"].to_numpy()
        ax.errorbar(est, y + offsets[model], xerr=[est-lo, hi-est], fmt=marker, color=color, ecolor=color, capsize=2.5, ms=4, elinewidth=1, label=name)
    ax.set_yticks(y); ax.set_yticklabels(LABELS); ax.set_xlim(*xlim); ax.set_xlabel(xlabel); ax.grid(axis="x", color=LIGHT, linewidth=.5); ax.set_axisbelow(True)

def figure2():
    c = pd.read_csv(WORK / "prediction_time_cohort_summary.csv"); m = pd.read_csv(WORK / "prediction_time_metrics.csv")
    fig = plt.figure(figsize=(7.2, 3.35)); gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1], wspace=.52)
    ax = fig.add_subplot(gs[0]); ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1); panel(ax, "a")
    boxes = [("MIMIC-IV 2008-2016", "development"), ("MIMIC-IV 2017-2019", "selection"), ("Frozen HGB", "refit 2008-2019"), ("Temporal / external scoring", "no refit or selection")]
    ys = [.78, .59, .40, .21]
    for i, (a, b) in enumerate(boxes):
        fill = PALE_BLUE if i == 0 else (PALE_TEAL if i == 2 else "white")
        ax.add_patch(Rectangle((.08, ys[i]), .84, .12, facecolor=fill, edgecolor=BLUE if i == 2 else GREY, linewidth=1))
        ax.text(.5, ys[i]+.075, a, ha="center", va="center", fontsize=7.2, fontweight="bold" if i in (0,2) else "normal")
        ax.text(.5, ys[i]+.036, b, ha="center", va="center", fontsize=6.5, color=GREY)
        if i < 3: ax.add_patch(FancyArrowPatch((.5, ys[i]), (.5, ys[i+1]+.12), arrowstyle="-|>", mutation_scale=8, linewidth=.7, color=GREY))
    ax.text(.5, .06, "Primary estimand: event before\nmin(landmark + 6 h, ICU/unit exit)", ha="center", va="bottom", fontsize=6.3, color=TEXT)
    ax = fig.add_subplot(gs[1]); forest(ax, m, "auroc", (.62, .90), "AUROC"); panel(ax, "b")
    ax = fig.add_subplot(gs[2]); forest(ax, m, "auprc", (0, .15), "AUPRC"); panel(ax, "c")
    handles, labels = fig.axes[1].get_legend_handles_labels(); fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(.68, -.02), fontsize=7)
    fig.subplots_adjust(left=.05, right=.98, top=.95, bottom=.19)
    save(fig, "Figure_2_study_design_and_discrimination")

def calibration_data(dataset):
    with np.load(WORK / f"prediction_time_{dataset}_hist_gradient_boosting.npz") as d: p, y = d["p"].astype(float), d["y"].astype(int)
    q = pd.qcut(p, 10, duplicates="drop"); frame = pd.DataFrame({"p":p,"y":y,"q":q}).groupby("q", observed=True).agg(predicted=("p","mean"), observed=("y","mean"))
    return frame

def figure3():
    m = pd.read_csv(WORK / "prediction_time_metrics.csv"); b = pd.read_csv(WORK / "prediction_time_brier_skill_summary.csv")
    fig = plt.figure(figsize=(7.2, 4.5)); gs = fig.add_gridspec(2, 3, hspace=.58, wspace=.48)
    for i, (d, lab) in enumerate(zip(DS, LABELS)):
        ax = fig.add_subplot(gs[0, i]); z = calibration_data(d); mx = max(z.predicted.max(), z.observed.max(), .02) * 1.08
        ax.plot([0,mx],[0,mx], "--", color=GREY, linewidth=.8); ax.plot(z.predicted, z.observed, "o-", color=BLUE, ms=3.2, linewidth=1.1)
        ax.set_xlim(0,mx); ax.set_ylim(0,mx); ax.set_xlabel("Predicted risk"); ax.set_ylabel("Observed fraction"); ax.set_title(lab.replace("\n", " "), fontsize=8); ax.grid(color=LIGHT, linewidth=.4); panel(ax, chr(ord('a')+i))
    ax = fig.add_subplot(gs[1,0]); sub=m[m.model.eq("hist_gradient_boosting")].set_index("dataset").loc[DS]; y=np.arange(3)[::-1]; est=sub.calibration_in_the_large.to_numpy(); lo=sub.calibration_in_the_large_ci_low.to_numpy(); hi=sub.calibration_in_the_large_ci_high.to_numpy(); ax.errorbar(est,y,xerr=[est-lo,hi-est],fmt="s",color=BLUE,ecolor=BLUE,capsize=2.5); ax.axvline(0,color=GREY,ls="--",lw=.8); ax.set_yticks(y); ax.set_yticklabels(LABELS); ax.set_xlim(-1.5,.6); ax.set_xlabel("CITL"); ax.grid(axis="x",color=LIGHT,lw=.5); panel(ax,"d")
    ax = fig.add_subplot(gs[1,1]); est=sub.calibration_slope.to_numpy(); lo=sub.calibration_slope_ci_low.to_numpy(); hi=sub.calibration_slope_ci_high.to_numpy(); ax.errorbar(est,y,xerr=[est-lo,hi-est],fmt="s",color=BLUE,ecolor=BLUE,capsize=2.5); ax.axvline(1,color=GREY,ls="--",lw=.8); ax.set_yticks(y); ax.set_yticklabels(LABELS); ax.set_xlim(.35,1.2); ax.set_xlabel("Calibration slope"); ax.grid(axis="x",color=LIGHT,lw=.5); panel(ax,"e")
    ax = fig.add_subplot(gs[1,2]); x=np.arange(3); width=.23; cols=[GREY,TEAL,"#8C6BB1"]; vals=[b.set_index("dataset").loc[DS,"brier_skill_uncalibrated_median"], b.set_index("dataset").loc[DS,"brier_skill_intercept_only_median"], b.set_index("dataset").loc[DS,"brier_skill_full_logistic_median"]];
    for j,v in enumerate(vals): ax.bar(x+(j-1)*width,v,width,color=cols[j],label=["Uncalibrated","Intercept only","Full logistic"][j])
    ax.axhline(0,color=TEXT,lw=.7); ax.set_xticks(x); ax.set_xticklabels(LABELS); ax.set_ylabel("Brier skill"); ax.grid(axis="y",color=LIGHT,lw=.5); panel(ax,"f"); ax.legend(fontsize=6,loc="lower left")
    fig.subplots_adjust(left=.10,right=.98,top=.94,bottom=.13)
    save(fig, "Figure_3_calibration_and_probability_skill")

def figure4():
    p = pd.read_csv(WORK / "prediction_time_policy_summary.csv")
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.3), sharex="col")
    for i, (d, lab) in enumerate(zip(DS, LABELS)):
        sub=p[p.dataset.eq(d)].set_index("strategy")
        names=["fixed_5_alerts_per_100_rows","target_80_percent_calibration_sensitivity"]; x=np.arange(2); short=["5 / 100","80% sens."]
        ax=axes[0,i]; raw=sub.loc[names,"raw_alerts_per_100_rows_median"]; emitted=sub.loc[names,"alerts_per_100_rows_median"]; ax.bar(x-.15,raw,width=.3,color=GREY,label="Before suppression"); ax.bar(x+.15,emitted,width=.3,color=TEAL,label="Emitted after 6 h"); ax.set_title(lab.replace("\n"," "),fontsize=8); ax.set_xticks(x); ax.set_xticklabels(short); ax.set_ylabel("Alerts / 100 rows"); ax.grid(axis="y",color=LIGHT,lw=.5); panel(ax,chr(ord('a')+i))
        ax=axes[1,i]; raw=sub.loc[names,"raw_landmark_sensitivity_median"]; post=sub.loc[names,"landmark_sensitivity_median"]; stay=sub.loc[names,"event_stay_sensitivity_median"]; ax.plot(x,raw,"o-",color=GREY,label="Raw landmark"); ax.plot(x,post,"s-",color=TEAL,label="Post-suppression landmark"); ax.plot(x,stay,"^-",color=BLUE,label="Event-stay"); ax.set_xticks(x); ax.set_xticklabels(short); ax.set_ylim(0,1.05); ax.set_ylabel("Sensitivity"); ax.grid(axis="y",color=LIGHT,lw=.5); panel(ax,chr(ord('d')+i))
    handles, labels = axes[0,0].get_legend_handles_labels(); handles2, labels2 = axes[1,0].get_legend_handles_labels(); fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.02),ncol=2,fontsize=7); fig.legend(handles2,labels2,loc="upper center",bbox_to_anchor=(.5,-.035),ncol=3,fontsize=7)
    fig.subplots_adjust(left=.09,right=.98,top=.93,bottom=.18,wspace=.47,hspace=.55)
    save(fig, "Figure_4_alert_burden_and_sensitivity")

def figure_s1():
    h=pd.read_csv(WORK/"prediction_time_hospital_calibration.csv"); o=pd.read_csv(WORK/"prediction_time_outcome_sensitivity.csv"); g=pd.read_csv(WORK/"prediction_time_one_hour_gap.csv"); e=pd.read_csv(WORK/"prediction_time_exit_summary.csv")
    fig,axes=plt.subplots(2,2,figsize=(7.2,4.4)); ax=axes[0,0]; ax.scatter(h.calibration_in_the_large,h.calibration_slope,s=12,c=BLUE,alpha=.7); ax.axvline(0,color=GREY,ls="--",lw=.7); ax.axhline(1,color=GREY,ls="--",lw=.7); ax.set_xlabel("Hospital CITL"); ax.set_ylabel("Hospital calibration slope"); ax.grid(color=LIGHT,lw=.4); panel(ax,"a")
    ax=axes[0,1]; sub=o.pivot(index="dataset",columns="analysis",values="auroc").loc[DS]; sub=sub[["norepinephrine_at_first","strict_norepinephrine_only"]]; sub.columns=["Norepinephrine at first", "Strict norepinephrine only"]; sub.plot.bar(ax=ax,color=[BLUE,TEAL],width=.72); ax.set_xticklabels(LABELS,rotation=0); ax.set_ylabel("AUROC"); ax.set_ylim(.55,.9); ax.grid(axis="y",color=LIGHT,lw=.4); ax.legend(fontsize=6, frameon=False); panel(ax,"b")
    ax=axes[1,0]; sub=g[g.analysis.eq("one_hour_gap_hgb")].set_index("dataset").loc[DS]; primary=pd.read_csv(WORK/"prediction_time_metrics.csv"); primary=primary[primary.model.eq("hist_gradient_boosting")].set_index("dataset").loc[DS]; x=np.arange(3); ax.plot(x,primary.auroc,"o-",color=GREY,label="Primary"); ax.plot(x,sub.auroc,"s-",color=BLUE,label="One-hour gap"); ax.set_xticks(x); ax.set_xticklabels(LABELS); ax.set_ylim(.6,.9); ax.set_ylabel("AUROC"); ax.grid(axis="y",color=LIGHT,lw=.4); ax.legend(fontsize=6); panel(ax,"c")
    ax=axes[1,1]; ax.bar(np.arange(3),e.set_index("dataset").loc[DS,"percent_exit_truncated_landmarks"],color=TEAL); ax.set_xticks(np.arange(3)); ax.set_xticklabels(LABELS); ax.set_ylabel("Exit-truncated landmarks (%)"); ax.grid(axis="y",color=LIGHT,lw=.4); panel(ax,"d")
    fig.subplots_adjust(left=.1,right=.98,top=.94,bottom=.13,wspace=.5,hspace=.58); save(fig,"Supplementary_Figure_S1_prediction_time_robustness",SUPP)

def main():
    figure1(); figure2(); figure3(); figure4(); figure_s1(); print(f"Wrote figures to {OUT}")

if __name__ == "__main__": main()
