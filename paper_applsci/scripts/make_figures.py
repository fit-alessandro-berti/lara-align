"""Build vector result figures from the included, checked evidence snapshot."""
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAPER = Path(__file__).resolve().parents[1]
D = json.loads((PAPER/"data/evidence.json").read_text())
R = D["original_results"]
BLUE, GREEN, AMBER, INK, MUTED = "#2066A8", "#087F72", "#BB641A", "#213547", "#566778"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.labelsize":10,
                     "axes.titlesize":11,"axes.titleweight":"bold","text.color":INK,
                     "axes.labelcolor":INK,"xtick.color":MUTED,"ytick.color":MUTED,
                     "axes.spines.top":False,"axes.spines.right":False,
                     "axes.edgecolor":"#ADB8C3","axes.linewidth":.7,
                     "pdf.fonttype":42,"svg.fonttype":"none","savefig.facecolor":"white"})


def save(fig, name):
    fig.savefig(PAPER/f"figures/{name}.pdf",bbox_inches="tight",pad_inches=.06)
    fig.savefig(PAPER/f"figures/{name}.svg",bbox_inches="tight",pad_inches=.06)
    plt.close(fig)


with (PAPER/"data/metrics.csv").open() as f:
    rows=list(csv.DictReader(f))
fig,ax=plt.subplots(figsize=(5.5,2.75),layout="constrained")
epochs=[int(r['epoch']) for r in rows]
ax.plot(epochs,[float(r['train_total']) for r in rows],color=BLUE,lw=2,label="Training (with perturbations)")
ax.plot(epochs,[float(r['val_total']) for r in rows],color=GREEN,lw=2,label="Validation")
ax.axvline(49,color=MUTED,ls="--",lw=1)
ax.annotate("Selected: epoch 49",(49,.684),xytext=(23,.75),fontsize=9,
            arrowprops={"arrowstyle":"->","color":MUTED},color=MUTED)
ax.set(xlabel="Training epoch",ylabel="Total loss",xlim=(1,50),ylim=(.65,1.3))
ax.grid(axis="y",alpha=.18);ax.legend(frameon=False,fontsize=9,loc="upper right")
save(fig,"training_curve")

hist=R['test_analysis_summary.json']['guided_gap_hist']
fig,ax=plt.subplots(figsize=(5.5,2.65),layout="constrained")
x=list(map(int,hist)); y=list(hist.values())
ax.bar(x,y,color=[BLUE]+[AMBER]*4,width=.6,zorder=3)
for a,b in zip(x,y):ax.text(a,b+10,str(b),ha="center",va="bottom",fontweight="bold")
ax.text(.98,.9,"466 of 512 candidates optimal\n46 candidates require a cost improvement",
        transform=ax.transAxes,ha="right",va="top",fontsize=10,color=INK)
ax.set(xlabel="Cost gap: candidate cost minus optimal cost",ylabel="Number of inputs",
       xticks=x,ylim=(0,530));ax.grid(axis="y",alpha=.18,zorder=0)
save(fig,"cost_gaps")

order=[('ordinary_tree','canonical_block','Tree / original identifiers'),
       ('ordinary_tree','isomorphic_renaming','Tree / renamed identifiers'),
       ('duplicate_vs_silent','duplicate_prefix','Duplicate prefix'),
       ('duplicate_vs_silent','silent_routing','Silent routing'),
       ('concurrent_vs_interleaved','parallel','Parallel'),
       ('concurrent_vs_interleaved','explicit_interleaving','Explicit interleaving'),
       ('m_nonfreechoice','canonical_block','M / block structure'),
       ('m_nonfreechoice','m_nonfreechoice','M / non-free choice')]
fig,ax=plt.subplots(figsize=(5.5,3.8),layout="constrained")
for i,(motif,rep,label) in enumerate(order):
    c=next(c for c in D['checked_classes'] if c['motif']==motif and c['representation']==rep)
    delta=100*(c['guided']['optimal']-c['unguided']['optimal'])/c['n']
    lo,hi=c['delta_ci95'];color=BLUE if delta>0 else AMBER if delta<0 else MUTED
    if i%2==0:ax.axhspan(i-.5,i+.5,color="#F2F5F8",zorder=0)
    ax.errorbar(delta,i,xerr=[[delta-lo],[hi-delta]],fmt='o',color=color,lw=1.8,capsize=3,zorder=4)
    ax.text(hi+1.3,i,f"{delta:+.1f}" if delta else "0.0",va='center',fontsize=9,color=color)
ax.axvline(0,color=MUTED,lw=.8,zorder=2)
ax.set(yticks=range(8),yticklabels=[o[2] for o in order],ylim=(7.6,-.7),xlim=(-8,43),
       xticks=[0,10,20,30,40],xlabel="Change in optimality rate (percentage points)")
ax.grid(axis='x',alpha=.15);save(fig,'guidance_effect')

rows=R['scaling_guided.json']['rows']
fig,axes=plt.subplots(1,2,figsize=(5.7,3.1),layout='constrained')
for ax,dev in zip(axes,[.15,.35]):
    rs=[r for r in rows if r['deviation_rate']==dev]
    for key,label,col,mark in [('fast_time_median','LARA fast',BLUE,'o'),
                                ('exact_time_median','Exact A*',GREEN,'s')]:
        ax.plot([r['size'] for r in rs],[r[key]*1000 for r in rs],label=label,
                color=col,marker=mark,markersize=5,lw=1.8)
    ax.set_title(f"Deviation rate {dev:.2f}")
    ax.set(xlabel="Visible leaf occurrences",xticks=[5,10,20,40],yscale='log',ylim=(.8,100),
           yticks=[1,10,100],yticklabels=['1','10','100'])
    ax.grid(axis='y',alpha=.18,which='major')
axes[0].set_ylabel('Median time per trace (ms, log scale)')
axes[0].legend(frameon=False,fontsize=9,loc='upper left')
save(fig,'scaling_latency')

fig,ax=plt.subplots(figsize=(5.5,2.8),layout='constrained')
for dev,mark,style in [(.15,'o','-'),(.35,'s','--')]:
    rs=[r for r in rows if r['deviation_rate']==dev]
    ax.plot([r['size'] for r in rs],[100*r['optimal_rate'] for r in rs],marker=mark,
            linestyle=style,color=BLUE if dev==.15 else AMBER,lw=2,
            label=f"Deviation rate {dev:.2f}")
ax.set(xlabel='Visible leaf occurrences',ylabel='Optimal candidates (%)',xticks=[5,10,20,40],ylim=(0,105))
ax.grid(axis='y',alpha=.18);ax.legend(frameon=False,fontsize=9)
save(fig,'scaling_quality')
print('Built five PDF/SVG figures from included evidence.')
