"""Figures of the paper. Figure 2 is simulated here; Figures 3-5 plot values recorded in results/
(X1: ext_localiser_*.json; X2: ext_failure_size.json; R1: rev_r1_results.json). Usage: python scripts/make_figures.py
Writes PDF and PNG files to figures/."""
import sys; import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from core import simulate
from networks import synthetic
plt.rcParams.update({'pdf.fonttype':42,'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':9,'axes.linewidth':0.6})
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures') + os.sep; os.makedirs(OUT, exist_ok=True)

# Figure 2: relative influence on the synthetic network
wn=synthetic(); J=wn.junction_name_list
h0,_=simulate(wn)
fig,axs=plt.subplots(1,2,figsize=(7.0,3.3))
cmap=plt.get_cmap('viridis'); norm=Normalize(0,1)
for ax,(leak,title) in zip(axs,[('G3_0','(a) Leak at a shadow entry (G3_0)'),('G2_3','(b) Leak inside the meshed grid (G2_3)')]):
    h,_=simulate(wn,{leak:5e-4})
    dh=(h0-h); rel=(dh[J]/dh[leak]).clip(0,1)
    for _,l in wn.links():
        a=wn.get_node(l.start_node_name).coordinates; b=wn.get_node(l.end_node_name).coordinates
        ax.plot([a[0],b[0]],[a[1],b[1]],color='0.7',lw=0.8,zorder=1)
    xy=np.array([wn.get_node(n).coordinates for n in J])
    sc=ax.scatter(xy[:,0],xy[:,1],c=rel.values,cmap=cmap,norm=norm,s=22,edgecolors='k',linewidths=0.3,zorder=2)
    for r in ('R1','R2'):
        c=wn.get_node(r).coordinates; ax.scatter(*c,marker='s',s=45,c='white',edgecolors='k',zorder=3); ax.annotate(r,c,xytext=(3,4),textcoords='offset points',fontsize=7)
    c=wn.get_node(leak).coordinates; ax.scatter(*c,marker='*',s=140,c='red',edgecolors='k',linewidths=0.4,zorder=4)
    ax.set_title(title,fontsize=9); ax.set_aspect('equal'); ax.axis('off')
    print(leak, 'shadow D-nodes rel:', rel[[f'D{k}' for k in range(8)]].round(4).values.min(), 'median others', np.median(rel[[n for n in J if not n.startswith('D') and n!=leak]]).round(3))
cb=fig.colorbar(sc,ax=axs,shrink=0.75,pad=0.02); cb.set_label(r'Relative influence $\Delta h_i/\Delta h_j$')
fig.savefig(OUT+'fig2_influence.pdf',bbox_inches='tight'); fig.savefig(OUT+'fig2_influence.png',dpi=200,bbox_inches='tight')

# Figure 3: X1 localiser comparison
meth=['nearest_sensor','graph_interp','linear','gnn_plain','reconstructor','physics_loss','hybrid_nodom','hybrid']
lab=['Largest-drop\nsensor','Graph\ninterp.','Linear\nprior','Data-only\nGNN','Recon-\nstructor','Physics-\nloss GNN','Hybrid\n(no flags)','Hybrid']
lt={'nearest_sensor':(0.701,0),'graph_interp':(0.701,0),'linear':(0.750,0),'gnn_plain':(0.603,0.0315),'reconstructor':(0.661,0.0265),'physics_loss':(0.657,0.0145),'hybrid_nodom':(0.796,0.008),'hybrid':(0.776,0.0315)}
ky={'nearest_sensor':(0.193,0),'graph_interp':(0.193,0),'linear':(0.458,0),'gnn_plain':(0.198,0.0105),'reconstructor':(0.181,0.0075),'physics_loss':(0.181,0.007),'hybrid_nodom':(0.416,0.0175),'hybrid':(0.416,0.0225)}
fig,ax=plt.subplots(figsize=(7.0,2.8)); x=np.arange(len(meth)); w=0.38
for k,(d,name,col) in enumerate([(lt,'L-Town (33 utility sensors)','#3b6ea5'),(ky,'KY4 (48 sensors, 5%)','#d08c2e')]):
    m=[d[a][0] for a in meth]; e=[d[a][1] for a in meth]
    ax.bar(x+(k-0.5)*w,m,w,yerr=e,capsize=2,color=col,label=name,error_kw=dict(lw=0.7))
ax.axhline(0.750,color='#3b6ea5',ls=':',lw=0.8); ax.axhline(0.458,color='#d08c2e',ls=':',lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(lab,fontsize=7.5); ax.set_ylabel('Share within 300 m'); ax.set_ylim(0,0.9)
ax.legend(frameon=False,fontsize=8,loc='upper left'); ax.spines[['top','right']].set_visible(False)
fig.savefig(OUT+'fig3_localiser.pdf',bbox_inches='tight'); fig.savefig(OUT+'fig3_localiser.png',dpi=200,bbox_inches='tight')

# Figure 4: X2 failure-size estimate
fig,axs=plt.subplots(1,2,figsize=(7.0,2.6))
names=['NEET-LIN','NEET-2PT','NEET-REG']
mre={'L-Town':[1.890,0.107,0.080],'KY4':[1.124,0.223,0.138]}
w50={'L-Town':[0.02,0.96,1.00],'KY4':[0.18,0.80,0.95]}
x=np.arange(3)
for k,(net,col) in enumerate([('L-Town','#3b6ea5'),('KY4','#d08c2e')]):
    axs[0].bar(x+(k-0.5)*w,mre[net],w,color=col,label=net)
    axs[1].bar(x+(k-0.5)*w,w50[net],w,color=col,label=net)
axs[0].axhline(0.5,color='k',ls='--',lw=0.8); axs[0].text(2.45,0.53,'decision\nthreshold',fontsize=7,ha='right',va='bottom')
axs[0].set_ylabel('Median relative error'); axs[1].set_ylabel('Share within 50% of $A^*$')
for a in axs:
    a.set_xticks(x); a.set_xticklabels(names,fontsize=8); a.spines[['top','right']].set_visible(False)
axs[0].set_title('(a) Error of the critical-size estimate',fontsize=9); axs[1].set_title('(b) Estimates within 50%',fontsize=9)
axs[1].legend(frameon=False,fontsize=8,loc='upper left'); axs[1].set_ylim(0,1.05)
fig.tight_layout(); fig.savefig(OUT+'fig4_failure.pdf',bbox_inches='tight'); fig.savefig(OUT+'fig4_failure.png',dpi=200,bbox_inches='tight')

# Figure 5: robustness of NEET-2PT (R1)
conds=['Nominal','Roughness','Demand','PDD','$C_d$','Combined','Drift','Wrong\nnode']
lt=[0.093,0.087,0.091,0.095,0.114,0.116,0.101,0.183]; ky=[0.210,0.224,0.210,0.210,0.240,0.244,0.210,0.293]
fig,axs=plt.subplots(1,2,figsize=(7.4,3.3),gridspec_kw=dict(width_ratios=[2.3,1]))
x=np.arange(len(conds)); w=0.38
axs[0].bar(x-w/2,lt,w,color='#3b6ea5',label='L-Town'); axs[0].bar(x+w/2,ky,w,color='#d08c2e',label='KY4')
axs[0].axhline(0.5,color='k',ls='--',lw=0.8); axs[0].set_xticks(x); axs[0].set_xticklabels(conds,fontsize=9,rotation=35,ha='right')
axs[0].set_ylabel('Median relative error'); axs[0].set_ylim(0,0.75); axs[0].legend(frameon=False,fontsize=8,loc='upper left')
axs[0].set_title('(a) Model route under model mismatch',fontsize=9)
nz=[0,0.01,0.05,0.10]; ltn=[0.093,0.304,0.679,0.725]; kyn=[0.210,0.467,0.658,0.724]
axs[1].plot(nz,ltn,'o-',color='#3b6ea5',label='L-Town'); axs[1].plot(nz,kyn,'s-',color='#d08c2e',label='KY4')
axs[1].axhline(0.5,color='k',ls='--',lw=0.8); axs[1].set_xlabel('Pressure noise (m)'); axs[1].set_ylim(0,0.8)
axs[1].set_title('(b) Measured route with noise',fontsize=9)
for a in axs: a.spines[['top','right']].set_visible(False)
fig.tight_layout(); fig.savefig(OUT+'fig5_robustness.pdf',bbox_inches='tight'); fig.savefig(OUT+'fig5_robustness.png',dpi=200,bbox_inches='tight')
