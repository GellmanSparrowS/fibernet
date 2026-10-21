"""Animate stored epochs only. No model imports, inference, fitting or solves."""
import os,tempfile
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MPLCONFIGDIR']=tempfile.mkdtemp(prefix='fibernet-learning-mpl-')
os.environ['XDG_CACHE_HOME']=tempfile.mkdtemp(prefix='fibernet-learning-cache-')
import json,pathlib,hashlib,time,resource
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image
R=pathlib.Path(__file__).resolve().parent;D=R/'data/gine';O=R/'media';O.mkdir(exist_ok=True)
t0=time.process_time();h=json.loads((D/'training.json').read_text());j=json.loads((D/'final_results.json').read_text());rows=j['rows'];sid='gine-9204-v2'
with np.load(D/f'inputs/{sid}.npz',allow_pickle=False) as f:a={k:f[k] for k in f.files}
with np.load(D/f'fields/{sid}.npz',allow_pickle=False) as f:b={k:f[k] for k in f.files}
ids={int(x):i for i,x in enumerate(b['node_ids'])};v=np.array([b['voltage_V'][ids[int(x)]] for x in a['node_ids']]);pos=a['positions'][:,:2]*1000;seg=pos[a['edge_index'].T];voltage=v[a['edge_index']].mean(0)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'text.color':'#dce6f2','axes.labelcolor':'#b8c9df','xtick.color':'#92a8c0','ytick.color':'#92a8c0','axes.edgecolor':'#36495e','axes.facecolor':'#101f30','figure.facecolor':'#081422','savefig.facecolor':'#081422'})
frames=[];seq=list(range(1,41,2))+[40]*4
for frame,k in enumerate(seq):
 fig=plt.figure(figsize=(8.4,11.88),dpi=100);gs=fig.add_gridspec(3,2,left=.10,right=.94,top=.85,bottom=.16,hspace=.42,wspace=.35,height_ratios=[1.55,1,.90]);ax0=fig.add_subplot(gs[0,0]);mat=LineCollection(seg,array=a['sigma_S_per_m']/1000,cmap='cool',linewidths=.65);ax0.add_collection(mat);ax0.autoscale();ax0.set_aspect('equal');ax0.set_title('Actual material assignment',loc='left',pad=10);ax0.set(xlabel='x (mm)',ylabel='y (mm)');ax0.text(0,-.25,'1 and 20 kS/m · 1,423 nodes',transform=ax0.transAxes,color='#b8c9df',fontsize=8);ax=fig.add_subplot(gs[0,1]);lc=LineCollection(seg,array=voltage,cmap='plasma',linewidths=.65);ax.add_collection(lc);ax.autoscale();ax.set_aspect('equal');ax.set_title('Saved solved voltage',loc='left',pad=10);ax.set(xlabel='x (mm)',ylabel='y (mm)');fig.colorbar(lc,ax=ax,label='Voltage (V)',fraction=.035,pad=.03)
 ax=fig.add_subplot(gs[1,:]);hist=h['history'][:k];ax.plot([x['epoch']+1 for x in hist],[x['online_train_mse'] for x in hist],color='#39d5cf',lw=2,label='Online training');ax.plot([x['epoch']+1 for x in hist],[x['validation_mse'] for x in hist],color='#fbb755',lw=2,label='Validation');ax.set(xlim=(1,40),ylim=(0,2.7),xlabel='Recorded epoch',ylabel='MSE (mS²)',title=f'Actual archived training history | epoch {k:02d}/40');ax.legend(frameon=False,labelcolor='#dce6f2');ax.grid(alpha=.12)
 ax=fig.add_subplot(gs[2,0]);xx=np.array([r['truth_mS'] for r in rows]);yy=np.array([r['gine_mS'] for r in rows]);ax.plot([0,4],[0,4],color='#5a7087',ls='--');ax.set(xlim=(0,4),ylim=(0,4),xlabel='Solved (mS)',ylabel='Prediction (mS)',title='Final evaluation, once')
 if k==40:ax.scatter(xx,yy,c=[r['variant'] for r in rows],cmap='viridis',s=25)
 else:ax.text(.5,.5,'Final points revealed\nafter training replay',ha='center',va='center',transform=ax.transAxes,color='#b8c9df')
 ax=fig.add_subplot(gs[2,1]);ax.axis('off');ax.text(0,.98,'48 NETWORKS / 12 PARENTS',weight='bold',fontsize=10,va='top');ax.text(0,.76,'24 train · 8 validation · 16 final\nSplit by parent, not by variant\nAdamW · seed 42 · 40 epochs',linespacing=1.7,va='top');ax.text(0,.35,'Final R² 0.831 · RMSE 0.480 mS\n4 held-out parent geometries\nTarget: network conductance',linespacing=1.7,color='#fbb755',va='top')
 fig.text(.10,.95,'FIBERNET / LEARN',color='#39d5cf',weight='bold',fontsize=12);fig.text(.10,.913,'Edge-aware GINE',weight='bold',fontsize=27);fig.text(.10,.88,'Conductance learning across material-assigned networks',fontsize=10,color='#b8c9df')
 fig.text(.10,.075,'Synthetic ohmic networks; fixed reference field; parent-level split.\nFinal evaluation: 16 material variants from 4 held-out parent geometries.',fontsize=8.5,color='#b8c9df',linespacing=1.7)
 fig.canvas.draw();im=Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy());frames.append(im)
 if frame in (0,10,len(seq)-1):im.save(O/f'gine_training_frame_{frame:02d}.png')
 if frame==len(seq)-1:fig.savefig(O/'gine_training_final.svg')
 plt.close(fig)
frames[0].save(O/'gine_training_history.gif',save_all=True,append_images=frames[1:],duration=300,loop=0,optimize=False)
receipt={'rendered_frames':len(frames),'encoded_frames':Image.open(O/'gine_training_history.gif').n_frames,'size_px':[840,1188],'duration_ms_per_frame':300,'cpu_s':time.process_time()-t0,'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'source_archive_sha256':'005dae4ae98d40e0bdb538d3e9fce52e4c433aac45b7e321d83cc6b4e2dbb633','source_tree':'8fabc86ee457c1002bab48544ef7b53f8737d185','training_or_inference_or_physics_calls':0,'history_sha256':hashlib.sha256((D/'training.json').read_bytes()).hexdigest(),'gif_sha256':hashlib.sha256((O/'gine_training_history.gif').read_bytes()).hexdigest()};(O/'gine_training_receipt.json').write_text(json.dumps(receipt,indent=2));print(receipt)
