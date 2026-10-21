"""Display three archived fixed environment actions, without interpolation or reruns."""
import os,tempfile
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MPLCONFIGDIR']=tempfile.mkdtemp(prefix='fibernet-learning-mpl-');os.environ['XDG_CACHE_HOME']=tempfile.mkdtemp(prefix='fibernet-learning-cache-')
import pathlib,json,hashlib,time,resource
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image
R=pathlib.Path(__file__).resolve().parent;D=R/'data/rl';O=R/'media';start=time.process_time()
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'text.color':'#dce6f2','axes.labelcolor':'#b8c9df','xtick.color':'#92a8c0','ytick.color':'#92a8c0','axes.edgecolor':'#36495e','axes.facecolor':'#101f30','figure.facecolor':'#081422','savefig.facecolor':'#081422'})
frames=[];data=[]
for case in ['H4','K3','T5']:
 b=D/'fibernet-rl-native-campaign-attempt03-20261016'/case;s=D/'fibernet-rl-native-stageA-20261016'/case
 load=lambda p:json.loads(p.read_text())
 g=load(b/'solver_graph.json');z=load(s/'zero_graph.json');raw=load(b/'raw_result.json');ret=load(b/'env_return.json');control=load(s/'controls.json');data.append((case,g,z,raw,ret,control))
for phase in range(3):
 fig,axes=plt.subplots(3,1,figsize=(8.4,11.88),dpi=100);fig.subplots_adjust(top=.83,bottom=.16,left=.10,right=.69,hspace=.31)
 for ax,(case,g,z,raw,ret,control) in zip(axes,data):
  graph=z if phase==0 else g;ids={int(n['id']):i for i,n in enumerate(graph['nodes'])};pos=np.array([n['position'] for n in graph['nodes']]);pairs=np.array([[ids[e['node_i']],ids[e['node_j']]] for e in graph['edges']]);
  if phase==2:pos=np.array(raw['deformed_positions'])
  lc=LineCollection(pos[pairs][:,:,:2],colors=['#39d5cf','#c49af7','#fbb755'][phase],linewidths=.9);ax.add_collection(lc);ax.autoscale();ax.set_aspect('equal');ax.set_title(f'{case}  |  {len(pos)} nodes · {len(pairs)} members',loc='left');ax.set(xlabel='x (model coordinate)',ylabel='y');ax.text(1.08,.85,'Fixed action',transform=ax.transAxes,weight='bold');act=control['actual_action_float32'];nonzero=[(i,v) for i,v in enumerate(act) if v];ax.text(1.08,.68,'Nonzero components:\n'+'\n'.join(f'a[{i}] = {v:g}' for i,v in nonzero),transform=ax.transAxes,va='top',linespacing=1.5,fontsize=8)
  if phase==2:ax.text(1.08,.22,f'Reward {ret["reward"]:.5f}\nTerminated: {ret["terminated"]}\nTruncated: {ret["truncated"]}',transform=ax.transAxes,va='top',linespacing=1.6,fontsize=8,color='#fbb755')
 stages=['01  RESET GEOMETRY','02  FIXED ACTION → NETWORK','03  RECORDED TRANSIENT RESPONSE'];fig.text(.10,.95,'FIBERNET / RL ENVIRONMENT',color='#39d5cf',weight='bold',fontsize=12);fig.text(.10,.91,'Structure → action → reward',weight='bold',fontsize=22);fig.text(.10,.865,stages[phase],fontsize=12,color='#fbb755')
 fig.text(.10,.065,'Prescribed actions; no trained policy.\nThree recorded environment steps shown as discrete states.\nNondimensional axial transient model: unit nodal mass, k = 1000, horizon 0.02.',fontsize=8.5,color='#b8c9df',linespacing=1.7)
 fig.canvas.draw();im=Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy());im.save(O/f'rl_environment_stage_{phase}.png');fig.savefig(O/f'rl_environment_stage_{phase}.svg');frames.append(im);plt.close(fig)
frames[0].save(O/'rl_environment_fixed_actions.gif',save_all=True,append_images=frames[1:],duration=1800,loop=0,optimize=False)
(O/'rl_environment_receipt.json').write_text(json.dumps({'source_archive_sha256':'0eefbb109cc8b88c52d6401156bfec7582ce8f5dddcca72264720007275f63fc','frames':3,'size_px':[840,1188],'duration_ms_per_frame':1800,'cpu_s':time.process_time()-start,'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'new_training_inference_physics_calls':0,'gif_sha256':hashlib.sha256((O/'rl_environment_fixed_actions.gif').read_bytes()).hexdigest()},indent=2))
