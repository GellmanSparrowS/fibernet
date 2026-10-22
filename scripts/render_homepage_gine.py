"""Presentation of archived GINE history/results; no training, inference or physics."""
from pathlib import Path
import argparse,hashlib,json,os
os.environ.setdefault('MPLCONFIGDIR','/tmp/fibernet-homepage-mpl-0813')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/fibernet-homepage-cache-0813')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(root,out):
 out.mkdir(parents=True,exist_ok=True);data=root/'docs/media/new/learning/data/gine'
 paths=[data/'training.json',data/'final_results.json'];before={str(p.relative_to(root)):digest(p) for p in paths}
 h=json.loads(paths[0].read_text());j=json.loads(paths[1].read_text());hist=h['history'];rows=j['rows']
 assert len(hist)==40 and len(rows)==16
 assert [r['epoch'] for r in hist]==list(range(40))
 xx=np.array([r['truth_mS'] for r in rows]);yy=np.array([r['gine_mS'] for r in rows]);assert np.isfinite(xx).all() and np.isfinite(yy).all()
 assert len({r['sample_id'] for r in rows})==16 and len({r['source_group'] for r in rows})==4
 BG='#081422';FG='#edf3fa';MUTED='#bed0e0';CYAN='#53dbd2';GOLD='#ffc468'
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':26,'text.color':FG,'axes.labelcolor':FG,'xtick.color':MUTED,'ytick.color':MUTED,'axes.edgecolor':'#547086','axes.facecolor':BG,'figure.facecolor':BG,'savefig.facecolor':BG,'svg.fonttype':'none','svg.hashsalt':'fibernet-homepage-gine-v1','axes.labelsize':26,'xtick.labelsize':24,'ytick.labelsize':24})
 frames=[];states=[];seq=list(range(1,40,2))+[40]
 def setup(title):
  fig=plt.figure(figsize=(9.6,6.4),dpi=100);ax=fig.add_axes([.16,.19,.80,.62]);fig.text(.06,.92,title,fontsize=27,weight='bold');ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='#6f8da5',alpha=.20,linewidth=1)
  return fig,ax
 def capture(fig,name=None):
  fig.canvas.draw();im=Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy());frames.append(im)
  if name:
   im.save(out/(name+'.png'));fig.savefig(out/(name+'.svg'),metadata={'Date':None})
   for w in [800,322]:im.resize((w,round(im.height*w/im.width)),Image.Resampling.LANCZOS).save(out/(name+f'_{w}.png'))
  plt.close(fig)
 for k in seq:
  fig,ax=setup(f'GINE training · {k:02d}/40')
  ax.plot(np.arange(1,k+1),[x['online_train_mse'] for x in hist[:k]],color=CYAN,lw=3.2,marker='o' if k==1 else None,markersize=8,label='Train')
  ax.plot(np.arange(1,k+1),[x['validation_mse'] for x in hist[:k]],color=GOLD,lw=3.2,marker='o' if k==1 else None,markersize=8,label='Validation')
  ax.set(xlim=(1,40),ylim=(0,2.7),xticks=[1,10,20,30,40],yticks=[0,1,2],xlabel='Epoch',ylabel='MSE (mS²)')
  ax.legend(frameon=False,fontsize=24,labelcolor=FG,loc='upper right',handlelength=1.7)
  states.append({'kind':'archived_history','epoch':k,'duration_ms':1200 if k==40 else 300})
  capture(fig,'gine_history' if k==40 else ('gine_start' if k==1 else None))
 fig,ax=setup('GINE · held-out evaluation')
 ax.plot([0,5],[0,5],color='#8295a9',ls='--',lw=2)
 ax.scatter(xx,yy,c=CYAN,s=130,edgecolor=BG,linewidth=.8,zorder=3)
 ax.set(xlim=(0,5),ylim=(0,5),xticks=[0,1,2,3,4,5],yticks=[0,1,2,3,4,5],xlabel='Solved conductance (mS)',ylabel='Prediction (mS)')
 m=j['metrics']['all']['gine'];ax.text(.035,.96,f"R² {m['r2']:.3f}\nRMSE {m['rmse_mS']:.3f} mS",transform=ax.transAxes,fontsize=24,va='top',color=GOLD,linespacing=1.45)
 fig.text(.96,.84,'16 cases · 4 held-out parents',fontsize=24,ha='right',color=MUTED)
 states.append({'kind':'final_held_out_evaluation','sample_ids':[r['sample_id'] for r in rows],'duration_ms':2200});capture(fig,'gine_final_evaluation')
 durations=[x['duration_ms'] for x in states]
 sheet=Image.new('RGB',(960,1280));sheet.paste(frames[-2],(0,0));sheet.paste(frames[-1],(0,640));palette=sheet.quantize(colors=256)
 q=[im.quantize(palette=palette,dither=Image.Dither.NONE) for im in frames]
 path=out/'gine_overview.gif';q[0].save(path,save_all=True,append_images=q[1:],duration=durations,loop=0,optimize=False,disposal=2)
 with Image.open(path) as im:
  assert im.n_frames==len(states) and im.info.get('loop')==0
  encoded=[(im.seek(i) or im.info['duration']) for i in range(im.n_frames)];assert encoded==durations
 assert before=={str(p.relative_to(root)):digest(p) for p in paths}
 receipt={'kind':'archived_learning_presentation','canvas':[960,640],'input_sha256':before,'states':states,'complete_history':hist,'final_rows':rows,'metrics_displayed':m,'all_16_final_points_inside_axes':bool(((xx>=0)&(xx<=5)&(yy>=0)&(yy<=5)).all()),'original_scatter_0_to_4_would_clip_truth_values':int((xx>4).sum()),'no_training_inference_or_physics':True,'gif_frames':len(states),'duration_ms':sum(durations),'gif_sha256':digest(path),'renderer_sha256':digest(__file__),'source_label_px':{'axis':26*100/72,'ticks':24*100/72},'visual_qa':'PENDING human inspection'}
 (out/'render_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({k:v for k,v in receipt.items() if k not in ['complete_history','final_rows','states']},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();main(a.source_root.resolve(),a.output_dir.resolve())
