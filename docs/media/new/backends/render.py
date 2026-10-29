"""Render saved backend states only; no solver calls, interpolation or camera animation.
Run from any directory with Python, NumPy and Pillow installed.
"""
from pathlib import Path
import json, html, time, resource, hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT=Path(__file__).resolve().parent
MEDIA=ROOT; DATA=MEDIA/'data'
BG='#08121f'; PANEL='#101f30'; FG='#e9f1f7'; MUTED='#94a9bc'; GOLD='#f4ba66'; CYAN='#4edacb'
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
class Canvas:
 def __init__(self):
  self.im=Image.new('RGB',(800,1132),BG);self.d=ImageDraw.Draw(self.im);self.svg=['<svg xmlns="http://www.w3.org/2000/svg" width="800" height="1132" viewBox="0 0 800 1132">','<rect width="800" height="1132" fill="'+BG+'"/>']
 def rect(self,b,fill):
  self.d.rectangle(b,fill=fill);x,y,x2,y2=b;self.svg.append(f'<rect x="{x}" y="{y}" width="{x2-x}" height="{y2-y}" fill="{fill}"/>')
 def line(self,pts,fill,width=1):
  pts=[tuple(map(float,p)) for p in pts]; self.d.line(pts,fill=fill,width=width);s=' '.join(f'{x:.2f},{y:.2f}' for x,y in pts);self.svg.append(f'<polyline points="{s}" fill="none" stroke="{fill}" stroke-width="{width}"/>')
 def text(self,xy,s,size=16,fill=FG):
  self.d.text(xy,s,font=ImageFont.truetype(FONT,size),fill=fill);x,y=xy;self.svg.append(f'<text x="{x}" y="{y+size}" font-family="DejaVu Sans,sans-serif" font-size="{size}" fill="{fill}">{html.escape(s)}</text>')
 def circle(self,p,r,color):
  x,y=p;self.d.ellipse((x-r,y-r,x+r,y+r),fill=color);self.svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r}" fill="{color}"/>')
def render(name,cfg):
 start=time.process_time();data_path=DATA/cfg['data_file']
 assert hashlib.sha256(data_path.read_bytes()).hexdigest()==cfg['data_sha256'], 'Saved data checksum mismatch'
 z=np.load(data_path, allow_pickle=False);p=z['positions'];edges=z['endpoints']
 assert p.ndim==3 and p.shape[2]==3 and np.isfinite(p).all()
 assert len(np.unique(z['node_ids']))==p.shape[1] and len(np.unique(z['edge_ids']))==len(edges)
 assert edges.min()>=0 and edges.max()<p.shape[1] and np.all(np.diff(z['times'])>0)
 disp=np.linalg.norm(p-p[0],axis=2);maxu=disp.max();shown=p[0]+cfg['magnification']*(p-p[0]);
 # A fixed orthographic projection. Planar Taichi geometry remains planar.
 matrix=np.array([[.83,-.55,0],[.26,.39,-.88]]) if name=='lammps' else np.array([[1.,0,0],[0,-1.,0]])
 proj=shown@matrix.T; ref=p[0]@matrix.T;lo=proj.min(axis=(0,1));hi=proj.max(axis=(0,1));scale=min(660/(hi[0]-lo[0]),570/(hi[1]-lo[1]));center=(hi+lo)/2
 proj=(proj-center)*scale+[400,450];ref=(ref-center)*scale+[400,450]
 ke=z['kinetic'];pe=z['potential'];energy_scale=max(np.max(ke),np.max(pe));frames=[]
 for i in range(len(p)):
  c=Canvas();c.text((28,19),'FIBERNET  /  SAVED SIMULATION',13,CYAN);c.text((28,44),cfg['title'],28);c.text((28,83),cfg['subtitle'],17,MUTED)
  c.rect((22,120,778,785),PANEL);c.rect((22,802,778,1062),PANEL)
  for a,b in edges:c.line([ref[a],ref[b]],'#2b3c4d')
  for a,b in edges:
   q=float((disp[i,a]+disp[i,b])/(2*maxu));rgb=np.array([53,121,145])*(1-q)+np.array([246,186,101])*q;color='#'+''.join(f'{int(v):02x}' for v in rgb);c.line([proj[i,a],proj[i,b]],color,2 if name=='lammps' else 1)
  if name=='lammps':
   for v in proj[i]:c.circle(v,3,CYAN)
  c.text((38,132),'Displacement ×1000 · fixed camera',17,FG);c.text((38,756),'Gray: initial geometry   Color: |u| (unscaled)',15,MUTED)
  unit='µs' if name=='lammps' else '(nondimensional)';t=z['times'][i]*(1e6 if name=='lammps' else 1)
  c.text((40,819),'PHYSICAL STATE',14,CYAN);c.text((40,847),f't = {t:.4g} {unit}',22);c.text((40,879),f"Step {z['steps'][i]} / {z['steps'][-1]} · {i+1}/{len(p)} saved views",15,MUTED)
  c.text((425,819),'MAX DISPLACEMENT',14,CYAN);val=disp[i].max()*(1e6 if name=='lammps' else 1);uunit='µm' if name=='lammps' else '(nondim.)';c.text((425,847),f'{val:.5g} {uunit}',22)
  c.text((40,925),'SAVED ENERGY',14,CYAN)
  for ys,col in [(ke,GOLD),(pe,CYAN)]:
   pts=np.column_stack((np.linspace(425,750,len(p)),1034-110*ys/energy_scale));c.line(pts,'#34495b');c.line(pts[:i+1],col,2) if i else None;c.circle(pts[i],3,col)
  eu='J' if name=='lammps' else 'nondim.';c.text((40,954),f'Kinetic     {ke[i]:.4e} {eu}',16,GOLD);c.text((40,985),f'Potential  {pe[i]:.4e} {eu}',16,CYAN)
  c.text((40,1026),'No frame interpolation',14,MUTED)
  c.text((28,1075),'Synthetic benchmark · saved states · loop resets after the final state',15,MUTED)
  note='Harmonic bonded network · SI units · short-time response' if name=='lammps' else 'Axial mass network · nondimensional · sampled cutoff checks only'
  c.text((28,1100),note,15,MUTED);frames.append(c.im.quantize(colors=96))
  if i==len(p)-1:
   (MEDIA/(name+'_saved_dynamics.svg')).write_text('\n'.join(c.svg)+"\n</svg>\n");c.im.save(MEDIA/(name+'_saved_dynamics.png'))
 duration=100 if name=='lammps' else 500;durations=[duration]*len(frames);durations[-1]=1200
 out=MEDIA/(name+'_saved_dynamics.gif');frames[0].save(out,save_all=True,append_images=frames[1:],duration=durations,loop=0,optimize=True,disposal=1)
 print(json.dumps({'asset':out.name,'frames':len(frames),'bytes':out.stat().st_size,'cpu_seconds':time.process_time()-start,'peak_rss_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'max_displacement':float(maxu)}))
if __name__=='__main__':
 for name,cfg in json.loads((DATA/'saved_dynamics_provenance.json').read_text()).items():render(name,cfg)
