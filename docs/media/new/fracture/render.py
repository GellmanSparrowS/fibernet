"""Render saved FiberNet observations; no solver imports or physics execution.
Run: python render.py (requires Pillow). Input/output paths are relative to this file.
"""
from pathlib import Path
import json, math, hashlib, html
from PIL import Image, ImageDraw, ImageFont
P=Path(__file__).resolve().parent
D=json.loads((P/'replay.json').read_text()); G=D['geometry']; R=D['result']; O=R['observations']
W,H=960,1358
BG='#091320'; PANEL='#101f30'; INK='#e7f1f7'; MUTED='#90a8bd'; GRID='#22364a'; CYAN='#54ddce'; RED='#ff655e'; ORANGE='#ffc16c'
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'; BOLD='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
class Canvas:
 def __init__(self):
  self.im=Image.new('RGB',(W,H),BG);self.d=ImageDraw.Draw(self.im);self.svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><rect width="960" height="1358" fill="{BG}"/>']
 def line(self,p,c=GRID,w=1):
  self.d.line(p,fill=c,width=w);self.svg.append(f'<polyline points="'+ ' '.join(f'{x:.2f},{y:.2f}' for x,y in p)+f'" fill="none" stroke="{c}" stroke-width="{w}"/>')
 def rect(self,b,c):
  self.d.rectangle(b,fill=c);x,y,u,v=b;self.svg.append(f'<rect x="{x}" y="{y}" width="{u-x}" height="{v-y}" fill="{c}"/>')
 def text(self,x,y,s,size=18,c=INK,b=False):
  self.d.text((x,y),s,font=ImageFont.truetype(BOLD if b else FONT,size),fill=c);self.svg.append(f'<text x="{x}" y="{y+size*.94}" font-family="DejaVu Sans,sans-serif" font-size="{size}" font-weight="{700 if b else 400}" fill="{c}">{html.escape(s)}</text>')
 def dot(self,x,y,r,c):
  self.d.ellipse((x-r,y-r,x+r,y+r),fill=c);self.svg.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{c}"/>')
 def save_svg(self,p):p.write_text('\n'.join(self.svg)+ '</svg>')
def frame(i):
 o=O[i];mask=o['mask']; prev=O[max(0,i-1)]['mask'];new=[e[0] for j,e in enumerate(G['original_edges']) if prev[j] and not mask[j]];lost=sum(G['initial_mask'])-sum(mask)
 c=Canvas();c.text(40,26,'FIBERNET  /  FRACTURE',16,CYAN,True);c.text(40,58,'When the load path breaks',36,INK,True)
 c.text(40,108,'Quasistatic member removal / reference geometry',20,MUTED)
 c.rect((40,151,920,225),PANEL)
 for x,label,val in [(60,'IMPOSED DISPLACEMENT',f"{o['displacement_m']*1000:.4f} mm"),(375,'REMOVED / INITIAL NOTCH',f'{lost} / 6 members'),(690,'OBSERVATION',f'{i+1:02d} / {len(O):02d}')]:
  c.text(x,161,label,12,MUTED,True);c.text(x,183,val,24,INK,True)
 # Actual geometry in mm, no deformation or interpolated poses.
 xy={int(k):(160+v[0]*8000,760-v[1]*8000) for k,v in G['positions'].items()}
 for k in range(9):
  x=160+80*k;c.line([(x,280),(x,760)],GRID);c.text(x-8,782,str(k*10),12,MUTED)
 for k in range(7):
  y=760-80*k;c.line([(160,y),(800,y)],GRID);c.text(120,y-8,str(k*10),12,MUTED)
 c.text(404,795,'REFERENCE x [mm]',13,MUTED);c.text(42,248,'y [mm]',13,MUTED)
 for j,(eid,a,b) in enumerate(G['original_edges']):
  pa,pb=xy[a],xy[b]
  if mask[j]:c.line([pa,pb],CYAN,2)
  else:
   col=RED if eid in new else ('#624651' if G['initial_mask'][j] else '#465569')
   for t in range(0,10,2):
    c.line([(pa[0]+(pb[0]-pa[0])*t/10,pa[1]+(pb[1]-pa[1])*t/10),(pa[0]+(pb[0]-pa[0])*(t+1)/10,pa[1]+(pb[1]-pa[1])*(t+1)/10)],col,4 if eid in new else 2)
 for nid,(x,y) in xy.items():c.dot(x,y,3,CYAN)
 for nid in G['loaded']:
  x,y=xy[nid];c.line([(x,y-8),(x,y-33)],ORANGE,2);c.line([(x-4,y-27),(x,y-33),(x+4,y-27)],ORANGE,2)
 for nid in G['fixed']:
  x,y=xy[nid];c.line([(x-7,y+9),(x+7,y+9)],MUTED,2)
 c.text(555,230,'Top row: prescribed +y',14,ORANGE);c.text(650,795,'Fixed bottom row',12,MUTED)
 for j,(eid,a,b) in enumerate(G['original_edges']):
  if eid in new:
   x=(xy[a][0]+xy[b][0])/2;y=(xy[a][1]+xy[b][1])/2;c.rect((x-6,y-19,x+55,y+2),BG);c.text(x-3,y-18,f'#{eid}',13,RED,True)
 c.text(40,829,'INTACT',12,CYAN,True);c.text(143,829,'NEW REMOVAL',12,RED,True);c.text(312,829,'DASHED: removed / initial notch',12,MUTED)
 phase=o['phase'].replace('_',' ');status='Material path lost' if not o['path_present'] else phase.capitalize();c.text(40,856,status,19,ORANGE,True)
 c.text(490,858,'New IDs: '+(', '.join(map(str,new)) if new else 'none'),15,MUTED)
 # Trace preserves observation ordering, including avalanches at repeated displacement.
 for x0,key,factor,ymax,title,col in [(83,'loaded_reaction_force_N',1,4,'Reaction [N]',CYAN),(543,'strain_energy_J',1e6,80,'Elastic energy [µJ]',ORANGE)]:
  y0=1081;pw=330;ph=135;c.text(x0,906,title,17,INK,True)
  for t in range(5):
   yy=y0-ph*t/4;c.line([(x0,yy),(x0+pw,yy)],GRID);c.text(x0-36,yy-8,f'{ymax*t/4:g}',12,MUTED)
  for t in range(4):
   xx=x0+pw*t/3;c.text(xx-12,y0+8,f'{.12*t/3:.2f}',12,MUTED)
  c.line([(x0,y0-ph),(x0,y0),(x0+pw,y0)],MUTED)
  pts=[(x0+z['displacement_m']*1000/.12*pw,y0-z[key]*factor/ymax*ph) for z in O[:i+1]]
  if len(pts)>1:c.line(pts,col,3)
  c.dot(*pts[-1],5,col);c.text(x0+27,1112,'Imposed displacement [mm]',13,MUTED)
 c.text(40,1153,'Synthetic notched frame · 63 nodes · 158 beams · linear-static Timoshenko model',13,MUTED)
 c.text(40,1175,'Quasistatic load-step sequence • elastic strain energy',12,MUTED)
 c.rect((40,1224,920,1317),PANEL)
 c.text(60,1239,'MODEL SNAPSHOT',12,CYAN,True)
 c.text(60,1263,'E = 1 GPa   |   radius = 0.5 mm   |   strength = 1 MPa',17,INK)
 c.text(60,1291,'Synthetic strengths; no material calibration, recoil or spatial separation.',13,MUTED)
 return c
if __name__=='__main__':
 frames=[frame(i).im for i in range(len(O))]
 frames[0].save(P/'fracture_sequence.gif',save_all=True,append_images=frames[1:],duration=[900]+[300]*(len(O)-2)+[1800],loop=0,optimize=True,disposal=2)
 frame(30).save_svg(P/'fracture_poster.svg')
 for i in [0,2,16,30]:
  with Image.open(P/'fracture_sequence.gif') as im:im.seek(i);im.convert('RGB').save(P/f'keyframe_{i:02d}.png')
 print('frames',len(frames),'gif_bytes',(P/'fracture_sequence.gif').stat().st_size)
