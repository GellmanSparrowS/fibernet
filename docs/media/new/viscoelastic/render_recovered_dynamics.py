"""Render recorded FiberNet generalized-Maxwell dynamics; never runs a solver."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

def main():
 p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 raw=a.source.read_bytes();r=json.loads(raw);source_hash=r.get('source_result_sha256',hashlib.sha256(raw).hexdigest());v=r['native_axial_viscoelastic'];s=v['samples'];q=v['sampled_native_state'];n=v['node_ids'];ids={k:i for i,k in enumerate(n)}
 x0=np.array(v['initial_positions']);x=np.array([z['positions'] for z in q]);vel=np.array([z['velocities'] for z in q]);t=np.array([z['time'] for z in q]);stress=np.array([z['stress_Pa'] for z in s]);edges=np.array([[ids[i],ids[j]] for i,j in v['edge_nodes']]);fixed=np.array([ids[i] for i in v['fixed_node_ids']]);mass=np.array(v['masses_kg'])
 kinetic=np.array([z['kinetic_energy_J'] for z in s]);recalc=.5*np.sum(mass[None,:,None]*vel**2,axis=(1,2));stored=np.array([sum(z['stored_energy_J']) for z in s]);loss=np.array([sum(z['cumulative']['physical_dissipation_J']) for z in s]);be=np.array([sum(z['cumulative']['be_numerical_dissipation_J']) for z in s]);u=np.linalg.norm(x-x0,axis=2)
 assert x.shape==(129,105,3) and edges.shape==(273,2)
 assert all(z['step']==w['step'] and z['time']==w['time'] for z,w in zip(s,q))
 assert np.allclose(kinetic,recalc,rtol=2e-14,atol=0)
 assert np.max(np.abs(x[:,fixed]-x0[fixed]))==0
 assert np.isfinite(x).all() and np.isfinite(stress).all()
 np.savez_compressed(a.output/'viscoelastic_network_recorded.npz',initial_positions_m=x0,positions_m=x,velocities_m_per_s=vel,time_s=t,stress_Pa=stress,edge_indices=edges,node_ids=n,edge_ids=v['edge_ids'],fixed_indices=fixed,kinetic_J=kinetic,stored_J=stored,physical_loss_J=loss,BE_loss_J=be)
 compact={k:v[k] for k in ['node_ids','edge_ids','edge_nodes','initial_positions','fixed_node_ids','masses_kg','dt','material','units']}
 compact['samples']=[{k:z[k] for k in ['step','time','stress_Pa','stored_energy_J','kinetic_energy_J','cumulative']} for z in s]
 compact['sampled_native_state']=[{k:z[k] for k in ['step','time','positions','velocities']} for z in q]
 (a.output/'viscoelastic_network_recorded.json').write_text(json.dumps({'source_result_sha256':source_hash,'native_axial_viscoelastic':compact},separators=(',',':'),allow_nan=False)+'\n')
 font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';bold='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
 def f(z,b=False):return ImageFont.truetype(bold if b else font,z)
 bg='#07141e';white='#e9f3f4';muted='#93afb9';cyan='#63ded3';gold='#f8c679';pink='#ed8c98';blue='#81b5f4'
 W,H=800,1132;mag=3000.;bound=np.max(np.abs(stress));idx=list(range(0,129,2));frames=[]
 scale=420/.013
 def coords(points):return np.column_stack((96+points[:,0]*scale,680-points[:,1]*scale))
 origin=coords(x0)
 def color(z):
  base=np.array([95,132,147.]);end=np.array([249,162,105.]) if z>=0 else np.array([78,208,234.]);return tuple((base+(end-base)*min(abs(z)/bound,1)).astype(int))
 for k in idx:
  im=Image.new('RGB',(W,H),bg);d=ImageDraw.Draw(im)
  d.text((42,30),'FIBERNET  /  MATERIAL MEMORY',font=f(15,True),fill=cyan)
  d.text((40,63),'Relaxation in motion',font=f(37,True),fill=white)
  d.text((42,116),'Recorded generalized-Maxwell network dynamics',font=f(18),fill=muted)
  d.line((42,157,758,157),fill='#27434e',width=1)
  d.text((42,178),'105 NODES  /  273 FIBERS  /  39 FIXED SUPPORTS',font=f(15,True),fill=white)
  d.text((42,202),f't = {t[k]:.4f} s       Deformation magnified {mag:,.0f}x',font=f(17),fill=gold)
  pts=coords(x0+mag*(x[k]-x0))
  for i,j in edges:d.line((*origin[i],*origin[j]),fill='#203540',width=1)
  for e,(i,j) in enumerate(edges):d.line((*pts[i],*pts[j]),fill=color(stress[k,e]),width=2)
  for i,point in enumerate(pts):
   xx,yy=point;d.ellipse((xx-2,yy-2,xx+2,yy+2),fill=white)
  for i in fixed:
   xx,yy=pts[i];d.rectangle((xx-3,yy-3,xx+3,yy+3),fill=bg,outline=cyan,width=1)
  d.text((566,289),'ACTUAL FIELD',font=f(12,True),fill=muted)
  d.text((566,313),'Axial stress',font=f(17,True),fill=white)
  d.text((566,341),'Pa',font=f(16),fill=muted)
  for j in range(160):d.line((571,381+j,590,381+j),fill=color(bound*(1-2*j/159)),width=1)
  for yy,txt in [(375,f'+{bound:.1f}'),(452,'0'),(532,f'-{bound:.1f}')]:d.text((603,yy),txt,font=f(14),fill=muted)
  d.text((566,570),'Max |u|',font=f(15),fill=muted);d.text((566,594),f'{u[k].max()*1e6:.3f} um',font=f(22,True),fill=white)
  d.text((42,701),'ENERGY TRANSFER',font=f(15,True),fill=cyan)
  d.text((42,729),'Initial motion becomes material dissipation',font=f(20,True),fill=white)
  chart=(64,793,734,964);xx0,y0,xx1,y1=chart
  d.line((xx0,y0,xx0,y1,xx1,y1),fill='#3d5865',width=1)
  series=[(kinetic,blue,'Kinetic'),(stored,gold,'Stored'),(loss,cyan,'Material loss'),(be,pink,'BE numerical loss')]
  for j,(values,c,label) in enumerate(series):
   d.text((64+(j%2)*340,756+(j//2)*20),label,font=f(13),fill=c)
   points=[(xx0+tt/.5*(xx1-xx0),y1-val/kinetic[0]*(y1-y0)) for tt,val in zip(t,values)]
   d.line(points,fill=tuple(int(c.lstrip('#')[i:i+2],16)//3 for i in (0,2,4)),width=1)
   if k>0:d.line(points[:k+1],fill=c,width=3)
  cursor=xx0+t[k]/.5*(xx1-xx0);d.line((cursor,y0,cursor,y1),fill='#d6e4e8',width=1)
  d.text((xx0,971),'0',font=f(12),fill=muted);d.text((xx1-40,971),'0.5 s',font=f(12),fill=muted)
  d.text((42,1001),'Energy normalized by initial kinetic energy. Gray network: rest geometry.',font=f(13),fill=muted)
  d.text((42,1030),'Synthetic SI parameters • small-strain axial fibers • no bending or contact',font=f(13),fill=white)
  d.text((42,1054),'Recorded response. Visual replay slowed ~13x.',font=f(13),fill=muted)
  d.text((42,1081),'Einf 100 kPa  |  branches 200 kPa / 0.02 s + 100 kPa / 0.1 s',font=f(13),fill=muted)
  frames.append(im)
 frames[0].save(a.output/'viscoelastic_network_dynamics.gif',save_all=True,append_images=frames[1:],duration=100,loop=0,optimize=True)
 select=[0,8,24,64];mont=Image.new('RGB',(800,1132),bg)
 for cell,k in enumerate(select):mont.paste(frames[k].resize((400,566)),((cell%2)*400,(cell//2)*566))
 mont.save(a.output/'viscoelastic_network_contact_sheet.png')
 report={'source_result_sha256':source_hash,'source_result_name':a.source.name,'render_only':True,'synthetic_parameters':True,'nodes':105,'edges':273,'fixed_nodes':39,'source_frames':129,'gif_frames':65,'source_time_s':[0,.5],'dt_s':v['dt'],'solver_steps':4096,'magnification':mag,'kinetic_energy_recompute_max_abs_error_J':float(np.max(np.abs(kinetic-recalc))),'max_displacement_m':float(u.max()),'physical_loss_fraction_final':float(loss[-1]/kinetic[0]),'BE_loss_fraction_final':float(be[-1]/kinetic[0]),'physical_energy_defect_fraction_final':float((kinetic[-1]+stored[-1]+loss[-1]-kinetic[0])/kinetic[0]),'assumptions':['small-strain axial generalized Maxwell','synthetic lumped masses','zero external force and numerical drag','initial-velocity excitation','no bending/contact/adhesion'],'source_material':v['material'],'units':v['units'],'playback':'Uniform sampled frames, 100 ms display each, no interpolated physics'}
 (a.output/'viscoelastic_network_dynamics.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
