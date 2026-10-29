"""Render archived steady electrical fields; never run a physics solver.

Usage: python scripts/render_saved_electrical_field.py --root .
Requires NumPy and Pillow. Input geometry is in metres; all physical fields
remain unchanged across frames. Apparent movement is solely a camera orbit.
"""
from pathlib import Path
import argparse, hashlib, json, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W,H,N=960,640,40
BG=(7,17,30); FG=(224,238,246); MUTED=(140,165,185)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--output',type=Path);a=ap.parse_args()
    base=a.root/'tests/data/joule_torus/run01';out=a.output or a.root/'docs/media';out.mkdir(parents=True,exist_ok=True)
    gp,ep=base/'torus_graph.json',base/'torus_electrical.json';g=json.loads(gp.read_text());e=json.loads(ep.read_text())
    assert e['status']=='converged' and e['schema']=='fibernet-linear-conductance-1'
    assert e['provenance']['length_unit']=='m'
    assert e['provenance']['units']=={'conductance':'S','voltage':'V','current':'A','power':'W'}
    nodes={n['id']:n for n in g['nodes']};edges={x['id']:x for x in g['edges']};ids=e['node_ids'];ei=e['edge_ids'];idx={k:i for i,k in enumerate(ids)}
    assert len(nodes)==len(ids)==len(set(ids)) and set(nodes)==set(ids)
    assert len(edges)==len(ei)==len(set(ei)) and set(edges)==set(ei)
    for k,(u,v) in zip(ei,e['endpoints_by_id']): assert (edges[k]['node_i'],edges[k]['node_j'])==(u,v)
    pos=np.array([nodes[k]['position'] for k in ids]);v=np.array(e['voltage_V']);cur=np.array(e['current_A']);power=np.array(e['edge_power_W']);pairs=np.array([[idx[u],idx[w]] for u,w in e['endpoints_by_id']]);cond=np.array(e['conductance_S'])
    dv=v[pairs[:,0]]-v[pairs[:,1]]
    assert np.allclose(cur,cond*dv,rtol=1e-10,atol=1e-14)
    assert np.allclose(power,cur*dv,rtol=1e-10,atol=1e-14)
    assert math.isclose(float(power.sum()),e['dissipated_power_W'],rel_tol=1e-12)
    pos=pos-pos.mean(axis=0);scale=275/np.max(np.linalg.norm(pos,axis=1));vmin,vmax=float(v.min()),float(v.max());imax=float(abs(cur).max())
    def color(value):
        t=np.clip((value-vmin)/(vmax-vmin),0,1);stops=np.array([[45,96,165],[37,202,212],[255,204,103]],float);q=min(int(t*2),1);return tuple(int(x) for x in np.rint(stops[q]*(1-(t*2-q))+stops[q+1]*(t*2-q)).astype(int))
    font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    def ft(n):return ImageFont.truetype(font,n)
    frames=[];svg=[]
    for f in range(N):
        angle=2*np.pi*f/N+.35;el=.63
        rx=np.array([math.cos(angle),math.sin(angle),0]);ry=np.array([-math.sin(angle)*math.sin(el),math.cos(angle)*math.sin(el),math.cos(el)]);rz=np.cross(rx,ry)
        p=np.column_stack([480+scale*pos@rx,350-scale*pos@ry,scale*pos@rz])
        im=Image.new('RGB',(W*2,H*2),BG);d=ImageDraw.Draw(im)
        if f==0:svg=['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="640" viewBox="0 0 960 640">','<rect width="960" height="640" fill="#07111e"/>']
        def line(xy,fill,width=1):
            d.line(tuple(z*2 for z in xy),fill=fill,width=max(2,round(width*2)))
            if f==0:svg.append(f'<path d="M {xy[0]:.2f} {xy[1]:.2f} L {xy[2]:.2f} {xy[3]:.2f}" fill="none" stroke="rgb{fill}" stroke-width="{width:.2f}" stroke-linecap="round"/>')
        def txt(x,y,s,size=16,fill=FG):
            d.text((x*2,y*2),s,font=ft(size*2),fill=fill)
            if f==0:svg.append(f'<text x="{x}" y="{y+size}" font-family="DejaVu Sans,sans-serif" font-size="{size}" fill="rgb{fill}">{s.replace("&","&amp;")}</text>')
        txt(38,24,'FIBERNET  /  TRANSPORT',14,(63,211,217));txt(38,51,'Electrical transport, in three dimensions',29)
        txt(38,95,'Synthetic steady electrical field · camera orbit',17,MUTED)
        txt(38,137,'80 junctions',16);txt(211,137,'162 conductive edges',16);txt(530,137,f'{power.sum():.4f} W  total dissipation',16)
        for k in np.argsort(p[pairs].mean(axis=1)[:,2]):
            u,w=pairs[k];c=color((v[u]+v[w])/2);width=1.0+5.0*abs(cur[k])/imax
            # A straight edge carries a constant saved current. Color interpolates
            # nodal voltage for display only; it is not an added solved field.
            for j in range(8):
                t0=j/8;t1=(j+1)/8;aa=p[u,:2]*(1-t0)+p[w,:2]*t0;bb=p[u,:2]*(1-t1)+p[w,:2]*t1;cc=color(v[u]*(1-(t0+t1)/2)+v[w]*(t0+t1)/2)
                line((*aa,*bb),cc,width)
        for k in np.argsort(p[:,2]):
            x,y,_=p[k];r=3.5;c=color(v[k]);d.ellipse(((x-r)*2,(y-r)*2,(x+r)*2,(y+r)*2),fill=c,outline=BG,width=2)
            if f==0:svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="rgb{c}" stroke="#07111e"/>')
        line((38,543,922,543),(38,60,79),1)
        txt(38,556,'Voltage [V]',15)
        for x in range(230):line((38+x,586,38+x,598),color(vmin+(vmax-vmin)*x/229))
        txt(38,605,f'{vmin:.1f}',13,MUTED);txt(244,605,f'{vmax:.1f}',13,MUTED)
        txt(350,556,'Edge width: |current| [A]',15)
        line((352,591,416,591),(99,216,216),1);line((461,591,525,591),(99,216,216),6)
        txt(352,605,'0',13,MUTED);txt(461,605,f'{imax:.3f}',13,MUTED)
        txt(643,556,'Fixed field · moving viewpoint',15);txt(643,586,'No thermal or time-dependent solve',13,MUTED)
        im=im.resize((W,H),Image.Resampling.LANCZOS)
        if f==0: palette=im.quantize(colors=64,method=Image.Quantize.MEDIANCUT)
        frames.append(im.quantize(palette=palette,dither=Image.Dither.NONE))
        if f==0:svg.append('</svg>');(out/'electrical_torus_field_final.svg').write_text('\n'.join(svg));im.save(out/'electrical_torus_field_preview.png')
    gif=out/'electrical_torus_field.gif';frames[0].save(gif,save_all=True,append_images=frames[1:],duration=100,loop=0,optimize=True,disposal=1)
    manifest={'schema':'fibernet-saved-electrical-visualization-1','source_commit':'c5f52d2805e3fe03e8cfb2109e295e3131e8e289','sources':{str(p.relative_to(a.root)):{'sha256':sha(p)} for p in (gp,ep)},'solver_input_sha256_recorded':e['provenance']['input_sha256'],'input_hash_note':'Solver-recorded input hash retained verbatim; source file SHA256 verified separately.','renderer':'scripts/render_saved_electrical_field.py','renderer_sha256':sha(Path(__file__)),'gif_sha256':sha(gif),'svg_sha256':sha(out/'electrical_torus_field_final.svg'),'nodes':len(ids),'edges':len(ei),'node_ids':ids,'edge_ids':ei,'units':e['provenance']['units'],'geometry_unit':'m','frames':N,'frame_duration_ms':100,'width':W,'height':H,'synthetic':True,'field_varies_over_frames':False,'animation':'camera orbit only','coupled':False,'dissipated_power_W':float(power.sum()),'current_absolute_max_A':imax,'voltage_range_V':[vmin,vmax],'display_encoding':{'color':'nodal voltage; linear interpolation along straight edge for display','width':'1 + 5 * abs(current_A) / max_abs_current_A pixels'},'verification':['stable node and edge ID sets match graph','directed edge endpoints match graph','I = G * voltage_difference from saved arrays','P = I * voltage_difference from saved arrays','sum edge power agrees with stored dissipation'],'assumptions':e['provenance']['assumptions']}
    (out/'electrical_torus_field.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'gif_bytes':gif.stat().st_size,'power_W':float(power.sum()),'current_max_A':imax,'voltage_range':[vmin,vmax]}))
if __name__=='__main__':main()
