"""Render all archived thermal frames, without importing FiberNet or running solvers.

python render.py --root .
Requires NumPy and Pillow. Output uses an A4 portrait aspect ratio. Temperature
on edges is the mean of recorded endpoint temperatures, a display mapping only.
The original 81 samples are retained in order with no temporal interpolation.
"""
from pathlib import Path
import argparse, hashlib, json, html, resource, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
W,H=960,1358
BG=(7,17,30);FG=(224,238,246);MUTED=(144,167,186);CYAN=(67,218,220);GOLD=(255,195,101);GRID=(39,58,76)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def color(v):
    stops=np.array([[47,83,145],[40,177,198],[153,224,180],[255,194,90]],float)
    z=float(np.clip(v,0,1))*3;i=min(int(z),2)
    return tuple(int(v) for v in np.rint(stops[i]*(1-(z-i))+stops[i+1]*(z-i)))
def main():
    started=time.process_time();ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);ap.add_argument('--output',type=Path);args=ap.parse_args();root=args.root
    source=root/'data';out=args.output or root;out.mkdir(parents=True,exist_ok=True)
    a=np.load(source/'thermal.npz',allow_pickle=False);summary=json.loads((source/'summary.json').read_text());ids=a['node_ids'];idx={int(v):k for k,v in enumerate(ids)};pairs=np.array([[idx[int(u)],idx[int(v)]] for u,v in a['endpoints_by_id']]);pos=a['positions'];t=a['times_s'];T=a['temperature_K'];cap=a['capacity_J_per_K']
    assert len(ids)==len(set(ids))==4340 and len(pairs)==5073 and len(set(a['edge_ids']))==5073
    assert T.shape==(81,4340) and np.array_equal(t,np.arange(81)*.25) and np.isfinite(T).all()
    assert np.all(pos[:,2]==0) and np.all(cap>0) and not a['interval_diagnostics_available'][0] and a['interval_diagnostics_available'][1:].all()
    assert np.isnan(a['component_energy_residual_J'][0]).all()
    # These are derived displays of saved state and saved ledger, not time stepping.
    stored=(T-T[0])@cap;sourceE=t*float(a['source_W'].sum());ledger=summary['independent_energy'];assert len(ledger)==80
    sink=np.r_[0.,-np.cumsum([x['reservoir_J'] for x in ledger])]
    assert np.allclose(stored,np.r_[0.,np.cumsum([x['stored_J'] for x in ledger])],rtol=1e-10,atol=1e-15)
    assert np.max(np.abs(sourceE-stored-sink))<1e-14
    low=float(T.min());high=float(T.max());xy=(pos[:,:2]-pos[:,:2].min(0));xy=xy/max(np.ptp(pos[:,:2],axis=0))*840;xy[:,0]+=60;xy[:,1]=1042-xy[:,1]
    font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';fonts={s:ImageFont.truetype(font,s) for s in (16,18,20,22,28,34)}
    palette=Image.new('P',(1,1));cols=[BG,FG,MUTED,CYAN,GOLD,GRID]+[color(i/15) for i in range(16)];palette.putpalette([q for c in cols for q in c]+[0]*(768-len(cols)*3))
    frames=[];svg=[];peak=T.max(axis=1)
    for f in range(len(t)):
        im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im);poster=f==len(t)-1
        if poster:svg=['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="1358" viewBox="0 0 960 1358">','<rect width="960" height="1358" fill="#07111e"/>']
        def line(points,c=GRID,width=1):
            d.line(points,fill=c,width=width)
            if poster:svg.append('<polyline points="'+' '.join(f'{x:.2f},{y:.2f}' for x,y in points)+'" fill="none" stroke="rgb'+str(c)+'" stroke-width="'+str(width)+'"/>')
        def txt(x,y,s,size=18,c=FG):
            d.text((x,y),s,font=fonts[size],fill=c)
            if poster:svg.append(f'<text x="{x}" y="{y+size}" font-family="DejaVu Sans,sans-serif" font-size="{size}" fill="rgb{c}">{html.escape(s)}</text>')
        txt(42,25,'FIBERNET  /  TRANSIENT TRANSPORT',18,CYAN)
        txt(42,57,'Joule heat through a complete network',34)
        txt(42,105,'4,340 junctions  /  5,073 edges  /  all 81 recorded states',20,MUTED)
        line([(42,146),(918,146)])
        txt(42,159,f'TIME   {t[f]:05.2f} s',28);txt(355,165,f'PEAK   {peak[f]:.4f} K',22,GOLD);txt(680,165,'INPUT   16.083 uW',20,CYAN)
        norm=(T[f]-low)/(high-low);edgevalues=norm[pairs].mean(axis=1)
        for (u,v),val in zip(pairs,edgevalues):line([tuple(xy[u]),tuple(xy[v])],color(val),1)
        # No position displacement or camera motion: changes are saved temperature.
        txt(60,1048,'x / y in metres  |  full extent approximately 0.020 x 0.020 m',16,MUTED)
        txt(42,1085,'Temperature [K]',18)
        for x in range(400):line([(236+x,1089),(236+x,1104)],color(x/399),1)
        txt(236,1109,f'{low:.3f}',16,MUTED);txt(572,1109,f'{high:.3f}',16,MUTED)
        txt(42,1143,'ENERGY [uJ]',18);txt(260,1143,f'Input {sourceE[f]*1e6:.1f}',16,CYAN);txt(440,1143,f'Stored {stored[f]*1e6:.1f}',16,GOLD);txt(640,1143,f'Reservoir {sink[f]*1e6:.1f}',16,FG)
        chart=(70,1180,898,1260);x0,y0,x1,y1=chart;maxE=sourceE[-1]*1e6
        line([(x0,y0),(x0,y1),(x1,y1)]);txt(8,1177,f'{maxE:.0f}',16,MUTED)
        for values,c in ((sourceE,CYAN),(stored,GOLD),(sink,FG)):
            pts=[(x0+(x1-x0)*t[j]/t[-1],y1-(y1-y0)*values[j]*1e6/maxE) for j in range(f+1)]
            if len(pts)>1:line(pts,c,2)
        txt(42,1265,'0',16,MUTED);txt(852,1265,'20 s',16,MUTED)
        txt(42,1292,'Synthetic one-way Joule heating • fixed-geometry thermal network',18)
        txt(42,1321,'Recorded dt = 0.25 s. Replay: 2.5x speed. No interpolated states.',16,MUTED)
        frames.append(im.quantize(palette=palette,dither=Image.Dither.NONE))
        if poster:
            svg.append('</svg>');(out/'transient_thermal_network_final.svg').write_text('\n'.join(svg)+'\n')
    gif=out/'transient_thermal_network.gif';frames[0].save(gif,save_all=True,append_images=frames[1:],duration=100,loop=0,optimize=True,disposal=1)
    m={'schema':'fibernet-archived-transient-thermal-media-1','renderer':'render.py','renderer_sha256':sha(Path(__file__)),'sources':{str(p.relative_to(root)):{'sha256':sha(p)} for p in sorted(source.iterdir())},'gif_sha256':sha(gif),'svg_sha256':sha(out/'transient_thermal_network_final.svg'),'width':W,'height':H,'layout':'A4 portrait aspect ratio','frames':81,'frame_duration_ms':100,'recorded_time_s':[float(t[0]),float(t[-1])],'recorded_dt_s':.25,'playback_speed':2.5,'synthetic':True,'nodes':4340,'edges':5073,'geometry_unit':'m','temperature_unit':'K','energy_unit':'J','peak_temperature_K':float(high),'source_power_W':float(a['source_W'].sum()),'display_encoding':{'edge_color':'mean of saved endpoint temperatures, fixed global scale','geometry':'all archived positions, unchanged','energy':'cumulative saved ledger and capacity-weighted recorded temperature change'},'coupling':'one-way electrical Joule heat to constant lumped thermal model','mechanical_deformation':False,'temperature_palette_levels':16,'time_interpolation':False,'scope':'Saved synthetic thermal scenario, not experimental material calibration or general electro-thermo-mechanical capability. See archived campaign analysis for timestep comparison; algebraic residual acceptance alone is not time accuracy.','initial_diagnostics':'Unavailable NaN initial interval retained; energy displays start at zero accumulated change.','verification':['all 81 original states in original order','unique stable node/edge IDs and endpoint resolution','finite temperatures and positive capacities','fixed planar original geometry','energy ledger matches capacity-weighted temperature change','cumulative energy closure below 1e-14 J'],'final_energy_J':{'input':float(sourceE[-1]),'stored':float(stored[-1]),'to_reservoir':float(sink[-1])}}
    (out/'transient_thermal_network.json').write_text(json.dumps(m,indent=2)+'\n')
    print(json.dumps({'cpu_s':time.process_time()-started,'maxrss_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'gif_bytes':gif.stat().st_size,'max_energy_closure_J':float(np.max(np.abs(sourceE-stored-sink))),'frames':len(frames)}))
if __name__=='__main__':main()
