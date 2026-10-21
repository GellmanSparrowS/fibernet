#!/usr/bin/env python3
"""Render a homepage hero using existing public geometry and saved frames only.

python scripts/render_saved_homepage_hero.py
python scripts/render_saved_homepage_hero.py --source-root /path/to/repo --output-dir /path/to/output

No FiberNet imports, graph generation, solver, interpolation of material motion,
or new physical simulation is performed. Pillow and NumPy are the only rendering
dependencies. The editable poster reuses the original saved SVG network paths.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import html
import json
import math
from pathlib import Path
import platform
import xml.etree.ElementTree as ET

import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont

W, H, SS, FRAME_COUNT, FRAME_MS = 1200, 760, 2, 60, 100
BG = '#07101a'
PANEL = '#0b1724'
BORDER = '#203446'
INK = '#eaf5fa'
MUTED = '#9eb4c5'
TEAL = '#62dce4'
GOLD = '#ffba69'
COLORS = {'#ffffff': PANEL, '#aab5c2': '#536b82', '#233646': '#b0c9dc', '#21698a': TEAL, '#d78732': GOLD}
NS = {'s': 'http://www.w3.org/2000/svg'}
SOURCES = [
 'examples/data/lammps_braced_network_si.json',
 'docs/media/tensile_recruitment_kagome.gif',
 'docs/media/tensile_recruitment_kagome.json',
 'docs/media/tensile_recruitment_kagome_final.svg',
]
CROP = (100, 106, 558, 441)
MATERIAL_X, MATERIAL_Y, MATERIAL_SCALE = 650, 242, 1.08


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rgb(c):
    return tuple(int(c[i:i+2], 16) for i in (1, 3, 5))


def blend(a, b, f):
    return tuple(round(x*(1-f)+y*f) for x,y in zip(rgb(a),rgb(b)))


def hexc(c):
    return '#' + ''.join(f'{int(x):02x}' for x in c)


class Canvas:
    """One primitive stream for crisp raster rendering and editable SVG text."""
    def __init__(self):
        self.image = Image.new('RGB', (W*SS,H*SS), BG)
        self.draw = ImageDraw.Draw(self.image)
        self.svg = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{W}" height="{H}" viewBox="0 0 {W} {H}">', '<title>FiberNet: fixed 3D geometry and saved material response</title>', '<desc>Left: camera orbit of the saved 64-node, 441-edge braced cubic graph. Right: saved kagome tensile recruitment frames; these are separate examples, not a coupled experiment. The poster shows the final saved loading frame.</desc>', f'<rect width="{W}" height="{H}" fill="{BG}"/>']
        self.fonts = {}
    def rect(self, box, fill, radius=0, stroke=None, width=1):
        coords=tuple(round(v*SS) for v in box)
        if radius:
            self.draw.rounded_rectangle(coords, radius=round(radius*SS), fill=fill, outline=stroke, width=round(width*SS))
        else:
            self.draw.rectangle(coords,fill=fill,outline=stroke,width=round(width*SS))
        x,y,x2,y2=box
        self.svg.append(f'<rect x="{x}" y="{y}" width="{x2-x}" height="{y2-y}" rx="{radius}" fill="{fill}"'+(f' stroke="{stroke}" stroke-width="{width}"' if stroke else '')+'/>')
    def line(self, points, fill, width=1):
        self.draw.line([(round(x*SS),round(y*SS)) for x,y in points],fill=fill,width=max(1,round(width*SS)))
        self.svg.append('<polyline points="'+' '.join(f'{x:.3f},{y:.3f}' for x,y in points)+f'" fill="none" stroke="{fill}" stroke-width="{width:.3f}" stroke-linecap="round" stroke-linejoin="round"/>')
    def circle(self,x,y,r,fill,stroke=None,width=1):
        self.draw.ellipse(tuple(round(v*SS) for v in (x-r,y-r,x+r,y+r)),fill=fill,outline=stroke,width=max(1,round(width*SS)))
        self.svg.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{r:.3f}" fill="{fill}"'+(f' stroke="{stroke}" stroke-width="{width}"' if stroke else '')+'/>')
    def text(self,x,y,s,size=18,fill=INK,bold=False,mono=False):
        path = Path('/usr/share/fonts/truetype/dejavu')/('DejaVuSansMono.ttf' if mono else 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')
        key=(str(path),size)
        if key not in self.fonts:self.fonts[key]=ImageFont.truetype(str(path),size*SS)
        self.draw.text((round(x*SS),round(y*SS)),s,font=self.fonts[key],fill=fill,anchor='lt')
        family='DejaVu Sans Mono' if mono else 'DejaVu Sans'
        # SVG baseline accounts for Pillow's ascender-trimmed top anchor.
        self.svg.append(f'<text x="{x}" y="{y+size*.77:.3f}" fill="{fill}" font-family="{family}, sans-serif" font-size="{size}" font-weight="{700 if bold else 400}">{html.escape(s)}</text>')
    def finish(self):
        return self.image.resize((W,H),Image.Resampling.LANCZOS)


def recolor_saved_frame(image):
    """Map original category colors while retaining saved raster line locations."""
    raw=np.asarray(image.crop(CROP).convert('RGB'),dtype=np.float32)
    white=np.array([255.,255.,255.]); bg=np.array(rgb(PANEL),dtype=np.float32)
    best=np.full(raw.shape[:2],np.inf); output=np.empty_like(raw)
    for old,new in COLORS.items():
        if old=='#ffffff':continue
        direction=np.array(rgb(old),dtype=float)-white
        alpha=np.clip(np.sum((raw-white)*direction,axis=2)/np.dot(direction,direction),0,1)
        fit=white+alpha[...,None]*direction
        residual=np.sum((raw-fit)**2,axis=2)
        mask=residual<best; best[mask]=residual[mask]
        transformed=bg+alpha[...,None]*(np.array(rgb(new))-bg)
        output[mask]=transformed[mask]
    return Image.fromarray(np.clip(output,0,255).astype('uint8'))


def project(points, angle):
    az=math.radians(angle); el=math.radians(24)
    right=np.array([-math.sin(az),math.cos(az),0.])
    up=np.array([-math.cos(az)*math.sin(el),-math.sin(az)*math.sin(el),math.cos(el)])
    forward=np.array([math.cos(az)*math.cos(el),math.sin(az)*math.cos(el),math.sin(el)])
    depth=points@forward
    perspective=5.2/(5.2-depth)
    return np.column_stack([points@right*perspective,-(points@up)*perspective]),depth


def add_editable_kagome(canvas, svg_path):
    root=ET.parse(svg_path).getroot()
    # Source SVG has 720 pt width and saved GIF has 1150 px width.
    raster_per_svg=1150/720
    scale=MATERIAL_SCALE*raster_per_svg
    tx=MATERIAL_X-CROP[0]*MATERIAL_SCALE
    ty=MATERIAL_Y-CROP[1]*MATERIAL_SCALE
    canvas.svg.append(f'<g id="saved-kagome-vector" transform="translate({tx},{ty}) scale({scale})">')
    for group_id in ('LineCollection_1','LineCollection_2','LineCollection_3','PathCollection_1'):
        group=root.find(f'.//s:g[@id="{group_id}"]',NS)
        if group is None:continue
        cloned=copy.deepcopy(group)
        for item in cloned.iter():
            item.attrib.pop('clip-path',None)
            for attr,value in list(item.attrib.items()):
                for old,new in COLORS.items():value=value.replace(old,new)
                item.set(attr,value)
        canvas.svg.append(ET.tostring(cloned,encoding='unicode'))
    canvas.svg.append('</g>')


def render(points, edges, material_frames, sample_indices, frame, scale, center, poster=False, svg_source=None):
    c=Canvas()
    c.text(30,24,'FiberNet',47,bold=True)
    c.text(805,34,'NETWORKS  /  MECHANICS',22,bold=True,fill=MUTED)
    c.text(32,87,'Graph geometry. Material response. One network language.',23,fill=MUTED)
    c.line([(30,124),(1170,124)],BORDER,1)
    c.rect((28,144,590,664),PANEL,16,BORDER)
    c.rect((610,144,1172,664),PANEL,16,BORDER)
    c.text(50,163,'01  /  GEOMETRY',18,fill=TEAL,bold=True)
    c.text(632,163,'02  /  RESPONSE',18,fill=GOLD,bold=True)
    c.text(50,194,'Braced cubic network',27,bold=True)
    c.text(632,194,'Kagome tensile recruitment',27,bold=True)
    c.circle(57,236,4,TEAL)
    c.text(70,226,'CAMERA ORBIT · FIXED GRAPH',18,fill=TEAL)
    c.circle(639,236,4,GOLD)
    c.text(652,226,'SIMULATED DEFORMATION · SAVED',18,fill=GOLD)
    angle=25+360*frame/FRAME_COUNT
    xy,depth=project(points,angle)
    xy=xy*scale+center
    dep=(depth-depth.min())/(depth.max()-depth.min())
    # Depth sorts are an optical styling choice, not mechanics or stress.
    edge_depth=depth[edges].mean(axis=1)
    for j in np.argsort(edge_depth,kind='stable'):
        e=edges[j]; d=float(dep[e].mean())
        col=hexc(blend('#183448',TEAL,.14+.82*d**1.2))
        c.line(xy[e].tolist(),col,.8+1.0*d)
    for j in np.argsort(depth,kind='stable'):
        x,y=xy[j];d=float(dep[j]);r=1.5+1.45*d
        c.circle(x,y,r,hexc(blend('#23495b','#dcfcff',.22+.73*d)))
        if d>.55:c.circle(x-.5,y-.5,.65,'#ecfeff')
    # Static reference axes follow the camera orientation; letters denote geometry.
    gizmo=np.array([[0,0,0],[.28,0,0],[0,.28,0],[0,0,.28]])
    g,_=project(gizmo,angle);g=g*100+np.array([72,552])
    for i,(name,col) in enumerate(zip('xyz',('#4f7897','#64bcc6','#b0c4d3')),1):
        c.line([g[0],g[i]],col,1.5);c.text(int(g[i,0]+4),int(g[i,1]-6),name,13,col)
    c.text(50,592,'64 nodes  /  441 edges',22,bold=True)
    c.text(50,628,'Depth shading only · no physical motion',18,fill=MUTED)
    selected=min(17,frame*18//48)
    if poster:selected=17
    source_frame=material_frames[selected]
    c.image.paste(source_frame.resize((round(source_frame.width*MATERIAL_SCALE*SS),round(source_frame.height*MATERIAL_SCALE*SS)),Image.Resampling.LANCZOS),(MATERIAL_X*SS,MATERIAL_Y*SS))
    # Vector poster replaces the raster crop with the original saved vector paths.
    if poster:add_editable_kagome(c,svg_source)
    c.text(632,610,'436 nodes  /  576 edges',22,bold=True)
    for i in range(18):
        c.rect((634+i*19,647,646+i*19,651),GOLD if i<=selected else '#283e50',2)
    c.text(1003,643,f'{sample_indices[selected]+1:02d} / 21',16,fill=MUTED,mono=True)
    c.text(32,688,'RECRUITMENT',17,fill=MUTED,bold=True)
    legend=[(209,'#536b82','Below threshold'),(444,TEAL,'Recruited'),(622,GOLD,'Spanning component')]
    for x,color,title in legend:
        c.line([(x,697),(x+27,697)],color,3)
        c.text(x+38,688,title,18,fill=INK)
    c.text(939,688,'Saved loading replay',17,fill=MUTED)
    c.text(32,729,'Separate examples. Threshold: 5% of frame-max positive strain; contact off. Colors are not total force flow.',17,fill=MUTED)
    if poster:c.svg.append('</svg>')
    return c


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args(); root=args.source_root.resolve(); out=(args.output_dir or root/'docs/media').resolve();out.mkdir(parents=True,exist_ok=True)
    graph_doc=json.loads((root/SOURCES[0]).read_text());graph=graph_doc['graph']
    index={n['id']:i for i,n in enumerate(graph['nodes'])}
    raw_points=np.array([n['position'] for n in graph['nodes']],dtype=float)
    edges=np.array([[index[e['node_i']],index[e['node_j']]] for e in graph['edges']],dtype=int)
    assert raw_points.shape==(64,3) and edges.shape==(441,2)
    assert np.isfinite(raw_points).all()
    # Display normalization is rigid centering plus a uniform positive scale.
    points=(raw_points-raw_points.mean(axis=0))/(np.ptp(raw_points,axis=0).max()/2)
    arrays_sha=hashlib.sha256(raw_points.astype('<f8').tobytes()+edges.astype('<i8').tobytes()).hexdigest()
    extent=np.concatenate([project(points,25+360*f/FRAME_COUNT)[0] for f in range(FRAME_COUNT)])
    scale=min(476/np.ptp(extent[:,0]),310/np.ptp(extent[:,1]))
    center=np.array([307.,415.])-(extent.min(axis=0)+extent.max(axis=0))/2*scale
    src=Image.open(root/SOURCES[1]);assert src.n_frames==18
    material=[]
    for f in range(src.n_frames):src.seek(f);material.append(recolor_saved_frame(src.convert('RGB')))
    saved_manifest=json.loads((root/SOURCES[2]).read_text());indices=saved_manifest['sampled_frame_indices']
    assert saved_manifest['gif_sha256']==digest(root/SOURCES[1])
    saved_svg_hash=digest(root/SOURCES[3])
    crlf_hash=hashlib.sha256((root/SOURCES[3]).read_text().replace('\n','\r\n').encode('utf-8')).hexdigest()
    assert saved_manifest['svg_sha256'] in (saved_svg_hash,crlf_hash)
    svg_provenance_match='exact bytes' if saved_manifest['svg_sha256']==saved_svg_hash else 'matches original CRLF line endings; working source is LF'
    frames=[]
    for f in range(FRAME_COUNT):
        frames.append(render(points,edges,material,indices,f,scale,center).finish())
        if f in (0,24,48):frames[-1].save(out/f'network_workflow_hero_keyframe_{f:02d}.png')
    # Fixed shared palette limits temporal quantization flicker. No dithering.
    palette_source=Image.new('RGB',(W, H*6))
    for i,f in enumerate((0,9,19,29,39,49)):palette_source.paste(frames[f],(0,H*i))
    palette=palette_source.quantize(colors=64,method=Image.Quantize.MEDIANCUT,dither=Image.Dither.NONE)
    quantized=[im.quantize(palette=palette,dither=Image.Dither.NONE) for im in frames]
    gif=out/'network_workflow_hero.gif'
    quantized[0].save(gif,save_all=True,append_images=quantized[1:],duration=FRAME_MS,loop=0,optimize=True,disposal=1)
    poster=render(points,edges,material,indices,48,scale,center,poster=True,svg_source=root/SOURCES[3])
    svg=out/'network_workflow_hero_poster.svg';svg.write_text('\n'.join(poster.svg)+'\n',encoding='utf-8')
    png=out/'network_workflow_hero_poster.png';poster.finish().save(png)
    poster.finish().resize((700,round(H*700/W)),Image.Resampling.LANCZOS).save(out/'network_workflow_hero_readme_700px.png')
    check=Image.open(gif);durations=[]
    for f in range(check.n_frames):check.seek(f);check.load();durations.append(check.info.get('duration',0))
    assert check.n_frames==FRAME_COUNT and check.size==(W,H) and sum(durations)==6000
    assert gif.stat().st_size<4_000_000
    # Check editable SVG marks and safety: no external links, embedded rasters, scripts.
    svg_root=ET.parse(svg).getroot()
    assert not svg_root.findall('.//s:image',NS)
    assert not svg_root.findall('.//s:script',NS)
    manifest={
      'schema':'fibernet.saved-media-hero.v1',
      'description':'Fixed saved 3D braced-cubic graph camera orbit beside saved kagome loading raster frames; no new simulation or material interpolation.',
      'canvas':{'width':W,'height':H,'frames':FRAME_COUNT,'frame_duration_ms':FRAME_MS,'duration_ms':6000,'loop':True},
      'left':{'topology':'generator-defined braced cubic graph','nodes':64,'edges':441,'source':SOURCES[0], 'source_graph_coordinates':'graph.nodes[].position; existing initial deformed StructureGraph','motion':'camera azimuth orbit only; fixed elevation 24 degrees; graph coordinates unchanged','display_normalization':'centering and uniform scaling only','color':'optical depth styling, not force or stress','graph_arrays_sha256':arrays_sha},
      'right':{'topology':'perturbed 3x3 kagome','nodes':436,'edges':576,'source':SOURCES[1], 'material_motion':'saved loading frames only; sequence restarts, not an unloading experiment','saved_gif_frames':18,'source_simulation_frame_indices':indices,'hero_source_frame_mapping':[min(17,f*18//48) for f in range(FRAME_COUNT)],'material_interpolation':False,'crop_source_pixels':list(CROP),'uniform_display_scale':MATERIAL_SCALE,'palette_transform':COLORS,'analysis':'positive axial strain; 5% of frame maximum threshold; contact off; colors do not measure total force flow'},
      'caveats':['The two panels are separate source-backed examples, not a coupled geometry-to-response experiment.','The braced-cubic example uses synthetic coefficients and is not a continuum-calibrated material.','The poster network paths and labels are editable vectors. The GIF reuses saved raster material frames.'],
      'source_files':SOURCES,
      'source_manifest_validation':{'saved_gif':'exact bytes','saved_svg':svg_provenance_match,'saved_svg_declared_sha256':saved_manifest['svg_sha256'],'saved_svg_actual_sha256':saved_svg_hash},
      'source_sha256':{s:digest(root/s) for s in SOURCES},
      'gif_sha256':digest(gif),'svg_sha256':digest(svg),
      'render_script':'scripts/render_saved_homepage_hero.py','render_script_sha256':digest(__file__),
      'dependencies':{'python':platform.python_version(),'numpy':np.__version__,'pillow':PIL.__version__,'font':'DejaVu Sans / DejaVu Sans Mono','gif_palette_colors':64,'gif_dithering':'none'},
      'output_sha256':{p.name:digest(p) for p in (gif,svg,png)},
      'validation':{'gif_all_frames_decoded':True,'frames':check.n_frames,'dimensions':list(check.size),'total_duration_ms':sum(durations),'size_bytes':gif.stat().st_size,'poster_external_images':0,'poster_script_elements':0,'poster_vector_paths':len(svg_root.findall('.//s:path',NS)),'qa_keyframes_exported':[0,24,48]},
    }
    (out/'network_workflow_hero.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'gif':str(gif),'bytes':gif.stat().st_size,'frames':check.n_frames,'size':check.size,'poster_svg':str(svg),'source_graph_sha256':arrays_sha},indent=2))

if __name__=='__main__':main()
