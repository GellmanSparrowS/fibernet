"""Render bounded interface tours from versioned FiberScope screenshots.

Run from any directory: python scripts/render_module_workflow_tours.py
Requires Pillow. No computation or application interaction is replayed.
"""
from pathlib import Path
import hashlib
import base64
import json
import math
from html import escape
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/media'
INPUT = ROOT / 'FiberScope/docs/screenshots'
COMMIT = 'c5f52d2805e3fe03e8cfb2109e295e3131e8e289'
EXPECTED = {
 'learning-en.png': 'f4ea7c9e15692dda87e2bacd1d8c88ddb1ab4613',
 'inverse-en.png': '92daef6f1aa2757f74807958ef695a460a90782e',
 'surface-en.png': 'cb2045f20aadf907201bf78c0994bf49a329d0f5',
 'manufacturing-en.png': '0da23fa08928bfcaef3fde36dcd5af216570e86d',
}
W, H = 960, 830
BG, INK, MUTED, LINE = '#0b1423', '#eff6ff', '#a5b6ca', '#283b52'
FONT = 'DejaVuSans.ttf'
def font(size, bold=False):
 return ImageFont.truetype('DejaVuSans-Bold.ttf' if bold else FONT, size)
def digest(data):
 return hashlib.sha256(data).hexdigest()
def git_blob(data):
 return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()

TOURS = [
 dict(name='learning_inverse_workflow', title='Learn from networks. Explore designs.', accent='#55d9ee',
  tabs=['Dataset & training', 'Diagnostics', 'Inverse search', 'Candidate review'],
  files=['learning-en.png','learning-en.png','inverse-en.png','inverse-en.png'],
  boxes=[(5,110,338,750),(345,110,1590,980),(5,110,325,980),(335,110,1590,980)],
  notes=['Inspect the saved dataset and training configuration.',
         'Read the saved loss and predicted-versus-actual panels.',
         'Inspect the saved inverse-search setup and evaluation log.',
         'Compare the saved candidate, target curve and convergence panel.'],
  footer='Saved diagnostics, not a live training run or a performance benchmark.',
  en='Animated tour of saved FiberScope learning and inverse-design interfaces. Only emphasis changes; plots and results remain the original screenshot data.',
  zh='FiberScope 学习与逆向设计的已保存界面导览。仅高亮提示发生变化，图表和结果保留原始截图内容；并非实时训练或性能基准。'),
 dict(name='surface_export_workflow', title='Map a surface. Prepare the geometry.', accent='#ffbf69',
  tabs=['Surface network', 'Surface controls', 'Fabrication model', 'Export options'],
  files=['surface-en.png','surface-en.png','manufacturing-en.png','manufacturing-en.png'],
  boxes=[(295,115,1585,980),(5,115,290,850),(350,150,1590,880),(5,915,1590,995)],
  notes=['Inspect the saved network mapped onto a three-dimensional surface.',
         'Locate the surface settings and fiber-geometry export controls.',
         'Inspect the saved fabrication geometry in the manufacturing view.',
         'Locate STL, 3MF and route CSV options in the saved interface.'],
  footer='Geometry and export-interface tour; no printer execution is shown.',
  en='Animated tour of saved FiberScope surface and manufacturing interfaces. Shows surface geometry and export controls, not a physical printer run or a solid-mechanics solution.',
  zh='FiberScope 曲面与制造的已保存界面导览。展示曲面几何和导出选项，不代表实际打印执行或三维实体力学求解。')
]

def frame(tour, stage, pulse, pictures):
 im=Image.new('RGB',(W,H),BG); d=ImageDraw.Draw(im); accent=tour['accent']
 d.text((24,17),'FIBERSCOPE  /  WORKFLOW ATLAS',font=font(12,True),fill=accent)
 d.text((24,40),tour['title'],font=font(28,True),fill=INK)
 d.text((24,82),'Interface tour · saved views',font=font(15),fill=MUTED)
 for i,label in enumerate(tour['tabs']):
  x=24+i*230
  d.rounded_rectangle((x,111,x+220,146),radius=7,fill='#173044' if i==stage else '#101e30',outline=accent if i==stage else LINE)
  d.text((x+10,120),f'{i+1:02}  {label}',font=font(13,True),fill=INK if i==stage else MUTED)
 im.paste(pictures[tour['files'][stage]],(24,163))
 box=tour['boxes'][stage]
 coords=tuple(round(v*0.57+(24 if j%2==0 else 163)) for j,v in enumerate(box))
 # Only the annotation pulses; every scientific pixel remains the saved view.
 a=0.7+0.3*math.sin(math.pi*pulse/7)
 rgb=tuple(round(int(accent[i:i+2],16)*a) for i in (1,3,5))
 d.rounded_rectangle(coords,radius=6,outline=rgb,width=3)
 d.rectangle((24,786,936,787),fill=LINE)
 d.text((24,751),tour['notes'][stage],font=font(17),fill=INK)
 d.text((24,800),tour['footer'],font=font(13),fill=MUTED)
 return im

def poster(tour):
 # Embed the verified original PNG so the editable SVG is self-contained.
 image_data = (INPUT / tour['files'][0]).read_bytes()
 if git_blob(image_data) != EXPECTED[tour['files'][0]]:
  raise ValueError('Poster screenshot changed; review provenance before rendering.')
 image_uri = 'data:image/png;base64,' + base64.b64encode(image_data).decode('ascii')
 s=[f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="830" viewBox="0 0 960 830">',
 '<rect width="960" height="830" fill="#0b1423"/>',
 '<g font-family="DejaVu Sans, sans-serif">',
 f'<text x="24" y="30" font-size="12" fill="{tour["accent"]}">FIBERSCOPE / WORKFLOW ATLAS</text>',
 f'<text x="24" y="66" font-size="28" font-weight="bold" fill="#eff6ff">{escape(tour["title"])}</text>',
 '<text x="24" y="98" font-size="15" fill="#a5b6ca">Interface tour · saved views</text>']
 for i,label in enumerate(tour['tabs']):
  x=24+i*230
  s += [f'<rect x="{x}" y="111" width="220" height="35" rx="7" fill="#101e30" stroke="#283b52"/>',f'<text x="{x+10}" y="134" font-size="13" fill="#eff6ff">{i+1:02} {escape(label)}</text>']
 s += [f'<image href="{image_uri}" x="24" y="163" width="912" height="570"/>',
 '<rect x="24" y="741" width="912" height="46" fill="#0b1423"/>',
 f'<text x="24" y="773" font-size="17" fill="#eff6ff">{escape(tour["notes"][0])}</text>',
 f'<text x="24" y="817" font-size="13" fill="#a5b6ca">{escape(tour["footer"])}</text>', '</g></svg>']
 return '\n'.join(s)+'\n'

def main():
 OUT.mkdir(parents=True,exist_ok=True); pictures={}; inputs=[]
 for name,expected in EXPECTED.items():
  data=(INPUT/name).read_bytes()
  if git_blob(data)!=expected: raise ValueError(f'Screenshot changed: {name}; review provenance before rendering.')
  with Image.open(INPUT/name) as im:
   if im.size != (1600,1000): raise ValueError(f'Unexpected dimensions: {name}')
   pictures[name]=im.convert('RGB').resize((912,570),Image.Resampling.LANCZOS)
  inputs.append(dict(path=f'FiberScope/docs/screenshots/{name}',git_blob=expected,sha256=digest(data),source_url=f'https://raw.githubusercontent.com/GellmanSparrowS/fibernet/{COMMIT}/FiberScope/docs/screenshots/{name}'))
 outputs=[]
 for tour in TOURS:
  frames=[frame(tour,i//8,i%8,pictures) for i in range(32)]
  # One palette prevents color flicker; no dithering keeps UI text crisp.
  sheet=Image.new('RGB',(W,H*4))
  for i in range(4): sheet.paste(frames[i*8],(0,H*i))
  palette=sheet.quantize(colors=256,method=Image.Quantize.MEDIANCUT)
  frames=[f.quantize(palette=palette,dither=Image.Dither.NONE) for f in frames]
  path=OUT/(tour['name']+'.gif')
  frames[0].save(path,save_all=True,append_images=frames[1:],duration=300,loop=0,optimize=True,disposal=1)
  (OUT/(tour['name']+'.svg')).write_text(poster(tour),encoding='utf-8')
  with Image.open(path) as gif:
   duration=sum((gif.seek(i) or gif.info['duration']) for i in range(gif.n_frames))
   outputs.append(dict(file=path.name,sha256=digest(path.read_bytes()),bytes=path.stat().st_size,width=W,height=H,frames=gif.n_frames,duration_ms=duration,caption_en=tour['en'],caption_zh=tour['zh']))
 manifest=dict(schema_version=1,source_commit=COMMIT,kind='saved_interface_tour',renderer='scripts/render_module_workflow_tours.py',inputs=inputs,outputs=outputs,poster_images='Embedded byte-for-byte source PNGs; self-contained editable SVG compositions.',animation='Only stage selection and annotation emphasis change. No scientific data are interpolated or generated.',scope='Screenshots are existing repository assets. These tours do not add physical validation, training validation, or fabrication validation.')
 (OUT/'module_workflow_tours.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(outputs,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
