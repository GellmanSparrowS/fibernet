"""Compose saved screenshots and genuine detail crops; no application execution."""
from pathlib import Path
import argparse,base64,hashlib,html,json
from PIL import Image,ImageDraw,ImageFont
W,H=960,640;BG='#081422';FG='#edf3fa';FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def blob(b):return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def main(root,out):
 out.mkdir(parents=True,exist_ok=True);inputs=root/'FiberScope/docs/screenshots';original=json.loads((root/'docs/media/module_workflow_tours.json').read_text());known={Path(x['path']).name:x for x in original['inputs']}
 names=['learning-en.png','inverse-en.png','surface-en.png','manufacturing-en.png'];pics={};hashes={}
 for name in names:
  p=inputs/name;raw=p.read_bytes();assert digest(p)==known[name]['sha256'] and blob(raw)==known[name]['git_blob'];pics[name]=Image.open(p).convert('RGB');assert pics[name].size==(1600,1000);hashes[str(p.relative_to(root))]=digest(p)
 stages=[
 ('Learning','learning-en.png',None),
 ('Learning controls','learning-en.png',[(14,146,328,293)]),
 ('Inverse design','inverse-en.png',None),
 ('Inverse-design controls','inverse-en.png',[(10,179,326,378)]),
 ('Surface mapping','surface-en.png',None),
 ('Surface controls','surface-en.png',[(12,345,288,520)]),
 ('Fabrication','manufacturing-en.png',None),
 ('Export controls','manufacturing-en.png',[(18,950,325,983),(332,950,639,983),(646,950,954,983)])]
 frames=[];records=[];lastsvg=None
 for index,(title,name,boxes) in enumerate(stages):
  im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im);label='Saved UI · '+title;d.text((24,15),label,font=ImageFont.truetype(FONT,36),fill=FG)
  svg=[f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><title>{html.escape(label)}</title>',f'<rect width="{W}" height="{H}" fill="{BG}"/>',f'<text x="24" y="49" font-family="DejaVu Sans" font-size="36" fill="{FG}">{html.escape(label)}</text>']
  b64=base64.b64encode((inputs/name).read_bytes()).decode();svg.append(f'<defs><image id="source" x="0" y="0" width="1600" height="1000" xlink:href="data:image/png;base64,{b64}"/></defs>')
  boxes=boxes or [(0,0,1600,1000)];placements=[]
  if len(boxes)==1:
   box=boxes[0];cw,ch=box[2]-box[0],box[3]-box[1];scale=min(928/cw,548/ch);tw,th=round(cw*scale),round(ch*scale);placements=[(box,(W-tw)//2,74+(548-th)//2,tw,th)]
  else:
   for i,box in enumerate(boxes):
    cw,ch=box[2]-box[0],box[3]-box[1];scale=2.8;tw,th=round(cw*scale),round(ch*scale);placements.append((box,(W-tw)//2,150+i*130,tw,th))
  record={'title':title,'source':str((inputs/name).relative_to(root)),'duration_ms':1500 if index%2==0 else 2200,'kind':'whole_screen_orientation' if index%2==0 else 'saved_ui_detail','placements':[]}
  for j,(box,x,y,tw,th) in enumerate(placements):
   crop=pics[name].crop(box);assert 0<=box[0]<box[2]<=1600 and 0<=box[1]<box[3]<=1000
   original_crop_hash=hashlib.sha256(crop.tobytes()).hexdigest();im.paste(crop.resize((tw,th),Image.Resampling.LANCZOS),(x,y))
   sx=tw/(box[2]-box[0]);sy=th/(box[3]-box[1]);assert abs(sx-sy)<.02
   svg.append(f'<defs><clipPath id="clip{j}"><rect x="{x}" y="{y}" width="{tw}" height="{th}"/></clipPath></defs><g clip-path="url(#clip{j})"><use xlink:href="#source" transform="translate({x-box[0]*sx},{y-box[1]*sy}) scale({sx},{sy})"/></g>')
   record['placements'].append({'source_box':list(box),'target_box':[x,y,x+tw,y+th],'source_crop_rgb_sha256':original_crop_hash})
  svg.append('</svg>');lastsvg='\n'.join(svg);frames.append(im);records.append(record)
  im.save(out/f'frame_{index:02d}.png')
  for w in [800,322]:im.resize((w,round(H*w/W)),Image.Resampling.LANCZOS).save(out/f'frame_{index:02d}_{w}.png')
 sheet=Image.new('RGB',(W,H*len(frames)))
 for i,f in enumerate(frames):sheet.paste(f,(0,i*H))
 pal=sheet.quantize(colors=256,method=Image.Quantize.MEDIANCUT);q=[f.quantize(palette=pal,dither=Image.Dither.NONE) for f in frames]
 target=out/'fiberscope_overview.gif';q[0].save(target,save_all=True,append_images=q[1:],duration=[r['duration_ms'] for r in records],loop=0,optimize=False,disposal=2)
 (out/'fiberscope_overview.svg').write_text(lastsvg)
 with Image.open(target) as gif:
  assert gif.n_frames==8 and gif.info['loop']==0
  assert [(gif.seek(i) or gif.info['duration']) for i in range(8)]==[r['duration_ms'] for r in records]
 assert hashes=={str((inputs/n).relative_to(root)):digest(inputs/n) for n in names}
 receipt={'kind':'saved_interface_tour','canvas':[W,H],'input_sha256':hashes,'stages':records,'whole_screen_text_is_orientation_not_mobile_reading':True,'scientific_plots_cropped':False,'original_sources_unchanged':True,'resampling':'Lanczos resizing and GIF palette quantization; no redraw of source UI contents','gif_sha256':digest(target),'renderer_sha256':digest(__file__),'duration_ms':sum(r['duration_ms'] for r in records),'no_application_training_physics_execution':True,'visual_qa':'PENDING human inspection'}
 (out/'render_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({k:v for k,v in receipt.items() if k!='stages'},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();main(a.source_root.resolve(),a.output_dir.resolve())
