"""Reflow retained SVG paths only. No trajectories, solvers or simulation imports."""
from pathlib import Path
import argparse,copy,hashlib,json,re,subprocess
from lxml import etree as E
from PIL import Image
NS='http://www.w3.org/2000/svg';N={'s':NS};W,H=960,1100
S=lambda x:'{'+NS+'}'+x
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def text(parent,x,y,value,size=36,anchor='start',color='#122636',**kw):
 n=E.SubElement(parent,S('text'),x=str(x),y=str(y),fill=color,**{'font-family':'DejaVu Sans','font-size':str(size),'text-anchor':anchor,**kw});n.text=value;return n
def main(root,out):
 out.mkdir(parents=True,exist_ok=True);receipts=[]
 for name in ['kagome','ring']:
  source=root/f'docs/media/route_preserving_intervention_{name}_final.svg';before=sha(source);tree=E.parse(str(source));old=tree.getroot()
  svg=E.Element(S('svg'),nsmap={None:NS,'xlink':'http://www.w3.org/1999/xlink'},width=str(W),height=str(H),viewBox=f'0 0 {W} {H}')
  E.SubElement(svg,S('title')).text=f'Static {name} intervention results at stretch ratio 1.40'
  for defs in old.findall(S('defs')):svg.append(copy.deepcopy(defs))
  E.SubElement(svg,S('rect'),width=str(W),height=str(H),fill='white')
  text(svg,24,45,f'Static {name} · stretch 1.40',36)
  for x,label,color in [(24,'Below threshold','#b4bec8'),(370,'Recruited','#277b9a'),(630,'Grip-spanning','#dc8935')]:
   E.SubElement(svg,S('line'),x1=str(x),x2=str(x+28),y1='85',y2='85',stroke=color,**{'stroke-width':'6'});text(svg,x+38,96,label,32)
  preserved=[];panels=[]
  for index in range(4):
   net=old.xpath(f'.//s:g[@id="axes_{index+1}"]',namespaces=N)[0];curve=old.xpath(f'.//s:g[@id="axes_{index+5}"]',namespaces=N)[0]
   title,ratio=net.xpath('.//s:text/text()',namespaces=N);ratio=ratio.rsplit(' ',1)[1];col=index%2;row=index//2;origin=col*480;dy=row*470
   text(svg,origin+240,145+dy,title,34,'middle');text(svg,origin+240,185+dy,f'Final {ratio}',34,'middle')
   # Both network and curve transformations are uniform; path coordinates stay verbatim.
   nb=net.xpath('./s:g[starts-with(@id,"patch_")]/s:path/@d',namespaces=N)[0];nums=[float(x) for x in re.findall(r'[-+]?(?:\d*\.\d+|\d+)',nb)];xs=nums[0::2];ys=nums[1::2];nx,ny=min(xs),min(ys);nw,nh=max(xs)-nx,max(ys)-ny
   ns=200/nh;tx=origin+240-(nw*ns)/2-nx*ns;ty=200+dy-ny*ns
   cb=curve.xpath('./s:g[starts-with(@id,"patch_")]/s:path/@d',namespaces=N)[0];nums=[float(x) for x in re.findall(r'[-+]?(?:\d*\.\d+|\d+)',cb)];xs=nums[0::2];ys=nums[1::2];cx,cy=min(xs),min(ys);cw,ch=max(xs)-cx,max(ys)-cy
   cs=325/cw;qx=origin+100-cx*cs;qy=420+dy-cy*cs
   for group,scale,dx,ddy in [(net,ns,tx,ty),(curve,cs,qx,qy)]:
    groupcopy=copy.deepcopy(group);expected=group.xpath('.//s:path/@d',namespaces=N)
    for labelnode in groupcopy.xpath('.//s:text',namespaces=N):labelnode.getparent().remove(labelnode)
    assert groupcopy.xpath('.//s:path/@d',namespaces=N)==expected
    wrap=E.SubElement(svg,S('g'),transform=f'translate({dx:.12f} {ddy:.12f}) scale({scale:.12f})');wrap.append(groupcopy)
    preserved.append({'group':group.get('id'),'path_count':len(expected),'path_data_sha256':hashlib.sha256(json.dumps(expected).encode()).hexdigest(),'uniform_scale':scale,'translation':[dx,ddy]})
   # Retained tick values/locations, new readable type. No resampling of numeric curves.
   for labelnode in curve.xpath('.//s:g[starts-with(@id,"xtick_")]//s:text',namespaces=N):
    value=labelnode.text;text(svg,float(labelnode.get('x'))*cs+qx,420+dy+ch*cs+36,value,34,'middle')
   for tick in curve.xpath('.//s:g[starts-with(@id,"ytick_")]',namespaces=N):
    label=tick.xpath('.//s:text',namespaces=N)[0];tick_y=float(tick.xpath('.//s:use/@y',namespaces=N)[0]);text(svg,origin+88,tick_y*cs+qy+12,label.text,34,'end')
   panels.append({'strategy':title,'final_reaction_ratio_display':ratio,'original_network_group':net.get('id'),'original_curve_group':curve.get('id')})
  text(svg,27,605,'Raw reaction / original final',32,'middle',transform='rotate(-90 27 605)')
  text(svg,480,1075,'Stretch ratio',36,'middle')
  target=out/f'intervention_{name}_grid.svg';E.ElementTree(svg).write(str(target),encoding='utf-8',xml_declaration=True)
  # Audit scientific path strings independently in the finished SVG.
  final=E.parse(str(target))
  for rec in preserved:
   group=final.xpath(f'//s:g[@id="{rec["group"]}"]',namespaces=N)[0];assert hashlib.sha256(json.dumps(group.xpath('.//s:path/@d',namespaces=N)).encode()).hexdigest()==rec['path_data_sha256']
  png=out/f'intervention_{name}_grid.png';subprocess.run(['inkscape',str(target),'--export-type=png',f'--export-filename={png}','--export-width=960'],check=True,capture_output=True)
  with Image.open(png) as im:
   assert im.size==(W,H)
   for width in [800,322]:im.resize((width,round(H*width/W)),Image.Resampling.LANCZOS).save(out/f'intervention_{name}_grid_{width}.png')
  assert before==sha(source)
  receipts.append({'source':str(source.relative_to(root)),'source_sha256':before,'output':target.name,'output_sha256':sha(target),'panels':panels,'preserved_scientific_path_groups':preserved,'all_original_path_strings_preserved':True,'source_texts':old.xpath('.//s:text/text()',namespaces=N),'no_solver_import_or_execution':True,'static_not_trajectory':True,'visual_qa':'PENDING human inspection'})
 receipt={'canvas':[W,H],'renderer_sha256':sha(__file__),'derivatives':receipts};(out/'render_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({'rendered':[r['output'] for r in receipts],'path_groups':sum(len(r['preserved_scientific_path_groups']) for r in receipts),'no_solver_execution':True}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();main(a.source_root.resolve(),a.output_dir.resolve())
