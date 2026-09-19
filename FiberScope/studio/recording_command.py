"""Parse explicit recording requests into deterministic workflow parameters."""
import re
from fslab.structure import all_unit_keys, unit_display


def recording_options(text):
    options=dict(samples=300,iterations=200,curved=True,open_slicer=True,target='J',
                 stretch=2.,surface_model='demo_pyramid.obj',create_demo=True,
                 demo_unit='square',diameter=2.,print_size=250.)
    aliases={'方形':'square','正方形':'square','六边形':'hexagon','圆环':'ring','三角形':'triangle'}
    aliases.update({unit_display(key,'zh'):key for key in all_unit_keys()})
    for name,key in sorted(aliases.items(),key=lambda item:-len(item[0])):
        if name in text:
            options['demo_unit']=key
            break
    for key,pattern,cast in (
        ('samples',r'(?:生成|准备)\s*(\d+)\s*个',int),
        ('iterations',r'(\d+)\s*次(?:预算|评估|迭代|搜索)',int),
        ('stretch',r'拉伸比\s*(\d+(?:\.\d+)?)',float),
        ('diameter',r'纤维直径\s*(\d+(?:\.\d+)?)\s*(?:毫米|mm)',float),
        ('print_size',r'适配\s*(\d+(?:\.\d+)?)\s*(?:毫米|mm)',float)):
        match=re.search(pattern,text,re.I)
        if match: options[key]=cast(match.group(1))
    target=re.search(r'以\s*([A-Za-z])\s*型?曲线为目标',text,re.I)
    if target: options['target']=target.group(1).upper()
    return options
