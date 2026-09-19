"""Create reproducible source/EXE submission archives with a SHA256 manifest.

python release_tools/package.py --exe path/to/FiberScope.exe --output path/to/deliverables
"""
from pathlib import Path
import argparse,hashlib,json,zipfile,re

class ReleasePackage:
    def build(self,executable,output):
        root=Path(__file__).resolve().parents[1];output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
        executable=Path(executable).resolve()
        paths=[p for p in root.rglob('*') if p.is_file() and not any(part in ('.git','__pycache__','_cache','_tmp','dist','build','build_vendor') for part in p.relative_to(root).parts) and p.suffix not in ('.pyc','.spec') and p.name not in ('smoke_result.txt','build_version.txt')]
        patterns=[rb'\bghp_[A-Za-z0-9]{30,}',rb'\bgithub_pat_[A-Za-z0-9_]{30,}',rb'\bms-[0-9a-f]{8}-[0-9a-f-]{27,}',rb'\bsk-[A-Za-z0-9_-]{25,}',rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----']
        for p in paths:
            if any(re.search(pattern,p.read_bytes()) for pattern in patterns):raise ValueError('Sensitive content: '+str(p.relative_to(root)))
        manifest={}
        dest=output/'AI4R_OPEN_ml-bio_代码材料.zip';temporary=dest.with_suffix('.zip.tmp')
        guide='''FiberScope 3.0 · GOAI 代码材料

1. Windows 10/11 x64：解压后运行 APP/FiberScope.exe，无需 Python。
2. 源码：source/FiberScope；FiberNet 核心：source/fibernet。
3. Python 3.10，在 source 目录运行：
   python -m pip install -r FiberScope/requirements.txt
   python FiberScope/run.py
4. 测试：python FiberScope/scripts/run_tests.py
5. 网页体验源码：source/web_demo；操作指令见 source/FiberScope/docs/finals/AI_RECORDING_3_0.md。
6. 在线体验：https://modelscope.cn/studios/GellmanSparrow/FiberScope
7. 开源仓库：https://github.com/GellmanSparrowS/fibernet

联网 AI 服务需评委自行配置；核心算法可离线运行。无 API 令牌、用户配置、原始参考资料或开发缓存。APP 为验收过的原始3.0 EXE；源码另外兼容公开仓库目录布局，不改变数值算法。
制作：复旦大学高分子科学系 杨云浩。致谢世界人工智能开源大赛。
'''
        with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            entries=[('APP/FiberScope.exe',executable)]+[('source/'+p.relative_to(root).as_posix(),p) for p in paths]
            for name,p in entries:
                z.write(p,name);manifest[name]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
            z.writestr('请先阅读.txt',guide);z.writestr('MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
        with zipfile.ZipFile(temporary) as z:
            if z.testzip() is not None:raise ValueError('ZIP integrity failure')
            for name,info in manifest.items():
                if hashlib.sha256(z.read(name)).hexdigest()!=info['sha256']:raise ValueError('Manifest mismatch')
        temporary.replace(dest)
        # The download asset includes the matching executable, source and all notices.
        asset=output/'FiberScope-3.0.0-Windows-x64.zip'
        import shutil
        shutil.copy2(dest,asset)
        digest=hashlib.sha256(dest.read_bytes()).hexdigest()
        (output/'SHA256SUMS.txt').write_text(f'{digest}  {dest.name}\n{digest}  {asset.name}\n'+hashlib.sha256(executable.read_bytes()).hexdigest()+'  APP/FiberScope.exe\n',encoding='utf-8')
        print(f'[package] {len(manifest)} files; {dest.stat().st_size} bytes; integrity and SHA256 PASS')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--exe',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();ReleasePackage().build(args.exe,args.output)
