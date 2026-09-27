"""Prepare a reviewed source snapshot; never copy caches, credentials or Git history.

Usage: python scripts/prepare_publication.py --destination PATH
The destination must already contain the selected upstream FiberNet source.
"""
from pathlib import Path
import argparse,shutil,json,hashlib,re

class Publication:
    def prepare(self,destination):
        source=Path(__file__).resolve().parents[1]
        root=Path(destination).resolve(); app=root/'FiberScope'
        if not (root/'fibernet/core').is_dir(): raise ValueError('upstream core missing')
        app.mkdir(parents=True,exist_ok=True)
        selected=[]
        for folder in ('fslab','studio','tests','scripts'):
            selected.extend((source/folder).glob('*.py'))
        selected=[p for p in selected if p.name not in ('wrap_reference27.py','prepare_reference26.py','validate_recording28.py','validate_final30.py','validate_precision29.py')]
        selected.extend((source/'assets').rglob('*'))
        selected.extend(source.glob('requirements*.txt'))
        selected.extend(source.glob('run.py'))
        selected.extend((source/'docs').glob('METHODS*.md'))
        selected.extend(source/p for p in ['docs/AI_TOOLS.md','docs/open_source/API.md','docs/finals/AI_RECORDING_3_0.md'])
        for path in selected:
            if not path.is_file() or '__pycache__' in path.parts: continue
            dest=app/path.relative_to(source);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
        shutil.copy2(root/'LICENSE',app/'LICENSE')
        # The source bridge resolves both the sibling checkout and monorepo.
        (app/'.gitignore').write_text('_cache/\n_tmp/\n__pycache__/\n*.pyc\nbuild/\nbuild_vendor/\ndist/\n*.spec\nsmoke_result.txt\nbuild_version.txt\n',encoding='utf-8')
        self.scan(root)
        print('[publication] staged application source and notices')

    def scan(self,root):
        patterns=[rb'ghp_[A-Za-z0-9]{30,}',rb'github_pat_[A-Za-z0-9_]{30,}',rb'ms-[0-9a-f]{8}-[0-9a-f-]{27,}',rb'sk-[A-Za-z0-9_-]{25,}',rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----']
        problems=[]
        for p in Path(root).rglob('*'):
            if not p.is_file() or '.git' in p.parts or '__pycache__' in p.parts: continue
            if p.suffix.lower() in ('.exe','.dll','.pyd','.png','.ico'): continue
            if any(re.search(rb'\b'+pattern,p.read_bytes()) for pattern in patterns):problems.append(str(p.relative_to(root)))
        if problems: raise ValueError('sensitive patterns in files: '+', '.join(problems))
        print('[publication] secret-pattern scan PASS')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--destination',required=True)
    Publication().prepare(parser.parse_args().destination)
