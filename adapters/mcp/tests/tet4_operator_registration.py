"""Strict, descriptor-anchored source admission without optional Meshio."""
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from fibernet_mcp.safety import ContractError,canonical,open_directory,sha
from fibernet_mcp.server import TREE
from fibernet_mcp.tet4_operator import load_operator_sources
from fibernet_mcp.tet4_contract import load_fixture,MAX_SOURCE

raw,digest,_=load_fixture('simple')
ITEM=dict(name='binding_test',sha256=digest)
def refused(parent,entries,builtins=()):
    fd=open_directory(parent)
    try:
        try:load_operator_sources(fd,entries,builtins)
        except (OSError,ContractError):return
    finally:os.close(fd)
    raise AssertionError('unexpected operator source admission')

def main():
    with TemporaryDirectory(prefix='fibernet-operator-admission-') as area:
        base=Path(area); outside=base/'outside'; outside.mkdir()
        other=outside/'binding_test.tet4.json';other.write_bytes(raw)
        parent=base/'manifest-root';parent.mkdir()
        sources=parent/'tet4-sources';target=sources/'binding_test.tet4.json'
        sources.symlink_to(outside,target_is_directory=True)
        refused(parent,[ITEM]); sources.unlink();sources.mkdir()
        target.symlink_to(other);refused(parent,[ITEM]);target.unlink()
        os.link(other,target);refused(parent,[ITEM]);target.unlink()
        os.mkfifo(target);refused(parent,[ITEM]);target.unlink()
        target.mkdir();refused(parent,[ITEM]);target.rmdir()
        with target.open('wb') as f:f.truncate(MAX_SOURCE+1)
        refused(parent,[ITEM]);target.write_bytes(raw)
        for entries in ([],[ITEM,ITEM],[ITEM|{'name':'../binding_test'}],
                [ITEM|{'path':'binding_test.tet4.json'}],[ITEM|{'sha256':'0'*64}],
                [ITEM|{'name':False}],[ITEM|{'name':'ñ'}]):
            refused(parent,entries)
        refused(parent,[ITEM],builtins=['binding_test'])
        # Directory pathname replacement cannot swap the admitted anchor.
        fd=open_directory(parent)
        try:
            previous=base/'renamed-original';parent.rename(previous);parent.mkdir()
            (parent/'tet4-sources').mkdir()
            (parent/'tet4-sources'/target.name).write_bytes(b'mismatched alternate root')
            row,=load_operator_sources(fd,[ITEM],[])
            assert row[:3]==(ITEM['name'],raw,digest)
        finally:os.close(fd)
        parent=previous
        clean=parent/'output'; clean.mkdir()
        sentinel=clean/'keep.txt';sentinel.write_text('keep')
        graph=parent/'graph.json';graph.write_bytes(canonical(dict(version='3.1',dimension=3,tolerance=1e-6,
             nodes=[dict(id=0,position=[0,0,0])],edges=[])))
        conf=dict(source_tree=TREE,graphs=[dict(path=str(graph),sha256=sha(graph.read_bytes()),
                  length_unit='unspecified',frame_id='unknown')],tet4_sources=[ITEM])
        manifest=parent/'operator.json'
        for overrides in ({'tet4_fixtures':['simple','welded']},{'tet4_sources':[ITEM|{'filename':'x'}]},
                          {'tet4_sources':[ITEM|{'sha256':'f'*64}]}):
            manifest.write_bytes(canonical(conf|overrides))
            p=subprocess.run([sys.executable,'-B','-m','fibernet_mcp.server','--manifest',str(manifest),
                              '--output-root',str(clean)],input=b'',capture_output=True,timeout=8)
            assert p.returncode==2 and p.stdout==b'' and set(clean.iterdir())=={sentinel} and sentinel.read_text()=='keep'
    print('operator fixture confinement, anchored-directory rename and actual rejected startup cleanup passed')
if __name__=='__main__':main()
