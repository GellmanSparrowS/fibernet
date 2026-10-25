"""Direct/lazy API smoke, also runnable against an isolated installed wheel."""
import importlib
import importlib.abc
import json
import subprocess
import sys
import unittest

class InstalledAPI(unittest.TestCase):
    def test_lazy_exports_without_optional_provider_import(self):
        code = '''
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'exudyn','taichi','torch','lammps','pyvista','matplotlib'}:
            raise AssertionError('optional provider import attempted: '+fullname)
sys.meta_path.insert(0,Block())
import fibernet.sim as sim
assert 'fibernet.sim.archived_ge' not in sys.modules
from fibernet.sim import ArchivedGEProjection, import_archived_ge_state
from fibernet.sim.archived_ge import ArchivedGEProjection as Direct
assert ArchivedGEProjection is Direct
assert callable(import_archived_ge_state)
assert {'ArchivedGEProjection','import_archived_ge_state'} <= set(dir(sim))
assert not any(n.split('.')[0] in {'exudyn','taichi','torch','lammps'} for n in sys.modules)
'''
        completed=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True,timeout=20)
        self.assertEqual(completed.returncode,0,completed.stdout+completed.stderr)

    def test_documented_namespaced_synthetic_call(self):
        from fibernet.sim import import_archived_ge_state
        from _archived_ge_fixtures import fixture,arguments
        graph,source,header,state=fixture()
        result=import_archived_ge_state(graph,**arguments(source,header,state))
        self.assertEqual(result.translations()['node_ids'],[11,91])
        self.assertEqual(result.sidecar()['edge_mapping']['77']['native_endpoint_physical_ids'],[11,91])
