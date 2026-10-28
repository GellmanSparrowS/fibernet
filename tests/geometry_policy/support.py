"""Provider-free AST support. No installed FiberNet module is imported.

Only chosen production definitions execute; named imports that would load the
package are removed and supplied as inert collaborators. This is source-level
control-flow coverage, not installed-package/provider integration.
"""
from __future__ import annotations
import ast
import builtins
import copy
import math
from pathlib import Path
import types

ROOT = Path(__file__).resolve().parents[2]
H = {}
_helper = ROOT / 'fibernet/sim/_taichi_geometry.py'
_helper_tree = ast.parse(_helper.read_text(encoding='utf-8'), filename=str(_helper))
if any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(_helper_tree)):
    raise AssertionError('The geometry-policy helper must remain import-free')
exec(compile(_helper_tree, str(_helper), 'exec'), H)
# The easy facade deliberately keeps these collaborators private so its
# historical wildcard exports remain unchanged.
H['_preflight_taichi_geometry'] = H['preflight_taichi_geometry']
H['_validate_geometry_policy'] = H['validate_geometry_policy']

class FakeBool: pass
class FakeInteger(int): pass
NP = types.SimpleNamespace(bool_=FakeBool, integer=FakeInteger, isfinite=math.isfinite)

PROVIDER_ROOTS = frozenset(('fibernet', 'numpy', 'taichi', 'networkx', 'scipy',
                          'gym', 'gymnasium', 'torch', 'tensorflow', 'jax',
                          'cupy', 'numba', 'lammps', 'exudyn', 'pybullet',
                          'pymunk', 'stable_baselines3', 'sklearn', 'pandas'))


def isolated_builtins():
    """Block package/provider imports only inside extracted production code."""
    original = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level or name.split('.', 1)[0] in PROVIDER_ROOTS:
            raise AssertionError('Unexpected package/provider import: ' + name)
        return original(name, globals, locals, fromlist, level)

    return dict(vars(builtins), __import__=guarded_import)

def graph(internal=False):
    # Nonmonotonic stable parallel edge IDs and actual-node segmentation are
    # distinguishable without loading graph/numeric providers.
    return types.SimpleNamespace(
        dimension=3, num_nodes=3, num_edges=2,
        nodes={8: object(), -3: object(), 91: object()},
        edges={9: types.SimpleNamespace(internal_points=[(0.5,0,0)] if internal else None),
               2: types.SimpleNamespace(internal_points=[])})

class StripPackageImports(ast.NodeTransformer):
    """Replace only explicit collaborator seams; reject new provider imports."""
    allowed_from = {
        (0, 'fibernet.sim._lammps_result', (('simulate_lammps', None),)),
        (0, 'fibernet.ml.beam_frame_fem', (('BeamFrameFEM', None),)),
        (0, 'fibernet.gen', (('pattern', None),)),
        (1, None, (('_generate_contract', 'contract'),)),
        (0, 'fibernet.sim._taichi_geometry', (
            ('preflight_taichi_geometry', None), ('GeometryPolicyError', None))),
        (0, 'fibernet.sim.accelerated', (('TaichiEngine', None),)),
        (1, '_taichi_viscoelastic', (('DynamicAxialMaterial', None),)),
        (1, '_energy_accounting', (('ExactLedger', None),)),
    }

    def visit_ImportFrom(self, node):
        if node.level or (node.module or '').split('.')[0] in {'fibernet','numpy','taichi','networkx'}:
            key = (node.level, node.module, tuple((a.name, a.asname) for a in node.names))
            if key not in self.allowed_from:
                raise AssertionError('Unreviewed provider import: ' + ast.dump(node))
            return ast.copy_location(ast.Pass(), node)
        return node
    def visit_Import(self, node):
        if any(alias.name.split('.')[0] in {'fibernet','numpy','taichi','networkx'} for alias in node.names):
            if [(a.name, a.asname) for a in node.names] != [('networkx', 'nx')]:
                raise AssertionError('Unreviewed provider import: ' + ast.dump(node))
            return ast.copy_location(ast.Pass(), node)
        return node

def definition(relative, name, class_name=None):
    tree = ast.parse((ROOT / relative).read_text(encoding='utf-8'))
    scope = tree.body if class_name is None else next(
        n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == class_name).body
    return copy.deepcopy(next(n for n in scope if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name == name))

def load_functions(relative, names, namespace, class_name=None):
    nodes=[]
    for name in names:
        node=definition(relative,name,class_name)
        if isinstance(node,ast.FunctionDef): node.decorator_list=[]
        nodes.append(StripPackageImports().visit(node))
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])
    ast.fix_missing_locations(module)
    namespace['__builtins__'] = isolated_builtins()
    exec(compile(module,str(ROOT/relative),'exec'),namespace)
    return namespace
