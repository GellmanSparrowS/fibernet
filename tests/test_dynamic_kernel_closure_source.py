"""Inspect actual freshly constructed kernel closures using inert decorators.

No numerical imports, kernel execution, material trial, or native initialization.
"""
import ast,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def check_source(path,expect_bound):
    tree=ast.parse(path.read_text())
    dynamics=next(n for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='TaichiEngine' for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='dynamics')
    allocate=next(n for n in dynamics.body if isinstance(n,ast.If) and any(isinstance(x,ast.FunctionDef) and x.name=='_cached_substep' for x in ast.walk(n)))
    body=[]
    for n in allocate.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Subscript) for t in n.targets):break
        body.append(n)
    # All field allocations and decorators are the actual AST. Kernel bodies
    # remain unmodified but are never invoked. Return the decorated functions.
    body.append(ast.Return(value=ast.Name(id='_cached_substep',ctx=ast.Load())))
    function=ast.FunctionDef(name='build',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=x) for x in ('material_mode','record_energy_ledger','dim','num_nodes','num_edges')],kwonlyargs=[],kw_defaults=[],defaults=[]),body=body,decorator_list=[])
    inspected=[]
    class TI:
        f64='f64';i32='i32'
        def field(self,*args,**kwargs):return object()
        def kernel(self,fn):
            values={};empty=[]
            for name,cell in zip(fn.__code__.co_freevars,fn.__closure__ or ()):
                try:values[name]=cell.cell_contents
                except ValueError:empty.append(name)
            inspected.append(dict(kernel=fn.__name__,empty=empty,names=sorted(values)))
            return fn
    ti=TI();ti.Vector=TI()
    ns=dict(ti=ti)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),'<actual allocation/kernel AST; inert fields>','exec'),ns)
    for material,ledger in [(False,False),(False,True),(True,False)]:ns['build'](material,ledger,3,2,1)
    empty=[r for r in inspected if r['empty']]
    if expect_bound:assert not empty,empty
    else:assert any(set(r['empty'])=={'ti_material_force','material_v_kick','material_v_drag','material_x_drift'} for r in empty),empty
    return inspected
def test_all_kernel_closures_are_bound():
    check_source(ROOT/'fibernet/sim/accelerated.py', True)

if __name__ == '__main__':
    test_all_kernel_closures_are_bound()
    print('PASS: all material and legacy kernel closure cells are bound; zero native calls')
