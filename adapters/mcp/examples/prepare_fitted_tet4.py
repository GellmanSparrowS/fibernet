"""Offline demonstration: freeze an existing fitted axial mesh and two load recipes.

From the repository root, with FiberNet, the adapter and meshio installed:
    python -m adapters.mcp.examples.prepare_fitted_tet4 operator-demo
"""
import argparse
from collections import Counter
from itertools import combinations
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from fibernet.io.tet4_meshio import write_tet4_vtu_bundle
from fibernet_mcp.safety import canonical
from fibernet_mcp.tet4_operator import export_bundle, export_snapshot

NAME='fitted-axial-junction'

def prepare(root, *, via_vtu=True):
    """Write an offline operator fixture and intact stable-ID control maps."""
    root=Path(root)
    root.mkdir()
    directory=root/'tet4-sources'
    directory.mkdir()
    path=Path(__file__).resolve().parents[3]/'examples'/'tet4_static_fitted_placement.py'
    spec=spec_from_file_location('_frozen_fitted_tet4_example',path)
    sample=module_from_spec(spec)
    spec.loader.exec_module(sample)
    bundle=sample._prepare()[0]['junction']
    recipe=sample._fixture()
    faces=Counter(tuple(sorted(int(bundle.node_ids[i]) for i in tri))
                  for row in bundle.tetrahedra for tri in combinations(row,3))
    minus={int(i) for f in recipe['fixed_minus_x_faces'] for i in f}
    plus={int(i) for f in recipe['loaded_plus_x_faces'] for i in f}
    assert minus and plus and not minus&plus
    side=sorted(f for f,n in faces.items() if n==1 and not (set(f)<=minus or set(f)<=plus))
    material_keys=sorted(bundle.materials)
    base=min(material_keys,key=lambda k:bundle.materials[k]['young_modulus_pa'])
    labels={str(k) for k in material_keys}
    # Synthetic electrical properties, NOT inferred from Young's modulus.
    # Existing non-base tags identify the frozen junction placement only.
    common=dict(conductivity_W_per_m_K_by_material=dict.fromkeys(sorted(labels),.25),
        electrode_voltage_V_by_id={**dict.fromkeys(map(str,sorted(minus)),0.),**dict.fromkeys(map(str,sorted(plus)),.3)},
        prescribed_K_by_id=dict.fromkeys(map(str,sorted(minus|plus)),300.),
        faces_by_node_id=[list(map(str,f)) for f in side],convection_W_per_m2_K=100.,ambient_K=299.95)
    cases={name:dict(common,conductivity_S_per_m_by_material={
                    str(k):2. if k==base else sigma for k in material_keys})
           for name,sigma in (('uniform_electrical',2.),('lower_junction_conductivity',.5))}
    if via_vtu:
        source=write_tet4_vtu_bundle(root/'fitted-axial-junction.zip',**vars(bundle))
        registration=export_bundle(source,NAME,directory)
    else:
        registration=export_snapshot(bundle,NAME,directory)
    controls=dict(tet4_sources=[registration],cases=cases)
    with (root/'fitted-controls.json').open('xb') as out: out.write(canonical(controls)+b'\n')
    return registration,cases,bundle

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('new_directory')
    args=parser.parse_args()
    entry,_,_=prepare(args.new_directory)
    print(canonical({'tet4_sources':[entry]}).decode())

if __name__=='__main__':main()
