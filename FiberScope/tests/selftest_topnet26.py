"""Reference-derived degree, twin identity, and simulation/export consistency."""
import sys
from pathlib import Path
from dataclasses import replace
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fslab.structure import StructureFactory, all_unit_keys
from fslab.manufacturing import compile_planar


def main():
    f = StructureFactory(n_pts_per_side=5)
    baseline = compile_planar(f)
    assert len(baseline.positions) == 256
    assert len(baseline.edges) == 288
    degree = np.bincount(baseline.edges.ravel(), minlength=len(baseline.positions))
    for point, expected in (([0,0,0],4), ([10,0,0],6), ([10,10,0],8)):
        node = np.flatnonzero(np.all(np.isclose(baseline.reference,point),axis=1))
        assert len(node)==1 and degree[node[0]]==expected
    changed = compile_planar(replace(f, line_displacements=[[0,.2]]*5))
    assert np.array_equal(baseline.route_edges,changed.route_edges)
    # Midpoints on a shared side remain distinct and separate in opposite directions.
    pair = np.flatnonzero(np.all(np.isclose(baseline.reference,[10,5,0]),axis=1))
    assert len(pair)==2
    assert np.allclose(sorted(changed.positions[pair,0]),[8,12])
    for unit in all_unit_keys():
        factory = StructureFactory(unit=unit,grid_x=2,grid_y=2,n_pts_per_side=2)
        model, graph = compile_planar(factory), factory.build()
        assert np.array_equal(model.positions,graph.node_positions()),unit
        assert np.array_equal(model.edges,graph.edge_array()),unit
        assert graph.metadata['topology_id']==model.topology_id
    print('[topnet26] reference degrees 4/6/8, 288 segments, independent shared twins, fixed route and all-unit simulation/export identity PASS')


if __name__ == '__main__':
    main()
