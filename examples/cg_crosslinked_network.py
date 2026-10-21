"""Pair-free 4 x 4 x 3 crosslinked CG network with explicit reference geometry.

Run using an existing, pinned RuntimeSpec JSON (no discovery or installation):
    python -m examples.cg_crosslinked_network --runtime-json RUNTIME.json --output OUTPUT

The default is 500 steps at 1e-7 s. For the same duration at half the timestep,
pass --steps 1000 --timestep-s 5e-8. Importing or building never launches LAMMPS;
run_example/main do. Coefficients are synthetic, not an EI/torsion calibration.
This is a small bounded example, not a large-scale performance demonstration.
"""
import argparse
import json
from pathlib import Path

from examples.cg_bent_branched_network import load_runtime
from fibernet import simulate
from fibernet.integrations import ResourceLimits, RuntimeSpec
from fibernet.sim.accelerated import SimResult

L = 1e-6
COUNTS = (48, 104, 64)


def build_case(*, steps=500, timestep_s=1e-7):
    """Return unloaded reference, loaded initial graph, and explicit CG request.

    Bond rest lengths remain 1e-6 m, never the loaded edge lengths. Cosine
    angles have straight (pi) rest angles, not their loaded initial angles.
    Twelve beads on the i=0 face are held; there are no nonbonded forces.
    """
    from fibernet.core.structure_graph import StructureGraph
    from fibernet.integrations import CGAngle, CGRequest, HarmonicBond, RunSpec
    coords = [(i, j, k) for i in range(4) for j in range(4) for k in range(3)]
    ids = {v: 101 + 13*n for n, v in enumerate(coords)}
    ref = {ids[v]: tuple(q*L for q in v) for v in coords}
    loaded = {}
    for (i, j, k), n in ids.items():
        loaded[n] = (1.02*i*L, j*L, k*L+.01*i*L+.03*L*(i*(3-i)/2)*(j*(3-j)/2))
    incidence = {}
    lookup = {}
    angles = {}
    for v in coords:
        for axis, bound in enumerate((4, 4, 3)):
            if v[axis]+1 < bound:
                w = list(v)
                w[axis] += 1
                w = tuple(w)
                e = 1001 + 17*len(incidence)
                incidence[e] = (ids[v], ids[w])
                lookup[frozenset((ids[v], ids[w]))] = e
    for v in coords:
        for axis, bound in enumerate((4, 4, 3)):
            if 0 < v[axis] < bound-1:
                u, w = list(v), list(v)
                u[axis] -= 1
                w[axis] += 1
                a, b, c = ids[tuple(u)], ids[v], ids[tuple(w)]
                angles[5003+19*len(angles)] = CGAngle(
                    a, b, c, lookup[frozenset((a, b))],
                    lookup[frozenset((b, c))], 3e-18)

    def graph(points, role):
        # Keep the original scene metadata for reproducible graph identities.
        return StructureGraph.from_dict({'dimension':3,
            'metadata':{'qualification_scene':'crosslinked-grid-4x4x3/1','geometry_role':role,
                        'coefficient_scope':'synthetic explicit CG; graph material/radius provenance only'},
            'nodes':[{'id':n,'position':list(p)} for n,p in points.items()],
            'edges':[{'id':e,'node_i':i,'node_j':j,'radius':1e-8,'internal_points':None}
                     for e,(i,j) in incidence.items()]})
    reference, initial = graph(ref,'unloaded_reference'), graph(loaded,'loaded_initial')
    request = CGRequest(length_unit='m',maximum_bond_length_m=3e-6,
        masses_kg={n:1e-15 for n in ids.values()},
        bonds={e:HarmonicBond(2e-6,L) for e in incidence}, angles=angles,
        projectile_node_id=None,projectile_pairs={},pair_encoding=None,profile=None,
        fixed_node_ids=tuple(ids[v] for v in coords if v[0]==0),
        run=RunSpec(timestep_s,steps,1,((-10e-6,10e-6),)*3))
    assert (len(initial.nodes),len(initial.edges),len(angles)) == COUNTS
    return reference, initial, request


def run_example(runtime, output_dir, *, steps=500, timestep_s=1e-7):
    """Run public simulate and verify detailed and summary export round-trips."""
    if type(runtime) is not RuntimeSpec:
        raise TypeError('an explicit RuntimeSpec is required')
    reference, initial, request = build_case(steps=steps, timestep_s=timestep_s)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    reference.save_json(output/'reference_graph.json')
    initial.save_json(output/'graph.json')
    result = simulate(
        initial, backend='lammps', mode='dynamics', cg_request=request,
        runtime=runtime, workdir=output/'native', record_state=True,
        limits=ResourceLimits(cpu_seconds=10, wall_seconds=20.,
                              output_bytes=16*1024**2, file_size_bytes=16*1024**2),
    )
    for detailed, filename in ((True, 'result_detailed.json'),
                               (False, 'result_summary.json')):
        path = output/filename
        result.save(path, detailed=detailed)
        reloaded = SimResult.load(path)
        if reloaded.to_dict(detailed) != result.to_dict(detailed):
            raise ValueError('saved CG result did not round-trip exactly')
        if reloaded.metadata['model_contract']['projectile_node_id'] is not None:
            raise ValueError('saved CG result lost its null projectile')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-json', required=True, type=Path,
                        help='existing pinned RuntimeSpec JSON, no runtime discovery')
    parser.add_argument('--output', required=True, type=Path,
                        help='new output directory; existing directories are refused')
    parser.add_argument('--steps', type=int, default=500, help='0..1000, every step saved')
    parser.add_argument('--timestep-s', type=float, default=1e-7)
    args = parser.parse_args(argv)
    result = run_example(load_runtime(args.runtime_json), args.output,
                         steps=args.steps, timestep_s=args.timestep_s)
    print(json.dumps({
        'output_directory': str(args.output.resolve()),
        'nodes': result.n_nodes, 'edges': result.n_edges,
        'potential_energy_J': result.energy,
        'maximum_axial_edge_force_N': result.max_force,
        'maximum_net_nodal_force_N': result.metadata['final_max_net_nodal_force_N'],
        'scope': 'synthetic harmonic axial bonds and cosine bending; no nonbonded forces, EI or torsion',
    }, indent=2))


if __name__ == '__main__':
    main()
