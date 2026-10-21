"""Explicit, pair-free 3-D CG example using an already qualified CLI runtime.

Run from the source root with an existing, pinned RuntimeSpec JSON:
    python -m examples.cg_bent_branched_network --runtime-json RUNTIME.json --output OUTPUT

This launches native LAMMPS. Importing/building the example does not. The
illustrative coefficients are not a continuum beam EI/torsion calibration.
"""
import argparse
import json
import math
from pathlib import Path

from fibernet import simulate
from fibernet.core.structure_graph import StructureGraph
from fibernet.integrations import (CGAngle, CGRequest, HarmonicBond,
                                   ResourceLimits, RunSpec, RuntimeSpec)
from fibernet.sim.accelerated import SimResult


def build_case(*, steps=20, timestep_s=1e-7):
    """Seven actual graph beads, six bonds, five explicitly selected angles."""
    positions = {
        101: (0., 0., 0.), 205: (1e-6, 0., 0.),
        309: (2e-6, .4e-6, 0.), 413: (3e-6, .4e-6, .3e-6),
        517: (1e-6, 1e-6, 0.), 621: (1.5e-6, 2e-6, .5e-6),
        725: (1e-6, -.8e-6, .6e-6),
    }
    incidence = {71: (101, 205), 83: (205, 309), 97: (309, 413),
                 109: (205, 517), 127: (517, 621), 139: (205, 725)}
    # Deserialize explicit stable IDs through the public authoritative graph API.
    # Graph material/radius records are provenance, not CG coefficient sources.
    graph = StructureGraph.from_dict({
        'dimension': 3,
        'nodes': [{'id': n, 'position': list(p)} for n, p in positions.items()],
        'edges': [{'id': e, 'node_i': i, 'node_j': j, 'radius': 1e-8,
                   'internal_points': None} for e, (i, j) in incidence.items()],
    })
    bonds = {e: HarmonicBond(2e-6, math.dist(positions[i], positions[j]))
             for e, (i, j) in incidence.items()}
    angles = {
        47: CGAngle(101, 205, 309, 71, 83, 3e-18),
        59: CGAngle(205, 309, 413, 83, 97, 3e-18),
        61: CGAngle(101, 205, 517, 71, 109, 3e-18),
        73: CGAngle(205, 517, 621, 109, 127, 3e-18),
        89: CGAngle(101, 205, 725, 71, 139, 3e-18),
    }
    request = CGRequest(
        length_unit='m', maximum_bond_length_m=3e-6,
        masses_kg={n: 1e-15 for n in graph.nodes}, bonds=bonds, angles=angles,
        projectile_node_id=None, projectile_pairs={}, pair_encoding=None,
        profile=None, fixed_node_ids=(101,),
        run=RunSpec(timestep_s, steps, 1, ((-10e-6, 10e-6),)*3),
    )
    return graph, request


def load_runtime(path):
    """Read caller-provided RuntimeSpec fields; never discover or install tools."""
    path = Path(path)
    if path.stat().st_size > 65536:
        raise ValueError('runtime JSON exceeds 64 KiB')
    record = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(record, dict):
        raise ValueError('runtime JSON must be an object of RuntimeSpec fields')
    for key in ('file_sha256', 'environment'):
        if key in record:
            record[key] = tuple(tuple(row) for row in record[key])
    return RuntimeSpec(**record)


def run_example(runtime, output_dir, *, steps=20, timestep_s=1e-7):
    """Execute through public simulate, then save and reload both export forms."""
    if type(runtime) is not RuntimeSpec:
        raise TypeError('an explicit RuntimeSpec is required')
    graph, request = build_case(steps=steps, timestep_s=timestep_s)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    graph.save_json(output/'graph.json')
    result = simulate(
        graph, backend='lammps', mode='dynamics', cg_request=request,
        runtime=runtime, workdir=output/'native',
        limits=ResourceLimits(cpu_seconds=5, wall_seconds=10.,
                              output_bytes=2*1024**2, file_size_bytes=2*1024**2),
        record_state=True,
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
    parser.add_argument('--steps', type=int, default=20, help='0..1000, every step saved')
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
        'scope': 'harmonic axial bonds and CG cosine bending; no nonbonded forces, EI or torsion',
    }, indent=2))


if __name__ == '__main__':
    main()
