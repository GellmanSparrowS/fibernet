"""Pure test geometry builders; importing this module never runs an example.

The two geometry builders retain the example inputs, but deliberately exclude
example entry points, simulation adapters, runtime discovery, and output I/O.
"""
import itertools
import math
from pathlib import Path

from fibernet.core.structure_graph import StructureGraph
from fibernet.integrations import CGAngle, CGRequest, HarmonicBond, RunSpec, RuntimeSpec


def inert_runtime(directory):
    """Describe portable nonexistent paths without accessing either path."""
    python = str(Path(directory) / 'inert-python-placeholder')
    lammps = str(Path(directory) / 'inert-lammps-placeholder')
    return RuntimeSpec(
        'cli_serial_static', python, lammps,
        expected_version=20250722, expected_banner='LAMMPS (22 Jul 2025)',
        file_sha256=((python, 'a' * 64), (lammps, 'b' * 64)),
    )


def build_branched_case(*, steps=20, timestep_s=1e-7):
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

def build_crosslinked_case(*, steps=500, timestep_s=1e-7):
    """Return unloaded reference, loaded initial graph, and explicit CG request.

    Bond rest lengths remain 1e-6 m, never the loaded edge lengths. Cosine
    angles have straight (pi) rest angles, not their loaded initial angles.
    Twelve beads on the i=0 face are held; there are no nonbonded forces.
    """
    L = 1e-6
    COUNTS = (48, 104, 64)
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

def capped_case(n=64, edge_count=128, angle_count=128, steps=1000):
    graph = StructureGraph.from_dict({'dimension':3, 'nodes': [
        {'id':i, 'position':[1e-6*(i%4), 1e-6*((i//4)%4), 1e-6*(i//16)]}
        for i in range(n)], 'edges':[]})
    edges = {}; neighbors = {i: [] for i in range(n)}
    for delta in range(1, n):
        for i in range(n-delta):
            if len(edges) == edge_count: break
            j = i+delta; eid = graph.add_edge(i, j, n_internal=0)
            edges[(i,j)] = eid; neighbors[i].append(j); neighbors[j].append(i)
        if len(edges) == edge_count: break
    bonds = {eid: HarmonicBond(2e-6, math.dist(graph.nodes[i].position, graph.nodes[j].position))
             for (i,j), eid in edges.items()}
    angles = {}
    for j, adjacent in neighbors.items():
        for i,k in itertools.combinations(sorted(adjacent), 2):
            if len(angles) == angle_count: break
            angles[len(angles)] = CGAngle(i,j,k,edges[tuple(sorted((i,j)))],
                                          edges[tuple(sorted((j,k)))],3e-18)
        if len(angles) == angle_count: break
    q = CGRequest('m', 20e-6, {i:1e-15 for i in graph.nodes}, bonds, angles,
                  None, {}, RunSpec(1e-7, steps, 1, ((-2e-6,30e-6),)*3))
    return graph, q
