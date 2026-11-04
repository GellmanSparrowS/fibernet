"""Small full-XYZ curved graph analysis and native JSON roundtrip.

Run with a new output directory. This synthetic geometry is not a manufacturing
route, physical simulation, material validation or training dataset qualification.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from fibernet import Material, StructureGraph
from fibernet.analysis import analyze_geometry_paths, analyze_topology

NODE_PAIRS = ((10, 30), (10, 70), (30, 110), (10, 190), (10, 10))
BUDGETS = dict(max_nodes=5, max_edges=6, max_points=8, max_pairs=5, max_path_work=55)


def make_graph():
    """Build through the existing native schema, preserving explicit sparse IDs."""
    materials = [Material(name='synthetic-A', youngs_modulus=2e6,
                          poissons_ratio=.25, density=900).to_dict(),
                 Material(name='synthetic-B', youngs_modulus=5e6,
                          poissons_ratio=.3, density=1100).to_dict()]
    nodes = [(10, [0, 0, 0]), (30, [2, 0, 0]), (70, [2, 2, 2]),
             (110, [0, 2, 0]), (190, [4, 4, 4])]
    edges = [(101, 10, 30, [[1, 0, 1]], 0),
             (205, 30, 10, [[1, 0, -2]], 1),
             (309, 30, 70, None, 0),
             (407, 70, 110, [[1, 2, 2]], 1),
             (503, 110, 10, None, 0),
             (601, 10, 70, None, 1)]
    record = dict(version='3.1', dimension=3, tolerance=1e-9,
                  next_node_id=200, next_edge_id=700, box_size=None,
                  metadata=dict(length_unit='m', coordinate_frame='synthetic XYZ',
                                purpose='geometry analysis only'),
                  nodes=[dict(id=nid, position=position,
                              metadata=dict(label='isolated' if nid == 190 else 'connected'))
                         for nid, position in nodes],
                  edges=[dict(id=eid, node_i=a, node_j=b, radius=.01+.002*kind,
                              material=materials[kind], internal_points=points,
                              segments=4, metadata=dict(label='edge-'+str(eid)))
                         for eid, a, b, points, kind in edges])
    return StructureGraph.from_dict(record)


def measure(graph):
    """Geometry metrics have explicit units; materials remain in graph records."""
    return dict(topology=asdict(analyze_topology(graph)),
                polyline=asdict(analyze_geometry_paths(
                    graph, length_unit='m', coordinate_frame='synthetic XYZ',
                    node_pairs=NODE_PAIRS, distance_mode='polyline', **BUDGETS)),
                chord=asdict(analyze_geometry_paths(
                    graph, length_unit='m', coordinate_frame='synthetic XYZ',
                    node_pairs=NODE_PAIRS, distance_mode='chord', **BUDGETS)))


def run(output_dir):
    """Refuse an existing directory; retain partial output on failure."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    graph = make_graph()
    before = json.loads(json.dumps(graph.to_dict(), allow_nan=False))
    report = measure(graph)
    if graph.to_dict() != before:
        raise RuntimeError('analysis changed its graph input')
    graph.save_json(str(output/'source_graph.json'))
    restored = StructureGraph.from_dict(json.loads((output/'source_graph.json').read_text()))
    if restored.to_dict() != before or measure(restored) != report:
        raise RuntimeError('native JSON roundtrip changed graph or analysis')
    restored.save_json(str(output/'restored_graph.json'))
    (output/'analysis.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir')
    run(parser.parse_args().output_dir)
