"""Export a true 3D Y network as two edge-once strokes, without added fibers.

Run from the source checkout with declared FiberNet dependencies installed:
    python -m examples.multistroke_manufacturing NEW_OUTPUT_DIRECTORY

This example generates only a graph and route files. It performs no simulation,
solid generation, travel planning, machine control or physical qualification.
"""
import argparse
import hashlib
from pathlib import Path

from fibernet import StructureGraph, manufacturing_trails_from_graph, export_mapped_trails


def main(output_directory):
    output = Path(output_directory)
    # Requiring a fresh directory also protects the saved authoritative graph.
    output.mkdir()
    graph = StructureGraph(dimension=3)
    graph.metadata.update(length_unit='m', coordinate_frame='lab_xyz',
                          description='Synthetic three-fiber spatial Y; stored material is provenance only')
    for point in [(0., 0., 0.), (.01, 0., 0.), (0., .01, .01), (0., -.01, .01)]:
        graph.add_node(point)
    for tip in (1, 2, 3):
        graph.add_edge(0, tip, radius=.0001, n_internal=0)
    source_path = output / 'source_graph.json'
    graph.save_json(str(source_path))
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    trails, snapshot = manufacturing_trails_from_graph(
        graph, graph_id='spatial-y-example', length_unit='m', coordinate_frame='lab_xyz')
    assert trails.stroke_count == 2
    assert sorted(trails.route_edges.tolist()) == [0, 1, 2]
    mapping = export_mapped_trails(output / 'trails.csv', trails, snapshot,
                                   sidecar_path=output / 'trails.json',
                                   source_input_sha256=source_sha256)
    assert hashlib.sha256((output / 'trails.csv').read_bytes()).hexdigest() == mapping['route_csv_sha256']
    print(f'{len(snapshot.edge_ids)} physical fibers, {trails.stroke_count} strokes; no added fibers')
    for stroke in range(trails.stroke_count):
        start, stop = trails.stroke_node_ptr[stroke:stroke + 2]
        rows = trails.route_nodes[start:stop]
        print(f'Stroke {stroke}: source node IDs {snapshot.node_ids[rows].tolist()}, closed={bool(trails.closed[stroke])}')
    print(f'Graph: {source_path}')
    print(f'Routes: {output / "trails.csv"} and {output / "trails.json"}')
    print('Process-specific travel and fabrication planning are separate steps')
    return mapping


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_directory', help='A new directory whose parent already exists')
    main(parser.parse_args().output_directory)
