"""Run the bounded experimental fracture API on an explicit, material-complete fracture geometry JSON."""
import argparse
import json
from pathlib import Path
from fibernet import StructureGraph, SNode, SEdge, Material
from fibernet._json import unique_object_pairs
import numpy as np
from . import FractureConfig, FractureSolver
from ._sparse import identifier, number


def load_geometry(path):
    """Load the original material-complete fracture geometry schema."""
    body=json.loads(Path(path).read_text(encoding='utf-8'))
    if set(body)!={'schema','nodes','edges'} or body['schema']!='fibernet-fracture-geometry-1':
        raise ValueError('Explicit fibernet-fracture-geometry-1 schema required')
    g=StructureGraph(dimension=3)
    for row in body['nodes']:
        if set(row)!={'id','position'} or type(row['id']) is not int or row['id']<0 or row['id'] in g.nodes:
            raise ValueError('Unique explicit node IDs and XYZ positions required')
        g.nodes[row['id']]=SNode(position=np.asarray(row['position'],dtype=float),node_id=row['id'])
    for row in body['edges']:
        required={'id','node_i','node_j','radius','material'}
        if not required<=set(row) or set(row)-required-{'internal_points'} or type(row['id']) is not int or row['id']<0 or row['id'] in g.edges:
            raise ValueError('Explicit edge IDs, endpoints, radius and material required')
        mat=row['material']
        if set(mat)!={'youngs_modulus','poissons_ratio','shear_modulus'}:
            raise ValueError('Explicit isotropic E, nu and G required per edge')
        g.edges[row['id']]=SEdge(node_i=row['node_i'],node_j=row['node_j'],edge_id=row['id'],
            radius=row['radius'],material=Material(**mat),
            internal_points=None if row.get('internal_points') is None else np.asarray(row['internal_points'],dtype=float))
    return g


def _load_structure_graph(path):
    """Require the physical input fields before native graph decoding."""
    body = json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique_object_pairs)
    if (not isinstance(body, dict) or body.get('version') != '3.1'
            or {'schema', 'fibers', 'crosslinks'}.intersection(body)
            or not isinstance(body.get('nodes'), list) or not isinstance(body.get('edges'), list)):
        raise ValueError('Explicit StructureGraph version 3.1 with nodes and edges required')

    node_ids = set()
    for node in body['nodes']:
        if not isinstance(node, dict) or not {'id', 'position'} <= node.keys():
            raise ValueError('Explicit node IDs and finite XYZ coordinates required')
        nid = identifier(node['id'])
        if nid in node_ids:
            raise ValueError(f'duplicate node ID: {nid}')
        node_ids.add(nid)
        position = node['position']
        if not isinstance(position, list) or len(position) != 3:
            raise ValueError(f'Node {nid} requires finite XYZ coordinates')
        for coordinate in position:
            number(coordinate, f'coordinate for node {nid}')

    edge_ids = set()
    elastic_fields = ('youngs_modulus', 'poissons_ratio', 'shear_modulus')
    for edge in body['edges']:
        if not isinstance(edge, dict) or not {'id', 'node_i', 'node_j', 'radius', 'material'} <= edge.keys():
            raise ValueError('Explicit edge IDs, endpoints, radius and material required')
        eid = identifier(edge['id'])
        if eid in edge_ids:
            raise ValueError(f'duplicate edge ID: {eid}')
        edge_ids.add(eid)
        i, j = identifier(edge['node_i']), identifier(edge['node_j'])
        if i == j or i not in node_ids or j not in node_ids:
            raise ValueError(f'Edge {eid} requires distinct existing endpoint IDs')
        number(edge['radius'], f'radius for edge {eid}', positive=True)
        material = edge['material']
        if not isinstance(material, dict) or any(material.get(key) is None for key in elastic_fields):
            raise ValueError(f'Edge {eid} requires explicit isotropic E, nu and G')
        for key in elastic_fields:
            number(material[key], f'{key} for edge {eid}')

    return StructureGraph.from_dict(body)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph',required=True)
    p.add_argument('--graph-format', choices=['fibernet-fracture-geometry-1', 'structure-graph-3.1'],
                   default='fibernet-fracture-geometry-1',
                   help='input geometry format (default: %(default)s)')
    p.add_argument('--config',required=True)
    p.add_argument('--fixed',type=int,nargs='+',required=True)
    p.add_argument('--loaded',type=int,nargs='+',required=True)
    p.add_argument('--length-unit',choices=['m','mm','um'],required=True)
    p.add_argument('--initially-broken',type=int,nargs='*',default=[])
    p.add_argument('--output',required=True)
    p.add_argument('--checkpoint')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if Path(args.output).exists():
        p.error('output already exists; choose a new filename')
    for name in (args.graph,args.config):
        if Path(name).stat().st_size > 16*1024**2:
            p.error('input exceeds the 16 MiB CLI limit')
    config=FractureConfig(**json.loads(Path(args.config).read_text(encoding='utf-8')))
    if args.graph_format == 'structure-graph-3.1':
        graph = _load_structure_graph(args.graph)
    else:
        graph = load_geometry(args.graph)
    result=FractureSolver(graph,config,fixed_nodes=args.fixed,loaded_nodes=args.loaded,
        length_unit=args.length_unit,initially_broken_edges=args.initially_broken).run(
        checkpoint=args.checkpoint,resume=args.resume)
    result.save(args.output)
    print(json.dumps(dict(termination_reason=result.termination_reason,
                         complete=result.data['complete'],events=len(result.events))))

if __name__=='__main__':
    main()
