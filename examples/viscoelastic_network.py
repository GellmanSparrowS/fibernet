"""Callable relaxation, creep/release and cyclic examples; no runs on import.

All parameters are synthetic. Graph coordinates/radii are mm, controls SI.
Run this file explicitly to write the three history results in a directory.
"""
import argparse
from pathlib import Path
import numpy as np
import fibernet as fn


def make_network():
    graph = fn.pattern_2d(unit='triangle', box=(4.,4.), grid=(3,3), radius=.04)
    # Exercise the graph's existing stable-ID roundtrip, retaining its type.
    record = graph.to_dict()
    node_map = {n['id']: 11+3*n['id'] for n in record['nodes']}
    for node in record['nodes']:
        node['id'] = node_map[node['id']]
    for edge in record['edges']:
        edge['id'] = 17+5*edge['id']
        edge['node_i'] = node_map[edge['node_i']]
        edge['node_j'] = node_map[edge['node_j']]
    record['next_node_id'] = 1+max(n['id'] for n in record['nodes'])
    record['next_edge_id'] = 1+max(e['id'] for e in record['edges'])
    return fn.StructureGraph.from_dict(record)


def inputs(graph, protocol, times):
    t = np.asarray(times)
    xmin=min(n.position[0] for n in graph.nodes.values())
    xmax=max(n.position[0] for n in graph.nodes.values())
    left=[i for i,n in graph.nodes.items() if abs(n.position[0]-xmin)<1e-9]
    right=[i for i,n in graph.nodes.items() if abs(n.position[0]-xmax)<1e-9]
    top=max(right,key=lambda i:graph.nodes[i].position[1])
    fixed={i:[[0.,0.,0.] for _ in t] for i in left}
    loads={}
    ramp=np.minimum(t/.2,1.)
    if protocol == 'relaxation':
        for i in right:
            fixed[i]=[[float(4e-6*x),None,None] for x in ramp]
        # Different transverse ramp timing produces nonproportional bending.
        for k,y in enumerate(1e-6*np.minimum(t/.4,1.)):
            fixed[top][k][1]=float(y)
    elif protocol == 'creep':
        amplitude=2e-5*np.minimum(t/.2,1.)*np.clip((2.-t)/.4,0.,1.)
        for i in right:
            loads[i]=np.column_stack((amplitude/len(right),np.zeros_like(t),np.zeros_like(t))).tolist()
        fixed[top]=[[None,float(3e-7*y),None] for y in np.minimum(t/.4,1.)]
        loads[top]=np.asarray(loads[top]); loads[top][:,2]=2e-10*np.minimum(t/.6,1.)
        loads[top]=loads[top].tolist()
    elif protocol == 'cyclic':
        for i in right:
            fixed[i]=[[float(3e-6*x),None,None] for x in np.sin(2*np.pi*t)]
        for k,y in enumerate(4e-7*(1-np.cos(4*np.pi*t))):
            fixed[top][k][1]=float(y)
    else:
        raise ValueError('protocol must be relaxation, creep or cyclic')
    return dict(material=fn.LinearViscoelasticMaterial(1e6,((2e6,.1),(1e6,1.))),
                times=t, constraints=fixed, loads=loads, length_unit='mm')


def run_protocol(protocol, steps=200):
    graph=make_network()
    return fn.simulate(graph,backend='fem',mode='history',
                       **inputs(graph,protocol,np.linspace(0.,2.,steps+1)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    for protocol in ('relaxation','creep','cyclic'):
        result=run_protocol(protocol)
        result.save(args.output/(protocol+'.json'))
        print(protocol, len(result.node_ids), 'nodes;', len(result.edge_ids), 'edges')


if __name__=='__main__':
    main()
