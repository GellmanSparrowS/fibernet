"""Affine geometry operations must not weld physical members or reallocate IDs."""
import copy
import numpy as np
import pytest
from fibernet.core.structure_graph import StructureGraph
from fibernet.core.material import Material
from fibernet.core.transforms import translate, rotate, mirror, scale, compose


def graph(dimension=3):
    g=StructureGraph.from_dict({'version':'3.1','dimension':dimension,
        'next_node_id':100,'next_edge_id':200,'box_size':[4,5,6],
        'nodes':[{'id':40,'position':[0,0,0], 'boundary':[True,False,True,False,True,False]},
                 {'id':7,'position':[2,1,1 if dimension==3 else 0]}],
        'edges':[{'id':99,'node_i':40,'node_j':7,'radius':.1,
                  'internal_points':[[.4,.8,.2 if dimension==3 else 0]],'segments':7},
                 {'id':3,'node_i':7,'node_j':40,'radius':.2,
                  'internal_points':[[1.4,.2,.8 if dimension==3 else 0]],'segments':9}]})
    g.metadata['nested']={'items':[1]}
    for n in g.nodes.values():n.metadata['nested']={'items':[n.node_id]}
    for e in g.edges.values():
        e.metadata['nested']={'items':[e.edge_id]}
        e.material=Material(name=str(e.edge_id),youngs_modulus=e.edge_id*1e6,
                            extra={'nested':[e.edge_id]})
    return g


CASES=[
    ('translate',lambda g:translate(g,[3,4,5]),lambda p:p+[3,4,5],1,None),
    ('rotate2d',lambda g:rotate(g,90,center=[1,2]),
     lambda p:np.array([-(p[1]-2)+1,p[0]-1+2,p[2]]),1,None),
    ('rotate3d',lambda g:rotate(g,90,axis=[1,0,0]),
     lambda p:np.array([p[0],-p[2],p[1]]),1,None),
    ('mirrorx',lambda g:mirror(g,'x',2),lambda p:np.array([4-p[0],p[1],p[2]]),1,0),
    ('mirrory',lambda g:mirror(g,'y',2),lambda p:np.array([p[0],4-p[1],p[2]]),1,1),
    ('mirrorz',lambda g:mirror(g,'z',2),lambda p:np.array([p[0],p[1],4-p[2]]),1,2),
    ('scale',lambda g:scale(g,-2,center=[1,2,3]),lambda p:(p-[1,2,3])*-2+[1,2,3],2,None),
    ('vector',lambda g:scale(g,[2,3,4]),lambda p:p*[2,3,4],1,None),
]


@pytest.mark.parametrize('name,operation,expected,radius_factor,axis',CASES)
def test_sparse_reversed_parallel_identity_geometry_and_ownership(name,operation,expected,radius_factor,axis):
    g=graph();before=copy.deepcopy(g.to_dict());h=operation(g)
    assert list(h.nodes)==[40,7] and list(h.edges)==[99,3]
    assert (h._next_node_id,h._next_edge_id)==(100,200)
    assert h._edge_set==g._edge_set and h._edge_set is not g._edge_set
    assert h.dimension==g.dimension and h.tolerance==g.tolerance
    for nid,n in g.nodes.items():
        assert h.nodes[nid].node_id==nid
        np.testing.assert_allclose(h.nodes[nid].position,expected(n.position),atol=1e-14)
        flags=list(n.boundary)
        if axis is not None:flags[2*axis],flags[2*axis+1]=flags[2*axis+1],flags[2*axis]
        assert h.nodes[nid].boundary==tuple(flags)
        assert not np.shares_memory(h.nodes[nid].position,n.position)
        h.nodes[nid].metadata['nested']['items'].append('new')
    for eid,e in g.edges.items():
        out=h.edges[eid]
        assert (out.edge_id,out.node_i,out.node_j)==(eid,e.node_i,e.node_j)
        assert out.radius==radius_factor*e.radius and out.segments==e.segments
        assert out.material.to_dict()==e.material.to_dict()
        np.testing.assert_allclose(out.internal_points,[expected(p) for p in e.internal_points],atol=1e-14)
        assert not np.shares_memory(out.internal_points,e.internal_points)
        out.material.extra['nested'].append('new')
        out.metadata['nested']['items'].append('new')
    assert not np.shares_memory(h._box_size,g._box_size)
    h._box_size[0]=999;h.metadata['nested']['items'].append('new')
    assert g.to_dict()==before


def test_rebuilt_spatial_hash_and_preserved_add_counters_and_edge_lookup():
    g=graph();h=translate(g,[10,20,30])
    assert h.add_node([10,20,30],merge=True)==40
    assert h.add_node([0,0,0],merge=True)==100
    assert h._next_node_id==101
    mapped=g._edge_set[frozenset((40,7))]
    assert h.add_edge(40,7)==mapped
    assert list(h.edges)==[99,3]
    assert h.add_edge(100,7)==200


@pytest.mark.parametrize('dimension',[2,3])
def test_empty_and_no_internal_points(dimension):
    empty=StructureGraph(dimension=dimension)
    empty._next_node_id=100;empty._next_edge_id=200
    h=translate(empty,[1,2]);assert not h.nodes and not h.edges
    assert (h._next_node_id,h._next_edge_id)==(100,200)
    g=graph(dimension);g.edges[99].internal_points=None
    g.edges[3].internal_points=np.empty((0,3))
    h=translate(g,[1,2]);assert h.edges[99].internal_points is None
    assert h.edges[3].internal_points.shape==(0,3)


@pytest.mark.parametrize('factor,expected_radius,expected_box',[
    (2,.2,[8,10,12]),(-2,.2,[-8,-10,-12]),([2,2,2],.1,[8,10,12]),
    ([2,3,4],.1,[8,15,24]),(0,0,[0,0,0])])
def test_existing_scale_policy(factor,expected_radius,expected_box):
    g=graph();h=scale(g,factor)
    assert h.edges[99].radius==expected_radius
    np.testing.assert_array_equal(h.box_size,expected_box)
    assert list(h.nodes)==[40,7] and list(h.edges)==[99,3]


def test_compose_retains_identity_and_shared_material_copy_policy():
    g=graph();g.edges[3].material=g.edges[99].material
    h=compose(g,lambda x:translate(x,[3,4,5]),lambda x:mirror(x,'z'))
    assert list(h.nodes)==[40,7] and list(h.edges)==[99,3]
    # Existing Graph.copy semantics independently deepcopy each edge's material.
    assert h.edges[3].material is not h.edges[99].material
    assert h.edges[3].material is not g.edges[3].material
    assert g.edges[3].material is g.edges[99].material
