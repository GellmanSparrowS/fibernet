"""Merge right-side mutable ownership; no welding or topology change."""
import copy
import numpy as np
import pytest
from fibernet.core.structure_graph import StructureGraph
from fibernet.core.material import Material

def fixture():
    left=StructureGraph.from_dict({'version':'3.1','dimension':3,'box_size':[2,3,4],'next_node_id':100,'next_edge_id':200,
        'metadata':{'side':'left','nested':{'v':[1]}},'nodes':[{'id':40,'position':[0,0,0]},{'id':7,'position':[1,1,1]}],
        'edges':[{'id':99,'node_i':40,'node_j':7,'radius':.1}]})
    right=StructureGraph.from_dict({'version':'3.1','dimension':3,'next_node_id':300,'next_edge_id':400,
        'metadata':{'side':'right'},'nodes':[{'id':90,'position':[0,0,0],'metadata':{'nested':{'v':[2]}}},
        {'id':3,'position':[1,1,1],'metadata':{'nested':{'v':[3]}}}],
        'edges':[{'id':42,'node_i':90,'node_j':3,'radius':.2,'segments':7,'internal_points':[[.2,.3,.4]],'metadata':{'nested':{'v':[4]}}},
                 {'id':2,'node_i':3,'node_j':90,'radius':.3,'segments':9,'internal_points':[[.8,.7,.6]],'metadata':{'nested':{'v':[5]}}}]})
    for eid,e in right.edges.items():e.material=Material(name=str(eid),youngs_modulus=eid*1e6,extra={'v':[eid]},stiffness_tensor=np.eye(6)*eid)
    return left,right


def state(g):return copy.deepcopy({'node_order':list(g.nodes),'edge_order':list(g.edges),'data':g.to_dict()})


@pytest.mark.parametrize('direction',['output_to_right','right_to_output'])
@pytest.mark.parametrize('field',['material_extra','material_tensor','curve','node_metadata','edge_metadata','position'])
def test_both_directions_independent(direction,field):
    left,right=fixture();left_before=state(left);out=left.merge(right)
    src,target=(out,right) if direction=='output_to_right' else (right,out)
    eid=200 if direction=='output_to_right' else 42
    nid=100 if direction=='output_to_right' else 90
    before=state(target)
    if field=='material_extra':src.edges[eid].material.extra['v'].append(999)
    elif field=='material_tensor':src.edges[eid].material.stiffness_tensor[0,0]=999
    elif field=='curve':src.edges[eid].internal_points[0,0]=999
    elif field=='node_metadata':src.nodes[nid].metadata['nested']['v'].append(999)
    elif field=='edge_metadata':src.edges[eid].metadata['nested']['v'].append(999)
    else:src.nodes[nid].position[0]=999
    assert state(target)==before
    assert state(left)==left_before


def test_topology_order_allocation_and_values_unchanged():
    left,right=fixture();before_left=state(left);before_right=state(right)
    out=left.merge(right)
    assert list(out.nodes)==[40,7,100,101]
    assert list(out.edges)==[99,200,201]
    assert (out._next_node_id,out._next_edge_id)==(102,202)
    assert [(e.node_i,e.node_j) for e in out.edges.values()]==[(40,7),(100,101),(101,100)]
    assert out.metadata==left.metadata and 'side' in out.metadata
    for source_id,new_id in [(42,200),(2,201)]:
        a,b=right.edges[source_id],out.edges[new_id]
        assert (a.radius,a.segments,a.material.to_dict(),a.metadata)==(b.radius,b.segments,b.material.to_dict(),b.metadata)
        np.testing.assert_array_equal(a.internal_points,b.internal_points)
    assert state(left)==before_left and state(right)==before_right
    assert out.add_edge(101,100)==200
    # Merge does not weld coincident positions; both input node populations remain.
    assert len(out.nodes)==4 and len(out.edges)==3


@pytest.mark.parametrize('points',[None,np.empty((0,3))])
def test_none_empty_curve_and_empty_rhs(points):
    left,right=fixture();right.edges[42].internal_points=points
    out=left.merge(right)
    if points is None:assert out.edges[200].internal_points is None
    else:assert out.edges[200].internal_points.shape==(0,3)
    assert state(left.merge(StructureGraph(dimension=3)))==state(left)


def test_shared_material_and_custom_deepcopy_follow_copy_policy():
    calls=[]
    class CustomMaterial(Material):
        def __deepcopy__(self,memo):
            calls.append(self.name)
            result=CustomMaterial(name=self.name)
            result.extra=copy.deepcopy(self.extra,memo)
            return result
    left,right=fixture();material=CustomMaterial(name='custom',extra={'v':[1]})
    right.edges[42].material=material;right.edges[2].material=material
    out=left.merge(right)
    assert calls==['custom','custom']
    assert out.edges[200].material is not out.edges[201].material
    out.edges[200].material.extra['v'].append(2)
    assert material.extra=={'v':[1]}
