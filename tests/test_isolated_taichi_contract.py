"""Request and graph interoperability tests without native execution."""
import json
import pytest
from fibernet.core.structure_graph import StructureGraph
from fibernet.sim.isolated_taichi import prepare_request, start_isolated_taichi

def graph():
    return StructureGraph.from_dict(dict(version='3.1',dimension=2,
        nodes=[dict(id=9,position=[0,0,0]),dict(id=2,position=[1,0,0]),dict(id=7,position=[1,1,0])],
        edges=[dict(id=18,node_i=9,node_j=2),dict(id=4,node_i=2,node_j=7),dict(id=11,node_i=9,node_j=2)]))

def controls():
    return dict(fixed_nodes=[0],external_force=[[0.,0.,0.],[.1,0.,0.],[0.,.1,0.]],spring_k=1.,dt=.001,num_steps=10,save_interval=5)

def test_order_parallel_edges_and_rows():
    g=graph();raw=prepare_request(g,**controls());req=json.loads(raw)
    assert req['node_ids']==[9,2,7] and req['edge_ids']==[18,4,11]
    rebuilt=StructureGraph.from_dict(req['graph'])
    assert list(rebuilt.nodes)==[9,2,7] and list(rebuilt.edges)==[18,4,11]
    assert req['controls']['external_force'][1]==[.1,0.,0.]
    assert prepare_request(rebuilt,**req['controls'])==raw
    assert list(g.nodes)==[9,2,7] # caller was not reordered

@pytest.mark.parametrize('name,value',[('state_observer',lambda _:None),('material',object()),('displacement_schedule',{})])
def test_unsupported_keyword_rejected_before_launch(name,value,monkeypatch):
    def no_launch(*a,**kw):raise AssertionError('must not launch')
    monkeypatch.setattr('fibernet.sim.isolated_taichi.subprocess.Popen',no_launch)
    with pytest.raises(TypeError):start_isolated_taichi(graph(),runtime=None,**controls(),**{name:value})

@pytest.mark.parametrize('name,value',[('fixed_nodes',[9]),('num_steps',True),('dt',float('nan')),('save_interval',0),('external_force',[[0.,0.]])])
def test_invalid_controls(name,value):
    c=controls();c[name]=value
    with pytest.raises(ValueError):prepare_request(graph(),**c)

def test_custom_metadata_rejected():
    g=graph();g.nodes[9].metadata['custom']=object()
    with pytest.raises(ValueError):prepare_request(g,**controls())

def test_curve_rejected():
    import numpy as np
    g=graph();g.edges[18].internal_points=np.array([[.5,.1,0.]])
    with pytest.raises(ValueError):prepare_request(g,**controls())

def test_record_identity_rejected():
    g=graph();g.nodes[9].node_id=42
    with pytest.raises(ValueError):prepare_request(g,**controls())

def test_signed_id_rejected():
    g=graph();node=g.nodes.pop(9);node.node_id=-9;g.nodes[-9]=node
    with pytest.raises(ValueError):prepare_request(g,**controls())

def test_integer_coordinates_rejected_before_launch():
    import numpy as np
    g=graph();g.nodes[9].position=np.array([0,0,0],dtype=np.int64)
    with pytest.raises(ValueError,match='float64'):prepare_request(g,**controls())

def test_material_normalization_and_extra_attribute_rejected():
    g=graph();g.edges[18].material.density=1000
    with pytest.raises(ValueError,match='binary64'):prepare_request(g,**controls())
    g=graph();g.edges[18].material.unlisted=1.0
    with pytest.raises(ValueError,match='undeclared'):prepare_request(g,**controls())

def test_malformed_empty_curve_rejected():
    import numpy as np
    g=graph();g.edges[18].internal_points=np.empty((0,2))
    with pytest.raises(ValueError):prepare_request(g,**controls())

def test_metadata_cycle_and_depth_bounded():
    g=graph();cycle=[];cycle.append(cycle);g.nodes[9].metadata['cycle']=cycle
    with pytest.raises(ValueError,match='cyclic'):prepare_request(g,**controls())
    g=graph();nested=[];top=nested
    for _ in range(20):child=[];nested.append(child);nested=child
    g.nodes[9].metadata['deep']=top
    with pytest.raises(ValueError,match='budget'):prepare_request(g,**controls())

def test_reverse_parallel_nonfirst_fixed_and_immutability():
    import copy
    g=graph();g.edges[11].node_i=2;g.edges[11].node_j=9
    c=controls();c['fixed_nodes']=[1];c['external_force']=[[.3,0.,0.],[0.,0.,0.],[0.,.7,0.]]
    before=copy.deepcopy(g.to_dict());materials={i:copy.deepcopy(vars(e.material)) for i,e in g.edges.items()}
    req=json.loads(prepare_request(g,**c));rebuilt=StructureGraph.from_dict(req['graph'])
    assert list(rebuilt.edges)==[18,4,11]
    assert (rebuilt.edges[11].node_i,rebuilt.edges[11].node_j)==(2,9)
    assert req['controls']['fixed_nodes']==[1] and req['controls']['external_force']==c['external_force']
    assert g.to_dict()==before and {i:vars(e.material) for i,e in g.edges.items()}==materials

def test_ordinary_constructor_and_default_material_supported():
    g=StructureGraph(dimension=2)
    a=g.add_node([0.,0.],merge=False);b=g.add_node([1.,0.],merge=False)
    g.add_edge(a,b)
    req=prepare_request(g,fixed_nodes=[0],external_force=[[0.,0.,0.],[.1,0.,0.]],spring_k=1.,dt=.001,num_steps=10,save_interval=5)
    assert json.loads(req)['node_ids']==[a,b]

def test_material_recomputation_is_rejected_before_launch():
    g=graph();g.edges[18].material.shear_modulus=None
    with pytest.raises(ValueError,match='fixed point'):prepare_request(g,**controls())

@pytest.mark.parametrize('bad',[True,1.0])
def test_record_id_type_not_just_numeric_equality(bad):
    g=StructureGraph(dimension=2)
    g.add_node([0.,0.],merge=False);g.add_node([1.,0.],merge=False);g.add_edge(0,1)
    g.nodes[1].node_id=bad
    c=dict(fixed_nodes=[0],external_force=[[0.,0.,0.],[.1,0.,0.]],spring_k=1.,dt=.001,num_steps=2,save_interval=1)
    with pytest.raises(ValueError):prepare_request(g,**c)
    g.nodes[1].node_id=1
    # A separate edge keyed 1 makes bool/float equal under loose comparison.
    e=g.edges.pop(0);g.edges[1]=e;e.edge_id=bad;g._next_edge_id=2
    with pytest.raises(ValueError):prepare_request(g,**c)
