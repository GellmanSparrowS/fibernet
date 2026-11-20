"""VTU must not manufacture observations or replace source record identities."""
from collections import UserList, deque
import json

import numpy as np
import pytest

from fibernet import Material, StructureGraph
from fibernet.io.meshio_adapter import to_meshio, write_vtu_bundle


def graph():
    g=StructureGraph(dimension=3)
    a=g.add_node([0.,0.,0.]);ghost=g.add_node([9.,9.,9.]);g.remove_node(ghost)
    b=g.add_node([1.,0.,0.]);g.add_node([2.,2.,0.])
    g.add_edge(a,b,radius=.1,material=Material(name='a'))
    g.add_edge(b,a,radius=.2,material=Material(name='b'),allow_parallel=True)
    return g


@pytest.mark.parametrize('field', ['position','boundary','box_size','radius','internal_points','node_id','edge_id','node_i'])
@pytest.mark.parametrize('masked', [True,False])
def test_masked_source_rejected_before_publication(tmp_path,field,masked):
    g=graph()
    if field=='position':g.nodes[0].position=np.ma.array([0.,0.,0.],mask=[masked,False,False])
    elif field=='boundary':g.nodes[0].boundary=np.ma.array([False]*6,mask=[masked]+[False]*5)
    # Preserve raw stored box data; the ordinary setter already coerces arrays.
    elif field=='box_size':g._box_size=np.ma.array([1.,1.,1.],mask=[masked,False,False])
    elif field=='radius':g.edges[0].radius=np.ma.array(.1,mask=masked)
    elif field=='internal_points':g.edges[0].internal_points=np.ma.array(np.empty((0,3)),mask=False)
    elif field=='node_id':g.nodes[0].node_id=np.ma.array(0,mask=masked)
    elif field=='edge_id':g.edges[0].edge_id=np.ma.array(0,mask=masked)
    elif field=='node_i':g.edges[0].node_i=np.ma.array(0,mask=masked)
    with pytest.raises(ValueError,match='masked arrays/scalars are unsupported'):
        write_vtu_bundle(tmp_path/'bundle',g,length_unit='mm')
    assert list(tmp_path.iterdir())==[]


@pytest.mark.parametrize('container', [list,tuple,deque,UserList,'object'])
@pytest.mark.parametrize('field',['position','boundary','box_size'])
def test_nested_masks_cannot_hide_inside_source_containers(tmp_path,container,field):
    g=graph();values=[np.ma.array(0.,mask=False),0.,0.]
    if field=='boundary':values=[np.ma.array(False,mask=False)]+[False]*5
    if container=='object':
        wrapped=np.empty(len(values),dtype=object)
        for i,v in enumerate(values):wrapped[i]=v
    else:wrapped=container(values)
    if field=='box_size':
        # Stored box data stays array-backed while holding a nested container.
        holder=np.empty(1,dtype=object);holder[0]=wrapped;g._box_size=holder
    else:setattr(g.nodes[0],field,wrapped)
    with pytest.raises(ValueError,match='masked arrays/scalars are unsupported'):
        write_vtu_bundle(tmp_path/'bundle',g,length_unit='m')
    assert list(tmp_path.iterdir())==[]


@pytest.mark.parametrize('field',['boundary','internal_points','box_size'])
def test_empty_masked_containers_are_rejected(field):
    g=graph();value=np.ma.array([],mask=False)
    if field=='internal_points':g.edges[0].internal_points=value
    elif field=='box_size':g._box_size=value
    else:g.nodes[0].boundary=value
    with pytest.raises(ValueError,match='masked arrays/scalars are unsupported'):to_meshio(g,length_unit='m')


@pytest.mark.parametrize('record',['node','edge'])
@pytest.mark.parametrize('bad',[True,-1,0.0,None,2**63,np.uint64(2**63)])
def test_record_identity_must_be_legal_int64(tmp_path,record,bad):
    g=graph()
    if record=='node':g.nodes[0].node_id=bad
    else:g.edges[0].edge_id=bad
    with pytest.raises(ValueError,match='record ID.*nonnegative int64'):
        write_vtu_bundle(tmp_path/'bundle',g,length_unit='m')
    assert list(tmp_path.iterdir())==[]


@pytest.mark.parametrize('record',['node','edge'])
def test_record_identity_must_match_dictionary_key(tmp_path,record):
    g=graph()
    if record=='node':g.nodes[0].node_id=900
    else:g.edges[0].edge_id=900
    with pytest.raises(ValueError,match='record ID.*dictionary key'):
        write_vtu_bundle(tmp_path/'bundle',g,length_unit='m')
    assert list(tmp_path.iterdir())==[]


def test_real_meshio_keeps_sparse_ids_isolates_parallel_edges_and_fields(tmp_path):
    meshio=pytest.importorskip('meshio');g=graph();data=g.to_dict();node_map={0:np.iinfo(np.int64).max,2:7,3:41}
    for n in data['nodes']:n['id']=int(node_map[n['id']])
    for e in data['edges']:e['node_i']=int(node_map[e['node_i']]);e['node_j']=int(node_map[e['node_j']]);e['id']=19 if e['id']==0 else 3
    data['nodes'].reverse();data['edges'].reverse();data.pop('next_node_id');data.pop('next_edge_id');g=StructureGraph.from_dict(data)
    for n in g.nodes.values():n.node_id=np.int64(n.node_id)
    for e in g.edges.values():e.edge_id=np.int64(e.edge_id);e.internal_points=np.empty((0,3))
    fields={i:[float(k),0.,0.] for k,i in enumerate(sorted(g.nodes))}
    out=write_vtu_bundle(tmp_path/'bundle',g,length_unit='mm',point_data_by_id={'u':fields},cell_data_by_id={'load':{3:2.,19:1.}},field_metadata={'point':{'u':{'units':'m','meaning':'translation'}},'cell':{'load':{'units':'N','meaning':'force'}}})
    mesh=meshio.read(out/'network.vtu');manifest=json.loads((out/'manifest.json').read_text())
    np.testing.assert_array_equal(mesh.point_data['node_id'],sorted(g.nodes));np.testing.assert_array_equal(mesh.cell_data['edge_id'][0],list(g.edges));assert len(mesh.points)==3 and len(mesh.cells[0].data)==2
    np.testing.assert_array_equal(mesh.point_data['u'],[fields[i] for i in sorted(g.nodes)]);np.testing.assert_array_equal(mesh.cell_data['load'][0],[2.,1.]);assert manifest['length_unit']=='mm' and manifest['coordinate_conversion']=='none'
    for edge_id,pair in zip(mesh.cell_data['edge_id'][0],mesh.cells[0].data):
        edge=g.edges[int(edge_id)];assert list(map(int,mesh.point_data['node_id'][pair]))==[edge.node_i,edge.node_j]


def test_plain_empty_internal_points_and_boundaries_keep_existing_behavior():
    pytest.importorskip('meshio');g=graph();g.edges[0].internal_points=np.empty((0,3));g.nodes[0].boundary=[]
    _,manifest=to_meshio(g,length_unit='m');assert manifest['node_boundary'][0]==[]


@pytest.mark.parametrize('location',['point','cell'])
def test_numeric_vector_memoryviews_keep_public_field_support(location):
    pytest.importorskip('meshio');g=graph();count=len(g.nodes) if location=='point' else len(g.edges)
    values=np.arange(count*3,dtype=float).reshape(count,3)
    kwargs={'length_unit':'m','field_metadata':{location:{'u':{'units':'m','meaning':'vector buffer'}}}}
    mesh,_=to_meshio(g,**kwargs,**{location+'_data':{'u':memoryview(values)}})
    observed=mesh.point_data['u'] if location=='point' else mesh.cell_data['u'][0]
    np.testing.assert_array_equal(observed,values)


def test_plain_geometry_and_boundary_memoryviews_keep_existing_support():
    pytest.importorskip('meshio');g=graph();g.nodes[0].position=memoryview(np.array([0.,0.,0.]));g.nodes[0].boundary=memoryview(np.array([False]*6));g.edges[0].internal_points=memoryview(np.empty((0,3)))
    mesh,manifest=to_meshio(g,length_unit='m');np.testing.assert_array_equal(mesh.points[0],[0.,0.,0.]);assert manifest['node_boundary'][0]==[False]*6
