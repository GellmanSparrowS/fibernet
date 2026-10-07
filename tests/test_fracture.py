"""Synthetic software/analytical checks; no private research fixtures."""
from dataclasses import replace
import json
import math
import numpy as np
import pytest
from fibernet import StructureGraph, Material
from fibernet.sim.fracture import FractureConfig, FractureSolver, _beam, _engine, _sparse


def config(**kw):
    c=FractureConfig(1e9,.3,1e6,1e6,axis=0,maximum_displacement_m=.001,memory_budget_mb=768)
    return replace(c,**kw)


def chain(unit='m', sparse=False):
    g=StructureGraph(dimension=3)
    scale=1000 if unit=='mm' else 1
    mat=Material(youngs_modulus=1e9,poissons_ratio=.3)
    for x in (0,.05,.1): g.add_node([x*scale,0,0])
    g.add_edge(0,1,radius=.001*scale,material=mat)
    g.add_edge(1,2,radius=.001*scale,material=mat)
    if sparse:
        ghost=g.add_node([1,1,1]);g.remove_node(ghost)
        g.add_node([.2,0,0]);g.add_edge(2,4,radius=.001*scale,material=mat)
    return g


def solver(g=None,c=None,**kw):
    return FractureSolver(g or chain(),c or config(),fixed_nodes=[0],loaded_nodes=[2],length_unit='m',**kw)


def test_axial_analytic_and_prebreak():
    s=solver(); original=s.geometry.copy()
    r=s.run()
    assert r.termination_reason=='material_path_loss'
    assert len(r.events)==2
    first=[x for x in r.observations if x['phase']=='event_pre'][0]
    assert first['displacement_m']==pytest.approx(1e-4,rel=1e-9)
    assert first['loaded_reaction_force_N']==pytest.approx(math.pi,rel=1e-9)
    assert first['strain_energy_J']==pytest.approx(.5*math.pi*1e-4,rel=1e-9)
    assert r.data['state']['positions_m']['1'] is None
    assert r.data['state']['rotations_rad']['1'] is None
    assert r.data['scope']['contact_verified'] is False
    assert s.geometry==original


def test_loading_bound_no_failure():
    r=solver(c=config(normal_strength_Pa=1e12,shear_strength_Pa=1e12)).run()
    assert r.events==[]
    assert r.data['state']['displacement_m']==pytest.approx(.001)
    assert r.data['state']['loaded_reaction_force_N']==pytest.approx(10*math.pi)


def test_units_and_sparse_ids():
    a=solver().run()
    b=FractureSolver(chain('mm'),config(),fixed_nodes=[0],loaded_nodes=[2],length_unit='mm').run()
    assert a.events==b.events
    r=solver(g=chain(sparse=True)).run()
    assert '4' in r.data['state']['positions_m']


def test_element_rigid_modes_and_covariance():
    sec=_beam.circular_properties(.001,1e9,.3)
    a=np.array([0.,0.,0.]);b=np.array([.1,.02,.03])
    k=_beam.element_stiffness(a,b,sec,kappa=.88)
    assert np.allclose(k,k.T)
    for d in range(3):
        v=np.zeros(12);v[d]=v[d+6]=1
        assert np.linalg.norm(k@v)<1e-7
    for d in range(3):
        rot=np.eye(3)[d];v=np.r_[np.cross(rot,a),rot,np.cross(rot,b),rot]
        assert np.linalg.norm(k@v)<1e-7
    q=np.array([[0,1,0],[-1,0,0],[0,0,1]])
    t=np.kron(np.eye(4),q)
    k2=_beam.element_stiffness(q@a,q@b,sec,kappa=.88)
    assert np.allclose(k2,t@k@t.T,rtol=1e-10,atol=1e-7)


def test_cantilever_bending_and_torsion():
    radius=.001; length=.1; e=1e9;nu=.3;kappa=6*(1+nu)/(7+6*nu)
    sec=_beam.circular_properties(radius,e,nu)
    k=_beam.local_stiffness(length,sec,kappa=kappa)
    f=np.array([0.,1.,0.,0.,0.,0.])
    u=np.linalg.solve(k[6:,6:],f)
    expected=length**3/(3*e*sec['Iz'])+length/(kappa*sec['G']*sec['A'])
    assert u[1]==pytest.approx(expected,rel=1e-10)
    f=np.array([0.,0.,0.,1.,0.,0.]);u=np.linalg.solve(k[6:,6:],f)
    assert u[3]==pytest.approx(length/(sec['G']*sec['J']))


def test_neartie_predicate_records_actual_criterion():
    s=solver();g=s.geometry;p=s.parameters.copy()
    states=[]
    def fake(p,g,mask,delta,guard):
        members=[dict(id=sid,criterion=(1 if sid==0 else 1-5e-11),local_end_resultants=[0.]*12)
                 for (sid,_,_),active in zip(g['original_edges'],mask) if active]
        return dict(members=members,mask=mask)
    _,events,_=_engine.advance(p,g,g['initial_mask'],0.,solve=fake,observe=lambda *x:states.append(x))
    assert len(events)==2
    assert events[1]['criterion']<1
    assert any(x[0]=='event_pre' for x in states)


def test_resume_and_tamper(tmp_path):
    path=tmp_path/'checkpoint.json';s=solver();a=s.run(checkpoint=str(path))
    b=s.run(checkpoint=str(path),resume=True)
    assert a.events==b.events
    assert a.data['state']==b.data['state']
    with pytest.raises(FileExistsError):s.run(checkpoint=str(path))
    with pytest.raises(ValueError,match='mismatch'):
        solver(c=config(event_tol=1e-8)).run(checkpoint=str(path),resume=True)
    content=json.loads(path.read_text());content['payload']['state']['mask'][0]=True
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError,match='mismatch'):s.run(checkpoint=str(path),resume=True)


def test_refuse_parallel_and_heterogeneous():
    g=chain();g.add_edge(0,1,radius=.001,material=g.edges[0].material,allow_parallel=True)
    with pytest.raises(ValueError,match='Parallel'):solver(g=g)
    g=chain();g.edges[1].radius*=2
    with pytest.raises(ValueError,match='radii'):solver(g=g)
    g=chain();g.edges[1].material=Material(youngs_modulus=2e9,poissons_ratio=.3)
    with pytest.raises(ValueError,match='homogeneous'):solver(g=g)


def test_refuse_curved_and_bad_inputs():
    g=chain();g.edges[0].internal_points=np.array([[.025,.01,0.]])
    with pytest.raises(ValueError,match='Curved'):solver(g=g)
    with pytest.raises(ValueError):config(event_tol=1e-6)
    with pytest.raises(ValueError):config(axis=True)
    with pytest.raises(ValueError):solver(initially_broken_edges=[0])
    with pytest.raises(ValueError):solver(c=config(reference_displacement_m=1.))


def test_save_no_overwrite(tmp_path):
    r=solver().run();path=tmp_path/'result.json';r.save(path)
    assert json.loads(path.read_text())['schema']=='fibernet-linear-fracture-result-1'
    with pytest.raises(FileExistsError):r.save(path)


def test_reject_independent_shear_modulus_and_joint_types():
    g=chain();g.edges[0].material.shear_modulus=1e6
    with pytest.raises(ValueError,match='homogeneous'):solver(g=g)
    g=chain();g.nodes[1].metadata['joint_type']='hinged'
    with pytest.raises(ValueError,match='welded'):solver(g=g)


def test_cli_loader_material_complete_sparse_ids(tmp_path):
    from fibernet.sim.fracture.__main__ import load_geometry
    body=dict(schema='fibernet-fracture-geometry-1',
        nodes=[dict(id=i,position=[x,0,0]) for i,x in ((10,0),(20,.05),(30,.1))],
        edges=[dict(id=i,node_i=a,node_j=b,radius=.001,
                    material=dict(youngs_modulus=1e9,poissons_ratio=.3,shear_modulus=1e9/2.6))
               for i,a,b in ((100,10,20),(200,20,30))])
    path=tmp_path/'geometry.json';path.write_text(json.dumps(body))
    g=load_geometry(path)
    assert sorted(g.nodes)==[10,20,30] and sorted(g.edges)==[100,200]
    r=FractureSolver(g,config(),fixed_nodes=[10],loaded_nodes=[30],length_unit='m').run()
    assert {x['segment_id'] for x in r.events}=={100,200}
    body['edges'][0].pop('material');path.write_text(json.dumps(body))
    with pytest.raises(ValueError,match='material'):load_geometry(path)


def test_constructor_binding_mutation_is_rejected():
    s=solver();s.parameters['sigma_Pa']=1e10
    with pytest.raises(ValueError,match='changed'):s.run()


def test_checkpoint_pause_and_resume_epochs(tmp_path):
    g=StructureGraph(dimension=3);m=Material(youngs_modulus=1e9,poissons_ratio=.3)
    for p in ([0,0,0],[.04,.015,0],[.04,-.04,.01],[.1,0,0]):g.add_node(p)
    for a,b in ((0,1),(1,3),(0,2),(2,3)):g.add_edge(a,b,radius=.001,material=m)
    s=FractureSolver(g,config(maximum_displacement_m=.1),fixed_nodes=[0],loaded_nodes=[3],length_unit='m')
    whole=s.run();path=tmp_path/'pause.json';part=s.run(checkpoint=str(path),stop_after_epochs=1)
    assert part.termination_reason=='paused'
    resumed=s.run(checkpoint=str(path),resume=True)
    assert whole.events==resumed.events
    assert whole.observations==resumed.observations
    assert whole.data['state']==resumed.data['state']
