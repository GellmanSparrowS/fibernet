"""Compiler, guarded protocol and optional pinned-native rotational BPM checks."""
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from fibernet import Material, StructureGraph, pattern_3d
from fibernet.integrations import (
    BPMBreak, BPMParticle, LAMMPSContractError, LAMMPSTask, ResourceLimits,
    RotationalBPMBond, RunSpec, RuntimeSpec, TaskState, prepare_rotational_bpm,
)
from fibernet.integrations import lammps_bpm as bpm
from fibernet.integrations.lammps_backend import validate_prepared


def _source_case():
    graph = StructureGraph(dimension=3, box_size=[100.,100.,100.])
    graph.metadata['name'] = 'BPM test geometry'
    for i, point in enumerate(((0,0,0),(20,0,0),(0,20,0),(0,0,20))):
        ghost = graph.add_node([-10-i,-10,-10])
        graph.remove_node(ghost)
        graph.add_node(point, boundary=(False,)*6, label=f'body {i}')
    nodes = list(graph.nodes)
    deleted = graph.add_edge(nodes[0],nodes[1],n_internal=0)
    graph.remove_edge(deleted)
    edges = [
        graph.add_edge(nodes[i],nodes[j],n_internal=0,radius=.29,
                       material=Material(name='provenance only', youngs_modulus=4.25e9),
                       source=[i,j])
        for i,j in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))
    ]
    kwargs = dict(
        length_unit='um', maximum_bond_length_m=60e-6,
        particles={n:BPMParticle(1e-12,4e-6) for n in nodes},
        bonds={sid:RotationalBPMBond(2.,.5,1e-14,4e-14,185e-9 if i==0 else 1e-3,
                                     1e-3,1e-9,1e-9)
               for i,sid in enumerate(edges)},
        run=RunSpec(1e-8,32,1,((-50e-6,80e-6),)*3),
        velocities_m_s={nodes[1]:(.5,0,0)},
        angular_velocities_rad_s={nodes[0]:(1e5,5e4,2e4)},
    )
    return graph, kwargs


def test_compiler_preserves_complete_graph_and_uses_explicit_si_tables():
    graph, kwargs = _source_case()
    snapshot = graph.to_dict()
    model = prepare_rotational_bpm(graph, checkpoint_step=20, **kwargs)
    assert validate_prepared(model) == model
    assert graph.to_dict()==snapshot
    assert json.loads(model.graph_json)==snapshot
    assert model.node_ids==(1,3,5,7) and model.edge_ids==(1,2,3,4,5,6)
    assert model.manifest()['schema']=='fibernet.lammps.rotational-bpm/1'
    assert model.manifest()['atom_to_node_id']=={'1':1,'2':3,'3':5,'4':7}
    assert model.manifest()['bond_type_to_edge_id']=={str(i):i for i in range(1,7)}
    assert 'units micro\n' in model.input_text
    command = next(row.split() for row in model.input_text.splitlines() if row.startswith('comm_modify cutoff '))
    assert float(command[2]) == pytest.approx(60.)
    assert command[3:] == ['vel','yes']
    first = next(row for row in model.input_text.splitlines() if row.startswith('bond_coeff 1 '))
    np.testing.assert_allclose(list(map(float,first.split()[2:])),
                               [2000.,500.,10.,40.,185.,1e6,1e6,1e6,0.,0.,0.,0.],rtol=2e-15)
    assert 'write_restart checkpoint.restart\nclear\nnewton on off\nread_restart checkpoint.restart\n' in model.input_text
    assert 'create_bonds single/bond 1 1 2' in model.input_text
    other = prepare_rotational_bpm(graph,**kwargs)
    assert 'read_restart' not in other.input_text
    graph.metadata['name']='later edit'
    assert json.loads(model.graph_json)==snapshot
    forged = replace(model,input_text=model.input_text+'print "foreign deck"\n')
    with pytest.raises(LAMMPSContractError,match='cannot be regenerated'):
        validate_prepared(forged)


@pytest.mark.parametrize('change, pattern', [
    ('bad-dt','timestep'), ('zero-steps','1–256'), ('bad-unit','length unit'),
    ('empty-mapping','particles'), ('checkpoint-end','checkpoint step'),
    ('short-reach','micrometre'), ('damping','damping'), ('parallel','unique'),
    ('microscale-floor','micrometre'),
])
def test_rejects_inputs_outside_the_bpm_pilot(change, pattern):
    graph, kwargs = _source_case()
    if change=='bad-dt':
        kwargs['run']=replace(kwargs['run'],timestep_s=float('nan'))
    elif change=='zero-steps':
        kwargs['run']=replace(kwargs['run'],steps=0)
    elif change=='bad-unit':
        kwargs['length_unit']='invalid'
    elif change=='empty-mapping':
        kwargs['particles']=None
    elif change=='checkpoint-end':
        kwargs['checkpoint_step']=32
    elif change=='short-reach':
        kwargs['maximum_bond_length_m']=10e-6
    elif change=='damping':
        kwargs['bonds'][1]=replace(kwargs['bonds'][1],radial_damping_N_s_per_m=1.)
    elif change=='parallel':
        edge=graph.add_edge(1,3,allow_parallel=True)
        kwargs['bonds'][edge]=kwargs['bonds'][1]
    elif change=='microscale-floor':
        graph.nodes[3].position[0]=.5
    with pytest.raises(LAMMPSContractError,match=pattern):
        prepare_rotational_bpm(graph,**kwargs)


def test_bpm_provider_requires_exact_release_and_pin(tmp_path, monkeypatch):
    graph, kwargs = _source_case()
    prepared=prepare_rotational_bpm(graph,**kwargs)
    binary=str(tmp_path/'native')
    runtime=RuntimeSpec('cli_serial_static',sys.executable,binary,expected_version=20260930,
                        expected_banner=bpm._NATIVE_BANNER,file_sha256=((binary,bpm._NATIVE_SHA256),))
    # The shared checks are still called for the correctly gated provider; this
    # test isolates the additional model-specific version/hash gate.
    seen=[]
    def ordinary_checks(self):
        seen.append(self)
        return self
    monkeypatch.setattr(RuntimeSpec,'checked',ordinary_checks)
    with pytest.raises(LAMMPSContractError,match='pinned official'):
        LAMMPSTask(prepared,replace(runtime,expected_version=20250722),tmp_path/'bad')
    with pytest.raises(LAMMPSContractError,match='pinned official'):
        LAMMPSTask(prepared,replace(runtime,file_sha256=((binary,'0'*64),)),tmp_path/'bad')
    assert not seen
    task=LAMMPSTask(prepared,runtime,tmp_path/'fresh')
    assert seen==[runtime]
    task.runtime=replace(runtime,expected_version=20250722)
    # Non-POSIX hosts reject native execution before the start-time provider check.
    expected = 'pinned official' if os.name == 'posix' else 'requires POSIX resource limits'
    with pytest.raises(LAMMPSContractError,match=expected):
        task.start()
    assert not task.workdir.exists()


def _stationary(path, *, reach_boundary=False, checkpoint=False, reach_override=None):
    graph=StructureGraph(dimension=3)
    left=graph.add_node([0.,0.,0.])
    point=(.007,.024,0.) if reach_boundary else (.02,0.,0.)
    right=graph.add_node(point)
    sid=graph.add_edge(left,right)
    native=[float(x*.001/1e-6) for x in point]
    # Native coordinates (7, 24, 0) have the exact binary64 norm 25.
    # Derive the SI bound from that norm, not a separately rounded SI norm.
    reach=math.dist((0.,0.,0.),native)*1e-6 if reach_boundary else .0001
    if reach_boundary:
        assert native == [7.,24.,0.] and reach/1e-6 == 25.
        # A separately rounded SI norm is one ULP above this same bound.
        assert math.dist((0.,0.,0.),[v*1e-6 for v in native]) > reach
    if reach_override is not None:
        reach=reach_override
    box=((-0.0002,0.0002),)*3
    prepared=prepare_rotational_bpm(
        graph,length_unit='mm',maximum_bond_length_m=reach,
        particles={n:BPMParticle(1e-12,4e-6) for n in (left,right)},
        bonds={sid:RotationalBPMBond(2,.5,1e-14,4e-14,1e-5,1e-5,1e-7,1e-7)},
        run=RunSpec(1e-8,2,1,box),checkpoint_step=1 if checkpoint else None,
    )
    def table(step, kind, cols, values):
        headers=(f'ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF {kind}\n{len(values)}\n'
                 'ITEM: BOX BOUNDS ff ff ff\n-200 200\n-200 200\n-200 200\n')
        name='ATOMS' if kind=='ATOMS' else 'ENTRIES'
        return headers+f'ITEM: {name} {cols}\n'+''.join(' '.join(map(str,row))+'\n' for row in values)
    atoms=[]
    for i,pos in enumerate(((0.,0.,0.),native),1):
        atoms.append([i,1,*pos,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1000,4])
    segments = (('',range(2)),('-after',range(1,3))) if checkpoint else (('',range(3)),)
    if checkpoint:
        atoms[0][15],atoms[0][18] = 1.,3.
        (path/'checkpoint.restart').write_bytes(b'owned test artifact')
    for suffix, steps in segments:
        (path/f'trajectory{suffix}.dump').write_text(
            'ITEM: UNITS\nmicro\n'+''.join(
                f'ITEM: TIME\n{step*.01:.17g}\n'+table(step,'ATOMS',bpm._ATOM_COLUMNS,atoms)
                for step in steps
            ),
            encoding='utf-8',
        )
        for name,cols,rows in (('active',bpm._EDGE_COLUMNS,[[1,1,2,1]]),('breaks',bpm._BREAK_COLUMNS,[])):
            (path/f'{name}{suffix}.dump').write_text(
                ''.join(table(step,'ENTRIES',cols,rows) for step in steps),encoding='utf-8')
    (path/'log.lammps').write_text(bpm._NATIVE_BANNER+'\n'+prepared.completion_marker+'\n',encoding='utf-8')
    return prepared


@pytest.mark.parametrize('damage, pattern', [
    ('units','expected micro'), ('time','time does not match'),
    ('nonfinite','nonfinite'), ('diameter','mass or diameter'),
    ('rupture','final topology'), ('tail','trailing'),
])
def test_rejects_damaged_bpm_outputs(tmp_path, damage, pattern):
    prepared=_stationary(tmp_path)
    file=tmp_path/('breaks.dump' if damage=='rupture' else 'trajectory.dump')
    text=file.read_text(encoding='utf-8')
    if damage=='units':
        text=text.replace('micro\n','si\n')
    elif damage=='time':
        text=text.replace('ITEM: TIME\n0.01\n','ITEM: TIME\n0.02\n',1)
    elif damage=='nonfinite':
        text=text.replace('1 1 0.0','1 1 nan',1)
    elif damage=='diameter':
        text=text.replace('1000 4','1000 7',1)
    elif damage=='rupture':
        text=text.replace('ITEM: TIMESTEP\n2\nITEM: NUMBER OF ENTRIES\n0',
                          'ITEM: TIMESTEP\n2\nITEM: NUMBER OF ENTRIES\n1')+'1 2 1 2\n'
    elif damage=='tail':
        text+='unexpected extra lines\n'
    file.write_text(text,encoding='utf-8')
    runtime=RuntimeSpec('cli_serial_static',sys.executable,expected_banner=bpm._NATIVE_BANNER)
    with pytest.raises(LAMMPSContractError,match=pattern):
        bpm.parse_task_output(prepared,tmp_path,runtime,1024**2)


def test_reach_equality_is_checked_in_native_coordinates(tmp_path):
    prepared=_stationary(tmp_path,reach_boundary=True)
    runtime=RuntimeSpec('cli_serial_static',sys.executable,expected_banner=bpm._NATIVE_BANNER)
    out=bpm.parse_task_output(prepared,tmp_path,runtime,1024**2)
    assert len(out.frames)==3 and out.frames[-1].active_edge_ids==(0,)

    reach=json.loads(prepared.contract_json)['maximum_bond_length_m']
    assert reach/1e-6 == 25.
    tighter=math.nextafter(reach,0.)
    assert tighter/1e-6 < 25.
    with pytest.raises(LAMMPSContractError,match='fit the declared reach'):
        _stationary(tmp_path,reach_boundary=True,reach_override=tighter)

    outside_y=math.nextafter(24.,math.inf)
    assert math.dist((0.,0.,0.),(7.,outside_y,0.)) > 25.
    path=tmp_path/'trajectory.dump'
    text=path.read_text()
    assert '2 1 7.0 24.0 0.0' in text
    path.write_text(text.replace('2 1 7.0 24.0 0.0',f'2 1 7.0 {outside_y!r} 0.0'))
    with pytest.raises(LAMMPSContractError,match='active bond left declared length domain'):
        bpm.parse_task_output(prepared,tmp_path,runtime,1024**2)


@pytest.mark.parametrize('updates, accepted', [
    ({15: math.nextafter(1., 2.),16: math.ulp(1.),18: math.nextafter(3., 4.)}, True),
    ({15: 1.000001}, False),
    ({20: 1e-6}, False),
    # These nonzero position and velocity changes survive flush-to-zero modes.
    ({2: sys.float_info.min}, False),
    ({5: sys.float_info.min}, False),
    # The stored quaternion starts at one. Even one ULP must fail at the seam.
    ({8: math.nextafter(1., 2.)}, False),
])
def test_restart_boundary_accepts_wrench_roundoff_only(tmp_path, updates, accepted):
    prepared=_stationary(tmp_path,checkpoint=True)
    file=tmp_path/'trajectory-after.dump'
    lines=file.read_text().splitlines()
    first_atom=next(i+1 for i,line in enumerate(lines) if line.startswith('ITEM: ATOMS '))
    fields=lines[first_atom].split()
    for column,value in updates.items():
        fields[column]=str(value)
    lines[first_atom]=' '.join(fields)
    file.write_text('\n'.join(lines)+'\n')
    runtime=RuntimeSpec('cli_serial_static',sys.executable,expected_banner=bpm._NATIVE_BANNER)
    if not accepted:
        with pytest.raises(LAMMPSContractError,match='restart boundary state differs'):
            bpm.parse_task_output(prepared,tmp_path,runtime,1024**2)
        return
    result=bpm.parse_task_output(prepared,tmp_path,runtime,1024**2)
    assert [f.step for f in result.frames]==[0,1,2]
    assert result.frames[1].forces_N[0]==(1e-9,0.,0.)
    assert result.frames[1].torques_N_m[0]==(3.*1e-15,0.,0.)


def _optional_native_runtime():
    inventory=os.environ.get('FIBERNET_TEST_BPM_RUNTIME_JSON')
    if not inventory:
        pytest.skip('supply an independently pinned official static runtime for native verification')
    raw=json.loads(Path(inventory).read_text())
    for key in ('environment','file_sha256'):
        raw[key]=tuple(map(tuple,raw[key]))
    return RuntimeSpec(**raw)


def test_optional_pinned_native_bpm_rupture_and_owned_checkpoint(tmp_path):
    runtime=_optional_native_runtime()
    graph, kwargs=_source_case()
    outputs=[]
    for split in (None,16,20,24):
        prepared=prepare_rotational_bpm(graph,checkpoint_step=split,**kwargs)
        task=LAMMPSTask(prepared,runtime,tmp_path/str(split),ResourceLimits(
            cpu_seconds=3,wall_seconds=5.,memory_bytes=1024**3,
            output_bytes=4*1024**2,file_size_bytes=4*1024**2))
        try:
            task.start()
            assert task.wait(timeout=7.)==TaskState.SUCCEEDED, task.error
        finally:
            if task.state==TaskState.RUNNING:
                task.cancel()
                task.wait(timeout=2.)
        result=task.result
        assert len(result.frames)==33 and result.node_ids==(1,3,5,7)
        assert result.breaks==(BPMBreak(20,2e-7,1,(1,3)),)
        assert result.frames[19].active_edge_ids==(1,2,3,4,5,6)
        assert result.frames[20].active_edge_ids==(2,3,4,5,6)
        if split is not None:
            owned = result.checkpoint
            raw_restart=(task.workdir/owned.filename).read_bytes()
            assert owned.step==split and owned.size_bytes==len(raw_restart)
            assert owned.sha256==hashlib.sha256(raw_restart).hexdigest()
        outputs.append(result)
    first=outputs[0]
    assert first.checkpoint is None
    assert all(o.frames==first.frames and o.breaks==first.breaks for o in outputs[1:])
    assert any(max(map(abs,t))>0 for f in first.frames for t in f.torques_N_m)
    mass=kwargs['particles'][1].mass_kg
    inertia=.4*mass*(kwargs['particles'][1].diameter_m/2)**2
    first_spin=np.asarray(first.frames[0].angular_velocities_rad_s)
    for frame in first.frames:
        force=np.asarray(frame.forces_N)
        torque=np.asarray(frame.torques_N_m)
        x=np.asarray(frame.positions_m)
        v=np.asarray(frame.velocities_m_s)
        w=np.asarray(frame.angular_velocities_rad_s)
        np.testing.assert_allclose(force.sum(axis=0),np.zeros(3),atol=1e-21)
        np.testing.assert_allclose((torque+np.cross(x,force)).sum(axis=0),np.zeros(3),atol=1e-24)
        np.testing.assert_allclose((mass*v).sum(axis=0),[mass*.5,0,0],rtol=1e-14,atol=1e-26)
        np.testing.assert_allclose((np.cross(x,mass*v)+inertia*w).sum(axis=0),
                                   inertia*first_spin.sum(axis=0),rtol=1e-8,atol=1e-30)


def test_optional_pinned_native_bpm_restarts_generated_octet_at_rupture(tmp_path):
    runtime=_optional_native_runtime()
    graph=pattern_3d(unit='octet',grid=(3,1,1),box=(40,40,40),n_internal=0)
    assert (len(graph.nodes),len(graph.edges))==(19,52)
    mass=1000*math.pi*4e-6**3/6
    particles={i:BPMParticle(mass*(1+.04*(i%5)),4e-6*(1+.03*(i%4))) for i in graph.nodes}
    bonds={i:RotationalBPMBond(.1+.01*(i%5),.05,5e-14,8e-14,5e-10 if i==0 else 1e-4,
                               1e-3,1e-8,1e-8) for i in graph.edges}
    velocities={}
    spins={}
    for i,node in graph.nodes.items():
        x,y,z=node.position
        velocities[i]=(.04+.003*x+.001*y,.02-.0013*y+.0012*z,-.03+.0009*x+.001*z)
        spins[i]=(2e4*(i%3-1),1.5e4*(i%5-2),1e4*(i%7-2))
    run=RunSpec(2e-9,48,1,((-50e-6,170e-6),(-50e-6,90e-6),(-50e-6,90e-6)))
    outputs=[]
    for checkpoint in (None,21):
        prepared=prepare_rotational_bpm(graph,length_unit='um',maximum_bond_length_m=65e-6,
                                         particles=particles,bonds=bonds,run=run,
                                         velocities_m_s=velocities,angular_velocities_rad_s=spins,
                                         checkpoint_step=checkpoint)
        task=LAMMPSTask(prepared,runtime,tmp_path/str(checkpoint),ResourceLimits(
            cpu_seconds=3,wall_seconds=5.,memory_bytes=1024**3,
            output_bytes=8*1024**2,file_size_bytes=8*1024**2))
        try:
            task.start()
            assert task.wait(timeout=7.)==TaskState.SUCCEEDED, task.error
        finally:
            if task.state==TaskState.RUNNING:
                task.cancel()
                task.wait(timeout=2.)
        outputs.append(task.result)
    full,split=outputs
    assert len(full.frames)==len(split.frames)==49
    assert full.breaks==split.breaks==(BPMBreak(21,21*run.timestep_s,0,(0,1)),)
    assert split.checkpoint.step==21
    assert split.frames[21]==full.frames[21]
    assert all(a.active_edge_ids==b.active_edge_ids for a,b in zip(full.frames,split.frames))
    for name,atol in (('positions_m',1e-18),('velocities_m_s',1e-14),('quaternions_wxyz',1e-15),
                      ('angular_velocities_rad_s',1e-9),('forces_N',1e-22),('torques_N_m',1e-26)):
        a=[getattr(f,name) for f in full.frames]
        b=[getattr(f,name) for f in split.frames]
        np.testing.assert_allclose(a,b,rtol=0,atol=atol)
