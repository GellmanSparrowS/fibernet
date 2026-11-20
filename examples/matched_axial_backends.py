"""Match an SI axial network in native LAMMPS and Taichi; see the companion guide."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from fibernet.core.structure_graph import StructureGraph
from fibernet.integrations import (
    HarmonicBond, LAMMPSTask, ResourceLimits, RunSpec, RuntimeSpec, TaskState,
    prepare_harmonic_beads,
)


def build_case(side=5):
    """Return one shared SI model, with IDs kept separate from array rows."""
    if side not in (1, 5):
        raise ValueError('side must be 1 (analytic spring) or 5 (braced network)')
    graph = StructureGraph(dimension=3, tolerance=1e-12)
    points = ([(0., 0., 0.), (.001, 0., 0.)] if side == 1 else
              [(x * .001, y * .001, 0.) for y in range(side) for x in range(side)])
    ids = [graph.add_node(p) for p in points]
    if side == 1:
        graph.add_edge(ids[0], ids[1], n_internal=0)
    else:
        for y in range(side):
            for x in range(side):
                a = y * side + x
                for b in ([a + 1] if x + 1 < side else []) + \
                         ([a + side] if y + 1 < side else []) + \
                         ([a + side + 1] if x + 1 < side and y + 1 < side else []):
                    graph.add_edge(ids[a], ids[b], n_internal=0)
    return _shared_case(graph, .0005 if side == 1 else .001)


def _shared_case(graph, duration_s):
    ids = sorted(graph.nodes)
    x0 = np.array([graph.nodes[i].position for i in ids])
    row = {node_id: index for index, node_id in enumerate(ids)}
    edge_ids = list(graph.edges)
    edges = np.array([(row[e.node_i], row[e.node_j]) for e in graph.edges.values()])
    rest = np.linalg.norm(x0[edges[:, 1]] - x0[edges[:, 0]], axis=1)
    force = np.zeros_like(x0)
    force[x0[:, 0] == x0[:, 0].max(), 0] = 1e-7
    return dict(graph=graph, node_ids=ids, edge_ids=edge_ids, x0=x0, edges=edges,
                rest=rest, force=force, mass=np.full(len(ids), 1e-9),
                fixed_rows=np.flatnonzero(x0[:, 0] == 0.),
                stiffness_N_per_m=np.full(len(edges), .25),
                duration_s=duration_s)


def build_braced_3d_case():
    """Deterministic 6 x 5 x 4 grid: 120 nodes, 573 axial bonds, 20 loads."""
    graph = StructureGraph(dimension=3, tolerance=1e-12)
    ids = {(x, y, z): graph.add_node((x*.001, y*.001, z*.001))
           for z in range(4) for y in range(5) for x in range(6)}
    directions = ((1,0,0), (0,1,0), (0,0,1), (1,1,0),
                  (1,0,1), (0,1,1), (1,1,1))
    for (x,y,z), a in ids.items():
        for dx,dy,dz in directions:
            neighbor = (x+dx, y+dy, z+dz)
            if neighbor in ids:
                graph.add_edge(a, ids[neighbor], n_internal=0)
    case = _shared_case(graph, .001)
    case.update(profile='harmonic_sampled_v1',
                box_m=((-.002,.009), (-.002,.008), (-.002,.007)))
    return case


def output_stride(case, steps):
    if case.get('profile') == 'harmonic_sampled_v1':
        if type(steps) is not int or steps not in (256,512,1024):
            raise ValueError('3D example requires 256, 512 or 1024 steps')
        return steps//128
    return 1


def prepare_case(case, steps):
    ids = case['node_ids']
    return prepare_harmonic_beads(
        case['graph'], length_unit='m', maximum_bond_length_m=.003,
        masses_kg=dict(zip(ids, case['mass'])),
        bonds={i: HarmonicBond(float(k), float(r)) for i, k, r in
               zip(case['edge_ids'], case['stiffness_N_per_m'], case['rest'])},
        run=RunSpec(case['duration_s'] / steps, steps, 1,
                    case.get('box_m', ((-.002,.008), (-.002,.008), (-.002,.002)))),
        fixed_node_ids=[ids[i] for i in case['fixed_rows']],
        forces_N={i: tuple(f) for i, f in zip(ids, case['force']) if np.any(f)},
        profile=case.get('profile', 'harmonic_full_history_v1'),
        output_stride=output_stride(case, steps))


def reference(case, times):
    """Exact forced spring, or separately integrated nonlinear nodal equations."""
    x0, mass, force = (case[k] for k in ('x0', 'mass', 'force'))
    n = len(x0)
    if n == 2:
        k = case['stiffness_N_per_m'][0]
        w = np.sqrt(k / mass[1])
        x = np.broadcast_to(x0, (len(times), n, 3)).copy()
        x[:, 1, 0] += force[1, 0] / k * (1 - np.cos(w * times))
        return x
    from scipy.integrate import solve_ivp
    edges, rest, k = (case[v] for v in ('edges', 'rest', 'stiffness_N_per_m'))

    def rhs(t, state):
        x, v = state[:3*n].reshape(n, 3), state[3*n:].reshape(n, 3)
        delta = x[edges[:, 1]] - x[edges[:, 0]]
        length = np.linalg.norm(delta, axis=1)
        member_force = (k * (length - rest) / length)[:, None] * delta
        nodal = force.copy()
        np.add.at(nodal, edges[:, 0], member_force)
        np.add.at(nodal, edges[:, 1], -member_force)
        acceleration = nodal / mass[:, None]
        acceleration[case['fixed_rows']] = 0.
        return np.r_[v.ravel(), acceleration.ravel()]

    sol = solve_ivp(rhs, (0., float(times[-1])), np.r_[x0.ravel(), np.zeros(3*n)],
                    method='DOP853', t_eval=times, rtol=2e-12, atol=1e-16,
                    max_step=case['duration_s'] / 1024)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y[:3*n].T.reshape(-1, n, 3)


def metrics(case, x, v, ref):
    delta = x[:, case['edges'][:, 1]] - x[:, case['edges'][:, 0]]
    potential = .5 * np.sum(case['stiffness_N_per_m'] *
                           (np.linalg.norm(delta, axis=2) - case['rest'])**2, axis=1)
    kinetic = .5 * np.sum(case['mass'][None, :, None] * v*v, axis=(1, 2))
    work = np.sum(case['force'] * (x - case['x0']), axis=(1, 2))
    scale = np.max(np.linalg.norm(ref - case['x0'], axis=2))
    return dict(position_error_peak_normalized=float(np.max(np.linalg.norm(x-ref, axis=2))/scale),
                energy_work_residual_peak_normalized=float(np.max(np.abs(kinetic+potential-work))/np.max(np.abs(work))),
                fixed_error_m=float(np.max(np.abs(x[:, case['fixed_rows']] - case['x0'][case['fixed_rows']]))))


def aligned_frames(case, native, state, steps, stride):
    """Validate full native histories, then select the common output times."""
    ids = case['node_ids']
    expected = np.arange(steps+1)*case['duration_s']/steps
    if len(native.trajectory) != steps+1 or len(native.thermo) != steps+1:
        raise RuntimeError('native full-history frame counts are not aligned')
    for history in (native.trajectory, native.thermo):
        if [f.step for f in history] != list(range(steps+1)):
            raise RuntimeError('native integration steps are not aligned')
    if any(list(f.node_ids) != ids for f in native.trajectory) or list(state['node_ids']) != ids:
        raise RuntimeError('stable node IDs are not aligned')
    native_times = np.array([f.time_s for f in native.trajectory])
    if not np.allclose(native_times, expected, rtol=1e-14, atol=1e-18):
        raise RuntimeError('native physical times are not aligned')
    frames, thermo = native.trajectory[::stride], native.thermo[::stride]
    times = native_times[::stride]
    state_times = np.asarray(state['times'])
    if state_times.shape != times.shape or not np.allclose(times, state_times, rtol=1e-14, atol=1e-18):
        raise RuntimeError('physical sample times are not aligned')
    for key in ('positions', 'velocities'):
        if np.shape(state[key]) != (len(times), len(ids), 3):
            raise RuntimeError('Taichi '+key+' shape is not aligned')
    for key in ('potential_energy', 'kinetic_energy'):
        if np.shape(state[key]) != (len(times),):
            raise RuntimeError('Taichi '+key+' shape is not aligned')
    return frames, thermo, times


def load_runtime(path):
    """Accept a caller-reviewed RuntimeSpec inventory; never discover/install it."""
    data = json.loads(path.read_text(encoding='utf-8'))
    for name in ('file_sha256', 'environment'):
        data[name] = tuple(map(tuple, data[name]))
    return RuntimeSpec(**data).checked()


def run_pair(case, steps, runtime, output):
    import fibernet as fn
    output.mkdir()
    graph_text = json.dumps(case['graph'].to_dict(), sort_keys=True)
    (output/'graph.json').write_text(graph_text, encoding='utf-8')
    prepared = prepare_case(case, steps)
    task = LAMMPSTask(prepared, runtime, output/'native', ResourceLimits(
        cpu_seconds=30, wall_seconds=40., memory_bytes=2*1024**3,
        output_bytes=32*1024**2, file_size_bytes=32*1024**2))
    started = time.perf_counter()
    try:
        task.start()
        task.wait(timeout=45.)
    finally:
        if task.state == TaskState.RUNNING:
            task.cancel()
            task.wait(timeout=2.)
    if task.state != TaskState.SUCCEEDED:
        raise RuntimeError('native LAMMPS failed: ' + str(task.error))
    native_wall_s = time.perf_counter()-started
    native = task.result
    ids = case['node_ids']
    stride = output_stride(case, steps)
    started = time.perf_counter()
    state = fn.simulate(
        case['graph'], backend='taichi', mode='dynamics', arch='cpu', num_threads=1,
        spring_k=case['stiffness_N_per_m']*case['rest'], dashpot=0., drag=0.,
        dt=case['duration_s']/steps, num_steps=steps, save_interval=stride,
        nodal_masses_by_id=dict(zip(ids, case['mass'])), record_state=True,
        state_units=dict(length='m', time='s', mass='kg', frame='world_xyz'),
        fixed_nodes=case['fixed_rows'].tolist(), external_force=case['force']).dynamics_state
    taichi_wall_s = time.perf_counter()-started
    frames, thermo, times = aligned_frames(case, native, state, steps, stride)
    arrays = dict(times_s=times, node_ids=np.array(ids), edge_ids=np.array(case['edge_ids']),
                  fixed_rows=case['fixed_rows'], x0=case['x0'], edges=case['edges'],
                  force_N=case['force'], mass_kg=case['mass'], rest_m=case['rest'],
                  stiffness_N_per_m=case['stiffness_N_per_m'],
                  lammps_x=np.array([f.positions_m for f in frames]),
                  lammps_v=np.array([f.velocities_m_s for f in frames]),
                  taichi_x=np.asarray(state['positions']), taichi_v=np.asarray(state['velocities']),
                  lammps_potential_J=np.array([f.potential_J for f in thermo]),
                  lammps_kinetic_J=np.array([f.kinetic_J for f in thermo]),
                  taichi_potential_J=np.asarray(state['potential_energy']),
                  taichi_kinetic_J=np.asarray(state['kinetic_energy']))
    started = time.perf_counter()
    ref = reference(case, times)
    reference_wall_s = time.perf_counter()-started
    arrays['reference_x'] = ref
    np.savez_compressed(output/'states.npz', **arrays)
    scale = np.max(np.linalg.norm(ref-case['x0'], axis=2))
    errors = {b: metrics(case, arrays[b+'_x'], arrays[b+'_v'], ref) for b in ('lammps', 'taichi')}
    for b in errors:
        errors[b]['position_error_absolute_m'] = float(np.max(np.linalg.norm(arrays[b+'_x']-ref, axis=2)))
    return dict(nodes=len(ids), bonds=len(case['edge_ids']),
                native_frames=len(native.trajectory), output_frames=len(times),
                native_file_bytes={p.name:p.stat().st_size for p in (output/'native').iterdir() if p.is_file()},
                states_file_bytes=(output/'states.npz').stat().st_size,
                output_forecast=task.output_forecast,
                native_task_wall_s=native_wall_s, taichi_call_wall_s=taichi_wall_s,
                reference_wall_s=reference_wall_s,
                timing_scope='Native: startup, integration, every-step I/O and parsing; Taichi: initialization/JIT, integration and state extraction.',
                reference_peak_displacement_m=float(scale),
                cross_backend_error_absolute_m=float(np.max(np.linalg.norm(arrays['lammps_x']-arrays['taichi_x'], axis=2))),
                steps=steps, dt_s=case['duration_s']/steps,
                prepared_identity=prepared.identity,
                graph_sha256=hashlib.sha256(graph_text.encode()).hexdigest(),
                states_sha256=hashlib.sha256((output/'states.npz').read_bytes()).hexdigest(),
                cross_backend_error_peak_normalized=float(np.max(np.linalg.norm(arrays['lammps_x']-arrays['taichi_x'], axis=2))/scale),
                **errors)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--runtime', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--case', choices=('original','braced3d'), default='original')
    parser.add_argument('--steps', nargs='+', type=int, choices=(256,512,1024), default=(256,512,1024))
    args = parser.parse_args(argv)
    if len(set(args.steps)) != len(args.steps):
        parser.error('--steps must not contain duplicate resolutions')
    cases = ({'braced3d': build_braced_3d_case()} if args.case == 'braced3d' else
             {str(side): build_case(side) for side in (1,5)})
    if not args.execute:
        if args.runtime or args.output:
            parser.error('--runtime/--output require --execute')
        print(json.dumps({name: dict(nodes=len(case['node_ids']), bonds=len(case['edge_ids']),
              plans=[dict(steps=n, output_stride=output_stride(case,n),
                          output_frames=n//output_stride(case,n)+1,
                          identity=prepare_case(case,n).identity) for n in args.steps])
              for name,case in cases.items()}, indent=2))
        return
    if args.runtime is None or args.output is None:
        parser.error('--execute requires --runtime and a new --output directory')
    runtime = load_runtime(args.runtime)
    args.output.mkdir(parents=True, exist_ok=False)
    # Preserve the exact caller-supplied inventory; never provision a native engine here.
    (args.output/'runtime.json').write_bytes(args.runtime.read_bytes())
    report = dict(schema='fibernet.matched-axial-backends.v1', cases={})
    for name,case in cases.items():
        report['cases'][name] = []
        for steps in args.steps:
            row = run_pair(case, steps, runtime, args.output/f'case{name}_{steps}')
            report['cases'][name].append(row)
            (args.output/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
