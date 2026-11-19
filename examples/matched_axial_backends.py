"""Match an SI axial network in native LAMMPS and Taichi; see the companion guide."""
import argparse
import hashlib
import json
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
                duration_s=.0005 if side == 1 else .001)


def prepare_case(case, steps):
    ids = case['node_ids']
    return prepare_harmonic_beads(
        case['graph'], length_unit='m', maximum_bond_length_m=.003,
        masses_kg=dict(zip(ids, case['mass'])),
        bonds={i: HarmonicBond(float(k), float(r)) for i, k, r in
               zip(case['edge_ids'], case['stiffness_N_per_m'], case['rest'])},
        run=RunSpec(case['duration_s'] / steps, steps, 1,
                    ((-.002, .008), (-.002, .008), (-.002, .002))),
        fixed_node_ids=[ids[i] for i in case['fixed_rows']],
        forces_N={i: tuple(f) for i, f in zip(ids, case['force']) if np.any(f)},
        profile='harmonic_full_history_v1')


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


def run_pair(case, steps, runtime, output):
    import fibernet as fn
    output.mkdir()
    graph_text = json.dumps(case['graph'].to_dict(), sort_keys=True)
    (output/'graph.json').write_text(graph_text, encoding='utf-8')
    prepared = prepare_case(case, steps)
    task = LAMMPSTask(prepared, runtime, output/'native', ResourceLimits(
        cpu_seconds=30, wall_seconds=40., memory_bytes=2*1024**3,
        output_bytes=32*1024**2, file_size_bytes=32*1024**2))
    try:
        task.start()
        task.wait(timeout=45.)
    finally:
        if task.state == TaskState.RUNNING:
            task.cancel()
            task.wait(timeout=2.)
    if task.state != TaskState.SUCCEEDED:
        raise RuntimeError('native LAMMPS failed: ' + str(task.error))
    native = task.result
    ids = case['node_ids']
    state = fn.simulate(
        case['graph'], backend='taichi', mode='dynamics', arch='cpu', num_threads=1,
        spring_k=case['stiffness_N_per_m']*case['rest'], dashpot=0., drag=0.,
        dt=case['duration_s']/steps, num_steps=steps, save_interval=1,
        nodal_masses_by_id=dict(zip(ids, case['mass'])), record_state=True,
        state_units=dict(length='m', time='s', mass='kg', frame='world_xyz'),
        fixed_nodes=case['fixed_rows'].tolist(), external_force=case['force']).dynamics_state
    if list(state['node_ids']) != ids or any(list(f.node_ids) != ids for f in native.trajectory):
        raise RuntimeError('stable node IDs are not aligned')
    times = np.array([f.time_s for f in native.trajectory])
    if not np.allclose(times, state['times'], rtol=1e-14, atol=1e-18):
        raise RuntimeError('physical sample times are not aligned')
    arrays = dict(times_s=times, x0=case['x0'], edges=case['edges'],
                  force_N=case['force'], mass_kg=case['mass'], rest_m=case['rest'],
                  stiffness_N_per_m=case['stiffness_N_per_m'],
                  lammps_x=np.array([f.positions_m for f in native.trajectory]),
                  lammps_v=np.array([f.velocities_m_s for f in native.trajectory]),
                  taichi_x=np.asarray(state['positions']), taichi_v=np.asarray(state['velocities']),
                  lammps_potential_J=np.array([f.potential_J for f in native.thermo]),
                  lammps_kinetic_J=np.array([f.kinetic_J for f in native.thermo]),
                  taichi_potential_J=np.asarray(state['potential_energy']),
                  taichi_kinetic_J=np.asarray(state['kinetic_energy']))
    ref = reference(case, times)
    arrays['reference_x'] = ref
    np.savez_compressed(output/'states.npz', **arrays)
    scale = np.max(np.linalg.norm(ref-case['x0'], axis=2))
    return dict(steps=steps, dt_s=case['duration_s']/steps,
                prepared_identity=prepared.identity,
                graph_sha256=hashlib.sha256(graph_text.encode()).hexdigest(),
                states_sha256=hashlib.sha256((output/'states.npz').read_bytes()).hexdigest(),
                cross_backend_error_peak_normalized=float(np.max(np.linalg.norm(arrays['lammps_x']-arrays['taichi_x'], axis=2))/scale),
                **{b: metrics(case, arrays[b+'_x'], arrays[b+'_v'], ref) for b in ('lammps', 'taichi')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--runtime', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not args.execute:
        if args.runtime or args.output:
            parser.error('--runtime/--output require --execute')
        print(json.dumps({str(side): dict(nodes=len(build_case(side)['node_ids']),
              bonds=len(build_case(side)['edge_ids']), identity=prepare_case(build_case(side),256).identity)
              for side in (1,5)}, indent=2))
        return
    if args.runtime is None or args.output is None:
        parser.error('--execute requires --runtime and a new --output directory')
    data = json.loads(args.runtime.read_text(encoding='utf-8'))
    for name in ('file_sha256', 'environment'):
        data[name] = tuple(map(tuple, data[name]))
    runtime = RuntimeSpec(**data).checked()
    args.output.mkdir(parents=True, exist_ok=False)
    # Preserve the exact caller-supplied inventory; never provision a native engine here.
    (args.output/'runtime.json').write_bytes(args.runtime.read_bytes())
    report = dict(schema='fibernet.matched-axial-backends.v1', cases={})
    for side in (1,5):
        report['cases'][str(side)] = []
        for steps in (256,512,1024):
            row = run_pair(build_case(side), steps, runtime, args.output/f'case{side}_{steps}')
            report['cases'][str(side)].append(row)
            (args.output/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
