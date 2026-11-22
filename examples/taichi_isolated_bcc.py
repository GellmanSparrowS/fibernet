"""Generate and run a small 3D BCC spring lattice on an isolated Taichi CPU."""
import argparse
import json
from pathlib import Path
import re
import shutil

GRID = (7, 6, 5)
DT_S, STEPS, EA_N, WALL_S = 1e-4, 256, 4., 35
TIP_M, FORCE_N = (.014, .012, .010), (.2, -.15, .1)


def prepare():
    """Build the public graph and validate its request without starting a worker."""
    import numpy as np
    from fibernet import pattern_3d
    from fibernet.sim.isolated_taichi import prepare_request

    graph = pattern_3d(unit="bcc", box=(.002,) * 3, grid=GRID, radius=.0001)
    node_ids = list(graph.nodes)
    row_by_id = {node_id: row for row, node_id in enumerate(node_ids)}
    reference = np.array([graph.nodes[node_id].position for node_id in node_ids])
    fixed_ids = [node_id for node_id, p in zip(node_ids, reference)
                 if abs(p[0]) < 1e-12]
    load_ids = [node_id for node_id, p in zip(node_ids, reference)
                if np.allclose(p, TIP_M, atol=1e-12, rtol=0)]
    if (graph.num_nodes, graph.num_edges, len(fixed_ids), len(load_ids)) != (546, 2542, 42, 1):
        raise ValueError("Unexpected BCC mesh or support/load region")
    load_id, = load_ids
    forces = [[0., 0., 0.] for _ in node_ids]
    forces[row_by_id[load_id]] = list(FORCE_N)
    controls = dict(fixed_nodes=[row_by_id[node_id] for node_id in fixed_ids],
                    external_force=forces, spring_k=EA_N, dt=DT_S,
                    num_steps=STEPS, save_interval=1, dashpot=0., drag=0.)
    request = prepare_request(graph, **controls)
    info = dict(nodes=graph.num_nodes, rods=graph.num_edges, fixed_node_ids=fixed_ids,
                loaded_node_id=load_id, request_bytes=len(request), dt_s=DT_S, steps=STEPS,
                force_N=list(FORCE_N), spring_EA_N=EA_N)
    return graph, controls, info


def summarize(graph, controls, info, result):
    """Check the first unit-mass kick/drift step against the applied node force."""
    import numpy as np

    node_ids, edge_ids = list(graph.nodes), list(graph.edges)
    if result.metadata["node_ids"] != node_ids or result.metadata["edge_ids"] != edge_ids:
        raise ValueError("Saved result does not match input graph row order")
    frames = np.asarray(result.positions_trajectory)
    if frames.shape != (STEPS + 1, len(node_ids), 3) or not np.isfinite(frames).all():
        raise ValueError("Expected every saved XYZ frame of the BCC solve")
    fixed = frames[:, controls["fixed_nodes"]]
    if np.any(fixed != fixed[0]):
        raise ValueError("A held node moved")
    row = node_ids.index(info["loaded_node_id"])
    first = frames[1, row] - frames[0, row]
    # At rest springs start unstretched. For 1 kg, the first kick/drift
    # displacement is F*dt**2/1 kg; this is not a static force balance.
    expected = np.asarray(FORCE_N) * DT_S**2
    if not np.allclose(first, expected, rtol=0, atol=1e-14):
        raise ValueError("First-step displacement does not match the unit-mass reference")
    return dict(info, mode="completed", frames=len(frames),
                first_displacement_m=first.tolist(), expected_first_displacement_m=expected.tolist(),
                final_displacement_m=(frames[-1, row] - frames[0, row]).tolist(),
                axial_potential_energy_J=float(result.energy))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true",
                        help="print request details without a worker (also the default with no runtime arguments)")
    parser.add_argument("--worker-python", help="absolute executable path of a qualified Linux CPython 3.12 worker")
    parser.add_argument("--worker-site", help="absolute package directory containing the worker's taichi/ package")
    parser.add_argument("--native-sha256", help="trusted SHA256 of the Taichi native .so, not the wheel")
    opts = parser.parse_args(argv)
    runtime_args = (opts.worker_python, opts.worker_site, opts.native_sha256)
    if any(v is not None for v in runtime_args):
        if opts.preflight or not all(runtime_args):
            parser.error("supply all three runtime arguments for a run, or use --preflight alone")
        if not all(Path(p).is_absolute() for p in runtime_args[:2]):
            parser.error("worker paths must be absolute")
        if not re.fullmatch("[0-9a-f]{64}", opts.native_sha256):
            parser.error("the native SHA256 must be 64 lowercase hex digits")
    graph, controls, info = prepare()
    if not any(runtime_args):
        print(json.dumps(dict(info, mode="preflight", worker_started=False), indent=2))
        return
    from fibernet.sim.isolated_taichi import WorkerRuntime, start_isolated_taichi

    runtime = WorkerRuntime(*runtime_args)
    task = start_isolated_taichi(graph, runtime=runtime, wall_seconds=WALL_S, **controls)
    with task:
        result = task.result(timeout=WALL_S + 10)
        output = summarize(graph, controls, info, result)
        encoded = json.dumps(output, allow_nan=False, indent=2)
    # A failed result or close intentionally retains the task directory,
    # including logs from a broker that has not acknowledged cancellation.
    shutil.rmtree(task.workdir)
    print(encoded)


if __name__ == "__main__":
    main()
