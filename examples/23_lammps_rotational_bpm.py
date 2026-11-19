"""Prepare an explicit rotational BPM model, or execute with a pinned runtime."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from fibernet import StructureGraph
from fibernet.integrations import (
    BPMParticle, LAMMPSTask, ResourceLimits, RotationalBPMBond, RunSpec, RuntimeSpec,
    TaskState, prepare_rotational_bpm,
)


def prepare_example(checkpoint_step=None):
    graph = StructureGraph(dimension=3)
    nodes = [graph.add_node(point) for point in ((0,0,0),(20,0,0),(0,20,0),(0,0,20))]
    edges = [graph.add_edge(nodes[i], nodes[j]) for i,j in
             ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))]
    particles = {node:BPMParticle(mass_kg=1e-12, diameter_m=4e-6) for node in nodes}
    bond = RotationalBPMBond(
        radial_stiffness_N_per_m=2., shear_stiffness_N_per_m=.5,
        twist_stiffness_N_m_per_rad=1e-14, bend_stiffness_N_m_per_rad=4e-14,
        radial_limit_N=1e-3, shear_limit_N=1e-3, twist_limit_N_m=1e-9, bend_limit_N_m=1e-9,
    )
    bonds = {edge:bond for edge in edges}
    bonds[edges[0]] = RotationalBPMBond(2., .5, 1e-14, 4e-14, 185e-9, 1e-3, 1e-9, 1e-9)
    return prepare_rotational_bpm(
        graph, length_unit='um', maximum_bond_length_m=60e-6,
        particles=particles, bonds=bonds, run=RunSpec(1e-8, 32, 1, ((-50e-6, 80e-6),)*3),
        velocities_m_s={nodes[1]:(.5,0.,0.)},
        angular_velocities_rad_s={nodes[0]:(1e5,5e4,2e4)}, checkpoint_step=checkpoint_step,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-step', type=int)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--runtime', type=Path)
    parser.add_argument('--workdir', type=Path)
    args = parser.parse_args()
    if args.execute != bool(args.runtime and args.workdir) or (
            not args.execute and (args.runtime or args.workdir)):
        parser.error('--execute requires both --runtime and a new --workdir')
    prepared = prepare_example(args.checkpoint_step)
    if not args.execute:
        print(json.dumps(dict(mode='prepared only', identity=prepared.identity,
                              sites=len(prepared.node_ids), bonds=len(prepared.edge_ids),
                              steps=prepared.steps, checkpoint_step=args.checkpoint_step), sort_keys=True))
        return
    payload = json.loads(args.runtime.read_text(encoding='utf-8'))
    for name in ('file_sha256','environment'):
        payload[name] = tuple(map(tuple, payload[name]))
    runtime = RuntimeSpec(**payload)
    limits = ResourceLimits(cpu_seconds=3, wall_seconds=5., memory_bytes=1024**3,
                            output_bytes=4*1024**2, file_size_bytes=4*1024**2)
    task = LAMMPSTask(prepared, runtime, args.workdir, limits)
    try:
        task.start()
        task.wait(timeout=7.)
    finally:
        if task.state == TaskState.RUNNING:
            task.cancel()
            task.wait(timeout=2.)
    if task.state != TaskState.SUCCEEDED:
        raise RuntimeError(task.error)
    result = task.result
    print(json.dumps(dict(state=task.state.value, frames=len(result.frames), identity=result.identity,
                          breaks=[asdict(event) for event in result.breaks],
                          checkpoint=asdict(result.checkpoint) if result.checkpoint else None),sort_keys=True))


if __name__ == '__main__':
    main()
