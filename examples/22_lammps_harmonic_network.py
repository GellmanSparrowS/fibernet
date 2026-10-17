"""Prepare the representative SI network; native execution is explicit/optional.

Default: python examples/22_lammps_harmonic_network.py
Native: add --execute --runtime reviewed_runtime.json --workdir NEW_DIRECTORY
See docs/integrations/LAMMPS_HARMONIC_CLI.md before supplying a runtime.
"""
import argparse
import json
from pathlib import Path

from fibernet.core.structure_graph import StructureGraph
from fibernet.integrations import (HarmonicBond, RunSpec, RuntimeSpec, ResourceLimits,
                                   LAMMPSTask, TaskState, prepare_harmonic_beads)


def read_json(path):
    def invalid(value):raise ValueError('nonfinite JSON constant: '+value)
    return json.loads(Path(path).read_text(encoding='utf-8'),parse_constant=invalid)


def prepare_example():
    data=read_json(Path(__file__).with_name('data')/'lammps_braced_network_si.json')
    if data['schema']!='fibernet.example.explicit-harmonic-network.v1':
        raise ValueError('unsupported example schema')
    graph=StructureGraph.from_dict(data['graph'])
    run=dict(data['run']);run['box_m']=tuple(map(tuple,run['box_m']))
    return prepare_harmonic_beads(graph,length_unit=data['geometry_length_unit'],
        maximum_bond_length_m=data['maximum_bond_length_m'],
        masses_kg={int(n):m for n,m in data['masses_kg'].items()},
        bonds={int(e):HarmonicBond(**value) for e,value in data['bonds'].items()},run=RunSpec(**run),
        fixed_node_ids=tuple(data['fixed_node_ids']),
        forces_N={int(n):tuple(f) for n,f in data['forces_N'].items()},
        velocities_m_s={int(n):tuple(v) for n,v in data['velocities_m_s'].items()})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--runtime',type=Path)
    parser.add_argument('--workdir',type=Path)
    args=parser.parse_args()
    prepared=prepare_example()
    if not args.execute:
        if args.runtime or args.workdir:parser.error('--runtime/--workdir require --execute')
        print(json.dumps(dict(mode='prepared only',sites=len(prepared.node_ids),bonds=len(prepared.edge_ids),
                              steps=prepared.steps,identity=prepared.identity),sort_keys=True))
        return
    if args.runtime is None or args.workdir is None:parser.error('--execute requires --runtime and a new --workdir')
    payload=read_json(args.runtime)
    for name in ('file_sha256','environment'):payload[name]=tuple(map(tuple,payload[name]))
    runtime=RuntimeSpec(**payload)
    limits=ResourceLimits(cpu_seconds=3,wall_seconds=5.,memory_bytes=1024**3,
                          output_bytes=4*1024**2,file_size_bytes=8*1024**2)
    task=LAMMPSTask(prepared,runtime,args.workdir,limits)
    try:
        task.start();task.wait(timeout=7.)
    finally:
        if task.state==TaskState.RUNNING:task.cancel();task.wait(timeout=2.)
    if task.state!=TaskState.SUCCEEDED:raise RuntimeError('LAMMPS task failed: '+str(task.error))
    # Native raw outputs are already retained. Return a small typed-result summary;
    # do not duplicate the full trajectory into an unbudgeted extra artifact.
    print(json.dumps(dict(state=task.state.value,identity=task.result.identity,
        frames=len(task.result.trajectory),initial_energy_J=task.result.thermo[0].total_J,
        final_energy_J=task.result.thermo[-1].total_J),sort_keys=True))


if __name__=='__main__':main()
