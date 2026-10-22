"""Inspect a saved temperature frame; solve only with explicit --solve.

The mechanics JSON must supply profile, length_unit, E_Pa,
expansion_alpha_per_K, stress_free_temperature_K, constraints, optional loads.
No electrical or thermal solver is called. Existing output files are refused.
"""
import argparse
import json
from pathlib import Path


def _id_mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError(name + ' must be a JSON object keyed by stable IDs')
    result = {}
    for key, item in value.items():
        if not key.isdecimal() or str(int(key)) != key or not 0 <= int(key) < 2**63:
            raise ValueError(name + ' contains a noncanonical stable ID')
        result[int(key)] = item
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph', required=True)
    parser.add_argument('--thermal', required=True)
    parser.add_argument('--mechanics', required=True)
    parser.add_argument('--frame', required=True, type=int)
    parser.add_argument('--solve', action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args(argv)
    if args.solve and not args.output:
        parser.error('--solve requires --output')
    if args.output and Path(args.output).exists():
        parser.error('output exists; choose a new path')
    from fibernet import StructureGraph
    from fibernet.sim.saved_temperature import read_saved_thermal_json, prepare_saved_temperature
    graph = StructureGraph.load_json(args.graph)
    mechanical = read_saved_thermal_json(args.mechanics)
    allowed = {'profile', 'length_unit', 'E_Pa', 'expansion_alpha_per_K',
               'stress_free_temperature_K', 'constraints', 'loads'}
    if set(mechanical) - allowed or allowed - {'loads'} - set(mechanical):
        raise ValueError('mechanics JSON requires exactly the documented inputs')
    frame = prepare_saved_temperature(graph, read_saved_thermal_json(args.thermal),
                                      frame_index=args.frame,
                                      length_unit=mechanical.pop('length_unit'))
    if not args.solve:
        print(json.dumps(frame.to_dict(), indent=2, allow_nan=False))
        return 0
    for name in ('expansion_alpha_per_K', 'constraints', 'loads'):
        if name in mechanical: mechanical[name] = _id_mapping(mechanical[name], name)
    # Reserve the output before running mechanics so another user's existing
    # result can never be overwritten after a physical computation.
    with Path(args.output).open('x', encoding='utf-8') as stream:
        result = frame.solve(graph, **mechanical)
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
