"""Replay saved electrothermal bindings without running any physics solver.

This checks decoding, graph binding and exact temperature transfer. It is not a
numerical reproduction or a performance benchmark. Run from the source checkout:
python -m benchmarks.saved_electrothermal_replay FILE --frames 0 5 10 --length-unit m
"""
import argparse
import hashlib
import json
from pathlib import Path

_MAX_FILE_BYTES = 64 * 1024 * 1024


def _file_sha256(path):
    digest = hashlib.sha256()
    size = 0
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            size += len(block)
            if size > _MAX_FILE_BYTES:
                raise ValueError('saved envelope exceeds 64 MiB')
            digest.update(block)
    return digest.hexdigest()


def replay_saved_electrothermal(path, *, frame_indices, length_unit, allow_partial=False):
    """Validate explicitly selected existing frames and their exact transfer."""
    if (type(frame_indices) not in (list, tuple) or not frame_indices
            or any(type(i) is not int or i < 0 for i in frame_indices)
            or len(set(frame_indices)) != len(frame_indices)):
        raise ValueError('frame_indices must contain distinct nonnegative integers')
    from fibernet.core.structure_graph import StructureGraph
    from fibernet.sim.saved_electrothermal import (
        read_saved_electrothermal_json, prepare_saved_electrothermal_temperature,
    )
    envelope_file_sha256 = _file_sha256(path)
    saved = read_saved_electrothermal_json(path)
    graph = StructureGraph.from_dict(saved['graph'])
    source = saved['source']
    frames = []
    for index in frame_indices:
        selected = prepare_saved_electrothermal_temperature(
            graph, saved, frame_index=index, length_unit=length_unit,
            allow_partial=allow_partial,
        )
        transferred = selected.mechanical_kwargs(graph)['temperature_K']
        expected = dict(zip(source['node_ids'], source['frames'][index]['temperature_K']))
        if (set(transferred) != set(expected)
                or any(float(transferred[n]).hex() != float(t).hex()
                       for n, t in expected.items())):
            raise ValueError('temperature transfer differs from the saved frame')
        frames.append({'frame_index': index, 'time_s': selected.to_dict()['time_s'],
                       'node_count': len(expected), 'exact_binary64_transfer': True})
    if _file_sha256(path) != envelope_file_sha256:
        raise ValueError('saved envelope file changed during replay')
    return {
        'schema': 'fibernet.saved_electrothermal_replay/1',
        'scope': 'saved-byte decoding, graph binding and exact temperature transfer only',
        'saved_envelope_file_sha256': envelope_file_sha256,
        'saved_canonical_content_sha256': saved['content_sha256'],
        'graph_sha256': saved['graph_sha256'],
        'source_completed': source['completed'],
        'source_frame_count': len(source['frames']),
        'node_count': len(source['node_ids']), 'edge_count': len(source['edge_ids']),
        'frames': frames,
    }


def replay_bundled_fixture(name, *, frame_indices, length_unit, allow_partial=False):
    """Replay one named bundled fixture through the unchanged strict JSON reader."""
    from tempfile import TemporaryDirectory
    from benchmarks.saved_electrothermal_fixtures import fixture_bytes
    raw = fixture_bytes(name)
    with TemporaryDirectory(prefix='fibernet-saved-fixture-') as directory:
        path = Path(directory) / name
        path.write_bytes(raw)
        report = replay_saved_electrothermal(
            path, frame_indices=frame_indices, length_unit=length_unit,
            allow_partial=allow_partial,
        )
    report['bundled_fixture'] = name
    return report

def main(argv=None):
    from benchmarks.saved_electrothermal_fixtures import FIXTURES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('saved_envelope', type=Path, nargs='?')
    parser.add_argument('--fixture', choices=tuple(FIXTURES))
    parser.add_argument('--frames', nargs='+', type=int, required=True)
    parser.add_argument('--length-unit', choices=('m', 'mm', 'um'), required=True)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args(argv)
    if (args.saved_envelope is None) == (args.fixture is None):
        parser.error('provide exactly one saved JSON path or --fixture')
    replay = replay_bundled_fixture if args.fixture is not None else replay_saved_electrothermal
    result = replay(
        args.fixture if args.fixture is not None else args.saved_envelope,
        frame_indices=args.frames, length_unit=args.length_unit,
        allow_partial=args.allow_partial,
    )
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
