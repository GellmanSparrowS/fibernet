"""Experimental offline 3-D saved-frame to prescribed-temperature mechanics.

The exact complete saved envelope is checked before a fresh, one-way solve.
It is not a restart, authenticity check or reciprocal coupling contract.
"""
from hashlib import sha256
import os
from pathlib import Path
import re
import stat
from typing import NamedTuple, Tuple

from .thermoelastic_3d import ThermoelasticSnapshot3D

__all__ = ["solve_saved_electrothermal_frame_3d"]

_MAX_ARCHIVE_BYTES = 4 * 1024**2
_MAX_NODES, _MAX_EDGES = 96, 192
_MAX_FRAMES, _MAX_ITERATION_ROWS = 64, 64 * 25


class _Saved3DReceipt(NamedTuple):
    file_sha256: str
    caller_byte_pin_checked: bool
    content_sha256: str
    graph_sha256: str
    reported_native_input_sha256: str
    selected_electrical_input_sha256: str
    frame_index: int
    frame_time_s: float
    source_frame_count: int
    length_unit: str
    temperatures_K_by_node_id: Tuple[Tuple[int, float], ...]


class _Saved3DRun(NamedTuple):
    snapshot: ThermoelasticSnapshot3D
    receipt: _Saved3DReceipt


def _path_fd_identity(info):
    # On Windows, path-based ctime may be birth time while fd-based ctime is
    # change time. Full mode bits can also depend on the provider. Compare
    # those only within the same provider; both endpoints must be regular.
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def _file_state(info):
    return (*_path_fd_identity(info), info.st_mode, info.st_ctime_ns)


def _read_bounded(path):
    # Reject nonregular/symlink leaves before opening. Recheck the owned fd,
    # and use nonblocking/no-follow flags where the platform provides them.
    path = Path(path)
    stated = path.lstat()
    if not stat.S_ISREG(stated.st_mode):
        raise ValueError("saved 3-D electrothermal input must be a regular file")
    if not 1 <= stated.st_size <= _MAX_ARCHIVE_BYTES:
        raise ValueError("saved 3-D electrothermal file exceeds the 4 MiB bound or is empty")
    flags = os.O_RDONLY
    for name in ("O_CLOEXEC", "O_NONBLOCK", "O_NOFOLLOW", "O_BINARY"):
        flags |= getattr(os, name, 0)
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        if (not stat.S_ISREG(opened.st_mode)
                or not 1 <= opened.st_size <= _MAX_ARCHIVE_BYTES
                or _path_fd_identity(stated) != _path_fd_identity(opened)):
            raise ValueError("saved 3-D electrothermal file changed before the bounded read")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            data = stream.read(_MAX_ARCHIVE_BYTES + 1)
        if (not 1 <= len(data) <= _MAX_ARCHIVE_BYTES
                or len(data) != opened.st_size
                or _file_state(os.fstat(fd)) != _file_state(opened)
                or _file_state(path.lstat()) != _file_state(stated)):
            raise ValueError("saved 3-D electrothermal file changed during the bounded read")
    finally:
        os.close(fd)
    return data


def _bounded_list(record, key, minimum, maximum):
    values = record.get(key)
    if type(values) is not list or not minimum <= len(values) <= maximum:
        raise ValueError(f"saved 3-D electrothermal {key} exceeds the prototype shape bounds")
    return values


def solve_saved_electrothermal_frame_3d(
    graph, saved_path, *, frame_index, stress_free_temperature_K,
    expansion_alpha_per_K_by_edge_id, constraints_by_node_id,
    loads_N_by_node_id=None, length_unit, applicability_limits,
    expected_file_sha256=None,
) -> _Saved3DRun:
    """Validate a complete local archive and solve just one selected 3-D frame.

    Only regular trusted-local files up to 4 MiB and complete source runs
    (up to 64 frames) are admitted. No concurrent file or graph mutation.
    An optional lowercase SHA256 pins exact input bytes, not authenticity.
    The existing saved owner validates every frame and full graph record.
    Mechanics then receives a detached graph and SI controls (geometry only
    in the declared unit), independent thermal expansion coefficients and
    positive stress-free Kelvin. Electrical temperature slopes are not used
    as expansion coefficients. No feedback, source solve or interpolation.
    """
    from fibernet.core.structure_graph import StructureGraph
    from .saved_electrothermal import _decode, _graph_record, _hash, _validate
    from .thermoelastic_3d import solve_thermoelastic_network_snapshot_3d

    if (type(graph) is not StructureGraph or graph.dimension != 3
            or not 2 <= len(graph.nodes) <= _MAX_NODES
            or not 1 <= len(graph.edges) <= _MAX_EDGES):
        raise ValueError("saved 3-D mechanics needs a bounded native 3-D graph")
    if type(length_unit) is not str or length_unit not in ("m", "mm", "um"):
        raise ValueError("length_unit must be explicitly m, mm or um")
    if type(frame_index) is not int or not 0 <= frame_index < _MAX_FRAMES:
        raise ValueError("frame_index must explicitly select a bounded accepted frame")
    if expected_file_sha256 is not None and (
        type(expected_file_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", expected_file_sha256) is None
    ):
        raise ValueError("expected_file_sha256 must be lowercase SHA256")
    data = _read_bounded(saved_path)
    byte_digest = sha256(data).hexdigest()
    if expected_file_sha256 is not None and byte_digest != expected_file_sha256:
        raise ValueError("saved 3-D electrothermal file differs from the expected SHA256")
    try:
        saved = _decode(data)
    except RecursionError as exc:
        raise ValueError("saved 3-D electrothermal JSON nesting exceeds the supported bound") from exc
    del data
    source, graph_record = saved.get("source"), saved.get("graph")
    if type(source) is not dict or type(graph_record) is not dict:
        raise ValueError("saved 3-D electrothermal envelope requires source and graph records")
    _bounded_list(graph_record, "nodes", 2, _MAX_NODES)
    _bounded_list(graph_record, "edges", 1, _MAX_EDGES)
    _bounded_list(source, "node_ids", 2, _MAX_NODES)
    _bounded_list(source, "edge_ids", 1, _MAX_EDGES)
    frames = _bounded_list(source, "frames", 2, _MAX_FRAMES)
    _bounded_list(source, "iteration_history", 1, _MAX_ITERATION_ROWS)
    if source.get("completed") is not True or saved.get("failure") is not None:
        raise ValueError("only complete saved 3-D electrothermal runs are admitted")
    if frame_index >= len(frames):
        raise ValueError("frame_index does not identify an existing accepted frame")
    try:
        source = _validate(graph, saved, length_unit)
    except (KeyError, IndexError, TypeError, RecursionError) as exc:
        raise ValueError("malformed saved 3-D electrothermal source") from exc
    # Must not materialize or dispatch a caller-created frame selector.
    detached = StructureGraph.from_dict(graph_record)
    if _hash(_graph_record(detached)) != saved["graph_sha256"]:
        raise ValueError("detached graph did not preserve the full validated record")
    frame = source["frames"][frame_index]
    temperatures = tuple(zip(source["node_ids"], frame["temperature_K"]))
    receipt = _Saved3DReceipt(
        byte_digest, expected_file_sha256 is not None, saved["content_sha256"],
        saved["graph_sha256"], source["input_sha256"], frame["electrical"]["provenance"]["input_sha256"],
        frame_index, frame["time_s"], len(source["frames"]), length_unit, temperatures,
    )
    snapshot = solve_thermoelastic_network_snapshot_3d(
        detached, temperature_K_by_node_id=dict(temperatures), length_unit=length_unit,
        stress_free_temperature_K=stress_free_temperature_K,
        expansion_alpha_per_K_by_edge_id=expansion_alpha_per_K_by_edge_id,
        constraints_by_node_id=constraints_by_node_id, loads_N_by_node_id=loads_N_by_node_id,
        applicability_limits=applicability_limits,
    )
    return _Saved3DRun(snapshot, receipt)
