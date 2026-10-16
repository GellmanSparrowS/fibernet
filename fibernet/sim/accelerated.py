"""
Taichi-accelerated simulations for fiber networks.

Two backends:
TaichiEngine is the primary simulation backend (mass-spring dynamics).
2. **TaichiEngine** — Mass-spring dynamics (explicit Verlet integration, Taichi parallel forces)

Both accept StructureGraph directly via high-level test methods.
"""

from __future__ import annotations

import time
import os
import stat
import tempfile
import threading
import warnings
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Union

import numpy as np

try:
    import taichi as ti
    HAS_TAICHI = True
except ImportError:
    HAS_TAICHI = False

from fibernet.core.structure_graph import StructureGraph


@dataclass
class SimResult:
    """Unified result container for spring and beam-frame simulations.

    Simple mode: basic displacements + energy
    Detailed mode: per-edge forces, max values, trajectory data

    metadata['node_ids'] identifies displacement, deformed-position and trajectory
    rows; metadata['edge_ids'] identifies edge-force and stretch rows. Producers
    preserve solver row order. Old results may omit these maps, so consumers must
    use their documented historical ordering rather than infer identity.
    """
    # Basic fields
    displacements: np.ndarray = None
    energy: float = 0.0
    time_seconds: float = 0.0
    mode: str = ""
    deformed_positions: np.ndarray = None
    positions_trajectory: List[np.ndarray] = field(default_factory=list)

    # Detailed fields (populated by compute_detailed or detailed=True)
    edge_forces: np.ndarray = None       # per-edge axial force magnitude
    edge_stretches: np.ndarray = None    # per-edge stretch ratio (L/L0)
    max_force: float = 0.0              # maximum edge force
    max_stretch: float = 0.0            # maximum edge stretch ratio
    mean_stretch: float = 0.0           # mean edge stretch ratio
    std_stretch: float = 0.0            # std of edge stretch ratios
    max_displacement: float = 0.0       # maximum node displacement magnitude
    n_nodes: int = 0
    n_edges: int = 0

    # History
    history: List[Dict] = field(default_factory=list)

    # Metadata
    metadata: Dict = field(default_factory=dict)

    def compute_detailed(self, graph, stiffness: Union[float, np.ndarray] = None):
        """Compute detailed metrics from deformed positions.

        Parameters
        ----------
        graph : StructureGraph
            Original (undeformed) graph.
        stiffness : float or (number of edges,) array, optional
            Coefficients in F = k * strain and E = 0.5 * k * L0 * strain**2.
        """
        from fibernet.sim.accelerated import _graph_to_arrays, _element_data

        pos_orig, elements, _, _ = _graph_to_arrays(graph)
        lengths, _ = _element_data(pos_orig, elements)

        self.n_nodes = len(pos_orig)
        self.n_edges = len(elements)

        if self.deformed_positions is not None:
            pos_def = self.deformed_positions
            # Per-edge stretch (vectorized)
            final_lengths = np.linalg.norm(pos_def[elements[:, 1]] - pos_def[elements[:, 0]], axis=1)
            self.edge_stretches = final_lengths / lengths
            self.max_stretch = float(np.max(self.edge_stretches))
            self.mean_stretch = float(np.mean(self.edge_stretches))
            self.std_stretch = float(np.std(self.edge_stretches))

            # Per-edge forces (F = k * (L - L0) / L0)
            if stiffness is not None:
                spring_coefficients = _normalize_spring_stiffness(
                    stiffness, len(elements)
                )
                strain = final_lengths / lengths - 1.0
                self.edge_forces = spring_coefficients * strain
                self.max_force = float(np.max(np.abs(self.edge_forces)))
                self.energy = float(
                    0.5 * np.sum(spring_coefficients * strain ** 2 * lengths)
                )

        # Max displacement
        if self.displacements is not None:
            self.max_displacement = float(np.max(np.linalg.norm(self.displacements, axis=1)))

        return self

    def to_dict(self, detailed: bool = False) -> Dict:
        """Convert to dictionary for CSV/JSON export.

        Parameters
        ----------
        detailed : bool
            If True, include per-edge arrays (large).
        """
        d = {
            "mode": self.mode,
            "energy": self.energy,
            "time_seconds": self.time_seconds,
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "max_force": self.max_force,
            "max_stretch": self.max_stretch,
            "mean_stretch": self.mean_stretch,
            "std_stretch": self.std_stretch,
            "max_displacement": self.max_displacement,
        }
        # Metadata shares the historical flat JSON namespace. Never let it
        # replace result fields (or smuggle detailed arrays into a summary).
        d.update({key: value for key, value in self.metadata.items()
                  if key not in self.__dataclass_fields__})
        if detailed:
            if self.edge_forces is not None:
                d["edge_forces"] = self.edge_forces.tolist()
            if self.edge_stretches is not None:
                d["edge_stretches"] = self.edge_stretches.tolist()
            if self.displacements is not None:
                d["displacements"] = self.displacements.tolist()
            if self.deformed_positions is not None:
                d["deformed_positions"] = self.deformed_positions.tolist()
            d["history"] = self.history
            if self.positions_trajectory:
                d["positions_trajectory"] = [frame.tolist() for frame in self.positions_trajectory]
        return d

    def save(self, path: str, detailed: bool = False):
        """Atomically save JSON via an exclusively owned sibling temporary.

        Existing ordinary file permissions are retained; new files are private
        (0600). Symlinks, nonregular targets and special mode bits are rejected.
        This is replacement atomicity, not crash durability or writer locking.
        """
        data = self.to_dict(detailed=detailed)
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            info = destination.lstat()
        except FileNotFoundError:
            mode = 0o600
        else:
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("result destination must be a regular file, not a symlink or nonregular target")
            mode = stat.S_IMODE(info.st_mode)
            if mode & 0o7000:
                raise ValueError("result destination has unsupported special mode bits")
        def convert(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            raise TypeError(f'Cannot serialize {type(value).__name__}')
        fd, name = tempfile.mkstemp(prefix="." + destination.name + ".result-",
                                    dir=destination.parent)
        temporary = Path(name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                fd = None  # The stream now owns the descriptor, including on error.
                json.dump(data, f, indent=2, default=convert)
            os.chmod(temporary, mode)
            temporary.replace(destination)
        finally:
            # Cleanup must not mask a serialization/replace error or report a
            # committed save as failed, even when warnings are treated as errors.
            for cleanup in ((lambda: os.close(fd)) if fd is not None else None,
                            lambda: temporary.unlink(missing_ok=True)):
                if cleanup is None:
                    continue
                try:
                    cleanup()
                except Exception as exc:
                    try:
                        warnings.warn(f"Result cleanup failed for owned temporary {temporary}: {exc!r}; "
                                      "the save outcome is unchanged", RuntimeWarning, stacklevel=2)
                    except Exception:
                        pass

    @staticmethod
    def load(path: str) -> "SimResult":
        with open(path) as f:
            data = json.load(f)
        # Older FEM summary files stored [axial, shear, moment] resultants
        # under the canonical one-dimensional edge_forces name. Recover the
        # axial forces while retaining the original resultants separately.
        if data.get("mode") == "fem_beam_frame" and data.get("edge_forces") is not None:
            legacy_forces = np.asarray(data["edge_forces"])
            if legacy_forces.ndim == 2 and legacy_forces.shape[1] == 3:
                data.setdefault("beam_resultants", data["edge_forces"])
                data["edge_forces"] = legacy_forces[:, 0].tolist()
        r = SimResult(
            mode=data.get("mode", ""),
            energy=data.get("energy", 0),
            time_seconds=data.get("time_seconds", 0),
            n_nodes=data.get("n_nodes", 0),
            n_edges=data.get("n_edges", 0),
            max_force=data.get("max_force", 0),
            max_stretch=data.get("max_stretch", 0),
            mean_stretch=data.get("mean_stretch", 0),
            std_stretch=data.get("std_stretch", 0),
            max_displacement=data.get("max_displacement", 0),
            history=data.get("history", []),
            metadata={k: v for k, v in data.items()
                      if k not in {"mode", "energy", "time_seconds", "n_nodes", "n_edges",
                                   "max_force", "max_stretch", "mean_stretch", "std_stretch",
                                   "max_displacement", "history",
                                   "edge_forces", "edge_stretches",
                                   "displacements", "deformed_positions",
                                   "positions_trajectory"}},
        )
        for key in ("edge_forces", "edge_stretches", "displacements", "deformed_positions"):
            if data.get(key) is not None:
                setattr(r, key, np.array(data[key]))
        if data.get("positions_trajectory"):
            r.positions_trajectory = [np.array(frame) for frame in data["positions_trajectory"]]
        return r


# Runtime ownership is process-wide because Taichi fields and kernels are global.
# Strong references avoid recycled Python/native thread IDs masquerading as owners.
_taichi_runtime = None
_taichi_program = None
_taichi_owner = None
_taichi_lifecycle_lock = threading.RLock()


def _taichi_runtime_state():
    """Narrow private-API adapter, source-verified against Taichi 1.7.4."""
    if not HAS_TAICHI:
        raise ImportError("Taichi required: pip install taichi==1.7.4")
    if tuple(getattr(ti, "__version__", ())) != (1, 7, 4):
        raise RuntimeError(
            "FiberNet runtime lifecycle support is verified for Taichi 1.7.4; "
            "install taichi==1.7.4 before using TaichiEngine."
        )
    try:
        runtime = ti.lang.impl.get_runtime()
        program = runtime.prog
    except AttributeError as exc:
        raise RuntimeError("Unsupported Taichi runtime API; expected Taichi 1.7.4") from exc
    return runtime, program


def _check_taichi_owner():
    if _taichi_owner is not None and threading.current_thread() is not _taichi_owner:
        raise RuntimeError(
            "TaichiEngine must run on its runtime-initializing thread "
            "(which may be a worker thread); cross-thread engine/cache use is unsupported."
        )


def _discard_taichi_caches():
    # Never inspect old fields: even querying their shape after reset can segfault.
    TaichiEngine._field_cache.clear()
    TaichiEngine._compute_forces_cache.clear()


def _ensure_taichi(arch: str = "cpu", num_threads: int = 4):
    global _taichi_runtime, _taichi_program, _taichi_owner
    if arch not in ("cpu", "gpu"):
        raise ValueError("arch must be 'cpu' or 'gpu'")
    with _taichi_lifecycle_lock:
        _check_taichi_owner()
        runtime, program = _taichi_runtime_state()
        if runtime is not _taichi_runtime or program is not _taichi_program:
            _discard_taichi_caches()
            if program is not None:
                raise RuntimeError(
                    "Taichi was initialized outside FiberNet; runtime thread ownership "
                    "cannot be verified. On the intended runtime-owning thread, call "
                    "ti.reset() and then construct/use TaichiEngine."
                )
        if program is None:
            if os.environ.get("TI_ARCH") is not None:
                raise RuntimeError("Remove TI_ARCH before TaichiEngine initialization; use arch= explicitly")
            ti.init(arch=ti.cpu if arch == "cpu" else ti.gpu,
                    cpu_max_num_threads=num_threads, enable_fallback=False)
            runtime, program = _taichi_runtime_state()
            if program is None:
                raise RuntimeError("Taichi initialization did not create a runtime")
            _taichi_runtime, _taichi_program = runtime, program
            _taichi_owner = threading.current_thread()
        actual_arch = ti.cfg.arch
        compatible = actual_arch == ti.cpu if arch == "cpu" else actual_arch in ti.gpu
        if not compatible:
            raise RuntimeError(
                f"Requested Taichi architecture {arch!r} conflicts with active "
                f"{actual_arch}; use the existing backend or reset on its owning thread."
            )


def _graph_to_arrays(graph: StructureGraph):
    """Extract positions, elements, node_id mapping from StructureGraph."""
    node_ids = list(graph.nodes.keys())
    nid_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    pos = np.array([graph.nodes[nid].position for nid in node_ids])
    elements = np.array([[nid_to_idx[e.node_i], nid_to_idx[e.node_j]] for e in graph.edges.values()])
    return pos, elements, node_ids, nid_to_idx


def _get_boundary_indices(pos: np.ndarray, tol: float = None, pct: float = 0.10) -> Dict[str, List[int]]:
    """Find boundary node indices for each side.
    
    Uses percentile-based detection for robust boundary identification
    across different structure types (crystal lattices, TPMS, etc.).
    
    Parameters
    ----------
    pos : (N, D) array of node positions
    tol : float, optional
        Fixed tolerance. If None, uses percentile-based detection.
    pct : float
        Percentile for boundary detection (0.10 = 10% of nodes at each end).
    """
    result = {}
    if tol is not None:
        # Legacy fixed tolerance mode
        bb_min, bb_max = pos.min(0), pos.max(0)
        result["left"] = list(np.where(pos[:, 0] < bb_min[0] + tol)[0])
        result["right"] = list(np.where(pos[:, 0] > bb_max[0] - tol)[0])
        result["bottom"] = list(np.where(pos[:, 1] < bb_min[1] + tol)[0])
        result["top"] = list(np.where(pos[:, 1] > bb_max[1] + tol)[0])
        if pos.shape[1] >= 3:
            result["back"] = list(np.where(pos[:, 2] < bb_min[2] + tol)[0])
            result["front"] = list(np.where(pos[:, 2] > bb_max[2] - tol)[0])
    else:
        # Percentile-based detection (robust for TPMS/complex structures)
        x_sorted = np.argsort(pos[:, 0])
        y_sorted = np.argsort(pos[:, 1])
        n = len(pos)
        n_bnd = max(int(n * pct), 1)
        result["left"] = list(x_sorted[:n_bnd])
        result["right"] = list(x_sorted[-n_bnd:])
        n_bnd_y = max(int(n * pct), 1)
        result["bottom"] = list(y_sorted[:n_bnd_y])
        result["top"] = list(y_sorted[-n_bnd_y:])
        if pos.shape[1] >= 3:
            z_sorted = np.argsort(pos[:, 2])
            n_bnd_z = max(int(n * pct), 1)
            result["back"] = list(z_sorted[:n_bnd_z])
            result["front"] = list(z_sorted[-n_bnd_z:])
    return result


def _element_data(pos, elements):
    """Compute element lengths, directions, areas."""
    diff = pos[elements[:, 1]] - pos[elements[:, 0]]
    lengths = np.linalg.norm(diff, axis=1)
    lengths = np.maximum(lengths, 1e-12)
    directions = diff / lengths[:, None]
    return lengths, directions



# ======================================================================
# ======================================================================


def _normalize_spring_stiffness(stiffness, num_edges):
    """Return one finite, nonnegative spring coefficient per edge.

    Convention: F = k * strain; energy = 0.5 * k * L0 * strain**2.
    k therefore has force units, not force/length units.
    """
    values = np.asarray(stiffness, dtype=np.float64)
    if values.ndim == 0:
        values = np.full(num_edges, float(values), dtype=np.float64)
    elif values.shape != (num_edges,):
        raise ValueError("stiffness must be scalar or have shape (num_edges,)")
    else:
        values = values.copy()
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("stiffness must be finite and nonnegative")
    return values


def _prepare_displacement_schedule(schedule, num_nodes, dim):
    """Normalize schedules expressed in integration-step coordinates.

    Empty schedules impose no constraint. Input order is immaterial.
    Duplicate timestamps are ambiguous and rejected. A single waypoint
    represents a constant displacement. Values outside the waypoint
    range are held at the nearest endpoint.
    """
    prepared = {}
    if schedule is None:
        return prepared
    for node, waypoints in schedule.items():
        if (
            isinstance(node, (bool, np.bool_))
            or not isinstance(node, (int, np.integer))
            or not 0 <= node < num_nodes
        ):
            raise ValueError("Schedule node must be a valid node-array index")
        rows = []
        for waypoint in waypoints:
            if len(waypoint) != 2:
                raise ValueError("Each waypoint must be (step, displacement)")
            step, displacement = waypoint
            step = float(step)
            vector = np.asarray(displacement, dtype=np.float64)
            if not np.isfinite(step) or step < 0:
                raise ValueError("Schedule steps must be finite and nonnegative")
            if vector.shape != (dim,) or not np.all(np.isfinite(vector)):
                raise ValueError(
                    f"Schedule displacements must be finite vectors of length {dim}"
                )
            rows.append((step, vector.copy()))
        if not rows:
            continue
        rows.sort(key=lambda row: row[0])
        if any(a[0] == b[0] for a, b in zip(rows, rows[1:])):
            raise ValueError("Schedule steps must be unique")
        prepared[int(node)] = rows
    return prepared


class TaichiEngine:
    """Mass-spring dynamics with Taichi parallel force computation.

    Axial spring model: F = k * (L - L0) / L0 * direction
    Explicit Verlet integration with damping.

    Runtime lifetime: sequential engines share one backend on the thread that
    first initializes FiberNet's runtime (including a worker thread). Other
    threads are rejected before field access. External reset between calls is
    supported on the owner; external init requires an explicit reset before
    reuse. External reset/init during a solve is unsupported. Taichi 1.7.4 is
    required by the version-tested private runtime-generation adapter.
    """

    # Cache Taichi fields and kernels to avoid SNode exhaustion
    # Key: (dim, num_nodes, num_edges)
    _field_cache = {}
    _compute_forces_cache = {}

    @classmethod
    def clear_field_cache(cls):
        """Drop cached field/kernel references on the runtime-owning thread.

        This does not destroy native Taichi SNodes. Subsequent simulations
        allocate new cache entries; this is not a native-memory reclamation API.
        """
        with _taichi_lifecycle_lock:
            _check_taichi_owner()
            _discard_taichi_caches()

    def __init__(self, arch: str = "cpu", num_threads: int = 4):
        _ensure_taichi(arch, num_threads)
        self.arch = arch
        self.num_threads = num_threads
        if arch == "cpu" and ti.cfg.cpu_max_num_threads != num_threads:
            warnings.warn(
                "TaichiEngine shares the active runtime's CPU thread count; "
                "num_threads only configures the first initialization.",
                RuntimeWarning, stacklevel=2,
            )

    def compute_forces(
        self,
        positions: np.ndarray,
        rest_lengths: np.ndarray,
        stiffness: np.ndarray,
        edges: np.ndarray,
    ) -> np.ndarray:
        """Compute spring forces in parallel (fields cached to avoid SNode leak)."""
        _ensure_taichi(self.arch, self.num_threads)
        num_nodes = positions.shape[0]
        num_edges = edges.shape[0]
        dim = positions.shape[1]

        cache_key = (dim, num_nodes, num_edges)
        if cache_key not in TaichiEngine._compute_forces_cache:
            pos = ti.Vector.field(dim, dtype=ti.f64, shape=num_nodes)
            forces = ti.Vector.field(dim, dtype=ti.f64, shape=num_nodes)
            edge_arr = ti.Vector.field(2, dtype=ti.i32, shape=num_edges)
            L0 = ti.field(dtype=ti.f64, shape=num_edges)
            k_field = ti.field(dtype=ti.f64, shape=num_edges)
            f_temp = ti.Vector.field(dim, dtype=ti.f64, shape=num_edges)

            @ti.kernel
            def _cf_compute():
                for e in range(num_edges):
                    i = edge_arr[e][0]
                    j = edge_arr[e][1]
                    diff = pos[j] - pos[i]
                    L = diff.norm()
                    if L > 1e-12:
                        strain_val = (L - L0[e]) / L0[e]
                        force_mag = k_field[e] * strain_val
                        f_temp[e] = force_mag * (diff / L)
                    else:
                        for d in ti.static(range(dim)):
                            f_temp[e][d] = 0.0

            @ti.kernel
            def _cf_zero():
                for n in range(num_nodes):
                    for d in ti.static(range(dim)):
                        forces[n][d] = 0.0

            @ti.kernel
            def _cf_accumulate():
                for e in range(num_edges):
                    i = edge_arr[e][0]
                    j = edge_arr[e][1]
                    for d in ti.static(range(dim)):
                        forces[i][d] += f_temp[e][d]
                        forces[j][d] -= f_temp[e][d]

            TaichiEngine._compute_forces_cache[cache_key] = {
                'pos': pos, 'forces': forces, 'edge_arr': edge_arr,
                'L0': L0, 'k_field': k_field, 'f_temp': f_temp,
                'compute': _cf_compute, 'zero': _cf_zero, 'accumulate': _cf_accumulate,
            }

        cached = TaichiEngine._compute_forces_cache[cache_key]
        pos = cached['pos']
        forces = cached['forces']
        edge_arr = cached['edge_arr']
        L0 = cached['L0']
        k_field = cached['k_field']

        pos.from_numpy(positions.astype(np.float64))
        edge_arr.from_numpy(edges.astype(np.int32))
        L0.from_numpy(rest_lengths.astype(np.float64))
        k_field.from_numpy(stiffness.astype(np.float64))

        cached['zero']()
        cached['compute']()
        cached['accumulate']()
        return forces.to_numpy()

    def dynamics(
        self,
        graph: StructureGraph,
        fixed_nodes: List[int] = None,
        displacement_schedule: Dict[int, List[Tuple[float, np.ndarray]]] = None,
        external_force: np.ndarray = None,
        stiffness: float = 1e5,
        damping: float = 0.3,
        dt: float = 1e-4,
        num_steps: int = 5000,
        save_interval: int = 500,
        spring_k: float = None,
        dashpot: float = 10.0,
        drag: float = 1.0,
        max_nodes: int = 20000,
    ) -> SimResult:
        """Run mass-spring dynamics with dashpot damping and air drag.

        Based on reference implementation with proper physics:
        - Spring force: F = -k * dir * (dist/rest - 1)
        - Dashpot damping: F_damp = -damp * (vi-vj).dot(dir) * dir * rest_len
        - Air drag: v *= exp(-drag * dt)
        - Constraints by position clamping + velocity zeroing

        Parameters
        ----------
        graph : StructureGraph
        fixed_nodes : list of node-array indices to fix
            Indices follow graph.nodes insertion order, not graph node IDs.
        displacement_schedule : dict mapping node-array index to waypoints
            Waypoints are (step, displacement) pairs, linearly interpolated and
            held outside their range. Keys follow graph.nodes insertion order.
            Displacements have pos_orig.shape[1] components (XYZ for StructureGraph).
        external_force : (N, 3) array
        stiffness : float
            Global force-like axial coefficient k in F = k * strain, overridden
            by spring_k. With consistent SI units k is in N (like EA), not N/m.
            Neither k nor node inertia is inferred from edge Material.
        damping : dashpot damping coefficient
        dt : time step
        num_steps : total integration steps
        save_interval : save trajectory every N steps
        spring_k : scalar or (number of edges,) stiffness array (overrides stiffness)
            Finite, nonnegative force-like coefficients in F = k * strain.
            Element energy is 0.5 * k * strain**2 * rest_length. A conventional
            force/extension spring constant is k / rest_length; no automatic
            conversion from Young's modulus or graph radius is performed.
        dashpot : dashpot damping coefficient
        drag : air drag coefficient
        """
        _check_taichi_owner()
        for name, value in (
            ("num_steps", num_steps),
            ("save_interval", save_interval),
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")

        # Memory guard: warn by default, hard block if FIBERNET_STRICT_MEMORY=1
        if graph.num_nodes > max_nodes:
            import os as _os
            msg = (
                f"Structure has {graph.num_nodes} nodes, exceeding max_nodes={max_nodes}. "
                f"Simulation may be slow or run out of memory. "
                f"Use a smaller grid or set max_nodes higher to suppress this warning."
            )
            if _os.environ.get("FIBERNET_STRICT_MEMORY", "0") == "1":
                raise ValueError(msg)
            import warnings as _warnings
            _warnings.warn(msg, ResourceWarning, stacklevel=2)

        t0 = time.time()
        pos_orig, elements, node_ids, _ = _graph_to_arrays(graph)
        dim = pos_orig.shape[1]
        num_nodes = len(node_ids)
        num_edges = len(elements)

        schedules = _prepare_displacement_schedule(
            displacement_schedule, num_nodes, dim
        )

        lengths, _ = _element_data(pos_orig, elements)
        rest_lengths_np = lengths.copy()

        stiff_np = _normalize_spring_stiffness(
            spring_k if spring_k is not None else stiffness, num_edges
        )

        _ensure_taichi(self.arch, self.num_threads)
        # Taichi fields and kernel (cached to avoid SNode exhaustion and kernel recompilation)
        cache_key = (dim, num_nodes, num_edges)
        if cache_key not in TaichiEngine._field_cache:
            # Allocate fields once
            ti_pos = ti.Vector.field(dim, dtype=ti.f64, shape=num_nodes)
            ti_vel = ti.Vector.field(dim, dtype=ti.f64, shape=num_nodes)
            ti_edges = ti.Vector.field(2, dtype=ti.i32, shape=num_edges)
            ti_L0 = ti.field(dtype=ti.f64, shape=num_edges)
            ti_k = ti.field(dtype=ti.f64, shape=num_edges)
            ti_fixed = ti.field(dtype=ti.i32, shape=num_nodes)
            ti_ext = ti.Vector.field(dim, dtype=ti.f64, shape=num_nodes)
            ti_pos0 = ti.Vector.field(dim, dtype=ti.f64, shape=num_nodes)
            ti_sched = ti.Vector.field(dim, dtype=ti.f64, shape=num_nodes)
            ti_sched_mask = ti.field(dtype=ti.i32, shape=num_nodes)
            ti_dt = ti.field(dtype=ti.f64, shape=())
            ti_dashpot = ti.field(dtype=ti.f64, shape=())
            ti_drag = ti.field(dtype=ti.f64, shape=())
            ti_ramp = ti.field(dtype=ti.f64, shape=())

            # Compile kernel once (closure captures fields + constants)
            @ti.kernel
            def _cached_substep():
                _dt = ti_dt[None]
                _dash = ti_dashpot[None]
                _drag = ti_drag[None]
                _ramp = ti_ramp[None]

                for e in range(num_edges):
                    ia = ti_edges[e][0]
                    ib = ti_edges[e][1]
                    d = ti_pos[ia] - ti_pos[ib]
                    dist = d.norm() + 1e-6
                    dir_ = d / dist
                    f_spring = -ti_k[e] * dir_ * (dist / ti_L0[e] - 1.0)
                    rel_v = ti_vel[ia] - ti_vel[ib]
                    f_damp = -_dash * rel_v.dot(dir_) * dir_ * ti_L0[e]
                    f_total = f_spring + f_damp
                    for k in ti.static(range(dim)):
                        ti.atomic_add(ti_vel[ia][k], f_total[k] * _dt)
                        ti.atomic_add(ti_vel[ib][k], -f_total[k] * _dt)

                for i in range(num_nodes):
                    for k in ti.static(range(dim)):
                        ti_vel[i][k] += ti_ext[i][k] * _dt
                    ti_vel[i] *= ti.exp(-_drag * _dt)
                    for k in ti.static(range(dim)):
                        ti_pos[i][k] += ti_vel[i][k] * _dt

                for i in range(num_nodes):
                    if ti_fixed[i] == 1:
                        for k in ti.static(range(dim)):
                            ti_pos[i][k] = ti_pos0[i][k]
                            ti_vel[i][k] = 0.0

                for i in range(num_nodes):
                    if ti_sched_mask[i] == 1:
                        for k in ti.static(range(dim)):
                            ti_pos[i][k] = ti_pos0[i][k] + ti_sched[i][k] * _ramp
                            ti_vel[i][k] = 0.0

            TaichiEngine._field_cache[cache_key] = {
                'ti_pos': ti_pos, 'ti_vel': ti_vel, 'ti_edges': ti_edges,
                'ti_L0': ti_L0, 'ti_k': ti_k, 'ti_fixed': ti_fixed,
                'ti_ext': ti_ext, 'ti_pos0': ti_pos0, 'ti_sched': ti_sched,
                'ti_sched_mask': ti_sched_mask, 'ti_dt': ti_dt,
                'ti_dashpot': ti_dashpot, 'ti_drag': ti_drag, 'ti_ramp': ti_ramp,
                'substep': _cached_substep,
            }

        # Retrieve cached fields and kernel
        cached = TaichiEngine._field_cache[cache_key]
        ti_pos = cached['ti_pos']
        ti_vel = cached['ti_vel']
        ti_edges = cached['ti_edges']
        ti_L0 = cached['ti_L0']
        ti_k = cached['ti_k']
        ti_fixed = cached['ti_fixed']
        ti_ext = cached['ti_ext']
        ti_pos0 = cached['ti_pos0']
        ti_sched = cached['ti_sched']
        ti_sched_mask = cached['ti_sched_mask']
        ti_dt = cached['ti_dt']
        ti_dashpot = cached['ti_dashpot']
        ti_drag = cached['ti_drag']
        ti_ramp = cached['ti_ramp']
        substep = cached['substep']

        # Reset field data (always, even on cache hit)
        ti_pos0.from_numpy(pos_orig.astype(np.float64))
        ti_pos.from_numpy(pos_orig.astype(np.float64))
        ti_vel.from_numpy(np.zeros_like(pos_orig))
        ti_edges.from_numpy(elements.astype(np.int32))
        ti_L0.from_numpy(rest_lengths_np.astype(np.float64))
        ti_k.from_numpy(stiff_np.astype(np.float64))

        fixed_arr = np.zeros(num_nodes, dtype=np.int32)
        for ni in (fixed_nodes or []):
            fixed_arr[ni] = 1
        ti_fixed.from_numpy(fixed_arr)

        ext_f = external_force if external_force is not None else np.zeros((num_nodes, dim))
        ti_ext.from_numpy(ext_f.astype(np.float64)[:, :dim])

        # Per-node, piecewise-linear schedules. Reset on every call,
        # including calls reusing fields from a previous simulation.
        schedule_targets = np.zeros((num_nodes, dim), dtype=np.float64)
        sched_mask = np.zeros(num_nodes, dtype=np.int32)
        for ni in schedules:
            sched_mask[ni] = 1
        ti_sched.from_numpy(schedule_targets)
        ti_sched_mask.from_numpy(sched_mask)

        # The kernel consumes absolute scheduled displacements.
        ti_ramp[None] = 1.0

        # Run loop
        ti_dt[None] = dt
        ti_dashpot[None] = dashpot
        ti_drag[None] = drag

        trajectory = [pos_orig.copy()]
        max_stretch_history = []

        for step in range(num_steps):
            if schedules:
                for ni, schedule in schedules.items():
                    schedule_targets[ni] = self._interpolate_schedule(
                        schedule, step + 1, num_steps
                    )
                ti_sched.from_numpy(schedule_targets)
            substep()

            if (step + 1) % save_interval == 0:
                cur_pos = ti_pos.to_numpy()
                trajectory.append(cur_pos.copy())
                # Vectorized edge length computation
                new_len = np.linalg.norm(cur_pos[elements[:, 1]] - cur_pos[elements[:, 0]], axis=1)
                max_stretch_history.append(float(np.max(new_len / rest_lengths_np)))

        pos_final = ti_pos.to_numpy()
        displacements = pos_final - pos_orig
        if dim < 3:
            displacements = np.hstack([displacements, np.zeros((num_nodes, 3 - dim))])

        result = SimResult(
            displacements=displacements,
            time_seconds=time.time() - t0,
            mode="dynamics",
            deformed_positions=pos_final,
            positions_trajectory=trajectory,
            history=[{"step": (i+1)*save_interval, "max_stretch": ms}
                     for i, ms in enumerate(max_stretch_history)],
            metadata={"node_ids": list(node_ids), "edge_ids": list(graph.edges),
                      "stiffness": float(stiffness), "damping": float(damping),
                      "dt": float(dt), "num_steps": int(num_steps)},
        )
        result.compute_detailed(graph, stiffness=stiff_np)
        return result
    def stretch_test(
        self,
        graph: StructureGraph,
        target_stretch: float = 2.0,
        stiffness: float = 1e5,
        damping: float = 0.3,
        num_steps: int = None,
        save_interval: int = 1000,
        ramp_fraction: float = 0.2,
        auto_steps: bool = True,
        max_nodes: int = 20000,
    ) -> SimResult:
        """Displacement-controlled uniaxial stretch with automatic step calculation.

        Two-phase simulation:
        1. Ramp phase: gradually move right boundary to target_stretch
        2. Relax phase: hold displacement, let waves propagate and settle

        Parameters
        ----------
        target_stretch : float
            Target stretch ratio (e.g., 2.0 = double the length)
        stiffness : float
            Force-like axial coefficient k in F = k * strain. With a consistent
            physical unit system this has force units (N in SI), not N/m.
            Energy is 0.5 * k * strain**2 * rest_length. Edge Material properties
            are not used to determine this coefficient.
        damping : float
            Velocity damping coefficient (0-1)
        num_steps : int, optional
            Total simulation steps. If None and auto_steps=True, calculated from graph diameter
        save_interval : int
            Save trajectory every N steps
        ramp_fraction : float
            Fraction of total steps for ramp phase (default 0.2 = 20%)
        auto_steps : bool
            Auto-calculate steps based on graph diameter if num_steps is None

        Returns
        -------
        SimResult
            Result with deformed_positions and trajectory
        """
        _check_taichi_owner()
        if not np.isfinite(ramp_fraction) or not 0 <= ramp_fraction <= 1:
            raise ValueError("ramp_fraction must be finite and between 0 and 1")
        if num_steps is not None and (
            isinstance(num_steps, (bool, np.bool_))
            or not isinstance(num_steps, (int, np.integer))
            or num_steps <= 0
        ):
            raise ValueError("num_steps must be a positive integer")

        import networkx as nx

        pos, elements, _, _ = _graph_to_arrays(graph)
        bnd = _get_boundary_indices(pos)
        L_x = pos[:, 0].max() - pos[:, 0].min()
        target_disp = L_x * (target_stretch - 1)

        # Auto-calculate steps based on graph diameter
        if num_steps is None and auto_steps:
            # Build graph to find diameter
            G = nx.Graph()
            for e in elements:
                G.add_edge(int(e[0]), int(e[1]))
            
            # Find max hops from right to left (sample for speed)
            right_set = set(bnd['right'])
            left_set = set(bnd['left'])
            max_hops = 0
            for rid in list(right_set)[:5]:
                for lid in list(left_set)[:5]:
                    try:
                        hops = nx.shortest_path_length(G, rid, lid)
                        max_hops = max(max_hops, hops)
                    except:
                        pass
            
            # 3D structures need more steps than 2D
            is_3d = pos.shape[1] == 3 and pos[:, 2].max() > 1e-6
            
            # Formula: base + hops * per_hop, capped reasonably
            base_steps = 3000 if is_3d else 2000
            per_hop = 300 if is_3d else 150
            num_steps = base_steps + max_hops * per_hop
            num_steps = min(num_steps, 20000)
        elif num_steps is None:
            num_steps = 5000

        ramp_steps = int(num_steps * ramp_fraction)
        schedule = {}
        target = np.array([target_disp, 0.0, 0.0])
        for ni in bnd["right"]:
            if ramp_steps == 0:
                schedule[ni] = [(0, target.copy())]
            else:
                schedule[ni] = [
                    (0, np.zeros(3)),
                    (ramp_steps, target.copy()),
                ]

        fixed = bnd["left"] + (bnd.get("bottom", [])[:1] if bnd.get("bottom") else [])

        return self.dynamics(
            graph,
            fixed_nodes=fixed,
            displacement_schedule=schedule,
            stiffness=stiffness,
            damping=damping,
            dt=1e-5,
            num_steps=num_steps,
            save_interval=save_interval,
            max_nodes=max_nodes,
        )

    @staticmethod
    def _interpolate_schedule(schedule, step, total_steps):
        """Linearly interpolate displacement from schedule."""
        if not schedule:
            return np.zeros(3)
        schedule = sorted(schedule, key=lambda x: x[0])
        if step <= schedule[0][0]:
            return np.array(schedule[0][1])
        if step >= schedule[-1][0]:
            return np.array(schedule[-1][1])
        for i in range(len(schedule) - 1):
            s0, d0 = schedule[i]
            s1, d1 = schedule[i + 1]
            if s0 <= step <= s1:
                t = (step - s0) / (s1 - s0)
                return np.array(d0) * (1 - t) + np.array(d1) * t
        return np.array(schedule[-1][1])


