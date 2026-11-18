"""
FiberNetwork - core data structure for a collection of fibers with connectivity.

Manages fiber collections, crosslinks, contacts, periodicity, and provides
methods for I/O, querying, and basic analysis.
"""

from __future__ import annotations

import numpy as np
# from scipy.spatial import cKDTree  # Lazy import to avoid C extension loading
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any, Union, TYPE_CHECKING
import json
import operator
from fibernet._json import unique_object_pairs as _unique_json_object_pairs
import warnings

from fibernet.core.fiber import Fiber, CrossSection
from fibernet.core.material import Material

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix as _CSRMatrix
else:
    _CSRMatrix = Any  # Runtime type inspection does not import SciPy.


def _hdf5_section_params_json(params: Dict[str, Any]) -> str:
    """Encode section parameters without coercing object keys to strings."""
    if not isinstance(params, dict):
        raise TypeError("HDF5 cross_section_params must be a JSON object")

    def require_string_keys(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if not isinstance(key, str):
                    raise TypeError("HDF5 cross_section_params object keys must be strings")
                require_string_keys(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                require_string_keys(child)

    # Let the JSON encoder report unsupported values and cycles before traversal.
    encoded = json.dumps(params, allow_nan=False)
    require_string_keys(params)
    return encoded


@dataclass
class Crosslink:
    """Represents a crosslink (junction) between two fibers.
    
    Parameters
    ----------
    fiber_i : int
        Index of first fiber.
    fiber_j : int
        Index of second fiber.
    param_i : float
        Parametric position on fiber_i (0 to 1).
    param_j : float
        Parametric position on fiber_j (0 to 1).
    position : np.ndarray
        3D coordinates of the crosslink point.
    crosslink_type : str
        Type: 'welded', 'pinned', 'sliding', 'bonded'.
    strength : float
        Crosslink strength in N (optional).
    stiffness : float
        Crosslink stiffness in N/m (optional).
    """
    fiber_i: int
    fiber_j: int
    param_i: float
    param_j: float
    position: np.ndarray
    crosslink_type: str = "welded"
    strength: float = float('inf')
    stiffness: float = float('inf')


@dataclass
class FiberNetwork:
    """A network of fibers with crosslinks and contacts.

    Parameters
    ----------
    fibers : list of Fiber
        The fibers in the network.
    crosslinks : list of Crosslink
        Crosslinks between fibers.
    box_size : np.ndarray or None
        Bounding box dimensions [Lx, Ly, Lz] for periodic systems.
    periodic : bool
        Whether the system uses periodic boundary conditions.
    dimension : int
        2 for 2D networks, 3 for 3D networks.
    metadata : dict
        Arbitrary metadata (generator info, parameters, etc.).
    """
    fibers: List[Fiber] = field(default_factory=list)
    crosslinks: List[Crosslink] = field(default_factory=list)
    box_size: Optional[np.ndarray] = None
    periodic: bool = False
    dimension: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_fibers(self) -> int:
        return len(self.fibers)

    @property
    def num_crosslinks(self) -> int:
        return len(self.crosslinks)

    @property
    def total_length(self) -> float:
        """Total length of all fibers."""
        return sum(f.length for f in self.fibers)

    @property
    def total_volume(self) -> float:
        """Total volume of fiber material."""
        return sum(f.length * f.cross_section_area for f in self.fibers)

    @property
    def mean_fiber_length(self) -> float:
        if not self.fibers:
            return 0.0
        return self.total_length / self.num_fibers

    @property
    def mean_radius(self) -> float:
        if not self.fibers:
            return 0.0
        return np.mean([f.radius for f in self.fibers])

    def density(self) -> float:
        """Estimate the fraction occupied by fibers in the network box.

        The planar estimate divides the sum of nominal fiber strip areas by
        the XY box area. Each strip has width twice that fiber's radius and
        length equal to its centerline arc length. Overlapping strips
        contribute separately. The 3D estimate uses fiber volumes based
        on the cross-sectional areas and the box volume.
        """
        if self.box_size is not None:
            dims = np.array(self.box_size, dtype=float)
        else:
            bb_min, bb_max = self.bounding_box()
            dims = bb_max - bb_min
        
        # Handle 2D networks (z-dimension is 0)
        if self.dimension == 2 or (len(dims) >= 3 and dims[2] < 1e-10):
            # 2D: area fraction
            area = dims[0] * dims[1] if dims[0] > 0 and dims[1] > 0 else 0.0
            if area <= 0:
                return 0.0
            # Keep the strip products and sum in native float precision.
            fiber_area = sum((2.0 * fiber.length) * float(fiber.radius) for fiber in self.fibers)
            return fiber_area / area
        else:
            # 3D: volume fraction
            vol = self.total_volume
            box_vol = float(np.prod(np.maximum(dims, 1e-12)))
            return vol / box_vol if box_vol > 0 else 0.0

    def bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Overall bounding box (min_corner, max_corner)."""
        if not self.fibers:
            return np.zeros(3), np.zeros(3)
        all_pts = np.vstack([f.centerline for f in self.fibers])
        return all_pts.min(axis=0), all_pts.max(axis=0)

    def add_fiber(self, fiber: Fiber) -> int:
        """Add a fiber to the network. Returns fiber index."""
        fiber.fiber_id = len(self.fibers)
        self.fibers.append(fiber)
        return fiber.fiber_id

    def add_crosslink(self, crosslink: Crosslink):
        """Add a crosslink to the network."""
        self.crosslinks.append(crosslink)

    def remove_fiber(self, index: int):
        """Remove fiber at given index and associated crosslinks."""
        if 0 <= index < len(self.fibers):
            self.fibers.pop(index)
            self.crosslinks = [
                cl for cl in self.crosslinks
                if cl.fiber_i != index and cl.fiber_j != index
            ]
            # Snapshot old indices so even a repeated Crosslink reference is
            # remapped once, while retaining each surviving object's fields.
            endpoints = [(cl, cl.fiber_i, cl.fiber_j) for cl in self.crosslinks]
            for cl, fiber_i, fiber_j in endpoints:
                cl.fiber_i = fiber_i - 1 if fiber_i > index else fiber_i
                cl.fiber_j = fiber_j - 1 if fiber_j > index else fiber_j
            for i, f in enumerate(self.fibers):
                f.fiber_id = i

    def get_fiber(self, index: int) -> Fiber:
        return self.fibers[index]

    def fiber_orientations(self) -> np.ndarray:
        """Get orientation vectors (direction) for all fibers."""
        if not self.fibers:
            return np.array([])
        return np.array([f.direction for f in self.fibers])

    def fiber_lengths(self) -> np.ndarray:
        return np.array([f.length for f in self.fibers])

    def connectivity_matrix(self, *, sparse: bool = False) -> Union[np.ndarray, _CSRMatrix]:
        """Return the symmetric crosslink-count matrix in fiber list order.

        The default is a dense NumPy array. Use ``sparse=True`` for a SciPy
        CSR matrix without a dense allocation. Each crosslink increments
        both directions, including two increments for a self-link.

        Sparse coordinates must be Python or NumPy integer scalars other
        than booleans. Negative indices in range wrap as in NumPy.
        """
        if not isinstance(sparse, (bool, np.bool_)):
            raise TypeError('sparse must be a boolean')
        n = self.num_fibers
        if sparse:
            from scipy.sparse import coo_matrix

            def coordinate(value):
                if (isinstance(value, (bool, np.bool_, np.timedelta64))
                        or not isinstance(value, (int, np.integer))):
                    raise TypeError('sparse connectivity requires integer fiber indices')
                index = operator.index(value)
                if not -n <= index < n:
                    raise IndexError(f'fiber index {index} is out of bounds for {n} fibers')
                return index % n

            endpoints = np.empty((len(self.crosslinks), 2), dtype=np.intp)
            for row, link in enumerate(self.crosslinks):
                endpoints[row] = coordinate(link.fiber_i), coordinate(link.fiber_j)
            entries = (endpoints.ravel(), endpoints[:, ::-1].ravel())
            return coo_matrix((np.ones(endpoints.size, dtype=int), entries),
                              shape=(n, n)).tocsr()

        conn = np.zeros((n, n), dtype=int)
        for cl in self.crosslinks:
            conn[cl.fiber_i, cl.fiber_j] += 1
            conn[cl.fiber_j, cl.fiber_i] += 1
        return conn

    def degree_distribution(self) -> np.ndarray:
        """Number of crosslinks per fiber."""
        n = self.num_fibers
        degrees = np.zeros(n, dtype=int)
        for cl in self.crosslinks:
            degrees[cl.fiber_i] += 1
            degrees[cl.fiber_j] += 1
        return degrees

    def detect_contacts(self, threshold: float = None, tree_leafsize: int = 16) -> List[Crosslink]:
        """Detect potential contacts (near-intersections) between fibers.
        
        Parameters
        ----------
        threshold : float
            Distance threshold for contact detection. Defaults to 2 * mean_radius.
        """
        if threshold is None:
            threshold = 2.0 * self.mean_radius if self.fibers else 1.0
        
        if not self.fibers:
            return []
        
        all_points = []
        point_to_fiber = []
        point_to_param = []
        
        for i, f in enumerate(self.fibers):
            for j, pt in enumerate(f.centerline):
                all_points.append(pt)
                point_to_fiber.append(i)
                point_to_param.append(j / max(len(f.centerline) - 1, 1))
        
        if len(all_points) < 2:
            return []
        
        all_points = np.array(all_points, dtype=np.float64)
        
        # Validate points
        if not np.all(np.isfinite(all_points)):
            return []
        
        # Use cKDTree on Linux, pure numpy on macOS/Windows (scipy KDTree segfaults)
        import sys as _sys
        _use_kdtree = _sys.platform == 'linux'
        
        if _use_kdtree:
            from scipy.spatial import cKDTree as _SciKDTree
            tree = _SciKDTree(all_points, leafsize=tree_leafsize)
            pairs = tree.query_pairs(threshold)
        else:
            # Fallback: use scipy.spatial.distance.cdist (no KDTree, safe on all platforms)
            pairs = set()
            try:
                from scipy.spatial.distance import cdist
                # Process in chunks to manage memory
                chunk_size = 2000
                n = len(all_points)
                for start in range(0, n, chunk_size):
                    end = min(start + chunk_size, n)
                    dists = cdist(all_points[start:end], all_points, metric='euclidean')
                    for local_i in range(end - start):
                        pi = start + local_i
                        for pj in range(pi + 1, n):
                            if dists[local_i, pj] <= threshold:
                                pairs.add((pi, pj))
            except Exception:
                pass  # If cdist also fails, return empty contacts
        
        contacts = []
        seen = set()
        for (pi, pj) in pairs:
            fi, fj = point_to_fiber[pi], point_to_fiber[pj]
            if fi == fj:
                continue
            key = (min(fi, fj), max(fi, fj))
            if key in seen:
                continue
            seen.add(key)
            mid_pt = 0.5 * (all_points[pi] + all_points[pj])
            contacts.append(Crosslink(
                fiber_i=fi, fiber_j=fj,
                param_i=point_to_param[pi], param_j=point_to_param[pj],
                position=mid_pt,
                crosslink_type="contact",
            ))
        
        return contacts

    def auto_crosslink(self, threshold: float = None, crosslink_type: str = "welded"):
        """Automatically detect and add crosslinks based on proximity."""
        contacts = self.detect_contacts(threshold)
        for c in contacts:
            c.crosslink_type = crosslink_type
            self.add_crosslink(c)


    def connect_components(self, max_gap: float = None, strategy: str = "nearest") -> int:
        """Join disconnected components at their selected nearest samples.

        Exactly coincident samples receive one weld and no new fiber. When
        the samples differ, a bridging fiber carries one weld at each end.
        
        Parameters
        ----------
        max_gap : float, optional
            Maximum gap distance to bridge. Defaults to 10 * mean_fiber_length.
        strategy : str
            "nearest" bridges nearest points, "centroid" bridges component centroids.
            
        Returns
        -------
        int : Number of components successfully joined.
        """
        from collections import defaultdict
        
        # Build adjacency from crosslinks
        adj = defaultdict(set)
        for cl in self.crosslinks:
            adj[cl.fiber_i].add(cl.fiber_j)
            adj[cl.fiber_j].add(cl.fiber_i)
        
        # Find components
        visited = set()
        components = []
        for start in range(self.num_fibers):
            if start not in visited:
                comp = set()
                queue = [start]
                while queue:
                    node = queue.pop(0)
                    if node in visited:
                        continue
                    visited.add(node)
                    comp.add(node)
                    queue.extend(adj[node] - visited)
                components.append(comp)
        
        if len(components) <= 1:
            return 0
        
        if max_gap is None:
            max_gap = 10 * self.mean_fiber_length if self.mean_fiber_length > 0 else 50.0
        
        mat = self.fibers[0].material if self.fibers else Material()
        components_joined = 0
        
        # Sort components by size (largest first)
        components.sort(key=len, reverse=True)
        
        # Bridge each component to the largest component
        main_comp = components[0]
        
        for comp in components[1:]:
            best_dist = float('inf')
            best_pts = None
            best_fibers = None
            best_samples = None
            
            # Find nearest pair of fibers between main and this component
            main_fibers = list(main_comp)
            other_fibers = list(comp)
            
            # Sample to keep computation reasonable
            if len(main_fibers) > 50:
                main_sample = np.random.choice(main_fibers, 50, replace=False)
            else:
                main_sample = main_fibers
            if len(other_fibers) > 50:
                other_sample = np.random.choice(other_fibers, 50, replace=False)
            else:
                other_sample = other_fibers
            
            for fi in main_sample:
                for fj in other_sample:
                    ci = self.fibers[fi].centerline
                    cj = self.fibers[fj].centerline
                    # Find nearest points
                    from scipy.spatial.distance import cdist
                    dists = cdist(ci, cj)
                    idx = np.unravel_index(np.argmin(dists), dists.shape)
                    d = dists[idx]
                    if d < best_dist:
                        best_dist = d
                        best_pts = (ci[idx[0]], cj[idx[1]])
                        best_fibers = (int(fi), int(fj))
                        best_samples = idx
            
            if best_dist <= max_gap and best_pts is not None:
                p1, p2 = best_pts
                fi, fj = best_fibers
                si, sj = best_samples
                # A single-sample fiber attaches at parameter zero.
                pi = int(si) / max(1, len(self.fibers[fi].centerline) - 1)
                pj = int(sj) / max(1, len(self.fibers[fj].centerline) - 1)
                
                if np.array_equal(p1, p2):
                    self.add_crosslink(Crosslink(
                        fiber_i=fi, fiber_j=fj,
                        param_i=pi, param_j=pj,
                        position=p1.copy(), crosslink_type="welded",
                    ))
                else:
                    # Add bridging fiber
                    bridge = Fiber.straight(
                        p1, p2,
                        radius=self.fibers[fi].radius,
                        material=mat,
                        fiber_id=self.num_fibers,
                        segments=4,
                    )
                    self.add_fiber(bridge)

                    # Add crosslinks at both ends
                    self.add_crosslink(Crosslink(
                        fiber_i=fi, fiber_j=self.num_fibers - 1,
                        param_i=pi, param_j=0.0,
                        position=p1.copy(), crosslink_type="welded",
                    ))
                    self.add_crosslink(Crosslink(
                        fiber_i=fj, fiber_j=self.num_fibers - 1,
                        param_i=pj, param_j=1.0,
                        position=p2.copy(), crosslink_type="welded",
                    ))

                # Merge this component into main
                main_comp.update(comp)
                components_joined += 1
        
        return components_joined

    def to_networkx(self):
        """Convert to a NetworkX graph for topology analysis."""
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("networkx is required. Install with: pip install networkx")
        G = nx.Graph()
        for i, f in enumerate(self.fibers):
            G.add_node(i, fiber=f, length=f.length, orientation=f.direction.tolist())
        for cl in self.crosslinks:
            G.add_edge(
                cl.fiber_i, cl.fiber_j,
                crosslink_type=cl.crosslink_type,
                position=cl.position.tolist(),
            )
        return G

    def to_dict(self) -> Dict[str, Any]:
        """Serialize network to dictionary."""
        return {
            "version": "0.1.0",
            "dimension": self.dimension,
            "periodic": self.periodic,
            "box_size": self.box_size.tolist() if self.box_size is not None else None,
            "metadata": self.metadata,
            "fibers": [f.to_dict() for f in self.fibers],
            "crosslinks": [
                {
                    "fiber_i": cl.fiber_i,
                    "fiber_j": cl.fiber_j,
                    "param_i": cl.param_i,
                    "param_j": cl.param_j,
                    "position": cl.position.tolist(),
                    "crosslink_type": cl.crosslink_type,
                    "strength": cl.strength,
                    "stiffness": cl.stiffness,
                }
                for cl in self.crosslinks
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FiberNetwork":
        """Create FiberNetwork from its native dictionary representation.

        Explicit StructureGraph node/edge arrays without either legacy network
        field are rejected rather than silently discarded. Other legacy header
        and version handling is unchanged; this is not full schema validation.
        """
        if (isinstance(data.get("nodes"), list)
                and isinstance(data.get("edges"), list)
                and "fibers" not in data and "crosslinks" not in data):
            raise ValueError(
                "StructureGraph nodes/edges cannot be loaded as FiberNetwork; "
                "use StructureGraph.from_dict(), StructureGraph.load_json(), "
                "or fibernet.api.load()")
        net = cls(
            dimension=data.get("dimension", 3),
            periodic=data.get("periodic", False),
            box_size=np.array(data["box_size"]) if data.get("box_size") else None,
            metadata=data.get("metadata", {}),
        )
        for fd in data.get("fibers", []):
            net.add_fiber(Fiber.from_dict(fd))
        for cd in data.get("crosslinks", []):
            net.add_crosslink(Crosslink(
                fiber_i=cd["fiber_i"], fiber_j=cd["fiber_j"],
                param_i=cd["param_i"], param_j=cd["param_j"],
                position=np.array(cd["position"]),
                crosslink_type=cd.get("crosslink_type", "welded"),
                strength=cd.get("strength", float('inf')),
                stiffness=cd.get("stiffness", float('inf')),
            ))
        return net

    def save_json(self, filepath: str):
        """Save network JSON, preserving an existing file if serialization fails.

        Serialization completes before opening the destination. Parent directories
        are not created. This protects against serialization errors, not partial
        filesystem writes or interruption after the destination has been opened.
        """
        payload = json.dumps(self.to_dict(), indent=2)
        with open(filepath, 'w') as f:
            f.write(payload)

    @classmethod
    def load_json(cls, filepath: str) -> "FiberNetwork":
        """Load network from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f, object_pairs_hook=_unique_json_object_pairs)
        return cls.from_dict(data)

    def save_hdf5(self, filepath: str, *, format_version: str = '0.1.0'):
        """Atomically replace a new or writable single-link regular HDF5 file.

        The default format, 0.1.0, preserves the legacy layout. Format 0.2.0
        also saves JSON cross-section parameters and the strength and stiffness
        of each crosslink.

        Accept str or PathLike[str], not bytes. Existing targets with no write
        mode bits, symlinks and nonregular/multiply-linked files are rejected.
        Mode checks are a conservative policy, not an ACL authorization check.
        Existing mode bits are retained, but ownership/ACLs/xattrs are not.
        New files have private temporary-file permissions. No durability or
        concurrent-directory-mutation guarantee is provided.
        """
        import os
        import stat
        import tempfile
        import h5py

        if not isinstance(format_version, str) or format_version not in ('0.1.0', '0.2.0'):
            raise ValueError("HDF5 format_version must be '0.1.0' or '0.2.0'")
        extended = format_version == '0.2.0'
        path = os.fspath(filepath)
        if not isinstance(path, str):
            raise TypeError("save_hdf5 requires str or PathLike[str], not bytes")
        target = os.path.abspath(path)
        metadata_json = json.dumps(self.metadata)
        materials_json = [json.dumps(f.material.to_dict()) for f in self.fibers]
        if extended:
            section_params_json = [_hdf5_section_params_json(f.cross_section_params)
                                   for f in self.fibers]
        try:
            previous = os.lstat(target)
        except FileNotFoundError:
            previous_mode = None
        else:
            if not stat.S_ISREG(previous.st_mode) or previous.st_nlink != 1:
                raise ValueError("save_hdf5 requires a new or single-link regular file")
            previous_mode = stat.S_IMODE(previous.st_mode)
            if not previous_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
                raise PermissionError("save_hdf5 rejects targets with no write mode bits")

        fd, temporary = tempfile.mkstemp(
            prefix=".fibernet-", suffix=".h5.tmp", dir=os.path.dirname(target)
        )
        try:
            try:
                os.close(fd)
            except BaseException as error:
                error.hdf5_descriptor_close_uncertain = True
                raise
            hf = h5py.File(temporary, 'w')
            try:
                hf.attrs['version'] = '0.2.0' if extended else '0.1.0'
                hf.attrs['dimension'] = self.dimension
                hf.attrs['periodic'] = self.periodic
                if self.box_size is not None:
                    hf.attrs['box_size'] = self.box_size
                hf.attrs['metadata'] = metadata_json
                fiber_grp = hf.create_group('fibers')
                for i, f in enumerate(self.fibers):
                    fg = fiber_grp.create_group(f'fiber_{i}')
                    fg.create_dataset('centerline', data=f.centerline)
                    fg.attrs['radius'] = f.radius
                    fg.attrs['fiber_id'] = f.fiber_id
                    fg.attrs['cross_section'] = f.cross_section.value
                    fg.attrs['segments'] = f.segments
                    fg.attrs['material'] = materials_json[i]
                    if extended:
                        fg.attrs['cross_section_params'] = section_params_json[i]
                cl_grp = hf.create_group('crosslinks')
                for i, cl in enumerate(self.crosslinks):
                    cg = cl_grp.create_group(f'cl_{i}')
                    cg.attrs['fiber_i'] = cl.fiber_i
                    cg.attrs['fiber_j'] = cl.fiber_j
                    cg.attrs['param_i'] = cl.param_i
                    cg.attrs['param_j'] = cl.param_j
                    cg.create_dataset('position', data=cl.position)
                    cg.attrs['crosslink_type'] = cl.crosslink_type
                    if extended:
                        for name in ('strength', 'stiffness'):
                            value = getattr(cl, name)
                            array = np.asarray(value)
                            if (array.ndim != 0 or array.dtype.kind not in 'iuf'
                                    or np.ma.is_masked(value)):
                                raise TypeError(f"HDF5 crosslink {name} must be a real numeric scalar")
                            cg.attrs[name] = value
            except BaseException as error:
                try:
                    hf.close()
                except BaseException as close_error:
                    error.hdf5_close_error = close_error
                raise
            else:
                hf.close()
            if previous_mode is not None:
                os.chmod(temporary, previous_mode)
            os.replace(temporary, target)
        except BaseException as error:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            except OSError as cleanup_error:
                error.hdf5_cleanup_error = cleanup_error
            raise

    def summary(self) -> str:
        """Print a summary of the network."""
        lines = [
            f"FiberNetwork Summary",
            f"  Fibers:      {self.num_fibers}",
            f"  Crosslinks:  {self.num_crosslinks}",
            f"  Dimension:   {self.dimension}D",
            f"  Total Length: {self.total_length:.3f}",
            f"  Mean Length:  {self.mean_fiber_length:.3f}",
            f"  Mean Radius:  {self.mean_radius:.4f}",
            f"  Volume Frac:  {self.density():.4f}",
            f"  Periodic:     {self.periodic}",
        ]
        if self.box_size is not None:
            lines.append(f"  Box Size:     {self.box_size}")
        bb_min, bb_max = self.bounding_box()
        lines.append(f"  BBox Min:     {bb_min}")
        lines.append(f"  BBox Max:     {bb_max}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"FiberNetwork(fibers={self.num_fibers}, crosslinks={self.num_crosslinks}, "
            f"dim={self.dimension}D, L_total={self.total_length:.2f})"
        )

    def describe(self) -> str:
        """Return a statistical summary of the network.

        Returns
        -------
        str
            Multi-line string with network statistics.
        """
        import numpy as np

        lines = [
            f"FiberNetwork Summary",
            f"{'='*40}",
            f"Fibers: {self.num_fibers}",
            f"Crosslinks: {self.num_crosslinks}",
            f"Dimension: {self.dimension}D",
        ]

        if self.num_fibers == 0:
            return '\n'.join(lines)

        lengths = np.array([f.length for f in self.fibers])
        radii = np.array([f.radius for f in self.fibers])
        tortuosities = np.array([f.tortuosity() for f in self.fibers])

        lines.extend([
            f"",
            f"Fiber Length:",
            f"  Mean: {np.mean(lengths):.3f}",
            f"  Std:  {np.std(lengths):.3f}",
            f"  Min:  {np.min(lengths):.3f}",
            f"  Max:  {np.max(lengths):.3f}",
            f"",
            f"Fiber Radius:",
            f"  Mean: {np.mean(radii):.4f}",
            f"  Std:  {np.std(radii):.4f}",
            f"",
            f"Tortuosity:",
            f"  Mean: {np.mean(tortuosities):.3f}",
            f"  Std:  {np.std(tortuosities):.3f}",
        ])

        # Connectivity
        if self.num_crosslinks > 0:
            lines.append(f"")
            lines.append(f"Connectivity:")
            lines.append(f"  Crosslinks per fiber: {2*self.num_crosslinks/self.num_fibers:.2f}")

        # Material
        mat_names = list(set(f.material.name for f in self.fibers))
        if len(mat_names) == 1:
            lines.append(f"Material: {mat_names[0]}")
        else:
            lines.append(f"Materials: {', '.join(mat_names)}")

        return '\n'.join(lines)

    def plot(self, **kwargs):
        """Quick visualization of the fiber network.

        For 2D networks, creates a 2D plot.
        For 3D networks, creates a 3D render.

        Parameters
        ----------
        **kwargs
            Passed to plot_network_2d or render_network_3d.
            
            2D options:
            - ax: matplotlib axes
            - color_by: 'uniform', 'stress', 'strain', 'material' (default: 'uniform')
            - colormap: matplotlib colormap name (default: 'viridis')
            - show_crosslinks: bool (default: True)
            - line_width: float (default: 1.0)
            - title: str
            - figsize: tuple (default: (8, 8))
            - save_path: str
            
            3D options:
            - color_by: 'uniform', 'stress', 'strain', 'material' (default: 'uniform')
            - colormap: str (default: 'viridis')
            - tube_radius: float
            - show_crosslinks: bool (default: True)
            - background: str (default: 'white')
            - window_size: tuple (default: (1024, 768))
            - save_path: str

        Returns
        -------
        fig : matplotlib.figure.Figure or None
            The matplotlib figure (2D only).

        Examples
        --------
        >>> net = gen.random_straight_2d(num_fibers=50, fiber_length=10, box_size=(30, 30))
        >>> net.plot(color_by='material', show_crosslinks=False)
        >>>
        >>> net_3d = gen.random_straight_3d(num_fibers=30, fiber_length=8, box_size=(20, 20, 20))
        >>> net_3d.plot(background='black')  # Opens 3D viewer
        """
        if self.dimension == 2:
            from fibernet.viz import plot_network_2d
            import inspect
            sig = inspect.signature(plot_network_2d)
            valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
            return plot_network_2d(self, **valid_kwargs)
        else:
            from fibernet.viz import render_network_3d
            import inspect
            sig = inspect.signature(render_network_3d)
            valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
            return render_network_3d(self, **valid_kwargs)

    def plot_statistics(self, figsize=(12, 4)):
        """Plot statistical distributions of network properties.

        Creates a figure with three subplots:
        1. Length distribution
        2. Orientation distribution (2D only)
        3. Tortuosity distribution

        Parameters
        ----------
        figsize : tuple, optional
            Figure size, by default (12, 4).

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib figure.

        Examples
        --------
        >>> net = gen.random_straight_2d(num_fibers=100, fiber_length=10, box_size=(30, 30))
        >>> fig = net.plot_statistics()
        >>> fig.savefig('statistics.png', dpi=150, bbox_inches='tight')
        """
        import matplotlib.pyplot as plt
        import numpy as np

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Length distribution
        lengths = [f.length for f in self.fibers]
        length_range = np.ptp(lengths)
        min_range = np.mean(lengths) * 1e-10  # Relative threshold
        if length_range > min_range:
            axes[0].hist(lengths, bins=20, edgecolor='black', alpha=0.7)
        else:
            # All same length (within precision) - use single bar
            axes[0].bar([np.mean(lengths)], [len(lengths)], width=np.mean(lengths)*0.1, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Fiber Length')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Length Distribution')
        axes[0].axvline(np.mean(lengths), color='red', linestyle='--', label=f'Mean: {np.mean(lengths):.2f}')
        axes[0].legend()

        # Orientation distribution
        if self.dimension == 2:
            orientations = [f.direction for f in self.fibers]
            angles = np.degrees(orientations)
            axes[1].hist(angles, bins=36, range=(-180, 180), edgecolor='black', alpha=0.7)
            axes[1].set_xlabel('Orientation Angle (degrees)')
            axes[1].set_ylabel('Count')
            axes[1].set_title('Orientation Distribution')
        else:
            # For 3D, plot polar angles
            orientations = [f.direction for f in self.fibers]
            # Extract theta (polar angle)
            thetas = [np.degrees(np.arccos(ori[2] / np.linalg.norm(ori))) for ori in orientations]
            axes[1].hist(thetas, bins=20, edgecolor='black', alpha=0.7)
            axes[1].set_xlabel('Polar Angle (degrees)')
            axes[1].set_ylabel('Count')
            axes[1].set_title('Orientation Distribution')

        # Tortuosity distribution
        tortuosities = [f.tortuosity() for f in self.fibers]
        tort_range = np.ptp(tortuosities)
        min_tort_range = np.mean(tortuosities) * 1e-10
        if tort_range > min_tort_range:
            axes[2].hist(tortuosities, bins=20, edgecolor='black', alpha=0.7)
        else:
            axes[2].bar([np.mean(tortuosities)], [len(tortuosities)], width=0.05, edgecolor='black', alpha=0.7)
        axes[2].set_xlabel('Tortuosity')
        axes[2].set_ylabel('Count')
        axes[2].set_title('Tortuosity Distribution')
        axes[2].axvline(np.mean(tortuosities), color='red', linestyle='--', label=f'Mean: {np.mean(tortuosities):.3f}')
        axes[2].legend()

        plt.tight_layout()
        return fig

    def validate(self) -> Dict[str, Any]:
        """Validate network integrity and return diagnostic information.

        Checks for:
        - Empty networks
        - Invalid fiber geometries (zero-length, nonfinite coordinates)
        - Invalid crosslinks (invalid fiber list indices)
        - Disconnected components

        Returns
        -------
        dict
            Validation results with keys:
            - 'valid': bool - Overall validity
            - 'errors': list of str - Critical issues
            - 'warnings': list of str - Non-critical issues
            - 'stats': dict - Network statistics

        If there are crosslink records, component counts use all fibers and
        only the records with valid references. Nonfinite coordinates leave
        the length statistics as NaN.

        Examples
        --------
        >>> net = gen.random_straight_2d(num_fibers=50, fiber_length=10, box_size=(30, 30))
        >>> result = net.validate()
        >>> if result['valid']:
        ...     print("Network is valid")
        ... else:
        ...     print(f"Errors: {result['errors']}")
        """
        from collections import defaultdict
        
        errors = []
        warnings = []
        stats = {
            'num_fibers': self.num_fibers,
            'num_crosslinks': self.num_crosslinks,
            'dimension': self.dimension,
        }

        # Check for empty network
        if self.num_fibers == 0:
            errors.append("Network has no fibers")
            return {'valid': False, 'errors': errors, 'warnings': warnings, 'stats': stats}

        # Check each fiber and retain its length for the summary.
        zero_length_count = 0
        nan_count = 0
        infinite_count = 0
        lengths = []
        for i, fiber in enumerate(self.fibers):
            coordinates = fiber.centerline
            if np.isfinite(coordinates).all():
                length = fiber.length
                if length < 1e-12:
                    zero_length_count += 1
            else:
                nan_count += int(np.isnan(coordinates).any())
                infinite_count += int(np.isinf(coordinates).any())
                if len(coordinates) == 1:
                    zero_length_count += 1
                length = np.nan
            lengths.append(length)
            # Check radius
            if fiber.radius <= 0:
                warnings.append(f"Fiber {i} has non-positive radius: {fiber.radius}")

        if nan_count > 0:
            errors.append(f"{nan_count} fiber(s) have NaN values in centerline")
        if infinite_count > 0:
            errors.append(f"{infinite_count} fiber(s) have infinite values in centerline")
        
        if zero_length_count > 0:
            errors.append(f"{zero_length_count} fiber(s) have zero length")

        def fiber_index(value):
            if (isinstance(value, (bool, np.bool_, np.timedelta64))
                    or not isinstance(value, (int, np.integer))):
                return None
            try:
                index = operator.index(value)
            except TypeError:
                return None
            return index if 0 <= index < self.num_fibers else None

        # Use only fully valid records for connectivity.
        adj = defaultdict(set)
        invalid_crosslinks = 0
        for cl in self.crosslinks:
            i = fiber_index(cl.fiber_i)
            j = fiber_index(cl.fiber_j)
            if i is None or j is None:
                invalid_crosslinks += 1
                continue
            adj[i].add(j)
            adj[j].add(i)

        if invalid_crosslinks > 0:
            errors.append(f"{invalid_crosslinks} crosslink(s) reference invalid fiber IDs")

        # Check connectivity (if crosslinks exist)
        if self.num_crosslinks > 0:
            # BFS to find connected components
            visited = set()
            components = 0
            for start in range(self.num_fibers):
                if start not in visited:
                    components += 1
                    queue = [start]
                    while queue:
                        node = queue.pop(0)
                        if node in visited:
                            continue
                        visited.add(node)
                        queue.extend(adj[node] - visited)
            
            stats['num_components'] = components
            if components > 1:
                warnings.append(f"Network has {components} disconnected components")
        else:
            warnings.append("No crosslinks - fibers may not be mechanically connected")

        # Add statistics
        stats['mean_length'] = np.mean(lengths)
        stats['total_length'] = np.sum(lengths)
        stats['mean_radius'] = np.mean([f.radius for f in self.fibers])

        valid = len(errors) == 0
        return {'valid': valid, 'errors': errors, 'warnings': warnings, 'stats': stats}


    def connect_components(self, max_gap: float = None, strategy: str = "nearest") -> int:
        """Join disconnected components at their selected nearest samples.

        Exactly coincident samples receive one weld and no new fiber. When
        the samples differ, a bridging fiber carries one weld at each end.
        
        Parameters
        ----------
        max_gap : float, optional
            Maximum gap distance to bridge. Defaults to 10 * mean_fiber_length.
        strategy : str
            "nearest" bridges nearest points, "centroid" bridges component centroids.
            
        Returns
        -------
        int : Number of components successfully joined.
        """
        from collections import defaultdict
        
        # Build adjacency from crosslinks
        adj = defaultdict(set)
        for cl in self.crosslinks:
            adj[cl.fiber_i].add(cl.fiber_j)
            adj[cl.fiber_j].add(cl.fiber_i)
        
        # Find components
        visited = set()
        components = []
        for start in range(self.num_fibers):
            if start not in visited:
                comp = set()
                queue = [start]
                while queue:
                    node = queue.pop(0)
                    if node in visited:
                        continue
                    visited.add(node)
                    comp.add(node)
                    queue.extend(adj[node] - visited)
                components.append(comp)
        
        if len(components) <= 1:
            return 0
        
        if max_gap is None:
            max_gap = 10 * self.mean_fiber_length if self.mean_fiber_length > 0 else 50.0
        
        mat = self.fibers[0].material if self.fibers else Material()
        components_joined = 0
        
        # Sort components by size (largest first)
        components.sort(key=len, reverse=True)
        
        # Bridge each component to the largest component
        main_comp = components[0]
        
        for comp in components[1:]:
            best_dist = float('inf')
            best_pts = None
            best_fibers = None
            best_samples = None
            
            # Find nearest pair of fibers between main and this component
            main_fibers = list(main_comp)
            other_fibers = list(comp)
            
            # Sample to keep computation reasonable
            if len(main_fibers) > 50:
                main_sample = np.random.choice(main_fibers, 50, replace=False)
            else:
                main_sample = main_fibers
            if len(other_fibers) > 50:
                other_sample = np.random.choice(other_fibers, 50, replace=False)
            else:
                other_sample = other_fibers
            
            for fi in main_sample:
                for fj in other_sample:
                    ci = self.fibers[fi].centerline
                    cj = self.fibers[fj].centerline
                    # Find nearest points
                    from scipy.spatial.distance import cdist
                    dists = cdist(ci, cj)
                    idx = np.unravel_index(np.argmin(dists), dists.shape)
                    d = dists[idx]
                    if d < best_dist:
                        best_dist = d
                        best_pts = (ci[idx[0]], cj[idx[1]])
                        best_fibers = (int(fi), int(fj))
                        best_samples = idx
            
            if best_dist <= max_gap and best_pts is not None:
                p1, p2 = best_pts
                fi, fj = best_fibers
                si, sj = best_samples
                # A single-sample fiber attaches at parameter zero.
                pi = int(si) / max(1, len(self.fibers[fi].centerline) - 1)
                pj = int(sj) / max(1, len(self.fibers[fj].centerline) - 1)
                
                if np.array_equal(p1, p2):
                    self.add_crosslink(Crosslink(
                        fiber_i=fi, fiber_j=fj,
                        param_i=pi, param_j=pj,
                        position=p1.copy(), crosslink_type="welded",
                    ))
                else:
                    # Add bridging fiber
                    bridge = Fiber.straight(
                        p1, p2,
                        radius=self.fibers[fi].radius,
                        material=mat,
                        fiber_id=self.num_fibers,
                        segments=4,
                    )
                    self.add_fiber(bridge)

                    # Add crosslinks at both ends
                    self.add_crosslink(Crosslink(
                        fiber_i=fi, fiber_j=self.num_fibers - 1,
                        param_i=pi, param_j=0.0,
                        position=p1.copy(), crosslink_type="welded",
                    ))
                    self.add_crosslink(Crosslink(
                        fiber_i=fj, fiber_j=self.num_fibers - 1,
                        param_i=pj, param_j=1.0,
                        position=p2.copy(), crosslink_type="welded",
                    ))

                # Merge this component into main
                main_comp.update(comp)
                components_joined += 1
        
        return components_joined

    def to_networkx(self):
        """Convert fiber network to NetworkX graph.

        Creates a graph where:
        - Nodes represent fibers
        - Edges represent crosslinks between fibers
        - Node attributes: length, radius, material
        - Edge attributes: stiffness, strength

        Returns
        -------
        networkx.Graph
            Graph representation of the network.

        Examples
        --------
        >>> net = gen.random_straight_2d(num_fibers=50, fiber_length=10, box_size=(30, 30))
        >>> G = net.to_networkx()
        >>> print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        >>> import networkx as nx
        >>> print(f"Clustering coefficient: {nx.average_clustering(G):.3f}")
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("networkx is required. Install with: pip install networkx")

        G = nx.Graph()

        # Add nodes (fibers)
        for fiber in self.fibers:
            G.add_node(fiber.fiber_id,
                      length=fiber.length,
                      radius=fiber.radius,
                      material=fiber.material.name,
                      youngs_modulus=fiber.material.youngs_modulus)

        # Add edges (crosslinks)
        for cl in self.crosslinks:
            G.add_edge(cl.fiber_i, cl.fiber_j,
                      stiffness=cl.stiffness,
                      strength=cl.strength,
                      crosslink_type=cl.crosslink_type)

        return G
