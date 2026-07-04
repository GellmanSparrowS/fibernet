"""
Taichi-accelerated simulations for large fiber networks.

Provides GPU/CPU-parallel implementations of:
- Beam FEM assembly (parallel element processing)
- Mass-spring dynamics (parallel force computation)
- Random network generation (parallel fiber deposition)

Uses Taichi's CPU backend for parallelism without GPU requirement.
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass

try:
    import taichi as ti
    HAS_TAICHI = True
except ImportError:
    HAS_TAICHI = False


@dataclass
class AcceleratedResult:
    """Container for accelerated simulation results."""
    displacements: np.ndarray = None
    forces: np.ndarray = None
    positions: List[np.ndarray] = None
    energy: float = 0.0
    time_seconds: float = 0.0


class TaichiEngine:
    """Taichi-accelerated computation engine.
    
    Parameters
    ----------
    arch : str
        'cpu' or 'gpu'. Defaults to 'cpu'.
    num_threads : int
        Number of CPU threads.
    """
    
    def __init__(self, arch: str = "cpu", num_threads: int = 4):
        if not HAS_TAICHI:
            raise ImportError("Taichi required. Install with: pip install taichi")
        
        try:
            initialized = ti.is_initialized()
        except AttributeError:
            initialized = False
        
        if not initialized:
            try:
                if arch == "cpu":
                    ti.init(arch=ti.cpu, cpu_max_num_threads=num_threads)
                elif arch == "gpu":
                    ti.init(arch=ti.gpu)
                else:
                    ti.init(arch=ti.cpu)
            except RuntimeError:
                pass
    
    def parallel_force_computation(
        self,
        positions: np.ndarray,
        rest_lengths: np.ndarray,
        stiffness: np.ndarray,
        edges: np.ndarray,
    ) -> np.ndarray:
        """Compute spring forces in parallel using Taichi.
        
        Parameters
        ----------
        positions : np.ndarray
            Node positions (N, 3).
        rest_lengths : np.ndarray
            Rest lengths for each edge.
        stiffness : np.ndarray
            Spring stiffness for each edge.
        edges : np.ndarray
            Edge connectivity (M, 2) - node indices.
        """
        num_nodes = positions.shape[0]
        num_edges = edges.shape[0]
        
        pos = ti.Vector.field(3, dtype=ti.f64, shape=num_nodes)
        forces = ti.Vector.field(3, dtype=ti.f64, shape=num_nodes)
        edge_arr = ti.Vector.field(2, dtype=ti.i32, shape=num_edges)
        L0 = ti.field(dtype=ti.f64, shape=num_edges)
        k = ti.field(dtype=ti.f64, shape=num_edges)
        f_temp = ti.Vector.field(3, dtype=ti.f64, shape=num_edges)
        
        pos.from_numpy(positions.astype(np.float64))
        edge_arr.from_numpy(edges.astype(np.int32))
        L0.from_numpy(rest_lengths.astype(np.float64))
        k.from_numpy(stiffness.astype(np.float64))
        
        @ti.kernel
        def compute_edge_forces():
            for e in range(num_edges):
                i = edge_arr[e][0]
                j = edge_arr[e][1]
                diff = pos[j] - pos[i]
                L = ti.sqrt(diff[0]**2 + diff[1]**2 + diff[2]**2)
                if L > 1e-12:
                    strain = (L - L0[e]) / L0[e]
                    force_mag = k[e] * strain
                    direction = diff / L
                    f_temp[e] = force_mag * direction
                else:
                    f_temp[e] = ti.Vector([0.0, 0.0, 0.0])
        
        @ti.kernel
        def zero_forces():
            for n in range(num_nodes):
                forces[n] = ti.Vector([0.0, 0.0, 0.0])
        
        @ti.kernel
        def accumulate_forces():
            for e in range(num_edges):
                i = edge_arr[e][0]
                j = edge_arr[e][1]
                ti.atomic_add(forces[i], f_temp[e])
                ti.atomic_add(forces[j], -f_temp[e])
        
        zero_forces()
        compute_edge_forces()
        accumulate_forces()
        
        return forces.to_numpy()
    
    def parallel_dynamics(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        masses: np.ndarray,
        rest_lengths: np.ndarray,
        stiffness: np.ndarray,
        edges: np.ndarray,
        dt: float = 1e-6,
        damping: float = 0.01,
        num_steps: int = 1000,
        save_interval: int = 100,
        fixed_nodes: Optional[List[int]] = None,
    ) -> AcceleratedResult:
        """Run parallel mass-spring dynamics using Taichi.
        
        Parameters
        ----------
        positions : np.ndarray
            Initial node positions (N, 3).
        velocities : np.ndarray
            Initial node velocities (N, 3).
        masses : np.ndarray
            Node masses (N,).
        edges : np.ndarray
            Edge connectivity (M, 2).
        dt : float
            Time step.
        num_steps : int
            Number of steps.
        """
        import time
        start_time = time.time()
        
        num_nodes = positions.shape[0]
        num_edges = edges.shape[0]
        fixed_set = set(fixed_nodes) if fixed_nodes else set()
        
        pos = ti.Vector.field(3, dtype=ti.f64, shape=num_nodes)
        vel = ti.Vector.field(3, dtype=ti.f64, shape=num_nodes)
        mass = ti.field(dtype=ti.f64, shape=num_nodes)
        frc = ti.Vector.field(3, dtype=ti.f64, shape=num_nodes)
        edge_arr = ti.Vector.field(2, dtype=ti.i32, shape=num_edges)
        L0 = ti.field(dtype=ti.f64, shape=num_edges)
        k_spring = ti.field(dtype=ti.f64, shape=num_edges)
        f_temp = ti.Vector.field(3, dtype=ti.f64, shape=num_edges)
        is_fixed = ti.field(dtype=ti.i32, shape=num_nodes)
        
        pos.from_numpy(positions.astype(np.float64))
        vel.from_numpy(velocities.astype(np.float64))
        mass.from_numpy(masses.astype(np.float64))
        edge_arr.from_numpy(edges.astype(np.int32))
        L0.from_numpy(rest_lengths.astype(np.float64))
        k_spring.from_numpy(stiffness.astype(np.float64))
        
        fixed_arr = np.zeros(num_nodes, dtype=np.int32)
        for n in fixed_set:
            fixed_arr[n] = 1
        is_fixed.from_numpy(fixed_arr)
        
        @ti.kernel
        def compute_forces():
            for n in range(num_nodes):
                frc[n] = ti.Vector([0.0, 0.0, 0.0])
            
            for e in range(num_edges):
                i = edge_arr[e][0]
                j = edge_arr[e][1]
                diff = pos[j] - pos[i]
                L = ti.sqrt(diff[0]**2 + diff[1]**2 + diff[2]**2)
                if L > 1e-12:
                    strain = (L - L0[e]) / L0[e]
                    force_mag = k_spring[e] * strain
                    direction = diff / L
                    f = force_mag * direction
                    ti.atomic_add(frc[i], f)
                    ti.atomic_add(frc[j], -f)
        
        @ti.kernel
        def integrate(damp: ti.f64, step_dt: ti.f64):
            for n in range(num_nodes):
                if is_fixed[n] == 0:
                    vel[n] = vel[n] + step_dt * (frc[n] - damp * vel[n]) / mass[n]
                    pos[n] = pos[n] + step_dt * vel[n]
        
        saved_positions = []
        for step in range(num_steps):
            compute_forces()
            integrate(damping, dt)
            
            if step % save_interval == 0:
                saved_positions.append(pos.to_numpy().copy())
        
        elapsed = time.time() - start_time
        
        return AcceleratedResult(
            positions=saved_positions,
            time_seconds=elapsed,
        )
    
    def parallel_generate_random_3d(
        self,
        num_fibers: int,
        fiber_length: float,
        box_size: Tuple[float, float, float],
        radius: float = 0.1,
        seed: int = 42,
    ) -> np.ndarray:
        """Generate random fiber endpoints in parallel.
        
        Returns array of shape (num_fibers, 2, 3) for start/end points.
        """
        rng = np.random.default_rng(seed)
        Lx, Ly, Lz = box_size
        
        centers = rng.uniform(0, 1, (num_fibers, 3)) * np.array([Lx, Ly, Lz])
        
        theta = rng.uniform(0, 2 * np.pi, num_fibers)
        cos_phi = rng.uniform(-1, 1, num_fibers)
        sin_phi = np.sqrt(1 - cos_phi**2)
        
        directions = np.column_stack([
            sin_phi * np.cos(theta),
            sin_phi * np.sin(theta),
            cos_phi,
        ])
        
        lengths = fiber_length * (0.5 + rng.uniform(size=num_fibers))
        
        starts = centers - 0.5 * lengths[:, None] * directions
        ends = centers + 0.5 * lengths[:, None] * directions
        
        endpoints = np.stack([starts, ends], axis=1)
        return endpoints


class TaichiFEMSolver:
    """
    Taichi-accelerated Finite Element Method solver for fiber networks.
    
    Provides GPU/CPU-parallel FEM assembly and solving for beam elements.
    
    Parameters
    ----------
    arch : str
        'cpu' or 'gpu'. Defaults to 'cpu'.
    num_threads : int
        Number of CPU threads.
    """
    
    def __init__(self, arch: str = "cpu", num_threads: int = 4):
        if not HAS_TAICHI:
            raise ImportError("Taichi required. Install with: pip install taichi")
        
        try:
            initialized = ti.is_initialized()
        except AttributeError:
            initialized = False
        
        if not initialized:
            try:
                if arch == "cpu":
                    ti.init(arch=ti.cpu, cpu_max_num_threads=num_threads)
                elif arch == "gpu":
                    ti.init(arch=ti.gpu)
                else:
                    ti.init(arch=ti.cpu)
            except RuntimeError:
                pass
    
    def solve_beam_network(
        self,
        node_positions: np.ndarray,
        elements: np.ndarray,
        youngs_modulus: float,
        radii: np.ndarray,
        fixed_nodes: List[int],
        applied_forces: np.ndarray,
    ) -> AcceleratedResult:
        """
        Solve beam network FEM using Taichi parallel assembly.
        
        Parameters
        ----------
        node_positions : np.ndarray
            Node positions (N, 3)
        elements : np.ndarray
            Element connectivity (E, 2) - node indices
        youngs_modulus : float
            Young's modulus (Pa)
        radii : np.ndarray
            Element radii (E,)
        fixed_nodes : list of int
            Fixed node indices
        applied_forces : np.ndarray
            Applied forces at nodes (N, 3)
        
        Returns
        -------
        AcceleratedResult
            Result with displacements and forces
        """
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
        import time
        
        start_time = time.time()
        
        num_nodes = node_positions.shape[0]
        num_elements = elements.shape[0]
        
        # Compute element properties
        lengths = np.zeros(num_elements)
        directions = np.zeros((num_elements, 3))
        
        for e in range(num_elements):
            i, j = elements[e]
            diff = node_positions[j] - node_positions[i]
            lengths[e] = np.linalg.norm(diff)
            if lengths[e] > 1e-12:
                directions[e] = diff / lengths[e]
        
        # Compute element stiffness (axial only for simplicity)
        areas = np.pi * radii**2
        axial_stiffness = youngs_modulus * areas / lengths
        
        # Parallel assembly using Taichi
        # Each element contributes to 6 DOFs (3 per node)
        num_dofs = num_nodes * 3
        
        # Use Taichi for parallel element stiffness computation
        ti_stiffness = ti.field(dtype=ti.f64, shape=(num_elements, 6, 6))
        ti_elements = ti.Vector.field(2, dtype=ti.i32, shape=num_elements)
        ti_directions = ti.Vector.field(3, dtype=ti.f64, shape=num_elements)
        ti_ks = ti.field(dtype=ti.f64, shape=num_elements)
        
        ti_elements.from_numpy(elements.astype(np.int32))
        ti_directions.from_numpy(directions.astype(np.float64))
        ti_ks.from_numpy(axial_stiffness.astype(np.float64))
        
        @ti.kernel
        def assemble_element_stiffness():
            for e in range(num_elements):
                k = ti_ks[e]
                d = ti_directions[e]
                
                # Direction cosines
                cx, cy, cz = d[0], d[1], d[2]
                
                # 3x3 outer product: d ⊗ d
                dd = ti.Matrix([[cx*cx, cx*cy, cx*cz],
                                [cy*cx, cy*cy, cy*cz],
                                [cz*cx, cz*cy, cz*cz]])
                
                # Local stiffness matrix (6x6)
                # [K] = k * [[dd, -dd], [-dd, dd]]
                for i in range(3):
                    for j in range(3):
                        ti_stiffness[e, i, j] = k * dd[i, j]
                        ti_stiffness[e, i, j+3] = -k * dd[i, j]
                        ti_stiffness[e, i+3, j] = -k * dd[i, j]
                        ti_stiffness[e, i+3, j+3] = k * dd[i, j]
        
        assemble_element_stiffness()
        
        # Convert to numpy for scipy assembly
        element_stiffness = ti_stiffness.to_numpy()
        
        # Assemble global stiffness (sparse)
        rows = []
        cols = []
        vals = []
        
        for e in range(num_elements):
            i, j = elements[e]
            dofs = [i*3, i*3+1, i*3+2, j*3, j*3+1, j*3+2]
            
            for local_i in range(6):
                for local_j in range(6):
                    if abs(element_stiffness[e, local_i, local_j]) > 1e-12:
                        rows.append(dofs[local_i])
                        cols.append(dofs[local_j])
                        vals.append(element_stiffness[e, local_i, local_j])
        
        K = sp.csr_matrix((vals, (rows, cols)), shape=(num_dofs, num_dofs))
        
        # Apply boundary conditions
        fixed_dofs = []
        for node in fixed_nodes:
            fixed_dofs.extend([node*3, node*3+1, node*3+2])
        
        free_dofs = [i for i in range(num_dofs) if i not in fixed_dofs]
        
        # Solve
        F = applied_forces.flatten()
        K_free = K[free_dofs, :][:, free_dofs]
        F_free = F[free_dofs]
        
        try:
            u_free = spla.spsolve(K_free, F_free)
        except:
            u_free = np.linalg.lstsq(K_free.toarray(), F_free, rcond=None)[0]
        
        # Full displacement vector
        u = np.zeros(num_dofs)
        u[free_dofs] = u_free
        displacements = u.reshape((num_nodes, 3))
        
        # Compute element forces
        element_forces = np.zeros(num_elements)
        for e in range(num_elements):
            i, j = elements[e]
            delta = displacements[j] - displacements[i]
            strain = np.dot(delta, directions[e]) / lengths[e]
            element_forces[e] = axial_stiffness[e] * lengths[e] * strain
        
        elapsed = time.time() - start_time
        
        return AcceleratedResult(
            displacements=displacements,
            forces=element_forces,
            energy=0.5 * np.sum(element_forces * np.zeros(num_elements)),
            time_seconds=elapsed,
        )
    
    def parallel_contact_detection(
        self,
        fiber_positions: np.ndarray,
        fiber_directions: np.ndarray,
        fiber_lengths: np.ndarray,
        radii: np.ndarray,
        box_size: Tuple[float, float, float],
        cell_size: float = None,
    ) -> List[Tuple[int, int, float]]:
        """
        Parallel contact detection using spatial hashing.
        
        Parameters
        ----------
        fiber_positions : np.ndarray
            Fiber center positions (N, 3)
        fiber_directions : np.ndarray
            Fiber direction vectors (N, 3)
        fiber_lengths : np.ndarray
            Fiber lengths (N,)
        radii : np.ndarray
            Fiber radii (N,)
        box_size : tuple
            (Lx, Ly, Lz)
        cell_size : float
            Grid cell size for spatial hashing
        
        Returns
        -------
        list of (int, int, float)
            Contact pairs with overlap distance
        """
        num_fibers = fiber_positions.shape[0]
        
        if cell_size is None:
            cell_size = 2 * np.max(radii) + np.max(fiber_lengths) * 0.1
        
        # Compute bounding boxes for each fiber
        half_lengths = 0.5 * fiber_lengths[:, None] * fiber_directions
        fiber_starts = fiber_positions - half_lengths
        fiber_ends = fiber_positions + half_lengths
        
        # Spatial hashing
        Lx, Ly, Lz = box_size
        nx = max(1, int(np.ceil(Lx / cell_size)))
        ny = max(1, int(np.ceil(Ly / cell_size)))
        nz = max(1, int(np.ceil(Lz / cell_size)))
        
        # Assign fibers to cells
        grid = {}
        for i in range(num_fibers):
            # Bounding box
            bbox_min = np.minimum(fiber_starts[i], fiber_ends[i]) - radii[i]
            bbox_max = np.maximum(fiber_starts[i], fiber_ends[i]) + radii[i]
            
            # Cell indices
            i_min = np.clip(np.floor(bbox_min / cell_size).astype(int), 0, [nx-1, ny-1, nz-1])
            i_max = np.clip(np.floor(bbox_max / cell_size).astype(int), 0, [nx-1, ny-1, nz-1])
            
            for cx in range(i_min[0], i_max[0] + 1):
                for cy in range(i_min[1], i_max[1] + 1):
                    for cz in range(i_min[2], i_max[2] + 1):
                        cell_key = (cx, cy, cz)
                        if cell_key not in grid:
                            grid[cell_key] = []
                        grid[cell_key].append(i)
        
        # Check for contacts within each cell
        contacts = []
        checked_pairs = set()
        
        for cell_fibers in grid.values():
            for idx_a in range(len(cell_fibers)):
                for idx_b in range(idx_a + 1, len(cell_fibers)):
                    i, j = cell_fibers[idx_a], cell_fibers[idx_b]
                    pair = (min(i, j), max(i, j))
                    
                    if pair in checked_pairs:
                        continue
                    checked_pairs.add(pair)
                    
                    # Compute minimum distance between two line segments
                    p1, d1, L1, r1 = fiber_positions[i], fiber_directions[i], fiber_lengths[i], radii[i]
                    p2, d2, L2, r2 = fiber_positions[j], fiber_directions[j], fiber_lengths[j], radii[j]
                    
                    # Line segment closest points
                    dist = self._segment_distance(p1, d1, L1, p2, d2, L2)
                    
                    # Check for overlap
                    overlap = (r1 + r2) - dist
                    if overlap > 0:
                        contacts.append((i, j, overlap))
        
        return contacts
    
    def _segment_distance(self, p1, d1, L1, p2, d2, L2):
        """Compute minimum distance between two line segments."""
        # Segment 1: p1 + t * d1 * L1, t in [-0.5, 0.5]
        # Segment 2: p2 + s * d2 * L2, s in [-0.5, 0.5]
        
        w0 = p1 - p2
        a = np.dot(d1, d1) * L1 * L1
        b = np.dot(d1, d2) * L1 * L2
        c = np.dot(d2, d2) * L2 * L2
        d = np.dot(d1, w0) * L1
        e = np.dot(d2, w0) * L2
        
        denom = a * c - b * b
        
        if abs(denom) < 1e-12:
            # Parallel segments
            t = 0.0
            s = e / c if abs(c) > 1e-12 else 0.0
        else:
            t = (b * e - c * d) / denom
            s = (a * e - b * d) / denom
        
        # Clamp to [-0.5, 0.5]
        t = np.clip(t, -0.5, 0.5)
        s = np.clip(s, -0.5, 0.5)
        
        # Closest points
        closest1 = p1 + t * d1 * L1
        closest2 = p2 + s * d2 * L2
        
        return np.linalg.norm(closest1 - closest2)
    
    def progressive_damage(
        self,
        node_positions: np.ndarray,
        elements: np.ndarray,
        youngs_modulus: float,
        radii: np.ndarray,
        fixed_nodes: List[int],
        strain_range: Tuple[float, float] = (0, 0.1),
        num_steps: int = 20,
        strength: np.ndarray = None,
        axis: int = 0,
    ) -> dict:
        """
        Progressive damage simulation with element failure.
        
        Parameters
        ----------
        node_positions : np.ndarray
            Node positions (N, 3)
        elements : np.ndarray
            Element connectivity (E, 2)
        youngs_modulus : float
            Young's modulus (Pa)
        radii : np.ndarray
            Element radii (E,)
        fixed_nodes : list of int
            Fixed node indices
        strain_range : tuple
            (min_strain, max_strain)
        num_steps : int
            Number of strain increments
        strength : np.ndarray
            Element strengths (Pa). If None, uses random distribution.
        axis : int
            Loading axis
        
        Returns
        -------
        dict
            Damage evolution data
        """
        import time
        from typing import Dict, Any
        
        start_time = time.time()
        
        num_nodes = node_positions.shape[0]
        num_elements = elements.shape[0]
        
        if strength is None:
            # Weibull distribution for fiber strength
            rng = np.random.default_rng(42)
            strength = 1e9 * (1 + 0.2 * rng.standard_normal(num_elements))
            strength = np.maximum(strength, 1e8)  # Minimum strength
        
        # Track damage
        active_elements = np.ones(num_elements, dtype=bool)
        strain_values = np.linspace(strain_range[0], strain_range[1], num_steps)
        
        damage_history = []
        stress_history = []
        broken_elements_history = []
        
        for step, strain in enumerate(strain_values):
            # Get active elements
            active_idx = np.where(active_elements)[0]
            if len(active_idx) == 0:
                break
            
            active_elems = elements[active_idx]
            active_radii = radii[active_idx]
            
            # Solve FEM
            applied_forces = np.zeros((num_nodes, 3))
            
            # Apply displacement on non-fixed nodes
            current_positions = node_positions.copy()
            bbox = np.max(node_positions, axis=0) - np.min(node_positions, axis=0)
            displacement = strain * bbox[axis]
            
            # Simple axial loading
            result = self.solve_beam_network(
                node_positions=current_positions,
                elements=active_elems,
                youngs_modulus=youngs_modulus,
                radii=active_radii,
                fixed_nodes=fixed_nodes,
                applied_forces=applied_forces,
            )
            
            # Compute element stresses
            areas = np.pi * active_radii**2
            element_stresses = np.zeros(len(active_idx))
            
            for e_local, e_global in enumerate(active_idx):
                i, j = elements[e_global]
                delta = result.displacements[j] - result.displacements[i]
                orig_length = np.linalg.norm(node_positions[j] - node_positions[i])
                if orig_length > 1e-12:
                    strain_e = np.linalg.norm(delta) / orig_length
                    element_stresses[e_local] = youngs_modulus * strain_e
            
            # Check for failure
            broken = 0
            for e_local, e_global in enumerate(active_idx):
                if abs(element_stresses[e_local]) > strength[e_global]:
                    active_elements[e_global] = False
                    broken += 1
            
            # Compute average stress
            if result.displacements is not None:
                bbox = np.max(node_positions, axis=0) - np.min(node_positions, axis=0)
                area = np.prod([bbox[i] for i in range(3) if i != axis])
                total_force = np.sum(np.abs(result.forces))
                avg_stress = total_force / area if area > 0 else 0
            else:
                avg_stress = 0
            
            damage_history.append(1.0 - np.sum(active_elements) / num_elements)
            stress_history.append(avg_stress)
            broken_elements_history.append(num_elements - np.sum(active_elements))
        
        elapsed = time.time() - start_time
        
        return {
            'strain': strain_values[:len(damage_history)],
            'stress': np.array(stress_history),
            'damage': np.array(damage_history),
            'broken_elements': np.array(broken_elements_history),
            'active_elements': active_elements,
            'time_seconds': elapsed,
        }
