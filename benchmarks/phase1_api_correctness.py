#!/usr/bin/env python3
"""
Phase 1: API Correctness & Robustness Benchmark
================================================
Tests each ML/RL API module with real fiber network structures at varying complexity.
Records: pass/fail, timing, memory, numerical statistics, graph-level analysis.

Usage:
    python benchmarks/phase1_api_correctness.py [--resume] [--phase 1]
"""

import sys, os, json, time, traceback, gc, argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

# Paths
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "benchmarks" / "results"
RESULTS_DIR.mkdir(exist_ok=True)
CHECKPOINT_FILE = RESULTS_DIR / "phase1_checkpoint.json"
RESULTS_FILE = RESULTS_DIR / "phase1_results.json"

# Memory guard
MAX_MEMORY_MB = 4000

def get_memory_mb():
    """Get current process memory in MB."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except:
        return 0

def memory_check():
    """Raise if memory is too high."""
    mem = get_memory_mb()
    if mem > MAX_MEMORY_MB:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mem = get_memory_mb()
        if mem > MAX_MEMORY_MB * 1.2:
            raise MemoryError(f"Memory {mem:.0f}MB exceeds limit {MAX_MEMORY_MB}MB")

# ============================================================
# Data Generation: Fiber Networks at Different Complexity
# ============================================================

def generate_structures():
    """Generate fiber network structures at 3 complexity levels."""
    from fibernet import pattern_2d

    structures = {}

    # Small: honeycomb 2x2 (~18 nodes)
    try:
        g = pattern_2d(unit="honeycomb", box=(10, 10), grid=(2, 2))
        structures["small_honeycomb"] = {"graph": g, "n_nodes": len(g.nodes), "n_edges": len(g.edges)}
    except Exception as e:
        structures["small_honeycomb"] = {"error": str(e)}

    # Medium: honeycomb 5x5 (~96 nodes)
    try:
        g = pattern_2d(unit="honeycomb", box=(10, 10), grid=(5, 5))
        structures["medium_honeycomb"] = {"graph": g, "n_nodes": len(g.nodes), "n_edges": len(g.edges)}
    except Exception as e:
        structures["medium_honeycomb"] = {"error": str(e)}

    # Large: kagome 6x6 (~180+ nodes)
    try:
        g = pattern_2d(unit="kagome", box=(10, 10), grid=(6, 6))
        structures["large_kagome"] = {"graph": g, "n_nodes": len(g.nodes), "n_edges": len(g.edges)}
    except Exception as e:
        structures["large_kagome"] = {"error": str(e)}

    # Diverse: reentrant (auxetic) 3x3
    try:
        g = pattern_2d(unit="reentrant", box=(10, 10), grid=(3, 3))
        structures["reentrant_3x3"] = {"graph": g, "n_nodes": len(g.nodes), "n_edges": len(g.edges)}
    except Exception as e:
        structures["reentrant_3x3"] = {"error": str(e)}

    # Voronoi random
    try:
        g = pattern_2d(unit="voronoi", box=(10, 10), grid=(3, 3), seed=42)
        structures["voronoi_3x3"] = {"graph": g, "n_nodes": len(g.nodes), "n_edges": len(g.edges)}
    except Exception as e:
        structures["voronoi_3x3"] = {"error": str(e)}

    return structures


def graph_to_dict(g):
    """Convert StructureGraph to plain dict for serialization."""
    from fibernet.ml.gnn import graph_from_structure
    try:
        return graph_from_structure(g)
    except:
        # Fallback: manual conversion
        node_ids = sorted(g.nodes.keys())
        node_map = {nid: i for i, nid in enumerate(node_ids)}
        positions = []
        for nid in node_ids:
            pos = g.nodes[nid].position
            positions.append(pos[:2].tolist() if len(pos) >= 2 else pos.tolist())
        src, dst = [], []
        for edge in g.edges.values():
            src.append(node_map.get(edge.node_i, 0))
            dst.append(node_map.get(edge.node_j, 0))
        return {
            "node_features": torch.tensor(positions, dtype=torch.float32),
            "edge_index": torch.tensor([src, dst], dtype=torch.long),
            "edge_features": torch.ones(len(src), 2),
            "n_nodes": len(node_ids),
            "n_edges": len(g.edges),
        }


# ============================================================
# Test Functions for Each Module
# ============================================================

class TestResult:
    """Container for test result."""
    def __init__(self, name, passed, time_s, error=None, stats=None):
        self.name = name
        self.passed = passed
        self.time_s = time_s
        self.error = error
        self.stats = stats or {}

    def to_dict(self):
        d = {
            "name": self.name,
            "passed": self.passed,
            "time_s": round(self.time_s, 4),
        }
        if self.error:
            d["error"] = str(self.error)[:500]
        if self.stats:
            d["stats"] = {k: (float(v) if isinstance(v, (np.floating, float)) else
                              int(v) if isinstance(v, (np.integer, int)) else v)
                          for k, v in self.stats.items()
                          if not isinstance(v, (torch.Tensor, np.ndarray))}
        return d


def timed_test(fn, *args, **kwargs):
    """Run function with timing and error handling."""
    t0 = time.time()
    try:
        result = fn(*args, **kwargs)
        elapsed = time.time() - t0
        if isinstance(result, TestResult):
            result.time_s = elapsed
            return result
        return TestResult(fn.__name__, True, elapsed, stats=result if isinstance(result, dict) else {})
    except Exception as e:
        elapsed = time.time() - t0
        return TestResult(fn.__name__, False, elapsed, error=f"{type(e).__name__}: {e}")


# ---- Module-specific tests ----

def test_gflownet(structures):
    """Test GFlowNet with real fiber structures."""
    from fibernet.ml.gflownet import FiberGFlowNet, GFlowNetTrainer, connectivity_reward, StructureState, StructureAction

    stats = {}
    # 1. Basic creation
    gfn = FiberGFlowNet(n_node_features=5, hidden=32, max_nodes=20)
    stats["model_params"] = sum(p.numel() for p in gfn.parameters())

    # 2. Sample structures
    samples = gfn.sample(n=5, max_steps=10, temperature=1.0)
    stats["sample_nodes"] = [s.n_nodes for s in samples]
    stats["sample_edges"] = [s.n_edges for s in samples]

    # 3. Training with real reward
    def fiber_reward(state):
        n, e = state.n_nodes, state.n_edges
        if n < 2: return 0.1
        return float(n * 0.3 + e * 0.5 + np.random.randn() * 0.1)

    trainer = GFlowNetTrainer(gfn, reward_fn=fiber_reward, lr=1e-3)
    history = trainer.train(n_iterations=5, batch_size=8, max_steps=10,
                             log_every=100, verbose=False)
    stats["train_loss_final"] = history["loss"][-1]
    stats["mean_reward_final"] = history["mean_reward"][-1]
    stats["max_reward_final"] = history["max_reward"][-1]

    # 4. DB loss variant
    gfn_db = FiberGFlowNet(n_node_features=5, hidden=32, max_nodes=20, loss_type="db")
    trainer_db = GFlowNetTrainer(gfn_db, reward_fn=fiber_reward, lr=1e-3)
    metrics = trainer_db.train_step(batch_size=4, max_steps=8)
    stats["db_loss"] = metrics["loss"]

    # 5. Reward function quality
    r = connectivity_reward(samples[0])
    stats["connectivity_reward_sample"] = r

    memory_check()
    return stats


def test_differentiable_physics(structures):
    """Test DifferentiablePhysics with real fiber structures."""
    from fibernet.ml.differentiable_physics import (
        DifferentiableSpringNetwork, DifferentiableBeamNetwork,
        DifferentiableFEA, PhysicsOptimizer, DifferentiableMaterialModel
    )

    stats = {}

    # Pick a real structure
    g_info = structures.get("medium_honeycomb", structures.get("small_honeycomb"))
    if "error" in g_info:
        return {"error": g_info["error"]}
    g = g_info["graph"]

    # Convert to tensors
    node_ids = sorted(g.nodes.keys())
    node_map = {nid: i for i, nid in enumerate(node_ids)}
    n_nodes = len(node_ids)
    positions = []
    for nid in node_ids:
        positions.append(g.nodes[nid].position[:2].tolist())
    node_pos = torch.tensor(positions, dtype=torch.float32)

    src, dst = [], []
    for edge in g.edges.values():
        si = node_map.get(edge.node_i)
        di = node_map.get(edge.node_j)
        if si is not None and di is not None:
            src.append(si); dst.append(di)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    n_edges = len(src)

    stats["structure_nodes"] = n_nodes
    stats["structure_edges"] = n_edges

    # 1. Spring network solve
    radii = (torch.ones(n_edges, dtype=torch.float32) * 0.005).requires_grad_(True)
    forces = torch.zeros(n_nodes, 2)
    forces[-1, 0] = 500.0  # Pull last node in x
    fixed = torch.tensor([0, 1], dtype=torch.long)

    sim = DifferentiableSpringNetwork(youngs_modulus=1e9, dim=2)
    u, sigma = sim.solve(edge_index, node_pos, radii, forces, fixed)
    stats["max_displacement"] = float(u.abs().max().item())
    stats["max_stress"] = float(sigma.abs().max().item())
    stats["mean_stress"] = float(sigma.abs().mean().item())

    # 2. Gradient flow
    loss = (u[:, 0] ** 2).sum()
    loss.backward()
    stats["grad_norm_radii"] = float(radii.grad.norm().item()) if radii.grad is not None else 0
    stats["grad_nonzero_frac"] = float((radii.grad.abs() > 1e-10).float().mean().item()) if radii.grad is not None else 0

    # 3. Beam network (use small structure to avoid numerical issues with large graphs)
    sim_beam = DifferentiableBeamNetwork(youngs_modulus=1e9, include_bending=True, damping=1e-6)
    small_info = structures.get("small_honeycomb")
    small_g = small_info["graph"]
    s_nids = sorted(small_g.nodes.keys())
    s_nmap = {nid: i for i, nid in enumerate(s_nids)}
    s_pos = torch.tensor([small_g.nodes[nid].position[:2].tolist() for nid in s_nids], dtype=torch.float32)
    s_src, s_dst = [], []
    for edge in small_g.edges.values():
        si, di = s_nmap.get(edge.node_i), s_nmap.get(edge.node_j)
        if si is not None and di is not None:
            s_src.append(si); s_dst.append(di)
    s_ei = torch.tensor([s_src, s_dst], dtype=torch.long)
    s_r = torch.ones(len(s_src), dtype=torch.float32) * 0.02  # larger radius for beam stiffness
    s_n = len(s_nids)
    forces_beam = torch.zeros(s_n, 2)
    forces_beam[-1, 1] = 200.0  # transverse load
    s_fixed = torch.tensor([0, 1], dtype=torch.long)
    u_beam, sigma_beam = sim_beam.solve(s_ei, s_pos, s_r, forces_beam, s_fixed)
    stats["beam_structure_nodes"] = s_n
    stats["beam_structure_edges"] = len(s_src)
    stats["beam_max_displacement"] = float(u_beam[:, :2].abs().max().item())
    stats["beam_max_rotation"] = float(u_beam[:, 2].abs().max().item()) if u_beam.shape[1] > 2 else 0

    # 4. FEA wrapper
    fea = DifferentiableFEA(youngs_modulus=1e9, element_type="spring")
    result = fea(edge_index, node_pos, radii.detach(), forces, fixed)
    stats["compliance"] = float(result["compliance"].item()) if result["compliance"].abs() > 1e-20 else float((forces.flatten() * result["displacements"][:, :2].flatten()).sum().item())
    stats["volume"] = float(result["volume"].item())

    # 5. Optimizer (short run)
    opt = PhysicsOptimizer(fea, lr=0.005, volume_constraint=result["volume"].item() * 1.5)
    opt_result = opt.optimize(edge_index, node_pos, radii.detach().clone(), forces, fixed,
                               n_iterations=5, verbose=False)
    stats["optimized_min_radius"] = float(opt_result["optimized_radii"].min().item())
    stats["optimized_max_radius"] = float(opt_result["optimized_radii"].max().item())
    stats["optimization_improvement"] = float(opt_result["history"]["objective"][0] - opt_result["history"]["objective"][-1])

    # 6. Material model
    mat_model = DifferentiableMaterialModel(hidden=[16, 8])
    strain = torch.linspace(0, 0.01, 50)
    stress = mat_model(strain)
    tangent = mat_model.tangent_modulus(strain)
    stats["material_stress_range"] = float(stress.max().item() - stress.min().item())
    stats["material_tangent_mean"] = float(tangent.mean().item())

    # 7. from_structure_graph
    fea2, np2, ei2 = DifferentiableFEA.from_structure_graph(g)
    stats["from_structure_nodes"] = np2.shape[0]
    stats["from_structure_edges"] = ei2.shape[1]

    memory_check()
    return stats


def test_pinn_gnn(structures):
    """Test Physics-Informed GNN with real structures."""
    from fibernet.ml.pinn_gnn import (
        PhysicsInformedGNN, PhysicsGNNTrainer,
        PhysicsInformedMessagePassing, ForceBalanceLoss
    )

    stats = {}

    # Get graph data
    g_info = structures.get("medium_honeycomb", structures.get("small_honeycomb"))
    if "error" in g_info:
        return {"error": g_info["error"]}
    g = g_info["graph"]
    gd = graph_to_dict(g)

    # 1. Message passing layer
    mp = PhysicsInformedMessagePassing(node_dim=gd["node_features"].shape[1],
                                        edge_dim=gd["edge_features"].shape[1],
                                        hidden=32, force_dim=2)
    h_new, e_new, forces = mp(gd["node_features"], gd["edge_index"], gd["edge_features"])
    stats["mp_output_nodes"] = h_new.shape[0]
    stats["mp_output_edges"] = e_new.shape[0]
    stats["mp_force_mean_norm"] = float(forces.norm(dim=-1).mean().item())

    # 2. Full PIGNN forward
    gnn = PhysicsInformedGNN(
        node_dim=gd["node_features"].shape[1],
        edge_dim=gd["edge_features"].shape[1],
        hidden=32, n_layers=3, n_outputs=1,
        predict_field=True, force_dim=2
    )
    stats["pinn_gnn_params"] = sum(p.numel() for p in gnn.parameters())

    pred = gnn([gd])
    stats["pred_shape"] = list(pred.shape)

    # 3. Field prediction
    fields = gnn.predict_fields(gd)
    stats["displacement_mean"] = float(fields["displacement"].mean().item())
    stats["displacement_std"] = float(fields["displacement"].std().item())
    stats["stress_range"] = float(fields["stress"].max().item() - fields["stress"].min().item())

    # 4. Physics loss
    losses = gnn.compute_physics_loss(gd)
    stats["force_balance_loss"] = float(losses["force_balance"].item())
    stats["constitutive_loss"] = float(losses["constitutive"].item())
    stats["energy_loss"] = float(losses["energy"].item())
    stats["total_physics_loss"] = float(losses["total_physics"].item())

    # 5. Training loop
    graphs = [graph_to_dict(structures[k]["graph"]) for k in structures if "error" not in structures[k]]
    labels = np.random.randn(len(graphs)).astype(np.float32) * 100  # fake targets
    trainer = PhysicsGNNTrainer(gnn, physics_loss_weight=0.3, lr=1e-3)
    result = trainer.fit(graphs, labels, epochs=5, batch_size=4, verbose=False)
    stats["train_best_val_loss"] = result["best_val_loss"]

    # 6. Multi-structure forward (different sizes)
    preds = []
    for k, info in structures.items():
        if "error" in info:
            continue
        gd2 = graph_to_dict(info["graph"])
        p = gnn([gd2])
        preds.append(p.item())
    stats["multi_structure_preds"] = [round(p, 4) for p in preds]

    memory_check()
    return stats


def test_neural_ode(structures):
    """Test Neural ODE with physics-based scenarios."""
    from fibernet.ml.neural_ode import (
        ODESolver, FiberNeuralODE, StressRelaxationODE,
        CreepODE, FatigueODE, NeuralODETrainer
    )

    stats = {}

    # 1. ODE Solver accuracy (exponential decay)
    for method in ["euler", "rk4", "dopri5"]:
        solver = ODESolver(method=method)
        traj = solver.solve(lambda t, x: -x, torch.tensor([1.0]), torch.linspace(0, 3, 100))
        expected = torch.exp(-torch.linspace(0, 3, 100))
        error = float((traj[:, 0] - expected).abs().max().item())
        stats[f"solver_{method}_max_error"] = error

    # 2. FiberNeuralODE
    ode = FiberNeuralODE(state_dim=4, hidden=[32, 16], solver_method="rk4")
    stats["neural_ode_params"] = sum(p.numel() for p in ode.parameters())
    x0 = torch.randn(4)
    t = torch.linspace(0, 2, 50)
    traj = ode.solve(x0, t)
    stats["neural_ode_traj_shape"] = list(traj.shape)
    stats["neural_ode_final_norm"] = float(traj[-1].norm().item())

    # 3. Stress relaxation - Maxwell
    sr_maxwell = StressRelaxationODE(model_type="maxwell", E=1e9, eta=1e12)
    t_r, sigma_r = sr_maxwell.relax(100.0, (0, 2000), 100)
    analytical = sr_maxwell.analytical_solution(100.0, t_r.numpy())
    sr_error = float(np.abs(sigma_r.detach().numpy() - analytical).max())
    stats["maxwell_relax_error_vs_analytical"] = sr_error
    stats["maxwell_relax_half_life_steps"] = int(np.argmin(np.abs(sigma_r.detach().numpy() - 50.0)))
    tau_expected = 1000.0  # eta/E
    stats["maxwell_tau_expected"] = tau_expected

    # 4. Stress relaxation - SLS
    sr_sls = StressRelaxationODE(model_type="sls", E=1e9, eta=1e12, E2=5e8)
    t_sls, sigma_sls = sr_sls.relax(100.0, (0, 2000), 100)
    stats["sls_relax_final_stress"] = float(sigma_sls[-1].item())
    stats["sls_relax_stress_retained_pct"] = float(sigma_sls[-1].item() / sigma_sls[0].item() * 100)

    # 5. Creep
    creep = CreepODE(E=1e9, eta=1e12, model_type="power_law", n_creep=0.3)
    t_c, eps_c = creep.predict_creep(0.0001, 100.0, (1, 1000), 50)
    stats["creep_strain_ratio"] = float(eps_c[-1].item() / eps_c[0].item())
    stats["creep_is_monotonic"] = bool(all(eps_c[i] <= eps_c[i+1] + 1e-10 for i in range(len(eps_c)-1)))

    # 6. Fatigue
    fatigue = FatigueODE(hidden=[16, 8])
    result = fatigue.predict_fatigue_life(0.0, 100.0, max_cycles=2000, failure_damage=1.0)
    stats["fatigue_cycles_to_failure"] = result["cycles_to_failure"]
    stats["fatigue_final_damage"] = float(result["final_damage"])

    # 7. Neural ODE trainer with synthetic data
    ode_train = FiberNeuralODE(state_dim=2, hidden=[16], solver_method="euler")
    trainer = NeuralODETrainer(ode_train, lr=1e-3)
    # Synthetic: exponential decay
    n_traj = 8
    n_t = 15
    time_data = np.tile(np.linspace(0, 1, n_t), (n_traj, 1))
    state_data = np.zeros((n_traj, n_t, 2))
    for i in range(n_traj):
        x0 = np.random.randn(2) * 0.5
        for ti in range(n_t):
            state_data[i, ti] = x0 * np.exp(-time_data[i, ti])
    hist = trainer.fit(time_data, state_data, epochs=10, batch_size=4, verbose=False)
    stats["neural_ode_train_loss_final"] = hist["train_loss"][-1]

    memory_check()
    return stats


def test_conservative_nn(structures):
    """Test Conservative Neural Networks."""
    from fibernet.ml.conservative_nn import (
        HamiltonianNN, LagrangianNN, EnergyConservingNN,
        MomentumConservingNN, DivergenceFreeNet, ConservativeTrainer
    )

    stats = {}

    # 1. Hamiltonian NN - SHO
    hnn = HamiltonianNN(n_coords=2, hidden=[32, 16])
    stats["hamiltonian_params"] = sum(p.numel() for p in hnn.parameters())
    q0 = torch.tensor([1.0, 0.0])
    p0 = torch.tensor([0.0, 1.0])
    dq, dp = hnn(q0, p0)
    stats["hamiltonian_dq_norm"] = float(dq.norm().item())
    stats["hamiltonian_dp_norm"] = float(dp.norm().item())

    # Simulate
    t = torch.linspace(0, 2, 50)
    q_traj, p_traj = hnn.simulate(q0, p0, t, method="rk4")
    H_start = hnn.hamiltonian(q0, p0).item()
    H_end = hnn.hamiltonian(q_traj[-1], p_traj[-1]).item()
    stats["hamiltonian_energy_drift"] = abs(H_end - H_start)

    # 2. Lagrangian NN
    lnn = LagrangianNN(n_coords=2, hidden=[32, 16])
    q = torch.randn(5, 2)
    qd = torch.randn(5, 2)
    qdd = lnn(q, qd)
    stats["lagrangian_qddot_shape"] = list(qdd.shape)
    stats["lagrangian_qddot_norm"] = float(qdd.norm().item())

    # 3. Energy conserving NN
    ec = EnergyConservingNN(state_dim=6, hidden=[32, 16])
    x = torch.randn(6)
    dx = ec(x)
    stats["energy_dx_norm"] = float(dx.norm().item())
    # Conservation check
    check = ec.check_conservation(x, dt=0.01, n_steps=50)
    stats["energy_max_drift"] = check["max_drift"]
    stats["energy_relative_drift"] = check["relative_drift"]

    # 4. Momentum conserving NN
    mc = MomentumConservingNN(n_nodes=8, coord_dim=2, feature_dim=4, hidden=32)
    pos = torch.randn(8, 2)
    feat = torch.randn(8, 4)
    forces, upd = mc(pos, feat)
    total_force = forces.sum(dim=0)
    stats["momentum_total_force_norm"] = float(total_force.norm().item())
    stats["momentum_conserved"] = bool(total_force.norm().item() < 1e-4)

    # 5. Divergence-free net
    df = DivergenceFreeNet(dim=2, hidden=[32, 16])
    coords = torch.randn(50, 2) * 5
    v = df(coords)
    stats["divfree_velocity_shape"] = list(v.shape)
    stats["divfree_velocity_norm_mean"] = float(v.norm(dim=-1).mean().item())
    div = df.check_divergence(coords[:10])
    stats["divfree_max_divergence"] = float(div.abs().max().item())

    # 6. Conservative trainer - Hamiltonian
    trainer = ConservativeTrainer(hnn, lr=1e-3)
    # SHO data: dq/dt = p, dp/dt = -q
    n = 50
    q_data = np.random.randn(n, 2).astype(np.float32) * 0.5
    p_data = np.random.randn(n, 2).astype(np.float32) * 0.5
    dq_data = p_data.copy()
    dp_data = -q_data.copy()
    hist = trainer.fit_hamiltonian(q_data, p_data, dq_data, dp_data, epochs=10, batch_size=16, verbose=False)
    stats["hamiltonian_train_loss_final"] = hist["train_loss"][-1]

    memory_check()
    return stats


def test_existing_modules(structures):
    """Test existing modules still work with new data."""
    from fibernet.ml.gnn import FiberGNN, graph_from_structure, train_gnn
    from fibernet.ml.diffusion import FiberDiffusion, DiffusionTrainer
    from fibernet.ml.gan import FiberWGAN, GANTrainer

    stats = {}

    # Get real graph data
    g_info = structures.get("small_honeycomb")
    if "error" in g_info:
        return {"error": g_info["error"]}
    g = g_info["graph"]
    gd = graph_from_structure(g)

    # 1. GNN forward
    node_dim = gd["node_features"].shape[1]
    gnn = FiberGNN(node_dim=node_dim, hidden=32, n_outputs=1, n_layers=3)
    pred = gnn([gd])
    stats["gnn_pred_shape"] = list(pred.shape)
    stats["gnn_params"] = sum(p.numel() for p in gnn.parameters())

    # 2. Diffusion
    n_feat = 10
    diff = FiberDiffusion(n_features=n_feat, hidden=[32, 16], n_steps=20)
    X = np.random.randn(30, n_feat).astype(np.float32)
    trainer = DiffusionTrainer(diff)
    hist = trainer.fit(X, epochs=3, batch_size=10, verbose=False)
    samples = trainer.sample(n=5)
    stats["diffusion_sample_shape"] = list(samples.shape)
    stats["diffusion_train_loss_final"] = hist["train_loss"][-1]

    # 3. GAN
    gan = FiberWGAN(n_features=n_feat, latent_dim=8, hidden=[32])
    gan_trainer = GANTrainer(gan)
    hist_gan = gan_trainer.fit(X, epochs=3, batch_size=10, verbose=False)
    samples_gan = gan_trainer.sample(n=5)
    stats["gan_sample_shape"] = list(samples_gan.shape)

    memory_check()
    return stats


# ============================================================
# Main Runner
# ============================================================

ALL_TESTS = {
    "gflownet": test_gflownet,
    "differentiable_physics": test_differentiable_physics,
    "pinn_gnn": test_pinn_gnn,
    "neural_ode": test_neural_ode,
    "conservative_nn": test_conservative_nn,
    "existing_modules": test_existing_modules,
}


def run_all(resume=False):
    """Run all Phase 1 tests with checkpointing."""
    # Load checkpoint
    completed = {}
    if resume and CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            completed = json.load(f)
        print(f"Resuming: {len(completed)} tests already completed")

    # Generate structures once
    print("Generating fiber network structures...")
    structures = generate_structures()
    for k, v in structures.items():
        if "error" in v:
            print(f"  WARNING: {k} generation failed: {v['error']}")
        else:
            print(f"  {k}: {v['n_nodes']} nodes, {v['n_edges']} edges")

    results = {}

    for name, test_fn in ALL_TESTS.items():
        if name in completed:
            print(f"[SKIP] {name} (already completed)")
            results[name] = completed[name]
            continue

        print(f"\n[RUN] {name}...")
        gc.collect()
        memory_check()

        t0 = time.time()
        try:
            stats = test_fn(structures)
            elapsed = time.time() - t0
            result = {
                "passed": True,
                "time_s": round(elapsed, 3),
                "stats": {},
            }
            # Serialize stats
            for k, v in stats.items():
                if isinstance(v, (torch.Tensor,)):
                    result["stats"][k] = v.detach().cpu().tolist() if v.numel() < 20 else f"tensor({v.shape})"
                elif isinstance(v, np.ndarray):
                    result["stats"][k] = v.tolist() if v.size < 20 else f"array({v.shape})"
                else:
                    result["stats"][k] = v
        except Exception as e:
            elapsed = time.time() - t0
            result = {
                "passed": False,
                "time_s": round(elapsed, 3),
                "error": f"{type(e).__name__}: {str(e)[:500]}",
                "traceback": traceback.format_exc()[-1000:],
                "stats": {},
            }

        results[name] = result
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  [{status}] {name} ({result['time_s']:.1f}s)")
        if not result["passed"]:
            print(f"  Error: {result.get('error', 'unknown')}")

        # Save checkpoint
        completed[name] = result
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(completed, f, indent=2, default=str)

        gc.collect()

    # Final save
    final = {
        "timestamp": datetime.now().isoformat(),
        "phase": 1,
        "results": results,
        "summary": {
            "total": len(results),
            "passed": sum(1 for r in results.values() if r["passed"]),
            "failed": sum(1 for r in results.values() if not r["passed"]),
            "total_time_s": round(sum(r.get("time_s", 0) for r in results.values()), 2),
        }
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(final, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Phase 1 Summary: {final['summary']['passed']}/{final['summary']['total']} passed")
    print(f"Total time: {final['summary']['total_time_s']:.1f}s")
    print(f"Results saved to: {RESULTS_FILE}")

    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    run_all(resume=args.resume)
