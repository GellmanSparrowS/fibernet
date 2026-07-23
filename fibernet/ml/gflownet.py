"""
GFlowNet (Generative Flow Networks) for FiberNet Structure Generation.

GFlowNets learn to sample discrete compositional objects (fiber network
topologies) proportionally to a reward function. This enables generation of
diverse, high-quality fiber structures that target specific properties.

Implements:
- FiberGFlowNet: Main model with policy + flow networks
- GFlowNetTrainer: Training with Trajectory Balance / Detailed Balance loss
- StructureState: State representation for fiber network construction
- GraphActions: Discrete action space (add node, add edge, stop)

Features
--------
- Trajectory Balance (TB) and Detailed Balance (DB) training
- Substructure-based construction (add node, add edge, skip)
- Reward shaping for targeted property generation
- Exploration with epsilon-greedy and temperature sampling
- Compatible with StructureGraph output format
- Batched trajectory collection for efficient training

References
----------
- Bengio et al., "GFlowNet Foundations" (JMLR 2023)
- Malkin et al., "GFlowNets for Biological Sequence Design" (2022)
- Article section 2: "GFlowNets establish bijective mappings between
  latent variables and network configurations"

Examples
--------
>>> from fibernet.ml.gflownet import FiberGFlowNet, GFlowNetTrainer
>>> gfn = FiberGFlowNet(n_node_features=5, n_edge_features=2, hidden=64)
>>> trainer = GFlowNetTrainer(gfn, reward_fn=my_reward_fn)
>>> trainer.train(n_iterations=1000, batch_size=16)
>>> structures = gfn.sample(n=10, max_steps=20)
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from dataclasses import dataclass, field
from copy import deepcopy

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _require_torch():
    if not HAS_TORCH:
        raise ImportError("PyTorch required: pip install torch")


# ======================================================================
# State & Action Definitions
# ======================================================================

@dataclass
class StructureAction:
    """Represents a discrete action in the structure-building MDP.

    Action types:
    - 'add_node': Add a new node with given features
    - 'add_edge': Connect two existing nodes
    - 'set_node_pos': Set position of an existing node
    - 'stop': Terminate construction
    """
    action_type: int  # 0=add_node, 1=add_edge, 2=set_pos, 3=stop
    target_idx: int = 0
    source_idx: int = 0
    features: Optional[np.ndarray] = None

    def to_tensor(self, max_nodes: int = 50, n_node_features: int = 5) -> torch.Tensor:
        """Encode action as fixed-size tensor."""
        _require_torch()
        parts = []
        # One-hot action type (4 dims)
        at = torch.zeros(4)
        at[self.action_type] = 1.0
        parts.append(at)
        # Indices normalized
        parts.append(torch.tensor([self.target_idx / max(max_nodes, 1),
                                    self.source_idx / max(max_nodes, 1)]))
        # Features (padded)
        if self.features is not None:
            f = torch.zeros(n_node_features)
            f[:len(self.features)] = torch.tensor(self.features, dtype=torch.float32)
            parts.append(f)
        else:
            parts.append(torch.zeros(n_node_features))
        return torch.cat(parts)


@dataclass
class StructureState:
    """Tracks the current state of a fiber network being constructed.

    Attributes
    ----------
    node_positions : list of np.ndarray
        Positions of current nodes.
    node_features : list of np.ndarray
        Features per node.
    edges : list of tuple
        List of (i, j) edge pairs.
    max_nodes : int
        Maximum allowed nodes.
    """
    node_positions: List[np.ndarray] = field(default_factory=list)
    node_features: List[np.ndarray] = field(default_factory=list)
    edges: List[Tuple[int, int]] = field(default_factory=list)
    max_nodes: int = 50
    box_size: float = 10.0

    @property
    def n_nodes(self) -> int:
        return len(self.node_positions)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def is_terminal(self) -> bool:
        return self._terminal

    def __post_init__(self):
        self._terminal = False

    def copy(self) -> 'StructureState':
        s = StructureState(
            node_positions=[p.copy() for p in self.node_positions],
            node_features=[f.copy() for f in self.node_features],
            edges=list(self.edges),
            max_nodes=self.max_nodes,
            box_size=self.box_size,
        )
        s._terminal = self._terminal
        return s

    def apply_action(self, action: StructureAction):
        """Apply an action to modify the state in-place."""
        if action.action_type == 0:  # add_node
            if self.n_nodes < self.max_nodes:
                if action.features is not None:
                    pos = action.features[:2] if len(action.features) >= 2 else np.random.uniform(0, self.box_size, 2)
                    feat = action.features
                else:
                    pos = np.random.uniform(0, self.box_size, 2)
                    feat = np.zeros(5)
                    feat[:2] = pos
                self.node_positions.append(pos.astype(np.float64))
                self.node_features.append(feat.astype(np.float32))
        elif action.action_type == 1:  # add_edge
            i, j = action.source_idx, action.target_idx
            if (0 <= i < self.n_nodes and 0 <= j < self.n_nodes
                    and i != j and (i, j) not in self.edges and (j, i) not in self.edges):
                self.edges.append((i, j))
        elif action.action_type == 2:  # set_pos
            if 0 <= action.target_idx < self.n_nodes and action.features is not None:
                self.node_positions[action.target_idx] = action.features[:2].astype(np.float64)
        elif action.action_type == 3:  # stop
            self._terminal = True

    def to_observation(self, n_node_features: int = 5, max_nodes: int = 50) -> np.ndarray:
        """Encode state as a fixed-size observation vector.

        Returns a flat vector encoding:
        - Node count, edge count (2)
        - Padded node positions (max_nodes * 2)
        - Padded adjacency (max_nodes * max_nodes, upper triangle)
        """
        parts = []
        # Global features
        parts.append([self.n_nodes / max_nodes, self.n_edges / (max_nodes * 2)])

        # Padded node positions
        pos_padded = np.zeros((max_nodes, 2))
        for i, pos in enumerate(self.node_positions):
            pos_padded[i] = pos[:2] / self.box_size
        parts.append(pos_padded.flatten())

        # Upper-triangle adjacency
        adj_size = max_nodes * (max_nodes - 1) // 2
        adj = np.zeros(adj_size)
        for (i, j) in self.edges:
            if i < j:
                idx = i * max_nodes - i * (i + 1) // 2 + (j - i - 1)
                if 0 <= idx < adj_size:
                    adj[idx] = 1.0
        parts.append(adj)

        return np.concatenate(parts).astype(np.float32)


if HAS_TORCH:

    # ==================================================================
    # Policy Network
    # ==================================================================

    class PolicyNetwork(nn.Module):
        """Policy network that outputs action logits given a state.

        Parameters
        ----------
        obs_dim : int
            Dimension of observation vector.
        hidden : int
            Hidden layer size.
        n_actions : int
            Total number of discrete action heads.
        n_node_features : int
            Number of node features.
        """

        def __init__(self, obs_dim: int, hidden: int = 128,
                     n_actions: int = 4, n_node_features: int = 5):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
            )
            # Action type head
            self.action_type_head = nn.Linear(hidden, n_actions)
            # Target index head
            self.target_head = nn.Linear(hidden, 50)  # max nodes
            # Source index head
            self.source_head = nn.Linear(hidden, 50)
            # Feature head (for add_node)
            self.feature_head = nn.Linear(hidden, n_node_features)

        def forward(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
            h = self.net(obs)
            return {
                "action_type": self.action_type_head(h),
                "target": self.target_head(h),
                "source": self.source_head(h),
                "features": self.feature_head(h),
            }

        def sample_action(self, obs: torch.Tensor, state: StructureState,
                          temperature: float = 1.0) -> Tuple[StructureAction, torch.Tensor]:
            """Sample an action from the policy, returning action and log-prob."""
            logits = self.forward(obs.unsqueeze(0) if obs.dim() == 1 else obs)

            # Action type
            at_logits = logits["action_type"] / temperature
            # Mask: can't add node if at max, can't add edge if < 2 nodes
            mask = torch.zeros_like(at_logits)
            if state.n_nodes >= state.max_nodes:
                mask[..., 0] = -1e9  # can't add node
            if state.n_nodes < 2:
                mask[..., 1] = -1e9  # can't add edge
            at_logits = at_logits + mask

            at_dist = torch.distributions.Categorical(logits=at_logits)
            action_type = at_dist.sample()
            log_prob = at_dist.log_prob(action_type)

            # Target
            t_logits = logits["target"] / temperature
            t_mask = torch.full_like(t_logits, -1e9)
            for i in range(min(state.n_nodes, 50)):
                t_mask[..., i] = 0.0
            t_logits = t_logits + t_mask
            t_dist = torch.distributions.Categorical(logits=t_logits)
            target = t_dist.sample()
            log_prob = log_prob + t_dist.log_prob(target)

            # Source
            s_logits = logits["source"] / temperature
            s_mask = torch.full_like(s_logits, -1e9)
            for i in range(min(state.n_nodes, 50)):
                s_mask[..., i] = 0.0
            s_logits = s_logits + s_mask
            s_dist = torch.distributions.Categorical(logits=s_logits)
            source = s_dist.sample()
            log_prob = log_prob + s_dist.log_prob(source)

            # Features (continuous, use tanh + scale)
            feat = torch.tanh(logits["features"]) * state.box_size

            action = StructureAction(
                action_type=action_type.item(),
                target_idx=target.item(),
                source_idx=source.item(),
                features=feat.squeeze(0).detach().cpu().numpy() if feat.dim() > 1 else feat.detach().cpu().numpy(),
            )
            return action, log_prob.squeeze()

    # ==================================================================
    # Flow Network (for DB loss)
    # ==================================================================

    class FlowNetwork(nn.Module):
        """Estimates log-flow at each state for Detailed Balance training."""

        def __init__(self, obs_dim: int, hidden: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            return self.net(obs).squeeze(-1)

    # ==================================================================
    # Main GFlowNet Model
    # ==================================================================

    class FiberGFlowNet(nn.Module):
        """GFlowNet for fiber network structure generation.

        Builds fiber network topologies step-by-step by adding nodes and edges,
        learning to sample structures proportionally to a reward function.

        Parameters
        ----------
        n_node_features : int
            Number of features per node.
        n_edge_features : int
            Number of features per edge.
        hidden : int
            Hidden layer size.
        max_nodes : int
            Maximum number of nodes in generated structures.
        box_size : float
            Size of the generation domain.
        loss_type : str
            "tb" (Trajectory Balance) or "db" (Detailed Balance).

        Examples
        --------
        >>> gfn = FiberGFlowNet(n_node_features=5, hidden=64)
        >>> trainer = GFlowNetTrainer(gfn, reward_fn=my_reward)
        >>> trainer.train(n_iterations=500, batch_size=16)
        >>> structures = gfn.sample(n=10, max_steps=15)
        """

        def __init__(
            self,
            n_node_features: int = 5,
            n_edge_features: int = 2,
            hidden: int = 128,
            max_nodes: int = 30,
            box_size: float = 10.0,
            loss_type: str = "tb",
        ):
            super().__init__()
            self.n_node_features = n_node_features
            self.n_edge_features = n_edge_features
            self.max_nodes = max_nodes
            self.box_size = box_size
            self.loss_type = loss_type

            # Observation dimension
            self.obs_dim = 2 + max_nodes * 2 + max_nodes * (max_nodes - 1) // 2

            self.policy = PolicyNetwork(
                self.obs_dim, hidden, n_actions=4, n_node_features=n_node_features
            )

            if loss_type == "db":
                self.flow_net = FlowNetwork(self.obs_dim, hidden)
            else:
                self.flow_net = None
                # Z (log-partition) parameter for TB loss
                self.log_Z = nn.Parameter(torch.zeros(1))

        def get_observation(self, state: StructureState) -> torch.Tensor:
            """Convert state to tensor observation."""
            obs = state.to_observation(self.n_node_features, self.max_nodes)
            return torch.tensor(obs, dtype=torch.float32)

        def sample_action(self, state: StructureState,
                          temperature: float = 1.0) -> Tuple[StructureAction, torch.Tensor]:
            """Sample an action from the policy."""
            obs = self.get_observation(state).to(next(self.parameters()).device)
            return self.policy.sample_action(obs, state, temperature)

        def forward_trajectory(self, max_steps: int = 20,
                                temperature: float = 1.0) -> Dict[str, Any]:
            """Generate one trajectory (sequence of states and actions).

            Returns
            -------
            dict with keys:
                - states: list of StructureState
                - actions: list of StructureAction
                - log_probs: list of tensors
                - final_state: terminal StructureState
            """
            state = StructureState(max_nodes=self.max_nodes, box_size=self.box_size)
            states = [state.copy()]
            actions = []
            log_probs = []

            for step in range(max_steps):
                obs = self.get_observation(state).to(next(self.parameters()).device)
                action, log_prob = self.policy.sample_action(obs, state, temperature)
                actions.append(action)
                log_probs.append(log_prob)
                state.apply_action(action)
                states.append(state.copy())

                if action.action_type == 3 or state.is_terminal:
                    break

            state._terminal = True
            return {
                "states": states,
                "actions": actions,
                "log_probs": log_probs,
                "final_state": state,
            }

        def sample(self, n: int = 10, max_steps: int = 20,
                   temperature: float = 1.0) -> List[StructureState]:
            """Sample n terminal structures.

            Parameters
            ----------
            n : int
                Number of structures to generate.
            max_steps : int
                Maximum construction steps.
            temperature : float
                Sampling temperature (higher = more exploration).

            Returns
            -------
            list of StructureState
                Generated terminal states.
            """
            self.eval()
            results = []
            with torch.no_grad():
                for _ in range(n):
                    traj = self.forward_trajectory(max_steps, temperature)
                    results.append(traj["final_state"])
            return results

        def to_structure_graph(self, state: StructureState):
            """Convert a terminal state to a StructureGraph.

            Returns
            -------
            StructureGraph or dict
                The generated fiber network structure.
            """
            try:
                from fibernet.core import StructureGraph, Node, Edge
                g = StructureGraph()
                for i, (pos, feat) in enumerate(zip(state.node_positions, state.node_features)):
                    g.add_node(Node(id=i, position=np.append(pos, [0.0])))
                for j, (i, k) in enumerate(state.edges):
                    g.add_edge(Edge(id=j, node_i=i, node_j=k))
                return g
            except ImportError:
                return {
                    "n_nodes": state.n_nodes,
                    "n_edges": state.n_edges,
                    "positions": state.node_positions,
                    "edges": state.edges,
                }

    # ==================================================================
    # Trainer
    # ==================================================================

    class GFlowNetTrainer:
        """Training loop for FiberGFlowNet.

        Supports Trajectory Balance (TB) and Detailed Balance (DB) losses.

        Parameters
        ----------
        model : FiberGFlowNet
            GFlowNet model.
        reward_fn : callable
            Maps StructureState → float reward (must be positive).
        lr : float
            Learning rate.
        loss_type : str
            "tb" or "db".
        max_grad_norm : float
            Gradient clipping norm.
        """

        def __init__(
            self,
            model: FiberGFlowNet,
            reward_fn: Callable[[StructureState], float],
            lr: float = 1e-3,
            max_grad_norm: float = 1.0,
        ):
            self.model = model
            self.reward_fn = reward_fn
            self.max_grad_norm = max_grad_norm

            params = list(model.parameters())
            self.optimizer = torch.optim.Adam(params, lr=lr)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=1000, eta_min=1e-5
            )

        def _compute_tb_loss(self, trajectories: List[Dict], rewards: List[float]) -> torch.Tensor:
            """Trajectory Balance loss: log(Z) + sum(log P_f) - log(R) = 0."""
            total_loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
            for traj, reward in zip(trajectories, rewards):
                log_pf = sum(traj["log_probs"])
                log_r = torch.tensor(math.log(max(reward, 1e-8)),
                                     device=total_loss.device)
                # TB loss: (log Z + log P_F - log R)^2
                loss = (self.model.log_Z + log_pf - log_r) ** 2
                total_loss = total_loss + loss
            return total_loss / len(trajectories)

        def _compute_db_loss(self, trajectories: List[Dict], rewards: List[float]) -> torch.Tensor:
            """Detailed Balance loss for consecutive state pairs."""
            total_loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
            count = 0
            for traj, reward in zip(trajectories, rewards):
                states = traj["states"]
                log_probs = traj["log_probs"]
                for t in range(len(log_probs)):
                    obs_s = self.model.get_observation(states[t]).to(total_loss.device)
                    log_f_s = self.model.flow_net(obs_s)

                    if t < len(states) - 1:
                        obs_sp = self.model.get_observation(states[t + 1]).to(total_loss.device)
                        log_f_sp = self.model.flow_net(obs_sp)
                        # DB: log F(s) + log P_F(s' | s) = log F(s') + log P_B(s | s')
                        # Simplified: just use forward flow consistency
                        db_loss = (log_f_s + log_probs[t] - log_f_sp) ** 2
                    else:
                        # Terminal: log F(s) = log R
                        log_r = torch.tensor(math.log(max(reward, 1e-8)),
                                            device=total_loss.device)
                        db_loss = (log_f_s - log_r) ** 2

                    total_loss = total_loss + db_loss
                    count += 1
            return total_loss / max(count, 1)

        def collect_trajectories(self, batch_size: int, max_steps: int = 20,
                                  temperature: float = 1.0) -> Tuple[List[Dict], List[float]]:
            """Collect a batch of trajectories and compute rewards."""
            self.model.eval()
            trajectories = []
            rewards = []
            with torch.no_grad():
                for _ in range(batch_size):
                    traj = self.model.forward_trajectory(max_steps, temperature)
                    reward = self.reward_fn(traj["final_state"])
                    trajectories.append(traj)
                    rewards.append(max(reward, 1e-8))
            return trajectories, rewards

        def train_step(self, batch_size: int = 16, max_steps: int = 20,
                        temperature: float = 1.0) -> Dict[str, float]:
            """Single training step."""
            self.model.train()
            trajectories, rewards = self.collect_trajectories(batch_size, max_steps, temperature)

            # Re-run trajectories with gradients for loss computation
            grad_trajectories = []
            for _ in range(batch_size):
                traj = self.model.forward_trajectory(max_steps, temperature)
                grad_trajectories.append(traj)

            if self.model.loss_type == "tb":
                loss = self._compute_tb_loss(grad_trajectories, rewards)
            else:
                loss = self._compute_db_loss(grad_trajectories, rewards)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            return {
                "loss": loss.item(),
                "mean_reward": np.mean(rewards),
                "max_reward": max(rewards),
            }

        def train(
            self,
            n_iterations: int = 500,
            batch_size: int = 16,
            max_steps: int = 20,
            temperature: float = 1.0,
            temperature_decay: float = 0.999,
            log_every: int = 50,
            verbose: bool = True,
        ) -> Dict[str, Any]:
            """Full training loop.

            Parameters
            ----------
            n_iterations : int
                Number of training iterations.
            batch_size : int
                Trajectories per iteration.
            max_steps : int
                Max steps per trajectory.
            temperature : float
                Initial sampling temperature.
            temperature_decay : float
                Temperature decay per iteration.
            log_every : int
                Log frequency.
            verbose : bool
                Print progress.

            Returns
            -------
            dict
                Training history.
            """
            history = {"loss": [], "mean_reward": [], "max_reward": []}
            temp = temperature

            for it in range(n_iterations):
                metrics = self.train_step(batch_size, max_steps, temp)
                history["loss"].append(metrics["loss"])
                history["mean_reward"].append(metrics["mean_reward"])
                history["max_reward"].append(metrics["max_reward"])

                temp *= temperature_decay
                self.scheduler.step()

                if verbose and (it % log_every == 0 or it == n_iterations - 1):
                    print(f"Iter {it:4d} | loss={metrics['loss']:.4f} | "
                          f"mean_r={metrics['mean_reward']:.3f} | "
                          f"max_r={metrics['max_reward']:.3f} | temp={temp:.3f}")

            return history

    # ==================================================================
    # Default Reward Functions
    # ==================================================================

    def connectivity_reward(state: StructureState) -> float:
        """Reward based on graph connectivity and density.

        Higher reward for well-connected structures with moderate density.
        """
        n = state.n_nodes
        e = state.n_edges
        if n < 2:
            return 0.1

        # Check connectivity via BFS
        adj = {i: set() for i in range(n)}
        for (i, j) in state.edges:
            adj[i].add(j)
            adj[j].add(i)

        visited = set()
        queue = [0]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            queue.extend(adj[node] - visited)

        connectivity = len(visited) / n
        # Reward connected, moderate density
        max_edges = n * (n - 1) / 2
        density = e / max(max_edges, 1)
        density_bonus = math.exp(-10 * (density - 0.3) ** 2)  # peak at 30%

        return float(connectivity * density_bonus * n)

    def property_target_reward(state: StructureState,
                                target_n_nodes: int = 15,
                                target_n_edges: int = 20,
                                target_connectivity: float = 1.0) -> float:
        """Reward targeting specific structural properties."""
        n = state.n_nodes
        e = state.n_edges
        if n < 2:
            return 0.1

        node_score = math.exp(-0.1 * (n - target_n_nodes) ** 2)
        edge_score = math.exp(-0.05 * (e - target_n_edges) ** 2)

        # Connectivity
        adj = {i: set() for i in range(n)}
        for (i, j) in state.edges:
            adj[i].add(j)
            adj[j].add(i)
        visited = set()
        queue = [0]
        while queue:
            nd = queue.pop(0)
            if nd in visited:
                continue
            visited.add(nd)
            queue.extend(adj[nd] - visited)
        conn = len(visited) / n

        return float(node_score * edge_score * conn * 10)

else:
    class PolicyNetwork:
        def __init__(self, *a, **kw):
            _require_torch()

    class FlowNetwork:
        def __init__(self, *a, **kw):
            _require_torch()

    class FiberGFlowNet:
        def __init__(self, *a, **kw):
            _require_torch()

    class GFlowNetTrainer:
        def __init__(self, *a, **kw):
            _require_torch()
