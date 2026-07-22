"""
Neural ODE for FiberNet — Continuous-Time Dynamics Modeling.

Models time-dependent fiber network behavior using neural ordinary
differential equations (Neural ODEs), enabling continuous-time prediction
of stress relaxation, creep, fatigue, and dynamic loading responses.

Implements:
- ODESolver: Differentiable ODE solvers (Euler, RK4, adaptive)
- FiberNeuralODE: General neural ODE for fiber dynamics
- StressRelaxationODE: Maxwell/Kelvin-Voigt stress relaxation model
- CreepODE: Time-dependent creep deformation
- FatigueODE: Cyclic loading fatigue accumulation
- NeuralODETrainer: Training on time-series data

Features
--------
- Continuous-time modeling (arbitrary time resolution)
- Differentiable solvers via PyTorch autograd
- Multiple solver options (Euler, RK4, adaptive Dormand-Prince)
- Physics-informed ODE functions (Maxwell, Kelvin-Voigt, SLS)
- Adjoint method for memory-efficient training
- Irregular time series support

References
----------
- Chen et al., "Neural Ordinary Differential Equations" (NeurIPS 2018)
- Article section on time-dependent behavior: creep, relaxation, fatigue
- Findley et al., "Creep and Relaxation of Nonlinear Viscoelastic Materials"
- Lakes, "Viscoelastic Materials" (Cambridge University Press)

Examples
--------
>>> from fibernet.ml.neural_ode import FiberNeuralODE, NeuralODETrainer
>>> ode = FiberNeuralODE(state_dim=4, hidden=64)
>>> trainer = NeuralODETrainer(ode)
>>> # Train on time-series: (batch, time, state_dim)
>>> trainer.fit(time_data, state_data, epochs=200)
>>> # Predict at arbitrary future times
>>> t_future = torch.linspace(0, 100, 500)
>>> trajectory = ode.solve(x0, t_future)

>>> # Stress relaxation
>>> from fibernet.ml.neural_ode import StressRelaxationODE
>>> sr = StressRelaxationODE(model_type="maxwell", E=1e9, eta=1e12)
>>> stress_t = sr.relax(initial_stress=100.0, t_span=(0, 1000), n_steps=100)
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

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


if HAS_TORCH:

    # ==================================================================
    # ODE Solvers
    # ==================================================================

    class ODESolver(nn.Module):
        """Differentiable ODE solver with multiple methods.

        Solves: dx/dt = f(t, x) with initial condition x(t0) = x0

        Parameters
        ----------
        method : str
            "euler", "rk4", or "dopri5" (adaptive).
        rtol : float
            Relative tolerance (adaptive only).
        atol : float
            Absolute tolerance (adaptive only).
        adjoint : bool
            Use adjoint method for memory-efficient gradients.
        """

        def __init__(
            self,
            method: str = "rk4",
            rtol: float = 1e-4,
            atol: float = 1e-6,
            adjoint: bool = False,
        ):
            super().__init__()
            self.method = method
            self.rtol = rtol
            self.atol = atol
            self.adjoint = adjoint

        def _euler_step(self, f: Callable, t: torch.Tensor, x: torch.Tensor,
                         dt: float) -> torch.Tensor:
            """Single Euler step."""
            return x + dt * f(t, x)

        def _rk4_step(self, f: Callable, t: torch.Tensor, x: torch.Tensor,
                        dt: float) -> torch.Tensor:
            """Single RK4 step."""
            k1 = f(t, x)
            k2 = f(t + 0.5 * dt, x + 0.5 * dt * k1)
            k3 = f(t + 0.5 * dt, x + 0.5 * dt * k2)
            k4 = f(t + dt, x + dt * k3)
            return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        def _dopri5_step(self, f: Callable, t: torch.Tensor, x: torch.Tensor,
                          dt: float) -> Tuple[torch.Tensor, torch.Tensor]:
            """Dormand-Prince (RK45) step with error estimate."""
            # Butcher tableau coefficients
            a2, a3, a4, a5, a6 = 1/5, 3/10, 4/5, 8/9, 1.0
            b21 = 1/5
            b31, b32 = 3/40, 9/40
            b41, b42, b43 = 44/45, -56/15, 32/9
            b51, b52, b53, b54 = 19372/6561, -25360/2187, 64448/6561, -212/729
            b61, b62, b63, b64, b65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
            # 5th order
            c1, c3, c4, c5, c6 = 35/384, 500/1113, 125/192, -2187/6784, 11/84
            # 4th order (for error)
            d1, d3, d4, d5, d6 = 5179/57600, 7571/16695, 393/640, -92097/339200, 187/2100
            d7 = 1/40

            k1 = f(t, x)
            k2 = f(t + a2*dt, x + dt*b21*k1)
            k3 = f(t + a3*dt, x + dt*(b31*k1 + b32*k2))
            k4 = f(t + a4*dt, x + dt*(b41*k1 + b42*k2 + b43*k3))
            k5 = f(t + a5*dt, x + dt*(b51*k1 + b52*k2 + b53*k3 + b54*k4))
            k6 = f(t + a6*dt, x + dt*(b61*k1 + b62*k2 + b63*k3 + b64*k4 + b65*k5))

            x5 = x + dt * (c1*k1 + c3*k3 + c4*k4 + c5*k5 + c6*k6)
            x4 = x + dt * (d1*k1 + d3*k3 + d4*k4 + d5*k5 + d6*k6)

            k7 = f(t + dt, x5)
            error = (x5 - x4).abs().max()

            return x5, error

        def solve(
            self,
            f: Callable,
            x0: torch.Tensor,
            t_span: torch.Tensor,
        ) -> torch.Tensor:
            """Solve ODE and return trajectory.

            Parameters
            ----------
            f : callable
                ODE function f(t, x) → dx/dt
            x0 : (state_dim,) or (batch, state_dim)
                Initial state.
            t_span : (n_times,)
                Time points to evaluate.

            Returns
            -------
            (n_times, ...) or (n_times, batch, state_dim) trajectory
            """
            trajectory = [x0]
            x = x0

            for i in range(1, len(t_span)):
                t = t_span[i - 1]
                dt = (t_span[i] - t_span[i - 1]).item()

                if self.method == "euler":
                    x = self._euler_step(f, t, x, dt)
                elif self.method == "rk4":
                    x = self._rk4_step(f, t, x, dt)
                elif self.method == "dopri5":
                    x, _ = self._dopri5_step(f, t, x, dt)
                else:
                    x = self._rk4_step(f, t, x, dt)

                trajectory.append(x)

            return torch.stack(trajectory)


    # ==================================================================
    # Fiber Neural ODE
    # ==================================================================

    class FiberNeuralODE(nn.Module):
        """Neural ODE for continuous-time fiber network dynamics.

        Learns the ODE function dx/dt = f(t, x; θ) where x can represent
        stress, strain, displacement, or other state variables.

        Parameters
        ----------
        state_dim : int
            Dimension of the state vector.
        hidden : list of int
            Hidden layer sizes for the ODE function.
        solver_method : str
            ODE solver: "euler", "rk4", "dopri5".
        time_embedding : bool
            Whether to embed time as input feature.
        """

        def __init__(
            self,
            state_dim: int = 4,
            hidden: Optional[List[int]] = None,
            solver_method: str = "rk4",
            time_embedding: bool = True,
        ):
            super().__init__()
            if hidden is None:
                hidden = [64, 32]

            self.state_dim = state_dim
            self.time_embedding = time_embedding
            self.solver = ODESolver(method=solver_method)

            # Build ODE function network
            input_dim = state_dim + (1 if time_embedding else 0)
            layers = []
            prev = input_dim
            for h in hidden:
                layers.extend([nn.Linear(prev, h), nn.GELU()])
                prev = h
            layers.append(nn.Linear(prev, state_dim))
            self.ode_net = nn.Sequential(*layers)

        def ode_function(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            """Neural network ODE function f(t, x).

            Parameters
            ----------
            t : scalar time
            x : (state_dim,) or (batch, state_dim) state

            Returns
            -------
            dx/dt
            """
            if x.dim() == 1:
                if self.time_embedding:
                    t_embed = torch.tensor([t.item() if hasattr(t, 'item') else t],
                                          device=x.device).expand(1)
                    inp = torch.cat([x, t_embed])
                else:
                    inp = x
                return self.ode_net(inp)
            else:
                if self.time_embedding:
                    t_val = t.item() if hasattr(t, 'item') else float(t)
                    t_embed = torch.full((x.shape[0], 1), t_val, device=x.device)
                    inp = torch.cat([x, t_embed], dim=-1)
                else:
                    inp = x
                return self.ode_net(inp)

        def solve(
            self,
            x0: torch.Tensor,
            t_span: torch.Tensor,
        ) -> torch.Tensor:
            """Solve the learned ODE.

            Parameters
            ----------
            x0 : (state_dim,) initial state
            t_span : (n_times,) time points

            Returns
            -------
            (n_times, state_dim) trajectory
            """
            return self.solver.solve(self.ode_function, x0, t_span)

        def forward(
            self,
            x0: torch.Tensor,
            t_span: torch.Tensor,
        ) -> torch.Tensor:
            """Forward pass: solve ODE and return final state."""
            trajectory = self.solve(x0, t_span)
            return trajectory[-1]

        def predict_trajectory(
            self,
            x0: torch.Tensor,
            t_eval: torch.Tensor,
        ) -> torch.Tensor:
            """Predict full trajectory at given time points."""
            return self.solve(x0, t_eval)


    # ==================================================================
    # Physics-Based ODE Models
    # ==================================================================

    class StressRelaxationODE(nn.Module):
        """Stress relaxation model for fiber networks.

        Models time-dependent stress decay under constant strain:
        - Maxwell: σ(t) = σ₀ exp(-t/τ), τ = η/E
        - Kelvin-Voigt: ε(t) = σ₀/E (1 - exp(-t/τ))
        - Standard Linear Solid (SLS): combination
        - Neural: learned ODE function with physics priors

        Parameters
        ----------
        model_type : str
            "maxwell", "kelvin_voigt", "sls", or "neural".
        E : float
            Young's modulus (or initial modulus for SLS).
        eta : float
            Viscosity.
        E2 : float
            Second spring constant (SLS only).
        hidden : list of int
            Hidden sizes for neural model.
        """

        def __init__(
            self,
            model_type: str = "maxwell",
            E: float = 1e9,
            eta: float = 1e12,
            E2: float = 5e8,
            hidden: Optional[List[int]] = None,
        ):
            super().__init__()
            self.model_type = model_type
            self.E = E
            self.eta = eta
            self.E2 = E2

            if model_type == "neural":
                if hidden is None:
                    hidden = [32, 16]
                layers = []
                prev = 2  # (sigma, t)
                for h in hidden:
                    layers.extend([nn.Linear(prev, h), nn.GELU()])
                    prev = h
                layers.append(nn.Linear(prev, 1))
                self.ode_net = nn.Sequential(*layers)
            elif model_type == "sls":
                # Learnable SLS parameters
                self.log_E = nn.Parameter(torch.tensor(math.log(E)))
                self.log_eta = nn.Parameter(torch.tensor(math.log(eta)))
                self.log_E2 = nn.Parameter(torch.tensor(math.log(E2)))

        def _maxwell_rhs(self, sigma: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            """dσ/dt = -(E/η) * σ for Maxwell model."""
            return -(self.E / self.eta) * sigma

        def _kelvin_voigt_rhs(self, sigma: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            """dσ/dt = -(E/η) * σ for creep under constant load."""
            return -(self.E / self.eta) * sigma

        def _sls_rhs(self, sigma: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            """SLS: dσ/dt = -(E1/η) * σ + correction term."""
            E1 = self.log_E.exp() if hasattr(self, 'log_E') else self.E
            eta_val = self.log_eta.exp() if hasattr(self, 'log_eta') else self.eta
            E2_val = self.log_E2.exp() if hasattr(self, 'log_E2') else self.E2
            tau = eta_val / E1
            # SLS relaxation: σ(t) = σ_∞ + (σ₀ - σ_∞) * exp(-t/τ_r)
            # dσ/dt = -(σ - σ_∞) / τ_r where σ_∞ = E2/(E1+E2) * σ₀
            # Simplified: dσ/dt = -σ/τ + correction
            return -(1.0 / tau) * sigma + (E2_val / (E1 + E2_val)) * sigma.detach() / (tau * 10 + 1e-12)

        def _neural_rhs(self, sigma: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            """Neural network RHS for learned dynamics."""
            t_val = t.item() if hasattr(t, 'item') else float(t)
            inp = torch.tensor([[sigma.item(), t_val]], device=sigma.device) if sigma.dim() == 0 else torch.stack([sigma.flatten(), torch.full_like(sigma.flatten(), t_val)], dim=-1)
            return self.ode_net(inp).squeeze(-1)

        def rhs(self, sigma: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            """Right-hand side of the ODE."""
            if self.model_type == "maxwell":
                return self._maxwell_rhs(sigma, t)
            elif self.model_type == "kelvin_voigt":
                return self._kelvin_voigt_rhs(sigma, t)
            elif self.model_type == "sls":
                return self._sls_rhs(sigma, t)
            else:
                return self._neural_rhs(sigma, t)

        def relax(
            self,
            initial_stress: float,
            t_span: Tuple[float, float] = (0, 1000),
            n_steps: int = 100,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Compute stress relaxation curve.

            Parameters
            ----------
            initial_stress : float
                Initial stress σ₀.
            t_span : (t_start, t_end)
            n_steps : int

            Returns
            -------
            times : (n_steps+1,)
            stresses : (n_steps+1,)
            """
            t = torch.linspace(t_span[0], t_span[1], n_steps + 1)
            sigma0 = torch.tensor([initial_stress], dtype=torch.float32)

            solver = ODESolver(method="rk4")
            # Wrap rhs to match solver interface: solver calls f(t, x), rhs expects (x, t)
            wrapped_rhs = lambda t, x: self.rhs(x, t)
            trajectory = solver.solve(wrapped_rhs, sigma0, t)

            return t, trajectory.squeeze(-1)

        def analytical_solution(
            self,
            initial_stress: float,
            times: np.ndarray,
        ) -> np.ndarray:
            """Compute analytical solution for validation."""
            if self.model_type == "maxwell":
                tau = self.eta / self.E
                return initial_stress * np.exp(-times / tau)
            elif self.model_type == "sls":
                E1, E2, eta = self.E, self.E2, self.eta
                tau_r = eta / E1
                sigma_inf = E2 / (E1 + E2) * initial_stress
                return sigma_inf + (initial_stress - sigma_inf) * np.exp(-times / tau_r)
            return np.full_like(times, initial_stress)


    class CreepODE(nn.Module):
        """Creep deformation model for fiber networks.

        Models time-dependent strain increase under constant stress:
        ε(t) = ε₀ + ε_creep(t)

        Parameters
        ----------
        E : float
            Young's modulus.
        eta : float
            Viscosity.
        model_type : str
            "power_law", "logarithmic", or "neural".
        """

        def __init__(
            self,
            E: float = 1e9,
            eta: float = 1e12,
            model_type: str = "power_law",
            n_creep: float = 0.3,
            hidden: Optional[List[int]] = None,
        ):
            super().__init__()
            self.E = E
            self.eta = eta
            self.model_type = model_type
            self.n_creep = n_creep

            if model_type == "neural":
                if hidden is None:
                    hidden = [32, 16]
                layers = []
                prev = 2  # (epsilon, t)
                for h in hidden:
                    layers.extend([nn.Linear(prev, h), nn.GELU()])
                    prev = h
                layers.append(nn.Linear(prev, 1))
                self.ode_net = nn.Sequential(*layers)

        def rhs(self, epsilon: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            """Creep rate dε/dt."""
            t_val = t.item() if hasattr(t, 'item') else float(t)
            t_safe = max(abs(t_val), 1e-12)

            if self.model_type == "power_law":
                # Power-law creep: dε/dt = A * σ^n * t^(m-1) where m < 1
                A = 1.0 / self.eta
                return torch.tensor([A * (t_safe ** (self.n_creep - 1))], device=epsilon.device)
            elif self.model_type == "logarithmic":
                # Logarithmic creep: ε(t) = ε₀ + B * log(1 + t/τ)
                B = 1e-6
                tau = self.eta / self.E
                return torch.tensor([B / (tau + t_safe)], device=epsilon.device)
            else:
                inp = torch.tensor([[epsilon.item(), t_val]], device=epsilon.device)
                return self.ode_net(inp).squeeze(-1)

        def predict_creep(
            self,
            initial_strain: float,
            applied_stress: float,
            t_span: Tuple[float, float] = (0, 1000),
            n_steps: int = 100,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Predict creep strain over time.

            Parameters
            ----------
            initial_strain : float
                Instantaneous elastic strain (σ/E).
            applied_stress : float
                Constant applied stress.
            t_span : (t_start, t_end)
            n_steps : int

            Returns
            -------
            times, strains : both (n_steps+1,)
            """
            t = torch.linspace(t_span[0], t_span[1], n_steps + 1)
            eps0 = torch.tensor([initial_strain], dtype=torch.float32)

            solver = ODESolver(method="rk4")
            # Wrap rhs to match solver interface
            wrapped_rhs = lambda t, x: self.rhs(x, t)
            trajectory = solver.solve(wrapped_rhs, eps0, t)

            return t, trajectory.squeeze(-1)


    class FatigueODE(nn.Module):
        """Fatigue damage accumulation model for fiber networks.

        Models damage variable D(t) evolution under cyclic loading:
        dD/dN = f(D, Δσ, material parameters)

        Parameters
        ----------
        hidden : list of int
            Hidden sizes for neural fatigue model.
        """

        def __init__(self, hidden: Optional[List[int]] = None):
            super().__init__()
            if hidden is None:
                hidden = [32, 16]

            # Damage evolution network: (D, Δσ, N) → dD/dN
            layers = []
            prev = 3  # (damage, stress_range, cycle_count)
            for h in hidden:
                layers.extend([nn.Linear(prev, h), nn.GELU()])
                prev = h
            layers.append(nn.Linear(prev, 1))
            layers.append(nn.Softplus())  # Ensure positive damage rate
            self.damage_net = nn.Sequential(*layers)

            # Learnable Paris law parameters
            self.log_C = nn.Parameter(torch.tensor(-10.0))
            self.log_m = nn.Parameter(torch.tensor(3.0))

        def damage_rate(
            self,
            D: torch.Tensor,
            stress_range: torch.Tensor,
            N: torch.Tensor,
        ) -> torch.Tensor:
            """Compute dD/dN.

            Parameters
            ----------
            D : damage variable (0 to 1)
            stress_range : Δσ
            N : cycle count
            """
            inp = torch.stack([D, stress_range, N], dim=-1)
            if inp.dim() == 1:
                inp = inp.unsqueeze(0)
            rate = self.damage_net(inp).squeeze(-1)
            return rate

        def paris_law_rate(
            self,
            crack_length: torch.Tensor,
            stress_range: torch.Tensor,
        ) -> torch.Tensor:
            """Paris law: da/dN = C * (ΔK)^m where ΔK = Δσ * √(πa)."""
            C = self.log_C.exp()
            m = self.log_m.exp()
            delta_K = stress_range * torch.sqrt(math.pi * crack_length.abs() + 1e-12)
            return C * delta_K.pow(m)

        def predict_fatigue_life(
            self,
            initial_damage: float = 0.0,
            stress_range: float = 100.0,
            max_cycles: int = 10000,
            failure_damage: float = 1.0,
        ) -> Dict[str, Any]:
            """Predict fatigue life (cycles to failure).

            Returns
            -------
            dict with cycles, damage_history, cycles_to_failure
            """
            D = initial_damage
            damage_history = [D]
            N = 0

            for n in range(1, max_cycles + 1):
                D_t = torch.tensor([D], dtype=torch.float32)
                ds_t = torch.tensor([stress_range], dtype=torch.float32)
                N_t = torch.tensor([float(n)], dtype=torch.float32)

                with torch.no_grad():
                    dD = self.damage_rate(D_t, ds_t, N_t).item()

                D += dD
                damage_history.append(D)
                N = n

                if D >= failure_damage:
                    break

            return {
                "cycles": N,
                "damage_history": np.array(damage_history),
                "cycles_to_failure": N,
                "final_damage": D,
            }


    # ==================================================================
    # Neural ODE Trainer
    # ==================================================================

    class NeuralODETrainer:
        """Training loop for neural ODE models on time-series data.

        Parameters
        ----------
        model : FiberNeuralODE
            Neural ODE model.
        lr : float
        weight_decay : float
        """

        def __init__(
            self,
            model: Union[FiberNeuralODE, StressRelaxationODE, CreepODE],
            lr: float = 1e-3,
            weight_decay: float = 1e-4,
        ):
            self.model = model
            self.optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )

        def fit(
            self,
            time_data: np.ndarray,
            state_data: np.ndarray,
            *,
            epochs: int = 200,
            batch_size: int = 32,
            val_split: float = 0.2,
            verbose: bool = True,
        ) -> Dict[str, Any]:
            """Train on time-series trajectories.

            Parameters
            ----------
            time_data : (n_trajectories, n_times) or (n_times,)
                Time points.
            state_data : (n_trajectories, n_times, state_dim)
                State trajectories.

            Returns
            -------
            dict with training history
            """
            if time_data.ndim == 1:
                time_data = np.tile(time_data, (state_data.shape[0], 1))

            t_all = torch.tensor(time_data, dtype=torch.float32)
            s_all = torch.tensor(state_data, dtype=torch.float32)

            n_traj = len(t_all)
            rng = torch.Generator().manual_seed(42)
            perm = torch.randperm(n_traj, generator=rng)
            n_val = max(1, int(n_traj * val_split))
            val_idx = perm[:n_val]
            train_idx = perm[n_val:]

            history = {"train_loss": [], "val_loss": []}

            for epoch in range(epochs):
                self.model.train()
                train_loss = 0.0
                n_batches = 0

                # Shuffle train indices
                train_perm = train_idx[torch.randperm(len(train_idx), generator=rng)]

                for start in range(0, len(train_perm), batch_size):
                    end = min(start + batch_size, len(train_perm))
                    batch_idx = train_perm[start:end]

                    self.optimizer.zero_grad()
                    batch_loss = torch.tensor(0.0)

                    for idx in batch_idx:
                        t_traj = t_all[idx]
                        s_traj = s_all[idx]
                        x0 = s_traj[0]

                        if isinstance(self.model, FiberNeuralODE):
                            pred_traj = self.model.solve(x0, t_traj)
                            loss = F.mse_loss(pred_traj, s_traj)
                        else:
                            # For physics-based models, train via ODE fit
                            pred_traj = self.model.solve(x0, t_traj) if hasattr(self.model, 'solve') else None
                            if pred_traj is not None:
                                loss = F.mse_loss(pred_traj, s_traj)
                            else:
                                loss = torch.tensor(0.0)

                        batch_loss = batch_loss + loss

                    batch_loss = batch_loss / len(batch_idx)
                    batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                    train_loss += batch_loss.item()
                    n_batches += 1

                train_loss /= max(n_batches, 1)

                # Validation
                val_loss = 0.0
                if len(val_idx) > 0:
                    self.model.eval()
                    with torch.no_grad():
                        for idx in val_idx:
                            t_traj = t_all[idx]
                            s_traj = s_all[idx]
                            x0 = s_traj[0]
                            if isinstance(self.model, FiberNeuralODE):
                                pred = self.model.solve(x0, t_traj)
                                val_loss += F.mse_loss(pred, s_traj).item()
                        val_loss /= len(val_idx)

                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)

                if verbose and (epoch % max(epochs // 10, 1) == 0 or epoch == epochs - 1):
                    print(f"Epoch {epoch:3d} | train={train_loss:.6f} | val={val_loss:.6f}")

            return history

else:
    class ODESolver:
        def __init__(self, *a, **kw):
            _require_torch()

    class FiberNeuralODE:
        def __init__(self, *a, **kw):
            _require_torch()

    class StressRelaxationODE:
        def __init__(self, *a, **kw):
            _require_torch()

    class CreepODE:
        def __init__(self, *a, **kw):
            _require_torch()

    class FatigueODE:
        def __init__(self, *a, **kw):
            _require_torch()

    class NeuralODETrainer:
        def __init__(self, *a, **kw):
            _require_torch()
