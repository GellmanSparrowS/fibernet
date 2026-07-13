#!/usr/bin/env python3
"""
RL Optimization: Bayesian + A2C for voronoi structure optimization

Optimize parameters to minimize max_force (or maximize mean_stretch uniformity).

Action space (5 continuous params):
- grid_x: [2, 5]
- grid_y: [2, 5]  
- n_internal: [5, 25]
- stiffness: [1e4, 1e6]
- damping: [0.1, 0.9]

Reward: -max_force (lower force = better structure)

Usage:
    python3 analysis_scripts/rl_optimize.py --method bayesian --n_iter 50
    python3 analysis_scripts/rl_optimize.py --method a2c --n_episodes 100
    python3 analysis_scripts/rl_optimize.py --method both
"""
import sys, json, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fibernet import pattern_2d, TaichiEngine

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output_data" / "rl_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_structure(params: dict, target_stretch: float = 1.5, num_steps: int = 5000) -> dict:
    """Generate + simulate a structure, return metrics."""
    engine = TaichiEngine()
    
    grid_x = max(2, min(5, int(round(params["grid_x"]))))
    grid_y = max(2, min(5, int(round(params["grid_y"]))))
    n_internal = max(5, min(25, int(round(params["n_internal"]))))
    stiffness = max(1e4, min(1e6, params["stiffness"]))
    damping = max(0.1, min(0.9, params["damping"]))
    seed = int(params.get("seed", 42))
    
    try:
        g = pattern_2d(unit="voronoi", box=(10, 10),
                       grid=(grid_x, grid_y), seed=seed,
                       n_internal=n_internal)
        result = engine.stretch_test(g, target_stretch=target_stretch,
                                      stiffness=stiffness, damping=damping,
                                      auto_steps=False, num_steps=num_steps)
        return {
            "max_force": result.max_force,
            "max_stretch": result.max_stretch,
            "mean_stretch": result.mean_stretch,
            "std_stretch": result.std_stretch,
            "n_nodes": g.num_nodes,
            "n_edges": g.num_edges,
            "success": True,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ───────────────── Bayesian Optimization ─────────────────

def run_bayesian(n_iter: int = 50):
    """Bayesian optimization to minimize max_force."""
    try:
        from skopt import gp_minimize
        from skopt.space import Real, Integer
        from skopt.utils import use_named_args
    except ImportError:
        print("scikit-optimize required: pip install scikit-optimize")
        return

    print("=" * 60)
    print("Bayesian Optimization (Gaussian Process)")
    print("=" * 60)

    space = [
        Integer(2, 5, name="grid_x"),
        Integer(2, 5, name="grid_y"),
        Integer(5, 25, name="n_internal"),
        Real(1e4, 1e6, name="stiffness", prior="log-uniform"),
        Real(0.1, 0.9, name="damping"),
    ]

    results_log = []

    @use_named_args(space)
    def objective(grid_x, grid_y, n_internal, stiffness, damping):
        params = {"grid_x": grid_x, "grid_y": grid_y, "n_internal": n_internal,
                  "stiffness": stiffness, "damping": damping}
        r = evaluate_structure(params, num_steps=5000)
        if r["success"]:
            # Minimize max_force, also penalize high std_stretch
            reward = r["max_force"] + 1e4 * r["std_stretch"]
            results_log.append({**params, **r, "reward": reward})
            print(f"  [{len(results_log):3d}] grid=({grid_x},{grid_y}) "
                  f"n_int={n_internal:2d} k={stiffness:.1e} d={damping:.2f} → "
                  f"max_F={r['max_force']:.0f} std={r['std_stretch']:.3f} R={reward:.0f}")
            return reward
        return 1e10  # penalty for failure

    t0 = time.time()
    result = gp_minimize(objective, space, n_calls=n_iter, random_state=42,
                         verbose=False)
    elapsed = time.time() - t0

    print(f"\nBayesian Optimization Results ({elapsed:.1f}s):")
    print(f"  Best reward: {result.fun:.0f}")
    print(f"  Best params: grid=({result.x[0]},{result.x[1]}), "
          f"n_internal={result.x[2]}, stiffness={result.x[3]:.1e}, damping={result.x[4]:.2f}")

    # Save results
    bayes_data = {
        "method": "bayesian",
        "n_iter": n_iter,
        "best_reward": float(result.fun),
        "best_params": {
            "grid_x": int(result.x[0]),
            "grid_y": int(result.x[1]),
            "n_internal": int(result.x[2]),
            "stiffness": float(result.x[3]),
            "damping": float(result.x[4]),
        },
        "all_results": results_log,
        "convergence": [float(x) for x in result.func_vals],
    }
    with open(OUTPUT_DIR / "bayesian_results.json", "w") as f:
        json.dump(bayes_data, f, indent=2, default=str)

    # Convergence plot
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0a0a0f')
    best_so_far = np.minimum.accumulate(result.func_vals)
    ax.plot(result.func_vals, 'o', color='#7c4dff', alpha=0.5, markersize=4, label="All trials")
    ax.plot(best_so_far, '-', color='#b388ff', linewidth=2, label="Best so far")
    ax.set_xlabel("Iteration", color='#d0d0d0')
    ax.set_ylabel("Reward (max_force + penalty)", color='#d0d0d0')
    ax.set_title("Bayesian Optimization Convergence", color='white', fontsize=14)
    ax.set_facecolor('#0a0a0f')
    ax.tick_params(colors='#d0d0d0')
    ax.legend(facecolor='#1a1a2a')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "bayesian_convergence.png", dpi=150,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"Results saved to {OUTPUT_DIR}/")
    return bayes_data


# ───────────────── A2C (Advantage Actor-Critic) ─────────────────

class VoronoiEnv:
    """Surrogate-based environment for RL optimization.
    
    State: normalized [grid_x, grid_y, n_internal] in [0, 1]
    Action: delta in [-0.5, 0.5] applied to state
    Reward: -predicted_max_force (higher = better, i.e. lower force)
    """
    
    def __init__(self, csv_path=None):
        import pandas as pd
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        
        if csv_path is None:
            csv_path = str(Path(__file__).resolve().parent.parent / "output_data" / "voronoi_100_results.csv")
        
        df = pd.read_csv(csv_path)
        
        self.param_cols = ["grid_x", "grid_y", "n_internal"]
        X_params = df[self.param_cols].values
        y_target = df["max_force"].values
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_params)
        self.surrogate = RandomForestRegressor(n_estimators=50, random_state=42)
        self.surrogate.fit(X_scaled, y_target)
        
        self.param_low = np.array([2.0, 2.0, 5.0])
        self.param_high = np.array([5.0, 5.0, 25.0])
        self.dim = 3
        self.obs_dim = 3
        self.action_dim = 3
        self._step_count = 0
        self.max_steps = 20
        self._state = None  # normalized [0,1]
    
    def _to_params(self, norm_state):
        """Convert normalized [0,1] state to actual params."""
        return norm_state * (self.param_high - self.param_low) + self.param_low
    
    def _predict(self, params):
        """Predict max_force from params."""
        params_round = np.array([round(params[0]), round(params[1]), round(params[2])])
        params_round = np.clip(params_round, self.param_low, self.param_high)
        X_scaled = self.scaler.transform(params_round.reshape(1, -1))
        return float(self.surrogate.predict(X_scaled)[0])
    
    def reset(self):
        self._step_count = 0
        # Start from random normalized state
        self._state = np.random.uniform(0, 1, self.dim)
        return self._state.copy()
    
    def step(self, action):
        """action: delta in [-0.5, 0.5] added to normalized state"""
        self._step_count += 1
        
        # Apply action (delta) and clip to [0, 1]
        self._state = np.clip(self._state + np.clip(action, -0.5, 0.5), 0, 1)
        
        # Convert to params and predict
        params = self._to_params(self._state)
        predicted_force = self._predict(params)
        
        # Reward: negative force (we want to minimize force)
        reward = -predicted_force / 1e5  # normalize reward scale
        
        done = self._step_count >= self.max_steps
        info = {"predicted_max_force": predicted_force, "params": params.tolist(), "success": True}
        
        return self._state.copy(), reward, done, info
    
    @property
    def action_space(self):
        return type('Space', (), {
            'sample': lambda: np.random.uniform(-0.3, 0.3, self.action_dim),
            'shape': (self.action_dim,)
        })()
    
    @property
    def observation_space(self):
        return type('Space', (), {'shape': (self.obs_dim,)})()



class A2CAgent:
    """Simple A2C agent with MLP policy."""
    
    def __init__(self, obs_dim, action_dim, lr=1e-3, gamma=0.99):
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except ImportError:
            raise ImportError("PyTorch required: pip install torch")
        
        self.gamma = gamma
        self.device = torch.device('cpu')
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim * 2),  # mean + log_std
        ).to(self.device)
        
        # Critic network (value)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr
        )
    
    def select_action(self, state):
        import torch
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        output = self.actor(state_t)
        action_dim = output.shape[1] // 2
        mean = output[:, :action_dim]
        log_std = output[:, action_dim:].clamp(-2, 2)
        std = log_std.exp()
        
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        action = action.clamp(-0.5, 0.5)
        log_prob = dist.log_prob(action).sum(-1)
        value = self.critic(state_t)
        
        return action.squeeze(0).cpu().numpy(), log_prob, value.squeeze()
    
    def update(self, states, actions, rewards, log_probs, values):
        import torch
        
        # Compute returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        log_probs_t = torch.stack(log_probs)
        values_t = torch.stack(values)
        
        # Advantage
        advantages = returns - values_t.detach()
        
        # Actor loss
        actor_loss = -(log_probs_t * advantages).mean()
        
        # Critic loss
        critic_loss = 0.5 * (returns - values_t).pow(2).mean()
        
        # Total loss
        loss = actor_loss + 0.5 * critic_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


def run_a2c(n_episodes: int = 100):
    """A2C reinforcement learning optimization."""
    print("=" * 60)
    print("A2C (Advantage Actor-Critic) Optimization")
    print("=" * 60)
    
    env = VoronoiEnv()
    agent = A2CAgent(env.obs_dim, env.action_dim, lr=3e-4)
    
    episode_rewards = []
    best_reward = -float('inf')
    best_params = None
    all_results = []
    
    for ep in range(n_episodes):
        state = env.reset()
        total_reward = 0
        states, actions, rewards, log_probs, values = [], [], [], [], []
        
        done = False
        while not done:
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            
            total_reward += reward
            state = next_state
        
        # Update
        if len(states) > 0:
            loss = agent.update(states, actions, rewards, log_probs, values)
        
        episode_rewards.append(total_reward)
        
        if info.get("success"):
            all_results.append({
                "episode": ep,
                "reward": total_reward,
                "max_force": info.get("predicted_max_force", info.get("max_force", 0)),
                "params": {
                    "grid_x": float(state[0]),
                    "grid_y": float(state[1]),
                    "n_internal": float(state[2]),
                }
            })
        
        if total_reward > best_reward:
            best_reward = total_reward
            best_params = state.copy()
        
        if (ep + 1) % 10 == 0:
            avg_r = np.mean(episode_rewards[-10:])
            print(f"  Episode {ep+1}/{n_episodes}: avg_reward={avg_r:.0f}, "
                  f"best={best_reward:.0f}")
    
    print(f"\nA2C Results:")
    print(f"  Best reward: {best_reward:.0f}")
    print(f"  Best params: grid=({best_params[0]:.0f},{best_params[1]:.0f}), "
          f"n_internal={best_params[2]:.0f}")
    
    # Save
    a2c_data = {
        "method": "a2c",
        "n_episodes": n_episodes,
        "best_reward": float(best_reward),
        "best_params": {
            "grid_x": float(best_params[0]),
            "grid_y": float(best_params[1]),
            "n_internal": float(best_params[2]),
        },
        "episode_rewards": [float(r) for r in episode_rewards],
        "all_results": all_results,
    }
    with open(OUTPUT_DIR / "a2c_results.json", "w") as f:
        json.dump(a2c_data, f, indent=2)
    
    # Convergence plot
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0a0a0f')
    # Smooth
    window = 10
    smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
    ax.plot(episode_rewards, alpha=0.3, color='#7c4dff', label="Raw")
    ax.plot(range(window-1, len(smoothed)+window-1), smoothed, color='#b388ff', linewidth=2, label=f"Smoothed (w={window})")
    ax.set_xlabel("Episode", color='#d0d0d0')
    ax.set_ylabel("Total Reward", color='#d0d0d0')
    ax.set_title("A2C Training Progress", color='white', fontsize=14)
    ax.set_facecolor('#0a0a0f')
    ax.tick_params(colors='#d0d0d0')
    ax.legend(facecolor='#1a1a2a')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "a2c_convergence.png", dpi=150,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    
    print(f"Results saved to {OUTPUT_DIR}/")
    return a2c_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="both", choices=["bayesian", "a2c", "both"])
    parser.add_argument("--n_iter", type=int, default=50, help="Bayesian iterations")
    parser.add_argument("--n_episodes", type=int, default=100, help="A2C episodes")
    args = parser.parse_args()
    
    if args.method in ("bayesian", "both"):
        run_bayesian(args.n_iter)
    
    if args.method in ("a2c", "both"):
        run_a2c(args.n_episodes)


if __name__ == "__main__":
    main()
