"""
RL infrastructure: networks, environment setup, plotting.
Students use this module but don't need to modify it.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import gymnasium as gym
import matplotlib.pyplot as plt

# ── Reproducibility ──────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")  # CPU is faster for this small-scale exercise


# ── Environment ──────────────────────────────────────────────────

def make_env(env_id, seed=SEED):
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.reset(seed=seed)
    return env


# ── Neural Networks ──────────────────────────────────────────────

class PolicyNetwork(nn.Module):
    """Simple MLP that outputs action logits."""
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, obs):
        return self.net(obs)

    def get_action(self, obs):
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action)

    def get_log_prob(self, obs, actions):
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions)


class ValueNetwork(nn.Module):
    """Simple MLP that outputs a scalar state-value estimate."""
    def __init__(self, obs_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


# ── Plotting ─────────────────────────────────────────────────────

def smooth(values, window=5):
    """Trailing moving average."""
    if len(values) < window:
        return values
    return [np.mean(values[max(0, i - window + 1):i + 1]) for i in range(len(values))]


def plot_comparison(results, title, save_path=None):
    """Plot per-iteration mean return for all algorithms on one figure."""
    plt.figure(figsize=(10, 5))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    for i, (name, returns) in enumerate(results.items()):
        c = colors[i % len(colors)]
        plt.plot(returns, alpha=0.2, color=c)
        plt.plot(smooth(returns), linewidth=2, color=c, label=name)
    plt.xlabel("Iteration")
    plt.ylabel("Mean Episode Return")
    plt.title(title)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"  Plot saved to {save_path}")
    plt.show()
    plt.close()
