"""
RL infrastructure: networks, rollout collection, plotting.
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


# ── Rollout Collection ───────────────────────────────────────────

def collect_rollouts(env, policy, value_net=None, num_steps=2048):
    """Collect `num_steps` of experience from the environment."""
    obs_list, act_list, logp_list, rew_list, done_list, val_list = [], [], [], [], [], []
    ep_returns = []
    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=DEVICE)

    for _ in range(num_steps):
        with torch.no_grad():
            action, log_prob = policy.get_action(obs)
            value = value_net(obs) if value_net is not None else torch.tensor(0.0)

        next_obs, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated

        obs_list.append(obs)
        act_list.append(action)
        logp_list.append(log_prob)
        rew_list.append(torch.tensor(reward, dtype=torch.float32, device=DEVICE))
        done_list.append(torch.tensor(float(done), dtype=torch.float32, device=DEVICE))
        val_list.append(value)

        if done:
            if "episode" in info:
                ep_returns.append(float(info["episode"]["r"]))
            next_obs, _ = env.reset()

        obs = torch.tensor(next_obs, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        last_value = value_net(obs) if value_net is not None else torch.tensor(0.0)

    return {
        "obs": torch.stack(obs_list),
        "actions": torch.stack(act_list),
        "log_probs": torch.stack(logp_list),
        "rewards": torch.stack(rew_list),
        "dones": torch.stack(done_list),
        "values": torch.stack(val_list),
        "last_value": last_value,
        "ep_returns": ep_returns,
    }


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
