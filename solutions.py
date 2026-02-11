"""
Reference solutions for the TODO functions in main.py.

To use:  set USE_SOLUTIONS = True in main.py
To test: python tester.py
"""

import torch


def compute_returns(rewards, dones, gamma=1.0):
    """Compute discounted returns (reward-to-go) for each timestep."""
    T = len(rewards)
    returns = torch.zeros(T, device=rewards.device)
    future_return = 0.0
    for t in reversed(range(T)):
        if dones[t]:
            future_return = 0.0
        future_return = rewards[t] + gamma * future_return
        returns[t] = future_return
    return returns


def compute_reinforce_loss(log_probs, returns):
    """REINFORCE policy gradient loss."""
    return -(log_probs * returns).mean()


def compute_ppo_loss(old_log_probs, new_log_probs, advantages, clip_eps=0.2):
    """PPO clipped surrogate objective."""
    ratio = (new_log_probs - old_log_probs).exp()
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    return loss
