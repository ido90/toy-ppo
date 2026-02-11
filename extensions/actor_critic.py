"""
Actor-Critic (optional extension)
=================================

Actor-Critic improves on REINFORCE by replacing the raw return G_t
with an advantage:  A_t = G_t - V(s_t)

The learned value function V(s) acts as a baseline, reducing variance
of the policy gradient without introducing bias.

Usage (in main.py __main__ block):
    from extensions.actor_critic import run_actor_critic
    results["Actor-Critic"] = run_actor_critic(env_id)
"""

import numpy as np
import torch
import torch.nn.functional as F

from rl_utils import (
    make_env, PolicyNetwork, ValueNetwork, collect_rollouts, DEVICE,
)
from main import compute_returns, NUM_ITERATIONS, NUM_STEPS, GAMMA, LR


def run_actor_critic(env_id):
    """Train a policy with Actor-Critic.

    Two networks:
    - Policy (actor):  π(a|s) — chooses actions
    - Value  (critic): V(s)   — estimates expected return from state s

    The advantage A_t = G_t - V(s_t) tells us how much better an action
    was compared to the average, reducing gradient variance.
    """
    print(f"\n  Training Actor-Critic on {env_id}...")
    env = make_env(env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy = PolicyNetwork(obs_dim, act_dim).to(DEVICE)
    value_net = ValueNetwork(obs_dim).to(DEVICE)
    policy_opt = torch.optim.Adam(policy.parameters(), lr=LR)
    value_opt = torch.optim.Adam(value_net.parameters(), lr=LR)

    iter_returns = []
    for i in range(NUM_ITERATIONS):
        rollout = collect_rollouts(env, policy, value_net, num_steps=NUM_STEPS)

        # Recompute log_probs and values with gradients
        log_probs = policy.get_log_prob(rollout["obs"], rollout["actions"])
        values = value_net(rollout["obs"])
        returns = compute_returns(rollout["rewards"], rollout["dones"])

        # Advantage = how much better than average
        advantages = returns - values.detach()

        # Policy loss: maximize E[log π(a|s) · A_t]
        policy_loss = -(log_probs * advantages).mean()
        policy_opt.zero_grad()
        policy_loss.backward()
        policy_opt.step()

        # Value loss: minimize (V(s) - G_t)^2
        value_loss = F.mse_loss(values, returns.detach())
        value_opt.zero_grad()
        value_loss.backward()
        value_opt.step()

        mean_ret = np.mean(rollout["ep_returns"]) if rollout["ep_returns"] else 0.0
        iter_returns.append(mean_ret)
        if (i + 1) % 20 == 0:
            print(f"    iter {i+1:3d} | Return: {mean_ret:.1f}")

    env.close()

    # Save trained policy
    import os
    from main import CHECKPOINT_DIR
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = f"{CHECKPOINT_DIR}/Actor-Critic_{env_id}.pt"
    torch.save({"obs_dim": int(obs_dim), "act_dim": int(act_dim),
                "policy_state_dict": policy.state_dict()}, path)
    print(f"    Saved checkpoint: {path}")

    return iter_returns


# ═══════════════════════════════════════════════════════════════════
#  MAIN — run all three methods together
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time
    from main import run_reinforce, run_ppo, ENV_IDS, DEVICE, NUM_ITERATIONS, NUM_STEPS, GAMMA, LR
    from rl_utils import plot_comparison

    print(f"Device: {DEVICE}")
    print(f"Config: {NUM_ITERATIONS} iterations × {NUM_STEPS} steps, γ={GAMMA}, lr={LR}")

    for env_id in ENV_IDS:
        print(f"\n{'='*60}")
        print(f"  Environment: {env_id}")
        print(f"{'='*60}")

        results = {}
        t0 = time.time()
        results["REINFORCE"] = run_reinforce(env_id)
        results["Actor-Critic"] = run_actor_critic(env_id)
        results["PPO"] = run_ppo(env_id)
        elapsed = time.time() - t0

        print(f"\n  Done ({elapsed:.0f}s). Plotting...")
        plot_comparison(results, f"All Methods — {env_id}",
                        save_path=f"plots/{env_id}_with_ac.png")

    print("\nAll done!")
