"""
Deep RL: From REINFORCE to PPO
==============================

This file contains the core algorithm logic for two policy gradient methods:
  1. REINFORCE  — vanilla policy gradient with Monte-Carlo returns
  2. PPO        — Proximal Policy Optimization with GAE, clipped surrogate, minibatch updates

Run:
    python main.py

Student tasks (5 TODOs):
    Implement the functions marked with TODO (search for "raise NotImplementedError").
    To check your implementations:  python tester.py
    To use reference solutions:     set USE_SOLUTIONS = True below.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_utils import (
    make_env, PolicyNetwork, ValueNetwork,
    plot_comparison, DEVICE,
)

CHECKPOINT_DIR = "checkpoints"

# Set to True to import reference solutions instead of using student code.
USE_SOLUTIONS = False

# ═══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

ENV_IDS = ["CartPole-v1", "LunarLander-v3"]

NUM_ITERATIONS = 100       # training iterations (each collects a full rollout)
NUM_STEPS      = 4096      # environment steps per rollout
GAMMA          = 1.0       # discount factor
LR             = 1e-3      # learning rate (REINFORCE)

# PPO-specific
PPO_LR             = LR / 10   # PPO uses a smaller learning rate
PPO_CLIP_EPS       = 0.2       # clipping range for the surrogate objective
PPO_UPDATE_EPOCHS  = 10        # passes over the rollout data per iteration
PPO_MINIBATCH_SIZE = 256       # minibatch size for PPO updates


# ═══════════════════════════════════════════════════════════════════
#  ROLLOUT COLLECTION
# ═══════════════════════════════════════════════════════════════════

def collect_rollout_step(obs, env, policy, value_net):
    """Execute one step of environment interaction.

    This is the core RL data-collection loop body. Unlike supervised learning,
    the agent generates its own training data by acting in the environment.
    See collect_rollouts() below for the loop that wraps this call.

    Args:
        obs:        current observation (as a tensor)
        env:        Gymnasium environment instance
        policy:     PolicyNetwork — call policy.get_action(obs) → (action, log_prob)
        value_net:  ValueNetwork or None — call value_net(obs) → scalar value

    Returns:
        action:    the action taken — scalar tensor (i.e. shape ())
        log_prob:  log π(action|obs) — scalar tensor
        value:     V(obs) estimate — scalar tensor (or 0 if value_net is None)
        reward:    reward received — float
        done:      whether the episode ended — bool
        info:      info dict received from env.step() — dict
        next_obs:  next observation — numpy array — after env.reset() if done
    """
    # ──────────────────────────────────────────────────────────────
    # TODO 1: Collect one environment step.  (8-10 lines)
    #
    # INTERFACES:
    # You will need the following interfaces to complete this function:
    # 1. Gymnasium (Gym) to interact with the environment:
    #    - first_observation, info = env.reset()  # reset the environment once an episode is done
    #    - next_observation, reward, terminated, truncated, info = env.step(action)  # simulate the next env step
    #    - done = terminated or truncated  # the episode is done if the time runs out or a terminal state is reached
    #    - For more details: https://gymnasium.farama.org/index.html
    # 2. The neural networks defined in rl_utils.py:
    #    - action, log_prob = policy.get_action(observation)  # sample an action from the policy
    #    - value = value_net(observation)  # estimate value of state
    #
    # TASKS:
    # 1. Within a torch.no_grad() block:
    #     - Use the policy network to sample an action
    #     - Use the value network to estimate the state value
    #       - If the value network is None (e.g. REINFORCE), use a zero tensor
    # 2. Simulate an environment step using the Gymnasium interface
    # 3. If done: reset the environment
    # 4. Return: action, log_prob, value, reward, done, info, next_obs
    # ──────────────────────────────────────────────────────────────
    raise NotImplementedError("TODO 1: Implement collect_rollout_step")


def collect_rollouts(env, policy, value_net=None, num_steps=2048):
    """Collect a rollout of `num_steps` transitions.

    A single rollout typically spans multiple episodes: when an episode ends
    (done=True), the environment resets and collection continues. The rollout
    may also end mid-episode; in that case, we return `last_value` as an estimte
    for the remaining return (i.e., sum of rewards that would have been collected
    if the episode continued).
    """
    obs_list, act_list, logp_list, rew_list, done_list, val_list = [], [], [], [], [], []
    ep_returns = []
    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=DEVICE)

    for _ in range(num_steps):
        action, log_prob, value, reward, done, info, next_obs = \
            collect_rollout_step(obs, env, policy, value_net)

        obs_list.append(obs)
        act_list.append(action)
        logp_list.append(log_prob)
        rew_list.append(torch.tensor(reward, dtype=torch.float32, device=DEVICE))
        done_list.append(torch.tensor(float(done), dtype=torch.float32, device=DEVICE))
        val_list.append(value)

        if done and "episode" in info:
            ep_returns.append(float(info["episode"]["r"]))

        obs = torch.tensor(next_obs, dtype=torch.float32, device=DEVICE)

    # Bootstrap value for the last observation (needed by GAE when rollout ends mid-episode)
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


# ═══════════════════════════════════════════════════════════════════
#  COMPUTING RETURNS AND ADVANTAGES
# ═══════════════════════════════════════════════════════════════════

def compute_returns(rewards, dones, gamma=GAMMA):
    """Compute discounted returns (reward-to-go) for each timestep.

    G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...
    Resets to 0 at episode boundaries.

    Args:
        rewards: tensor of shape (T,) — reward at each timestep
        dones:   tensor of shape (T,) — 1.0 if episode ended at that step, else 0.0
        gamma:   discount factor

    Returns:
        returns: tensor of shape (T,) — discounted return G_t for each timestep
    """
    # ──────────────────────────────────────────────────────────────
    # TODO 2: Implement discounted returns.  (5-9 lines)
    # ──────────────────────────────────────────────────────────────
    raise NotImplementedError("TODO 2: Implement compute_returns")


def compute_gae(rewards, values, dones, last_value, gamma=GAMMA, gae_lambda=1.0):
    """Generalized Advantage Estimation (GAE).

    Computes advantages using a mix of TD residuals, controlled by gae_lambda.
    With gae_lambda=1.0 this reduces to Monte-Carlo advantages (G_t - V(s_t)),
    but with proper bootstrapping at rollout boundaries.

    Returns:
        advantages: A_t for the policy gradient
        returns:    V-targets for the value function loss
    """
    T = len(rewards)
    advantages = torch.zeros(T, device=rewards.device)
    last_gae = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = last_value
        else:
            next_value = values[t + 1]
        next_non_terminal = 1.0 - dones[t]

        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values.detach()
    return advantages.detach(), returns


def compute_reinforce_loss(log_probs, returns):
    """REINFORCE policy gradient loss.

    The policy gradient theorem gives:  ∇J ≈ E[ ∇log π(a|s) · G_t ]
    We minimize the negative of this (since optimizers minimize).

    Args:
        log_probs: tensor of shape (T,) — log π(a_t | s_t)
        returns:   tensor of shape (T,) — discounted return G_t

    Returns:
        loss: scalar tensor — the policy gradient loss
    """
    # ──────────────────────────────────────────────────────────────
    # TODO 3: Implement the REINFORCE loss.  (1-2 lines)
    # ──────────────────────────────────────────────────────────────
    raise NotImplementedError("TODO 3: Implement compute_reinforce_loss")


def compute_value_loss(value_net, obs, returns):
    """Value function loss: how well does V(s) predict the actual returns?

    Args:
        value_net: ValueNetwork — call value_net(obs) to get V(s) predictions
        obs:       tensor of shape (B, obs_dim) — observations
        returns:   tensor of shape (B,) — target returns (from GAE or Monte-Carlo)

    Returns:
        loss: scalar tensor — mean squared error between predictions and targets
    """
    # ──────────────────────────────────────────────────────────────
    # TODO 4: Implement the value function loss.  (1-2 lines)
    # ──────────────────────────────────────────────────────────────
    raise NotImplementedError("TODO 4: Implement compute_value_loss")


def compute_ppo_loss(old_log_probs, new_log_probs, advantages, clip_eps=PPO_CLIP_EPS):
    """PPO clipped surrogate objective.

    Clips the policy ratio to [1 - eps, 1 + eps] to prevent
    destructively large policy updates.

    Args:
        old_log_probs: tensor of shape (B,) — log π_old(a|s) from the rollout
        new_log_probs: tensor of shape (B,) — log π_new(a|s) from current policy
        advantages:    tensor of shape (B,) — advantage estimates A_t
        clip_eps:      clipping range (default 0.2)

    Returns:
        loss: scalar tensor — the clipped surrogate loss
    """
    # ──────────────────────────────────────────────────────────────
    # TODO 5: Implement the PPO clipped surrogate loss.  (3-4 lines)
    # ──────────────────────────────────────────────────────────────
    raise NotImplementedError("TODO 5: Implement compute_ppo_loss")


# ── Override with reference solutions if enabled ─────────────────
if USE_SOLUTIONS:
    from solutions import (
        collect_rollout_step, compute_returns,
        compute_reinforce_loss, compute_value_loss, compute_ppo_loss,
    )
    print("(Using reference solutions)")


# ═══════════════════════════════════════════════════════════════════
#  REINFORCE
# ═══════════════════════════════════════════════════════════════════

def run_reinforce(env_id):
    """Train a policy with REINFORCE (vanilla policy gradient).

    The policy gradient is:  ∇J ≈ E[ ∇log π(a|s) · G_t ]
    where G_t is the discounted return from timestep t.
    """
    print(f"\n  Training REINFORCE on {env_id}...")
    env = make_env(env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy = PolicyNetwork(obs_dim, act_dim).to(DEVICE)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

    iter_returns = []
    for i in range(NUM_ITERATIONS):
        rollout = collect_rollouts(env, policy, num_steps=NUM_STEPS)

        # Recompute log_probs with gradients (rollout ones are detached)
        log_probs = policy.get_log_prob(rollout["obs"], rollout["actions"])
        returns = compute_returns(rollout["rewards"], rollout["dones"])
        loss = compute_reinforce_loss(log_probs, returns)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mean_ret = np.mean(rollout["ep_returns"]) if rollout["ep_returns"] else 0.0
        iter_returns.append(mean_ret)
        if (i + 1) % 20 == 0:
            print(f"    iter {i+1:3d} | Return: {mean_ret:.1f}")

    env.close()

    # Save trained policy
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = f"{CHECKPOINT_DIR}/REINFORCE_{env_id}.pt"
    torch.save({"obs_dim": int(obs_dim), "act_dim": int(act_dim),
                "policy_state_dict": policy.state_dict()}, path)
    print(f"    Saved checkpoint: {path}")

    return iter_returns


# ═══════════════════════════════════════════════════════════════════
#  PPO
# ═══════════════════════════════════════════════════════════════════

def run_ppo(env_id):
    """Train a policy with Proximal Policy Optimization (PPO).

    Key differences from REINFORCE:
    - Uses a learned value function V(s) as a variance-reducing baseline
    - Clips the policy update ratio to prevent destructive updates
    - Reuses each rollout for multiple gradient steps (epochs x minibatches)
    """
    print(f"\n  Training PPO on {env_id}...")
    env = make_env(env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy = PolicyNetwork(obs_dim, act_dim).to(DEVICE)
    value_net = ValueNetwork(obs_dim).to(DEVICE)
    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(value_net.parameters()), lr=PPO_LR,
    )

    iter_returns = []
    for i in range(NUM_ITERATIONS):
        rollout = collect_rollouts(env, policy, value_net, num_steps=NUM_STEPS)
        mean_ret = np.mean(rollout["ep_returns"]) if rollout["ep_returns"] else 0.0
        iter_returns.append(mean_ret)

        # Compute advantages using GAE
        advantages, returns = compute_gae(
            rollout["rewards"], rollout["values"], rollout["dones"],
            rollout["last_value"],
        )
        # Normalize advantages (standard practice for PPO)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        b_obs = rollout["obs"]
        b_actions = rollout["actions"]
        b_old_log_probs = rollout["log_probs"].detach()

        # Multiple epochs of minibatch updates on the same rollout
        batch_size = len(b_obs)
        for epoch in range(PPO_UPDATE_EPOCHS):
            indices = torch.randperm(batch_size)
            for start in range(0, batch_size, PPO_MINIBATCH_SIZE):
                mb_idx = indices[start : start + PPO_MINIBATCH_SIZE]

                # policy loss
                new_log_probs = policy.get_log_prob(b_obs[mb_idx], b_actions[mb_idx])
                policy_loss = compute_ppo_loss(
                    b_old_log_probs[mb_idx], new_log_probs, advantages[mb_idx],
                )

                # value loss
                value_loss = compute_value_loss(value_net, b_obs[mb_idx], returns[mb_idx])

                # total loss
                loss = policy_loss + 0.5 * value_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(policy.parameters()) + list(value_net.parameters()), 0.5,
                )
                optimizer.step()

        if (i + 1) % 20 == 0:
            print(f"    iter {i+1:3d} | Return: {mean_ret:.1f}")

    env.close()

    # Save trained policy
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = f"{CHECKPOINT_DIR}/PPO_{env_id}.pt"
    torch.save({"obs_dim": int(obs_dim), "act_dim": int(act_dim),
                "policy_state_dict": policy.state_dict()}, path)
    print(f"    Saved checkpoint: {path}")

    return iter_returns


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Config: {NUM_ITERATIONS} iterations × {NUM_STEPS} steps, γ={GAMMA}, lr={LR}")

    for env_id in ENV_IDS:
        print(f"\n{'='*60}")
        print(f"  Environment: {env_id}")
        print(f"{'='*60}")

        results = {}

        t0 = time.time()
        results["REINFORCE"] = run_reinforce(env_id)
        results["PPO"] = run_ppo(env_id)
        elapsed = time.time() - t0
        # To also run Actor-Critic:  python -m extensions.actor_critic

        print(f"\n  Done ({elapsed:.0f}s). Plotting...")
        plot_comparison(results, f"REINFORCE vs PPO — {env_id}",
                        save_path=f"plots/{env_id}.png")


    print("\nAll done!")
