"""
Visualize a trained (or random) agent and save as a GIF.

Usage:
    python visualize.py                          # PPO on CartPole (default)
    python visualize.py --agent random           # random agent
    python visualize.py --agent REINFORCE        # trained REINFORCE agent
    python visualize.py --env LunarLander-v3     # different environment

Requires: pip install gymnasium[classic-control]  (for rendering)
          pip install gymnasium[box2d]            (for LunarLander)
"""

import argparse
import gymnasium as gym
import torch
import numpy as np
from PIL import Image

from rl_utils import PolicyNetwork, DEVICE


def load_policy(agent, env_id):
    """Load a trained policy from checkpoint, or return None for random."""
    if agent == "random":
        return None

    path = f"checkpoints/{agent}_{env_id}.pt"
    try:
        ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    except FileNotFoundError:
        print(f"  Checkpoint not found: {path}")
        print(f"  Run 'python main.py' first to train agents.")
        raise SystemExit(1)

    policy = PolicyNetwork(ckpt["obs_dim"], ckpt["act_dim"]).to(DEVICE)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    print(f"  Loaded: {path}")
    return policy


def collect_frames(env, policy=None, max_steps=500):
    """Run one episode, return (frames, total_reward)."""
    obs, _ = env.reset()
    frames = []
    total_reward = 0.0

    for _ in range(max_steps):
        frames.append(env.render())

        if policy is None:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
                action, _ = policy.get_action(obs_t)
                action = action.item()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    return frames, total_reward


def save_gif(frames, path, fps=30):
    """Save list of RGB arrays as an animated GIF."""
    images = [Image.fromarray(f) for f in frames]
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=1000 // fps, loop=0)
    print(f"  Saved: {path}  ({len(frames)} frames)")


def main():
    parser = argparse.ArgumentParser(description="Visualize an RL agent")
    parser.add_argument("--env", default="CartPole-v1",
                        help="Gymnasium environment ID")
    parser.add_argument("--agent", default="random",
                        choices=["random", "REINFORCE", "PPO", "Actor-Critic"],
                        help="Which agent to visualize")
    parser.add_argument("--out", default=None,
                        help="Output GIF path (default: plots/{agent}_{env}.gif)")
    args = parser.parse_args()

    print(f"  Env: {args.env}  |  Agent: {args.agent}")

    policy = load_policy(args.agent, args.env)
    env = gym.make(args.env, render_mode="rgb_array")

    frames, ret = collect_frames(env, policy)
    env.close()
    print(f"  Episode return: {ret:.1f}  ({len(frames)} steps)")

    out_path = args.out or f"plots/{args.agent}_{args.env}.gif"
    save_gif(frames, out_path)


if __name__ == "__main__":
    main()
