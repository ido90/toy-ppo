# Toy PPO

A hands-on deep RL exercise — Policy Gradient from REINFORCE to PPO.

The main training pipeline is implemented compactly in `main.py`. Students are expected to read through the code, understand the end-to-end flow, and implement five core functions.

This repo was written by Ido Greenberg for the course [RL-for-Real](https://docs.google.com/document/d/1fmfYp7EH9fqcB7CWWBvrZ40MtCN89Sr_o3o3EG9hWyE), organized by NVIDIA Research in collaboration with Google Research, Mentee Robotics, Tel-Aviv University, Bar-Ilan University, and the Technion.

## What you'll implement

| TODO | Function | What it does | #Lines |
|------|----------|-------------|-------|
| 1 | `collect_rollout_step` | One step of agent-environment interaction | 8–10 |
| 2 | `compute_returns` | Discounted future return from each timestep | 5–9 |
| 3 | `compute_reinforce_loss` | Vanilla policy gradient loss | 1–2 |
| 4 | `compute_value_loss` | Value function regression loss | 1–2 |
| 5 | `compute_ppo_loss` | PPO's clipped surrogate objective | 3–4 |

Everything else (networks, training loops, plotting) is provided so you can read and trace the full pipeline.

## Setup

### 1. Install Miniconda (if needed)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

### 2. Create Environment

```bash
conda env create -f environment.yml
conda activate simple_ppo
```

### 3. Fix library path (needed on most Linux systems)

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

This ensures conda's C++ runtime is used instead of the system's, avoiding `CXXABI` import errors.

## Exercise

**1. Read the code** — review `main.py` and follow the training flow. Note what changes between `run_reinforce()` and `run_ppo()`.

**2. Implement the TODOs** — fill in the five functions listed above.

**3. Test** — verify your implementations with the unit tests (no training needed):

```bash
python tester.py
```

**4. Train** — run REINFORCE and PPO on CartPole and LunarLander:

```bash
python main.py
```

Total expected runtime (for all 4 training runs) is 8-15 minutes (depending on CPU).

**5. Visualize** — render a trained agent as a GIF:

```bash
python visualize.py --agent random   # an untrained agent for reference
python visualize.py --agent PPO --env CartPole-v1
```

Visualize your agent behavior for each task. Does it look like the agent learned well?

You may read about the tasks and their reward functions:
- [**CartPole-v1**](https://gymnasium.farama.org/environments/classic_control/cart_pole/): balance a pole on a cart.
- [**LunarLander-v3**](https://gymnasium.farama.org/environments/box2d/lunar_lander/): land a spacecraft.

### Reference solutions

If you get stuck, set `USE_SOLUTIONS = True` in `main.py` to import the reference implementations.

## Optional extensions

**6. MountainCar** — repeat the experiments for MountainCar-v0 (modify `ENV_IDS` in `main.py`). Does the learning work? Can you guess why? Propose a few possible solutions, then pick one and make it work.

**7. Beat the defaults** — can you learn faster or reach higher returns? Try tuning hyperparameters (learning rate, gamma, network size) or modifying the training loop.

**8. GAE-lambda** — erase and re-implement `compute_gae()` in `main.py`.

**9. Actor-Critic** — experiment with all three methods (REINFORCE, Actor-Critic, PPO) and compare learning curves:

```bash
python -m extensions.actor_critic
```

## File structure

```
main.py              — exercise file: config, algorithms, training (start here)
rl_utils.py          — infrastructure: networks, environment setup, plotting
solutions.py         — reference solutions for the TODOs
tester.py            — unit tests for your implementations
visualize.py         — render a trained/random agent as a GIF
environment.yml      — conda environment specification
extensions/
  actor_critic.py    — optional: Actor-Critic method
```
