# Toy PPO

A hands-on deep RL exercise — Policy Gradient from REINFORCE to PPO.

The main training pipeline is implemented compactly in `main.py`. Students are expected to read through the code, understand the end-to-end flow, and implement three small core functions.

This repo was written by Ido Greenberg for the course [RL-for-Real](https://docs.google.com/document/d/1fmfYp7EH9fqcB7CWWBvrZ40MtCN89Sr_o3o3EG9hWyE), organized by NVIDIA Research in collaboration with Google Research, Mentee Robotics, Tel-Aviv University, Bar-Ilan University, and the Technion.

## What you'll implement

| TODO | Function | What it does |
|------|----------|-------------|
| 1 | `compute_returns` | Compute future return from each timestep |
| 2 | `compute_reinforce_loss` | Vanilla policy gradient loss |
| 3 | `compute_ppo_loss` | PPO's clipped surrogate objective |

Everything else (networks, rollout collection, training loops, plotting) is provided so you can read and trace the full pipeline.

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

**2. Implement the TODOs** — fill in the three functions listed above.

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

If you get stuck, set `USE_SOLUTIONS = True` in `main.py` to import the reference implementations.

## File structure

```
main.py              — exercise file: config, algorithms, training (start here)
rl_utils.py          — infrastructure: networks, rollout collection, plotting
solutions.py         — reference solutions for the TODOs
tester.py            — unit tests for your implementations
visualize.py         — render a trained/random agent as a GIF
environment.yml      — conda environment specification
extensions/
  actor_critic.py    — optional: Actor-Critic method
```

## Environments

- [**CartPole-v1**](https://gymnasium.farama.org/environments/classic_control/cart_pole/) — balance a pole on a cart (easy, fast training)
- [**LunarLander-v3**](https://gymnasium.farama.org/environments/box2d/lunar_lander/) — land a spacecraft (harder, clearer algorithm differentiation)

## Optional extension

Run all three methods (REINFORCE, Actor-Critic, PPO) together:

```bash
python -m extensions.actor_critic
```
