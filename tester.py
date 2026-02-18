"""
Test your TODO implementations without running the full training pipeline.

Usage:
    python tester.py

Tests each function independently with small, deterministic inputs.
"""

import torch
import sys

from main import (
    collect_rollout_step, compute_returns, compute_reinforce_loss,
    compute_value_loss, compute_ppo_loss, GAMMA, DEVICE,
)
from rl_utils import make_env, PolicyNetwork, ValueNetwork


def test_collect_rollout_step():
    """Test collect_rollout_step with a real environment."""
    print("Testing collect_rollout_step...")

    env = make_env("CartPole-v1", seed=0)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    policy = PolicyNetwork(obs_dim, act_dim).to(DEVICE)
    value_net = ValueNetwork(obs_dim).to(DEVICE)

    obs, _ = env.reset(seed=0)
    obs = torch.tensor(obs, dtype=torch.float32, device=DEVICE)

    # ── Test 1: returns correct types ──────────────────────────────
    action, log_prob, value, reward, done, info, next_obs = \
        collect_rollout_step(obs, env, policy, value_net)

    if not isinstance(action, torch.Tensor):
        print(f"  FAILED: action should be a tensor, got {type(action).__name__}")
        return False
    if not isinstance(log_prob, torch.Tensor):
        print(f"  FAILED: log_prob should be a tensor, got {type(log_prob).__name__}")
        return False
    if not isinstance(value, torch.Tensor):
        print(f"  FAILED: value should be a tensor, got {type(value).__name__}")
        return False
    if not isinstance(reward, (int, float)):
        print(f"  FAILED: reward should be a number, got {type(reward).__name__}")
        return False
    if not isinstance(done, bool):
        print(f"  FAILED: done should be a bool, got {type(done).__name__}")
        return False
    import numpy as _np
    if not isinstance(next_obs, _np.ndarray):
        print(f"  FAILED: next_obs should be a numpy array, got {type(next_obs).__name__}")
        return False
    print("  return types ................ OK")

    # ── Test 2: shapes are correct ─────────────────────────────────
    if action.shape != ():
        print(f"  FAILED: action should be scalar, got shape {action.shape}")
        return False
    if log_prob.shape != ():
        print(f"  FAILED: log_prob should be scalar, got shape {log_prob.shape}")
        return False
    if next_obs.shape[0] != obs.shape[0]:
        print(f"  FAILED: next_obs dim {next_obs.shape[0]} != obs dim {obs.shape[0]}")
        return False
    print("  shapes ...................... OK")

    # ── Test 3: log_prob is negative (valid probability) ───────────
    if log_prob.item() > 0:
        print(f"  FAILED: log_prob should be <= 0, got {log_prob.item():.4f}")
        return False
    print("  log_prob is negative ........ OK")

    # ── Test 4: value_net=None returns zero value ──────────────────
    obs2, _ = env.reset(seed=0)
    obs2 = torch.tensor(obs2, dtype=torch.float32, device=DEVICE)
    _, _, value_none, _, _, _, _ = collect_rollout_step(obs2, env, policy, None)
    if value_none.item() != 0.0:
        print(f"  FAILED: value should be 0 when value_net=None, got {value_none.item()}")
        return False
    print("  value_net=None → zero ....... OK")

    # ── Test 5: runs multiple steps without error ──────────────────
    obs3, _ = env.reset(seed=0)
    obs3 = torch.tensor(obs3, dtype=torch.float32, device=DEVICE)
    for _ in range(50):
        _, _, _, _, _, _, next_obs3 = collect_rollout_step(obs3, env, policy, value_net)
        obs3 = torch.tensor(next_obs3, dtype=torch.float32, device=DEVICE)
    print("  50 consecutive steps ........ OK")

    env.close()
    print("  PASSED")
    return True


def test_compute_returns():
    """Test compute_returns with known examples."""
    print("Testing compute_returns...")

    # ── Test 1: gamma=1, two episodes ────────────────────────────
    rewards = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
    dones   = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0])
    expected = torch.tensor([4.0, 3.0, 2.0, 1.0, 1.0])
    result = compute_returns(rewards, dones, gamma=1.0)

    if not torch.allclose(result, expected, atol=1e-5):
        print(f"  FAILED (gamma=1.0)")
        print(f"    Expected: {expected.tolist()}")
        print(f"    Got:      {result.tolist()}")
        return False
    print("  gamma=1.0 basic ............. OK")

    # ── Test 2: gamma=0.5, same episodes ─────────────────────────
    expected_disc = torch.tensor([1.875, 1.75, 1.5, 1.0, 1.0])
    result_disc = compute_returns(rewards, dones, gamma=0.5)

    if not torch.allclose(result_disc, expected_disc, atol=1e-5):
        print(f"  FAILED (gamma=0.5)")
        print(f"    Expected: {expected_disc.tolist()}")
        print(f"    Got:      {result_disc.tolist()}")
        return False
    print("  gamma=0.5 discounting ....... OK")

    # ── Test 3: episode boundary isolation ────────────────────────
    rewards2 = torch.tensor([10.0, 1.0, 1.0, 1.0])
    dones2   = torch.tensor([1.0,  0.0, 0.0, 0.0])
    expected2 = torch.tensor([10.0, 2.71, 1.9, 1.0])
    result2 = compute_returns(rewards2, dones2, gamma=0.9)

    if not torch.allclose(result2, expected2, atol=1e-5):
        print(f"  FAILED (episode boundary isolation)")
        print(f"    Expected: {expected2.tolist()}")
        print(f"    Got:      {result2.tolist()}")
        return False
    print("  episode boundary isolation .. OK")

    print("  PASSED")
    return True


def test_compute_reinforce_loss():
    """Test compute_reinforce_loss with known examples."""
    print("Testing compute_reinforce_loss...")

    # ── Test 1: correct value ─────────────────────────────────────
    log_probs = torch.tensor([-0.5, -1.0, -0.2])
    returns   = torch.tensor([3.0,  2.0,  1.0])
    expected = -(-0.5 * 3.0 + -1.0 * 2.0 + -0.2 * 1.0) / 3.0
    result = compute_reinforce_loss(log_probs, returns)

    if not torch.allclose(result, torch.tensor(expected), atol=1e-4):
        print(f"  FAILED (wrong value)")
        print(f"    Expected: {expected:.4f}")
        print(f"    Got:      {result.item():.4f}")
        return False
    if result.dim() != 0:
        print(f"  FAILED (result should be a scalar tensor, got dim={result.dim()})")
        return False
    print("  correct value ............... OK")

    # ── Test 2: gradients flow ────────────────────────────────────
    log_probs_g = torch.tensor([-0.5, -1.0], requires_grad=True)
    returns_g   = torch.tensor([2.0, 1.0])
    loss = compute_reinforce_loss(log_probs_g, returns_g)
    loss.backward()

    if log_probs_g.grad is None:
        print(f"  FAILED (no gradient on log_probs — did you detach it?)")
        return False
    print("  gradients flow .............. OK")

    print("  PASSED")
    return True


def test_compute_value_loss():
    """Test compute_value_loss with known examples."""
    print("Testing compute_value_loss...")

    # Build a tiny value network for testing
    vnet = ValueNetwork(obs_dim=2).to(DEVICE)

    # ── Test 1: returns a scalar loss ──────────────────────────────
    obs = torch.randn(4, 2, device=DEVICE)
    returns = torch.tensor([1.0, 2.0, 3.0, 4.0], device=DEVICE)
    result = compute_value_loss(vnet, obs, returns)

    if result.dim() != 0:
        print(f"  FAILED (result should be a scalar, got dim={result.dim()})")
        return False
    if result.item() < 0:
        print(f"  FAILED (MSE loss should be non-negative, got {result.item():.4f})")
        return False
    print("  returns scalar loss ......... OK")

    # ── Test 2: zero loss when predictions match ──────────────────
    with torch.no_grad():
        perfect_returns = vnet(obs)
    result_zero = compute_value_loss(vnet, obs, perfect_returns)

    if not torch.allclose(result_zero, torch.tensor(0.0), atol=1e-5):
        print(f"  FAILED (should be ~0 for perfect targets, got {result_zero.item():.6f})")
        return False
    print("  zero loss when perfect ...... OK")

    # ── Test 3: gradients flow to value network ───────────────────
    obs_g = torch.randn(3, 2, device=DEVICE)
    returns_g = torch.tensor([5.0, 5.0, 5.0], device=DEVICE)
    loss = compute_value_loss(vnet, obs_g, returns_g)
    loss.backward()

    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in vnet.parameters())
    if not has_grad:
        print(f"  FAILED (no gradient flowing to value network)")
        return False
    print("  gradients flow to network ... OK")

    print("  PASSED")
    return True


def test_compute_ppo_loss():
    """Test compute_ppo_loss with known examples."""
    print("Testing compute_ppo_loss...")

    # ── Test 1: ratio=1 (same policy), no clipping ───────────────
    old_log_probs = torch.tensor([-1.0, -1.0, -1.0])
    new_log_probs = torch.tensor([-1.0, -1.0, -1.0])
    advantages    = torch.tensor([1.0,  -1.0,  0.5])
    expected = -advantages.mean()
    result = compute_ppo_loss(old_log_probs, new_log_probs, advantages, clip_eps=0.2)

    if not torch.allclose(result, expected, atol=1e-4):
        print(f"  FAILED (ratio=1, no clipping)")
        print(f"    Expected: {expected.item():.4f}")
        print(f"    Got:      {result.item():.4f}")
        return False
    print("  ratio=1, no clipping ....... OK")

    # ── Test 2: ratio within bounds, no clipping ─────────────────
    old_lp = torch.tensor([-1.0])
    new_lp = torch.tensor([-0.9])
    adv    = torch.tensor([2.0])
    ratio = (new_lp - old_lp).exp()
    expected_noclip = -(ratio * adv).mean()
    result_noclip = compute_ppo_loss(old_lp, new_lp, adv, clip_eps=0.2)

    if not torch.allclose(result_noclip, expected_noclip, atol=1e-4):
        print(f"  FAILED (within bounds, should not clip)")
        print(f"    Expected: {expected_noclip.item():.4f}")
        print(f"    Got:      {result_noclip.item():.4f}")
        return False
    print("  within bounds, no clip ..... OK")

    # ── Test 3: high ratio + positive advantage → clip at 1+eps ──
    old_lp = torch.tensor([-2.0])
    new_lp = torch.tensor([-0.5])
    adv    = torch.tensor([1.0])
    result_clip_high = compute_ppo_loss(old_lp, new_lp, adv, clip_eps=0.2)
    expected_clip_high = torch.tensor(-1.2)

    if not torch.allclose(result_clip_high, expected_clip_high, atol=1e-4):
        print(f"  FAILED (high ratio, positive advantage)")
        print(f"    Expected: {expected_clip_high.item():.4f}")
        print(f"    Got:      {result_clip_high.item():.4f}")
        return False
    print("  clip high ratio + pos adv .. OK")

    # ── Test 4: low ratio + negative advantage → clip at 1-eps ───
    old_lp = torch.tensor([-0.5])
    new_lp = torch.tensor([-2.0])
    adv    = torch.tensor([-1.0])
    result_clip_low = compute_ppo_loss(old_lp, new_lp, adv, clip_eps=0.2)
    expected_clip_low = torch.tensor(0.8)

    if not torch.allclose(result_clip_low, expected_clip_low, atol=1e-4):
        print(f"  FAILED (low ratio, negative advantage)")
        print(f"    Expected: {expected_clip_low.item():.4f}")
        print(f"    Got:      {result_clip_low.item():.4f}")
        return False
    print("  clip low ratio + neg adv ... OK")

    print("  PASSED")
    return True


if __name__ == "__main__":
    print("=" * 50)
    print("  Testing your TODO implementations")
    print("=" * 50)
    print()

    tests = [
        ("collect_rollout_step",  test_collect_rollout_step),
        ("compute_returns",       test_compute_returns),
        ("compute_reinforce_loss", test_compute_reinforce_loss),
        ("compute_value_loss",    test_compute_value_loss),
        ("compute_ppo_loss",      test_compute_ppo_loss),
    ]

    passed = 0
    failed = 0
    errors = 0

    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
            else:
                failed += 1
        except NotImplementedError as e:
            print(f"NOT IMPLEMENTED: {e}")
            errors += 1
        except Exception as e:
            print(f"ERROR: {e}")
            errors += 1

    print()
    total = len(tests)
    print(f"Results: {passed} passed, {failed} failed, {errors} not implemented (out of {total})")

    if failed > 0:
        sys.exit(1)
    if errors > 0:
        sys.exit(2)

    print("\nAll tests passed! You can now run:  python main.py")
