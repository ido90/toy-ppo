"""
Test your TODO implementations without running the full training pipeline.

Usage:
    python tester.py

Tests each function independently with small, deterministic inputs.
"""

import torch
import sys

# Import student implementations from main.py
# (USE_SOLUTIONS must be False in main.py for this to test student code)
from main import compute_returns, compute_reinforce_loss, compute_ppo_loss, GAMMA


def test_compute_returns():
    """Test compute_returns with known examples."""
    print("Testing compute_returns...")

    # ── Test 1: gamma=1, two episodes ────────────────────────────
    # Episode 1: [1, 1, 1, done]  Episode 2: [1, 1] (not done)
    rewards = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
    dones   = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0])

    # Episode 1 returns = [4, 3, 2, 1],  Episode 2 = [1] (only 1 step before rollout ends)
    expected = torch.tensor([4.0, 3.0, 2.0, 1.0, 1.0])
    result = compute_returns(rewards, dones, gamma=1.0)

    if not torch.allclose(result, expected, atol=1e-5):
        print(f"  FAILED (gamma=1.0)")
        print(f"    Expected: {expected.tolist()}")
        print(f"    Got:      {result.tolist()}")
        return False
    print("  gamma=1.0 basic ............. OK")

    # ── Test 2: gamma=0.5, same episodes ─────────────────────────
    # Episode 1: G_0 = 1 + 0.5 + 0.25 + 0.125 = 1.875
    #            G_1 = 1 + 0.5 + 0.25 = 1.75
    #            G_2 = 1 + 0.5 = 1.5
    #            G_3 = 1.0  (terminal)
    # Episode 2: G_4 = 1.0
    expected_disc = torch.tensor([1.875, 1.75, 1.5, 1.0, 1.0])
    result_disc = compute_returns(rewards, dones, gamma=0.5)

    if not torch.allclose(result_disc, expected_disc, atol=1e-5):
        print(f"  FAILED (gamma=0.5)")
        print(f"    Expected: {expected_disc.tolist()}")
        print(f"    Got:      {result_disc.tolist()}")
        return False
    print("  gamma=0.5 discounting ....... OK")

    # ── Test 3: episode boundary isolation ────────────────────────
    # Two episodes with different rewards: [10, done] [1, 1, 1]
    # Returns should NOT bleed across the boundary.
    rewards2 = torch.tensor([10.0, 1.0, 1.0, 1.0])
    dones2   = torch.tensor([1.0,  0.0, 0.0, 0.0])

    # gamma=0.9: Ep1 = [10], Ep2 = [1 + 0.9 + 0.81, 1 + 0.9, 1] = [2.71, 1.9, 1.0]
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

    # loss = -mean([-1.5, -2.0, -0.2]) = -(-1.2333) = 1.2333
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


def test_compute_ppo_loss():
    """Test compute_ppo_loss with known examples."""
    print("Testing compute_ppo_loss...")

    # ── Test 1: ratio=1 (same policy), no clipping ───────────────
    old_log_probs = torch.tensor([-1.0, -1.0, -1.0])
    new_log_probs = torch.tensor([-1.0, -1.0, -1.0])  # same policy → ratio = 1
    advantages    = torch.tensor([1.0,  -1.0,  0.5])

    # loss = -mean(1 * advantages) = -0.1667
    expected = -advantages.mean()
    result = compute_ppo_loss(old_log_probs, new_log_probs, advantages, clip_eps=0.2)

    if not torch.allclose(result, expected, atol=1e-4):
        print(f"  FAILED (ratio=1, no clipping)")
        print(f"    Expected: {expected.item():.4f}")
        print(f"    Got:      {result.item():.4f}")
        return False
    print("  ratio=1, no clipping ....... OK")

    # ── Test 2: ratio within bounds, no clipping ─────────────────
    # ratio = exp(-0.9 - (-1.0)) = exp(0.1) ≈ 1.105, within [0.8, 1.2]
    old_lp = torch.tensor([-1.0])
    new_lp = torch.tensor([-0.9])
    adv    = torch.tensor([2.0])

    ratio = (new_lp - old_lp).exp()  # ≈ 1.105
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
    new_lp = torch.tensor([-0.5])   # ratio = exp(1.5) ≈ 4.48, way above 1.2
    adv    = torch.tensor([1.0])

    # min(4.48 * 1, 1.2 * 1) = 1.2 → loss = -1.2
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
    new_lp = torch.tensor([-2.0])   # ratio = exp(-1.5) ≈ 0.22, below 0.8
    adv    = torch.tensor([-1.0])

    # ratio*adv = 0.22*(-1) = -0.22,  clipped*adv = 0.8*(-1) = -0.8
    # min(-0.22, -0.8) = -0.8 → loss = -(-0.8) = 0.8
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

    results = []
    results.append(("compute_returns",       test_compute_returns))
    results.append(("compute_reinforce_loss", test_compute_reinforce_loss))
    results.append(("compute_ppo_loss",       test_compute_ppo_loss))

    passed = 0
    failed = 0
    errors = 0

    for name, test_fn in results:
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
    print(f"Results: {passed} passed, {failed} failed, {errors} not implemented")

    if failed > 0:
        sys.exit(1)
    if errors > 0:
        sys.exit(2)

    print("\nAll tests passed! You can now run:  python main.py")
