import numpy as np
import pytest

from src.simulator import simulate_path, run_monte_carlo
from src.strategies import inventory_quotes, symmetric_quotes
from src.config import GAMMA_SCENARIOS, PAPER_SPREAD_COMPONENTS, K, SIGMA, T, DT, S0, A


# ---------------------------------------------------------------------------
# Helper: default kwargs for simulate_path
# ---------------------------------------------------------------------------
_DEFAULT = dict(sigma=SIGMA, k=K, A=A, T=T, dt=DT, S0=S0, q0=0, X0=0.0)


def _make_rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# 1. Reservation price == mid-price when inventory is zero
# ---------------------------------------------------------------------------
def test_reservation_price_at_zero_inventory():
    s = 100.0
    _, _, _, _, r, _ = inventory_quotes(s=s, q=0, gamma=0.1, sigma=2.0, k=1.5, T=1.0, t=0.5)
    assert r == pytest.approx(s)


# ---------------------------------------------------------------------------
# 2. Reservation price == mid-price at maturity (T - t = 0)
# ---------------------------------------------------------------------------
def test_reservation_price_at_maturity():
    s = 105.0
    for q in [-3, 0, 5]:
        _, _, _, _, r, _ = inventory_quotes(s=s, q=q, gamma=0.1, sigma=2.0, k=1.5, T=1.0, t=1.0)
        assert r == pytest.approx(s), f"q={q}: expected r={s}, got r={r}"


# ---------------------------------------------------------------------------
# 3. Constant spread component matches paper values (within 0.1)
# ---------------------------------------------------------------------------
def test_spread_constants_match_paper():
    for gamma, expected_c in PAPER_SPREAD_COMPONENTS.items():
        _, _, delta_ask, _ = symmetric_quotes(s=100.0, gamma=gamma, k=K)
        c = 2.0 * delta_ask
        assert abs(c - expected_c) < 0.1, (
            f"gamma={gamma}: computed spread={c:.4f}, paper={expected_c}"
        )


# ---------------------------------------------------------------------------
# 4. Cash sign consistency: ask fill → cash up by p^a; bid fill → cash down by p^b
# ---------------------------------------------------------------------------
def test_cash_sign_consistency():
    rng = _make_rng(123)
    path = simulate_path(strategy='inventory', gamma=0.1, **_DEFAULT, rng=rng)
    N = len(path.r)
    for i in range(N):
        expected = path.X[i] + path.ask_fills[i] * path.p_ask[i] - path.bid_fills[i] * path.p_bid[i]
        assert path.X[i + 1] == pytest.approx(expected), (
            f"step {i}: X[{i+1}]={path.X[i+1]:.6f} != expected={expected:.6f}"
        )


# ---------------------------------------------------------------------------
# 5. Inventory sign consistency: ask fill → q−1; bid fill → q+1
# ---------------------------------------------------------------------------
def test_inventory_sign_consistency():
    rng = _make_rng(456)
    path = simulate_path(strategy='symmetric', gamma=0.1, **_DEFAULT, rng=rng)
    N = len(path.r)
    for i in range(N):
        expected = int(path.q[i]) - int(path.ask_fills[i]) + int(path.bid_fills[i])
        assert int(path.q[i + 1]) == expected, (
            f"step {i}: q[{i+1}]={path.q[i+1]} != expected={expected}"
        )


# ---------------------------------------------------------------------------
# 6. Smoke test: single path runs without error for all strategies and gammas
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("strategy", ["inventory", "symmetric"])
@pytest.mark.parametrize("gamma", GAMMA_SCENARIOS)
def test_single_path_runs_without_error(strategy, gamma):
    rng = _make_rng(7)
    path = simulate_path(strategy=strategy, gamma=gamma, **_DEFAULT, rng=rng)
    assert isinstance(path.terminal_profit, float)
    assert len(path.S) == len(path.q) == len(path.X)
    assert len(path.r) == len(path.p_ask) == len(path.p_bid)


# ---------------------------------------------------------------------------
# 7. Terminal profit == X_T + q_T * S_T
# ---------------------------------------------------------------------------
def test_terminal_profit_definition():
    for strategy in ('inventory', 'symmetric'):
        rng = _make_rng(99)
        path = simulate_path(strategy=strategy, gamma=0.1, **_DEFAULT, rng=rng)
        expected = float(path.X[-1] + path.q[-1] * path.S[-1])
        assert path.terminal_profit == pytest.approx(expected), (
            f"{strategy}: terminal_profit={path.terminal_profit:.6f} != {expected:.6f}"
        )


# ---------------------------------------------------------------------------
# 8. Gamma sensitivity: inventory has lower std(profit) than symmetric at gamma=0.5
# ---------------------------------------------------------------------------
def test_gamma_sensitivity_inventory_lower_std():
    gamma = 0.5
    n_paths = 100
    _, profits_inv, _ = run_monte_carlo(
        strategy='inventory', gamma=gamma, n_paths=n_paths, base_seed=42,
    )
    _, profits_sym, _ = run_monte_carlo(
        strategy='symmetric', gamma=gamma, n_paths=n_paths, base_seed=42,
    )
    std_inv = np.std(profits_inv)
    std_sym = np.std(profits_sym)
    assert std_inv < std_sym, (
        f"Expected std_inventory < std_symmetric, got {std_inv:.4f} vs {std_sym:.4f}"
    )


# ---------------------------------------------------------------------------
# 9. Extreme inventory (q=10) does NOT crash the simulator
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("strategy", ["inventory", "symmetric"])
def test_negative_quote_distance_handling(strategy):
    kwargs = dict(**_DEFAULT)
    kwargs["q0"] = 10
    rng = _make_rng(11)
    # Should complete without raising any exception
    path = simulate_path(strategy=strategy, gamma=0.1, **kwargs, rng=rng)
    assert np.isfinite(path.terminal_profit)


# ---------------------------------------------------------------------------
# 10. Common random numbers: same seed → identical mid-price path S for both strategies
# ---------------------------------------------------------------------------
def test_common_random_numbers():
    seed = 42
    path_inv = simulate_path(
        strategy='inventory', gamma=0.1, **_DEFAULT, rng=np.random.default_rng(seed),
    )
    path_sym = simulate_path(
        strategy='symmetric', gamma=0.1, **_DEFAULT, rng=np.random.default_rng(seed),
    )
    np.testing.assert_array_equal(
        path_inv.S, path_sym.S,
        err_msg="S-paths differ despite identical seeds — CRN violated",
    )
