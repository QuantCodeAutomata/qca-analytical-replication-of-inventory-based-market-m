"""
Pytest test suite for src/formulas.py — Avellaneda-Stoikov analytical formulas.

Parameter values follow the paper (Section 3.3) and src/config.py:
    s0=100, T=1, sigma=2, A=140, k=1.5
    gamma in {0.1, 0.01, 0.5}
    q     in {-2, -1, 0, 1, 2}
"""

import numpy as np
import pytest

from src.formulas import (
    ask_quote,
    bid_quote,
    delta_ask,
    delta_bid,
    inactive_value,
    reservation_ask,
    reservation_bid,
    reservation_price,
    spread_constant,
    total_spread,
)

# ---------------------------------------------------------------------------
# Shared test parameters
# ---------------------------------------------------------------------------

S0: float = 100.0
T: float = 1.0
SIGMA: float = 2.0
K: float = 1.5

GAMMAS = [0.1, 0.01, 0.5]
QS = [-2, -1, 0, 1, 2]

# A representative mid-time before maturity
T_MID: float = 0.5

# Paper reference values for spread_constant (Table 1, gamma / k = gamma / 1.5)
PAPER_SPREAD: dict[float, float] = {0.1: 1.29, 0.01: 1.33, 0.5: 1.15}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _all_gamma_q_pairs():
    return [(g, q) for g in GAMMAS for q in QS]


# ---------------------------------------------------------------------------
# Reservation price tests
# ---------------------------------------------------------------------------

def test_reservation_price_zero_inventory():
    """At q=0 the reservation price equals the mid-price for any t."""
    for gamma in GAMMAS:
        for t in [0.0, T_MID, T]:
            r = reservation_price(S0, q=0, gamma=gamma, sigma=SIGMA, T=T, t=t)
            assert np.isclose(r, S0, rtol=1e-6), (
                f"r(q=0) != s0: got {r} for gamma={gamma}, t={t}"
            )


def test_reservation_price_at_maturity():
    """At t=T the time-to-maturity is 0, so r(s, q, T) = s for any q."""
    for gamma in GAMMAS:
        for q in QS:
            r = reservation_price(S0, q=q, gamma=gamma, sigma=SIGMA, T=T, t=T)
            assert np.isclose(r, S0, rtol=1e-6), (
                f"r(t=T) != s0: got {r} for gamma={gamma}, q={q}"
            )


@pytest.mark.parametrize("gamma,q", _all_gamma_q_pairs())
def test_reservation_price_average(gamma, q):
    """Reservation mid-price equals arithmetic mean of reservation ask and bid."""
    t = T_MID
    r_a = reservation_ask(S0, q, gamma, SIGMA, T, t)
    r_b = reservation_bid(S0, q, gamma, SIGMA, T, t)
    r = reservation_price(S0, q, gamma, SIGMA, T, t)
    assert np.isclose(r, (r_a + r_b) / 2.0, rtol=1e-6), (
        f"r != (r_a+r_b)/2: gamma={gamma}, q={q}"
    )


# ---------------------------------------------------------------------------
# Spread tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gamma,q", _all_gamma_q_pairs())
def test_delta_sum_equals_spread(gamma, q):
    """delta_a + delta_b must equal the total spread for every (gamma, q)."""
    t = T_MID
    d_a = delta_ask(q, gamma, SIGMA, K, T, t)
    d_b = delta_bid(q, gamma, SIGMA, K, T, t)
    psi = total_spread(gamma, SIGMA, K, T, t)
    assert np.isclose(d_a + d_b, psi, rtol=1e-6), (
        f"delta_a+delta_b != psi: gamma={gamma}, q={q}; "
        f"delta_a={d_a}, delta_b={d_b}, psi={psi}"
    )


@pytest.mark.parametrize("gamma", GAMMAS)
def test_spread_constants_paper_values(gamma):
    """Spread constant 2/gamma*ln(1+gamma/k) should match paper Table 1 (tol 0.1)."""
    c = spread_constant(gamma, K)
    ref = PAPER_SPREAD[gamma]
    assert np.isclose(c, ref, atol=0.1), (
        f"spread_constant mismatch: gamma={gamma}, computed={c:.4f}, paper={ref}"
    )


def test_gamma_zero_limit():
    """As gamma -> 0 the spread constant converges to 2/k (verified at gamma=1e-8)."""
    tiny_gamma = 1e-8
    c = spread_constant(tiny_gamma, K)
    expected = 2.0 / K
    assert np.isclose(c, expected, rtol=1e-6), (
        f"spread_constant({tiny_gamma}, {K}) = {c}, expected 2/k = {expected}"
    )


# ---------------------------------------------------------------------------
# Inventory-direction tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("q", [1, 2])
def test_positive_inventory_lowers_reservation(gamma, q):
    """With q > 0 the reservation price lies strictly below the mid-price (t < T)."""
    r = reservation_price(S0, q=q, gamma=gamma, sigma=SIGMA, T=T, t=0.0)
    assert r < S0, f"r(q={q}) >= s0 for gamma={gamma}: r={r}"


@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("q", [-1, -2])
def test_negative_inventory_raises_reservation(gamma, q):
    """With q < 0 the reservation price lies strictly above the mid-price (t < T)."""
    r = reservation_price(S0, q=q, gamma=gamma, sigma=SIGMA, T=T, t=0.0)
    assert r > S0, f"r(q={q}) <= s0 for gamma={gamma}: r={r}"


# ---------------------------------------------------------------------------
# Quote ordering test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gamma,q", _all_gamma_q_pairs())
def test_ask_above_bid(gamma, q):
    """Optimal ask quote must always strictly exceed the optimal bid quote."""
    t = T_MID
    p_a = ask_quote(S0, q, gamma, SIGMA, K, T, t)
    p_b = bid_quote(S0, q, gamma, SIGMA, K, T, t)
    assert p_a > p_b, (
        f"ask <= bid: gamma={gamma}, q={q}; ask={p_a}, bid={p_b}"
    )


# ---------------------------------------------------------------------------
# Inactive value function tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gamma,q", _all_gamma_q_pairs())
def test_inactive_value_negative(gamma, q):
    """Inactive-agent value function must always be strictly negative."""
    t = T_MID
    v = inactive_value(x=0.0, s=S0, q=q, gamma=gamma, sigma=SIGMA, T=T, t=t)
    assert v < 0, f"inactive_value >= 0: gamma={gamma}, q={q}, v={v}"


@pytest.mark.parametrize("gamma,q", _all_gamma_q_pairs())
def test_inactive_value_ratio_ask_indifference(gamma, q):
    """v(x + r_a, s, q-1, t) == v(x, s, q, t): utility-indifference for sells."""
    x, t = 0.0, T_MID
    r_a = reservation_ask(S0, q, gamma, SIGMA, T, t)
    v_before = inactive_value(x, S0, q, gamma, SIGMA, T, t)
    v_after = inactive_value(x + r_a, S0, q - 1, gamma, SIGMA, T, t)
    assert np.isclose(v_after, v_before, rtol=1e-6), (
        f"Ask indifference failed: gamma={gamma}, q={q}; "
        f"v_before={v_before:.6e}, v_after={v_after:.6e}"
    )


@pytest.mark.parametrize("gamma,q", _all_gamma_q_pairs())
def test_inactive_value_ratio_bid_indifference(gamma, q):
    """v(x - r_b, s, q+1, t) == v(x, s, q, t): utility-indifference for buys."""
    x, t = 0.0, T_MID
    r_b = reservation_bid(S0, q, gamma, SIGMA, T, t)
    v_before = inactive_value(x, S0, q, gamma, SIGMA, T, t)
    v_after = inactive_value(x - r_b, S0, q + 1, gamma, SIGMA, T, t)
    assert np.isclose(v_after, v_before, rtol=1e-6), (
        f"Bid indifference failed: gamma={gamma}, q={q}; "
        f"v_before={v_before:.6e}, v_after={v_after:.6e}"
    )


# ---------------------------------------------------------------------------
# Symmetry / edge-case tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gamma", GAMMAS)
def test_ask_distance_symmetric_at_zero_inventory(gamma):
    """At q=0 ask and bid quote distances from mid are identical."""
    t = T_MID
    d_a = delta_ask(q=0, gamma=gamma, sigma=SIGMA, k=K, T=T, t=t)
    d_b = delta_bid(q=0, gamma=gamma, sigma=SIGMA, k=K, T=T, t=t)
    assert np.isclose(d_a, d_b, rtol=1e-6), (
        f"delta_a != delta_b at q=0: gamma={gamma}; delta_a={d_a}, delta_b={d_b}"
    )


def test_edge_cases():
    """Sanity checks for extreme inventory levels and t=0 vs t=T boundaries."""
    t0 = 0.0  # start of day

    # Extreme positive inventory: reservation price should be well below mid-price
    for gamma in GAMMAS:
        r = reservation_price(S0, q=2, gamma=gamma, sigma=SIGMA, T=T, t=t0)
        assert r < S0, f"Extreme q=+2: r={r} not below S0 for gamma={gamma}"

    # Extreme negative inventory: reservation price should be well above mid-price
    for gamma in GAMMAS:
        r = reservation_price(S0, q=-2, gamma=gamma, sigma=SIGMA, T=T, t=t0)
        assert r > S0, f"Extreme q=-2: r={r} not above S0 for gamma={gamma}"

    # Single unit: indifference prices bracket the mid-price symmetrically when q=0
    for gamma in GAMMAS:
        r_a = reservation_ask(S0, q=0, gamma=gamma, sigma=SIGMA, T=T, t=t0)
        r_b = reservation_bid(S0, q=0, gamma=gamma, sigma=SIGMA, T=T, t=t0)
        assert r_a > S0, f"q=0 r_a not above S0 for gamma={gamma}"
        assert r_b < S0, f"q=0 r_b not below S0 for gamma={gamma}"
        assert np.isclose(r_a - S0, S0 - r_b, rtol=1e-6), (
            f"q=0 r_a and r_b not symmetric around S0 for gamma={gamma}"
        )

    # At maturity all quote distances collapse to spread_constant/2 only
    for gamma in GAMMAS:
        for q in QS:
            d_a = delta_ask(q, gamma, SIGMA, K, T, t=T)
            d_b = delta_bid(q, gamma, SIGMA, K, T, t=T)
            half_c = spread_constant(gamma, K) / 2.0
            assert np.isclose(d_a, half_c, rtol=1e-6), (
                f"delta_a(t=T) != c/2: gamma={gamma}, q={q}"
            )
            assert np.isclose(d_b, half_c, rtol=1e-6), (
                f"delta_b(t=T) != c/2: gamma={gamma}, q={q}"
            )
