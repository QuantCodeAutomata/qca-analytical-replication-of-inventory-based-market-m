"""
Benchmark quoting strategies for Avellaneda-Stoikov convergence analysis.

Benchmark A  – paper-faithful constant-spread symmetric benchmark.
Benchmark B  – full time-varying spread psi_t, but centred at the mid-price
               S_t rather than the inventory-adjusted reservation price r_t.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Shared spread primitive
# ---------------------------------------------------------------------------

def compute_spread_constant(gamma: float, k: float) -> float:
    """Numerically stable constant spread component  2/γ · ln(1 + γ/k).

    Uses numpy.log1p(gamma / k) to avoid catastrophic cancellation when γ is
    small.  As γ → 0 the expression converges to 2/k (L'Hôpital / Taylor).
    """
    return (2.0 / gamma) * np.log1p(gamma / k)


# ---------------------------------------------------------------------------
# Benchmark A  –  constant spread, centred at mid-price
# ---------------------------------------------------------------------------

def benchmark_a_quotes(
    s: float,
    gamma: float,
    k: float,
) -> tuple[float, float, float, float]:
    """Benchmark A (paper-faithful constant-spread symmetric benchmark).

    c_gamma = 2/γ · ln(1 + γ/k)   [numpy.log1p for stability]
    p^a = s + c_gamma/2
    p^b = s − c_gamma/2

    Parameters
    ----------
    s : float
        Current mid-price.
    gamma : float
        Risk-aversion coefficient.
    k : float
        Intensity decay rate.

    Returns
    -------
    (p_ask, p_bid, delta_ask, delta_bid)
        Quoted prices and their distances from the mid-price.
    """
    c_gamma = compute_spread_constant(gamma, k)
    half = c_gamma / 2.0
    return s + half, s - half, half, half


# ---------------------------------------------------------------------------
# Benchmark B  –  time-varying spread psi_t, centred at mid-price
# ---------------------------------------------------------------------------

def benchmark_b_quotes(
    s: float,
    q: float,
    gamma: float,
    sigma: float,
    k: float,
    T: float,
    t: float,
) -> tuple[float, float, float, float]:
    """Benchmark B (alternative: full time-varying spread psi_t centred at s).

    psi_t = γ·σ²·(T−t) + 2/γ · ln(1 + γ/k)   [numpy.log1p]
    p^a   = s + psi_t/2
    p^b   = s − psi_t/2

    The spread is *identical* to the inventory strategy, but the quotes are
    centred at the raw mid-price s instead of the reservation price r_t.  The
    inventory q is accepted for API uniformity but is not used.

    Parameters
    ----------
    s : float
        Current mid-price.
    q : float
        Current inventory (unused; included for API consistency).
    gamma, sigma, k, T, t : float
        Model parameters and current time.

    Returns
    -------
    (p_ask, p_bid, delta_ask, delta_bid)
    """
    psi_t = gamma * sigma ** 2 * (T - t) + (2.0 / gamma) * np.log1p(gamma / k)
    half = psi_t / 2.0
    return s + half, s - half, half, half


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def quote_center_distance(S: float, r: float) -> float:
    """Absolute distance between mid-price and reservation price |r − S|."""
    return abs(r - S)
