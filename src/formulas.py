"""
Closed-form analytical formulas for the Avellaneda-Stoikov (2008) market-making model.

All functions assume:
- Arithmetic Brownian motion for the mid-price (driftless).
- CARA utility: u(W) = -exp(-gamma * W).
- Exponential order-arrival intensities: lambda(delta) = A * exp(-k * delta).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Value function
# ---------------------------------------------------------------------------

def inactive_value(
    x: float | np.ndarray,
    s: float | np.ndarray,
    q: float | np.ndarray,
    gamma: float,
    sigma: float,
    T: float,
    t: float | np.ndarray,
) -> float | np.ndarray:
    """Closed-form inactive-agent value function under driftless ABM and CARA utility.

    Equals the expected CARA utility of terminal wealth for an agent who
    holds q units of stock and cash x, with mid-price currently at s:

        v(x, s, q, t) = E[-exp(-gamma * (x + q * S_T))]
                      = -exp(-gamma*x)
                        * exp(-gamma*q*s)
                        * exp((gamma^2 * q^2 * sigma^2 * (T-t)) / 2)

    Parameters
    ----------
    x : float or ndarray
        Cash wealth.
    s : float or ndarray
        Current mid-price.
    q : float or ndarray
        Inventory (number of shares held).
    gamma : float
        Absolute risk-aversion coefficient (> 0).
    sigma : float
        Mid-price volatility (arithmetic Brownian motion).
    T : float
        Trading horizon.
    t : float or ndarray
        Current time (0 <= t <= T).

    Returns
    -------
    float or ndarray
        Value function (always strictly negative).
    """
    tau = T - t
    return (
        -np.exp(-gamma * x)
        * np.exp(-gamma * q * s)
        * np.exp(0.5 * gamma**2 * q**2 * sigma**2 * tau)
    )


# ---------------------------------------------------------------------------
# Reservation (indifference) prices
# ---------------------------------------------------------------------------

def reservation_ask(
    s: float | np.ndarray,
    q: float | np.ndarray,
    gamma: float,
    sigma: float,
    T: float,
    t: float | np.ndarray,
) -> float | np.ndarray:
    """Reservation ask price from CARA utility indifference.

    The price at which selling one unit leaves the agent indifferent:
        v(x + r_a, s, q-1, t) = v(x, s, q, t)

    Closed form:
        r^a(s, q, t) = s + ((1 - 2q) / 2) * gamma * sigma^2 * (T - t)

    Parameters
    ----------
    s : float or ndarray
        Current mid-price.
    q : float or ndarray
        Current inventory.
    gamma : float
        Absolute risk-aversion coefficient.
    sigma : float
        Mid-price volatility.
    T : float
        Trading horizon.
    t : float or ndarray
        Current time.

    Returns
    -------
    float or ndarray
        Reservation ask price.
    """
    tau = T - t
    return s + ((1.0 - 2.0 * q) / 2.0) * gamma * sigma**2 * tau


def reservation_bid(
    s: float | np.ndarray,
    q: float | np.ndarray,
    gamma: float,
    sigma: float,
    T: float,
    t: float | np.ndarray,
) -> float | np.ndarray:
    """Reservation bid price from CARA utility indifference.

    The price at which buying one unit leaves the agent indifferent:
        v(x - r_b, s, q+1, t) = v(x, s, q, t)

    Closed form:
        r^b(s, q, t) = s + ((-1 - 2q) / 2) * gamma * sigma^2 * (T - t)

    Parameters
    ----------
    s : float or ndarray
        Current mid-price.
    q : float or ndarray
        Current inventory.
    gamma : float
        Absolute risk-aversion coefficient.
    sigma : float
        Mid-price volatility.
    T : float
        Trading horizon.
    t : float or ndarray
        Current time.

    Returns
    -------
    float or ndarray
        Reservation bid price.
    """
    tau = T - t
    return s + ((-1.0 - 2.0 * q) / 2.0) * gamma * sigma**2 * tau


def reservation_price(
    s: float | np.ndarray,
    q: float | np.ndarray,
    gamma: float,
    sigma: float,
    T: float,
    t: float | np.ndarray,
) -> float | np.ndarray:
    """Midpoint reservation / indifference price.

    Arithmetic mean of the reservation ask and bid:
        r(s, q, t) = (r^a + r^b) / 2 = s - q * gamma * sigma^2 * (T - t)

    Agents with long inventory (q > 0) quote a lower mid-price to encourage
    sells; agents with short inventory (q < 0) quote higher to encourage buys.

    Parameters
    ----------
    s : float or ndarray
        Current mid-price.
    q : float or ndarray
        Current inventory.
    gamma : float
        Absolute risk-aversion coefficient.
    sigma : float
        Mid-price volatility.
    T : float
        Trading horizon.
    t : float or ndarray
        Current time.

    Returns
    -------
    float or ndarray
        Reservation / indifference mid-price.
    """
    tau = T - t
    return s - q * gamma * sigma**2 * tau


# ---------------------------------------------------------------------------
# Spread components
# ---------------------------------------------------------------------------

def spread_constant(gamma: float, k: float) -> float:
    """Constant spread component arising from the exponential intensity model.

    Derived from the Hamilton-Jacobi-Bellman equation with lambda(delta) = A*exp(-k*delta):
        c(gamma, k) = (2 / gamma) * ln(1 + gamma / k)

    Uses numpy.log1p(gamma / k) for numerical stability when gamma << k.

    Parameters
    ----------
    gamma : float
        Absolute risk-aversion coefficient (> 0).
    k : float
        Intensity decay rate (> 0).

    Returns
    -------
    float
        Constant spread component; converges to 2/k as gamma -> 0.
    """
    return (2.0 / gamma) * np.log1p(gamma / k)


def total_spread(
    gamma: float,
    sigma: float,
    k: float,
    T: float,
    t: float | np.ndarray,
) -> float | np.ndarray:
    """Total optimal bid-ask spread.

    Sum of the inventory-risk component and the constant intensity component:
        psi(gamma, sigma, k, T, t) = gamma * sigma^2 * (T - t)
                                     + (2 / gamma) * ln(1 + gamma / k)

    Parameters
    ----------
    gamma : float
        Absolute risk-aversion coefficient.
    sigma : float
        Mid-price volatility.
    k : float
        Intensity decay rate.
    T : float
        Trading horizon.
    t : float or ndarray
        Current time.

    Returns
    -------
    float or ndarray
        Optimal total spread.
    """
    tau = T - t
    return gamma * sigma**2 * tau + spread_constant(gamma, k)


# ---------------------------------------------------------------------------
# Quote distances from mid-price
# ---------------------------------------------------------------------------

def delta_ask(
    q: float | np.ndarray,
    gamma: float,
    sigma: float,
    k: float,
    T: float,
    t: float | np.ndarray,
) -> float | np.ndarray:
    """Optimal ask quote distance from the current mid-price.

        delta^a(q, t) = ((1 - 2q) / 2) * gamma * sigma^2 * (T - t)
                        + (1 / gamma) * ln(1 + gamma / k)

    Parameters
    ----------
    q : float or ndarray
        Current inventory.
    gamma : float
        Absolute risk-aversion coefficient.
    sigma : float
        Mid-price volatility.
    k : float
        Intensity decay rate.
    T : float
        Trading horizon.
    t : float or ndarray
        Current time.

    Returns
    -------
    float or ndarray
        Ask distance; positive means quote is above mid-price.
    """
    tau = T - t
    return ((1.0 - 2.0 * q) / 2.0) * gamma * sigma**2 * tau + np.log1p(gamma / k) / gamma


def delta_bid(
    q: float | np.ndarray,
    gamma: float,
    sigma: float,
    k: float,
    T: float,
    t: float | np.ndarray,
) -> float | np.ndarray:
    """Optimal bid quote distance from the current mid-price.

        delta^b(q, t) = ((1 + 2q) / 2) * gamma * sigma^2 * (T - t)
                        + (1 / gamma) * ln(1 + gamma / k)

    Parameters
    ----------
    q : float or ndarray
        Current inventory.
    gamma : float
        Absolute risk-aversion coefficient.
    sigma : float
        Mid-price volatility.
    k : float
        Intensity decay rate.
    T : float
        Trading horizon.
    t : float or ndarray
        Current time.

    Returns
    -------
    float or ndarray
        Bid distance; positive means quote is below mid-price.
    """
    tau = T - t
    return ((1.0 + 2.0 * q) / 2.0) * gamma * sigma**2 * tau + np.log1p(gamma / k) / gamma


# ---------------------------------------------------------------------------
# Optimal quote prices
# ---------------------------------------------------------------------------

def ask_quote(
    s: float | np.ndarray,
    q: float | np.ndarray,
    gamma: float,
    sigma: float,
    k: float,
    T: float,
    t: float | np.ndarray,
) -> float | np.ndarray:
    """Optimal ask quote price.

    Reservation price shifted up by half the total spread:
        p^a(s, q, t) = r(s, q, t) + psi(gamma, sigma, k, T, t) / 2

    Equivalently: p^a = s + delta^a.

    Parameters
    ----------
    s : float or ndarray
        Current mid-price.
    q : float or ndarray
        Current inventory.
    gamma : float
        Absolute risk-aversion coefficient.
    sigma : float
        Mid-price volatility.
    k : float
        Intensity decay rate.
    T : float
        Trading horizon.
    t : float or ndarray
        Current time.

    Returns
    -------
    float or ndarray
        Optimal ask quote.
    """
    return reservation_price(s, q, gamma, sigma, T, t) + total_spread(gamma, sigma, k, T, t) / 2.0


def bid_quote(
    s: float | np.ndarray,
    q: float | np.ndarray,
    gamma: float,
    sigma: float,
    k: float,
    T: float,
    t: float | np.ndarray,
) -> float | np.ndarray:
    """Optimal bid quote price.

    Reservation price shifted down by half the total spread:
        p^b(s, q, t) = r(s, q, t) - psi(gamma, sigma, k, T, t) / 2

    Equivalently: p^b = s - delta^b.

    Parameters
    ----------
    s : float or ndarray
        Current mid-price.
    q : float or ndarray
        Current inventory.
    gamma : float
        Absolute risk-aversion coefficient.
    sigma : float
        Mid-price volatility.
    k : float
        Intensity decay rate.
    T : float
        Trading horizon.
    t : float or ndarray
        Current time.

    Returns
    -------
    float or ndarray
        Optimal bid quote.
    """
    return reservation_price(s, q, gamma, sigma, T, t) - total_spread(gamma, sigma, k, T, t) / 2.0
