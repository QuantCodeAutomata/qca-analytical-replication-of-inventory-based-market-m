import numpy as np
from src.config import A, K


def inventory_quotes(
    s: float,
    q: int,
    gamma: float,
    sigma: float,
    k: float,
    T: float,
    t: float,
) -> tuple[float, float, float, float, float, float]:
    """
    Inventory-aware quoting policy (Avellaneda-Stoikov Section 3.2).
    Returns (p_ask, p_bid, delta_ask, delta_bid, r, psi)

    r    = s - q*gamma*sigma^2*(T-t)
    psi  = gamma*sigma^2*(T-t) + 2/gamma*ln(1 + gamma/k)
    delta_a = (1-2q)/2 * gamma*sigma^2*(T-t) + 1/gamma * ln(1 + gamma/k)
    delta_b = (1+2q)/2 * gamma*sigma^2*(T-t) + 1/gamma * ln(1 + gamma/k)
    p_ask = s + delta_a  (= r + psi/2)
    p_bid = s - delta_b  (= r - psi/2)
    """
    tau = T - t
    gst2 = gamma * sigma**2 * tau          # γσ²(T-t)
    log_term = np.log1p(gamma / k) / gamma  # (1/γ)·ln(1 + γ/k)

    r = s - q * gst2
    psi = gst2 + 2.0 * log_term

    delta_ask = (1 - 2 * q) / 2.0 * gst2 + log_term
    delta_bid = (1 + 2 * q) / 2.0 * gst2 + log_term

    p_ask = s + delta_ask   # ≡ r + psi/2
    p_bid = s - delta_bid   # ≡ r - psi/2

    return p_ask, p_bid, delta_ask, delta_bid, r, psi


def symmetric_quotes(
    s: float,
    gamma: float,
    k: float,
) -> tuple[float, float, float, float]:
    """
    Symmetric benchmark: constant spread c = 2/gamma * ln(1 + gamma/k) centred at s.
    Returns (p_ask, p_bid, delta_ask, delta_bid)
    Uses numpy.log1p for numerical stability.
    """
    c = 2.0 / gamma * np.log1p(gamma / k)
    half_c = c / 2.0
    p_ask = s + half_c
    p_bid = s - half_c
    return p_ask, p_bid, half_c, half_c


def fill_intensities(
    delta_a: float,
    delta_b: float,
    A: float,
    k: float,
) -> tuple[float, float]:
    """λ^a = A*exp(-k*delta_a),  λ^b = A*exp(-k*delta_b)"""
    lambda_a = A * np.exp(-k * delta_a)
    lambda_b = A * np.exp(-k * delta_b)
    return lambda_a, lambda_b
