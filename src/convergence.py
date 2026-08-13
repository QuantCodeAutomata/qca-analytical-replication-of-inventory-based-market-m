"""
Convergence analysis: gamma-to-zero and benchmark ambiguity study.

Compares three quoting strategies across a grid of risk-aversion values γ:
  • Inventory strategy  – full Avellaneda-Stoikov optimal quotes (simulator.py)
  • Benchmark A         – constant spread c_γ = 2/γ·ln(1+γ/k), centred at S_t
  • Benchmark B         – time-varying spread ψ_t, centred at S_t (not r_t)

Common Random Numbers (CRN) are used so that variance reduction makes
differences between strategies attributable only to quoting policy.

Actual interface for simulate_path (src.simulator)
--------------------------------------------------
simulate_path(strategy, gamma, sigma, k, A, T, dt, S0, q0, X0, rng) -> PathData
    PathData.terminal_profit : float      – X_T + q_T · S_T
    PathData.terminal_q      : int        – q_T
    PathData.r               : ndarray(N) – reservation prices r_t per step
    PathData.S               : ndarray(N+1) – mid-price (incl. S_0 and S_T)

CRN draw order within each simulation step (must match simulator.py exactly):
    1. rng.random()  – u_a: ask-fill Bernoulli (fill if u_a < λ_a · dt)
    2. rng.random()  – u_b: bid-fill Bernoulli (fill if u_b < λ_b · dt)
    3. rng.random()  – u_s: mid-price direction (eps=+1 if u_s<0.5 else -1)

Mid-price model: S_{i+1} = S_i + σ·√dt·eps  (binary random walk, not Gaussian).
"""

import os
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use('Agg')

from src.benchmark_variants import (
    benchmark_a_quotes,
    benchmark_b_quotes,
    quote_center_distance,
)
from src.simulator import simulate_path


# ---------------------------------------------------------------------------
# Internal helpers – result extraction
# ---------------------------------------------------------------------------

def _extract(result) -> tuple[float, float, float]:
    """Unpack a simulate_path return value into (terminal_profit, final_q, qcd_sum).

    Handles the ``PathData`` dataclass from ``src.simulator`` (primary case),
    a plain ``dict``, or a bare sequence as fallbacks.

    ``qcd_sum`` is Σ_t |r_t − S_t| over all N steps of the path, computed
    from PathData.r (shape N) and PathData.S[:-1] (shape N).
    """
    # PathData dataclass (src.simulator) ← primary case
    if hasattr(result, 'terminal_profit') and hasattr(result, 'r'):
        qcd_sum = float(np.sum(np.abs(result.r - result.S[:-1])))
        return float(result.terminal_profit), float(result.terminal_q), qcd_sum
    # dict fallback
    if isinstance(result, dict):
        return (
            float(result['terminal_profit']),
            float(result['final_q']),
            float(result.get('quote_center_diff_sum', 0.0)),
        )
    # Sequence fallback
    tp = float(result[0])
    fq = float(result[1])
    qcd = float(result[2]) if len(result) > 2 else 0.0
    return tp, fq, qcd


# ---------------------------------------------------------------------------
# Local benchmark simulation functions
# ---------------------------------------------------------------------------

def _simulate_path_bA(
    gamma: float,
    sigma: float,
    k: float,
    A: float,
    T: float,
    dt: float,
    S0: float,
    q0: float,
    x0: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Simulate one path under Benchmark A (constant-spread, mid-price centred).

    Chronology per step: compute quotes → determine fills → update cash &
    inventory → advance mid-price (identical to simulator.py).

    CRN draw order per step (must mirror simulator.py exactly):
        u_a = rng.random()  – ask Bernoulli  (fill if u_a < λ_a · dt)
        u_b = rng.random()  – bid Bernoulli  (fill if u_b < λ_b · dt)
        u_s = rng.random()  – mid-price direction (eps=+1 if u_s<0.5 else -1)

    Returns
    -------
    (terminal_profit, final_q)
    """
    n_steps = int(round(T / dt))
    S = float(S0)
    q = float(q0)
    x = float(x0)
    sqrt_dt = np.sqrt(dt)

    for i in range(n_steps):
        # --- quotes (constant spread, centred at mid-price) ---
        p_ask, p_bid, delta_ask, delta_bid = benchmark_a_quotes(S, gamma, k)

        lambda_ask = A * np.exp(-k * delta_ask)
        lambda_bid = A * np.exp(-k * delta_bid)
        prob_ask = lambda_ask * dt   # Bernoulli approx matching simulator.py
        prob_bid = lambda_bid * dt

        # --- CRN draws: u_a, u_b, u_s (matches simulator.py order) ---
        u_a = rng.random()
        u_b = rng.random()
        u_s = rng.random()

        # --- fills ---
        if u_a < prob_ask:
            x += p_ask
            q -= 1.0
        if u_b < prob_bid:
            x -= p_bid
            q += 1.0

        # --- mid-price update (binary random walk, matches simulator.py) ---
        eps = 1 if u_s < 0.5 else -1
        S += sigma * sqrt_dt * eps

    return x + q * S, q


def _simulate_path_bB(
    gamma: float,
    sigma: float,
    k: float,
    A: float,
    T: float,
    dt: float,
    S0: float,
    q0: float,
    x0: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Simulate one path under Benchmark B (time-varying spread, mid-price centred).

    psi_t = γ·σ²·(T−t) + 2/γ·ln(1+γ/k)  recomputed every step.

    CRN draw order per step (must mirror simulator.py exactly):
        u_a = rng.random()  – ask Bernoulli  (fill if u_a < λ_a · dt)
        u_b = rng.random()  – bid Bernoulli  (fill if u_b < λ_b · dt)
        u_s = rng.random()  – mid-price direction (eps=+1 if u_s<0.5 else -1)

    Returns
    -------
    (terminal_profit, final_q)
    """
    n_steps = int(round(T / dt))
    S = float(S0)
    q = float(q0)
    x = float(x0)
    sqrt_dt = np.sqrt(dt)

    for i in range(n_steps):
        t = i * dt

        # --- quotes (time-varying spread, centred at mid-price) ---
        p_ask, p_bid, delta_ask, delta_bid = benchmark_b_quotes(S, q, gamma, sigma, k, T, t)

        lambda_ask = A * np.exp(-k * delta_ask)
        lambda_bid = A * np.exp(-k * delta_bid)
        prob_ask = lambda_ask * dt   # Bernoulli approx matching simulator.py
        prob_bid = lambda_bid * dt

        # --- CRN draws: u_a, u_b, u_s (matches simulator.py order) ---
        u_a = rng.random()
        u_b = rng.random()
        u_s = rng.random()

        # --- fills ---
        if u_a < prob_ask:
            x += p_ask
            q -= 1.0
        if u_b < prob_bid:
            x -= p_bid
            q += 1.0

        # --- mid-price update (binary random walk, matches simulator.py) ---
        eps = 1 if u_s < 0.5 else -1
        S += sigma * sqrt_dt * eps

    return x + q * S, q


# ---------------------------------------------------------------------------
# Convergence experiment
# ---------------------------------------------------------------------------

def run_convergence_experiment(
    gamma_grid: List[float],
    n_paths: int = 1000,
    sigma: float = 2.0,
    k: float = 1.5,
    A: float = 140.0,
    T: float = 1.0,
    dt: float = 0.005,
    S0: float = 100.0,
    base_seed: int = 42,
) -> pd.DataFrame:
    """Run the gamma-to-zero convergence / benchmark ambiguity experiment.

    For each γ in *gamma_grid* the function simulates *n_paths* independent
    trajectories under three strategies (inventory, Benchmark A, Benchmark B)
    using Common Random Numbers (CRN): every path index uses the same seed
    for all three strategies so that mid-price realisations and fill dice rolls
    are shared, isolating the effect of the quoting policy.

    Parameters
    ----------
    gamma_grid : list of float
        Risk-aversion values to sweep (descending towards zero recommended).
    n_paths : int
        Monte-Carlo paths per γ.
    sigma, k, A, T, dt, S0 : float
        Model parameters forwarded to all simulators.
    base_seed : int
        Path *i* uses ``np.random.default_rng(base_seed + i)`` for all three
        strategies, guaranteeing CRN alignment.

    Returns
    -------
    pd.DataFrame
        One row per γ with columns:
        gamma,
        mean_profit_inv, std_profit_inv,
        mean_profit_bA,  std_profit_bA,
        mean_profit_bB,  std_profit_bB,
        mean_q_inv,      std_q_inv,
        mean_q_bA,       std_q_bA,
        mean_q_bB,       std_q_bB,
        mean_quote_center_diff,
        abs_diff_mean_profit_A,
        abs_diff_std_profit_A,  abs_diff_std_q_A,
        abs_diff_std_profit_B,  abs_diff_std_q_B.
    """
    n_steps = int(round(T / dt))
    records = []

    for gamma in gamma_grid:
        profits_inv, profits_bA, profits_bB = [], [], []
        qs_inv, qs_bA, qs_bB = [], [], []
        qcd_total = 0.0  # Σ_{paths, steps} |r_t − S_t|

        for path_idx in range(n_paths):
            seed = base_seed + path_idx

            # ---- Inventory strategy (CRN seed) ----
            rng_inv = np.random.default_rng(seed)
            result = simulate_path(
                strategy='inventory',
                gamma=gamma, sigma=sigma, k=k, A=A,
                T=T, dt=dt, S0=S0, q0=0, X0=0.0,
                rng=rng_inv,
            )
            tp_inv, fq_inv, qcd_sum = _extract(result)
            profits_inv.append(tp_inv)
            qs_inv.append(fq_inv)
            qcd_total += qcd_sum

            # ---- Benchmark A (same CRN seed) ----
            rng_bA = np.random.default_rng(seed)
            tp_bA, fq_bA = _simulate_path_bA(
                gamma=gamma, sigma=sigma, k=k, A=A,
                T=T, dt=dt, S0=S0, q0=0.0, x0=0.0,
                rng=rng_bA,
            )
            profits_bA.append(tp_bA)
            qs_bA.append(fq_bA)

            # ---- Benchmark B (same CRN seed) ----
            rng_bB = np.random.default_rng(seed)
            tp_bB, fq_bB = _simulate_path_bB(
                gamma=gamma, sigma=sigma, k=k, A=A,
                T=T, dt=dt, S0=S0, q0=0.0, x0=0.0,
                rng=rng_bB,
            )
            profits_bB.append(tp_bB)
            qs_bB.append(fq_bB)

        arr_inv = np.asarray(profits_inv)
        arr_bA  = np.asarray(profits_bA)
        arr_bB  = np.asarray(profits_bB)
        arr_qi  = np.asarray(qs_inv)
        arr_qA  = np.asarray(qs_bA)
        arr_qB  = np.asarray(qs_bB)

        mean_qcd = qcd_total / (n_paths * n_steps)

        records.append({
            'gamma':                  gamma,
            'mean_profit_inv':        float(arr_inv.mean()),
            'std_profit_inv':         float(arr_inv.std()),
            'mean_profit_bA':         float(arr_bA.mean()),
            'std_profit_bA':          float(arr_bA.std()),
            'mean_profit_bB':         float(arr_bB.mean()),
            'std_profit_bB':          float(arr_bB.std()),
            'mean_q_inv':             float(arr_qi.mean()),
            'std_q_inv':              float(arr_qi.std()),
            'mean_q_bA':              float(arr_qA.mean()),
            'std_q_bA':               float(arr_qA.std()),
            'mean_q_bB':              float(arr_qB.mean()),
            'std_q_bB':               float(arr_qB.std()),
            'mean_quote_center_diff': float(mean_qcd),
            'abs_diff_mean_profit_A': float(abs(arr_inv.mean() - arr_bA.mean())),
            'abs_diff_std_profit_A':  float(abs(arr_inv.std()  - arr_bA.std())),
            'abs_diff_std_q_A':       float(abs(arr_qi.std()   - arr_qA.std())),
            'abs_diff_std_profit_B':  float(abs(arr_inv.std()  - arr_bB.std())),
            'abs_diff_std_q_B':       float(abs(arr_qi.std()   - arr_qB.std())),
        })

        print(
            f'  γ={gamma:.4f} | '
            f'E[π_inv]={arr_inv.mean():.2f}  '
            f'E[π_bA]={arr_bA.mean():.2f}  '
            f'E[π_bB]={arr_bB.mean():.2f}  '
            f'|qcd|={mean_qcd:.4f}'
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_convergence(df: pd.DataFrame, save_dir: str) -> None:
    """Produce three convergence plots (log-scale x-axis) and save as PNGs.

    Figures
    -------
    1. ``exp3_profit_std_diff.png``  –  |std_profit_inv − std_profit_bX| vs γ
    2. ``exp3_inv_std_diff.png``     –  |std_q_inv − std_q_bX| vs γ
    3. ``exp3_quote_center_diff.png``–  mean |r_t − S_t| vs γ
    """
    gamma = df['gamma'].values

    # --- Plot 1: profit std divergence ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(gamma, df['abs_diff_std_profit_A'], 'o-', label='vs Benchmark A')
    ax.plot(gamma, df['abs_diff_std_profit_B'], 's--', label='vs Benchmark B')
    ax.set_xscale('log')
    ax.set_xlabel('γ  (log scale)')
    ax.set_ylabel('|σ(π_inv) − σ(π_benchmark)|')
    ax.set_title('Profit Std Convergence: Inventory vs Benchmarks')
    ax.legend()
    ax.grid(True, which='both', linestyle=':', alpha=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'exp3_profit_std_diff.png'), dpi=150)
    plt.close(fig)

    # --- Plot 2: inventory std divergence ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(gamma, df['abs_diff_std_q_A'], 'o-', label='vs Benchmark A')
    ax.plot(gamma, df['abs_diff_std_q_B'], 's--', label='vs Benchmark B')
    ax.set_xscale('log')
    ax.set_xlabel('γ  (log scale)')
    ax.set_ylabel('|σ(q_inv) − σ(q_benchmark)|')
    ax.set_title('Inventory Std Convergence: Inventory vs Benchmarks')
    ax.legend()
    ax.grid(True, which='both', linestyle=':', alpha=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'exp3_inv_std_diff.png'), dpi=150)
    plt.close(fig)

    # --- Plot 3: quote-centre distance ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(gamma, df['mean_quote_center_diff'], 'D-', color='tab:green')
    ax.set_xscale('log')
    ax.set_xlabel('γ  (log scale)')
    ax.set_ylabel('E[|r_t − S_t|]')
    ax.set_title('Mean Quote-Centre Distance vs γ  (inventory strategy)')
    ax.grid(True, which='both', linestyle=':', alpha=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'exp3_quote_center_diff.png'), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Numerical limit check
# ---------------------------------------------------------------------------

def check_spread_constant_limit(k: float = 1.5, gamma_small: float = 1e-8) -> float:
    """Numerically verify 2/γ · log1p(γ/k) → 2/k as γ → 0.

    Evaluates the expression at *gamma_small* using numpy.log1p and returns
    the result.  The caller can assert it is within tolerance of 2/k.

    Parameters
    ----------
    k : float
        Intensity decay rate.
    gamma_small : float
        Small positive γ at which to evaluate the limit approximation.

    Returns
    -------
    float
        Computed value of 2/γ · log1p(γ/k) at γ = gamma_small.
    """
    return (2.0 / gamma_small) * np.log1p(gamma_small / k)
