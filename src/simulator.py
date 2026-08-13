import numpy as np
from dataclasses import dataclass
from typing import Literal

from src.strategies import inventory_quotes, fill_intensities, symmetric_quotes


@dataclass
class PathData:
    """Stores full pathwise data for one simulation path."""
    S: np.ndarray           # mid-price time series, length N+1
    r: np.ndarray           # reservation price,    length N
    p_ask: np.ndarray       # ask quote,            length N
    p_bid: np.ndarray       # bid quote,            length N
    delta_ask: np.ndarray   # ask distance from mid,length N
    delta_bid: np.ndarray   # bid distance from mid,length N
    q: np.ndarray           # inventory (int),      length N+1
    X: np.ndarray           # cash,                 length N+1
    ask_fills: np.ndarray   # 0/1 indicators,       length N
    bid_fills: np.ndarray   # 0/1 indicators,       length N
    lambda_a: np.ndarray    # ask intensities,      length N
    lambda_b: np.ndarray    # bid intensities,      length N
    terminal_profit: float  # X_T + q_T * S_T
    terminal_q: int         # q_T
    violations: int         # steps where lambda*dt > 1


def simulate_path(
    strategy: Literal['inventory', 'symmetric'],
    gamma: float,
    sigma: float,
    k: float,
    A: float,
    T: float,
    dt: float,
    S0: float,
    q0: int,
    X0: float,
    rng: np.random.Generator,
) -> PathData:
    """
    Simulate one path of the Avellaneda-Stoikov market-making model.

    CRITICAL chronology (from paper):
    1. At each step t_i: compute quotes from current state (S, q, X)
    2. Compute fill intensities lambda^a, lambda^b
    3. Check lambda*dt > 1 (log violation, do NOT clip)
    4. Draw Bernoulli fills: I^a ~ Bern(lambda^a * dt), I^b ~ Bern(lambda^b * dt)
    5. Update cash and inventory if fills occur
    6. THEN update mid-price: S += sigma*sqrt(dt) * eps  where eps in {-1,+1} p=0.5
    """
    N = int(round(T / dt))
    sqrt_dt = np.sqrt(dt)

    # State arrays (N+1 length include t=0 initial and t=T terminal)
    S = np.empty(N + 1)
    q = np.empty(N + 1, dtype=np.int64)
    X = np.empty(N + 1)

    # Per-step arrays (length N)
    r_arr       = np.empty(N)
    p_ask_arr   = np.empty(N)
    p_bid_arr   = np.empty(N)
    da_arr      = np.empty(N)
    db_arr      = np.empty(N)
    la_arr      = np.empty(N)
    lb_arr      = np.empty(N)
    ask_fills   = np.empty(N, dtype=np.int64)
    bid_fills   = np.empty(N, dtype=np.int64)

    S[0] = S0
    q[0] = q0
    X[0] = X0
    violations = 0

    for i in range(N):
        t_i = i * dt
        s_i = S[i]
        q_i = int(q[i])

        # Step 1: Compute quotes
        if strategy == 'inventory':
            pa, pb, da, db, ri, _ = inventory_quotes(s_i, q_i, gamma, sigma, k, T, t_i)
        else:  # symmetric
            pa, pb, da, db = symmetric_quotes(s_i, gamma, k)
            ri = s_i  # symmetric has no inventory-adjusted reservation price

        r_arr[i]     = ri
        p_ask_arr[i] = pa
        p_bid_arr[i] = pb
        da_arr[i]    = da
        db_arr[i]    = db

        # Step 2: Fill intensities
        la, lb = fill_intensities(da, db, A, k)
        la_arr[i] = la
        lb_arr[i] = lb

        # Step 3: Log violations (do NOT clip probabilities)
        if la * dt > 1.0 or lb * dt > 1.0:
            violations += 1

        # Step 4: Bernoulli fills — consume random numbers in fixed order for CRN
        u_a = rng.random()
        u_b = rng.random()
        ia = int(u_a < la * dt)
        ib = int(u_b < lb * dt)
        ask_fills[i] = ia
        bid_fills[i] = ib

        # Step 5: Update cash and inventory
        X[i + 1] = X[i] + ia * pa - ib * pb
        q[i + 1] = q_i - ia + ib

        # Step 6: Update mid-price (eps in {-1,+1} with p=0.5)
        u_s = rng.random()
        eps = 1 if u_s < 0.5 else -1
        S[i + 1] = s_i + sigma * sqrt_dt * eps

    terminal_profit = float(X[N] + q[N] * S[N])

    return PathData(
        S=S,
        r=r_arr,
        p_ask=p_ask_arr,
        p_bid=p_bid_arr,
        delta_ask=da_arr,
        delta_bid=db_arr,
        q=q,
        X=X,
        ask_fills=ask_fills,
        bid_fills=bid_fills,
        lambda_a=la_arr,
        lambda_b=lb_arr,
        terminal_profit=terminal_profit,
        terminal_q=int(q[N]),
        violations=violations,
    )


def run_monte_carlo(
    strategy: Literal['inventory', 'symmetric'],
    gamma: float,
    n_paths: int,
    sigma: float = 2.0,
    k: float = 1.5,
    A: float = 140.0,
    T: float = 1.0,
    dt: float = 0.005,
    S0: float = 100.0,
    q0: int = 0,
    X0: float = 0.0,
    base_seed: int = 42,
    common_random: bool = True,
) -> tuple[list[PathData], np.ndarray, np.ndarray]:
    """
    Run Monte Carlo simulation over n_paths independent paths.

    CRN: when common_random=True path i uses seed = base_seed + i.
    Calling this function with the same base_seed for both strategies
    produces identical S-paths (common random numbers), isolating the
    effect of the quoting strategy.

    Returns (paths, terminal_profits, terminal_qs).
    """
    paths: list[PathData] = []
    terminal_profits = np.empty(n_paths)
    terminal_qs = np.empty(n_paths, dtype=np.int64)

    for i in range(n_paths):
        seed = base_seed + i          # same offset for both strategies → CRN
        rng = np.random.default_rng(seed)
        path = simulate_path(
            strategy=strategy,
            gamma=gamma,
            sigma=sigma,
            k=k,
            A=A,
            T=T,
            dt=dt,
            S0=S0,
            q0=q0,
            X0=X0,
            rng=rng,
        )
        paths.append(path)
        terminal_profits[i] = path.terminal_profit
        terminal_qs[i] = path.terminal_q

    return paths, terminal_profits, terminal_qs
