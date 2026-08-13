"""
Symbolic derivation and verification of Avellaneda-Stoikov closed-form formulas.

Uses SymPy to confirm:
- Utility-indifference conditions for reservation ask and bid prices.
- Reservation price equals average of reservation ask and bid.
- Optimal half-spreads sum to the total spread.
- Spread constant converges to 2/k as risk aversion vanishes.

Run as a script:  python -m src.symbolic_checks
"""

import sympy


# ---------------------------------------------------------------------------
# Module-level symbol declarations (exported for inspection / reuse)
# ---------------------------------------------------------------------------

gamma, sigma, q, t, T_sym, s, x, k_sym = sympy.symbols(
    "gamma sigma q t T s x k", positive=True
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _v(x_: sympy.Expr, s_: sympy.Expr, q_: sympy.Expr, t_: sympy.Expr) -> sympy.Expr:
    """Symbolic inactive-agent value function.

    v(x, s, q, t) = -exp(-gamma*x) * exp(-gamma*q*s)
                    * exp(gamma^2 * q^2 * sigma^2 * (T - t) / 2)
    """
    tau_ = T_sym - t_
    return (
        -sympy.exp(-gamma * x_)
        * sympy.exp(-gamma * q_ * s_)
        * sympy.exp(gamma**2 * q_**2 * sigma**2 * tau_ / sympy.Integer(2))
    )


def _exponent(expr: sympy.Expr) -> sympy.Expr:
    """Extract and sum the arguments of all Mul-combined exp factors in expr.

    Works for expressions of the form -exp(a)*exp(b)*exp(c) by taking log of
    the absolute value and expanding, giving a + b + c directly.
    """
    return sympy.expand(sympy.log(sympy.Abs(expr)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_symbolic_checks() -> dict:
    """Derive and verify all Avellaneda-Stoikov closed-form relationships.

    Performs five symbolic checks:

    1. **indifference_ask_check** – Verifies that the known reservation ask
       formula solves v(x + r_a, s, q-1, t) = v(x, s, q, t) by confirming
       the exponent difference is identically zero.

    2. **indifference_bid_check** – Same for the bid side:
       v(x - r_b, s, q+1, t) = v(x, s, q, t).

    3. **r_avg_check** – Confirms the reservation mid-price equals the
       arithmetic mean of reservation ask and bid.

    4. **delta_sum_check** – Confirms delta^a + delta^b equals total spread psi.

    5. **gamma_zero_limit** – Computes lim_{gamma->0} (2/gamma)*ln(1+gamma/k),
       which should equal 2/k (the spread component at zero risk aversion).

    Returns
    -------
    dict
        Keys: 'indifference_ask_check', 'indifference_bid_check',
               'r_avg_check', 'delta_sum_check', 'gamma_zero_limit'.
        Bool entries are True when the identity holds exactly.
        'gamma_zero_limit' is the SymPy expression for the limit.
    """
    tau = T_sym - t  # time-to-maturity

    # ------------------------------------------------------------------
    # Symbolic formulas (as derived in the paper)
    # ------------------------------------------------------------------
    r_a_sym = s + ((1 - 2 * q) / sympy.Integer(2)) * gamma * sigma**2 * tau
    r_b_sym = s + ((-1 - 2 * q) / sympy.Integer(2)) * gamma * sigma**2 * tau
    r_sym = s - q * gamma * sigma**2 * tau

    c_sym = (sympy.Integer(2) / gamma) * sympy.log(1 + gamma / k_sym)
    psi_sym = gamma * sigma**2 * tau + c_sym

    delta_a_sym = (
        ((1 - 2 * q) / sympy.Integer(2)) * gamma * sigma**2 * tau
        + sympy.log(1 + gamma / k_sym) / gamma
    )
    delta_b_sym = (
        ((1 + 2 * q) / sympy.Integer(2)) * gamma * sigma**2 * tau
        + sympy.log(1 + gamma / k_sym) / gamma
    )

    # ------------------------------------------------------------------
    # 1 & 2 — Utility-indifference derivation of reservation prices
    #
    # Strategy: introduce a free symbol (r_var), solve the condition that
    # the log|ratio| = 0, and then verify the solution matches our formula.
    # ------------------------------------------------------------------
    r_var = sympy.Symbol("r_var", real=True)

    # Ask side: v(x + r_var, s, q-1, t) = v(x, s, q, t)
    #           <=> log|v(x+r_var,s,q-1,t)| - log|v(x,s,q,t)| = 0
    ask_exponent_diff = sympy.expand(
        _exponent(_v(x + r_var, s, q - 1, t)) - _exponent(_v(x, s, q, t))
    )
    solved_r_a = sympy.solve(ask_exponent_diff, r_var)[0]
    indifference_ask_check = bool(sympy.simplify(solved_r_a - r_a_sym) == sympy.Integer(0))

    # Bid side: v(x - r_var, s, q+1, t) = v(x, s, q, t)
    bid_exponent_diff = sympy.expand(
        _exponent(_v(x - r_var, s, q + 1, t)) - _exponent(_v(x, s, q, t))
    )
    solved_r_b = sympy.solve(bid_exponent_diff, r_var)[0]
    indifference_bid_check = bool(sympy.simplify(solved_r_b - r_b_sym) == sympy.Integer(0))

    # ------------------------------------------------------------------
    # 3 — Reservation mid-price is the mean of ask and bid
    # ------------------------------------------------------------------
    r_avg_check = bool(
        sympy.simplify((r_a_sym + r_b_sym) / sympy.Integer(2) - r_sym) == sympy.Integer(0)
    )

    # ------------------------------------------------------------------
    # 4 — Half-spreads sum to total spread
    # ------------------------------------------------------------------
    delta_sum_check = bool(
        sympy.simplify(delta_a_sym + delta_b_sym - psi_sym) == sympy.Integer(0)
    )

    # ------------------------------------------------------------------
    # 5 — Spread constant limit as gamma -> 0
    # ------------------------------------------------------------------
    gamma_zero_limit = sympy.limit(c_sym, gamma, 0)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    _print_results(
        solved_r_a=solved_r_a,
        r_a_sym=r_a_sym,
        solved_r_b=solved_r_b,
        r_b_sym=r_b_sym,
        indifference_ask_check=indifference_ask_check,
        indifference_bid_check=indifference_bid_check,
        r_avg_check=r_avg_check,
        delta_sum_check=delta_sum_check,
        gamma_zero_limit=gamma_zero_limit,
        k_sym=k_sym,
    )

    return {
        "indifference_ask_check": indifference_ask_check,
        "indifference_bid_check": indifference_bid_check,
        "r_avg_check": r_avg_check,
        "delta_sum_check": delta_sum_check,
        "gamma_zero_limit": gamma_zero_limit,
    }


def _print_results(
    solved_r_a, r_a_sym,
    solved_r_b, r_b_sym,
    indifference_ask_check, indifference_bid_check,
    r_avg_check, delta_sum_check,
    gamma_zero_limit, k_sym,
) -> None:
    sep = "=" * 62
    print(sep)
    print("Avellaneda-Stoikov Symbolic Verification")
    print(sep)

    print("\n[1] Reservation ask — indifference condition")
    print(f"    Solved r_a  : {solved_r_a}")
    print(f"    Formula r_a : {r_a_sym}")
    print(f"    Match       : {indifference_ask_check}")

    print("\n[2] Reservation bid — indifference condition")
    print(f"    Solved r_b  : {solved_r_b}")
    print(f"    Formula r_b : {r_b_sym}")
    print(f"    Match       : {indifference_bid_check}")

    print("\n[3] Reservation mid-price: r = (r_a + r_b) / 2")
    print(f"    Check       : {r_avg_check}")

    print("\n[4] Half-spread sum: delta_a + delta_b = psi")
    print(f"    Check       : {delta_sum_check}")

    print("\n[5] Spread constant limit: lim(gamma->0) (2/gamma)*ln(1+gamma/k)")
    print(f"    Limit value : {gamma_zero_limit}")
    print(f"    Equals 2/k  : {sympy.simplify(gamma_zero_limit - 2 / k_sym) == sympy.Integer(0)}")

    print(f"\n{sep}")
    all_ok = all([
        indifference_ask_check, indifference_bid_check,
        r_avg_check, delta_sum_check,
        sympy.simplify(gamma_zero_limit - 2 / k_sym) == sympy.Integer(0),
    ])
    print(f"All checks passed: {all_ok}")
    print(sep)


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_symbolic_checks()
