"""
Exp 1: Analytical replication of Avellaneda-Stoikov inventory-based market-making formulas.

Validates the finite-horizon reservation-price and optimal-quote formulas using
sympy for symbolic derivations and numpy for numerical spot-checks.

Run:
    python -m exp.run_exp1
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.config import (
    S0, SIGMA, T, A, K,
    GAMMA_SCENARIOS, PAPER_SPREAD_COMPONENTS,
)
from src.formulas import (
    reservation_ask, reservation_bid, reservation_price,
    spread_constant, total_spread, delta_ask, delta_bid,
    ask_quote, bid_quote, inactive_value,
)
from src.symbolic_checks import run_symbolic_checks

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def _fmt(val: float, decimals: int = 6) -> str:
    return f"{val:.{decimals}f}"


def run_symbolic_validation() -> dict:
    """Run all SymPy symbolic derivations and report pass/fail."""
    print("\n=== Symbolic Derivations (SymPy) ===")
    results = run_symbolic_checks()
    for name, val in results.items():
        status = "PASS" if val else "FAIL"
        print(f"  [{status}] {name}: {val}")
    return results


def run_formula_spot_checks() -> pd.DataFrame:
    """Numerical spot-checks of the main analytical formulas."""
    print("\n=== Numerical Formula Spot-Checks ===")

    s = S0
    t = 0.0           # start of horizon → tau = T - t = T = 1
    tau = T - t
    q_vals = [-2, -1, 0, 1, 2]
    rows = []

    for gamma in GAMMA_SCENARIOS:
        for q in q_vals:
            r_a  = reservation_ask(s, q, gamma, SIGMA, T, t)
            r_b  = reservation_bid(s, q, gamma, SIGMA, T, t)
            r    = reservation_price(s, q, gamma, SIGMA, T, t)
            psi  = total_spread(gamma, SIGMA, K, T, t)
            da   = delta_ask(q, gamma, SIGMA, K, T, t)
            db   = delta_bid(q, gamma, SIGMA, K, T, t)
            pa   = ask_quote(s, q, gamma, SIGMA, K, T, t)
            pb   = bid_quote(s, q, gamma, SIGMA, K, T, t)

            # Identity checks
            r_avg_ok      = np.isclose(r, (r_a + r_b) / 2, rtol=1e-10)
            delta_sum_ok  = np.isclose(da + db, psi, rtol=1e-10)
            ask_above_bid = pa > pb

            rows.append({
                "gamma": gamma, "q": q,
                "r_a": r_a, "r_b": r_b, "r": r,
                "spread_psi": psi, "da": da, "db": db,
                "p_ask": pa, "p_bid": pb,
                "r_avg_ok": r_avg_ok,
                "delta_sum_ok": delta_sum_ok,
                "ask_above_bid": ask_above_bid,
            })

    df = pd.DataFrame(rows)
    n_pass = (df["r_avg_ok"] & df["delta_sum_ok"] & df["ask_above_bid"]).sum()
    print(f"  Identity checks passed: {n_pass}/{len(df)}")
    return df


def run_spread_constant_checks() -> pd.DataFrame:
    """Verify constant spread components against paper's reported values."""
    print("\n=== Spread Constant Spot-Checks (vs Paper) ===")
    rows = []
    for gamma in GAMMA_SCENARIOS:
        c    = spread_constant(gamma, K)
        ref  = PAPER_SPREAD_COMPONENTS[gamma]
        diff = abs(c - ref)
        ok   = diff < 0.1
        status = "PASS" if ok else "FAIL"
        print(f"  gamma={gamma:.3f}: c={c:.4f}, paper={ref:.2f}, diff={diff:.4f} [{status}]")
        rows.append({"gamma": gamma, "computed": c, "paper_ref": ref, "diff": diff, "pass": ok})
    return pd.DataFrame(rows)


def run_gamma_zero_limit_check() -> pd.DataFrame:
    """Verify convergence of spread_constant to 2/k as gamma -> 0."""
    print("\n=== Gamma -> 0 Limit Check ===")
    gamma_vals = [0.1, 0.01, 0.001, 1e-6, 1e-8]
    expected = 2.0 / K
    rows = []
    for g in gamma_vals:
        c = spread_constant(g, K)
        diff = abs(c - expected)
        print(f"  gamma={g:.1e}: c={c:.8f}, expected={expected:.8f}, diff={diff:.2e}")
        rows.append({"gamma": g, "c": c, "expected": expected, "diff": diff})
    return pd.DataFrame(rows)


def run_indifference_verification() -> pd.DataFrame:
    """Verify v(x+r_a, s, q-1, t) == v(x, s, q, t) and similarly for bid."""
    print("\n=== Utility Indifference Verification ===")
    x, s, t = 50.0, S0, 0.0
    q_vals = [-2, -1, 0, 1, 2]
    rows = []
    for gamma in GAMMA_SCENARIOS:
        for q in q_vals:
            r_a = reservation_ask(s, q, gamma, SIGMA, T, t)
            r_b = reservation_bid(s, q, gamma, SIGMA, T, t)

            v0    = inactive_value(x, s, q,   gamma, SIGMA, T, t)
            v_ask = inactive_value(x + r_a, s, q - 1, gamma, SIGMA, T, t)
            v_bid = inactive_value(x - r_b, s, q + 1, gamma, SIGMA, T, t)

            ask_ok = np.isclose(v_ask, v0, rtol=1e-9)
            bid_ok = np.isclose(v_bid, v0, rtol=1e-9)
            rows.append({
                "gamma": gamma, "q": q,
                "ask_indiff_ok": ask_ok, "bid_indiff_ok": bid_ok,
            })

    df = pd.DataFrame(rows)
    n_ask = df["ask_indiff_ok"].sum()
    n_bid = df["bid_indiff_ok"].sum()
    print(f"  Ask indifference: {n_ask}/{len(df)} PASS")
    print(f"  Bid indifference: {n_bid}/{len(df)} PASS")
    return df


def save_results(
    sym_results: dict,
    spot_df: pd.DataFrame,
    spread_df: pd.DataFrame,
    limit_df: pd.DataFrame,
    indiff_df: pd.DataFrame,
) -> None:
    """Save all exp_1 results to results/ directory."""
    spot_df.to_csv(RESULTS_DIR / "exp1_formula_spot_checks.csv", index=False)
    spread_df.to_csv(RESULTS_DIR / "exp1_spread_constants.csv", index=False)
    limit_df.to_csv(RESULTS_DIR / "exp1_gamma_zero_limit.csv", index=False)
    indiff_df.to_csv(RESULTS_DIR / "exp1_indifference_check.csv", index=False)
    print("\n  Results saved to results/exp1_*.csv")


def main() -> None:
    print("=" * 60)
    print("Exp 1: Avellaneda-Stoikov Analytical Formula Validation")
    print("=" * 60)

    sym_results  = run_symbolic_validation()
    spot_df      = run_formula_spot_checks()
    spread_df    = run_spread_constant_checks()
    limit_df     = run_gamma_zero_limit_check()
    indiff_df    = run_indifference_verification()

    save_results(sym_results, spot_df, spread_df, limit_df, indiff_df)

    # Summary assertion: all symbolic checks pass
    assert all(sym_results.values()), "Some symbolic checks FAILED"
    assert spread_df["pass"].all(), "Some spread constant checks FAILED"
    print("\n[PASS] All exp_1 validations completed successfully.")


if __name__ == "__main__":
    main()
