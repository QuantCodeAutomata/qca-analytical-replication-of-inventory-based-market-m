"""
Exp 2: Monte Carlo replication of Section 3.3 — inventory-aware vs symmetric quoting.

Runs 1000 paths per gamma scenario for both strategies, compares terminal P&L and
final inventory distributions, and generates representative path plots and histograms.

Run:
    python -m exp.run_exp2
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.config import (
    S0, SIGMA, T, DT, Q0, X0, A, K,
    GAMMA_SCENARIOS, N_PATHS, RANDOM_SEED,
    PAPER_MC_REFERENCE, PAPER_SPREAD_COMPONENTS,
)
from src.simulator import run_monte_carlo
from src.metrics import build_results_table, log_violations, compute_summary
from src.plots import plot_representative_path, plot_all_gammas
from src.formulas import spread_constant

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def run_scenario(gamma: float) -> dict:
    """Run inventory + symmetric Monte Carlo for one gamma value.

    Returns dict with paths and terminal arrays for both strategies.
    """
    print(f"\n  --- gamma={gamma} ---")

    paths_inv, profits_inv, qs_inv = run_monte_carlo(
        strategy="inventory",
        gamma=gamma,
        n_paths=N_PATHS,
        sigma=SIGMA,
        k=K,
        A=A,
        T=T,
        dt=DT,
        S0=S0,
        q0=Q0,
        X0=X0,
        base_seed=RANDOM_SEED,
    )
    log_violations(paths_inv, gamma, "inventory")

    paths_sym, profits_sym, qs_sym = run_monte_carlo(
        strategy="symmetric",
        gamma=gamma,
        n_paths=N_PATHS,
        sigma=SIGMA,
        k=K,
        A=A,
        T=T,
        dt=DT,
        S0=S0,
        q0=Q0,
        X0=X0,
        base_seed=RANDOM_SEED,  # same seed → CRN with inventory strategy
    )
    log_violations(paths_sym, gamma, "symmetric")

    # Summary statistics
    c = spread_constant(gamma, K)
    inv_stats = compute_summary(profits_inv, qs_inv, c, "inventory", gamma)
    sym_stats = compute_summary(profits_sym, qs_sym, c, "symmetric", gamma)

    print(f"    [Inventory ] mean_profit={inv_stats['mean_profit']:7.2f} "
          f"std_profit={inv_stats['std_profit']:6.2f} "
          f"mean_q={inv_stats['mean_q']:6.3f} std_q={inv_stats['std_q']:5.2f}")
    print(f"    [Symmetric ] mean_profit={sym_stats['mean_profit']:7.2f} "
          f"std_profit={sym_stats['std_profit']:6.2f} "
          f"mean_q={sym_stats['mean_q']:6.3f} std_q={sym_stats['std_q']:5.2f}")

    return {
        "paths_inv": paths_inv,
        "paths_sym": paths_sym,
        "profits_inv": profits_inv,
        "profits_sym": profits_sym,
        "qs_inv": qs_inv,
        "qs_sym": qs_sym,
        "inv_stats": inv_stats,
        "sym_stats": sym_stats,
    }


def compare_with_paper(gamma: float, inv_stats: dict, sym_stats: dict) -> pd.DataFrame:
    """Compute difference between simulated and paper reference values."""
    ref = PAPER_MC_REFERENCE[gamma]
    rows = []
    for strat, stats in [("inventory", inv_stats), ("symmetric", sym_stats)]:
        r = ref[strat]
        rows.append({
            "gamma": gamma,
            "strategy": strat,
            "sim_mean_profit": stats["mean_profit"],
            "paper_mean_profit": r["mean_profit"],
            "sim_std_profit": stats["std_profit"],
            "paper_std_profit": r["std_profit"],
            "sim_mean_q": stats["mean_q"],
            "paper_mean_q": r["mean_q"],
            "sim_std_q": stats["std_q"],
            "paper_std_q": r["std_q"],
        })
    return pd.DataFrame(rows)


def validate_qualitative_properties(all_results: dict) -> None:
    """Assert the paper's qualitative findings hold."""
    print("\n=== Qualitative Property Checks ===")
    for gamma, res in all_results.items():
        inv_std_p = np.std(res["profits_inv"])
        sym_std_p = np.std(res["profits_sym"])
        inv_std_q = np.std(res["qs_inv"])
        sym_std_q = np.std(res["qs_sym"])

        prop_profit_ok = inv_std_p < sym_std_p
        prop_q_ok      = inv_std_q < sym_std_q

        print(f"  gamma={gamma}: std(profit) inv={inv_std_p:.2f} < sym={sym_std_p:.2f} → "
              f"{'PASS' if prop_profit_ok else 'FAIL'}")
        print(f"  gamma={gamma}: std(q)      inv={inv_std_q:.2f} < sym={sym_std_q:.2f} → "
              f"{'PASS' if prop_q_ok else 'FAIL'}")


def main() -> None:
    print("=" * 60)
    print("Exp 2: Monte Carlo Replication — Section 3.3")
    print("=" * 60)

    all_results = {}
    comparison_dfs = []

    for gamma in GAMMA_SCENARIOS:
        res = run_scenario(gamma)
        all_results[gamma] = res
        cmp_df = compare_with_paper(gamma, res["inv_stats"], res["sym_stats"])
        comparison_dfs.append(cmp_df)

    # Build and save summary table
    build_input = {
        g: {
            "inventory": (res["profits_inv"], res["qs_inv"]),
            "symmetric": (res["profits_sym"], res["qs_sym"]),
        }
        for g, res in all_results.items()
    }
    summary_df = build_results_table(build_input)
    summary_df.to_csv(RESULTS_DIR / "exp2_summary_table.csv")
    print("\n=== Monte Carlo Summary Table ===")
    print(summary_df.to_string())

    # Comparison with paper reference values
    cmp_all = pd.concat(comparison_dfs, ignore_index=True)
    cmp_all.to_csv(RESULTS_DIR / "exp2_paper_comparison.csv", index=False)
    print("\n=== Comparison with Paper Reference ===")
    print(cmp_all.to_string(index=False))

    # Qualitative validation
    validate_qualitative_properties(all_results)

    # --- Plots ---
    # 1. Representative path for gamma=0.1 (first path, deterministic seed)
    rep_path = all_results[0.1]["paths_inv"][0]
    plot_representative_path(
        rep_path,
        gamma=0.1,
        dt=DT,
        save_path=str(RESULTS_DIR / "exp2_representative_path_gamma01.png"),
    )
    print("\n  Representative path figure saved.")

    # 2. Profit histograms for each gamma
    hist_input = {
        g: {
            "inventory": res["profits_inv"],
            "symmetric": res["profits_sym"],
        }
        for g, res in all_results.items()
    }
    plot_all_gammas(hist_input, save_dir=str(RESULTS_DIR))
    print("  Profit histogram figures saved.")

    print("\n[PASS] Exp 2 completed successfully.")


if __name__ == "__main__":
    main()
