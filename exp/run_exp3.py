"""
Exp 3: Gamma-to-zero convergence and benchmark ambiguity study.
Extension of Avellaneda-Stoikov replication.

Run from the project root:
    python -m exp.run_exp3
or
    python exp/run_exp3.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.config import GAMMA_GRID_EXT, N_PATHS, SIGMA, K, A, T, DT, S0
from src.convergence import (
    run_convergence_experiment,
    plot_convergence,
    check_spread_constant_limit,
)

RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)


def main() -> None:
    print('=== Exp 3: Gamma Convergence and Benchmark Ambiguity ===')

    # 1. Verify spread constant limit  2/γ · log1p(γ/k) → 2/k as γ → 0
    limit_val = check_spread_constant_limit(k=K)
    print(f'  2/gamma * log1p(gamma/k) at gamma=1e-8: {limit_val:.6f} (expected: {2/K:.6f})')
    assert abs(limit_val - 2 / K) < 1e-4, 'Limit check failed'
    print('  Limit check passed.')

    # 2. Run convergence experiment across GAMMA_GRID_EXT
    print(f'\nRunning convergence experiment over γ grid: {GAMMA_GRID_EXT}')
    print(f'  n_paths={N_PATHS}, sigma={SIGMA}, k={K}, A={A}, T={T}, dt={DT}, S0={S0}\n')
    df = run_convergence_experiment(
        gamma_grid=GAMMA_GRID_EXT,
        n_paths=N_PATHS,
        sigma=SIGMA,
        k=K,
        A=A,
        T=T,
        dt=DT,
        S0=S0,
        base_seed=42,
    )

    # 3. Save results table
    out_csv = RESULTS_DIR / 'exp3_convergence_table.csv'
    df.to_csv(out_csv, index=False)
    print(f'\nConvergence table saved to {out_csv}')
    print(df.to_string())

    # 4. Produce and save convergence plots
    plot_convergence(df, save_dir=str(RESULTS_DIR))
    print('\nConvergence plots saved.')


if __name__ == '__main__':
    main()
