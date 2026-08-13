import numpy as np
import pandas as pd

from src.simulator import PathData
from src.config import K


def compute_summary(
    profits: np.ndarray,
    qs: np.ndarray,
    spread: float,
    strategy_name: str,
    gamma: float,
) -> dict:
    """Compute mean/std profit and mean/std final inventory."""
    return {
        "gamma": gamma,
        "strategy": strategy_name,
        "spread": spread,
        "mean_profit": float(np.mean(profits)),
        "std_profit": float(np.std(profits)),
        "mean_q": float(np.mean(qs)),
        "std_q": float(np.std(qs)),
    }


def build_results_table(results: dict) -> pd.DataFrame:
    """
    Build comparison DataFrame indexed by (gamma, strategy).

    results structure:
        {gamma: {'inventory': (profits, qs), 'symmetric': (profits, qs)}}
    """
    rows = []
    for gamma, strategies in results.items():
        # Constant-spread component is the same for both strategies at a given gamma
        spread = 2.0 / gamma * np.log1p(gamma / K)
        for strategy_name, (profits, qs) in strategies.items():
            row = compute_summary(profits, qs, spread, strategy_name, gamma)
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.set_index(["gamma", "strategy"])
    return df


def log_violations(paths: list[PathData], gamma: float, strategy: str) -> None:
    """Log total lambda*dt > 1 violations across all paths."""
    total = sum(p.violations for p in paths)
    print(
        f"[violations] gamma={gamma:.4f}  strategy={strategy:<12s}"
        f"  total_violations={total}  paths={len(paths)}"
    )
