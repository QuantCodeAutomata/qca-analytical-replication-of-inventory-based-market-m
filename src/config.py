"""
Shared model parameters for Avellaneda-Stoikov market-making replication.
All values taken directly from the paper (Section 3.3 unless noted).
"""

# ----- Mid-price process -----
S0: float = 100.0       # Initial mid-price
SIGMA: float = 2.0      # Constant volatility (arithmetic Brownian motion)

# ----- Horizon / time grid -----
T: float = 1.0          # Trading horizon
DT: float = 0.005       # Time step  →  N = T/DT = 200 steps
N_STEPS: int = int(T / DT)  # 200

# ----- Initial state -----
Q0: int = 0             # Initial inventory
X0: float = 0.0         # Initial cash wealth

# ----- Intensity parameters (exponential intensities) -----
A: float = 140.0        # Intensity scale  λ(δ) = A·exp(−k·δ)
K: float = 1.5          # Intensity decay rate

# ----- Monte Carlo -----
N_PATHS: int = 1000     # Paths per scenario
RANDOM_SEED: int = 42   # Base seed for reproducibility

# ----- Risk-aversion scenarios from the paper -----
GAMMA_SCENARIOS: list[float] = [0.1, 0.01, 0.5]

# ----- Extension experiment gamma grid (exp_3) -----
GAMMA_GRID_EXT: list[float] = [0.5, 0.1, 0.01, 0.005, 0.001, 0.0001]

# ----- Paper reference values for numerical spot-checks (exp_1) -----
# Constant spread component  2/γ · ln(1 + γ/k)
PAPER_SPREAD_COMPONENTS: dict[float, float] = {
    0.1:  1.29,
    0.01: 1.33,
    0.5:  1.15,
}

# ----- Paper Monte Carlo reference table (exp_2) -----
# Structure: gamma → {strategy: {metric: value}}
PAPER_MC_REFERENCE: dict = {
    0.1: {
        "inventory": {
            "spread": 1.29, "mean_profit": 62.94, "std_profit": 5.89,
            "mean_q": 0.10, "std_q": 2.80,
        },
        "symmetric": {
            "spread": 1.29, "mean_profit": 67.21, "std_profit": 13.43,
            "mean_q": -0.018, "std_q": 8.66,
        },
    },
    0.01: {
        "inventory": {
            "spread": 1.33, "mean_profit": 66.78, "std_profit": 8.76,
            "mean_q": -0.02, "std_q": 4.70,
        },
        "symmetric": {
            "spread": 1.33, "mean_profit": 67.36, "std_profit": 13.40,
            "mean_q": -0.31, "std_q": 8.65,
        },
    },
    0.5: {
        "inventory": {
            "spread": 1.15, "mean_profit": 33.92, "std_profit": 4.72,
            "mean_q": -0.02, "std_q": 1.88,
        },
        "symmetric": {
            "spread": 1.15, "mean_profit": 66.20, "std_profit": 14.53,
            "mean_q": 0.25, "std_q": 9.06,
        },
    },
}
