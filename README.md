# Analytical Replication: Inventory-Based Market Making (Avellaneda-Stoikov)

Replication and validation of the finite-horizon Avellaneda-Stoikov market-making framework, covering:

- **Exp 1**: Symbolic and numerical verification of all closed-form reservation-price and optimal-quote formulas.
- **Exp 2**: Monte Carlo simulation comparing inventory-aware quoting versus a symmetric benchmark (1 000 paths × 3 risk-aversion scenarios).
- **Exp 3**: Convergence analysis studying the γ → 0 limit and sensitivity to benchmark-spread definition.

## Repository Structure

```
data/                   # Placeholder (no external data required)
src/
  config.py             # All model parameters (S0, σ, T, A, k, γ scenarios)
  formulas.py           # Closed-form numpy functions (v, r_a, r_b, r, ψ, δ^a, δ^b)
  symbolic_checks.py    # SymPy derivations and symbolic identity verification
  strategies.py         # inventory_quotes(), symmetric_quotes(), fill_intensities()
  simulator.py          # simulate_path(), run_monte_carlo(), PathData dataclass
  metrics.py            # Summary statistics and results table builder
  plots.py              # Path figure and profit histogram generation
  benchmark_variants.py # Benchmark A (constant spread) and Benchmark B (time-varying)
  convergence.py        # Gamma-grid convergence experiment and plots
exp/
  run_exp1.py           # Analytical formula validation runner
  run_exp2.py           # Monte Carlo replication runner
  run_exp3.py           # Convergence extension runner
tests/
  test_formulas.py      # Pytest suite for all closed-form formulas (14 test functions)
  test_simulator.py     # Pytest suite for simulation engine (16 test functions)
results/
  RESULTS.md            # Summary of all experiment findings
  exp1_*.csv            # Formula spot-check tables
  exp2_*.csv            # MC summary and paper comparison
  exp3_*.csv            # Convergence tables
  *.png                 # Figures
```

## Model Parameters (Paper Section 3.3)

| Parameter | Value | Description |
|-----------|-------|-------------|
| S₀        | 100   | Initial mid-price |
| σ         | 2     | Arithmetic Brownian volatility |
| T         | 1     | Horizon |
| dt        | 0.005 | Time step (N = 200 steps) |
| A         | 140   | Intensity scale λ(δ) = A·exp(−k·δ) |
| k         | 1.5   | Intensity decay rate |
| γ         | 0.1, 0.01, 0.5 | Risk-aversion scenarios |

## Key Formulas

```
r(s,q,t)   = s − q·γ·σ²·(T−t)                          # reservation price
ψ(t)       = γ·σ²·(T−t) + (2/γ)·ln(1 + γ/k)            # optimal spread
p^a        = r + ψ/2                                      # ask quote
p^b        = r − ψ/2                                      # bid quote
δ^a        = ((1−2q)/2)·γ·σ²·(T−t) + (1/γ)·ln(1+γ/k)
δ^b        = ((1+2q)/2)·γ·σ²·(T−t) + (1/γ)·ln(1+γ/k)
```

## Running the Experiments

```bash
# Install dependencies (if needed)
pip install sympy numpy pandas matplotlib scipy

# Run all tests
pytest tests/ -v

# Run individual experiments
python -m exp.run_exp1
python -m exp.run_exp2
python -m exp.run_exp3
```

## Library Choices (Context7 Verified)

- `sympy` — symbolic derivation and identity verification (Context7: `/sympy/sympy`)
- `numpy` — numerical evaluation, Monte Carlo simulation (Context7: `/numpy/numpy`)
- `pandas` — results tables
- `matplotlib` — figures (Agg backend for headless environments)

No specialized quant finance package required; all model logic is custom as the AS model is a bespoke continuous-time framework.
