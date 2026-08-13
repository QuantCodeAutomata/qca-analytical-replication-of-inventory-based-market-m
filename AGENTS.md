# Avellaneda-Stoikov Market-Making Replication — Agent Notes

## Project Structure

```
src/
  config.py          # Shared model parameters (S0, SIGMA, T, K, A, GAMMA_SCENARIOS, …)
  formulas.py        # exp_1: Closed-form AS analytical formulas (numpy)
  symbolic_checks.py # exp_1: SymPy derivation & verification of all formulas
tests/
  test_formulas.py   # Pytest suite for formulas.py (112 tests, all pass)
exp/                 # Experiment modules (exp_2, exp_3, …)
data/, results/      # Data and output artifacts
```

## Key Model Facts (Avellaneda-Stoikov 2008)

- **Mid-price**: driftless arithmetic Brownian motion  σ dW_t
- **Utility**: CARA  u(W) = −exp(−γW)
- **Intensities**: λ(δ) = A·exp(−k·δ)   (exponential, parametrised by A and k)

### Core Formulas (`src/formulas.py`)

| Symbol | Formula |
|--------|---------|
| `inactive_value` | −exp(−γx)·exp(−γqs)·exp(γ²q²σ²τ/2),  τ=T−t |
| `reservation_ask` | s + ((1−2q)/2)·γσ²τ |
| `reservation_bid` | s + ((−1−2q)/2)·γσ²τ |
| `reservation_price` | s − q·γσ²τ   (mean of ask & bid) |
| `spread_constant` | (2/γ)·ln(1+γ/k)  — uses `np.log1p(γ/k)` for stability |
| `total_spread` (ψ) | γσ²τ + spread_constant |
| `delta_ask` | ((1−2q)/2)·γσ²τ + (1/γ)·ln(1+γ/k) |
| `delta_bid` | ((1+2q)/2)·γσ²τ + (1/γ)·ln(1+γ/k) |
| `ask_quote` | r + ψ/2 |
| `bid_quote` | r − ψ/2 |

Key identities:
- δ^a + δ^b = ψ  (always)
- r = (r^a + r^b)/2  (always)
- lim(γ→0) spread_constant = 2/k
- Paper spread values: γ=0.1 → 1.29,  γ=0.01 → 1.33,  γ=0.5 → 1.15  (k=1.5)

### Indifference Conditions

Ask (sell 1 unit): v(x+r_a, s, q−1, t) = v(x, s, q, t)
Bid (buy  1 unit): v(x−r_b, s, q+1, t) = v(x, s, q, t)

These hold exactly; tested in `test_formulas.py` for all (γ, q) combinations.

## Symbolic Checks (`src/symbolic_checks.py`)

`run_symbolic_checks()` → dict with 5 keys, all verified True:

- `indifference_ask_check`: sympy.solve confirms r_a formula
- `indifference_bid_check`: sympy.solve confirms r_b formula
- `r_avg_check`: r = (r_a+r_b)/2 identity
- `delta_sum_check`: δ^a+δ^b = ψ identity
- `gamma_zero_limit`: sympy.limit(c, γ, 0) = 2/k (SymPy expression)

Run as a script: `python -m src.symbolic_checks`

## Testing

```bash
pytest tests/test_formulas.py -v   # 112 tests, < 1 s
```

Parameters used: s0=100, T=1, σ=2, k=1.5, γ∈{0.1, 0.01, 0.5}, q∈{−2,−1,0,1,2}.
All assertions use `np.isclose(rtol=1e-6)` except paper-value spot-checks (`atol=0.1`).

## Dependencies

- numpy (formulas & tests)
- sympy (symbolic_checks)
- pandas (metrics table)
- matplotlib (plots — Agg backend, non-interactive)
- pytest (tests)

---

## exp_2 — Monte Carlo Simulation Engine

### New modules (all under `src/`)

| File | Purpose |
|------|---------|
| `strategies.py` | `inventory_quotes`, `symmetric_quotes`, `fill_intensities` |
| `simulator.py`  | `PathData` dataclass, `simulate_path`, `run_monte_carlo` |
| `metrics.py`    | `compute_summary`, `build_results_table`, `log_violations` |
| `plots.py`      | `plot_representative_path`, `plot_profit_histograms`, `plot_all_gammas` |

Tests: `tests/test_simulator.py` — 16 tests, all pass (`pytest tests/test_simulator.py -v`).

### Simulation Chronology (critical — one step i, t_i = i·dt)
1. Compute quotes from state (S[i], q[i], X[i])
2. Compute λ^a = A·exp(−k·δ_a),  λ^b = A·exp(−k·δ_b)
3. Log violation if λ·dt > 1 — do NOT clip probabilities
4. Draw fills: u_a, u_b ~ U(0,1);  I^a = 1{u_a < λ^a·dt};  I^b = 1{u_b < λ^b·dt}
5. X[i+1] = X[i] + I^a·p_ask − I^b·p_bid;  q[i+1] = q[i] − I^a + I^b
6. u_s ~ U(0,1);  eps = 1 if u_s < 0.5 else −1;  S[i+1] = S[i] + σ·√dt·eps

### PathData Array Sizes (N = int(T/dt) = 200)
- `S`, `q`, `X` : length **N+1**  (t=0 initial through t=T terminal)
- `r`, `p_ask`, `p_bid`, `delta_ask`, `delta_bid`,
  `lambda_a`, `lambda_b`, `ask_fills`, `bid_fills` : length **N**

### Common Random Numbers (CRN)
`run_monte_carlo` uses `seed = base_seed + i` for path i (fixed regardless of strategy).
Calling with the same `base_seed` for both 'inventory' and 'symmetric' gives
identical S-paths, isolating strategy effect. Three uniforms consumed per step
(u_a, u_b, u_s) in fixed order — order must not change between strategies.

### Inventory strategy formulas
- τ = T − t
- r = s − q·γ·σ²·τ
- δ_a = (1−2q)/2·γσ²τ + (1/γ)·ln(1+γ/k)
- δ_b = (1+2q)/2·γσ²τ + (1/γ)·ln(1+γ/k)
- ψ = γσ²τ + (2/γ)·ln(1+γ/k)
- p_ask = s + δ_a  (= r + ψ/2),  p_bid = s − δ_b  (= r − ψ/2)

### Symmetric strategy formulas
- c = (2/γ)·ln(1+γ/k)   — use `np.log1p(γ/k)` for numerical stability
- δ_ask = δ_bid = c/2,  p_ask = s + c/2,  p_bid = s − c/2

### Key result (paper Table 1)
Inventory strategy achieves significantly lower std(profit) than symmetric,
especially at high γ (γ=0.5 inventory σ≈5 vs symmetric σ≈14).
Terminal profit = X_T + q_T · S_T.

---

## exp_3 — Gamma-to-zero Convergence and Benchmark Ambiguity

### New modules

| File | Purpose |
|------|---------|
| `src/benchmark_variants.py` | Benchmark A & B quote functions; `compute_spread_constant`; `quote_center_distance` |
| `src/convergence.py`        | `run_convergence_experiment`, `plot_convergence`, `check_spread_constant_limit` |
| `exp/run_exp3.py`           | Experiment runner (limit check → MC → CSV → plots) |

### Benchmark strategies

**Benchmark A** — constant spread, centred at mid-price:
- `c_γ = 2/γ · log1p(γ/k)`
- `p^a = S + c_γ/2`, `p^b = S − c_γ/2`
- Identical to the paper's "symmetric benchmark" (`symmetric_quotes`).

**Benchmark B** — time-varying spread ψ_t, centred at mid-price (not r_t):
- `ψ_t = γσ²(T−t) + 2/γ·log1p(γ/k)`
- `p^a = S + ψ_t/2`, `p^b = S − ψ_t/2`
- Same spread as inventory strategy but no inventory shift of the quote centre.

### Convergence experiment design

- Sweeps `GAMMA_GRID_EXT = [0.5, 0.1, 0.01, 0.005, 0.001, 0.0001]`
- **CRN**: path i uses `np.random.default_rng(base_seed + i)` for all three strategies.
- Local `_simulate_path_bA` / `_simulate_path_bB` in `convergence.py` mirror
  the exact chronology and random-draw order of `simulator.py`:
  `u_a = rng.random()` → `u_b = rng.random()` → `u_s = rng.random()` per step.
- Fill model: Bernoulli `P(fill) = λ·dt` (matches `simulator.py`).
- Mid-price: binary random walk `eps ∈ {−1,+1}` (matches `simulator.py`).
- `quote_center_diff_sum` extracted from `PathData.r` and `PathData.S[:-1]`.

### Outputs (written to `results/`)

- `exp3_convergence_table.csv` — one row per γ with 19 metric columns
- `exp3_profit_std_diff.png`  — |σ(π_inv)−σ(π_bX)| vs γ (log-scale)
- `exp3_inv_std_diff.png`     — |σ(q_inv)−σ(q_bX)| vs γ (log-scale)
- `exp3_quote_center_diff.png`— mean |r_t−S_t| vs γ (log-scale)

### Limit check

`check_spread_constant_limit(k, gamma_small=1e-8)` numerically verifies
`2/γ·log1p(γ/k) → 2/k` as `γ→0`; tolerance `< 1e-4`.
