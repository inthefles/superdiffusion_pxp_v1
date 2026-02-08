# PXP Transport Simulation Workflow

This document describes the complete workflow of the PXP energy transport simulation code.

## Overview

The code simulates **superdiffusive energy transport** in the PXP (Rydberg blockade) model using Time-Evolving Block Decimation (TEBD) with tensor networks. It computes energy-energy autocorrelation functions to extract the dynamical exponent $z$ that characterizes transport regimes.

**Reference:** Ljubotina et al., Phys. Rev. X 13, 011033 (2023)

---

## Workflow Pipeline

### 1. Initialization (`hilbert.jl`)

**Purpose:** Set up the constrained Hilbert space with Rydberg blockade constraint.

**Key Functions:**
- `PXPSites(N)` → Create ITensor site indices for N spins
- `create_merged_sites(N)` → Merge pairs of sites (2i-1, 2i) into merged sites

**Physical Constraint:**
- No two adjacent sites can both be excited: $|\uparrow\uparrow\rangle$ is forbidden
- Valid Hilbert space dimension grows as Fibonacci: $D(N) = F_{N+2}$

**Site Merging Strategy:**
```
Original sites:  1  2  3  4  5  6  ...  N
                 └──┘  └──┘  └──┘
Merged sites:      1    2    3    ...  N/2
```
- Each merged site has dimension 3 (only valid 2-site configurations)
- Converts 3-site PXP terms → 2-site gates for TEBD efficiency

**Output:**
- `MergedSiteInfo`: Contains merged site indices and mapping information

---

### 2. Operator Construction (`operators.jl`)

**Purpose:** Build the local energy density operator for correlation measurements.

**Key Functions:**
- `center_energy_density_merged(sites; Ω)` → Energy density MPO at center

**Energy Density Operator:**
$$h_l = \Omega P_{l-1} \sigma^x_l P_{l+1}$$

where $P_i = \frac{1 - \sigma^z_i}{2}$ projects onto $|\downarrow\rangle$ state.

**MPO Representation:**
- Represented as Matrix Product Operator (MPO)
- Bond dimension $\chi = 4$ for merged sites
- Supports variants: PNP ($h_l^{\text{PNP}}$), PNPNP ($h_l^{\text{PNPNP}}$)

**Output:**
- `h0::MPO`: Initial energy density operator at center site

---

### 3. Time Evolution (`tebd.jl`)

**Purpose:** Evolve the energy density operator in the Heisenberg picture.

#### 3.1 Evolution Strategy

**Heisenberg Picture:**
$$h(t) = U^\dagger(t) \, h_0 \, U(t)$$

where $U(t) = \exp(-iHt)$ is the time evolution operator.

**Why Heisenberg picture?**
- More efficient than evolving states at infinite temperature
- Directly gives time-dependent operator needed for correlation function

#### 3.2 Trotter Decomposition

**4th-order Trotter splitting:**
$$U(\delta t) \approx \prod_{\text{bonds}} \exp(-i H_{\text{local}} \delta t)$$

with error $O(\delta t^5)$.

**Key Functions:**
- `make_trotter_gates_merged(info, dt, params)` → Build 2-site unitary gates

#### 3.3 Local Hamiltonian

For each pair of adjacent merged sites, construct 9×9 local Hamiltonian:
$$H_{\text{local}} = H_{\text{PXP}} + \lambda H_{\text{PXPZ}} + \delta H_{\text{PNP}}$$

**Model variants (via `TEBDParams`):**
- **PXP:** $\Omega$ (Rabi frequency)
- **PXPZ:** $+ \lambda \sum_i P_{i-1} \sigma^z_i P_{i+1}$ (diagonal field)
- **PNP:** $+ \delta \sum_i n_i$ (chemical potential)
- **PNPNP:** $+ \xi$ (5-site term, not implemented - requires 3-site gates)

#### 3.4 Gate Application

**For each time step:**

1. **Build gates:** $g = \exp(-i H_{\text{local}} \delta t)$
2. **Apply to MPO:** Use ITensors built-in function
   ```julia
   h_new = apply([gate], h_old; apply_dag=true, maxdim=χ, cutoff=ε)
   ```
   - `apply_dag=true`: Implements $U^\dagger h U$ (Heisenberg evolution)
   - `maxdim`: Maximum bond dimension for SVD truncation
   - `cutoff`: SVD threshold for discarding small singular values

3. **Save snapshots:** Store MPO at selected time points

**Key Function:**
- `run_tebd_evolution(info, h0, tmax, params)` → Returns `(times, mpos)`

**Parameters (`TEBDParams`):**
- `dt`: Time step (default 0.05)
- `maxdim` ($\chi$): Bond dimension (128-512 for convergence)
- `cutoff`: SVD truncation threshold ($10^{-10}$)
- `order`: Trotter order (2 or 4)
- `Ω, λ, δ, ξ`: Physical Hamiltonian parameters

**Output:**
- `times::Vector{Float64}`: Time points $[t_0, t_1, \ldots, t_{\text{max}}]$
- `mpos::Vector{MPO}`: Evolved operators $[h(t_0), h(t_1), \ldots]$

---

### 4. Observable Computation (`observables.jl`)

**Purpose:** Extract physical observables from evolved operators.

#### 4.1 Autocorrelation Function

**Definition:**
$$C(t) = \langle h_0(0) h_0(t) \rangle - \langle h_0 \rangle^2$$

where in the Heisenberg picture:
- $h_0(0) = h_0$ is the initial energy density operator
- $h_0(t) = U^\dagger(t) h_0 U(t)$ is the time-evolved operator

For the PXP model:
$$C(t) = \text{Tr}[h_0 \, h(t)] - (\text{Tr}[h_0])^2$$

**Key simplifications:**
1. **Projector already included:** Since $h_0 = \Omega P_{l-1} \sigma^x_l P_{l+1}$ already contains the projectors, we use $\langle ... \rangle = \text{Tr}[...]$ directly
2. **Trace invariance:** For unitary evolution, $\text{Tr}[h(t)] = \text{Tr}[U^\dagger h_0 U] = \text{Tr}[h_0]$ (cyclicity of trace), so $\langle h_0(t) \rangle = \langle h_0 \rangle$
3. **No normalization:** We don't divide by Hilbert space dimension $D$ since projectors already restrict to physical subspace

This is the **connected correlation function** that isolates genuine quantum correlations from mean-field contributions.

**Physical Meaning:**
- Measures how energy-energy correlations spread from initial localized excitation
- Decay characterizes transport: faster decay → faster spreading
- Connected part isolates genuine correlations from trivial mean-field contributions

**Key Functions:**
- `trace_mpo(M)` → Compute $\text{Tr}[M]$ by contracting physical indices
- `inner(A::MPO, B::MPO)` → ITensors built-in function for $\text{Tr}[A^\dagger B]$
- `autocorrelation(h0, ht)` → Compute $C(t) = \text{Tr}[h_0 h_t] - (\text{Tr}[h_0])^2$
- `compute_correlation_function(sites, h0, times, mpos)` → Returns $C(t)$ for all time points

**Implementation notes:**
- Uses ITensors' built-in `inner(A, B)` instead of custom implementation
- Since $h_0$ and $h_t$ are Hermitian, $\text{Tr}[h_0 h_t] = \text{Tr}[h_0^\dagger h_t] = \text{inner}(h_0, h_t)$
- Simplified from original implementation by recognizing projectors are already built into operators

#### 4.2 Dynamical Exponent

**Scaling form:**
$$C(t) \sim t^{-\alpha} \sim t^{-2/z}$$

where $z$ is the dynamical exponent characterizing transport.

**Extraction method:**
$$z^{-1}(t) = -\frac{d \log C}{d \log t}$$

Computed via numerical derivative on log-log plot.

**Key Function:**
- `instantaneous_exponent(times, C)` → Returns $(t_{\text{mid}}, z^{-1})$

**Transport Classification:**
| Exponent | Regime | Physics |
|----------|--------|---------|
| $z = 1$ | Ballistic | Energy spreads linearly: $\langle x^2 \rangle \sim t^2$ |
| $z = 3/2$ | KPZ superdiffusive | Pure PXP model, $\langle x^2 \rangle \sim t^{4/3}$ |
| $z = 2$ | Diffusive | Normal diffusion, $\langle x^2 \rangle \sim t$ |

**Output:**
- `C::Vector{Float64}`: Correlation values at each time
- `z_inv::Vector{Float64}`: Instantaneous exponent $z^{-1}(t)$

---

### 5. Data Management (`io.jl`)

**Purpose:** Package and persist simulation results.

#### 5.1 Result Structure

**`SimulationResult` struct contains:**
- `times`: Time points
- `correlations`: $C(t)$ values
- `exponents`: $z^{-1}(t)$ values
- `exponent_times`: Corresponding time points for exponents
- `params`: Dictionary of all simulation parameters

#### 5.2 File I/O

**Save function:**
```julia
save_simulation(filename, result)
```
Saves to JLD2 format at `data/pxp_transport_MODEL_N64_chi128.jld2`

**Load function:**
```julia
result = load_simulation(filename)
```

**Key Functions:**
- `generate_filename(params)` → Auto-generate descriptive filename
- `print_summary(result)` → Display statistics (mean $z$ for $t > 5$)

**Output:**
- Persistent `.jld2` files for later analysis
- Console summary with transport classification

---

### 6. Visualization (`plot_results.jl`)

**Purpose:** Generate publication-quality plots from saved data.

#### 6.1 Single Simulation Plots

**Correlation function plot:**
- Log-log plot of $C(t)$ vs $t$
- Reference lines: $t^{-4/3}$ (KPZ), $t^{-2}$ (ballistic)

**Exponent plot:**
- $z^{-1}(t)$ vs $t$
- Reference lines: 2/3 (KPZ), 1.0 (ballistic), 0.5 (diffusive)

**Key Functions:**
- `plot_correlation(result)` → Correlation decay plot
- `plot_exponent(result)` → Dynamical exponent plot

#### 6.2 Multi-Simulation Comparison

**Overlay multiple datasets:**
- Compare different system sizes
- Compare different models (PXP vs PXPZ vs PNP)
- Compare different bond dimensions (convergence check)

**Key Function:**
- `compare_results(results::Vector{SimulationResult})`

**Usage:**
```bash
julia --project=. scripts/plot_results.jl data/pxp_transport_*.jld2
```

**Output:**
- Interactive plots (if Plots.jl available)
- ASCII tables (fallback)
- Saved PNG files

---

## Complete Data Flow Diagram

```
User Input (N, χ, t_max, ...)
         │
         ▼
┌────────────────────────┐
│  Hilbert Space Setup   │  hilbert.jl
│  - PXPSites(N)        │
│  - create_merged_sites│
└───────────┬────────────┘
            │ MergedSiteInfo
            ▼
┌────────────────────────┐
│  Operator Construction │  operators.jl
│  - h₀ = center_energy │
│    _density_merged()  │
└───────────┬────────────┘
            │ h₀::MPO
            ▼
┌────────────────────────┐
│   TEBD Evolution       │  tebd.jl
│  - make_trotter_gates  │
│  - run_tebd_evolution  │
│  - apply gates to MPO  │
└───────────┬────────────┘
            │ times, mpos
            ▼
┌────────────────────────┐
│  Observable Extraction │  observables.jl
│  - compute_correlation │
│  - instantaneous_exp   │
└───────────┬────────────┘
            │ C(t), z⁻¹(t)
            ▼
┌────────────────────────┐
│   Package Results      │  io.jl
│  - SimulationResult    │
│  - save_simulation     │
└───────────┬────────────┘
            │ .jld2 file
            ▼
┌────────────────────────┐
│   Visualization        │  plot_results.jl
│  - Load saved data     │
│  - Generate plots      │
└────────────────────────┘
```

---

## Entry Points

### Command Line

**Basic usage:**
```bash
julia --project=. scripts/run_simulation.jl
```

**With parameters:**
```bash
julia --project=. scripts/run_simulation.jl --N 64 --chi 128 --tmax 20.0
```

**Model variants:**
```bash
# PXPZ at ballistic point
julia --project=. scripts/run_simulation.jl --lambda 0.655

# PNP with chemical potential
julia --project=. scripts/run_simulation.jl --delta 0.5
```

### Programmatic API

**High-level function:**
```julia
using PXPTransport

# Run complete simulation
result = run_pxp_simulation(
    N = 64,
    maxdim = 128,
    tmax = 20.0,
    dt = 0.05,
    λ = 0.024,  # PXPZ integrable point
    save_to_file = true
)

print_summary(result)
```

**Low-level control:**
```julia
# Setup
info = create_merged_sites(64)
h0 = center_energy_density_merged(info.merged_sites; Ω=1.0)

# Evolution
params = TEBDParams(dt=0.05, maxdim=128, λ=0.024)
times, mpos = run_tebd_evolution(info, h0, 20.0, params)

# Analysis
C = compute_correlation_function(info.merged_sites, h0, times, mpos)
t_mid, z_inv = instantaneous_exponent(times, C)
```

---

## Key Design Principles

1. **Site Merging:** Reduces computational cost by converting 3-site terms to 2-site gates
   - Each merged site: 2 original spins → dimension 3
   - Local Hamiltonian: 9×9 instead of full $2^N$ space

2. **Heisenberg Picture:** Evolves operators instead of states
   - More efficient at infinite temperature
   - Directly computes $h(t)$ needed for $C(t)$

3. **TEBD Algorithm:** Tensor network method for large systems
   - Scales polynomially: $O(N \chi^3)$ vs exponential ED
   - Controlled approximation via bond dimension $\chi$

4. **Modular Design:** Clear separation of concerns
   - `hilbert.jl`: Constrained space
   - `operators.jl`: Observable construction
   - `tebd.jl`: Time evolution engine
   - `observables.jl`: Physical measurements
   - `io.jl`: Data persistence

5. **Reproducibility:** All parameters saved with results
   - Auto-generated filenames encode parameters
   - Full parameter dictionary stored in `.jld2`
   - Easy to reproduce and compare simulations

---

## Typical Use Cases

### 1. Pure PXP Transport (Baseline)
```julia
result = run_pxp_simulation(N=64, maxdim=128, tmax=20.0)
# Expected: z ≈ 3/2 (KPZ superdiffusion)
```

### 2. PXPZ at Integrable Point
```julia
result = run_pxp_simulation(N=64, maxdim=128, λ=0.024, tmax=20.0)
# Expected: z → 1 (ballistic, integrable)
```

### 3. Convergence Study
```julia
for χ in [64, 128, 256, 512]
    run_pxp_simulation(N=64, maxdim=χ, tmax=20.0)
end
# Compare results to verify convergence in χ
```

### 4. Finite-Size Scaling
```julia
for N in [32, 64, 128, 256]
    run_pxp_simulation(N=N, maxdim=256, tmax=min(20.0, N/4))
end
# Check finite-size effects: t_max < N/(2v)
```

---

## Performance Considerations

**Computational Scaling:**
- Memory: $O(N \chi^2)$ to store MPO
- Time per step: $O(N \chi^3)$ for gate applications
- Total time: $\sim$ (number of steps) × $O(N \chi^3)$

**Typical Parameters:**
- Small systems (N=32): χ=64, dt=0.05, ~1 minute
- Medium systems (N=64): χ=128, dt=0.05, ~10 minutes
- Large systems (N=256): χ=256, dt=0.05, ~hours

**Accuracy vs Cost Trade-off:**
- Larger χ → more accurate but slower
- Smaller dt → more accurate but more steps
- 4th-order Trotter → better accuracy, same cost per step

**Convergence Checks:**
1. Increase χ until results stable (typically χ=256-512)
2. Decrease dt until 4th-order error negligible
3. Check t_max < N/(2v) to avoid finite-size effects

---

## References

- **Paper:** Ljubotina et al., "Superdiffusive Energy Transport in Kinetically Constrained Models", Phys. Rev. X 13, 011033 (2023)
- **Method:** Time-Evolving Block Decimation (TEBD) for quantum many-body systems
- **Physics:** KPZ universality class, superdiffusive transport, Rydberg atom quantum simulators
