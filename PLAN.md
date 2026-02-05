# Implementation Plan: PXP Energy Transport

## Overview

Reproduce superdiffusive energy transport results from Ljubotina et al., Phys. Rev. X 13, 011033 (2023) using ITensors.jl.

**User preferences:**
- Multi-file modular structure
- JLD2 output format
- Initial test: $N=64$, $\chi=128$
- Site-merging TEBD approach

---

## Project Structure

```
superdiffusion_pxp/
├── PLAN.md                  # This plan
├── Project.toml             # Dependencies
├── src/
│   ├── PXPTransport.jl      # Main module file
│   ├── hamiltonian.jl       # Hamiltonians (PXP, PXPZ, PNP, PNPNP)
│   ├── hilbert.jl           # Constrained Hilbert space utilities
│   ├── operators.jl         # Energy density, projector construction
│   ├── tebd.jl              # TEBD evolution with site merging
│   ├── observables.jl       # Correlation functions, exponent extraction
│   └── io.jl                # JLD2 save/load utilities
├── scripts/
│   ├── run_simulation.jl    # Main entry point
│   └── plot_results.jl      # Visualization
├── test/
│   ├── runtests.jl          # Test runner
│   ├── test_hamiltonian.jl  # Hamiltonian tests
│   └── test_ed_benchmark.jl # ED comparison
└── data/                    # Output directory
```

---

## Module Design

### 1. `hamiltonian.jl`

**Functions:**
```julia
PXP_hamiltonian(sites; Ω=1.0) → MPO
PXPZ_hamiltonian(sites; Ω=1.0, λ=0.0) → MPO
PNP_hamiltonian(sites; Ω=1.0, δ=0.0) → MPO
PNPNP_hamiltonian(sites; Ω=1.0, ξ=0.0) → MPO
```

**Implementation:**
- Use `OpSum` (AutoMPO) for clean construction
- Express $P_i = (1 - \sigma^z_i)/2$ to use standard Pauli operators

### 2. `hilbert.jl`

**Functions:**
```julia
PXPSites(N::Int) → Vector{Index}           # Site indices
constrained_dim(N::Int) → Int               # Fibonacci dimension
is_valid_state(state::Int, N::Int) → Bool   # Check constraint
```

### 3. `operators.jl`

**Functions:**
```julia
energy_density(sites, l::Int; Ω=1.0) → MPO  # h_l = P_{l-1} σ^x_l P_{l+1}
projector_mpo(sites) → MPO                   # Global 𝒫 (bond dim 2)
```

### 4. `tebd.jl` (Core algorithm)

**Site-merging approach:**
1. Merge pairs of sites: $(1,2), (3,4), \ldots$
2. Constraint reduces merged local dim from 4 to 3: $\{|\downarrow\downarrow\rangle, |\uparrow\downarrow\rangle, |\downarrow\uparrow\rangle\}$
3. PXP term becomes 2-site gate on merged sites
4. Apply standard 2nd-order TEBD, compose for 4th-order

**Functions:**
```julia
merge_sites(sites) → merged_sites, mapping
make_trotter_gates(merged_sites, dt; Ω, λ, δ, ξ, order=4) → Vector{ITensor}
apply_gates!(M::MPO, gates; maxdim, cutoff) → MPO
evolve_tebd(h0::MPO, params, tmax; dt, maxdim, cutoff, save_every) → (times, MPOs)
```

### 5. `observables.jl`

**Functions:**
```julia
trace_mpo(M::MPO) → ComplexF64
correlation(h0::MPO, ht::MPO, P::MPO) → Float64  # ⟨h₀(0)h₀(t)⟩_c
instantaneous_exponent(times, corr) → (t_mid, z_inv)
spatial_profile(h0::MPO, ht_list::Vector{MPO}, P::MPO) → Vector{Float64}
```

### 6. `io.jl`

**Functions:**
```julia
save_simulation(filename, times, correlations, params)  # JLD2
load_simulation(filename) → (times, correlations, params)
```

---

## TEBD Algorithm Detail

### 4th-order Trotter Decomposition

For $U(\delta t) = e^{-iH\delta t}$, use symmetric decomposition:

$$
U_4(\delta t) = U_2(p\delta t) \cdot U_2(p\delta t) \cdot U_2((1-4p)\delta t) \cdot U_2(p\delta t) \cdot U_2(p\delta t)
$$

where $p = 1/(4 - 4^{1/3}) \approx 0.4145$ and $U_2$ is 2nd-order Trotter.

### Operator Evolution

Heisenberg picture: $h(t) = e^{iHt} h(0) e^{-iHt}$

For MPO evolution:
1. Apply $U^\dagger$ to bra (primed) indices
2. Apply $U$ to ket (unprimed) indices

Simplification: Since $h(0)$ is Hermitian and we only need $\text{Tr}[h(0) h(t)]$, can evolve in one direction and use symmetry.

---

## Data Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Parameters │     │   TEBD      │     │  Analysis   │
│  N, χ, δt   │ ──▶ │  Evolution  │ ──▶ │  z(t), C(t) │
│  Ω, λ, δ, ξ │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
                          │
                          ▼
                    ┌─────────────┐
                    │  JLD2 File  │
                    │  times, C,  │
                    │  params     │
                    └─────────────┘
```

---

## Implementation Status

1. **Phase 1: Foundation** ✓
   - [x] `Project.toml` with dependencies
   - [x] `hilbert.jl`: Site construction, dimension formulas
   - [x] `hamiltonian.jl`: PXP and deformations

2. **Phase 2: Core TEBD** ✓
   - [x] `tebd.jl`: Site merging, gate construction, evolution
   - [x] `operators.jl`: Energy density MPO

3. **Phase 3: Observables** ✓
   - [x] `observables.jl`: Correlation, exponent extraction
   - [x] `io.jl`: JLD2 save/load

4. **Phase 4: Integration** ✓
   - [x] `PXPTransport.jl`: Module wrapper
   - [x] `run_simulation.jl`: Entry script
   - [x] Test files

5. **Phase 5: Extensions** ✓
   - [x] PXPZ, PNP, PNPNP deformations
   - [x] ED benchmark tests
   - [x] Plotting script

---

## Verification Plan

### Unit Tests
- Hamiltonian Hermiticity
- Constraint dimension matches Fibonacci formula
- Energy density is local (low bond dimension)

### ED Benchmark
- Compare TEBD vs ED for $N \leq 12$
- Expect $<1\%$ relative error at short times

### Physics Checks
- $C(0) > 0$ (positive autocorrelation)
- $1/z \to 1$ for PNPNP at $\xi = 1$ (ballistic)
- Oscillation peaks near $t \approx 5.1, 10.2$

### Convergence
- Run with $\chi = 64, 128, 256$
- Results should stabilize

---

## Usage

```bash
# Install dependencies
julia --project=. -e "using Pkg; Pkg.instantiate()"

# Run tests
julia --project=. test/runtests.jl

# Run simulation
julia --project=. scripts/run_simulation.jl --N 64 --chi 128

# Plot results
julia --project=. scripts/plot_results.jl data/pxp_transport_PXP_N64_chi128.jld2
```

---

## Dependencies

```toml
[deps]
ITensors = "9136182c-28ba-11e9-034c-db9fb085ebd5"
JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
```
