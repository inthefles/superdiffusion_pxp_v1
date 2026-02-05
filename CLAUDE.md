# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Julia-based research code for simulating superdiffusive energy transport in the PXP (Rydberg blockade) model using tensor network methods. Reproduces results from Ljubotina et al., Phys. Rev. X 13, 011033 (2023).

**Physics Context:** The PXP model describes Rydberg atoms with kinetic constraint (no adjacent excited states). The project studies how energy spreads through this constrained system, with transport regimes characterized by dynamical exponent z: ballistic (z=1), superdiffusive/KPZ (z=3/2), diffusive (z=2).

## Commands

### Run Simulation
```bash
julia --project=. scripts/run_simulation.jl [--N 64] [--chi 128] [--lambda 0.655] [--delta 0.5] [--tmax 20.0]
```

### Run Tests
```bash
julia --project=. test/runtests.jl
```

### Programmatic Usage
```julia
using PXPTransport
result = run_pxp_simulation(N=64, maxdim=128, tmax=20.0, λ=0.024)
```

### Install Dependencies
```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Architecture

### Core Modules (in `src/`)

| Module | Purpose |
|--------|---------|
| `hilbert.jl` | Constrained Hilbert space with Rydberg blockade. Key: `PXPSites(N)`, `constrained_dim(N)` (Fibonacci), `enumerate_valid_states(N)` |
| `hamiltonian.jl` | PXP model and deformations (PXPZ, PNP, PNPNP) as MPOs via ITensors OpSum |
| `operators.jl` | Local energy density $h_l = \Omega P_{l-1} \sigma^x_l P_{l+1}$ and projector MPOs. Includes merged-site versions with bond dimension 4 |
| `tebd.jl` | Time-Evolving Block Decimation with 4th-order Trotter. Site merging pairs (2i-1, 2i) → merged site with dim 3. Heisenberg picture evolution of observables |
| `observables.jl` | Correlation functions: `autocorrelation(h0, ht)`, `instantaneous_exponent(times, C)` for extracting $z^{-1}(t)$ |
| `io.jl` | JLD2 serialization: `save_simulation()`, `load_simulation()`, `SimulationResult` struct |

### Key Design Patterns

- **Site Merging:** Converts 3-site PXP terms to 2-site gates on merged sites (dimension 3 each)
- **Heisenberg Picture:** Evolves operators rather than states for efficiency at infinite temperature
- **MPO Representation:** Keeps operators in compact tensor form for large systems (N up to 1024)

### Hamiltonian Models

- **PXP:** $H = \Omega \sum_i P_{i-1} \sigma^x_i P_{i+1}$ (base model)
- **PXPZ:** adds $\lambda \sum_i P_{i-1} \sigma^z_i P_{i+1}$ ($\lambda \approx 0.024$ is integrable)
- **PNP:** adds $\delta \sum_i n_i$ (superdiffusive at $\delta \geq 0.4$)
- **PNPNP:** adds 5-site term ($\xi = 1$ is integrable hard-square point, ballistic)

where $P_i = (1 - \sigma^z_i)/2$ is the projector onto $|\downarrow\rangle$.

## Parameters

**Numerical:**
- `N`: System size (must be even for site merging), typically 64-1024
- `maxdim` (χ): Bond dimension, 128-512 for convergence
- `dt`: Time step, 0.05 with 4th-order Trotter (error $O(dt^5)$)
- `tmax`: Evolution time, limited by finite-size effects at $t < N/(2v)$
- `cutoff`: SVD truncation threshold, typically 1e-10

**Physical:**
- `Ω`: Rabi frequency (default 1.0, sets energy/time scale)
- `λ`, `δ`, `ξ`: Deformation strengths for PXPZ, PNP, PNPNP models

## Validation

- **ED Benchmark:** `test/test_ed_benchmark.jl` provides exact diagonalization reference for N≤12
- **Known Limits:** PNPNP ξ=1 → z=1 (ballistic); PXPZ λ=0.5 → z=2 (diffusive)
- **Convergence:** Compare χ=256 vs χ=512; time step refinement with 4th-order Trotter

## Dependencies

Core: ITensors.jl (v0.3-0.8), JLD2, LinearAlgebra, Reexport
Optional: Plots.jl for visualization
