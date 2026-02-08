# PXPTransport.jl
# Main module for PXP energy transport simulations
#
# Reproduces superdiffusive transport results from:
# Ljubotina et al., Phys. Rev. X 13, 011033 (2023)
#
# Usage:
#   using PXPTransport
#   sites = PXPSites(64)
#   H = PXP_hamiltonian(sites)
#   h0 = center_energy_density(sites)
#   params = TEBDParams(dt=0.05, maxdim=128)
#   times, mpos = run_tebd_evolution(sites, h0, 20.0, params)
#   C = compute_correlation_function(sites, h0, times, mpos)

module PXPTransport

using ITensors
using ITensorMPS

# Include submodules in dependency order
include("hilbert.jl")
include("hamiltonian.jl")
include("operators.jl")
include("tebd.jl")
include("observables.jl")
include("io.jl")

# Re-export commonly used functions
export PXPSites, constrained_dim, enumerate_valid_states
export infinite_temperature_state, all_down_state, neel_state

export PXP_hamiltonian, PXPZ_hamiltonian, PNP_hamiltonian, PNPNP_hamiltonian
export check_hermiticity

export energy_density, center_energy_density
export projector_mpo, identity_mpo
export sz_operator, sx_operator, number_operator
# Paper-based implementation (χ=2)
export energy_density_original, projector_mpo_original
export merge_mpo_pairs
export energy_density_merged, center_energy_density_merged
export projector_mpo_merged, identity_mpo_merged
# PNP / PNPNP energy densities
export energy_density_pnp_original, energy_density_pnpnp_original
export energy_density_pnp_merged, energy_density_pnpnp_merged
export center_energy_density_pnp_merged, center_energy_density_pnpnp_merged

export TEBDParams, evolve_mpo, run_tebd_evolution
export make_trotter_gates, make_trotter_gates_merged
export create_merged_sites

export trace_mpo, autocorrelation, compute_correlation_function
export instantaneous_exponent, fit_exponent
export check_unitarity

export SimulationResult, save_simulation, load_simulation
export default_params, generate_filename, print_summary

"""
    run_pxp_simulation(; N=64, maxdim=128, tmax=20.0, dt=0.05,
                       Ω=1.0, λ=0.0, δ=0.0, ξ=0.0,
                       cutoff=1e-10, order=4,
                       save_to_file=true, data_dir="data")

High-level function to run a complete PXP transport simulation.

# Arguments
- `N`: System size (must be even for site merging)
- `maxdim`: Maximum bond dimension for MPO truncation
- `tmax`: Maximum evolution time
- `dt`: Time step
- `Ω`: Rabi frequency (default 1.0)
- `λ`: PXPZ deformation parameter
- `δ`: Chemical potential (PNP model)
- `ξ`: PNPNP deformation parameter
- `cutoff`: SVD cutoff for truncation
- `order`: Trotter order (2 or 4)
- `save_to_file`: Whether to save results to JLD2
- `data_dir`: Directory for output files

# Returns
- `result::SimulationResult`: Contains times, correlations, exponents, and params

# Example
```julia
result = run_pxp_simulation(N=64, maxdim=128, tmax=20.0)
print_summary(result)
```
"""
function run_pxp_simulation(; N=64, maxdim=128, tmax=20.0, dt=0.05,
                            Ω=1.0, λ=0.0, δ=0.0, ξ=0.0,
                            cutoff=1e-10, order=4,
                            save_to_file=true, data_dir="data")

    @info "Starting PXP transport simulation"
    @info "  N = $N, χ = $maxdim, t_max = $tmax"

    # Create merged site representation
    info = create_merged_sites(N)
    @info "  Created $(info.N_merged) merged sites from $N original sites"

    # Construct energy density at center using merged site representation
    h0 = center_energy_density_merged(info.merged_sites; Ω=Ω)
    @info "  Created energy density operator at merged site $(div(info.N_merged+1, 2))"

    # TEBD parameters
    params = TEBDParams(dt=dt, maxdim=maxdim, cutoff=cutoff, order=order,
                        Ω=Ω, λ=λ, δ=δ, ξ=ξ)

    # Run time evolution
    @info "  Running TEBD evolution..."
    nsteps = round(Int, tmax / dt)
    save_times = collect(range(0, tmax, length=min(nsteps+1, 401)))

    times, mpos = run_tebd_evolution(info, h0, tmax, params; save_times=save_times)
    @info "  Evolution complete, $(length(times)) time points saved"

    # Compute correlation function
    @info "  Computing correlation function..."
    C = compute_correlation_function(info.merged_sites, h0, times, mpos)

    # Extract exponents
    t_mid, z_inv = instantaneous_exponent(times, C)

    # Package results
    param_dict = default_params(N=N, maxdim=maxdim, cutoff=cutoff, dt=dt, tmax=tmax,
                                 Ω=Ω, λ=λ, δ=δ, ξ=ξ, order=order)

    result = SimulationResult(times, C, z_inv, t_mid, param_dict)

    # Save if requested
    if save_to_file
        mkpath(data_dir)
        filename = joinpath(data_dir, generate_filename(param_dict))
        save_simulation(filename, result)
    end

    print_summary(result)
    return result
end

export run_pxp_simulation

end # module
