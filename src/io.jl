# io.jl
# Input/Output utilities for PXP transport simulations
#
# Uses JLD2 for efficient Julia-native serialization.

using JLD2
using Printf

"""
    SimulationResult

Container for simulation results.
"""
struct SimulationResult
    times::Vector{Float64}
    correlations::Vector{Float64}
    exponents::Vector{Float64}        # Instantaneous 1/z values
    exponent_times::Vector{Float64}   # Times for exponent values
    params::Dict{String,Any}
end

"""
    save_simulation(filename::String, result::SimulationResult)

Save simulation results to a JLD2 file.
"""
function save_simulation(filename::String, result::SimulationResult)
    # Ensure .jld2 extension
    if !endswith(filename, ".jld2")
        filename = filename * ".jld2"
    end

    jldsave(filename;
            times=result.times,
            correlations=result.correlations,
            exponents=result.exponents,
            exponent_times=result.exponent_times,
            params=result.params)

    @info "Saved simulation to $filename"
end

"""
    save_simulation(filename::String, times, correlations, params;
                    exponents=nothing, exponent_times=nothing)

Save simulation results with individual components.
"""
function save_simulation(filename::String,
                         times::Vector{Float64},
                         correlations::Vector{Float64},
                         params::Dict{String,Any};
                         exponents::Union{Vector{Float64},Nothing}=nothing,
                         exponent_times::Union{Vector{Float64},Nothing}=nothing)
    if isnothing(exponents)
        exponents = Float64[]
    end
    if isnothing(exponent_times)
        exponent_times = Float64[]
    end

    result = SimulationResult(times, correlations, exponents, exponent_times, params)
    save_simulation(filename, result)
end

"""
    load_simulation(filename::String) -> SimulationResult

Load simulation results from a JLD2 file.
"""
function load_simulation(filename::String)
    data = load(filename)

    times = get(data, "times", Float64[])
    correlations = get(data, "correlations", Float64[])
    exponents = get(data, "exponents", Float64[])
    exponent_times = get(data, "exponent_times", Float64[])
    params = get(data, "params", Dict{String,Any}())

    return SimulationResult(times, correlations, exponents, exponent_times, params)
end

"""
    default_params(; kwargs...) -> Dict{String,Any}

Create a parameter dictionary with default values.
"""
function default_params(; N=64, maxdim=128, cutoff=1e-10, dt=0.05, tmax=20.0,
                        Ω=1.0, λ=0.0, δ=0.0, ξ=0.0, order=4)
    return Dict{String,Any}(
        "N" => N,
        "maxdim" => maxdim,
        "cutoff" => cutoff,
        "dt" => dt,
        "tmax" => tmax,
        "Ω" => Ω,
        "λ" => λ,
        "δ" => δ,
        "ξ" => ξ,
        "order" => order,
        "model" => _model_name(λ, δ, ξ),
        "timestamp" => Dates.format(Dates.now(), "yyyy-mm-dd_HH:MM:SS")
    )
end

function _model_name(λ, δ, ξ)
    if abs(λ) > 1e-10 && abs(δ) < 1e-10 && abs(ξ) < 1e-10
        return "PXPZ"
    elseif abs(δ) > 1e-10 && abs(λ) < 1e-10 && abs(ξ) < 1e-10
        return "PNP"
    elseif abs(ξ) > 1e-10 && abs(λ) < 1e-10 && abs(δ) < 1e-10
        return "PNPNP"
    else
        return "PXP"
    end
end

"""
    generate_filename(params::Dict; prefix="pxp_transport") -> String

Generate a descriptive filename based on simulation parameters.
"""
function generate_filename(params::Dict; prefix="pxp_transport")
    N = get(params, "N", 0)
    maxdim = get(params, "maxdim", 0)
    model = get(params, "model", "PXP")
    λ = get(params, "λ", 0.0)
    δ = get(params, "δ", 0.0)
    ξ = get(params, "ξ", 0.0)

    parts = ["$(prefix)", "$(model)", "N$(N)", "chi$(maxdim)"]

    if abs(λ) > 1e-10
        push!(parts, @sprintf("lam%.3f", λ))
    end
    if abs(δ) > 1e-10
        push!(parts, @sprintf("del%.3f", δ))
    end
    if abs(ξ) > 1e-10
        push!(parts, @sprintf("xi%.3f", ξ))
    end

    return join(parts, "_") * ".jld2"
end

"""
    print_summary(result::SimulationResult)

Print a summary of simulation results.
"""
function print_summary(result::SimulationResult)
    params = result.params

    println("=" ^ 60)
    println("PXP Transport Simulation Summary")
    println("=" ^ 60)

    # Model info
    println("Model: $(get(params, "model", "PXP"))")
    println("System size: N = $(get(params, "N", "?"))")
    println("Bond dimension: χ = $(get(params, "maxdim", "?"))")

    # Deformation parameters
    λ = get(params, "λ", 0.0)
    δ = get(params, "δ", 0.0)
    ξ = get(params, "ξ", 0.0)
    if abs(λ) > 1e-10
        @printf("λ (PXPZ): %.4f\n", λ)
    end
    if abs(δ) > 1e-10
        @printf("δ (PNP): %.4f\n", δ)
    end
    if abs(ξ) > 1e-10
        @printf("ξ (PNPNP): %.4f\n", ξ)
    end

    # Time evolution
    println("\nTime evolution:")
    @printf("  dt = %.4f, t_max = %.1f\n", get(params, "dt", 0.0), get(params, "tmax", 0.0))
    @printf("  Trotter order: %d\n", get(params, "order", 4))

    # Results summary
    if !isempty(result.correlations)
        println("\nCorrelation function:")
        @printf("  C(0) = %.6f\n", result.correlations[1])
        @printf("  C(t_max) = %.6e\n", result.correlations[end])
        @printf("  Decay factor: %.2f\n", result.correlations[1] / result.correlations[end])
    end

    if !isempty(result.exponents)
        # Find average exponent in late time regime
        late_idx = findall(t -> t > 5.0, result.exponent_times)
        if !isempty(late_idx)
            z_inv_avg = mean(result.exponents[late_idx])
            z_avg = 1.0 / z_inv_avg
            @printf("\nDynamical exponent (t > 5):\n")
            @printf("  1/z = %.4f ± %.4f\n", z_inv_avg, std(result.exponents[late_idx]))
            @printf("  z = %.4f\n", z_avg)
        end
    end

    println("=" ^ 60)
end

# Import Dates for timestamp
import Dates
using Statistics: mean, std

export SimulationResult, save_simulation, load_simulation
export default_params, generate_filename, print_summary
