#!/usr/bin/env julia
# plot_results.jl
# Visualization script for PXP transport simulation results
#
# Usage:
#   julia --project=. scripts/plot_results.jl <data_file.jld2>
#   julia --project=. scripts/plot_results.jl data/*.jld2  # Multiple files

using Pkg
Pkg.activate(dirname(@__DIR__))

using PXPTransport
using Printf

# Try to load Plots, but provide ASCII fallback
const HAS_PLOTS = try
    using Plots
    true
catch
    @warn "Plots.jl not available, using ASCII output"
    false
end

"""
    plot_correlation(result::SimulationResult; output=nothing)

Plot the correlation function C(t) vs t on log-log scale.
"""
function plot_correlation(result::SimulationResult; output=nothing)
    times = result.times
    C = result.correlations
    params = result.params

    # Filter positive values for log plot
    valid = (times .> 0) .& (C .> 0)
    t_valid = times[valid]
    C_valid = C[valid]

    if HAS_PLOTS
        p = plot(t_valid, C_valid,
                 xscale=:log10, yscale=:log10,
                 xlabel="t", ylabel="C(t)",
                 label="Data",
                 linewidth=2,
                 title="Energy Autocorrelation - $(get(params, "model", "PXP"))")

        # Add power law guide for z=3/2 (superdiffusive)
        if length(t_valid) > 2
            t_guide = range(t_valid[2], t_valid[end], length=100)
            # C(t) ~ t^{-2/z}, for z=3/2: C ~ t^{-4/3}
            C_guide = C_valid[2] * (t_guide / t_valid[2]) .^ (-4/3)
            plot!(p, t_guide, C_guide, linestyle=:dash, label="t^{-4/3} (z=3/2)",
                  color=:gray)

            # Also show ballistic z=1 guide
            C_ball = C_valid[2] * (t_guide / t_valid[2]) .^ (-2)
            plot!(p, t_guide, C_ball, linestyle=:dot, label="t^{-2} (z=1)",
                  color=:lightgray)
        end

        if !isnothing(output)
            savefig(p, output)
            @info "Saved correlation plot to $output"
        else
            display(p)
        end

        return p
    else
        # ASCII output
        println("\nCorrelation Function C(t):")
        println("-" ^ 50)
        @printf("%10s  %15s\n", "t", "C(t)")
        println("-" ^ 50)
        for (t, c) in zip(t_valid[1:min(20, end)], C_valid[1:min(20, end)])
            @printf("%10.4f  %15.6e\n", t, c)
        end
        if length(t_valid) > 20
            println("... ($(length(t_valid) - 20) more points)")
        end
    end
end

"""
    plot_exponent(result::SimulationResult; output=nothing)

Plot the instantaneous exponent 1/z(t) vs t.
"""
function plot_exponent(result::SimulationResult; output=nothing)
    t_mid = result.exponent_times
    z_inv = result.exponents
    params = result.params

    if isempty(t_mid)
        @warn "No exponent data available"
        return
    end

    if HAS_PLOTS
        p = plot(t_mid, z_inv,
                 xlabel="t", ylabel="1/z(t)",
                 label="Data",
                 linewidth=2,
                 title="Instantaneous Exponent - $(get(params, "model", "PXP"))",
                 ylims=(0, 1.5))

        # Reference lines
        hline!(p, [2/3], linestyle=:dash, label="2/3 (z=3/2, superdiffusive)",
               color=:blue)
        hline!(p, [1.0], linestyle=:dot, label="1 (z=1, ballistic)",
               color=:green)
        hline!(p, [0.5], linestyle=:dashdot, label="1/2 (z=2, diffusive)",
               color=:red)

        if !isnothing(output)
            savefig(p, output)
            @info "Saved exponent plot to $output"
        else
            display(p)
        end

        return p
    else
        # ASCII output
        println("\nInstantaneous Exponent 1/z(t):")
        println("-" ^ 50)
        @printf("%10s  %15s\n", "t", "1/z")
        println("-" ^ 50)
        for (t, z) in zip(t_mid[1:min(20, end)], z_inv[1:min(20, end)])
            @printf("%10.4f  %15.6f\n", t, z)
        end
        if length(t_mid) > 20
            println("... ($(length(t_mid) - 20) more points)")
        end

        # Summary statistics
        println("\nStatistics for t > 5:")
        late_idx = findall(t -> t > 5.0, t_mid)
        if !isempty(late_idx)
            z_late = z_inv[late_idx]
            @printf("  Mean 1/z:   %.4f\n", mean(z_late))
            @printf("  Std 1/z:    %.4f\n", std(z_late))
            @printf("  Mean z:     %.4f\n", 1.0 / mean(z_late))
        end
    end
end

"""
    compare_results(results::Vector{SimulationResult}; output=nothing)

Compare correlation functions from multiple simulations.
"""
function compare_results(results::Vector{SimulationResult}; output=nothing)
    if HAS_PLOTS
        p = plot(xlabel="t", ylabel="C(t)",
                 xscale=:log10, yscale=:log10,
                 title="Correlation Comparison")

        for result in results
            times = result.times
            C = result.correlations
            params = result.params

            valid = (times .> 0) .& (C .> 0)
            label_str = "$(get(params, "model", "?")) N=$(get(params, "N", "?"))"

            if get(params, "λ", 0.0) != 0
                label_str *= " λ=$(get(params, "λ", 0.0))"
            end

            plot!(p, times[valid], C[valid], label=label_str, linewidth=2)
        end

        if !isnothing(output)
            savefig(p, output)
        else
            display(p)
        end

        return p
    else
        println("\nComparison of $(length(results)) simulations:")
        for (i, result) in enumerate(results)
            params = result.params
            println("\n[$i] $(get(params, "model", "?")) - N=$(get(params, "N", "?"))")
            if !isempty(result.exponents)
                late_idx = findall(t -> t > 5.0, result.exponent_times)
                if !isempty(late_idx)
                    z_mean = 1.0 / mean(result.exponents[late_idx])
                    @printf("    z ≈ %.3f\n", z_mean)
                end
            end
        end
    end
end

using Statistics: mean, std

function main()
    if isempty(ARGS)
        println("""
        Usage: julia --project=. scripts/plot_results.jl <data_files.jld2>

        Examples:
          julia --project=. scripts/plot_results.jl data/pxp_transport_PXP_N64_chi128.jld2
          julia --project=. scripts/plot_results.jl data/*.jld2
        """)
        return
    end

    results = SimulationResult[]
    for filename in ARGS
        if isfile(filename) && endswith(filename, ".jld2")
            @info "Loading $filename"
            result = load_simulation(filename)
            push!(results, result)
        else
            @warn "Skipping $filename (not a .jld2 file)"
        end
    end

    if isempty(results)
        @error "No valid data files found"
        return
    end

    if length(results) == 1
        # Single file: detailed plots
        result = results[1]
        print_summary(result)

        println("\n" * "=" ^ 60)
        println("Plotting correlation function...")
        plot_correlation(result)

        println("\nPlotting exponent...")
        plot_exponent(result)
    else
        # Multiple files: comparison
        println("Comparing $(length(results)) simulations...")
        compare_results(results)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
