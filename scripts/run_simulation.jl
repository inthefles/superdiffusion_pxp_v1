#!/usr/bin/env julia
# run_simulation.jl
# Main entry point for running PXP transport simulations
#
# Usage:
#   julia --project=. scripts/run_simulation.jl [options]
#
# Examples:
#   julia --project=. scripts/run_simulation.jl                    # Default parameters
#   julia --project=. scripts/run_simulation.jl --N 64 --chi 128   # Custom size
#   julia --project=. scripts/run_simulation.jl --lambda 0.655     # PXPZ model

using Pkg
Pkg.activate(dirname(@__DIR__))

using PXPTransport
using Printf

function parse_args(args)
    params = Dict{Symbol,Any}(
        :N => 64,
        :maxdim => 128,
        :tmax => 20.0,
        :dt => 0.05,
        :Ω => 1.0,
        :λ => 0.0,
        :δ => 0.0,
        :ξ => 0.0,
        :cutoff => 1e-10,
        :order => 4,
        :data_dir => joinpath(dirname(@__DIR__), "data"),
        :help => false
    )

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--help" || arg == "-h"
            params[:help] = true
        elseif arg == "--N" || arg == "-N"
            i += 1
            params[:N] = parse(Int, args[i])
        elseif arg == "--chi" || arg == "--maxdim"
            i += 1
            params[:maxdim] = parse(Int, args[i])
        elseif arg == "--tmax"
            i += 1
            params[:tmax] = parse(Float64, args[i])
        elseif arg == "--dt"
            i += 1
            params[:dt] = parse(Float64, args[i])
        elseif arg == "--omega" || arg == "--Omega"
            i += 1
            params[:Ω] = parse(Float64, args[i])
        elseif arg == "--lambda"
            i += 1
            params[:λ] = parse(Float64, args[i])
        elseif arg == "--delta"
            i += 1
            params[:δ] = parse(Float64, args[i])
        elseif arg == "--xi"
            i += 1
            params[:ξ] = parse(Float64, args[i])
        elseif arg == "--cutoff"
            i += 1
            params[:cutoff] = parse(Float64, args[i])
        elseif arg == "--order"
            i += 1
            params[:order] = parse(Int, args[i])
        elseif arg == "--data-dir"
            i += 1
            params[:data_dir] = args[i]
        else
            @warn "Unknown argument: $arg"
        end
        i += 1
    end

    return params
end

function print_help()
    println("""
    PXP Transport Simulation
    ========================

    Simulates energy transport in the PXP model using TEBD.
    Reference: Ljubotina et al., Phys. Rev. X 13, 011033 (2023)

    Usage:
      julia --project=. scripts/run_simulation.jl [options]

    Options:
      --N <int>         System size (default: 64)
      --chi, --maxdim   Maximum bond dimension (default: 128)
      --tmax <float>    Maximum evolution time (default: 20.0)
      --dt <float>      Time step (default: 0.05)
      --omega <float>   Rabi frequency Ω (default: 1.0)
      --lambda <float>  PXPZ deformation λ (default: 0.0)
      --delta <float>   Chemical potential δ for PNP (default: 0.0)
      --xi <float>      PNPNP deformation ξ (default: 0.0)
      --cutoff <float>  SVD truncation cutoff (default: 1e-10)
      --order <int>     Trotter order, 2 or 4 (default: 4)
      --data-dir <path> Output directory (default: data/)
      --help, -h        Show this help message

    Examples:
      # Pure PXP model (superdiffusive, z ≈ 3/2)
      julia --project=. scripts/run_simulation.jl --N 64 --chi 128

      # PXPZ at ballistic point (z = 1)
      julia --project=. scripts/run_simulation.jl --lambda 0.655

      # PNPNP at ballistic point (z = 1)
      julia --project=. scripts/run_simulation.jl --xi 1.0

    Output:
      Results are saved to data/pxp_transport_*.jld2
    """)
end

function main()
    params = parse_args(ARGS)

    if params[:help]
        print_help()
        return
    end

    println("=" ^ 60)
    println("PXP Transport Simulation")
    println("=" ^ 60)
    println()
    @printf("System size:     N = %d\n", params[:N])
    @printf("Bond dimension:  χ = %d\n", params[:maxdim])
    @printf("Time evolution:  t_max = %.1f, dt = %.4f\n", params[:tmax], params[:dt])
    @printf("Trotter order:   %d\n", params[:order])

    # Show model type
    if abs(params[:λ]) > 1e-10
        @printf("Model: PXPZ (λ = %.4f)\n", params[:λ])
    elseif abs(params[:δ]) > 1e-10
        @printf("Model: PNP (δ = %.4f)\n", params[:δ])
    elseif abs(params[:ξ]) > 1e-10
        @printf("Model: PNPNP (ξ = %.4f)\n", params[:ξ])
    else
        println("Model: PXP")
    end
    println()

    # Run simulation
    result = run_pxp_simulation(
        N = params[:N],
        maxdim = params[:maxdim],
        tmax = params[:tmax],
        dt = params[:dt],
        Ω = params[:Ω],
        λ = params[:λ],
        δ = params[:δ],
        ξ = params[:ξ],
        cutoff = params[:cutoff],
        order = params[:order],
        save_to_file = true,
        data_dir = params[:data_dir]
    )

    println("\nSimulation complete!")
    return result
end

# Run if called as script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
