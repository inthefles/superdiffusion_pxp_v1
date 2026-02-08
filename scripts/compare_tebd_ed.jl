# compare_tebd_ed.jl
# Compare TEBD and ED results for energy-energy correlation function

using PXPTransport
using ITensors
using LinearAlgebra
using Statistics
using Plots
using Printf

# Include ED functions from test file
include("../test/test_ed_benchmark.jl")

println("="^60)
println("Comparing TEBD vs ED for N=20")
println("="^60)

# Parameters
N = 10  # Hilbert space dim = 144 (safe for both ED and TEBD)
Ω = 1.0
tmax = 15.0  # Extended time to check convergence
dt = 0.05
maxdim = 256

# Time points to evaluate
times = collect(0.0:1.0:tmax)

println("\nSystem parameters:")
println("  N = $N")
println("  Ω = $Ω")
println("  tmax = $tmax")
println("  dt = $dt (TEBD)")
println("  maxdim = $maxdim (TEBD)")
println()

# =============================================================================
# Exact Diagonalization
# =============================================================================
println("Running Exact Diagonalization...")
flush(stdout)

l_center = div(N + 1, 2)

# Build Hamiltonian and energy density
H_mat = build_pxp_matrix(N; Ω=Ω)
h_mat = build_energy_density_matrix(N, l_center; Ω=Ω)

# Compute correlation function at each time
C_ed = Float64[]
for t in times
    C_t = exact_correlation(H_mat, h_mat, t)
    push!(C_ed, C_t)
    if t % 2.0 < 0.1
        println("  ED: t = $t, C(t) = $(round(C_t, sigdigits=6))")
        flush(stdout)
    end
end

println("ED calculation complete!")
println()

# =============================================================================
# TEBD
# =============================================================================
println("Running TEBD...")
flush(stdout)

# Use high-level API for robustness
result = run_pxp_simulation(
    N=N,
    maxdim=maxdim,
    tmax=tmax,
    dt=dt,
    Ω=Ω,
    λ=0.0,  # No PXPZ deformation
    δ=0.0,  # No PNP deformation
    save_to_file=false
)

times_tebd_all = result.times
C_tebd_all = result.correlations

# Interpolate TEBD results to match ED time points
using Interpolations
itp = LinearInterpolation(times_tebd_all, C_tebd_all)
C_tebd = [itp(t) for t in times]
times_tebd = times

println("TEBD calculation complete!")
println()

# =============================================================================
# Comparison
# =============================================================================
println("="^60)
println("Results Comparison")
println("="^60)
println()
println("Time    C_ED         C_TEBD       Abs Error    Rel Error")
println("-"^60)

errors_abs = abs.(C_ed .- C_tebd)
errors_rel = errors_abs ./ (abs.(C_ed) .+ 1e-14)

for i in 1:length(times)
    t = times[i]
    if i == 1 || i == length(times) || t % 2.0 < 0.1
        @printf("%5.1f   %.6e   %.6e   %.2e   %.2e\n",
                t, C_ed[i], C_tebd[i], errors_abs[i], errors_rel[i])
    end
end

println()
println("Statistics:")
println("  Max absolute error: $(maximum(errors_abs))")
println("  Mean absolute error: $(mean(errors_abs))")
println("  Max relative error: $(maximum(errors_rel))")
println("  Mean relative error: $(mean(errors_rel))")
println()

# =============================================================================
# Plotting
# =============================================================================
println("Creating plots...")

# Plot 1: Correlation functions
p1 = plot(times, C_ed,
         label="ED (exact)",
         marker=:circle,
         markersize=4,
         linewidth=2,
         xlabel="Time",
         ylabel="C(t)",
         title="Energy-Energy Correlation Function (N=$N)",
         legend=:topright,
         dpi=300)

plot!(p1, times, C_tebd,
      label="TEBD (χ=$maxdim)",
      marker=:square,
      markersize=3,
      linewidth=2,
      linestyle=:dash)

# Plot 2: Absolute error
p2 = plot(times, errors_abs,
         label="",
         marker=:circle,
         markersize=3,
         linewidth=2,
         xlabel="Time",
         ylabel="|C_ED - C_TEBD|",
         title="Absolute Error",
         yscale=:log10,
         legend=false,
         dpi=300)

# Plot 3: Relative error
p3 = plot(times, errors_rel,
         label="",
         marker=:circle,
         markersize=3,
         linewidth=2,
         xlabel="Time",
         ylabel="|C_ED - C_TEBD| / |C_ED|",
         title="Relative Error",
         yscale=:log10,
         legend=false,
         dpi=300)

# Plot 4: Log-log plot of correlation
p4 = plot(times[2:end], abs.(C_ed[2:end]),
         label="ED (exact)",
         marker=:circle,
         markersize=4,
         linewidth=2,
         xlabel="Time",
         ylabel="|C(t)|",
         title="Correlation Decay (log-log)",
         xscale=:log10,
         yscale=:log10,
         legend=:topright,
         dpi=300)

plot!(p4, times[2:end], abs.(C_tebd[2:end]),
      label="TEBD (χ=$maxdim)",
      marker=:square,
      markersize=3,
      linewidth=2,
      linestyle=:dash)

# Power law reference lines
t_ref = times[2:end]
# Ballistic: C ~ t^(-2)
plot!(p4, t_ref, 0.5 * t_ref.^(-2),
      label="t^(-2) (ballistic)",
      linestyle=:dot,
      linewidth=2,
      color=:gray)
# Superdiffusive (KPZ): C ~ t^(-2/3)
plot!(p4, t_ref, 5.0 * t_ref.^(-2/3),
      label="t^(-2/3) (KPZ)",
      linestyle=:dot,
      linewidth=2,
      color=:orange)

# Combine all plots
p_combined = plot(p1, p2, p3, p4,
                  layout=(2, 2),
                  size=(1200, 1000),
                  margin=5Plots.mm)

# Save plots
output_file = "comparison_tebd_ed_N$(N).png"
savefig(p_combined, output_file)
println("Saved plot to: $output_file")

# Also save individual high-res plot of correlation function
savefig(p1, "correlation_N$(N).png")
println("Saved correlation plot to: correlation_N$(N).png")

println()
println("="^60)
println("Comparison complete!")
println("="^60)

# Display the plot
display(p_combined)
