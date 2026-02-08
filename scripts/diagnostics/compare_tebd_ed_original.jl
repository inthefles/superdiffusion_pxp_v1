# compare_tebd_ed_original.jl
# Compare TEBD and ED using the SAME energy density operator (original representation)

using PXPTransport
using ITensors
using LinearAlgebra
using Statistics
using Plots
using Printf

# Include ED functions from test file
include("../test/test_ed_benchmark.jl")

println("="^60)
println("TEBD vs ED: Using Original-Site Energy Density")
println("="^60)

# Parameters
N = 10  # Original sites
Ω = 1.0
tmax = 5.0
dt = 0.05
maxdim = 128

# Time points to evaluate
times = collect(0.0:0.5:tmax)

println("\nSystem parameters:")
println("  N = $N (original sites)")
println("  Ω = $Ω")
println("  tmax = $tmax")
println("  dt = $dt (TEBD)")
println("  maxdim = $maxdim (TEBD)")
println()

# =============================================================================
# Build Energy Density Operator (Original Representation)
# =============================================================================

# Choose center site
l_center = div(N + 1, 2)  # Site 5 or 6 for N=10
println("Using energy density at original site $l_center")
println()

# =============================================================================
# Exact Diagonalization
# =============================================================================
println("Running Exact Diagonalization...")
flush(stdout)

# Build Hamiltonian and energy density
H_mat = build_pxp_matrix(N; Ω=Ω)
h_mat = build_energy_density_matrix(N, l_center; Ω=Ω)

println("  Matrix dimensions: $(size(H_mat))")
println("  Tr[h] = $(tr(h_mat)) (should be ~0 for bulk)")
println("  Tr[h²] = $(tr(h_mat^2))")

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
# TEBD with Original-Site Energy Density
# =============================================================================
println("Running TEBD...")
flush(stdout)

# Create merged sites for TEBD gates
info = create_merged_sites(N)
println("  Created $(info.N_merged) merged sites from $N original sites")

# Create energy density in ORIGINAL site representation
# This is the same operator as ED uses, but as an MPO
sites_original = PXPSites(N)
h0_original = center_energy_density(sites_original; Ω=Ω)

println("  Created energy density at original site $l_center")

# Check it matches ED
tr_h0 = real(trace_mpo(h0_original))
norm_h0_sq = real(inner(h0_original, h0_original))
println("  Tr[h₀] = $tr_h0 (ED: $(tr(h_mat)))")
println("  Tr[h₀²] = $norm_h0_sq (ED: $(tr(h_mat^2)))")

if abs(norm_h0_sq - tr(h_mat^2)) / tr(h_mat^2) < 1e-6
    println("  ✓ TEBD and ED operators match!")
else
    println("  ⚠ Warning: TEBD and ED operators differ by $(abs(norm_h0_sq - tr(h_mat^2)) / tr(h_mat^2) * 100)%")
end
println()

# Convert to merged representation for evolution
h0_merged = merge_mpo_pairs(h0_original, info.merged_sites)

# TEBD evolution (evolves the merged MPO)
params = TEBDParams(dt=dt, maxdim=maxdim, cutoff=1e-10, order=4, Ω=Ω)
times_tebd_all, mpos_merged = run_tebd_evolution(info, h0_merged, tmax, params;
                                                  save_times=times)

# Compute correlation function using merged sites
C_tebd = compute_correlation_function(info.merged_sites, h0_merged, times, mpos_merged)

println("TEBD calculation complete!")
println()

# =============================================================================
# Comparison
# =============================================================================
println("="^60)
println("Results Comparison")
println("="^60)
println()
println("Both using energy density at original site $l_center")
println()
println("Time    C_ED         C_TEBD       Abs Error    Rel Error    Ratio")
println("-"^70)

errors_abs = abs.(C_ed .- C_tebd)
errors_rel = errors_abs ./ (abs.(C_ed) .+ 1e-14)
ratio = C_tebd ./ (C_ed .+ 1e-14)

for i in 1:length(times)
    t = times[i]
    @printf("%5.1f   %.6e   %.6e   %.2e   %.2e   %.3f\n",
            t, C_ed[i], C_tebd[i], errors_abs[i], errors_rel[i], ratio[i])
end

println()
println("Statistics:")
println("  Max absolute error: $(maximum(errors_abs))")
println("  Mean absolute error: $(mean(errors_abs))")
println("  Max relative error: $(maximum(errors_rel))")
println("  Mean relative error: $(mean(errors_rel))")
println("  Mean ratio: $(mean(ratio)) (should be ≈1.0)")
println()

# =============================================================================
# Plotting
# =============================================================================
println("Creating plots...")

# Plot 1: Direct comparison
p1 = plot(times, C_ed,
         label="ED (exact)",
         marker=:circle,
         markersize=4,
         linewidth=2,
         xlabel="Time",
         ylabel="C(t)",
         title="Energy Autocorrelation (N=$N)",
         legend=:topright,
         dpi=300)

plot!(p1, times, C_tebd,
      label="TEBD (χ=$maxdim)",
      marker=:square,
      markersize=3,
      linewidth=2,
      linestyle=:dash)

# Plot 2: Log-log
valid = (times .> 0) .& (C_ed .> 0) .& (C_tebd .> 0)
t_valid = times[valid]

p2 = plot(t_valid, abs.(C_ed[valid]),
         label="ED (exact)",
         marker=:circle,
         markersize=4,
         linewidth=2,
         xlabel="Time",
         ylabel="|C(t)|",
         title="Correlation Decay (log-log)",
         xscale=:log10,
         yscale=:log10,
         legend=:bottomleft,
         dpi=300)

plot!(p2, t_valid, abs.(C_tebd[valid]),
      label="TEBD (χ=$maxdim)",
      marker=:square,
      markersize=3,
      linewidth=2,
      linestyle=:dash)

# Plot 3: Absolute error
p3 = plot(times, errors_abs,
         marker=:circle,
         markersize=3,
         linewidth=2,
         xlabel="Time",
         ylabel="|C_ED - C_TEBD|",
         title="Absolute Error",
         yscale=:log10,
         legend=false,
         dpi=300)

# Plot 4: Ratio
p4 = plot(times, ratio,
         marker=:circle,
         markersize=3,
         linewidth=2,
         xlabel="Time",
         ylabel="C_TEBD / C_ED",
         title="Ratio (should be ≈1.0)",
         legend=false,
         dpi=300)
hline!(p4, [1.0], linestyle=:dash, color=:gray, linewidth=2)

# Combine plots
p_combined = plot(p1, p2, p3, p4,
                  layout=(2, 2),
                  size=(1200, 1000),
                  margin=5Plots.mm)

# Save plots
output_file = "comparison_same_operator_N$(N).png"
savefig(p_combined, output_file)
println("Saved plot to: $output_file")

println()
println("="^60)
println("Comparison complete!")
println("="^60)

# Display the plot
display(p_combined)
