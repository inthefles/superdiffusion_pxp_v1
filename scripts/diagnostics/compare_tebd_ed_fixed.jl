# compare_tebd_ed_fixed.jl
# Careful comparison of TEBD and ED with matching energy density operators

using PXPTransport
using ITensors
using LinearAlgebra
using Statistics
using Plots
using Printf

# Include ED functions from test file
include("../test/test_ed_benchmark.jl")

println("="^60)
println("Comparing TEBD vs ED with Consistent Operators")
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
# Exact Diagonalization
# =============================================================================
println("Running Exact Diagonalization...")
flush(stdout)

# Choose center site in original representation
# For even N, use N/2
l_center_original = div(N, 2)
println("  Using energy density at original site $l_center_original")

# Build Hamiltonian and energy density
H_mat = build_pxp_matrix(N; Ω=Ω)
h_mat = build_energy_density_matrix(N, l_center_original; Ω=Ω)

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
# TEBD with Matching Operator
# =============================================================================
println("Running TEBD...")
flush(stdout)

# Create merged sites
info = create_merged_sites(N)
N_merged = info.N_merged
println("  Created $N_merged merged sites from $N original sites")

# Determine which merged site contains the center
# Original site l_center_original is in merged site ceil(l_center_original / 2)
l_center_merged = div(l_center_original + 1, 2)
println("  Original site $l_center_original is in merged site $l_center_merged")

# Create energy density at the corresponding merged site
h0 = energy_density_merged(info.merged_sites, l_center_merged; Ω=Ω)

println("  Created energy density at merged site $l_center_merged")

# Check normalization
tr_h0 = real(trace_mpo(h0))
norm_h0 = sqrt(abs(inner(h0, h0)))
println("  Tr[h₀] = $tr_h0 (should be ~0)")
println("  ||h₀|| = $norm_h0")

# TEBD evolution
params = TEBDParams(dt=dt, maxdim=maxdim, cutoff=1e-10, order=4, Ω=Ω)
times_tebd_all, mpos = run_tebd_evolution(info, h0, tmax, params;
                                          save_times=times)

# Compute correlation function
C_tebd = compute_correlation_function(info.merged_sites, h0, times, mpos)

println("TEBD calculation complete!")
println()

# =============================================================================
# Comparison
# =============================================================================
println("="^60)
println("Results Comparison")
println("="^60)
println()
println("Energy density locations:")
println("  ED:   original site $l_center_original")
println("  TEBD: merged site $l_center_merged (contains original sites $(2*l_center_merged-1) and $(2*l_center_merged))")
println()
println("Time    C_ED         C_TEBD       Abs Error    Rel Error    Ratio")
println("-"^70)

errors_abs = abs.(C_ed .- C_tebd)
errors_rel = errors_abs ./ (abs.(C_ed) .+ 1e-14)
ratio = C_tebd ./ (C_ed .+ 1e-14)

for i in 1:length(times)
    t = times[i]
    if i == 1 || i == length(times) || t % 2.0 < 0.1
        @printf("%5.1f   %.6e   %.6e   %.2e   %.2e   %.3f\n",
                t, C_ed[i], C_tebd[i], errors_abs[i], errors_rel[i], ratio[i])
    end
end

println()
println("Statistics:")
println("  Max absolute error: $(maximum(errors_abs))")
println("  Mean absolute error: $(mean(errors_abs))")
println("  Max relative error: $(maximum(errors_rel))")
println("  Mean relative error: $(mean(errors_rel))")
println("  Mean C_TEBD/C_ED ratio: $(mean(ratio))")
println()

# Check if there's a constant scaling factor
println("Analysis:")
println("  C_TEBD(0) / C_ED(0) = $(C_tebd[1] / C_ed[1])")
println()
println("Note: The merged-site energy density operator is not exactly")
println("      the same as the original-site operator, so a constant")
println("      scaling factor is expected. The important test is whether")
println("      the decay dynamics match.")

# =============================================================================
# Plotting
# =============================================================================
println("\nCreating plots...")

# Plot 1: Correlation functions (normalized)
C_ed_norm = C_ed ./ C_ed[1]
C_tebd_norm = C_tebd ./ C_tebd[1]

p1 = plot(times, C_ed_norm,
         label="ED (exact)",
         marker=:circle,
         markersize=4,
         linewidth=2,
         xlabel="Time",
         ylabel="C(t) / C(0)",
         title="Normalized Energy Autocorrelation (N=$N)",
         legend=:topright,
         dpi=300)

plot!(p1, times, C_tebd_norm,
      label="TEBD (χ=$maxdim)",
      marker=:square,
      markersize=3,
      linewidth=2,
      linestyle=:dash)

# Plot 2: Absolute values on log-log
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
         legend=:topright,
         dpi=300)

plot!(p2, t_valid, abs.(C_tebd[valid]),
      label="TEBD (χ=$maxdim)",
      marker=:square,
      markersize=3,
      linewidth=2,
      linestyle=:dash)

# Plot 3: Relative error over time
p3 = plot(times, errors_rel,
         label="",
         marker=:circle,
         markersize=3,
         linewidth=2,
         xlabel="Time",
         ylabel="Relative Error",
         title="Relative Error: |C_ED - C_TEBD| / |C_ED|",
         legend=false,
         dpi=300)

# Plot 4: Ratio over time
p4 = plot(times, ratio,
         label="",
         marker=:circle,
         markersize=3,
         linewidth=2,
         xlabel="Time",
         ylabel="C_TEBD / C_ED",
         title="Ratio of TEBD to ED",
         legend=false,
         dpi=300)
hline!(p4, [1.0], linestyle=:dash, color=:gray, label="Perfect match")

# Combine plots
p_combined = plot(p1, p2, p3, p4,
                  layout=(2, 2),
                  size=(1200, 1000),
                  margin=5Plots.mm)

# Save plots
output_file = "comparison_tebd_ed_fixed_N$(N).png"
savefig(p_combined, output_file)
println("Saved plot to: $output_file")

println()
println("="^60)
println("Comparison complete!")
println("="^60)

# Display the plot
display(p_combined)
