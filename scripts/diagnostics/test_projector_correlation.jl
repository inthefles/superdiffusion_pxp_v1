# test_projector_correlation.jl
# Test correlation function with explicit projector vs without

using PXPTransport
using ITensors
using LinearAlgebra
using Statistics
using Printf

# Include ED functions
include("../test/test_ed_benchmark.jl")

println("="^70)
println("Testing Correlation Function with Explicit Projector")
println("="^70)
println()

# Parameters
N = 10
Ω = 1.0
tmax = 10.0
dt = 0.05
maxdim = 256

times = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]

println("System parameters:")
println("  N = $N")
println("  Ω = $Ω")
println("  tmax = $tmax")
println("  maxdim = $maxdim")
println()

# =============================================================================
# ED (Exact Diagonalization) - Reference
# =============================================================================
println("Running ED (reference)...")

l_center = div(N + 1, 2)
H_mat = build_pxp_matrix(N; Ω=Ω)
h_mat = build_energy_density_matrix(N, l_center; Ω=Ω)

C_ed = [exact_correlation(H_mat, h_mat, t) for t in times]

println("  ED: Tr[h²] = $(tr(h_mat^2))")
println("  ED: C(0) = $(C_ed[1])")
println()

# =============================================================================
# TEBD Setup
# =============================================================================
println("Setting up TEBD...")

# Create original sites and operators
sites_original = PXPSites(N)
h0_original = center_energy_density(sites_original; Ω=Ω)
P_original = projector_mpo(sites_original)

println("  Created energy density at original site $l_center")
println("  Tr[h₀] = $(real(trace_mpo(h0_original)))")
println("  Tr[h₀²] = $(real(inner(h0_original, h0_original)))")
println("  Tr[P] = $(real(trace_mpo(P_original)))")
println()

# Convert to merged representation
info = create_merged_sites(N)
h0_merged = merge_mpo_pairs(h0_original, info.merged_sites)
P_merged = merge_mpo_pairs(P_original, info.merged_sites)

println("  Merged to $(info.N_merged) sites")
println("  Tr[h₀_merged] = $(real(trace_mpo(h0_merged)))")
println("  Tr[h₀²_merged] = $(real(inner(h0_merged, h0_merged)))")
println("  Tr[P_merged] = $(real(trace_mpo(P_merged)))")
println()

# =============================================================================
# TEBD Evolution
# =============================================================================
println("Running TEBD evolution...")

params = TEBDParams(dt=dt, maxdim=maxdim, cutoff=1e-10, order=4, Ω=Ω)
times_all, mpos_merged = run_tebd_evolution(info, h0_merged, tmax, params;
                                            save_times=times)

println("  Evolution complete")
println()

# =============================================================================
# Compute Correlations Three Ways
# =============================================================================
println("Computing correlations...")

# Method 1: Without projector (current implementation)
println("  Method 1: C(t) = Tr[h₀ hₜ] - Tr[h₀]²")
C_no_proj = compute_correlation_function(info.merged_sites, h0_merged,
                                         times, mpos_merged)

# Method 2: With explicit projector
println("  Method 2: C(t) with ⟨...⟩ = Tr[P ...]/Tr[P]")
C_with_proj = compute_correlation_with_projector(info.merged_sites, h0_merged,
                                                 times, mpos_merged, P_merged)

println()

# =============================================================================
# Comparison
# =============================================================================
println("="^70)
println("Results Comparison")
println("="^70)
println()

println("Method 1 (no projector): C(0) = $(C_no_proj[1])")
println("Method 2 (with projector): C(0) = $(C_with_proj[1])")
println("ED (reference): C(0) = $(C_ed[1])")
println()

println("Time    C_ED         C_no_proj    C_with_proj  Err1      Err2")
println("-"^70)

for i in 1:length(times)
    t = times[i]
    err1 = abs(C_ed[i] - C_no_proj[i]) / (abs(C_ed[i]) + 1e-14)
    err2 = abs(C_ed[i] - C_with_proj[i]) / (abs(C_ed[i]) + 1e-14)

    @printf("%5.1f   %11.4e  %11.4e  %11.4e  %.3f  %.3f\n",
            t, C_ed[i], C_no_proj[i], C_with_proj[i], err1, err2)
end

println()
println("Summary:")
println("  Method 1 (no projector) mean rel error: $(mean(abs.(C_ed .- C_no_proj) ./ (abs.(C_ed) .+ 1e-14)))")
println("  Method 2 (with projector) mean rel error: $(mean(abs.(C_ed .- C_with_proj) ./ (abs.(C_ed) .+ 1e-14)))")
println()

# Check if using projector improves agreement
ratio_improvement = mean(abs.(C_ed .- C_no_proj)) / mean(abs.(C_ed .- C_with_proj))
println("  Error reduction factor: $(ratio_improvement)x")
if ratio_improvement > 1.1
    println("  ✓ Using explicit projector IMPROVES agreement with ED")
elseif ratio_improvement < 0.9
    println("  ✗ Using explicit projector WORSENS agreement with ED")
else
    println("  ≈ Using explicit projector has minimal effect")
end

println()
println("="^70)
println("Test complete!")
println("="^70)
