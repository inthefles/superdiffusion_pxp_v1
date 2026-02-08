# diagnose_inner_product.jl
# Investigate why inner(h0_merged, h0_merged) gives wrong result

using PXPTransport
using ITensors
using ITensorMPS
using LinearAlgebra
using Printf

println("="^70)
println("Diagnosing Inner Product for Merged MPO")
println("="^70)
println()

N = 10
Ω = 1.0
l_center = 5

# Create original and merged MPOs
sites_original = PXPSites(N)
h0_original = center_energy_density(sites_original; Ω=Ω)

info = create_merged_sites(N)
h0_merged = merge_mpo_pairs(h0_original, info.merged_sites)

println("Setup:")
println("  Original MPO: N = $N sites")
println("  Merged MPO: N_merged = $(info.N_merged) sites")
println()

# =============================================================================
# Check what inner() actually computes
# =============================================================================
println("1. Understanding inner(MPO, MPO)")
println("-"^70)

println("  For original MPO:")
inner_orig = inner(h0_original, h0_original)
println("    inner(h0_original, h0_original) = $inner_orig")
println()

println("  For merged MPO:")
inner_merged = inner(h0_merged, h0_merged)
println("    inner(h0_merged, h0_merged) = $inner_merged")
println()

println("  Ratio: merged/original = $(real(inner_merged) / real(inner_orig))")
println()

# =============================================================================
# Manual computation of inner product
# =============================================================================
println("2. Manual Computation of Inner Product")
println("-"^70)
println()

println("  Method: Contract all tensors step by step")
println()

# For MPO inner product: ⟨M|M⟩ = Tr[M† M]
# This means contracting M with dag(M), matching physical indices

# Start from the left
result = ITensor(1.0)

for i in 1:length(h0_merged)
    global result
    # Get the tensor and its conjugate
    T_i = h0_merged[i]
    T_i_dag = dag(T_i)

    # Contract with running result
    # First contract T_i_dag
    result = result * T_i_dag
    # Then contract T_i
    result = result * T_i

    if i <= 3
        println("  After site $i:")
        println("    Remaining indices: $(inds(result))")
        println("    Number of indices: $(length(inds(result)))")
    end
end

manual_inner = scalar(result)
println()
println("  Manual inner product result: $manual_inner")
println("  ITensors inner() result: $inner_merged")
println("  Match: $(abs(manual_inner - inner_merged) < 1e-10)")
println()

# =============================================================================
# Check individual tensor contributions
# =============================================================================
println("3. Individual Tensor Norms vs Full Inner Product")
println("-"^70)
println()

println("  Computing ||T_i||² for each tensor:")
sum_tensor_norms = 0.0
for i in 1:length(h0_merged)
    T_i = h0_merged[i]
    T_i_dag = dag(T_i)

    # Contract T_i with T_i_dag
    T_contracted = T_i * T_i_dag
    norm_sq = real(scalar(T_contracted))

    sum_tensor_norms += norm_sq
    @printf("    Site %d: ||T_%d||² = %.6f\n", i, i, norm_sq)
end

println()
@printf("  Sum of individual norms: Σ ||T_i||² = %.6f\n", sum_tensor_norms)
@printf("  Full inner product: ⟨M|M⟩ = %.6f\n", real(inner_merged))
@printf("  Ratio: %.6f\n", real(inner_merged) / sum_tensor_norms)
println()

println("  Why are they different?")
println("  - Σ ||T_i||² treats each tensor independently")
println("  - ⟨M|M⟩ contracts the full MPO network")
println("  - The difference comes from bond index contractions")
println()

# =============================================================================
# Understand the bond structure
# =============================================================================
println("4. Analyzing Bond Structure")
println("-"^70)
println()

println("  Checking bond dimensions:")
for i in 1:(length(h0_merged)-1)
    link_i = linkind(h0_merged, i)
    if !isnothing(link_i)
        @printf("    Bond %d: dimension = %d\n", i, dim(link_i))
    end
end
println()

# =============================================================================
# Compare with a simpler case: single-site MPO
# =============================================================================
println("5. Test Case: Single Merged Site")
println("-"^70)
println()

# Create a minimal system with just one merged site
N_small = 2
sites_small = PXPSites(N_small)
h_small = center_energy_density(sites_small; Ω=Ω)

info_small = create_merged_sites(N_small)
h_small_merged = merge_mpo_pairs(h_small, info_small.merged_sites)

println("  Small system: N = $N_small → 1 merged site")
println()

inner_small_orig = inner(h_small, h_small)
inner_small_merged = inner(h_small_merged, h_small_merged)

println("  Original (2 sites): inner = $inner_small_orig")
println("  Merged (1 site): inner = $inner_small_merged")
println("  Ratio: $(real(inner_small_merged) / real(inner_small_orig))")
println()

# For single tensor, should match
T_single = h_small_merged[1]
T_single_dag = dag(T_single)
T_contracted = T_single * T_single_dag
single_norm = real(scalar(T_contracted))

println("  Single tensor norm: $single_norm")
println("  Match with inner: $(abs(single_norm - real(inner_small_merged)) < 1e-10)")
println()

# =============================================================================
# Checking trace_mpo
# =============================================================================
println("6. Checking trace_mpo Function")
println("-"^70)
println()

tr_orig = trace_mpo(h0_original)
tr_merged = trace_mpo(h0_merged)

println("  Tr[h_original] = $tr_orig")
println("  Tr[h_merged] = $tr_merged")
println("  (Both should be ~0 for traceless operator)")
println()

# =============================================================================
# The real issue: What does inner(MPO, MPO) compute?
# =============================================================================
println("7. What Does inner(MPO, MPO) Actually Compute?")
println("-"^70)
println()

println("  For an MPO M with physical indices s_i (unprimed) and s'_i (primed):")
println("  inner(M, M) should compute Tr[M† M]")
println()
println("  This involves:")
println("  1. Taking dag(M) which swaps primes: s_i ↔ s'_i")
println("  2. Contracting dag(M) with M")
println("  3. The physical indices contract: s_i in dag(M) with s_i in M")
println("  4. The primed indices contract: s'_i in dag(M) with s'_i in M")
println()

println("  For merged sites with dimension 3:")
println("  - Each site has 3×3 = 9 matrix elements")
println("  - The contraction should sum over all these")
println()

# Check if the issue is with how merged sites contract
println("  Checking physical index dimensions:")
for i in 1:min(3, length(h0_merged))
    s_i = siteind(h0_merged, i)
    @printf("    Site %d: dimension = %d\n", i, dim(s_i))
end

println()
println("="^70)
println("Diagnosis complete!")
println("="^70)
println()

println("SUMMARY:")
println("  Σ ||T_i||² = $(sum_tensor_norms) (sum of individual tensor norms)")
println("  ⟨M|M⟩ = $(real(inner_merged)) (full MPO inner product)")
println("  Ratio = $(real(inner_merged) / sum_tensor_norms)")
println()
println("  The large difference suggests the bond contractions are")
println("  causing significant cancellation or there's an issue with")
println("  how the merged MPO network is being contracted.")
