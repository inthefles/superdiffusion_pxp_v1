# diagnose_merge_mpo.jl
# Detailed diagnostics for merge_mpo_pairs function

using PXPTransport
using ITensors
using ITensorMPS
using LinearAlgebra
using Printf

include("../test/test_ed_benchmark.jl")

println("="^70)
println("Diagnosing merge_mpo_pairs Function")
println("="^70)
println()

N = 10
Ω = 1.0
l_center = div(N + 1, 2)

println("System: N = $N, center site = $l_center")
println()

# =============================================================================
# Reference: ED Matrix
# =============================================================================
println("1. ED Matrix (reference)")
println("-"^70)

h_mat = build_energy_density_matrix(N, l_center; Ω=Ω)
println("  Size: $(size(h_mat))")
println("  Tr[h] = $(tr(h_mat))")
println("  Tr[h²] = $(tr(h_mat^2))")
println("  ||h||_F = $(norm(h_mat))")
println()

# =============================================================================
# Original MPO
# =============================================================================
println("2. Original MPO (ITensor on N=$N sites)")
println("-"^70)

sites_original = PXPSites(N)
h0_original = center_energy_density(sites_original; Ω=Ω)

tr_h0_orig = real(trace_mpo(h0_original))
tr_h0sq_orig = real(inner(h0_original, h0_original))

println("  Number of tensors: $(length(h0_original))")
println("  Tr[h₀] = $tr_h0_orig")
println("  Tr[h₀²] = $tr_h0sq_orig")
println("  Ratio to ED: $(tr_h0sq_orig / tr(h_mat^2))")
println()

# Check individual tensor structure
println("  Tensor structure:")
for i in 1:length(h0_original)
    inds_i = inds(h0_original[i])
    println("    Site $i: $(length(inds_i)) indices, dims = $(dims(inds_i))")
end
println()

# =============================================================================
# Manual Merging Test - Step by Step
# =============================================================================
println("3. Manual Merging - First Pair (sites 1-2)")
println("-"^70)

# Get first two tensors
T1 = h0_original[1]
T2 = h0_original[2]

println("  T1 indices: $(inds(T1))")
println("  T2 indices: $(inds(T2))")

# Contract them
T_combined = T1 * T2
println("  T_combined indices: $(inds(T_combined))")
println()

# Get physical indices
s1 = siteind(h0_original, 1)
s2 = siteind(h0_original, 2)

println("  Physical index s1: $s1, dim = $(dim(s1))")
println("  Physical index s2: $s2, dim = $(dim(s2))")
println()

# Check matrix elements
println("  Checking T_combined matrix elements:")
SPIN_DOWN = 1
SPIN_UP = 2

count_nonzero = 0
max_val = 0.0

for sp1 in 1:2, sp2 in 1:2, s1_val in 1:2, s2_val in 1:2
    try
        val = T_combined[s1'=>sp1, dag(s1)=>s1_val, s2'=>sp2, dag(s2)=>s2_val]
        if abs(val) > 1e-12
            count_nonzero += 1
            max_val = max(max_val, abs(val))
            if count_nonzero <= 5
                println("    [$sp1,$sp2|$s1_val,$s2_val] = $val")
            end
        end
    catch
        # Index not present
    end
end
println("  Total non-zero elements: $count_nonzero / 16")
println("  Max |element|: $max_val")
println()

# =============================================================================
# Check what merge_mpo_pairs produces
# =============================================================================
println("4. Merged MPO (via merge_mpo_pairs)")
println("-"^70)

info = create_merged_sites(N)
h0_merged = merge_mpo_pairs(h0_original, info.merged_sites)

tr_h0_merged = real(trace_mpo(h0_merged))
tr_h0sq_merged = real(inner(h0_merged, h0_merged))

println("  Number of merged tensors: $(length(h0_merged))")
println("  Tr[h₀_merged] = $tr_h0_merged")
println("  Tr[h₀²_merged] = $tr_h0sq_merged")
println("  Ratio to original: $(tr_h0sq_merged / tr_h0sq_orig)")
println("  Ratio to ED: $(tr_h0sq_merged / tr(h_mat^2))")
println()

# Check merged tensor structure
println("  Merged tensor structure:")
for i in 1:length(h0_merged)
    inds_i = inds(h0_merged[i])
    println("    Merged site $i: $(length(inds_i)) indices, dims = $(dims(inds_i))")
end
println()

# =============================================================================
# Direct Comparison: Manual vs merge_mpo_pairs for first merged site
# =============================================================================
println("5. Comparing Manual Merge vs merge_mpo_pairs")
println("-"^70)

# Manual merge of first merged site
s_merged_1 = info.merged_sites[1]
d_merged = 3

# Build manually
W_manual = zeros(ComplexF64, d_merged, d_merged)
function merged_state(s1, s2)
    SPIN_DOWN = 1
    SPIN_UP = 2
    if s1 == SPIN_DOWN && s2 == SPIN_DOWN
        return 1  # DD
    elseif s1 == SPIN_UP && s2 == SPIN_DOWN
        return 2  # UD
    elseif s1 == SPIN_DOWN && s2 == SPIN_UP
        return 3  # DU
    else
        return 0  # UU - forbidden
    end
end

for sp1 in 1:2, sp2 in 1:2
    for s1_val in 1:2, s2_val in 1:2
        sp_merged = merged_state(sp1, sp2)
        s_merged_val = merged_state(s1_val, s2_val)
        if sp_merged > 0 && s_merged_val > 0
            try
                val = T_combined[s1'=>sp1, dag(s1)=>s1_val, s2'=>sp2, dag(s2)=>s2_val]
                W_manual[sp_merged, s_merged_val] += val
            catch
                # This element doesn't exist in T_combined
            end
        end
    end
end

println("  Manual merge matrix W_manual (3×3):")
for i in 1:3
    @printf("    ")
    for j in 1:3
        @printf("%10.6f  ", real(W_manual[i,j]))
    end
    println()
end
println("  Tr[W_manual] = $(tr(W_manual))")
println("  Tr[W_manual²] = $(real(tr(W_manual' * W_manual)))")
println()

# Get the actual merged tensor from merge_mpo_pairs
T_merged_1 = h0_merged[1]
println("  Merged tensor from merge_mpo_pairs:")
println("  Indices: $(inds(T_merged_1))")

# Try to extract the matrix
s_m = info.merged_sites[1]
W_from_function = zeros(ComplexF64, 3, 3)
for i in 1:3, j in 1:3
    try
        W_from_function[i,j] = T_merged_1[s_m'=>i, dag(s_m)=>j]
    catch
        W_from_function[i,j] = 0.0
    end
end

println("  Matrix from merge_mpo_pairs (3×3):")
for i in 1:3
    @printf("    ")
    for j in 1:3
        @printf("%10.6f  ", real(W_from_function[i,j]))
    end
    println()
end
println("  Tr[W_function] = $(tr(W_from_function))")
println("  Tr[W_function²] = $(real(tr(W_from_function' * W_from_function)))")
println()

# Check if they match
diff = norm(W_manual - W_from_function)
println("  ||W_manual - W_function|| = $diff")
if diff < 1e-10
    println("  ✓ Manual and function results MATCH")
else
    println("  ✗ Manual and function results DIFFER")
end

println()
println("="^70)
println("Diagnosis complete!")
println("="^70)
