# check_energy_location.jl
# Find where the energy density is actually located after merging

using PXPTransport
using ITensors
using ITensorMPS
using LinearAlgebra
using Printf

println("="^70)
println("Locating Energy Density in Merged Representation")
println("="^70)
println()

N = 10
Ω = 1.0
l_center = 5  # Center at original site 5

println("System: N = $N, energy density at original site $l_center")
println()

# Create original MPO
sites_original = PXPSites(N)
h0_original = center_energy_density(sites_original; Ω=Ω)

# Merge
info = create_merged_sites(N)
h0_merged = merge_mpo_pairs(h0_original, info.merged_sites)

println("Original sites → Merged sites mapping:")
for i_merged in 1:info.N_merged
    i_left = 2 * i_merged - 1
    i_right = 2 * i_merged
    println("  Merged site $i_merged ← original sites ($i_left, $i_right)")
end
println()

# Check norm of each merged tensor
println("Norm contribution from each merged site:")
println("(checking ||T_i||_F for each tensor)")
println()

total_norm_sq = 0.0
for i in 1:length(h0_merged)
    global total_norm_sq
    # Get the tensor
    T_i = h0_merged[i]

    # Compute Frobenius norm of this tensor
    # Contract T with its conjugate
    T_dag = dag(T_i)
    T_contracted = T_i * T_dag
    norm_i_sq = real(scalar(T_contracted))
    total_norm_sq += norm_i_sq

    i_left = 2 * i - 1
    i_right = 2 * i

    @printf("  Merged site %d (orig %d-%d): ||T||² = %.6e\n",
            i, i_left, i_right, norm_i_sq)
end

println()
@printf("  Total: Σ ||T_i||² = %.6e\n", total_norm_sq)
@printf("  Tr[h²_merged] = %.6e\n", real(inner(h0_merged, h0_merged)))
println()

# Check the center merged site more carefully
i_merged_center = div(l_center + 1, 2)  # Site 5 is in merged site 3
println("Original site $l_center is in merged site $i_merged_center")
println()

# Extract the matrix for the center merged site
T_center = h0_merged[i_merged_center]
s_m = info.merged_sites[i_merged_center]

println("Center merged tensor:")
println("  Indices: $(inds(T_center))")
println()

# Try to extract the local matrix (without bond indices)
# This is tricky because the tensor also has bond indices

println("Attempting to extract physical matrix...")
# For a bulk tensor with indices (s', s, link_l, link_r)
# We need to trace over bond indices

n_indices = length(inds(T_center))
println("  Number of indices: $n_indices")

if n_indices == 4
    # Bulk tensor: (s', s, link_l, link_r)
    # Sum over bond indices to get effective 2-index tensor
    inds_T = inds(T_center)

    # Identify which are physical and which are bonds
    s_prime = nothing
    s_unprime = nothing
    link_indices = []

    for idx in inds_T
        if hasplev(idx)
            s_prime = idx
        elseif hastags(idx, "Site")
            s_unprime = idx
        else
            push!(link_indices, idx)
        end
    end

    println("  Physical (primed): $s_prime")
    println("  Physical (unprimed): $s_unprime")
    println("  Bond indices: $(length(link_indices))")

    # Contract over bond indices by setting them equal
    # Or just look at norm
    println()
    println("  ||T_center||_F² = $(real(inner(T_center, T_center')))")
end

println()
println("="^70)
println("Analysis complete!")
println("="^70)
