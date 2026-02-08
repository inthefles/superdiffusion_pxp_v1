# test_inner_carefully.jl
# Carefully test what inner(MPO, MPO) computes

using PXPTransport
using ITensors
using ITensorMPS
using LinearAlgebra
using Printf

println("="^70)
println("Testing inner(MPO, MPO) Carefully")
println("="^70)
println()

# =============================================================================
# Test 1: Simple 2-site system to understand inner()
# =============================================================================
println("Test 1: Simple 2-site system with known operator")
println("-"^70)

# Create simple 2-site Pauli X operator: X ⊗ I
sites = siteinds("S=1/2", 2)

# Create MPO for X ⊗ I using OpSum
os = OpSum()
os += "Sx", 1
X_at_1 = MPO(os, sites)

println("  Created MPO for Sx at site 1")
println("  Number of tensors: $(length(X_at_1))")

# Check what inner computes
inner_XX = inner(X_at_1, X_at_1)
println("  inner(X, X) = $inner_XX")

# For Pauli X: X† X = I, so Tr[X† X] = Tr[I] = 2^N = 4 for 2 qubits
println("  Expected: Tr[X† X] = 4 (for 2 qubits)")
println("  Match: $(abs(inner_XX - 4.0) < 1e-10)")
println()

# =============================================================================
# Test 2: Check with a traceless operator
# =============================================================================
println("Test 2: Traceless operator (Sz)")
println("-"^70)

os_z = OpSum()
os_z += "Sz", 1
Sz_at_1 = MPO(os_z, sites)

inner_ZZ = inner(Sz_at_1, Sz_at_1)
println("  inner(Sz, Sz) = $inner_ZZ")

# For Pauli Z on first site: Sz = (1/2) σz
# Tr[(1/2 σz)²] = (1/4) Tr[σz²] = (1/4) Tr[I] = (1/4) × 4 = 1
println("  Expected: ~1.0 for Sz ⊗ I")
println("  Match: $(abs(inner_ZZ - 1.0) < 1e-3)")
println()

# =============================================================================
# Test 3: Check merged sites with simple operator
# =============================================================================
println("Test 3: Merged sites with dimension 3")
println("-"^70)

# Create merged site indices (dimension 3)
s_merged = Index(3, "Site,n=1")
sites_merged = [s_merged]

# Create simple diagonal operator on merged site
# M[i,j] = δ_ij * i (so M[1,1]=1, M[2,2]=2, M[3,3]=3)
M_mat = zeros(Float64, 3, 3)
for i in 1:3
    M_mat[i,i] = Float64(i)
end

println("  Operator matrix:")
for i in 1:3
    @printf("    ")
    for j in 1:3
        @printf("%5.2f ", M_mat[i,j])
    end
    println()
end

M_tensor = ITensor(M_mat, s_merged', dag(s_merged))
M_mpo = MPO([M_tensor])

inner_MM = inner(M_mpo, M_mpo)
println()
println("  inner(M, M) = $inner_MM")

# Expected: Tr[M† M] = Tr[M²] = 1² + 2² + 3² = 1 + 4 + 9 = 14
expected_MM = tr(M_mat' * M_mat)
println("  Expected: Tr[M† M] = $expected_MM")
println("  Match: $(abs(inner_MM - expected_MM) < 1e-10)")
println()

# =============================================================================
# Test 4: What does inner() do for 2-tensor MPO?
# =============================================================================
println("Test 4: Two-site merged MPO")
println("-"^70)

s1_merged = Index(3, "Site,n=1")
s2_merged = Index(3, "Site,n=2")
link = Index(2, "Link")

# Create two tensors with a bond
# T1: (s1', s1, link)
T1_arr = zeros(Float64, 3, 3, 2)
T1_arr[1,1,1] = 1.0
T1_arr[2,2,2] = 1.0
T1 = ITensor(T1_arr, s1_merged', dag(s1_merged), link)

# T2: (s2', s2, link)
T2_arr = zeros(Float64, 3, 3, 2)
T2_arr[1,1,1] = 1.0
T2_arr[3,3,2] = 1.0
T2 = ITensor(T2_arr, s2_merged', dag(s2_merged), dag(link))

M2_mpo = MPO([T1, T2])

inner_M2M2 = inner(M2_mpo, M2_mpo)
println("  inner(M2, M2) = $inner_M2M2")

# Manual calculation: contract the network
# inner() should compute Tr[M† M]
# For two sites, this is more complex...

println()

# =============================================================================
# Test 5: Compare original vs merged MPO for same operator
# =============================================================================
println("Test 5: Same operator, different representations")
println("-"^70)

N = 4  # Small system
sites_orig = PXPSites(N)
h_orig = center_energy_density(sites_orig; Ω=1.0)

println("  Original MPO: $N sites")
inner_orig = inner(h_orig, h_orig)
println("  inner(h_orig, h_orig) = $inner_orig")

# Merge
info = create_merged_sites(N)
h_merged = merge_mpo_pairs(h_orig, info.merged_sites)

println()
println("  Merged MPO: $(info.N_merged) sites")
inner_merged = inner(h_merged, h_merged)
println("  inner(h_merged, h_merged) = $inner_merged")

println()
println("  Ratio: merged/original = $(real(inner_merged) / real(inner_orig))")

# If merge_mpo_pairs is correct, these SHOULD match (or be proportional)
# If they don't, either:
# 1. merge_mpo_pairs has a bug
# 2. inner() handles merged sites differently
# 3. The operators are fundamentally different after merging

println()
println("="^70)
println("Analysis")
println("="^70)
println()

println("If inner() is working correctly:")
println("  - Tests 1-3 should pass ✓")
println("  - Test 5 ratio should be close to 1.0")
println()
println("If merge_mpo_pairs loses information:")
println("  - Tests 1-3 pass")
println("  - Test 5 ratio << 1.0")
println()
