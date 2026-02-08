# verify_gate_indices.jl
# Verify that gate application has correct index structure

using PXPTransport
using ITensors
using ITensorMPS
using LinearAlgebra
using Printf

println("="^70)
println("Verifying Gate Application Index Structure")
println("="^70)
println()

N = 10  # Need enough sites for center energy density
Ω = 1.0

# Create merged sites
info = create_merged_sites(N)
println("System: N = $N original sites → $(info.N_merged) merged sites")
println()

# =============================================================================
# 1. Check Energy Density MPO Index Structure
# =============================================================================
println("1. Energy Density MPO Index Structure")
println("-"^70)

h0 = center_energy_density_merged(info.merged_sites; Ω=Ω)
println("  Created energy density MPO")
println("  Number of tensors: $(length(h0))")
println()

println("  Index structure for each tensor:")
for i in 1:length(h0)
    T = h0[i]
    inds_T = inds(T)
    println("    Tensor $i:")
    for (j, idx) in enumerate(inds_T)
        prime_str = plev(idx) > 0 ? "'" : ""
        tags_str = tags(idx)
        println("      Index $j: dim=$(dim(idx)), tags=$tags_str, plev=$(plev(idx))")
    end
    println()
end

# =============================================================================
# 2. Check Gate Index Structure
# =============================================================================
println("2. Gate Index Structure")
println("-"^70)

dt = 0.1

# Build gates for merged sites
gates = make_trotter_gates_merged(info, dt; Ω=Ω, order=2)

println("  Created $(length(gates)) gates")
println()

# Look at the first gate
if length(gates) > 0
    gate = gates[1]
    println("  First gate index structure:")
    inds_gate = inds(gate)
    for (j, idx) in enumerate(inds_gate)
        tags_str = tags(idx)
        println("    Index $j: dim=$(dim(idx)), tags=$tags_str, plev=$(plev(idx))")
    end
    println()

    # Check if it's unitary
    s1 = info.merged_sites[1]
    s2 = info.merged_sites[2]

    println("  Gate should have indices: (s1', s2', dag(s1), dag(s2))")
    println("  Where:")
    println("    - s1', s2' are OUTPUT (primed) for bra side")
    println("    - dag(s1), dag(s2) are INPUT (unprimed) for ket side")
    println()
end

# =============================================================================
# 3. Manual Gate Application Test
# =============================================================================
println("3. Manual Gate Application Test")
println("-"^70)

# Create a simple test MPO - just identity on merged sites
println("  Creating test MPO (identity operator)...")
test_mpo = identity_mpo_merged(info.merged_sites)

println("  Initial MPO norm: ||M||² = $(real(inner(test_mpo, test_mpo)))")
println()

# Apply gate manually to first two sites
println("  Applying gate to sites 1-2...")

if length(gates) > 0
    gate = gates[1]

    # Extract 2-site MPO
    M_sub = MPO([test_mpo[1], test_mpo[2]])

    println("  Before gate application:")
    println("    M_sub[1] indices: $(inds(test_mpo[1]))")
    println("    M_sub[2] indices: $(inds(test_mpo[2]))")
    println("    Gate indices: $(inds(gate))")
    println()

    # Apply using ITensors
    println("  Method 1: Using apply([gate], M; apply_dag=true)")
    M_evolved = apply([gate], M_sub; apply_dag=true, maxdim=100, cutoff=1e-12)

    println("    After gate application:")
    println("    M_evolved[1] indices: $(inds(M_evolved[1]))")
    println("    M_evolved[2] indices: $(inds(M_evolved[2]))")
    println()

    norm_before = real(inner(M_sub, M_sub))
    norm_after = real(inner(M_evolved, M_evolved))

    println("    ||M||² before: $norm_before")
    println("    ||M||² after:  $norm_after")
    println("    Ratio: $(norm_after / norm_before)")
    println()

    if abs(norm_after / norm_before - 1.0) < 1e-6
        println("    ✓ Norm preserved (unitary evolution)")
    else
        println("    ✗ Norm NOT preserved!")
    end
    println()
end

# =============================================================================
# 4. What apply_dag=true Does
# =============================================================================
println("4. Understanding apply_dag=true")
println("-"^70)
println()

println("  For Heisenberg evolution: O(t) = U†(t) O(0) U(t)")
println()
println("  The MPO O has indices:")
println("    - Bra indices: s1', s2' (primed)")
println("    - Ket indices: dag(s1), dag(s2) (unprimed)")
println()
println("  The gate U has indices:")
println("    - Output: s1', s2' (primed)")
println("    - Input:  dag(s1), dag(s2) (unprimed)")
println()
println("  apply([gate], M; apply_dag=true) should compute:")
println("    - Contract U with ket indices of M")
println("    - Contract U† with bra indices of M")
println()
println("  This gives: U† M U, which is the Heisenberg evolution")
println()

# =============================================================================
# 5. Verify with Known Operator
# =============================================================================
println("5. Verify with Known Operator Evolution")
println("-"^70)
println()

# For identity gate (H=0), should have U=I, so O(t) = O(0)
println("  Test: Identity gate (H=0) should preserve operator")

# Create identity Hamiltonian
H_zero = zeros(ComplexF64, 9, 9)
for i in 1:9
    H_zero[i,i] = 0.0
end

U_identity = exp(-im * dt * H_zero)
println("  Unitarity check: ||U†U - I|| = $(norm(U_identity' * U_identity - I(9)))")

s1 = info.merged_sites[1]
s2 = info.merged_sites[2]
gate_identity = ITensor(U_identity, s1', s2', dag(s1), dag(s2))

# Apply to test MPO
M_test = MPO([test_mpo[1], test_mpo[2]])
M_evolved_identity = apply([gate_identity], M_test; apply_dag=true, maxdim=100, cutoff=1e-12)

# Check if they match
diff = inner(M_test, M_evolved_identity)
norm_M = inner(M_test, M_test)

println("  ⟨M_original | M_evolved⟩ = $diff")
println("  ⟨M_original | M_original⟩ = $norm_M")
println("  Overlap: $(real(diff) / real(norm_M))")
println()

if abs(real(diff) / real(norm_M) - 1.0) < 1e-6
    println("  ✓ Identity gate preserves operator (correct!)")
else
    println("  ✗ Identity gate changes operator (WRONG!)")
end

println()
println("="^70)
println("Analysis Complete")
println("="^70)
