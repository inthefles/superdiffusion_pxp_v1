# correct_gate_application.jl
# Figure out the correct way to apply gates to MPOs

using ITensors
using ITensorMPS: MPO
using LinearAlgebra

println("Understanding MPO index structure...")
println()

# Create simple 2-site system
s = [Index(2, "s$i") for i in 1:2]

# Create an MPO representing Pauli X on site 1
# X = |0⟩⟨1| + |1⟩⟨0|
M1 = ITensor(s[1], s[1]')
M1[s[1]=>1, s[1]'=>2] = 1.0  # |0⟩⟨1|
M1[s[1]=>2, s[1]'=>1] = 1.0  # |1⟩⟨0|

M2 = ITensor(s[2], s[2]')
M2[s[2]=>1, s[2]'=>1] = 1.0  # |0⟩⟨0|
M2[s[2]=>2, s[2]'=>2] = 1.0  # |1⟩⟨1|

M_op = MPO([M1, M2])

println("Created MPO for X⊗I operator")
println("Norm²: $(real(inner(M_op, M_op)))")
println()

# For operator X, Tr(X†X) = Tr(XX) = Tr(I) = 2
# So ||X||² should be 2
# For X⊗I in 2-site system: ||X⊗I||² = Tr((X⊗I)†(X⊗I)) = Tr(I⊗I) = 4

println("Expected norm² for X⊗I: 4")
println("Actual norm²: $(real(inner(M_op, M_op)))")
println()

# Now test Heisenberg evolution with identity
# O(t) = U† O U where U = exp(-iHt)
# For H=0, U=I, so O(t) = O(0)

println("Testing Heisenberg evolution with U=I...")
println()

# Create identity gate
H_local = zeros(ComplexF64, 4, 4)  # 2x2 local Hilbert space
dt = 0.1
U = exp(-im * dt * H_local)  # U = I

println("Gate unitarity: ||U†U - I|| = $(norm(U' * U - I(4)))")

# Convert to ITensor gate
# Gate should have indices: (s1_out, s2_out, s1_in, s2_in)
# Following ITensors convention: primed = out, unprimed = in
gate = ITensor(U, s[1]', s[2]', dag(s[1]), dag(s[2]))

println("Gate created with indices:")
for idx in inds(gate)
    println("  $idx")
end
println()

# Manual application of U† O U
println("Method 1: Direct calculation of U† O U")
println()

# For a 2-site MPO M with tensors M1, M2:
# The full operator is O = sum_{abcd} M1[a,c] M2[b,d] |ab⟩⟨cd|
#
# Heisenberg evolution: O' = U† O U
# where U acts on ket |ab⟩ and U† acts on bra ⟨cd|
#
# O'[ab, cd] = sum_{a'b'c'd'} U†[ab, a'b'] M[a'b', c'd'] U[c'd', cd]
#            = sum_{a'b'c'd'} U*[a'b', ab] M[a'b', c'd'] U[c'd', cd]
#
# In tensor notation with gate = U[ab', ab]:
# Step 1: contract gate with MPO's ket indices
# Step 2: contract gate† with MPO's bra indices

# Actually, thinking about this more carefully:
# An MPO M represents: M = sum_{s,s'} M[s,s'] |s⟩⟨s'|
# Heisenberg: M(t) = U† M U = sum_{s,s'} M[s,s'] U†|s⟩⟨s'|U
#                            = sum_{s,s',a,b} M[s,s'] U†[s,a] ⟨b|U δ_{s',b}
#                            = sum_{s,s',a} M[s,s'] U*[a,s] U[s',a]
# Wait, that's not quite right either...

# Let me think in terms of matrix elements:
# M_{ab,cd} = matrix element of operator M
# After evolution: M'_{ab,cd} = sum_{a'b'c'd'} (U†)_{ab,a'b'} M_{a'b',c'd'} U_{c'd',cd}
#
# U is the forward evolution so it goes from 'in' states to 'out' states
# U† is backward, from 'out' to 'in'
#
# Actually for time evolution of operators:
# M(t) = e^{iHt} M(0) e^{-iHt}
# So: M'[out, in] = sum_{out',in'} e^{iHt}[out,out'] M[out',in'] e^{-iHt}[in',in]

println("The correct formula for operator evolution:")
println("M'[s_out, s_in'] = Σ_{s',s''} U†[s_out, s'] M[s', s''] U[s'', s_in']")
println()
println("Where U = exp(-iHt) is the state evolution operator")
println("and U† = exp(+iHt)")
println()

# For MPO in ITensors:
# M has indices (s_physical, s_physical')
# where s_physical is unprimed (ket side) and s_physical' is primed (bra side)
#
# We want: M_new[s, s'] = Σ_{a,b} U†[s,a] M[a,b] U[b,s']
#
# Since U = exp(-iHt), we have U†[s,a] = conj(U[a,s]) = conj(U)[s,a]
#
# In ITensors gate notation: gate has (s_out', s_in)
# So gate[s',s] corresponds to U[s,s']
#
# We need to:
# 1. Apply gate to contract with M's unprimed index (ket)
# 2. Apply conj(gate) to contract with M's primed index (bra)

println("This suggests the operation should be:")
println("M_temp = M * gate  (contract gate's unprimed input with M's unprimed)")
println("M_new = conj(gate) * M_temp  (contract conj(gate) with M's primed)")
println()
println("But we need to be careful about index structure...")
