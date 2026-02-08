# test_gate_application.jl
# Test the gate application logic directly

using PXPTransport
using ITensors
using ITensorMPS: MPO
using LinearAlgebra

println("="^70)
println("Testing Gate Application to MPO")
println("="^70)
println()

# Create a simple 2-site MPO to test
# Use identity operator for simplicity
s1 = Index(3, "s1")
s2 = Index(3, "s2")

# Create identity MPO: I = sum_i |i><i|
# In MPO form: I[s, s'] = delta_{s, s'}
I1 = ITensor(s1, s1')
I2 = ITensor(s2, s2')

for i in 1:3
    I1[s1=>i, s1'=>i] = 1.0
    I2[s2=>i, s2'=>i] = 1.0
end

# Create MPO = I1 * I2 (product operator)
M = MPO([I1, I2])

println("Initial MPO norm²: ||M||² = $(real(inner(M, M)))")

# Create a simple unitary gate
# Just use identity for now
H_local = zeros(ComplexF64, 9, 9)
for i in 1:9
    H_local[i,i] = 0.0  # Zero Hamiltonian -> U = I
end
dt = 0.1
U = exp(-im * dt * H_local)

println("Gate unitarity check: ||U†U - I|| = $(norm(U' * U - I(9)))")

# Convert to ITensor
gate = ITensor(U, s1', s2', dag(s1), dag(s2))

# Apply gate using the function
M_test = copy(M)
apply_2site_gate_to_merged_MPO!(M_test, gate, 1; maxdim=100, cutoff=1e-10)

println("After applying identity gate:")
println("  ||M||² = $(real(inner(M_test, M_test)))")
println("  Ratio = $(real(inner(M_test, M_test)) / real(inner(M, M)))")

# Now try with a non-trivial Hamiltonian
println()
println("Testing with non-trivial Hamiltonian...")
H_local_real = pxp_local_hamiltonian(; Ω=1.0)
U_real = exp(-im * dt * H_local_real)

println("Real gate unitarity: ||U†U - I|| = $(norm(U_real' * U_real - I(9)))")

gate_real = ITensor(U_real, s1', s2', dag(s1), dag(s2))

M_test2 = copy(M)
apply_2site_gate_to_merged_MPO!(M_test2, gate_real, 1; maxdim=100, cutoff=1e-10)

println("After applying PXP gate:")
println("  ||M||² = $(real(inner(M_test2, M_test2)))")
println("  Ratio = $(real(inner(M_test2, M_test2)) / real(inner(M, M)))")

println()
println("="^70)
