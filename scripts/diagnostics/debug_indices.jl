# debug_indices.jl
# Understand the index structure of gates and MPOs

using PXPTransport
using ITensors
using ITensorMPS: MPO
using LinearAlgebra

println("Debugging Index Structure")
println("="^70)

# Create simple system
N = 8
info = create_merged_sites(N)
sites = info.merged_sites

println("Merged sites:")
for (i, s) in enumerate(sites)
    println("  Site $i: $s")
end
println()

# Create a simple MPO
h0 = center_energy_density_merged(sites; Ω=1.0)

println("MPO structure:")
for (i, tensor) in enumerate(h0)
    println("  Tensor $i indices:")
    for idx in inds(tensor)
        println("    $idx")
    end
end
println()

# Create a gate
H_local = pxp_local_hamiltonian(; Ω=1.0)
dt = 0.1
U = exp(-im * dt * H_local)
gate = ITensor(U, sites[1]', sites[2]', dag(sites[1]), dag(sites[2]))

println("Gate indices:")
for idx in inds(gate)
    println("  $idx")
end
println()

# Check if dag(sites[1]) == sites[1]
println("Index comparison:")
println("  sites[1] = $(sites[1])")
println("  dag(sites[1]) = $(dag(sites[1]))")
println("  sites[1] == dag(sites[1]): $(sites[1] == dag(sites[1]))")
println("  sites[1]' = $(sites[1]')")
println()

# Try contraction
M_test = h0[1] * h0[2]
println("Combined MPO indices:")
for idx in inds(M_test)
    println("  $idx")
end
println()

# See what happens when we contract
println("Testing contraction: M_test * gate")
try
    M_temp = M_test * gate
    println("Result indices:")
    for idx in inds(M_temp)
        println("  $idx")
    end
    println()

    # Now try gate_dag
    gate_dag = swapprime(dag(gate), 0 => 1)
    println("gate_dag indices:")
    for idx in inds(gate_dag)
        println("  $idx")
    end
    println()

    println("Testing: gate_dag * M_temp")
    M_new = gate_dag * M_temp
    println("Final result indices:")
    for idx in inds(M_new)
        println("  $idx")
    end

catch e
    println("Error: $e")
end
