# debug_tebd.jl
# Systematic debugging of TEBD implementation

using PXPTransport
using ITensors
using ITensors: inds
using LinearAlgebra
using Printf
using Statistics

println("="^70)
println("TEBD Implementation Debugging")
println("="^70)
println()

# Test parameters
N = 6  # Small system that sometimes works
Ω = 1.0
dt = 0.05

# Create merged sites
info = create_merged_sites(N)
println("System: N=$N, N_merged=$(info.N_merged)")
println()

# =============================================================================
# Test 1: Check Hermiticity of Local Hamiltonian
# =============================================================================
println("Test 1: Hermiticity of Local Hamiltonian")
println("-"^70)

H_local = pxp_local_hamiltonian(; Ω=Ω, λ=0.0, δ=0.0, ξ=0.0)
println("Local Hamiltonian size: $(size(H_local))")

# Check Hermiticity
hermiticity_error = norm(H_local - H_local')
println("Hermiticity error: ||H - H†|| = $hermiticity_error")

if hermiticity_error > 1e-10
    println("❌ PROBLEM: Hamiltonian is not Hermitian!")
    println("Max element difference: $(maximum(abs.(H_local - H_local')))")
else
    println("✓ Hamiltonian is Hermitian")
end

# Check eigenvalues are real
eigenvalues = eigvals(H_local)
max_imag = maximum(abs.(imag(eigenvalues)))
println("Max imaginary part of eigenvalues: $max_imag")

if max_imag > 1e-10
    println("❌ PROBLEM: Eigenvalues have imaginary parts!")
else
    println("✓ Eigenvalues are real")
end

println()

# =============================================================================
# Test 2: Check Unitarity of Trotter Gates
# =============================================================================
println("Test 2: Unitarity of Trotter Gates")
println("-"^70)

# Create gates with 2nd order Trotter (simpler)
gates_2nd = make_trotter_gates_merged(info, dt; Ω=Ω, λ=0.0, δ=0.0, ξ=0.0, order=2)
println("Number of gates (2nd order): $(length(gates_2nd))")

# Check unitarity of each gate
global max_unitarity_error = 0.0
problematic_gates = []

for (i, gate) in enumerate(gates_2nd)
    # Convert gate to matrix
    gate_inds = ITensors.inds(gate)
    if length(gate_inds) != 4
        println("  Gate $i: has $(length(gate_inds)) indices (expected 4)")
        continue
    end

    # Get combined index for rows and columns
    row_inds = (gate_inds[1], gate_inds[2])
    col_inds = (gate_inds[3], gate_inds[4])

    # Convert to dense matrix (4D tensor → 2D matrix)
    gate_array = Array(gate, row_inds..., col_inds...)
    # gate_array is (d1, d2, d1', d2') where d1=d2=3 (merged site dim)
    # Reshape to (d1*d2, d1'*d2') = (9, 9)
    d1, d2, d1p, d2p = size(gate_array)
    gate_matrix = reshape(gate_array, d1*d2, d1p*d2p)

    # Check unitarity: U†U should be identity
    unitarity_error = norm(gate_matrix' * gate_matrix - I(d1*d2))

    if unitarity_error > 1e-8
        push!(problematic_gates, (i, unitarity_error))
        global max_unitarity_error = max(max_unitarity_error, unitarity_error)
    end
end

if length(problematic_gates) > 0
    println("❌ PROBLEM: Found $(length(problematic_gates)) non-unitary gates!")
    println("Max unitarity error: ||U†U - I|| = $max_unitarity_error")
    println("Problematic gates:")
    for (i, err) in problematic_gates[1:min(5, length(problematic_gates))]
        @printf("  Gate %d: error = %.2e\n", i, err)
    end
else
    println("✓ All gates are unitary (within tolerance)")
end

println()

# =============================================================================
# Test 3: Check Energy Density Operator Properties
# =============================================================================
println("Test 3: Energy Density Operator Properties")
println("-"^70)

h0_mpo = center_energy_density_merged(info.merged_sites; Ω=Ω)
println("Energy density MPO length: $(length(h0_mpo))")

# Check for NaNs/Infs
global has_nan = false
global has_inf = false
for (i, tensor) in enumerate(h0_mpo)
    arr = array(tensor)
    if any(isnan, arr)
        println("  Site $i: contains NaN")
        global has_nan = true
    end
    if any(isinf, arr)
        println("  Site $i: contains Inf")
        global has_inf = true
    end
end

if has_nan || has_inf
    println("❌ PROBLEM: Initial operator contains NaN or Inf!")
else
    println("✓ Initial operator is finite")
end

# Compute operator norm (Frobenius norm approximation)
println()
println("Computing operator norm...")
h0_norm_sq = inner(h0_mpo, h0_mpo)
h0_norm_sq_real = real(h0_norm_sq)
println("||h0||² = $h0_norm_sq_real")

if h0_norm_sq_real < 0
    println("❌ PROBLEM: Negative norm squared!")
elseif isnan(h0_norm_sq_real) || isinf(h0_norm_sq_real)
    println("❌ PROBLEM: Norm is NaN or Inf!")
else
    println("✓ Operator norm is positive and finite")
end

println()

# =============================================================================
# Test 4: Single TEBD Step Evolution
# =============================================================================
println("Test 4: Single TEBD Step")
println("-"^70)

h_evolved = copy(h0_mpo)

h_before = real(inner(h_evolved, h_evolved))
println("Before evolution:")
println("  ||h||² = $h_before")

# Apply one even-odd sweep
try
    tebd_step_merged!(h_evolved, gates_2nd, info.merged_sites; maxdim=64, cutoff=1e-10)

    h_norm_after = real(inner(h_evolved, h_evolved))
    println("After evolution:")
    println("  ||h||² = $h_norm_after")

    # Check for problems
    growth_factor = h_norm_after / h_before
    if isnan(h_norm_after) || isinf(h_norm_after)
        println("❌ PROBLEM: Norm became NaN or Inf after evolution!")
    elseif growth_factor > 10.0
        println("❌ PROBLEM: Norm grew by factor of $growth_factor!")
    elseif growth_factor < 0.1
        println("❌ PROBLEM: Norm decayed by factor of $growth_factor!")
    else
        println("✓ Norm changed by factor of $growth_factor (reasonable)")
    end

    # Check for NaN/Inf in evolved operator
    has_nan_after = false
    for (i, tensor) in enumerate(h_evolved)
        arr = array(tensor)
        if any(isnan, arr) || any(isinf, arr)
            println("  ❌ Site $i contains NaN or Inf after evolution")
            has_nan_after = true
        end
    end

    if !has_nan_after
        println("✓ No NaN/Inf in evolved operator")
    end

catch e
    println("❌ ERROR during TEBD step:")
    println("  $e")
end

println()

# =============================================================================
# Test 5: Multiple TEBD Steps
# =============================================================================
println("Test 5: Multiple TEBD Steps (10 steps)")
println("-"^70)

h_test = copy(h0_mpo)
norms = [real(inner(h_test, h_test))]

for step in 1:10
    try
        tebd_step_merged!(h_test, gates_2nd, info.merged_sites; maxdim=64, cutoff=1e-10)
        norm_sq = real(inner(h_test, h_test))
        push!(norms, norm_sq)

        if isnan(norm_sq) || isinf(norm_sq)
            println("  Step $step: ❌ Norm = $norm_sq")
            break
        else
            @printf("  Step %d: ||h||² = %.6e (ratio = %.3f)\n",
                    step, norm_sq, norm_sq / norms[1])
        end
    catch e
        println("  Step $step: ❌ Error: $e")
        break
    end
end

println()

# Analyze norm growth
if length(norms) > 1
    growth_factors = [norms[i] / norms[i-1] for i in 2:length(norms)]
    avg_growth = exp(mean(log.(growth_factors)))

    println("Norm growth analysis:")
    println("  Average growth factor per step: $avg_growth")

    if avg_growth > 1.1
        println("  ❌ PROBLEM: Norm is growing exponentially!")
    elseif avg_growth < 0.9
        println("  ❌ PROBLEM: Norm is decaying exponentially!")
    else
        println("  ✓ Norm is approximately conserved")
    end
end

println()
println("="^70)
println("Debugging Summary")
println("="^70)
println("Check the results above to identify the source of the bug.")
println("="^70)
