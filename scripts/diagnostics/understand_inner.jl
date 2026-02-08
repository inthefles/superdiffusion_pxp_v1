# understand_inner.jl
# Understand exactly what inner(MPO, MPO) computes

using ITensors
using ITensorMPS
using LinearAlgebra
using Printf

println("="^70)
println("Understanding what inner(MPO, MPO) Computes")
println("="^70)
println()

# Create simple test case: identity operator on 2 qubits
sites = siteinds("S=1/2", 2)

# Identity operator
os_I = OpSum()
os_I += "Id", 1
os_I += "Id", 2
I_mpo = MPO(os_I, sites)

# Pauli operators
os_X = OpSum()
os_X += "Sx", 1
X_mpo = MPO(os_X, sites)

os_Z = OpSum()
os_Z += "Sz", 1
Z_mpo = MPO(os_Z, sites)

println("Test operators:")
println("  I: identity on site 1")
println("  X: Sx on site 1")
println("  Z: Sz on site 1")
println()

# Check inner products
println("Inner products:")
@printf("  inner(I, I) = %.6f\n", inner(I_mpo, I_mpo))
@printf("  inner(X, X) = %.6f\n", inner(X_mpo, X_mpo))
@printf("  inner(Z, Z) = %.6f\n", inner(Z_mpo, Z_mpo))
@printf("  inner(X, Z) = %.6f\n", inner(X_mpo, Z_mpo))
println()

# Now compute the same quantities as dense matrices
println("Converting to dense matrices:")

# Function to convert MPO to dense matrix
function mpo_to_matrix(M::MPO)
    d = 2  # dimension per site
    N = length(M)
    D = d^N

    mat = zeros(ComplexF64, D, D)

    # Enumerate all basis states
    for i_out in 0:(D-1)
        for i_in in 0:(D-1)
            # Extract bit patterns
            out_bits = [((i_out >> (N-k-1)) & 1) + 1 for k in 0:(N-1)]
            in_bits = [((i_in >> (N-k-1)) & 1) + 1 for k in 0:(N-1)]

            # Contract MPO for this matrix element
            val = 1.0 + 0.0im
            for k in 1:N
                s_k = siteind(M, k)
                indices_k = inds(M[k])

                # Build index value dictionary
                vals = Dict()
                vals[s_k'] = out_bits[k]
                vals[dag(s_k)] = in_bits[k]

                # Add link indices if present
                if k > 1
                    link_l = commonind(M[k-1], M[k])
                    if !isnothing(link_l)
                        # This gets tricky... need to sum over bond indices
                        # Skip for now - use ITensors contraction
                    end
                end

                if k == 1 && length(M) == 1
                    val *= M[k][s_k'=>out_bits[k], dag(s_k)=>in_bits[k]]
                end
            end

            if length(M) == 1
                mat[i_out+1, i_in+1] = val
            end
        end
    end

    # For multi-site, use ITensors' built-in contraction
    if length(M) > 1
        # Contract the MPO into a single tensor
        result = M[1]
        for k in 2:length(M)
            result = result * M[k]
        end

        # Extract matrix elements
        for i_out in 0:(D-1)
            for i_in in 0:(D-1)
                out_bits = [((i_out >> (N-k-1)) & 1) + 1 for k in 0:(N-1)]
                in_bits = [((i_in >> (N-k-1)) & 1) + 1 for k in 0:(N-1)]

                vals = Dict()
                for k in 1:N
                    s_k = siteind(M, k)
                    vals[s_k'] = out_bits[k]
                    vals[dag(s_k)] = in_bits[k]
                end

                mat[i_out+1, i_in+1] = result[vals...]
            end
        end
    end

    return mat
end

# Convert to matrices
I_mat = mpo_to_matrix(I_mpo)
X_mat = mpo_to_matrix(X_mpo)
Z_mat = mpo_to_matrix(Z_mpo)

println("  Identity operator:")
println("  Tr[I] = $(tr(I_mat))")
println("  Tr[I† I] = $(tr(I_mat' * I_mat))")
println()

println("  Sx operator:")
println("  Tr[X] = $(tr(X_mat))")
println("  Tr[X† X] = $(tr(X_mat' * X_mat))")
println()

println("  Sz operator:")
println("  Tr[Z] = $(tr(Z_mat))")
println("  Tr[Z† Z] = $(tr(Z_mat' * Z_mat))")
println()

println("Comparison:")
println("  inner(I, I) / Tr[I† I] = $(inner(I_mpo, I_mpo) / tr(I_mat' * I_mat))")
println("  inner(X, X) / Tr[X† X] = $(inner(X_mpo, X_mpo) / tr(X_mat' * X_mat))")
println("  inner(Z, Z) / Tr[Z† Z] = $(inner(Z_mpo, Z_mpo) / tr(Z_mat' * Z_mat))")
println()

println("="^70)
println("Conclusion:")
println("="^70)
println()
println("If ratios are all equal to some constant factor,")
println("then inner() computes a normalized version of Tr[M† M]")
