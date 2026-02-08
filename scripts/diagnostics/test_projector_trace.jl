# test_projector_trace.jl
# Test the trace of the global projector in merged representation

using PXPTransport
using ITensors
using ITensorMPS
using LinearAlgebra
using Printf

println("="^70)
println("Testing Trace of Global Projector")
println("="^70)
println()

N = 10
println("System size: N = $N")
println()

# =============================================================================
# Expected trace: dimension of constrained Hilbert space
# =============================================================================
println("1. Expected Trace (Constrained Hilbert Space Dimension)")
println("-"^70)

D_constrained = constrained_dim(N)
println("  Constrained dimension D = $D_constrained (Fibonacci number)")
println("  This is the number of valid states with Rydberg blockade")
println()

# =============================================================================
# Original representation
# =============================================================================
println("2. Projector in Original Representation")
println("-"^70)

sites_original = PXPSites(N)
P_original = projector_mpo(sites_original)

println("  Created projector MPO on $N sites")
println("  Number of tensors: $(length(P_original))")

# Check tensor structure
println("  Bond dimensions:")
for i in 1:(length(P_original)-1)
    link = linkind(P_original, i)
    if !isnothing(link)
        @printf("    Bond %d: χ = %d\n", i, dim(link))
    end
end
println()

# Compute trace
tr_P_orig = real(trace_mpo(P_original))
println("  Tr[P_original] = $tr_P_orig")
println("  Expected: $D_constrained")
println("  Match: $(abs(tr_P_orig - D_constrained) / D_constrained < 1e-10)")
println()

# =============================================================================
# Merged representation
# =============================================================================
println("3. Projector in Merged Representation")
println("-"^70)

info = create_merged_sites(N)
println("  Created $(info.N_merged) merged sites from $N original sites")

# Method 1: Build directly using merged sites
P_merged_direct = projector_mpo_merged(info.merged_sites)
println()
println("  Method 1: projector_mpo_merged()")
println("  Number of tensors: $(length(P_merged_direct))")

# Check bond dimensions
println("  Bond dimensions:")
for i in 1:(length(P_merged_direct)-1)
    link = linkind(P_merged_direct, i)
    if !isnothing(link)
        @printf("    Bond %d: χ = %d\n", i, dim(link))
    end
end
println()

# Compute trace
tr_P_merged_direct = real(trace_mpo(P_merged_direct))
println("  Tr[P_merged_direct] = $tr_P_merged_direct")
println("  Expected: $D_constrained")
println("  Ratio to expected: $(tr_P_merged_direct / D_constrained)")
println()

# Method 2: Merge the original projector
P_merged_converted = merge_mpo_pairs(P_original, info.merged_sites)
println("  Method 2: merge_mpo_pairs(P_original)")

tr_P_merged_converted = real(trace_mpo(P_merged_converted))
println("  Tr[P_merged_converted] = $tr_P_merged_converted")
println("  Expected: $D_constrained")
println("  Ratio to expected: $(tr_P_merged_converted / D_constrained)")
println()

# =============================================================================
# Comparison
# =============================================================================
println("4. Comparison")
println("-"^70)
println()

@printf("  Tr[P_original]           = %.6e\n", tr_P_orig)
@printf("  Tr[P_merged_direct]      = %.6e\n", tr_P_merged_direct)
@printf("  Tr[P_merged_converted]   = %.6e\n", tr_P_merged_converted)
@printf("  Expected (D_constrained) = %.6e\n", Float64(D_constrained))
println()

# Ratios
println("  Ratios:")
@printf("    P_merged_direct / P_original   = %.6f\n", tr_P_merged_direct / tr_P_orig)
@printf("    P_merged_converted / P_original = %.6f\n", tr_P_merged_converted / tr_P_orig)
@printf("    P_merged_direct / expected     = %.6f\n", tr_P_merged_direct / D_constrained)
@printf("    P_merged_converted / expected  = %.6f\n", tr_P_merged_converted / D_constrained)
println()

# =============================================================================
# Analysis
# =============================================================================
println("5. Analysis")
println("-"^70)
println()

println("The projector should satisfy:")
println("  - Tr[P] = dimension of constrained Hilbert space = $D_constrained")
println("  - This should be preserved in both representations")
println()

if abs(tr_P_orig - D_constrained) / D_constrained < 1e-6
    println("  ✓ Original projector trace is correct")
else
    println("  ✗ Original projector trace is WRONG!")
end

if abs(tr_P_merged_direct - D_constrained) / D_constrained < 1e-6
    println("  ✓ Merged projector (direct) trace is correct")
else
    println("  ✗ Merged projector (direct) trace is WRONG!")
    println("    Error: $(abs(tr_P_merged_direct - D_constrained) / D_constrained * 100)%")
end

if abs(tr_P_merged_converted - D_constrained) / D_constrained < 1e-6
    println("  ✓ Merged projector (converted) trace is correct")
else
    println("  ✗ Merged projector (converted) trace is WRONG!")
    println("    Error: $(abs(tr_P_merged_converted - D_constrained) / D_constrained * 100)%")
end

println()
println("="^70)
println("Test complete!")
println("="^70)
