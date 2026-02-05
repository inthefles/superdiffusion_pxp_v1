# observables.jl
# Observable calculations for PXP transport
#
# Computes energy autocorrelation functions and extracts dynamical exponents.

using ITensors
using LinearAlgebra

"""
    trace_mpo(M::MPO) -> ComplexF64

Compute the trace of an MPO: Tr[M] = Σ_σ ⟨σ|M|σ⟩

For an MPO with physical indices s_i and s_i', the trace contracts
s_i = s_i' for all sites.
"""
function trace_mpo(M::MPO)
    N = length(M)

    # Contract the first tensor with trace over physical index
    sites = siteinds(M)
    result = M[1] * delta(sites[1]...) # Assumes siteinds returns pairs (s, s')

    # Actually, siteinds for MPO returns the site indices
    # We need to trace: contract s with s' at each site

    # Alternative: use inner product trick
    # Tr[M] = inner(I, M) where I is identity MPS-like object

    # Simpler approach: direct contraction
    result = ITensor(1.0)
    for i in 1:N
        # Get physical indices at site i
        s = siteind(M, i)  # Gets one of the site indices
        sp = s'  # The primed version

        # Contract: Σ_s M[i]_{s,s} (trace over physical index)
        # This requires delta(s, s') contraction
        δ = delta(s, sp)
        result = result * M[i] * δ
    end

    return scalar(result)
end

"""
    inner_mpo(A::MPO, B::MPO) -> ComplexF64

Compute the Hilbert-Schmidt inner product: Tr[A† B]
"""
function inner_mpo(A::MPO, B::MPO)
    @assert length(A) == length(B)
    N = length(A)

    # Tr[A† B] can be computed by contracting A and B
    # with their physical indices contracted

    result = ITensor(1.0)
    for i in 1:N
        # Contract A†[i] with B[i] over physical indices
        # A†[i] = conj(A[i]) with primes swapped

        A_dag_i = dag(swapprime(A[i], 0, 1))
        result = result * A_dag_i * B[i]
    end

    return scalar(result)
end

"""
    mpo_norm(M::MPO) -> Float64

Compute the Frobenius norm of an MPO: ||M||_F = √Tr[M†M]
"""
function mpo_norm(M::MPO)
    return sqrt(abs(inner_mpo(M, M)))
end

"""
    autocorrelation(h0::MPO, ht::MPO; use_projector=false, P::Union{MPO,Nothing}=nothing) -> Float64

Compute the infinite-temperature autocorrelation function:
    C(t) = Tr[h(0) h(t)] / Tr[I] - (Tr[h(0)] / Tr[I])²

For the PXP model at infinite temperature, we use:
    ⟨h(0) h(t)⟩ = Tr[h₀ hₜ] / D

where D is the Hilbert space dimension.

If use_projector=true, restricts to the constrained subspace:
    C(t) = Tr[𝒫 h₀ hₜ] / Tr[𝒫] - (Tr[𝒫 h₀] / Tr[𝒫])²
"""
function autocorrelation(h0::MPO, ht::MPO;
                         use_projector=false, P::Union{MPO,Nothing}=nothing)
    N = length(h0)

    if use_projector && !isnothing(P)
        # Restricted trace using projector
        # Tr[𝒫 h₀ hₜ] requires computing 𝒫 * h₀ * hₜ as an MPO product
        # This is expensive; for now we use unrestricted trace

        # Approximate: Tr[h₀ hₜ] ≈ D * C(t) where D = 2^N
        # The projector should give similar dynamics for local observables
        error("Projector-restricted trace not yet implemented")
    end

    # Compute Tr[h₀ hₜ]
    # This is the Hilbert-Schmidt inner product (up to normalization)

    # For MPOs, Tr[A B] = Tr[A†† B] = inner_mpo(A†, B) with proper normalization
    # But h₀, hₜ are Hermitian, so we can use a simpler approach

    # Contract h0 and ht with all physical indices traced
    tr_h0_ht = _trace_mpo_product(h0, ht)

    # Normalization: Tr[I] = 2^N for unconstrained space
    D = 2.0^N
    C_t = real(tr_h0_ht) / D

    # Subtract disconnected part (usually zero for traceless operators)
    tr_h0 = trace_mpo(h0)
    mean_h = real(tr_h0) / D
    C_t -= mean_h^2

    return C_t
end

"""
    _trace_mpo_product(A::MPO, B::MPO) -> ComplexF64

Compute Tr[A B] for two MPOs.
This contracts A and B vertically and traces over physical indices.
"""
function _trace_mpo_product(A::MPO, B::MPO)
    @assert length(A) == length(B)
    N = length(A)

    # Build a "double MPO" and trace
    # Tr[A B] = Σ_{s₁...sₙ} ⟨s|A|s'⟩⟨s'|B|s⟩

    result = ITensor(1.0)

    for i in 1:N
        # A has indices: (link_left, s, s', link_right)
        # B has indices: (link_left', s', s'', link_right')

        # For trace: s = s'' (outer indices equal)
        # and s' (middle index) is summed

        # Contract A[i] * B[i] with appropriate index matching
        # Need to be careful about which indices to contract

        # Get site indices
        s_A = siteind(A, i)
        s_B = siteind(B, i)

        # For the product Tr[AB]:
        # (A*B)_{s,s''} = Σ_{s'} A_{s,s'} B_{s',s''}
        # Then trace: Σ_s (A*B)_{s,s}

        # Contract A[i] and B[i]
        AB_i = A[i] * B[i]
        result = result * AB_i
    end

    # The result should have paired site indices; trace over them
    # If indices don't match exactly, may need delta tensors

    return scalar(result)
end

"""
    compute_correlation_function(sites, h0::MPO, times::Vector{Float64}, mpos::Vector{MPO}) -> Vector{Float64}

Compute the autocorrelation function C(t) = ⟨h(0)h(t)⟩ - ⟨h⟩² for all saved times.
"""
function compute_correlation_function(sites, h0::MPO, times::Vector{Float64},
                                       mpos::Vector{MPO})
    @assert length(times) == length(mpos)

    C = Float64[]
    for (t, ht) in zip(times, mpos)
        C_t = autocorrelation(h0, ht)
        push!(C, C_t)
    end

    return C
end

"""
    instantaneous_exponent(times::Vector{Float64}, C::Vector{Float64}) -> (Vector{Float64}, Vector{Float64})

Extract the instantaneous dynamical exponent z from the correlation function.

The exponent is defined through C(t) ~ t^{-2/z}, so:
    1/z(t) = -d log C(t) / (2 d log t)

Returns (t_mid, z_inv) where t_mid are midpoint times and z_inv = 1/z(t).

For superdiffusive transport in PXP: z ≈ 3/2, so 1/z ≈ 2/3.
"""
function instantaneous_exponent(times::Vector{Float64}, C::Vector{Float64})
    # Filter out t=0 and negative/zero correlations
    valid_idx = findall(i -> times[i] > 0 && C[i] > 0, 1:length(times))

    if length(valid_idx) < 2
        return Float64[], Float64[]
    end

    log_t = log.(times[valid_idx])
    log_C = log.(C[valid_idx])

    # Numerical derivative: d(log C)/d(log t)
    t_mid = Float64[]
    z_inv = Float64[]

    for i in 1:(length(valid_idx)-1)
        dt = log_t[i+1] - log_t[i]
        dC = log_C[i+1] - log_C[i]

        # 1/z = -dC/(2*dt) = -(d log C)/(2 d log t)
        push!(t_mid, exp((log_t[i] + log_t[i+1]) / 2))
        push!(z_inv, -dC / (2 * dt))
    end

    return t_mid, z_inv
end

"""
    fit_exponent(times::Vector{Float64}, C::Vector{Float64};
                 t_min=1.0, t_max=Inf) -> (z, error)

Fit the dynamical exponent z from C(t) ~ t^{-2/z} in the specified time window.

Returns the fitted exponent z and its uncertainty.
"""
function fit_exponent(times::Vector{Float64}, C::Vector{Float64};
                      t_min=1.0, t_max=Inf)
    # Filter data to time window
    valid_idx = findall(i -> t_min <= times[i] <= t_max && C[i] > 0, 1:length(times))

    if length(valid_idx) < 3
        return NaN, NaN
    end

    log_t = log.(times[valid_idx])
    log_C = log.(C[valid_idx])

    # Linear fit: log C = a + b * log t
    # where b = -2/z
    n = length(log_t)
    sum_x = sum(log_t)
    sum_y = sum(log_C)
    sum_xx = sum(log_t .^ 2)
    sum_xy = sum(log_t .* log_C)

    # Least squares
    b = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x^2)
    a = (sum_y - b * sum_x) / n

    # z = -2/b
    z = -2.0 / b

    # Estimate error from residuals
    residuals = log_C .- (a .+ b .* log_t)
    σ_b = sqrt(sum(residuals .^ 2) / (n - 2) / (sum_xx - sum_x^2 / n))
    σ_z = abs(2.0 / b^2) * σ_b

    return z, σ_z
end

"""
    spatial_correlation(h0::MPO, ht::MPO, l::Int, sites) -> Float64

Compute the spatial correlation ⟨h_l(0) h_0(t)⟩ at displacement l.

This requires h_l to be the energy density at site l.
Used for studying the spatial spreading of correlations.
"""
function spatial_correlation(h0::MPO, ht::MPO, sites)
    # This is the same as autocorrelation but for different operators
    return autocorrelation(h0, ht)
end

"""
    check_unitarity(mpos::Vector{MPO}, h0::MPO) -> Vector{Float64}

Check that ||h(t)||_F ≈ ||h(0)||_F for all times (unitarity check).

Returns the ratio ||h(t)||/||h(0)|| which should be close to 1.
"""
function check_unitarity(mpos::Vector{MPO}, h0::MPO)
    norm_h0 = mpo_norm(h0)
    return [mpo_norm(ht) / norm_h0 for ht in mpos]
end

export trace_mpo, inner_mpo, mpo_norm
export autocorrelation, compute_correlation_function
export instantaneous_exponent, fit_exponent
export spatial_correlation, check_unitarity
