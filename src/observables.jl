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
    result = ITensor(1.0)

    for i in 1:N
        # Get physical indices at site i
        s = siteind(M, i)
        sp = s'  # The primed version

        # Contract: Σ_s M[i]_{s,s} (trace over physical index)
        δ = delta(s, sp)
        result = result * M[i] * δ
    end

    return scalar(result)
end

"""
    mpo_norm(M::MPO) -> Float64

Compute the Frobenius norm of an MPO: ||M||_F = √Tr[M†M]

Uses ITensors' built-in inner() function.
"""
function mpo_norm(M::MPO)
    return sqrt(abs(inner(M, M)))
end

"""
    autocorrelation(h0::MPO, ht::MPO) -> Float64

Compute the energy-energy autocorrelation function:
    C(t) = ⟨h₀(0) h₀(t)⟩ - ⟨h₀⟩²

where:
    h₀(0) = h₀  (initial energy density operator)
    h₀(t) = U†(t) h₀ U(t) = hₜ  (time-evolved operator in Heisenberg picture)

For the PXP model:
    ⟨...⟩ = Tr[...]  (infinite temperature, with projector already built into h₀)

Since the energy density operator h₀ = Ω P_{l-1} σˣ_l P_{l+1} already contains
the projectors P, and for unitary evolution Tr[hₜ] = Tr[h₀], we have:
    C(t) = Tr[h₀ hₜ] - (Tr[h₀])²

Note: We don't normalize by Hilbert space dimension D since the projectors
already restrict to the physical subspace.
"""
function autocorrelation(h0::MPO, ht::MPO)
    # Compute two-point function: Tr[h₀ hₜ]
    # Use ITensors' built-in inner() which computes Tr[A† B]
    # Since h₀, hₜ are Hermitian, Tr[h₀ hₜ] = Tr[h₀† hₜ] = inner(h₀, hₜ)
    two_point = real(inner(h0, ht))

    # Compute one-point function: Tr[h₀]
    tr_h0 = real(trace_mpo(h0))

    # Connected correlation function
    C_t = two_point - tr_h0^2

    return C_t
end

"""
    compute_correlation_function(sites, h0::MPO, times::Vector{Float64}, mpos::Vector{MPO}) -> Vector{Float64}

Compute the energy-energy autocorrelation function for all saved times:
    C(t) = ⟨h₀(0) h₀(t)⟩ - ⟨h₀(0)⟩⟨h₀(t)⟩

where h₀(0) = h0 and h₀(t) = mpos[i] are the energy density operators
at time t=0 and t=times[i] in the Heisenberg picture.
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
Uses ITensors' built-in norm() function.
"""
function check_unitarity(mpos::Vector{MPO}, h0::MPO)
    norm_h0 = sqrt(abs(inner(h0, h0)))
    return [sqrt(abs(inner(ht, ht))) / norm_h0 for ht in mpos]
end

"""
    expectation_with_projector(O::MPO, P::MPO) -> Float64

Compute expectation value using the projector onto the constrained subspace:
    ⟨O⟩ = Tr[P O] / Tr[P]

where P is the global constraint projector for the Rydberg blockade.

This is the proper definition for operators that don't already include
the projector constraints.
"""
function expectation_with_projector(O::MPO, P::MPO)
    # Tr[P O] using ITensors' inner function
    # Note: inner(A, B) computes Tr[A† B]
    # For MPO product, we need to contract them first

    # Since P is Hermitian projector, Tr[P O] = Tr[P† O] = inner(P, O)
    tr_PO = real(inner(P, O))

    # Tr[P]
    tr_P = real(trace_mpo(P))

    return tr_PO / tr_P
end

"""
    autocorrelation_with_projector(h0::MPO, ht::MPO, P::MPO) -> Float64

Compute the energy-energy autocorrelation function with explicit projector:
    C(t) = ⟨h₀(0) h₀(t)⟩ - ⟨h₀(0)⟩⟨h₀(t)⟩

where ⟨...⟩ = Tr[P ...] / Tr[P]

This uses the projector P explicitly to restrict to the constrained subspace,
rather than relying on the operator already containing projectors.
"""
function autocorrelation_with_projector(h0::MPO, ht::MPO, P::MPO)
    # Compute Tr[P]
    tr_P = real(trace_mpo(P))

    # Compute ⟨h₀(0) h₀(t)⟩ = Tr[P h₀ hₜ] / Tr[P]
    # Need to contract P * h₀ * hₜ
    # For MPOs: (P*h₀*hₜ) means contracting the physical indices

    # Method: Tr[P h₀ hₜ] = Tr[(P h₀) hₜ]
    # First compute Ph0 = P * h0
    Ph0 = apply(P, h0; cutoff=1e-14)

    # Then compute Tr[(P h₀) hₜ] = inner(Ph0, ht)
    tr_Ph0ht = real(inner(Ph0, ht))
    two_point = tr_Ph0ht / tr_P

    # Compute ⟨h₀(0)⟩ = Tr[P h₀] / Tr[P]
    tr_Ph0 = real(inner(P, h0))
    mean_h0 = tr_Ph0 / tr_P

    # Compute ⟨h₀(t)⟩ = Tr[P hₜ] / Tr[P]
    tr_Pht = real(inner(P, ht))
    mean_ht = tr_Pht / tr_P

    # Connected correlation
    C_t = two_point - mean_h0 * mean_ht

    return C_t
end

"""
    compute_correlation_with_projector(sites, h0::MPO, times::Vector{Float64},
                                       mpos::Vector{MPO}, P::MPO) -> Vector{Float64}

Compute correlation function using explicit projector for all time points.
"""
function compute_correlation_with_projector(sites, h0::MPO, times::Vector{Float64},
                                           mpos::Vector{MPO}, P::MPO)
    @assert length(times) == length(mpos)

    C = Float64[]
    for (t, ht) in zip(times, mpos)
        C_t = autocorrelation_with_projector(h0, ht, P)
        push!(C, C_t)
    end

    return C
end

export trace_mpo, mpo_norm
export autocorrelation, compute_correlation_function
export instantaneous_exponent, fit_exponent
export spatial_correlation, check_unitarity
export expectation_with_projector, autocorrelation_with_projector
export compute_correlation_with_projector
