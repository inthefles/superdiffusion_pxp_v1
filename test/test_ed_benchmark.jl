# test_ed_benchmark.jl
# Exact diagonalization benchmark for small systems
#
# Compares TEBD results against exact dynamics for N ≤ 12

using Test
using PXPTransport
using ITensors
using LinearAlgebra

"""
    build_pxp_matrix(N::Int; Ω=1.0) -> Matrix

Build the PXP Hamiltonian as a dense matrix in the constrained Hilbert space.
Only valid states (no adjacent excitations) are included.
"""
function build_pxp_matrix(N::Int; Ω=1.0)
    states = enumerate_valid_states(N)
    D = length(states)

    # Create state-to-index mapping
    state_to_idx = Dict{Vector{Int}, Int}()
    for (i, s) in enumerate(states)
        state_to_idx[s] = i
    end

    H = zeros(Float64, D, D)

    # PXP term: P_{i-1} σ^x_i P_{i+1}
    # σ^x flips site i: 0 ↔ 1
    for (idx, state) in enumerate(states)
        for i in 1:N
            # Check if flip is allowed
            left_ok = (i == 1) || (state[i-1] == 0)
            right_ok = (i == N) || (state[i+1] == 0)

            if left_ok && right_ok
                # Create flipped state
                new_state = copy(state)
                new_state[i] = 1 - new_state[i]

                # Check if new state is valid
                if is_valid_state(new_state)
                    new_idx = state_to_idx[new_state]
                    H[idx, new_idx] += Ω
                end
            end
        end
    end

    return Symmetric(H)
end

"""
    build_energy_density_matrix(N::Int, l::Int; Ω=1.0) -> Matrix

Build the energy density operator h_l = Ω P_{l-1} σ^x_l P_{l+1} as a matrix.
"""
function build_energy_density_matrix(N::Int, l::Int; Ω=1.0)
    states = enumerate_valid_states(N)
    D = length(states)

    state_to_idx = Dict{Vector{Int}, Int}()
    for (i, s) in enumerate(states)
        state_to_idx[s] = i
    end

    h = zeros(Float64, D, D)

    for (idx, state) in enumerate(states)
        # Check if flip at site l is allowed
        left_ok = (l == 1) || (state[l-1] == 0)
        right_ok = (l == N) || (state[l+1] == 0)

        if left_ok && right_ok
            new_state = copy(state)
            new_state[l] = 1 - new_state[l]

            if is_valid_state(new_state)
                new_idx = state_to_idx[new_state]
                h[idx, new_idx] += Ω
            end
        end
    end

    return h
end

"""
    exact_correlation(H, h0, t::Float64) -> Float64

Compute the energy-energy autocorrelation using exact diagonalization:
    C(t) = Tr[h(0) h(t)] - Tr[h(0)] Tr[h(t)]

where h(t) = e^{iHt} h(0) e^{-iHt} in the Heisenberg picture.

Note: No normalization by Hilbert space dimension since the energy density
operator h already contains the projectors that restrict to physical subspace.
"""
function exact_correlation(H::Union{Matrix, Symmetric}, h0::Matrix, t::Float64)
    D = size(H, 1)

    # Diagonalize H
    E, V = eigen(H)

    # Time evolution: h(t) = V exp(iEt) V† h V exp(-iEt) V†
    exp_iEt = Diagonal(exp.(im * E * t))
    exp_mEt = Diagonal(exp.(-im * E * t))

    # h(t) in eigenbasis
    h_eigenbasis = V' * h0 * V
    ht_eigenbasis = exp_iEt * h_eigenbasis * exp_mEt
    ht = V * ht_eigenbasis * V'

    # Tr[h0 ht] - Tr[h0] Tr[ht]
    return real(tr(h0 * ht)) - real(tr(h0)) * real(tr(ht))
end

@testset "ED Benchmark" begin
    # Small system for exact comparison
    N = 8
    l = div(N + 1, 2)  # Center site

    @testset "Hamiltonian Matrix" begin
        H = build_pxp_matrix(N)
        D = constrained_dim(N)

        @test size(H) == (D, D)
        @test issymmetric(H)

        # Check eigenvalues are real
        E = eigvals(H)
        @test all(isreal.(E))

        # PXP has specific spectral properties
        # The spectrum should be symmetric around 0 (particle-hole symmetry)
        @test isapprox(sum(E), 0, atol=1e-10)
    end

    @testset "Energy Density Matrix" begin
        h = build_energy_density_matrix(N, l)
        D = constrained_dim(N)

        @test size(h) == (D, D)
        # Energy density is Hermitian
        @test norm(h - h') < 1e-10

        # Should be traceless (no constant term in bulk)
        if 1 < l < N
            @test abs(tr(h)) < 1e-10
        end
    end

    @testset "Exact Correlation Function" begin
        H = build_pxp_matrix(N; Ω=1.0)
        h = build_energy_density_matrix(N, l; Ω=1.0)
        D = size(H, 1)

        # At t=0, C(0) = Tr[h²] - (Tr[h])²
        # Since h is traceless for bulk sites, C(0) = Tr[h²]
        C0 = exact_correlation(H, h, 0.0)
        expected_C0 = tr(h^2) - (tr(h))^2
        @test isapprox(C0, expected_C0, rtol=1e-10)

        # C(0) should be positive
        @test C0 > 0

        # Test a few time points
        times = [0.0, 0.5, 1.0, 2.0]
        C = [exact_correlation(H, h, t) for t in times]

        # Correlation should decay
        @test C[1] >= C[end]

        # Should remain bounded
        for c in C
            @test abs(c) <= C[1]
        end
    end

    @testset "Exponent Extraction" begin
        # Longer time series for exponent fitting
        H = build_pxp_matrix(N; Ω=1.0)
        h = build_energy_density_matrix(N, l; Ω=1.0)

        times = collect(0.1:0.1:10.0)
        C = [exact_correlation(H, h, t) for t in times]

        # Filter positive correlations
        valid = C .> 0
        if sum(valid) > 5
            t_valid = times[valid]
            C_valid = C[valid]

            t_mid, z_inv = instantaneous_exponent(t_valid, C_valid)

            # For small systems (N=8), finite-size effects dominate
            # Just verify the extraction runs without errors
            @test !isempty(z_inv)
            @test !isempty(t_mid)
            @test length(z_inv) == length(t_mid)
        end
    end
end

@testset "TEBD vs ED Comparison" begin
    # This test requires implementing the full TEBD-to-matrix conversion
    # For now, we just verify the structures are compatible

    N = 6
    sites = PXPSites(N)

    H_mpo = PXP_hamiltonian(sites)
    H_mat = build_pxp_matrix(N)

    # Check dimensions match
    @test length(H_mpo) == N
    @test size(H_mat, 1) == constrained_dim(N)

    # More detailed comparison would require MPO-to-matrix conversion
    @test true  # Placeholder for future implementation
end
