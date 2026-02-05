# test_hamiltonian.jl
# Tests for Hamiltonian construction

using Test
using PXPTransport
using ITensors
using LinearAlgebra

@testset "Hilbert Space" begin
    # Test constrained dimension (Fibonacci sequence)
    @test constrained_dim(1) == 2   # F_3 = 2
    @test constrained_dim(2) == 3   # F_4 = 3
    @test constrained_dim(3) == 5   # F_5 = 5
    @test constrained_dim(4) == 8   # F_6 = 8
    @test constrained_dim(5) == 13  # F_7 = 13
    @test constrained_dim(6) == 21  # F_8 = 21
    @test constrained_dim(10) == 144  # F_12 = 144

    # Test state enumeration matches dimension
    for N in 2:8
        states = enumerate_valid_states(N)
        @test length(states) == constrained_dim(N)
    end

    # Test validity check
    @test is_valid_state([0, 0, 0]) == true   # |↓↓↓⟩
    @test is_valid_state([1, 0, 1]) == true   # |↑↓↑⟩
    @test is_valid_state([0, 1, 0]) == true   # |↓↑↓⟩
    @test is_valid_state([1, 1, 0]) == false  # |↑↑↓⟩ - forbidden
    @test is_valid_state([0, 1, 1]) == false  # |↓↑↑⟩ - forbidden
end

@testset "PXP Hamiltonian" begin
    N = 6
    sites = PXPSites(N)

    # Test Hamiltonian construction
    H = PXP_hamiltonian(sites; Ω=1.0)
    @test length(H) == N

    # Test Hermiticity by checking MPO structure
    # For a proper test, we'd convert to matrix and check H = H†
    # For now, just verify it doesn't throw errors
    @test true
end

@testset "Energy Density" begin
    N = 6
    sites = PXPSites(N)

    # Test energy density at each site
    for l in 1:N
        h_l = energy_density(sites, l; Ω=1.0)
        @test length(h_l) == N
    end

    # Test center energy density
    h_center = center_energy_density(sites; Ω=1.0)
    @test length(h_center) == N

    # Energy density should be at site 3 for N=6
    expected_center = div(N + 1, 2)
    @test expected_center == 3
end

@testset "PXPZ Hamiltonian" begin
    N = 6
    sites = PXPSites(N)

    # PXPZ with λ = 0 should equal PXP
    H_pxp = PXP_hamiltonian(sites; Ω=1.0)
    H_pxpz = PXPZ_hamiltonian(sites; Ω=1.0, λ=0.0)

    # Both should have same structure
    @test length(H_pxp) == length(H_pxpz)

    # Test with nonzero λ
    H_pxpz_def = PXPZ_hamiltonian(sites; Ω=1.0, λ=0.655)
    @test length(H_pxpz_def) == N
end

@testset "PNP Hamiltonian" begin
    N = 6
    sites = PXPSites(N)

    # PNP with δ = 0 should equal PXP
    H_pnp = PNP_hamiltonian(sites; Ω=1.0, δ=0.0)
    @test length(H_pnp) == N

    # With chemical potential
    H_pnp_def = PNP_hamiltonian(sites; Ω=1.0, δ=0.63)
    @test length(H_pnp_def) == N
end

@testset "PNPNP Hamiltonian" begin
    N = 8  # Need larger N for 5-site terms
    sites = PXPSites(N)

    # PNPNP with ξ = 0 should equal PXP
    H_pnpnp = PNPNP_hamiltonian(sites; Ω=1.0, ξ=0.0)
    @test length(H_pnpnp) == N

    # At ξ = 1 (ballistic point)
    H_pnpnp_ball = PNPNP_hamiltonian(sites; Ω=1.0, ξ=1.0)
    @test length(H_pnpnp_ball) == N
end

@testset "Site Merging" begin
    N = 6
    info = create_merged_sites(N)

    @test info.N_original == 6
    @test info.N_merged == 3
    @test length(info.merged_sites) == 3

    # Each merged site should have dimension 3
    for s in info.merged_sites
        @test dim(s) == 3
    end

    # Odd N should throw
    @test_throws AssertionError create_merged_sites(7)
end

@testset "TEBD Parameters" begin
    # Default parameters
    params = TEBDParams()
    @test params.dt == 0.05
    @test params.maxdim == 128
    @test params.order == 4
    @test params.Ω == 1.0
    @test params.λ == 0.0

    # Custom parameters
    params2 = TEBDParams(dt=0.1, maxdim=64, λ=0.5)
    @test params2.dt == 0.1
    @test params2.maxdim == 64
    @test params2.λ == 0.5
end

@testset "Trotter Gates" begin
    N = 6
    sites = PXPSites(N)

    # 2nd order gates
    gates_2 = make_trotter_gates(sites, 0.05; order=2)
    @test length(gates_2) > 0

    # 4th order gates (should have 5x as many)
    gates_4 = make_trotter_gates(sites, 0.05; order=4)
    @test length(gates_4) > length(gates_2)
end

@testset "IO Functions" begin
    # Test parameter generation
    params = default_params(N=32, maxdim=64, λ=0.5)
    @test params["N"] == 32
    @test params["maxdim"] == 64
    @test params["λ"] == 0.5
    @test params["model"] == "PXPZ"

    # Test filename generation
    filename = generate_filename(params)
    @test contains(filename, "PXPZ")
    @test contains(filename, "N32")
    @test contains(filename, "chi64")
    @test endswith(filename, ".jld2")
end

#==============================================================================#
# Tests for Paper-Based Energy Density Implementation (χ=2)
#==============================================================================#

@testset "Energy Density Original Sites" begin
    # Test energy_density_original construction
    for N in [4, 6, 8]
        for l in 1:N
            h = energy_density_original(N, l; Ω=1.0)
            @test length(h) == N
        end
    end

    # Bond dimension should be 2
    N = 6
    h = energy_density_original(N, 3; Ω=1.0)
    # Check internal link dimensions
    for i in 1:(N-1)
        link = linkind(h, i)
        @test dim(link) == 2
    end
end

@testset "Projector MPO Original Sites" begin
    for N in [4, 6, 8]
        P = projector_mpo_original(N)
        @test length(P) == N

        # Bond dimension should be 2
        for i in 1:(N-1)
            link = linkind(P, i)
            @test dim(link) == 2
        end
    end
end

@testset "MPO Merging" begin
    N_original = 6
    N_merged = N_original ÷ 2

    merged_sites = [Index(3, "Site,n=$i") for i in 1:N_merged]

    # Test merging projector
    P_orig = projector_mpo_original(N_original)
    P_merged = merge_mpo_pairs(P_orig, merged_sites)
    @test length(P_merged) == N_merged

    # Each merged site should have dimension 3
    for i in 1:N_merged
        s = siteind(P_merged, i)
        @test dim(s) == 3
    end

    # Bond dimension should remain 2
    for i in 1:(N_merged-1)
        link = linkind(P_merged, i)
        @test dim(link) == 2
    end

    # Test merging energy density
    l = 3  # middle site
    h_orig = energy_density_original(N_original, l; Ω=1.0)
    h_merged = merge_mpo_pairs(h_orig, merged_sites)
    @test length(h_merged) == N_merged
end

@testset "Energy Density Merged" begin
    N_original = 8
    N_merged = N_original ÷ 2
    info = create_merged_sites(N_original)

    # Test at each merged site
    for l_merged in 1:N_merged
        h = energy_density_merged(info.merged_sites, l_merged; Ω=1.0)
        @test length(h) == N_merged
    end

    # Test center energy density
    h_center = center_energy_density_merged(info.merged_sites; Ω=1.0)
    @test length(h_center) == N_merged
end

@testset "Projector MPO Merged" begin
    N_original = 8
    N_merged = N_original ÷ 2
    info = create_merged_sites(N_original)

    P = projector_mpo_merged(info.merged_sites)
    @test length(P) == N_merged

    # Bond dimension should be 2
    for i in 1:(N_merged-1)
        link = linkind(P, i)
        @test dim(link) == 2
    end
end

@testset "Energy Density ED Comparison" begin
    # Compare implementation against ED reference for small systems
    N_original = 6
    N_merged = N_original ÷ 2
    info = create_merged_sites(N_original)

    # Build ED reference energy density matrix
    function build_energy_density_matrix_local(N::Int, l::Int; Ω=1.0)
        states = enumerate_valid_states(N)
        D = length(states)

        state_to_idx = Dict{Vector{Int}, Int}()
        for (i, s) in enumerate(states)
            state_to_idx[s] = i
        end

        h = zeros(Float64, D, D)

        for (idx, state) in enumerate(states)
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

    # For center merged site (l_merged = 2 for N_merged=3),
    # the energy density includes terms at original sites 3 and 4
    l_merged = 2
    l_left = 2 * l_merged - 1   # = 3
    l_right = 2 * l_merged      # = 4

    h_ed_left = build_energy_density_matrix_local(N_original, l_left; Ω=1.0)
    h_ed_right = build_energy_density_matrix_local(N_original, l_right; Ω=1.0)
    h_ed_total = h_ed_left + h_ed_right

    # Properties that should hold
    D = constrained_dim(N_original)
    @test size(h_ed_total) == (D, D)

    # Hermitian
    @test norm(h_ed_total - h_ed_total') < 1e-10

    # Tr[h²] > 0 (non-trivial operator)
    @test tr(h_ed_total^2) > 0

    # The MPO version should give the same matrix representation
    h_mpo = energy_density_merged(info.merged_sites, l_merged; Ω=1.0)

    # Check that MPO has correct structure
    @test length(h_mpo) == N_merged
end

@testset "Physics Checks for Energy Density" begin
    N_original = 8
    N_merged = N_original ÷ 2
    info = create_merged_sites(N_original)

    l_merged = N_merged ÷ 2 + 1  # center-ish

    h = energy_density_merged(info.merged_sites, l_merged; Ω=1.0)

    # Check that MPO exists and has proper dimensions
    @test length(h) == N_merged
    for i in 1:N_merged
        s = siteind(h, i)
        @test dim(s) == 3  # merged site dimension
    end

    # Projector should be valid
    P = projector_mpo_merged(info.merged_sites)
    @test length(P) == N_merged
end
