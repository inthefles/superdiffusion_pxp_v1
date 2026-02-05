# hamiltonian.jl
# Hamiltonian construction for PXP model and its deformations
#
# Reference: Ljubotina et al., Phys. Rev. X 13, 011033 (2023)
#
# The PXP Hamiltonian describes Rydberg atoms with blockade constraint:
#   H_PXP = Ω Σ_i P_{i-1} σ^x_i P_{i+1}
# where P_i = (1 - σ^z_i)/2 = |↓⟩⟨↓| is the projector onto the ground state.

using ITensors
using LinearAlgebra

"""
    PXP_hamiltonian(sites; Ω=1.0) -> MPO

Construct the PXP Hamiltonian as an MPO:
    H = Ω Σ_i P_{i-1} σ^x_i P_{i+1}

where P_i = (1 - σ^z_i)/2 projects onto the ground state |↓⟩.

For boundary sites:
- Site 1: H includes P_0 = 1 (no constraint on left)
- Site N: H includes P_{N+1} = 1 (no constraint on right)

This uses open boundary conditions (OBC).
"""
function PXP_hamiltonian(sites; Ω=1.0)
    N = length(sites)
    os = OpSum()

    for i in 1:N
        # P_{i-1} σ^x_i P_{i+1}
        # P_j = (1 - σ^z_j)/2 = (1/2) - (1/2)σ^z_j
        # But for OpSum, we use the projector form directly

        if i == 1
            # Boundary: no P_{i-1}, only σ^x_1 P_2
            # P_2 σ^x_1 = (1/2)(1 - σ^z_2) σ^x_1
            #           = (1/2) σ^x_1 - (1/2) σ^x_1 σ^z_2
            os += Ω/2, "Sx", 1
            os += -Ω/2, "Sx", 1, "Sz", 2
        elseif i == N
            # Boundary: no P_{i+1}, only P_{N-1} σ^x_N
            os += Ω/2, "Sx", N
            os += -Ω/2, "Sz", N-1, "Sx", N
        else
            # Bulk: P_{i-1} σ^x_i P_{i+1}
            # = (1/4)(1 - σ^z_{i-1})(1 - σ^z_{i+1}) σ^x_i
            # = (1/4) σ^x_i - (1/4) σ^z_{i-1} σ^x_i
            #   - (1/4) σ^x_i σ^z_{i+1} + (1/4) σ^z_{i-1} σ^x_i σ^z_{i+1}
            os += Ω/4, "Sx", i
            os += -Ω/4, "Sz", i-1, "Sx", i
            os += -Ω/4, "Sx", i, "Sz", i+1
            os += Ω/4, "Sz", i-1, "Sx", i, "Sz", i+1
        end
    end

    return MPO(os, sites)
end

"""
    PXPZ_hamiltonian(sites; Ω=1.0, λ=0.0) -> MPO

Construct the PXPZ Hamiltonian (Eq. 6 in paper):
    H = Ω Σ_i P_{i-1} σ^x_i P_{i+1} + λ Σ_i P_{i-1} σ^z_i P_{i+1}

The λ term adds a local field within the constrained subspace.
At λ ≈ 0.655, transport becomes ballistic (z = 1).
"""
function PXPZ_hamiltonian(sites; Ω=1.0, λ=0.0)
    N = length(sites)
    os = OpSum()

    # PXP terms
    for i in 1:N
        if i == 1
            os += Ω/2, "Sx", 1
            os += -Ω/2, "Sx", 1, "Sz", 2
        elseif i == N
            os += Ω/2, "Sx", N
            os += -Ω/2, "Sz", N-1, "Sx", N
        else
            os += Ω/4, "Sx", i
            os += -Ω/4, "Sz", i-1, "Sx", i
            os += -Ω/4, "Sx", i, "Sz", i+1
            os += Ω/4, "Sz", i-1, "Sx", i, "Sz", i+1
        end
    end

    # PZP terms (λ deformation)
    if abs(λ) > 1e-14
        for i in 1:N
            if i == 1
                os += λ/2, "Sz", 1
                os += -λ/2, "Sz", 1, "Sz", 2
            elseif i == N
                os += λ/2, "Sz", N
                os += -λ/2, "Sz", N-1, "Sz", N
            else
                os += λ/4, "Sz", i
                os += -λ/4, "Sz", i-1, "Sz", i
                os += -λ/4, "Sz", i, "Sz", i+1
                os += λ/4, "Sz", i-1, "Sz", i, "Sz", i+1
            end
        end
    end

    return MPO(os, sites)
end

"""
    PNP_hamiltonian(sites; Ω=1.0, δ=0.0) -> MPO

Construct the PNP Hamiltonian (Eq. 7 in paper):
    H = Ω Σ_i P_{i-1} N_i P_{i+1} + δ Σ_i n_i

where N_i = σ^x_i (flip operator) and n_i = (1 + σ^z_i)/2 = |↑⟩⟨↑|
is the excitation number operator.

The δ term is a chemical potential for excitations.
At δ ≈ 0.63, transport becomes ballistic.
"""
function PNP_hamiltonian(sites; Ω=1.0, δ=0.0)
    N = length(sites)
    os = OpSum()

    # PNP terms (same as PXP since N_i = σ^x_i)
    for i in 1:N
        if i == 1
            os += Ω/2, "Sx", 1
            os += -Ω/2, "Sx", 1, "Sz", 2
        elseif i == N
            os += Ω/2, "Sx", N
            os += -Ω/2, "Sz", N-1, "Sx", N
        else
            os += Ω/4, "Sx", i
            os += -Ω/4, "Sz", i-1, "Sx", i
            os += -Ω/4, "Sx", i, "Sz", i+1
            os += Ω/4, "Sz", i-1, "Sx", i, "Sz", i+1
        end
    end

    # Chemical potential: δ Σ_i n_i = δ Σ_i (1 + σ^z_i)/2
    if abs(δ) > 1e-14
        for i in 1:N
            os += δ/2, "Id", i  # constant term per site
            os += δ/2, "Sz", i  # σ^z contribution
        end
    end

    return MPO(os, sites)
end

"""
    PNPNP_hamiltonian(sites; Ω=1.0, ξ=0.0) -> MPO

Construct the PNPNP Hamiltonian (Eq. 8 in paper):
    H = Ω Σ_i P_{i-1} N_i P_{i+1} + ξ Σ_i P_{i-2} N_{i-1} P_i N_{i+1} P_{i+2}

This is a 5-site term that introduces next-nearest-neighbor interactions.
At ξ = 1, transport becomes ballistic (exact mapping to free fermions).
"""
function PNPNP_hamiltonian(sites; Ω=1.0, ξ=0.0)
    N = length(sites)
    os = OpSum()

    # PNP terms
    for i in 1:N
        if i == 1
            os += Ω/2, "Sx", 1
            os += -Ω/2, "Sx", 1, "Sz", 2
        elseif i == N
            os += Ω/2, "Sx", N
            os += -Ω/2, "Sz", N-1, "Sx", N
        else
            os += Ω/4, "Sx", i
            os += -Ω/4, "Sz", i-1, "Sx", i
            os += -Ω/4, "Sx", i, "Sz", i+1
            os += Ω/4, "Sz", i-1, "Sx", i, "Sz", i+1
        end
    end

    # PNPNP terms: 5-site interactions
    # P_{i-2} N_{i-1} P_i N_{i+1} P_{i+2}
    # where P = (1-σ^z)/2 and N = σ^x
    if abs(ξ) > 1e-14
        for i in 2:(N-1)
            # Central site is i, with σ^x at i-1 and i+1
            # Projectors at i-2, i, i+2
            # Need to handle boundaries carefully

            if i == 2
                # No P_{i-2}, have P_i P_{i+2}
                # = (1/4)(1-σ^z_2)(1-σ^z_4) σ^x_1 σ^x_3
                if N >= 4
                    _add_5site_term!(os, ξ, i-1, i+1, nothing, i, i+2, N)
                end
            elseif i == N-1
                # No P_{i+2}, have P_{i-2} P_i
                if N >= 4
                    _add_5site_term!(os, ξ, i-1, i+1, i-2, i, nothing, N)
                end
            else
                # Full 5-site term
                if i >= 3 && i <= N-2
                    _add_5site_term!(os, ξ, i-1, i+1, i-2, i, i+2, N)
                end
            end
        end
    end

    return MPO(os, sites)
end

"""
Helper function to add 5-site PNPNP term to OpSum.
Sites for σ^x: sx1, sx2
Sites for projectors: p1, p2, p3 (can be nothing for boundary)
"""
function _add_5site_term!(os, ξ, sx1, sx2, p1, p2, p3, N)
    # Count number of projectors
    projs = filter(!isnothing, [p1, p2, p3])
    n_proj = length(projs)

    # Prefactor: (1/2)^n_proj from each projector P = (1-σ^z)/2
    prefactor = ξ / (2^n_proj)

    # Generate all 2^n_proj combinations of (1, -σ^z) for projectors
    for mask in 0:(2^n_proj - 1)
        sign = 1
        ops = Tuple{Any,Int}[]

        for (j, p) in enumerate(projs)
            if (mask >> (j-1)) & 1 == 1
                # -σ^z term
                sign *= -1
                push!(ops, ("Sz", p))
            end
            # else: identity (1), no operator needed
        end

        # Add the σ^x operators
        push!(ops, ("Sx", sx1))
        push!(ops, ("Sx", sx2))

        # Sort by site index for ITensors
        sort!(ops, by=x->x[2])

        # Build the term
        coeff = sign * prefactor
        if length(ops) == 2
            os += coeff, ops[1][1], ops[1][2], ops[2][1], ops[2][2]
        elseif length(ops) == 3
            os += coeff, ops[1][1], ops[1][2], ops[2][1], ops[2][2], ops[3][1], ops[3][2]
        elseif length(ops) == 4
            os += coeff, ops[1][1], ops[1][2], ops[2][1], ops[2][2],
                        ops[3][1], ops[3][2], ops[4][1], ops[4][2]
        elseif length(ops) == 5
            os += coeff, ops[1][1], ops[1][2], ops[2][1], ops[2][2],
                        ops[3][1], ops[3][2], ops[4][1], ops[4][2],
                        ops[5][1], ops[5][2]
        end
    end
end

"""
    check_hermiticity(H::MPO; tol=1e-10) -> Bool

Verify that an MPO is Hermitian by checking H = H†.
"""
function check_hermiticity(H::MPO; tol=1e-10)
    H_dag = dag(swapprime(H, 0, 1))
    diff = norm(H - H_dag)
    return diff < tol
end

export PXP_hamiltonian, PXPZ_hamiltonian, PNP_hamiltonian, PNPNP_hamiltonian
export check_hermiticity
