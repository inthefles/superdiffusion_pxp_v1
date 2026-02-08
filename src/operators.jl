# operators.jl
# Local operators for PXP model
#
# Defines energy density operators and projectors for computing
# autocorrelation functions.

using ITensors
using ITensorMPS

"""
    energy_density(sites, l::Int; Ω=1.0) -> MPO

Construct the local energy density operator h_l as an MPO:
    h_l = Ω P_{l-1} σ^x_l P_{l+1}

This is the local term in the PXP Hamiltonian at site l.
The energy density is used for computing the autocorrelation function
C(t) = ⟨h_l(0) h_l(t)⟩ - ⟨h_l⟩².

For boundary sites, the missing projectors are replaced by identity.
"""
function energy_density(sites, l::Int; Ω=1.0)
    N = length(sites)
    @assert 1 <= l <= N "Site index l must be between 1 and N"

    os = OpSum()

    if l == 1
        # h_1 = Ω σ^x_1 P_2 = Ω (1/2)(1 - σ^z_2) σ^x_1
        os += Ω/2, "Sx", 1
        os += -Ω/2, "Sx", 1, "Sz", 2
    elseif l == N
        # h_N = Ω P_{N-1} σ^x_N
        os += Ω/2, "Sx", N
        os += -Ω/2, "Sz", N-1, "Sx", N
    else
        # h_l = Ω P_{l-1} σ^x_l P_{l+1}
        os += Ω/4, "Sx", l
        os += -Ω/4, "Sz", l-1, "Sx", l
        os += -Ω/4, "Sx", l, "Sz", l+1
        os += Ω/4, "Sz", l-1, "Sx", l, "Sz", l+1
    end

    return MPO(os, sites)
end

"""
    center_energy_density(sites; Ω=1.0) -> MPO

Construct the energy density at the center of the chain.
For even N, uses site N/2. For odd N, uses site (N+1)/2.
"""
function center_energy_density(sites; Ω=1.0)
    N = length(sites)
    l = div(N + 1, 2)  # Center site
    return energy_density(sites, l; Ω=Ω)
end

"""
    projector_mpo(sites) -> MPO

Construct the global constraint projector 𝒫 = Π_i (1 - n_i n_{i+1}) as an MPO.

This projector removes states with adjacent excitations (|↑↑⟩).
It has bond dimension 2 since it's a product of local terms.

The projector is useful for:
1. Verifying states remain in the constrained subspace
2. Computing traces restricted to the physical Hilbert space
"""
function projector_mpo(sites)
    N = length(sites)
    os = OpSum()

    # Start with identity
    for i in 1:N
        os += 1.0, "Id", i
    end

    # Subtract n_i n_{i+1} for each bond
    # n_i = (1 + σ^z_i)/2
    # n_i n_{i+1} = (1/4)(1 + σ^z_i)(1 + σ^z_{i+1})
    #             = (1/4)(1 + σ^z_i + σ^z_{i+1} + σ^z_i σ^z_{i+1})
    for i in 1:(N-1)
        os += -1/4, "Id", i
        os += -1/4, "Sz", i
        os += -1/4, "Sz", i+1
        os += -1/4, "Sz", i, "Sz", i+1
    end

    return MPO(os, sites)
end

"""
    identity_mpo(sites) -> MPO

Construct the identity operator as an MPO.
"""
function identity_mpo(sites)
    return MPO(sites, "Id")
end

"""
    sz_operator(sites, l::Int) -> MPO

Construct σ^z_l operator as an MPO.
"""
function sz_operator(sites, l::Int)
    os = OpSum()
    os += 1.0, "Sz", l
    return MPO(os, sites)
end

"""
    sx_operator(sites, l::Int) -> MPO

Construct σ^x_l operator as an MPO.
"""
function sx_operator(sites, l::Int)
    os = OpSum()
    os += 1.0, "Sx", l
    return MPO(os, sites)
end

"""
    number_operator(sites, l::Int) -> MPO

Construct the number operator n_l = (1 + σ^z_l)/2 = |↑⟩⟨↑|_l as an MPO.
"""
function number_operator(sites, l::Int)
    os = OpSum()
    os += 0.5, "Id", l
    os += 0.5, "Sz", l
    return MPO(os, sites)
end

"""
    total_number_operator(sites) -> MPO

Construct the total excitation number N = Σ_l n_l as an MPO.
"""
function total_number_operator(sites)
    N = length(sites)
    os = OpSum()
    for l in 1:N
        os += 0.5, "Id", l
        os += 0.5, "Sz", l
    end
    return MPO(os, sites)
end

#==============================================================================#
# Elementary Tensors on Original Spin-1/2 Sites (Paper Eqs A2-A8)
#==============================================================================#

"""
State encoding for original spin-1/2 sites:
|↓⟩ → 1
|↑⟩ → 2

Virtual bond encoding for projector (χ=2):
1 = safe (previous site was ↓)
2 = danger (previous site was ↑)

Down projector 𝒟 and flip operator 𝒳 are χ=1 (local operators).
"""
const SPIN_DOWN = 1
const SPIN_UP = 2

#------------------------------------------------------------------------------
# Projector Tensors (χ=2)
# Paper Eq A2: L = (I, |↓⟩⟨↓|), M defined by Eq A4, R = ((1,1)ᵀ for ↓, (1,0)ᵀ for ↑)
#------------------------------------------------------------------------------

"""
    _projector_tensor_left() -> Array{ComplexF64, 3}

Left boundary tensor for projector 𝒫. Shape: W[s', s, β] with β ∈ {1,2}.
This is the "L" in L * M * M * ... * R.

Paper Eq A2: L = (|↑⟩⟨↑| + |↓⟩⟨↓|, |↓⟩⟨↓|) = (I, |↓⟩⟨↓|)
- Column β=1: Identity (passes both ↓ and ↑)
- Column β=2: |↓⟩⟨↓| (passes only ↓)
"""
function _projector_tensor_left()
    d, χ = 2, 2
    W = zeros(ComplexF64, d, d, χ)
    # β=1: Identity
    W[SPIN_DOWN, SPIN_DOWN, 1] = 1.0
    W[SPIN_UP, SPIN_UP, 1] = 1.0
    # β=2: |↓⟩⟨↓|
    W[SPIN_DOWN, SPIN_DOWN, 2] = 1.0
    # W[SPIN_UP, SPIN_UP, 2] = 0.0 (already zero)
    return W
end

"""
    _projector_tensor_bulk() -> Array{ComplexF64, 4}

Bulk tensor for projector 𝒫. Shape: W[s', s, α, β] with α,β ∈ {1,2}.
This is the "M" in L * M * M * ... * R.

Paper Eq A4: M[α,β] where:
- α=1 (safe): accepts any state, outputs β based on current state
- α=2 (danger): only accepts ↓ (forbids ↑↑), outputs β based on current state
Output β: 1 if current=↓ (safe), 2 if current=↑ (danger for next site)
"""
function _projector_tensor_bulk()
    d, χ = 2, 2
    W = zeros(ComplexF64, d, d, χ, χ)
    for α in 1:χ
        for s in [SPIN_DOWN, SPIN_UP]
            # If danger (α=2) and current=↑: forbidden (would be ↑↑)
            if α == 2 && s == SPIN_UP
                continue
            end
            # Output: safe (β=1) if ↓, danger (β=2) if ↑
            β = (s == SPIN_UP) ? 2 : 1
            W[s, s, α, β] = 1.0
        end
    end
    return W
end

"""
    _projector_tensor_right() -> Array{ComplexF64, 3}

Right boundary tensor for projector 𝒫. Shape: W[s', s, α] with α ∈ {1,2}.
This is the "R" in L * M * M * ... * R.

Paper Eq A5: R[↓] = (1,0)ᵀ, R[↑] = (0,1)ᵀ
"""
function _projector_tensor_right()
    d, χ = 2, 2
    W = zeros(ComplexF64, d, d, χ)
    # ↓ accepts α=1
    W[SPIN_DOWN, SPIN_DOWN, 1] = 1.0
    # ↑ accepts only α=2 (safe)
    W[SPIN_UP, SPIN_UP, 2] = 1.0
    return W
end

#------------------------------------------------------------------------------
# Down Projector 𝒟 = |↓⟩⟨↓| (χ=1 local operator)
#------------------------------------------------------------------------------

"""
    _down_projector_tensor() -> Array{ComplexF64, 4}

Down projector 𝒟 = |↓⟩⟨↓|. Shape: W[s', s, α, β] with α,β ∈ {1}.
This is a χ=1 local operator (trivial virtual bonds).
"""
function _down_projector_tensor()
    d = 2
    W = zeros(ComplexF64, d, d, 1, 1)
    W[SPIN_DOWN, SPIN_DOWN, 1, 1] = 1.0
    return W
end

#------------------------------------------------------------------------------
# Flip Operator 𝒳 = σˣ (χ=1 local operator)
#------------------------------------------------------------------------------

"""
    _flip_operator_tensor(Ω::Float64) -> Array{ComplexF64, 4}

Flip operator 𝒳 = Ω σˣ. Shape: W[s', s, α, β] with α,β ∈ {1}.
This is a χ=1 local operator (trivial virtual bonds).
"""
function _flip_operator_tensor(Ω::Float64)
    d = 2
    W = zeros(ComplexF64, d, d, 1, 1)
    W[SPIN_DOWN, SPIN_UP, 1, 1] = Ω
    W[SPIN_UP, SPIN_DOWN, 1, 1] = Ω
    return W
end

#------------------------------------------------------------------------------
# Identity tensor (χ=1) for single-site projector regions
#------------------------------------------------------------------------------

"""
    _identity_tensor() -> Array{ComplexF64, 4}

Identity operator. Shape: W[s', s, α, β] with α,β ∈ {1}.
Used when projector region has only one site (no constraint needed).
"""
function _identity_tensor()
    d = 2
    W = zeros(ComplexF64, d, d, 1, 1)
    W[SPIN_DOWN, SPIN_DOWN, 1, 1] = 1.0
    W[SPIN_UP, SPIN_UP, 1, 1] = 1.0
    return W
end

#------------------------------------------------------------------------------
# Number Operator 𝒩 = |↑⟩⟨↑| (χ=1 local operator)
#------------------------------------------------------------------------------

"""
    _number_operator_tensor(δ::Float64) -> Array{ComplexF64, 4}

Number operator 𝒩 = δ |↑⟩⟨↑|. Shape: W[s', s, α, β] with α,β ∈ {1}.
This is a χ=1 local operator (trivial virtual bonds).
Used as the local term in the PNP energy density: h_l^{PNP} includes δ n_l.
"""
function _number_operator_tensor(δ::Float64)
    d = 2
    W = zeros(ComplexF64, d, d, 1, 1)
    W[SPIN_UP, SPIN_UP, 1, 1] = δ
    return W
end

#==============================================================================#
# Projector-Sandwich Helper
#==============================================================================#

"""
    _projector_sandwich_mpo(N::Int, first_op::Int, last_op::Int,
                            central_tensors::Vector{Array{ComplexF64,4}};
                            sites=nothing) -> MPO

Generic MPO builder for a single operator block sandwiched by constraint
projectors:

    h = 𝒫_{1..first_op-1} · [central_tensors] · 𝒫_{last_op+1..N}

Each element of `central_tensors` is a (2,2,1,1) χ=1 local-operator tensor
(e.g. `_down_projector_tensor()`, `_flip_operator_tensor(Ω)`,
`_number_operator_tensor(δ)`).  The central block occupies original sites
`first_op` through `last_op` inclusive, so
`length(central_tensors) == last_op - first_op + 1`.

Left projector spans sites 1 … first_op−1; right projector spans
sites last_op+1 … N.  Each side must have at least 1 site, so
`first_op ≥ 2` and `last_op ≤ N−1`.

# Arguments
- `sites`: optional pre-allocated `Vector{Index}` of length N.
  When provided the returned MPO shares those indices, allowing two such
  MPOs to be added with `add()`.  When `nothing`, fresh `siteinds("S=1/2", N)`
  are created internally.
"""
function _projector_sandwich_mpo(N::Int, first_op::Int, last_op::Int,
                                 central_tensors::Vector{Array{ComplexF64,4}};
                                 sites=nothing)
    @assert first_op >= 2 "first_op must be ≥ 2 (need ≥ 1 left-projector site). Got first_op=$first_op"
    @assert last_op <= N-1 "last_op must be ≤ N-1 (need ≥ 1 right-projector site). Got last_op=$last_op, N=$N"
    @assert length(central_tensors) == last_op - first_op + 1 "central_tensors length must equal last_op - first_op + 1"

    d = 2  # spin-1/2

    if sites === nothing
        sites = siteinds("S=1/2", N)
    end
    M = MPO(sites)

    # --- Bond-dimension profile ---
    # Left projector: sites 1 … first_op-1
    #   1 site  → χ=1 (identity, no internal bond)
    #   ≥2 sites → χ=2 internally (L-M-…-M-R)
    # Central: sites first_op … last_op → χ=1
    # Right projector: sites last_op+1 … N
    #   1 site  → χ=1
    #   ≥2 sites → χ=2 internally
    left_P_sites  = first_op - 1          # number of left-projector sites
    right_P_sites = N - last_op           # number of right-projector sites

    χ = Vector{Int}(undef, N-1)
    for i in 1:(N-1)
        if i <= first_op - 2
            # inside left projector (only exists when left_P_sites ≥ 2)
            χ[i] = 2
        elseif i >= first_op - 1 && i <= last_op
            # transition left-P → central, within central, central → right-P
            χ[i] = 1
        else
            # inside right projector
            χ[i] = 2
        end
    end

    links = [Index(χ[i], "Link,l=$i") for i in 1:(N-1)]

    for i in 1:N
        s = sites[i]

        # ============================================================
        # Left projector region (sites 1 … first_op-1)
        # ============================================================
        if i <= first_op - 1
            if left_P_sites == 1
                # Single site: identity (χ=1 output)
                W = _identity_tensor()
                M[i] = ITensor(W[:,:,1,:], s', dag(s), links[i])

            elseif i == 1
                # L tensor (outputs χ=2)
                W = _projector_tensor_left()
                M[i] = ITensor(W, s', dag(s), links[i])

            elseif i < first_op - 1
                # M tensor (χ=2 both sides)
                W = _projector_tensor_bulk()
                M[i] = ITensor(W, s', dag(s), dag(links[i-1]), links[i])

            else  # i == first_op - 1
                # R tensor, reshaped to (d, d, 2, 1) for χ=1 right bond
                W = _projector_tensor_right()
                W_reshaped = reshape(W, (d, d, 2, 1))
                M[i] = ITensor(W_reshaped, s', dag(s), dag(links[i-1]), links[i])
            end

        # ============================================================
        # Central operator block (sites first_op … last_op)
        # ============================================================
        elseif i >= first_op && i <= last_op
            W = central_tensors[i - first_op + 1]   # (2,2,1,1)
            if i == 1  # can't happen given first_op≥2, but safe
                M[i] = ITensor(W[:,:,1,:], s', dag(s), links[i])
            elseif i == N  # can't happen given last_op≤N-1
                M[i] = ITensor(W[:,:,:,1], s', dag(s), dag(links[i-1]))
            else
                M[i] = ITensor(W, s', dag(s), dag(links[i-1]), links[i])
            end

        # ============================================================
        # Right projector region (sites last_op+1 … N)
        # ============================================================
        else
            if right_P_sites == 1
                # Single site: identity (χ=1 input)
                W = _identity_tensor()
                M[i] = ITensor(W[:,:,:,1], s', dag(s), dag(links[i-1]))

            elseif i == last_op + 1
                # L tensor, reshaped to (d, d, 1, 2) for χ=1 left bond
                W = _projector_tensor_left()
                W_reshaped = reshape(W, (d, d, 1, 2))
                M[i] = ITensor(W_reshaped, s', dag(s), dag(links[i-1]), links[i])

            elseif i < N
                # M tensor (χ=2 both sides)
                W = _projector_tensor_bulk()
                M[i] = ITensor(W, s', dag(s), dag(links[i-1]), links[i])

            else  # i == N
                # R tensor (χ=2 input, no right bond)
                W = _projector_tensor_right()
                M[i] = ITensor(W, s', dag(s), dag(links[i-1]))
            end
        end
    end

    return M
end

#==============================================================================#
# Energy Density MPO Construction (Bulk Case)
#==============================================================================#

"""
    energy_density_original(N::Int, l::Int; Ω=1.0) -> MPO

Construct the energy density operator h_l = Ω P_{l-1} σˣ_l P_{l+1} as an MPO
on N original spin-1/2 sites.

Structure (Eq A6): h = P · D · X · D · P
where P = L M M ... M R is a complete projector (χ=2 internally, terminates to χ=1).

MPO structure:
- Sites 1 to l-2: Left projector [L - M - ... - M - R]
- Site l-1: Down projector D (χ=1)
- Site l: Flip operator X (χ=1)
- Site l+1: Down projector D (χ=1)
- Sites l+2 to N: Right projector [L - M - ... - M - R]

Bond dimensions: 2 - 2 - ... - 2 - 1 - 1 - 1 - 2 - 2 - ... - 2
                [1]           [l-2][l-1][l][l+1][l+2]        [N]

Note: Requires bulk sites (3 ≤ l ≤ N-2). For boundary cases, use OpSum-based energy_density().
"""
function energy_density_original(N::Int, l::Int; Ω=1.0)
    @assert 3 <= l <= N-2 "Site index l must be in bulk (3 ≤ l ≤ N-2). Got l=$l, N=$N"

    d = 2  # spin-1/2

    # Create standard S=1/2 site indices
    sites = siteinds("S=1/2", N)
    M = MPO(sites)

    # Determine bond dimensions
    # Left projector (sites 1 to l-2): χ=2 internally if multiple sites, else χ=1
    # D-X-D region (sites l-1 to l+1): χ=1
    # Right projector (sites l+2 to N): χ=2 internally if multiple sites, else χ=1
    χ = Vector{Int}(undef, N-1)

    for i in 1:(N-1)
        if i <= l-3
            # Within left projector (between L/M sites)
            χ[i] = 2
        elseif i >= l-2 && i <= l+1
            # Between left P and D, within D-X-D, between D and right P
            χ[i] = 1
        else
            # Within right projector (between L/M/R sites)
            χ[i] = 2
        end
    end

    # Create link indices
    links = [Index(χ[i], "Link,l=$i") for i in 1:(N-1)]

    # Determine left projector structure
    left_P_sites = l - 2  # number of sites in left projector
    # Determine right projector structure
    right_P_sites = N - (l + 1)  # number of sites in right projector

    for i in 1:N
        s = sites[i]

        # === Left projector region (sites 1 to l-2) ===
        if i <= l - 2
            if left_P_sites == 1
                # Single site: identity (no constraint, outputs χ=1)
                W = _identity_tensor()
                M[i] = ITensor(W[:,:,1,:], s', dag(s), links[i])
            elseif i == 1
                # L tensor (outputs χ=2)
                W = _projector_tensor_left()
                M[i] = ITensor(W, s', dag(s), links[i])
            elseif i < l - 2
                # M tensor (χ=2 both sides)
                W = _projector_tensor_bulk()
                M[i] = ITensor(W, s', dag(s), dag(links[i-1]), links[i])
            else  # i == l - 2
                # R tensor with trivial χ=1 right bond (reshape from (2,2,2) to (2,2,2,1))
                W = _projector_tensor_right()
                W_reshaped = reshape(W, (d, d, 2, 1))
                M[i] = ITensor(W_reshaped, s', dag(s), dag(links[i-1]), links[i])
            end

        # === D-X-D region (sites l-1, l, l+1) ===
        elseif i == l - 1
            W = _down_projector_tensor()
            M[i] = ITensor(W, s', dag(s), dag(links[i-1]), links[i])

        elseif i == l
            W = _flip_operator_tensor(Ω)
            M[i] = ITensor(W, s', dag(s), dag(links[i-1]), links[i])

        elseif i == l + 1
            W = _down_projector_tensor()
            M[i] = ITensor(W, s', dag(s), dag(links[i-1]), links[i])

        # === Right projector region (sites l+2 to N) ===
        else
            if right_P_sites == 1
                # Single site: identity (no constraint, accepts χ=1)
                W = _identity_tensor()
                M[i] = ITensor(W[:,:,:,1], s', dag(s), dag(links[i-1]))
            elseif i == l + 2
                # L tensor with trivial χ=1 left bond (reshape from (2,2,2) to (2,2,1,2))
                W = _projector_tensor_left()
                W_reshaped = reshape(W, (d, d, 1, 2))
                M[i] = ITensor(W_reshaped, s', dag(s), dag(links[i-1]), links[i])
            elseif i < N
                # M tensor (χ=2 both sides)
                W = _projector_tensor_bulk()
                M[i] = ITensor(W, s', dag(s), dag(links[i-1]), links[i])
            else  # i == N
                # R tensor (χ=2 input, no right bond)
                W = _projector_tensor_right()
                M[i] = ITensor(W, s', dag(s), dag(links[i-1]))
            end
        end
    end

    return M
end

"""
    projector_mpo_original(N::Int) -> MPO

Construct the global constraint projector 𝒫 as an MPO on N original spin-1/2 sites.

The projector enforces no |↑↑⟩ states. Bond dimension is χ=2.
Structure: L(1×2) * M(2×2) * ... * M(2×2) * R(2×1) → scalar
"""
function projector_mpo_original(N::Int)
    d = 2
    χ = 2

    sites = siteinds("S=1/2", N)
    M = MPO(sites)

    # Create link indices
    links = [Index(χ, "Link,l=$i") for i in 1:(N-1)]

    for i in 1:N
        s = sites[i]

        if i == 1 && i == N
            # Single site: identity (no constraint)
            W = zeros(ComplexF64, d, d)
            W[SPIN_DOWN, SPIN_DOWN] = 1.0
            W[SPIN_UP, SPIN_UP] = 1.0
            M[i] = ITensor(W, s', dag(s))
        elseif i == 1
            W = _projector_tensor_left()
            M[i] = ITensor(W, s', dag(s), links[i])
        elseif i == N
            W = _projector_tensor_right()
            M[i] = ITensor(W, s', dag(s), dag(links[i-1]))
        else
            W = _projector_tensor_bulk()
            M[i] = ITensor(W, s', dag(s), dag(links[i-1]), links[i])
        end
    end

    return M
end

#==============================================================================#
# MPO Merging: Original Sites → Merged Sites
#==============================================================================#

"""
    merge_mpo_pairs(M_original::MPO, merged_sites::Vector{Index}) -> MPO

Merge an MPO from original spin-1/2 sites to merged sites by contracting pairs.

Each pair of adjacent tensors (M[2i-1], M[2i]) is contracted along their
shared virtual bond to produce a single tensor on merged site i.

Physical indices are combined: (s_{2i-1}, s_{2i}) → s_merged
Mapping: (↓,↓)→DD(1), (↑,↓)→UD(2), (↓,↑)→DU(3)
The (↑,↑) state is excluded from the merged basis.

Handles varying bond dimensions (e.g., χ=2 for projector, χ=1 for local operators).
"""
function merge_mpo_pairs(M_original::MPO, merged_sites::Vector{Index})
    N_original = length(M_original)
    N_merged = length(merged_sites)
    @assert N_original == 2 * N_merged "Original MPO must have twice as many sites"

    d_orig = 2   # original spin-1/2 dimension
    d_merged = 3 # merged site dimension (DD, UD, DU)

    # State mapping: (s1, s2) in original → state in merged
    function merged_state(s1, s2)
        if s1 == SPIN_DOWN && s2 == SPIN_DOWN
            return 1  # DD
        elseif s1 == SPIN_UP && s2 == SPIN_DOWN
            return 2  # UD
        elseif s1 == SPIN_DOWN && s2 == SPIN_UP
            return 3  # DU
        else
            return 0  # UU - forbidden
        end
    end

    # Determine merged bond dimensions from original MPO
    # Bond after merged site i = bond after original site 2i
    χ_merged = Int[]
    for i_merged in 1:(N_merged-1)
        i_right = 2 * i_merged
        link = linkind(M_original, i_right)
        push!(χ_merged, dim(link))
    end

    # Create link indices for merged MPO
    links_merged = [Index(χ, "Link,l=$i") for (i, χ) in enumerate(χ_merged)]

    M_merged = MPO(merged_sites)

    for i_merged in 1:N_merged
        i_left = 2 * i_merged - 1
        i_right = 2 * i_merged

        s_merged = merged_sites[i_merged]

        # Get original tensors and contract
        T_left = M_original[i_left]
        T_right = M_original[i_right]
        T_combined = T_left * T_right

        # Get physical indices
        s1 = siteind(M_original, i_left)
        s2 = siteind(M_original, i_right)

        is_left_boundary = (i_merged == 1)
        is_right_boundary = (i_merged == N_merged)

        # Get bond dimensions for this merged site
        χ_l = is_left_boundary ? 0 : χ_merged[i_merged-1]
        χ_r = is_right_boundary ? 0 : χ_merged[i_merged]

        if is_left_boundary && is_right_boundary
            # Single merged site
            W_merged = zeros(ComplexF64, d_merged, d_merged)
            for sp1 in 1:d_orig, sp2 in 1:d_orig
                for s1_val in 1:d_orig, s2_val in 1:d_orig
                    sp_merged = merged_state(sp1, sp2)
                    s_merged_val = merged_state(s1_val, s2_val)
                    if sp_merged > 0 && s_merged_val > 0
                        val = T_combined[s1'=>sp1, dag(s1)=>s1_val, s2'=>sp2, dag(s2)=>s2_val]
                        W_merged[sp_merged, s_merged_val] += val
                    end
                end
            end
            M_merged[i_merged] = ITensor(W_merged, s_merged', dag(s_merged))

        elseif is_left_boundary
            link_r_orig = linkind(M_original, i_right)
            W_merged = zeros(ComplexF64, d_merged, d_merged, χ_r)
            for sp1 in 1:d_orig, sp2 in 1:d_orig
                for s1_val in 1:d_orig, s2_val in 1:d_orig
                    sp_merged = merged_state(sp1, sp2)
                    s_merged_val = merged_state(s1_val, s2_val)
                    if sp_merged > 0 && s_merged_val > 0
                        for β in 1:χ_r
                            val = T_combined[s1'=>sp1, dag(s1)=>s1_val,
                                            s2'=>sp2, dag(s2)=>s2_val,
                                            link_r_orig=>β]
                            W_merged[sp_merged, s_merged_val, β] += val
                        end
                    end
                end
            end
            M_merged[i_merged] = ITensor(W_merged, s_merged', dag(s_merged), links_merged[i_merged])

        elseif is_right_boundary
            link_l_orig = linkind(M_original, i_left - 1)
            W_merged = zeros(ComplexF64, d_merged, d_merged, χ_l)
            for sp1 in 1:d_orig, sp2 in 1:d_orig
                for s1_val in 1:d_orig, s2_val in 1:d_orig
                    sp_merged = merged_state(sp1, sp2)
                    s_merged_val = merged_state(s1_val, s2_val)
                    if sp_merged > 0 && s_merged_val > 0
                        for α in 1:χ_l
                            val = T_combined[s1'=>sp1, dag(s1)=>s1_val,
                                            s2'=>sp2, dag(s2)=>s2_val,
                                            dag(link_l_orig)=>α]
                            W_merged[sp_merged, s_merged_val, α] += val
                        end
                    end
                end
            end
            M_merged[i_merged] = ITensor(W_merged, s_merged', dag(s_merged), dag(links_merged[i_merged-1]))

        else
            # Bulk
            link_l_orig = linkind(M_original, i_left - 1)
            link_r_orig = linkind(M_original, i_right)
            W_merged = zeros(ComplexF64, d_merged, d_merged, χ_l, χ_r)
            for sp1 in 1:d_orig, sp2 in 1:d_orig
                for s1_val in 1:d_orig, s2_val in 1:d_orig
                    sp_merged = merged_state(sp1, sp2)
                    s_merged_val = merged_state(s1_val, s2_val)
                    if sp_merged > 0 && s_merged_val > 0
                        for α in 1:χ_l, β in 1:χ_r
                            val = T_combined[s1'=>sp1, dag(s1)=>s1_val,
                                            s2'=>sp2, dag(s2)=>s2_val,
                                            dag(link_l_orig)=>α, link_r_orig=>β]
                            W_merged[sp_merged, s_merged_val, α, β] += val
                        end
                    end
                end
            end
            M_merged[i_merged] = ITensor(W_merged, s_merged', dag(s_merged),
                                         dag(links_merged[i_merged-1]), links_merged[i_merged])
        end
    end

    return M_merged
end

#==============================================================================#
# New Energy Density on Merged Sites (Paper's Approach)
#==============================================================================#

"""
    energy_density_merged(merged_sites::Vector{Index}, l_merged::Int; Ω=1.0) -> MPO

Construct the energy density operator on merged sites following the paper's
Appendix A approach: build on original sites, then merge.

For merged site l_merged containing original sites (2l-1, 2l), the energy
density includes two PXP terms:
- Term 1: h_{2l-1} = P_{2l-2} σˣ_{2l-1} P_{2l}  (flip left spin)
- Term 2: h_{2l} = P_{2l-1} σˣ_{2l} P_{2l+1}    (flip right spin)

Note: Requires l_merged to be in bulk (not near boundaries) so that both
original sites 2l_merged-1 and 2l_merged are in bulk (at least 3 sites
from boundaries). This means N_merged >= 4 and 2 <= l_merged <= N_merged-1.
"""
function energy_density_merged(merged_sites::Vector{Index}, l_merged::Int; Ω=1.0)
    N_merged = length(merged_sites)
    N_original = 2 * N_merged

    # Original site indices for this merged site
    l_left = 2 * l_merged - 1   # left spin of merged site
    l_right = 2 * l_merged      # right spin of merged site

    # Check bulk constraints
    @assert 3 <= l_left <= N_original-2 "Left original site $l_left must be in bulk"
    @assert 3 <= l_right <= N_original-2 "Right original site $l_right must be in bulk"

    # Build Term 1: h_{2l-1} = P_{2l-2} σˣ_{2l-1} P_{2l}
    h1_original = energy_density_original(N_original, l_left; Ω=Ω)

    # Build Term 2: h_{2l} = P_{2l-1} σˣ_{2l} P_{2l+1}
    h2_original = energy_density_original(N_original, l_right; Ω=Ω)

    # Merge both terms
    h1_merged = merge_mpo_pairs(h1_original, merged_sites)
    h2_merged = merge_mpo_pairs(h2_original, merged_sites)

    # Add the two MPOs
    return h1_merged
    #add(h1_merged, h2_merged; alg="directsum")
end

"""
    center_energy_density_merged(merged_sites::Vector{Index}; Ω=1.0) -> MPO

Construct the energy density at the center merged site using the paper's approach.
"""
function center_energy_density_merged(merged_sites::Vector{Index}; Ω=1.0)
    N = length(merged_sites)
    l = div(N + 1, 2)  # Center merged site
    return energy_density_merged(merged_sites, l; Ω=Ω)
end

"""
    projector_mpo_merged(merged_sites::Vector{Index}) -> MPO

Construct the global constraint projector 𝒫 on merged sites using the paper's
approach: build on original sites with χ=2, then merge.

The merged MPO has bond dimension χ=2.
"""
function projector_mpo_merged(merged_sites::Vector{Index})
    N_merged = length(merged_sites)
    N_original = 2 * N_merged

    # Build projector on original sites
    P_original = projector_mpo_original(N_original)

    # Merge to get projector on merged sites
    return merge_mpo_pairs(P_original, merged_sites)
end

"""
    identity_mpo_merged(merged_sites::Vector{Index}) -> MPO

Construct the identity operator as an MPO on merged sites.
"""
function identity_mpo_merged(merged_sites::Vector{Index})
    N = length(merged_sites)
    Id3 = Matrix{ComplexF64}(I, 3, 3)

    M = MPO(merged_sites)

    for i in 1:N
        s = merged_sites[i]
        if i == 1
            link_r = Index(1, "Link,l=$i")
            M[i] = ITensor(Id3, s', dag(s)) * ITensor([1.0], link_r)
        elseif i == N
            link_l = Index(1, "Link,l=$(i-1)")
            M[i] = ITensor(Id3, s', dag(s)) * ITensor([1.0], dag(link_l))
        else
            link_l = Index(1, "Link,l=$(i-1)")
            link_r = Index(1, "Link,l=$i")
            M[i] = ITensor(Id3, s', dag(s)) * ITensor([1.0], dag(link_l), link_r)
        end
    end

    return M
end

#==============================================================================#
# PNP Energy Density (original sites)
#==============================================================================#

"""
    energy_density_pnp_original(N::Int, l::Int; Ω=1.0, δ=0.0) -> MPO

Construct the PNP energy density at original site `l`:

    h_l^{PNP} = Ω P_{l-1} σˣ_l P_{l+1}  +  δ n_l

The first term is the usual PXP 3-site block (D·X·D sandwiched by projectors).
The second term is the number operator δ |↑⟩⟨↑| at site `l`, also sandwiched
by projectors (single-site central block).

Both terms share the same site indices so they can be added directly.

Requires `3 ≤ l ≤ N-2` (PXP block needs at least 1 projector site each side).
"""
function energy_density_pnp_original(N::Int, l::Int; Ω=1.0, δ=0.0)
    @assert 3 <= l <= N-2 "PNP original-site energy density requires 3 ≤ l ≤ N-2. Got l=$l, N=$N"

    # Shared site indices so the two MPOs can be added
    sites = siteinds("S=1/2", N)

    # Term 1: PXP  —  central block is D · X · D at sites l-1, l, l+1
    term_pxp = _projector_sandwich_mpo(N, l-1, l+1,
        [_down_projector_tensor(), _flip_operator_tensor(Ω), _down_projector_tensor()];
        sites=sites)

    # Term 2: number operator  —  central block is δ n at site l
    term_n = _projector_sandwich_mpo(N, l-1, l+1,
        [_down_projector_tensor(), _number_operator_tensor(δ), _down_projector_tensor()];
        sites=sites)

    return add(term_pxp, term_n; alg="directsum")
end

#==============================================================================#
# PNPNP Energy Density (original sites)
#==============================================================================#

"""
    energy_density_pnpnp_original(N::Int, l::Int; Ω=1.0, ξ=0.0) -> MPO

Construct the PNPNP energy density at original site `l`:

    h_l^{PNPNP} = Ω P_{l-1} σˣ_l P_{l+1}
                 + ξ P_{l-2} σˣ_{l-1} P_l σˣ_{l+1} P_{l+2}

The first term is the PXP 3-site block (D·X·D).
The second term is the 5-site PNPNP block (D·X·D·X·D) with the ξ coefficient
placed on the first flip tensor.

Both terms share the same site indices.

Requires `4 ≤ l ≤ N-3` (5-site block needs `first_op = l-2 ≥ 2` and
`last_op = l+2 ≤ N-1`).
"""
function energy_density_pnpnp_original(N::Int, l::Int; Ω=1.0, ξ=0.0)
    @assert 4 <= l <= N-3 "PNPNP original-site energy density requires 4 ≤ l ≤ N-3. Got l=$l, N=$N"

    sites = siteinds("S=1/2", N)

    # Term 1: PXP  —  D · X(Ω) · D  at sites l-1, l, l+1
    term_pxp = _projector_sandwich_mpo(N, l-1, l+1,
        [_down_projector_tensor(), _flip_operator_tensor(Ω), _down_projector_tensor()];
        sites=sites)

    # Term 2: 5-site PNPNP  —  D · X(ξ) · D · X(1) · D  at sites l-2 … l+2
    # The ξ coefficient is carried by the first flip tensor (site l-1).
    term_5site = _projector_sandwich_mpo(N, l-2, l+2,
        [_down_projector_tensor(),
         _flip_operator_tensor(ξ),
         _down_projector_tensor(),
         _flip_operator_tensor(1.0),
         _down_projector_tensor()];
        sites=sites)

    return add(term_pxp, term_5site; alg="directsum")
end

#==============================================================================#
# PNP / PNPNP Energy Density on Merged Sites
#==============================================================================#

"""
    energy_density_pnp_merged(merged_sites::Vector{Index}, l_merged::Int;
                              Ω=1.0, δ=0.0) -> MPO

Construct the PNP energy density on merged sites at merged site `l_merged`.

Merged site `l_merged` contains original sites `(2l-1, 2l)`.  The energy
density sums all PNP terms whose "center" falls on either original site:

- PXP at `2l-1` and PXP at `2l`
- number at `2l-1` and number at `2l`

Each is built on original sites via `energy_density_pnp_original`, merged with
`merge_mpo_pairs`, then all four merged MPOs are summed.

Requires `2 ≤ l_merged ≤ N_merged-1` (both original sites must satisfy
`3 ≤ l_orig ≤ N_orig-2`).
"""
function energy_density_pnp_merged(merged_sites::Vector{Index}, l_merged::Int;
                                   Ω=1.0, δ=0.0)
    N_merged   = length(merged_sites)
    N_original = 2 * N_merged

    l_left  = 2 * l_merged - 1   # left original site
    l_right = 2 * l_merged       # right original site

    @assert 3 <= l_left  <= N_original-2 "Left original site $l_left out of bulk"
    @assert 3 <= l_right <= N_original-2 "Right original site $l_right out of bulk"

    # Build four original-site MPOs and merge each
    h_pnp_left  = energy_density_pnp_original(N_original, l_left;  Ω=Ω, δ=δ)
    h_pnp_right = energy_density_pnp_original(N_original, l_right; Ω=Ω, δ=δ)

    h_left_merged  = merge_mpo_pairs(h_pnp_left,  merged_sites)
    h_right_merged = merge_mpo_pairs(h_pnp_right, merged_sites)

    return add(h_left_merged, h_right_merged; alg="directsum")
end

"""
    energy_density_pnpnp_merged(merged_sites::Vector{Index}, l_merged::Int;
                                Ω=1.0, ξ=0.0) -> MPO

Construct the PNPNP energy density on merged sites at merged site `l_merged`.

Merged site `l_merged` contains original sites `(2l-1, 2l)`.  The energy
density sums:

- PXP at `2l-1` and PXP at `2l`
- 5-site PNPNP at `2l-1` and 5-site PNPNP at `2l`

Each is built via `energy_density_pnpnp_original`, merged, then summed.

Requires `3 ≤ l_merged ≤ N_merged-2` (5-site term at original site `2l-1`
needs `2l-1 ≥ 4`, i.e. `l_merged ≥ 3`; at `2l` needs `2l ≤ N_orig-3`,
i.e. `l_merged ≤ N_merged-2`).
"""
function energy_density_pnpnp_merged(merged_sites::Vector{Index}, l_merged::Int;
                                     Ω=1.0, ξ=0.0)
    N_merged   = length(merged_sites)
    N_original = 2 * N_merged

    l_left  = 2 * l_merged - 1
    l_right = 2 * l_merged

    @assert 4 <= l_left  <= N_original-3 "Left original site $l_left out of PNPNP bulk"
    @assert 4 <= l_right <= N_original-3 "Right original site $l_right out of PNPNP bulk"

    h_pnpnp_left  = energy_density_pnpnp_original(N_original, l_left;  Ω=Ω, ξ=ξ)
    h_pnpnp_right = energy_density_pnpnp_original(N_original, l_right; Ω=Ω, ξ=ξ)

    h_left_merged  = merge_mpo_pairs(h_pnpnp_left,  merged_sites)
    h_right_merged = merge_mpo_pairs(h_pnpnp_right, merged_sites)

    return add(h_left_merged, h_right_merged; alg="directsum")
end

#==============================================================================#
# Center-site Convenience Functions
#==============================================================================#

"""
    center_energy_density_pnp_merged(merged_sites::Vector{Index};
                                     Ω=1.0, δ=0.0) -> MPO

PNP energy density at the center merged site.
"""
function center_energy_density_pnp_merged(merged_sites::Vector{Index};
                                          Ω=1.0, δ=0.0)
    N = length(merged_sites)
    l = div(N + 1, 2)
    return energy_density_pnp_merged(merged_sites, l; Ω=Ω, δ=δ)
end

"""
    center_energy_density_pnpnp_merged(merged_sites::Vector{Index};
                                       Ω=1.0, ξ=0.0) -> MPO

PNPNP energy density at the center merged site.
"""
function center_energy_density_pnpnp_merged(merged_sites::Vector{Index};
                                            Ω=1.0, ξ=0.0)
    N = length(merged_sites)
    l = div(N + 1, 2)
    return energy_density_pnpnp_merged(merged_sites, l; Ω=Ω, ξ=ξ)
end

#==============================================================================#
# DEPRECATED: Old Merged Site Implementation (χ=4)
# Commented out in favor of paper-based approach (χ=2)
#==============================================================================#

#=
"""
    energy_density_merged(merged_sites::Vector{Index}, l::Int; Ω=1.0) -> MPO

Construct the energy density operator on merged sites following the method
described in Ljubotina et al., Phys. Rev. X 13, 011033 (2023), Appendix A.

The energy density is defined as (Eq. A1):
    h_ℓ(0) = 𝒫 P_{ℓ-1} σ^x_ℓ P_{ℓ+1}

where 𝒫 is the global projector onto the constrained Hilbert space (no |↑↑⟩).

In MPO form (Eq. A6):
    h(ℓ, 0) = 𝒫_{1,ℓ-2} 𝒟_{ℓ-1} 𝒳_ℓ 𝒟_{ℓ+1} 𝒫_{ℓ+2,N-ℓ-1}

where:
- 𝒟_i = |↓⟩⟨↓|_i (projects to down, Eq. A7)
- 𝒳_i = |↓⟩⟨↑| + |↑⟩⟨↓| (σ^x operator, Eq. A8)
- 𝒫 is the projector MPO with bond dimension 2 (Eqs. A2-A5)

For merged sites where pairs (2i-1, 2i) form site i with states:
- |DD⟩ = |↓↓⟩ (state 1, or |0⟩ in spin-1)
- |UD⟩ = |↑↓⟩ (state 2, or |−⟩ in spin-1)
- |DU⟩ = |↓↑⟩ (state 3, or |+⟩ in spin-1)

The constraint forbids |+−⟩ = |DU,UD⟩ between adjacent merged sites.

The energy density at merged site l includes PXP terms from original sites
(2l-1) and (2l):
- Term 1: P_{2l-2} σ^x_{2l-1} P_{2l} (flip left spin)
- Term 2: P_{2l-1} σ^x_{2l} P_{2l+1} (flip right spin)

# Arguments
- `merged_sites::Vector{Index}`: The merged site indices
- `l::Int`: The merged site index where the energy density is localized
- `Ω::Float64`: Rabi frequency

# Returns
An MPO on merged sites representing the local energy density with proper
projector structure.
"""
function energy_density_merged(merged_sites::Vector{Index}, l::Int; Ω=1.0)
    N = length(merged_sites)
    @assert 1 <= l <= N "Merged site index l must be between 1 and $N"

    # State encoding for merged sites (matching paper's spin-1 convention)
    # |↓↓⟩ → 1 (|0⟩), |↑↓⟩ → 2 (|−⟩), |↓↑⟩ → 3 (|+⟩)
    DD, UD, DU = 1, 2, 3
    d = 3  # merged site dimension

    # Build the MPO following Eq. A6 structure translated to merged sites
    # Bond dimension: 4 to encode projector state (2) × operator stage (2)
    # Virtual indices: (proj_safe=0/danger=1) × (before_op=0/after_op=1)
    # Linearized: 0 = (safe, before), 1 = (danger, before),
    #             2 = (safe, after),  3 = (danger, after)

    χ = 4  # bond dimension

    # Helper to convert (proj_state, op_stage) to linear index (1-based for Julia)
    lin(p, o) = p + 2*o + 1  # p ∈ {0,1}, o ∈ {0,1} → 1,2,3,4

    M = MPO(merged_sites)

    for i in 1:N
        s = merged_sites[i]

        # Determine tensor dimensions based on position
        if i == 1
            χ_l, χ_r = 1, (i == N ? 1 : χ)
        elseif i == N
            χ_l, χ_r = χ, 1
        else
            χ_l, χ_r = χ, χ
        end

        # Initialize tensor data
        W = zeros(ComplexF64, d, d, χ_l, χ_r)

        # Fill tensor based on position relative to operator site l
        if i < l - 1
            # Before operator region: just projector
            W = _projector_tensor_merged(d, χ_l, χ_r, :before, i == 1, false)
        elseif i == l - 1
            # Adjacent to operator on the left
            # For Term 1: need to check right spin of this site = ↓
            # States with right spin ↓: DD (1), UD (2)
            W = _projector_tensor_with_right_constraint_merged(d, χ_l, χ_r, Ω, i == 1)
        elseif i == l
            # Operator site: apply energy density
            W = _energy_density_tensor_merged(d, χ_l, χ_r, Ω, l == 1, l == N)
        elseif i == l + 1
            # Adjacent to operator on the right
            # For Term 2: need to check left spin of this site = ↓
            # States with left spin ↓: DD (1), DU (3)
            W = _projector_tensor_with_left_constraint_merged(d, χ_l, χ_r, Ω, i == N)
        else
            # After operator region: just projector
            W = _projector_tensor_merged(d, χ_l, χ_r, :after, false, i == N)
        end

        # Create ITensor from data
        if i == 1 && i == N
            # Single site (N=1): no link indices
            M[i] = ITensor(W[:, :, 1, 1], s', dag(s))
        elseif i == 1
            link_r = Index(χ_r, "Link,l=$i")
            M[i] = ITensor(W[:, :, 1, :], s', dag(s), link_r)
        elseif i == N
            link_l = Index(χ_l, "Link,l=$(i-1)")
            M[i] = ITensor(W[:, :, :, 1], s', dag(s), dag(link_l))
        else
            link_l = Index(χ_l, "Link,l=$(i-1)")
            link_r = Index(χ_r, "Link,l=$i")
            M[i] = ITensor(W, s', dag(s), dag(link_l), link_r)
        end
    end

    return M
end

"""
Build projector tensor for merged sites (implementing Eqs. A2-A5 in merged basis).

The projector enforces: no |DU⟩|UD⟩ = |↓↑↑↓⟩ between adjacent merged sites.

Virtual indices track:
- proj_state: 0 = safe (previous not DU), 1 = danger (previous is DU)
- op_stage: 0 = before operator, 1 = after operator
"""
function _projector_tensor_merged(d::Int, χ_l::Int, χ_r::Int, stage::Symbol,
                                   is_left_boundary::Bool, is_right_boundary::Bool)
    DD, UD, DU = 1, 2, 3
    lin(p, o) = p + 2*o + 1

    W = zeros(ComplexF64, d, d, χ_l, χ_r)

    op_stage = (stage == :before) ? 0 : 1

    if is_left_boundary && is_right_boundary
        # Single site case
        for state in [DD, UD, DU]
            W[state, state, 1, 1] = 1.0
        end
    elseif is_left_boundary
        # Left boundary: initialize projector state based on current state
        for state in [DD, UD, DU]
            # DD, UD → safe (proj=0), DU → danger (proj=1)
            proj_out = (state == DU) ? 1 : 0
            W[state, state, 1, lin(proj_out, op_stage)] = 1.0
        end
    elseif is_right_boundary
        # Right boundary: accept based on projector constraint
        for proj_in in [0, 1]
            for state in [DD, UD, DU]
                # If danger (proj_in=1) and state=UD, forbidden
                if proj_in == 1 && state == UD
                    continue
                end
                W[state, state, lin(proj_in, op_stage), 1] = 1.0
            end
        end
    else
        # Bulk: transition projector state
        for proj_in in [0, 1]
            for state in [DD, UD, DU]
                # If danger and UD: forbidden
                if proj_in == 1 && state == UD
                    continue
                end
                # Output proj state based on current state
                proj_out = (state == DU) ? 1 : 0
                W[state, state, lin(proj_in, op_stage), lin(proj_out, op_stage)] = 1.0
            end
        end
    end

    return W
end

"""
Tensor at site l-1: projector with constraint that right spin must be ↓.
This implements 𝒟_{2l-2} for Term 1 in merged-site language.
"""
function _projector_tensor_with_right_constraint_merged(d::Int, χ_l::Int, χ_r::Int,
                                                         Ω::Float64, is_left_boundary::Bool)
    DD, UD, DU = 1, 2, 3
    lin(p, o) = p + 2*o + 1

    W = zeros(ComplexF64, d, d, χ_l, χ_r)

    # States with right spin = ↓: DD, UD (allowed for Term 1)
    # State DU has right spin = ↑ (not allowed for Term 1, but still passes projector)

    if is_left_boundary
        for state in [DD, UD, DU]
            proj_out = (state == DU) ? 1 : 0
            # Transition from before (op_stage=0) to after (op_stage=1)
            # Only DD, UD contribute to Term 1 (right spin ↓)
            if state == DD || state == UD
                # This state satisfies P_{2l-2}, so it can contribute to Term 1
                W[state, state, 1, lin(proj_out, 1)] = 1.0
            else
                # DU doesn't satisfy P_{2l-2}, passes as identity for projector only
                # But we still need to allow it for Term 2 which doesn't need this constraint
                W[state, state, 1, lin(proj_out, 1)] = 1.0
            end
        end
    else
        for proj_in in [0, 1]
            for state in [DD, UD, DU]
                if proj_in == 1 && state == UD
                    continue  # Forbidden by projector
                end
                proj_out = (state == DU) ? 1 : 0
                # Transition to op_stage=1 (at/after operator)
                W[state, state, lin(proj_in, 0), lin(proj_out, 1)] = 1.0
            end
        end
    end

    return W
end

"""
Tensor at operator site l: implements the energy density operator.

For merged site l containing original sites (2l-1, 2l):
- Term 1: P_{2l-2} σ^x_{2l-1} P_{2l} → DD ↔ UD (flip left spin, need right spin=↓)
- Term 2: P_{2l-1} σ^x_{2l} P_{2l+1} → DD ↔ DU (flip right spin, need left spin=↓)

Both terms require internal constraints:
- Term 1: Right spin of merged site must be ↓ (states DD, UD only)
- Term 2: Left spin of merged site must be ↓ (states DD, DU only)
"""
function _energy_density_tensor_merged(d::Int, χ_l::Int, χ_r::Int, Ω::Float64,
                                        is_left_boundary::Bool, is_right_boundary::Bool)
    DD, UD, DU = 1, 2, 3
    lin(p, o) = p + 2*o + 1

    W = zeros(ComplexF64, d, d, χ_l, χ_r)

    # Local operator matrix for energy density
    # Term 1: DD ↔ UD (flip left spin), requires P_{2l-2} (from l-1) and P_{2l} (right=↓, automatic for DD,UD)
    # Term 2: DD ↔ DU (flip right spin), requires P_{2l-1} (left=↓, automatic for DD,DU) and P_{2l+1} (from l+1)

    if is_left_boundary && is_right_boundary
        # Single site: both terms active without neighbor constraints
        # Term 1: DD ↔ UD
        W[DD, UD, 1, 1] = Ω
        W[UD, DD, 1, 1] = Ω
        # Term 2: DD ↔ DU
        W[DD, DU, 1, 1] = Ω
        W[DU, DD, 1, 1] = Ω
    elseif is_left_boundary
        # Left boundary: no l-1, so Term 1 has no P_{2l-2} constraint (boundary condition)
        for proj_out in [0, 1]
            # Term 1: DD ↔ UD, both output safe (DD→safe, UD→safe)
            if proj_out == 0  # Output from DD or UD
                W[DD, UD, 1, lin(0, 1)] = Ω
                W[UD, DD, 1, lin(0, 1)] = Ω
            end
            # Term 2: DD ↔ DU
            # DD→safe, DU→danger
            W[DD, DU, 1, lin(1, 1)] = Ω  # DU output → danger
            W[DU, DD, 1, lin(0, 1)] = Ω  # DD output → safe
        end
    elseif is_right_boundary
        # Right boundary: no l+1, so Term 2 has no P_{2l+1} constraint
        for proj_in in [0, 1]
            # Term 1: DD ↔ UD (need P_{2l-2} checked at l-1)
            # P_{2l} (right spin=↓) is automatic for DD, UD
            if !(proj_in == 1)  # Safe input or will check UD constraint
                W[DD, UD, lin(proj_in, 1), 1] = Ω
                W[UD, DD, lin(proj_in, 1), 1] = Ω
            end
            # Handle danger input
            if proj_in == 1
                # Can't have UD as input state (forbidden), but DD is ok
                W[DD, UD, lin(1, 1), 1] = Ω  # DD→UD ok even with danger input
                # UD→DD not allowed if proj_in=1 (would mean prev was DU, curr is UD: forbidden)
            end

            # Term 2: DD ↔ DU (need P_{2l-1}, i.e., left spin=↓: DD, DU only)
            # No P_{2l+1} constraint at boundary
            W[DD, DU, lin(proj_in, 1), 1] = Ω
            W[DU, DD, lin(proj_in, 1), 1] = Ω
        end
    else
        # Bulk: full constraints
        for proj_in in [0, 1]
            # Determine which input states are allowed
            # If proj_in=1 (danger), UD input is forbidden

            # Term 1: DD ↔ UD
            # P_{2l-2} checked at l-1 (encoded in transition to op_stage=1)
            # P_{2l}: right spin=↓, automatic for DD, UD
            # Output projector state: DD→0, UD→0

            # From DD (allowed for both proj_in)
            W[DD, UD, lin(proj_in, 1), lin(0, 1)] = Ω

            # From UD (only if proj_in=0, i.e., safe)
            if proj_in == 0
                W[UD, DD, lin(0, 1), lin(0, 1)] = Ω
            end

            # Term 2: DD ↔ DU
            # P_{2l-1}: left spin=↓, automatic for DD, DU
            # P_{2l+1} will be checked at l+1

            # From DD (allowed for both proj_in)
            W[DD, DU, lin(proj_in, 1), lin(1, 1)] = Ω  # DU output → danger

            # From DU (only if proj_in=0, since DU followed by UD would be forbidden,
            # but here we're outputting DD which is safe)
            if proj_in == 0
                W[DU, DD, lin(0, 1), lin(0, 1)] = Ω  # DD output → safe
            end
            # If proj_in=1, we can't have DU as previous state's output leading here...
            # Actually, if proj_in=1, it means previous site output DU
            # Current site can be DD (allowed) or DU (allowed), not UD
            if proj_in == 1
                W[DU, DD, lin(1, 1), lin(0, 1)] = Ω  # DU→DD, output safe
            end
        end
    end

    return W
end

"""
Tensor at site l+1: projector with constraint that left spin must be ↓.
This implements 𝒟_{2l+1} for Term 2 in merged-site language.
"""
function _projector_tensor_with_left_constraint_merged(d::Int, χ_l::Int, χ_r::Int,
                                                        Ω::Float64, is_right_boundary::Bool)
    DD, UD, DU = 1, 2, 3
    lin(p, o) = p + 2*o + 1

    W = zeros(ComplexF64, d, d, χ_l, χ_r)

    # States with left spin = ↓: DD, DU (allowed for Term 2)
    # UD has left spin = ↑

    if is_right_boundary
        for proj_in in [0, 1]
            for state in [DD, UD, DU]
                if proj_in == 1 && state == UD
                    continue  # Forbidden by projector (DU followed by UD)
                end
                # All valid states pass through; P_{2l+1} constraint is on left spin
                # DD and DU satisfy P_{2l+1}, UD does not
                # But we include all for the projector; Term 2 weight was set at site l
                W[state, state, lin(proj_in, 1), 1] = 1.0
            end
        end
    else
        for proj_in in [0, 1]
            for state in [DD, UD, DU]
                if proj_in == 1 && state == UD
                    continue  # Forbidden
                end
                proj_out = (state == DU) ? 1 : 0
                W[state, state, lin(proj_in, 1), lin(proj_out, 1)] = 1.0
            end
        end
    end

    return W
end

"""
    center_energy_density_merged(merged_sites::Vector{Index}; Ω=1.0) -> MPO

Construct the energy density at the center merged site.
"""
function center_energy_density_merged(merged_sites::Vector{Index}; Ω=1.0)
    N = length(merged_sites)
    l = div(N + 1, 2)  # Center merged site
    return energy_density_merged(merged_sites, l; Ω=Ω)
end

"""
    projector_mpo_merged(merged_sites::Vector{Index}) -> MPO

Construct the global constraint projector 𝒫 on merged sites.

The projector enforces: no |DU⟩|UD⟩ = |↓↑↑↓⟩ configurations between
adjacent merged sites. This is the Rydberg blockade constraint.

The projector has bond dimension 2, tracking whether the previous
merged site was DU (danger state) or not (safe state).
"""
function projector_mpo_merged(merged_sites::Vector{Index})
    N = length(merged_sites)
    DD, UD, DU = 1, 2, 3
    d = 3

    M = MPO(merged_sites)

    for i in 1:N
        s = merged_sites[i]

        if i == 1 && i == N
            # Single site: just identity (no constraint needed)
            W = zeros(ComplexF64, d, d)
            for state in [DD, UD, DU]
                W[state, state] = 1.0
            end
            M[i] = ITensor(W, s', dag(s))
        elseif i == 1
            # Left boundary: output projector state
            χ_r = 2
            link_r = Index(χ_r, "Link,l=$i")
            W = zeros(ComplexF64, d, d, χ_r)
            for state in [DD, UD, DU]
                # DD, UD → safe (index 1), DU → danger (index 2)
                proj_out = (state == DU) ? 2 : 1
                W[state, state, proj_out] = 1.0
            end
            M[i] = ITensor(W, s', dag(s), link_r)
        elseif i == N
            # Right boundary: check constraint
            χ_l = 2
            link_l = Index(χ_l, "Link,l=$(i-1)")
            W = zeros(ComplexF64, d, d, χ_l)
            for proj_in in 1:2
                for state in [DD, UD, DU]
                    # If danger (proj_in=2) and state=UD, forbidden
                    if proj_in == 2 && state == UD
                        continue
                    end
                    W[state, state, proj_in] = 1.0
                end
            end
            M[i] = ITensor(W, s', dag(s), dag(link_l))
        else
            # Bulk: transition projector state
            χ_l = 2
            χ_r = 2
            link_l = Index(χ_l, "Link,l=$(i-1)")
            link_r = Index(χ_r, "Link,l=$i")
            W = zeros(ComplexF64, d, d, χ_l, χ_r)
            for proj_in in 1:2
                for state in [DD, UD, DU]
                    if proj_in == 2 && state == UD
                        continue
                    end
                    proj_out = (state == DU) ? 2 : 1
                    W[state, state, proj_in, proj_out] = 1.0
                end
            end
            M[i] = ITensor(W, s', dag(s), dag(link_l), link_r)
        end
    end

    return M
end

"""
    identity_mpo_merged(merged_sites::Vector{Index}) -> MPO

Construct the identity operator as an MPO on merged sites.
"""
function identity_mpo_merged(merged_sites::Vector{Index})
    N = length(merged_sites)
    Id3 = Matrix{ComplexF64}(I, 3, 3)

    M = MPO(merged_sites)

    for i in 1:N
        s = merged_sites[i]
        if i == 1
            link_r = Index(1, "Link,l=$i")
            M[i] = ITensor(Id3, s', dag(s)) * ITensor([1.0], link_r)
        elseif i == N
            link_l = Index(1, "Link,l=$(i-1)")
            M[i] = ITensor(Id3, s', dag(s)) * ITensor([1.0], dag(link_l))
        else
            link_l = Index(1, "Link,l=$(i-1)")
            link_r = Index(1, "Link,l=$i")
            M[i] = ITensor(Id3, s', dag(s)) * ITensor([1.0], dag(link_l), link_r)
        end
    end

    return M
end
=#

export energy_density, center_energy_density
export projector_mpo, identity_mpo
export sz_operator, sx_operator, number_operator, total_number_operator
# Paper-based implementation (χ=2)
export energy_density_original, projector_mpo_original
export merge_mpo_pairs
export energy_density_merged, center_energy_density_merged
export identity_mpo_merged, projector_mpo_merged
# PNP / PNPNP energy densities
export energy_density_pnp_original, energy_density_pnpnp_original
export energy_density_pnp_merged, energy_density_pnpnp_merged
export center_energy_density_pnp_merged, center_energy_density_pnpnp_merged

 