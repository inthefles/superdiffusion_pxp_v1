# hilbert.jl
# Constrained Hilbert space utilities for PXP model
#
# The PXP model has the Rydberg blockade constraint: no two adjacent sites
# can both be in the excited state |↑⟩. This constrains the Hilbert space
# dimension to grow as Fibonacci numbers.

using ITensors

"""
    constrained_dim(N::Int) -> Int

Compute the dimension of the constrained Hilbert space for N sites
with open boundary conditions. The dimension follows the Fibonacci sequence:
    D(N) = F_{N+2}
where F_n is the n-th Fibonacci number (F_1 = F_2 = 1).

For periodic boundary conditions, the dimension is:
    D_PBC(N) = F_{N-1} + F_{N+1} = L_N (Lucas numbers)
but we use OBC here.
"""
function constrained_dim(N::Int)
    if N <= 0
        return 0
    elseif N == 1
        return 2  # |↓⟩, |↑⟩
    elseif N == 2
        return 3  # |↓↓⟩, |↓↑⟩, |↑↓⟩
    end
    # Fibonacci recurrence: F_{n+2} = F_{n+1} + F_n
    F_prev, F_curr = 2, 3  # F_3, F_4
    for _ in 3:N
        F_prev, F_curr = F_curr, F_prev + F_curr
    end
    return F_curr
end

"""
    is_valid_state(state::Vector{Int}) -> Bool

Check if a state configuration satisfies the Rydberg blockade constraint.
state[i] = 0 for |↓⟩ (ground state), state[i] = 1 for |↑⟩ (excited state).
Returns true if no two adjacent sites are both excited.
"""
function is_valid_state(state::Vector{Int})
    for i in 1:(length(state)-1)
        if state[i] == 1 && state[i+1] == 1
            return false
        end
    end
    return true
end

"""
    enumerate_valid_states(N::Int) -> Vector{Vector{Int}}

Enumerate all valid states in the constrained Hilbert space.
Useful for exact diagonalization benchmarks on small systems.
"""
function enumerate_valid_states(N::Int)
    valid_states = Vector{Int}[]
    for config in 0:(2^N - 1)
        state = [(config >> (i-1)) & 1 for i in 1:N]
        if is_valid_state(state)
            push!(valid_states, state)
        end
    end
    return valid_states
end

"""
    PXPSites(N::Int; conserve_qns::Bool=false) -> Vector{Index}

Create site indices for the PXP model using ITensors.
Uses standard S=1/2 spin sites with states |↓⟩ (Dn) and |↑⟩ (Up).

Note: The constraint is enforced dynamically through the Hamiltonian
structure, not through modified site indices.
"""
function PXPSites(N::Int; conserve_qns::Bool=false)
    return siteinds("S=1/2", N; conserve_qns=conserve_qns)
end

"""
    infinite_temperature_state(sites) -> MPO

Create the infinite temperature density matrix ρ ∝ I (identity) as an MPO.
This is the thermal equilibrium state at β = 0.

For computing ⟨h(0)h(t)⟩ = Tr[ρ h(0) h(t)] / Tr[ρ], we use ρ ∝ I
so ⟨h(0)h(t)⟩ = Tr[h(0) h(t)] / Tr[I].
"""
function infinite_temperature_state(sites)
    N = length(sites)
    # Identity operator as MPO: each local tensor is identity
    identity_mpo = MPO(sites, "Id")
    return identity_mpo
end

"""
    projector_state(sites) -> MPS

Create the all-down state |↓↓...↓⟩ as an MPS.
This is a valid state in the constrained Hilbert space.
"""
function all_down_state(sites)
    return MPS(sites, "Dn")
end

"""
    neel_state(sites) -> MPS

Create the Néel-like state |↓↑↓↑...⟩ as an MPS.
This is a valid state in the constrained Hilbert space.
"""
function neel_state(sites)
    N = length(sites)
    states = [isodd(i) ? "Dn" : "Up" for i in 1:N]
    return MPS(sites, states)
end

"""
    merged_local_dim() -> Int

Return the local dimension for merged (paired) sites.
After site merging, the constraint eliminates |↑↑⟩, leaving 3 states:
- |1⟩ = |↓↓⟩
- |2⟩ = |↑↓⟩
- |3⟩ = |↓↑⟩
"""
merged_local_dim() = 3

"""
    merged_site_indices(N::Int) -> Vector{Index}

Create site indices for the merged (paired) site representation.
N must be even. Returns N/2 indices, each with dimension 3.
"""
function merged_site_indices(N::Int)
    @assert iseven(N) "N must be even for site merging"
    N_merged = N ÷ 2
    return [Index(3, "Site,n=$i") for i in 1:N_merged]
end

export constrained_dim, is_valid_state, enumerate_valid_states
export PXPSites, infinite_temperature_state, all_down_state, neel_state
export merged_local_dim, merged_site_indices
