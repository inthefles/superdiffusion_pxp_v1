# tebd.jl
# Time-Evolving Block Decimation for PXP model
#
# Implements TEBD with site-merging approach to handle 3-site PXP terms.
# Uses 4th-order Trotter-Suzuki decomposition for high accuracy.
#
# Reference: Ljubotina et al., Phys. Rev. X 13, 011033 (2023)

using ITensors
using LinearAlgebra

#==============================================================================#
# Site Merging
#==============================================================================#

"""
    MergedSiteInfo

Stores information about the site-merging mapping.
After merging pairs (1,2), (3,4), ..., the constrained Hilbert space
on each merged site has dimension 3 (excluding |↑↑⟩).
"""
struct MergedSiteInfo
    N_original::Int      # Original number of sites (must be even)
    N_merged::Int        # Number of merged sites = N_original/2
    merged_sites::Vector{Index}  # Site indices for merged representation
end

"""
    create_merged_sites(N::Int) -> MergedSiteInfo

Create site indices for the merged representation.
Each merged site has dimension 3 corresponding to:
- state 1: |↓↓⟩
- state 2: |↑↓⟩
- state 3: |↓↑⟩
"""
function create_merged_sites(N::Int)
    @assert iseven(N) "N must be even for site merging"
    N_merged = N ÷ 2
    merged_sites = [Index(3, "Site,n=$i") for i in 1:N_merged]
    return MergedSiteInfo(N, N_merged, merged_sites)
end

"""
State encoding for merged sites:
|↓↓⟩ → 1
|↑↓⟩ → 2
|↓↑⟩ → 3
"""
const MERGED_DD = 1  # |↓↓⟩
const MERGED_UD = 2  # |↑↓⟩
const MERGED_DU = 3  # |↓↑⟩

#==============================================================================#
# Gate Construction
#==============================================================================#

"""
    pxp_local_hamiltonian(; Ω=1.0, λ=0.0, δ=0.0, ξ=0.0) -> Matrix

Construct the local Hamiltonian matrix for a pair of adjacent merged sites.
This handles the 3-site PXP term which becomes a 2-site term after merging.

For merged sites i and i+1 (corresponding to original sites 2i-1, 2i, 2i+1, 2i+2):
- The PXP terms P_{2i-1} σ^x_{2i} P_{2i+1} and P_{2i} σ^x_{2i+1} P_{2i+2}
  connect states on these merged sites.

Returns a 9×9 matrix acting on the tensor product of two merged sites.
"""
function pxp_local_hamiltonian(; Ω=1.0, λ=0.0, δ=0.0, ξ=0.0)
    # Basis states for two merged sites: (merged_i, merged_{i+1})
    # Each takes values 1,2,3 corresponding to DD, UD, DU
    # Total: 9 states, but some are further constrained

    # Local dimension
    d = 3  # |DD⟩, |UD⟩, |DU⟩

    # Hamiltonian matrix
    H = zeros(ComplexF64, d^2, d^2)

    # State indexing: (s1, s2) → (s1-1)*d + s2
    idx(s1, s2) = (s1-1)*d + s2

    # PXP term structure:
    # P_{i-1} σ^x_i P_{i+1} flips site i if neighbors are down
    # After merging, we need to track which original site is flipped

    # Consider merged sites A and B
    # A = (a1, a2), B = (b1, b2) where a,b ∈ {↓,↑}
    # Original sites: a1=site 1, a2=site 2, b1=site 3, b2=site 4

    # Term 1: P_1 σ^x_2 P_3 (flip site 2, check sites 1 and 3)
    # In merged: A=(s_a1, s_a2), B=(s_b1, s_b2)
    # Need: a1=↓, b1=↓, flip a2
    # States where a1=↓: A∈{DD, DU} = {1, 3}
    # States where b1=↓: B∈{DD, DU} = {1, 3}

    # A=DD (1) → flip a2 → A=DU (3), if b1=↓ (B∈{1,3})
    # A=DU (3) → flip a2 → A=DD (1), if b1=↓ (B∈{1,3})
    for B in [MERGED_DD, MERGED_DU]
        # DD ↔ DU transitions
        H[idx(MERGED_DD, B), idx(MERGED_DU, B)] += Ω
        H[idx(MERGED_DU, B), idx(MERGED_DD, B)] += Ω
    end

    # Term 2: P_2 σ^x_3 P_4 (flip site 3=b1, check sites 2=a2 and 4=b2)
    # Need: a2=↓, b2=↓
    # States where a2=↓: A∈{DD, UD} = {1, 2}
    # States where b2=↓: B∈{DD, UD} = {1, 2}

    # B=DD → flip b1 → B=UD (if allowed, but UD then has b1=↑)
    # Wait, B=DD means (b1=↓, b2=↓), flipping b1 gives (b1=↑, b2=↓) = UD
    for A in [MERGED_DD, MERGED_UD]
        # DD ↔ UD transitions for B
        H[idx(A, MERGED_DD), idx(A, MERGED_UD)] += Ω
        H[idx(A, MERGED_UD), idx(A, MERGED_DD)] += Ω
    end

    # Additional constraint: boundary between merged sites
    # States (A, B) with A=DU (a2=↑) and B=UD (b1=↑) are forbidden!
    # This is the |↑↑⟩ constraint across the boundary.
    # However, in our local Hamiltonian on 2 merged sites, we should
    # NOT include matrix elements involving this forbidden state.

    # Actually, let's reconsider. The state (DU, UD) means:
    # A = DU = (a1=↓, a2=↑), B = UD = (b1=↑, b2=↓)
    # Original: |↓↑↑↓⟩ which has adjacent ↑↑ at sites 2,3 - FORBIDDEN!

    # So we should zero out rows/columns for this state
    forbidden_idx = idx(MERGED_DU, MERGED_UD)
    H[forbidden_idx, :] .= 0
    H[:, forbidden_idx] .= 0

    # PXPZ deformation: add λ P_{i-1} σ^z_i P_{i+1}
    if abs(λ) > 1e-14
        # This adds diagonal terms based on σ^z of the middle site
        # Similar analysis needed...
        # For now, we'll add this as a simpler diagonal correction
        # σ^z = +1 for ↓, -1 for ↑

        # Term for site 2 (a2): P_1 σ^z_2 P_3, need a1=↓, b1=↓
        for A in [MERGED_DD, MERGED_DU]  # a1=↓
            for B in [MERGED_DD, MERGED_DU]  # b1=↓
                if idx(A, B) != forbidden_idx
                    # σ^z on a2: +1 if a2=↓ (A=DD), -1 if a2=↑ (A=DU)
                    sz_a2 = (A == MERGED_DD) ? 1.0 : -1.0
                    H[idx(A, B), idx(A, B)] += λ * sz_a2
                end
            end
        end

        # Term for site 3 (b1): P_2 σ^z_3 P_4, need a2=↓, b2=↓
        for A in [MERGED_DD, MERGED_UD]  # a2=↓
            for B in [MERGED_DD, MERGED_UD]  # b2=↓
                if idx(A, B) != forbidden_idx
                    # σ^z on b1: +1 if b1=↓ (B=DD), -1 if b1=↑ (B=UD)
                    sz_b1 = (B == MERGED_DD) ? 1.0 : -1.0
                    H[idx(A, B), idx(A, B)] += λ * sz_b1
                end
            end
        end
    end

    # PNP deformation: δ Σ_i n_i (chemical potential)
    if abs(δ) > 1e-14
        # n_i = (1 + σ^z_i)/2, counts excitations
        # For each merged site, count excitations in both sub-sites
        for s1 in 1:d
            for s2 in 1:d
                if idx(s1, s2) != forbidden_idx
                    # Count excitations in s1
                    n1 = (s1 == MERGED_UD) ? 1 : ((s1 == MERGED_DU) ? 1 : 0)
                    # Actually: DD=0, UD=1 (first up), DU=1 (second up)
                    n_A = (s1 == MERGED_DD) ? 0 : 1
                    n_B = (s2 == MERGED_DD) ? 0 : 1
                    H[idx(s1, s2), idx(s1, s2)] += δ * (n_A + n_B)
                end
            end
        end
    end

    return H
end

"""
    make_trotter_gates_merged(info::MergedSiteInfo, dt::Float64;
                              Ω=1.0, λ=0.0, δ=0.0, ξ=0.0, order=2) -> Vector{ITensor}

Construct Trotter gates for time evolution on merged sites.

For 2nd order Trotter:
    U(dt) ≈ U_odd(dt/2) U_even(dt) U_odd(dt/2)

For 4th order Trotter:
    U(dt) ≈ U_2(p*dt)² U_2((1-4p)*dt) U_2(p*dt)²
    where p = 1/(4 - 4^(1/3)) ≈ 0.4145
"""
function make_trotter_gates_merged(info::MergedSiteInfo, dt::Float64;
                                   Ω=1.0, λ=0.0, δ=0.0, ξ=0.0, order=2)
    sites = info.merged_sites
    N = info.N_merged

    # Local Hamiltonian for adjacent merged sites
    H_local = pxp_local_hamiltonian(; Ω=Ω, λ=λ, δ=δ, ξ=ξ)

    # Exponentiate to get local time evolution
    function make_gate(τ)
        U = exp(-im * τ * H_local)
        return U
    end

    if order == 2
        # 2nd order: U_odd(dt/2) U_even(dt) U_odd(dt/2)
        return _make_second_order_gates(sites, N, make_gate, dt)
    elseif order == 4
        # 4th order composition
        p = 1.0 / (4.0 - 4.0^(1/3))
        gates = ITensor[]

        # U_2(p*dt) twice
        gates_p = _make_second_order_gates(sites, N, make_gate, p*dt)
        append!(gates, gates_p)
        append!(gates, gates_p)

        # U_2((1-4p)*dt)
        gates_mid = _make_second_order_gates(sites, N, make_gate, (1-4p)*dt)
        append!(gates, gates_mid)

        # U_2(p*dt) twice
        append!(gates, gates_p)
        append!(gates, gates_p)

        return gates
    else
        error("Trotter order must be 2 or 4")
    end
end

function _make_second_order_gates(sites, N, make_gate, dt)
    gates = ITensor[]
    d = 3  # merged site dimension

    # Odd bonds: (1,2), (3,4), ... with half time step
    for i in 1:2:(N-1)
        U = make_gate(dt/2)
        s1, s2 = sites[i], sites[i+1]
        gate = ITensor(U, s1', s2', dag(s1), dag(s2))
        push!(gates, gate)
    end

    # Even bonds: (2,3), (4,5), ... with full time step
    for i in 2:2:(N-1)
        U = make_gate(dt)
        s1, s2 = sites[i], sites[i+1]
        gate = ITensor(U, s1', s2', dag(s1), dag(s2))
        push!(gates, gate)
    end

    # Odd bonds again with half time step
    for i in 1:2:(N-1)
        U = make_gate(dt/2)
        s1, s2 = sites[i], sites[i+1]
        gate = ITensor(U, s1', s2', dag(s1), dag(s2))
        push!(gates, gate)
    end

    return gates
end

#==============================================================================#
# Gate Application to MPO (Site-Merging Approach)
#==============================================================================#

"""
    apply_2site_gate_to_merged_MPO!(M::MPO, gate::ITensor, i::Int;
                                     maxdim::Int=256, cutoff::Float64=1e-10)

Apply a 2-site gate acting on merged sites i and i+1 to an MPO.
The MPO structure is restored via SVD after the gate application.

For Heisenberg evolution O(t) = U† O(0) U:
- The gate U acts on the ket (unprimed) indices
- The gate U† acts on the bra (primed) indices

This function contracts the gate with the two adjacent MPO tensors,
then uses SVD to restore the canonical two-tensor form.

# Arguments
- `M::MPO`: The MPO to modify (modified in place)
- `gate::ITensor`: The 2-site unitary gate
- `i::Int`: The left site index (gate acts on sites i and i+1)
- `maxdim::Int`: Maximum bond dimension after SVD truncation
- `cutoff::Float64`: SVD singular value cutoff

# Returns
The modified MPO (same object as input, modified in place)
"""
function apply_2site_gate_to_merged_MPO!(M::MPO, gate::ITensor, i::Int;
                                          maxdim::Int=256, cutoff::Float64=1e-10)
    j = i + 1
    N = length(M)
    @assert 1 <= i < N "Gate sites ($i, $j) must be valid adjacent sites"

    # Contract the two adjacent MPO tensors
    M_combined = M[i] * M[j]

    # For Heisenberg evolution: O(t) = U† O U
    # Apply U to ket indices (unprimed site indices of M)
    # Apply U† to bra indices (primed site indices of M)
    #
    # The gate has structure: gate = ITensor(U, s1', s2', dag(s1), dag(s2))
    # where s1', s2' are primed (output) and dag(s1), dag(s2) are unprimed (input)
    #
    # For MPO: M has indices (s, s') at each site
    # We need: M_new[s', s] = sum_{s1, s1'} U†[s', s1'] M[s1', s1] U[s1, s]
    #
    # Step 1: Apply U from the right (ket side)
    # Contract gate's unprimed input indices with M's unprimed indices
    M_temp = M_combined * gate

    # Step 2: Apply U† from the left (bra side)
    # Create U† by conjugating and swapping prime levels
    gate_dag = dag(gate)
    # The conjugate gate has primed indices as inputs and unprimed as outputs
    # We need to swap so it contracts with M's primed indices
    # gate_dag: dag(s1)', dag(s2)', s1, s2 -> need to contract s1, s2 with M's s1', s2'
    # Use replaceprime to shift indices appropriately
    gate_dag = replaceprime(gate_dag, 1 => 2)  # s' -> s''
    gate_dag = replaceprime(gate_dag, 0 => 1)  # s -> s'
    gate_dag = replaceprime(gate_dag, 2 => 0)  # s'' -> s

    M_new = M_temp * gate_dag

    # Remove any prime level 2 indices that might remain
    M_new = replaceprime(M_new, 2 => 1)

    # SVD to restore MPO form
    # Collect indices that should go to the left tensor (site i)
    # This includes: site indices at i (both primed and unprimed) and left link
    left_inds = Index[]

    # Get site indices at position i
    s_i = siteind(M, i)
    push!(left_inds, s_i)
    if hasind(M_new, s_i')
        push!(left_inds, s_i')
    end

    # Get left link index if not at left boundary
    if i > 1
        l_left = linkind(M, i-1)
        if hasind(M_new, l_left)
            push!(left_inds, l_left)
        end
    end

    # Perform SVD
    U, S, V = svd(M_new, Tuple(left_inds); maxdim=maxdim, cutoff=cutoff)

    # Assign back to MPO
    M[i] = U
    M[j] = S * V

    return M
end

"""
    tebd_step_merged!(M::MPO, gates::Vector{ITensor}, merged_sites::Vector{Index};
                      maxdim::Int=256, cutoff::Float64=1e-10)

Apply all gates in a single Trotter step to a merged-site MPO.

The gates should be created by `make_trotter_gates_merged()` and act on
pairs of adjacent merged sites.

# Arguments
- `M::MPO`: The MPO on merged sites (modified in place)
- `gates::Vector{ITensor}`: The Trotter gates
- `merged_sites::Vector{Index}`: The site indices for merged sites
- `maxdim::Int`: Maximum bond dimension
- `cutoff::Float64`: SVD cutoff

# Returns
The modified MPO
"""
function tebd_step_merged!(M::MPO, gates::Vector{ITensor}, merged_sites::Vector{Index};
                           maxdim::Int=256, cutoff::Float64=1e-10)
    for gate in gates
        # Find which merged sites this gate acts on by checking indices
        gate_site_indices = Int[]

        for (idx, s) in enumerate(merged_sites)
            # Check if gate has this site index (primed or unprimed)
            if hasind(gate, s) || hasind(gate, s')
                push!(gate_site_indices, idx)
            end
        end

        if length(gate_site_indices) == 2
            i = minimum(gate_site_indices)
            # Apply the 2-site gate
            apply_2site_gate_to_merged_MPO!(M, gate, i; maxdim=maxdim, cutoff=cutoff)
        elseif length(gate_site_indices) == 1
            # Single-site gate (if any) - apply directly
            idx = gate_site_indices[1]
            M[idx] = M[idx] * gate
            # Prime management for single-site gates
            M[idx] = replaceprime(M[idx], 2 => 1)
        end
    end

    return M
end

#==============================================================================#
# DEPRECATED: Standard TEBD on original sites (3-site gate approach)
# Using site-merging approach instead (see above)
#==============================================================================#

#=

"""
    make_trotter_gates(sites, dt::Float64; Ω=1.0, λ=0.0, δ=0.0, order=4) -> Vector{ITensor}

Construct Trotter gates for PXP evolution on original (non-merged) sites.
Uses a 3-site gate decomposition for the PXP term.

This is an alternative to the site-merging approach.
"""
function make_trotter_gates(sites, dt::Float64; Ω=1.0, λ=0.0, δ=0.0, order=4)
    N = length(sites)

    if order == 4
        p = 1.0 / (4.0 - 4.0^(1/3))
        gates = ITensor[]
        append!(gates, _make_2nd_order_pxp_gates(sites, N, p*dt; Ω=Ω, λ=λ, δ=δ))
        append!(gates, _make_2nd_order_pxp_gates(sites, N, p*dt; Ω=Ω, λ=λ, δ=δ))
        append!(gates, _make_2nd_order_pxp_gates(sites, N, (1-4p)*dt; Ω=Ω, λ=λ, δ=δ))
        append!(gates, _make_2nd_order_pxp_gates(sites, N, p*dt; Ω=Ω, λ=λ, δ=δ))
        append!(gates, _make_2nd_order_pxp_gates(sites, N, p*dt; Ω=Ω, λ=λ, δ=δ))
        return gates
    else
        return _make_2nd_order_pxp_gates(sites, N, dt; Ω=Ω, λ=λ, δ=δ)
    end
end

function _make_2nd_order_pxp_gates(sites, N, dt; Ω=1.0, λ=0.0, δ=0.0)
    gates = ITensor[]

    # PXP has 3-site terms, so we use 3-site gates
    # Split into: odd sites (1,3,5,...) and even sites (2,4,6,...)

    # Build local 3-site Hamiltonian matrix
    # Basis: |s_{i-1} s_i s_{i+1}⟩ where s ∈ {↓, ↑} = {1, 2}
    function pxp_3site_hamiltonian(boundary_left, boundary_right)
        d = 2  # local dimension
        H = zeros(ComplexF64, d^3, d^3)

        # Indexing: |s1 s2 s3⟩ → (s1-1)*d^2 + (s2-1)*d + s3
        idx(s1, s2, s3) = (s1-1)*d^2 + (s2-1)*d + s3

        # P_{i-1} σ^x_i P_{i+1}: flip s2 if s1=↓ and s3=↓
        # s=1 means ↓, s=2 means ↑
        if boundary_left && boundary_right
            # Both boundaries: just σ^x_i
            for s1 in 1:d, s3 in 1:d
                H[idx(s1, 1, s3), idx(s1, 2, s3)] += Ω
                H[idx(s1, 2, s3), idx(s1, 1, s3)] += Ω
            end
        elseif boundary_left
            # Left boundary: σ^x_i P_{i+1}
            for s1 in 1:d
                # s3 must be ↓ (s3=1) for projector
                H[idx(s1, 1, 1), idx(s1, 2, 1)] += Ω
                H[idx(s1, 2, 1), idx(s1, 1, 1)] += Ω
            end
        elseif boundary_right
            # Right boundary: P_{i-1} σ^x_i
            for s3 in 1:d
                # s1 must be ↓ (s1=1) for projector
                H[idx(1, 1, s3), idx(1, 2, s3)] += Ω
                H[idx(1, 2, s3), idx(1, 1, s3)] += Ω
            end
        else
            # Bulk: P_{i-1} σ^x_i P_{i+1}
            # Both s1=1 and s3=1
            H[idx(1, 1, 1), idx(1, 2, 1)] += Ω
            H[idx(1, 2, 1), idx(1, 1, 1)] += Ω
        end

        # Add λ P σ^z P term if needed
        if abs(λ) > 1e-14
            # Similar structure but with σ^z (diagonal)
            for s1 in 1:d, s2 in 1:d, s3 in 1:d
                proj = 1.0
                if !boundary_left && s1 != 1
                    proj = 0.0
                end
                if !boundary_right && s3 != 1
                    proj = 0.0
                end
                if proj > 0
                    sz = (s2 == 1) ? 1.0 : -1.0  # ↓=+1, ↑=-1
                    H[idx(s1, s2, s3), idx(s1, s2, s3)] += λ * sz * proj
                end
            end
        end

        return H
    end

    # Odd sites: i = 1, 3, 5, ... (with dt/2)
    for i in 1:2:N
        boundary_left = (i == 1)
        boundary_right = (i == N)

        if i == 1 && N >= 2
            # 2-site gate for boundary
            H_2site = _pxp_2site_left_boundary(Ω, λ)
            U = exp(-im * (dt/2) * H_2site)
            s1, s2 = sites[1], sites[2]
            gate = ITensor(U, s1', s2', dag(s1), dag(s2))
            push!(gates, gate)
        elseif i == N && N >= 2
            # 2-site gate for right boundary
            H_2site = _pxp_2site_right_boundary(Ω, λ)
            U = exp(-im * (dt/2) * H_2site)
            s1, s2 = sites[N-1], sites[N]
            gate = ITensor(U, s1', s2', dag(s1), dag(s2))
            push!(gates, gate)
        elseif i > 1 && i < N
            # Bulk 3-site gate
            H_3site = pxp_3site_hamiltonian(false, false)
            U = exp(-im * (dt/2) * H_3site)
            s1, s2, s3 = sites[i-1], sites[i], sites[i+1]
            gate = ITensor(U, s1', s2', s3', dag(s1), dag(s2), dag(s3))
            push!(gates, gate)
        end
    end

    # Even sites: i = 2, 4, 6, ... (with full dt)
    for i in 2:2:N
        if i > 1 && i < N
            H_3site = pxp_3site_hamiltonian(false, false)
            U = exp(-im * dt * H_3site)
            s1, s2, s3 = sites[i-1], sites[i], sites[i+1]
            gate = ITensor(U, s1', s2', s3', dag(s1), dag(s2), dag(s3))
            push!(gates, gate)
        end
    end

    # Odd sites again with dt/2
    for i in 1:2:N
        if i == 1 && N >= 2
            H_2site = _pxp_2site_left_boundary(Ω, λ)
            U = exp(-im * (dt/2) * H_2site)
            s1, s2 = sites[1], sites[2]
            gate = ITensor(U, s1', s2', dag(s1), dag(s2))
            push!(gates, gate)
        elseif i == N && N >= 2
            H_2site = _pxp_2site_right_boundary(Ω, λ)
            U = exp(-im * (dt/2) * H_2site)
            s1, s2 = sites[N-1], sites[N]
            gate = ITensor(U, s1', s2', dag(s1), dag(s2))
            push!(gates, gate)
        elseif i > 1 && i < N
            H_3site = pxp_3site_hamiltonian(false, false)
            U = exp(-im * (dt/2) * H_3site)
            s1, s2, s3 = sites[i-1], sites[i], sites[i+1]
            gate = ITensor(U, s1', s2', s3', dag(s1), dag(s2), dag(s3))
            push!(gates, gate)
        end
    end

    return gates
end

function _pxp_2site_left_boundary(Ω, λ)
    # Site 1: σ^x_1 P_2
    d = 2
    H = zeros(ComplexF64, d^2, d^2)
    idx(s1, s2) = (s1-1)*d + s2

    # Flip s1 if s2=↓
    H[idx(1, 1), idx(2, 1)] += Ω
    H[idx(2, 1), idx(1, 1)] += Ω

    # λ term
    if abs(λ) > 1e-14
        for s1 in 1:d
            sz = (s1 == 1) ? 1.0 : -1.0
            H[idx(s1, 1), idx(s1, 1)] += λ * sz
        end
    end

    return H
end

function _pxp_2site_right_boundary(Ω, λ)
    # Site N: P_{N-1} σ^x_N
    d = 2
    H = zeros(ComplexF64, d^2, d^2)
    idx(s1, s2) = (s1-1)*d + s2

    # Flip s2 if s1=↓
    H[idx(1, 1), idx(1, 2)] += Ω
    H[idx(1, 2), idx(1, 1)] += Ω

    # λ term
    if abs(λ) > 1e-14
        for s2 in 1:d
            sz = (s2 == 1) ? 1.0 : -1.0
            H[idx(1, s2), idx(1, s2)] += λ * sz
        end
    end

    return H
end
=#

#==============================================================================#
# MPO Time Evolution (Heisenberg picture) - Site-Merging Approach
#==============================================================================#

"""
    evolve_mpo_merged(O::MPO, info::MergedSiteInfo, t::Float64;
                      dt=0.05, maxdim=128, cutoff=1e-10, order=4,
                      Ω=1.0, λ=0.0, δ=0.0, ξ=0.0) -> MPO

Evolve an operator O in the Heisenberg picture using the site-merging approach:
    O(t) = e^{iHt} O e^{-iHt}

The operator O must be defined on merged sites (dimension 3 per site).
Uses TEBD with specified Trotter order (2 or 4).

# Arguments
- `O::MPO`: Initial operator on merged sites
- `info::MergedSiteInfo`: Site-merging information
- `t::Float64`: Total evolution time
- `dt::Float64`: Time step size
- `maxdim::Int`: Maximum bond dimension
- `cutoff::Float64`: SVD cutoff
- `order::Int`: Trotter order (2 or 4)
- `Ω, λ, δ, ξ`: PXP Hamiltonian parameters

# Returns
The time-evolved operator O(t) as an MPO
"""
function evolve_mpo_merged(O::MPO, info::MergedSiteInfo, t::Float64;
                           dt=0.05, maxdim=128, cutoff=1e-10, order=4,
                           Ω=1.0, λ=0.0, δ=0.0, ξ=0.0)
    merged_sites = info.merged_sites

    # Compute number of time steps
    nsteps = max(1, round(Int, abs(t) / dt))
    actual_dt = t / nsteps

    # Make Trotter gates on merged sites for one time step
    gates = make_trotter_gates_merged(info, actual_dt; Ω=Ω, λ=λ, δ=δ, ξ=ξ, order=order)

    O_t = copy(O)

    for step in 1:nsteps
        # Apply all gates in this Trotter step
        tebd_step_merged!(O_t, gates, merged_sites; maxdim=maxdim, cutoff=cutoff)
    end

    return O_t
end

"""
    evolve_mpo(O::MPO, sites, t::Float64; ...) -> MPO

Convenience wrapper for evolve_mpo_merged that creates the merged site info.
The input MPO O must already be on merged sites.

See `evolve_mpo_merged` for full documentation.
"""
function evolve_mpo(O::MPO, sites::Vector{<:Index}, t::Float64;
                    dt=0.05, maxdim=128, cutoff=1e-10, order=4,
                    Ω=1.0, λ=0.0, δ=0.0, ξ=0.0)
    # Assume sites are already merged sites
    N_merged = length(sites)
    N_original = 2 * N_merged
    info = MergedSiteInfo(N_original, N_merged, sites)

    return evolve_mpo_merged(O, info, t;
                             dt=dt, maxdim=maxdim, cutoff=cutoff, order=order,
                             Ω=Ω, λ=λ, δ=δ, ξ=ξ)
end

#==============================================================================#
# High-level evolution interface
#==============================================================================#

"""
    TEBDParams

Parameters for TEBD simulation.
"""
struct TEBDParams
    dt::Float64          # Time step
    maxdim::Int          # Maximum bond dimension
    cutoff::Float64      # SVD cutoff
    order::Int           # Trotter order (2 or 4)
    Ω::Float64           # Rabi frequency
    λ::Float64           # PXPZ deformation
    δ::Float64           # Chemical potential
    ξ::Float64           # PNPNP deformation
end

TEBDParams(; dt=0.05, maxdim=128, cutoff=1e-10, order=4,
           Ω=1.0, λ=0.0, δ=0.0, ξ=0.0) =
    TEBDParams(dt, maxdim, cutoff, order, Ω, λ, δ, ξ)

"""
    run_tebd_evolution(info::MergedSiteInfo, h0::MPO, tmax::Float64, params::TEBDParams;
                       save_times=nothing) -> (times, mpos)

Run TEBD evolution of operator h0 on merged sites and return time-evolved operators.

# Arguments
- `info::MergedSiteInfo`: Site-merging information
- `h0::MPO`: Initial operator on merged sites
- `tmax::Float64`: Maximum evolution time
- `params::TEBDParams`: TEBD parameters
- `save_times`: Optional array of times at which to save snapshots

# Returns
- `times::Vector{Float64}`: Times at which operators were saved
- `mpos::Vector{MPO}`: Time-evolved operators at each saved time
"""
function run_tebd_evolution(info::MergedSiteInfo, h0::MPO, tmax::Float64, params::TEBDParams;
                            save_times=nothing)
    if isnothing(save_times)
        nsteps = round(Int, tmax / params.dt)
        save_times = range(0, tmax, length=nsteps+1)
    end

    times = Float64[]
    mpos = MPO[]

    push!(times, 0.0)
    push!(mpos, copy(h0))

    current_mpo = copy(h0)
    current_t = 0.0

    for target_t in save_times[2:end]
        evolution_time = target_t - current_t
        if evolution_time > 0
            current_mpo = evolve_mpo_merged(current_mpo, info, evolution_time;
                                            dt=params.dt, maxdim=params.maxdim,
                                            cutoff=params.cutoff, order=params.order,
                                            Ω=params.Ω, λ=params.λ, δ=params.δ,
                                            ξ=params.ξ)
            current_t = target_t
        end
        push!(times, current_t)
        push!(mpos, copy(current_mpo))
    end

    return times, mpos
end

export MergedSiteInfo, create_merged_sites
export pxp_local_hamiltonian, make_trotter_gates_merged
export apply_2site_gate_to_merged_MPO!, tebd_step_merged!
export evolve_mpo_merged, evolve_mpo
export TEBDParams, run_tebd_evolution
