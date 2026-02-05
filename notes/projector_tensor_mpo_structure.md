# MPO Structure of `_projector_tensor_merged`

This document explains how the MPO tensors are constructed in the `_projector_tensor_merged` function from `src/operators.jl`.

## Setup

### Physical States (dimension $d=3$)

| Index | Symbol | State | Meaning |
|-------|--------|-------|---------|
| 1 | DD | $\|\downarrow\downarrow\rangle$ | Both spins down |
| 2 | UD | $\|\uparrow\downarrow\rangle$ | Left up, right down |
| 3 | DU | $\|\downarrow\uparrow\rangle$ | Left down, right up |

Note: $\|\uparrow\uparrow\rangle$ is forbidden by the Rydberg constraint.

### Virtual Index Encoding (dimension $\chi=4$)

The virtual bond encodes two pieces of information:
- **proj_state**: Whether previous site was DU (danger) or not (safe)
- **op_stage**: Whether the operator has been applied (after) or not (before)

```
lin(proj_state, op_stage) = proj_state + 2*op_stage + 1
```

| Index | proj_state | op_stage | Meaning |
|-------|------------|----------|---------|
| 1 | 0 (safe) | 0 (before) | Safe, operator not yet applied |
| 2 | 1 (danger) | 0 (before) | Previous site was DU, operator not yet applied |
| 3 | 0 (safe) | 1 (after) | Safe, operator has been applied |
| 4 | 1 (danger) | 1 (after) | Previous site was DU, operator has been applied |

### Tensor Shape

$$W[\text{out}, \text{in}, \text{left bond}, \text{right bond}] \equiv W^{s's}_{\alpha\beta}$$

where:
- $s'$ = output physical index (bra)
- $s$ = input physical index (ket)
- $\alpha$ = left virtual bond
- $\beta$ = right virtual bond

---

## Case 1: Left Boundary (`is_left_boundary=true`)

Bond dimensions: $\chi_L = 1$, $\chi_R = 4$

### For `stage = :before` (op_stage = 0)

| State | proj_out | Tensor element |
|-------|----------|----------------|
| DD | 0 | $W[1,1,1,1] = 1$ |
| UD | 0 | $W[2,2,1,1] = 1$ |
| DU | 1 | $W[3,3,1,2] = 1$ |

As a column vector (indexed by right bond), with each entry being a $3\times 3$ physical operator:

$$W^{\text{left}}_{\text{before}} = \begin{pmatrix} |DD\rangle\langle DD| + |UD\rangle\langle UD| \\ |DU\rangle\langle DU| \\ 0 \\ 0 \end{pmatrix}$$

**Interpretation**:
- DD and UD output to "safe, before" (index 1)
- DU outputs to "danger, before" (index 2)
- No output to "after" indices (3, 4)

### For `stage = :after` (op_stage = 1)

| State | proj_out | Tensor element |
|-------|----------|----------------|
| DD | 0 | $W[1,1,1,3] = 1$ |
| UD | 0 | $W[2,2,1,3] = 1$ |
| DU | 1 | $W[3,3,1,4] = 1$ |

$$W^{\text{left}}_{\text{after}} = \begin{pmatrix} 0 \\ 0 \\ |DD\rangle\langle DD| + |UD\rangle\langle UD| \\ |DU\rangle\langle DU| \end{pmatrix}$$

**Interpretation**: Same logic but outputs to "after" indices (3, 4).

---

## Case 2: Right Boundary (`is_right_boundary=true`)

Bond dimensions: $\chi_L = 4$, $\chi_R = 1$

### For `stage = :before` (op_stage = 0)

| proj_in | State | Allowed? | Tensor element |
|---------|-------|----------|----------------|
| 0 | DD | ✓ | $W[1,1,1,1] = 1$ |
| 0 | UD | ✓ | $W[2,2,1,1] = 1$ |
| 0 | DU | ✓ | $W[3,3,1,1] = 1$ |
| 1 | DD | ✓ | $W[1,1,2,1] = 1$ |
| 1 | UD | ✗ | **forbidden** (DU followed by UD) |
| 1 | DU | ✓ | $W[3,3,2,1] = 1$ |

As a row vector (indexed by left bond):

$$W^{\text{right}}_{\text{before}} = \begin{pmatrix} \mathbb{1}_3 & |DD\rangle\langle DD| + |DU\rangle\langle DU| & 0 & 0 \end{pmatrix}$$

where $\mathbb{1}_3 = |DD\rangle\langle DD| + |UD\rangle\langle UD| + |DU\rangle\langle DU|$ is the identity on the 3-dimensional merged site.

**Interpretation**:
- From "safe, before" (index 1): all states pass
- From "danger, before" (index 2): UD is forbidden (would create $|DU\rangle|UD\rangle = |\downarrow\uparrow\uparrow\downarrow\rangle$)
- From "after" indices (3, 4): no input expected

### For `stage = :after` (op_stage = 1)

$$W^{\text{right}}_{\text{after}} = \begin{pmatrix} 0 & 0 & \mathbb{1}_3 & |DD\rangle\langle DD| + |DU\rangle\langle DU| \end{pmatrix}$$

**Interpretation**: Same logic but accepts from "after" indices (3, 4).

---

## Case 3: Bulk (Neither Boundary)

Bond dimensions: $\chi_L = 4$, $\chi_R = 4$

### For `stage = :before` (op_stage = 0)

Non-zero tensor elements:

| proj_in | State | proj_out | Element |
|---------|-------|----------|---------|
| 0 | DD | 0 | $W[1,1,1,1] = 1$ |
| 0 | UD | 0 | $W[2,2,1,1] = 1$ |
| 0 | DU | 1 | $W[3,3,1,2] = 1$ |
| 1 | DD | 0 | $W[1,1,2,1] = 1$ |
| 1 | UD | — | **forbidden** |
| 1 | DU | 1 | $W[3,3,2,2] = 1$ |

As a $4 \times 4$ matrix where each entry is a $3 \times 3$ physical operator:

$$W^{\text{bulk}}_{\text{before}} = \begin{pmatrix}
|DD\rangle\langle DD| + |UD\rangle\langle UD| & |DU\rangle\langle DU| & 0 & 0 \\
|DD\rangle\langle DD| & |DU\rangle\langle DU| & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0
\end{pmatrix}$$

### For `stage = :after` (op_stage = 1)

$$W^{\text{bulk}}_{\text{after}} = \begin{pmatrix}
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & |DD\rangle\langle DD| + |UD\rangle\langle UD| & |DU\rangle\langle DU| \\
0 & 0 & |DD\rangle\langle DD| & |DU\rangle\langle DU|
\end{pmatrix}$$

---

## Visual Summary: Bulk Tensor Block Structure

For `:before` stage, the $4\times 4$ bond matrix has a $2\times 2$ active block in the upper-left:

```
              Right bond →
              safe,bef  danger,bef  safe,aft  danger,aft
             ┌─────────┬──────────┬─────────┬──────────┐
Left         │ DD+UD   │    DU    │    0    │    0     │  safe,before
bond         ├─────────┼──────────┼─────────┼──────────┤
↓            │   DD    │    DU    │    0    │    0     │  danger,before
             ├─────────┼──────────┼─────────┼──────────┤
             │    0    │    0     │    0    │    0     │  safe,after
             ├─────────┼──────────┼─────────┼──────────┤
             │    0    │    0     │    0    │    0     │  danger,after
             └─────────┴──────────┴─────────┴──────────┘
```

For `:after` stage, the active block is in the lower-right:

```
              Right bond →
              safe,bef  danger,bef  safe,aft  danger,aft
             ┌─────────┬──────────┬─────────┬──────────┐
Left         │    0    │    0     │    0    │    0     │  safe,before
bond         ├─────────┼──────────┼─────────┼──────────┤
↓            │    0    │    0     │    0    │    0     │  danger,before
             ├─────────┼──────────┼─────────┼──────────┤
             │    0    │    0     │ DD+UD   │    DU    │  safe,after
             ├─────────┼──────────┼─────────┼──────────┤
             │    0    │    0     │   DD    │    DU    │  danger,after
             └─────────┴──────────┴─────────┴──────────┘
```

---

## Key Observations

1. **Diagonal in physical indices**: The projector doesn't change the physical state, it only filters configurations. All non-zero elements have the form $|s\rangle\langle s|$.

2. **Block diagonal in op_stage**: The "before" and "after" subspaces don't mix within projector tensors. This ensures the operator is applied exactly once.

3. **Projector state transitions**:
   - DD, UD → safe (proj_out = 0): Right spin is $\downarrow$, safe for next site
   - DU → danger (proj_out = 1): Right spin is $\uparrow$, next site cannot be UD

4. **Forbidden transition**: When proj_in = 1 (danger) and current state is UD, the matrix element is zero. This kills the path corresponding to $|DU\rangle|UD\rangle = |\downarrow\uparrow\uparrow\downarrow\rangle$, which violates the Rydberg blockade.

5. **Finite automaton structure**: The virtual bond implements a finite automaton that tracks whether we're in a "danger" state. The automaton has two states per op_stage:
   - safe: Previous merged site ended with $\downarrow$ (DD or UD)
   - danger: Previous merged site ended with $\uparrow$ (DU)

---

## Connection to Paper (Ljubotina et al.)

This implementation follows Appendix A of Phys. Rev. X 13, 011033 (2023):

- Equations A2-A5 define the projector MPO structure
- The bond dimension 2 projector (safe/danger) is extended to dimension 4 to also track the operator stage (before/after)
- The constraint $|DU\rangle|UD\rangle$ forbidden corresponds to the paper's constraint that no adjacent $|\uparrow\uparrow\rangle$ can occur at the boundary between merged sites
