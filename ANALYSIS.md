# Paper Analysis: Superdiffusive Energy Transport in Kinetically Constrained Models

**Reference:** M. Ljubotina, J.-Y. Desaules, M. Serbyn, and Z. Papić, Phys. Rev. X **13**, 011033 (2023)
**DOI:** [10.1103/PhysRevX.13.011033](https://doi.org/10.1103/PhysRevX.13.011033)

---

## 1. Physical Model

### Hamiltonian

**PXP Model (base):**

$$
H_{\text{PXP}} = \Omega \sum_i P_{i-1} \sigma^x_i P_{i+1}
$$

where:
- $P_i = |\downarrow\rangle\langle\downarrow|_i$ is the projector onto spin-down state
- $\sigma^x_i$ is the Pauli-X matrix at site $i$
- $\Omega = 1$ (Rabi frequency, set to unity)

**Deformations:**

1. **PXPZ deformation** (integrability-enhancing):
$$
H_{\text{PXPZ}} = H_{\text{PXP}} - \lambda \sum_i (\sigma^z_{i-2} + \sigma^z_{i+2}) P_{i-1} \sigma^x_i P_{i+1}
$$

2. **PNP deformation** (chemical potential):
$$
H_{\text{PNP}} = H_{\text{PXP}} + \delta \sum_i P_{i-1} n_i P_{i+1}
$$
where $n_i = |\uparrow\rangle\langle\uparrow|_i$.

3. **PNPNP deformation** (hard-square integrable point):
$$
H_{\text{PNPNP}} = H_{\text{PXP}} + \xi \sum_i P_{i-2} n_{i-1} P_i n_{i+1} P_{i+2}
$$

### Hilbert Space

- **Local degrees of freedom:** Spin-1/2 ($|\uparrow\rangle$, $|\downarrow\rangle$)
- **Constraint:** No two consecutive up spins allowed (Rydberg blockade)
  $$
  |\uparrow\uparrow\rangle \text{ states are forbidden}
  $$
- **Dimension:** Scales as Fibonacci numbers
  $$
  \mathcal{D}(N) \sim \phi^N, \quad \phi = \frac{1+\sqrt{5}}{2} \approx 1.618
  $$
- **Working space:** Largest connected sector excluding all $|\uparrow\uparrow\rangle$ pairs

### Symmetries

- Translation symmetry (with PBC)
- Spatial inversion
- Particle-hole symmetry (maps $|\uparrow\rangle \leftrightarrow |\downarrow\rangle$)
- Approximate su(2) algebra structure (responsible for quantum many-body scars)

---

## 2. Observables

### Primary Observable: Energy-Energy Correlation Function

$$
\langle h_0(0) h_\ell(t) \rangle_c = \langle h_0(0) h_\ell(t) \rangle - \langle h_0(0) \rangle \langle h_\ell(t) \rangle
$$

**Definitions:**
- Energy density operator at site $\ell$:
  $$
  h_\ell(0) = P_{\ell-1} \sigma^x_\ell P_{\ell+1}
  $$
- Time-evolved operator:
  $$
  h_\ell(t) = e^{iH t} h_\ell(0) e^{-iH t}
  $$
- Infinite-temperature expectation value:
  $$
  \langle O \rangle \equiv \text{Tr}(\mathcal{P} O)
  $$
  where $\mathcal{P} = \prod_i (\mathbb{1}_{i,i+1} - n_i n_{i+1})$ projects onto constrained space.

**Physical meaning:** Tracks spreading of energy perturbation at infinite temperature.

### Derived Quantities

- **Instantaneous dynamical exponent:**
  $$
  z^{-1}(t) = -\frac{d \ln \langle h_0(0) h_0(t) \rangle_c}{d \ln t}
  $$

- **Transport classification:**
  | $z$ value | $1/z$ value | Transport type |
  |-----------|-------------|----------------|
  | $z = 1$ | $1/z = 1$ | Ballistic |
  | $z = 3/2$ | $1/z \approx 0.67$ | Superdiffusive (KPZ) |
  | $z = 2$ | $1/z = 0.5$ | Diffusive |
  | $z > 2$ | $1/z < 0.5$ | Subdiffusive |

---

## 3. Numerical Methods

### Algorithm: TEBD (Time-Evolving Block Decimation)

**Method:** Evolve the energy density operator $h_\ell(0)$ as a Matrix Product Operator (MPO) in the Heisenberg picture.

**Key features:**
- Fourth-order Suzuki-Trotter decomposition for time evolution
- Site-merging: Combine pairs of sites to handle 3-site PXP terms as effective 2-site gates
- Constraint reduces local Hilbert space dimension from 4 to 3 per pair
- Randomized SVD for speedup at late times when bond dimension saturates

**Evolution scheme:**
$$
h(t + \delta t) = e^{iH\delta t} h(t) e^{-iH\delta t}
$$

### Algorithm: Exact Diagonalization (for benchmarking)

**Method:** Full diagonalization in constrained basis for small systems.

**Typical pure state approach:**
$$
|\psi\rangle = \frac{1}{\mathcal{N}} \sum_k c_k |\phi_k\rangle
$$
where $c_k$ are random Gaussian coefficients, enabling efficient infinite-temperature averaging.

### Approximations

| Approximation | Validity | Control Parameter |
|---------------|----------|-------------------|
| Trotter decomposition | $\delta t \ll 1/\|H\|$ | Time step $\delta t$ |
| MPO truncation | Bounded entanglement growth | Bond dimension $\chi$ |
| Finite system size | $t < N/(2v)$ where $v$ is spreading velocity | System size $N$ |

### Error Sources

- **Trotter error:** $O(\delta t^5)$ for 4th-order; controlled by decreasing $\delta t$
- **Truncation error:** Controlled by increasing bond dimension $\chi$
- **Finite-size effects:** Correlation must not reach boundaries

---

## 4. Parameters & Regimes

### System Parameters

| Parameter | Symbol | Values in Paper | Notes |
|-----------|--------|-----------------|-------|
| System size | $N$ | 512 – 1024 | Larger for longer times |
| Rabi frequency | $\Omega$ | 1.0 | Sets energy/time units |
| PXPZ strength | $\lambda$ | 0, 0.024, 0.05, 0.5 | 0.024 ≈ integrable point |
| PNP strength | $\delta$ | 0 – 2.0 | Chemical potential |
| PNPNP strength | $\xi$ | 0, 0.5, 1.0, 2.0 | $\xi=1$: integrable |

### Numerical Parameters

| Parameter | Symbol | Values | Convergence Criterion |
|-----------|--------|--------|----------------------|
| Bond dimension | $\chi$ | 256, 384, 512 | Results stable across $\chi$ values |
| Time step | $\delta t$ | 0.2 | 4th-order Trotter |
| Max time | $t_{\max}$ | 200 – 300 | Before finite-size effects |

### Physical Regimes

- **Short time ($t \lesssim 30$):** Oscillatory behavior due to su(2) representations; peaks at $t \approx 5.1, 10.2, 15.3, 20.4$
- **Long time ($t \gtrsim 50$):** Power-law decay; extract transport exponent
- **PXP ($\lambda = \delta = \xi = 0$):** Superdiffusive, $1/z$ slowly decreasing from ~1
- **PNPNP ($\xi = 1$):** Ballistic, $z = 1$
- **PNP ($\delta \geq 0.4$):** Stable superdiffusion, $z \approx 3/2$

---

## 5. Validation Criteria

### Known Limits

- **Integrable point ($\xi = 1$):** Expect ballistic transport $z = 1$
- **Large PXPZ ($\lambda = 0.5$):** Expect diffusion $z = 2$
- **$t \to 0$:** Correlation function $\langle h_0(0) h_0(0) \rangle_c$ should be finite and positive

### Benchmarks

- ED comparison for $N \leq 35$: TEBD and ED should match within 1% relative error
- Bond dimension convergence: Results for $\chi = 256$ and $\chi = 512$ should agree

### Sanity Checks

- [ ] Autocorrelation $C(t=0) > 0$
- [ ] $C(t)$ decays monotonically at late times (after oscillations damp)
- [ ] $1/z(t)$ bounded between 0 and ~1.5
- [ ] Correlation profile symmetric about center
- [ ] No finite-size effects (correlation hasn't reached boundary)

---

## 6. Key Results to Reproduce

### Figure 1: PXP Energy Autocorrelation

**What it shows:** Connected energy autocorrelation $\langle h_0(0) h_0(t) \rangle_c$ for pure PXP model.

**Key features:**
- (a) Two regimes: oscillatory (I) at short times, power-law decay (II) at long times
- (b) $1/z(t)$ approaches ~1 at $t \sim 100$, then slowly decreases toward ~2/3
- (c) Log-log plot consistent with possible power-law approach to diffusion

**Parameters:** $N = 1024$, $\chi = 512$, $t$ up to 300

### Figure 3: Integrable PNPNP Point

**What it shows:** Transport at integrable hard-square point ($\xi = 1$).

**Key features:**
- (a) Stable ballistic exponent $z = 1$ after initial transient
- (b) Flat spatial profile of correlation function

**Parameters:** $N = 512$, $\chi$ up to 384

### Figure 4: PXPZ Deformation Effects

**What it shows:** Effect of PXPZ deformation on transport.

**Key features:**
- $\lambda = 0.05$: Enhanced oscillations, similar late-time $z$ as PXP
- $\lambda = 0.024$: Fast decay, nearly ballistic ($1/z \to 1$ from above)
- $\lambda = 0.5$: Rapid onset of diffusion ($1/z \to 0.5$)
- Spatial profile becomes flat near $\lambda \approx 0.024$

**Parameters:** $N = 768$, $\chi = 384$ (256 for spatial profiles)

### Figure 5: Stable Superdiffusion with PNP

**What it shows:** Chemical potential deformation gives robust superdiffusion.

**Key features:**
- (a) $\delta \geq 0.4$: Clear superdiffusive exponent $z \approx 1.5$
- (b) Single-parameter scaling collapse with $1/z \approx 0.669$
- Ballistic peaks at edges expected to vanish as $t \to \infty$

**Parameters:** $N = 1024$, $\chi = 256$ (384 for $\delta = 2$)

### Quantitative Targets

| Quantity | Expected Value | Tolerance |
|----------|----------------|-----------|
| $1/z$ for PNPNP $\xi=1$ | 1.0 | ±0.05 |
| $1/z$ for PNP $\delta=0.5$ at $t=200$ | ~0.67 | ±0.05 |
| $1/z$ for PXPZ $\lambda=0.5$ | 0.5 | ±0.05 |
| Oscillation period (PXP) | ~5.1 | ±0.2 |

---

## Notes

- The paper uses a massively parallel TEBD implementation on PRACE supercomputer; full-scale reproduction ($N=1024$, $\chi=512$, $t=300$) requires significant computational resources
- For initial testing, smaller systems ($N=64-128$, $\chi=128$) with shorter times ($t \lesssim 50$) are sufficient to verify qualitative behavior
- The connection between superdiffusion and KPZ universality class remains an open question
