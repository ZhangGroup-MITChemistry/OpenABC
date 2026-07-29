# STARS Simulations

This tutorial demonstrates how to run one-component and two-component STARS
(**ST**ickers **A**nd **R**andom **S**pacers) simulations through the OpenABC
Python API.

STARS generalizes the classical stickers-and-spacers picture of associative
polymers by adding *heterogeneous, nonspecific* interactions among spacer
segments on top of *specific, valence-limited* sticker interactions. Sequence
complexity presented in the model as the spread of the spacer-interaction
distribution: a low-complexity sequence corresponds to a narrow spectrum of
spacer interaction strengths, a high-complexity sequence to a broad one. The
theory was introduced by Sood and Zhang [1]; the off-lattice implementation
documented here follows Zhang *et al.* [2] and is built on OpenABC [3].

Each chain is specified by a binary sequence:

- `0`: spacer-only backbone position
- `1`: parent spacer position carrying an auxiliary sticker bead

For example `010` is a three-bead backbone whose central spacer carries one
auxiliary sticker bead. Sticker beads interact through an orientation-dependent
(hydrogen-bond-like) attraction that is optimal only when the two parent→sticker
vectors are collinear, so a single sticker cannot satisfy the optimal geometry
with more than one partner — valency is limited by geometry rather than by an
explicit bond-assignment algorithm. Spacer beads interact through
excluded-volume repulsion plus an optional finite-range pair potential whose
per-pair strengths are drawn from a Gaussian distribution.

## Citations

If you use this model, please cite:

1. **STARS model.** A. Sood and B. Zhang, *Preserving condensate structure and
   composition by lowering sequence complexity*, **Biophysical Journal** 123
   (2024) 1815–1826. doi:[10.1016/j.bpj.2024.05.026](https://doi.org/10.1016/j.bpj.2024.05.026)
   (PMID 38824391)
2. **This off-lattice implementation.** Y. Zhang, A. Sood, A. Athreya and
   B. Zhang, *OpenABC Simulations of the Stickers and Random Spacers Model
   Reveal the Role of Sequence Complexity in Condensate Organization*.
3. **OpenABC.** S. Liu, C. Wang, A. P. Latham, X. Ding and B. Zhang, *OpenABC
   enables flexible, simplified, and efficient GPU accelerated simulations of
   biomolecular condensates*, **PLoS Computational Biology** 19 (2023) e1011442.
   doi:[10.1371/journal.pcbi.1011442](https://doi.org/10.1371/journal.pcbi.1011442)

Supporting references for the individual energy terms are listed under
[Energy Function](#energy-function).

## Files

- `run_1comp_api.py`: minimal one-component STARS simulation (sticker
  interactions only).
- `run_2comp_api.py`: minimal two-component STARS simulation with independent
  A–A, A–B and B–B sticker strengths.
- `output-1comp/` and `output-2comp/`: default output directories created by the
  example scripts.

The model itself lives in `openabc/forcefields/stars/`:

- `stars_model.py`: the four public entry points `STARS_1comp`, `STARS_2comp`,
  `STARS_1comp_from_npy`, `STARS_2comp_from_npy`, re-exported from
  `openabc.forcefields`.
- `utils.py`: sequence parsing, topology construction, the individual force
  builders, and the slab-preparation / I/O helpers described below.

## Units

The model is formulated in reduced units of mass *m*, length σ and energy ε.
To supply the dimensional quantities OpenMM requires, the implementation sets

| reduced unit | value in the code |
| --- | --- |
| mass *m* | 1 Da (every particle gets mass 1.0) |
| length σ | 1 nm |
| energy ε | 1 kJ mol⁻¹ |

Consequently `temperature=1.0` means *T* = ε/k_B ≈ 120.3 K, i.e. k_BT = ε, and
every energy argument (`kaa`, `kab`, `kbb`, `mean_eps_*`, `std_eps_*`) is
interpreted in units of k_BT — the builders multiply by `k_BT` internally.
Distances (`r0`, `tau`, `cutoff_distance`, `initial_box`) are in σ = nm.

## Energy Function

The total potential energy is

$$
U = U_\mathrm{bond} + U_\mathrm{EV} + U_\mathrm{st-st} + U_\mathrm{sp-sp}
$$

Conventional stickers-and-spacers simulations are recovered by switching the
spacer term off (`include_spacers=False`) or by using a uniform spacer strength
(`std_eps_* = 0`).

Each term maps onto exactly one OpenMM force object, and each force sits in its
own force group so that subsets can be activated during box compression (see
[Force groups](#force-groups)).

### 1. Bonded interactions — `Class2BondPotential`

Applied between adjacent backbone beads and between each auxiliary sticker bead
and its parent backbone bead, using a class-2 bond potential [4]:

$$
U_\mathrm{bond} = \sum_{\langle i,j\rangle}
  \left[ k_2 (r_{ij}-r_0)^2 + k_3 (r_{ij}-r_0)^3 + k_4 (r_{ij}-r_0)^4 \right]
$$

Implemented as an OpenMM `CustomBondForce` with
`k2*(r-bond_length)^2 + k3*(r-bond_length)^3 + k4*(r-bond_length)^4`.
Defaults in `add_class2bond_forces`: `r0 = 1.0 σ`, `k2 = 100 ε/σ²`,
`k3 = 100 ε/σ³`, `k4 = 100 ε/σ⁴`. Periodic boundary conditions are enabled for
this force.

### 2. Excluded volume — `ExcludedVolumePotential`

A purely repulsive Weeks–Chandler–Andersen potential [5] on nonbonded
spacer–spacer and spacer–sticker pairs:

$$
U_\mathrm{WCA}(r) = 4\varepsilon\left[(\sigma/r)^{12}-(\sigma/r)^{6}\right] + \varepsilon,
\qquad r < r^\mathrm{EV}_\mathrm{cut}
$$

and zero beyond the cutoff. Implemented as a `CustomNonbondedForce` with
`LJ * step(Outer_Cutoff - r)`, `Epsilon = 1 k_BT`, `Sigma = 1 σ`, and
`r^EV_cut = cutoff_distance = 2^(1/6) σ ≈ 1.1225 σ` — the Lennard-Jones minimum,
so the potential is continuous and purely repulsive.

Two exclusion rules matter here and both follow the Methods section:

- **Directly bonded pairs** (adjacent backbone beads; each sticker with its
  parent) are excluded via `createExclusionsFromBonds(..., 1)`.
- **Sticker–sticker pairs carry no excluded volume at all.** Only the
  interaction groups (spacer, spacer) and (spacer, sticker) are registered,
  because sticker–sticker contacts are described entirely by the directional
  potential below — which is what allows the two sticker beads to overlap at the
  optimal bonded geometry.

### 3. Sticker–sticker interactions — `HbondPotential-AA/-AB/-BB`

An orientation-dependent potential, analogous to hydrogen-bond terms used in
coarse-grained nucleic-acid models [6, 7]:

$$
U_\mathrm{st-st} = \sum_{i<j} K_\mathrm{sticker}\,
  \exp\!\left[ k_r (r_{ij}-r_0^\mathrm{st})^2
             + k_\theta (\theta_1-\theta_0)^2
             + k_\theta (\theta_2-\theta_0)^2 \right]
$$

where `r_ij` is the sticker–sticker distance and θ₁, θ₂ are the two angles
formed by each sticker bead and its parent backbone bead. Implemented as an
OpenMM `CustomHbondForce`:

```text
K * exp( kr*(distance(d1,a1)-r0)^2
       + ka*((angle(d2,d1,a2)-theta0)^2)
       + ka*((angle(a2,a1,d2)-theta0)^2) )
```

Manuscript parameterization, and the API defaults:

```python
kr = -2.0      # k_r, in sigma^-2
ka = -5.0      # k_theta, dimensionless
r0 = 0.0       # r_0^st, in sigma
# theta0 = pi, a local constant in add_hbond_forces
```

Because `kr` and `ka` are negative, the exponential is a Gaussian peaked at
`r_ij = r0` and `θ₁ = θ₂ = π`, so the most favorable configuration has the two
sticker beads overlapping with their parent→sticker vectors collinear. `K` is
negative for attraction and more negative values mean stronger binding
(`E_sticker ≈ K_sticker` at the optimal geometry).

Three separate `CustomHbondForce` objects are always created when
`include_hbonds=True`, one per sticker-type pair, so that the three strengths
are independent:

| force name | donors | acceptors | strength kwarg | manuscript symbol |
| --- | --- | --- | --- | --- |
| `HbondPotential-AA` | `S` | `S` | `kaa` | *K*<sub>AA</sub> |
| `HbondPotential-AB` | `T` | `S` | `kab` | *K*<sub>AB</sub> |
| `HbondPotential-BB` | `T` | `T` | `kbb` | *K*<sub>BB</sub> |

`S` and `T` are the sticker beads of the A and B components, respectively (see
[Bead naming](#bead-naming)). A strength left at its default of `0.0` makes that
force present but energetically inert, which is exactly how the manuscript
switches a channel off (e.g. `KAB = 0` for spacer-mediated client recruitment).
For the AA and BB forces the self-pair (a sticker bonding to itself) is
explicitly excluded.

For efficiency the interaction is truncated at a finite radial cutoff,

$$
r^\mathrm{st}_\mathrm{cut} = r_0^\mathrm{st} + \sqrt{\ln\gamma / k_r}
$$

where γ is the value of the radial exponential factor at the cutoff. It defaults
to `1e-6` in `add_hbond_forces` and is not forwarded from the STARS entry
points, so it is effectively fixed at 10⁻⁶; with the default `kr` and
`r0` this gives `r^st_cut ≈ 2.628 σ`, the distance at which the radial factor has
decayed to 10⁻⁶. Note that this γ is *not* the `gamma` argument of the STARS
entry points, which belongs to the spacer potential.

### 4. Spacer–spacer interactions — `RandomSpacers`

The nonspecific, finite-range spacer potential of the STARS theory [1]:

$$
U_\mathrm{sp-sp} = \sum_{i<j} \tfrac{1}{2}\,\epsilon_{ij}
  \left[ 1 + \tanh\!\left( \alpha (r_0^\mathrm{sp} - r_{ij}) \right) \right]
$$

Implemented as a `CustomNonbondedForce` with a tabulated per-pair strength:

```text
0.5 * epsilon(pindex1,pindex2) * (1 + tanh(alpha*(tau - r)) - 2*gamma)
```

Two implementation details are worth spelling out:

- `tau` **is** the manuscript's `r_0^sp`, the midpoint of the interaction range
  (1.5 σ in the paper); `alpha` is the sharpness α (4.5 σ⁻¹ in the paper).
- The extra `- 2*gamma` shifts the prefactor to zero at the truncation distance,
  so the potential is continuous there. The truncation is

  $$
  r^\mathrm{sp}_\mathrm{cut} = r_0^\mathrm{sp} - \tfrac{1}{\alpha}\tanh^{-1}(2\gamma-1)
  $$

  With the manuscript values `tau = 1.5`, `alpha = 4.5`, `gamma = 1e-4`, this
  gives `r^sp_cut ≈ 2.523 σ`, where the prefactor has decayed to γ = 10⁻⁴.

Negative `ε_ij` is attractive, positive is repulsive; `ε_ij = 0` leaves only
excluded volume. Strengths are drawn per **atom pair** from Gaussian
distributions,

$$
\epsilon_{ij} \sim \mathcal{N}(\bar\epsilon, \Delta\epsilon^2)
$$

with independent (mean, standard deviation) pairs for the A–A, A–B and B–B
spacer blocks. `ε̄` sets the average nonspecific attraction; `Δε` sets the
*heterogeneity*, i.e. the model's proxy for sequence complexity.

Construction of the *N* × *N* matrix, in order: the A–A and B–B blocks are drawn
and symmetrized (`triu(X) + triu(X,1).T`); the A–B block is drawn once and copied
into the B–A block transposed, so ε_ij = ε_ji throughout; all rows and columns
belonging to sticker beads are zeroed, since stickers do not participate in the
spacer potential; the matrix is scaled by k_BT and registered as an OpenMM
`Discrete2DFunction`. Only the (spacer, spacer) interaction group is active, and
directly bonded pairs are excluded.

Two practical consequences:

- **Reproducibility.** The ε matrix is sampled with `numpy.random.normal` from
  the global NumPy RNG at build time. Call `numpy.random.seed(...)` before
  constructing the simulation if you need the same disorder realization twice,
  and treat different seeds as independent disorder replicas.
- **Memory.** The tabulated function holds *N*² entries for *N* particles, so
  memory grows quadratically with system size. This, not the force evaluation,
  is usually the practical ceiling on how large a `include_spacers=True` system
  can be.

### Force groups

| group | force name | term |
| --- | --- | --- |
| 0 | `CMMotionRemover` | center-of-mass motion removal |
| 1 | `Class2BondPotential` | *U*<sub>bond</sub> |
| 2 | `ExcludedVolumePotential` | *U*<sub>EV</sub> |
| 3 | `HbondPotential-AB` | *U*<sub>st−st</sub> (A–B) |
| 4 | `HbondPotential-AA` | *U*<sub>st−st</sub> (A–A) |
| 5 | `HbondPotential-BB` | *U*<sub>st−st</sub> (B–B) |
| 6 | `RandomSpacers` | *U*<sub>sp−sp</sub> |

These names and groups are what `compress_box_npt` uses to run a compression
stage with only a subset of the potential active.

### Bead naming

`parse_sequence` and the topology builders label beads by component, and the
force builders select particles by these names — so do not rename them:

| bead | name | role |
| --- | --- | --- |
| A-component backbone | `A` | spacer |
| A-component sticker | `S` | auxiliary sticker bead |
| B-component backbone | `B` | spacer |
| B-component sticker | `T` | auxiliary sticker bead |

Each auxiliary sticker bead is added to the topology **immediately after its
parent backbone bead**, and `add_hbond_forces` relies on this by registering the
parent as index `d - 1`. Any external tool that reorders atoms will silently
break the sticker geometry.

## Parameter Reference

All four entry points share the same keyword set; the two-component versions
take `seqA`/`seqB` and `nA`/`nB` in place of `seq`/`nChains`, and the `_from_npy`
variants add `position_npy`.

```python
from openabc.forcefields import (
    STARS_1comp, STARS_2comp, STARS_1comp_from_npy, STARS_2comp_from_npy,
)
```

### System definition

| kwarg | default | meaning |
| --- | --- | --- |
| `seq` / `seqA`, `seqB` | required | binary sequence(s); `0` = spacer, `1` = parent spacer with sticker |
| `nChains` / `nA`, `nB` | required | number of chains per component |
| `initial_box` | `None` | initial cubic box edge in nm. If `None`, the heuristic `L = (nChains or nA+nB)*padding + 10` nm is used, and the chosen value is printed |
| `padding` | `2.5` | per-chain padding used by that heuristic (nm) |
| `platform_name` | `None` | OpenMM platform: `CPU`, `CUDA`, `OpenCL`, `Reference`. `None` lets OpenMM choose |
| `position_npy` | — | (`_from_npy` only) `(N,3)` array of coordinates in **Å**; divided by 10 internally. Atom count must match the topology exactly |

Initial placement of chains (`build_topology_positions`) is on a square grid
with `bond_length = 1.0 σ` along *z*, `chain_spacing = 4.0 σ` between chains and
the sticker offset `side_offset = 1.0 σ`. In the two-component builder all A
chains are placed before all B chains.

### Integrator and thermostat

| kwarg | default | meaning |
| --- | --- | --- |
| `integrator_type` | `'Langevin'` | `'Langevin'` (production) or `'Verlet'` (no thermostat) |
| `temperature` | `1.0` | in units of ε/k_B, so `1.0` ⇒ k_BT = ε ≈ 120.3 K |
| `friction_coeff` | `0.1` | Langevin friction in ps⁻¹ |
| `timestep` | `1.0` | integration timestep in fs |

### Sticker term (`include_hbonds=True`)

| kwarg | default | manuscript | meaning |
| --- | --- | --- | --- |
| `include_hbonds` | `False` | — | build the three sticker forces |
| `kr` | `-2.0` | *k_r* = −2 σ⁻² | radial stiffness (negative) |
| `ka` | `-5.0` | *k*<sub>θ</sub> = −5 | angular stiffness (negative) |
| `r0` | `0.0` | *r*₀<sup>st</sup> = 0 | optimal sticker–sticker separation, σ |
| `kaa` | `0.0` | *K*<sub>AA</sub> | A–A sticker strength, k_BT |
| `kab` | `0.0` | *K*<sub>AB</sub> | A–B sticker strength, k_BT |
| `kbb` | `0.0` | *K*<sub>BB</sub> | B–B sticker strength, k_BT |
| `selector` | `None` | *N*<sub>sticker</sub> | activate only a subset of stickers, see below |

θ₀ = π is a local constant inside `add_hbond_forces`, and its cutoff γ defaults
to 10⁻⁶ there (not reachable from the STARS entry points). Note the
tutorial scripts pass their `--*-strength` values straight through, and
`include_hbonds` defaults to `False` in the API but `True` in the scripts.

**Selecting a sticker subset with `selector`.** The manuscript varies sticker
valency by activating only a subset of the auxiliary sticker beads present on a
chain, and by assigning some A stickers to A–A and others to A–B interactions.
`selector` is the mechanism: a dict keyed by sticker-type pair, whose value
slices the global list of that type.

```python
selector = {
    ('S', 'T'): 'all',      # A-B: use every S acceptor
    ('S', 'S'): '::2',      # A-A: every other S bead
    ('T', 'T'): [0, 2, 4],  # B-B: explicit indices
}
```

Accepted rule forms: `'all'`, an `int` *n* (stride, i.e. `lst[::n]`), a slice
string such as `'::2'` or `'10:40'`, a list/tuple of indices, or the string form
of such a list (`'[0,2,4]'`). Indices refer to position within the concatenated
list of all `S` (or all `T`) beads in the system, not within a single chain. For
`('S','S')` and `('T','T')` the rule is applied to donors and acceptors alike;
for `('S','T')` only the acceptor (`S`) list is sliced. Sticker beads not
selected by any rule still exist in the topology and still feel bonds and
excluded volume — they are simply inert in that sticker channel.

### Spacer term (`include_spacers=True`)

| kwarg | default | manuscript | meaning |
| --- | --- | --- | --- |
| `include_spacers` | `False` | — | build `RandomSpacers` |
| `mean_eps_AA` | `0.0` | ε̄<sub>AA</sub> | mean A–A spacer strength, k_BT |
| `std_eps_AA` | `0.0` | Δε<sub>AA</sub> | A–A heterogeneity, k_BT |
| `mean_eps_AB` | `0.0` | ε̄<sub>AB</sub> | mean A–B spacer strength, k_BT |
| `std_eps_AB` | `0.0` | Δε<sub>AB</sub> | A–B heterogeneity, k_BT |
| `mean_eps_BB` | `0.0` | ε̄<sub>BB</sub> | mean B–B spacer strength, k_BT |
| `std_eps_BB` | `0.0` | Δε<sub>BB</sub> | B–B heterogeneity, k_BT |
| `alpha` | `4.5` | α = 4.5 σ⁻¹ | sharpness of the tanh switching |
| `tau` | `None` | *r*₀<sup>sp</sup> = 1.5 σ | midpoint of the interaction range, σ |
| `gamma` | `None` | γ = 10⁻⁴ | residual prefactor at the truncation distance |

**`tau` and `gamma` have no usable defaults.** They are `None` in the signature
and are only consumed when `include_spacers=True`, so a spacer-enabled run must
supply them explicitly — use `tau=1.5, gamma=1e-4` to reproduce the work in manuscript.
Also note that `alpha` is carried in the signature even when spacers are off
(the tutorial scripts pass `alpha=4.5` while running sticker-only), where it has
no effect; it is a spacer parameter, never the sticker radial parameter `kr`.

### Excluded volume

| kwarg | default | meaning |
| --- | --- | --- |
| `cutoff_distance` | `2**(1/6)` | *r*<sub>cut</sub><sup>EV</sup> in σ; also the neighbor-list cutoff of the WCA force |

`Epsilon = 1 k_BT`, `Sigma = 1 σ` and the class-2 bond constants
(`k2 = k3 = k4 = 100`, `bond_length = 1.0 σ`) are not exposed through the STARS
entry points. To change them, call `add_excluded_volume_forces` /
`add_class2bond_forces` from `openabc.forcefields.stars.utils` while building a
system yourself.

### Manuscript system compositions

For reference, the systems reported in Zhang *et al.* [2] (full details in their
Table S1):

- **One component:** 100 identical A chains; 74 backbone spacer beads per chain,
  24 of them sticker-bearing, each carrying one auxiliary sticker bead. Only a
  subset of those 24 stickers is activated for A–A interactions in the main-text
  runs — this is what `selector` is for.
- **Two components:** A chains assign 12 stickers to A–A and 12 to A–B; B chains
  carry backbone spacers plus auxiliary stickers assigned to A–B. *K*<sub>AA</sub>
  and *K*<sub>AB</sub> are then varied independently.

### Reproducing the manuscript parameter set

```python
import numpy as np
from openabc.forcefields import STARS_1comp

np.random.seed(2024)                 # fix the spacer disorder realization

seq = "".join("1" if i % 3 == 0 else "0" for i in range(74))  # 74 spacers, 25 stickers

sim = STARS_1comp(
    seq=seq,                         # replace with your own sticker pattern
    nChains=100,
    integrator_type="Langevin",
    temperature=1.0,                 # k_BT = epsilon
    friction_coeff=0.1,              # ps^-1
    timestep=1.0,                    # fs
    cutoff_distance=2 ** (1 / 6),    # WCA cutoff
    include_hbonds=True,
    kr=-2.0, ka=-5.0, r0=0.0,        # directional sticker potential
    kaa=-10.0,                       # K_sticker in k_BT
    include_spacers=True,
    mean_eps_AA=0.0, std_eps_AA=2.0, # eps_bar and Delta_eps in k_BT
    alpha=4.5, tau=1.5, gamma=1e-4,  # spacer range and truncation
    initial_box=30.0,
    platform_name="CUDA",
)
```

## Simulation Protocol Utilities

The tutorial scripts run a single-box minimize → equilibrate → produce sequence.
The slab protocol used in the manuscript is assembled from helpers in
`openabc/forcefields/stars/utils.py`; they are not wrapped by the tutorial CLI,
so call them directly.

### The manuscript's staged protocol

1. Place chains in extended conformations on a 3D grid, avoiding steric overlap;
   in two-component systems place A chains first, then B chains at least 10σ from
   the A-rich region.
2. Energy minimize, then compress under NPT with only bonded and excluded-volume
   interactions active (two-component systems additionally keep weakened A–A and
   A–B sticker forces, see below).
3. Minimize again and relax in NVT for 2 × 10⁵ steps, still with bonded and
   excluded-volume interactions only.
4. Set the slab box to 30σ × 30σ × 300σ.
5. (Optional) Equilibrate for 5 × 10³ NVT steps with the *full* production potential.
6. Run production NVT for 5 × 10⁷ steps, saving configurations every 5 × 10³ steps.

Two notes on step 1: `build_topology_positions` uses `chain_spacing = 4.0 σ`
whereas the manuscript specifies a minimum separation of 5σ, and the builder
lays chains out on a square *XY* grid with the backbone along *z* rather than
sequentially along *x*, *y*, *z*. Use `pack_chains_in_elongated_box` (and, for
two components, `shift_atoms_by_name`) to reproduce the manuscript's placement,
or supply your own coordinates through `STARS_*_from_npy`.

### `compress_box_npt(...)`

Compresses an initially dilute box toward a slab geometry
(30σ × 30σ × 300σ in the manuscript), writing `npt-log.tsv` and `npt.dcd` into
`outdir`. Selected arguments:

| argument | required? | meaning |
| --- | --- | --- |
| `simulation`, `system`, `integrator` | yes | the objects returned / used by the STARS entry point |
| `nA`, `outdir` | yes | number of A chains (`nB` defaults to `0` for one-component runs) and output directory |
| `target_box_length` | yes | target *X*/*Y* edge (an OpenMM `Quantity`) |
| `slab_extension_factor` | yes | sets the *Z* threshold `target_box_length * slab_extension_factor`; > 1 gives the elongated slab |
| `box_reduction` | yes | box-edge decrement per iteration (a `Quantity`) |
| `n_steps`, `save_freq` | yes | dynamics steps per iteration, and reporter interval |
| `max_iterations` | `1000` | iteration cap |
| `compression_axes` | `'xy'` | `'xy'` = Monte Carlo barostat, cube → slab; `'z'` = manual *Z* ratchet at fixed *X*/*Y*; `'xyz'` = manual, all three |
| `compression_force_names` | `None` | force names (see [Force groups](#force-groups)) active during compression; `None` = all |
| `compression_hbond_strengths` | `None` | temporary sticker strengths during compression, e.g. `{'kaa': -10.0, 'kab': -10.0}`, restored afterwards |
| `barostat_pressure`, `barostat_interval` | `1.0 bar` / `25` | `MonteCarloBarostat` settings, used only for `compression_axes='xy'` |
| `initial_box_length` / `initial_box_lengths` | `None` | starting cube edge, or an explicit `(x, y, z)` triple; if both are `None`, the `(nA+nB)*padding + 10` nm heuristic is used |

The manuscript protocol maps onto these directly: one-component systems compress
with only `('Class2BondPotential', 'ExcludedVolumePotential')` active; two-component
systems additionally keep the A–A and A–B sticker forces active but temporarily
weakened to −10 k_BT, with spacer interactions excluded throughout compression.
In the manual (`'z'`, `'xyz'`) modes the function temporarily reduces the timestep
to ≤ 0.2 fs and raises the Langevin friction to ≥ 5 ps⁻¹ for stability, then
restores the production values.

### Other helpers

| function | purpose |
| --- | --- |
| `pack_chains_in_elongated_box(sim, xy_nm, z_nm, chain_spacing_nm=3.0, margin_nm=2.0, min_layer_gap_nm=None, placement_xy_nm=None, interleave_components=False)` | repack whole chains into a narrow-*XY* / long-*Z* box; keeps each component in one layer, expanding the box if the request is too small. `interleave_components=True` mixes A and B in a shared layer instead of stacking them along *z*; `min_layer_gap_nm` enforces a minimum separation between the A and B layers |
| `shift_atoms_by_name(sim, atom_names, shift_nm, axis='z', wrap=True)` | translate selected bead types along one axis — how B chains are placed on one side of the slab for the client-loading simulations |
| `recenter_to_box_center(sim, box_dims_nm)` | move the centroid of all beads to the box center |
| `write_pdb_and_psf(sim, pdb_filename, psf_filename, outdir)` | write current coordinates as PDB, plus a PSF when ParmEd is installed (skipped with a message otherwise) |
| `parse_sequence`, `build_topology_positions`, `build_topology_positions_2comp` | sequence → bead labels → topology and starting coordinates |
| `add_class2bond_forces`, `add_excluded_volume_forces`, `add_hbond_forces`, `add_random_spacer_forces` | the individual force builders, for custom system assembly. Calling them directly is the only way to change `k2`/`k3`/`k4`/`bond_length`, the WCA `epsilon`/`sigma`, or the sticker cutoff `gamma`. (θ₀ = π is a local constant and not an argument; `add_random_spacer_forces` accepts `skip_spacers` but its body is commented out, so the flag currently has no effect.) |
| `tanh_r_cut(tau, alpha, gamma)` | the spacer truncation distance, if you want to check `r_cut^sp` before building a system |

## Run A One-Component Simulation

From this directory:

```bash
python run_1comp_api.py
```

This runs 10 chains using the default sequence:

```text
0010010010001
```

Useful options:

```bash
python run_1comp_api.py \
  --sequence 0010010010001 \
  --n-chains 16 \
  --box-length 30.0 \
  --temperature 1.0 \
  --friction 0.1 \
  --timestep 1.0 \
  --hbond-strength -1.5 \
  --steps 10000 \
  --report-interval 100 \
  --platform CPU \
  --output output-1comp
```

Disable sticker attractions:

```bash
python run_1comp_api.py --no-hbonds
```

`--hbond-strength` sets `kaa` (i.e. *K*<sub>AA</sub> in k_BT); more negative is
stronger. The script fixes `kr=-2.0`, `ka=-5.0`, `r0=0.0` and `alpha=4.5`, and
does not enable the spacer term — use the API directly for STARS runs with
`include_spacers=True`.

## Run A Two-Component Simulation

From this directory:

```bash
python run_2comp_api.py
```

This runs 10 A chains and 10 B chains using the default sequences:

```text
A: 0010010010001
B: 0100100100100
```

Useful options:

```bash
python run_2comp_api.py \
  --sequence-a 0010010010001 \
  --sequence-b 0100100100100 \
  --n-a 16 \
  --n-b 16 \
  --box-length 30.0 \
  --aa-strength -1.0 \
  --ab-strength -2.0 \
  --bb-strength -0.5 \
  --steps 10000 \
  --report-interval 100 \
  --platform CPU \
  --output output-2comp
```

Disable sticker attractions:

```bash
python run_2comp_api.py --no-hbonds
```

The three strengths map to `kaa`, `kab` and `kbb`. Setting `--ab-strength 0`
reproduces the manuscript's spacer-mediated recruitment condition — but the
spacer interactions that then drive A–B association must be supplied through the
API (`include_spacers=True`, `mean_eps_AB`, `std_eps_AB`), not through this
script.

## Outputs

Each script writes into `--output`:

- `initial.pdb`: minimized starting structure.
- `trajectory.dcd`: simulation trajectory (`--report-interval`, periodic images
  enforced).
- `simulation.log`: tab-separated OpenMM state data (step, time, potential and
  kinetic energy, temperature, speed).
- `final.pdb`: final structure after the requested number of steps.

`compress_box_npt` additionally writes `npt-log.tsv` and `npt.dcd`.

## Analysis Conventions

For reference, the manuscript quantifies trajectories as follows. These
definitions are analysis conventions, not part of the force field module.

- **Sticker contact indicator.** Stickers *i* and *j* are in contact when
  `r_ij < r_c = 0.5 σ` **and** `|φ_ij| > φ_c = 0.9 π`, where φ_ij is the dihedral
  angle of the four beads (parent *a*, sticker *i*, sticker *j*, parent *b*).
- **Sticker valence.** *v_i(t)* is the number of contacts of sticker *i*;
  the fractions with *v* = 0, 1 and > 1 report unbound, singly bound and
  (geometrically suppressed) higher-valence stickers.
- **Degree of conversion.** *p*(t) = 2 *N*<sub>contact</sub>(t) / *N*<sub>sticker</sub>,
  partitioned into intrachain and interchain contributions.
- **Mean-squared displacement.** Coordinates unwrapped across periodic
  boundaries, first 20 % of the trajectory discarded, collective drift of the
  sticker population subtracted, then fitted to MSD(τ) = 6*D*τ + *b* over lag
  times between 30 % and 90 % of the maximum lag. Lag times in MD steps, MSD in
  σ², *D* in σ² per MD step.
- **Dense-phase identification.** One-component: the aligned slab is centered in
  the box and the dense phase is evaluated in a fixed 20 σ central core window,
  against two 60 σ dilute reference windows; a system is called phase separated
  when ρ_core/ρ_dilute ≥ 2. Two-component: the dense domain is selected
  adaptively from the combined profile ρ_tot(z) = ρ_A(z) + ρ_B(z) as the widest
  contiguous region above ρ_cut = ρ₂₀ + 0.01(ρ₉₅ − ρ₂₀), rejecting candidates
  narrower than 5 nm, then narrowed to a co-mixed interval where both components
  contribute ≥ 10 % of the local profile. Trajectory means and uncertainties come
  from five contiguous time blocks.

## Notes

- The default OpenMM platform in both scripts is `CPU`. Use `--platform CUDA` or
  `--platform OpenCL` when available; GPU acceleration is the point of running
  STARS through OpenABC.
- Temperature is specified in reduced STARS units: `--temperature 1.0`
  corresponds to one kJ mol⁻¹ of thermal energy, i.e. k_BT = ε.
- Two distinct quantities are both called γ. The sticker radial cutoff uses
  γ = 10⁻⁶, fixed inside `add_hbond_forces`; the `gamma` keyword of the STARS
  entry points is the spacer truncation parameter (10⁻⁴ in the manuscript).
- Setting `integrator_type='Verlet'` removes the thermostat; use it only for
  diagnostics, not production runs.

## References

1. A. Sood, B. Zhang. *Preserving condensate structure and composition by
   lowering sequence complexity.* Biophys. J. **123**, 1815–1826 (2024).
   doi:10.1016/j.bpj.2024.05.026
2. Y. Zhang, A. Sood, A. Athreya, B. Zhang. *OpenABC Simulations of the Stickers
   and Random Spacers Model Reveal the Role of Sequence Complexity in Condensate
   Organization.*
3. S. Liu, C. Wang, A. P. Latham, X. Ding, B. Zhang. *OpenABC enables flexible,
   simplified, and efficient GPU accelerated simulations of biomolecular
   condensates.* PLoS Comput. Biol. **19**, e1011442 (2023).
   doi:10.1371/journal.pcbi.1011442
4. H. Sun. *COMPASS: an ab initio force-field optimized for condensed-phase
   applications — overview with details on alkane and benzene compounds.*
   J. Phys. Chem. B **102**, 7338–7364 (1998). doi:10.1021/jp980939v
5. S. P. Tan, H. Adidharma, M. Radosz. *Weeks–Chandler–Andersen model for
   solid–liquid equilibria in Lennard-Jones systems.* J. Phys. Chem. B **106**,
   7878–7881 (2002). doi:10.1021/jp013579b
6. N. A. Denesyuk, D. Thirumalai. *Coarse-grained model for predicting RNA
   folding thermodynamics.* J. Phys. Chem. B **117**, 4901–4911 (2013).
   doi:10.1021/jp401087x
7. I. Riveros, B. Zhang. *NEAT-DNA: a chemically accurate, sequence-dependent
   coarse-grained model for large-scale DNA simulations.* J. Chem. Theory
   Comput. **22**, 3709–3719 (2026). doi:10.1021/acs.jctc.5c01966
