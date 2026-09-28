# Generalized STARS simulations

This tutorial runs a generalized three-component STARS (**ST**ickers **A**nd
**R**andom **S**pacers) system through the OpenABC Python API. Components A, B,
and C each have a binary sequence and a chain count, with independently
configurable sticker and spacer interactions for all six unique component pairs.
The same interface supports additional components by extending those lists and
pair dictionaries.

STARS combines directional sticker attraction with heterogeneous, nonspecific
spacer interactions. The spacer-strength standard deviation represents sequence
complexity: a narrow distribution describes low complexity and a broad
distribution describes high complexity. See references [1–3] below.

## Files and setup

- [`run_multicomp_api.py`](run_multicomp_api.py): runnable generalized example.
- [`pair_example.text`](pair_example.text): interaction assignments to copy into
  the runner's optional pair-override section; this is not an input file read by
  the CLI.
- [`GENERALIZED.md`](GENERALIZED.md): detailed API and selector guide.
- [`generalized.py`](../../openabc/forcefields/stars/generalized.py): model
  construction, exported as `STARS` and `STARS_from_npy` from
  `openabc.forcefields`.

Use an environment with OpenABC and its dependencies, including NumPy and
OpenMM, installed. From the repository root, install the local package if needed:

```bash
python -m pip install -e .
cd tutorials/STARS-simulations
python run_multicomp_api.py --help
```

## Run the three-component example

```bash
python run_multicomp_api.py --n-comp 3 --steps 100 --platform CPU
```

By default each component has two chains with sequence `001001`. This gives
36 backbone beads and 12 auxiliary sticker beads, for 48 particles total.
The runner enables both sticker and spacer forces, sets same-component sticker
strengths to −10 and cross-component sticker strengths to zero, and sets all
spacer means and standard deviations to zero. Thus spacer forces are present
but have zero energy until their pair parameters are changed. All stickers are
eligible by default.

To change the composition:

```bash
python run_multicomp_api.py \
  --n-comp 3 \
  --sequences 001001001 010010 1001001 \
  --n-chains 10 8 6 \
  --steps 10000 \
  --platform CPU
```

Both lists must contain exactly `--n-comp` entries. Sequences must be nonempty
binary strings; chain counts must be nonnegative integers with a positive total.
Use `--platform CUDA` or `--platform OpenCL` when available. `--steps` must be
nonnegative. Interaction values are edited in the script, not supplied as CLI
flags.

The runner constructs the system, minimizes for up to 1000 iterations,
initializes velocities with seed 123, advances the requested steps, and prints
component count, particle count, and final potential energy. This is a short
NVT construction example; it does not implement a slab/compression protocol.

## Sequences and component pairs

A `0` contributes a backbone spacer bead. A `1` contributes a backbone spacer
with an auxiliary sticker immediately after it in topology order. For example,
`010` contains three backbone beads and one sticker. Stickers have mass 1 Da;
they are not virtual sites.

Component IDs are zero-based: A = 0, B = 1, C = 2. Every pair dictionary uses
canonical `(i, j)` keys with `i <= j`; specify cross pairs only once.

| Pair | Dictionary key | Sticker context parameter |
| --- | --- | --- |
| A–A | `(0, 0)` | `k_hb_0_0` |
| A–B | `(0, 1)` | `k_hb_0_1` |
| A–C | `(0, 2)` | `k_hb_0_2` |
| B–B | `(1, 1)` | `k_hb_1_1` |
| B–C | `(1, 2)` | `k_hb_1_2` |
| C–C | `(2, 2)` | `k_hb_2_2` |

`sticker_strengths` sets directional attraction, `spacer_means` sets average
nonspecific strength, and `spacer_stds` sets its nonnegative standard deviation
(not variance). Omitted numeric pairs default to zero in the API. Negative
strengths are attractive; positive spacer strengths are repulsive. Enable
`include_hbonds` and `include_spacers` explicitly: both default to `False` in
the API, although the runner enables them.

## Build a configured system with the API

```python
from openabc.forcefields import STARS

sim = STARS(
    n_comp=3,
    sequences=["001001001", "010010", "1001001"],
    n_chains=[10, 8, 6],
    sticker_strengths={
        (0, 0): -15.0, (0, 1): -17.0, (0, 2): -12.0,
        (1, 1): -8.0, (1, 2): 0.0, (2, 2): 0.0,
    },
    selectors={
        (0, 0): "1::2",
        (0, 1): {"i": "::2", "j": "all"},
        (0, 2): {"i": "::2", "j": "all"},
    },
    spacer_means={(0, 0): -0.2, (1, 1): -0.5, (1, 2): -0.5},
    spacer_stds={(0, 0): 0.1},
    include_hbonds=True, include_spacers=True,
    kr=-2.0, ka=-5.0, r0=0.0, hbond_gamma=1e-6,
    alpha=4.5, tau=1.5, gamma=1e-4,
    initial_box=40.0,
    temperature=1.0, friction_coeff=0.1, timestep=1.0,
    seed=123, platform_name="CPU",
)
sim.minimizeEnergy(maxIterations=1000)
sim.context.setVelocitiesToTemperature(sim.integrator.getTemperature(), 123)
sim.step(100)
```

The A–B and A–C channels above share the same selected A sticker pool. Selectors
control eligibility independently for each pair; they do not assign permanent
bonds or enforce an explicit valence cap.

Selectors accept `"all"`/`None`, slice strings such as `"::2"` and `"1::2"`, a
positive integer stride, or a list of unique sticker ordinals such as `[0, 3, 5]`.
Use `[]` to select no stickers. Ordinals span **all chains of a component** in
topology order and do not restart per chain. A diagonal rule selects both sides;
a simple cross-pair rule selects only the lower-index component. Use
`{"i": rule_i, "j": rule_j}` to select both sides of a cross pair. Missing
rules or sides select all stickers. Numeric `0` is not a valid stride.

## Energy terms and units

Let $E_0 = 1\ \mathrm{kJ\ mol^{-1}}$ and $\sigma = 1\ \mathrm{nm}$ be the
fixed reference energy and length. The total potential is

$$
U = U_{\mathrm{bond}} + U_{\mathrm{EV}} + U_{\mathrm{st-st}} + U_{\mathrm{sp-sp}}.
$$

The terms below describe the implemented potential, including its truncations
and pair-counting conventions. Setting `include_hbonds=False` or
`include_spacers=False` omits the corresponding force objects.

### Bonded interactions — `Class2BondPotential`

Every pair of adjacent backbone beads and every parent–sticker pair is connected
by a class-2 bond [4]:

$$
U_{\mathrm{bond}} = \sum_{\langle a,b\rangle}
\left[k_2(r_{ab}-\ell_0)^2 + k_3(r_{ab}-\ell_0)^3
+ k_4(r_{ab}-\ell_0)^4\right].
$$

The OpenMM `CustomBondForce` expression is

```text
k2*(r-bond_length)^2 + k3*(r-bond_length)^3 + k4*(r-bond_length)^4
```

Defaults are $\ell_0=\sigma$, $k_2=100E_0/\sigma^2$,
$k_3=100E_0/\sigma^3$, and $k_4=100E_0/\sigma^4$.
The quadratic term sets the local stiffness; the cubic and quartic terms make
the bond response anharmonic. Periodic boundary conditions are enabled.
These constants are not exposed by `STARS`; custom system assembly can use
`add_class2bond_forces` in `openabc.forcefields.stars.utils`.

### Excluded volume — `ExcludedVolumePotential`

Nonbonded backbone–backbone and backbone–sticker pairs interact through the
repulsive Weeks–Chandler–Andersen potential [5]:

$$
U_{\mathrm{EV}}(r)=
\begin{cases}
4E_0\left[(\sigma/r)^{12}-(\sigma/r)^6\right]+E_0,
& r < r_{\mathrm{cut}}^{\mathrm{EV}},\\
0, & r\ge r_{\mathrm{cut}}^{\mathrm{EV}}.
\end{cases}
$$

The default cutoff is $r_{\mathrm{cut}}^{\mathrm{EV}}=2^{1/6}\sigma$, the
Lennard-Jones minimum. The added $E_0$ makes the energy zero at this cutoff,
so the default potential is continuous and purely repulsive. OpenMM uses

```text
LJ * step(Outer_Cutoff - r);
LJ = 4*Epsilon*((Sigma/r)^12 - (Sigma/r)^6) + Epsilon
```

with `Epsilon=1` kJ/mol and `Sigma=1` nm. `cutoff_distance` changes both the
expression cutoff and neighbor-list cutoff; the zero-energy-at-cutoff property
applies to the default WCA cutoff, not arbitrary values.

Directly bonded pairs are excluded with `createExclusionsFromBonds(..., 1)`.
Sticker–sticker pairs have **no excluded-volume interaction**, permitting
stickers to overlap at the optimum of their directional attraction.

### Directional sticker interactions — `HbondPotential-i-j`

Each of the six component pairs has its own `CustomHbondForce` when
`include_hbonds=True`. For a participating donor sticker $a$ and acceptor
sticker $b$, the energy is

$$
u_{ab}^{(i,j)} = K_{ij}\exp\left[
 k_r(r_{ab}-r_0)^2 + k_\theta(\theta_1-\pi)^2
 + k_\theta(\theta_2-\pi)^2\right],
$$

where $K_{ij}$ is `sticker_strengths[i, j]` in units of $E_0$,
$k_r$ is `kr`, and $k_\theta$ is `ka`. The angles are defined by
(parent of donor, donor sticker, parent of acceptor) and
(parent of acceptor, acceptor sticker, parent of donor), respectively.
The implementation is analogous to directional interactions in coarse-grained
nucleic-acid models [6, 7]:

```text
k_hb_i_j * exp(kr*(distance(d1,a1)-r0)^2
             + ka*(angle(d2,d1,a2)-pi)^2
             + ka*(angle(a2,a1,d2)-pi)^2)
```

Defaults are `kr=-2.0` nm⁻², `ka=-5.0`, and `r0=0.0` nm. The exponential is
maximal at overlapping sticker positions and angles of $\pi$, corresponding
to collinear parent-to-sticker vectors. Negative $K_{ij}$ gives attraction;
more negative values strengthen it. Directionality suppresses simultaneous
optimal contacts, but there is no explicit bond-assignment or saturation rule.
Selectors determine which stickers may participate in each channel.

For a cross-component pair $i<j$, component $j$ supplies donors and component
$i$ supplies acceptors, so each eligible sticker pair contributes once.
For a same-component channel, both donor/acceptor orientations contribute for
each distinct eligible pair. Thus its total isolated-pair energy is twice the
single-orientation expression above. Self-pairs are excluded; other intrachain
sticker pairs remain eligible. Equal diagonal and cross prefactors therefore
do not imply equal pair well depths.

The radial cutoff is

$$
r_{\mathrm{cut}}^{\mathrm{st}} = r_0 +
\sqrt{\frac{\ln(\gamma_{\mathrm{st}})}{k_r}},
\qquad \gamma_{\mathrm{st}}=\texttt{hbond\_gamma}.
$$

At the default `hbond_gamma=1e-6`, the cutoff is about 2.628 nm and the radial
factor has decayed to $10^{-6}$. The sticker potential is truncated without an
energy shift, leaving a small residual at the cutoff. This tolerance is
independent of the spacer `gamma`. All sticker forces share force group 3;
zero-strength channels remain present but energetically inert.

### Random spacer interactions — `RandomSpacers`

For nonbonded backbone beads $a,b$ belonging to components $i,j$, a quenched
pair strength is sampled at construction:

$$
\epsilon_{ab}/E_0 \sim
\mathcal{N}\left(\mu_{ij},\Delta_{ij}^{\,2}\right),
$$

where $\mu_{ij}=\texttt{spacer\_means[i,j]}$ and
$\Delta_{ij}=\texttt{spacer\_stds[i,j]}$ is the standard deviation.
The finite-range potential [1] is

$$
u_{ab}^{\mathrm{sp}}(r)=
\begin{cases}
\frac{\epsilon_{ab}}{2}
\left[1+\tanh\bigl(\alpha(\tau-r)\bigr)-2\gamma\right],
& r<r_{\mathrm{cut}}^{\mathrm{sp}},\\
0, & r\ge r_{\mathrm{cut}}^{\mathrm{sp}}.
\end{cases}
$$

OpenMM implements this as a `CustomNonbondedForce` with a tabulated strength:

```text
0.5*epsilon(pindex1,pindex2)*(1+tanh(alpha*(tau-r))-2*gamma)
```

Here `tau` is the switching midpoint (1.5 nm by default), and `alpha` is the
sharpness, fixed at 4.5 nm⁻¹. Larger separation reduces the interaction smoothly.
Negative epsilon is attractive and positive epsilon is repulsive. A zero
standard deviation gives a uniform strength within a component pair; a nonzero
standard deviation creates particle-pair heterogeneity and can sample both
signs. The strengths remain fixed throughout dynamics rather than being
redrawn at each time step.

The subtraction of `2*gamma` makes the potential zero at

$$
r_{\mathrm{cut}}^{\mathrm{sp}} = \tau -
\frac{1}{\alpha}\operatorname{atanh}(2\gamma-1).
$$

For `tau=1.5`, `alpha=4.5`, and `gamma=1e-4`, this is about 2.523 nm.
The energy is continuous there; the implementation does not additionally
smooth the force to zero at the cutoff.

The full particle-indexed epsilon matrix is constructed as follows:

1. Draw the A–A, B–B, and C–C backbone blocks and symmetrize each using its
   upper triangle.
2. Draw A–B, A–C, and B–C backbone blocks once each and copy their transposes
   into the reverse blocks, ensuring $\epsilon_{ab}=\epsilon_{ba}$.
3. Leave all sticker rows and columns zero and register the matrix as an
   OpenMM `Discrete2DFunction`.

Only backbone–backbone interaction groups are active. Directly bonded pairs
are excluded. With all means and standard deviations zero, this term contributes
no energy, while excluded volume still acts. Disorder reproducibility and
quadratic memory cost are described below.

### Units and parameter defaults

Mass is 1 Da per particle, length is nm, and energy strengths use a **fixed
1 kJ/mol reference**, independent of thermostat temperature. `temperature=1.0`
means kBT = 1 kJ/mol (about 120.3 K). Friction is in ps⁻¹ and timestep in fs.
Changing temperature does not rescale the potential coefficients.

| API parameter | Default | Meaning |
| --- | --- | --- |
| `kr`, `ka`, `r0` | `-2.0`, `-5.0`, `0.0` | Sticker radial/angular shape and optimal distance |
| `hbond_gamma` | `1e-6` | Sticker radial cutoff tolerance |
| `alpha` | `4.5` | Spacer sharpness in nm⁻¹; this value is fixed |
| `tau`, `gamma` | `1.5`, `1e-4` | Spacer midpoint in nm and cutoff tolerance |
| `temperature` | `1.0` | Reduced thermostat temperature |
| `friction_coeff`, `timestep` | `0.1`, `1.0` | Langevin friction and timestep |
| `initial_box` | `None` | Cubic edge in nm; automatic estimate when omitted |
| `seed` | `None` | Spacer-disorder and Langevin random seed |

Sticker cutoff is `r0 + sqrt(log(hbond_gamma)/kr)` (about 2.628 nm by default).
Spacer cutoff is `tau - atanh(2*gamma-1)/alpha` (about 2.523 nm).
An explicit box must be at least twice every active cutoff; suitable packing
and avoidance of periodic overlaps remain the user's responsibility.

A nonzero `seed` reproduces spacer disorder and seeds Langevin dynamics;
OpenMM treats seed zero as automatic. Without a seed, spacer sampling uses
NumPy's global random state. Spacer tables scale as N², including all particles:
the float64 array alone occupies `8*N*N` bytes, with additional construction
and OpenMM copies.

## Coordinates, force groups, and output

Use `sim.stars_components[c]["backbone"]` and
`sim.stars_components[c]["stickers"]` for authoritative component atom indices.
Topology order is component, chain, sequence position, with each sticker directly
after its parent. A/B/C backbone names are `A`, `B`, `A2`; sticker names are
`S`, `T`, `S2`. Use component metadata for component-specific analysis and packing.
Preparation helpers based on A/B or S/T names do not distinguish every component.

| Force group | Force name |
| --- | --- |
| 0 | `CMMotionRemover` |
| 1 | `Class2BondPotential` |
| 2 | `ExcludedVolumePotential` |
| 3 | All `HbondPotential-i-j` forces |
| 6 | `RandomSpacers` |

All sticker pairs share group 3. Change an enabled sticker pair at runtime with
`sim.context.setParameter("k_hb_0_2", -12.0)`. Custom compression code must use
these generalized force names and context parameters.

Pass a finite `(number_of_atoms, 3)` array as `positions_nm` for custom starting
coordinates. For saved NumPy coordinates:

```python
from openabc.forcefields import STARS_from_npy

sim = STARS_from_npy(
    n_comp=3, sequences=["01", "10", "11"], n_chains=[2, 2, 1],
    position_npy="positions.npy", coordinate_unit="nm",
    initial_box=30.0, platform_name="CPU",
)
```

The file wrapper defaults to angstrom; set `coordinate_unit="nm"` explicitly
for nanometer files. Add OpenMM reporters to save trajectories and state data;
the supplied runner does not create output files. For example, before stepping:

```python
from openmm import app

sim.reporters.append(app.DCDReporter("trajectory.dcd", 100))
sim.reporters.append(app.StateDataReporter(
    "simulation.log", 100, step=True, time=True, potentialEnergy=True,
    kineticEnergy=True, temperature=True, separator="\t",
))
```

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

