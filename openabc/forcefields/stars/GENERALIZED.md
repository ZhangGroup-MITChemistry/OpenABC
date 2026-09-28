# Generalized STARS

`STARS` supports any positive integer `n_comp`, subject to available memory and
compute. Each component has one binary sequence and a nonnegative chain count;
at least one chain must exist. Component IDs are **zero-based**: 0, 1, ..., n_comp-1.
Use a list of exactly `n_comp` sequences and chain counts. A `1` adds a sticker
side bead to that sequence position's backbone bead; a `0` has only the backbone.
All particles, including stickers, are massive particles of mass 1 Da, as in the
legacy implementation; they are not OpenMM virtual sites.

## Three-component example

With OpenABC:

```python
from openabc.forcefields import STARS

sim = STARS(
    n_comp=3,
    sequences=["001001001", "010010", "1001001"],
    n_chains=[10, 8, 6],
    include_hbonds=True,
    sticker_strengths={
        (0, 0): -1.0, (0, 1): -2.0, (0, 2): -0.5,
        (1, 1): -1.5, (1, 2): -0.8, (2, 2): -1.0,
    },
    selectors={
        (0, 0): "1::2",                  # same subset on both sides
        (0, 1): "::2",                   # select component 0; all of 1
        (1, 2): {"i": "all", "j": "::2"},  # independently select both sides
    },
    include_spacers=True,
    spacer_means={(0, 0): -0.2, (0, 1): -0.3, (1, 2): -0.1},
    spacer_stds={(0, 0): 0.1, (0, 1): 0.05},
    kr=-2.0, ka=-5.0, r0=0.0, hbond_gamma=1e-6,
    alpha=4.5, tau=1.5, gamma=1e-4,
    initial_box=40.0, temperature=1.0,
    friction_coeff=0.1, timestep=1.0,
    seed=123, platform_name="CPU",
)
sim.minimizeEnergy(maxIterations=1000)
sim.step(100)
```

For the local package, put the **parent `api_code` directory** on `PYTHONPATH`
and replace the import with `from stars import STARS`. On the cluster, that parent
can be `/home/yumzhang/orcd/pool/work/2-nuc/api_code`. The same implementation is
also exported from `stars.stars_model` and `openabc.forcefields.stars.stars_model`.
No installation of OpenABC is required to use the local package (NumPy and OpenMM
are required). ParmEd is optional for the existing PSF-writing helper.

## Pair parameters and selectors

All pair dictionaries use **canonical `(i, j)` keys with `i <= j`**. A cross pair
is specified once; do not also specify `(j, i)`. Reversed, out-of-range, floating
point, and boolean component IDs are rejected.

| Argument | Per-pair meaning | Omitted pair |
|---|---|---|
| `sticker_strengths` | Sticker H-bond prefactor | 0 |
| `spacer_means` | Gaussian mean epsilon for each backbone-particle pair | 0 |
| `spacer_stds` | Gaussian epsilon standard deviation, >= 0 | 0 |
| `selectors` | Sticker participation | All stickers |

`include_hbonds` and `include_spacers` default to `False`, matching the API
constructors. **Enable each force explicitly** when supplying its pair parameters.
Negative sticker strengths and negative spacer epsilon values are attractive.
Disorder is sampled separately for each particle pair, not once per component
pair, and the matrix is symmetric. A nonzero standard deviation can produce
both attractive and repulsive values. `seed` reproduces the epsilon realization;
it also sets the Langevin random seed (OpenMM treats seed 0 as automatic).
When seed is omitted, epsilon sampling uses NumPy's global random state, as before.

Selectors accept `"all"`/`None`, a slice string (`"1::2"`, `"::2"`), a positive
integer stride, or a list/tuple of unique nonnegative sticker ordinals. Use `[]`
to select no stickers. Invalid and out-of-range indices raise an error rather
than silently disappearing. Selection is over the entire component's sticker
list, spanning its chains, in topology order; it is **not reset on each chain**.

For a diagonal pair, one rule selects both donors and acceptors. For a cross
pair, a simple rule selects only component `i` (the lower index), preserving the
old AB convention. A mapping `{"i": rule_i, "j": rule_j}` selects both sides;
omitted sides mean all. Side mappings are only valid for cross pairs.

## Force conventions and units

The bond, excluded-volume, sticker, and spacer energy equations are unchanged.
Sticker energy is

```text
k_ij * exp(kr*(r-r0)^2 + ka*(theta1-pi)^2 + ka*(theta2-pi)^2)
```

The generalized interface validates `kr < 0`, `ka <= 0`, `r0 >= 0`, finite
numeric parameters, and both cutoff tolerances strictly between 0 and 1.
The sticker cutoff is `r0 + sqrt(log(hbond_gamma)/kr)`; `hbond_gamma` is now
explicitly independent of spacer `gamma`. The spacer cutoff is
`tau - atanh(2*gamma-1)/alpha` and must be positive when enabled.

Defaults follow the API: `kr=-2`, `ka=-5`, `r0=0`, excluded-volume cutoff
`2**(1/6)` nm, friction 0.1 ps^-1, and timestep 1 fs. **The new interface fixes
`alpha=4.5`** and rejects another value. It supplies usable spacer defaults
`tau=1.5`, `gamma=1e-4`; legacy wrappers retain their existing tau/gamma defaults.
The new default sticker tolerance is the legacy value `hbond_gamma=1e-6`.

Distances are nm; energy strengths are multiples of a fixed 1 kJ/mol reference.
Temperature 1 corresponds to kBT = 1 kJ/mol. Changing the thermostat temperature
does not rescale the potential coefficients. kr is in nm^-2 and alpha in nm^-1.

**Same-component sticker pairs retain legacy double counting:** two distinct
selected stickers contribute both donor/acceptor orientations. Cross-component
pairs contribute one orientation. Consequently, equal diagonal and cross
prefactors do not give equal isolated-pair well depths. Self-pairs are excluded;
other same-chain sticker interactions remain allowed. No saturation/valence
constraint has been added. Excluded volume still omits sticker-sticker pairs.

## Coordinates, labels, and preparation

Initial topology order is component, chain, sequence position, with each sticker
immediately after its parent backbone bead. `sim.stars_components[c]` contains
the authoritative `backbone` and `stickers` atom-index tuples for component `c`.
Names A/S and B/T are retained for components 0/1; later components use A2/S2,
A3/S3, etc. Use metadata rather than atom-name prefixes to distinguish components.
The generalized force construction uses explicit component index lists.

The default cubic box retains the chain-count estimate but enlarges it when
needed to fit long chains and active cutoffs. An explicit box must be at least
twice every active cutoff. Users supplying positions or a custom box must still
choose physically suitable packing and avoid overlaps across periodic boundaries.

Pass a finite `(number_of_atoms, 3)` array with `positions_nm=...`, or use:

```python
from openabc.forcefields import STARS_from_npy
sim = STARS_from_npy(
    n_comp=3, sequences=["01", "10", "11"], n_chains=[2, 2, 1],
    position_npy="positions.npy", coordinate_unit="angstrom",
    initial_box=30.0, platform_name="CPU",
)
```

The file wrapper defaults to angstrom to match existing files; use
`coordinate_unit="nm"` for nanometer files.

Force groups: 0 = center-of-mass remover, 1 = bonds, 2 = excluded volume,
3 = **all** sticker forces, 6 = spacers. Sticker forces are named
`HbondPotential-i-j`, with context parameters `k_hb_i_j`. Sharing group 3 avoids
the 32-force-group limit for many components. Pair-specific strengths can be
changed with `sim.context.setParameter("k_hb_0_2", value)`.

The existing local slab runners and backup directories are not modified. The
new example demonstrates model construction and short NVT dynamics; it does
not reproduce their compression/slab protocol. Legacy preparation helpers that
identify components by A/B or S/T names are not general multi-component packers.
Do not assume they distinguish later components; use `stars_components` when
implementing component-specific rearrangements. If reusing force-name selection
in compression, supply the new force names and parameter names explicitly.

## Compatibility and cost

`STARS_1comp`, `STARS_2comp`, and their coordinate wrappers remain available.
Local constructor defaults are aligned with the API defaults. Existing explicit
arguments still take precedence, including the unchanged root one-component
runner's `alpha=4.0`; use `STARS` for the new fixed-alpha interface.

The two installed copies of `generalized.py` are identical. Regression tests
compare one/two-component energies and forces by term using identical random
disorder; three-component sticker forces are checked against energy derivatives;
ten-component dynamics exercise more than 32 component pairs.

Spacer disorder retains the dense N x N epsilon table, where N includes all
particles. Memory is at least 8*N*N bytes before temporary arrays, the Python
list conversion, and OpenMM's copy. H-bond force count grows as n_comp*(n_comp+1)/2.
There is no hard-coded component-count cap, but large systems require sufficient
memory and may be expensive.

From the generalized OpenABC repository, run:

```sh
PYTHONDONTWRITEBYTECODE=1 python tests/test-stars/test_generalized.py
```

To exercise the local library using the same test file, set
`STARS_TEST_PACKAGE=stars` and `PYTHONPATH` to your local `api_code` directory.
