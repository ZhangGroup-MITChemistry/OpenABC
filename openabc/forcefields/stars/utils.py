import openmm
import openmm as mm
import openmm.app as app
from openmm import unit, Discrete2DFunction
import numpy as np
import math
import random
import ast
from collections import defaultdict
from openmm import unit
from openmm.openmm import Vec3, MonteCarloBarostat
# new: also bring in DCDReporter
from openmm.app import StateDataReporter, Simulation, DCDReporter
from openmm.app import Topology
# Monkey-patch for legacy calls (if needed)
from openmm.app.element import Element
import os
from openmm.app import PDBFile
from typing import Optional, Tuple

try:
    import parmed as pmd
except ImportError:
    pmd = None
    print("Warning: ParmEd not found; PSF writing functions will be disabled. ")



mm.Element = Element

T = 1 * unit.kilojoule_per_mole / \
    (unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA)
k_BT = T * unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
length_scale = 1.0 * unit.nanometer

def parse_sequence(
    seq: str,
    chain: str = 'A',
) -> list:
    """
    Convert a binary string sequence into bead-type labels for specified chain.

    chain 'A': spacer 'A', sticker 'S';
    chain 'B': spacer 'B', sticker 'T'.
    """
    if chain == 'A':
        spacer_label, sticker_label = 'A', 'S'
    elif chain == 'B':
        spacer_label, sticker_label = 'B', 'T'
    else:
        raise ValueError("chain must be 'A' or 'B'")
    return [sticker_label if c == '1' else spacer_label for c in seq]


def build_topology_positions(
    bead_types: list,
    num_chains: int,
    bond_length: float = 1.0,
    side_offset: float = 1.0,
    chain_spacing: float = 4.0,
) -> Tuple[app.Topology, list]:
    """
    Build topology and positions for 1-component system.
    Backbone beads always named 'A', stickers are virtual beads named 'S'.
    """
    topology = app.Topology()
    element_C = mm.Element.getBySymbol('C')
    positions = []
    chain_idx = 0
    spacer_label, sticker_label = 'A', 'S'
    grid_width = int(math.ceil(math.sqrt(num_chains)))
    for _ in range(num_chains):
        chain = topology.addChain()
        residue = topology.addResidue('MOL', chain)
        prev_atom = None
        grid_x = (chain_idx % grid_width) * chain_spacing
        grid_y = (chain_idx // grid_width) * chain_spacing
        for i, bead_flag in enumerate(bead_types):
            # backbone atom always spacer_label
            atom = topology.addAtom(spacer_label, element_C, residue)
            positions.append(mm.Vec3(
                grid_x,
                grid_y,
                i * bond_length
            ) * unit.nanometer)
            if prev_atom is not None:
                topology.addBond(prev_atom, atom)
            prev_atom = atom
            # if bead_flag denotes sticker, add sticker virtual atom
            if bead_flag == sticker_label:
                sc_atom = topology.addAtom(sticker_label, element_C, residue)
                positions.append(mm.Vec3(
                    grid_x + side_offset,
                    grid_y,
                    i * bond_length
                ) * unit.nanometer)
                topology.addBond(atom, sc_atom)
        chain_idx += 1
    return topology, positions


def build_topology_positions_2comp(
    bead_types_A: list,
    bead_types_B: list,
    nA: int,
    nB: int,
    bond_length: float = 1.0,
    side_offset: float = 1.0,
    chain_spacing: float = 4.0,
) -> Tuple[app.Topology, list]:
    """
    Build topology and positions for 2-component system.
    Backbone beads always named 'A' or 'B'; stickers are virtual beads named 'S' or 'T'.
    """
    topology = app.Topology()
    element_C = mm.Element.getBySymbol('C')
    positions = []
    chain_idx = 0
    grid_width = int(math.ceil(math.sqrt(nA + nB)))
    # A-chains
    spacer_A, sticker_A = 'A', 'S'
    for _ in range(nA):
        chain = topology.addChain()
        residue = topology.addResidue('MOL', chain)
        prev_atom = None
        grid_x = (chain_idx % grid_width) * chain_spacing
        grid_y = (chain_idx // grid_width) * chain_spacing
        for i, bead_flag in enumerate(bead_types_A):
            atom = topology.addAtom(spacer_A, element_C, residue)
            positions.append(mm.Vec3(
                grid_x,
                grid_y,
                i * bond_length
            ) * unit.nanometer)
            if prev_atom is not None:
                topology.addBond(prev_atom, atom)
            prev_atom = atom
            if bead_flag == sticker_A:
                sc = topology.addAtom(sticker_A, element_C, residue)
                positions.append(mm.Vec3(
                    grid_x + side_offset,
                    grid_y,
                    i * bond_length
                ) * unit.nanometer)
                topology.addBond(atom, sc)
        chain_idx += 1
    # B-chains
    spacer_B, sticker_B = 'B', 'T'
    for _ in range(nB):
        chain = topology.addChain()
        residue = topology.addResidue('MOL', chain)
        prev_atom = None
        grid_x = (chain_idx % grid_width) * chain_spacing
        grid_y = (chain_idx // grid_width) * chain_spacing
        for i, bead_flag in enumerate(bead_types_B):
            atom = topology.addAtom(spacer_B, element_C, residue)
            positions.append(mm.Vec3(
                grid_x,
                grid_y,
                i * bond_length
            ) * unit.nanometer)
            if prev_atom is not None:
                topology.addBond(prev_atom, atom)
            prev_atom = atom
            if bead_flag == sticker_B:
                sc = topology.addAtom(sticker_B, element_C, residue)
                positions.append(mm.Vec3(
                    grid_x + side_offset,
                    grid_y,
                    i * bond_length
                ) * unit.nanometer)
                topology.addBond(atom, sc)
        chain_idx += 1
    return topology, positions


def add_class2bond_forces(
    system: mm.System,
    topology: app.Topology,
    k2: float = 100.0,
    k3: float = 100.0,
    k4: float = 100.0,
    bond_length: float = 1.0,
):
    expr = 'k2*(r-bond_length)^2 + k3*(r-bond_length)^3 + k4*(r-bond_length)^4'
    force = mm.CustomBondForce(expr)
    force.setForceGroup(1)
    force.addGlobalParameter('k2', k2 * k_BT / length_scale**2)
    force.addGlobalParameter('k3', k3 * k_BT / length_scale**3)
    force.addGlobalParameter('k4', k4 * k_BT / length_scale**4)
    force.addGlobalParameter('bond_length', bond_length * length_scale)
    force.setUsesPeriodicBoundaryConditions(True)
    force.setName('Class2BondPotential')
    for bond in topology.bonds():
        force.addBond(bond[0].index, bond[1].index)
    system.addForce(force)


def add_excluded_volume_forces(
    system: mm.System,
    topology: app.Topology,
    epsilon: float = 1.0,
    sigma: float = 1.0,
    cutoff: float = 2**(1/6),
):
    energy = (
        'LJ * step(Outer_Cutoff - r);'
        'LJ = 4.0*Epsilon*((Sigma/r)^12 - (Sigma/r)^6) + Epsilon'
    )
    force = mm.CustomNonbondedForce(energy)
    force.setForceGroup(2)
    force.addGlobalParameter('Epsilon', epsilon * k_BT)
    force.addGlobalParameter('Sigma', sigma * length_scale)
    force.addGlobalParameter('Outer_Cutoff', cutoff * length_scale)

    # 3) Cutoff method
    force.setNonbondedMethod(force.CutoffPeriodic)
    force.setCutoffDistance(cutoff * length_scale)
    force.setName('ExcludedVolumePotential')

    for _ in topology.atoms():
        force.addParticle()

    # 5) Build index lists for spacers vs. stickers
    spacer_idxs  = [a.index for a in topology.atoms() if a.name[0] in ('A','B')]
    sticker_idxs = [a.index for a in topology.atoms() if a.name[0] in ('S','T')]

    # 6) Only include spacer–spacer and spacer–sticker interactions
    force.addInteractionGroup(spacer_idxs, spacer_idxs)
    force.addInteractionGroup(spacer_idxs, sticker_idxs)

    # 7) Exclude bonded pairs
    force.createExclusionsFromBonds(
        [(b[0].index, b[1].index) for b in topology.bonds()], 1
    )
    system.addForce(force)


def add_hbond_forces(
    system: mm.System,
    topology: Topology,
    kr: float,
    ka: float,
    r0: float,
    kab: float = 0.0,
    kaa: float = 0.0,
    kbb: float = 0.0,
    selector: dict = None,
    gamma: float = 1e-6,
):
    """
    Adds three CustomHbondForce objects:
      - HbondPotential-AB (strength = kab)
      - HbondPotential-AA (strength = kaa)
      - HbondPotential-BB (strength = kbb)

    `selector` may contain rules for slicing, e.g.:
      selector = {
        ('S','T'): 'all',
        ('S','S'): '::2',
        ('T','T'): [0,2,4],
      }
    """

    def hbond_r_cut(kr, r0, gamma):
        return r0 + np.sqrt(np.log(gamma) / kr)

    theta0    = math.pi
    cutoff_nm = hbond_r_cut(kr, r0, gamma) * length_scale

    # 1) gather sticker indices
    indices_S = [a.index for a in topology.atoms() if a.name.startswith('S')]
    indices_T = [a.index for a in topology.atoms() if a.name.startswith('T')]

    # 2) helper to apply selector slices
    def parse_slice(s):
        parts = [int(p) if p else None for p in s.split(':')]
        return slice(*parts)
    def apply_rule(lst, rule):
        if rule == 'all':
            return lst
        if isinstance(rule, int):
            return lst[::rule]
        if isinstance(rule, str):
            if rule.strip().startswith('['):
                rule = ast.literal_eval(rule)
                return [lst[i] for i in rule if 0 <= i < len(lst)]
            return lst[parse_slice(rule)]
        if isinstance(rule, (list, tuple)):
            return [lst[i] for i in rule if 0 <= i < len(lst)]
        return []

    # 3) builder for each H-bond force
    def _build_force(name, kparam, kvalue, donors, acceptors, force_group, exclude_self=False):
        expr = (
            f"{kparam} * exp("
            "kr*(distance(d1,a1)-r0)^2 + "
            "ka*((angle(d2,d1,a2)-theta0)^2) + "
            "ka*((angle(a2,a1,d2)-theta0)^2)"
            ")"
        )
        f = mm.CustomHbondForce(expr)
        f.setName(name)
        f.setForceGroup(force_group)
        f.setNonbondedMethod(f.CutoffPeriodic)
        f.setCutoffDistance(cutoff_nm)
        # global params
        f.addGlobalParameter('kr',  kr / length_scale**2)
        f.addGlobalParameter('ka',  ka)
        f.addGlobalParameter(kparam, kvalue * k_BT)
        f.addGlobalParameter('r0',   r0 * length_scale)
        f.addGlobalParameter('theta0', theta0)
        # donors & acceptors
        d_tags = [f.addDonor(d, d-1, -1) for d in donors]
        a_tags = [f.addAcceptor(a, a-1, -1) for a in acceptors]
        # optionally exclude self-pairs (for AA/BB)
        if exclude_self:
            for dt, at in zip(d_tags, a_tags):
                f.addExclusion(dt, at)
        system.addForce(f)

    # --- A–B H-bonds ---
    donors_ab   = list(indices_T)
    acceptors_ab= list(indices_S)
    if selector and ('S','T') in selector:
        # apply only to acceptors (could also slice donors if desired)
        acceptors_ab = apply_rule(acceptors_ab, selector[('S','T')])
    _build_force("HbondPotential-AB", 'kab', kab, donors_ab, acceptors_ab, force_group=3, exclude_self=False)

    # --- A–A H-bonds ---
    donors_aa    = list(indices_S)
    acceptors_aa = list(indices_S)
    if selector and ('S','S') in selector:
        donors_aa = apply_rule(donors_aa, selector[('S','S')])
        acceptors_aa = apply_rule(acceptors_aa, selector[('S','S')])
    _build_force("HbondPotential-AA", 'kaa', kaa, donors_aa, acceptors_aa, force_group=4, exclude_self=True)

    # --- B–B H-bonds ---
    donors_bb    = list(indices_T)
    acceptors_bb = list(indices_T)
    if selector and ('T','T') in selector:
        donors_bb = apply_rule(donors_bb, selector[('T','T')])
        acceptors_bb = apply_rule(acceptors_bb, selector[('T','T')])
    _build_force("HbondPotential-BB", 'kbb', kbb, donors_bb, acceptors_bb, force_group=5, exclude_self=True)


def tanh_r_cut(tau, alpha, gamma):
    return tau - (1.0/alpha) * np.arctanh(2.0 * gamma - 1.0)




def add_random_spacer_forces(
    system: openmm.System,
    topology: app.Topology,
    alpha: float,
    tau: float,
    gamma: float,
    mean_eps_AA: float = 0.0,
    std_eps_AA: float = 0.0,
    mean_eps_AB: float = 0.0,
    std_eps_AB: float = 0.0,
    mean_eps_BB: float = 0.0,
    std_eps_BB: float = 0.0,
    skip_spacers: bool = False,
):
    """
    Exactly mimics the golden‐role RandomSpacers:
      • unique ε_ij for each spacer–spacer atom pair
      • zero out sticker–bead interactions
      • optionally skip ~¼ of spacers if skip_spacers=True
      • scale by k_BT
      • tanh‐based CustomNonbondedForce over spacer–spacer
    """
    import numpy as np
    import openmm
    from openmm import unit
    from .stars_model import k_BT, length_scale

    def tanh_r_cut(tau, alpha, gamma):
        return tau - (1.0/alpha) * np.arctanh(2*gamma - 1)

    # 1) identify atom indices
    indices_A = [a.index for a in topology.atoms() if a.name.startswith('A')]
    indices_B = [a.index for a in topology.atoms() if a.name.startswith('B')]
    indices_S = [a.index for a in topology.atoms() if a.name.startswith('S')]
    indices_T = [a.index for a in topology.atoms() if a.name.startswith('T')]
    all_spacers = indices_A + indices_B

    N = topology.getNumAtoms()
    epsilon_mat = np.zeros((N, N), dtype=np.float64)

    # 2) fill A–A block
    rand_aa = np.random.normal(mean_eps_AA, std_eps_AA, size=(len(indices_A), len(indices_A)))
    tri_aa = np.triu(rand_aa) + np.triu(rand_aa, k=1).T
    epsilon_mat[np.ix_(indices_A, indices_A)] = tri_aa

    # 3) fill B–B block
    rand_bb = np.random.normal(mean_eps_BB, std_eps_BB, size=(len(indices_B), len(indices_B)))
    tri_bb = np.triu(rand_bb) + np.triu(rand_bb, k=1).T
    epsilon_mat[np.ix_(indices_B, indices_B)] = tri_bb

    # 4) fill A–B & B–A blocks
    rand_ab = np.random.normal(mean_eps_AB, std_eps_AB, size=(len(indices_A), len(indices_B)))
    epsilon_mat[np.ix_(indices_A, indices_B)] = rand_ab
    epsilon_mat[np.ix_(indices_B, indices_A)] = rand_ab.T

    # 5) zero out sticker interactions
    for idx in indices_S + indices_T:
        epsilon_mat[idx, :] = 0.0
        epsilon_mat[:, idx] = 0.0

    # 6) optional dynamic skip of ~1/4 spacers
#    if skip_spacers:
#        skip_A = sorted(indices_A)[::3]
#        skip_A2 = [i+1 for i in skip_A]
#        skip_B = sorted(indices_B)[::3]
#        skip_set = set(skip_A) | set(skip_A2) | set(skip_B)
#        for i in skip_set:
#            epsilon_mat[i, :] = 0.0
#            epsilon_mat[:, i] = 0.0

    # 7) scale by k_BT (in kJ/mol)
    epsilon_mat *= k_BT.value_in_unit(unit.kilojoule_per_mole)

    # 8) build Discrete2DFunction
    flat_eps = epsilon_mat.ravel().tolist()
    eps_func = openmm.openmm.Discrete2DFunction(N, N, flat_eps)

    # 9) create the CustomNonbondedForce
    energy = "0.5 * epsilon(pindex1,pindex2) * (1 + tanh(alpha*(tau - r))-2*gamma)"
    force = openmm.CustomNonbondedForce(energy)
    force.setName('RandomSpacers')
    force.setForceGroup(6)
    force.addPerParticleParameter('pindex')
    force.addTabulatedFunction('epsilon', eps_func)
    force.addGlobalParameter('alpha', alpha/length_scale)
    force.addGlobalParameter('tau', tau*length_scale)
    force.addGlobalParameter('gamma', gamma)
    for atom in topology.atoms():
        force.addParticle([atom.index])

    # 10) cutoff & exclusions
    cutoff = tanh_r_cut(tau, alpha, gamma) * length_scale
    force.setNonbondedMethod(force.CutoffPeriodic)
    force.setCutoffDistance(cutoff)
    force.createExclusionsFromBonds(
        [(b[0].index, b[1].index) for b in topology.bonds()], 1
    )
    force.addInteractionGroup(all_spacers, all_spacers)

    system.addForce(force)




def compress_box_npt(
    simulation: Simulation,
    system,
    integrator,
    nA: int,
    outdir: str,
    #initial_box_length: unit.Quantity,
    target_box_length: unit.Quantity,
    slab_extension_factor: float,
    save_freq: int,
    n_steps: int,
    box_reduction: unit.Quantity,
    barostat_pressure: unit.Quantity = 1.0 * unit.bar,
    barostat_interval: int = 25,
    max_iterations: int = 1000,
    initial_box_length: Optional[float] = None,
    initial_box_lengths: Optional[tuple] = None,
    padding: float = 2.5,
    nB: int = 0,                            # if 1-comp, user can omit nB
    compression_force_names: Optional[tuple] = None,
    compression_hbond_strengths: Optional[dict] = None,
    compression_axes: str = "xy",
) -> Optional[int]:
    """
    NPT-compress the box toward the target geometry.

    compression_axes="xy" preserves the legacy behavior: start from a cube and
    shrink X/Y, switching to slab geometry when X <= target*slab_extension_factor.
    compression_axes="z" starts from a rectangular box and ratchets Z down toward
    target*slab_extension_factor while X/Y are held at the initial rectangular
    X/Y lengths.
    compression_axes="xyz" starts from a rectangular box and ratchets X/Y down
    toward target_box_length and Z down toward target*slab_extension_factor.
    """

    # 1) Convert all quantities to plain floats in nanometers
    nChains = nA + nB
    if initial_box_lengths is not None:
        init_box_nm = tuple(
            value.value_in_unit(unit.nanometer) if hasattr(value, 'value_in_unit') else float(value)
            for value in initial_box_lengths
        )
        if len(init_box_nm) != 3:
            raise ValueError("initial_box_lengths must contain exactly three values")
    else:
        if initial_box_length is None:
            # simple heuristic: one padding length per chain
            init_nm = nChains * padding + 10
        elif hasattr(initial_box_length, 'value_in_unit'):
            init_nm = initial_box_length.value_in_unit(unit.nanometer)
        else:
            init_nm = float(initial_box_length)
        init_box_nm = (init_nm, init_nm, init_nm)

    #    init_nm      = initial_box_length.value_in_unit(unit.nanometer)
    target_nm    = target_box_length.value_in_unit(unit.nanometer)
    slab_thresh  = target_nm * slab_extension_factor
    reduce_nm    = box_reduction.value_in_unit(unit.nanometer)
    axes = compression_axes.lower()
    if axes not in {"xy", "z", "xyz"}:
        raise ValueError("compression_axes must be 'xy', 'z', or 'xyz'")

    def _set_orthorhombic_box(x_nm, y_nm, z_nm):
        x_nm, y_nm, z_nm = float(x_nm), float(y_nm), float(z_nm)
        if min(x_nm, y_nm, z_nm) <= 0:
            raise ValueError(f"Invalid box lengths: {x_nm}, {y_nm}, {z_nm} nm")
        simulation.context.setPeriodicBoxVectors(
            Vec3(x_nm, 0.0, 0.0) * unit.nanometer,
            Vec3(0.0, y_nm, 0.0) * unit.nanometer,
            Vec3(0.0, 0.0, z_nm) * unit.nanometer,
        )

    def _vector_length_nm(v):
        return np.sqrt(sum(v[i].value_in_unit(unit.nanometer)**2 for i in range(3)))

    def _box_lengths_nm():
        vectors = simulation.context.getState().getPeriodicBoxVectors()
        return tuple(_vector_length_nm(v) for v in vectors)

    def _position_bounds_nm():
        state = simulation.context.getState(getPositions=True, enforcePeriodicBox=False)
        pos_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        return pos_nm.min(axis=0), pos_nm.max(axis=0)

    def _wrap_positions_into_box():
        state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
        positions = state.getPositions(asNumpy=True)
        pos_nm = positions.value_in_unit(unit.nanometer)
        if not np.isfinite(pos_nm).all():
            raise ValueError("Particle coordinate is NaN during NPT compression.")
        simulation.context.setPositions(positions)

    def _assert_positions_sane(stage):
        state = simulation.context.getState(getPositions=True, enforcePeriodicBox=False)
        pos_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        if not np.isfinite(pos_nm).all():
            raise RuntimeError(f"Particle coordinate is NaN during {stage}.")

        # Whole chains may sit slightly outside the primary box under PBC, but
        # million-nm coordinates mean the integrator has blown up.
        coord_limit = 10.0 * max(_box_lengths_nm()) + 100.0
        max_abs = float(np.max(np.abs(pos_nm)))
        if max_abs > coord_limit:
            raise RuntimeError(
                f"Particle coordinates became unreasonably large during {stage} "
                f"(max |x|={max_abs:.3f} nm; limit={coord_limit:.3f} nm)."
            )

    def _minimize_current_box(stage, max_iterations=200):
        try:
            simulation.minimizeEnergy(maxIterations=int(max_iterations))
            _wrap_positions_into_box()
            _assert_positions_sane(stage)
        except Exception as exc:
            raise RuntimeError(f"Energy minimization failed during {stage}.") from exc

    def _run_steps_in_chunks(total_steps, chunk_steps, stage):
        remaining = int(total_steps)
        chunk_steps = max(1, int(chunk_steps))
        while remaining > 0:
            this_chunk = min(chunk_steps, remaining)
            simulation.step(this_chunk)
            _assert_positions_sane(stage)
            remaining -= this_chunk

    def _set_temporary_hbond_parameters(strengths):
        if not strengths:
            return {}

        restore_values = {}
        for param_name, strength in strengths.items():
            if strength is None:
                continue
            try:
                old_value = simulation.context.getParameter(param_name)
            except Exception:
                print(
                    "[compress_box_npt] WARNING: requested temporary "
                    f"{param_name}={strength}, but this parameter is not available."
                )
                continue

            new_value = (float(strength) * k_BT).value_in_unit(unit.kilojoule_per_mole)
            simulation.context.setParameter(param_name, new_value)
            restore_values[param_name] = old_value
            print(
                "[compress_box_npt] Temporary NPT H-bond strength: "
                f"{param_name}={float(strength):.3f}"
            )
        return restore_values

    def _restore_hbond_parameters(restore_values):
        for param_name, old_value in restore_values.items():
            simulation.context.setParameter(param_name, old_value)
        if restore_values:
            restored = ", ".join(sorted(restore_values))
            print(f"[compress_box_npt] Restored production H-bond parameters: {restored}")

    manual_axes = axes in {"z", "xyz"}

    # 2) Clear old reporters & add barostat when OpenMM should perform MC
    # volume moves.  In elongated manual-compression modes the box is changed
    # below, while dynamics relaxes with the selected force groups.
    simulation.reporters = []
    barostat_idx = None
    if manual_axes:
        print(f"[compress_box_npt] {axes.upper()} mode: manual box reduction with no OpenMM barostat.")
    else:
        barostat = MonteCarloBarostat(barostat_pressure, T)
        barostat.setFrequency(barostat_interval)
        barostat_idx = system.addForce(barostat)

    def _force_group_mask(force_names=None):
        if force_names is None:
            names = None
        else:
            names = set(force_names)

        mask = 0
        active_names = []
        for f in system.getForces():
            name = f.getName()
            if names is None or name in names or f.getForceGroup() == 0:
                mask |= 1 << f.getForceGroup()
                active_names.append(name)

        if mask == 0:
            raise ValueError("No force groups selected for NPT compression")
        return mask, active_names

    # 3) Select the forces used during box compression.
    # By default this is all forces; callers can pass only bonds/excluded volume
    # to avoid attraction-driven collapse while the box is shrinking.
    all_groups = 0
    for f in system.getForces():
        all_groups |= 1 << f.getForceGroup()
    compression_groups, active_names = _force_group_mask(compression_force_names)
    integrator.setIntegrationForceGroups(compression_groups)
    print("[compress_box_npt] Active compression forces:", ", ".join(active_names))

    # 4) Reinitialize so barostat & force-groups stick
    simulation.context.reinitialize(preserveState=True)
    hbond_restore_values = _set_temporary_hbond_parameters(compression_hbond_strengths)
    original_step_size = None
    original_friction = None
    if manual_axes:
        if hasattr(integrator, "getStepSize") and hasattr(integrator, "setStepSize"):
            original_step_size = integrator.getStepSize()
            original_step_fs = original_step_size.value_in_unit(unit.femtosecond)
            compression_step_fs = min(original_step_fs, 0.2)
            if compression_step_fs < original_step_fs:
                integrator.setStepSize(compression_step_fs * unit.femtosecond)
                print(
                    "[compress_box_npt] Manual mode: temporary timestep "
                    f"{compression_step_fs:.3f} fs for compression relaxation "
                    f"(will restore {original_step_fs:.3f} fs)."
                )
        if hasattr(integrator, "getFriction") and hasattr(integrator, "setFriction"):
            original_friction = integrator.getFriction()
            inverse_ps = unit.picosecond**-1
            original_friction_ps = original_friction.value_in_unit(inverse_ps)
            compression_friction_ps = max(original_friction_ps, 5.0)
            if compression_friction_ps > original_friction_ps:
                integrator.setFriction(compression_friction_ps * inverse_ps)
                print(
                    "[compress_box_npt] Manual mode: temporary Langevin friction "
                    f"{compression_friction_ps:.3f} 1/ps for damping "
                    f"(will restore {original_friction_ps:.3f} 1/ps)."
                )

    # 5) Initial axis-align to the requested orthorhombic box
    _set_orthorhombic_box(*init_box_nm)
    _wrap_positions_into_box()
    pmin, pmax = _position_bounds_nm()
    print(
        "[compress_box_npt] Initial packed state: "
        f"box=({init_box_nm[0]:.4f}, {init_box_nm[1]:.4f}, {init_box_nm[2]:.4f}) nm; "
        f"pos_min=({pmin[0]:.4f}, {pmin[1]:.4f}, {pmin[2]:.4f}) nm; "
        f"pos_max=({pmax[0]:.4f}, {pmax[1]:.4f}, {pmax[2]:.4f}) nm"
    )

    # 6) Add DCD + StateData reporters
    #os.makedirs(outdir, exist_ok=True)
    simulation.reporters.append(
        StateDataReporter(
            f"{outdir}/npt-log.tsv", save_freq,
            step=True, time=True, temperature=True,
            potentialEnergy=True, kineticEnergy=True,
            totalEnergy=True, volume=True, density=True,
            elapsedTime=True, progress=True,
            remainingTime=True, speed=True,
            totalSteps=int(n_steps) * int(max_iterations), separator='\t'
        )
    )
    simulation.reporters.append(
        DCDReporter(f"{outdir}/npt.dcd", save_freq)
    )

    # 7) Compression loop
    for it in range(max_iterations):
        Lx, Ly, Lz = _box_lengths_nm()
        _set_orthorhombic_box(Lx, Ly, Lz)
        pmin, pmax = _position_bounds_nm()

        print(f"[compress_box_npt] Iter {it}: Lx={Lx:.4f} nm, Ly={Ly:.4f} nm, Lz={Lz:.4f} nm")
        print(
            "[compress_box_npt] Position bounds: "
            f"x=[{pmin[0]:.4f}, {pmax[0]:.4f}], "
            f"y=[{pmin[1]:.4f}, {pmax[1]:.4f}], "
            f"z=[{pmin[2]:.4f}, {pmax[2]:.4f}] nm"
        )

        # done?
        if axes == "z":
            if Lz <= slab_thresh:
                print(f"-> reached target Z {slab_thresh} nm; stopping.")
                break
        elif axes == "xyz":
            if Lx <= target_nm and Ly <= target_nm and Lz <= slab_thresh:
                print(f"-> reached target X/Y {target_nm} nm and Z {slab_thresh} nm; stopping.")
                break
        else:
            if Lx <= target_nm and Ly <= target_nm:
                print(f"-> reached target {target_nm} nm; stopping.")
                break

        # choose geometry
        if axes == "z":
            x_nm, y_nm = init_box_nm[0], init_box_nm[1]
            z_nm = max(slab_thresh, Lz - reduce_nm)
        elif axes == "xyz":
            x_nm = Lx if Lx <= target_nm else max(target_nm, Lx - reduce_nm)
            y_nm = Ly if Ly <= target_nm else max(target_nm, Ly - reduce_nm)
            z_nm = Lz if Lz <= slab_thresh else max(slab_thresh, Lz - reduce_nm)
        else:
            if Lx <= slab_thresh:
                # slab: fix Z at slab_thresh, shrink X/Y
                new_xy = max(target_nm, Lx - reduce_nm)
                x_nm, y_nm = new_xy, new_xy
                z_nm       = slab_thresh
            else:
                # full cubic shrink
                new_L = max(target_nm, Lx - reduce_nm)
                x_nm = y_nm = z_nm = new_L

        # reapply an explicitly orthorhombic box before stepping
        _set_orthorhombic_box(x_nm, y_nm, z_nm)
        if manual_axes:
            _wrap_positions_into_box()
            _minimize_current_box(f"{axes.upper()}-compression iter {it} after box shrink", max_iterations=200)
            simulation.context.setVelocitiesToTemperature(T)
            saved_state = simulation.context.getState(
                getPositions=True, getVelocities=True, enforcePeriodicBox=False
            )
            try:
                _run_steps_in_chunks(
                    int(n_steps),
                    chunk_steps=min(25, max(1, int(n_steps))),
                    stage=f"{axes.upper()}-compression iter {it} relaxation",
                )
            except Exception as exc:
                print(
                    f"[compress_box_npt] WARNING: {axes.upper()} relaxation became unstable "
                    f"at iter {it}; restoring pre-relaxation coordinates and "
                    "continuing after minimization."
                )
                print(f"[compress_box_npt] Relaxation error was: {exc}")
                _set_orthorhombic_box(x_nm, y_nm, z_nm)
                simulation.context.setPositions(saved_state.getPositions())
                simulation.context.setVelocities(saved_state.getVelocities())
                _minimize_current_box(f"{axes.upper()}-compression iter {it} fallback", max_iterations=500)
                simulation.context.setVelocitiesToTemperature(T)
            _wrap_positions_into_box()
        else:
            # advance NPT
            try:
                simulation.step(int(n_steps))
            except Exception as exc:
                if "periodic box vector" not in str(exc):
                    raise
                print("[compress_box_npt] Reapplying orthorhombic box after OpenMM box-vector error.")
                _set_orthorhombic_box(x_nm, y_nm, z_nm)
                simulation.context.reinitialize(preserveState=True)
                _set_orthorhombic_box(x_nm, y_nm, z_nm)
                state = simulation.context.getState(getPositions=True)
                pos_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
                if not np.isfinite(pos_nm).all():
                    raise RuntimeError(
                        "Particle coordinates became NaN after an OpenMM periodic "
                        "box-vector error during NPT compression."
                    ) from exc
                simulation.step(int(n_steps))
    else:
        print(f"WARNING: did not reach {target_nm} nm after {max_iterations} iterations")

    if manual_axes:
        if original_step_size is not None and hasattr(integrator, "setStepSize"):
            integrator.setStepSize(original_step_size)
        if original_friction is not None and hasattr(integrator, "setFriction"):
            integrator.setFriction(original_friction)
    _restore_hbond_parameters(hbond_restore_values)
    integrator.setIntegrationForceGroups(all_groups)
    return barostat_idx




def recenter_to_box_center(simulation, box_dims_nm):
    """
    Recenters the system coordinates to the geometric center of the periodic box.

    Parameters
    ----------
    simulation : openmm.app.Simulation
        Your running Simulation object.
    box_dims_nm : tuple of 3 floats
        The box lengths (x, y, z) in nanometers.
    """
    # 1) grab positions
    state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
    pos = state.getPositions(asNumpy=True)  # shape (N,3), Quantity nm
    pos_nm = pos.value_in_unit(unit.nanometer)  # plain ndarray (N,3)

    # 2) compute the new center from the provided box lengths
    #    box_dims_nm is e.g. (target_xy, target_xy, slab_z)
    center = np.array(box_dims_nm) / 2.0  # (3,)

    # 3) shift all positions so their centroid lands at `center`
    centroid = pos_nm.mean(axis=0)
    new_pos_nm = pos_nm - centroid + center

    # 4) write them back
    simulation.context.setPositions(new_pos_nm * unit.nanometer)


def shift_atoms_by_name(simulation, atom_names, shift_nm, axis="z", wrap=True):
    """
    Shift atoms with selected topology names along one axis.

    When a complete chain is selected, periodic wrapping is applied once to
    the chain as a whole so bonded atoms are not split across the box in the
    written coordinates.
    """
    axis_index = {"x": 0, "y": 1, "z": 2}[axis.lower()]
    selected_names = set(atom_names)
    selected = []
    chain_indices = []
    for chain in simulation.topology.chains():
        indices = [
            atom.index
            for residue in chain.residues()
            for atom in residue.atoms()
        ]
        selected_in_chain = [
            atom.index
            for residue in chain.residues()
            for atom in residue.atoms()
            if atom.name in selected_names
        ]
        selected.extend(selected_in_chain)
        if selected_in_chain:
            chain_indices.append((indices, selected_in_chain))
    if not selected:
        print(f"[shift_atoms_by_name] No atoms matched names: {atom_names}")
        return 0

    state = simulation.context.getState(getPositions=True, enforcePeriodicBox=False)
    pos_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    pos_nm[selected, axis_index] += float(shift_nm)

    if wrap:
        vectors = simulation.context.getState().getPeriodicBoxVectors()
        box_nm = np.array([
            np.sqrt(sum(v[i].value_in_unit(unit.nanometer) ** 2 for i in range(3)))
            for v in vectors
        ])
        box_length = box_nm[axis_index]
        for indices, selected_in_chain in chain_indices:
            if len(selected_in_chain) == len(indices):
                chain_center = pos_nm[indices, axis_index].mean()
                image = math.floor(chain_center / box_length)
                pos_nm[indices, axis_index] -= image * box_length
            else:
                pos_nm[selected_in_chain, axis_index] = np.mod(
                    pos_nm[selected_in_chain, axis_index],
                    box_length,
                )

    simulation.context.setPositions(pos_nm * unit.nanometer)
    return len(selected)


def pack_chains_in_elongated_box(
    simulation,
    xy_nm,
    z_nm,
    chain_spacing_nm=3.0,
    margin_nm=2.0,
    min_layer_gap_nm=None,
    placement_xy_nm=None,
    interleave_components=False,
):
    """
    Reposition complete chains into a narrow XY / long Z orthorhombic box.
    """
    xy_nm = float(xy_nm)
    z_nm = float(z_nm)
    placement_xy_nm = xy_nm if placement_xy_nm is None else float(placement_xy_nm)
    if placement_xy_nm > xy_nm:
        raise ValueError("placement_xy_nm cannot be larger than xy_nm")
    if min_layer_gap_nm is None:
        min_layer_gap_nm = chain_spacing_nm

    state = simulation.context.getState(getPositions=True, enforcePeriodicBox=False)
    pos_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    chains = list(simulation.topology.chains())
    if not chains:
        return (xy_nm, xy_nm, z_nm)

    chain_atom_indices = []
    chain_components = []
    for chain in chains:
        atoms = [atom for residue in chain.residues() for atom in residue.atoms()]
        indices = [atom.index for atom in atoms]
        chain_atom_indices.append(indices)
        atom_names = {atom.name for atom in atoms}
        chain_components.append("B" if atom_names.intersection({"B", "T"}) else "A")

    component_order = []
    for component in chain_components:
        if component not in component_order:
            component_order.append(component)

    component_groups = {
        component: [
            index for index, chain_component in enumerate(chain_components)
            if chain_component == component and chain_atom_indices[index]
        ]
        for component in component_order
    }

    segment_specs = []
    if interleave_components and {"A", "B"}.issubset(component_groups):
        mixed_entries = []
        for component_index, component in enumerate(("A", "B")):
            group = component_groups[component]
            mixed_entries.extend(
                (
                    (local_index + 0.5) / len(group),
                    component_index,
                    chain_index,
                )
                for local_index, chain_index in enumerate(group)
            )
        mixed_group = [
            chain_index
            for _, _, chain_index in sorted(mixed_entries)
        ]
        segment_specs = [("mixed", None, mixed_group)]
        print(
            "[pack_chains_in_elongated_box] Mixed XY packing: "
            f"{len(component_groups['A'])} A chains and "
            f"{len(component_groups['B'])} B chains in one shared layer"
        )
    else:
        segment_specs = [
            (component, None, group)
            for component, group in component_groups.items()
        ]

    # Keep every packing segment in one XY layer.  If the requested packing
    # area is too small, expand X/Y instead of splitting the segment into
    # multiple, spatially separated Z layers.
    max_segment_size = max(
        (len(group) for _, _, group in segment_specs),
        default=0,
    )
    required_grid_width = max(1, int(math.ceil(math.sqrt(max_segment_size))))
    required_placement_xy = (
        2.0 * margin_nm
        + (required_grid_width - 1) * chain_spacing_nm
    )
    if placement_xy_nm < required_placement_xy:
        print(
            "[pack_chains_in_elongated_box] Requested XY is too small for "
            "single-layer component packing. Expanding placement XY from "
            f"{placement_xy_nm:.3f} to {required_placement_xy:.3f} nm."
        )
        placement_xy_nm = required_placement_xy
    if xy_nm < placement_xy_nm:
        print(
            "[pack_chains_in_elongated_box] Expanding box XY from "
            f"{xy_nm:.3f} to {placement_xy_nm:.3f} nm."
        )
        xy_nm = placement_xy_nm

    usable_xy = max(placement_xy_nm - 2.0 * margin_nm, chain_spacing_nm)
    grid_width = max(1, int(math.floor(usable_xy / chain_spacing_nm)) + 1)

    layouts = []
    required_z = 2.0 * margin_nm
    for segment_index, (component, block_index, group) in enumerate(segment_specs):
        if not group:
            continue
        max_span_z = max(
            pos_nm[chain_atom_indices[index], 2].max() - pos_nm[chain_atom_indices[index], 2].min()
            for index in group
        )
        layer_chunks = [list(group)]
        n_layers = 1
        layer_spacing = 0.0
        block_height = max_span_z
        layouts.append({
            "component": component,
            "block_index": block_index,
            "indices": group,
            "layer_chunks": layer_chunks,
            "max_span_z": max_span_z,
            "n_layers": n_layers,
            "layer_spacing": layer_spacing,
            "block_height": block_height,
        })
        required_z += block_height
        if segment_index != len(segment_specs) - 1:
            required_z += float(min_layer_gap_nm)

    if z_nm < required_z:
        print(
            "[pack_chains_in_elongated_box] Requested Z is too small for "
            f"component-aware packing. Expanding Z from {z_nm:.3f} to {required_z:.3f} nm."
        )
        z_nm = required_z

    simulation.context.setPeriodicBoxVectors(
        Vec3(xy_nm, 0.0, 0.0) * unit.nanometer,
        Vec3(0.0, xy_nm, 0.0) * unit.nanometer,
        Vec3(0.0, 0.0, z_nm) * unit.nanometer,
    )

    z_cursor = margin_nm
    for layout in layouts:
        print(
            "[pack_chains_in_elongated_box] Component "
            f"{layout['component']}"
            f"{'' if layout['block_index'] is None else '-' + str(layout['block_index'])}: "
            f"{len(layout['indices'])} chains, "
            f"{layout['n_layers']} layers, span={layout['max_span_z']:.3f} nm, "
            f"layer_spacing={layout['layer_spacing']:.3f} nm, "
            f"layer_counts={[len(chunk) for chunk in layout['layer_chunks']]}"
        )
        for layer, layer_group in enumerate(layout["layer_chunks"]):
            for in_layer, chain_index in enumerate(layer_group):
                indices = chain_atom_indices[chain_index]
                grid_x = in_layer % grid_width
                grid_y = in_layer // grid_width
                target_min = np.array([
                    margin_nm + grid_x * chain_spacing_nm,
                    margin_nm + grid_y * chain_spacing_nm,
                    z_cursor + layer * layout["layer_spacing"],
                ])
                current_min = pos_nm[indices].min(axis=0)
                pos_nm[indices] += target_min - current_min
        z_cursor += layout["block_height"] + float(min_layer_gap_nm)

    pos_nm = np.mod(pos_nm, np.array([xy_nm, xy_nm, z_nm]))
    simulation.context.setPositions(pos_nm * unit.nanometer)
    return (xy_nm, xy_nm, z_nm)


def write_pdb_and_psf(simulation, pdb_filename, psf_filename, outdir):
    """
    Write the current positions of the simulation to PDB, and—
    if ParmEd is installed—also write a PSF. Otherwise skip PSF.

    Parameters
    ----------
    simulation : openmm.app.Simulation
        Your running Simulation object.
    pdb_filename : str
        Name of the output PDB file (e.g. 'snapshot.pdb').
    psf_filename : str
        Name of the output PSF file (e.g. 'snapshot.psf') — only used if parmed is present.
    outdir : str
        Directory path where files will be written.
    """
    os.makedirs(outdir, exist_ok=True)

    # always write PDB
    state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
    simulation.topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())
    positions = state.getPositions()
    pdb_path = os.path.join(outdir, pdb_filename)
    with open(pdb_path, 'w') as f:
        PDBFile.writeFile(simulation.topology, positions, f)
    print(f"PDB file saved to {pdb_path}")

    # skip PSF if ParmEd missing
    if pmd is None:
        print("ParmEd not installed; skipping PSF writing.")
        return

    # else, write PSF
    structure = pmd.openmm.load_topology(
        simulation.topology, simulation.system, xyz=positions
    )
    for atom in structure.atoms:
        atom.type = atom.type[:4]
        atom.name = atom.name[:4]
    psf_path = os.path.join(outdir, psf_filename)
    structure.save(psf_path, overwrite=True, format='psf')
    print(f"PSF file saved to {psf_path}")
