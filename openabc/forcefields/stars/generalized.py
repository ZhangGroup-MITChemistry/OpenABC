"""STARS with an explicit, arbitrary number of components.

Component IDs are zero-based integers. Pair dictionaries use canonical (i, j)
keys with i <= j. Energies retain the legacy STARS conventions, including
ordered same-component H-bond pairs and per-particle-pair spacer disorder.
"""

import math
from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np
import openmm as mm
from openmm import app, unit

from .utils import add_class2bond_forces, add_excluded_volume_forces, T


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


def _number(value, name, minimum=None, strict=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number in model units")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and (value <= minimum if strict else value < minimum):
        raise ValueError(f"{name} must be {'>' if strict else '>='} {minimum}")
    return value


def _pairs(values, n_comp, name, nonnegative=False, numeric=True):
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a dictionary keyed by (i, j)")
    result = {}
    for pair, value in values.items():
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError(f"{name} keys must be (i, j) tuples")
        i, j = (_integer(x, f"{name} component ID") for x in pair)
        if not 0 <= i <= j < n_comp:
            raise ValueError(f"{name} pair {pair} must satisfy 0 <= i <= j < n_comp")
        result[i, j] = (_number(value, f"{name}[{pair}]", 0 if nonnegative else None)
                        if numeric else value)
    return result


def _select(indices, rule):
    """Select ordinals in the component-wide sticker list (not atom indices)."""
    if rule is None or (isinstance(rule, str) and rule == 'all'):
        return list(indices)
    if isinstance(rule, Integral) and not isinstance(rule, (bool, np.bool_)):
        return list(indices[::_integer(rule, 'selector stride', 1)])
    if isinstance(rule, str):
        if ':' not in rule:
            raise ValueError("selector must be 'all', a slice such as '1::2', or indices")
        try:
            parts = [int(p) if p.strip() else None for p in rule.split(':')]
            if len(parts) not in (2, 3) or (len(parts) == 3 and parts[2] == 0):
                raise ValueError()
            return list(indices[slice(*parts)])
        except ValueError as exc:
            raise ValueError(f"Invalid selector slice: {rule!r}") from exc
    if isinstance(rule, (list, tuple)):
        ordinals = [_integer(x, 'selector index') for x in rule]
        if len(set(ordinals)) != len(ordinals) or any(x >= len(indices) for x in ordinals):
            raise ValueError('selector indices must be unique and within the sticker list')
        return [indices[x] for x in ordinals]
    raise TypeError('Unsupported selector; use all, a slice, positive stride, or index list')


def _topology(sequences, n_chains):
    topology = app.Topology()
    carbon = app.element.Element.getBySymbol('C')
    positions, components = [], []
    width = int(math.ceil(math.sqrt(sum(n_chains))))
    chain_number = 0
    for component, (sequence, count) in enumerate(zip(sequences, n_chains)):
        # Preserve legacy A/S and B/T names; all later names remain compatible
        # with the excluded-volume helper's first-character role detection.
        backbone_name = 'A' if component == 0 else ('B' if component == 1 else f'A{component}')
        sticker_name = 'S' if component == 0 else ('T' if component == 1 else f'S{component}')
        backbone, stickers = [], []
        for _ in range(count):
            chain = topology.addChain()
            residue = topology.addResidue('MOL', chain)
            previous = None
            x, y = (chain_number % width) * 4.0, (chain_number // width) * 4.0
            for z, flag in enumerate(sequence):
                atom = topology.addAtom(backbone_name, carbon, residue)
                backbone.append(atom.index)
                positions.append([x, y, float(z)])
                if previous is not None:
                    topology.addBond(previous, atom)
                previous = atom
                if flag == '1':
                    sticker = topology.addAtom(sticker_name, carbon, residue)
                    stickers.append(sticker.index)
                    positions.append([x + 1.0, y, float(z)])
                    topology.addBond(atom, sticker)
            chain_number += 1
        components.append({'backbone': tuple(backbone), 'stickers': tuple(stickers)})
    return topology, np.asarray(positions), components


def STARS(n_comp, sequences, n_chains, *, sticker_strengths=None,
          spacer_means=None, spacer_stds=None, selectors=None,
          include_hbonds=False, include_spacers=False,
          kr=-2.0, ka=-5.0, r0=0.0, hbond_gamma=1e-6,
          alpha=4.5, tau=1.5, gamma=1e-4,
          integrator_type='Langevin', temperature=1.0, friction_coeff=0.1,
          timestep=1.0, cutoff_distance=2**(1/6), initial_box=None,
          padding=2.5, platform_name=None, seed=None, positions_nm=None):
    """Build a generalized STARS Simulation; see GENERALIZED.md for examples.

    n_comp must be a positive integer (not a bool or float). sequences and
    n_chains each contain n_comp entries. Missing pair strengths/means/stds are
    zero; missing selectors use all stickers. A cross-pair scalar selector
    selects the lower-index component only, as in legacy S/T selection. To
    select both sides use {'i': rule_i, 'j': rule_j}; a diagonal pair accepts
    one rule applied to both donor and acceptor lists.

    alpha is fixed at 4.5 in this new interface. hbond_gamma is independent of
    spacer gamma. Energy parameters use the fixed 1 kJ/mol reference energy,
    not the instantaneous thermal energy at the requested temperature.
    """
    n_comp = _integer(n_comp, 'n_comp', 1)
    if isinstance(sequences, str) or not isinstance(sequences, (list, tuple)):
        raise TypeError('sequences must be a list or tuple of binary strings')
    if not isinstance(n_chains, (list, tuple)):
        raise TypeError('n_chains must be a list or tuple of integers')
    if len(sequences) != n_comp or len(n_chains) != n_comp:
        raise ValueError('sequences and n_chains must each have exactly n_comp entries')
    if any(not isinstance(s, str) or not s or set(s) - {'0', '1'} for s in sequences):
        raise ValueError('each sequence must be a nonempty binary string containing only 0/1')
    counts = [_integer(n, f'n_chains[{i}]') for i, n in enumerate(n_chains)]
    if not sum(counts):
        raise ValueError('at least one chain is required')
    for flag, name in [(include_hbonds, 'include_hbonds'), (include_spacers, 'include_spacers')]:
        if not isinstance(flag, bool):
            raise TypeError(f'{name} must be bool')
    strengths = _pairs(sticker_strengths, n_comp, 'sticker_strengths')
    means = _pairs(spacer_means, n_comp, 'spacer_means')
    stds = _pairs(spacer_stds, n_comp, 'spacer_stds', nonnegative=True)
    selections = _pairs(selectors, n_comp, 'selectors', numeric=False)
    kr, ka = _number(kr, 'kr'), _number(ka, 'ka')
    if kr >= 0 or ka > 0:
        raise ValueError('kr must be negative and ka must be <= 0')
    r0 = _number(r0, 'r0', 0)
    for value, name in [(hbond_gamma, 'hbond_gamma'), (gamma, 'gamma')]:
        if not 0 < _number(value, name) < 1:
            raise ValueError(f'{name} must be between 0 and 1, exclusively')
    if _number(alpha, 'alpha') != 4.5:
        raise ValueError('the generalized model uses alpha=4.5')
    tau = _number(tau, 'tau', 0)
    temperature = _number(temperature, 'temperature', 0, strict=True)
    friction_coeff = _number(friction_coeff, 'friction_coeff', 0)
    timestep = _number(timestep, 'timestep', 0, strict=True)
    cutoff_distance = _number(cutoff_distance, 'cutoff_distance', 0, strict=True)
    padding = _number(padding, 'padding', 0, strict=True)
    if integrator_type not in ('Langevin', 'Verlet'):
        raise ValueError('integrator_type must be Langevin or Verlet')
    if seed is not None:
        seed = _integer(seed, 'seed')
        if seed > 2**31 - 1:
            raise ValueError('seed must be <= 2**31 - 1')
    hb_cut = r0 + math.sqrt(math.log(hbond_gamma) / kr)
    spacer_cut = tau - math.atanh(2 * gamma - 1) / alpha
    if include_spacers and spacer_cut <= 0:
        raise ValueError('spacer cutoff must be positive; check tau and gamma')
    topology, positions, components = _topology(sequences, counts)
    # Validate selectors even if H-bonds are currently disabled.
    selected = {}
    for i in range(n_comp):
        for j in range(i, n_comp):
            rule = selections.get((i, j))
            if isinstance(rule, Mapping):
                if i == j or set(rule) - {'i', 'j'}:
                    raise ValueError("side-specific selectors require i < j and keys 'i'/'j'")
                left, right = rule.get('i'), rule.get('j')
            else:
                left, right = rule, (rule if i == j else None)
            selected[i, j] = (_select(components[i]['stickers'], left),
                              _select(components[j]['stickers'], right))
    if positions_nm is not None:
        positions = np.asarray(positions_nm, dtype=float)
        if positions.shape != (topology.getNumAtoms(), 3) or not np.isfinite(positions).all():
            raise ValueError('positions_nm must be a finite (number_of_atoms, 3) array')
    largest_cutoff = max(cutoff_distance, hb_cut if include_hbonds else 0,
                         spacer_cut if include_spacers else 0)
    if initial_box is None:
        # Retain legacy heuristic, but ensure long chains and all cutoffs fit.
        box = max(sum(counts) * padding + 10,
                  float(np.ptp(positions, axis=0).max()) + 2 * padding,
                  2 * largest_cutoff + padding)
    else:
        box = _number(initial_box, 'initial_box', 0, strict=True)
        if box < 2 * largest_cutoff:
            raise ValueError('initial_box must be at least twice every active force cutoff')
    system = mm.System()
    for _ in topology.atoms():
        system.addParticle(1.0)
    vectors = tuple(mm.Vec3(*(box if k == axis else 0 for k in range(3))) * unit.nanometer
                    for axis in range(3))
    system.setDefaultPeriodicBoxVectors(*vectors)
    topology.setPeriodicBoxVectors(vectors)
    add_class2bond_forces(system, topology)
    add_excluded_volume_forces(system, topology, cutoff=cutoff_distance)
    if include_hbonds:
        for (i, j), (left, right) in selected.items():
            parameter = f'k_hb_{i}_{j}'
            expr = (f'{parameter}*exp(kr*(distance(d1,a1)-r0)^2 + '
                    'ka*(angle(d2,d1,a2)-theta0)^2 + '
                    'ka*(angle(a2,a1,d2)-theta0)^2)')
            force = mm.CustomHbondForce(expr)
            force.setName(f'HbondPotential-{i}-{j}')
            # All sticker forces share a group: no 32-group component limit.
            force.setForceGroup(3)
            force.setNonbondedMethod(force.CutoffPeriodic)
            force.setCutoffDistance(hb_cut)
            for key, val in [('kr', kr), ('ka', ka), ('r0', r0),
                             ('theta0', math.pi), (parameter, strengths.get((i, j), 0.0))]:
                force.addGlobalParameter(key, val)
            for atom in right:
                force.addDonor(atom, atom - 1, -1)
            for atom in left:
                force.addAcceptor(atom, atom - 1, -1)
            if i == j:
                for index in range(len(left)):
                    force.addExclusion(index, index)
            system.addForce(force)
    if include_spacers:
        rng = np.random if seed is None else np.random.RandomState(seed)
        size = topology.getNumAtoms()
        epsilon = np.zeros((size, size))
        # Draw diagonals first, then cross blocks, matching legacy RNG order
        # exactly for n_comp=1/2 and keeping disorder symmetric for all pairs.
        for i in range(n_comp):
            indices = components[i]['backbone']
            block = rng.normal(means.get((i, i), 0), stds.get((i, i), 0),
                               (len(indices), len(indices)))
            epsilon[np.ix_(indices, indices)] = np.triu(block) + np.triu(block, 1).T
        for i in range(n_comp):
            for j in range(i + 1, n_comp):
                a, b = components[i]['backbone'], components[j]['backbone']
                block = rng.normal(means.get((i, j), 0), stds.get((i, j), 0), (len(a), len(b)))
                epsilon[np.ix_(a, b)] = block
                epsilon[np.ix_(b, a)] = block.T
        force = mm.CustomNonbondedForce(
            '0.5*epsilon(pindex1,pindex2)*(1+tanh(alpha*(tau-r))-2*gamma)')
        force.setName('RandomSpacers')
        force.setForceGroup(6)
        force.addPerParticleParameter('pindex')
        force.addTabulatedFunction('epsilon', mm.Discrete2DFunction(size, size, epsilon.ravel().tolist()))
        for name, value in [('alpha', alpha), ('tau', tau), ('gamma', gamma)]:
            force.addGlobalParameter(name, value)
        for atom in topology.atoms():
            force.addParticle([atom.index])
        force.setNonbondedMethod(force.CutoffPeriodic)
        force.setCutoffDistance(spacer_cut)
        force.createExclusionsFromBonds([(a.index, b.index) for a, b in topology.bonds()], 1)
        backbone = [x for c in components for x in c['backbone']]
        force.addInteractionGroup(backbone, backbone)
        system.addForce(force)
    cm = mm.CMMotionRemover()
    cm.setForceGroup(0)
    system.addForce(cm)
    if integrator_type == 'Langevin':
        integrator = mm.LangevinIntegrator(temperature * T, friction_coeff / unit.picosecond,
                                           timestep * unit.femtosecond)
        if seed is not None:
            integrator.setRandomNumberSeed(seed)
    else:
        integrator = mm.VerletIntegrator(timestep * unit.femtosecond)
    platform = mm.Platform.getPlatformByName(platform_name) if platform_name else None
    simulation = app.Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(positions * unit.nanometer)
    simulation.stars_components = tuple(components)
    return simulation


def STARS_from_npy(n_comp, sequences, n_chains, position_npy, *,
                   coordinate_unit='angstrom', **kwargs):
    """Load (N, 3) positions; legacy files default to angstrom, converted to nm."""
    if coordinate_unit not in ('angstrom', 'nm'):
        raise ValueError("coordinate_unit must be 'angstrom' or 'nm'")
    if 'positions_nm' in kwargs:
        raise ValueError('use either position_npy or positions_nm')
    positions = np.load(position_npy, allow_pickle=False)
    positions = np.asarray(positions, dtype=float) / (10.0 if coordinate_unit == 'angstrom' else 1.0)
    return STARS(n_comp, sequences, n_chains, positions_nm=positions, **kwargs)
