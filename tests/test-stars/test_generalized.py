"""Numerical regression tests. Set STARS_TEST_PACKAGE to stars or an API package.

Run with python -B test_generalized.py. No simulation output files are written.
"""
import gc
import importlib
import math
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from openmm import unit

package = importlib.import_module(os.environ.get('STARS_TEST_PACKAGE', 'openabc.forcefields'))
STARS = package.STARS


def state(simulation, groups=-1):
    snapshot = simulation.context.getState(getEnergy=True, getForces=True, groups=groups)
    return (snapshot.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
            snapshot.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer))


class GeneralizedSTARS(unittest.TestCase):
    def tearDown(self):
        gc.collect()

    def build(self, n_comp=3, sequences=None, n_chains=None, **kwargs):
        return STARS(n_comp, sequences or ['101'] * n_comp,
                     n_chains or [1] * n_comp, platform_name='Reference',
                     initial_box=30.0, **kwargs)

    def test_legacy_energies_and_forces(self):
        # All potential terms, random disorder, exclusions and selectors must
        # agree at identical nondegenerate coordinates, separately by term.
        for n_comp in (1, 2):
            for restricted in (False, True):
                with self.subTest(n_comp=n_comp, restricted=restricted):
                    parameters = dict(initial_box=30.0, platform_name='Reference',
                                      kr=-2.0, ka=-5.0, r0=0.0, kaa=-1.1,
                                      kab=-0.7, kbb=-1.7, alpha=4.5, tau=1.5, gamma=1e-4,
                                      cutoff_distance=2**(1/6), friction_coeff=0.1, timestep=1.0,
                                      include_hbonds=True, include_spacers=True,
                                      mean_eps_AA=-0.3, std_eps_AA=0.2,
                                      mean_eps_AB=-0.4, std_eps_AB=0.1,
                                      mean_eps_BB=-0.6, std_eps_BB=0.3)
                    selectors = {(0, 0): '1::2'} if restricted else {}
                    old_selectors = {('S', 'S'): '1::2'} if restricted else {}
                    if n_comp == 2 and restricted:
                        selectors.update({(0, 1): '::2', (1, 1): [0]})
                        old_selectors.update({('S', 'T'): '::2', ('T', 'T'): [0]})
                    np.random.seed(123)
                    if n_comp == 1:
                        old = package.STARS_1comp('101', 2, selector=old_selectors, **parameters)
                    else:
                        old = package.STARS_2comp('101', '101', 1, 1, selector=old_selectors, **parameters)
                    coords = old.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
                    coords[5:, 0] -= 2.6
                    coords += np.random.RandomState(14).normal(0, 0.08, coords.shape)
                    old.context.setPositions(coords)
                    pairs = {(0, 0): -1.1}
                    means, stds = {(0, 0): -0.3}, {(0, 0): 0.2}
                    if n_comp == 2:
                        pairs.update({(0, 1): -0.7, (1, 1): -1.7})
                        means.update({(0, 1): -0.4, (1, 1): -0.6})
                        stds.update({(0, 1): 0.1, (1, 1): 0.3})
                    new = self.build(n_comp, n_chains=[2] if n_comp == 1 else [1, 1],
                                     include_hbonds=True, include_spacers=True,
                                     sticker_strengths=pairs, spacer_means=means, spacer_stds=stds,
                                     selectors=selectors, seed=123, positions_nm=coords)
                    for old_groups, new_groups in [(2, 2), (4, 4), (56, 8), (64, 64), (-1, -1)]:
                        old_e, old_f = state(old, old_groups)
                        new_e, new_f = state(new, new_groups)
                        np.testing.assert_allclose(new_e, old_e, rtol=1e-11, atol=1e-9)
                        np.testing.assert_allclose(new_f, old_f, rtol=1e-10, atol=1e-8)
                    del old, new

    def test_three_component_pair_energies_and_gradients(self):
        # Every cross component pair, including 1-2, must use its own strength.
        positions = np.array([[-1., 0, 0], [0, 0, 0],
                              [1.7, .1, 0], [.7, .1, 0],
                              [.2, 1.5, .2], [.2, .5, .2]])
        strengths = {(0, 1): -0.7, (0, 2): -1.2, (1, 2): -2.1}
        sim = self.build(3, ['1'] * 3, include_hbonds=True,
                         sticker_strengths=strengths, positions_nm=positions)
        def angle(a, b, c):
            u, v = a-b, c-b
            return math.acos(np.clip(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))
        expected = 0
        for (i, j), k in strengths.items():
            a1, a2, d1, d2 = positions[2*i+1], positions[2*i], positions[2*j+1], positions[2*j]
            expected += k * math.exp(-2*np.sum((d1-a1)**2)
                                    -5*(angle(d2, d1, a2)-math.pi)**2
                                    -5*(angle(a2, a1, d2)-math.pi)**2)
        energy, forces = state(sim, 8)
        self.assertAlmostEqual(energy, expected, places=12)
        h = 1e-5
        for atom in range(6):
            for axis in range(3):
                plus, minus = positions.copy(), positions.copy()
                plus[atom, axis] += h
                minus[atom, axis] -= h
                sim.context.setPositions(plus)
                ep = state(sim, 8)[0]
                sim.context.setPositions(minus)
                em = state(sim, 8)[0]
                self.assertAlmostEqual(forces[atom, axis], -(ep-em)/(2*h), places=6)

    def test_same_component_counting(self):
        positions = [[-1, 0, 0], [0, 0, 0], [1.7, 0, 0], [.7, 0, 0]]
        sim = self.build(1, ['1'], [2], include_hbonds=True,
                         sticker_strengths={(0, 0): -1.3}, positions_nm=positions)
        self.assertAlmostEqual(state(sim, 8)[0], -2*1.3*math.exp(-2*.7**2), places=12)

    def test_ten_components_no_force_group_limit(self):
        sim = self.build(10, ['01'] * 10, include_hbonds=True,
                         sticker_strengths={(0, 9): -1}, include_spacers=True,
                         spacer_means={(0, 9): -0.1}, seed=3)
        self.assertEqual(len(sim.stars_components), 10)
        self.assertEqual(sim.system.getNumForces(), 59)
        energy, force = state(sim)
        self.assertTrue(np.isfinite(energy) and np.isfinite(force).all())
        sim.context.setVelocitiesToTemperature(1*unit.kelvin, 3)
        sim.step(3)
        self.assertTrue(np.isfinite(state(sim)[0]))

    def test_spacer_pair_matrix_and_exclusions(self):
        sim = self.build(3, include_spacers=True,
                         spacer_means={(0, 2): -0.4, (1, 2): -0.8}, seed=11)
        force = next(f for f in sim.system.getForces() if f.getName() == 'RandomSpacers')
        nx, ny, values = force.getTabulatedFunction(0).getFunctionParameters()
        eps = np.array(values).reshape(nx, ny)
        c = sim.stars_components
        np.testing.assert_equal(eps[np.ix_(c[0]['backbone'], c[2]['backbone'])], -0.4)
        np.testing.assert_equal(eps[np.ix_(c[1]['backbone'], c[2]['backbone'])], -0.8)
        np.testing.assert_equal(eps[np.ix_(c[0]['backbone'], c[1]['backbone'])], 0)
        np.testing.assert_equal(eps, eps.T)
        for component in c:
            np.testing.assert_equal(eps[list(component['stickers'])], 0)
        self.assertEqual(force.getNumExclusions(), sim.topology.getNumBonds())

    def test_selectors_both_sides_and_disabled_pair(self):
        sim = self.build(selectors={(0, 1): {'i': [0], 'j': '1::2'}, (1, 2): []}, include_hbonds=True)
        forces = {f.getName(): f for f in sim.system.getForces()}
        cross = forces['HbondPotential-0-1']
        self.assertEqual(cross.getNumDonors(), 1)
        self.assertEqual(cross.getNumAcceptors(), 1)
        self.assertEqual(forces['HbondPotential-1-2'].getNumAcceptors(), 0)

    def test_input_validation(self):
        cases = [dict(n_comp=2.0), dict(n_comp=True), dict(n_comp=0),
                 dict(sequences=['01']), dict(sequences=['012']*3),
                 dict(n_chains=[1., 1, 1]), dict(n_chains=[0, 0, 0]),
                 dict(kr=0), dict(kr=2), dict(ka=1), dict(alpha=4),
                 dict(gamma=0), dict(hbond_gamma=1), dict(timestep=-1),
                 dict(sticker_strengths={(2, 1): -1}),
                 dict(sticker_strengths={(0, 3): -1}), dict(spacer_stds={(0, 1): -1}),
                 dict(selectors={(0, 1): '::0'}), dict(selectors={(0, 1): [99]}),
                 dict(selectors={(0, 0): {'i': 'all'}}), dict(positions_nm=[[0, 0, 0]])]
        for params in cases:
            with self.subTest(params=params), self.assertRaises((TypeError, ValueError)):
                self.build(**params)

    def test_hbond_gamma_and_coordinates(self):
        sim = self.build(1, ['01'], hbond_gamma=1e-4, include_hbonds=True)
        hb = next(f for f in sim.system.getForces() if f.getName().startswith('Hbond'))
        self.assertAlmostEqual(hb.getCutoffDistance().value_in_unit(unit.nanometer), math.sqrt(math.log(1e-4)/-2))
        del sim
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'coords.npy'
            coordinates = np.array([[0, 0, 0], [0, 0, 10], [10, 0, 10]])
            np.save(path, coordinates)
            sim = package.STARS_from_npy(1, ['01'], [1], path, platform_name='Reference')
            actual = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            np.testing.assert_allclose(actual, coordinates/10)


if __name__ == '__main__':
    unittest.main(verbosity=2)
