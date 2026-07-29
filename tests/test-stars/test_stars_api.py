"""Smoke tests for the public OpenABC STARS API.

Run from the repository root with:
    python tests/test-stars/test_stars_api.py
"""

from pathlib import Path
import tempfile
import unittest

import numpy as np
from openmm import unit

from openabc.forcefields import (
    STARS_1comp,
    STARS_1comp_from_npy,
    STARS_2comp,
    STARS_2comp_from_npy,
)


class STARSAPITest(unittest.TestCase):
    """Verify that STARS builds runnable OpenMM simulations via OpenABC."""

    platform_name = "Reference"

    def assert_finite_energy(self, simulation):
        energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
        self.assertTrue(np.isfinite(energy.value_in_unit(unit.kilojoule_per_mole)))

    def test_one_component_constructor(self):
        simulation = STARS_1comp(
            seq="101",
            nChains=2,
            initial_box=20.0,
            platform_name=self.platform_name,
        )

        self.assertEqual(simulation.topology.getNumAtoms(), 10)
        self.assertEqual(simulation.system.getNumForces(), 3)
        self.assert_finite_energy(simulation)

    def test_two_component_optional_forces(self):
        simulation = STARS_2comp(
            seqA="101",
            seqB="01",
            nA=1,
            nB=1,
            initial_box=20.0,
            include_hbonds=True,
            kr=-4.0,
            ka=-1.0,
            r0=1.0,
            kab=-1.0,
            kaa=-1.0,
            kbb=-1.0,
            include_spacers=True,
            alpha=2.0,
            tau=2.0,
            gamma=0.01,
            platform_name=self.platform_name,
        )

        self.assertEqual(simulation.topology.getNumAtoms(), 8)
        force_names = {force.getName() for force in simulation.system.getForces()}
        self.assertTrue(
            {
                "Class2BondPotential",
                "ExcludedVolumePotential",
                "HbondPotential-AB",
                "HbondPotential-AA",
                "HbondPotential-BB",
                "RandomSpacers",
            }.issubset(force_names)
        )
        self.assert_finite_energy(simulation)

    def test_numpy_coordinate_wrappers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            one_comp_reference = STARS_1comp(
                seq="101",
                nChains=1,
                initial_box=20.0,
                platform_name=self.platform_name,
            )
            one_comp_coords = np.zeros((one_comp_reference.topology.getNumAtoms(), 3))
            one_comp_coords[:, 0] = np.arange(one_comp_coords.shape[0]) * 10.0
            one_comp_path = tmpdir / "one-component-positions.npy"
            np.save(one_comp_path, one_comp_coords)

            one_comp = STARS_1comp_from_npy(
                seq="101",
                nChains=1,
                position_npy=one_comp_path,
                initial_box=20.0,
                platform_name=self.platform_name,
            )
            one_comp_positions = one_comp.context.getState(getPositions=True).getPositions(asNumpy=True)
            self.assertAlmostEqual(
                one_comp_positions[1][0].value_in_unit(unit.nanometer),
                1.0,
            )

            two_comp_reference = STARS_2comp(
                seqA="1",
                seqB="0",
                nA=1,
                nB=1,
                initial_box=20.0,
                platform_name=self.platform_name,
            )
            two_comp_coords = np.zeros((two_comp_reference.topology.getNumAtoms(), 3))
            two_comp_coords[:, 2] = np.arange(two_comp_coords.shape[0]) * 10.0
            two_comp_path = tmpdir / "two-component-positions.npy"
            np.save(two_comp_path, two_comp_coords)

            two_comp = STARS_2comp_from_npy(
                seqA="1",
                seqB="0",
                nA=1,
                nB=1,
                position_npy=two_comp_path,
                initial_box=20.0,
                platform_name=self.platform_name,
            )
            two_comp_positions = two_comp.context.getState(getPositions=True).getPositions(asNumpy=True)
            self.assertAlmostEqual(
                two_comp_positions[2][2].value_in_unit(unit.nanometer),
                2.0,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
