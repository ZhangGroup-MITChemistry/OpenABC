"""Run a minimal one-component STARS simulation through the OpenABC API.

For example:
    python run_1comp_api.py
    python run_1comp_api.py --n-chains 16 --steps 10000 --output output-1comp

The sequence is binary: ``0`` is a spacer and ``1`` is a sticker.  STARS
parameters are expressed in reduced units, so the default temperature of 1.0
corresponds to one kJ mol^-1 of thermal energy.
"""

import argparse
from pathlib import Path

import openmm.app as app
from openmm import unit

from openabc.forcefields import STARS_1comp


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", default="0010010010001")
    parser.add_argument("--n-chains", type=int, default=10)
    parser.add_argument("--box-length", type=float, default=30.0, help="Initial cubic box length in nm.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Reduced STARS temperature.")
    parser.add_argument("--friction", type=float, default=0.1, help="Langevin friction in ps^-1.")
    parser.add_argument("--timestep", type=float, default=1.0, help="Integration timestep in fs.")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--report-interval", type=int, default=100)
    parser.add_argument("--hbond-strength", type=float, default=-1.0, help="A-A sticker attraction in kBT.")
    parser.add_argument("--no-hbonds", dest="include_hbonds", action="store_false")
    parser.set_defaults(include_hbonds=True)
    parser.add_argument("--platform", default="CPU", help="OpenMM platform, e.g. CPU, CUDA, OpenCL, Reference.")
    parser.add_argument("--output", type=Path, default=Path("output-1comp"))
    return parser.parse_args()


def write_pdb(simulation, output_path):
    state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
    simulation.topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())
    with output_path.open("w") as handle:
        app.PDBFile.writeFile(simulation.topology, state.getPositions(), handle)


def main():
    args = parse_args()
    if args.n_chains < 1:
        raise ValueError("--n-chains must be at least 1")
    if args.steps < 0 or args.report_interval < 1:
        raise ValueError("--steps must be non-negative and --report-interval must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    simulation = STARS_1comp(
        seq=args.sequence,
        nChains=args.n_chains,
        integrator_type="Langevin",
        temperature=args.temperature,
        friction_coeff=args.friction,
        timestep=args.timestep,
        cutoff_distance=2**(1/6),
        include_hbonds=args.include_hbonds,
        kr=-2.0,
        ka=-5.0,
        r0=0.0,
        kaa=args.hbond_strength,
        alpha=4.5,
        initial_box=args.box_length,
        platform_name=args.platform,
    )

    simulation.minimizeEnergy(maxIterations=1000)
    write_pdb(simulation, args.output / "initial.pdb")

    simulation.reporters.append(
        app.DCDReporter(str(args.output / "trajectory.dcd"), args.report_interval, enforcePeriodicBox=True)
    )
    simulation.reporters.append(
        app.StateDataReporter(
            str(args.output / "simulation.log"),
            args.report_interval,
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            temperature=True,
            speed=True,
            totalSteps=args.steps,
            separator="\t",
        )
    )

    temperature = args.temperature * unit.kilojoule_per_mole / (
        unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
    )
    simulation.context.setVelocitiesToTemperature(temperature)
    simulation.step(args.steps)
    write_pdb(simulation, args.output / "final.pdb")
    print(f"Finished {args.steps} steps. Output written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
