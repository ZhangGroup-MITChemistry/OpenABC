"""Run a minimal two-component STARS simulation through the OpenABC API.

For example:
    python run_2comp_api.py
    python run_2comp_api.py --n-a 16 --n-b 16 --steps 10000 --output output-2comp

Both sequences are binary: ``0`` is a spacer and ``1`` is a sticker.  The
three h-bond strengths respectively control A-A, A-B, and B-B sticker pairs.
"""

import argparse
from pathlib import Path

import openmm.app as app
from openmm import unit

from openabc.forcefields import STARS_2comp


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-a", default="0010010010001")
    parser.add_argument("--sequence-b", default="0100100100100")
    parser.add_argument("--n-a", type=int, default=4, help="Number of A-component chains.")
    parser.add_argument("--n-b", type=int, default=4, help="Number of B-component chains.")
    parser.add_argument("--box-length", type=float, default=40.0, help="Initial cubic box length in nm.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Reduced STARS temperature.")
    parser.add_argument("--friction", type=float, default=1.0, help="Langevin friction in ps^-1.")
    parser.add_argument("--timestep", type=float, default=2.0, help="Integration timestep in fs.")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--report-interval", type=int, default=100)
    parser.add_argument("--aa-strength", type=float, default=-1.0, help="A-A sticker attraction in kBT.")
    parser.add_argument("--ab-strength", type=float, default=-1.0, help="A-B sticker attraction in kBT.")
    parser.add_argument("--bb-strength", type=float, default=-1.0, help="B-B sticker attraction in kBT.")
    parser.add_argument("--no-hbonds", dest="include_hbonds", action="store_false")
    parser.set_defaults(include_hbonds=True)
    parser.add_argument("--platform", default="CPU", help="OpenMM platform, e.g. CPU, CUDA, OpenCL, Reference.")
    parser.add_argument("--output", type=Path, default=Path("output-2comp"))
    return parser.parse_args()


def write_pdb(simulation, output_path):
    state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
    simulation.topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())
    with output_path.open("w") as handle:
        app.PDBFile.writeFile(simulation.topology, state.getPositions(), handle)


def main():
    args = parse_args()
    if args.n_a < 0 or args.n_b < 0 or args.n_a + args.n_b < 1:
        raise ValueError("--n-a and --n-b must be non-negative, with at least one total chain")
    if args.steps < 0 or args.report_interval < 1:
        raise ValueError("--steps must be non-negative and --report-interval must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    simulation = STARS_2comp(
        seqA=args.sequence_a,
        seqB=args.sequence_b,
        nA=args.n_a,
        nB=args.n_b,
        integrator_type="Langevin",
        temperature=args.temperature,
        friction_coeff=args.friction,
        timestep=args.timestep,
        include_hbonds=args.include_hbonds,
        kr=-4.0,
        ka=-1.0,
        r0=1.0,
        kaa=args.aa_strength,
        kab=args.ab_strength,
        kbb=args.bb_strength,
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
