"""Minimal generalized STARS simulation. Run with --help for composition options.

Example: python run_multicomp_api.py --n-comp 3 --sequences 001001 010010 100100 \
             --n-chains 2 2 2 --steps 100

Edit the clearly labeled interaction defaults and optional pair overrides below.
Same-component sticker strengths default to -10; cross-component strengths,
spacer means, and spacer deltas default to zero. All stickers are eligible.
"""
import argparse

from openabc.forcefields import STARS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-comp', type=int, default=3)
    parser.add_argument('--sequences', nargs='+', default=None)
    parser.add_argument('--n-chains', nargs='+', type=int, default=None)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--platform', default='CPU')
    args = parser.parse_args()
    if args.steps < 0:
        parser.error('--steps must be nonnegative')
    sequences = args.sequences if args.sequences is not None else ['001001'] * args.n_comp
    counts = args.n_chains if args.n_chains is not None else [2] * args.n_comp

    # Component IDs: A=0, B=1, C=2, D=3, ...
    # List every unique pair once: i <= j. For 10 components, this gives
    # 55 pairs (10 same-component pairs and 45 cross-component pairs).
    pairs = [
        (i, j)
        for i in range(args.n_comp)
        for j in range(i, args.n_comp)
    ]

    # DEFAULTS: edit these four values to change the whole interaction table.
    same_component_sticker_strength = -10.0
    cross_component_sticker_strength = 0.0
    default_spacer_mean = 0.0
    default_spacer_delta = 0.0  # Standard deviation, not variance.

    sticker_strengths = {
        (i, j): (same_component_sticker_strength if i == j
                 else cross_component_sticker_strength)
        for i, j in pairs
    }
    spacer_means = {pair: default_spacer_mean for pair in pairs}
    spacer_stds = {pair: default_spacer_delta for pair in pairs}
    selectors = {}  # Missing rules mean ALL stickers; do not use numeric 0.

    # OPTIONAL PAIR OVERRIDES: uncomment/edit only the settings you want.
    # Later assignments replace the defaults above for that pair.
    # To give EVERY pair sticker strength -10, set both sticker defaults to -10.
    #
    # selectors[0, 0] = "1::2"  # A-A: same selected A stickers on both sides.
    # if args.n_comp >= 2:
    #     sticker_strengths[0, 1] = -17.0  # A-B
    #     selectors[0, 1] = {"i": "::2", "j": "all"}
    #     # i is A and j is B in this pair.
    # if args.n_comp >= 3:
    #     sticker_strengths[0, 2] = -17.0  # A-C
    #     selectors[0, 2] = {"i": "::2", "j": "all"}
    #     # i is A and j is C here; AB/AC share the same A sticker pool.
    #     sticker_strengths[1, 2] = 0.0    # No B-C sticker attraction.
    #     spacer_means[1, 2] = -0.5       # Optional B-C spacer attraction.
    #     spacer_stds[1, 2] = 0.0
    #
    # For the three-component setup, e.g. set A-A=-15 and B-B=C-C=0:
    # sticker_strengths[0, 0] = -15.0
    # if args.n_comp >= 2:
    #     sticker_strengths[1, 1] = 0.0
    # if args.n_comp >= 3:
    #     sticker_strengths[2, 2] = 0.0
    #
    # Selector rules: "all", "::2" (0,2,4,...), "1::2" (1,3,5,...),
    # or an ordinal list such as [0,3,5]. [] selects no stickers.
    # Ordinals span all chains of a component; they do not reset per chain.


    simulation = STARS(
        n_comp=args.n_comp, sequences=sequences, n_chains=counts,
        sticker_strengths=sticker_strengths, selectors=selectors,
        spacer_means=spacer_means, spacer_stds=spacer_stds,
        include_hbonds=True, include_spacers=True,
        kr=-2.0, ka=-5.0, r0=0.0, hbond_gamma=1e-6,
        alpha=4.5, tau=1.5, gamma=1e-4,
        temperature=1.0, friction_coeff=0.1, timestep=1.0,
        seed=123, platform_name=args.platform,
    )
    simulation.minimizeEnergy(maxIterations=1000)
    # Temperature 1 in STARS corresponds to 1 kJ/mol thermal energy.
    simulation.context.setVelocitiesToTemperature(simulation.integrator.getTemperature(), 123)
    simulation.step(args.steps)
    energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    print(f'{args.n_comp} components; {simulation.system.getNumParticles()} particles; energy={energy}')


if __name__ == '__main__':
    main()
