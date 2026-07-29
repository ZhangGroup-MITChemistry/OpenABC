import openmm as mm
import openmm.app as app
from openmm import unit
from openmm.openmm import CMMotionRemover
import numpy as np
from openmm.openmm import Vec3, MonteCarloBarostat
from openmm.app    import StateDataReporter, Simulation
from openmm import VerletIntegrator
from typing import Optional, Tuple
from .utils import (
    parse_sequence,
    build_topology_positions,
    build_topology_positions_2comp,
    add_class2bond_forces,
    add_excluded_volume_forces,
    add_hbond_forces,
    add_random_spacer_forces,
)

T = 1 * unit.kilojoule_per_mole / \
    (unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA)
k_BT = T * unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
length_scale = 1.0 * unit.nanometer


def STARS_1comp(
    seq: str,
    nChains: int,
#    initial_box: float,
    integrator_type: str = 'Langevin',
    temperature: float = 1.0,
    friction_coeff: float = 1.0,
    timestep: float = 2.0,
    cutoff_distance: float = 1.0,
    include_hbonds: bool = False,
    include_spacers: bool = False,
    kr: float = None,
    ka: float = None,
    kab: float = 0.0,
    kaa: float = 0.0,
    kbb: float = 0.0,
    r0: float = None,
    selector: dict = None,
    # spacer‐epsilon parameters
    mean_eps_AA: float = 0.0,
    std_eps_AA: float = 0.0,
    mean_eps_AB: float = 0.0,
    std_eps_AB: float = 0.0,
    mean_eps_BB: float = 0.0,
    std_eps_BB: float = 0.0,
    alpha: float = None,
    tau: float = None,
    gamma: float = None,
    platform_name: str = None,
    initial_box: Optional[float] = None,
    padding: float = 2.5,
) -> app.Simulation:
    # Build topology and positions
    bead_types = parse_sequence(seq, chain='A')
    topology, positions = build_topology_positions(bead_types, nChains)

    # Create system
    system = mm.System()
    for _ in topology.atoms():
        system.addParticle(1.0)

    if initial_box is None:
        # simple heuristic: one padding length per chain
        L = (nChains * padding + 10 ) * unit.nanometer
        print(f"Auto-estimated initial box (1comp): {L} "
              f"(nChains={nChains}, padding={padding})")
    else:
        L = initial_box * unit.nanometer
        print(f"Using user-specified initial box: {L} ")
    L_nm = L.value_in_unit(unit.nanometer)
    system.setDefaultPeriodicBoxVectors(
        mm.Vec3(L_nm, 0, 0) * unit.nanometer,
        mm.Vec3(0, L_nm, 0) * unit.nanometer,
        mm.Vec3(0, 0, L_nm) * unit.nanometer,
    )


    # Add forces
    add_class2bond_forces(system, topology)
    add_excluded_volume_forces(system, topology, cutoff=cutoff_distance)
    if include_hbonds:
        add_hbond_forces(
            system,
            topology,
            kr=kr,
            ka=ka,
            r0=r0,
            kab=kab,
            kaa=kaa,
            kbb=kbb,
            selector=selector,
         )

    if include_spacers:
        add_random_spacer_forces(
             system, topology,
             alpha=alpha, tau=tau, gamma=gamma,
             mean_eps_AA=mean_eps_AA, std_eps_AA=std_eps_AA,
             mean_eps_AB=mean_eps_AB, std_eps_AB=std_eps_AB,
             mean_eps_BB=mean_eps_BB, std_eps_BB=std_eps_BB,
        )

#    if include_spacers:
#        spacer_params = {
#            ('A','A'): (mean_eps_AA, std_eps_AA),
#            ('A','B'): (mean_eps_AB, std_eps_AB),
#            ('B','B'): (mean_eps_BB, std_eps_BB),
#        }
#        eps_func = make_eps_function(spacer_params)
#        add_random_spacer_forces(system, topology, eps_func, alpha, tau, gamma)

    # Center-of-mass remover
    cm = CMMotionRemover()
    cm.setForceGroup(0)
    system.addForce(cm)

    # Integrator
    if integrator_type == 'Langevin':
        integrator = mm.LangevinIntegrator(
            temperature * T,
            friction_coeff / unit.picosecond,
            timestep * unit.femtosecond,
        )
    elif integrator_type == 'Verlet':
        integrator = VerletIntegrator(
            timestep * unit.femtosecond
        )
    else:
        raise ValueError(f"Unsupported integrator: {integrator_type} \ncurrent version only supports Langevin or Verlet")

    # Simulation
    platform = mm.Platform.getPlatformByName(platform_name) if platform_name else None
    sim = app.Simulation(topology, system, integrator, platform)
    sim.context.setPositions(positions)
    return sim


def STARS_2comp(
    seqA: str,
    seqB: str,
    nA: int,
    nB: int,
#    initial_box: float,
    integrator_type: str = 'Langevin',
    temperature: float = 1.0,
    friction_coeff: float = 1.0,
    timestep: float = 2.0,
    cutoff_distance: float = 1.0,
    include_hbonds: bool = False,
    include_spacers: bool = False,
    # H-bond params
    kr: float = None,
    ka: float = None,
    kab: float = 0.0,
    kaa: float = 0.0,
    kbb: float = 0.0,
    r0: float = None,
    selector: dict = None,
    # Spacer ε
    mean_eps_AA: float = 0.0,
    std_eps_AA: float = 0.0,
    mean_eps_AB: float = 0.0,
    std_eps_AB: float = 0.0,
    mean_eps_BB: float = 0.0,
    std_eps_BB: float = 0.0,
    # tanh-cutoff
    alpha: float = None,
    tau: float = None,
    gamma: float = None,
    platform_name: str = None,
    initial_box: Optional[float] = None,
    padding: float = 2.5,
) -> app.Simulation:
    beadA = parse_sequence(seqA, chain='A')
    beadB = parse_sequence(seqB, chain='B')
    spacer_params = {
        ('A','A'): (mean_eps_AA, std_eps_AA),
        ('A','B'): (mean_eps_AB, std_eps_AB),
        ('B','B'): (mean_eps_BB, std_eps_BB),
    }
    topology, positions = build_topology_positions_2comp(beadA, beadB, nA, nB)
    system = mm.System()
    for _ in topology.atoms(): system.addParticle(1.0)


    total_chains = nA + nB
    if initial_box is None:
        # simple heuristic: one padding length per chain
        L = (total_chains * padding + 10 ) * unit.nanometer
        print(f"Auto-estimated initial box (2 comp): {L} "
              f"(total Chains={total_chains}, padding={padding})")
    else:
        L = initial_box * unit.nanometer
        print(f"Using user-specified initial box: {L} ")
    L_nm = L.value_in_unit(unit.nanometer)
    system.setDefaultPeriodicBoxVectors(
        mm.Vec3(L_nm, 0, 0) * unit.nanometer,
        mm.Vec3(0, L_nm, 0) * unit.nanometer,
        mm.Vec3(0, 0, L_nm) * unit.nanometer,
    )



    add_class2bond_forces(system, topology)
    add_excluded_volume_forces(system, topology, cutoff=cutoff_distance)
    if include_hbonds:
        add_hbond_forces(
            system=system,
            topology=topology,
            kr=kr,
            ka=ka,
            r0=r0,
            kab=kab,
            kaa=kaa,
            kbb=kbb,
            selector=selector,
        )

    if include_spacers:
        add_random_spacer_forces(
            system=system,
            topology=topology,
            alpha=alpha,
            tau=tau,
            gamma=gamma,
            mean_eps_AA=mean_eps_AA,
            std_eps_AA=std_eps_AA,
            mean_eps_AB=mean_eps_AB,
            std_eps_AB=std_eps_AB,
            mean_eps_BB=mean_eps_BB,
            std_eps_BB=std_eps_BB,
        )

    cm = CMMotionRemover()
    cm.setForceGroup(0)
    system.addForce(cm)
    if integrator_type=='Langevin':
        integrator=mm.LangevinIntegrator(
            temperature*T, friction_coeff/unit.picosecond,
            timestep*unit.femtosecond)
    elif integrator_type=='Verlet':
        integrator=VerletIntegrator(timestep*unit.femtosecond)
    else:
        raise ValueError(f"Unsupported integrator: {integrator_type}")
    platform=mm.Platform.getPlatformByName(platform_name) if platform_name else None
    sim=app.Simulation(topology, system, integrator, platform)
    sim.context.setPositions(positions)
    return sim

# --- New "from_npy" wrapper functions ---

def STARS_1comp_from_npy(
    seq: str,
    nChains: int,
#    initial_box: float,
    position_npy: str,
    integrator_type: str = 'Langevin',
    temperature: float = 1.0,
    friction_coeff: float = 1.0,
    timestep: float = 2.0,
    cutoff_distance: float = 1.0,
    include_hbonds: bool = False,
    include_spacers: bool = False,
    kr: float = None,
    ka: float = None,
    kab: float = 0.0,
    kaa: float = 0.0,
    kbb: float = 0.0,
    r0: float = None,
    selector: dict = None,
    mean_eps_AA: float = 0.0,
    std_eps_AA: float = 0.0,
    mean_eps_AB: float = 0.0,
    std_eps_AB: float = 0.0,
    mean_eps_BB: float = 0.0,
    std_eps_BB: float = 0.0,
    alpha: float = None,
    tau: float = None,
    gamma: float = None,
    platform_name: str = None,
    initial_box: Optional[float] = None,
    padding: float = 2.5,
) -> app.Simulation:
    """
    Exactly like STARS_1comp, but override bead coordinates with a .npy array.
    """
    sim = STARS_1comp(
            seq=seq,
            nChains=nChains,
            integrator_type=integrator_type,
            temperature=temperature,
            friction_coeff=friction_coeff,
            timestep=timestep,
            cutoff_distance=cutoff_distance,
            include_hbonds=include_hbonds,
            include_spacers=include_spacers,
            kr=kr,
            ka=ka,
            kab=kab,
            kaa=kaa,
            kbb=kbb,
            r0=r0,
            selector=selector,
            mean_eps_AA=mean_eps_AA,
            std_eps_AA=std_eps_AA,
            mean_eps_AB=mean_eps_AB,
            std_eps_AB=std_eps_AB,
            mean_eps_BB=mean_eps_BB,
            std_eps_BB=std_eps_BB,
            alpha=alpha,
            tau=tau,
            gamma=gamma,
            platform_name=platform_name,
            initial_box=initial_box,
            padding=padding,
    )

    # Load and validate coords
    coords = np.load(position_npy)
    natoms = sim.topology.getNumAtoms()
    if coords.shape[0] != natoms:
        raise ValueError(
            f"NumPy file has {coords.shape[0]} atoms but system expects {natoms}."
        )
    # Convert to nm and Vec3
    coords_nm = coords / 10.0
    vecs = [Vec3(*xyz) for xyz in coords_nm]
    sim.context.setPositions(vecs)
    return sim


def STARS_2comp_from_npy(
    seqA: str,
    seqB: str,
    nA: int,
    nB: int,
#    initial_box: float,
    position_npy: str,
    integrator_type: str = 'Langevin',
    temperature: float = 1.0,
    friction_coeff: float = 1.0,
    timestep: float = 2.0,
    cutoff_distance: float = 1.0,
    include_hbonds: bool = False,
    include_spacers: bool = False,
    # H-bond params
    kr: float = None,
    ka: float = None,
    kab: float = 0.0,
    kaa: float = 0.0,
    kbb: float = 0.0,     # ← NEW
    r0: float = None,
    selector: dict = None,
    # Spacer ε
    mean_eps_AA: float = 0.0,
    std_eps_AA: float = 0.0,
    mean_eps_AB: float = 0.0,
    std_eps_AB: float = 0.0,
    mean_eps_BB: float = 0.0,
    std_eps_BB: float = 0.0,
    # tanh-cutoff
    alpha: float = None,
    tau: float = None,
    gamma: float = None,
    platform_name: str = None,
    initial_box: Optional[float] = None,
    padding: float = 2.5,
) -> app.Simulation:
    """
    Exactly like STARS_2comp, but override bead coordinates with a .npy array,
    and now also takes kbb for B–B H-bond strength.
    """
    # 1) build the base sim, passing kbb through
    sim = STARS_2comp(
        seqA=seqA,
        seqB=seqB,
        nA=nA,
        nB=nB,
        integrator_type=integrator_type,
        temperature=temperature,
        friction_coeff=friction_coeff,
        timestep=timestep,
        cutoff_distance=cutoff_distance,
        include_hbonds=include_hbonds,
        include_spacers=include_spacers,
        kr=kr,
        ka=ka,
        kab=kab,
        kaa=kaa,
        kbb=kbb,
        r0=r0,
        selector=selector,
        mean_eps_AA=mean_eps_AA,
        std_eps_AA=std_eps_AA,
        mean_eps_AB=mean_eps_AB,
        std_eps_AB=std_eps_AB,
        mean_eps_BB=mean_eps_BB,
        std_eps_BB=std_eps_BB,
        alpha=alpha,
        tau=tau,
        gamma=gamma,
        platform_name=platform_name,
        initial_box=initial_box,
        padding=padding,
    )


    # 2) load & validate coords
    coords = np.load(position_npy)  # (N,3) in Å
    natoms = sim.topology.getNumAtoms()
    if coords.shape[0] != natoms:
        raise ValueError(f"NumPy file has {coords.shape[0]} atoms but system expects {natoms}.")

    # 3) convert & set positions
    coords_nm = coords / 10.0
    vecs = [Vec3(*xyz) for xyz in coords_nm]
    sim.context.setPositions(vecs)

    return sim
