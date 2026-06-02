from pathlib import Path
import json

try:
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
except ImportError:
    import simtk.openmm as mm
    import simtk.openmm.app as app
    import simtk.unit as unit

from openabc.forcefields.MOFF2.forcefields import MOFF2Model


# =========================
# Clean user inputs
# =========================

protein = "bba"

reference_dir = Path(
        "./"
)

aa_pdb = reference_dir / f"{protein}.pdb"

output_dir = Path("outputs") / protein
ca_pdb = output_dir / f"{protein}_ca.pdb"

temperature_K = 300.0
ionic_strength_mM = 150.0
target_rg_nm = None

timestep_fs = 10.0

# Tiny smoke-test values. Increase for production.
n_relax_steps = 50
n_production_steps = 500
report_interval = 10

res_group_mapping_name = "default"
platform_name = "CPU"  # use "CUDA" for GPU production runs


# =========================
# Validate inputs
# =========================

if not aa_pdb.exists():
    raise FileNotFoundError(f"Required input PDB not found: {aa_pdb}")

print(f"Protein: {protein}")
print(f"Atomistic PDB: {aa_pdb}")
print(f"Temperature: {temperature_K} K")
print(f"Ionic strength: {ionic_strength_mM} mM")
print(f"Reference Rg: {target_rg_nm} nm")


# =========================
# Prepare output folder
# =========================

output_dir.mkdir(parents=True, exist_ok=True)

input_parameters = {
    "protein": protein,
    "aa_pdb": str(aa_pdb),
    "ca_pdb": str(ca_pdb),
    "temperature": temperature_K,
    "ionic_strength": ionic_strength_mM,
    "target_rg_nm": target_rg_nm,
    "timestep_fs": timestep_fs,
    "n_relax_steps": n_relax_steps,
    "n_production_steps": n_production_steps,
    "report_interval": report_interval,
    "platform_name": platform_name,
    "res_group_mapping": res_group_mapping_name,
}

with open(output_dir / "input_parameters.json", "w") as f:
    json.dump(input_parameters, f, indent=4)
    f.write("\n")


# =========================
# Build MOFF2 OP system
# =========================

# This constructor:
#   1. parses the atomistic PDB into a CA model;
#   2. computes DSSP;
#   3. keeps native pairs inside continuous ordered H/E segments;
#   4. corrects HIS and terminal charges;
#   5. adds bonds, angles, dihedrals, and native pairs.
model = MOFF2Model.from_folded_pdb(
    aa_pdb=str(aa_pdb),
    ca_pdb=str(ca_pdb),
    bond_force_group=1,
    angle_force_group=2,
    dihedral_force_group=3,
    native_pair_force_group=4,
)

# This method uses packaged parameters:
#   openabc/forcefields/MOFF2/forcefields/parameters/MOFF2.pkl
# and adds:
#   AH + Gaussian contacts, Debye-Huckel electrostatics, density terms.
model.add_moff2_forces(
    temperature=temperature_K,
    ionic_strength=ionic_strength_mM,
    res_group_mapping=res_group_mapping_name,
    contact_force_group=5,
    elec_force_group=6,
    density_force_group_start=7,
)

with open(output_dir / "system.xml", "w") as f:
    f.write(mm.XmlSerializer.serialize(model.system))


# =========================
# Run short simulation
# =========================

T = temperature_K * unit.kelvin
friction_coeff = 1.0 / unit.picosecond
timestep = timestep_fs * unit.femtosecond

integrator = mm.LangevinMiddleIntegrator(T, friction_coeff, timestep)
init_coord = app.PDBFile(str(ca_pdb)).getPositions()

properties = {"Precision": "mixed"} if platform_name == "CUDA" else {}

model.set_simulation(
    integrator,
    platform_name=platform_name,
    init_coord=init_coord,
    properties=properties,
)

model.simulation.minimizeEnergy()
model.simulation.step(n_relax_steps)

output_dcd = output_dir / "output.dcd"
model.add_reporters(report_interval=report_interval, output_dcd=str(output_dcd))

model.simulation.context.setVelocitiesToTemperature(T)
model.simulation.step(n_production_steps)


# =========================
# Save final outputs
# =========================

checkpoint_path = output_dir / "checkpoint.chk"
model.simulation.saveCheckpoint(str(checkpoint_path))

state = model.simulation.context.getState(
    getPositions=True,
    getVelocities=True,
    enforcePeriodicBox=True,
)

with open(output_dir / "state.xml", "w") as f:
    f.write(mm.XmlSerializer.serialize(state))

final_pdb = output_dir / f"{protein}_final.pdb"
with open(final_pdb, "w") as f:
    app.PDBFile.writeFile(
        model.simulation.topology,
        state.getPositions(),
        f,
    )

print(f"Saved input parameters: {output_dir / 'input_parameters.json'}")
print(f"Saved CA PDB: {ca_pdb}")
print(f"Saved system XML: {output_dir / 'system.xml'}")
print(f"Saved trajectory: {output_dcd}")
print(f"Saved checkpoint: {checkpoint_path}")
print(f"Saved state XML: {output_dir / 'state.xml'}")
print(f"Saved final PDB: {final_pdb}")
