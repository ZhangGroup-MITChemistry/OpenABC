from pathlib import Path
import json
import shutil

import mdtraj

try:
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
except ImportError:
    import simtk.openmm as mm
    import simtk.openmm.app as app
    import simtk.unit as unit

from openabc.forcefields.parsers import HPSParser
from openabc.forcefields.MOFF2.forcefields import MOFF2Model


# =========================
# Clean user inputs
# =========================

protein = "A1-LCD+NLS"

input_pdb = Path(
    "A1-LCD+NLS_ca.pdb"
)

output_dir = Path("outputs") / protein

temperature_K = 298.0
ionic_strength_mM = 150.0
timestep_fs = 10.0

# tiny smoke-test values
n_relax_steps = 50
n_production_steps = 500
report_interval = 10

res_group_mapping_name = "default"
platform_name = "CPU"


# =========================
# Prepare files
# =========================

if not input_pdb.exists():
    raise FileNotFoundError(f"Input PDB not found: {input_pdb}")

output_dir.mkdir(parents=True, exist_ok=True)

ca_pdb = output_dir / f"{protein}_ca.pdb"
shutil.copyfile(input_pdb, ca_pdb)

input_parameters = {
    "protein": protein,
    "input_pdb": str(input_pdb),
    "temperature": temperature_K,
    "ionic_strength": ionic_strength_mM,
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
# Build MOFF2 system
# =========================

model = MOFF2Model()
model.append_mol(HPSParser(str(ca_pdb)))

model.protein_bonds.loc[:, "k_bond"] = 8000.0
model.protein_bonds.loc[:, "r0"] = 0.386

ca_traj = mdtraj.load_pdb(str(ca_pdb))
assert ca_traj.n_chains == 1

for _, row in model.atoms.iterrows():
    if row["resname"] == "HIS":
        assert row["charge"] == 0.5

charge = model.atoms["charge"].to_numpy()
charge[0] += 1
charge[-1] -= 1
model.atoms["charge"] = charge

top = app.PDBFile(str(ca_pdb)).getTopology()
model.create_system(top=top, box_a=1000, box_b=1000, box_c=1000)

model.add_protein_bonds(force_group=1)

# Uses packaged:
# openabc/forcefields/MOFF2/forcefields/parameters/MOFF2.pkl
model.add_moff2_forces(
    temperature=temperature_K,
    ionic_strength=ionic_strength_mM,
    res_group_mapping=res_group_mapping_name,
    contact_force_group=2,
    elec_force_group=3,
    density_force_group_start=4,
)

with open(output_dir / "system.xml", "w") as f:
    f.write(mm.XmlSerializer.serialize(model.system))


# =========================
# Run simulation
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
model.add_reporters(
    report_interval=report_interval,
    output_dcd=str(output_dcd),
)

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
print(f"Saved system XML: {output_dir / 'system.xml'}")
print(f"Saved trajectory: {output_dcd}")
print(f"Saved checkpoint: {checkpoint_path}")
print(f"Saved state XML: {output_dir / 'state.xml'}")
print(f"Saved final PDB: {final_pdb}")
