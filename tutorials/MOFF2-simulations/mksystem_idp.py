#!/usr/bin/env python3
from pathlib import Path
import json
import shutil
import sys

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
from openabc.utils.insert import insert_molecules


# =========================
# Clean user inputs
# =========================

protein = "A1-LCD+NLS"

input_pdb = Path(
        f"{protein}_ca.pdb"
)

output_dir = Path(
    f"output/LLPS_{protein}"
)

temperature_K = 293.0
ionic_strength_mM = 150.0
res_group_mapping_name = "default"

n_mol = 100
box_a_nm = 25.0
box_b_nm = 25.0
box_c_nm = 300.0

timestep_fs = 10.0
n_steps = 5000
output_interval = 500
platform_name = "CPU"  # use "CUDA" for production runs


# =========================
# Validate and prepare files
# =========================

if not input_pdb.exists():
    raise FileNotFoundError(f"Input CA PDB not found: {input_pdb}")

output_dir.mkdir(parents=True, exist_ok=True)

ca_pdb = output_dir / f"{protein}_ca.pdb"
shutil.copyfile(input_pdb, ca_pdb)

input_parameters = {
    "protein": protein,
    "input_pdb": str(input_pdb),
    "temperature": temperature_K,
    "ionic_strength": ionic_strength_mM,
    "res_group_mapping": res_group_mapping_name,
    "n_mol": n_mol,
    "box_a": box_a_nm,
    "box_b": box_b_nm,
    "box_c": box_c_nm,
    "timestep_fs": timestep_fs,
    "n_steps": n_steps,
    "output_interval": output_interval,
    "platform_name": platform_name,
}

with open(output_dir / "input_parameters.json", "w") as f:
    json.dump(input_parameters, f, indent=4)
    f.write("\n")


# =========================
# Build one MOFF2 IDP chain
# =========================

protein_model = MOFF2Model()
protein_model.append_mol(HPSParser(str(ca_pdb)))

protein_model.protein_bonds.loc[:, "k_bond"] = 8000.0
protein_model.protein_bonds.loc[:, "r0"] = 0.386

ca_traj = mdtraj.load_pdb(str(ca_pdb))
assert ca_traj.n_chains == 1

his_mask = protein_model.atoms["resname"] == "HIS"
protein_model.atoms.loc[his_mask, "charge"] = 0.5

charges = protein_model.atoms["charge"].to_numpy()
charges[0] += 1.0
charges[-1] -= 1.0
protein_model.atoms["charge"] = charges


# =========================
# Insert chains into a slab box
# =========================

start_pdb = output_dir / "start.pdb"
if not start_pdb.exists():
    insert_molecules(
        str(ca_pdb),
        str(start_pdb),
        n_mol,
        box=[box_a_nm, box_b_nm, box_c_nm],
    )


# =========================
# Build the MOFF2 LLPS system
# =========================

model = MOFF2Model()
for _ in range(n_mol):
    model.append_mol(protein_model)

top = app.PDBFile(str(start_pdb)).getTopology()
model.create_system(
    top=top,
    box_a=box_a_nm,
    box_b=box_b_nm,
    box_c=box_c_nm,
)

model.add_protein_bonds(force_group=1)
model.add_moff2_forces(
    temperature=temperature_K,
    ionic_strength=ionic_strength_mM,
    res_group_mapping=res_group_mapping_name,
    contact_force_group=2,
    elec_force_group=3,
    density_force_group_start=4,
)

temperature = temperature_K * unit.kelvin
pressure = 1.0 * unit.bar
model.system.addForce(mm.MonteCarloBarostat(pressure, temperature))

with open(output_dir / "system.xml", "w") as f:
    f.write(mm.XmlSerializer.serialize(model.system))


# =========================
# Run a short NPT simulation
# =========================

integrator = mm.LangevinMiddleIntegrator(
    temperature,
    1.0 / unit.picosecond,
    timestep_fs * unit.femtosecond,
)

properties = {"Precision": "mixed"} if platform_name == "CUDA" else {}
init_coord = app.PDBFile(str(start_pdb)).getPositions()

model.set_simulation(
    integrator,
    platform_name=platform_name,
    init_coord=init_coord,
    properties=properties,
)

model.simulation.minimizeEnergy()

output_dcd = output_dir / "output_NPT.dcd"
model.simulation.reporters.append(
    app.DCDReporter(str(output_dcd), output_interval, enforcePeriodicBox=True)
)
model.simulation.reporters.append(
    app.StateDataReporter(
        sys.stdout,
        output_interval,
        step=True,
        potentialEnergy=True,
        kineticEnergy=True,
        totalEnergy=True,
        temperature=True,
        speed=True,
    )
)

model.simulation.context.setVelocitiesToTemperature(temperature)

for start in range(0, n_steps, output_interval):
    model.simulation.step(output_interval)
    state = model.simulation.context.getState(
        getPositions=False,
        enforcePeriodicBox=True,
    )
    box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
    print(
        f"Step {start + output_interval}: "
        f"box=({box[0][0]:.2f}, {box[1][1]:.2f}, {box[2][2]:.2f}) nm"
    )

final_state = model.simulation.context.getState(
    getPositions=True,
    getVelocities=True,
    enforcePeriodicBox=True,
)

with open(output_dir / "NPT_final_state.xml", "w") as f:
    f.write(mm.XmlSerializer.serialize(final_state))

final_pdb = output_dir / f"{protein}_final.pdb"
with open(final_pdb, "w") as f:
    app.PDBFile.writeFile(
        model.simulation.topology,
        final_state.getPositions(),
        f,
    )

print(f"Saved input parameters: {output_dir / 'input_parameters.json'}")
print(f"Saved start structure: {start_pdb}")
print(f"Saved system XML: {output_dir / 'system.xml'}")
print(f"Saved trajectory: {output_dcd}")
print(f"Saved final state XML: {output_dir / 'NPT_final_state.xml'}")
print(f"Saved final PDB: {final_pdb}")
