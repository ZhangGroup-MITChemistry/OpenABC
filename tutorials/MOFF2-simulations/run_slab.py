#!/usr/bin/env python3
import os, sys, argparse, numpy as np
try:
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
except ImportError:
    import simtk.openmm as mm
    import simtk.openmm.app as app
    import simtk.unit as unit
import pickle
import mdtraj
import shutil
import glob
import json
import sys

protein='A1-LCD+NLS'
output_dir=f'output/LLPS_{protein}'


system_path = os.path.join(f"{output_dir}/system.xml")
state_path = os.path.join(f"{output_dir}/NPT_final_state.xml")
top_path = os.path.join(f"{output_dir}/start.pdb")
temperature = 298.0
timestep = 10.0
n_steps = 5000
output_interval = 500
box_a=25
box_b=25
box_c=300

with open(system_path) as f:
    system = mm.XmlSerializer.deserialize(f.read())
with open(state_path) as f:
    npt_state = mm.XmlSerializer.deserialize(f.read())

# Remove barostat
for i, fce in enumerate(list(system.getForces())):
    if isinstance(fce, mm.MonteCarloBarostat):
        system.removeForce(i)
        print(f"Removed barostat (force {i})")

top = app.PDBFile(top_path).getTopology()
integrator = mm.NoseHooverIntegrator(temperature * unit.kelvin,
                                     0.01 / unit.picosecond,
                                     timestep * unit.femtosecond)
platform = mm.Platform.getPlatformByName("CPU")
properties = {"Precision": "mixed"} if platform == "CUDA" else {}
sim = app.Simulation(top, system, integrator, platform, properties)

# Load previous state
sim.context.setPeriodicBoxVectors(*npt_state.getPeriodicBoxVectors())
sim.context.setPositions(npt_state.getPositions())
vel = npt_state.getVelocities()
if vel is not None:
    sim.context.setVelocities(vel)
else:
    sim.context.setVelocitiesToTemperature(temperature * unit.kelvin)

# Override box if requested
if (box_a, box_b, box_c) != (0, 0, 0):
    new_a = mm.Vec3(box_a, 0, 0) * unit.nanometer
    new_b = mm.Vec3(0, box_b, 0) * unit.nanometer
    new_c = mm.Vec3(0, 0, box_c) * unit.nanometer
    sim.context.setPeriodicBoxVectors(new_a, new_b, new_c)
    print(f"Set new box: {box_a} × {box_b} × {box_c} nm")

    # Recenter positions (keep units)
    pos_np = sim.context.getState(getPositions=True).getPositions(asNumpy=True)
    center = 0.5 * np.array([box_a, box_b, box_c]) * unit.nanometer
    pos_np -= np.mean(pos_np, axis=0)
    pos_np += center
    sim.context.setPositions(pos_np)
    print("Recentered coordinates into new box.")

# Minimize and run NVT
sim.minimizeEnergy(maxIterations=200)
# Reporters
output_dcd = os.path.join(output_dir, f"slab.dcd")
dcd_reporter = app.DCDReporter(output_dcd, output_interval, enforcePeriodicBox=True)
state_data_reporter = app.StateDataReporter(
    sys.stdout, output_interval,
    step=True, time=True, potentialEnergy=True,
    kineticEnergy=True, totalEnergy=True,
    temperature=True, speed=True
)
sim.reporters.append(dcd_reporter)
sim.reporters.append(state_data_reporter)

checkpoint_interval = 50000
chk_path = os.path.join(output_dir, f"chk.chk")
xml_prefix = os.path.join(output_dir, f"chk")

for step in range(0, n_steps, checkpoint_interval):
    sim.step(checkpoint_interval)
    sim.saveCheckpoint(chk_path)
    state = sim.context.getState(
        getPositions=True, 
        getVelocities=True, 
        enforcePeriodicBox=True
    )
    xml_path = f"{xml_prefix}.xml"
    with open(xml_path, "w") as f:
        f.write(mm.XmlSerializer.serialize(state))

print("✅ Simulation completed successfully.")


