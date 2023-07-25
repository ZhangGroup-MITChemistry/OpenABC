import numpy as np
import pandas as pd
try:
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
except ImportError:
    import simtk.openmm as mm
    import simtk.openmm.app as app
    import simtk.unit as unit
import sys
import os

top = app.PDBFile('start.pdb').getTopology()
init_coord = app.PDBFile('start.pdb').getPositions()
system_xml = 'system.xml'
with open(system_xml, 'r') as f:
    system = mm.XmlSerializer.deserialize(f.read())

pressure = 1*unit.bar
temperature = 150*unit.kelvin
system.addForce(mm.MonteCarloBarostat(pressure, temperature))

friction_coeff = 1/unit.picosecond
timestep = 10*unit.femtosecond
integrator = mm.LangevinMiddleIntegrator(temperature, friction_coeff, timestep)
properties = {'Precision': 'mixed'}
platform_name = 'CUDA'
platform = mm.Platform.getPlatformByName(platform_name)
simulation = app.Simulation(top, system, integrator, platform, properties)
simulation.context.setPositions(init_coord)
simulation.minimizeEnergy()
simulation.context.setVelocitiesToTemperature(temperature)

output_dcd = 'output.dcd'
output_interval = 100000
dcd_reporter = app.DCDReporter(output_dcd, output_interval, enforcePeriodicBox=True)
state_reporter = app.StateDataReporter(sys.stdout, output_interval, step=True, time=True, potentialEnergy=True,
                                       kineticEnergy=True, totalEnergy=True, temperature=True, speed=True)
simulation.reporters.append(dcd_reporter)
simulation.reporters.append(state_reporter)
n_steps = 5000000
simulation.step(n_steps)

state = simulation.context.getState(getPositions=True, getVelocities=True, getForces=True, getEnergy=True,
                                    getParameters=True, enforcePeriodicBox=True)
box_vec = state.getPeriodicBoxVectors(asNumpy=True)
print('Final box vectors:')
print(box_vec)
with open('NPT_final_state.xml', 'w') as f:
    f.write(mm.XmlSerializer.serialize(state))


