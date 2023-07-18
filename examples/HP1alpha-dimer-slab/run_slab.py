import numpy as np
import pandas as pd
import sys
import os
import argparse
import simtk.unit as unit
import simtk.openmm as mm
import simtk.openmm.app as app

parser = argparse.ArgumentParser()
parser.add_argument('--temperature', type=float, default=300.0, help='simulation temperature in unit kelvin')
parser.add_argument('--box_a', type=float, default=25.0, help='simulation x-axis box length')
parser.add_argument('--box_b', type=float, default=25.0, help='simulation y-axis box length')
parser.add_argument('--box_c', type=float, default=500.0, help='simulation z-axis box length')
parser.add_argument('--output_dcd', default='output.dcd', help='output dcd file path')
parser.add_argument('--output_interval', type=int, default=20000, help='output interval')
parser.add_argument('--steps', type=int, default=200000000, help='number of steps')
args = parser.parse_args()
print('command line args: ' + ' '.join(sys.argv))

system_xml = '/home/gridsan/sliu/Projects/openbc-product-simulations/HP1alpha-dimer-slab/build-system/system.xml'
with open(system_xml, 'r') as f:
    system = mm.XmlSerializer.deserialize(f.read())
box_vec_a = np.array([args.box_a, 0, 0])*unit.nanometer
box_vec_b = np.array([0, args.box_b, 0])*unit.nanometer
box_vec_c = np.array([0, 0, args.box_c])*unit.nanometer
system.setDefaultPeriodicBoxVectors(box_vec_a, box_vec_b, box_vec_c)
print('System box vectors are:')
print(system.getDefaultPeriodicBoxVectors())

top = app.PDBFile('start.pdb').getTopology()

npt_final_state_xml = 'NPT_final_state.xml'
with open(npt_final_state_xml, 'r') as f:
    npt_final_state = mm.XmlSerializer.deserialize(f.read())
init_coord = np.array(npt_final_state.getPositions().value_in_unit(unit.nanometer))
# move the geometric center of atom coordinates to box center
init_coord -= np.mean(init_coord, axis=0)
init_coord += 0.5*np.array([args.box_a, args.box_b, args.box_c])

start_temperature = 150
collision = 1/unit.picosecond
timestep = 5*unit.femtosecond
integrator = mm.NoseHooverIntegrator(start_temperature*unit.kelvin, collision, timestep)
platform_name = 'CUDA'
platform = mm.Platform.getPlatformByName(platform_name)
properties = {'Precision': 'mixed'}
simulation = app.Simulation(top, system, integrator, platform, properties)
simulation.context.setPositions(init_coord)
simulation.minimizeEnergy()
simulation.context.setVelocitiesToTemperature(start_temperature*unit.kelvin)
n_iterations = 100
for temperature_i in np.linspace(start_temperature, args.temperature, n_iterations):
    integrator.setTemperature(temperature_i*unit.kelvin)
    simulation.step(1000)

dcd_reporter = app.DCDReporter(args.output_dcd, args.output_interval, enforcePeriodicBox=True)
state_reporter = app.StateDataReporter(sys.stdout, args.output_interval, step=True, time=True, potentialEnergy=True, 
                                       kineticEnergy=True, totalEnergy=True, temperature=True, speed=True)
simulation.reporters.append(dcd_reporter)
simulation.reporters.append(state_reporter)
simulation.step(args.steps)


