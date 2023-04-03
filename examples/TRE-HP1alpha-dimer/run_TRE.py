import numpy as np
import pandas as pd
import simtk.openmm as mm
import simtk.openmm.app as app
import simtk.unit as unit
import sys
import os

__location__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append('../..')
from openabc.utils.replica_exchange import TemperatureReplicaExchange

'''
Run replica exchange for HP1alpha dimer. 
'''

# prepare system
top = app.PDBFile('hp1alpha_dimer_CA.pdb').getTopology()
with open('hp1alpha_dimer_system.xml', 'r') as f:
    system = mm.XmlSerializer.deserialize(f.read())

# set replica exchange
backend = 'gloo'
n_replicas = 6
rank = int(os.environ['SLURM_PROCID'])
temperatures = 1/np.linspace(1/300, 1/400, n_replicas)
print(f'Replica {rank} uses temperature {temperatures[rank]} K. ')
friction_coeff = 1/unit.picosecond
timestep = 10*unit.femtosecond
integrator = mm.LangevinMiddleIntegrator(temperatures[rank]*unit.kelvin, friction_coeff, timestep)
top = app.PDBFile('hp1alpha_dimer_CA.pdb').getTopology()
positions = app.PDBFile('hp1alpha_dimer_CA.pdb').getPositions()
replica_exchange = TemperatureReplicaExchange(backend, n_replicas, rank, positions, top, system, temperatures, integrator, platform_name='CUDA')
replica_exchange.add_reporters(report_interval=10000, output_dcd=f'output-dcd/output.{rank}.dcd')
n_steps = 1000000
exchange_interval = 1000
replica_exchange.run_replica_exchange(n_steps, exchange_interval)


