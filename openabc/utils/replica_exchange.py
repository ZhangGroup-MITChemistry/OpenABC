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
from openabc.lib import GAS_CONST
import sys
import os
try:
    import torch
except ImportError:
    torch = None
import math
import time

"""
The code is adapted from Xinqiang Ding's script. 
"""

# set gas constant R
GAS_CONST_value = GAS_CONST.value_in_unit(unit.kilojoule_per_mole / unit.kelvin)

class TemperatureReplicaExchange(object):
    """
    Temperature replica exchange class for performing temperature replica exchange (TRE) simulations. 
    
    Note only positions and scaled velocities are exchanged, while other internal states of the simulations are not exchanged. This means the class may not be properly applied to simulations involving other internal states. For example, Nose-Hoover integrator may not work properly as it includes internal chain states. 
    
    The parallelization is achieved with `torch.distributed`. 
    
    An example of setting environment variables in a slurm job script:

        export WORLD_SIZE=$((${SLURM_NNODES}*${SLURM_NTASKS_PER_NODE}))
    
        export MASTER_PORT=$(expr 30000 + $(echo -n ${SLURM_JOBID} | tail -c 4))
    
        master_addr=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
    
        export MASTER_ADDR=${master_addr}
    
    """
    
    def __init__(self, backend, n_replicas, rank, positions, top, system, temperatures, integrator, 
                 platform_name='CUDA', properties={'Precision': 'mixed'}):
        """
        Initialize temperature replica exchange object. 
        
        Parameters
        ----------
        backend : str
            Backend for torch
        
        n_replicas : int
            The number of replicas.
        
        rank : int
            The index of the current replica. 
        
        positions : np.ndarray, shape is (n_atoms, 3) or (n_replicas, n_atoms, 3)
            The initial coordinates. 
            If shape is (n_atoms, 3), then all replicas start from the same initial coordinates. 
            If shape is (n_replicas, n_atoms, 3), then the i-th replica starts from initial coordinate positions[i]. 
        
        top: OpenMM Topology
            The OpenMM topology. 
        
        system: OpenMM System
            The OpenMM system.
        
        temperatures: array-like of numbers or quantities, shape is (n_replicas,)
            The temperatures for all replicas.
        
        integrator: OpenMM Integrator
            The OpenMM integrator. 
        
        platform_name: str
            The OpenMM simulation platform name. This can be OpenCL or CUDA or CPU or Reference. 
        
        properties : dict or None
            The OpenMM simulation platform properties. 
        
        References
        ----------
        https://pytorch.org/docs/stable/distributed.html

        """
        assert torch is not None, 'torch is not installed.'
        self.backend = backend
        self.n_replicas = n_replicas
        assert self.n_replicas == int(os.environ['WORLD_SIZE'])
        self.rank = rank
        torch.distributed.init_process_group(
            backend,
            world_size=self.n_replicas,
            rank=self.rank,
        )
        self.top = top
        self.system = system
        assert len(temperatures) == n_replicas
        
        def _temperature_to_float(t):
            if unit.is_quantity(t):
                return t.value_in_unit(unit.kelvin)
            else:
                return float(t)
        self.temperatures = [_temperature_to_float(t) for t in temperatures]
        self.integrator = integrator
        self.integrator.setTemperature(self.temperatures[self.rank]) # reset temperature to ensure it is at the target temperature
        platform = mm.Platform.getPlatformByName(platform_name)
        if platform in ['CUDA', 'OpenCL']:
            self.simulation = app.Simulation(self.top, self.system, self.integrator, platform, properties)
        else:
            self.simulation = app.Simulation(self.top, self.system, self.integrator, platform)
        assert positions.ndim in [2, 3]
        if positions.ndim == 2:
            self.simulation.context.setPositions(positions)
        else:
            self.simulation.context.setPositions(positions[self.rank])
        self.simulation.minimizeEnergy()
        self.simulation.context.setVelocitiesToTemperature(self.temperatures[self.rank])
    
    def add_reporters(self, report_interval, report_state=True, report_dcd=True, output_dcd=None):
        """
        Add reporters for OpenMM simulation.
        
        Whether to use PBC is read from self.system information. 
        
        Parameters
        ----------
        report_interval : int
            Report interval.
        
        report_state : bool
            Whether to report simulation state. 
        
        report_dcd : bool
            Whether to report trajectory as dcd file. 
        
        output_dcd : str or None
            The output dcd file path. If None, then the output dcd file path is set as output.{self.rank}.dcd. 
        
        """
        use_pbc = self.system.usesPeriodicBoundaryConditions()
        if report_state:
            state_data_reporter = app.StateDataReporter(
                sys.stdout,
                report_interval,
                step=True,
                time=True, 
                potentialEnergy=True,
                kineticEnergy=True,
                totalEnergy=True, 
                temperature=True,
                speed=True,
            )
            self.simulation.reporters.append(state_data_reporter)
        if report_dcd:
            if output_dcd is None:
                output_dcd = f'output.{self.rank}.dcd'
            dcd_reporter = app.DCDReporter(output_dcd, report_interval, enforcePeriodicBox=use_pbc)
            self.simulation.reporters.append(dcd_reporter)
    
    def run_replica_exchange(self, n_steps, exchange_interval, verbose=True):
        """
        Perform replica exchange simulation. 
        
        Exchange atom positions and rescaled velocities. Other state variables are not exchanged. 
        
        The temperature of each replica remains unchanged. 
        
        Parameters
        ----------
        n_steps : int
            Total number of simulation steps for each replica.
        
        exchange_interval : int
            Exchange interval. 
        
        verbose : bool
            Whether to report exchange acceptance ratio and simulation speed. 
         
        """
        n_iterations = int(n_steps / exchange_interval)
        n_steps = n_iterations * exchange_interval # reset n_steps based on interval
        n_attempts = 0
        n_accepted_attempts = 0
        start_time = time.time()
        for i in range(n_iterations):
            self.simulation.step(exchange_interval)
            state = self.simulation.context.getState(
                getPositions=True,
                getEnergy=True,
                getVelocities=True, 
                enforcePeriodicBox=True,
            )
            pos = torch.from_numpy(np.array(state.getPositions().value_in_unit(unit.nanometer)))
            vel = torch.from_numpy(np.array(state.getVelocities().value_in_unit(unit.nanometer / unit.picosecond)))
            pe = torch.tensor([state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)])
            all_pos = [torch.zeros_like(pos) for _ in range(self.n_replicas)]
            all_vel = [torch.zeros_like(vel) for _ in range(self.n_replicas)]
            all_pe = [torch.zeros(1) for _ in range(self.n_replicas)]
            torch.distributed.all_gather(all_pos, pos)
            torch.distributed.all_gather(all_vel, vel)
            torch.distributed.all_gather(all_pe, pe)
            all_pe = torch.stack(all_pe).reshape(-1)
            if self.rank == 0:
                for j in range(self.n_replicas - 1):
                    n_attempts += 1
                    delta_pe = all_pe[j] - all_pe[j + 1]
                    delta_beta = (1 / self.temperatures[j] - 1 / self.temperatures[j + 1]) / GAS_CONST_value
                    if np.random.uniform(0, 1) < math.exp(delta_beta * delta_pe):
                        n_accepted_attempts += 1
                        all_pos[j], all_pos[j + 1] = all_pos[j + 1], all_pos[j]
                        alpha = (self.temperatures[j] / self.temperatures[j + 1])**0.5
                        all_vel[j], all_vel[j + 1] = alpha * all_vel[j + 1], all_vel[j] / alpha
                        all_pe[j], all_pe[j + 1] = all_pe[j + 1], all_pe[j]
            else:
                all_pos = None
                all_vel = None
            torch.distributed.scatter(pos, all_pos, src=0)
            torch.distributed.scatter(vel, all_vel, src=0)
            self.simulation.context.setPositions(pos.numpy())
            self.simulation.context.setVelocities(vel.numpy())
        end_time = time.time()
        if (self.rank == 0) and verbose:
            acceptance_ratio = n_accepted_attempts / n_attempts
            print(f'Replica exchange acceptance ratio is {acceptance_ratio}.')
            timestep = self.integrator.getStepSize().value_in_unit(unit.nanosecond)
            speed_ns_per_day = (24 * 3600 / (end_time - start_time)) * (timestep * n_steps)
            print(f'Simulation speed is {speed_ns_per_day} ns/day')



