# TRE-HP1alpha-dimer

Example of running temperature replica exchange (TRE) simulations for HP1alpha dimer

First run `python run_build_system.py` to build and save the system as xml file. Then use `sbatch run.slurm` to submit job to GPU nodes and run the simulation. You may need to change settings in `run.slurm` based on the GPUs you have. 

