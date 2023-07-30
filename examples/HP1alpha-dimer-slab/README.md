# HP1alpha-dimer-slab

The scripts in this directory show the whole workflow of running slab simulation. Starting from preparing the system, the molecules are placed into a cubic box, then run an NPT simulation to compress, and place the compressed condensate into an elongated box to run a slab simulation. Finally the density profile and concentrations of two phases are computed based on the slab simulation trajectory. As the simulations are time consuming, we provide python and slurm scripts instead of jupyter notebooks as the example.

Run command "sbatch run.slurm" to run all the steps, and finally you will get a density profile. The users should change the settings in run.slurm based on the avilable resources. 

Here we provide explanations for the scripts: 

*run.slurm*: The slurm script for running the whole workflow.

*build_system.py*: The python script for building the simulation system, which includes all the force field information. The script also prepare the initial structure as start.pdb. 

*compress.py*: The python script for running an NPT simulation at 1 bar and 150 K. This compresses the molecules into a dense condensate. 

*run_slab.py*: Perform the 200-million-step slab simulation, which is an NVT simulation, at the given temperature. By default (the values set in run.slurm), the temperature is T=260 K, and the simulation box size is 25 nm x 25 nm x 400 nm. A snapshot is saved every 20 thousand steps. The output trajectory file is slab_${T}K.dcd. 

*align_slab_traj.py*: Translate the molecules in the output trajectory box so that in every snapshot the largest cluster is placed at the center of the box. By default the aligned trajectory is aligned_slab_${T}K.dcd, 

*compute_density.py*: Compute the density profile and the concentrations of the two phases based on the aligned trajectory.

*draw_density.py*: Draw the density profile based on the aligned trajectory. 

