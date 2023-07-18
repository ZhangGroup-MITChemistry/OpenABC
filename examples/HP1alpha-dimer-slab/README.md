# HP1alpha-dimer-slab

The scripts in this directory show the whole workflow of running slab simulation. Starting from preparing the system, the molecules are placed into a cubic box, then run an NPT simulation to compress, and finally place the compressed condensate into an elongated box to run a slab simulation.

Run the script "run.slurm" to run all the steps, and finally you will get a density profile. The users should change the settings in run.slurm based on the avilable resources. 

