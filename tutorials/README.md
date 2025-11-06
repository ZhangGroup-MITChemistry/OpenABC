# Tutorials

This folder provides tutorials for setting up and performing simulations with MOFF, MRG or HPS models. Each tutorial includes jupyter notebooks and necessary input files. 

- MOFF-protein-condensate explains 
    1. setting simulations of a protein with both folded and disordered domains using MOFF; 
    2. setting up condensate simulations with MOFF; 
    3. converting coarse-grained configurations with only alpha carbons to atomistic structures;
    4. analyzing slab simulation trajectories to compute density profiles. 

- HPS-protein-condensate explains
    1. setting up simulations of a single protein with HPS;
    2. setting up condensate simulations with HPS. 

- MOFF-protein-MRG-dsDNA-condensate explains setting up complex condensate simulations with MOFF and MRG-DNA.

- MRG-dsDNA provides an example for running simulations of a dsDNA with MRG. 

- Mpipi-protein-RNA-condensate shows how to run protein-RNA condensate simulation. 

- SMOG-3SPN2-two-nucleosomes shows how to run the simulation of two nucleosomes with SMOG and 3SPN2 models. 

- NEAT-DNA-dsDNA provides an example for running simulations of a dsDNA with NEAT-DNA. 

- NEAT-DNA-MOFF-protein-nucleosome provides an example for running simulations of a nucleosome with NEAT-DNA and MOFF. 

- build-new-forcefield provides the tutorial for building a new force field. 

To view the structures and trajectories, we use nglview package in some tutorials to render the structures or trajectories directly in the jupyter notebook. Based on our test, nglview is not very robust and sometimes it fails to render. Increase notebook output limit (with command `jupyter notebook --NotebookApp.iopub_data_rate_limit=10000000`), rerun the code block, or restart the kernel may be helpful. Users can also use other popular tools to view the structures, such as VMD and Mol* Viewer. 

