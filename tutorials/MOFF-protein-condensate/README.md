# MOFF-protein-condensate

Tutorials on simulations with MOFF, which was introduced in the following references:

    1. Latham, A. P., & Zhang, B. (2020). "Maximum Entropy Optimized Force Field for Intrinsically Disordered Proteins." Journal of Chemical Theory and Computation, 16: 773–781. 
    2. Latham, A. P., & Zhang, B. (2021). "Consistent force field captures homologue-resolved HP1 phase separation." Journal of chemical theory and computation 17: 3134-3144.


- The jupyter notebook file *hp1alpha_dimer.ipynb* provides detailed instructions on setting up and running simulations of a HP1alpha dimer.
    1. It explains how to extract native contacts from the initial configuration with the shadow algorithm to preserve tertiary structures; 
    2. It also illustrates how to avoid biases in disordered regions to keep their conformational flexibility. 
    3. Example scripts are also provided to rigidify folded domains for enhanced sampling and larger simulation time steps. 

- The jupyter notebook file *hp1alpha_condensate.ipynb* provides detailed instructions on
    1. setting up and running slab simulations for hp1alpha dimer, which consists of the following steps
        - NPT compression to create a densely packaged system. 
        - Place the condensed system into a rectangle box with dense-dilute interface along the z-axis. 
    2. converting coarse-grained configurations to atomistic structures. 

- The jupyter notebook file *condensate_analysis.ipynb* includes python scripts for analyzing slab simulations and computing density profiles. Such analyses are essential for computing phase diagrams. 

