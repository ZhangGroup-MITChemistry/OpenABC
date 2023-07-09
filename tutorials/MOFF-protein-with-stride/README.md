# MOFF-protein-with-stride

Tutorial on setting up MOFF protein simulation with input stride file. This follows the original setup of GROMACS version of MOFF with IBB as an example: <https://github.com/ZhangGroup-MITChemistry/MOFF/tree/main/Example/IBB>. 

While parameterizing MOFF on both ordered and disordered proteins, folded potentials were used to stabilize individual secondary structure elements (specifically alpha-helices and beta-strands), but not between different secondary structure elements. This tutorial demonstrates how to do this, by removing all native pairs except those that stabilize continuous, ordered secondary structure domains.
