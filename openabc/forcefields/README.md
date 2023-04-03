# forcefields

For each single force field (for example, MOFF or MRG, which captures a single type of molecule), we have a parser for that forcefield. 

For each valid combination of force field (for example, MOFF+MRG, which is a combination of a protein model and a DNA model), we have a container for that combination. Each combination is dealed independently because combination leads to new nonbonded forces. For example, MOFF+MRG combination leads to new protein-DNA forces. 

