# openabc

openabc is a package for preparing CG biomolecular simulations performed by OpenMM. 

OpenMM is a tool to set and perform molecular dynamics (MD) simulations. With GPU accleration, OpenMM is extremly suitable for doing bio-molecular condensate (BC) simulations. 

By default, all the units are kept consistent with OpenMM. The length unit is nm, the angle unit is radian, the time unit is ps, and the energy unit is kJ/mol.  

Another useful package used many times in the package is MDTraj, which is a useful tool to load and analyze bio-molecular structures. By default MDTraj also uses nm and ps units. For the degree part, while the angles in `mdtraj.Trajectory` are saved as degree unit, the output angles of `mdtraj.compute_angles` are in unit radians. So be careful with angle units when using MDTraj. 


