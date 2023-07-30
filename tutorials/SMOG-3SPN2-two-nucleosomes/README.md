# SMOG-3SPN2-two-nucleosomes

Tutorial about how to simulation two nucleosomes with SMOG and 3SPN2 models. The models are introduced in the following references: 

    3SPN2 model: (1) Hinckley, Daniel M., et al. "An experimentally-informed coarse-grained 3-site-per-nucleotide model of DNA: Structure, thermodynamics, and dynamics of hybridization." The Journal of chemical physics 139.14 (2013). (2) Freeman, Gordon S., et al. "Coarse-grained modeling of DNA curvature." The Journal of chemical physics 141.16 (2014).
    
    Open3SPN2 model: Lu, Wei, et al. "OpenAWSEM with Open3SPN2: A fast, flexible, and accessible framework for large-scale coarse-grained biomolecular simulations." PLoS computational biology 17.2 (2021): e1008308.
    
    SMOG model: Noel, Jeffrey K., et al. "SMOG 2: a versatile software package for generating structure-based models." PLoS computational biology 12.3 (2016): e1004794.

Note 3SPN2 has already been implemented in OpenMM framework as Open3SPN2, and we further adapt Open3SPN2 into OpenABC framework. Please follow the steps in *two_nucl.ipynb* to learn how to use these two models. 

Note to run simulation with 3SPN2, user have to install x3dna, since 3SPN2 B_curved DNA get force field parameters from the DNA template built by x3dna. 

