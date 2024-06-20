# README-explicit-ion-nucl-dev

This is the note for the development of the branch "explicit-ion-nucl-dev". This branch is developed for doing explicit ion simulation of nucleosomes. The model was first applied in Ref [1] in LAMMPS, and here we aim to move the model to OpenMM thus we can use GPU acceleration. Basically the model is to add explicit ions to the SMOG+3SPN2 model framework with certain modifications. 

For the modifications, the main challenge is to chagne the electrostatic potential from Debye-Huckel potential to unscreened Coulombic potential. 



## References
[1] Lin, Xingcheng, and Bin Zhang. "Explicit ion modeling predicts physicochemical interactions for chromatin organization." Elife 12 (2024): RP90073.



