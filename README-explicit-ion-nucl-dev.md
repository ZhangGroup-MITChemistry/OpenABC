# README-explicit-ion-nucl-dev

This is the note for the development of the branch "explicit-ion-nucl-dev". This branch is developed for doing explicit ion simulation of nucleosomes. The model was first applied in Ref [1] in LAMMPS, and here we aim to move the model to OpenMM thus we can use GPU acceleration. Basically the model is to add explicit ions to the SMOG+3SPN2 model framework with certain modifications. 

We mainly modify nonbonded terms, which include nonbonded contacts (including VDWL and hydration) and electrostatic interactions. There are three new types of CG atoms: sodium, magnesium, and chloride ions. For the modifications, the main technical challenge is to change the electrostatic potential from Debye-Huckel potential to unscreened Coulombic potential. 

We use AA+ and AA- to represent amino acids with positive or negative charges, respectively. 

## Nonbonded potential

The nonbonded potential is

$$
U_{nonbonded}=U_{vdwl}+U_{hydr}+U_{elec}
$$

We will explain each term below.

### Van der Waals term

$U_{vdwl}$ is the Van der Waals potential described as LJ potential. This term is involved for all the interactions between protein, DNA, and ions (i.e. between all the i-j pairs, i is from protein or D or I, j is from protein or D or I).

$$
U_{vdwl}=4\epsilon\left[\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^{6}\right]
$$

For $U_{vdwl}$, existing vdwl parameters in SMOG+3SPN2 are inherited except: (1) phosphate-phosphate vdwl parameters change; (2) protein-DNA vdwl cutoff changes. vdwl parameters involving ions and phosphate-phosphate pair parameters are shown in Tables 1 and 2 of ref [1]. Protein-DNA vdwl cutoff is changed to $2^{1/6}\sigma$, where $\sigma=0.57 nm$. 

### Hydration term

$U_{hydr}$ is the hydration term. This term is motivated by the hydration shell. The potential is 

$$
U_{hydr} = \frac{H}{\sigma_h \sqrt{2\pi}} \exp\left[-\frac{(r-r_{mh})^2}{2\sigma_h^2}\right]
$$

Note this term only exists between two charged species except when both CG atoms are charged amino acids (i.e. except (AA+, AA+), (AA+, AA-), (AA-, AA-) pairs). Charged species include phosphate, ions, AA+, and AA-. Some pairs have two sets of hydration potentials, while other pairs only have one set. The parameters are shown in Table 1 of ref [1]. In the code, we use notations $\mu = r_{mh}$, $\eta = \sigma_h$, $\gamma = \frac{H}{\sigma_h \sqrt{2\pi}}$. So that the hydration potential is written as

$$
U_{hydr} = \gamma \exp{\left[-\frac{(r-\mu)^2}{2\eta^2}\right]}
$$

The cutoff for the each hydration term is set as $\mu + 10\eta$ so that the numerical error due to cutoff is negligible. 

In the code, the vdwl and hydration terms are combined in one `CustomNonbondedForce`. 






## References
[1] Lin, Xingcheng, and Bin Zhang. "Explicit ion modeling predicts physicochemical interactions for chromatin organization." Elife 12 (2024): RP90073.



