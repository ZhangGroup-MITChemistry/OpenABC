## OpenABC

OpenABC stands for **Open**MM GPU-**A**ccelerated simulations of **B**iomolecular **C**ondensates. It is flexible and implements multiple popular coarse-grained force fields for simulations, including the hydropathy scale (HPS) model, MOFF C<sub>$\alpha$</sub> model, and the molecular renormalization group (MRG)-CG DNA model. The package dramatically simplifies the simulation setup: users only need a few lines of python code to carry out condensate simulations starting from initial configurations of a single protein or DNA. The package is integrated with OpenMM, a GPU-accelerated MD simulation engine, enabling efficient simulations with advanced sampling techniques. We include tools for converting coarse-grained configurations to atomistic structures for further simulations with all-atom force fields. We provide tutorials in Jupyter notebooks to demonstrate the various capabilities. We anticipate OpenABC to significantly facilitate the application of existing computer models for simulating biomolecular condensates and the continued development of new force fields.

<img src="./image/flowchart.png" width="500px"><img>

## Tutorials

Detailed tutorials with all the necessary input files are provided in "tutorials" directory. 


## Manual

The output html manual file is docs/index.html. 

The manual is also shown in: https://zhanggroup-mitchemistry.github.io/OpenABC/

Instructions for class methods and functions are also included as comments in the source code. 


## Environment

We recommend using openmm 7.5.1 for using OpenABC, as OpenABC is built based on openmm 7.5.1. 

Install openmm 7.5.1 with the following command: 

```
conda install -c conda-forge openmm=7.5.1
```

Other required packages: numpy, pandas, mdanalysis, mdtraj, openmmplumed. 

If running replica exchange with `openabc.utils.replica_exchange`, then torch is also required. 


## Installation

The user can either download the package from github, or use pip install:

```
pip install openabc
```


## Usage

If openabc is not within the default python module searching paths, please add it to the searching paths. One way is to use `sys.path.append(dir_path)`, and replace `dir_path` with the directory path where openabc is saved. 

Here is an example of setting up a MOFF system composed of 100 copies of proteins. 

```python
from openabc.forcefields.parsers import MOFFParser
from openabc.forcefields import MOFFMRGModel
from openabc.utils.insert import insert_molecules
import simtk.openmm.app as app

# Parse structural and topological information
protein = MOFFParser.from_atomistic_pdb('all_atom.pdb', 'Calpha.pdb')

# Build initial condensate configuration with N = 100 proteins
N = 100
a, b, c = 100, 100, 100 # box sizes
insert_molecules('Calpha.pdb', 'start.pdb', n_mol=N, box=[a, b, c])

# Create molecule container and OpenMM system
condensate = MOFFMRGModel()
for i in range(N):
    condensate.append_mol(protein)
top = app.PDBFile('start.pdb').getTopology()
condensate.create_system(top, box_a=a, box_b=b, box_c=c)
condensate.add_all_default_forces()

```

Please read the tutorials for more instructions. 


## Tip

For version 1.0.2: When using `openabc.utils.insert.insert_molecules`, users may see such warnings: `UserWarning: Found no information for attr: 'formalcharges' Using default value of '0'` and `warnings.warn("Found no information for attr: '{}'"`. Such warning can be ignored as atom charge is not required when we insert molecules. To suppress such warning, users can add such lines at the beginning of the script:

```
import warnings
warnings.filterwarnings('ignore')
```

Starting from version 1.0.3, we use other ways to write pdb file for `openabc.utils.insert.insert_molecules`, and such warning will not appear. 


## Extension

If the user intends to add new force fields, then the user has to write new parsers, new models, and expressions of new forces. Take HPS model as an example, the main components are `openabc/forcefields/parsers/hps_parser.py` and `openabc/forcefields/hps_model.py`. `openabc/forcefields/parsers/hps_parser.py` includes a parser that can parse each individual protein and get all the bonded interactions. `openabc/forcefields/hps_model.py` includes a container-like class that can hold multiple protein parser objects and add forces. Definitions of different potentials are saved in `openabc/forcefields/functional_terms/*_terms.py`. 


## Important notes and updates

Some important notes and updates are listed: 

### Version 1.0.2

The first version available with pip install. 

### Version 1.0.3 

Update `openabc.utils.insert.insert_molecules`, MDAnalysis warnings will not appear. Both "FastNS" and "distance_array" methods are supported. 


## Citations

Please cite the following paper if you use openabc package: 

    "OpenABC Enables Flexible, Simplified, and Efficient GPU Accelerated Simulations of Biomolecular Condensates", doi: https://doi.org/10.1101/2023.04.19.537533


