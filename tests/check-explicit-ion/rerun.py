import numpy as np
import pandas as pd
try:
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
except ImportError:
    import simtk.openmm as mm
    import simtk.openmm.app as app
    import simtk.unit as unit
import mdtraj
import sys
import os
sys.path.insert(0, '../..') # ensure use the specific openabc we aim to test
from openabc.forcefields.parsers import SMOGParser, DNA3SPN2Parser
from openabc.forcefields import SMOG3SPN2ExplicitIonModel

"""
Rerun a protein-DNA system with explicit ion model.
"""

protein_parser = SMOGParser.from_atomistic_pdb('protein.pdb', 'cg_protein.pdb')
dna_parser = DNA3SPN2Parser.from_atomistic_pdb('dna.pdb', 'cg_dna.pdb')
model = SMOG3SPN2ExplicitIonModel()
model.append_mol(protein_parser)
model.append_mol(dna_parser)

n_NA_ions = 514
n_CL_ions = 482
model.append_ions('NA', n_NA_ions)
model.append_ions('CL', n_CL_ions)




