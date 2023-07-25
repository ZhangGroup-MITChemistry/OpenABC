import numpy as np
import pandas as pd
import sys
import os

sys.path.append('../../..')
import openabc.utils.helper_functions as helper_functions
from openabc.lib import _amino_acids, _dna_nucleotides

nucl = helper_functions.parse_pdb('1kx5.pdb')
nucl = nucl.loc[nucl['recname'] == 'ATOM'].copy()

dna = nucl.loc[nucl['resname'].isin(_dna_nucleotides)].copy()
protein = nucl.loc[nucl['resname'].isin(_amino_acids)].copy()

dna['serial'] = list(range(len(dna.index)))
protein['serial'] = list(range(len(protein.index)))

helper_functions.write_pdb(dna, 'dna.pdb')
helper_functions.write_pdb(protein, 'histone.pdb')



