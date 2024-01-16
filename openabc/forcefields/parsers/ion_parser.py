import numpy as np
import pandas as pd
from openabc.utils import parse_pdb

class IonParser(object):
    """
    Ion parser.
    """
    def __init__(self, pdb):
        """
        Initialize by loading pdb to parse ions. Non-ion atoms will be removed.
        
        Parameters
        ----------
        pdb : str
            Path to pdb file.
        
        """
        self.pdb = pdb
        atoms = parse_pdb(pdb)
        atoms = atoms[atoms['resname'].isin(['NA', 'MG', 'CL'])].copy()
        atoms = atoms.reset_index(drop=True)
        self.atoms = atoms
        
    
    
