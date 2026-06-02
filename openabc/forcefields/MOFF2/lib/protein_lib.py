_amino_acids = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS',
                'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                'LEU', 'LYS', 'MET', 'PHE', 'PRO',
                'SER', 'THR', 'TRP', 'TYR', 'VAL']

_amino_acid_1_letter_to_3_letters_dict = dict(A='ALA', R='ARG', N='ASN', D='ASP', C='CYS', 
                                              Q='GLN', E='GLU', G='GLY', H='HIS', I='ILE', 
                                              L='LEU', K='LYS', M='MET', F='PHE', P='PRO', 
                                              S='SER', T='THR', W='TRP', Y='TYR', V='VAL')

_amino_acid_3_letters_to_1_letter_dict = {v: k for k, v in _amino_acid_1_letter_to_3_letters_dict.items()}

# _amino_acid_mass_dict is consistent with the values used by the HPS model
_amino_acid_mass_dict = dict(ALA=71.08, ARG=156.20, ASN=114.10, ASP=115.10, CYS=103.10, 
                             GLN=128.10, GLU=129.10, GLY=57.05, HIS=137.10, ILE=113.20, 
                             LEU=113.20, LYS=128.20, MET=131.20, PHE=147.20, PRO=97.12, 
                             SER=87.08, THR=101.10, TRP=186.20, TYR=163.20, VAL=99.07)

# _hps_amino_acid_sigma_dict includes the size parameter (unit: nm) of amino acids in HPS model
_hps_amino_acid_sigma_dict = dict(ALA=0.504, ARG=0.656, ASN=0.568, ASP=0.558, CYS=0.548, 
                                  GLN=0.602, GLU=0.592, GLY=0.450, HIS=0.608, ILE=0.618,
                                  LEU=0.618, LYS=0.636, MET=0.618, PHE=0.636, PRO=0.556,
                                  SER=0.518, THR=0.562, TRP=0.678, TYR=0.646, VAL=0.586)

_res_group_mappings = {
    "default": {res: 0 for res in _amino_acids},
    "hydrophobic_polar": {
        **{res: "hydrophobic" for res in ['ALA', 'GLY', 'ILE', 'LEU', 'MET', 'PHE', 'PRO', 'TRP', 'VAL']},
        **{res: "polar" for res in ['ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'HIS', 'LYS', 'SER', 'THR', 'TYR']}
    },
    "hydrophobic_polar_positive_negative": {
        **{res: "01hydrophobic" for res in ['ALA', 'GLY', 'ILE', 'LEU', 'MET', 'PHE', 'PRO', 'TRP', 'VAL']},
        **{res: "02polar" for res in ['CYS', 'SER', 'THR', 'ASN', 'GLN', 'TYR']},
        **{res: "03positive" for res in ['ARG', 'LYS', 'HIS']},
        **{res: "04negative" for res in ['ASP', 'GLU']}
    }
}
