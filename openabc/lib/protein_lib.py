_amino_acids = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS',
                'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                'LEU', 'LYS', 'MET', 'PHE', 'PRO',
                'SER', 'THR', 'TRP', 'TYR', 'VAL']

_amino_acid_1_letter_to_3_letters_dict = dict(A='ALA', R='ARG', N='ASN', D='ASP', C='CYS', 
                                              Q='GLN', E='GLU', G='GLY', H='HIS', I='ILE', 
                                              L='LEU', K='LYS', M='MET', F='PHE', P='PRO', 
                                              S='SER', T='THR', W='TRP', Y='TYR', V='VAL')

_amino_acid_3_letters_to_1_letter_dict = {v: k for k, v in _amino_acid_1_letter_to_3_letters_dict.items()}

