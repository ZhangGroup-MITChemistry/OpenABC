from Bio.PDB.Polypeptide import (
    index_to_one,
    one_to_index,
    three_to_index,
    index_to_three,
)

_amino_acids = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS',
                'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                'LEU', 'LYS', 'MET', 'PHE', 'PRO',
                'SER', 'THR', 'TRP', 'TYR', 'VAL']
assert _amino_acids == sorted([index_to_three(i) for i in range(20)])

_amino_acid_one_to_three_dict = {index_to_one(i): index_to_three(i) for i in range(20)}
_amino_acid_three_to_one_dict = {v: k for k, v in _amino_acid_one_to_three_dict.items()}

_amino_acid_1_letter_to_3_letters_dict = _amino_acid_one_to_three_dict
_amino_acid_3_letters_to_1_letter_dict = _amino_acid_three_to_one_dict


