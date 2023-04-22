try:
    ipy_str = str(type(get_ipython()))
    if 'zmqshell' in ipy_str:
        from tqdm.notebook import tqdm
    if 'terminal' in ipy_str:
        from tqdm import tqdm
except:
    from tqdm import tqdm
import mdtraj
import os
import argparse
import subprocess
import warnings
from openabc.utils.helper_functions import parse_pdb, write_pdb

__location__ = os.path.dirname(os.path.abspath(__file__))

__author__ = 'Cong Wang'

"""
This python script calls REMO (https://zhanggroup.org/REMO/) to reconstruct atomic configurations for protein condensates.
"""


default_REMO_path = __location__ + '/REMO'

def args_parse():
    """
    The function uses argparse module to create and manage the command-line interface.

    Returns
    -------
    parser.parse_args(): the parsed command-line arguments object.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', type=str, help='A string that represents the path to the input protein PDB file')
    parser.add_argument('-N', '--number_of_chains', nargs='+', type=int, help='A list of integers, where each element represents the number of chains for a particular protein type')
    parser.add_argument('-n', '--number_of_residues', nargs='+', type=int, help='A list of integers, where each element represents the number of residues for a particular protein type')
    parser.add_argument('-d', '--debug', action='store_true', help='A boolen variable that has a default value of False. By default this script does not store temporary files')
    return parser.parse_args()


def split_protein(input_pdbfile, num_chains, num_residues, output_path):
    """
    Split a PDB file containing multiple chains into individual files, each containing one chain.
    
    Parameters
    ----------
    input_pdbfile : str
        Input pdb file. 
    
    num_chains : list
        Number of protein chains in the system. Each element of the list represent one type of protein.
        
    num_residues : list
        Length of protein chains in the system. Each element of the list represent one type of protein. 
        Note `num_chains` and `num_residues` share the same length unit. 

    output_path : str
        The folder where user wants to save output pdb files.  
           
    """
    pdb = parse_pdb(input_pdbfile)
    assert len(num_chains) == len(num_residues)
    protein_types = len(num_chains)
    current_index = 0
    os.makedirs(f'{output_path}', exist_ok=True)
    for i in range(protein_types):
        for j in range(num_chains[i]):
            pdb_i_j = pdb[current_index: current_index + num_residues[i]]
            write_pdb(pdb_i_j, f'{output_path}/protein_{i}_chain_{j}.pdb')
            current_index += num_residues[i]
    if current_index != len(pdb):
        raise ValueError(f'Number of CA in pdb file not consistant with num_chains and num_residues.')
    return


def single_chain_CA2AA(input_pdbfile, output_pdbfile, REMO_path=default_REMO_path, debug=False):
    """
    Reconstruct all-atom representation for a single-chain protein from its alpha carbons using REMO.
    
    Parameters
    ----------
    input_pdbfile : str
        Input pdb file that contains one protein chain. 
    
    output_pdbfile : str
        output pdb file.
        
    REMO_path : str
        Path where REMO program locates.

    debug : bool, default=False
        If True, temporary files will not be deleted.
    
    References
    ----------
    Yunqi Li and Yang Zhang. REMO: A new protocol to refine full atomic protein models from C-alpha traces by optimizing hydrogen-bonding networks. Proteins, 2009, 76: 665-676.
    https://zhanggroup.org/REMO/.
    """
    REMO_path = os.path.expanduser(REMO_path)
    if not os.path.exists(f'{REMO_path}/REMO.pl'):
        raise FileNotFoundError(f'REMO.pl not found in {REMO_path}')
    output_pdbfile_path = '/'.join(output_pdbfile.split('/')[:-1])
    os.makedirs(output_pdbfile_path, exist_ok=True)
    if debug:
        subprocess.run([f'{REMO_path}/REMO.pl', '0', input_pdbfile])
    else: 
        subprocess.run([f'{REMO_path}/REMO.pl', '0', input_pdbfile], 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.STDOUT)
    subprocess.run(['mv', f'{input_pdbfile}.h', output_pdbfile])
    subprocess.run(['rm', 'PRHB'])
    return


def align_protein(input_pdbfile, reference_pdbfile, output_pdbfile, reference_atoms='CA'):
    """
    Align a protein with the reference protein accoding to positions of alpha carbons.
    
    Parameters
    ----------
    input_pdbfile : str
        Input pdb file to be aligned. 
    
    reference_pdbfile : str
        Pdb file that be used as reference.
        Note 'input_pdbfile' and 'reference_pdbfile' must represent the same system.

    output_pdbfile : str
        Aligned pdb file.

    reference_atoms : str, default='CA'
        Name of the atoms that used as reference atoms. Currently only support alpha carbons.
    
    References
    ----------
    Yunqi Li and Yang Zhang. REMO: A new protocol to refine full atomic protein models from C-alpha traces by optimizing hydrogen-bonding networks. Proteins, 2009, 76: 665-676.
    https://zhanggroup.org/REMO/.
    """
    input_pdb = mdtraj.load(input_pdbfile)
    ref_pdb = mdtraj.load(reference_pdbfile)
    input_pdb_CA_index = input_pdb.top.select(f'name=={reference_atoms}')
    ref_pdb_CA_index= ref_pdb.top.select(f'name=={reference_atoms}')
    assert len(input_pdb_CA_index) == len(ref_pdb_CA_index)
    input_pdb_aligned = input_pdb.superpose(
                                            ref_pdb, 
                                            frame=0, 
                                            atom_indices=input_pdb_CA_index, 
                                            ref_atom_indices=ref_pdb_CA_index
                                            )
    output_pdbfile_path = '/'.join(output_pdbfile.split('/')[:-1])
    os.makedirs(output_pdbfile_path, exist_ok=True)
    input_pdb_aligned.save(output_pdbfile)
    return


def combine_proteins(pdbfile_lis, output_pdbflie):
    """
    Merge multiple pdb files into a single pdb file.
    
    Parameters
    ----------
    pdbfile_lis : str
        List that contains name of pdb files to be merged. 
    
    output_pdbfile : str
        Merged pdb file.
        
    """
    pdb_list = [mdtraj.load(i) for i in pdbfile_lis]
    pdb_combined = pdb_list[0]
    for i in range(len(pdb_list) - 1):
        pdb_combined = pdb_combined.stack(pdb_list[i + 1])
    pdb_combined.save(output_pdbflie)
    return


def multiple_chains_CA2AA(input_pdbfile, num_chains, num_residues, REMO_path=default_REMO_path, debug=False):
    """
    Reconstruct all-atom representation of a multiple-chain protein from its alpha carbons. If the name of input file is 'XXX.pdb', then output will be saved as 'XXX_AA.pdb' in the folder where 'XXX.pdb' is located.
    
    Parameters
    ----------
    input_pdbfile : str
        Input pdb file. 

    num_chains : list
        Number of protein chains in the system. Each element of the list represent one type of protein.
        
    num_residues : list
        Length of protein chains in the system. Each element of the list represent one type of protein. 
        Note `num_chains` and `num_residues` share the same length unit. 

    REMO_path : str
        Path where REMO program locates.

    debug : bool, default=False
        If True, temporary files will not be deleted.

    References
    ----------
    Yunqi Li and Yang Zhang. REMO: A new protocol to refine full atomic protein models from C-alpha traces by optimizing hydrogen-bonding networks. Proteins, 2009, 76: 665-676.
    https://zhanggroup.org/REMO/.
    """
    assert len(num_chains) == len(num_residues)
    pdb_name = input_pdbfile.split('/')[-1].split('.')[0]
    split_protein(input_pdbfile, num_chains, num_residues, output_path=f'{pdb_name}/ca_splitted')
    aligned_protein_list = []
    for i in tqdm(range(len(num_chains))):
        for j in tqdm(range(num_chains[i])):
            single_chain_CA2AA(
                                 f'{pdb_name}/ca_splitted/protein_{i}_chain_{j}.pdb', 
                                 f'{pdb_name}/aa_splitted_noaligned/protein_{i}_chain_{j}.pdb', 
                                 REMO_path=REMO_path, 
                                 debug=debug
                                 )
            align_protein(
                          f'{pdb_name}/aa_splitted_noaligned/protein_{i}_chain_{j}.pdb', 
                          f'{pdb_name}/ca_splitted/protein_{i}_chain_{j}.pdb', 
                          f'{pdb_name}/aa_splitted_aligned/protein_{i}_chain_{j}.pdb', 
                          reference_atoms='CA'
                          )
            aligned_protein_list.append(f'{pdb_name}/aa_splitted_aligned/protein_{i}_chain_{j}.pdb')
    combine_proteins(aligned_protein_list, f'{pdb_name}_AA.pdb')
    if not debug:
        subprocess.run(['rm', '-r', pdb_name])
    return


if __name__ == '__main__':
    args = args_parse()
    protein_ca = args.input_file
    num_chains = args.number_of_chains
    num_residues = args.number_of_residues
    debug = args.debug
    multiple_chains_CA2AA(protein_ca, num_chains, num_residues, REMO_path=default_REMO_path, debug=debug)

