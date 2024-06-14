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
import warnings
try:
    import pdbfixer
except ImportError:
    pdbfixer = None
try:
    import openmm.unit as unit
except ImportError:
    import simtk.unit as unit
import sys
import os
from openabc.lib import (_amino_acids, _amino_acid_1_letter_to_3_letters_dict, 
                         _dna_nucleotides, _dna_WC_pair_dict, 
                         _rna_nucleotides)

"""
Some code is adapted from Open3SPN2. 

Open3SPN2 and OpenAWSEM paper: 
Lu, Wei, et al. "OpenAWSEM with Open3SPN2: A fast, flexible, and accessible framework for large-scale coarse-grained biomolecular simulations." PLoS computational biology 17.2 (2021): e1008308.
"""

_nucleotides = _dna_nucleotides + _rna_nucleotides

def parse_pdb(pdb_file):
    """
    Load pdb file as pandas dataframe.
    Note there should be only a single frame in the pdb file.
    
    Parameters
    ----------
    pdb_file : str
        Path for the pdb file. 
    
    Returns
    -------
    pdb_atoms : pd.DataFrame
        A pandas dataframe includes atom information. 
    
    """
    # load with mdtraj to ensure there is only 1 frame
    traj = mdtraj.load_pdb(pdb_file)
    assert traj.n_frames == 1
    def pdb_line(line):
        return dict(recname=str(line[0:6]).strip(),
                    serial=int(line[6:11]),
                    name=str(line[12:16]).strip(),
                    altLoc=str(line[16:17]),
                    resname=str(line[17:20]).strip(),
                    chainID=str(line[21:22]),
                    resSeq=int(line[22:26]),
                    iCode=str(line[26:27]),
                    x=float(line[30:38]),
                    y=float(line[38:46]),
                    z=float(line[46:54]),
                    occupancy=0.0 if line[54:60].strip() == '' else float(line[54:60]),
                    tempFactor=0.0 if line[60:66].strip() == '' else float(line[60:66]),
                    element=str(line[76:78].strip()),
                    charge=str(line[78:80].strip()))
    with open(pdb_file, 'r') as pdb:
        lines = []
        for line in pdb:
            if (len(line) > 6) and (line[:6] in ['ATOM  ', 'HETATM']):
                lines += [pdb_line(line)]
    pdb_atoms = pd.DataFrame(lines)
    pdb_atoms = pdb_atoms[['recname', 'serial', 'name', 'altLoc',
                           'resname', 'chainID', 'resSeq', 'iCode',
                           'x', 'y', 'z', 'occupancy', 'tempFactor',
                           'element', 'charge']]
    return pdb_atoms


def write_pdb(pdb_atoms, pdb_file, write_TER=False):
    """
    Write pandas dataframe to pdb file. 
    
    Parameters
    ----------
    pdb_atoms : pd.DataFrame
        A pandas dataframe includes atom information. 
    
    pdb_file : str
        Output path for the pdb file. 
    
    write_TER : bool
        Whether to write TER between two chains. 

    """
    chainID = None
    with open(pdb_file, 'w') as pdb:
        for i, atom in pdb_atoms.iterrows():
            if chainID is not None:
                if write_TER and (atom['chainID'] != chainID):
                    pdb.write('TER\n')
            chainID = atom['chainID']
            pdb_line = f'{atom.recname:<6}{int(atom.serial):>5} {atom["name"]:^4}{atom.altLoc:1}'+\
                       f'{atom.resname:<3} {atom.chainID:1}{int(atom.resSeq):>4}{atom.iCode:1}   '+\
                       f'{atom.x:>8.3f}{atom.y:>8.3f}{atom.z:>8.3f}' +\
                       f'{atom.occupancy:>6.2f}{atom.tempFactor:>6.2f}'+' ' * 10 +\
                       f'{atom.element:>2}{atom.charge:>2}'
            assert len(pdb_line) == 80, f'An item in the atom table is longer than expected ({len(pdb_line)})\n{pdb_line}'
            pdb.write(pdb_line + '\n')
        pdb.write('END\n')


def atomistic_pdb_to_ca_pdb(atomistic_pdb, ca_pdb, write_TER=False):
    """
    Convert atomistic pdb to protein CA pdb. 
    
    Parameters
    ----------
    atomistic_pdb : str
        Path for the atomistic pdb file. 
    
    ca_pdb : str
        Output path for the CA pdb file. 
    
    write_TER : bool
        Whether to write TER between two chains. 
    
    """
    atomistic_pdb_atoms = parse_pdb(atomistic_pdb)
    ca_pdb_atoms = pd.DataFrame(columns=atomistic_pdb_atoms.columns)
    for i, row in atomistic_pdb_atoms.iterrows():
        if (row['resname'] in _amino_acids) and (row['name'] == 'CA'):
            ca_pdb_atoms.loc[len(ca_pdb_atoms.index)] = row
    ca_pdb_atoms['serial'] = list(range(1, len(ca_pdb_atoms.index) + 1))
    ca_pdb_atoms.loc[:, 'charge'] = '' # remove charge
    write_pdb(ca_pdb_atoms, ca_pdb, write_TER)
    

def atomistic_pdb_to_nucleotide_pdb(atomistic_pdb, cg_nucleotide_pdb, write_TER=False):
    """
    Convert DNA or RNA atomistic pdb to nucleotide pdb (i.e. one CG bead per nucleotide). 
    The position of each CG nucleotide bead is the geometric center of all the nucleotide atoms in the pdb.
    
    Parameters
    ----------
    atomistic_pdb : str
        Path for the atomistic pdb file. 

    cg_nucleotide_pdb : str
        Output path for the CG pdb file. 
    
    write_TER : bool
        Whether to write TER between two chains. 
    
    """
    atomistic_pdb_atoms = parse_pdb(atomistic_pdb)
    atomistic_pdb_atoms = atomistic_pdb_atoms.loc[atomistic_pdb_atoms['resname'].isin(_nucleotides)].copy()
    atomistic_pdb_atoms.index = list(range(len(atomistic_pdb_atoms.index)))
    chainID = atomistic_pdb_atoms['chainID']
    resSeq = atomistic_pdb_atoms['resSeq']
    atomistic_pdb_atoms.index = chainID.astype(str) + '_' + resSeq.astype(str)
    cg_nucleotide_pdb_atoms = pd.DataFrame(columns=atomistic_pdb_atoms.columns)
    for each in atomistic_pdb_atoms.index.drop_duplicates():
        residue_atoms = atomistic_pdb_atoms.loc[each]
        coord = np.mean(residue_atoms[['x', 'y', 'z']].to_numpy(), axis=0)
        row = residue_atoms.iloc[0].copy()
        row[['x', 'y', 'z']] = coord
        resname = row['resname']
        # set atom name to distinguish DNA and RNA nucleotides
        if resname in _dna_nucleotides:
            name = 'DN'
        elif resname in _rna_nucleotides:
            name = 'RN'
        else:
            name = ''
        row[['name', 'occupancy', 'tempFactor', 'element', 'charge']] = [name, 1.0, 1.0, '', '']
        cg_nucleotide_pdb_atoms.loc[len(cg_nucleotide_pdb_atoms.index)] = row
    cg_nucleotide_pdb_atoms['serial'] = list(range(1, len(cg_nucleotide_pdb_atoms.index) + 1))
    write_pdb(cg_nucleotide_pdb_atoms, cg_nucleotide_pdb, write_TER)


def build_straight_CA_chain(sequence, r0=0.38):
    """
    Build a straight portein CA atom chain with given sequence. 
    
    Parameters
    ----------
    sequence : str
        Protein chain sequence (1 letter for each amino acid). 
    
    r0: float or int
        Distance in unit nm between two neighboring CA atoms. 
    
    Returns
    -------
    df_atoms : pd.DataFrame
        A pandas dataframe includes atom information. 
    
    """
    n_atoms = len(sequence)
    data = []
    for i in range(n_atoms):
        resname = _amino_acid_1_letter_to_3_letters_dict[sequence[i]]
        atom_i_dict = {'recname': 'ATOM', 'name': 'CA', 'altLoc': '', 'resname': resname, 'chainID': 'A', 
                       'iCode': '', 'occupancy': 1.0, 'tempFactor': 1.0, 'element': 'C', 'charge': ''}
        data.append(atom_i_dict)
    df_atoms = pd.DataFrame(data)
    df_atoms['serial'] = list(range(1, n_atoms + 1))
    df_atoms['resSeq'] = list(range(1, n_atoms + 1))
    df_atoms.loc[:, 'x'] = 0
    df_atoms.loc[:, 'y'] = 0
    z = r0*np.arange(n_atoms)
    z -= np.mean(z)
    df_atoms['z'] = z*10 # convert nm to angstroms
    df_atoms['z'] = df_atoms['z'].round(3)
    return df_atoms


def build_straight_chain(n_atoms, chainID, r0):
    """
    Build a straight chain. 
    Each atom is viewed as one residue. 
    
    Parameters
    ----------
    n_atoms : int
        The number of atoms along the chain. 
    
    chainID : floar or int or str
        Chain ID. 
    
    r0 : float or int
        Distance in unit nm between neighboring atoms along the chain. 
    
    Returns
    -------
    df_atoms : pd.DataFrame
        A pandas dataframe includes atom information. 
        
    """
    data = []
    for i in range(n_atoms):
        atom_i_dict = {'recname': 'ATOM', 'name': '', 'altLoc': '', 'resname': '', 'chainID': chainID, 
                       'iCode': '', 'occupancy': 1.0, 'tempFactor': 1.0, 'element': '', 'charge': ''}
        data.append(atom_i_dict)
    df_atoms = pd.DataFrame(data)
    df_atoms['serial'] = list(range(1, n_atoms + 1))
    df_atoms['resSeq'] = list(range(1, n_atoms + 1))
    df_atoms.loc[:, 'x'] = 0
    df_atoms.loc[:, 'y'] = 0
    z = r0*np.arange(n_atoms)
    z -= np.mean(z)
    df_atoms['z'] = z*10 # convert nm to angstroms
    df_atoms['z'] = df_atoms['z'].round(3)
    return df_atoms


def make_mol_whole(coord, box_a, box_b, box_c):
    """
    Use this function to make the molecule whole. 
    This is useful to recover the whole molecule if it is split by box boundary. 
    The position of the first atom is kept. 
    The other atoms are moved so that each one is in the closest periodic image relative to the previous one.
    
    Parameters
    ----------
    coord : np.ndarray, shape = (n_atoms, 3)
        Input atom coordinates. 
    
    box_a : float or int
        Box length along x-axis. 
    
    box_b : float or int
        Box length along y-axis. 
    
    box_c : float or int
        Box length along z-axis. 
    
    Returns
    -------
    new_coord : np.ndarray, shape = (n_atoms, 3)
        Output atom coordinates. 
    
    """
    n_atoms = coord.shape[0]
    new_coord = np.zeros((n_atoms, 3))
    for i in range(n_atoms):
        if i == 0:
            new_coord[i] = coord[i]
        else:
            r0 = new_coord[i - 1]
            r1 = coord[i]
            delta_r = r1 - r0
            delta_x, delta_y, delta_z = delta_r[0], delta_r[1], delta_r[2]
            while delta_x < -0.5*box_a:
                delta_x += box_a
            while delta_x >= 0.5*box_a:
                delta_x -= box_a
            while delta_y < -0.5*box_b:
                delta_y += box_b
            while delta_y >= 0.5*box_b:
                delta_y -= box_b
            while delta_z < -0.5*box_c:
                delta_z += box_c
            while delta_z >= 0.5*box_c:
                delta_z -= box_c
            new_coord[i] = r0 + np.array([delta_x, delta_y, delta_z])
    return new_coord
                

def move_atoms_to_closest_pbc_image(coord, ref_point, box_a, box_b, box_c):
    """
    Move the atom coordinates to the periodic image closest to ref_point. 
    
    Parameters
    ----------
    coord : np.ndarray, shape = (n_atoms, 3)
        Input atom coordinates. 
    
    ref_point : np.ndarray, shape = (3,)
        The reference point coordinate. 
    
    box_a : float or int
        Box length along x-axis. 
    
    box_b : float or int
        Box length along y-axis. 
    
    box_c : float or int
        Box length along z-axis. 
    
    Returns
    -------
    new_coord : np.ndarray, shape = (n_atoms, 3)
        Output atom coordinates. 
    
    """
    n_atoms = coord.shape[0]
    new_coord = np.zeros((n_atoms, 3))
    for i in range(n_atoms):
        delta_r = coord[i] - ref_point
        delta_x, delta_y, delta_z = delta_r[0], delta_r[1], delta_r[2]
        while delta_x < -0.5*box_a:
            delta_x += box_a
        while delta_x >= 0.5*box_a:
            delta_x -= box_a
        while delta_y < -0.5*box_b:
            delta_y += box_b
        while delta_y >= 0.5*box_b:
            delta_y -= box_b
        while delta_z < -0.5*box_c:
            delta_z += box_c
        while delta_z >= 0.5*box_c:
            delta_z -= box_c
        new_coord[i] = ref_point + np.array([delta_x, delta_y, delta_z])
    return new_coord
        

def compute_rg(coord, mass):
    """
    Compute radius of gyration.
    
    Parameters
    ----------
    coord : np.ndarray, shape = (n_atoms, 3)
        Input atom coordinates. 
    
    mass : np.ndarray, shape = (n_atoms,)
        Input atom mass. 
    
    Returns
    -------
    rg : float
        Radius of gyration. 
    
    """
    weights = mass/np.sum(mass)
    r_COM = np.average(coord, axis=0, weights=weights)
    rg_square = np.average(np.sum((coord - r_COM)**2, axis=1), weights=weights)
    rg = rg_square**0.5
    return rg


def fix_pdb(pdb_file):
    """
    Fix pdb file with pdbfixer. The fixed structure is output as pandas dataframe. 
    This function is adapted from open3SPN2.
    
    Parameters
    ----------
    pdb_file : str
        PDB file path. 
    
    Returns
    -------
    atoms : pd.DataFrame
        Output fixed structure. 

    """
    assert pdbfixer is not None, 'pdbfixer is not installed.'
    fixer = pdbfixer.PDBFixer(filename=pdb_file)
    fixer.findMissingResidues()
    chains = list(fixer.topology.chains())
    keys = fixer.missingResidues.keys()
    for key in list(keys):
        chain_tmp = chains[key[0]]
        if key[1] in [0, len(list(chain_tmp.residues()))]:
            del fixer.missingResidues[key]

    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    
    columns = ['recname', 'serial', 'name', 'altLoc',
               'resname', 'chainID', 'resSeq', 'iCode',
               'x', 'y', 'z', 'occupancy', 'tempFactor',
               'element', 'charge']
    data = []
    for atom, pos in zip(fixer.topology.atoms(), fixer.positions):
        residue = atom.residue
        chain = residue.chain
        pos = pos.value_in_unit(unit.angstrom)
        data += [dict(zip(columns, ['ATOM', int(atom.id), atom.name, '',
                                    residue.name, chain.id, int(residue.id), '',
                                    pos[0], pos[1], pos[2], 0, 0,
                                    atom.element.symbol, '']))]
    atoms = pd.DataFrame(data)
    atoms = atoms[columns]
    atoms.index = atoms['serial']
    return atoms


def get_WC_paired_sequence(sequence):
    """
    Get WC-paired sequence. 
    
    Parameters
    ----------
    sequence : str
        Input sequence. 
        
    Returns
    -------
    paired_sequence : str
        W-C paired sequence. 
    
    """
    paired_sequence = ''
    for i in range(len(sequence)):
        paired_sequence += _dna_WC_pair_dict[sequence[i]]
    paired_sequence = paired_sequence[::-1]
    return paired_sequence
    

def compute_PE(traj, system, platform_name='Reference', properties={'Precision': 'double'}, groups=-1):
    """
    Compute the potential energy of samples in the trajectory with the given openmm system.
    
    Parameters
    ----------
    traj : mdtraj.Trajectory
        The trajectory including all the samples.
    
    system : openmm.System
        The openmm system.
    
    platform_name : str
        The platform name for running openmm to evaulate energy.
    
    properties : dict
        Properties for running openmm. Only used when platform_name is CUDA or OpenCL.
    
    groups : int or set
        The force groups included when computing energy.
        See openmm context.getState for details about this parameter.
    
    Returns
    -------
    energy : np.ndarray, shape = (traj.n_frames,)
        The energy of each sample in the trajectory in unit kJ/mol.
    
    """
    top = traj.topology.to_openmm()
    # use any integrator with any parameter
    timestep = 1.0 * unit.femtosecond
    integrator = mm.VerletIntegrator(timestep)
    platform = mm.Platform.getPlatformByName(platform_name)
    if platform_name in ['CUDA', 'OpenCL']:
        simulation = app.Simulation(top, system, integrator, platform, properties)
    else:
        simulation = app.Simulation(top, system, integrator, platform)
    use_pbc = system.usesPeriodicBoundaryConditions()
    energy = []
    for i in range(traj.n_frames):
        simulation.context.setPositions(traj.xyz[i])
        state = simulation.context.getState(getEnergy=True, enforcePeriodicBox=use_pbc, groups=groups)
        energy.append(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
    energy = np.array(energy)
    return energy

