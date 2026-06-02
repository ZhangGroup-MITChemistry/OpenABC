import numpy as np
import pandas as pd
import mdtraj
import warnings

def dual_boundary_tanh_step(r, r_min, r_max, eta):
    """
    Dual boundary tanh step function. 
    
    Parameters
    ----------
    r : np.ndarray
        The input array.
    
    r_min : float
        The left boundary. r_min <= r_max. 
    
    r_max : float
        The right boundary. r_min <= r_max. 
    
    eta : positive float
        The width of the step.
    
    Returns
    -------
    switch : np.ndarray, same shape as r
        The step function values.
        
    """
    assert isinstance(r, np.ndarray)
    assert r_min <= r_max
    assert eta > 0
    switch = (1 + np.tanh(eta * (r - r_min))) * (1 + np.tanh(eta * (r_max - r))) / 4
    return switch


def select_native_pairs_continuous_ss_dssp(old_native_pairs, aa_pdb, frame=0):
    """
    Select native pairs within continuous secondary structures based on DSSP.
    aa_pdb should only have 1 chain.

    Parameters
    ----------
    old_native_pairs : pd.DataFrame
        The old native pairs.
    
    aa_pdb : str
        All-atom pdb file.
    
    frame : int
        The frame index of aa_pdb to compute DSSP.
    
    Returns
    -------
    new_native_pairs : pd.DataFrame
        The new native pairs.
    
    """
    traj = mdtraj.load_pdb(aa_pdb)
    assert traj.n_chains == 1
    atoms, _ = traj.topology.to_dataframe()
    ca_atoms = atoms[(atoms['name'] == 'CA') & (atoms['element'] == 'C')]
    dssp = mdtraj.compute_dssp(traj)[frame].tolist()
    if 'NA' in dssp:
        warnings.warn(f'NA in dssp of {aa_pdb}')
    assert len(dssp) == len(ca_atoms.index)
    new_native_pairs = pd.DataFrame(columns=old_native_pairs.columns)
    for _, row in old_native_pairs.iterrows():
        a1 = int(row['a1'])
        a2 = int(row['a2'])
        if a1 > a2:
            a1, a2 = a2, a1
        if dssp[a1] in ['H', 'E']:
            if all([x == dssp[a1] for x in dssp[a1:a2 + 1]]):
                new_native_pairs.loc[len(new_native_pairs.index)] = row
    return new_native_pairs


def select_native_pairs_all_ss_dssp(old_native_pairs, aa_pdb, frame=0):
    """
    Select native pairs within secondary structures based on DSSP.
    Note all the native pairs from ordered secondary structures are kept.
    This means those native pairs from discontinuous secondary structures are also kept.  
    aa_pdb should only have 1 chain.

    Parameters
    ----------
    old_native_pairs : pd.DataFrame
        The old native pairs.
    
    aa_pdb : str
        All-atom pdb file.
    
    frame : int
        The frame index of aa_pdb to compute DSSP.
    
    Returns
    -------
    new_native_pairs : pd.DataFrame
        The new native pairs.
    
    """
    traj = mdtraj.load_pdb(aa_pdb)
    assert traj.n_chains == 1
    atoms, _ = traj.topology.to_dataframe()
    ca_atoms = atoms[(atoms['name'] == 'CA') & (atoms['element'] == 'C')]
    dssp = mdtraj.compute_dssp(traj)[frame].tolist()
    if 'NA' in dssp:
        warnings.warn(f'NA in dssp of {aa_pdb}')
    assert len(dssp) == len(ca_atoms.index)
    new_native_pairs = pd.DataFrame(columns=old_native_pairs.columns)
    for _, row in old_native_pairs.iterrows():
        a1 = int(row['a1'])
        a2 = int(row['a2'])
        if (dssp[a1] in ['H', 'E']) and (dssp[a2] in ['H', 'E']):
            new_native_pairs.loc[len(new_native_pairs.index)] = row
    return new_native_pairs

def write_pdb(atoms_df, filename):
    """
    Write a PDB file from a DataFrame with columns:
    ['serial','name','resname','chainID','x','y','z', ...]
    If 'resid' is missing, auto-generate it (renumbered per chain).
    """
    import numpy as np

    atoms = atoms_df.copy()

    # If no chainID column, assume single chain A
    if "chainID" not in atoms.columns:
        atoms["chainID"] = "A"

    # If no resid column, add one (sequential per chain)
    if "resid" not in atoms.columns:
        atoms["resid"] = 0
        for chain_id in atoms["chainID"].unique():
            mask = atoms["chainID"] == chain_id
            atoms.loc[mask, "resid"] = np.arange(1, mask.sum() + 1)

    with open(filename, "w") as f:
        for _, row in atoms.iterrows():
            f.write(
                "{:<6s}{:>5d} {:<4s}{:1s}{:>3s} {:1s}{:>4d}    "
                "{:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}\n".format(
                    row.get("recname", "ATOM"),
                    int(row.get("serial", 1)),
                    row.get("name", "CA"),
                    row.get("altLoc", ""),
                    row.get("resname", "GLY"),
                    row.get("chainID", "A"),
                    int(row.get("resid", 1)),
                    float(row["x"]),
                    float(row["y"]),
                    float(row["z"]),
                    float(row.get("occupancy", 1.0)),
                    float(row.get("tempFactor", 0.0)),
                    row.get("element", "C"),
                )
            )
        f.write("END\n")

