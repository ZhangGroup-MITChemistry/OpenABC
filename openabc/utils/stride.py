import numpy as np
import pandas as pd

def parse_stride(stride_file):
    """
    Parse stride output file. 
    
    Parameters
    ----------
    stride_file : str
        The stride output file. 
    
    Returns
    -------
    df_data : pd.DataFrame
        A dataframe with the secondary structure information. 
    
    """
    with open(stride_file, 'r') as f:
        stride_lines = f.readlines()
    df_data = pd.DataFrame(columns=['resname', 'pdb_resid', 'ordinal_resid', 'ss_abbr', 'ss', 'phi', 'psi', 'rsaa'])
    for each_line in stride_lines:
        if len(each_line) >= 1:
            elements = each_line.split()
            if elements[0] == 'ASG':
                resname = each_line[5:8]
                pdb_resid = int(each_line[11:15].strip())
                ordinal_resid = int(each_line[16:20].strip())
                ss_abbr = each_line[24] # secondary structure name abbreviation
                ss = each_line[26:39].strip() # full secondary structure name
                phi = float(each_line[42:49].strip())
                psi = float(each_line[52:59].strip())
                rsaa = float(each_line[64:69].strip()) # residue solvent accessible area
                row = [resname, pdb_resid, ordinal_resid, ss_abbr, ss, phi, psi, rsaa]
                df_data.loc[len(df_data.index)] = row
    return df_data


