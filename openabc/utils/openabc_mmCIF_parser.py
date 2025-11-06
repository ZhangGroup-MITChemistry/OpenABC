import os
import pandas as pd

mmCIF_PDBx_keyword_order = ["group_PDB",
                            "id",
                            "type_symbol",
                            "label_atom_id",
                            "label_alt_id",
                            "label_comp_id",
                            "label_asym_id",
                            "label_entity_id",
                            "label_seq_id",
                            "pdbx_PDB_ins_code",
                            "Cartn_x",
                            "Cartn_y",
                            "Cartn_z",
                            "occupancy",
                            "B_iso_or_equiv",
                            "pdbx_formal_charge",
                            "auth_seq_id",
                            "auth_comp_id",
                            "auth_asym_id",
                            "auth_atom_id",
                            "pdbx_PDB_model_num"
                            ]
# ""


#_atom_site.group_PDB 
#_atom_site.id 
#_atom_site.type_symbol 
#_atom_site.label_atom_id 
#_atom_site.label_alt_id 
#_atom_site.label_comp_id 
#_atom_site.label_asym_id 
#_atom_site.label_entity_id 
#_atom_site.label_seq_id 
#_atom_site.pdbx_PDB_ins_code 
#_atom_site.Cartn_x 
#_atom_site.Cartn_y 
#_atom_site.Cartn_z 
#_atom_site.occupancy 
#_atom_site.B_iso_or_equiv 
#_atom_site.pdbx_formal_charge 
#_atom_site.auth_seq_id 
#_atom_site.auth_comp_id 
#_atom_site.auth_asym_id 
#_atom_site.auth_atom_id 
#_atom_site.pdbx_PDB_model_num 


mmCIF_to_PDB_keyword_map = {'group_PDB':'recname',
                            'id':'serial',
                            'label_atom_id':'name', 
                            'label_alt_id':'altLoc',
                            'label_comp_id':'resname', 
                            'label_asym_id':'chainID', 
                            'label_seq_id':'resSeq', 
                            'pdbx_PDB_ins_code':'iCode',
                            'Cartn_x':'x', 
                            'Cartn_y':'y', 
                            'Cartn_z':'z', 
                            'occupancy':'occupancy', 
                            'B_iso_or_equiv':'tempFactor',
                            'type_symbol':'element', 
                            'pdbx_formal_charge':'charge',
                            "auth_seq_id":'resSeq',
                            "auth_comp_id":'resname',
                            "auth_asym_id":'chainID',
                            "auth_atom_id":'name'}

PDB_to_mmCIF_keyword_map = {v:k for k,v in mmCIF_to_PDB_keyword_map.items()}

def convert_to_pdb_fmt(table,entries,keymap):
    for k in table.keys():
        e = entries[keymap[k]]
        if k == 'serial':
            e = int(e)
        if k == 'resSeq':
            e = int(e)
        if k == 'x' or k == 'y' or k == 'z':
            e = float(e)
        if k == 'occupancy' or k == 'tempFactor':
            e = float(e)
        table[k].append(e)
    return table

def process_loop_block(line:str) -> tuple[str,str|None]:
    if line[0] != "_":
        return 'fr',None

    tok = line.split(".")
    if tok[0] != "_atom_site":
        return 'fc',None
    return '',tok[1]

def parse_table_entry(line:str):
    within_token = False
    token_delim = " "
    entries = []
    current_entry = []
    for c in line:
        #print(c,end="")
        if c != " " and within_token == False: 
            within_token = True
            if c == "\"" or c == "\'":
                token_delim = c
                #print()
                continue
            else:
                token_delim = " "
        if within_token:
            if c == token_delim:
                within_token = False
                entry = "".join(current_entry)
                if entry == "." or entry == "?":
                    entry = None
                entries.append(entry)
                current_entry = []
                #print()
                continue
            current_entry.append(c)
            #print("<- Token Entry")
    if within_token:
        entries.append("".join(current_entry))
        current_entry = []
        #print()
    return entries

def write_mmCIF(atoms,filename):
    """
    Write pandas dataframe to mmCIF file. This is currently only guaranteed to correctly write mmCIF files
    for NEAT-DNA.
    
    Parameters
    ----------
    atoms : pd.DataFrame
        A pandas dataframe includes atom information. 
    
    filename : str
        Output path for the mmCIF file. 
    
    """
    with open(filename,'w') as f:
        myatoms = atoms.copy()
        #myatoms['name'] = myatoms['name'].apply(lambda x: f"'{x}'")
        myatoms = myatoms.replace(r'^\s*$', None, regex=True)
        myatoms[mmCIF_to_PDB_keyword_map['label_alt_id']] = "."
        myatoms["label_entity_id"] = "1"
        myatoms["pdbx_PDB_model_num"] = "1"
        print('data_CG_model_OpenABC',file=f)
        print('#',file=f)
        print('loop_',file=f)
        outcolumns=mmCIF_PDBx_keyword_order
        valcolumns=[mmCIF_to_PDB_keyword_map.get(k,k) for k in mmCIF_PDBx_keyword_order]
        for k in outcolumns:
            print(f'_atom_site.{k}',file=f)
        print(myatoms.to_csv(sep=' ',na_rep='?',
                           float_format='%.3f',
                           columns=valcolumns,
                           index=False,
                           header=False),file=f,end='')
        print('#',file=f)

def parse_mmCIF(mmCIF_file:str):
    """
    Load mmCIF file as pandas dataframe.
    Note there should be only a single frame in the mmCIF file.
    
    Parameters
    ----------
    mmCIF_file : str
        Path for the mmCIF file. 
    
    Returns
    -------
    atom_table : pd.DataFrame
        A pandas dataframe with atom information in OpenABC compatible format
    
    """
    atom_table_order = []
    atom_table = {}
    atom_table_key_idx = {}
    in_loop_block = False
    reading_table = False
    sz_f = os.path.getsize(mmCIF_file)
    sz_cur = 0
    n_data = 0
    with open(mmCIF_file, "r+") as f:
        print(f"Processing mmCIF file: {mmCIF_file}")
        for i,line in enumerate(f):
            sz_cur += len(line.encode('utf-8'))
            percent = int(100 * sz_cur/sz_f)
            print("\r", end="")
            print(f"Reading: {percent}%", end="")
            line = line.rstrip()
            if len(line) == 0:
                continue
            if line.startswith("data_"):
                n_data += 1
                continue
            assert n_data < 2, f"mmCIF File ({mmCIF_file}) contains multiple frames: {n_data}. Cannot handle more than 1 frame!"
            if in_loop_block:
                a_codes, table_entry = process_loop_block(line) 
                if table_entry is not None:
                    atom_table_order.append(table_entry)
                for a_code in a_codes:
                    if a_code == 'f':
                        in_loop_block = False
                    if a_code == 'c':
                        continue
                    if a_code == 'r':
                        reading_table = True
                        for i,ato in enumerate(atom_table_order):
                            if (ato in mmCIF_to_PDB_keyword_map.keys()) and ('auth' not in ato):
                                atom_table[mmCIF_to_PDB_keyword_map[ato]] = []
                                atom_table_key_idx[mmCIF_to_PDB_keyword_map[ato]] = i
            if line == "loop_":
                in_loop_block = True
            if reading_table:
                if line[0] == "_":
                    reading_table = False
                else:
                    entries = parse_table_entry(line)
                    atom_table = convert_to_pdb_fmt(atom_table,entries,atom_table_key_idx)
    atom_table = pd.DataFrame.from_dict(atom_table)
    atom_table = atom_table.fillna(value="")
    print(" (Done)")
    return atom_table
