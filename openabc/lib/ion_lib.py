import pandas as pd

df_ion_template = pd.DataFrame(dict(recname=['ATOM'], 
                                    serial=[0], 
                                    name=[None], 
                                    altLoc=[''], 
                                    resname=[None], 
                                    chainID=['X'], 
                                    resSeq=[0], 
                                    iCode=[''], 
                                    x=[0.0], 
                                    y=[0.0], 
                                    z=[0.0], 
                                    occupancy=[0.0], 
                                    tempFactor=[0.0], 
                                    element=[None], 
                                    charge=[None]))

# sodium ion
df_sodium_ion = df_ion_template.copy()
df_sodium_ion.loc[:, ['name', 'resname']] = 'NA'
df_sodium_ion.loc[:, 'element'] = 'Na'
df_sodium_ion.loc[:, 'charge'] = 1.0
df_sodium_ion.loc[:, 'mass'] = 22.99

# chloride ion
df_chloride_ion = df_ion_template.copy()
df_chloride_ion.loc[:, ['name', 'resname']] = 'CL'
df_chloride_ion.loc[:, 'element'] = 'Cl'
df_chloride_ion.loc[:, 'charge'] = -1.0
df_chloride_ion.loc[:, 'mass'] = 35.45

# magnesium ion
df_magnesium_ion = df_ion_template.copy()
df_magnesium_ion.loc[:, ['name', 'resname']] = 'MG'
df_magnesium_ion.loc[:, 'element'] = 'Mg'
df_magnesium_ion.loc[:, 'charge'] = 2.0
df_magnesium_ion.loc[:, 'mass'] = 24.305

