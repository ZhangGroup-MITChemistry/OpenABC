import numpy as np
import pandas as pd
import sys
import os

"""
Use this script to produce the .csv files for MOFF nonbonded interaction parameters. 
The nonbonded interaction form is: abs(epsilon)*(sigma^12)/(r^12)-0.5*epsilon*(1+tanh(eta*(r0-r))).
Set alpha=abs(epsilon)*(sigma^12).
"""
df_contact_parameters = pd.DataFrame(columns=['atom_type1', 'atom_type2', 'alpha', 'epsilon', 'sigma'])

with open('template_MOFF.top', 'r') as input_reader:
    template_MOFF_lines = input_reader.readlines()

n_lines = len(template_MOFF_lines)
for i in range(n_lines):
    if 'nonbond_params' in template_MOFF_lines[i]:
        start_line_index = i + 2
    if 'moleculetype' in template_MOFF_lines[i]:
        end_line_index = i - 1

for i in range(start_line_index, end_line_index + 1):
    elements = template_MOFF_lines[i].split()
    if len(elements) == 5:
        atom_type1 = elements[0]
        atom_type2 = elements[1]
        epsilon = float(elements[3])
        alpha = float(elements[4])
        sigma = (alpha/abs(epsilon))**(1/12)
        df_contact_parameters.loc[len(df_contact_parameters.index)] = [atom_type1, atom_type2, alpha, epsilon, sigma]

df_contact_parameters.to_csv('MOFF_contact_parameters.csv', index=False)




