import numpy as np
import pandas as pd

"""
Convert LAMMPS input data file to VMD readable file.
"""

with open('data.prot_dna_ions', 'r') as f:
    lines = f.readlines()

n_lines = len(lines)
for i in range(n_lines):
    if 'Atoms' in lines[i]:
        start_line_index = i + 1
    elif 'Bonds' in lines[i]:
        end_line_index = i - 1
        break

for i in range(start_line_index, end_line_index + 1):
    old_line = lines[i]
    elements = old_line.split()
    if len(elements) > 1:
        elements = elements[:2] + elements[3:]
        for j in range(len(elements)):
            if j <= 2:
                elements[j] = f'{int(elements[j]):>7}'
            else:
                elements[j] = f'{float(elements[j]):>10}'
    new_line = ' '.join(elements) + '\n'
    lines[i] = new_line

# manually insert an empty line before 'Bonds'
lines.insert(end_line_index + 1, '\n')
with open('data.prot_dna_ions.4vmd', 'w') as f:
    for each_line in lines:
        f.write(each_line)


