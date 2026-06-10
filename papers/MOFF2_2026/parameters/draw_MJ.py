import numpy as np
import pandas as pd
from openabc.lib import _amino_acids, _amino_acid_3_letters_to_1_letter_dict
import sys
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
mpl.rcParams['pdf.fonttype'] = 42
plt.rcParams.update({'font.size': 12})

"""
Draw the MJ potential parameters.
"""

os.makedirs('pictures', exist_ok=True)

raw_MJ = pd.read_csv('raw_MJ.csv')
MJ_matrix = np.zeros((20, 20))
for _, row in raw_MJ.iterrows():
    i = _amino_acids.index(row['amino acid1'])
    j = _amino_acids.index(row['amino acid2'])
    MJ_matrix[i, j] = row['epsilon (RT)']
    MJ_matrix[j, i] = row['epsilon (RT)']
MJ_matrix *= -1.0 # convert to larger positive values indicating stronger attraction
MJ_matrix = (MJ_matrix - np.min(MJ_matrix)) / (np.max(MJ_matrix) - np.min(MJ_matrix))
MJ_min = 0.0
MJ_max = 1.0
MJ_matrix = MJ_matrix * (MJ_max - MJ_min) + MJ_min
aa_labels = [_amino_acid_3_letters_to_1_letter_dict[i] for i in _amino_acids]
plt.imshow(MJ_matrix, cmap='coolwarm', origin='upper')
plt.colorbar()
xticks = np.arange(20)
yticks = np.arange(20)
plt.xticks(xticks, labels=aa_labels)
plt.yticks(yticks, labels=aa_labels)
plt.title(f'Scaled and shifted MJ \nLarger value means attraction')
plt.tight_layout()
plt.savefig(f'pictures/MJ_min_{MJ_min}_max_{MJ_max}.pdf')
plt.close()


