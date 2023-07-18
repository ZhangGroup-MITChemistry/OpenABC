import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
plt.rcParams.update({'font.size': 12})

if not os.path.exists('pictures'):
    os.makedirs('pictures')

input_csv = sys.argv[1]
output_plot = sys.argv[2]

df = pd.read_csv(input_csv)
z = df['z (nm)']
rho = df['rho (g/L)'] # g/L is equivalent to mg/mL
plt.plot(z, rho)
plt.axvline(-10, color='grey', linestyle='--')
plt.axvline(10, color='grey', linestyle='--')
plt.axvline(-50, color='grey', linestyle='--')
plt.axvline(50, color='grey', linestyle='--')
plt.xlabel('z (nm)')
plt.ylabel('Density (mg/mL)')
plt.title(r'HP1$\alpha$ dimer')
plt.savefig(output_plot)
plt.close()


