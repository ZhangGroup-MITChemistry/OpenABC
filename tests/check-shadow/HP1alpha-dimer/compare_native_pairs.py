import numpy as np
import pandas as pd
import sys
import os

pairs1 = pd.read_csv('native_pairs.csv')
pairs2 = pd.read_csv('smog_native_pairs.csv')
pairs3 = pd.read_csv('legacy_native_pairs.csv')
pairs1.loc[:, 'our shadow'] = 1
pairs2.loc[:, 'smog'] = 1
pairs3.loc[:, 'our legacy shadow'] = 1

merged_pairs = pd.merge(pairs1, pairs2, 'outer', on=['a1', 'a2'])
merged_pairs = pd.merge(merged_pairs, pairs3, 'outer', on=['a1', 'a2'])
merged_pairs = merged_pairs.fillna(0)
merged_pairs.to_csv('merged_native_pairs.csv', index=False)

diff_pairs = merged_pairs[(merged_pairs['our shadow'] == 0) | (merged_pairs['smog'] == 0) | 
                          (merged_pairs['our legacy shadow'] == 0)]
print(diff_pairs)


