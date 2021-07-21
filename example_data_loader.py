import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


REPO_PATH = ''
DATASET = 'synthetic_gp_systemtics/sb1_dataset_gp_systematics.p'

# Load data.
pickle_path = os.path.join(REPO_PATH, DATASET)
with open(pickle_path, 'rb') as f:
    sb1_dataset = pickle.load(f)

# Data columns.
print(sb1_dataset.columns)

# Iterate binaries.
for idx, binary in sb1_dataset.iterrows():

    # Draw.
    plt.plot(binary['jd'], binary['rvs_orbit_primary'],
             c='#000000', lw=1, ls='--', label='Orbital motion')
    plt.scatter(binary['jd'], binary['rvs_observed_primary'],
                c='#bc5090', s=10, label='Observed motion')
    plt.xlabel('JD')
    plt.ylabel('Velocity / $\\rm{km\,s^{-1}}$')
    plt.title('$e={}$, $\omega={}$, $k_1={}$'.format(
        round(binary['e'], 2), round(binary['w'] * 180 / np.pi, 2),
        round(binary['k1'], 2)))
    plt.tight_layout()
    plt.show()
