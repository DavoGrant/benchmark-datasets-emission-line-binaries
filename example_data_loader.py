import os
import ast
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


REPO_PATH = ''
DATASET = 'synthetic_gp_systematics/sb1_dataset_gp_systematics.p'

# Load data.
if os.path.splitext(DATASET)[1] == '.p':

    # Dataset is a python pickle.
    pickle_path = os.path.join(REPO_PATH, DATASET)
    with open(pickle_path, 'rb') as f:
        sb1_dataset = pickle.load(f)

elif os.path.splitext(DATASET)[1] == '.csv':

    # Dataset is an ascii csv file.
    def np_array_from_string(array_string):
        array_string = ','.join(array_string.replace('[', '').replace(']', '').split())
        return np.array(ast.literal_eval(array_string))

    csv_path = os.path.join(REPO_PATH, DATASET)
    sb1_dataset = pd.read_csv(
        csv_path, header='infer',
        converters={'jd': np_array_from_string,
                    'energy': np_array_from_string,
                    'rvs_orbit_primary': np_array_from_string,
                    'rvs_systematics_primary': np_array_from_string,
                    'rvs_total_primary': np_array_from_string,
                    'rvs_observed_primary': np_array_from_string})

else:
    raise ValueError('Dataset filetype not recognised.')

# Data columns.
print('Synthetic objects loaded={}'.format(len(sb1_dataset)))
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
