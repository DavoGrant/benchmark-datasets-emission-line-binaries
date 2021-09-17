import pickle
import george
import numpy as np
import pandas as pd
from george import kernels
import matplotlib.pyplot as plt

from generator.keplerian_motion.orbits import StellarDynamics


# Make reproducible.
np.random.seed(111)
N_SYSTEMS = 3**5
N_OBSERVATIONS = 100
DRAW = False

# Data columns.
meta = ['P', 'T0', 'e', 'w', 'k1', 'g1', 'sigma_w']
data = ['jd', 'energy', 'rvs_orbit_primary',
        'rvs_systematics_primary', 'rvs_total_primary',
        'rvs_observed_primary']

# Set up data structure for storage.
sb1_dataset = pd.DataFrame(columns=meta + data)
for a in data:
    sb1_dataset[a] = sb1_dataset[a].astype('object')

# Line transitions.
line_energy = 23.07

# Generate orbital parameters.
p_param = 50.
t0_param = 2458932.
e_params = np.random.uniform(0., 0.95, size=N_SYSTEMS)
w_params = np.random.uniform(0., 360, size=N_SYSTEMS)
k1_params = np.random.uniform(20., 250., size=N_SYSTEMS)
g1_param = 0.

# Systematics.
a_fraction_params = np.random.uniform(0., 0.5, size=N_SYSTEMS)
l1_params = np.random.uniform(0.1, 30., size=N_SYSTEMS)

# White Gaussian noise.
sigma = 5.

# Iterate synthesising systems.
for i in range(N_SYSTEMS):

    # Instantiate.
    sd = StellarDynamics()

    # System params.
    sd.period = p_param
    sd.time_of_periastron = t0_param
    sd.eccentricity = e_params[i]
    sd.argument_of_periastron_primary = w_params[i] * np.pi / 180.

    # Observing epochs.
    epoch_picks = np.sort(np.random.beta(
        a=1 - e_params[i], b=1 - e_params[i], size=N_OBSERVATIONS))
    epochs = np.append(epoch_picks[epoch_picks >= 0.5],
                       epoch_picks[epoch_picks < 0.5] + 1.0)
    sd.jd = ((epochs - 1) * p_param) + t0_param
    sd.auto_calculate_params(phase=True)

    # SB1 params.
    sd.semi_amplitude_primary = k1_params[i]
    sd.rv_offset_primary = g1_param

    # Compute primary radial velocities.
    rvs_pri = sd.keplerian_radial_velocity_primary

    # Add systematics: gp sqr exp.
    A2 = (a_fraction_params[i] * k1_params[i])**2
    l12 = l1_params[i]**2
    k = A2 * kernels.ExpSquaredKernel(l12)
    gp = george.GP(k)
    sys_pri = gp.sample(sd.jd)

    # Add Gaussian noise.
    rvs_observed_pri = rvs_pri + sys_pri + np.random.normal(
        loc=0., scale=sigma, size=N_OBSERVATIONS)

    if DRAW:
        plt.scatter(sd.jd, rvs_pri)
        plt.scatter(sd.jd, rvs_observed_pri)
        plt.tight_layout()
        plt.show()

    # Add meta.
    sb1_dataset.at[i, 'P'] = sd.period
    sb1_dataset.at[i, 'T0'] = sd.time_of_periastron
    sb1_dataset.at[i, 'e'] = sd.eccentricity
    sb1_dataset.at[i, 'w'] = sd.argument_of_periastron_primary
    sb1_dataset.at[i, 'k1'] = sd.semi_amplitude_primary
    sb1_dataset.at[i, 'g1'] = sd.rv_offset_primary
    sb1_dataset.at[i, 'sigma_w'] = sigma

    # Add data.
    sb1_dataset.at[i, 'jd'] = sd.jd
    sb1_dataset.at[i, 'energy'] = np.ones(sd.jd.shape) * line_energy
    sb1_dataset.at[i, 'rvs_orbit_primary'] = rvs_pri
    sb1_dataset.at[i, 'rvs_systematics_primary'] = sys_pri
    sb1_dataset.at[i, 'rvs_total_primary'] = rvs_pri + sys_pri
    sb1_dataset.at[i, 'rvs_observed_primary'] = rvs_observed_pri

with open('sb1_dataset_systematics.p', 'wb') as f:
    pickle.dump(sb1_dataset, f)
