# Benchmark datasets for predicting the orbits of emission-line binaries

Last updated: 21.07.2021<br>
Version: 0.1<br>
Python version: 3.8<br>

## Contents
[Overview](#overview)<br>
[Quick start](#quick-start)<br>
[The datasets](#the-datasets)<br>
[Publications](#publications)<br>

## Overview
This repository contains three synthetic benchmark datasets for testing
algorithms for inferring the orbits of emission-line binaries. Each dataset 
is formulated with a different noise specification -- white noise only, 
polynomial systematics, and GP systematics -- to challenge models to 
make accurate and reliable predictions, especially in the presence of 
difficult systematics. This work originates from Grant and Blundell 2021: see 
this publication for full details and further context.

## Quick start
To immediately get going with benchmarking your models follow these steps. 
You will need python and pip to follow this guide. First clone this repository
```
git clone https://github.com/DavoGrant/benchmark-datasets-emission-line-binaries.git
```
Next, install the python modules required.
```
cd benchmark-datasets-emission-line-binaries
pip install -r requirements.txt
```
Now you can immediately view the benchmark datasets with the supplied script 
example_data_loader.py. This script will loop over the SB1 data per synthetic 
object, including the injected true orbital parameters.
```
python example_data_loader.py
```
You can switch between datasets (gp systematics is the default) using the DATASET 
variable at the top of the script. 

Using this script as a template, simply apply your models to the synthetic data 
of your choosing, or all of the data, and check that you are recovering the true 
orbital parameters. In this way, you can ensure your models are going to make 
accurate and reliable parameter inferences before going on to run them on real data.

Additionally, if you would like to compare your scores to others who have made use 
of these benchmark datasets, then you will want to score your model's results 
quantitatively. To help with these efforts we have provided an example function in
the script example_scoring.py. This computes the log posterior density at the truth
per synthetic object per orbital parameter. If you can beat the best known scores with 
your models, for the three benchmark datasets, then be sure to report them and help the
field keep advancing towards better and better inference techniques.

## The datasets
In this section we describe the data structures, how each dataset was generated, and 
how to load the datasets. The datasets are all SB1 (single-lined spectroscopic binary) 
data.

Each dataset is a pandas.Dataframe object. Each row (pd.Series) holds the data for each 
of 243 synthetic objects per dataset. The data types of a given row are:

| Key | Data type | Description |
| ------------- | ------------- | ------------- |
| Index  | int  |  Synthetic object identifier
| P  | float  |  Period (days)  |
| T0  | float  | Time of periastron (jd)  |
| e  | float  |  Eccentricity  |
| w  | float  |  Argument of periastron (radians)  |
| k1  | float  |  RV Semi-amplitude (km/s)  |
| g1  | float  |  RV offset (km/s)  |
| sigma_w  | float  |  White noise (km/s)  |
| jd  | numpy.ndarray  |  Observation epochs (jd)  |
| energy  | numpy.ndarray  |  n/a  |
| rvs_orbit_primary  | numpy.ndarray  |  Orbital motion (km/s)  |
| rvs_systematics_primary  | numpy.ndarray  |  Systematics (km/s)  |
| rvs_total_primary  | numpy.ndarray  |  Orbit + systematics (km/s)  |
| rvs_observed_primary  | numpy.ndarray  |  Radial velocities (km/s)  |

The orbital parameters required to fully specify SB1 data are the time of periastron, 
orbital period, eccentricity, argument of periastron, primary star's semi-amplitude, 
and primary star's radial velocity offset. We generate 100 observations from one 
orbital period for every system. The observations are drawn from a beta distribution with 
shape parameters alpha = beta = 1 - e, with values corresponding to the orbital phase, so 
there is always sufficient orbital coverage to infer all six of the orbital parameters.

The noise added to each dataset is designed to challenge potential models to be robust 
against the various problems that may be embedded in the radial velocities extracted 
from emission-line stars:
1. Benchmark dataset 1 (/synthetic_Gaussian_white_noise) we add only Gaussian white noise 
to the velocities.
1. Benchmark dataset 2 (/synthetic_polynomial_systematics) we add Gaussian white noise and 
correlated noise to the velocities by injecting randomly generated polynomials.
1. Benchmark dataset 3 (/synthetic_gp_systemtics) we add Gaussian white noise and 
correlated noise to the velocities by injecting functions randomly drawn from a 
Gaussian process prior having a squared exponential kernel.

The algorithms to generate the datasets can be found in the directory /generator. Here, 
you can see the parameter distributions sampled from for the both the orbital parameters 
and the systematic formulations, eg. the GP prior in dataset 3.

The three benchmark datasets are each supplied as both python pickles and ascii 
csv files. The easiest method to load a dataset is from a pickle using a code snippet 
such as
```python
pickle_path = 'path/to/dataset.p'
with open(pickle_path, 'rb') as f:
    benchmark_dataset = pickle.load(f)
```
This method ensures the data formats remain as intended. However, if you prefer you 
can load dataset from the ascii file using
```python
def np_array_from_string(array_string):
    array_string = ','.join(array_string.replace('[', '').replace(']', '').split())
    return np.array(ast.literal_eval(array_string))

csv_path = os.path.join(REPO_PATH, DATASET)
benchmark_dataset = pd.read_csv(
    csv_path, header='infer',
    converters={'jd': np_array_from_string,
                'energy': np_array_from_string,
                'rvs_orbit_primary': np_array_from_string,
                'rvs_systematics_primary': np_array_from_string,
                'rvs_total_primary': np_array_from_string,
                'rvs_observed_primary': np_array_from_string})
```
You can refer to the script example_data_loader.py, mentioned in the quick start 
documentation, for a minimal example if you need more help getting going.

## Publications
The associated paper is available at:
https://arxiv.org/abs/2110.10537

