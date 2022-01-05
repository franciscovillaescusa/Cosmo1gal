# Structure

There are four different folders:

- **data**: this folder contains the codes to generate the data and the data itself.
- **neural_networks**: this folder contains the codes and results obtained from analyzing the data with neural networks.
- **XBG**: this folder contains the codes and results obtained by analyzing the data with gradient boosting trees.
- **other**: this folder contains other codes written for the data analysis.

We now describe in a bit more detail the content of each folder.

## data

This folder contains the following files:

- `preprocess.py`. This script will read the Subfind catalogues and generate the input files for both the neural networks and the gradient boosting trees. The output of this script are two files:
	- `galaxies_X_z=Z.ZZ.txt`. These files contain the galaxy properties for each galaxy in all simulations. X can be `IllustrisTNG` or `SIMBA`, and `Z.ZZ` is the redshift.
	- `offset_X_z=Z.ZZ.txt`. These files contain the offset to identify galaxies belonging to different simulations. This file is used to create the training, validation, and testing sets splitting galaxies across simulations. X can be `IllustrisTNG` or `SIMBA`, and `Z.ZZ` is the redshift.
- `Omega_b.py`. This script is similar to `preprocess.py` and is used to generate the galaxy and offset files for the simulations with different values of $\Omega_{\rm b}$. This script will generate the file:
	-  `galaxies_Omega_b.txt` that contains the galaxy properties for the simulations with different values of $\Omega_{\rm b}$.
- `latin_hypercube_params_X.txt`. This file contains the value of the cosmological and astrophysical parameters for each simulation. X can be  `IllustrisTNG` or `SIMBA`.

## neural networks

## XGB

## other

