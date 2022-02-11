# Prerequisites

The packages needed to run the codes are:
- python3
- numpy
- [Pytorch](https://pytorch.org)
- [Optuna](https://optuna.org)
- [XGB](https://xgboost.readthedocs.io/en/stable/)
- [SHAP](https://shap.readthedocs.io/en/latest/) (only to compute SHAP values)
- [Pylians3](https://pylians3.readthedocs.io/en/master/) (only for Fisher analysis)

If the user wants to add more galaxy properties to the ones considered here he/she will need to download the [CAMELS Subfind catalogues](https://camels.readthedocs.io/en/latest/subfind.html).

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
- `Omega_b.py`. This script is similar to `preprocess.py` and is used to generate the galaxy and offset files for the simulations with different values of Omega_b. This script will generate the file:
	-  `galaxies_Omega_b.txt` that contains the galaxy properties for the simulations with different values of Omega_b.
- `latin_hypercube_params_X.txt`. This file contains the value of the cosmological and astrophysical parameters for each simulation. X can be  `IllustrisTNG` or `SIMBA`.

We note that the `galaxies_*` files are too heavy to be stored in GitHub. We provide access to them through:
-	[Globus](https://app.globus.org/file-manager?origin_id=fb43264c-8b52-11ec-900b-3132d806a822&origin_path=%2F)
-	[Url](https://users.flatironinstitute.org/~fvillaescusa/priv/WrswOUVzX5cGFgbhh7Z7hUCJyrStY/PUBLIC)

## neural networks

This folder contains the codes, databases, and weights of the neural networks. There are different files:

- `architecture.py`. This script contains the different neural network architectures.
- `data.py`. This script reads the data and prepare it to train the networks.
- `main_LFI.py`. This script is used to train the networks.
- `test_LFI.py`. This script is used to test the networks.
- `train_with_feedback_LFI.py`. This script is used to train models where the value of the astrophysical parameters are known.
- `analyze_databases.py`. This script will read the different databases and print some information about their best trials.
- `analyze_results.py`. This script is used to analyze the results after training the networks.
- `shap_values.py`. This script is used to compute the shape values.

There are also different folders:
- `databases`. This folder contains the databases.
- `losses`. This folder contains the losses of the different models.
- `models`. This folder contains the network weights for the different models.
- `Results.txt`. This folder contains the results of testing the models.
- `shap`. This folder contains the SHAP values of the model.

Unfortunately, the folders are too heavy to be hosted in GitHub. We however provide access to them through:
-	[Globus](https://app.globus.org/file-manager?origin_id=fb43264c-8b52-11ec-900b-3132d806a822&origin_path=%2F)
-	[Url](https://users.flatironinstitute.org/~fvillaescusa/priv/WrswOUVzX5cGFgbhh7Z7hUCJyrStY/PUBLIC)


## XGB

This folder contains the scripts, databases, and results of performing the analysis using gradient boosting trees.


## other

This folder contains the codes used to carry out the Fisher matrix calculation.
