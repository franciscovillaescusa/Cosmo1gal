# This code carries out hyperparameter optimization using gradient boosted trees
# USAGE: python3 RF.py -discard 5 6  # will remove element 5 and train; then element 6 and train
# USAGE: python3 RF.py -discard 1 -discarded 5 6 
# elements 5 & 6 are already discarded. Will remove element 1 and train
import numpy as np
import optuna
import xgboost as xgb
from xgboost import XGBRegressor
import sys, os, argparse

"""
# read the elements to be removed and the ones already discarded
parser = argparse.ArgumentParser(description="description of routine")
parser.add_argument("-discard", nargs='+', type=int, default=[],
                    help="number of the element to delete")
parser.add_argument("-discarded", nargs='+', type=int, default=[],
                    help="number of the discarded elements")
args      = parser.parse_args()
discard   = args.discard
discarded = args.discarded
"""


# get the dataset
def get_dataset(f_input, f_off, f_output, seed, mode):

    # load the data
    data = np.loadtxt(f_input)
    params = np.loadtxt(f_output)
    off, length = np.loadtxt(f_off, dtype=np.int64, unpack=True)

    # get the size and offset depending on the type of dataset
    sims = params.shape[0]
    if mode == 'train':
        size, offset = int(sims*0.85), int(sims*0.00)
    elif mode == 'valid':
        size, offset = int(sims*0.10), int(sims*0.85)
    elif mode == 'test':
        size, offset = int(sims*0.05), int(sims*0.95)
    elif mode == 'all':
        size, offset = int(sims*1.00), int(sims*0.00)
    else:
        raise Exception('Wrong name!')

    # randomly shuffle the sims. Instead of 0 1 2 3...999 have a
    # random permutation. E.g. 5 9 0 29...342
    np.random.seed(seed)
    indexes = np.arange(sims)  # shuffle the order of the simulations
    np.random.shuffle(indexes)
    indexes = indexes[offset:offset+size]  # select indexes of mode

    # get the indexes of the galaxies in the considered set
    Ngal = 0
    for i in indexes:
        Ngal += length[i]
    print('Number of galaxies in the %s set: %d' % (mode, Ngal))

    # define the arrays containing the properties and the parameter values
    prop = np.zeros((Ngal, data.shape[1]),   dtype=np.float32)
    pars = np.zeros((Ngal, params.shape[1]), dtype=np.float32)

    # fill the arrays
    count = 0
    for i in indexes:
        for j in range(length[i]):
            prop[count] = data[off[i] + j]
            pars[count] = params[i]
            count += 1
    print('processed %d galaxies' % count)

    return prop, pars[:, 0]


##################################### INPUT ###########################################
# file parameters
root     = '/mnt/ceph/users/camels/Software/1galaxy_cosmo'
sim      = 'IllustrisTNG'
f_input  = '%s/galaxies_%s_UKg_z=0.00.txt'%(root,sim)
f_output = '%s/latin_hypercube_params_%s.txt'%(root,sim)
f_off    = '%s/offset_%s_UKg_z=0.00.txt'%(root,sim)
sim2     = 'SIMBA'
f_input2 = '%s/galaxies_%s_UKg_z=0.00.txt'%(root,sim2)
f_output2 = '%s/latin_hypercube_params_%s.txt'%(root,sim2)
f_off2   = '%s/offset_%s_UKg_z=0.00.txt'%(root,sim2)
seed     = 1
n_jobs   = -1

# properties parameters
elements = [1,4,7]

# optuna parameters
study_name     = 'properties_2_Omegam'
n_trials       = 100  # set to None for infinite
startup_trials = 25
#######################################################################################

# 0-Mgas 1-Mstar 2-Mbh 3-Mtot 4-Vmax 5-Vdisp 6-Zg
# 7-Zs 8-SFR 9-J 10-V 11-Rstar 12-Rtot 13-Rvmax 14-U 15-K 16-g

learning_rate    = 0.06
max_depth        = 16 
min_child_weight = 7.17
gamma            = 0.05
colsample_bytree = 0.84
n_estimators     = 200

# get datasets
input_train, output_train = get_dataset(f_input, f_off, f_output, seed, 'train')
input_valid, output_valid = get_dataset(f_input, f_off, f_output, seed, 'valid')
input_test,  output_test  = get_dataset(f_input, f_off, f_output, seed, 'test')

input_test2,  output_test2 = get_dataset(f_input2, f_off2, f_output2, seed, 'test')

# only take the considered galaxy properties
input_train = input_train[:, elements]
input_valid = input_valid[:, elements]
input_test  = input_test[:, elements]
input_test2 = input_test2[:,elements]

# define and fit the model
XGBR = xgb.XGBRegressor(n_estimators=n_estimators,
                        objective='reg:squarederror',
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                        min_child_weight=min_child_weight,
                        gamma=gamma, n_jobs=n_jobs,
                        colsample_bytree=colsample_bytree)
fit = XGBR.fit(input_train, output_train)

# predict validation and test scores
pred_valid = XGBR.predict(input_valid)
pred_test  = XGBR.predict(input_test)
pred_test2 = XGBR.predict(input_test2)
        
print('calculating the loss')
valid_mse = np.sqrt(np.mean((pred_valid - output_valid)**2))
test_mse  = np.sqrt(np.mean((pred_test  - output_test)**2))
test2_mse = np.sqrt(np.mean((pred_test2  - output_test2)**2))

print(valid_mse,test_mse,test2_mse)

unique = np.unique(output_test)
mean = np.zeros(unique.shape[0], dtype=np.float32)
std  = np.zeros(unique.shape[0], dtype=np.float32)
for i,u in enumerate(unique):
    indexes = np.where(output_test==u)[0]
    values = pred_test[indexes]
    mean[i] = np.mean(values)
    std[i]  = np.std(values)
np.savetxt('borrar.txt', np.transpose([unique,mean,std]))

unique = np.unique(output_test2)
mean = np.zeros(unique.shape[0], dtype=np.float32)
std  = np.zeros(unique.shape[0], dtype=np.float32)
for i,u in enumerate(unique):
    indexes = np.where(output_test2==u)[0]
    values = pred_test2[indexes]
    mean[i] = np.mean(values)
    std[i]  = np.std(values)
np.savetxt('borrar2.txt', np.transpose([unique,mean,std]))
