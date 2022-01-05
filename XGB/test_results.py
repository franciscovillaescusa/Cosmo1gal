# This code carries out hyperparameter optimization using gradient boosted trees
# USAGE: python3 RF.py -discard 5 6  # will remove element 5 and train; then element 6 and train
# USAGE: python3 RF.py -discard 1 -discarded 5 6 
# elements 5 & 6 are already discarded. Will remove element 1 and train
import numpy as np
import optuna
import xgboost as xgb
from xgboost import XGBRegressor
import sys, os


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
root      = '/mnt/ceph/users/camels/Software/1galaxy_cosmo/data'
sim       = 'SIMBA'
f_input   = '%s/galaxies_%s_z=0.00.txt'%(root,sim)
f_output  = '%s/latin_hypercube_params_%s.txt'%(root,sim)
f_off     = '%s/offset_%s_z=0.00.txt'%(root,sim)
sim2      = 'IllustrisTNG'
f_input2  = '%s/galaxies_%s_z=0.00.txt'%(root,sim2)
f_output2 = '%s/latin_hypercube_params_%s.txt'%(root,sim2)
f_off2    = '%s/offset_%s_z=0.00.txt'%(root,sim2)
seed      = 1
n_jobs    = -1

# properties parameters
elements = [1,4,7]

# optuna parameters
study_name = 'properties_2_Omegam'
storage    = 'sqlite:///%s/databases/TPE_%s_Mstar_Vmax_Zs.db'%(sim,sim)
#######################################################################################

# 0-Mgas 1-Mstar 2-Mbh 3-Mtot 4-Vmax 5-Vdisp 6-Zg
# 7-Zs 8-SFR 9-J 10-V 11-Rstar 12-Rtot 13-Rvmax 14-U 15-K 16-g

# load the optuna study
study = optuna.load_study(study_name=study_name, storage=storage)

# get the scores of the study trials
values = np.zeros(len(study.trials))
completed = 0
for i,t in enumerate(study.trials):
    values[i] = t.value
    if t.value is not None:  completed += 1

# get the info of the best trial
indexes = np.argsort(values)
for i in [0]:  #choose the best-model here, e.g. [0], or [1]
    trial = study.trials[indexes[i]]
    print("\nTrial number {}".format(trial.number))
    print("Value: %.5e"%trial.value)
    print(" Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    learning_rate    = trial.params['learning_rate']
    max_depth        = trial.params['max_depth']
    min_child_weight = trial.params['min_child_weight']
    gamma            = trial.params['gamma']
    colsample_bytree = trial.params['colsample_bytree']
    n_estimators     = trial.params['n_estimators']

# get datasets
input_train, output_train  = get_dataset(f_input, f_off, f_output, seed, 'train')
input_valid, output_valid  = get_dataset(f_input, f_off, f_output, seed, 'valid')
input_test,  output_test   = get_dataset(f_input, f_off, f_output, seed, 'test')
#input_train[:,[11,12,13]]/=(1.0+2.0)
#input_valid[:,[11,12,13]]/=(1.0+2.0)
#input_test[:,[11,12,13]]/=(1.0+2.0)

# only take the considered galaxy properties
input_train = input_train[:, elements]
input_valid = input_valid[:, elements]
input_test  = input_test[:, elements]

# define and fit the model
XGBR = xgb.XGBRegressor(n_estimators=n_estimators,
                        objective='reg:squarederror',
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                        min_child_weight=min_child_weight,
                        gamma=gamma, n_jobs=n_jobs,
                        colsample_bytree=colsample_bytree)
XGBR.fit(input_train, output_train)

# predict validation and test scores
pred_valid = XGBR.predict(input_valid)
pred_test  = XGBR.predict(input_test)
        
print('calculating the loss')
valid_mse = np.sqrt(np.mean((pred_valid - output_valid)**2))
test_mse  = np.sqrt(np.mean((pred_test  - output_test)**2))
print(valid_mse, test_mse)

unique = np.unique(output_test)
mean = np.zeros(unique.shape[0], dtype=np.float32)
std  = np.zeros(unique.shape[0], dtype=np.float32)
for i,u in enumerate(unique):
    indexes = np.where(output_test==u)[0]
    values = pred_test[indexes]
    mean[i] = np.mean(values)
    std[i]  = np.std(values)
np.savetxt('borrar.txt', np.transpose([unique,mean,std]))



input_test,  output_test = get_dataset(f_input2, f_off2, f_output2, seed, 'test')
input_test = input_test[:,elements]
pred_test = XGBR.predict(input_test)
test_mse = np.sqrt(np.mean((pred_test  - output_test)**2))
unique = np.unique(output_test)
mean = np.zeros(unique.shape[0], dtype=np.float32)
std  = np.zeros(unique.shape[0], dtype=np.float32)
for i,u in enumerate(unique):
    indexes = np.where(output_test==u)[0]
    values = pred_test[indexes]
    mean[i] = np.mean(values)
    std[i]  = np.std(values)
test_mse  = np.sqrt(np.mean((pred_test  - output_test)**2))
print(test_mse)
np.savetxt('borrar2.txt', np.transpose([unique,mean,std]))

