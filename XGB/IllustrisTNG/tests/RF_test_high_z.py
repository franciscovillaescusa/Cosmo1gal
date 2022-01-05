# This code carries out hyperparameter optimization using gradient boosted trees
# USAGE: python3 RF.py -discard 5 6  # will remove element 5 and train; then element 6 and train
# USAGE: python3 RF.py -discard 1 -discarded 5 6 
# elements 5 & 6 are already discarded. Will remove element 1 and train
import numpy as np
import optuna
import xgboost as xgb
from xgboost import XGBRegressor
import sys, os


# optuna trials
class Objective(object):
    def __init__(self, input_train, output_train, input_valid, output_valid,
                 input_test, output_test, n_jobs):

        self.input_train  = input_train
        self.output_train = output_train
        self.input_valid  = input_valid
        self.output_valid = output_valid
        self.input_test   = input_test
        self.output_test  = output_test
        self.n_jobs       = n_jobs


    def __call__(self, trial):

        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.6)
        max_depth = trial.suggest_int('max_depth', 4, 20)
        min_child_weight = trial.suggest_float('min_child_weight', 0.5, 10)
        gamma = trial.suggest_float('gamma', 0.05, 0.4, log=True)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1)
        n_estimators = trial.suggest_int('n_estimators', 10, 200)
        
        # define and fit the model
        XGBR = xgb.XGBRegressor(n_estimators=n_estimators,
                                objective='reg:squarederror',
                                learning_rate=learning_rate,
                                max_depth=max_depth,
                                min_child_weight=min_child_weight,
                                gamma=gamma, n_jobs=self.n_jobs,
                                colsample_bytree=colsample_bytree)
        fit = XGBR.fit(self.input_train, self.output_train)

        # predict validation and test scores
        pred_valid = XGBR.predict(self.input_valid)
        pred_test  = XGBR.predict(self.input_test)
        
        print('calculating the loss')
        valid_mse = np.sqrt(np.mean((pred_valid - self.output_valid)**2))
        test_mse  = np.sqrt(np.mean((pred_test  - self.output_test)**2))

        # save results to file
        f = open(fout, 'a')
        f.write('%d %.5e %.5e \n' % (trial.number, valid_mse, test_mse))
        f.close()

        return valid_mse


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
root     = '/mnt/ceph/users/camels/Software/1galaxy_cosmo/data_preprocessing'
sim      = 'IllustrisTNG'
f_input  = '%s/galaxies_%s_z=2.00.txt'%(root,sim)
f_output = '%s/latin_hypercube_params_%s.txt'%(root,sim)
f_off    = '%s/offset_%s_z=2.00.txt'%(root,sim)
seed     = 1
n_jobs   = -1

# optuna parameters
study_name     = 'properties_2_Omegam'
n_trials       = 100  # set to None for infinite
startup_trials = 25

# elements to consider
elements = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
suffix = 'all_z=2'
#######################################################################################

# 0-Mgas 1-Mstar 2-Mbh 3-Mtot 4-Vmax 5-Vdisp 6-Zg
# 7-Zs 8-SFR 9-J 10-V 11-Rstar 12-Rtot 13-Rvmax 14-U 15-K 16-g

# get datasets
input_train, output_train = get_dataset(f_input, f_off, f_output, seed, 'train')
input_valid, output_valid = get_dataset(f_input, f_off, f_output, seed, 'valid')
input_test,  output_test  = get_dataset(f_input, f_off, f_output, seed, 'test')

# only take the considered galaxy properties
input_train = input_train[:, elements]
input_valid = input_valid[:, elements]
input_test  = input_test[:, elements]

# get the name of the files
storage = 'sqlite:///TPE_%s_%s.db' % (sim, suffix)
fout = 'loss_XGB_Om_%s_%s.txt' % (sim, suffix)

f = open(fout, 'a');  f.write('# elements: %s\n'%elements);  f.close()    

print('checking the size of samples')
print(input_train.shape, output_train.shape)
print(input_valid.shape, output_valid.shape)
print(input_test.shape,  output_test.shape)

objective = Objective(input_train, output_train, input_valid, output_valid,
                      input_test, output_test, n_jobs)
sampler = optuna.samplers.TPESampler(n_startup_trials=startup_trials)
study = optuna.create_study(study_name=study_name, storage=storage, sampler=sampler)
study.optimize(objective, n_trials)

print('Minimum mse: %.3f', study.best_value)
print('Best parameter: ', str(study.best_params))

trial = study.best_trial
print("Best trial: number {}".format(trial.number))
print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
