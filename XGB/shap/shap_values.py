import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
import sys, os
import shap, optuna

from pylab import *
import numpy as np
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse
rcParams["mathtext.fontset"]='cm'

# get the best-values from optuna
def get_best_values(study_name, storage):

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
        print('')
        learning_rate    = trial.params['learning_rate']
        max_depth        = trial.params['max_depth']
        min_child_weight = trial.params['min_child_weight']
        gamma            = trial.params['gamma']
        colsample_bytree = trial.params['colsample_bytree']
        n_estimators     = trial.params['n_estimators']


    return learning_rate, max_depth, min_child_weight, gamma, colsample_bytree, n_estimators


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
root     = '/mnt/ceph/users/camels/Software/1galaxy_cosmo/data'
sim      = 'IllustrisTNG'
f_input  = '%s/galaxies_%s_z=0.00.txt'%(root,sim)
f_output = '%s/latin_hypercube_params_%s.txt'%(root,sim)
f_off    = '%s/offset_%s_z=0.00.txt'%(root,sim)
seed     = 1

storage = 'sqlite:///../IllustrisTNG/databases/TPE_IllustrisTNG_all.db'

# optuna parameters
study_name = 'properties_2_Omegam'

# properties parameters
elements = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

"""
learning_rate = 0.1
max_depth = 15
min_child_weight = 1
gamma = 0.1
colsample_bytree = 0.9
n_estimators = 200
"""
#######################################################################################

# 0-Mgas 1-Mstar 2-Mbh 3-Mtot 4-Vmax 5-Vdisp 6-Zg
# 7-Zs 8-SFR 9-J 10-V 11-Rstar 12-Rtot 13-Rvmax 14-U 15-K 16-g

names = [r'$M_{\rm g}$', r'$M_*$', r'$M_{\rm bh}$', r'$M_{\rm t}$', r'$V_{\rm max}$', 
         r'$\sigma_v$', r'$Z_{\rm g}$', r'$Z_*$', r'${\rm SFR}$', r'$J$', r'${\rm V}$',
         r'$R_*$', r'$R_{\rm t}$', r'$R_{\rm max}$', r'${\rm U}$', r'${\rm K}$',
         r'${\rm g}$']

# get best hyperparameter values from optuna
learning_rate, max_depth, min_child_weight, gamma, colsample_bytree, n_estimators = \
             get_best_values(study_name, storage)                   

# get datasets
input_train, output_train = get_dataset(f_input, f_off, f_output, seed, 'train')
input_valid, output_valid = get_dataset(f_input, f_off, f_output, seed, 'valid')
input_test,  output_test  = get_dataset(f_input, f_off, f_output, seed, 'test')

# only take the considered galaxy properties
input_train = input_train[:, elements]
input_valid = input_valid[:, elements]
input_test  = input_test[:, elements]

XGBR = xgb.XGBRegressor(n_estimators=n_estimators,
                        objective='reg:squarederror',
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                        min_child_weight=min_child_weight,
                        gamma=gamma, n_jobs=-1,
                        colsample_bytree=colsample_bytree)

XGBR.fit(input_train, output_train)

print('fit done!')
   
# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
#explainer = shap.Explainer(XGBR)
#shap_values = explainer(input_test)

explainer = shap.explainers.Exact(XGBR.predict, input_test)
shap_values = explainer(input_test, max_evals=200000)
print(shap_values)
print(shap_values.shape)
print(shap_values.values)
print(np.mean(np.absolute(shap_values.values), axis=0))
print(np.mean(np.absolute(shap_values.values), axis=1))
print(np.min(shap_values.base_values))
print(np.max(shap_values.base_values))

np.savetxt('borrar.txt', shap_values.values)

# visualize the first prediction's explanation
#shap.plots.waterfall(shap_values[0])
plot = shap.summary_plot(shap_values.values, input_test, feature_names=names)
savefig('borrar.pdf', bbox_inches='tight')

#indexes = np.where(input_test[:,3]>1e11)[0]
#plot = shap.summary_plot(shap_values.values[indexes], input_test[indexes], feature_names=names)
#savefig('borrar2.pdf', bbox_inches='tight') 

#shap.plots.bar(shap_values, feature_names=names)
#savefig('borrar3.pdf', bbox_inches='tight')

print('making the prediction')
pred_valid = XGBR.predict(input_valid)
pred_test = XGBR.predict(input_test)

print('calculating the loss')
valid_loss = (pred_valid - output_valid)**2
valid_mse = np.sqrt(np.mean(valid_loss))

test_loss = (pred_test - output_test)**2
test_mse = np.sqrt(np.mean(test_loss))

print(valid_mse, test_mse)

fout = 'borrar2.txt'
np.savetxt(fout, pred_test)




