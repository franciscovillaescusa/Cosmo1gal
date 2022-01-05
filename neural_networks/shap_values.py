import numpy as np
import sys, os, time
sys.path.append('/mnt/ceph/users/camels/Software/1galaxy_cosmo/neural_networks')
import torch
import torch.nn as nn
import data, architecture
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

def test_model(input_size, output_size, n_layers, hidden, dr, device, g, h, fmodel,
               fout):

    # generate the architecture
    model = architecture.dynamic_model2(input_size, output_size, n_layers, hidden, dr)
    model.to(device)  

    # load best-model, if it exists
    print('Loading model...')
    if os.path.exists(fmodel):  
        model.load_state_dict(torch.load(fmodel, map_location=torch.device(device)))
    else:  
        raise Exception('model doesnt exists!!!')

    # define the matrix containing the true & predicted value of the parameters + errors
    params      = len(g)
    results     = np.zeros((test_points, 3*params), dtype=np.float32)
    shap_results = np.zeros((test_points, 2*len(features)+1), dtype=np.float32)
    shap_values = np.zeros((test_points, len(features)), dtype=np.float64)
    input_values = np.zeros((test_points, len(features)), dtype=np.float32)
    input_values = torch.tensor(input_values)
    points = 0
    for x,y in test_loader:
        bs = x.shape[0]         #batch size
        input_values[points:points+bs] = x
        points += bs

    input_values = input_values.to(device)
    indexes = np.arange(input_values.shape[0])
    indexes = np.random.permutation(indexes)
    indexes = indexes[:2000]
    explainer = shap.DeepExplainer(model, input_values[indexes])

    # test the model
    test_loss1, test_loss = torch.zeros(len(g)).to(device), 0.0
    test_loss2, points    = torch.zeros(len(g)).to(device), 0
    model.eval()
    for x, y in test_loader:
        print(points)
        #with torch.no_grad():
        bs    = x.shape[0]         #batch size
        x     = x.to(device)       #maps
        y     = y.to(device)[:,g]  #parameters
        p     = model(x)           #NN output
        y_NN  = p[:,g]             #posterior mean
        e_NN  = p[:,h]             #posterior std
        loss1 = torch.mean((y_NN - y)**2,                axis=0)
        loss2 = torch.mean(((y_NN - y)**2 - e_NN**2)**2, axis=0)
        loss  = torch.mean(torch.log(loss1) + torch.log(loss2))
        test_loss1 += loss1*bs
        test_loss2 += loss2*bs
        #results[points:points+bs,0*params:1*params] = y.cpu().numpy()
        #results[points:points+bs,1*params:2*params] = y_NN.cpu().numpy()
        #results[points:points+bs,2*params:3*params] = e_NN.cpu().numpy()
        shap_values[points:points+bs] = explainer.shap_values(x)[0]
        shap_results[points:points+bs, 0] = y[:,0].cpu().numpy()
        shap_results[points:points+bs, 1:1+len(features)] = x.cpu().numpy()
        shap_results[points:points+bs, 1+len(features):1+2*len(features)] = shap_values[points:points+bs]
        points     += bs
         
    np.savetxt('shap_values_SIMBA.txt', shap_results)
       
    test_loss = torch.log(test_loss1/points) + torch.log(test_loss2/points)
    test_loss = torch.mean(test_loss).item()
    print('Test loss:', test_loss)

    # Set the index of the specific example to explain
    print(shap_values)
    print(shap_values.shape)

    plot = shap.summary_plot(shap_values, input_values.cpu().numpy(), 
                             feature_names=names)
    savefig('shap_values_SIMBA.pdf', bbox_inches='tight')


    # denormalize results here
    minimum = np.array([0.1, 0.6, 0.25, 0.25, 0.5, 0.5])[g]
    maximum = np.array([0.5, 1.0, 4.00, 4.00, 2.0, 2.0])[g]
    results[:,0*params:1*params] = results[:,0*params:1*params]*(maximum-minimum)+minimum
    results[:,1*params:2*params] = results[:,1*params:2*params]*(maximum-minimum)+minimum
    results[:,2*params:3*params] = results[:,2*params:3*params]*(maximum-minimum)

    # save results to file
    #np.savetxt(fout, results)


##################################### INPUT ###########################################
root        = '/mnt/ceph/users/camels/Software/1galaxy_cosmo/data'
sim         = 'SIMBA' #'IllustrisTNG'
f_prop      = '%s/galaxies_%s_z=0.00.txt'%(root,sim)
f_prop_norm = None
f_off       = '%s/offset_%s_z=0.00.txt'%(root,sim)
f_params    = '%s/latin_hypercube_params_%s.txt'%(root,sim)  
seed        = 1   
mode        = 'test'  #'train','valid','test' or 'all'

prefix      = 'all_Om_s8_A1_A2_A3_A4'
features    = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]

# optuna parameters
study_name = 'gal_2_params'
storage = 'sqlite:///../databases/TPE_%s_%s.db'%(sim,prefix)

# architecture parameters
input_size  = len(features) #number of bins in Pk
output_size = 12 #number of parameters to predict (posterior mean + std)

# training parameters
batch_size = 512
g          = [0,1,2,3,4,5]
h          = [6,7,8,9,10,11]
#######################################################################################

# 0-Mgas 1-Mstar 2-Mbh 3-Mtot 4-Vmax 5-Vdisp 6-Zg
# 7-Zs 8-SFR 9-J 10-V 11-Rstar 12-Rtot 13-Rvmax 14-U 15-K 16-g

names = [r'$M_{\rm g}$', r'$M_*$', r'$M_{\rm bh}$', r'$M_{\rm t}$', r'$V_{\rm max}$', 
         r'$\sigma_v$', r'$Z_{\rm g}$', r'$Z_*$', r'${\rm SFR}$', r'$J$', r'${\rm V}$',
         r'$R_*$', r'$R_{\rm t}$', r'$R_{\rm max}$', r'${\rm U}$', r'${\rm K}$',
         r'${\rm g}$']

# use GPUs if available
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')

# get the data
test_loader = data.create_dataset(mode, seed, f_prop, f_off, f_prop_norm, f_params, 
                                  features, batch_size, shuffle=False)
test_points = 0
for x,y in test_loader:  test_points += x.shape[0]

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
for best_trial in [0]:  #choose the best-model here, e.g. [0], or [1]
    trial = study.trials[indexes[best_trial]]
    print("\nTrial number {}".format(trial.number))
    print("Value: %.5e"%trial.value)
    print(" Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    n_layers = trial.params['n_layers']
    lr       = trial.params['lr']
    wd       = trial.params['wd']
    hidden   = np.zeros(n_layers, dtype=np.int32)
    dr       = np.zeros(n_layers, dtype=np.float32)
    for i in range(n_layers):
        hidden[i] = trial.params['n_units_l%d'%i]
        dr[i]     = trial.params['dropout_l%d'%i]

    fmodel = '../models/models_%s_%s/model_%d.pt'%(sim,prefix,trial.number)
    fout   = 'Results_%s_%s_%d.txt'%(sim,prefix,best_trial)
    test_model(input_size, output_size, n_layers, hidden, dr, device, g, h, fmodel, fout)


sys.exit()
















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





