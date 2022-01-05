import numpy as np
import sys, os, time
import torch
import torch.nn as nn
import data, architecture
import optuna
#from captum.attr import IntegratedGradients

def test_model(input_size, output_size, n_layers, hidden, dr, device, g, h, fmodel, fout):

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
    
    # test the model
    test_loss1, test_loss = torch.zeros(len(g)).to(device), 0.0
    test_loss2, points    = torch.zeros(len(g)).to(device), 0
    model.eval()
    for x, y in test_loader:
        with torch.no_grad():
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
            results[points:points+bs,0*params:1*params] = y.cpu().numpy()
            results[points:points+bs,1*params:2*params] = y_NN.cpu().numpy()
            results[points:points+bs,2*params:3*params] = e_NN.cpu().numpy()
            points     += bs
    test_loss = torch.log(test_loss1/points) + torch.log(test_loss2/points)
    test_loss = torch.mean(test_loss).item()
    print('Test loss:', test_loss)

    """
    # interpret model
    importances = np.zeros((test_points, input_size), dtype=np.float64)
    ig = IntegratedGradients(model)
    points = 0
    for x, y in test_loader:
        bs    = x.shape[0]                      #batch size
        x     = x.requires_grad_().to(device)   #maps
        attr, delta = ig.attribute(x, target=0, return_convergence_delta=True)
        attr = attr.detach().numpy()
        importances[points:points+bs] = np.absolute(attr)
        points += bs
    importances = np.mean(importances, axis=0)
    print('Computed integrated gradients of %d galaxies'%points)
    print(importances)
    """

    # denormalize results here
    minimum = np.array([0.1, 0.6, 0.25, 0.25, 0.5, 0.5])[g]
    maximum = np.array([0.5, 1.0, 4.00, 4.00, 2.0, 2.0])[g]
    results[:,0*params:1*params] = results[:,0*params:1*params]*(maximum-minimum)+minimum
    results[:,1*params:2*params] = results[:,1*params:2*params]*(maximum-minimum)+minimum
    results[:,2*params:3*params] = results[:,2*params:3*params]*(maximum-minimum)

    # save results to file
    np.savetxt(fout, results)


################################### INPUT ############################################
# data parameters
root          = '/mnt/ceph/users/camels/Software/1galaxy_cosmo/data'

sim_train     = 'IllustrisTNG'
f_prop_train  = '%s/galaxies_%s_z=0.00.txt'%(root,sim_train)

sim_test      = 'IllustrisTNG'
f_prop_test   = '%s/galaxies_%s_z=0.00.txt'%(root,sim_test)
f_off_test    = '%s/offset_%s_z=0.00.txt'%(root,sim_test)
f_params_test = '%s/latin_hypercube_params_%s.txt'%(root,sim_test)  
seed          = 1   
mode          = 'test'  #'train','valid','test' or 'all'

#prefix      = 'Mtot_Vmax_Vdisp_Zs'
#features    = [3,4,5,7]

#prefix      = 'Mtot_Vmax_Vdisp'
#features    = [3,4,5]

#prefix      = 'Mstar_Mtot_Vmax_Vdisp'
#features    = [1,3,4,5]

#prefix      = 'all'
#features    = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]

prefix      = 'all_UKg'
features    = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

#prefix      = 'all_Om_s8_A1_A2_A3_A4'
#features    = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]#,14,15,16]

#prefix      = 'all-UKg_Om_s8_A1_A2_A3_A4_z=0.0'
#features    = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]

#prefix      = 'all-Vmax-Vdisp-Mtot-Rtot-Rmax_Om_s8_A1_A2_A3_A4_z=0.0'
#features    = [0,1,2,6,7,8,9,10,11,14,15,16] #features used as input

#prefix      = 'all_Om_s8_A1_A2_A3_A4_z=3'
#features    = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]

#prefix      = 'all+A1_A2_A3_A4_z=0'
#features    = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

#prefix      = 'Mstar_Vmax_A1_A2_A3_A4_z=0'
#features    = [1,4]

#prefix      = 'all_Om'
#features    = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]#,14,15,16]

#prefix      = 'Mgas_Vmax_Zgas'
#features    = [0,4,6]

#prefix      = 'Mstar_Mtot_Vmax_Vdisp_Zs'
#features    = [1,3,4,5,7]

#prefix      = 'Mstar_Vdisp_Zs_Rstar_K'
#features    = [1,5,7,11,15]

#prefix      = 'Mstar_Vmax_Zs_Rstar_Rmax'
#features    = [1,4,7,11,13]

#prefix      = 'Mgas_Mstar_Vmax_Zg_K_z=0'
#features    = [0,1,4,6,15]


# architecture parameters
input_size  = len(features) #number of bins in Pk
output_size = 12 #number of parameters to predict (posterior mean + std)

# training parameters
batch_size = 128
g          = [0,1,2,3,4,5]
h          = [6,7,8,9,10,11]

# optuna parameters
study_name = 'gal_2_params'
storage    = 'sqlite:///databases/TPE_%s_%s.db'%(sim_train,prefix)
######################################################################################

# 0-Mgas 1-Mstar 2-Mbh 3-Mtot 4-Vmax 5-Vdisp 6-Zg
# 7-Zs 8-SFR 9-J 10-V 11-Rstar 12-Rtot 13-Rvmax 14-U 15-K 16-g

# use GPUs if available
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')

# get the data
test_loader = data.create_dataset(mode, seed, f_prop_test, f_off_test, f_prop_train, 
                                  f_params_test, features, batch_size, shuffle=False)
#test_loader = data.create_dataset_with_feedback(mode, seed, f_prop, f_off, f_prop_norm, f_params, 
#                                  features, batch_size, shuffle=False)
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
for best_trial in [0,1,2,3]:  #choose the best-model here, e.g. [0], or [1]
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

    fmodel = 'models/models_%s_%s/model_%d.pt'%(sim_train,prefix,trial.number)
    print(fmodel)
    fout   = 'Results/Results_train_%s_%s_%d_test_%s.txt'%(sim_train,prefix,best_trial,sim_test)
    test_model(input_size, output_size, n_layers, hidden, dr, device, g, h, fmodel, fout)
