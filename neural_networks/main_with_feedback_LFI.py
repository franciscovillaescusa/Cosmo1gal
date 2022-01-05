import numpy as np
import sys, os, time
import torch
import torch.nn as nn
import data, architecture
import optuna


class Objective(object):
    def __init__(self, input_size, output_size, max_layers, max_neurons_layers, device,
                 epochs, seed, batch_size):

        self.input_size         = input_size
        self.output_size        = output_size
        self.max_layers         = max_layers
        self.max_neurons_layers = max_neurons_layers
        self.device             = device
        self.epochs             = epochs
        self.seed               = seed
        self.batch_size         = batch_size

    def __call__(self, trial):

        # name of the files that will contain the losses and model weights
        fout   = 'losses/losses_%s_%s/loss_%d.txt'%(sim, prefix, trial.number)
        fmodel = 'models/models_%s_%s/model_%d.pt'%(sim, prefix, trial.number)

        # generate the architecture
        model = architecture.dynamic_model(trial, self.input_size, self.output_size, 
                            self.max_layers, self.max_neurons_layers).to(self.device)

        # get the weight decay and learning rate values
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        wd = trial.suggest_float("wd", 1e-8, 1e0,  log=True)

        # define the optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.5, 0.999), 
                                      weight_decay=wd)

        # get the data
        train_loader = data.create_dataset_with_feedback('train', self.seed, f_prop, f_off, 
                                           f_prop_norm, f_params, features, 
                                           self.batch_size, shuffle=True, 
                                           num_workers=num_workers)
        valid_loader = data.create_dataset_with_feedback('valid', self.seed, f_prop, f_off, 
                                           f_prop_norm, f_params, features,
                                           self.batch_size, shuffle=False, 
                                           num_workers=num_workers)

        # train/validate model
        min_valid = 1e40
        for epoch in range(self.epochs):

            # do training
            train_loss1, train_loss = torch.zeros(len(g)).to(device), 0.0
            train_loss2, points     = torch.zeros(len(g)).to(device), 0
            model.train()
            for x, y in train_loader:
                bs   = x.shape[0]         #batch size
                x    = x.to(device)       #maps
                y    = y.to(device)[:,g]  #parameters
                p    = model(x)           #NN output
                y_NN = p[:,g]             #posterior mean
                e_NN = p[:,h]             #posterior std
                loss1 = torch.mean((y_NN - y)**2,                axis=0)
                loss2 = torch.mean(((y_NN - y)**2 - e_NN**2)**2, axis=0)
                loss  = torch.mean(torch.log(loss1) + torch.log(loss2))
                train_loss1 += loss1*bs
                train_loss2 += loss2*bs
                points      += bs
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            train_loss = torch.log(train_loss1/points) + torch.log(train_loss2/points)
            train_loss = torch.mean(train_loss).item()

            # do validation
            valid_loss1, valid_loss = torch.zeros(len(g)).to(device), 0.0
            valid_loss2, points     = torch.zeros(len(g)).to(device), 0
            model.eval()
            for x, y in valid_loader:
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
                    valid_loss1 += loss1*bs
                    valid_loss2 += loss2*bs
                    points     += bs
            valid_loss = torch.log(valid_loss1/points) + torch.log(valid_loss2/points)
            valid_loss = torch.mean(valid_loss).item()

            # save best model if found
            if valid_loss<min_valid:  
                min_valid = valid_loss
                torch.save(model.state_dict(), fmodel)
            f = open(fout, 'a')
            f.write('%d %.5e %.5e\n'%(epoch, train_loss, valid_loss))
            f.close()

        print(features)
        print(f_prop)
        return min_valid

##################################### INPUT ##########################################
# legend
# 0-Mgas 1-Mstar 2-Mbh 3-Mtot 4-Vmax 5-Vdisp 6-Zg 
# 7-Zs 8-SFR 9-J 10-V 11-Rstar 12-Rtot 13-Rvmax

# data parameters
root        = '/mnt/ceph/users/camels/Software/1galaxy_cosmo/data'
sim         = 'IllustrisTNG'
f_prop      = '%s/galaxies_%s_z=0.00.txt'%(root,sim)
f_prop_norm = None
f_off       = '%s/offset_%s_z=0.00.txt'%(root,sim)
f_params    = '%s/latin_hypercube_params_%s.txt'%(root,sim)
seed        = 1
num_workers = 8
prefix      = 'all+A1_A2_A3_A4_z=0'
#prefix      = 'Mstar_Vdisp_Zs_Rstar_K'
#prefix      = 'Mgas_Mstar_Vmax_Zg_K_z=0'

# architecture parameters; for SIMBA use [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
features           = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16] #features used as input
#features           = [1,5,7,11,15] #features used as input
input_size         = len(features)+4 #number of subhalo properties
output_size        = 2 #number of parameters to predict (posterior mean + std)
max_layers         = 5
max_neurons_layers = 1500

# training parameters
batch_size = 1024
epochs     = 800
g          = [0]   #minimize loss using parameters 0 and 1
h          = [1] #minimize loss using errors of parameters 0 and 1

# optuna parameters
study_name       = 'gal_2_params'
n_trials         = 100 #set to None for infinite
storage          = 'sqlite:///databases/TPE_%s_%s.db'%(sim,prefix)
n_jobs           = 1
n_startup_trials = 30 #random sample the space before using the sampler
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

# create output folders if they dont exist
for fout in ['models/models_%s_%s'%(sim,prefix), 'losses/losses_%s_%s'%(sim,prefix)]:
    if not(os.path.exists(fout)):  os.system('mkdir %s'%fout)

# define the optuna study and optimize it
objective = Objective(input_size, output_size, max_layers, max_neurons_layers, 
                      device, epochs, seed, batch_size)
sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials)
study = optuna.create_study(study_name=study_name, sampler=sampler, storage=storage,
                            load_if_exists=True)
study.optimize(objective, n_trials, n_jobs=n_jobs)
