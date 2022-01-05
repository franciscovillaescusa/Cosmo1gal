import torch 
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import sys, os, time

# This class creates the dataset 
class make_dataset():

    def __init__(self, mode, seed, f_prop, f_off, f_prop_norm, f_params, features):

        # read data, scale it, and normalize it
        data = np.loadtxt(f_prop)
        data = data[:,features]
        data[np.where(data==0.0)] += 1e-30
        indexes = np.where(data>0.0)
        data[indexes] = np.log10(data[indexes])
        if f_prop_norm is None:
            mean, std = np.mean(data, axis=0), np.std(data, axis=0)
        else:
            data_norm = np.loadtxt(f_prop_norm)
            data_norm = data_norm[:,features]
            data_norm[np.where(data_norm==0.0)] += 1e-30
            indexes = np.where(data_norm>0.0)
            data_norm[indexes] = np.log10(data_norm[indexes])
            mean, std = np.mean(data_norm, axis=0), np.std(data_norm, axis=0)
        data = (data - mean)/std
        off, length = np.loadtxt(f_off, dtype=np.int64, unpack=True)

        # read the value of the cosmological & astrophysical parameters; normalize them
        params  = np.loadtxt(f_params)
        minimum = np.array([0.1, 0.6, 0.25, 0.25, 0.5, 0.5])
        maximum = np.array([0.5, 1.0, 4.00, 4.00, 2.0, 2.0])
        params  = (params - minimum)/(maximum - minimum)

        # get the size and offset depending on the type of dataset
        sims = params.shape[0]
        if   mode=='train':  size, offset = int(sims*0.85), int(sims*0.00)
        elif mode=='valid':  size, offset = int(sims*0.10), int(sims*0.85)
        elif mode=='test':   size, offset = int(sims*0.05), int(sims*0.95)
        elif mode=='all':    size, offset = int(sims*1.00), int(sims*0.00)
        else:                raise Exception('Wrong name!')

        # randomly shuffle the sims. Instead of 0 1 2 3...999 have a 
        # random permutation. E.g. 5 9 0 29...342
        np.random.seed(seed)
        indexes = np.arange(sims) #shuffle the order of the simulations
        np.random.shuffle(indexes)
        indexes = indexes[offset:offset+size] #select indexes of mode

        # get the indexes of the galaxies in the considered set
        Ngal = 0
        for i in indexes:
            Ngal += length[i]
        print('Number of galaxies in the %s set: %d'%(mode,Ngal))

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
        print('processed %d galaxies'%count)

        # define size, input and output matrices
        self.size   = Ngal
        self.input  = torch.tensor(prop, dtype=torch.float)
        self.output = torch.tensor(pars, dtype=torch.float)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.input[idx], self.output[idx]

# This class creates the dataset 
class make_dataset_with_feedback():

    def __init__(self, mode, seed, f_prop, f_off, f_prop_norm, f_params, features):

        # read data, scale it, and normalize it
        data = np.loadtxt(f_prop)
        data = data[:,features]
        data[np.where(data==0.0)] += 1e-30
        indexes = np.where(data>0.0)
        data[indexes] = np.log10(data[indexes])
        if f_prop_norm is None:
            mean, std = np.mean(data, axis=0), np.std(data, axis=0)
        else:
            data_norm = np.loadtxt(f_prop_norm)
            data_norm = data_norm[:,features]
            data_norm[np.where(data_norm==0.0)] += 1e-30
            indexes = np.where(data_norm>0.0)
            data_norm[indexes] = np.log10(data_norm[indexes])
            mean, std = np.mean(data_norm, axis=0), np.std(data_norm, axis=0)
        data = (data - mean)/std
        off, length = np.loadtxt(f_off, dtype=np.int64, unpack=True)

        # read the value of the cosmological & astrophysical parameters; normalize them
        params  = np.loadtxt(f_params)
        minimum = np.array([0.1, 0.6, 0.25, 0.25, 0.5, 0.5])
        maximum = np.array([0.5, 1.0, 4.00, 4.00, 2.0, 2.0])
        params  = (params - minimum)/(maximum - minimum)

        # get the size and offset depending on the type of dataset
        sims = params.shape[0]
        if   mode=='train':  size, offset = int(sims*0.85), int(sims*0.00)
        elif mode=='valid':  size, offset = int(sims*0.10), int(sims*0.85)
        elif mode=='test':   size, offset = int(sims*0.05), int(sims*0.95)
        elif mode=='all':    size, offset = int(sims*1.00), int(sims*0.00)
        else:                raise Exception('Wrong name!')

        # randomly shuffle the sims. Instead of 0 1 2 3...999 have a 
        # random permutation. E.g. 5 9 0 29...342
        np.random.seed(seed)
        indexes = np.arange(sims) #shuffle the order of the simulations
        np.random.shuffle(indexes)
        indexes = indexes[offset:offset+size] #select indexes of mode

        # get the indexes of the galaxies in the considered set
        Ngal = 0
        for i in indexes:
            Ngal += length[i]
        print('Number of galaxies in the %s set: %d'%(mode,Ngal))

        # define the arrays containing the properties and the parameter values
        prop = np.zeros((Ngal, data.shape[1]+4),   dtype=np.float32)
        pars = np.zeros((Ngal, params.shape[1]), dtype=np.float32)

        # fill the arrays
        num_features = len(features)
        count = 0
        for i in indexes:
            for j in range(length[i]):
                prop[count, :num_features] = data[off[i] + j]
                prop[count, num_features:] = params[i,[2,3,4,5]]
                pars[count] = params[i]
                count += 1
        print('processed %d galaxies'%count)

        # define size, input and output matrices
        self.size   = Ngal
        self.input  = torch.tensor(prop, dtype=torch.float)
        self.output = torch.tensor(pars, dtype=torch.float)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.input[idx], self.output[idx]


# This routine creates a dataset loader
# mode ---------------> 'train', 'valid', 'test' or 'all'
# seed ---------------> random seed to split data among training, validation and testing
# f_Pk ---------------> file containing the power spectra
# f_Pk_norm ----------> file containing the power spectra to normalize data
# f_params -----------> files with the value of the cosmological + astrophysical params
# features -----------> tuple with the indexes of the features to use as input
# batch_size ---------> batch size
# shuffle ------------> whether to shuffle the data or not
# num_workers --------> number of cpus to load the data
def create_dataset(mode, seed, f_prop, f_off, f_prop_norm, f_params, features,
                   batch_size, shuffle, num_workers=1):
    data_set = make_dataset(mode, seed, f_prop, f_off, f_prop_norm, f_params, features)
    return DataLoader(dataset=data_set, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers)

def create_dataset_with_feedback(mode, seed, f_prop, f_off, f_prop_norm, f_params, 
                                 features, batch_size, shuffle, num_workers=1):
    data_set = make_dataset_with_feedback(mode, seed, f_prop, f_off, f_prop_norm, 
                                          f_params, features)
    return DataLoader(dataset=data_set, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers)
