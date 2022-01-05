import torch 
import torch.nn as nn
import numpy as np
import sys, os, time
import optuna


######## 1 hidden layer ##########
# inp ---------> size of input data
# h1 ----------> size of first hidden layer
# out ---------> size of output data
# dr ----------> dropout rate
class model_1hl(nn.Module):
    
    def __init__(self, inp, h1, out, dr):
        super(model_1hl, self).__init__()
        
        self.fc1 = nn.Linear(inp, h1) 
        self.fc2 = nn.Linear(h1,  out)
	
        self.dropout   = nn.Dropout(p=dr)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
        # initialize the weights of the different layers
        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose2d) or \
                 isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
       
    # forward pass
    def forward(self, x):
        out = self.dropout(self.LeakyReLU(self.fc1(x)))
        out = self.fc2(out)         
        return out
##################################

######## 2 hidden layers #########
# inp ---------> size of input data
# h1 ----------> size of first hidden layer
# h2 ----------> size of second hidden layer
# out ---------> size of output data
# dr ----------> dropout rate
class model_2hl(nn.Module):
    
    def __init__(self, inp, h1, h2, out, dr):
        super(model_2hl, self).__init__()
        
        self.fc1 = nn.Linear(inp, h1) 
        self.fc2 = nn.Linear(h1,  h2)
        self.fc3 = nn.Linear(h2,  out)
	
        self.dropout   = nn.Dropout(p=dr)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU()
        
        # initialize the weights of the different layers
        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose2d) or \
                 isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
       
    # forward pass
    def forward(self, x):
        out = self.dropout(self.LeakyReLU(self.fc1(x)))
        out = self.dropout(self.LeakyReLU(self.fc2(out)))
        out = self.fc3(out)         
        return out
##################################

######## 3 hidden layers #########
# inp ---------> size of input data
# h1 ----------> size of first hidden layer
# h2 ----------> size of second hidden layer
# h3 ----------> size of third  hidden layer
# out ---------> size of output data
# dr ----------> dropout rate
class model_3hl(nn.Module):
    
    def __init__(self, inp, h1, h2, h3, out, dr):
        super(model_3hl, self).__init__()
        
        self.fc1 = nn.Linear(inp, h1) 
        self.fc2 = nn.Linear(h1,  h2)
        self.fc3 = nn.Linear(h2,  h3)
        self.fc4 = nn.Linear(h3,  out)
	
        self.dropout   = nn.Dropout(p=dr)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU()
        
        # initialize the weights of the different layers
        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose2d) or \
                 isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
       
    # forward pass
    def forward(self, x):
        out = self.dropout(self.LeakyReLU(self.fc1(x)))
        out = self.dropout(self.LeakyReLU(self.fc2(out)))
        out = self.dropout(self.LeakyReLU(self.fc3(out)))
        out = self.fc4(out)         
        return out
##################################

######## 4 hidden layers #########
# inp ---------> size of input data
# h1 ----------> size of first hidden layer
# h2 ----------> size of second hidden layer
# h3 ----------> size of third  hidden layer
# h4 ----------> size of fourth hidden layer
# out ---------> size of output data
# dr ----------> dropout rate
class model_4hl(nn.Module):
    
    def __init__(self, inp, h1, h2, h3, h4, out, dr):
        super(model_4hl, self).__init__()
        
        self.fc1 = nn.Linear(inp, h1) 
        self.fc2 = nn.Linear(h1,  h2)
        self.fc3 = nn.Linear(h2,  h3)
        self.fc4 = nn.Linear(h3,  h4)
        self.fc5 = nn.Linear(h4,  out)
	
        self.dropout   = nn.Dropout(p=dr)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU()
        
        # initialize the weights of the different layers
        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose2d) or \
                 isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
       
    # forward pass
    def forward(self, x):
        out = self.dropout(self.LeakyReLU(self.fc1(x)))
        out = self.dropout(self.LeakyReLU(self.fc2(out)))
        out = self.dropout(self.LeakyReLU(self.fc3(out)))
        out = self.dropout(self.LeakyReLU(self.fc4(out)))
        out = self.fc5(out)         
        return out
##################################


# This routine returns an architecture that is built inside the routine itself
# It can have from 1 to max_layers hidden layers. The user specifies the size of the
# input and output together with the maximum number of neurons in each layers
# trial -------------> optuna variable
# input_size --------> size of the input
# output_size -------> size of the output
# max_layers --------> maximum number of hidden layers to consider (default=3)
# max_neurons_layer -> the maximum number of neurons a layer can have (default=500)
def dynamic_model(trial, input_size, output_size, max_layers=3, max_neurons_layers=500):

    # define the tuple containing the different layers
    layers = []

    # get the number of hidden layers
    n_layers = trial.suggest_int("n_layers", 1, max_layers)

    # get the hidden layers
    in_features = input_size
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 4, max_neurons_layers)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.LeakyReLU(0.2))
        p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.8)
        layers.append(nn.Dropout(p))
        in_features = out_features

    # get the last layer
    layers.append(nn.Linear(out_features, output_size))

    # return the model
    return nn.Sequential(*layers)

def dynamic_model2(input_size, output_size, n_layers, hidden, dr):

    # define the tuple containing the different layers
    layers = []

    # get the hidden layers
    in_features = input_size
    for i in range(n_layers):
        out_features = hidden[i]
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Dropout(dr[i]))
        in_features = out_features

    # get the last layer
    layers.append(nn.Linear(out_features, output_size))

    # return the model
    return nn.Sequential(*layers)
