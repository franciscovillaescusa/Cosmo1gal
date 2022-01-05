import numpy as np
import sys,os,glob
import optuna

# get the best validation loss of the database
def loss_database(study_name, storage):

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
        #print("Trial number {}".format(trial.number))
        #print("Value: %.5e"%trial.value)
        #print(" Params: ")
        #for key, value in trial.params.items():
        #    print("    {}: {}".format(key, value))
        return trial.value


######################################## INPUT ########################################
# optuna parameters
sim = 'IllustrisTNG'
study_name = 'gal_2_params'
#######################################################################################


prefixes = glob.glob('databases/TPE_%s_*.db'%sim)
losses   = np.zeros(len(prefixes))

for i,prefix in enumerate(prefixes):

    storage = 'sqlite:///%s'%(prefix)
    losses[i] = loss_database(study_name, storage)
    
indexes = np.argsort(losses)

for i in indexes:
    print('%.3e ----> %s'%(losses[i],prefixes[i][27:-3]))
