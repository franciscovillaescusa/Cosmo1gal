import numpy as np
import sys,os


###################################### INPUT ##########################################
sim    = 'IllustrisTNG'
#prefix = 'Mtot_Vmax_Vdisp_Zs'
#prefix = 'Mstar_Mtot_Vmax_Vdisp'
#prefix = 'Mtot_Vmax_Vdisp'
#prefix = 'Mgas_Vmax_Zgas'
#prefix = 'Mstar_Mtot_Vmax_Vdisp_Zs'
#prefix = 'all'
prefix = 'all_UKg'
trial  = 1
fin    = 'Results_%s_%s_%d.txt'%(sim,prefix,trial)
fout   = 'stats_%s_%s_%d.txt'%(sim,prefix,trial)
#######################################################################################

# read the data
data = np.loadtxt(fin)

# get the unique values of Omega_m 
Om_unique = np.unique(data[:,0])
Om_unique = np.sort(Om_unique)

# do a loop over all unique values of Omega_m
f = open(fout, 'w')
for Om in Om_unique:

    # select the subhalos with that value of Omega_m
    indexes = np.where(data[:,0]==Om)[0]

    # compute MSE
    MSE = np.mean((data[indexes,0]-data[indexes,1])**2)

    # compute accuracy
    acc = np.mean(np.absolute(data[indexes,2])/data[indexes,1])

    # compute bias
    bias = np.mean(data[indexes,0]-data[indexes,1])/np.mean(data[indexes,2])
    bias = np.absolute(bias)

    # compute mean Om and mean error
    mean_Om  = np.mean(data[indexes,1])
    mean_dOm = np.mean(np.absolute(data[indexes,2]))

    print('Om: %.4f %.3e %.3f %.3e %.3f %.3e %d'%(Om,MSE,acc,bias,mean_Om,mean_dOm,len(indexes)))
    f.write('%.4f %.3e %.3f %.3e %.3f %.3e %d\n'%(Om,MSE,acc,bias,mean_Om,mean_dOm,len(indexes)))
f.close()
