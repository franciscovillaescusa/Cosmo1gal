import numpy as np 
import sys, os

# This routine takes a covariance matrix and computes its inverse and conditional number
def Inv_Cov(Cov):

    print('\n####################################################')
    # find eigenvalues and eigenvector of the covariance
    v1,w1 = np.linalg.eig(Cov)
    print('Max eigenvalue    Cov = %.3e'%np.max(v1))
    print('Min eigenvalue    Cov = %.3e'%np.min(v1))
    print('Condition number  Cov = %.3e'%(np.max(v1)/np.min(v1)))
    print(' ')

    # compute the inverse of the covariance
    ICov = np.linalg.inv(Cov)

    # find eigenvalues and eigenvector of the covariance
    v2,w2 = np.linalg.eig(ICov)
    print('Max eigenvalue   ICov = %.3e'%np.max(v2))
    print('Min eigenvalue   ICov = %.3e'%np.min(v2))
    print('Condition number ICov = %.3e'%(np.max(v2)/np.min(v2)))
    
    # check the product of the covariance and its inverse gives the identity matrix
    Equal = np.allclose(np.dot(Cov, ICov), np.eye(Cov.shape[0]))
    print('\nHas the inverse been properly found?',Equal)
    print('####################################################\n')
    
    return ICov

#################################### INPUT ###########################################
dOm = 0.01
ds8 = 0.02

k_min = 10.0 #h/Mpc
k_max = 30.0  #h/Mpc

realizations = 4500 #number of Pk for covariance
######################################################################################

# read Pk for derivatives
k1, Pk_Om_p = np.loadtxt('Pk/Pk_Om_p_z=0.0.txt', unpack=True)
k2, Pk_Om_m = np.loadtxt('Pk/Pk_Om_m_z=0.0.txt', unpack=True)
k3, Pk_s8_p = np.loadtxt('Pk/Pk_s8_p_z=0.0.txt', unpack=True)
k4, Pk_s8_m = np.loadtxt('Pk/Pk_s8_m_z=0.0.txt', unpack=True)

# sanity checks
indexes1 = np.where(k1!=k2)
indexes2 = np.where(k1!=k3)
indexes3 = np.where(k1!=k4)
print(indexes1, indexes2, indexes3)

# take only the modes where k_min < k < k_max
indexes = np.where((k1>=k_min) & (k1<k_max))[0]
bins = indexes.shape[0]

# compute derivatives
dPk_Om = (Pk_Om_p - Pk_Om_m)/(2.0*dOm)
dPk_s8 = (Pk_s8_p - Pk_s8_m)/(2.0*ds8)
dPk_Om = dPk_Om[indexes]
dPk_s8 = dPk_s8[indexes]
np.savetxt('derivative_Om_grid.txt', np.transpose([k1[indexes], dPk_Om]))
np.savetxt('derivative_s8_grid.txt', np.transpose([k1[indexes], dPk_s8]))

# define array with the data for the covariance matrix
data = np.zeros((realizations,bins), dtype=np.float64)

# fill the covariance matrix
for i in range(realizations):
    data[i] = np.loadtxt('Pk/Pk_fiducial_%d_z=0.0.txt'%(i+2))[:,1][indexes]
data_mean, data_std = np.mean(data, axis=0), np.std(data,  axis=0)

# compute the covariance
Cov = np.zeros((bins,bins), dtype=np.float64)

# compute the covariance matrix
for i in range(bins):
    for j in range(i,bins):
        Cov[i,j] = np.sum((data[:,i]-data_mean[i])*(data[:,j]-data_mean[j])) 
        if j>i:  Cov[j,i] = Cov[i,j]
Cov /= (realizations-1.0)

# compute the inverse of the covariance matrix
ICov = Inv_Cov(Cov)

# define the Fisher matrix
F = np.zeros((2,2), dtype=np.float64)
F[0,0] = np.dot(dPk_Om, np.dot(ICov, dPk_Om))
F[1,1] = np.dot(dPk_s8, np.dot(ICov, dPk_s8))
F[0,1] = 0.5*(np.dot(dPk_Om, np.dot(ICov, dPk_s8)) + 
              np.dot(dPk_s8, np.dot(ICov, dPk_Om)))
F[1,0] = F[0,1]


C = np.linalg.inv(F)

dOm = np.sqrt(C[0,0])
ds8 = np.sqrt(C[1,1])

print('Error Om: %.3f'%dOm)
print('Error s8: %.3f'%ds8)
