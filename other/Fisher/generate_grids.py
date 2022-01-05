import numpy as np
import sys,os
import density_field_library as DFL
import Pk_library as PKL

def generate_grid(grid, BoxSize, seed, Rayleigh_sampling, threads, verbose, Pk_file,
                  fout):

    # read power spectrum; k and Pk have to be floats, not doubles
    k, Pk = np.loadtxt(Pk_file, unpack=True)
    k, Pk = k.astype(np.float32), Pk.astype(np.float32)
    
    # generate a 3D Gaussian density field
    df_3D = DFL.gaussian_field_3D(grid, k, Pk, Rayleigh_sampling, seed,
                                  BoxSize, threads, verbose)

    # compute Pk of the field and save results to file
    Pk = PKL.Pk(df_3D, BoxSize, axis, MAS, threads, verbose)
    np.savetxt(fout, np.transpose([Pk.k3D, Pk.Pk[:,0]]))
    

##################################### INPUT ##########################################
# density field parameters
grid              = 256    #grid size
BoxSize           = 25.0   #Mpc/h
threads           = 28     #number of openmp threads
verbose           = True   #whether to print some information

# Pk parameters
axis = 0
MAS  = 'None'
######################################################################################


# grids to compute the derivatives
Rayleigh_sampling = 0 
seed              = 1 
for cosmo in ['Om_p', 'Om_m', 's8_p', 's8_m']:
    Pk_file = '%s/Pk_m_z=0.0.txt'%cosmo
    fout = 'Pk/Pk_%s_z=0.0.txt'%cosmo
    generate_grid(grid, BoxSize, seed, Rayleigh_sampling, threads, verbose, 
                  Pk_file, fout)

# grids to compute covariance
Rayleigh_sampling = 1
Pk_file = 'fiducial/Pk_m_z=0.0.txt'
for seed in range(2,5000):
    print('Density field: %d'%seed)
    fout = 'Pk/Pk_fiducial_%d_z=0.0.txt'%seed
    generate_grid(grid, BoxSize, seed, Rayleigh_sampling, threads, verbose, 
                  Pk_file, fout)
