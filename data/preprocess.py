# This code reads all galaxy catalogues and creates two files. One file contains
# the properties of the galaxies for all simulations and the other contains the 
# offset and length of the galaxy catalogue of each simulation

import numpy as np
import sys,os,h5py

#################################### INPUT ###########################################
root         = '/mnt/ceph/users/camels/Sims'
snapnum      = 14 #33(z=0) 25(z=0.5) 19(z=1) 14(z=1.5) 10(z=2) 4(z=3)
Nstars_thres = 20
properties   = 17 #14 without U,K,g
realizations = 1000
######################################################################################

redshift = {33:0.0, 25:0.5, 14:1.5, 19:1.0, 10:2.0, 4:3.0}[snapnum]
print('Redshift: %.2f'%redshift)

# do a loop over IllustrisTNG and SIMBA
for sim in ['IllustrisTNG', 'SIMBA']:

    print('\n%s'%sim)

    # get the name of the files
    fout = 'galaxies_%s_z=%.2f.txt'%(sim,redshift)
    foff = 'offset_%s_z=%.2f.txt'%(sim,redshift)

    # find how many galaxies are there in total
    offset = np.zeros((realizations,2), dtype=np.int64)
    Ngal   = 0
    for i in range(realizations):

        # get the name of the galaxy catalogue
        fin = '%s/%s/LH_%d/fof_subhalo_tab_%03d.hdf5'%(root,sim,i,snapnum)

        # find the number of galaxies in it
        f = h5py.File(fin, 'r')
        Nstars = f['/Subhalo/SubhaloLenType'][:,4]
        f.close()

        indexes = np.where(Nstars>Nstars_thres)[0]
    
        offset[i,0] = Ngal
        offset[i,1] = len(indexes)
        Ngal += len(indexes)

    # define the matrix containing all galaxies with their properties
    print('Total number of galaxies found: %d'%Ngal)
    gal_prop = np.zeros((Ngal,properties), dtype=np.float32)

    # fill the matrix
    count = 0
    for i in range(realizations):

        # get the name of the galaxy catalogue
        fin = '%s/%s/LH_%d/fof_subhalo_tab_%03d.hdf5'%(root,sim,i,snapnum)

        # find the number of galaxies in it
        f      = h5py.File(fin, 'r')
        
        Mg     = f['/Subhalo/SubhaloMassType'][:,0]*1e10 #Msun/h
        Mstar  = f['/Subhalo/SubhaloMassType'][:,4]*1e10 #Msun/h
        Mbh    = f['/Subhalo/SubhaloBHMass'][:]*1e10     #Msun/h
        Mtot   = f['/Subhalo/SubhaloMass'][:]*1e10       #Msun/h
        
        Vmax   = f['/Subhalo/SubhaloVmax'][:]
        Vdisp  = f['/Subhalo/SubhaloVelDisp'][:]
        
        Zg     = f['/Subhalo/SubhaloGasMetallicity'][:]
        Zs     = f['/Subhalo/SubhaloStarMetallicity'][:]
        SFR    = f['/Subhalo/SubhaloSFR'][:]
        J      = f['/Subhalo/SubhaloSpin'][:]
        V      = f['/Subhalo/SubhaloVel'][:]
        J      = np.sqrt(J[:,0]**2 + J[:,1]**2 + J[:,2]**2)
        V      = np.sqrt(V[:,0]**2 + V[:,1]**2 + V[:,2]**2)

        Rstar  = f['/Subhalo/SubhaloHalfmassRadType'][:,4]/1e3 #Mpc/h
        Rtot   = f['/Subhalo/SubhaloHalfmassRad'][:]/1e3       #Mpc/h
        Rvmax  = f['/Subhalo/SubhaloVmaxRad'][:]/1e3           #Mpc/h

        U      = f['/Subhalo/SubhaloStellarPhotometrics'][:,0]
        K      = f['/Subhalo/SubhaloStellarPhotometrics'][:,3]
        g      = f['/Subhalo/SubhaloStellarPhotometrics'][:,4]

        Nstars = f['/Subhalo/SubhaloLenType'][:,4]
        f.close()

        # only take galaxies with more than 20 stars
        indexes = np.where(Nstars>Nstars_thres)[0]
        Ngal    = len(indexes)
        
        # fill the matrix
        gal_prop[count:count+Ngal,0]  = Mg[indexes]
        gal_prop[count:count+Ngal,1]  = Mstar[indexes]
        gal_prop[count:count+Ngal,2]  = Mbh[indexes]
        gal_prop[count:count+Ngal,3]  = Mtot[indexes]
        gal_prop[count:count+Ngal,4]  = Vmax[indexes]
        gal_prop[count:count+Ngal,5]  = Vdisp[indexes]
        gal_prop[count:count+Ngal,6]  = Zg[indexes]
        gal_prop[count:count+Ngal,7]  = Zs[indexes]
        gal_prop[count:count+Ngal,8]  = SFR[indexes]
        gal_prop[count:count+Ngal,9]  = J[indexes]
        gal_prop[count:count+Ngal,10] = V[indexes]
        gal_prop[count:count+Ngal,11] = Rstar[indexes]
        gal_prop[count:count+Ngal,12] = Rtot[indexes]
        gal_prop[count:count+Ngal,13] = Rvmax[indexes]
        gal_prop[count:count+Ngal,14] = U[indexes]
        gal_prop[count:count+Ngal,15] = K[indexes]
        gal_prop[count:count+Ngal,16] = g[indexes]
        count += Ngal

    print('Galaxies processed:             %d'%count)

    # save data to files
    header1 = '| offset in file | length |'
    header2 = '| gas mass | stellar mass | black-hole mass | total mass | Vmax | velocity dispersion | gas metallicity | stars metallicity | star-formation rate | spin | peculiar velocity | stellar radius | total radius | Vmax radius | U | K | g'
    np.savetxt(foff, offset,   fmt='%d',  header=header1)
    np.savetxt(fout, gal_prop, fmt='%.6e',header=header2)
