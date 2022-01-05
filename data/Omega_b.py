import numpy as np
import sys,os,h5py

galaxies     = 600
Nstars_thres = 20

data = np.zeros((600,5), dtype=np.float64)

root = '/mnt/home/sgenel/CAMELS/Sims/IllustrisTNG_extras/Paco_1gal_checks'

offset = 0
for Om in [0.2, 0.3, 0.4]:
    for Ob in [0.025, 0.075]:

        #print(Om,Ob)

        fin = '%s_Om_%.1f_Ob_%.3f/fof_subhalo_tab_033.hdf5'%(root,Om,Ob)
        
        f = h5py.File(fin, 'r')
        Nstars = f['/Subhalo/SubhaloLenType'][:,4]
        Mstar  = f['/Subhalo/SubhaloMassType'][:,4]*1e10 #Msun/h
        Vmax   = f['/Subhalo/SubhaloVmax'][:]
        Zs     = f['/Subhalo/SubhaloStarMetallicity'][:]
        f.close()

        indexes = np.where(Nstars>Nstars_thres)[0]
        Mstar   = Mstar[indexes]
        Vmax    = Vmax[indexes]
        Zs      = Zs[indexes]

        ids = np.random.choice(np.arange(len(Zs)), 100, replace=False)

        data[offset:offset+100, 0] = Mstar[ids]
        data[offset:offset+100, 1] = Vmax[ids]
        data[offset:offset+100, 2] = Zs[ids]
        data[offset:offset+100, 3] = np.ones(100)*Om
        data[offset:offset+100, 4] = np.ones(100)*Ob
        offset += 100

np.savetxt('galaxies_Omega_b.txt', data)
