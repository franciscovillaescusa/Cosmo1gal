import numpy as np
import sys,os
import integration_library as IL

#################################### INPUT ###########################################
dOm = 0.01
ds8 = 0.02

kmax  = 64.0    #h/Mpc
kmins = np.logspace(np.log10(0.3), np.log10(kmax*0.9), 100) #values of kmin

# integral parameters
x2       = kmax
eps      = 1e-10
h1       = 1e-15
hmin     = 0.0
function = 'linear'
bins     = 100000
verbose  = False
######################################################################################

# read power spectra
k,  Pk     = np.loadtxt('fiducial/Pk_m_z=0.0.txt', unpack=True)
k1, Pk_Omp = np.loadtxt('Om_p/Pk_m_z=0.0.txt', unpack=True)
k2, Pk_Omm = np.loadtxt('Om_m/Pk_m_z=0.0.txt', unpack=True)
k3, Pk_s8p = np.loadtxt('s8_p/Pk_m_z=0.0.txt', unpack=True)
k4, Pk_s8m = np.loadtxt('s8_m/Pk_m_z=0.0.txt', unpack=True)

# safety check
indexes1 = np.where(k1!=k2)
indexes2 = np.where(k1!=k3)
indexes3 = np.where(k1!=k4)
indexes4 = np.where(k1!=k)
print(indexes1, indexes2)
print(indexes3, indexes4)

# compute derivatives
#dlogPk_dOm = (np.log(Pk_Omp) - np.log(Pk_Omm))/(2.0*dOm)
#dlogPk_ds8 = (np.log(Pk_s8p) - np.log(Pk_s8m))/(2.0*ds8)
#dlogPk_dOm_interp = np.interp(x, k1, dlogPk_dOm)
#dlogPk_ds8_interp = np.interp(x, k1, dlogPk_ds8)
dPk_dOm = (Pk_Omp - Pk_Omm)/(2.0*dOm)
dPk_ds8 = (Pk_s8p - Pk_s8m)/(2.0*ds8)

np.savetxt('derivative_Om.txt', np.transpose([k1, dPk_dOm]))
np.savetxt('derivative_s8.txt', np.transpose([k1, dPk_ds8]))

f = open('constraints_vs_kmin.txt', 'w')
for kmin in kmins:
    x1 = kmin
    x  = np.linspace(x1, x2, bins)

    dPk_dOm_interp = np.interp(x, k1, dPk_dOm)
    dPk_ds8_interp = np.interp(x, k1, dPk_ds8)
    Pk_interp      = np.interp(x, k1, Pk)

    # calculate the corresponding volume
    V = (2.0*np.pi/kmin)**3

    # define the Fisher matrix
    F = np.zeros((2,2), dtype=np.float64)

    # compute Fisher
    yinit = np.zeros(1, dtype=np.float64)
    #integrant = dlogPk_dOm_interp*dlogPk_dOm_interp*x**2*V/(2.0*np.pi)**2
    integrant = dPk_dOm_interp*dPk_dOm_interp/Pk_interp**2*x**2*V/(2.0*np.pi)**2
    F[0,0] = IL.odeint(yinit, x1, x2, eps, h1, hmin, x, integrant, function,
                       verbose=verbose)[0]
    
    yinit = np.zeros(1, dtype=np.float64)
    #integrant = dlogPk_dOm_interp*dlogPk_ds8_interp*x**2*V/(2.0*np.pi)**2
    integrant = dPk_dOm_interp*dPk_ds8_interp/Pk_interp**2*x**2*V/(2.0*np.pi)**2
    F[0,1] = IL.odeint(yinit, x1, x2, eps, h1, hmin, x, integrant, function, 
                       verbose=verbose)[0]
    
    F[1,0] = F[0,1]

    yinit = np.zeros(1, dtype=np.float64)
    #integrant = dlogPk_ds8_interp*dlogPk_ds8_interp*x**2*V/(2.0*np.pi)**2
    integrant = dPk_ds8_interp*dPk_ds8_interp/Pk_interp**2*x**2*V/(2.0*np.pi)**2
    F[1,1] = IL.odeint(yinit, x1, x2, eps, h1, hmin, x, integrant, function, 
                       verbose=verbose)[0]

    C = np.linalg.inv(F)

    dOm = np.sqrt(C[0,0])
    ds8 = np.sqrt(C[1,1])

    print('%.3f %.3f %.3f'%(kmin,dOm,ds8))
    f.write('%.3e %.3e %.3e\n'%(kmin,dOm,ds8))
f.close()
