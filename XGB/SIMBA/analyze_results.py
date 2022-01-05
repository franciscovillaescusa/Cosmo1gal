import numpy as np
import sys,os

prop = {0:'Mgas', 1:'Mstar', 2:'Mbh', 3:'Mtot', 4:'Vmax', 5:'Vdisp', 6:'Zg', 
        7:'Zs', 8:'SFR', 9:'J', 10:'V', 11:'Rstar', 12:'Rtot', 13:'Rvmax',
        14:'U', 15:'K', 16:'g'}


for discarded in ['', '9_', '9_2_', '9_2_6_', '9_2_6_10_', '9_2_6_10_12_',
                  '9_2_6_10_12_8_', '9_2_6_10_12_8_3_', '9_2_6_10_12_8_3_5_',
                  '9_2_6_10_12_8_3_5_0_', '9_2_6_10_12_8_3_5_0_11_',
                  '9_2_6_10_12_8_3_5_0_11_7_', '9_2_6_10_12_8_3_5_0_11_7_13_']:

    results = np.zeros(14, dtype=np.float32)
    epochs  = np.zeros(14, dtype=np.int32)
    for i in range(14):
        fin = 'losses/loss_XGB_Om_SIMBA_%s%d.txt'%(discarded,i)
        if not(os.path.exists(fin)):  continue
        epoch, valid_loss, test_loss = np.loadtxt(fin, unpack=True)
        results[i] = np.min(valid_loss)
        epochs[i]  = epoch[-1]

    print('##########################',discarded)
    ids = np.argsort(results)

    for i in ids:
        if results[i]==0:  continue
        print('%07s -----> %.3e %02d'%(prop[i],results[i],epochs[i]))

