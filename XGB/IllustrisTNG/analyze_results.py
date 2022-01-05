import numpy as np
import sys,os

prop = {0:'Mgas', 1:'Mstar', 2:'Mbh', 3:'Mtot', 4:'Vmax', 5:'Vdisp', 6:'Zg', 
        7:'Zs', 8:'SFR', 9:'J', 10:'V', 11:'Rstar', 12:'Rtot', 13:'Rvmax',
        14:'U', 15:'K', 16:'g'}


for discarded in ['','9_','9_12_', '9_12_14_', '9_12_14_8_', '9_12_14_8_2_',
                  '9_12_14_8_2_16_', '9_12_14_8_2_16_10_', '9_12_14_8_2_16_10_6_',
                  '9_12_14_8_2_16_10_6_13_', '9_12_14_8_2_16_10_6_13_5_', 
                  '9_12_14_8_2_16_10_6_13_5_3_', '9_12_14_8_2_16_10_6_13_5_3_0_', 
                  '9_12_14_8_2_16_10_6_13_5_3_0_15_', 
                  '9_12_14_8_2_16_10_6_13_5_3_0_15_11_', 
                  '9_12_14_8_2_16_10_6_13_5_3_0_15_11_7_', 
                  '14_15_16_', '14_15_16_9_', '14_15_16_9_10_', '14_15_16_9_10_12_',
                  '14_15_16_9_10_12_13_',
              ]:

    results = np.zeros(17, dtype=np.float32)
    epochs  = np.zeros(17, dtype=np.int32)
    for i in range(17):
        fin = 'losses/loss_XGB_Om_IllustrisTNG_%s%d.txt'%(discarded,i)
        if not(os.path.exists(fin)):  continue
        epoch, valid_loss, test_loss = np.loadtxt(fin, unpack=True)
        results[i] = np.min(valid_loss)
        epochs[i]  = epoch[-1]

    print('##########################',discarded)
    ids = np.argsort(results)

    for i in ids:
        if results[i]==0:  continue
        print('%07s -----> %.3e %02d'%(prop[i],results[i],epochs[i]))

