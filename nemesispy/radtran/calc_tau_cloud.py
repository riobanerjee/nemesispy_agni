import numpy as np
from numba import jit
from nemesispy.radtran.calc_mmw import calc_mmw

# def calc_tau_cloud(wave_grid, H_layer, P_layer, U_layer, VMR_layer, T_layer, dH, Htop, Hknee, Xdeep, Xscat, mmw):
@jit(nopython=True)
def calc_tau_cloud(wave_grid, H_layer, P_layer, T_layer, Hknee, mmw, power=4.0):

    NLAYER = len(H_layer)
    NWAVE = len(wave_grid)
    tau_cloud = np.zeros((NWAVE,NLAYER))

    for ilayer in range(NLAYER):
        if H_layer[ilayer] < Hknee:
            tau_cloud[:,ilayer] = 1.0e3
        # else:
        #     for iwave in range(NWAVE):
        #         tau_cloud[iwave, ilayer] = 1.0/wave_grid[iwave]**power

    # xsec = 1.0
    # Q = np.zeros((NWAVE, NLAYER))
    # if Htop < Hknee:
    #     print('warning in cloud :: cloud top below cloud knee')
    #     return tau_cloud
    

    # for ilayer in range(NLAYER):
    #     for iwave in range(NWAVE):
    #         if H_layer[ilayer] > Hknee:
    #             Q[iwave, ilayer] = 1.0
    #         else:
    #             Q[iwave ,ilayer] = 1.0e30
    #         rho = P_layer[ilayer] * 0.1013 * mmw / (8.314 * T_layer[ilayer])
    #         nd = rho * Q[iwave, ilayer]
    #         tau_cloud[iwave, ilayer] = xsec * rho * Q[iwave, ilayer]
    
    return tau_cloud