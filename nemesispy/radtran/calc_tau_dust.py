from numba import jit
import numpy as np

@jit(nopython=True)
def calc_tau_dust(A_layer,phase_func):
    NMODE = phase_func.shape[0]
    NWAVE = phase_func.shape[1]
    NLAYER = A_layer.shape[0]
    
    tau_dust_ext = np.zeros((NWAVE,NLAYER))
    tau_dust_scat = np.zeros((NWAVE,NLAYER))
    lfrac = np.zeros((NWAVE,NMODE, NLAYER))  
    
    if np.any(A_layer > 0):
        xexts = phase_func[:,:,0]
        xscats = phase_func[:,:,1]

        tau_dust_m_ext = np.zeros((NMODE,NWAVE,NLAYER))
        tau_dust_m_scat = np.zeros((NMODE,NWAVE,NLAYER))

        for imode in range(NMODE):
            for iwave in range(NWAVE):
                for ilayer in range(NLAYER):
                    tau_dust_m_ext[imode,iwave,ilayer] = xexts[imode,iwave]*A_layer[ilayer,imode]
                    tau_dust_m_scat[imode,iwave,ilayer] = xscats[imode,iwave]*A_layer[ilayer,imode]

        tau_dust_ext = np.sum(tau_dust_m_ext,axis=0)
        tau_dust_scat = np.sum(tau_dust_m_scat,axis=0)           

        for imode in range(NMODE):
            for iwave in range(NWAVE):
                for ilayer in range(NLAYER):
                    if tau_dust_scat[iwave,ilayer]>0:
                        lfrac[iwave,imode,ilayer] = tau_dust_m_scat[imode,iwave,ilayer]/tau_dust_scat[iwave,ilayer]
                    
    return tau_dust_ext, tau_dust_scat, lfrac