from numba import njit
import numpy as np
from nemesispy.radtran.scatter import scloud11wave

@njit
def calc_spectrum_scloud11(wave_grid, phase_func,radground,
                                        tau_total,tau_rayleigh,
                                        omegas,lfrac,planck,angles):
    """
    Wrapper for scloud11wave.
    
    Parameters
    ----------
    wave_grid(NWAVE) : ndarray
        Wavelengths (um) grid for calculating spectra.
    phase_func(NMODE,NWAVE,6) : ndarray
        Contains fitted phase function parameters and 
        cross-sections for NMODE modes
    radground(NWAVE,NMU): ndarray
        Incident intensity at the bottom of the atmosphere
    tau_total(NWAVE,NLAY): ndarray
        Total optical thickness of each layer
    tau_total(NWAVE,NLAY): ndarray
        Rayleigh optical thickness of each layer 
    omegas(NWAVE,NLAY): ndarray
        Single scattering albedo of each layer
    lfrac(NMODE,NLAY): ndarray
        Fraction of scattering contributed by each mode in each layer
    planck(NWAVE,NLAY): ndarray
        Mean Planck function in each layer
    """
    
    # Set up quadrature
    mu = np.array([0.165278957666387,0.477924949810444,0.738773865105505,
        0.919533908166459,1.00000000000000])
    wtmu = np.array([0.327539761183898,0.292042683679684,0.224889342063117,
            0.133305990851069,2.222222222222220E-002])
    
    nf = int(angles[1]/3) # Number of fourier components
    
    # Calculate radiance
    spectrum = scloud11wave(phasarr = phase_func, 
                            radg = radground.transpose(),
                            sol_ang = angles[0], 
                            emiss_ang = angles[1], 
                            solar = np.ones_like(wave_grid), 
                            aphi = angles[2], 
                            lowbc = 1, 
                            galb = np.zeros_like(wave_grid), 
                            mu1 = mu, 
                            wt1 = wtmu, 
                            nf = nf,
                            vwaves = wave_grid, 
                            bnu = planck, 
                            tau = tau_total, 
                            tauray = tau_rayleigh,
                            omegas = omegas, 
                            nphi = 100,
                            iray = 1, 
                            lfrac = lfrac, 
                            imie=0)
    return spectrum