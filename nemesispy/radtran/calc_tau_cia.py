#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""
Calculate collision-induced-absorption optical path.
"""
import numpy as np
from numba import jit
import pandas as pd
# from scipy.interpolate import pchip_interpolate
from nemesispy.common.constants import N_A
import os

@jit(nopython=True)
def co2cia(wave_grid):
    """
    Calculates CO2-CO2 CIA absorption coefficients.
    Parameters
    ----------
    wave_grid : ndarray

    Returns
    -------
    co2cia : ndarray
        CO2-CO2 CIA absorption coefficients.
    """

    # CO2 CIA
    NWAVE = len(wave_grid)
    co2cia = np.zeros((NWAVE))
    for iwave in range(NWAVE):
        xl = wave_grid[iwave]
        if 2.15 < xl < 2.55:
            a_co2 = 4e-8
        elif 1.7 < xl < 1.76:
            a_co2 = 6e-9
        elif 1.25 < xl < 1.35:
            a_co2 = 1.5e-9
        elif 1.125 < xl < 1.225:
            a_co2 = 0.5*(0.31+0.79)*1e-9
        elif 1.06 < xl < 1.125:
            a_co2 = 0.5*(0.29+0.67)*1e-9
        else:
            a_co2 = 0.0
        co2cia[iwave] = a_co2

    return co2cia

# At 400K, but not very T dependent anyway
# @jit(nopython=True)
# def n2n2cia(wave_grid):
#     """
#     Calculates N2-N2 CIA absorption coefficients.
#     Parameters
#     ----------
#     wave_grid : ndarray

#     Returns
#     -------
#     n2n2cia : ndarray
#         N2-N2 CIA absorption coefficients.
#     """
#     __location__ = os.environ['nemesispy_dir']
#     cia_path = os.path.join(__location__, "nemesispy/data/cia/n2n2.cia")
#     df = pd.read_csv(cia_path, header=None, skiprows=1, delim_whitespace=True)
#     df.columns = ['wavenumber', 'absorption_coef']
#     wavenumber = df['wavenumber'].values
#     wavenumber = wavenumber[::-1]
#     abs_coef = df['absorption_coef'].values
#     # only k2v ?
#     abs_coef = abs_coef[::-1]
#     wavelengths = 1e4/wavenumber[:]
#     # amagat = 2.68675E19
#     amagat = 44.615 * N_A / 1e6
#     abs_coef = abs_coef[:] * (amagat**2)
#     # aprx_coef = pchip_interpolate(wavelengths, abs_coef, wave_grid)
#     aprx_coef = np.interp(wave_grid, wavelengths, abs_coef)

#     # units?
#     return aprx_coef

@jit(nopython=True)
def calc_tau_cia(wave_grid, K_CIA, ISPACE,
    ID, TOTAM, T_layer, P_layer, VMR_layer, DELH,
    cia_nu_grid, TEMPS, INORMAL, NPAIR=9, aprx_coef=None):
    """
    Calculates
    Parameters
    ----------
    wave_grid : ndarray
        Wavenumber (cm-1) or wavelength array (um) at which to compute
        CIA opacities.
    ID : ndarray
        Gas ID
    # ISO : ndarray
    #     Isotop ID.
    VMR_layer : TYPE
        DESCRIPTION.
    ISPACE : int
        Flag indicating whether the calculation must be performed in
        wavenumbers (0) or wavelength (1)
    K_CIA(NPAIR,NTEMP,NWAVE) : ndarray
         CIA cross sections for each pair at each temperature level and wavenumber.
    cia_nu_grid : TYPE
        DESCRIPTION.
    INORMAL : int


    Returns
    -------
    tau_cia_layer(NWAVE,NLAY) : ndarray
        CIA optical depth in each atmospheric layer.
    """

    # Need to pass NLAY from a atm profile
    NPAIR = 9

    NLAY,NVMR = VMR_layer.shape
    ISO = np.zeros((NVMR))

    # mixing ratios of the relevant gases
    qh2 = np.zeros((NLAY))
    qhe = np.zeros((NLAY))
    qn2 = np.zeros((NLAY))
    qch4 = np.zeros((NLAY))
    qco2 = np.zeros((NLAY))
    # IABSORB = np.ones(5,dtype='int32') * -1

    NWAVEC = 17

    # get mixing ratios from VMR grid
    for iVMR in range(NVMR):
        if ID[iVMR] == 39: # hydrogen
            qh2[:] = VMR_layer[:,iVMR]
            # IABSORB[0] = iVMR
        if ID[iVMR] == 40: # helium
            qhe[:] = VMR_layer[:,iVMR]
            # IABSORB[1] = iVMR
        if ID[iVMR] == 22: # nitrogen
            qn2[:] = VMR_layer[:,iVMR]
            # IABSORB[2] = iVMR
        if ID[iVMR] == 6: # methane
            qch4[:] = VMR_layer[:,iVMR]
            # IABSORB[3] = iVMR
        if ID[iVMR] == 2: # co2
            qco2[:] = VMR_layer[:,iVMR]
            # IABSORB[4] = iVMR

    # calculating the opacity
    XLEN = DELH * 1.0e2 # cm
    TOTAM = TOTAM * 1.0e-4 # cm-2

    ### back to FORTRAN ORIGINAL
    P0 = 101325
    T0 = 273.15
    AMAGAT = 2.68675E19 #molecules cm-3
    KBOLTZMANN = 1.381E-23
    MODBOLTZA = 10 * KBOLTZMANN/1.013

    tau = (P_layer/P0)**2 * (T0/T_layer)**2 * DELH
    height1 = P_layer * MODBOLTZA * T_layer

    height = XLEN * 1e2
    amag1 = TOTAM /height/AMAGAT
    tau = height*amag1**2

    amag1 = TOTAM / XLEN / AMAGAT # number density
    tau = XLEN*amag1**2# optical path, why fiddle around with XLEN

    # define the calculatiion wavenumbers
    if ISPACE == 0: # input wavegrid is already in wavenumber (cm^-1)
        WAVEN = wave_grid
    elif ISPACE == 1:
        WAVEN = 1.e4/wave_grid
        isort = np.argsort(WAVEN)
        WAVEN = WAVEN[isort] # ascending wavenumbers

    # if WAVEN.min() < cia_nu_grid.min() or WAVEN.max()>cia_nu_grid.max():
    #     print('warning in CIA :: Calculation wavelengths expand a larger range than in .cia file')

    # calculate the CIA opacity at the correct temperature and wavenumber
    NWAVEC = len(wave_grid)  # Number of calculation wavelengths
    tau_cia_layer = np.zeros((NWAVEC,NLAY))

    for ilay in range(NLAY):
        # interpolating to the correct temperature
        temp1 = T_layer[ilay]
        it = (np.abs(TEMPS-temp1)).argmin()

        # want to sandwich the T point
        if TEMPS[it] >= temp1:
            ithi = it
            if it==0:
                # edge case, layer T < T grid
                temp1 = TEMPS[it]
                itl = 0
                ithi = 1
            else:
                itl = it - 1

        elif TEMPS[it]<temp1:
            NT = len(TEMPS)
            itl = it
            if it == NT - 1:
                # edge case, layer T > T grid
                temp1 = TEMPS[it]
                ithi = NT - 1
                itl = NT - 2
            else:
                ithi = it + 1

        # find opacities for the chosen T
        ktlo = K_CIA[:,itl,:]
        kthi = K_CIA[:,ithi,:]

        fhl = (temp1 - TEMPS[itl])/(TEMPS[ithi]-TEMPS[itl])
        fhh = (TEMPS[ithi]-temp1)/(TEMPS[ithi]-TEMPS[itl])

        kt = ktlo * (1.-fhl) + kthi * (1.-fhh)

        # checking that interpolation can be performed to the calculation wavenumbers
        inwave = np.where( (cia_nu_grid>=WAVEN.min()) & (cia_nu_grid<=WAVEN.max()) )
        inwave = inwave[0]

        if len(inwave)>0:

            k_cia = np.zeros((NWAVEC,NPAIR))
            inwave1 = np.where( (WAVEN>=cia_nu_grid.min())&(WAVEN<=cia_nu_grid.max()) )
            inwave1 = inwave1[0]

            for ipair in range(NPAIR):

                # wavenumber interpolation
                # f = interpolate.interp1d(cia_nu_grid,kt[ipair,:])
                # k_cia[inwave1,ipair] = f(WAVEN[inwave1])

                # use numpy for numba integration
                k_cia[inwave1,ipair] = np.interp(WAVEN[inwave1],cia_nu_grid,kt[ipair,:])

            #Combining the CIA absorption of the different pairs (included in .cia file)
            sum1 = np.zeros(NWAVEC)
            # print(k_cia[:,2])
            if INORMAL==0: # equilibrium hydrogen (1:1)
                sum1[:] = sum1[:] + k_cia[:,0] * qh2[ilay] * qh2[ilay] \
                    + k_cia[:,1] * qhe[ilay] * qh2[ilay]

            elif INORMAL==1: # normal hydrogen (3:1)
                sum1[:] = sum1[:] + k_cia[:,2] * qh2[ilay] * qh2[ilay]\
                    + k_cia[:,3] * qhe[ilay] * qh2[ilay]

            sum1[:] = sum1[:] + k_cia[:,4] * qh2[ilay] * qn2[ilay]
            sum1[:] = sum1[:] + k_cia[:,5] * qn2[ilay] * qch4[ilay]
            sum1[:] = sum1[:] + k_cia[:,6] * qn2[ilay] * qn2[ilay]
            sum1[:] = sum1[:] + k_cia[:,7] * qch4[ilay] * qch4[ilay]
            sum1[:] = sum1[:] + k_cia[:,8] * qh2[ilay] * qch4[ilay]

            # look up CO2-CO2 CIA coefficients (external)
            """
            TO BE DONE
            """
            # k_co2 = sum1*0
            # k_co2 = co2cia(wave_grid)

            # sum1[:] = sum1[:] + k_co2[:] * qco2[ilay] * qco2[ilay]

            #Look up N2-N2 NIR CIA coefficients
            if aprx_coef is not None:
                sum1[:] = sum1[:] + aprx_coef[::-1] * qn2[ilay] * qn2[ilay]

            # TO BE DONE

            #Look up N2-H2 NIR CIA coefficients
            """
            TO BE DONE
            """
            # TO BE DONE

            tau_cia_layer[:,ilay] = sum1[:] * tau[ilay]

    if ISPACE==1:
        tau_cia_layer[:,:] = tau_cia_layer[isort,:]

    return tau_cia_layer