#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""
Routines to calculate thermal emission spectra from a planetary atmosphere
using the correlated-k method to combine gaseous opacities.
We inlcude collision-induced absorption from H2-H2, H2-he, H2-N2, N2-Ch4, N2-N2,
Ch4-Ch4, H2-Ch4 pairs and Rayleigh scattering from H2 molecules and
He molecules.

As of now the routines are fully accelerated using numba.jit.
"""
import numpy as np
import random
from numba import jit
from nemesispy.radtran.calc_planck import calc_planck
from nemesispy.radtran.calc_tau_gas import calc_tau_gas
from nemesispy.radtran.calc_tau_gas import calc_tau_gas_comp
from nemesispy.radtran.calc_tau_cia import calc_tau_cia
from nemesispy.radtran.calc_tau_rayleigh import calc_tau_rayleigh
from nemesispy.radtran.calc_tau_cloud import calc_tau_cloud
from nemesispy.radtran.calc_tau_dust import calc_tau_dust
from nemesispy.radtran.calc_spectrum_scloud11 import calc_spectrum_scloud11
from nemesispy.common.constants import ATM
import time


@jit(nopython=True)
def calc_transm(wave_grid, H_layer, H_base, U_layer, P_layer,P_base, T_layer, VMR_layer,
    k_gas_w_g_p_t, P_grid, T_grid, del_g, ScalingFactor, R_plt, R_star, solspec,
    k_cia, ID, cia_nu_grid, cia_T_grid, dH, mmw, Ptop, power, cia_contrib_on, gas_contrib_on,
    rayleigh_contrib_on, cloud_contrib_on, aprx_coef, hazemult, h2_frac=None):
    """
    Calculate transmission spectrum using the correlated-k method.

    Parameters
    ----------
    wave_grid(NWAVE) : ndarray
        Wavelengths (um) grid for calculating spectra.
    H_layer(NLAYER) : ndarray
        Height of each layer.
        Unit: m
    U_layer(NLAYER) : ndarray
        Surface density of gas particles in each layer.
        Unit: no. of particle/m^2
    P_layer(NLAYER) : ndarray
        Atmospheric pressure grid.
        Unit: Pa
    T_layer(NLAYER) : ndarray
        Atmospheric temperature grid.
        Unit: K
    VMR_layer(NLAYER,NGAS) : ndarray
        Array of volume mixing ratios for NGAS.
        Has dimensioin: NLAYER x NGAS
    k_gas_w_g_p_t(NGAS,NWAVE,NG,NPRESSK,NTEMPK) : ndarray
        k-coefficients.
        Has dimension: NGAS x NWAVE x NG x NPRESSK x NTEMPK.
    P_grid(NPRESSK) : ndarray
        Pressure grid on which the k-coeff's are pre-computed.
        We want SI unit (Pa) here.
    T_grid(NTEMPK) : ndarray
        Temperature grid on which the k-coeffs are pre-computed. In Kelvin
    del_g : ndarray
        Quadrature weights of the g-ordinates.
    ScalingFactor(NLAYER) : ndarray
        Scale stuff to line of sight
    R_plt : real
        Planetary radius
        Unit: m
    solspec : ndarray
        Stellar spectra, used when the unit of the output is in fraction
        of stellar irradiance.

        Stellar flux at planet's distance (W cm-2 um-1 or W cm-2 (cm-1)-1)

    Returns
    -------
    spectrum : ndarray
        Output spectrum (W cm-2 um-1 sr-1)
    """
    NGAS, NWAVE, NG, NPRESS, NTEMP = k_gas_w_g_p_t.shape
    NLAYER = len(P_layer)

    # Initiate arrays to record total optical paths
    tau_total_w_g_l = np.zeros((NWAVE,NG,NLAYER))
    # Collision induced absorptioin optical path (NWAVE x NLAYER)
    tau_cia = calc_tau_cia(wave_grid=wave_grid,K_CIA=k_cia,ISPACE=1,
        ID=ID,TOTAM=U_layer,T_layer=T_layer,P_layer=P_layer,VMR_layer=VMR_layer,
        DELH=dH,cia_nu_grid=cia_nu_grid,TEMPS=cia_T_grid,INORMAL=1,NPAIR=9, aprx_coef=aprx_coef)
    
    # Rayleigh scattering optical path (NWAVE x NLAYER)

    tau_rayleigh = calc_tau_rayleigh(wave_grid=wave_grid,U_layer=U_layer,fH2=h2_frac)
    # FORTRAN straight transcript
    tau_gas = calc_tau_gas(k_gas_w_g_p_t, P_layer, T_layer, VMR_layer, U_layer,
        P_grid, T_grid, del_g)
    
    # Cloud optical path (NWAVE x NLAYER)
    if hazemult is not None:
        tau_cloud = calc_tau_cloud(wave_grid, P_layer, T_layer, mmw, Ptop, power, hazemult)
    else:
        tau_cloud = calc_tau_cloud(wave_grid, P_layer, T_layer, mmw, Ptop, power)


    cia_on = 1.0 # CO2 CIA?
    gas_on = 1.0
    rayleigh_on = 1.0 # Need CO2 Rayleigh scattering
    cloud_on = 1.0

    if cia_contrib_on == False:
        cia_on = 0.0
    if gas_contrib_on == False:
        gas_on = 0.0
    if rayleigh_contrib_on == False:
        rayleigh_on = 0.0
    if cloud_contrib_on == False:
        cloud_on = 0.0

    # Merge all different opacities
    for iwave in range(NWAVE):
        for ig in range(NG):
            for ilayer in range(NLAYER):
                tau_total_w_g_l[iwave,ig,ilayer] = rayleigh_on * tau_rayleigh[iwave,ilayer] \
                    + gas_on * tau_gas[iwave,ig,ilayer] \
                    + cia_on * tau_cia[iwave,ilayer] \
                    + cloud_on * tau_cloud[iwave,ilayer]

    # Create list of atmospheric paths for transit geometry
    paths = []
    NPATH = NLAYER
    for ipath in range(NPATH):
        path = list(range(NPATH-1, NPATH-ipath-2, -1))\
        + list(range(NPATH-ipath-1, NPATH, 1))
        paths.append(path)
    paths = paths[::-1]


    y1 = np.zeros((NPATH, NWAVE))
    area = np.zeros(NWAVE)
    for ipath in range(NPATH):

        tau_cum_w_g = np.zeros((NWAVE,NG))
        h_pathbase = R_plt + H_base[ipath]

        for iwave in range(NWAVE):
            for ig in range(NG):
                for ilayer, layer_id in enumerate(paths[ipath]):
                    # Scale to the line-of-sight opacities
                    tau_cum_w_g[iwave,ig] =  tau_cum_w_g[iwave,ig] + \
                    tau_total_w_g_l[iwave,ig,layer_id]*ScalingFactor[ipath, ilayer]
                  
        # print(tau_cum_w_g)
        tr_w_g = 1.0 - np.exp(-tau_cum_w_g)
        tr_w = np.zeros(NWAVE)

        for iwave in range(NWAVE):
            for ig in range(NG):
                tr_w[iwave] = tr_w[iwave] + tr_w_g[iwave, ig] * del_g[ig]
            

        # if np.isnan(tr_w).any():
        #     print("Error: NaN in optical depth")
        #     break
        
        y1[ipath] = 2 * np.pi * h_pathbase * tr_w
        
    for ipath in range(NPATH-1):
        area = area + 0.5 * (y1[ipath+1] + y1[ipath]) * dH[ipath]
    
    area_star = np.pi * R_star**2
    area_plt_bottom = np.pi * (R_plt + H_base[0])**2
    td_percent = 100.0 * (area + area_plt_bottom)/area_star

    return td_percent

@jit(nopython=True)
def calc_transm_dual(wave_grid, H_layer, H_base, U_layer, U_layer_n, P_layer, P_layer_n,
                     P_base, T_layer, T_layer_n, VMR_layer, VMR_layer_n,
    k_gas_w_g_p_t, P_grid, T_grid, del_g, ScalingFactor, R_plt, R_star, solspec,
    k_cia, ID, cia_nu_grid, cia_T_grid, dH, mmw, mmw_n, Ptop, power, Ptop_n, power_n,cia_contrib_on, gas_contrib_on,
    rayleigh_contrib_on, cloud_contrib_on, aprx_coef, hazemult, hazemult_n, h2_frac=None):
    """
    Calculate transmission spectrum using the correlated-k method.

    Parameters
    ----------
    wave_grid(NWAVE) : ndarray
        Wavelengths (um) grid for calculating spectra.
    H_layer(NLAYER) : ndarray
        Height of each layer.
        Unit: m
    U_layer(NLAYER) : ndarray
        Surface density of gas particles in each layer.
        Unit: no. of particle/m^2
    P_layer(NLAYER) : ndarray
        Atmospheric pressure grid.
        Unit: Pa
    T_layer(NLAYER) : ndarray
        Atmospheric temperature grid.
        Unit: K
    VMR_layer(NLAYER,NGAS) : ndarray
        Array of volume mixing ratios for NGAS.
        Has dimensioin: NLAYER x NGAS
    k_gas_w_g_p_t(NGAS,NWAVE,NG,NPRESSK,NTEMPK) : ndarray
        k-coefficients.
        Has dimension: NGAS x NWAVE x NG x NPRESSK x NTEMPK.
    P_grid(NPRESSK) : ndarray
        Pressure grid on which the k-coeff's are pre-computed.
        We want SI unit (Pa) here.
    T_grid(NTEMPK) : ndarray
        Temperature grid on which the k-coeffs are pre-computed. In Kelvin
    del_g : ndarray
        Quadrature weights of the g-ordinates.
    ScalingFactor(NLAYER) : ndarray
        Scale stuff to line of sight
    R_plt : real
        Planetary radius
        Unit: m
    solspec : ndarray
        Stellar spectra, used when the unit of the output is in fraction
        of stellar irradiance.

        Stellar flux at planet's distance (W cm-2 um-1 or W cm-2 (cm-1)-1)

    Returns
    -------
    spectrum : ndarray
        Output spectrum (W cm-2 um-1 sr-1)
    """
    NGAS, NWAVE, NG, NPRESS, NTEMP = k_gas_w_g_p_t.shape
    NLAYER = len(P_layer)

    # Initiate arrays to record total optical paths
    tau_total_w_g_l = np.zeros((NWAVE,NG,NLAYER))
    # Collision induced absorptioin optical path (NWAVE x NLAYER)
    tau_cia = calc_tau_cia(wave_grid=wave_grid,K_CIA=k_cia,ISPACE=1,
        ID=ID,TOTAM=U_layer,T_layer=T_layer,P_layer=P_layer,VMR_layer=VMR_layer,
        DELH=dH,cia_nu_grid=cia_nu_grid,TEMPS=cia_T_grid,INORMAL=1,NPAIR=9, aprx_coef=aprx_coef)
    
    # Rayleigh scattering optical path (NWAVE x NLAYER)

    tau_rayleigh = calc_tau_rayleigh(wave_grid=wave_grid,U_layer=U_layer,fH2=h2_frac)
    # FORTRAN straight transcript
    tau_gas = calc_tau_gas(k_gas_w_g_p_t, P_layer, T_layer, VMR_layer, U_layer,
        P_grid, T_grid, del_g)
    
    # Cloud optical path (NWAVE x NLAYER)
    if hazemult is not None:
        tau_cloud = calc_tau_cloud(wave_grid, P_layer, T_layer, mmw, Ptop, power, hazemult)
    else:
        tau_cloud = calc_tau_cloud(wave_grid, P_layer, T_layer, mmw, Ptop, power)


    cia_on = 1.0 # CO2 CIA?
    gas_on = 1.0
    rayleigh_on = 1.0 # Need CO2 Rayleigh scattering
    cloud_on = 1.0

    if cia_contrib_on == False:
        cia_on = 0.0
    if gas_contrib_on == False:
        gas_on = 0.0
    if rayleigh_contrib_on == False:
        rayleigh_on = 0.0
    if cloud_contrib_on == False:
        cloud_on = 0.0

    # Merge all different opacities
    for iwave in range(NWAVE):
        for ig in range(NG):
            for ilayer in range(NLAYER):
                tau_total_w_g_l[iwave,ig,ilayer] = rayleigh_on * tau_rayleigh[iwave,ilayer] \
                    + gas_on * tau_gas[iwave,ig,ilayer] \
                    + cia_on * tau_cia[iwave,ilayer] \
                    + cloud_on * tau_cloud[iwave,ilayer]
                
    ## NIGHTSIDE
    tau_total_w_g_l_n = np.zeros((NWAVE,NG,NLAYER))
    # Collision induced absorptioin optical path (NWAVE x NLAYER)
    P_grid_n = P_grid
    T_grid_n = T_grid

    tau_cia = calc_tau_cia(wave_grid=wave_grid,K_CIA=k_cia,ISPACE=1,
        ID=ID,TOTAM=U_layer_n,T_layer=T_layer_n,P_layer=P_layer_n,VMR_layer=VMR_layer_n,
        DELH=dH,cia_nu_grid=cia_nu_grid,TEMPS=cia_T_grid,INORMAL=1,NPAIR=9, aprx_coef=aprx_coef)
    
    # Rayleigh scattering optical path (NWAVE x NLAYER)

    tau_rayleigh = calc_tau_rayleigh(wave_grid=wave_grid,U_layer=U_layer_n,fH2=h2_frac)
    # FORTRAN straight transcript
    tau_gas = calc_tau_gas(k_gas_w_g_p_t, P_layer_n, T_layer_n, VMR_layer_n, U_layer_n,
        P_grid_n, T_grid_n, del_g)
    
    # Cloud optical path (NWAVE x NLAYER)
    if hazemult_n is not None:
        tau_cloud = calc_tau_cloud(wave_grid, P_layer_n, T_layer_n, mmw_n, Ptop_n, power_n, hazemult_n)
    else:
        tau_cloud = calc_tau_cloud(wave_grid, P_layer_n, T_layer_n, mmw_n, Ptop_n, power_n)


    cia_on = 1.0 # CO2 CIA?
    gas_on = 1.0
    rayleigh_on = 1.0 # Need CO2 Rayleigh scattering
    cloud_on = 1.0

    if cia_contrib_on == False:
        cia_on = 0.0
    if gas_contrib_on == False:
        gas_on = 0.0
    if rayleigh_contrib_on == False:
        rayleigh_on = 0.0
    if cloud_contrib_on == False:
        cloud_on = 0.0

    # Merge all different opacities
    for iwave in range(NWAVE):
        for ig in range(NG):
            for ilayer in range(NLAYER):
                tau_total_w_g_l_n[iwave,ig,ilayer] = rayleigh_on * tau_rayleigh[iwave,ilayer] \
                    + gas_on * tau_gas[iwave,ig,ilayer] \
                    + cia_on * tau_cia[iwave,ilayer] \
                    + cloud_on * tau_cloud[iwave,ilayer]

    # Create list of atmospheric paths for transit geometry
    paths = []
    NPATH = NLAYER
    for ipath in range(NPATH):
        path = list(range(NPATH-1, NPATH-ipath-2, -1))\
        + list(range(-NPATH+ipath+1, -NPATH, -1))
        paths.append(path)
    paths = paths[::-1]


    y1 = np.zeros((NPATH, NWAVE))
    area = np.zeros(NWAVE)
    for ipath in range(NPATH):

        tau_cum_w_g = np.zeros((NWAVE,NG))
        h_pathbase = R_plt + H_base[ipath]

        for iwave in range(NWAVE):
            for ig in range(NG):
                for ilayer, layer_id in enumerate(paths[ipath]):
                    # Scale to the line-of-sight opacities
                    if layer_id >= 0:
                        tau_cum_w_g[iwave,ig] =  tau_cum_w_g[iwave,ig] + \
                        tau_total_w_g_l[iwave,ig,layer_id]*ScalingFactor[ipath, ilayer]
                    else:
                        layer_id = -layer_id
                        tau_cum_w_g[iwave,ig] =  tau_cum_w_g[iwave,ig] + \
                        tau_total_w_g_l_n[iwave,ig,layer_id]*ScalingFactor[ipath, ilayer]
                  
        # print(tau_cum_w_g)
        tr_w_g = 1.0 - np.exp(-tau_cum_w_g)
        tr_w = np.zeros(NWAVE)

        for iwave in range(NWAVE):
            for ig in range(NG):
                tr_w[iwave] = tr_w[iwave] + tr_w_g[iwave, ig] * del_g[ig]
            

        # if np.isnan(tr_w).any():
        #     print("Error: NaN in optical depth")
        #     break
        
        y1[ipath] = 2 * np.pi * h_pathbase * tr_w
        
    for ipath in range(NPATH-1):
        area = area + 0.5 * (y1[ipath+1] + y1[ipath]) * dH[ipath]
    
    area_star = np.pi * R_star**2
    area_plt_bottom = np.pi * (R_plt + H_base[0])**2
    td_percent = 100.0 * (area + area_plt_bottom)/area_star

    return td_percent

@jit(nopython=True)
def calc_contrib(wave_grid, H_layer, H_base, U_layer, P_layer,P_base, T_layer, VMR_layer,
    k_gas_w_g_p_t, P_grid, T_grid, del_g, ScalingFactor, R_plt, R_star, solspec,
    k_cia, ID, cia_nu_grid, cia_T_grid, dH, mmw, Ptop, power, cia_contrib_on, gas_contrib_on,
    rayleigh_contrib_on, cloud_contrib_on, aprx_coef, jlayer, hazemult):
    """
    Calculate transmission spectrum using the correlated-k method.

    Parameters
    ----------
    wave_grid(NWAVE) : ndarray
        Wavelengths (um) grid for calculating spectra.
    H_layer(NLAYER) : ndarray
        Height of each layer.
        Unit: m
    U_layer(NLAYER) : ndarray
        Surface density of gas particles in each layer.
        Unit: no. of particle/m^2
    P_layer(NLAYER) : ndarray
        Atmospheric pressure grid.
        Unit: Pa
    T_layer(NLAYER) : ndarray
        Atmospheric temperature grid.
        Unit: K
    VMR_layer(NLAYER,NGAS) : ndarray
        Array of volume mixing ratios for NGAS.
        Has dimensioin: NLAYER x NGAS
    k_gas_w_g_p_t(NGAS,NWAVE,NG,NPRESSK,NTEMPK) : ndarray
        k-coefficients.
        Has dimension: NGAS x NWAVE x NG x NPRESSK x NTEMPK.
    P_grid(NPRESSK) : ndarray
        Pressure grid on which the k-coeff's are pre-computed.
        We want SI unit (Pa) here.
    T_grid(NTEMPK) : ndarray
        Temperature grid on which the k-coeffs are pre-computed. In Kelvin
    del_g : ndarray
        Quadrature weights of the g-ordinates.
    ScalingFactor(NLAYER) : ndarray
        Scale stuff to line of sight
    R_plt : real
        Planetary radius
        Unit: m
    solspec : ndarray
        Stellar spectra, used when the unit of the output is in fraction
        of stellar irradiance.

        Stellar flux at planet's distance (W cm-2 um-1 or W cm-2 (cm-1)-1)

    Returns
    -------
    spectrum : ndarray
        Output spectrum (W cm-2 um-1 sr-1)
    """
    NGAS, NWAVE, NG, NPRESS, NTEMP = k_gas_w_g_p_t.shape
    NLAYER = len(P_layer)

    # Initiate arrays to record total optical paths
    tau_total_w_g_l = np.zeros((NWAVE,NG,NLAYER))
    # Collision induced absorptioin optical path (NWAVE x NLAYER)
    tau_cia = calc_tau_cia(wave_grid=wave_grid,K_CIA=k_cia,ISPACE=1,
        ID=ID,TOTAM=U_layer,T_layer=T_layer,P_layer=P_layer,VMR_layer=VMR_layer,
        DELH=dH,cia_nu_grid=cia_nu_grid,TEMPS=cia_T_grid,INORMAL=1,NPAIR=9, aprx_coef=aprx_coef)
    
    # Rayleigh scattering optical path (NWAVE x NLAYER)
    tau_rayleigh = calc_tau_rayleigh(wave_grid=wave_grid,U_layer=U_layer)

    # FORTRAN straight transcript
    tau_gas = calc_tau_gas(k_gas_w_g_p_t, P_layer, T_layer, VMR_layer, U_layer,
        P_grid, T_grid, del_g)
    
    # Cloud optical path (NWAVE x NLAYER)
    if hazemult is not None:
        tau_cloud = calc_tau_cloud(wave_grid, P_layer, T_layer, mmw, Ptop, power, hazemult)
    else:
        tau_cloud = calc_tau_cloud(wave_grid, P_layer, T_layer, mmw, Ptop, power)
    # Values 1.0 if on, 0.0 if off
    cia_on = 1.0 # CO2 CIA?
    gas_on = 1.0
    rayleigh_on = 1.0 # Need CO2 Rayleigh scattering
    cloud_on = 1.0

    # Merge all different opacities
    for iwave in range(NWAVE):
        for ig in range(NG):
            for ilayer in range(NLAYER):
                if jlayer != ilayer:
                    tau_total_w_g_l[iwave,ig,ilayer] = rayleigh_on * tau_rayleigh[iwave,ilayer] \
                        + gas_on * tau_gas[iwave,ig,ilayer] \
                        + cia_on * tau_cia[iwave,ilayer] \
                        + cloud_on * tau_cloud[iwave,ilayer]

    # Create list of atmospheric paths for transit geometry
    paths = []
    NPATH = NLAYER
    for ipath in range(NPATH):
        path = list(range(NPATH-1, NPATH-ipath-2, -1))\
        + list(range(NPATH-ipath-1, NPATH, 1))
        paths.append(path)
    paths = paths[::-1]


    y1 = np.zeros((NPATH, NWAVE))
    area = np.zeros(NWAVE)
    for ipath in range(NPATH):

        tau_cum_w_g = np.zeros((NWAVE,NG))
        h_pathbase = R_plt + H_base[ipath]

        for iwave in range(NWAVE):
            for ig in range(NG):
                for ilayer, layer_id in enumerate(paths[ipath]):
                    # Scale to the line-of-sight opacities
                    tau_cum_w_g[iwave,ig] =  tau_cum_w_g[iwave,ig] + \
                    tau_total_w_g_l[iwave,ig,layer_id]*ScalingFactor[ipath, ilayer]
                  
        # print(tau_cum_w_g)
        tr_w_g = 1.0 - np.exp(-tau_cum_w_g)
        tr_w = np.zeros(NWAVE)

        for iwave in range(NWAVE):
            for ig in range(NG):
                tr_w[iwave] = tr_w[iwave] + tr_w_g[iwave, ig] * del_g[ig]
            

        # if np.isnan(tr_w).any():
        #     print("Error: NaN in optical depth")
        #     break
        
        y1[ipath] = 2 * np.pi * h_pathbase * tr_w
        
    for ipath in range(NPATH-1):
        area = area + 0.5 * (y1[ipath+1] + y1[ipath]) * dH[ipath]
    
    area_star = np.pi * R_star**2
    area_plt_bottom = np.pi * (R_plt + H_base[0])**2
    td_percent = 100.0 * (area + area_plt_bottom)/area_star

    return td_percent


@jit(nopython=True)
def calc_radiance(wave_grid, U_layer, P_layer, T_layer, VMR_layer, mmw, Ptop,
    k_gas_w_g_p_t, P_grid, T_grid, del_g, ScalingFactor, R_plt, solspec,
    k_cia, ID, cia_nu_grid, cia_T_grid, dH, A_layer, phase_func, angles):
    """
    Calculate emission spectrum using the correlated-k method.

    Parameters
    ----------
    wave_grid(NWAVE) : ndarray
        Wavelengths (um) grid for calculating spectra.
    U_layer(NLAYER) : ndarray
        Surface density of gas particles in each layer.
        Unit: no. of particle/m^2
    P_layer(NLAYER) : ndarray
        Atmospheric pressure grid.
        Unit: Pa
    T_layer(NLAYER) : ndarray
        Atmospheric temperature grid.
        Unit: K
    VMR_layer(NLAYER,NGAS) : ndarray
        Array of volume mixing ratios for NGAS.
        Has dimensioin: NLAYER x NGAS
    A_layer(NLAYER,NGAS) : ndarray
        Array of aerosol opacities for NMODE.
        Has dimension: NLAYER x NMODE
    phase_func(NMODE,NWAVE,6) : ndarray
        Contains fitted phase function parameters and 
        cross-sections for NMODE
    k_gas_w_g_p_t(NGAS,NWAVE,NG,NPRESSK,NTEMPK) : ndarray
        k-coefficients.
        Has dimension: NGAS x NWAVE x NG x NPRESSK x NTEMPK.
    P_grid(NPRESSK) : ndarray
        Pressure grid on which the k-coeff's are pre-computed.
        We want SI unit (Pa) here.
    T_grid(NTEMPK) : ndarray
        Temperature grid on which the k-coeffs are pre-computed. In Kelvin
    del_g : ndarray
        Quadrature weights of the g-ordinates.
    ScalingFactor(NLAYER) : ndarray
        Scale stuff to line of sight
    R_plt : real
        Planetary radius
        Unit: m
    solspec : ndarray
        Stellar spectra, used when the unit of the output is in fraction
        of stellar irradiance.
        Stellar flux at planet's distance (W cm-2 um-1 or W cm-2 (cm-1)-1)

    Returns
    -------
    spectrum : ndarray
        Output spectrum (W cm-2 um-1 sr-1)
    """
    # Reorder atmospheric layers from top to bottom
    ScalingFactor = ScalingFactor[::-1]
    P_layer = P_layer[::-1] # layer pressures (Pa)
    T_layer = T_layer[::-1] # layer temperatures (K)
    U_layer = U_layer[::-1] # layer absorber amounts (no./m^2)
    VMR_layer = VMR_layer[::-1,:] # layer volume mixing ratios
    A_layer = A_layer[::-1,:] # aerosol 
    dH = dH[::-1] # lengths of each layer

    MULTIPLE_SCATTERING_CONDITION = np.any(A_layer > 0)
    
    # Record array dimensions
    NGAS, NWAVE, NG, NPRESS, NTEMP = k_gas_w_g_p_t.shape
    NLAYER = len(P_layer)
    NMODE = A_layer.shape[1]
        
    if phase_func is None:
        phase_func = np.zeros((NMODE,NWAVE,6))
        
    # Initiate arrays to record total optical paths
    tau_total_w_g_l = np.zeros((NWAVE,NG,NLAYER))
    omegas = np.zeros((NWAVE,NG,NLAYER))

    # Collision induced absorptioin optical path (NWAVE x NLAYER)
    tau_cia = calc_tau_cia(wave_grid=wave_grid,K_CIA=k_cia,ISPACE=1,
        ID=ID,TOTAM=U_layer,T_layer=T_layer,P_layer=P_layer,VMR_layer=VMR_layer,
        DELH=dH,cia_nu_grid=cia_nu_grid,TEMPS=cia_T_grid,INORMAL=1,NPAIR=9)

    # Rayleigh scattering optical path (NWAVE x NLAYER)
    tau_rayleigh = calc_tau_rayleigh(wave_grid=wave_grid,U_layer=U_layer)

    # Dust scattering optical path (NWAVE x NLAYER)
    if phase_func is not None:
        tau_dust_ext, tau_dust_scat, lfrac = calc_tau_dust(A_layer, phase_func)
    else:
        tau_dust_ext = np.zeros((NWAVE,NLAYER))
        tau_dust_scat = np.zeros((NWAVE,NLAYER))
        lfrac = np.zeros((NWAVE,NMODE, NLAYER)) 
                    
    # FORTRAN straight transcript
    tau_gas = calc_tau_gas(k_gas_w_g_p_t, P_layer, T_layer, VMR_layer, U_layer,
        P_grid, T_grid, del_g)

    # Merge all different opacities
    for iwave in range(NWAVE):
        for ilayer in range(NLAYER):
            for ig in range(NG):
                tau_total_w_g_l[iwave,ig,ilayer] = tau_gas[iwave,ig,ilayer] \
                    + tau_cia[iwave,ilayer] \
                    + tau_dust_ext[iwave,ilayer] \
                    + tau_rayleigh[iwave,ilayer]
                
                omegas[iwave,ig,ilayer] = (tau_rayleigh[iwave,ilayer]\
                                           + tau_dust_scat[iwave,ilayer])\
                                            /tau_total_w_g_l[iwave,ig,ilayer]

    # Scale to the line-of-sight opacities
    tau_total_w_g_l *=  ScalingFactor
    tau_rayleigh *= ScalingFactor

    # Defining the units of the output spectrum / divide by stellar spectrum
    # radextra = np.sum(dH[:-1])
    # xfac = np.pi*4.*np.pi*((R_plt+radextra)*1e2)**2./solspec[:]
    xfac = np.pi*4.*np.pi*(R_plt*1e2)**2./solspec[:]

    # Calculating atmospheric gases contribution
    tau_cum_w_g = np.zeros((NWAVE,NG))
    tr_old_w_g = np.ones((NWAVE,NG))
    spec_w_g = np.zeros((NWAVE,NG))

    
    if not MULTIPLE_SCATTERING_CONDITION:
        for ilayer in range(NLAYER):
            for iwave in range(NWAVE):
                for ig in range(NG):
                    tau_cum_w_g[iwave,ig] \
                        =  tau_total_w_g_l[iwave,ig,ilayer] + tau_cum_w_g[iwave,ig]
            tr_w_g = np.exp(-tau_cum_w_g[:,:]) # transmission function
            bb = calc_planck(wave_grid[:], T_layer[ilayer]) # blackbody function

            for iwave in range(NWAVE):
                for ig in range(NG):
                    spec_w_g[iwave,ig] = spec_w_g[iwave,ig] \
                        + (tr_old_w_g[iwave,ig]-tr_w_g[iwave,ig])\
                        * bb[iwave] * xfac[iwave]

            tr_old_w_g = tr_w_g

        # Add radiation from below deepest layer
        radground = calc_planck(wave_grid,T_layer[-1])
        for ig in range(NG):
            spec_w_g[:,ig] = spec_w_g[:,ig] \
                + tr_old_w_g[:,ig] * radground[:] * xfac[:]



    else:
        radground = calc_planck(wave_grid,T_layer[-1]) * np.ones((5,NWAVE))
        planck = np.zeros((NWAVE,NLAYER))
        for ilayer in range(NLAYER):
            planck[:,ilayer] = calc_planck(wave_grid,T_layer[ilayer])

        
        
        for ig in range(NG):
            
            spec_w_g[:,ig] = calc_spectrum_scloud11(wave_grid, phase_func,radground,
                                                    tau_total_w_g_l[:,ig],tau_rayleigh,
                                                    omegas[:,ig],lfrac,planck,angles) * xfac
    # Integrate over g-ordinates
    spectrum = np.zeros((NWAVE))
    for iwave in range(NWAVE):
        for ig in range(NG):
            spectrum[iwave] += spec_w_g[iwave,ig] * del_g[ig]
    return spectrum


@jit(nopython=True)
def calc_weighting(wave_grid, U_layer, P_layer, T_layer, VMR_layer,
    k_gas_w_g_p_t, P_grid, T_grid, del_g, ScalingFactor, R_plt, solspec,
    k_cia, ID, cia_nu_grid, cia_T_grid, dH):
    """
    Calculate emission spectrum using the correlated-k method.

    Parameters
    ----------
    wave_grid(NWAVE) : ndarray
        Wavelengths (um) grid for calculating spectra.
    U_layer(NLAYER) : ndarray
        Surface density of gas particles in each layer.
        Unit: no. of particle/m^2
    P_layer(NLAYER) : ndarray
        Atmospheric pressure grid.
        Unit: Pa
    T_layer(NLAYER) : ndarray
        Atmospheric temperature grid.
        Unit: K
    VMR_layer(NLAYER,NGAS) : ndarray
        Array of volume mixing ratios for NGAS.
        Has dimensioin: NLAYER x NGAS
    k_gas_w_g_p_t(NGAS,NWAVE,NG,NPRESSK,NTEMPK) : ndarray
        k-coefficients.
        Has dimension: NGAS x NWAVE x NG x NPRESSK x NTEMPK.
    P_grid(NPRESSK) : ndarray
        Pressure grid on which the k-coeff's are pre-computed.
        We want SI unit (Pa) here.
    T_grid(NTEMPK) : ndarray
        Temperature grid on which the k-coeffs are pre-computed. In Kelvin
    del_g : ndarray
        Quadrature weights of the g-ordinates.
    ScalingFactor(NLAYER) : ndarray
        Scale stuff to line of sight
    R_plt : real
        Planetary radius
        Unit: m
    solspec : ndarray
        Stellar spectra, used when the unit of the output is in fraction
        of stellar irradiance.

        Stellar flux at planet's distance (W cm-2 um-1 or W cm-2 (cm-1)-1)

    Returns
    -------
    spectrum : ndarray
        Output spectrum (W cm-2 um-1 sr-1)
    """
    # Reorder atmospheric layers from top to bottom
    ScalingFactor = ScalingFactor[::-1]
    P_layer = P_layer[::-1] # layer pressures (Pa)
    T_layer = T_layer[::-1] # layer temperatures (K)
    U_layer = U_layer[::-1] # layer absorber amounts (no./m^2)
    VMR_layer = VMR_layer[::-1,:] # layer volume mixing ratios
    dH = dH[::-1] # lengths of each layer

    # Record array dimensions
    NGAS, NWAVE, NG, NPRESS, NTEMP = k_gas_w_g_p_t.shape
    NLAYER = len(P_layer)

    # Initiate arrays to record total optical paths
    tau_total_w_g_l = np.zeros((NWAVE,NG,NLAYER))

    # Collision induced absorptioin optical path (NWAVE x NLAYER)
    tau_cia = calc_tau_cia(wave_grid=wave_grid,K_CIA=k_cia,ISPACE=1,
        ID=ID,TOTAM=U_layer,T_layer=T_layer,P_layer=P_layer,VMR_layer=VMR_layer,
        DELH=dH,cia_nu_grid=cia_nu_grid,TEMPS=cia_T_grid,INORMAL=1,NPAIR=9)

    # Rayleigh scattering optical path (NWAVE x NLAYER)
    tau_rayleigh = calc_tau_rayleigh(wave_grid=wave_grid,U_layer=U_layer)

    # Dust scattering optical path (NWAVE x NLAYER)
    tau_dust = np.zeros((NWAVE,NLAYER))

    # FORTRAN straight transcript
    tau_gas = calc_tau_gas(k_gas_w_g_p_t, P_layer, T_layer, VMR_layer, U_layer,
        P_grid, T_grid, del_g)

    # Merge all different opacities
    for iwave in range(NWAVE):
        for ilayer in range(NLAYER):
            for ig in range(NG):
                tau_total_w_g_l[iwave,ig,ilayer] = tau_gas[iwave,ig,ilayer] \
                    + tau_cia[iwave,ilayer] \
                    + tau_dust[iwave,ilayer] \
                    + tau_rayleigh[iwave,ilayer]

    # Calculating atmospheric gases contribution
    tau_cum_w_g = np.zeros((NWAVE,NG))
    tr_old_w_g = np.ones((NWAVE,NG))
    weighting_l_w_g = np.zeros((NLAYER,NWAVE,NG))
    for ilayer in range(NLAYER):
        for iwave in range(NWAVE):
            for ig in range(NG):
                tau_cum_w_g[iwave,ig] \
                    =  tau_total_w_g_l[iwave,ig,ilayer] + tau_cum_w_g[iwave,ig]
        tr_w_g = np.exp(-tau_cum_w_g[:,:]) # transmission function
        bb = calc_planck(wave_grid[:], T_layer[ilayer]) # blackbody function

        for iwave in range(NWAVE):
            for ig in range(NG):
                weighting_l_w_g[ilayer,iwave,ig] = (tr_old_w_g[iwave,ig]-tr_w_g[iwave,ig])
        tr_old_w_g = tr_w_g


    weighting_l_w = np.zeros((NLAYER,NWAVE))

    # Integrate over g-ordinates
    for iwave in range(NWAVE):
        for ig in range(NG):
            weighting_l_w[:,iwave] += weighting_l_w_g[:,iwave,ig] * del_g[ig]

    # Normalise
    for iwave in range(NWAVE):
        weighting_l_w[:,iwave] = weighting_l_w[:,iwave]/sum(weighting_l_w[:,iwave])

    # Invert pressure coordinate
    for iwave in range(NWAVE):
        weighting_l_w[:,iwave] = weighting_l_w[::-1,iwave]

    return weighting_l_w

@jit(nopython=True)
def calc_weighting_transm(wave_grid, H_layer,H_base, U_layer, P_layer, P_base, T_layer, VMR_layer,
    k_gas_w_g_p_t, P_grid, T_grid, del_g, ScalingFactor, R_plt, R_star,solspec,
    k_cia, ID, cia_nu_grid, cia_T_grid, dH, mmw, Ptop, power, aprx_coef, hazemult):
    """
    Calculate transm spectrum weighing.

    Parameters
    ----------
    wave_grid(NWAVE) : ndarray
        Wavelengths (um) grid for calculating spectra.
    U_layer(NLAYER) : ndarray
        Surface density of gas particles in each layer.
        Unit: no. of particle/m^2
    P_layer(NLAYER) : ndarray
        Atmospheric pressure grid.
        Unit: Pa
    T_layer(NLAYER) : ndarray
        Atmospheric temperature grid.
        Unit: K
    VMR_layer(NLAYER,NGAS) : ndarray
        Array of volume mixing ratios for NGAS.
        Has dimensioin: NLAYER x NGAS
    k_gas_w_g_p_t(NGAS,NWAVE,NG,NPRESSK,NTEMPK) : ndarray
        k-coefficients.
        Has dimension: NGAS x NWAVE x NG x NPRESSK x NTEMPK.
    P_grid(NPRESSK) : ndarray
        Pressure grid on which the k-coeff's are pre-computed.
        We want SI unit (Pa) here.
    T_grid(NTEMPK) : ndarray
        Temperature grid on which the k-coeffs are pre-computed. In Kelvin
    del_g : ndarray
        Quadrature weights of the g-ordinates.
    ScalingFactor(NLAYER) : ndarray
        Scale stuff to line of sight
    R_plt : real
        Planetary radius
        Unit: m
    solspec : ndarray
        Stellar spectra, used when the unit of the output is in fraction
        of stellar irradiance.

        Stellar flux at planet's distance (W cm-2 um-1 or W cm-2 (cm-1)-1)

    Returns
    -------
    spectrum : ndarray
        Output spectrum (W cm-2 um-1 sr-1)
    """
    # Reorder atmospheric layers from top to bottom
    # ScalingFactor = ScalingFactor[::-1]
    # P_layer = P_layer[::-1] # layer pressures (Pa)
    # T_layer = T_layer[::-1] # layer temperatures (K)
    # U_layer = U_layer[::-1] # layer absorber amounts (no./m^2)
    # VMR_layer = VMR_layer[::-1,:] # layer volume mixing ratios
    # dH = dH[::-1] # lengths of each layer

    # Record array dimensions
    NGAS, NWAVE, NG, NPRESS, NTEMP = k_gas_w_g_p_t.shape
    NLAYER = len(P_layer)

    # Initiate arrays to record total optical paths
    tau_total_w_g_l = np.zeros((NWAVE,NG,NLAYER))

    # Collision induced absorptioin optical path (NWAVE x NLAYER)
    tau_cia = calc_tau_cia(wave_grid=wave_grid,K_CIA=k_cia,ISPACE=1,
        ID=ID,TOTAM=U_layer,T_layer=T_layer,P_layer=P_layer,VMR_layer=VMR_layer,
        DELH=dH,cia_nu_grid=cia_nu_grid,TEMPS=cia_T_grid,INORMAL=1,NPAIR=9, aprx_coef=aprx_coef)
    
    # Rayleigh scattering optical path (NWAVE x NLAYER)
    tau_rayleigh = calc_tau_rayleigh(wave_grid=wave_grid,U_layer=U_layer)

    # FORTRAN straight transcript
    tau_gas = calc_tau_gas(k_gas_w_g_p_t, P_layer, T_layer, VMR_layer, U_layer,
        P_grid, T_grid, del_g)
    
    # Cloud optical path (NWAVE x NLAYER)
    if hazemult is not None:
        tau_cloud = calc_tau_cloud(wave_grid, P_layer, T_layer, mmw, Ptop, power, hazemult)
    else:
        tau_cloud = calc_tau_cloud(wave_grid, P_layer, T_layer, mmw, Ptop, power)

    # FORTRAN straight transcript
    tau_gas = calc_tau_gas(k_gas_w_g_p_t, P_layer, T_layer, VMR_layer, U_layer,
        P_grid, T_grid, del_g)

    # Merge all different opacities
    for iwave in range(NWAVE):
        for ilayer in range(NLAYER):
            for ig in range(NG):
                tau_total_w_g_l[iwave,ig,ilayer] = tau_gas[iwave,ig,ilayer] \
                    + tau_cia[iwave,ilayer] \
                    + tau_cloud[iwave,ilayer] \
                    + tau_rayleigh[iwave,ilayer]

    # Integrate over
    paths = []
    NPATH = NLAYER
    for ipath in range(NPATH):
        path = list(range(NPATH-1, NPATH-ipath-2, -1))\
        + list(range(NPATH-ipath-1, NPATH, 1))
        paths.append(path)
    paths = paths[::-1]

    
    # t0 = time.time()

    weighting_p_w_g = np.zeros((NPATH,NWAVE,NG))
    y1 = np.zeros((NPATH,NWAVE))
    area = np.zeros((NPATH,NWAVE))
    td_percent = np.zeros((NPATH,NWAVE))

    for ipath in range(NPATH):
        tau_cum_w_g = np.zeros((NWAVE,NG))
        h_pathbase = R_plt + H_base[ipath]

        for ilayer, layer_id in enumerate(paths[ipath]):
            for iwave in range(NWAVE):
                for ig in range(NG):
                    tau_cum_w_g[iwave,ig] =  tau_cum_w_g[iwave,ig] + \
                    tau_total_w_g_l[iwave,ig,layer_id]*ScalingFactor[ipath, ilayer]

        tr_w_g = 1.0 - np.exp(-tau_cum_w_g)
        tr_w = np.zeros(NWAVE)
        for iwave in range(NWAVE):
            for ig in range(NG):
                tr_w[iwave] = tr_w[iwave] + tr_w_g[iwave, ig] * del_g[ig]
        y1[ipath] = 2 * np.pi * h_pathbase * tr_w
        
        area[ipath] = y1[ipath] * dH[ipath]

        area_star = np.pi * R_star**2
        area_plt_bottom = np.pi * (R_plt + H_base[0])**2
        td_percent[ipath] = 100.0 * (area[ipath] + area_plt_bottom)/area_star
                
    #     # look into masked arrays
    #     tr_w_g = 1.0 - np.exp(-tau_cum_w_g)
    #     for iwave in range(NWAVE):
    #         for ig in range(NG):
    #             weighting_p_w_g[ipath,iwave,ig] = tau_cum_w_g[iwave,ig] * dH[layer_id]
                

          
    #     # look into masked arrays
    #     tr_w_g = 1.0 - np.exp(-tau_cum_w_g)
    #     tr_w = np.zeros(NWAVE)
    #     for iwave in range(NWAVE):
    #         for ig in range(NG):
    #             tr_w[iwave] = tr_w[iwave] + tr_w_g[iwave, ig] * del_g[ig]
    #     y1[ipath] = 2 * np.pi * h_pathbase * tr_w
        
    # for ipath in range(NPATH-1):
    #     area = area + 0.5 * (y1[ipath+1] + y1[ipath]) * dH[ipath]

    # area_star = np.pi * R_star**2
    # area_plt_bottom = np.pi * (R_plt + H_base[0])**2
    # td_percent = 100.0 * (area + area_plt_bottom)/area_star

    # return wave_grid, td_percent

    return td_percent, P_layer

@jit(nopython=True)
def calc_component(wave_grid, H_layer, H_base, U_layer, P_layer,P_base, T_layer, VMR_layer,
    k_gas_w_g_p_t, P_grid, T_grid, del_g, ScalingFactor, R_plt, R_star, solspec,
    k_cia, ID, cia_nu_grid, cia_T_grid, dH, mmw, Ptop, power, cia_contrib_on, gas_contrib_on,
    rayleigh_contrib_on, cloud_contrib_on, aprx_coef, hazemult, gas_index):
    """
    Calculate transmission spectrum using the correlated-k method.

    Parameters
    ----------
    wave_grid(NWAVE) : ndarray
        Wavelengths (um) grid for calculating spectra.
    H_layer(NLAYER) : ndarray
        Height of each layer.
        Unit: m
    U_layer(NLAYER) : ndarray
        Surface density of gas particles in each layer.
        Unit: no. of particle/m^2
    P_layer(NLAYER) : ndarray
        Atmospheric pressure grid.
        Unit: Pa
    T_layer(NLAYER) : ndarray
        Atmospheric temperature grid.
        Unit: K
    VMR_layer(NLAYER,NGAS) : ndarray
        Array of volume mixing ratios for NGAS.
        Has dimensioin: NLAYER x NGAS
    k_gas_w_g_p_t(NGAS,NWAVE,NG,NPRESSK,NTEMPK) : ndarray
        k-coefficients.
        Has dimension: NGAS x NWAVE x NG x NPRESSK x NTEMPK.
    P_grid(NPRESSK) : ndarray
        Pressure grid on which the k-coeff's are pre-computed.
        We want SI unit (Pa) here.
    T_grid(NTEMPK) : ndarray
        Temperature grid on which the k-coeffs are pre-computed. In Kelvin
    del_g : ndarray
        Quadrature weights of the g-ordinates.
    ScalingFactor(NLAYER) : ndarray
        Scale stuff to line of sight
    R_plt : real
        Planetary radius
        Unit: m
    solspec : ndarray
        Stellar spectra, used when the unit of the output is in fraction
        of stellar irradiance.

        Stellar flux at planet's distance (W cm-2 um-1 or W cm-2 (cm-1)-1)

    Returns
    -------
    spectrum : ndarray
        Output spectrum (W cm-2 um-1 sr-1)
    """
    NGAS, NWAVE, NG, NPRESS, NTEMP = k_gas_w_g_p_t.shape
    NLAYER = len(P_layer)

    # Initiate arrays to record total optical paths
    tau_total_w_g_l = np.zeros((NWAVE,NG,NLAYER))
    # Collision induced absorptioin optical path (NWAVE x NLAYER)
    tau_cia = calc_tau_cia(wave_grid=wave_grid,K_CIA=k_cia,ISPACE=1,
        ID=ID,TOTAM=U_layer,T_layer=T_layer,P_layer=P_layer,VMR_layer=VMR_layer,
        DELH=dH,cia_nu_grid=cia_nu_grid,TEMPS=cia_T_grid,INORMAL=1,NPAIR=9, aprx_coef=aprx_coef)
    
    # Rayleigh scattering optical path (NWAVE x NLAYER)
    tau_rayleigh = calc_tau_rayleigh(wave_grid=wave_grid,U_layer=U_layer)

    # FORTRAN straight transcript
    tau_gas = calc_tau_gas_comp(k_gas_w_g_p_t, P_layer, T_layer, VMR_layer, U_layer,
        P_grid, T_grid, del_g, gas_index)
    
    # Cloud optical path (NWAVE x NLAYER)
    if hazemult is not None:
        tau_cloud = calc_tau_cloud(wave_grid, P_layer, T_layer, mmw, Ptop, power, hazemult)
    else:
        tau_cloud = calc_tau_cloud(wave_grid, P_layer, T_layer, mmw, Ptop, power)


    cia_on = 1.0 # CO2 CIA?
    gas_on = 1.0
    rayleigh_on = 1.0 # Need CO2 Rayleigh scattering
    cloud_on = 1.0

    if cia_contrib_on == False:
        cia_on = 0.0
    if gas_contrib_on == False:
        gas_on = 0.0
    if rayleigh_contrib_on == False:
        rayleigh_on = 0.0
    if cloud_contrib_on == False:
        cloud_on = 0.0

    # Merge all different opacities
    for iwave in range(NWAVE):
        for ig in range(NG):
            for ilayer in range(NLAYER):
                tau_total_w_g_l[iwave,ig,ilayer] = rayleigh_on * tau_rayleigh[iwave,ilayer] \
                    + gas_on * tau_gas[iwave,ig,ilayer] \
                    + cia_on * tau_cia[iwave,ilayer] \
                    + cloud_on * tau_cloud[iwave,ilayer]

    # Create list of atmospheric paths for transit geometry
    paths = []
    NPATH = NLAYER
    for ipath in range(NPATH):
        path = list(range(NPATH-1, NPATH-ipath-2, -1))\
        + list(range(NPATH-ipath-1, NPATH, 1))
        paths.append(path)
    paths = paths[::-1]


    y1 = np.zeros((NPATH, NWAVE))
    area = np.zeros(NWAVE)
    for ipath in range(NPATH):

        tau_cum_w_g = np.zeros((NWAVE,NG))
        h_pathbase = R_plt + H_base[ipath]

        for iwave in range(NWAVE):
            for ig in range(NG):
                for ilayer, layer_id in enumerate(paths[ipath]):
                    # Scale to the line-of-sight opacities
                    tau_cum_w_g[iwave,ig] =  tau_cum_w_g[iwave,ig] + \
                    tau_total_w_g_l[iwave,ig,layer_id]*ScalingFactor[ipath, ilayer]
                  
        # print(tau_cum_w_g)
        tr_w_g = 1.0 - np.exp(-tau_cum_w_g)
        tr_w = np.zeros(NWAVE)

        for iwave in range(NWAVE):
            for ig in range(NG):
                tr_w[iwave] = tr_w[iwave] + tr_w_g[iwave, ig] * del_g[ig]
            

        # if np.isnan(tr_w).any():
        #     print("Error: NaN in optical depth")
        #     break
        
        y1[ipath] = 2 * np.pi * h_pathbase * tr_w
        
    for ipath in range(NPATH-1):
        area = area + 0.5 * (y1[ipath+1] + y1[ipath]) * dH[ipath]
    
    area_star = np.pi * R_star**2
    area_plt_bottom = np.pi * (R_plt + H_base[0])**2
    td_percent = 100.0 * (area + area_plt_bottom)/area_star

    return td_percent