#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""
Interface class for running forward models.
"""
import numpy as np
from nemesispy.radtran.calc_mmw import calc_mmw
from nemesispy.radtran.read import read_kls
from nemesispy.radtran.calc_radiance import calc_radiance
from nemesispy.radtran.calc_radiance import calc_transm
from nemesispy.radtran.calc_radiance import calc_transm_dual
from nemesispy.radtran.calc_radiance import calc_contrib
from nemesispy.radtran.calc_radiance import calc_component
from nemesispy.radtran.read import read_cia
from nemesispy.radtran.calc_layer import calc_layer
from nemesispy.radtran.calc_layer import calc_layer_transm, calc_layer_transm_dual
from nemesispy.common.calc_trig import gauss_lobatto_weights
from nemesispy.common.constants import AMU,N_A
from nemesispy.common.interpolate_gcm import interp_gcm
from nemesispy.common.calc_hydrostat import calc_hydrostat, calc_hydrostat_alt, calc_hydrostat_pref_test
from nemesispy.radtran.calc_radiance import calc_weighting
from nemesispy.radtran.calc_radiance import calc_weighting_transm
from scipy.interpolate import CubicSpline
from nemesispy.radtran.scatter import makephase

import time
class ForwardModel():

    def __init__(self):
        """
        Attributes to store data that doesn't change during a retrieval
        """
        # planet and planetary system data
        self.M_plt = None
        self.R_plt = None
        self.M_star = None # currently not used
        self.R_star = None # currently not used
        self.T_star = None # currently not used
        self.semi_major_axis = None # currently not used
        self.NLAYER = None
        self.is_planet_model_set = False

        # opacity data
        self.gas_id_list = None
        self.iso_id_list = None
        self.wave_grid = None
        self.g_ord = None
        self.del_g = None
        self.k_table_P_grid = None
        self.k_table_T_grid = None
        self.k_gas_w_g_p_t = None
        self.cia_nu_grid = None
        self.cia_T_grid = None
        self.k_cia_pair_t_w = None
        self.is_opacity_data_set = False

        # phase curve debug data
        self.fov_H_model = None
        self.fake_fov_H_model = None
        fov_latitudes = None
        fov_longitudes = None
        fov_emission_angles = None
        fov_weights = None
        self.total_weight = None

        # Multiple scattering
        self.phase_func = None

    def set_planet_model(self, M_plt, R_plt, gas_id_list, iso_id_list, NLAYER,
        gas_name_list=None, solspec=None, R_star=None, T_star=None,
        semi_major_axis=None, P_plt=None):
        """
        Store the planetary system parameters
        """
        self.M_plt = M_plt
        self.R_plt = R_plt
        self.R_star = R_star
        self.P_plt = P_plt
        # self.T_star = T_star
        # self.semi_major_axis = semi_major_axis
        self.gas_name_list = gas_name_list
        self.gas_id_list = gas_id_list
        self.iso_id_list = iso_id_list
        self.NLAYER = NLAYER

        self.is_planet_model_set = True

    def set_opacity_data(self, kta_file_paths, cia_file_path,
            truncate_upper=-1,truncate_lower=-1,step=-1):
        """
        Read gas ktables and cia opacity files and store as class attributes.
        """
        k_gas_id_list, k_iso_id_list, wave_grid, g_ord, del_g, k_table_P_grid,\
            k_table_T_grid, k_gas_w_g_p_t = read_kls(kta_file_paths)
        # print('gas',k_gas_id_list)
        if truncate_upper > 0:
            wave_grid = wave_grid[:truncate_upper]
            k_gas_w_g_p_t = k_gas_w_g_p_t[:,:truncate_upper,:,:,:]
        if truncate_lower > 0:
            wave_grid = wave_grid[truncate_lower:]
            k_gas_w_g_p_t = k_gas_w_g_p_t[:,truncate_lower:,:,:,:]
        if step > 0:
            wave_grid = wave_grid[::step]
            k_gas_w_g_p_t = k_gas_w_g_p_t[:,::step,:,:,:]

        ##
        # if obs_wv is not None:
        #     wave_grid = obs_wv
        """
        Some gases (e.g. H2 and He) have no k table data so gas id lists need
        to be passed somewhere else.
        """
        self.k_gas_id_list = k_gas_id_list
        self.k_iso_id_list = k_iso_id_list
        self.wave_grid = wave_grid
        self.g_ord = g_ord
        self.del_g = del_g

        self.k_gas_w_g_p_t = k_gas_w_g_p_t # key
        # print('Loaded ktables', k_gas_w_g_p_t, k_gas_w_g_p_t.shape)
        # P_deep = 1.0
        # P_top = -7.0
        # P_fortran = np.logspace(P_deep, P_top, 22)
        # P_python = P_fortran * 101325.0
        self.k_table_P_grid = k_table_P_grid
        self.k_table_T_grid = k_table_T_grid
        # self.k_table_T_grid = np.ones(27) * 1500.0


        cia_nu_grid, cia_T_grid, k_cia_pair_t_w = read_cia(cia_file_path)
        self.cia_nu_grid = cia_nu_grid
        self.cia_T_grid = cia_T_grid
        self.k_cia_pair_t_w = k_cia_pair_t_w

        self.is_opacity_data_set = True

    def calc_weighting_function(self, P_model, T_model, VMR_model,
        path_angle=0, solspec=[]):

        NPRO = len(P_model)
        mmw = np.zeros(P_model.shape)
        for ipro in range(NPRO):
            mmw[ipro] = calc_mmw(self.gas_id_list,VMR_model[ipro,:])
        H_model = calc_hydrostat(P=P_model, T=T_model, mmw=mmw,
            M_plt=self.M_plt, R_plt=self.R_plt)
        H_layer,P_layer,T_layer,VMR_layer,U_layer,dH,scale \
            = calc_layer(
            self.R_plt, H_model, P_model, T_model, VMR_model,
            self.gas_id_list, self.NLAYER, path_angle, layer_type=1,
            H_0=0.0
            )

        if len(solspec)==0:
            solspec = np.ones(len(self.wave_grid))

        weighting_function = calc_weighting(self.wave_grid, U_layer, P_layer, T_layer,
            VMR_layer, self.k_gas_w_g_p_t, self.k_table_P_grid,
            self.k_table_T_grid, self.del_g, ScalingFactor=scale,
            R_plt=self.R_plt, solspec=solspec, k_cia=self.k_cia_pair_t_w,
            ID=self.gas_id_list,cia_nu_grid=self.cia_nu_grid,
            cia_T_grid=self.cia_T_grid, dH=dH)

        return weighting_function

    def calc_weighting_function_transm(self, P_model, T_model, VMR_model,
        path_angle, Pref=None, Ptop=None, power=None, solspec=[], ray_on=True,\
              hazemult =None, aprx_coef=None, set_mmw=None):

        NPRO = len(P_model)
        mmw = np.zeros(P_model.shape)

        for ipro in range(NPRO):
            mmw[ipro] = calc_mmw(self.gas_id_list,VMR_model[ipro,:])
        
        if set_mmw is not None:
            mmw = np.ones(NPRO) * set_mmw
        
        if Pref is None:
            H_model = calc_hydrostat(P=P_model, T=T_model, mmw=mmw,
                M_plt=self.M_plt, R_plt=self.R_plt)
        else:
            H_model = calc_hydrostat_pref_test(P=P_model, T=T_model, mmw=mmw,
                M_plt=self.M_plt, R_plt=self.R_plt, Pref=Pref)
            
        H_layer,H_base,P_layer,P_base,T_layer,VMR_layer,U_layer,dH,scale \
            = calc_layer_transm(
            self.R_plt, H_model, P_model, T_model, VMR_model,
            self.gas_id_list, self.NLAYER, path_angle, layer_type=1,
            H_0=0.0
            )

        if len(solspec)==0:
            solspec = np.ones(len(self.wave_grid))

        weighting_function = calc_weighting_transm(self.wave_grid, H_layer,H_base, U_layer, P_layer, P_base, T_layer,
            VMR_layer, self.k_gas_w_g_p_t, self.k_table_P_grid,
            self.k_table_T_grid, self.del_g, ScalingFactor=scale,
            R_plt=self.R_plt, R_star=self.R_star,solspec=solspec, k_cia=self.k_cia_pair_t_w,
            ID=self.gas_id_list,cia_nu_grid=self.cia_nu_grid,
            cia_T_grid=self.cia_T_grid, dH=dH, mmw=mmw, Ptop=Ptop, power=power, aprx_coef=aprx_coef, hazemult=hazemult)

        return weighting_function

    def calc_point_spectrum(self, H_model, P_model, T_model, VMR_model, Ptop,
        angles = np.array([0.,0.,0.]), solspec=[], A_model=None):
        """
        Calculate average layer properties from model inputs,
        then compute the spectrum at a single point on the disc.
        """
        path_angle = angles[1]
        
        NPRO = len(P_model)
        mmw = np.zeros(P_model.shape)
        for ipro in range(NPRO):
            mmw[ipro] = calc_mmw(self.gas_id_list,VMR_model[ipro,:])
        
        if A_model is not None:
            for imode in range(A_model.shape[1]):
                A_model[:,imode] = A_model[:,imode] * mmw / N_A / AMU * 1e-4

        
        H_layer,P_layer,T_layer,VMR_layer,U_layer,A_layer,dH,scale \
            = calc_layer(
            self.R_plt, H_model, P_model, T_model, VMR_model,
            self.gas_id_list, self.NLAYER, path_angle, layer_type=1,
            H_0=0.0, A_model=A_model
            )

        
        if len(solspec)==0:
            solspec = np.ones(len(self.wave_grid))
            
        point_spectrum = calc_radiance(self.wave_grid, U_layer, P_layer, T_layer,
            VMR_layer, mmw, Ptop, self.k_gas_w_g_p_t, self.k_table_P_grid,
            self.k_table_T_grid, self.del_g, ScalingFactor=scale,
            R_plt=self.R_plt, solspec=solspec, k_cia=self.k_cia_pair_t_w,
            ID=self.gas_id_list,cia_nu_grid=self.cia_nu_grid,
            cia_T_grid=self.cia_T_grid, dH=dH, A_layer=A_layer, phase_func=self.phase_func, angles=angles)
        return point_spectrum
    
    def calc_transm_spectrum(self, P_model, T_model, VMR_model,
        path_angle, Pref=None, Ptop=None, power=None, solspec=[], cia_contrib_on=True, gas_contrib_on=True,
        rayleigh_contrib_on=True, cloud_contrib_on=True, hazemult=None, aprx_coef=None, set_mmw=None, h2_frac=None,
        layer_type=1, dual=False, T_model_n=None, VMR_model_n=None, Ptop_n=None, power_n=None, hazemult_n=None, 
        cia_contrib_on_n=True, gas_contrib_on_n=True, spin=False):
        """
        Calculate average layer properties from model inputs,
        then compute the transmission spectrum.
        Uses the hydrostatic balance equation to calculate layer height.
        """

        NPRO = len(P_model)
        mmw = np.zeros(P_model.shape)
        for ipro in range(NPRO):
            mmw[ipro] = calc_mmw(self.gas_id_list,VMR_model[ipro,:])
        
        if set_mmw is not None:
            mmw = np.ones(NPRO) * set_mmw
        
        if Pref is None:
            H_model = calc_hydrostat(P=P_model, T=T_model, mmw=mmw,
                M_plt=self.M_plt, R_plt=self.R_plt, P_plt=self.P_plt, spin=spin)
        else:
            H_model = calc_hydrostat_pref_test(P=P_model, T=T_model, mmw=mmw,
                M_plt=self.M_plt, R_plt=self.R_plt, Pref=Pref)
                        
        if H_model[0] == -999:
            print("Hydrostatic balance failed")
            return -999*np.ones(len(self.wave_grid))
                    

        if len(solspec)==0:
            solspec = np.ones(len(self.wave_grid))

        if dual:
            mmw_n = np.zeros(P_model.shape)
            for ipro in range(NPRO):
                mmw_n[ipro] = calc_mmw(self.gas_id_list,VMR_model_n[ipro,:])
            P_model = calc_hydrostat_alt(H=H_model, T=T_model, mmw=mmw,
                                         M_plt=self.M_plt, R_plt=self.R_plt)
            P_model_n = calc_hydrostat_alt(H=H_model, T=T_model_n, mmw=mmw_n,
                                            M_plt=self.M_plt, R_plt=self.R_plt)

            H_layer,H_base,P_layer,P_base,T_layer,VMR_layer,U_layer,U_layer_n,dH,scale \
                = calc_layer_transm_dual(
                self.R_plt, H_model, P_model, P_model_n, T_model, T_model_n, VMR_model,
                self.gas_id_list, self.NLAYER, path_angle, layer_type=1,
                H_0=np.min(H_model))
            
            transm_spectrum = calc_transm_dual(self.wave_grid, H_layer,H_base, U_layer, U_layer_n, P_model, P_model_n, P_base,T_model,
                                               T_model_n,
                VMR_model, VMR_model_n, self.k_gas_w_g_p_t, self.k_table_P_grid,
                self.k_table_T_grid, self.del_g, ScalingFactor=scale,
                R_plt=self.R_plt, R_star=self.R_star, solspec=solspec, k_cia=self.k_cia_pair_t_w,
                ID=self.gas_id_list,cia_nu_grid=self.cia_nu_grid,
                cia_T_grid=self.cia_T_grid, dH=dH, mmw=mmw, mmw_n=mmw_n,Ptop=Ptop, power=power,Ptop_n=Ptop_n, power_n=power_n, cia_contrib_on=cia_contrib_on,
                gas_contrib_on=gas_contrib_on, rayleigh_contrib_on=rayleigh_contrib_on, cloud_contrib_on=cloud_contrib_on,
                aprx_coef=aprx_coef, hazemult=hazemult, hazemult_n=hazemult_n, h2_frac=h2_frac)

        else:
            H_layer,H_base,P_layer,P_base,T_layer,VMR_layer,U_layer,dH,scale \
            = calc_layer_transm(
            self.R_plt, H_model, P_model, T_model, VMR_model,
            self.gas_id_list, self.NLAYER, path_angle, layer_type=1,
            H_0=np.min(H_model))

            transm_spectrum = calc_transm(self.wave_grid, H_layer,H_base, U_layer, P_layer, P_base,T_layer,
                VMR_layer, self.k_gas_w_g_p_t, self.k_table_P_grid,
                self.k_table_T_grid, self.del_g, ScalingFactor=scale,
                R_plt=self.R_plt, R_star=self.R_star, solspec=solspec, k_cia=self.k_cia_pair_t_w,
                ID=self.gas_id_list,cia_nu_grid=self.cia_nu_grid,
                cia_T_grid=self.cia_T_grid, dH=dH, mmw=mmw, Ptop=Ptop, power=power, cia_contrib_on=cia_contrib_on,
                gas_contrib_on=gas_contrib_on, rayleigh_contrib_on=rayleigh_contrib_on, cloud_contrib_on=cloud_contrib_on,
                aprx_coef=aprx_coef, hazemult=hazemult,h2_frac=h2_frac)
                
                
        return transm_spectrum
    
    def calc_transm_component(self, P_model, T_model, VMR_model,
        path_angle, Pref=None, Ptop=None, power=None, solspec=[], cia_contrib_on=True, gas_contrib_on=True,
        rayleigh_contrib_on=True, cloud_contrib_on=True, hazemult =None, aprx_coef=None, set_mmw=None, gas_index=[0]):
        """
        Calculate average layer properties from model inputs,
        then compute the transmission spectrum.
        Uses the hydrostatic balance equation to calculate layer height.
        """

        NPRO = len(P_model)
        mmw = np.zeros(P_model.shape)

        for ipro in range(NPRO):
            mmw[ipro] = calc_mmw(self.gas_id_list,VMR_model[ipro,:])
        
        if set_mmw is not None:
            mmw = np.ones(NPRO) * set_mmw
        
        if Pref is None:
            H_model = calc_hydrostat(P=P_model, T=T_model, mmw=mmw,
                M_plt=self.M_plt, R_plt=self.R_plt)
        else:
            H_model = calc_hydrostat_pref_test(P=P_model, T=T_model, mmw=mmw,
                M_plt=self.M_plt, R_plt=self.R_plt, Pref=Pref)
                        
        if H_model[0] == -999:
            print("Hydrostatic balance failed")
            return -999*np.ones(len(self.wave_grid))
            
        # print(P_model, T_model, mmw, H_model)
        
        # if np.isnan(H_model).any() or np.isinf(H_model).any():
        #     print("Hydrostatic balance failed")
        #     return -1, -1

        H_layer,H_base,P_layer,P_base,T_layer,VMR_layer,U_layer,dH,scale \
            = calc_layer_transm(
            self.R_plt, H_model, P_model, T_model, VMR_model,
            self.gas_id_list, self.NLAYER, path_angle, layer_type=1,
            H_0=np.min(H_model))

        if len(solspec)==0:
            solspec = np.ones(len(self.wave_grid))

        transm_spectrum = calc_component(self.wave_grid, H_layer,H_base, U_layer, P_layer, P_base,T_layer,
            VMR_layer, self.k_gas_w_g_p_t, self.k_table_P_grid,
            self.k_table_T_grid, self.del_g, ScalingFactor=scale,
            R_plt=self.R_plt, R_star=self.R_star, solspec=solspec, k_cia=self.k_cia_pair_t_w,
            ID=self.gas_id_list,cia_nu_grid=self.cia_nu_grid,
            cia_T_grid=self.cia_T_grid, dH=dH, mmw=mmw, Ptop=Ptop, power=power, cia_contrib_on=cia_contrib_on,
            gas_contrib_on=gas_contrib_on, rayleigh_contrib_on=rayleigh_contrib_on, cloud_contrib_on=cloud_contrib_on,
            aprx_coef=aprx_coef, hazemult=hazemult, gas_index=gas_index)

                
        return transm_spectrum
    

    def calc_transm_contrib(self, P_model, T_model, VMR_model,
        path_angle, Pref=None, Ptop=None, power=None, solspec=[], cia_contrib_on=True, gas_contrib_on=True,
        rayleigh_contrib_on=True, cloud_contrib_on=True, hazemult=None, aprx_coef=None, set_mmw=None):
        """
        Calculate average layer properties from model inputs,
        then compute the transmission spectrum.
        Uses the hydrostatic balance equation to calculate layer height.
        """

        NPRO = len(P_model)
        mmw = np.zeros(P_model.shape)

        for ipro in range(NPRO):
            mmw[ipro] = calc_mmw(self.gas_id_list,VMR_model[ipro,:])
        
        if set_mmw is not None:
            mmw = np.ones(P_model.shape) * set_mmw
        
        if Pref is None:
            H_model = calc_hydrostat(P=P_model, T=T_model, mmw=mmw,
                M_plt=self.M_plt, R_plt=self.R_plt)
        else:
            H_model = calc_hydrostat_pref_test(P=P_model, T=T_model, mmw=mmw,
                M_plt=self.M_plt, R_plt=self.R_plt, Pref=Pref)
        
        if H_model[0] == -999:
            print("Hydrostatic balance failed")
            return -999*np.ones(len(self.wave_grid))
            
        # print(P_model, T_model, mmw, H_model)
        
        # if np.isnan(H_model).any() or np.isinf(H_model).any():
        #     print("Hydrostatic balance failed")
        #     return -1, -1

        H_layer,H_base,P_layer,P_base,T_layer,VMR_layer,U_layer,dH,scale \
            = calc_layer_transm(
            self.R_plt, H_model, P_model, T_model, VMR_model,
            self.gas_id_list, self.NLAYER, path_angle, layer_type=1,
            H_0=np.min(H_model))

        if len(solspec)==0:
            solspec = np.ones(len(self.wave_grid))

        Rnom = calc_transm(self.wave_grid, H_layer,H_base, U_layer, P_layer, P_base,T_layer,\
                            VMR_layer, self.k_gas_w_g_p_t, self.k_table_P_grid,\
                            self.k_table_T_grid, self.del_g, ScalingFactor=scale,\
                            R_plt=self.R_plt, R_star=self.R_star, solspec=solspec, k_cia=self.k_cia_pair_t_w,\
                            ID=self.gas_id_list,cia_nu_grid=self.cia_nu_grid,\
                            cia_T_grid=self.cia_T_grid, dH=dH, mmw=mmw,Ptop=Ptop, power=power, cia_contrib_on=cia_contrib_on,
            gas_contrib_on=gas_contrib_on, rayleigh_contrib_on=rayleigh_contrib_on, cloud_contrib_on=cloud_contrib_on,
            aprx_coef=aprx_coef, hazemult=hazemult)

        Rj = np.zeros((self.NLAYER, len(self.wave_grid)))
        contribs = np.zeros((self.NLAYER, len(self.wave_grid)))

        for jlayer in range(self.NLAYER):
            Rj[jlayer, :] = calc_contrib(self.wave_grid, H_layer,H_base, U_layer, P_layer, P_base,T_layer,
            VMR_layer, self.k_gas_w_g_p_t, self.k_table_P_grid,
            self.k_table_T_grid, self.del_g, ScalingFactor=scale,
            R_plt=self.R_plt, R_star=self.R_star, solspec=solspec, k_cia=self.k_cia_pair_t_w,
            ID=self.gas_id_list,cia_nu_grid=self.cia_nu_grid,
            cia_T_grid=self.cia_T_grid, dH=dH, mmw=mmw, Ptop=Ptop, power=power, cia_contrib_on=cia_contrib_on,
            gas_contrib_on=gas_contrib_on, rayleigh_contrib_on=rayleigh_contrib_on, cloud_contrib_on=cloud_contrib_on,
            aprx_coef=aprx_coef, jlayer=jlayer, hazemult=hazemult)
            contribs[jlayer, :] = Rnom**2 - Rj[jlayer, :]**2
        contribs = contribs / np.sum(contribs, axis=0)
              
        return contribs, P_layer

    def calc_point_spectrum_hydro(self, P_model, T_model, VMR_model, Ptop=None,
        angles=np.array([0.,0.,0.]), solspec=[], A_model=None):
        """
        Use the hydrodynamic equation to calculate layer height
        First get layer properties from model inputs
        Then calculate the spectrum at a single point on the disc.
        """
        NPRO = len(P_model)
        mmw = np.zeros(P_model.shape)

        for ipro in range(NPRO):
            mmw[ipro] = calc_mmw(self.gas_id_list,VMR_model[ipro,:])

        H_model = calc_hydrostat(P=P_model, T=T_model, mmw=mmw,
            M_plt=self.M_plt, R_plt=self.R_plt)

        point_spectrum = self.calc_point_spectrum(H_model, P_model,
            T_model, VMR_model, Ptop, angles, solspec, A_model=A_model)

        return point_spectrum

    

    def calc_disc_spectrum(self,phase,nmu,P_model,
        global_model_P_grid,global_T_model,global_VMR_model,
        mod_lon,mod_lat,solspec,global_aerosol_model):
        """
        Parameters
        ----------
        phase : real
            Orbital phase, increase from 0 at primary transit to 180 and secondary
            eclipse.
        aerosol_mode: 

        """
        # initialise output array
        disc_spectrum = np.zeros(len(self.wave_grid))

        # get locations and angles for disc averaging
        nav, wav = gauss_lobatto_weights(phase, nmu)
        wav = np.around(wav,decimals=8)
        fov_latitudes = wav[0,:]
        fov_longitudes = wav[1,:]
        fov_stellar_zen = wav[2,:]
        fov_emission_angles = wav[3,:]
        fov_stellar_azi = wav[4,:]
        fov_weights = wav[5,:]

        for iav in range(nav):
            xlon = fov_longitudes[iav]
            xlat = fov_latitudes[iav]
            T_model, VMR_model = interp_gcm(
                lon=xlon,lat=xlat, p=P_model,
                gcm_lon=mod_lon, gcm_lat=mod_lat,
                gcm_p=global_model_P_grid,
                gcm_t=global_T_model, gcm_vmr=global_VMR_model,
                substellar_point_longitude_shift=180)

            path_angle = fov_emission_angles[iav]
            weight = fov_weights[iav]
            NPRO = len(P_model)
            mmw = np.zeros(NPRO)
            for ipro in range(NPRO):
                mmw[ipro] = calc_mmw(self.gas_id_list,VMR_model[ipro,:])
            H_model = calc_hydrostat(P=P_model, T=T_model, mmw=mmw,
                M_plt=self.M_plt, R_plt=self.R_plt)

            point_spectrum = self.calc_point_spectrum(
                H_model, P_model, T_model, VMR_model, path_angle,
                solspec=solspec)

            disc_spectrum += point_spectrum * weight
        return disc_spectrum

    def calc_disc_spectrum_uniform(self, nmu, P_model, T_model, VMR_model,
        H_model=[],solspec=[], A_model=None):
        """Caculate the disc integrated spectrum of a homogeneous atmosphere
        A_Model = Aerosol
        """
        # initialise output array
        disc_spectrum = np.zeros(len(self.wave_grid))
        nav, wav = gauss_lobatto_weights(0, nmu)
        
        fov_sol_angles = wav[2,:]
        fov_emission_angles = wav[3,:]
        fov_azimuth_angles = wav[4,:]
        fov_weights = wav[5,:]

        # Hydrostatic case
        if len(H_model) == 0:
            for iav in range(nav):
                sol_angle = fov_sol_angles[iav]
                path_angle = fov_emission_angles[iav]
                azi_angle = fov_azimuth_angles[iav]
                
                angles = np.array([sol_angle,path_angle,azi_angle])
                
                weight = fov_weights[iav]
                point_spectrum = self.calc_point_spectrum_hydro(
                    P_model, T_model, VMR_model, angles,
                    solspec=solspec, A_model=A_model)
                disc_spectrum += point_spectrum * weight
        else:
            for iav in range(nav):
                sol_angle = fov_sol_angles[iav]
                path_angle = fov_emission_angles[iav]
                azi_angle = fov_azimuth_angles[iav]
                
                angles = np.array([sol_angle,path_angle,azi_angle])
                
                weight = fov_weights[iav]
                point_spectrum = self.calc_point_spectrum(
                    H_model, P_model, T_model, VMR_model, angles,
                    solspec=solspec, A_model=A_model)
                disc_spectrum += point_spectrum * weight
        return disc_spectrum
    

    def set_phase_function(self, mean_size, size_variance, n_imag, n_imag_wave_grid, n_real_reference,
                   n_real_reference_wave = None, normalising_wave = None, ispace=1):

        """
        Calculates extinction and scattering cross-sections, along with fitted phase function parameters,
        for an aerosol species with defined properties, assuming a standard gamma distribution.

        Parameters
        ----------
        wave_grid(NWAVE) : ndarray
            Wavelengths (um) grid for calculating parameters.
        mean_size : float
            Mean size of the particles
        size_variance : float
            Size variance of the particles
        n_imag(NWAVEIMAG) : ndarray
            Grid of imaginary refractive indices
        n_imag_wave_grid(NWAVEIMAG) : ndarray
            Grid of wavelengths that n_imag lies on
        n_real_reference : float
            Reference real refractive index
        n_real_reference_wave : float
            Wavelength that the n_real_reference corresponds to. Defaults to wave_grid[0]
        normalising_wave : float
            Wavelength that the cross-sections should be normalised at. Defaults to wave_grid[0]

        Returns
        ---------
        phase_func(1,NWAVE,6) : ndarray
            Contains the cross sections and phase function parameters for use with multiple scattering.

        """
        wave_grid = self.wave_grid

        phase_func = np.zeros((1,len(wave_grid),6))
        
        if n_real_reference_wave is None:
            n_real_reference_wave = wave_grid[0]
        if normalising_wave is None:
            normalising_wave = wave_grid[0]      

        n_real_reference_wave = np.array([n_real_reference_wave]) #placeholders before adding multiple modes
        normalising_wave = np.array([normalising_wave])
        
            
        # Assuming iscat = 1 (standard gamma)
        size_distribution_parameters = np.array([mean_size,size_variance,(1-3*size_variance)/size_variance])

        size_integration_bounds = np.array([0.015*np.min(n_imag_wave_grid), 0.0, 0.015*np.min(n_imag_wave_grid)]) 

        small_phase_func = makephase(wave_grid = n_imag_wave_grid,
                             iscat = np.array([1]),
                             dsize = size_distribution_parameters,
                             rs = size_integration_bounds,
                             nimag_wave_grid = n_imag_wave_grid,
                             nreal_ref = np.array([n_real_reference]),
                             nimag = n_imag,
                             refwave = n_real_reference_wave,
                             ispace = ispace)[None,:]
    
    
        spline = CubicSpline(n_imag_wave_grid, 
                              small_phase_func[0,:, :], 
                              axis=0)

        phase_func[0,:, :] = spline(wave_grid)
        xext_norm = spline(normalising_wave)[0][0]

        phase_func[0,:, :2] = phase_func[0,:, :2] / xext_norm


#         for iphas in range(2,5):
#             phase_func[0,:, iphas] = np.interp(wave_grid,n_imag_wave_grid,\
#                                                small_phase_func[0,:, iphas])
        self.phase_func = phase_func
