import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nemesispy.common.info_mol_id import mol_id
from nemesispy.common.constants import *
import os
from nemesispy.radtran.forward_model import ForwardModel
import spectres
from time import time

# read from prf
prf_header = ['height', 'press', 'temp', 'h2o', 'co2', 'co', 'ch4', 'h2', 'he', 'na', 'k']
prf = pd.read_csv('./example_data/hd189.prf', skiprows=11, header=None, sep='\s+')
prf.columns = prf_header
H_model = prf['height'].values * 1e3
P_model = prf['press'].values * ATM
T_model = prf['temp'].values
VMR_model = prf[['h2o', 'co2', 'co', 'ch4', 'h2', 'he', 'na', 'k']].values

# planet reflected spectrum
planet_spec = pd.read_csv('./example_data/hd189_data.txt', header=None, sep='\s+')
planet_wave = planet_spec[0].values
planet_flux = planet_spec[1].values
planet_err = planet_spec[2].values

# stellar spectrum
st_spec = pd.read_csv('./example_data/hd189.dat', skiprows=3, header=None, sep='\s+')
st_wave_full = st_spec[0].values
st_flux_full = st_spec[1].values / (4*np.pi)
st_wave = planet_wave.copy()
st_flux = spectres.spectres(planet_wave, st_wave_full, st_flux_full)

# enstatite
enstatite = pd.read_csv('./example_data/enstatite.dat', skiprows=2, header=None, sep='\s+')
en_wave = enstatite[0].values
en_real = enstatite[1].values
en_imag = enstatite[2].values

# Wavelengths grid for the spectrum (microns)
wave_grid = planet_wave
nwave = len(wave_grid)
# Orbital phase grid (degree)
phase_grid = np.array([ 22.5,  45. ,  67.5,  90. , 112.5, 135. , 157.5, 180. ,
    202.5, 225. , 247.5, 270. , 292.5, 315. , 337.5])
nphase = len(phase_grid)
# Pick resolution for the disc average
nmu = 5 # Number of mu bins
# Reference planetary parameters
M_plt = 2167.664 * 1e24
R_plt = 84592.31 * 1e3 # m
# List of gas species to include in the model using identifiers
gas_names_active = ['H2O', 'CO2', 'CO', 'CH4', 'Na', 'K']
nvmr = len(gas_names_active)
# Spectrally inactive gases to include
gas_names_inactive = ['H2', 'He']
gas_id = [mol_id[gas] for gas in gas_names_active[0:4]] + [mol_id[gas] for gas in gas_names_inactive] + [mol_id[gas] for gas in gas_names_active[4:]]
gas_id = np.array(gas_id)
iso_id = np.zeros_like(gas_id) # Isotopologue identifier
lowres_file_paths = [f'{gas}_hd189_refl.kta' for gas in gas_names_active]
# __location__ = os.environ['nemesispy_scatter']
__location__ = '.'
ktable_path = os.path.join(__location__, "nemesispy/data/ktables")
for ipath,path in enumerate(lowres_file_paths):
    lowres_file_paths[ipath] = os.path.join(ktable_path,path)
cia_folder_path = os.path.join(__location__ , "nemesispy/data/cia")
cia_file_path = os.path.join(cia_folder_path,'exocia_hitran12_200-3800K.tab')
# Define the atmospheric model
NLAYER = len(P_model) # Number of layers

# Create a ForwardModel object
FM = ForwardModel()
FM.set_planet_model(
    M_plt=M_plt,R_plt=R_plt,
    gas_id_list=gas_id,iso_id_list=iso_id,
    NLAYER=NLAYER
    )
FM.set_opacity_data(
    kta_file_paths=lowres_file_paths,
    cia_file_path=cia_file_path
)

FM2 = ForwardModel()
FM2.set_planet_model(
    M_plt=M_plt,R_plt=R_plt,
    gas_id_list=gas_id,iso_id_list=iso_id,
    NLAYER=NLAYER
    )
FM2.set_opacity_data(
    kta_file_paths=lowres_file_paths,
    cia_file_path=cia_file_path
)

# Using 20 calculation wavelengths for Makephase,
# evenly distributed over our wavelength range
NWAVE_NIMAG = len(en_wave)
n_imag_wave_grid = en_wave

# Adding mode 1 - gamma distribution (iscat=1)
n_imag =  en_imag
# n_real is set by reference for now
n_real = en_real
A_model = 1e0 * np.ones((NLAYER, 1))
wave_grid = FM.wave_grid

FM.clear_phase_function()
ms = 0.01
FM.add_phase_function(mean_size = ms, 
                    size_variance = 0.0, 
                    n_imag = n_imag, 
                    n_imag_wave_grid = n_imag_wave_grid, 
                    n_real_reference = 0.5,
                    n_real_reference_wave= 0.4,
                    iscat = 4)

A_model = 0.5 * np.ones((NLAYER, 1))

toc = time()

point = FM2.calc_point_spectrum_hydro(P_model = P_model, 
                                     T_model = T_model, 
                                     VMR_model = VMR_model, 
                                     angles=np.array([0.0,0.0,180.0]),
                                     solspec=st_flux) # angles: stellar, emission, azimuth


point_aerosol = FM.calc_point_spectrum_hydro(P_model = P_model, 
                                     T_model = T_model, 
                                     VMR_model = VMR_model, 
                                     A_model = A_model,
                                     angles=np.array([0.0,0.0,180.0]),
                                     solspec=st_flux) # angles: stellar, emission, azimuth


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), height_ratios=[3, 1], sharex=True)

toc = time()
for i in range(2000):
    point = FM2.calc_point_spectrum_hydro(P_model = P_model, 
                                     T_model = T_model, 
                                     VMR_model = VMR_model, 
                                     angles=np.array([0.0,0.0,180.0]),
                                     solspec=st_flux) # angles: stellar, emission, azimuth
    # ax1.plot(wave_grid, point, label=f'NS_{i}', color='#88CCEE', lw=3, alpha=0.9)
    
tic = time()
print(f'No scattering ran in {tic-toc}s')

toc = time()
for i in range(2000):
    A_model = 0.5 * np.ones((NLAYER, 1))
    point_aerosol = FM.calc_point_spectrum_hydro(P_model = P_model, 
                                     T_model = T_model, 
                                     VMR_model = VMR_model, 
                                     A_model = A_model,
                                     angles=np.array([0.0,0.0,180.0]),
                                     solspec=st_flux) # angles: stellar, emission, azimuth
    # ax1.plot(wave_grid, point_aerosol, label=f'S_{i}', lw=2, alpha=0.7)

tic = time()
print(f'Scattering ran in {tic-toc}s')

ax1.plot(wave_grid, point, label='No scattering or aerosol', color='#88CCEE', lw=3, alpha=0.9)
ax1.plot(wave_grid, point_aerosol, label='Scattering, constant aerosol', color='#CC6677', lw=3, alpha=0.7, linestyle='--')
# ax1.errorbar(wave_grid, planet_flux, yerr=planet_err, xerr=0.022, fmt='o', color='k', \
#             ms=7, elinewidth=1.5, capsize=2, label='HD189 data', alpha=0.3)
ax1.set_ylabel('Flux Ratio')
ax1.legend()
ax2.plot(wave_grid, (point_aerosol - point), color='k', lw=3, alpha=0.7)
ax2.set_ylabel('$\Delta$')
plt.xlabel('Wavelength ($\mu$m)')
plt.savefig('HD189.png', dpi=600)