#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 09:51:26 2024

@author: ty
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
import sys

mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = r"\usepackage{bm}"

# mpl.rcParams['mathtext.fontset'] = 'stix'
# mpl.rcParams['font.family'] = 'STIXGeneral'

fits = {'400_air':dict(t0=46.203,sig_hi=7835.644,sig_0=-29.423,tau=0.066,
 sig_lo=-7830.405,sig_lin=0.959,err_tau=0.001),
'400_vac':dict(t0=65.825,sig_hi=49.032,sig_0=-24.719,tau=0.050,
 sig_lo=-48.506,sig_lin=1.457,err_tau=0.002),
'600_air':dict(t0=19.060,sig_hi=51.089,sig_0=-20.299,tau=0.101,
 sig_lo=-46.332,sig_lin=2.208,err_tau=0.002),
'600_vac':dict(t0=24.857,sig_hi=-28.386,sig_0=-27.239,tau=0.168,
 sig_lo=32.580,sig_lin=1.853,err_tau=0.005),
'960_air':dict(t0=87.247,sig_hi=-167.368,sig_0=-20.531,tau=0.116,
 sig_lo=170.963,sig_lin=1.515,err_tau=0.003),
'960_vac':dict(t0=51.188,sig_hi=-102.882,sig_0=-44.741,tau=0.213,
 sig_lo=113.609,sig_lin=1.200,err_tau=0.004)}

# --------------------------------------------------------------------------------------------------

def replace_zeros(x,array):
    
    _inds = np.flatnonzero(array == 0)    
    _inds = np.r_[_inds,np.flatnonzero(np.isnan(array))]
    _inds = np.r_[_inds,np.flatnonzero(np.isinf(array))]
    _inds = np.unique(_inds)
    
    array[_inds] = np.interp(x[_inds],np.delete(x,_inds),np.delete(array,_inds))
    
# --------------------------------------------------------------------------------------------------

def get_data(file_name,sample_len,sample_area):
    
    """
    len is in cm, area is in mm^2
    """
    
    data = np.loadtxt(file_name,delimiter=';',skiprows=1,dtype=object)
    
    voltage = data[:,1].astype(str)
    current = data[:,3].astype(str)
    time = data[:,-1].astype(float)/60 # seconds to minutes

    voltage = np.char.strip(voltage,'V').astype(float) # V
    current = np.char.strip(current,'A').astype(float)*1000 # mA
    
    _inds = np.flatnonzero(voltage == 0)    
    time = np.delete(time,_inds)
    voltage = np.delete(voltage,_inds)
    current = np.delete(current,_inds)
    
    resistance = voltage/current

    field = voltage/sample_len # V/cm
    current_den = current/sample_area # mA/mm^2
    resistivity = resistance*(sample_area*0.01)/sample_len # rho = R*A/L => ohm*cm
    
    conductivity = 1/resistivity

    return current, current_den, voltage, field, resistivity, conductivity, time

# --------------------------------------------------------------------------------------------------

def _obj_func_hi(time,sig_hi, sig_0, tau, sig_lo, sig_lin):
    return sig_hi - sig_0 * np.exp(-time/tau) + sig_lo - sig_lin * time
        
# --------------------------------------------------------------------------------------------------

def parse_file_name(file_name):
    
    _f = file_name.split('-')
    
    temp = _f[1].strip('C')
    environ = _f[3].lower()
    
    return temp, environ

# --------------------------------------------------------------------------------------------------

def plot_and_fit(file_name,sample_len,sample_area,field_ylims,current_ylims,rho_ylims,t_lo_lims,t_hi_lims):
    
    current, current_den, voltage, field, resistivity, conductivity, time = \
            get_data(file_name,sample_len,sample_area)
            
    temp, environ = parse_file_name(file_name)
    
    fig, ax = plt.subplots(2,2,figsize=(4,5),
                           gridspec_kw={'height_ratios':[1,1],'width_ratios':[1,1],
                                        'hspace':0.1,'wspace':0.1}) 
    
    field_ax = ax[0,0]; field_twin = ax[0,1]
    current_ax = field_ax.twinx(); current_twin = field_twin.twinx()
    rho_ax = ax[1,0]; rho_twin = ax[1,1]
    
    field_ax.plot(time,field,lw=1,marker='o',ms=1,c='b')
    field_twin.plot(time,field,lw=1,marker='o',ms=1,c='b')
    
    current_ax.plot(time,current_den,lw=1,marker='d',ms=1,c='r',ls=(0,(2,1)))
    current_twin.plot(time,current_den,lw=1,marker='d',ms=1,c='r',ls=(0,(2,1)))
    
    field_ax.set_zorder(0)
    current_ax.set_zorder(1)
    
    rho_ax.plot(time,conductivity,lw=1,marker='o',ms=1,c='m')
    rho_twin.plot(time,conductivity,lw=1,marker='o',ms=1,c='m')
    
    params = fits[f'{temp}_{environ}']
    t0 = params['t0']
    tau = params['tau']
    err_tau = params['err_tau']
    del params['t0'], params['err_tau']
    _fit = _obj_func_hi(time-t0,**params)
    rho_twin.plot(time,_fit,lw=1,marker='o',ms=0,c='k')
    
    axes = [field_ax,current_ax,rho_ax, field_twin,current_twin,rho_twin]
    
    for _ax in axes:
        for axis in ['top','bottom','left','right']:
            _ax.spines[axis].set_linewidth(1.5)
        _ax.minorticks_on()
        _ax.tick_params(which='both',width=1,labelsize='large')
        _ax.tick_params(which='major',length=5)
        _ax.tick_params(which='minor',length=2)
        _ax.set_rasterized = True
        
    field_ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    field_twin.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    rho_twin.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    rho_ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    
    field_ax.spines.right.set_visible(False)
    current_ax.spines.right.set_visible(False)
    rho_ax.spines.right.set_visible(False)
    
    field_ax.tick_params(axis='y', colors='b')
    current_twin.tick_params(axis='y', colors='r')
    rho_ax.tick_params(axis='y', colors='m')
    
    field_ax.spines.right.set_visible(False)
    current_ax.spines.right.set_visible(False)
    rho_ax.spines.right.set_visible(False)
    
    field_ax.tick_params(axis='y',which='both',right=False,labelright=False)
    current_ax.tick_params(axis='y',which='both',right=False,labelright=False)
    rho_ax.tick_params(axis='y',which='both',right=False,labelright=False)
    
    field_twin.tick_params(axis='y',which='both',left=False,labelleft=False)
    current_twin.tick_params(axis='y',which='both',left=False,labelleft=False)
    rho_twin.tick_params(axis='y',which='both',left=False,labelleft=False)
    
    field_twin.spines.left.set_visible(False)
    current_twin.spines.left.set_visible(False)
    rho_twin.spines.left.set_visible(False)
    
    d = 0.1  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-d, -d), (d, d)], markersize=5,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    field_ax.plot((1,1), (0,0), transform=field_ax.transAxes, **kwargs)
    field_ax.plot((1,1), (1,1), transform=field_ax.transAxes, **kwargs)
    field_twin.plot((0,0), (0,0), transform=field_twin.transAxes, **kwargs)
    field_twin.plot((0,0), (1,1), transform=field_twin.transAxes, **kwargs)
    
    field_ax.plot([1,1],[0,1], transform=field_ax.transAxes, lw=1, ls=(0,(4,1,2,1)),ms=0, c='k')
    field_ax.plot([1,1],[0,1], transform=field_ax.transAxes, lw=1, ls=(0,(4,1,2,1)),ms=0, c='k')
    field_twin.plot([0,0],[0,1], transform=field_twin.transAxes, lw=1, ls=(0,(4,1,2,1)),ms=0, c='k')
    field_twin.plot([0,0],[0,1], transform=field_twin.transAxes, lw=1, ls=(0,(4,1,2,1)),ms=0, c='k')
    
    rho_ax.plot((1,1), (0,0), transform=rho_ax.transAxes, **kwargs)
    rho_ax.plot((1,1), (1,1), transform=rho_ax.transAxes, **kwargs)
    rho_twin.plot((0,0), (0,0), transform=rho_twin.transAxes, **kwargs)
    rho_twin.plot((0,0), (1,1), transform=rho_twin.transAxes, **kwargs)
    
    rho_ax.plot([1,1],[0,1], transform=rho_ax.transAxes, lw=1, ls=(0,(4,1,2,1)),ms=0, c='k')
    rho_ax.plot([1,1],[0,1], transform=rho_ax.transAxes, lw=1, ls=(0,(4,1,2,1)),ms=0, c='k')
    rho_twin.plot([0,0],[0,1], transform=rho_twin.transAxes, lw=1, ls=(0,(4,1,2,1)),ms=0, c='k')
    rho_twin.plot([0,0],[0,1], transform=rho_twin.transAxes, lw=1, ls=(0,(4,1,2,1)),ms=0, c='k')
    
    field_ax.set_xlim(t_lo_lims)
    field_ax.set_ylim(field_ylims)
    current_ax.set_xlim(t_lo_lims)
    current_ax.set_ylim(current_ylims)
    rho_ax.set_xlim(t_lo_lims)
    rho_ax.set_ylim(rho_ylims)
    
    field_twin.set_xlim(t_hi_lims)
    field_twin.set_ylim(field_ylims)
    current_twin.set_xlim(t_hi_lims)
    current_twin.set_ylim(current_ylims)
    rho_twin.set_xlim(t_hi_lims)
    rho_twin.set_ylim(rho_ylims)
    
    field_ax.set_xticklabels([])
    current_ax.set_xticklabels([])
    
    field_twin.set_xticklabels([])
    current_twin.set_xticklabels([])
    
    rho_twin.annotate(rf'$\tau=$ {tau*60:.2f}'+'\n'+rf' $  \pm$ {err_tau*60:.2f} s',
                      xy=(0.5,0.7),xycoords='axes fraction',fontsize='large')
    
    field_ax.set_ylabel('E-field [V/cm]',fontsize='large',labelpad=5,c='b')
    current_twin.set_ylabel(r'Current density [mA/mm$^2$]',fontsize='large',labelpad=3,c='r')
    rho_ax.set_ylabel(r'$\sigma$ [($\Omega$-cm)$^{-1}$]',fontsize='large',labelpad=5,c='m')
    
    fig.supxlabel('Time [m]',fontsize='large',y=0.02)
    
    fig.suptitle(rf'T={temp}$^\circ$C, flashed in {environ}',fontsize='large',y=0.94)
    
    fig_name = f'TiO2_{temp}_C_{environ}.png'
    plt.savefig(fig_name,dpi=300,bbox_inches='tight')
    
    
# --------------------------------------------------------------------------------------------------


file_name = 'TiO2/900C-400C-TiO2-Air-new-3.csv'
sample_len = 0.217
sample_area = 1.7545

field_ylims = [0,600]
current_ylims = [0,150]
rho_ylims = [0,80]

# 3 to 1 ratio
t_lo_lims = [0,3] 
t_hi_lims = [45,48] 

plot_and_fit(file_name,sample_len,sample_area,field_ylims,current_ylims,rho_ylims,t_lo_lims,t_hi_lims)

# --------------------------------

file_name = 'TiO2/900C-400C-TiO2-Vac-new-3.csv'
sample_len = 0.204
sample_area = 1.6775

field_ylims = [0,400]
current_ylims = [0,150]
rho_ylims = [0,80]

# 3 to 1 ratio
t_lo_lims = [0,3] 
t_hi_lims = [64.5,67.5] 

plot_and_fit(file_name,sample_len,sample_area,field_ylims,current_ylims,rho_ylims,t_lo_lims,t_hi_lims)

# --------------------------------

file_name = 'TiO2/900C-600C-TiO2-Air-new-4.csv'
sample_len = 0.211
sample_area = 1.8535

field_ylims = [0,400]
current_ylims = [0,150]
rho_ylims = [0,80]

# 3 to 1 ratio
t_lo_lims = [0,3] 
t_hi_lims = [18,21] 

plot_and_fit(file_name,sample_len,sample_area,field_ylims,current_ylims,rho_ylims,t_lo_lims,t_hi_lims)

# --------------------------------

file_name = 'TiO2/900C-600C-TiO2-Vac-new-3.csv'
sample_len = 0.194
sample_area = 1.4245

field_ylims = [0,400]
current_ylims = [0,150]
rho_ylims = [0,100]

# 3 to 1 ratio
t_lo_lims = [0,3] 
t_hi_lims = [23.5,26.5] 

plot_and_fit(file_name,sample_len,sample_area,field_ylims,current_ylims,rho_ylims,t_lo_lims,t_hi_lims)

# --------------------------------

file_name = 'TiO2/900C-960C-TiO2-Air-new-2.csv'
sample_len = 0.214
sample_area = 1.969

field_ylims = [0,400]
current_ylims = [0,150]
rho_ylims = [0,90]

# 3 to 1 ratio
t_lo_lims = [0,3] 
t_hi_lims = [86,89] 

plot_and_fit(file_name,sample_len,sample_area,field_ylims,current_ylims,rho_ylims,t_lo_lims,t_hi_lims)

# --------------------------------

file_name = 'TiO2/900C-960C-TiO2-Vac-new-4.csv'
sample_len = 0.227
sample_area = 1.5675

field_ylims = [0,400]
current_ylims = [0,150]
rho_ylims = [0,100]

# 3 to 1 ratio
t_lo_lims = [0,3] 
t_hi_lims = [50,53] 

plot_and_fit(file_name,sample_len,sample_area,field_ylims,current_ylims,rho_ylims,t_lo_lims,t_hi_lims)

