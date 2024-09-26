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

fits = {'400_air_lo':dict(t0=0.188,sig_c=4.343,sig_l=39.432,
sig_0=23.648,tau=0.004,err_tau=0.000),
'400_air_hi':dict(t0=45.865,sig_c=6.419,sig_l=1.835,
sig_0=1146.990,tau=0.084,err_tau=0.002),
'400_vac_lo':dict(t0=0.111,sig_c=7.062,sig_l=62.062,
sig_0=56.165,tau=0.006,err_tau=0.000),
'400_vac_hi':dict(t0=65.621,sig_c=0.712,sig_l=1.226,
sig_0=1364.121,tau=0.051,err_tau=0.002),
'600_air_lo':dict(t0=0.073,sig_c=3.662,sig_l=69.085,
sig_0=38.318,tau=0.001,err_tau=0.000),
'600_air_hi':dict(t0=18.610,sig_c=6.601,sig_l=2.828,
sig_0=3754.236,tau=0.087,err_tau=0.002),
'600_vac_lo':dict(t0=0.075,sig_c=6.068,sig_l=80.774,
sig_0=19.747,tau=0.004,err_tau=0.000),
'600_vac_hi':dict(t0=24.054,sig_c=5.243,sig_l=1.588,
sig_0=2955.353,tau=0.171,err_tau=0.005),
'960_air_lo':dict(t0=0.103,sig_c=4.072,sig_l=53.706,
sig_0=35.626,tau=0.001,err_tau=0.000),
'960_air_hi':dict(t0=86.751,sig_c=5.359,sig_l=2.330,
sig_0=1763.697,tau=0.110,err_tau=0.003),
'960_vac_lo':dict(t0=0.726,sig_c=10.489,sig_l=52.339,
sig_0=16.691,tau=0.005,err_tau=0.000),
'960_vac_hi':dict(t0=50.396,sig_c=12.231,sig_l=1.444,
sig_0=2160.705,tau=0.205,err_tau=0.004)}

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

def _obj_func_lo(time, sig_c, sig_l, sig_0, tau, t0):
    t = time-t0
    return sig_c + sig_l * t + sig_0 * np.exp(t/tau) 

def _obj_func_hi(time, sig_c, sig_l, sig_0, tau, t0):
    t = time-t0
    return sig_0 * np.exp(-t/tau) + sig_c - sig_l * t
        
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
    
    params = fits[f'{temp}_{environ}_hi']
    tau = params['tau']
    err_tau = params['err_tau']
    del params['err_tau']
    _fit = _obj_func_hi(time,**params)
    rho_twin.plot(time,_fit,lw=1,marker='o',ms=0,c='k',ls=(0,(4,1,2,1)))
    rho_twin.annotate(rf'$\tau=$ {tau*60:.2f}'+'\n'+rf' $  \pm$ {err_tau*60:.2f} s',
                      xy=(0.5,0.5),xycoords='axes fraction',fontsize='large')
    
    params = fits[f'{temp}_{environ}_lo']
    tau = params['tau']
    err_tau = params['err_tau']
    del params['err_tau']
    _fit = _obj_func_lo(time,**params)
    rho_ax.plot(time,_fit,lw=1,marker='o',ms=0,c='k',ls=(0,(4,1,2,1)))
    rho_ax.annotate(rf'$\tau=$ {tau*60:.2f}'+'\n'+rf' $  \pm$ {err_tau*60:.2f} s',
                      xy=(0.32,0.3),xycoords='axes fraction',fontsize='large')
    
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
    
    field_ax.set_ylabel('E-field [V/cm]',fontsize='large',labelpad=5,c='b')
    current_twin.set_ylabel(r'Current density [mA/mm$^2$]',fontsize='large',labelpad=3,c='r')
    rho_ax.set_ylabel(r'$\sigma$ [($\Omega$-cm)$^{-1}$]',fontsize='large',labelpad=5,c='m')
    
    fig.supxlabel('Time [m]',fontsize='large',y=0.02)
    
    fig.suptitle(rf'T={temp}$^\circ$C, TiO$_2$ flashed in {environ}',fontsize='large',y=0.94)
    
    field_ax.annotate('(a)',xy=(0.05,0.9),xycoords='axes fraction',fontsize='large')
    field_twin.annotate('(b)',xy=(0.825,0.9),xycoords='axes fraction',fontsize='large')
    rho_ax.annotate('(c)',xy=(0.05,0.9),xycoords='axes fraction',fontsize='large')
    rho_twin.annotate('(d)',xy=(0.825,0.9),xycoords='axes fraction',fontsize='large')
    
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

