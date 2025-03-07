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

fits = {'21_air_lo':dict(t0=0.444,sig_c=155.223,sig_l=-110.647,
                             sig_0=4.289,tau=0.017423,err_tau=0.001419),
        '21_air_hi':dict(t0=4.368,sig_c=136.970,sig_l=-20.038,
                             sig_0=503.595,tau=0.017025,err_tau=0.000915)}

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
    
    try:
        data = np.loadtxt(file_name,delimiter=';',skiprows=1,dtype=object)
    except:
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
    
    resistance = voltage/current*1000 # Ohm

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

def plot_and_fit(file_name,sample_len,sample_area,field_ylims,
                 current_ylims,rho_ylims,t_lo_lims,t_hi_lims,temp,environ):
    
    current, current_den, voltage, field, resistivity, conductivity, time = \
            get_data(file_name,sample_len,sample_area)
            
    # temp, environ = parse_file_name(file_name)
    
    fig, ax = plt.subplots(1,2,figsize=(4,2),
                           gridspec_kw={'height_ratios':[1],'width_ratios':[1,1],
                                        'hspace':0.1,'wspace':0.1}) 
    
    field_ax = ax[0]; field_twin = ax[1]
    rho_ax = field_ax.twinx(); rho_twin = field_twin.twinx()
    
    field_ax.plot(time,field,lw=1,marker='o',ms=1,c='b')
    field_twin.plot(time,field,lw=1,marker='o',ms=1,c='b')
    
    field_ax.set_zorder(0)
    rho_ax.set_zorder(1)
    
    rho_ax.plot(time,conductivity,lw=1,marker='o',ms=1,c='m')
    rho_twin.plot(time,conductivity,lw=1,marker='o',ms=1,c='m')
    
    params = fits[f'21_air_hi']
    tau = params['tau']
    err_tau = params['err_tau']
    del params['err_tau']
    _fit = _obj_func_hi(time,**params)
    rho_twin.plot(time,_fit,lw=1,marker='o',ms=0,c='k',ls=(0,(4,1,2,1)))
    rho_twin.annotate(rf'$\tau=$ {tau*60:.2f}'+'\n'+rf' $  \pm$ {err_tau*60:.2f} s',
                      xy=(0.3,0.65),xycoords='axes fraction',fontsize='large')
    
    params = fits[f'21_air_lo']
    tau = params['tau']
    err_tau = params['err_tau']
    del params['err_tau']
    _fit = _obj_func_lo(time,**params)
    rho_ax.plot(time,_fit,lw=1,marker='o',ms=0,c='k',ls=(0,(4,1,2,1)))
    rho_ax.annotate(rf'$\tau=$ {tau*60:.2f}'+'\n'+rf' $  \pm$ {err_tau*60:.2f} s',
                      xy=(0.35,0.25),xycoords='axes fraction',fontsize='large')
    
    axes = [field_ax,rho_ax, field_twin, rho_twin]
    
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
    rho_ax.spines.right.set_visible(False)
    
    field_ax.tick_params(axis='y', colors='b')
    rho_twin.tick_params(axis='y', colors='m')
    
    field_ax.spines.right.set_visible(False)
    rho_ax.spines.right.set_visible(False)
    
    field_ax.tick_params(axis='y',which='both',right=False,labelright=False)
    rho_ax.tick_params(axis='y',which='both',right=False,labelright=False)
    
    field_twin.tick_params(axis='y',which='both',left=False,labelleft=False)
    rho_twin.tick_params(axis='y',which='both',left=False,labelleft=False)
    
    field_twin.spines.left.set_visible(False)
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
    rho_ax.set_xlim(t_lo_lims)
    rho_ax.set_ylim(rho_ylims)
    
    field_twin.set_xlim(t_hi_lims)
    field_twin.set_ylim(field_ylims)
    rho_twin.set_xlim(t_hi_lims)
    rho_twin.set_ylim(rho_ylims)
    
    field_ax.set_ylabel('E-field [V/cm]',fontsize='large',labelpad=5,c='b')
    rho_twin.set_ylabel(r'$\sigma$ [($\Omega$-cm)$^{-1}$]',fontsize='large',labelpad=5,c='m')

    _T_K = int(temp)+274

    fig.supxlabel('Time [minutes]',fontsize='large',y=-0.1)    
    fig.suptitle(rf'T={_T_K} K, Si flashed in {environ}',fontsize='large',y=1.0)
    
    # field_ax.annotate('(a)',xy=(0.05,0.9),xycoords='axes fraction',fontsize='large')
    # field_twin.annotate('(b)',xy=(0.825,0.9),xycoords='axes fraction',fontsize='large')
    # rho_ax.annotate('(c)',xy=(0.05,0.9),xycoords='axes fraction',fontsize='large')
    # rho_twin.annotate('(d)',xy=(0.825,0.9),xycoords='axes fraction',fontsize='large')
    
    fig_name = f'Si_{_T_K}_K_{environ}.png'
    print(fig_name)
    plt.savefig(fig_name,dpi=300,bbox_inches='tight')
    
    
# --------------------------------------------------------------------------------------------------

file_name = 'Si/Room-Si-Flash-attempt-1.csv'
temp, environ = 21, 'air'
sample_len = 0.592
sample_area = 1.7394

field_ylims = [0,40]
current_ylims = [0,150]
rho_ylims = [0,0.6]

# 3 to 1 ratio
t_lo_lims = [0.4,0.8] 
t_hi_lims = [4.3,4.7] 

plot_and_fit(file_name,sample_len,sample_area,field_ylims,
             current_ylims,rho_ylims,t_lo_lims,t_hi_lims,temp,environ)

# --------------------------------------------------------------------------------------------------

# file_name = 'Si-Flash Data/Room-Si-Flash-1.csv'
# sample_len = 0.459
# sample_area = 3.6225

# field_ylims = [0,100]
# current_ylims = [0,150]
# rho_ylims = [0,600]

# # 3 to 1 ratio
# t_lo_lims = [80,120] 
# t_hi_lims = [4.3,4.7] 

# plot_and_fit(file_name,sample_len,sample_area,field_ylims,current_ylims,rho_ylims,t_lo_lims,t_hi_lims)


