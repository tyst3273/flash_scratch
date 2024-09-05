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

def _obj_func_decay(time,sig_c,sig_0,tau):
    return sig_c - sig_0 * np.exp(-time/tau)

def _obj_func(time,sig_c,sig_0,tau):
    return sig_c + sig_0 * np.exp(time/tau)
        
# --------------------------------------------------------------------------------------------------

def fit_conductivity(time,conductivity,lims,buffer=10,decay=True):
    
    """
    fit exponential decay constant for current:
        
    sig = sig_c + sig_0 * exp(-t/tau)
    """
    
    _lo = np.flatnonzero(time >= lims[0]).min()
    _hi = np.flatnonzero(time <= lims[1]).max()
    
    _inds = np.flatnonzero(conductivity == 0)    
    _inds = np.r_[_inds,np.flatnonzero(np.isnan(conductivity))]
    _inds = np.r_[_inds,np.flatnonzero(np.isinf(conductivity))]
    _inds = np.unique(_inds)
    
    _t = np.delete(time,_inds)
    _c = np.delete(conductivity,_inds)
    
    _t = _t[_lo+buffer:_hi-buffer] 
    _c = _c[_lo+buffer:_hi-buffer]
    _const = max(_c[0],_c[-1])
    # print(_const)
    
    _inds = np.flatnonzero((_const - _c)/_const > 0.2 )

    _t = _t[_inds]
    _c = _c[_inds]
    
    _t0 = _t[0]
    _t -= _t0
    
    if decay:
        f = _obj_func_decay
    else:
        f = _obj_func
        
    params = [1,100,10]
    popt, pcov = curve_fit(f,_t,_c,p0=params)
    
    perr = np.sqrt(np.diag(pcov))
    
    sig_c, sig_0, tau = popt
    err_c, err_0, err_tau = perr
    fit = f(_t,*popt)
    
    # plt.plot(_t,_c,c='r')
    # plt.plot(_t,fit,c='b')
    # exit()
    
    return _t, _t0, sig_c, sig_0, tau, fit, err_c, err_0, err_tau

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
    
    print(temp,environ)
    
    # ----------------------------------------------------------------------------------------------
    # fit the data in the low-lims, hi-lims ranges
    
    lo_t, lo_t0, lo_sig_c, lo_sig_0, lo_tau, lo_fit, lo_err_c, lo_err_0, lo_err_tau = \
            fit_conductivity(time,conductivity,t_lo_lims,decay=False)
            
    hi_t, hi_t0, hi_sig_c, hi_sig_0, hi_tau, hi_fit, hi_err_c, hi_err_0, hi_err_tau = \
            fit_conductivity(time,conductivity,t_hi_lims,decay=True)
        
    # ----------------------------------------------------------------------------------------------
    
    # interpolate out 0's for plotting
    # replace_zeros(time,field)
    # replace_zeros(time,current_den)
    # replace_zeros(time,conductivity)
    
    fig, ax = plt.subplots(2,2,figsize=(4,5),
                           gridspec_kw={'height_ratios':[1,1],'width_ratios':[1,3],
                                        'hspace':0.125,'wspace':0.1}) 
    
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
    
    rho_ax.plot(lo_t+lo_t0,lo_fit,c='k',ls=(0,(4,1,1,1)),lw=2)
    
    rho_twin.plot(hi_t+hi_t0,hi_fit,c='k',ls=(0,(4,1,1,1)),lw=2)
    
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
    
    # field_ax.spines.left.set_color('b')
    # current_ax.spines.left.set_color('b')
    # field_twin.spines.right.set_color('r')
    # current_twin.spines.right.set_color('r')
    # rho_ax.spines.left.set_color('m')
    
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
    
    # rho_ax.set_yscale('log')
    # rho_twin.set_yscale('log')
    
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
    
    field_twin.annotate(rf'T={temp}$^\circ$C, flashed in {environ}',xy=(0.125,0.85),xycoords='axes fraction',
                        fontsize='large')
    
    rho_twin.annotate(rf'$\sigma(t)\sim \exp(-t/\tau)$',xy=(0.3,0.85),xycoords='axes fraction',
                        fontsize='large')
    rho_twin.annotate(rf'$\tau=${hi_tau*60:.3f} s',xy=(0.4,0.7),xycoords='axes fraction',
                        fontsize='large')
    
    
    fig_name = f'TiO2_{temp}_C_{environ}.png'
    plt.savefig(fig_name,dpi=300,bbox_inches='tight')
    
    
# --------------------------------------------------------------------------------------------------


file_name = 'TiO2/900C-400C-TiO2-Air-new-3.csv'
sample_len = 0.204
sample_area = 1.6775

field_ylims = [0,600]
current_ylims = [0,150]
rho_ylims = [0,70]

t_lo_lims = [0,1/3] # should be 1/3 long as high lim
t_hi_lims = [46,47] 

plot_and_fit(file_name,sample_len,sample_area,field_ylims,current_ylims,rho_ylims,t_lo_lims,t_hi_lims)




