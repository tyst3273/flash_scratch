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

def _obj_func_lo(time, sig_c, sig_l, sig_0, tau, t0):
    t = time-t0
    return sig_c + sig_l * t + sig_0 * np.exp(t/tau) 

def _obj_func_hi(time, sig_c, sig_l, sig_0, tau, t0):
    t = time-t0
    return sig_0 * np.exp(-t/tau) + sig_c - sig_l * t
        
# --------------------------------------------------------------------------------------------------

def fit_lo(time,conductivity,lims):
    
    """
    fit exponential decay constant for current: add a linear term for early
    stage of incubation
        
    sig = sig_c + sig_l*t + sig_0 * exp(-t/tau)
    """
    
    _lo = np.flatnonzero(time >= lims[0]).min()
    _hi = np.flatnonzero(time <= lims[1]).max()
    
    _t = time[_lo:_hi] 
    _c = conductivity[_lo:_hi]
    plt.plot(_t,_c,marker='o',ms=4,c='m')
    
    f = _obj_func_lo
    
    # sig_c, sig_l, sig_0, tau, t0
    params = [1,10,50,0.2,0.15]
    popt, pcov = curve_fit(f,_t,_c,p0=params)
    
    perr = np.sqrt(np.diag(pcov))
    
    sig_c, sig_l, sig_0, tau, t0 = popt
    err_c, err_l, err_0, err_tau, err_t0 = perr
    fit = f(_t,*popt)
    
    return _t, t0, sig_c, sig_l, sig_0, tau, fit, err_c, err_0, err_tau

# --------------------------------------------------------------------------------------------------

def fit_hi(time,conductivity,lims):
    
    """
    fit exponential decay constant for current: add a linear term for early
    stage of incubation
        
    sig = sig_hi( 1 - exp(-t/tau) ) + sig_lo
    """
    
    _lo = np.flatnonzero(time >= lims[0]).min()
    _hi = np.flatnonzero(time <= lims[1]).max()
    
    _t = time[_lo:_hi] 
    _c = conductivity[_lo:_hi]
    plt.plot(_t,_c,marker='o',ms=4,c='m')
    
    f = _obj_func_hi
    
    # sig_c, sig_l, sig_0, tau, t0 
    params = [5,1,50,1,-0.1]
    popt, pcov = curve_fit(f,_t,_c,p0=params)
    
    perr = np.sqrt(np.diag(pcov))
    
    sig_c, sig_l, sig_0, tau, t0 = popt
    err_c, err_l, err_0, err_tau, err_t0 = perr
    fit = f(_t,*popt)
    
    return _t, t0, sig_c, sig_l, sig_0, tau, fit, err_c, err_0, err_tau

# --------------------------------------------------------------------------------------------------

def parse_file_name(file_name):
    
    _f = file_name.split('-')
    
    temp = _f[1].strip('C')
    environ = _f[3].lower()
    
    return temp, environ

# --------------------------------------------------------------------------------------------------

def plot_and_fit(file_name,sample_len,sample_area,xlims,ylims,which):
    
        
    current, current_den, voltage, field, resistivity, conductivity, time = \
            get_data(file_name,sample_len,sample_area)
            
    temp, environ = parse_file_name(file_name)
    
    # ----------------------------------------------------------------------------------------------
    # fit the data in the low-lims, hi-lims ranges
    
    if which == 'lo':
        t, t0, sig_c, sig_l, sig_0, tau, fit, err_c, err_0, err_tau = fit_lo(time,conductivity,xlims)
        _label = f'{temp}_{environ}_{which}'
        _m = f"'{_label}':dict(t0={t0:.3f},sig_c={sig_c:.3f},sig_l={sig_l:.3f},\n"\
            f"sig_0={sig_0:.3f},tau={tau:.6f},err_tau={err_tau:.6f})\n"
        print(_m)
        
        d = 0.01
        
    elif which == 'hi':
        t, t0, sig_c, sig_l, sig_0, tau, fit, err_c, err_0, err_tau = fit_hi(time,conductivity,xlims)
        _label = f'{temp}_{environ}_{which}'
        _m = f"'{_label}':dict(t0={t0:.3f},sig_c={sig_c:.3f},sig_l={sig_l:.3f},\n"\
            f"sig_0={sig_0:.3f},tau={tau:.6f},err_tau={err_tau:.6f})\n"
        print(_m)
        
        d = 0.1
            
    # ----------------------------------------------------------------------------------------------
    
    fig, ax = plt.subplots(figsize=(4,4),gridspec_kw={'hspace':0.125,'wspace':0.2}) 
    
    ax.plot(time,conductivity,lw=1,marker='o',ms=4,c='m')
    
    ax.plot(t,fit,c='k',ls=(0,(4,1,1,1)),lw=2)

    axes = [ax]
    
    for _ax in axes:
        for axis in ['top','bottom','left','right']:
            _ax.spines[axis].set_linewidth(1.5)
        _ax.minorticks_on()
        _ax.tick_params(which='both',width=1,labelsize='large')
        _ax.tick_params(which='major',length=5)
        _ax.tick_params(which='minor',length=2)
        _ax.set_rasterized = True
    
    ax.set_xlim(xlims[0]-d,xlims[1]+d)
    ax.set_ylim(ylims)
    
    ax.set_title(rf'T={temp}$^\circ$C, flashed in {environ}',fontsize='large')
    
    if which == 'lo':
        ax.annotate(r'$\sigma(t)\sim \exp(t/\tau)$',xy=(0.3,0.9),xycoords='axes fraction',
                            fontsize='large')
    else:
        ax.annotate(r'$\sigma(t)\sim \exp(-t/\tau)$',xy=(0.3,0.9),xycoords='axes fraction',
                            fontsize='large')
        
    ax.annotate(rf'$\tau=$ {tau*60:.3f} $\pm$ {err_tau*60:.3f} s',xy=(0.35,0.825),xycoords='axes fraction',
                        fontsize='large')
    
    ax.set_ylabel(r'$\sigma$ [($\Omega$-cm)$^{-1}$]',fontsize='large',labelpad=5,c='m')
    ax.set_xlabel('Time [m]',fontsize='large',y=-0.05)
    
    fig_name = f'TiO2_{temp}_C_{environ}_{which}_fit.png'
    plt.savefig(fig_name,dpi=300,bbox_inches='tight')
    
# --------------------------------------------------------------------------------------------------

file_name = 'Si/Room-Si-Flash-attempt-1.csv'
sample_len = 0.592
sample_area = 1.7394

ylims = [0,600]
xlims = [0.42,0.515] 
which = 'lo'
plot_and_fit(file_name,sample_len,sample_area,xlims,ylims,which)

ylims = [0,600]
xlims = [4.375,4.7] 
which = 'hi'
plot_and_fit(file_name,sample_len,sample_area,xlims,ylims,which)
