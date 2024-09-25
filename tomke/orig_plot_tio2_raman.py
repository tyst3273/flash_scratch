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
import os

mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = r"\usepackage{bm}"

# mpl.rcParams['mathtext.fontset'] = 'stix'
# mpl.rcParams['font.family'] = 'STIXGeneral'

kB = 8.617333e-5 # eV/K
h = 4.135667e-15 # eV*s
c = c = 299792458 # m/s
invcm_2_eV = 0.001/8.064516

lambda_0 = 532 
E_laser = h*c/(lambda_0*10**(-9))

# --------------------------------------------------------------------------------------------------

def get_temps(anti_params,stokes_params,anti_err,stokes_err):

    temps = []
    temp_errs = []
    
    for ii in range(len(anti_params)):
        
        anti = anti_params[ii]
        stokes = stokes_params[ii]
        
        A_as = np.abs(anti[1])
        A_s = np.abs(stokes[1])
        ratio = 0.786*A_as/A_s
        # ratio = A_as/A_s
        
        err_as = anti_err[ii][1]/A_as
        err_s = stokes_err[ii][1]/A_s
        err_ratio = (err_s+err_as)*ratio
        
        w_as = -anti[2]
        w_s = stokes[2]
        w = (w_s - w_as)/2
        
        E = w*invcm_2_eV 
        T = -E / (kB * np.log(ratio / (((E_laser+E)/(E_laser-E))**4 )))
        
        T_err = -E*(-1)*(kB*(np.log(ratio)-np.log((E_laser+E)/(E_laser-E))**4))**(-2) \
                *(kB*1/ratio-kB)*err_ratio
        
        temps.append(T)
        temp_errs.append(T_err)
     
    return temps, temp_errs

# --------------------------------------------------------------------------------------------------

def _obj_func(w,s,A,w0,Gamma):
    return s + A * w * Gamma / ((w**2 - w0**2)**2 +  ( w * Gamma)**2)

# --------------------------------------------------------------------------------------------------

def fit_data(data_set):

    params = []
    errs = []
    fits = []

    for dd in data_set:

        x = dd[:,0]
        y = dd[:,1]
        
        x = np.abs(x)

        _inds = np.flatnonzero(x < 550)
        _inds = np.intersect1d(_inds,np.flatnonzero(x > 450))

        x = x[_inds]
        y = y[_inds]

        # shift, A, w0, Gamma
        p0 = [1,500,520,3]
        popt, pcov = curve_fit(_obj_func,np.abs(x),y,p0)

        fit = _obj_func(x,*popt)

        popt[-1] = np.abs(popt[-1])

        params.append(popt)
        errs.append(np.sqrt(np.diag(pcov)))
        fits.append([x,fit])

    return params, errs, fits

# --------------------------------------------------------------------------------------------------

def plot_and_fit(directory,sample_len,sample_area):
    
    _files = os.listdir(directory)
    
    stokes = []
    anti = []
    
    for f in _files:
        if f.startswith('AS'):
            anti.append(f)
        else:
            stokes.append(f)
            
    currents = sorted([int(_.split('_')[1].strip('mA.txt')) for _ in stokes])
    
    anti = [f'AS_{ii}mA.txt' for ii in currents]
    stokes = [f'S_{ii}mA.txt' for ii in currents]

    anti_data = []
    for f in anti:
        data = np.loadtxt(os.path.join(directory,f))
        anti_data.append(data)
    
    stokes_data = []
    for f in stokes:
        data = np.loadtxt(os.path.join(directory,f))
        stokes_data.append(data)
        
    anti_params, anti_errs, anti_fits = fit_data(anti_data)
    stokes_params, stokes_errs, stokes_fits = fit_data(stokes_data)  
    
    temps, temp_errs = get_temps(anti_params,stokes_params,anti_errs,stokes_errs)
    
    fig, ax = plt.subplots(1,2,figsize=(4,3),
                           gridspec_kw={'height_ratios':[1],'width_ratios':[1,1],
                                        'hspace':0.1,'wspace':0.1}) 
    
    anti_ax = ax[0]; stokes_ax = ax[1]
        
    c = plt.cm.rainbow(np.linspace(0,1,len(currents)))
    shift = 10
    for ii, cc in enumerate(currents):
        
        d = stokes_data[ii]
        s = stokes_params[ii][0]
        stokes_ax.plot(d[:,0],d[:,1]+ii*shift-s,marker='o',ms=2,lw=1,zorder=1000-ii,c=c[ii])
        stokes_ax.plot([-1000,1000],[ii*shift,ii*shift],ms=0,lw=1,ls=(0,(1,1)),c=(0.25,0.25,0.25))
        x, y = stokes_fits[ii]
        stokes_ax.plot(x,y+ii*shift-s,marker='o',ms=0,lw=1,zorder=1000-ii,c='k')
        
        d = anti_data[ii]
        s = anti_params[ii][0]
        anti_ax.plot(d[:,0],d[:,1]+ii*shift-s,marker='o',ms=2,lw=1,zorder=1000-ii,c=c[ii])
        anti_ax.plot([-1000,1000],[ii*shift,ii*shift],ms=0,lw=1,ls=(0,(1,1)),c=(0.25,0.25,0.25))
        x, y = anti_fits[ii]
        anti_ax.plot(-x,y+ii*shift-s,marker='o',ms=0,lw=1,zorder=1000-ii,c='k')
        
        T = temps[ii]
        stokes_ax.annotate(rf'T={T:.1f}K',fontsize='medium',
                            xy=(432,ii*shift+3),xycoords='data',zorder=1000,
                            annotation_clip=False)
        anti_ax.annotate(rf'{cc:d} mA',fontsize='medium',
                            xy=(-468,ii*shift+3),xycoords='data',zorder=1000)
        
    axes = [anti_ax, stokes_ax]
    
    for _ax in axes:
        for axis in ['top','bottom','left','right']:
            _ax.spines[axis].set_linewidth(1.5)
        _ax.minorticks_on()
        _ax.tick_params(which='both',width=1,labelsize='large')
        _ax.tick_params(which='major',length=5)
        _ax.tick_params(which='minor',length=2)
        _ax.set_rasterized = True
    
    anti_ax.spines.right.set_visible(False)
    anti_ax.tick_params(axis='y',which='both',right=False,labelright=False)

    stokes_ax.spines.left.set_visible(False)
    stokes_ax.tick_params(axis='y',which='both',left=False,labelleft=False)
    
    d = 0.1  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-d, -d), (d, d)], markersize=5,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    
    anti_ax.plot((1,1), (0,0), transform=anti_ax.transAxes, **kwargs)
    anti_ax.plot((1,1), (1,1), transform=anti_ax.transAxes, **kwargs)
    stokes_ax.plot((0,0), (0,0), transform=stokes_ax.transAxes, **kwargs)
    stokes_ax.plot((0,0), (1,1), transform=stokes_ax.transAxes, **kwargs)
    
    anti_ax.plot((1,1), (0,0), transform=anti_ax.transAxes, **kwargs)
    anti_ax.plot((1,1), (1,1), transform=anti_ax.transAxes, **kwargs)
    stokes_ax.plot((0,0), (0,0), transform=stokes_ax.transAxes, **kwargs)
    stokes_ax.plot((0,0), (1,1), transform=stokes_ax.transAxes, **kwargs)
    
    anti_ax.plot([1,1],[0,1], transform=anti_ax.transAxes, lw=1, ls=(0,(4,1,2,1)),ms=0, c='k')
    stokes_ax.plot([0,0],[0,1], transform=stokes_ax.transAxes, lw=1, ls=(0,(4,1,2,1)),ms=0, c='k')
    
    xlim = [-550,-430]
    anti_ax.set_xlim(xlim)
    
    xlim = [430,550]
    stokes_ax.set_xlim(xlim)
    
    ylim = [-2,65]
    anti_ax.set_ylim(ylim)
    stokes_ax.set_ylim(ylim)
    
    # anti_off_ax.xaxis.set_major_formatter(FormatStrFormatter('%1d'))
    # stokes_off_ax.xaxis.set_major_formatter(FormatStrFormatter('%1d'))
    
    # anti_ax.annotate(rf'fan on',xy=(0.1,0.9),xycoords='axes fraction',fontsize='large')  
    anti_ax.set_ylabel('Intensity [arb. units]',fontsize='large',labelpad=5)
   
    
    fig.supxlabel('Raman shift [1/cm]',fontsize='large',y=-0.04)
    
    fig.suptitle(r'Si on flashing TiO$_2$',fontsize='large',y=0.96)
    
    fig_name = 'si_on_tio2_raman.png'
    plt.savefig(fig_name,dpi=300,bbox_inches='tight')
    
    w0 = []
    w0_err = []
    g = []
    g_err = []
    for ii in range(len(currents)):
        
        w_s = stokes_params[ii][2]
        w_as = anti_params[ii][2]
        w = (w_s+w_as)/2
        w0.append(w)
        
        e_s = stokes_errs[ii][2]
        e_as = anti_errs[ii][2]
        e = np.sqrt(e_s**2+e_as**2) #/2
        w0_err.append(e)
        
        g_s = stokes_params[ii][3]
        g_as = anti_params[ii][3]
        _g = (g_s+g_as)/2
        g.append(_g)
        
        e_s = stokes_errs[ii][3]
        e_as = anti_errs[ii][3]
        e = np.sqrt(e_s**2+e_as**2) #/2
        g_err.append(e)
        
        
    data = np.c_[temps,temp_errs,w0,w0_err,g,g_err,currents]
    np.savetxt('si_on_tio2_data.txt',data,fmt='%.6f',header='T [K], dT [K], w0 [1/cm], dw0 [1/cm], G [1/cm], dG [1/cm], I [mA]')
        
    return temps, temp_errs, w0, w0_err, g, g_err
 
# --------------------------------------------------------------------------------------------------

directory = '20240809/corrected Raman data export'
sample_len = 0.220
sample_area = 1.9345 # mm^2

temps, temp_errs, w0, w0_err, g, g_err = plot_and_fit(directory,sample_len,sample_area)
    
# plot_vs_temps(on_temps, on_errs, off_temps, off_errs,
#     on_w0, on_w0_err, on_g, on_g_err, off_w0, off_w0_err, off_g, off_g_err)


