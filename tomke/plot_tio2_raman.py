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
invcm_2_eV = 0.001/8.064516

lambda_0 = 532 # nm
nu_0 = 1/(lambda_0*1e-9)/100 # 1/cm
E_laser = nu_0 * invcm_2_eV

# --------------------------------------------------------------------------------------------------    

def bose(x,T,scale=1.138):
    
    return 1/(np.exp(np.abs(x)*invcm_2_eV/(kB*T*scale))-1)
    
# --------------------------------------------------------------------------------------------------    

def _obj_func_better(w,s1,s2,A,w0,Gamma,T):
    
    aw = w[np.flatnonzero(w < 0)] 
    sw = w[np.flatnonzero(w > 0)] 
    
    aw = aw[np.argsort(np.abs(aw))]
    aw = np.abs(aw)
    
    ab = bose(aw,T)
    sb = bose(sw,T)+1
    
    ay = s1 + A * aw * Gamma / ((aw**2 - w0**2)**2 +  ( aw * Gamma)**2) * ab #* (nu_0 - aw)**4
    sy = s2 + A * sw * Gamma / ((sw**2 - w0**2)**2 +  ( sw * Gamma)**2) * sb #* (nu_0 + sw)**4
    
    f = np.r_[ay,sy]
    
    return f

# --------------------------------------------------------------------------------------------------

def fit_data_better(stokes_data,anti_data):

    params = []
    errs = []
    fits = []
    
    for ii in range(len(stokes_data)):

        ad = anti_data[ii]
        sd = stokes_data[ii]
        
        ax = ad[:,0]
        ay = ad[:,1]
        _inds = np.flatnonzero(np.abs(ax) < 550)
        _inds = np.intersect1d(_inds,np.flatnonzero(np.abs(ax) > 450))
        ax = ax[_inds]
        ay = ay[_inds]
        
        sx = sd[:,0]
        sy = sd[:,1]
        sx = np.abs(sx)
        _inds = np.flatnonzero(sx < 650)
        _inds = np.intersect1d(_inds,np.flatnonzero(sx > 450))
        sx = sx[_inds]
        sy = sy[_inds]
        
        x = np.r_[ax,sx]
        y = np.r_[ay,sy]

        # s1, s2, A, w0, Gamma, temp
        p0 = [1 , 1, 500, 500. , 5. , 3000. ]
        popt, pcov = curve_fit(_obj_func_better,x,y,p0)
        fit = _obj_func_better(x,*popt)
        
        ax = x[np.flatnonzero(x < 0)] 
        ay = fit[np.flatnonzero(x < 0)]
        sx = x[np.flatnonzero(x > 0)] 
        sy = fit[np.flatnonzero(x > 0)]
        # ax = ax[np.argsort(np.abs(ax))]
        # ax = np.abs(ax)
        
        # plt.plot(ax,ay,c='r',lw=0,marker='o',ms=2)
        # plt.plot(sx,sy,c='r',lw=0,marker='o',ms=2)
        # plt.plot(x,fit,c='b',lw=0,marker='o',ms=2)
        
        params.append(popt)
        errs.append(np.sqrt(np.diag(pcov)))
        fits.append([ax,ay,sx,sy])

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
        
    params, errs, fits = fit_data_better(stokes_data,anti_data)
    
    fig, ax = plt.subplots(1,2,figsize=(4,3),
                           gridspec_kw={'height_ratios':[1],'width_ratios':[1,1],
                                        'hspace':0.1,'wspace':0.1}) 
    
    anti_ax = ax[0]; stokes_ax = ax[1]
        
    c = plt.cm.rainbow(np.linspace(0,1,len(currents)))
    shift = 10
    for ii, cc in enumerate(currents):
        
        d = stokes_data[ii]
        s = params[ii][1]
        stokes_ax.plot(d[:,0],d[:,1]+ii*shift-s,marker='o',ms=2,lw=1,zorder=1000-ii,c=c[ii])
        stokes_ax.plot([-1000,1000],[ii*shift,ii*shift],ms=0,lw=1,ls=(0,(1,1)),c=(0.25,0.25,0.25))
        x = fits[ii][2]; y = fits[ii][3]
        stokes_ax.plot(x,y+ii*shift-s,marker='o',ms=0,lw=1,zorder=1000-ii,c='k')
        
        d = anti_data[ii]
        s = params[ii][0]
        anti_ax.plot(d[:,0],d[:,1]+ii*shift-s,marker='o',ms=2,lw=1,zorder=1000-ii,c=c[ii])
        anti_ax.plot([-1000,1000],[ii*shift,ii*shift],ms=0,lw=1,ls=(0,(1,1)),c=(0.25,0.25,0.25))
        x = fits[ii][0]; y = fits[ii][1]
        anti_ax.plot(x,y+ii*shift-s,marker='o',ms=0,lw=1,zorder=1000-ii,c='k')
        
        T = params[ii][5]
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
    temps = []
    temp_errs = []
    g = []
    g_err = []
    for ii in range(len(currents)):
        
        _w = params[ii][3]
        w0.append(_w)
        
        _e = errs[ii][3]
        w0_err.append(_e)
        
        _g = params[ii][4]
        g.append(_g)
        
        _e = errs[ii][4]
        g_err.append(_e)
        
        _T = params[ii][5]
        temps.append(_T)

        _e = errs[ii][5]
        temp_errs.append(_e)
          
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


