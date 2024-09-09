#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 09:51:26 2024

@author: ty
"""

ref_w_T = [29.457364341085253, 89.92248062015503, 305.4263565891473, 389.14728682170545, 457.36434108527135, 536.4341085271318, 632.5581395348838, 772.0930232558139]
ref_w = [523.2428115015974, 523.6900958466454, 519.5686900958466, 518.8019169329074, 516.4376996805112, 513.9456869009584, 513.6900958466454, 509.6645367412141]

ref_g_T = [26.044226044226036, 85.01228501228502, 306.14250614250614, 391.64619164619165, 460.9336609336609, 543.4889434889435, 642.2604422604422, 785.2579852579853]
ref_g = [2.2098298676748582, 2.6030245746691874, 4.054820415879018, 5.0529300567107756, 5.521739130434783, 6.413988657844991, 7.306238185255198, 8.984877126654064]

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
        
        A_as = np.abs(anti[1])#-anti[0]
        A_s = np.abs(stokes[1])#-stokes[0]
        ratio = A_as/A_s
        
        err_as = anti_err[ii][1]/A_as
        err_s = stokes_err[ii][1]/A_s
        err_ratio = (err_s+err_as)*ratio
        
        w_as = anti[2]
        w_s = stokes[2]
        w = (w_s - w_as)/2
        
        E = w*invcm_2_eV 
        T = -E / (kB * np.log(ratio / (((E_laser+E)/(E_laser-E))**4 )))
        
        T_err = -E*(-1)*(kB*(np.log(ratio)-np.log((E_laser+E)/(E_laser-E))**4))**(-2)*(kB*1/ratio-kB)*err_ratio
        
        temps.append(T)
        temp_errs.append(T_err)
    
    return temps, temp_errs

# --------------------------------------------------------------------------------------------------

def _obj_func(w,s,A,w0,Gamma):
    return s + A * np.abs(w * Gamma) / ((w**2 - w0**2)**2 +  ( w * Gamma)**2)
        
# --------------------------------------------------------------------------------------------------

def fit_data(data_set):
    
    params = []
    errs = []
    fits = []
    
    for dd in data_set:
        
        x = dd[:,0]
        y = dd[:,1]
            
        # shift, A, w0, Gamma
        p0 = [0.1,20,520,1]
        popt, pcov = curve_fit(_obj_func,np.abs(x),y,p0)
        
        if x[0] < 0:
            popt[2] *= -1
        
        fit = _obj_func(x,*popt)
         
        popt[-1] = np.abs(popt[-1])
        # plt.show()
        # exit()
        
        params.append(popt)
        errs.append(np.sqrt(np.diag(pcov)))
        fits.append([x,fit])
        
    return params, errs, fits

# --------------------------------------------------------------------------------------------------

def plot_and_fit(directory,sample_len,sample_area):
    
    _files = os.listdir(directory)
    stokes_off = []
    anti_off = []
    stokes_on = []
    anti_on = []
    for f in _files:
        if f.endswith('fan_off.txt'):
            if f.startswith('AS'):
                anti_off.append(f)
            else:
                stokes_off.append(f)
        else:
            if f.startswith('AS'):
                anti_on.append(f)
            else:
                stokes_on.append(f)
    
    on_currents = sorted([int(_.split('_')[1].strip('mA')) for _ in stokes_on])
    off_currents = sorted([int(_.split('_')[1].strip('mA')) for _ in stokes_off])
    
    anti_on = [f'AS_{ii}mA_fan_on.txt' for ii in on_currents]
    stokes_on = [f'S_{ii}mA_fan_on.txt' for ii in on_currents]
    
    anti_off = [f'AS_{ii}mA_fan_off.txt' for ii in off_currents]
    stokes_off = [f'S_{ii}mA_fan_off.txt' for ii in off_currents]

    anti_on_data = []
    for f in anti_on:
        data = np.loadtxt(os.path.join(directory,f))
        anti_on_data.append(data)
        
    anti_off_data = []
    for f in anti_off:
        data = np.loadtxt(os.path.join(directory,f))
        anti_off_data.append(data)
    
    stokes_on_data = []
    for f in stokes_on:
        data = np.loadtxt(os.path.join(directory,f))
        stokes_on_data.append(data)
        
    stokes_off_data = []
    for f in stokes_off:
        data = np.loadtxt(os.path.join(directory,f))
        stokes_off_data.append(data)
      
    anti_on_params, anti_on_errs, anti_on_fits = fit_data(anti_on_data)
    stokes_on_params, stokes_on_errs, stokes_on_fits = fit_data(stokes_on_data)  
    anti_off_params, anti_off_errs, anti_off_fits = fit_data(anti_off_data)
    stokes_off_params, stokes_off_errs, stokes_off_fits = fit_data(stokes_off_data)
    
    on_temps, on_errs = get_temps(anti_on_params,stokes_on_params,anti_on_errs,stokes_on_errs)
    off_temps, off_errs = get_temps(anti_off_params,stokes_off_params,anti_off_errs,stokes_off_errs)
    
    fig, ax = plt.subplots(2,2,figsize=(4,5),
                           gridspec_kw={'height_ratios':[1,1],'width_ratios':[1,1],
                                        'hspace':0.1,'wspace':0.1}) 
    
    anti_on_ax = ax[0,0]; stokes_on_ax = ax[0,1]
    anti_off_ax = ax[1,0]; stokes_off_ax = ax[1,1]
        
    c = plt.cm.rainbow(np.linspace(0,1,len(on_currents)))
    shift = 10
    for ii, cc in enumerate(on_currents):
        
        d = stokes_on_data[ii]
        stokes_on_ax.plot(d[:,0],d[:,1]+ii*shift,marker='o',ms=2,lw=1,zorder=1000-ii,c=c[ii])
        stokes_on_ax.plot([-1000,1000],[ii*shift,ii*shift],ms=0,lw=1,ls=(0,(1,1)),c=(0.25,0.25,0.25))
        x, y = stokes_on_fits[ii]
        stokes_on_ax.plot(x,y+ii*shift,marker='o',ms=0,lw=1,zorder=1000-ii,c='k')
        
        d = anti_on_data[ii]
        anti_on_ax.plot(d[:,0],d[:,1]+ii*shift,marker='o',ms=2,lw=1,zorder=1000-ii,c=c[ii])
        anti_on_ax.plot([-1000,1000],[ii*shift,ii*shift],ms=0,lw=1,ls=(0,(1,1)),c=(0.25,0.25,0.25))
        x, y = anti_on_fits[ii]
        anti_on_ax.plot(x,y+ii*shift,marker='o',ms=0,lw=1,zorder=1000-ii,c='k')
        
        T = on_temps[ii]
        stokes_on_ax.annotate(rf'T={T:.1f}K',fontsize='medium',
                            xy=(445,ii*shift+2),xycoords='data')
        anti_on_ax.annotate(rf'{cc:d} mA',fontsize='medium',
                            xy=(-480,ii*shift+2),xycoords='data')
        
    c = plt.cm.rainbow(np.linspace(0,1,len(off_currents)))
    shift = 10
    for ii, cc in enumerate(off_currents):
        
        d = stokes_off_data[ii]
        stokes_off_ax.plot(d[:,0],d[:,1]+ii*shift,marker='o',ms=2,lw=1,zorder=1000-ii,c=c[ii])
        stokes_off_ax.plot([-1000,1000],[ii*shift,ii*shift],ms=0,lw=1,ls=(0,(1,1)),c=(0.25,0.25,0.25))
        x, y = stokes_off_fits[ii]
        stokes_off_ax.plot(x,y+ii*shift,marker='o',ms=0,lw=1,zorder=1000-ii,c='k')
        
        d = anti_off_data[ii]
        anti_off_ax.plot(d[:,0],d[:,1]+ii*shift,marker='o',ms=2,lw=1,zorder=1000-ii,c=c[ii])
        anti_off_ax.plot([-1000,1000],[ii*shift,ii*shift],ms=0,lw=1,ls=(0,(1,1)),c=(0.25,0.25,0.25))
        x, y = anti_off_fits[ii]
        anti_off_ax.plot(x,y+ii*shift,marker='o',ms=0,lw=1,zorder=1000-ii,c='k')
        
        T = off_temps[ii]
        stokes_off_ax.annotate(rf'T={T:.1f}K',fontsize='medium',
                            xy=(445,ii*shift+2),xycoords='data')
        anti_off_ax.annotate(rf'{cc:d} mA',fontsize='medium',
                            xy=(-480,ii*shift+2),xycoords='data')

    axes = [anti_on_ax, stokes_on_ax, anti_off_ax, stokes_off_ax]
    
    for _ax in axes:
        for axis in ['top','bottom','left','right']:
            _ax.spines[axis].set_linewidth(1.5)
        _ax.minorticks_on()
        _ax.tick_params(which='both',width=1,labelsize='large')
        _ax.tick_params(which='major',length=5)
        _ax.tick_params(which='minor',length=2)
        _ax.set_rasterized = True
    
    anti_on_ax.spines.right.set_visible(False)
    anti_on_ax.tick_params(axis='y',which='both',right=False,labelright=False)
    anti_off_ax.spines.right.set_visible(False)
    anti_off_ax.tick_params(axis='y',which='both',right=False,labelright=False)
    
    stokes_on_ax.spines.left.set_visible(False)
    stokes_on_ax.tick_params(axis='y',which='both',left=False,labelleft=False)
    stokes_off_ax.spines.left.set_visible(False)
    stokes_off_ax.tick_params(axis='y',which='both',left=False,labelleft=False)
    
    d = 0.1  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-d, -d), (d, d)], markersize=5,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    
    anti_on_ax.plot((1,1), (0,0), transform=anti_on_ax.transAxes, **kwargs)
    anti_on_ax.plot((1,1), (1,1), transform=anti_on_ax.transAxes, **kwargs)
    stokes_on_ax.plot((0,0), (0,0), transform=stokes_on_ax.transAxes, **kwargs)
    stokes_on_ax.plot((0,0), (1,1), transform=stokes_on_ax.transAxes, **kwargs)
    
    anti_on_ax.plot((1,1), (0,0), transform=anti_on_ax.transAxes, **kwargs)
    anti_on_ax.plot((1,1), (1,1), transform=anti_on_ax.transAxes, **kwargs)
    stokes_on_ax.plot((0,0), (0,0), transform=stokes_on_ax.transAxes, **kwargs)
    stokes_on_ax.plot((0,0), (1,1), transform=stokes_on_ax.transAxes, **kwargs)
    
    anti_on_ax.plot([1,1],[0,1], transform=anti_on_ax.transAxes, lw=1, ls=(0,(4,1,2,1)),ms=0, c='k')
    anti_off_ax.plot([1,1],[0,1], transform=anti_off_ax.transAxes, lw=1, ls=(0,(4,1,2,1)),ms=0, c='k')
    stokes_on_ax.plot([0,0],[0,1], transform=stokes_on_ax.transAxes, lw=1, ls=(0,(4,1,2,1)),ms=0, c='k')
    stokes_off_ax.plot([0,0],[0,1], transform=stokes_off_ax.transAxes, lw=1, ls=(0,(4,1,2,1)),ms=0, c='k')
    
    xlim = [-550,-440]
    anti_on_ax.set_xlim(xlim)
    anti_off_ax.set_xlim(xlim)
    
    xlim = [440,550]
    stokes_on_ax.set_xlim(xlim)
    stokes_off_ax.set_xlim(xlim)
    
    ylim = [-5,70]
    anti_on_ax.set_ylim(ylim)
    stokes_on_ax.set_ylim(ylim)
    
    ylim = [-5,80]
    anti_off_ax.set_ylim(ylim)
    stokes_off_ax.set_ylim(ylim)
    
    anti_off_ax.xaxis.set_major_formatter(FormatStrFormatter('%1d'))
    stokes_off_ax.xaxis.set_major_formatter(FormatStrFormatter('%1d'))
    
    anti_on_ax.set_xticklabels([])
    stokes_on_ax.set_xticklabels([])
    
    anti_on_ax.annotate(rf'fan on',xy=(0.1,0.9),xycoords='axes fraction',fontsize='large')
    anti_off_ax.annotate(rf'fan off',xy=(0.1,0.9),xycoords='axes fraction',fontsize='large')
    
    anti_on_ax.set_ylabel('Intensity [arb. units]',fontsize='large',labelpad=5)
    anti_off_ax.set_ylabel('Intensity [arb. units]',fontsize='large',labelpad=5)
    
    fig.supxlabel('Raman shift [1/cm]',fontsize='large',y=0.02)
    
    fig.suptitle('Flashing Si',fontsize='large',y=0.93)
    
    fig_name = 'si_raman.png'
    plt.savefig(fig_name,dpi=300,bbox_inches='tight')

    on_w0 = []
    on_w0_err = []
    on_g = []
    on_g_err = []
    for ii in range(len(on_currents)):
        
        w_s = stokes_on_params[ii][2]
        w_as = anti_on_params[ii][2]
        w = (w_s-w_as)/2
        on_w0.append(w)
        
        e_s = stokes_on_errs[ii][2]
        e_as = anti_on_errs[ii][2]
        e = np.sqrt(e_s**2+e_as**2)/2
        on_w0_err.append(e)
        
        g_s = stokes_on_params[ii][3]
        g_as = anti_on_params[ii][3]
        g = (g_s+g_as)/2
        on_g.append(g)
        
        e_s = stokes_on_errs[ii][3]
        e_as = anti_on_errs[ii][3]
        e = np.sqrt(e_s**2+e_as**2)/2
        on_g_err.append(e)
     
    off_w0 = []
    off_w0_err = []
    off_g = []
    off_g_err = []
    for ii in range(len(off_currents)):
        
        w_s = stokes_off_params[ii][2]
        w_as = anti_off_params[ii][2]
        w = (w_s-w_as)/2
        off_w0.append(w)
        
        e_s = stokes_off_errs[ii][2]
        e_as = anti_off_errs[ii][2]
        e = np.sqrt(e_s**2+e_as**2) #/2
        off_w0_err.append(e)
        
        g_s = stokes_off_params[ii][3]
        g_as = anti_off_params[ii][3]
        g = (g_s+g_as)/2
        off_g.append(g)
        
        e_s = stokes_off_errs[ii][3]
        e_as = anti_off_errs[ii][3]
        e = np.sqrt(e_s**2+e_as**2) #/2
        off_g_err.append(e)
        
    return on_temps, on_errs, off_temps, off_errs, \
        on_w0, on_w0_err, on_g, on_g_err, off_w0, off_w0_err, off_g, off_g_err, \
        on_currents, off_currents
        
        
# --------------------------------------------------------------------------------------------------

def plot_vs_temps(on_temps, on_errs, off_temps, off_errs, on_w0, on_w0_err, on_g, on_g_err, 
                off_w0, off_w0_err, off_g, off_g_err,on_currents,off_currents):
    
    fig, ax = plt.subplots(2,1,figsize=(4,6),
                           gridspec_kw={'hspace':0.1,'wspace':0.075}) 
    
    w_ax = ax[0]; g_ax = ax[1]
            
    w_ax.errorbar(off_temps,off_w0,yerr=off_w0_err,xerr=off_errs,ms=6,lw=0,
                  c='b',marker='o',label='Si - fan off')
    w_ax.errorbar(on_temps,on_w0,yerr=on_w0_err,xerr=on_errs,ms=6,lw=0,
                  c='r',marker='s',label='Si - fan on',
                  markerfacecolor='none',markeredgewidth=1.5)
    
    g_ax.errorbar(off_temps,off_g,yerr=off_g_err,xerr=off_errs,ms=6,lw=0,
                  c='b',marker='o')
    g_ax.errorbar(on_temps,on_g,yerr=on_g_err,xerr=on_errs,ms=6,lw=0,
                  c='r',marker='s',markerfacecolor='none',markeredgewidth=1.5)
    
    _T, _dT, _w, _dw, _g, _dg, _I = np.loadtxt('si_on_tio2_data.txt',unpack=True)
    w_ax.errorbar(_T,_w,yerr=_dw,xerr=_dT,ms=6,lw=0,elinewidth=1, #ls=(0,(4,2,2,2)),
                  c='m',marker='^',label='Si on TiO2',markerfacecolor=None,
                  markeredgewidth=1.5,zorder=1000)
    g_ax.errorbar(_T,_g,yerr=_dg,xerr=_dT,ms=6,lw=0,elinewidth=1, #ls=(0,(4,2,2,2)),
                  c='m',marker='^',markerfacecolor=None,markeredgewidth=1.5)

    w_ax.plot(ref_w_T,ref_w,marker='x',ms=6,c='k',lw=0,zorder=1000,mew=2,label='ref.')
    g_ax.plot(ref_g_T,ref_g,marker='x',ms=6,c='k',lw=0,zorder=1000,mew=2)
    
    w_ax.legend(frameon=False,fontsize='large',loc='upper right',
                bbox_to_anchor=(1.0,1.0),handletextpad=0.1)
                #labelspacing=0.1,handlelength=0.5,handletextpad=0.7)
    
    _T_fit = np.r_[on_temps,off_temps]
    _w_fit = np.r_[on_w0,off_w0]
    _g_fit = np.r_[on_g,off_g]
    
    _inds = np.argsort(_T_fit)
    _T_fit = _T_fit[_inds]
    _w_fit = _w_fit[_inds]
    _g_fit = _g_fit[_inds]

    _T_plot = np.r_[_T,_T_fit]
    _T_plot = np.sort(_T_plot)
    
    coeff = np.polynomial.polynomial.polyfit(_T_fit,_w_fit,deg=1)

    w_ax.plot(_T_plot,coeff[0]+_T_plot*coeff[1],lw=1,ls=(0,(4,2,2,2)),c='k')
    
    coeff = np.polynomial.polynomial.polyfit(_T_fit,_g_fit,deg=1)

    g_ax.plot(_T_plot,coeff[0]+_T_plot*coeff[1],lw=1,ls=(0,(4,2,2,2)),c='k')
    
    axes = [w_ax,g_ax]
    
    #g_ax.yaxis.tick_right()
    #g_ax.yaxis.set_label_position("right")
    
    for _ax in axes:
        for axis in ['top','bottom','left','right']:
            _ax.spines[axis].set_linewidth(1.5)
        _ax.minorticks_on()
        _ax.tick_params(which='both',width=1,labelsize='large')
        _ax.tick_params(which='major',length=5)
        _ax.tick_params(which='minor',length=2)
        _ax.set_rasterized = True

    xlim = [200,1900]
    w_ax.set_xlim(xlim)
    g_ax.set_xlim(xlim)
    
    ylim = [495,522]
    w_ax.set_ylim(ylim)
    
    ylim = [3,17]
    g_ax.set_ylim(ylim)

    w_ax.set_xticklabels([])
    
    w_ax.set_ylabel('Raman shift [1/cm]',fontsize='large',labelpad=5)
    g_ax.set_ylabel('HWHM [1/cm]',fontsize='large',labelpad=5)
    
    fig.supxlabel('Temperature [K]',fontsize='large',y=0.03)
    
    # fig.suptitle(rf'Flashing Si',fontsize='large',y=0.92)
    
    fig_name = f'si_temps.png'
    plt.savefig(fig_name,dpi=300,bbox_inches='tight')
        
# --------------------------------------------------------------------------------------------------

def plot_vs_currents(on_temps, on_errs, off_temps, off_errs, on_w0, on_w0_err, on_g, on_g_err,
                off_w0, off_w0_err, off_g, off_g_err,on_currents,off_currents):

    fig, ax = plt.subplots(1,figsize=(4,4),
                           gridspec_kw={'hspace':0.1,'wspace':0.075})

    T_ax = ax #; g_ax = ax[1]

    T_ax.errorbar(off_currents,off_temps,yerr=off_errs,ms=6,lw=0,
                  c='b',marker='o',label='Si - fan off')
    T_ax.errorbar(on_currents,on_temps,yerr=on_errs,ms=6,lw=0,
                  c='r',marker='s',label='Si - fan on',
                  markerfacecolor='none',markeredgewidth=1.5)

    _T, _dT, _w, _dw, _g, _dg, _I = np.loadtxt('si_on_tio2_data.txt',unpack=True)
    _I /= 1.9345
    T_ax.errorbar(_I,_T,yerr=_dT,ms=6,lw=0,elinewidth=1, #ls=(0,(4,2,2,2)),
                  c='m',marker='^',label='Si on TiO2',markerfacecolor=None,
                  markeredgewidth=1.5,zorder=1000)

    #w_ax.plot(ref_w_T,ref_w,marker='x',ms=6,c='k',lw=0,zorder=1000,mew=2,label='ref.')
    #g_ax.plot(ref_g_T,ref_g,marker='x',ms=6,c='k',lw=0,zorder=1000,mew=2)

    T_ax.legend(frameon=False,fontsize='large',loc='upper right',
                bbox_to_anchor=(1.0,1.0),handletextpad=0.1)
                #labelspacing=0.1,handlelength=0.5,handletextpad=0.7)

    _T_fit = np.r_[on_temps[1:],off_temps[1:]]
    _I_fit = np.r_[on_currents[1:],off_currents[1:]]
    _inds = np.argsort(_I_fit)
    _T_fit = _T_fit[_inds]
    _I_fit = _I_fit[_inds]
    _I_plot = np.r_[on_currents,off_currents]
    _I_plot = np.sort(_I_plot)
    coeff = np.polynomial.polynomial.polyfit(_I_fit,_T_fit,deg=1)
    T_ax.plot(_I_plot,coeff[0]+_I_plot*coeff[1],lw=1,ls=(0,(4,2,2,2)),c='k')

    _T_fit = _T[1:]
    _I_fit = _I[1:]
    _inds = np.argsort(_I_fit)
    _T_fit = _T_fit[_inds]
    _I_fit = _I_fit[_inds]
    _I_plot = _I
    _I_plot = np.sort(_I_plot)
    coeff = np.polynomial.polynomial.polyfit(_I_fit,_T_fit,deg=1)
    T_ax.plot(_I_plot,coeff[0]+_I_plot*coeff[1],lw=1,ls=(0,(4,2,2,2)),c='k')

    axes = [T_ax]

    #g_ax.yaxis.tick_right()
    #g_ax.yaxis.set_label_position("right")

    for _ax in axes:
        for axis in ['top','bottom','left','right']:
            _ax.spines[axis].set_linewidth(1.5)
        _ax.minorticks_on()
        _ax.tick_params(which='both',width=1,labelsize='large')
        _ax.tick_params(which='major',length=5)
        _ax.tick_params(which='minor',length=2)
        _ax.set_rasterized = True

    xlim = [-50,750]
    T_ax.set_xlim(xlim)

    ylim = [200,2000]
    T_ax.set_ylim(ylim)

    T_ax.set_ylabel('Temperature [K]',fontsize='large',labelpad=5)
    T_ax.set_xlabel(r'Current density [mA/mm$^2$]',fontsize='large')
    #fig.supxlabel('Temperature [K]',fontsize='large',y=0.03)

    #fig.suptitle(rf'POWER DISSIPATION INSTEAD',fontsize='large',y=0.95,c='r')

    fig_name = f'temps_vs_currents.png'
    plt.savefig(fig_name,dpi=300,bbox_inches='tight')

# --------------------------------------------------------------------------------------------------


directory = '20240730_Si_flash/20240730/corrected_Raman_data_export'
sample_len = 0.230
sample_area = 1.1011 # mm^2

on_temps, on_errs, off_temps, off_errs, on_w0, on_w0_err, on_g, on_g_err, off_w0, \
    off_w0_err, off_g, off_g_err, on_currents, off_currents = \
            plot_and_fit(directory,sample_len,sample_area)
    
plot_vs_temps(on_temps, on_errs, off_temps, off_errs,
    on_w0, on_w0_err, on_g, on_g_err, off_w0, off_w0_err, off_g, off_g_err, on_currents, \
    off_currents)

on_currents = np.array(on_currents)/sample_area
off_currents = np.array(off_currents)/sample_area

plot_vs_currents(on_temps, on_errs, off_temps, off_errs,
    on_w0, on_w0_err, on_g, on_g_err, off_w0, off_w0_err, off_g, off_g_err, on_currents, \
    off_currents)



