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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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

def get_temps(anti_params,stokes_params,anti_err,stokes_err):

    temps = []
    temp_errs = []
    
    for ii in range(len(anti_params)):
        
        anti = anti_params[ii]
        stokes = stokes_params[ii]
        
        A_as = np.abs(anti[1])#-anti[0]
        A_s = np.abs(stokes[1])#-stokes[0]
        ratio = 0.9625*A_as/A_s
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
        p0 = [1,100,500,3]
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
   
    # fig, ax = plt.subplots(2,2,figsize=(4,5),
    #                        gridspec_kw={'height_ratios':[1,1],'width_ratios':[1,1],
    #                                     'hspace':0.1,'wspace':0.1}) 
    
    # anti_on_ax = ax[0,0]; stokes_on_ax = ax[0,1]
    # anti_off_ax = ax[1,0]; stokes_off_ax = ax[1,1]
        
    # c = plt.cm.rainbow(np.linspace(0,1,len(on_currents)))
    # shift = 10
    # for ii, cc in enumerate(on_currents):
        
    #     d = stokes_on_data[ii]
    #     stokes_on_ax.plot(d[:,0],d[:,1]+ii*shift,marker='o',ms=2,lw=1,zorder=1000-ii,c=c[ii])
    #     stokes_on_ax.plot([-1000,1000],[ii*shift,ii*shift],ms=0,lw=1,ls=(0,(1,1)),c=(0.25,0.25,0.25))
    #     x, y = stokes_on_fits[ii]
    #     stokes_on_ax.plot(x,y+ii*shift,marker='o',ms=0,lw=1,zorder=1000-ii,c='k')
        
    #     d = anti_on_data[ii]
    #     anti_on_ax.plot(d[:,0],d[:,1]+ii*shift,marker='o',ms=2,lw=1,zorder=1000-ii,c=c[ii])
    #     anti_on_ax.plot([-1000,1000],[ii*shift,ii*shift],ms=0,lw=1,ls=(0,(1,1)),c=(0.25,0.25,0.25))
    #     x, y = anti_on_fits[ii]
    #     anti_on_ax.plot(-x,y+ii*shift,marker='o',ms=0,lw=1,zorder=1000-ii,c='k')
        
    #     T = on_temps[ii]
    #     stokes_on_ax.annotate(rf'T={T:.1f}K',fontsize='medium',
    #                         xy=(435,ii*shift+2),xycoords='data')
    #     anti_on_ax.annotate(rf'{cc:d} mA',fontsize='medium',
    #                         xy=(-475,ii*shift+2),xycoords='data')
        
    # c = plt.cm.rainbow(np.linspace(0,1,len(off_currents)))
    # shift = 10
    # for ii, cc in enumerate(off_currents):
        
    #     d = stokes_off_data[ii]
    #     stokes_off_ax.plot(d[:,0],d[:,1]+ii*shift,marker='o',ms=2,lw=1,zorder=1000-ii,c=c[ii])
    #     stokes_off_ax.plot([-1000,1000],[ii*shift,ii*shift],ms=0,lw=1,ls=(0,(1,1)),c=(0.25,0.25,0.25))
    #     x, y = stokes_off_fits[ii]
    #     stokes_off_ax.plot(x,y+ii*shift,marker='o',ms=0,lw=1,zorder=1000-ii,c='k')
        
    #     d = anti_off_data[ii]
    #     anti_off_ax.plot(d[:,0],d[:,1]+ii*shift,marker='o',ms=2,lw=1,zorder=1000-ii,c=c[ii])
    #     anti_off_ax.plot([-1000,1000],[ii*shift,ii*shift],ms=0,lw=1,ls=(0,(1,1)),c=(0.25,0.25,0.25))
    #     x, y = anti_off_fits[ii]
    #     anti_off_ax.plot(-x,y+ii*shift,marker='o',ms=0,lw=1,zorder=1000-ii,c='k')
        
    #     T = off_temps[ii]
    #     stokes_off_ax.annotate(rf'T={T:.1f}K',fontsize='medium',
    #                         xy=(435,ii*shift+2),xycoords='data')
    #     anti_off_ax.annotate(rf'{cc:d} mA',fontsize='medium',
    #                         xy=(-475,ii*shift+2),xycoords='data')

    # axes = [anti_on_ax, stokes_on_ax, anti_off_ax, stokes_off_ax]
    
    # for _ax in axes:
    #     for axis in ['top','bottom','left','right']:
    #         _ax.spines[axis].set_linewidth(1.5)
    #     _ax.minorticks_on()
    #     _ax.tick_params(which='both',width=1,labelsize='large')
    #     _ax.tick_params(which='major',length=5)
    #     _ax.tick_params(which='minor',length=2)
    #     _ax.set_rasterized = True
    
    # anti_on_ax.spines.right.set_visible(False)
    # anti_on_ax.tick_params(axis='y',which='both',right=False,labelright=False)
    # anti_off_ax.spines.right.set_visible(False)
    # anti_off_ax.tick_params(axis='y',which='both',right=False,labelright=False)
    
    # stokes_on_ax.spines.left.set_visible(False)
    # stokes_on_ax.tick_params(axis='y',which='both',left=False,labelleft=False)
    # stokes_off_ax.spines.left.set_visible(False)
    # stokes_off_ax.tick_params(axis='y',which='both',left=False,labelleft=False)
    
    # d = 0.1  # proportion of vertical to horizontal extent of the slanted line
    # kwargs = dict(marker=[(-d, -d), (d, d)], markersize=5,
    #               linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    
    # anti_on_ax.plot((1,1), (0,0), transform=anti_on_ax.transAxes, **kwargs)
    # anti_on_ax.plot((1,1), (1,1), transform=anti_on_ax.transAxes, **kwargs)
    # stokes_on_ax.plot((0,0), (0,0), transform=stokes_on_ax.transAxes, **kwargs)
    # stokes_on_ax.plot((0,0), (1,1), transform=stokes_on_ax.transAxes, **kwargs)
    
    # anti_on_ax.plot((1,1), (0,0), transform=anti_on_ax.transAxes, **kwargs)
    # anti_on_ax.plot((1,1), (1,1), transform=anti_on_ax.transAxes, **kwargs)
    # stokes_on_ax.plot((0,0), (0,0), transform=stokes_on_ax.transAxes, **kwargs)
    # stokes_on_ax.plot((0,0), (1,1), transform=stokes_on_ax.transAxes, **kwargs)
    
    # anti_on_ax.plot([1,1],[0,1], transform=anti_on_ax.transAxes, 
    #                 lw=1, ls=(0,(4,1,2,1)),ms=0, c='k')
    # anti_off_ax.plot([1,1],[0,1], transform=anti_off_ax.transAxes, 
    #                 lw=1, ls=(0,(4,1,2,1)),ms=0, c='k')
    # stokes_on_ax.plot([0,0],[0,1], transform=stokes_on_ax.transAxes, 
    #                 lw=1, ls=(0,(4,1,2,1)),ms=0, c='k')
    # stokes_off_ax.plot([0,0],[0,1], transform=stokes_off_ax.transAxes, 
    #                 lw=1, ls=(0,(4,1,2,1)),ms=0, c='k')
    
    # xlim = [-550,-430]
    # anti_on_ax.set_xlim(xlim)
    # anti_off_ax.set_xlim(xlim)
    
    # xlim = [430,550]
    # stokes_on_ax.set_xlim(xlim)
    # stokes_off_ax.set_xlim(xlim)
    
    # ylim = [-5,70]
    # anti_on_ax.set_ylim(ylim)
    # stokes_on_ax.set_ylim(ylim)
    
    # ylim = [-5,80]
    # anti_off_ax.set_ylim(ylim)
    # stokes_off_ax.set_ylim(ylim)
    
    # anti_off_ax.xaxis.set_major_formatter(FormatStrFormatter('%1d'))
    # stokes_off_ax.xaxis.set_major_formatter(FormatStrFormatter('%1d'))
    
    # anti_on_ax.set_xticklabels([])
    # stokes_on_ax.set_xticklabels([])
    
    # anti_on_ax.annotate(rf'fan on',xy=(0.1,0.9),xycoords='axes fraction',fontsize='large')
    # anti_off_ax.annotate(rf'fan off',xy=(0.1,0.9),xycoords='axes fraction',fontsize='large')
    
    # anti_on_ax.set_ylabel('Intensity [arb. units]',fontsize='large',labelpad=5)
    # anti_off_ax.set_ylabel('Intensity [arb. units]',fontsize='large',labelpad=5)
    
    # fig.supxlabel('Raman shift [1/cm]',fontsize='large',y=0.02)
    
    # fig.suptitle('Flashing Si',fontsize='large',y=0.93)
    
    # anti_on_ax.annotate(rf'(a)',xy=(0.825,0.9),xycoords='axes fraction',fontsize='large')  
    # stokes_on_ax.annotate(rf'(b)',xy=(0.825,0.9),xycoords='axes fraction',fontsize='large')  
    
    # anti_off_ax.annotate(rf'(c)',xy=(0.825,0.9),xycoords='axes fraction',fontsize='large')  
    # stokes_off_ax.annotate(rf'(d)',xy=(0.825,0.9),xycoords='axes fraction',fontsize='large')  
    
    # fig_name = 'si_raman.png'
    # plt.savefig(fig_name,dpi=300,bbox_inches='tight')

    on_w0 = []
    on_w0_err = []
    on_g = []
    on_g_err = []
    for ii in range(len(on_currents)):
        
        w_s = stokes_on_params[ii][2]
        w_as = anti_on_params[ii][2]
        w = (w_s+w_as)/2
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
        w = (w_s+w_as)/2
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
        
    data = np.c_[on_temps,on_errs,on_w0,on_w0_err,on_g,on_g_err,on_currents]
    np.savetxt('si_fan_on_data.txt',data,fmt='%.6f',header='T [K], dT [K], w0 [1/cm], dw0 [1/cm], G [1/cm], dG [1/cm], I [mA]')
    
    data = np.c_[off_temps,off_errs,off_w0,off_w0_err,off_g,off_g_err,off_currents]
    np.savetxt('si_fan_off_data.txt',data,fmt='%.6f',header='T [K], dT [K], w0 [1/cm], dw0 [1/cm], G [1/cm], dG [1/cm], I [mA]')
    
    header = ' '.join([str(_) for _ in on_currents])
    anti_on_fits = np.array(anti_on_fits) 
    e = np.atleast_2d(anti_on_fits[0,0,:])
    fits = anti_on_fits[:,1,:]
    data = np.append(e,fits,axis=0)
    np.savetxt('si_AS_on_fits.txt',data,fmt='%.3e',header=header)
     
    stokes_on_fits = np.array(stokes_on_fits)        
    e = np.atleast_2d(stokes_on_fits[0,0,:])
    fits = stokes_on_fits[:,1,:]
    data = np.append(e,fits,axis=0)
    np.savetxt('si_S_on_fits.txt',data,fmt='%.3e',header=header)
    
    header = ' '.join([str(_) for _ in off_currents])
    anti_off_fits = np.array(anti_off_fits) 
    e = np.atleast_2d(anti_off_fits[0,0,:])
    fits = anti_off_fits[:,1,:]
    data = np.append(e,fits,axis=0)
    np.savetxt('si_AS_off_fits.txt',data,fmt='%.3e',header=header)
     
    stokes_off_fits = np.array(stokes_off_fits)        
    e = np.atleast_2d(stokes_off_fits[0,0,:])
    fits = stokes_off_fits[:,1,:]
    data = np.append(e,fits,axis=0)
    np.savetxt('si_S_off_fits.txt',data,fmt='%.3e',header=header)
    
    return on_temps, on_errs, off_temps, off_errs, \
        on_w0, on_w0_err, on_g, on_g_err, off_w0, off_w0_err, off_g, off_g_err, \
        on_currents, off_currents
        
# --------------------------------------------------------------------------------------------------

def plot_vs_power(on_temps, on_errs, off_temps, off_errs, on_w0, on_w0_err, on_g, on_g_err,
                off_w0, off_w0_err, off_g, off_g_err,on_currents,off_currents):
    
    sample_len = 0.230 # cm
    sample_area = 1.1011 # mm^2
    sample_area *= 0.01

    print(sample_area/sample_len)

    fig, ax = plt.subplots(1,figsize=(4,4),
                           gridspec_kw={'hspace':0.1,'wspace':0.075})

    T_ax = ax #; g_ax = ax[1]
   
    _I, _P, _p_off = np.loadtxt('../david/Si_off_power.txt',unpack=True)
    _P /= 1000; _I /= 1000
    print(_P/_I)
    R = _P/_I**2
    r = R*sample_area/sample_len # Ohm * cm
    T_ax.plot(off_temps[1:],1/r,marker='o',ms=8,lw=1,c='r',label='Si - fan off',ls=(0,(4,1,2,1)))
    
    _I, _P, _p_on = np.loadtxt('../david/Si_on_power.txt',unpack=True)
    _P /= 1000; _I /= 1000
    R = _P/_I**2
    r = R*sample_area/sample_len # Ohm * cm
    T_ax.plot(on_temps[1:],1/r[1:],marker='o',ms=8,lw=1,c='b',label='Si - fan on',ls=(0,(4,1,2,1)))

    # run 2/3 - resistivity
    x = [0.8, 0.9461928934010153, 1.0984771573604062, 1.2568527918781727, 1.3908629441624365, 1.537055837563452, 1.6345177664974622, 1.7746192893401016, 1.8903553299492388, 1.969543147208122, 2.0609137055837565, 2.14010152284264, 2.2741116751269037, 2.402030456852792, 2.5116751269035538, 2.651776649746193, 2.773604060913706, 2.9197969543147213, 3.078172588832488, 3.230456852791878]
    y = [0.011038044913440246, 0.030483766577128934, 0.07956742329247525, 0.2530383077694783, 0.5899746255923562, 1.6293328133760168, 3.207210395951558, 7.691825299412011, 16.94988151390346, 29.80332411852676, 49.528245211413356, 60.34443910169926, 64.75517923684654, 59.499018813225625, 54.669499512746235, 48.15019377307908, 43.01089020952117, 37.88186678859618, 32.89704301901153, 29.385782166876627]
    x = np.array(x)
    y = np.array(y)
    T_ax.plot(10**3/x,1/y,marker='^',ms=4,lw=1,c='m',label='10.1103/PhysRev.167.765',zorder=1000)

    # David
    x = [318.28631138975965, 379.7283176593521, 428.735632183908, 479.2058516196447, 528.944618599791, 580.1462904911181, 629.8850574712644, 678.1609195402299, 730.0940438871473, 779.8328108672936, 829.5715778474399, 880.0417972831766, 948.0668756530825, 978.7878787878788, 1028.526645768025]
    y = [486.66666666666663, 349.6296296296296, 331.1111111111111, 68.14814814814814, 37.77777777777778, 25.185185185185183, 27.407407407407405, 41.48148148148148, 34.07407407407407, 100.74074074074073, 100, 15.555555555555555, 4.444444444444445, 3.7037037037037033, 3.7037037037037033]
    x = np.array(x) 
    y = np.array(y)/10
    T_ax.plot(x,1/y,marker='s',ms=4,lw=1,c='g',label='David',zorder=1000)

    # https://lampz.tugraz.at/~hadley/psd/L4/conductivity.php -- conductivity
    x, y = np.loadtxt('si_cond_tugraz.txt',unpack=True)
    T_ax.plot(x,y,marker='^',ms=0,lw=2,c='k',label='TUGraz',zorder=1000)

    for axis in ['top','bottom','left','right']:
        T_ax.spines[axis].set_linewidth(1.5)
    T_ax.minorticks_on()
    T_ax.tick_params(which='both',width=1,labelsize='large')
    T_ax.tick_params(which='major',length=5)
    T_ax.tick_params(which='minor',length=2)
    T_ax.set_rasterized = True

    T_ax.set_ylabel(r'$\sigma$ [$\Omega$-cm]$^{-1}$',fontsize='large',labelpad=5)
    T_ax.set_xlabel('Temperature [K]',fontsize='large')

    T_ax.axis([550,950,0,3])
    T_ax.legend(fontsize='medium',loc='upper left',frameon=False)

    plt.savefig('si_temp_vs_resistivity.png',dpi=300,bbox_inches='tight')
    plt.show()

# --------------------------------------------------------------------------------------------------


directory = '20240730_Si_flash/20240730/corrected_Raman_data_export'
sample_len = 0.230
sample_area = 1.1011 # mm^2

on_temps, on_errs, off_temps, off_errs, on_w0, on_w0_err, on_g, on_g_err, off_w0, \
    off_w0_err, off_g, off_g_err, on_currents, off_currents = \
            plot_and_fit(directory,sample_len,sample_area)

plot_vs_power(on_temps, on_errs, off_temps, off_errs,
    on_w0, on_w0_err, on_g, on_g_err, off_w0, off_w0_err, off_g, off_g_err, on_currents, \
    off_currents)




