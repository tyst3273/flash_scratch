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



fig, ax = plt.subplots(1,3,figsize=(8,2.5),gridspec_kw={'hspace':0.075,'wspace':0.25}) 
# twin_ax = ax.twinx()

# ----------------------

j, E, p = np.loadtxt('Si_off_power_density.txt',unpack=True)
E *= 10
d = j[1:]-j[:-1]
ind = np.flatnonzero(d < 0)[0]+1

_E = E[:ind]
_j = j[:ind]
rho = _E/_j
ax[0].plot(_j,_E,lw=1,marker='o',ms=5,c='b')
for ii in range(ind):
    ax[0].annotate(rf'{ii+1}',xy=(_j[ii]-15,_E[ii]-10),fontsize='medium',xycoords='data')
    
_E = E[ind-1:]
_j = j[ind-1:]
rho = _E/_j
ax[0].plot(_j,_E,lw=1,marker='o',ms=5,c='b')#,mfc='none')
for ii, jj in enumerate(range(ind,E.size)):
    ax[0].annotate(rf'{jj+1}',xy=(_j[ii+1]+10,_E[ii+1]+2.5),fontsize='medium',xycoords='data')
    
# ----------------------

j, E, p = np.loadtxt('Si_on_power_density.txt',unpack=True)
E *= 10
d = j[1:].round()-j[:-1].round()
ind = np.flatnonzero(d < 0)[0]+1

_E = E[:ind]
_j = j[:ind]
ax[1].plot(_j,_E,lw=1,marker='s',ms=5,c='r',mfc='none')
for ii in range(ind):
    ax[1].annotate(rf'{ii+1}',xy=(_j[ii]+10,_E[ii]+2.5),fontsize='medium',xycoords='data')
        
# ind += 1
_E = E[ind-1:]
_j = j[ind-1:]
ax[1].plot(_j,_E,lw=1,marker='s',ms=5,c='r',mfc='none')
for ii, jj in enumerate(range(ind,E.size)):
    ax[1].annotate(rf'{jj+1}',xy=(_j[ii+1]-50,_E[ii+1]-7.5),fontsize='medium',xycoords='data')
    
# ----------------------

j, E, p = np.loadtxt('TiO2_power_density.txt',unpack=True)
E *= 10
ax[2].plot(j,E,lw=1,marker='^',ms=5,c='m')
for ii in range(j.size):
    ax[2].annotate(rf'{ii+1}',xy=(j[ii]+1,E[ii]+2),fontsize='medium',xycoords='data')
    
# ----------------------

axes = list(ax)

for _ax in axes:
    for axis in ['top','bottom','left','right']:
        _ax.spines[axis].set_linewidth(1.5)
    _ax.minorticks_on()
    _ax.tick_params(which='both',width=1,labelsize='large')
    _ax.tick_params(which='major',length=5)
    _ax.tick_params(which='minor',length=2)
    _ax.set_rasterized = True

ax[0].set_xlim(-20,825)
ax[1].set_xlim(-20,825)
ax[2].set_xlim(17,55)

ax[0].set_ylim(20,110.5)
ax[1].set_ylim(20,110.5)
ax[2].set_ylim(190,400)

# ax[1].set_yticklabels([])
# ax[2].set_yticklabels([])
    
ax[0].annotate(rf'Si - Fan off',xy=(0.5,0.9),
                   xycoords='axes fraction',fontsize='large')
ax[1].annotate(rf'Si - Fan on',xy=(0.5,0.9),
                   xycoords='axes fraction',fontsize='large')
ax[2].annotate(rf'TiO$_2$',xy=(0.75,0.9),
                   xycoords='axes fraction',fontsize='large')

ax[0].set_xlabel(r'Current density [mA/mm$^2$]',fontsize='large',labelpad=1)
ax[1].set_xlabel(r'Current density [mA/mm$^2$]',fontsize='large',labelpad=1)
ax[2].set_xlabel(r'Current density [mA/mm$^2$]',fontsize='large',labelpad=1)

ax[0].set_ylabel(r'Electric field [V/cm]',fontsize='large',labelpad=5)

# fig_name = f'TiO2_{temp}_C_{environ}_{which}_fit.png'
plt.savefig('iv_hysteresis.pdf',dpi=300,bbox_inches='tight')

    

