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

def plot_and_fit(file_name,sample_len,sample_area,slices):

    current, current_den, voltage, field, resistivity, conductivity, time = \
            get_data(file_name,sample_len,sample_area)

    which = file_name.split('/')[-1].replace('-','_').split('_')[-1][:-4]
    
    fig, ax = plt.subplots(figsize=(4,5)) 
    
    twin = ax.twinx()
    
    ax.plot(time,voltage,lw=1,marker='o',ms=1,c='b')
    twin.plot(time,current,lw=1,marker='o',ms=1,c='r')
    
    ax.set_zorder(0)
    twin.set_zorder(1)
    

    axes = [ax,twin]
    
    for _ax in axes:
        for axis in ['top','bottom','left','right']:
            _ax.spines[axis].set_linewidth(1.5)
        _ax.minorticks_on()
        _ax.tick_params(which='both',width=1,labelsize='large')
        _ax.tick_params(which='major',length=5)
        _ax.tick_params(which='minor',length=2)
        _ax.set_rasterized = True
    
    ax.tick_params(axis='y', colors='b')
    twin.tick_params(axis='y', colors='r')
    
    ax.tick_params(axis='y',which='both',right=False,labelright=False)
    twin.tick_params(axis='y',which='both',left=False,labelleft=False)
    
    # xlims = [0,100]
    # ax.set_xlim(xlims)
    # twin.set_xlim(xlims)
    
    # ax.set_ylim([0,100])
    # twin.set_ylim([0,200])
    
    ax.set_ylabel(r'Voltage',fontsize='large',labelpad=5,c='b')
    twin.set_ylabel(r'Current [mA]',fontsize='large',labelpad=3,c='r')
    
    ax.set_xlabel('Time [m]',fontsize='large',y=0.02)
    
    # fig_name = f'TiO2_{temp}_C_{environ}.png'
    # plt.savefig(fig_name,dpi=300,bbox_inches='tight')
    
    # get_power(slices,time,current,voltage,current_den,field,which)
    
# --------------------------------------------------------------------------------------------------

def get_power(slices,time,current,voltage,current_den,field,which):
    
    buffer = 500
    
    power = []
    power_density = []
    currents = []
    
    for s in slices:
        
        if s[1] < 0:
            s[1] = time.max()*60
        
        _lo = np.flatnonzero(time*60 >= s[0]).min()+buffer
        _hi = np.flatnonzero(time*60 <= s[1]).max()-buffer
        
        _t = time[_lo:_hi]
        print(_t.min(),_t.max())
        _c = current[_lo:_hi].mean()
        _v = voltage[_lo:_hi].mean()
        _j = current_den[_lo:_hi].mean()
        _e = field[_lo:_hi].mean()
        
        _p = _v*_c # milli-watts
        _p_den = _j*_e / 10 # V * mA / cm / mm^2 => mW / mm^3
        
        power.append(_p)
        power_density.append(_p_den)
        currents.append((_c/10).round(0)*10)
        
    currents = np.array(currents,dtype=int)
    power = np.array(power,dtype=float)
    power_density = np.array(power_density,dtype=float)
    
    _sort = np.argsort(currents)
    currents = currents[_sort]
    power = power[_sort]
    power_density = power_density[_sort]
    
    data = np.c_[currents,power,power_density]
    np.savetxt(f'{which}_power.txt',data,fmt='%d %.3f %.3f',
               header='current, power [mW], power-density [mW/mm^3]')

# --------------------------------------------------------------------------------------------------

file_name = 'Raman V Data/Raman-Ti02.csv'
sample_len = 0.220
sample_area = 1.9345

slices = [[0,1000],[1420,2400],[3270,4170],[4170,5000],[5000,6000],[6000,6900]]
plot_and_fit(file_name,sample_len,sample_area,slices)

# --------------------------------------------------------------------------------------------------

# file_name = 'Raman V Data/Raman50V_Si.csv'
# sample_len = 0.230
# sample_area = 1.011

# slices = [[0,1000],[1420,2400],[3270,4170],[4170,5000],[5000,6000],[6000,-1]]
# plot_and_fit(file_name,sample_len,sample_area,slices)