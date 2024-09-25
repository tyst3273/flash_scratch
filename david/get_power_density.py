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

def get_data(file_name):
    
    """
    len is in cm, area is in mm^2
    """
    
    if isinstance(file_name,str):
        file_name = [file_name]
    
    count = 0
    for f in file_name:
        
        if count == 0:
            data = np.loadtxt(f,delimiter=';',skiprows=1,dtype=object)
        else:
            data = np.r_[data,np.loadtxt(f,delimiter=';',skiprows=1,dtype=object)]    
            
        count += 1
    
    voltage = data[:,1].astype(str)
    current = data[:,3].astype(str)
    time = data[:,-1].astype(float)

    voltage = np.char.strip(voltage,'V').astype(float) # V
    current = np.char.strip(current,'A').astype(float) # A
    
    _inds = np.flatnonzero(voltage == 0)    
    time = np.delete(time,_inds)
    voltage = np.delete(voltage,_inds)
    current = np.delete(current,_inds)
    
    _sort = np.argsort(time)
    time = time[_sort]
    voltage = voltage[_sort]
    current = current[_sort]

    return current, voltage, time

# --------------------------------------------------------------------------------------------------

def plot_and_fit(file_name,sample_volume,sample_len,sample_area,slices,which):

    current, voltage, time = get_data(file_name)
       
    fig, ax = plt.subplots(figsize=(4,5)) 
    
    twin = ax.twinx()
    
    ax.plot(time,voltage,lw=1,marker='o',ms=1,c='b')
    twin.plot(time,current*1000,lw=1,marker='o',ms=1,c='r')
    
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
    
    xlims = [8500,8550]
    
    ax.set_ylabel(r'Voltage [V]',fontsize='large',labelpad=5,c='b')
    twin.set_ylabel(r'Current [mA]',fontsize='large',labelpad=3,c='r')
    
    ax.set_xlabel('Time [s]',fontsize='large',y=0.02)
    
    fig_name = f'{which}.png'
    plt.savefig(fig_name,dpi=300,bbox_inches='tight')
    
    get_power(slices,time,current,voltage,sample_volume,sample_len,sample_area,which)
    
# --------------------------------------------------------------------------------------------------

def get_power(slices,time,current,voltage,sample_volume,sample_len,sample_area,which):
        
    buffer = 0
    
    avg_voltage = []
    power = []
    power_density = []
    currents = []
    
    for s in slices:
            
        _lo = np.flatnonzero(time >= s[0]).min()+buffer
        _hi = np.flatnonzero(time <= s[1]).max()-buffer
            
        _t = time[_lo:_hi]
        _c = current[_lo:_hi].mean()
        _v = voltage[_lo:_hi].mean()
            
        _power = _v*_c * 1000 # milli-Watts
        _power_den = _v * _c * 1000 / sample_volume # mW / mm^3
                    
        avg_voltage.append(_v)
        power.append(_power)
        power_density.append(_power_den)
        currents.append((_c*100).round()*10) # mA
            
    avg_voltage = np.array(avg_voltage,dtype=float)
    currents = np.array(currents,dtype=float)
    power = np.array(power,dtype=float)
    power_density = np.array(power_density,dtype=float)
    
    e_field = avg_voltage / sample_len # V/cm
    current_den = currents / sample_area # mA / mm^2

    data = np.c_[current_den,e_field,power_density]
    np.savetxt(f'{which}_power_density.txt',data,fmt='%.3f %.3f %.3f',
               header='j [mA/mm^2], E [V/mm], power-density [mW/mm^3]')

# --------------------------------------------------------------------------------------------------

file_name = 'Raman V Data/Raman-TiO2.csv'
sample_len = 2.20
sample_area = 1.9345
sample_volume = sample_len * sample_area

slices = [[100,800],
          [1500,2300],
          [4200,4900],
          [5100,5900],
          [6100,6900]]
plot_and_fit(file_name,sample_volume,sample_len,sample_area,slices,which='TiO2')

# --------------------------------------------------------------------------------------------------

# fan off

# Si-Flash-Room-Air-Raman.csv,Si-Flash-Room-Air-Raman_1.csv  and Si-Flash-Room-Air-Raman_2.csv

file_name = ['Raman V Data/Si-Flash-Room-Air-Raman.csv','Raman V Data/Si-Flash-Room-Air-Raman_1.csv',
             'Raman V Data/Si-Flash-Room-Air-Raman_2.csv']
sample_len = 2.30
sample_area = 1.011
sample_volume = sample_len * sample_area

slices = [[200,320],
          [400,1800],
          [2000,2900],
          [3000,4000],
          [4100,5900],
          [6100,8400],
          [8600,9600],
          [9700,11500],
          [11600,12500],
          [12600,14000]]
plot_and_fit(file_name,sample_volume,sample_len,sample_area,slices,which='Si_off')

# --------------------------------------------------------------------------------------------------

# fan on

# Si-Flash-Room-Air-Raman_2.csv,Si-Flash-Room-Air-Raman-2_1.csv  and Si-Flash-Room-Air-Raman-2_2.csv

file_name = ['Raman V Data/Si-Flash-Room-Air-Raman-2.csv','Raman V Data/Si-Flash-Room-Air-Raman-2_1.csv',
             'Raman V Data/Si-Flash-Room-Air-Raman-2_2.csv']
sample_len = 2.30
sample_area = 1.011
sample_volume = sample_len * sample_area

slices = [[250,1300],
          [1450,2350],
          [2500,3650],
          [3800,4700],
          [5000,9600],
          [9800,10600],
          [10750,11750],
          [11900,13000],
          [13100,13950],
          [14100,15000]]
plot_and_fit(file_name,sample_volume,sample_len,sample_area,slices,which='Si_on')
