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

def parse_file_name(file_name):
    
    _f = file_name.split('-')
    
    temp = _f[1].strip('C')
    environ = _f[3].lower()
    
    return temp, environ

# --------------------------------------------------------------------------------------------------

def plot(file_name,sample_len,sample_area):
    
    current, current_den, voltage, field, resistivity, conductivity, time = \
            get_data(file_name,sample_len,sample_area)
            
    # temp, environ = parse_file_name(file_name)
    
    fig, ax = plt.subplots(figsize=(6,6)) 

    ax.scatter(voltage,current,marker='o',s=10,color='k')
    # ax.plot(voltage,current,marker='o',ms=2,color='k')

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
    ax.minorticks_on()
    ax.tick_params(which='both',width=1,labelsize='large')
    ax.tick_params(which='major',length=5)
    ax.tick_params(which='minor',length=2)
    ax.set_rasterized = True
    
    # ax.set_xlim(t_lo_lims)
    # ax.set_ylim(field_ylims)
    
    ax.set_ylabel('Voltage [V]',fontsize='large')
    ax.set_xlabel('Current [A]',fontsize='large')
    
    # plt.savefig(fig_name,dpi=300,bbox_inches='tight')
    plt.show()
    
# --------------------------------------------------------------------------------------------------

# file_name = '../Si-Flash Data/Room-Si-Flash-attempt-1.csv'
file_name = '../Si-Flash Data/Room-Si-Flash-1-900-1000.csv'
sample_len = 0.592
sample_area = 1.7394

plot(file_name,sample_len,sample_area)

# --------------------------------------------------------------------------------------------------


