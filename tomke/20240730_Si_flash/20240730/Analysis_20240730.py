# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 18:42:23 2024

@author: Tomke
"""

import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
import statistics
import csv
import os

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

# %% Label Size for plots
plt.rc ('font', size = 14) # steuert die Standardtextgröße
plt.rc ('axes', titlesize = 14) # Schriftgröße des Titels
plt.rc ('axes', labelsize = 14) # Schriftgröße der x- und y-Beschriftungen
plt.rc ('xtick', labelsize = 14) #Schriftgröße der x-Tick-Labels
plt.rc ('ytick', labelsize = 14) #Schriftgröße der y-Tick-Labels
plt.rc ('legend', fontsize = 14) #Schriftgröße der Legende

# %% Load Data
def load_data(path):#Load data from txt file
    data_list = []
    rs_list = []
    lines = np.loadtxt(path, skiprows=1,  usecols = (0,1,2), delimiter=',')
    for i in range(len(lines)):
        data_list.append(lines[i][2])
        rs_list.append(lines[i][1])
    return data_list, rs_list

def save_data(rs, data, name):#save data to txt file
    zip(rs, data)
    with open(str(name)+'.txt', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(zip(rs, data))
        
# %% Load Data Accumulations
def load_data_acc(path, N):#Load data from txt file, N = number of acc
    data = np.array([])
    rs_list = []
    lines = np.loadtxt(path, skiprows=1,  usecols = (0,1,2), delimiter=',')
    data_list = []
    for i in range(len(lines)):
        data_list.append(lines[i][2])
        rs_list.append(lines[i][1])
        
    data = np.array(data_list)
    data = data.reshape([N, 1340], order = 'C')
    rs = np.array(rs_list)
    rs = rs.reshape([N, 1340], order = 'C')
    return data, rs

def median_acc(path, N):
    data, w = load_data_acc(path, N)
    med = []
    rs = []
    std = []
    if N == 1:
        med = data[0]
        rs = w[0]
        std = [0,]*1340
    else:
        for i in range(len(data[0])):
            values = []
            for j in range(N):
                values.append(data[j][i])
            rs.append(w[0][i])
            med.append(statistics.median(values))                              # Median values
            std.append(statistics.stdev(values))                               # standard deviation
    return med, std, rs
    
# %% Background Correction
def bg_subtraction(path_bg, N_bg, data):
    bg, std, rs = median_acc(path_bg, N_bg)
    #smooth bg
    bg_sm = [bg[0],bg[1]]
    for i in range(2, len(bg)-2):
        bg_sm.append((bg[i-2]+bg[i-1]+bg[i]+bg[i+1]+bg[i+2])/5)
    bg_sm.append(bg[len(bg)-2])
    bg_sm.append(bg[len(bg)-1])
    
    data_sub = np.array(data)-np.array(bg_sm)
    '''
    plt.figure()
    plt.plot(data, 'k')
    plt.plot(bg, 'k')
    plt.plot(bg_sm, 'r')
    #plt.plot(data_sub, 'g')
    plt.xlim(500, 750)
    plt.show()
    '''
    return data_sub

# %% Normalization
def norm(data, std, time, power):#Normalization to compare different measurements
    data_norm = np.array(data)/(time*power)
    std = np.array(std)/(time*power)
    return data_norm, std
   
# %% Lorentzian Fitting with lmfit
def Lor_fitting(data, std, rs, fit_range, plot_range, pos):                    # plot_range = range on x-Axis shown in the figure
    i_end=abs(np.array(rs)-fit_range[0]).argmin()                              # fit_range = range on x-Axis used for the fit algorithm
    i_start=abs(np.array(rs)-fit_range[1]).argmin()
    x = rs
    y = data
    weight = []                                                                # add 1 for all data point in fit range and 0 outside fit range
    for i in range(len(data)):
        if i < i_start:
            weight.append(0)
        elif i > i_end:
            weight.append(0)
        else:
            weight.append(1)
    
    
    def fitfunc(x, y0, a, A1, G1, x0_1):                                       # Phonon Lorentz function
        # x = frequency, y0 = constant background, 
        # a*x allows background slope
        # A1 = Phonon Amplitude, G1 = Phonon width/ damping, 
        # x0_1 = Phonon frequency
        return y0+a*x+A1*x*G1/((x**2-x0_1**2)**2+G1**2*x**2)
    
    fitmodel = Model(fitfunc, independent_vars=['x'])
    params = fitmodel.make_params(y0=0, a=0.0000, A1=1E4, G1=3, x0_1=pos)   # set some start values
    result = fitmodel.fit(y, params, x=x, weights = weight)
    print('######## Lorentz fitting #########')
    print(result.fit_report())
    res = np.array(y)-np.array(result.best_fit)                                # calculate residual 
    '''
    ##### plot fitting result #################################################
    plt.figure(dpi=200)
    plt.title('Lorentz Fit')
    if std[0] == std[1] and std[0] == 0:
        plt.plot(rs, np.array(data), 'k', label='data median')
    else:
        plt.plot(rs, np.array(data), 'k', label='data median')
        plt.errorbar(rs, np.array(data), yerr = std, linestyle='-', color = 'black', alpha=0.4, capsize = 1.2, linewidth=1.1)
    #plt.plot(x, result.init_fit, '--b', label='initial')
    plt.plot(x, np.array(result.best_fit), 'r', label='best fit')
    plt.plot(x, res, c = 'gray', label = 'residual')
    plt.plot([plot_range[0], plot_range[1]], [0,0], 'k-.')
    plt.xlabel('Raman Shift (cm$^{-1}$)')
    plt.ylabel('Counts/(mW*sec)')
    plt.tick_params(left = True, right = True, bottom = True, top = True)
    plt.tick_params(axis = 'both', direction = 'in')
    plt.legend(frameon = False)
    #plt.grid(linestyle=':')
    plt.xlim(plot_range[0], plot_range[1])
    m = np.amax(data[abs(np.array(rs)-fit_range[1]).argmin():abs(np.array(rs)-fit_range[0]).argmin()])
    plt.ylim(-m*0.5, m*1.8)
    plt.yticks([])
    plt.show()
    '''
    # Extract final fit parameters ############################################
    width = result.params['G1'].value
    width_std = result.params['G1'].stderr
    
    pos = result.params['x0_1'].value
    pos_std = result.params['x0_1'].stderr
    
    amp = result.params['A1'].value
    amp_std = result.params['A1'].stderr
    
    parameters = [width, width_std, pos, pos_std, amp, amp_std]
    return parameters

def Lor_fitting_ref(data, std, rs, fit_range, plot_range, pos, g):                    # plot_range = range on x-Axis shown in the figure
    i_end=abs(np.array(rs)-fit_range[0]).argmin()                              # fit_range = range on x-Axis used for the fit algorithm
    i_start=abs(np.array(rs)-fit_range[1]).argmin()
    x = rs
    y = data
    weight = []                                                                # add 1 for all data point in fit range and 0 outside fit range
    for i in range(len(data)):
        if i < i_start:
            weight.append(0)
        elif i > i_end:
            weight.append(0)
        else:
            weight.append(1)
    
    
    def fitfunc(x, y0, A1, G1, x0_1):                                       # Phonon Lorentz function
        # x = frequency, y0 = constant background, 
        # A1 = Phonon Amplitude, G1 = Phonon width/ damping, 
        # x0_1 = Phonon frequency
        return y0+A1*x*G1/((x**2-x0_1**2)**2+G1**2*x**2)
    
    fitmodel = Model(fitfunc, independent_vars=['x'])
    params = fitmodel.make_params()# set some start values
    params.add('y0', value=0.001, min=-0.01, max=10)                                   # set some start values
    params.add('A1', value=10E3, min=0, max=10E10)
    params.add('G1', value=g, min=-50, max=50)
    params.add('x0_1', value=pos, min=pos-50, max=pos+50)
    result = fitmodel.fit(y, params, x=x, weights = weight)
    print('######## Lorentz fitting #########')
    print(result.fit_report())
    res = np.array(y)-np.array(result.best_fit)                                # calculate residual 
    
    '''
    ##### plot fitting result #################################################
    plt.figure(dpi=200)
    plt.title('Lorentz Fit')
    if std[0] == std[1] and std[0] == 0:
        plt.plot(rs, np.array(data), 'k', label='data median')
    else:
        plt.plot(rs, np.array(data), 'k', label='data median')
        plt.errorbar(rs, np.array(data), yerr = std, linestyle='-', color = 'black', alpha=0.4, capsize = 1.2, linewidth=1.1)
    plt.plot(x, result.init_fit, '--b', label='initial')
    plt.plot(x, np.array(result.best_fit), 'r', label='best fit')
    plt.plot(x, res, c = 'gray', label = 'residual')
    plt.plot([plot_range[0], plot_range[1]], [0,0], 'k-.')
    plt.xlabel('Raman Shift (cm$^{-1}$)')
    plt.ylabel('Counts/(mW*sec)')
    plt.tick_params(left = True, right = True, bottom = True, top = True)
    plt.tick_params(axis = 'both', direction = 'in')
    plt.legend(frameon = False)
    #plt.grid(linestyle=':')
    plt.xlim(plot_range[0], plot_range[1])
    m = np.amax(data[abs(np.array(rs)-fit_range[1]).argmin():abs(np.array(rs)-fit_range[0]).argmin()])
    plt.ylim(-m*0.5, m*1.8)
    plt.yticks([])
    plt.show()
    '''
    # Extract final fit parameters ############################################
    width = result.params['G1'].value
    width_std = result.params['G1'].stderr
    
    pos = result.params['x0_1'].value
    pos_std = result.params['x0_1'].stderr
    
    amp = result.params['A1'].value
    amp_std = result.params['A1'].stderr
    
    parameters = [width, width_std, pos, pos_std, amp, amp_std]
    return parameters

# %%  Temperature Calculation

def Temp(rs_S, lambda_0, amp_S, amp_AS, amp_S_std, amp_AS_std):     
    E_S = rs_S/8.064516*0.001                                                  #Raman Shift energy in eV
    kB = 8.617333 * 10**(-5)                                                   #eV/K
    h = 4.135667 * 10**(-15)                                                   #eV*s
    c = 299792458                                                              #m/s
    v0 = h*c/(lambda_0*10**(-9))                                               #Laser energy in eV         
    ratio = -amp_AS/amp_S                                                      #Amplitudes from Lorentz fit                              
    #ratio_AS_intensity_to_S_intensity = ((v0+E_S)/(v0-E_S))**4 * np.exp(-E_S/(kB*T))
    T = -E_S/(kB*np.log(ratio/((v0+E_S)/(v0-E_S))**4))
    err_S = abs(amp_S_std/amp_S)#relative error from Stokes amplitude
    err_AS = abs(amp_AS_std/amp_AS)#relative error from anti-Stokes amplitude
    ratio_err = (err_S + err_AS)*ratio#derived error of ratio
    T_err = -E_S*(-1)*(kB*(np.log(ratio)-np.log((v0+E_S)/(v0-E_S))**4))**(-2)*(kB*1/ratio-kB)*ratio_err #error propagation law: (d T/d ratio)*ratio_err
    #errors are smaller than 3 K, plus additional uncertainties in the experiments... max 10 K errors for the calculated temperatures
    return T, T_err

def Corr_S(rs_S, T):
    kB = 8.617333 * 10**(-5)                                                   #eV/K
    h = 4.135667 * 10**(-15)                                                   #eV*s
    c = 299792458   
    lambda_0 = 532                                                           #m/s
    v0 = h*c/(lambda_0*10**(-9)) 
    energy_S = np.array(rs_S)*(-1)/8.064516*0.001 
    N = 1340
    Bose = []
    for i in range(N):
        Bose.append((1/(np.exp(-energy_S[i]/(kB*T))-1)+1)*((v0+energy_S[i])/v0)**4)
    return Bose

def Corr_AS(rs_AS, T):
    kB = 8.617333 * 10**(-5)                                                   #eV/K
    h = 4.135667 * 10**(-15)                                                   #eV*s
    c = 299792458
    lambda_0 = 532                                                                #m/s
    v0 = h*c/(lambda_0*10**(-9)) 
    energy_AS = np.array(rs_AS)*(-1)/8.064516*0.001 
    N = 1340
    Bose = []
    for i in range(N):
        Bose.append((1/(np.exp(energy_AS[i]/(kB*T))-1))*((v0+energy_AS[i])/v0)**4)
    return Bose

# %%  Ref data

std = [0]*1340
from ref_data import *

# import reference data from PRB and fit them with Lorentz function to get the widths for comparison
shift_20K = raman_20K_stokes['x']
intensity_20K = raman_20K_stokes['y']
parameters = Lor_fitting_ref(intensity_20K, std, shift_20K, [-515,-528], [-500,-530], -522, -3)
width_ref_20K = -parameters[0]#due to definition of raman shift axis (negativ), width and shifts are negativ... but we want to plot the absolute values
width_ref_20K_err = parameters[1]

shift_460K = raman_460K_stokes['x']
intensity_460K = raman_460K_stokes['y']
parameters = Lor_fitting_ref(intensity_460K, std, shift_460K, [-500,-527], [-500,-530], -515, -5)
width_ref_460K = -parameters[0]
width_ref_460K_err = parameters[1]

shift_770K = raman_770K_stokes['x']
intensity_770K = raman_770K_stokes['y']
parameters = Lor_fitting_ref(intensity_770K, std, shift_770K, [-495,-522], [-490,-530], -510, -8)
width_ref_770K = -parameters[0]
width_ref_770K_err = parameters[1]

width_ref = [width_ref_20K, width_ref_460K, width_ref_770K]
width_ref_err = [width_ref_20K_err, width_ref_460K_err, width_ref_770K_err]

# same with Anti-Stokes data
shift_AS_770K = raman_AS_770K_stokes['x']
intensity_AS_770K = raman_AS_770K_stokes['y']
parameters = Lor_fitting_ref(intensity_AS_770K, std, shift_AS_770K, [530,500], [530,500], 510, 8)
width_ref_770K_AS = parameters[0]
width_ref_770K_AS_err = parameters[1]

shift_AS_460K = raman_AS_460K_stokes['x']
intensity_AS_460K = raman_AS_460K_stokes['y']
parameters = Lor_fitting_ref(intensity_AS_460K, std, shift_AS_460K, [530,500], [530,500], 515, 5)
width_ref_460K_AS = parameters[0]
width_ref_460K_AS_err = parameters[1]

width_ref_AS = [width_ref_460K_AS, width_ref_770K_AS]
width_ref_AS_err = [width_ref_460K_AS_err, width_ref_770K_AS_err]


# %% Data import and fitting
### laser  for reference ###
path = os.path.join(__location__, 'raw_data/laser_fanoff_0mA_30mW_1sec_5acc_531nm.txt')
laser_0_fanoff, std, rs = median_acc(path, 5)
path = os.path.join(__location__, 'raw_data/laser_fanoff_50mA_30mW_1sec_5acc_531nm.txt')
laser_50_fanoff, std, rs = median_acc(path, 5)
path = os.path.join(__location__, 'raw_data/laser_fanoff_150mA_30mW_1sec_5acc_531nm.txt')
laser_150_fanoff, std, rs = median_acc(path, 5)
path = os.path.join(__location__, 'raw_data/laser_fanoff_300mA_30mW_1sec_5acc_531nm.txt')
laser_300_fanoff, std, rs = median_acc(path, 5)
path = os.path.join(__location__, 'raw_data/laser_fanoff_450mA_30mW_1sec_5acc_531nm.txt')
laser_450_fanoff, std, rs = median_acc(path, 5)
path = os.path.join(__location__, 'raw_data/laser_fanoff_600mA_30mW_1sec_5acc_531nm.txt')
laser_600_fanoff, std, rs = median_acc(path, 5)
path = os.path.join(__location__, 'raw_data/laser_fanoff_750mA_30mW_1sec_5acc_531nm.txt')
laser_750_fanoff, std, rs = median_acc(path, 5)

plt.figure(dpi=200)
plt.plot(rs, laser_0_fanoff, c = 'blue', label = '0 A')
plt.plot(rs, laser_50_fanoff, c = 'green', label = '50 mA')
plt.plot(rs, laser_150_fanoff, c = 'k', label = '150 mA')
plt.plot(rs, laser_300_fanoff, c = 'r', label = '300 mA')
plt.plot(rs, laser_450_fanoff, c = 'y', label = '450 mA')
plt.plot(rs, laser_600_fanoff, c = 'violet', label = '600 mA')
plt.plot(rs, laser_750_fanoff, c = 'purple', label = '750 mA')
plt.plot([0, 0], [-1000000, 10000000], 'k--')
plt.ylim(58000, 70000)
#plt.plot([320, 830], [0,0], 'k--')
plt.xlim(-50, 50)
plt.legend(frameon = False)
plt.xlabel('Raman shift (cm$^{-1}$)')
plt.ylabel('Intensity (Counts / mW*sec)')
plt.yticks([])
plt.tick_params(left = True, right = True, bottom = True, top = True)
plt.tick_params(axis = 'both', direction = 'in')
plt.title('Stokes Raman data')
plt.show()

# %%
# Raman data with Fan on ######################################################

# import data from measurements and fit with Lorentzians
# S = Stokes, AS = anti-Stokes
x_axis = [0, 150, 300, 450, 600, 750]#current in mA
width_S = []
width_S_std = []
pos_S = []
pos_S_std = []
amp_S = []
amp_S_std = []
width_AS = []
width_AS_std = []
pos_AS = []
pos_AS_std = []
amp_AS = []
amp_AS_std = []


#0 Volts ## Stokes ############################################################
print('######### 0 Volts - Stokes')
path = os.path.join(__location__, 'raw_data/Si_0mA_30mW_10sec_20acc_545nm.txt')
path_bg = os.path.join(__location__, 'raw_data/Si_0mA_30mW_10sec_5acc_545nm_bg.txt')

time = 10 #integration time in sec
power = 30 #laser power in mW

data, std, rs = median_acc(path, 20) #load data and calculate median from the collected accumulations
data_corr = bg_subtraction(path_bg, 5, data) #subtract background measurement
data_norm, std = norm(data_corr, std, time, power) #normalize data to power and integration time

S_0 = data_norm
rs_S = np.array(rs)
parameters = Lor_fitting(data_norm, std, rs_S, [420,580], [485,545], 520) #Lorentz fitting

width_S.append(parameters[0])
width_S_std.append(parameters[1])
pos_S.append(parameters[2])
pos_S_std.append(parameters[3])
amp_S.append(parameters[4])
amp_S_std.append(parameters[5])


#0 Volts ## Anti-Stokes #######################################################
print('######### 0 Volts - Anti-Stokes')
path = os.path.join(__location__, 'raw_data/Si_0mA_30mW_10sec_20acc_517nm.txt')
path_bg = os.path.join(__location__, 'raw_data/Si_0mA_30mW_10sec_5acc_517nm_bg.txt')

time = 10 #sec
power = 30 #mW

data, std, rs = median_acc(path, 20)
data_corr = bg_subtraction(path_bg, 5, data)
data_norm, std = norm(data_corr, std, time, power)

AS_0 = data_norm
rs_AS = np.array(rs)

parameters = Lor_fitting(data_norm, std, rs_AS, [-580,-420], [-580,-420], 520)

width_AS.append(parameters[0])
width_AS_std.append(parameters[1])
pos_AS.append(parameters[2])
pos_AS_std.append(parameters[3])
amp_AS.append(parameters[4])
amp_AS_std.append(parameters[5])

#150 mAmps 0p5xCurrent ## Stokes ###############################################
print('######### 150 mAmps - Stokes')
path = os.path.join(__location__, 'raw_data/Si_150mA_30mW_10sec_20acc_545nm.txt')
path_bg = os.path.join(__location__, 'raw_data/Si_150mA_30mW_10sec_5acc_545nm_bg.txt')

time = 10 #sec
power = 30 #mW

data, std, rs = median_acc(path, 20)
data_corr = bg_subtraction(path_bg, 5, data)
data_norm, std = norm(data_corr, std, time, power)

S_150 = data_norm
parameters = Lor_fitting(data_norm, std, rs_S, [420,580], [420,580], 520)

width_S.append(parameters[0])
width_S_std.append(parameters[1])
pos_S.append(parameters[2])
pos_S_std.append(parameters[3])
amp_S.append(parameters[4])
amp_S_std.append(parameters[5])


#150 mAmps ## Anti-Stokes ##########################################
print('######### 150 mAmps - Anti-Stokes')
path = os.path.join(__location__, 'raw_data/Si_150mA_30mW_10sec_20acc_517nm.txt')
path_bg = os.path.join(__location__, 'raw_data/Si_150mA_30mW_10sec_5acc_517nm_bg.txt')

time = 10 #sec
power = 30 #mW

data, std, rs = median_acc(path, 20)
data_corr = bg_subtraction(path_bg, 5, data)
data_norm, std = norm(data_corr, std, time, power)

AS_150 = data_norm
parameters = Lor_fitting(data_norm, std, rs_AS, [-580,-420], [-580,-420], 520)

width_AS.append(parameters[0])
width_AS_std.append(parameters[1])
pos_AS.append(parameters[2])
pos_AS_std.append(parameters[3])
amp_AS.append(parameters[4])
amp_AS_std.append(parameters[5])


#300 mAmps ## Stokes ###########################################################
print('######### 300 mAmps Stokes')
path = os.path.join(__location__, 'raw_data/Si_300mA_30mW_10sec_20acc_545nm.txt')
path_bg = os.path.join(__location__, 'raw_data/Si_300mA_30mW_10sec_5acc_545nm_bg.txt')

time = 10 #sec
power = 30 #mW

data, std, rs = median_acc(path, 20)
data_corr = bg_subtraction(path_bg, 5, data)
data_norm, std = norm(data_corr, std, time, power)

S_300 = data_norm
parameters = Lor_fitting(data_norm, std, rs_S, [420,580], [420,580], 520)

width_S.append(parameters[0])
width_S_std.append(parameters[1])
pos_S.append(parameters[2])
pos_S_std.append(parameters[3])
amp_S.append(parameters[4])
amp_S_std.append(parameters[5])


# 300 mAmps ## Anti-Stokes ######################################################
print('######### 300 mAmps Anti-Stokes')
path = os.path.join(__location__, 'raw_data/Si_300mA_30mW_10sec_20acc_517nm.txt')
path_bg = os.path.join(__location__, 'raw_data/Si_300mA_30mW_10sec_5acc_517nm_bg.txt')

time = 10 #sec
power = 30 #mW

data, std, rs = median_acc(path, 20)
data_corr = bg_subtraction(path_bg, 5, data)
data_norm, std = norm(data_corr, std, time, power)

AS_300 = data_norm
parameters = Lor_fitting(data_norm, std, rs_AS, [-580,-420], [-580,-420], 520)

width_AS.append(parameters[0])
width_AS_std.append(parameters[1])
pos_AS.append(parameters[2])
pos_AS_std.append(parameters[3])
amp_AS.append(parameters[4])
amp_AS_std.append(parameters[5])


#450 mAmps # Stokes #################################################
print('######### 450 mAmps - Stokes')
path = os.path.join(__location__, 'raw_data/Si_450mA_30mW_10sec_20acc_545nm.txt')
path_bg = os.path.join(__location__, 'raw_data/Si_450mA_30mW_10sec_5acc_545nm_bg.txt')
bg_50Vx2 = load_data(path_bg)

time = 10 #sec
power = 30 #mW

data, std, rs = median_acc(path, 20)
data_corr = bg_subtraction(path_bg, 5, data)
data_norm, std = norm(data_corr, std, time, power)

S_450 = data_norm
parameters = Lor_fitting(data_norm, std, rs_S, [420,580], [420,580], 520)

width_S.append(parameters[0])
width_S_std.append(parameters[1])
pos_S.append(parameters[2])
pos_S_std.append(parameters[3])
amp_S.append(parameters[4])
amp_S_std.append(parameters[5])


#450 mAmps ## Anti-Stokes ############################################
print('######### 450 mAmps - Anti-Stokes')
path = os.path.join(__location__, 'raw_data/Si_450mA_30mW_10sec_20acc_517nm.txt')
path_bg = os.path.join(__location__, 'raw_data/Si_450mA_30mW_10sec_5acc_517nm_bg.txt')

time = 10 #sec
power = 30 #mW

data, std, rs = median_acc(path, 20)
data_corr = bg_subtraction(path_bg, 5, data)
data_norm, std = norm(data_corr, std, time, power)

AS_450 = data_norm
parameters = Lor_fitting(data_norm, std, rs_AS, [-580,-420], [-580,-420], 520)

width_AS.append(parameters[0])
width_AS_std.append(parameters[1])
pos_AS.append(parameters[2])
pos_AS_std.append(parameters[3])
amp_AS.append(parameters[4])
amp_AS_std.append(parameters[5])

#600 mAmps # Stokes #################################################
print('######### 600 mAmps - Stokes')
path = os.path.join(__location__, 'raw_data/Si_600mA_30mW_10sec_20acc_545nm.txt')
path_bg = os.path.join(__location__, 'raw_data/Si_600mA_30mW_10sec_5acc_545nm_bg.txt')

time = 10 #sec
power = 30 #mW

data, std, rs = median_acc(path, 20)
data_corr = bg_subtraction(path_bg, 5, data)
data_norm, std = norm(data_corr, std, time, power)

S_600 = data_norm
parameters = Lor_fitting(data_norm, std, rs_S, [420,580], [420,580], 520)

width_S.append(parameters[0])
width_S_std.append(parameters[1])
pos_S.append(parameters[2])
pos_S_std.append(parameters[3])
amp_S.append(parameters[4])
amp_S_std.append(parameters[5])


#600 mAmps ## Anti-Stokes ############################################
print('######### 600 mAmps - Anti-Stokes')
path = os.path.join(__location__, 'raw_data/Si_600mA_30mW_10sec_20acc_517nm.txt')
path_bg = os.path.join(__location__, 'raw_data/Si_600mA_30mW_10sec_5acc_517nm_bg.txt')

time = 10 #sec
power = 30 #mW

data, std, rs = median_acc(path, 20)
data_corr = bg_subtraction(path_bg, 5, data)
data_norm, std = norm(data_corr, std, time, power)

AS_600 = data_norm
parameters = Lor_fitting(data_norm, std, rs_AS, [-580,-420], [-580,-420], 520)

width_AS.append(parameters[0])
width_AS_std.append(parameters[1])
pos_AS.append(parameters[2])
pos_AS_std.append(parameters[3])
amp_AS.append(parameters[4])
amp_AS_std.append(parameters[5])

#750 mAmps # Stokes #################################################
print('######### 750 mAmps - Stokes')
path = os.path.join(__location__, 'raw_data/Si_750mA_30mW_10sec_20acc_545nm.txt')
path_bg = os.path.join(__location__, 'raw_data/Si_750mA_30mW_10sec_5acc_545nm_bg.txt')

time = 10 #sec
power = 30 #mW

data, std, rs = median_acc(path, 20)
data_corr = bg_subtraction(path_bg, 5, data)
data_norm, std = norm(data_corr, std, time, power)

S_750 = data_norm
parameters = Lor_fitting(data_norm, std, rs_S, [420,580], [420,580], 520)

width_S.append(parameters[0])
width_S_std.append(parameters[1])
pos_S.append(parameters[2])
pos_S_std.append(parameters[3])
amp_S.append(parameters[4])
amp_S_std.append(parameters[5])


#750 mAmps ## Anti-Stokes ############################################
print('######### 750 mAmps - Anti-Stokes')
path = os.path.join(__location__, 'raw_data/Si_750mA_30mW_10sec_20acc_517nm.txt')
path_bg = os.path.join(__location__, 'raw_data/Si_750mA_30mW_10sec_5acc_517nm_bg.txt')

time = 10 #sec
power = 30 #mW

data, std, rs = median_acc(path, 20)
data_corr = bg_subtraction(path_bg, 5, data)
data_norm, std = norm(data_corr, std, time, power)

AS_750 = data_norm
parameters = Lor_fitting(data_norm, std, rs_AS, [-580,-420], [-580,-420], 520)

width_AS.append(parameters[0])
width_AS_std.append(parameters[1])
pos_AS.append(parameters[2])
pos_AS_std.append(parameters[3])
amp_AS.append(parameters[4])
amp_AS_std.append(parameters[5])

# %%
####### Calculate Temperatures with Bose Function #############################
Temperatures = [Temp(pos_S[0], 531.79, amp_S[0], amp_AS[0], amp_S_std[0], amp_AS_std[0])[0],
        Temp(pos_S[1], 531.79, amp_S[1], amp_AS[1], amp_S_std[1], amp_AS_std[1])[0],
        Temp(pos_S[2], 531.79, amp_S[2], amp_AS[2],amp_S_std[2], amp_AS_std[2])[0],
        Temp(pos_S[3], 531.79, amp_S[3], amp_AS[3], amp_S_std[3], amp_AS_std[3])[0],
        Temp(pos_S[4], 531.79, amp_S[4], amp_AS[4], amp_S_std[4], amp_AS_std[4])[0],
        Temp(pos_S[5], 531.79, amp_S[5], amp_AS[5], amp_S_std[5], amp_AS_std[5])[0]]

err = [Temp(pos_S[0], 531.79, amp_S[0], amp_AS[0], amp_S_std[0], amp_AS_std[0])[1],
        Temp(pos_S[1], 531.79, amp_S[1], amp_AS[1], amp_S_std[1], amp_AS_std[1])[1],
        Temp(pos_S[2], 531.79, amp_S[2], amp_AS[2],amp_S_std[2], amp_AS_std[2])[1],
        Temp(pos_S[3], 531.79, amp_S[3], amp_AS[3], amp_S_std[3], amp_AS_std[3])[1],
        Temp(pos_S[4], 531.79, amp_S[4], amp_AS[4], amp_S_std[4], amp_AS_std[4])[1],
        Temp(pos_S[5], 531.79, amp_S[5], amp_AS[5], amp_S_std[5], amp_AS_std[5])[1]]

# %% Bose correction of Raman spectra

S_0_bose = S_0/Corr_S(rs_S, Temperatures[0])
AS_0_bose = AS_0/Corr_AS(rs_AS, Temperatures[0])

S_150_bose = S_150/Corr_S(rs_S, Temperatures[1])
AS_150_bose = AS_150/Corr_AS(rs_AS, Temperatures[1])

S_300_bose = S_300/Corr_S(rs_S, Temperatures[2])
AS_300_bose = AS_300/Corr_AS(rs_AS, Temperatures[2])

S_450_bose = S_450/Corr_S(rs_S, Temperatures[3])
AS_450_bose = AS_450/Corr_AS(rs_AS, Temperatures[3])

S_600_bose = S_600/Corr_S(rs_S, Temperatures[4])
AS_600_bose = AS_600/Corr_AS(rs_AS, Temperatures[4])

S_750_bose = S_750/Corr_S(rs_S, Temperatures[5])
AS_750_bose = AS_750/Corr_AS(rs_AS, Temperatures[5])

# %%

# Raman data with Fan off #######################################################

x_axis_fanoff = [0, 50, 150, 300, 450, 600, 750]
width_S_fanoff = []
width_S_fanoff_std = []
pos_S_fanoff = []
pos_S_fanoff_std = []
amp_S_fanoff = []
amp_S_fanoff_std = []
width_AS_fanoff = []
width_AS_fanoff_std = []
pos_AS_fanoff = []
pos_AS_fanoff_std = []
amp_AS_fanoff = []
amp_AS_fanoff_std = []

#0 Volts ## Stokes ############################################################
print('######### 0 Volts - Stokes')
path = os.path.join(__location__, 'raw_data/Si_fanoff_0mA_30mW_10sec_20acc_545nm.txt')
path_bg = os.path.join(__location__, 'raw_data/Si_fanoff_0mA_30mW_10sec_5acc_545nm_bg.txt')

time = 10 #sec
power = 30 #mW

data, std, rs = median_acc(path, 20)
data_corr = bg_subtraction(path_bg, 5, data)
data_norm, std = norm(data_corr, std, time, power)

S_0_fanoff = data_norm
rs_S_fanoff = np.array(rs)

parameters = Lor_fitting(data_norm, std, rs_S, [420,580], [485,545], 520)

width_S_fanoff.append(parameters[0])
width_S_fanoff_std.append(parameters[1])
pos_S_fanoff.append(parameters[2])
pos_S_fanoff_std.append(parameters[3])
amp_S_fanoff.append(parameters[4])
amp_S_fanoff_std.append(parameters[5])


#0 Volts ## Anti-Stokes #######################################################
print('######### 0 Volts - Anti-Stokes')
path = os.path.join(__location__, 'raw_data/Si_fanoff_0mA_30mW_10sec_20acc_517nm.txt')
path_bg = os.path.join(__location__, 'raw_data/Si_fanoff_0mA_30mW_10sec_5acc_517nm_bg.txt')

time = 10 #sec
power = 30 #mW

data, std, rs = median_acc(path, 20)
data_corr = bg_subtraction(path_bg, 5, data)
data_norm, std = norm(data_corr, std, time, power)

AS_0_fanoff = data_norm
rs_AS_fanoff = np.array(rs)

parameters = Lor_fitting(data_norm, std, rs_AS, [-580,-420], [-580,-420], 520)

width_AS_fanoff.append(parameters[0])
width_AS_fanoff_std.append(parameters[1])
pos_AS_fanoff.append(parameters[2])
pos_AS_fanoff_std.append(parameters[3])
amp_AS_fanoff.append(parameters[4])
amp_AS_fanoff_std.append(parameters[5])

#50 Volts ## Stokes ############################################################
print('######### 50 Volts - Stokes')
path = os.path.join(__location__, 'raw_data/Si_fanoff_50mA_30mW_10sec_20acc_545nm.txt')
path_bg = os.path.join(__location__, 'raw_data/Si_fanoff_50mA_30mW_10sec_5acc_545nm_bg.txt')

time = 10 #sec
power = 30 #mW

data, std, rs = median_acc(path, 20)
data_corr = bg_subtraction(path_bg, 5, data)
data_norm, std = norm(data_corr, std, time, power)

S_50_fanoff = data_norm
parameters = Lor_fitting(data_norm, std, rs_S, [420,580], [485,545], 520)

width_S_fanoff.append(parameters[0])
width_S_fanoff_std.append(parameters[1])
pos_S_fanoff.append(parameters[2])
pos_S_fanoff_std.append(parameters[3])
amp_S_fanoff.append(parameters[4])
amp_S_fanoff_std.append(parameters[5])


#50 Volts ## Anti-Stokes #######################################################
print('######### 50 Volts - Anti-Stokes')
path = os.path.join(__location__, 'raw_data/Si_fanoff_50mA_30mW_10sec_20acc_517nm.txt')
path_bg = os.path.join(__location__, 'raw_data/Si_fanoff_50mA_30mW_10sec_5acc_517nm_bg.txt')

time = 10 #sec
power = 30 #mW

data, std, rs = median_acc(path, 20)
data_corr = bg_subtraction(path_bg, 5, data)
data_norm, std = norm(data_corr, std, time, power)

AS_50_fanoff = data_norm
parameters = Lor_fitting(data_norm, std, rs_AS, [-580,-420], [-580,-420], 520)

width_AS_fanoff.append(parameters[0])
width_AS_fanoff_std.append(parameters[1])
pos_AS_fanoff.append(parameters[2])
pos_AS_fanoff_std.append(parameters[3])
amp_AS_fanoff.append(parameters[4])
amp_AS_fanoff_std.append(parameters[5])

#150 mAmps 0p5xCurrent ## Stokes ###############################################
print('######### 150 mAmps - Stokes')
path = os.path.join(__location__, 'raw_data/Si_fanoff_150mA_30mW_10sec_20acc_545nm.txt')
path_bg = os.path.join(__location__, 'raw_data/Si_fanoff_150mA_30mW_10sec_5acc_545nm_bg.txt')

time = 10 #sec
power = 30 #mW

data, std, rs = median_acc(path, 20)
data_corr = bg_subtraction(path_bg, 5, data)
data_norm, std = norm(data_corr, std, time, power)

S_150_fanoff = data_norm
parameters = Lor_fitting(data_norm, std, rs_S, [420,580], [420,580], 520)

width_S_fanoff.append(parameters[0])
width_S_fanoff_std.append(parameters[1])
pos_S_fanoff.append(parameters[2])
pos_S_fanoff_std.append(parameters[3])
amp_S_fanoff.append(parameters[4])
amp_S_fanoff_std.append(parameters[5])


#150 mAmps ## Anti-Stokes ##########################################
print('######### 150 mAmps - Anti-Stokes')
path = os.path.join(__location__, 'raw_data/Si_fanoff_150mA_30mW_10sec_20acc_517nm.txt')
path_bg = os.path.join(__location__, 'raw_data/Si_fanoff_150mA_30mW_10sec_5acc_517nm_bg.txt')

time = 10 #sec
power = 30 #mW

data, std, rs = median_acc(path, 20)
data_corr = bg_subtraction(path_bg, 5, data)
data_norm, std = norm(data_corr, std, time, power)

AS_150_fanoff = data_norm
parameters = Lor_fitting(data_norm, std, rs_AS, [-580,-420], [-580,-420], 520)

width_AS_fanoff.append(parameters[0])
width_AS_fanoff_std.append(parameters[1])
pos_AS_fanoff.append(parameters[2])
pos_AS_fanoff_std.append(parameters[3])
amp_AS_fanoff.append(parameters[4])
amp_AS_fanoff_std.append(parameters[5])


#300 mAmps ## Stokes ###########################################################
print('######### 300 mAmps Stokes')
path = os.path.join(__location__, 'raw_data/Si_fanoff_300mA_30mW_10sec_20acc_545nm.txt')
path_bg = os.path.join(__location__, 'raw_data/Si_fanoff_300mA_30mW_10sec_5acc_545nm_bg.txt')

time = 10 #sec
power = 30 #mW

data, std, rs = median_acc(path, 20)
data_corr = bg_subtraction(path_bg, 5, data)
data_norm, std = norm(data_corr, std, time, power)

S_300_fanoff = data_norm
parameters = Lor_fitting(data_norm, std, rs_S, [420,580], [420,580], 520)

width_S_fanoff.append(parameters[0])
width_S_fanoff_std.append(parameters[1])
pos_S_fanoff.append(parameters[2])
pos_S_fanoff_std.append(parameters[3])
amp_S_fanoff.append(parameters[4])
amp_S_fanoff_std.append(parameters[5])


# 300 mAmps ## Anti-Stokes ######################################################
print('######### 300 mAmps Anti-Stokes')
path = os.path.join(__location__, 'raw_data/Si_fanoff_300mA_30mW_10sec_20acc_517nm.txt')
path_bg = os.path.join(__location__, 'raw_data/Si_fanoff_300mA_30mW_10sec_5acc_517nm_bg.txt')

time = 10 #sec
power = 30 #mW

data, std, rs = median_acc(path, 20)
data_corr = bg_subtraction(path_bg, 5, data)
data_norm, std = norm(data_corr, std, time, power)

AS_300_fanoff = data_norm
parameters = Lor_fitting(data_norm, std, rs_AS, [-580,-420], [-580,-420], 520)

width_AS_fanoff.append(parameters[0])
width_AS_fanoff_std.append(parameters[1])
pos_AS_fanoff.append(parameters[2])
pos_AS_fanoff_std.append(parameters[3])
amp_AS_fanoff.append(parameters[4])
amp_AS_fanoff_std.append(parameters[5])


#450 mAmps # Stokes #################################################
print('######### 450 mAmps - Stokes')
path = os.path.join(__location__, 'raw_data/Si_fanoff_450mA_30mW_10sec_20acc_545nm.txt')
path_bg = os.path.join(__location__, 'raw_data/Si_fanoff_450mA_30mW_10sec_5acc_545nm_bg.txt')

time = 10 #sec
power = 30 #mW

data, std, rs = median_acc(path, 20)
data_corr = bg_subtraction(path_bg, 5, data)
data_norm, std = norm(data_corr, std, time, power)

S_450_fanoff = data_norm
parameters = Lor_fitting(data_norm, std, rs_S, [420,580], [420,580], 520)

width_S_fanoff.append(parameters[0])
width_S_fanoff_std.append(parameters[1])
pos_S_fanoff.append(parameters[2])
pos_S_fanoff_std.append(parameters[3])
amp_S_fanoff.append(parameters[4])
amp_S_fanoff_std.append(parameters[5])


#450 mAmps ## Anti-Stokes ############################################
print('######### 450 mAmps - Anti-Stokes')
path = os.path.join(__location__, 'raw_data/Si_fanoff_450mA_30mW_10sec_20acc_517nm.txt')
path_bg = os.path.join(__location__, 'raw_data/Si_fanoff_450mA_30mW_10sec_5acc_517nm_bg.txt')

time = 10 #sec
power = 30 #mW

data, std, rs = median_acc(path, 20)
data_corr = bg_subtraction(path_bg, 5, data)
data_norm, std = norm(data_corr, std, time, power)

AS_450_fanoff = data_norm
parameters = Lor_fitting(data_norm, std, rs_AS, [-580,-420], [-580,-420], 520)

width_AS_fanoff.append(parameters[0])
width_AS_fanoff_std.append(parameters[1])
pos_AS_fanoff.append(parameters[2])
pos_AS_fanoff_std.append(parameters[3])
amp_AS_fanoff.append(parameters[4])
amp_AS_fanoff_std.append(parameters[5])

#600 mAmps # Stokes #################################################
print('######### 600 mAmps - Stokes')
path = os.path.join(__location__, 'raw_data/Si_fanoff_600mA_30mW_10sec_20acc_545nm.txt')
path_bg = os.path.join(__location__, 'raw_data/Si_fanoff_600mA_30mW_10sec_5acc_545nm_bg.txt')

time = 10 #sec
power = 30 #mW

data, std, rs = median_acc(path, 20)
data_corr = bg_subtraction(path_bg, 5, data)
data_norm, std = norm(data_corr, std, time, power)

S_600_fanoff = data_norm
parameters = Lor_fitting(data_norm, std, rs_S, [420,580], [420,580], 520)

width_S_fanoff.append(parameters[0])
width_S_fanoff_std.append(parameters[1])
pos_S_fanoff.append(parameters[2])
pos_S_fanoff_std.append(parameters[3])
amp_S_fanoff.append(parameters[4])
amp_S_fanoff_std.append(parameters[5])


#600 mAmps ## Anti-Stokes ############################################
print('######### 600 mAmps - Anti-Stokes')
path = os.path.join(__location__, 'raw_data/Si_fanoff_600mA_30mW_10sec_20acc_517nm.txt')
path_bg = os.path.join(__location__, 'raw_data/Si_fanoff_600mA_30mW_10sec_5acc_517nm_bg.txt')

time = 10 #sec
power = 30 #mW

data, std, rs = median_acc(path, 20)
data_corr = bg_subtraction(path_bg, 5, data)
data_norm, std = norm(data_corr, std, time, power)

AS_600_fanoff = data_norm
parameters = Lor_fitting(data_norm, std, rs_AS, [-580,-420], [-580,-420], 520)

width_AS_fanoff.append(parameters[0])
width_AS_fanoff_std.append(parameters[1])
pos_AS_fanoff.append(parameters[2])
pos_AS_fanoff_std.append(parameters[3])
amp_AS_fanoff.append(parameters[4])
amp_AS_fanoff_std.append(parameters[5])

#750 mAmps # Stokes #################################################
print('######### 750 mAmps - Stokes')
path = os.path.join(__location__, 'raw_data/Si_fanoff_750mA_30mW_10sec_20acc_545nm.txt')
path_bg = os.path.join(__location__, 'raw_data/Si_fanoff_750mA_30mW_10sec_5acc_545nm_bg.txt')

time = 10 #sec
power = 30 #mW

data, std, rs = median_acc(path, 20)
data_corr = bg_subtraction(path_bg, 5, data)
data_norm, std = norm(data_corr, std, time, power)

S_750_fanoff = data_norm
parameters = Lor_fitting(data_norm, std, rs_S, [420,580], [420,580], 520)

width_S_fanoff.append(parameters[0])
width_S_fanoff_std.append(parameters[1])
pos_S_fanoff.append(parameters[2])
pos_S_fanoff_std.append(parameters[3])
amp_S_fanoff.append(parameters[4])
amp_S_fanoff_std.append(parameters[5])


#750 mAmps ## Anti-Stokes ############################################
print('######### 750 mAmps - Anti-Stokes')
path = os.path.join(__location__, 'raw_data/Si_fanoff_750mA_30mW_10sec_20acc_517nm.txt')
path_bg = os.path.join(__location__, 'raw_data/Si_fanoff_750mA_30mW_10sec_5acc_517nm_bg.txt')

time = 10 #sec
power = 30 #mW

data, std, rs = median_acc(path, 20)
data_corr = bg_subtraction(path_bg, 5, data)
data_norm, std = norm(data_corr, std, time, power)

AS_750_fanoff = data_norm
parameters = Lor_fitting(data_norm, std, rs_AS, [-580,-420], [-580,-420], 520)

width_AS_fanoff.append(parameters[0])
width_AS_fanoff_std.append(parameters[1])
pos_AS_fanoff.append(parameters[2])
pos_AS_fanoff_std.append(parameters[3])
amp_AS_fanoff.append(parameters[4])
amp_AS_fanoff_std.append(parameters[5])

# %%
####### Calculate Temperatures with Bose Function #############################
Temperatures_fanoff = [Temp(pos_S_fanoff[0], 531.79, amp_S_fanoff[0], amp_AS_fanoff[0], amp_S_fanoff_std[0], amp_AS_fanoff_std[0])[0],
        Temp(pos_S_fanoff[1], 531.79, amp_S_fanoff[1], amp_AS_fanoff[1], amp_S_fanoff_std[1], amp_AS_fanoff_std[1])[0],
        Temp(pos_S_fanoff[2], 531.79, amp_S_fanoff[2], amp_AS_fanoff[2], amp_S_fanoff_std[2], amp_AS_fanoff_std[2])[0],
        Temp(pos_S_fanoff[3], 531.79, amp_S_fanoff[3], amp_AS_fanoff[3], amp_S_fanoff_std[3], amp_AS_fanoff_std[3])[0],
        Temp(pos_S_fanoff[4], 531.79, amp_S_fanoff[4], amp_AS_fanoff[4], amp_S_fanoff_std[4], amp_AS_fanoff_std[4])[0],
        Temp(pos_S_fanoff[5], 531.79, amp_S_fanoff[5], amp_AS_fanoff[5], amp_S_fanoff_std[5], amp_AS_fanoff_std[5])[0],
        Temp(pos_S_fanoff[6], 531.79, amp_S_fanoff[6], amp_AS_fanoff[6], amp_S_fanoff_std[6], amp_AS_fanoff_std[6])[0]]

err_fanoff = [Temp(pos_S_fanoff[0], 531.79, amp_S_fanoff[0], amp_AS_fanoff[0], amp_S_fanoff_std[0], amp_AS_fanoff_std[0])[1],
        Temp(pos_S_fanoff[1], 531.79, amp_S_fanoff[1], amp_AS_fanoff[1], amp_S_fanoff_std[1], amp_AS_fanoff_std[1])[1],
        Temp(pos_S_fanoff[2], 531.79, amp_S_fanoff[2], amp_AS_fanoff[2], amp_S_fanoff_std[2], amp_AS_fanoff_std[2])[1],
        Temp(pos_S_fanoff[3], 531.79, amp_S_fanoff[3], amp_AS_fanoff[3], amp_S_fanoff_std[3], amp_AS_fanoff_std[3])[1],
        Temp(pos_S_fanoff[4], 531.79, amp_S_fanoff[4], amp_AS_fanoff[4], amp_S_fanoff_std[4], amp_AS_fanoff_std[4])[1],
        Temp(pos_S_fanoff[5], 531.79, amp_S_fanoff[5], amp_AS_fanoff[5], amp_S_fanoff_std[5], amp_AS_fanoff_std[5])[1],
        Temp(pos_S_fanoff[6], 531.79, amp_S_fanoff[6], amp_AS_fanoff[6], amp_S_fanoff_std[6], amp_AS_fanoff_std[6])[1]]

# %% Bose correction

S_0_fanoff_bose = S_0_fanoff/Corr_S(rs_S, Temperatures_fanoff[0])
AS_0_fanoff_bose = AS_0_fanoff/Corr_AS(rs_AS, Temperatures_fanoff[0])

S_50_fanoff_bose = S_50_fanoff/Corr_S(rs_S, Temperatures_fanoff[1])
AS_50_fanoff_bose = AS_50_fanoff/Corr_AS(rs_AS, Temperatures_fanoff[1])

S_150_fanoff_bose = S_150_fanoff/Corr_S(rs_S, Temperatures_fanoff[2])
AS_150_fanoff_bose = AS_150_fanoff/Corr_AS(rs_AS, Temperatures_fanoff[2])

S_300_fanoff_bose = S_300_fanoff/Corr_S(rs_S, Temperatures_fanoff[3])
AS_300_fanoff_bose = AS_300_fanoff/Corr_AS(rs_AS, Temperatures_fanoff[3])

S_450_fanoff_bose = S_450_fanoff/Corr_S(rs_S, Temperatures_fanoff[4])
AS_450_fanoff_bose = AS_450_fanoff/Corr_AS(rs_AS, Temperatures_fanoff[4])

S_600_fanoff_bose = S_600_fanoff/Corr_S(rs_S, Temperatures_fanoff[5])
AS_600_fanoff_bose = AS_600_fanoff/Corr_AS(rs_AS, Temperatures_fanoff[5])

S_750_fanoff_bose = S_750_fanoff/Corr_S(rs_S, Temperatures_fanoff[6])
AS_750_fanoff_bose = AS_750_fanoff/Corr_AS(rs_AS, Temperatures_fanoff[6])



# %% Figures

######### Plot Stokes data - Fan on ####################################################
plt.figure(dpi=200)
plt.plot(np.array(rs_S), S_0, c = 'blue', label = '0 A')
plt.plot(np.array(rs_AS), AS_0, c = 'blue')
plt.plot(np.array(rs_S), S_150, c = 'green', label = '150 mA')
plt.plot(np.array(rs_AS), AS_150, c = 'green')
plt.plot(np.array(rs_S), S_300, c = 'k', label = '300 mA')
plt.plot(np.array(rs_AS), AS_300, c = 'k')
plt.plot(np.array(rs_S), S_450, c = 'r', label = '450 mA')
plt.plot(np.array(rs_AS), AS_450, c = 'r')
plt.plot(np.array(rs_S), S_600, c = 'y', label = '600 mA')
plt.plot(np.array(rs_AS), AS_600, c = 'y')
plt.plot(np.array(rs_S), S_750, c = 'violet', label = '750 mA')
plt.plot(np.array(rs_AS), AS_750, c = 'violet')
plt.ylim(-5, 70)
plt.plot([320, 830], [0,0], 'k--')
plt.xlim(470, 560)
plt.legend(frameon = False)
plt.xlabel('Raman shift (cm$^{-1}$)')
plt.ylabel('Intensity (Counts / mW*sec)')
plt.yticks([])
plt.tick_params(left = True, right = True, bottom = True, top = True)
plt.tick_params(axis = 'both', direction = 'in')
plt.title('Stokes Raman data - Fan on')
plt.show()

########## Plot anti-Stokes data - Fan on ##############################################
plt.figure(dpi=200)
plt.plot(np.array(rs_S), S_0, c = 'blue', label = '0 A')
plt.plot(np.array(rs_AS), AS_0, c = 'blue')
plt.plot(np.array(rs_S), S_150, c = 'green', label = '150 mA')
plt.plot(np.array(rs_AS), AS_150, c = 'green')
plt.plot(np.array(rs_S), S_300, c = 'k', label = '300 mA')
plt.plot(np.array(rs_AS), AS_300, c = 'k')
plt.plot(np.array(rs_S), S_450, c = 'r', label = '450 mA')
plt.plot(np.array(rs_AS), AS_450, c = 'r')
plt.plot(np.array(rs_S), S_600, c = 'y', label = '600 mA')
plt.plot(np.array(rs_AS), AS_600, c = 'y')
plt.plot(np.array(rs_S), S_750, c = 'violet', label = '750 mA')
plt.plot(np.array(rs_AS), AS_750, c = 'violet')
plt.ylim(-5, 70)
plt.plot([-830, -320], [0,0], 'k--')
plt.xlim(-560, -470)
plt.legend(frameon = False)
plt.xlabel('Raman shift (cm$^{-1}$)')
plt.ylabel('Intensity (Counts / mW*sec)')
plt.yticks([])
plt.tick_params(left = True, right = True, bottom = True, top = True)
plt.tick_params(axis = 'both', direction = 'in')
plt.title('Anti-Stokes Raman data - Fan on')
plt.show()

########## Plot anti-Stokes data - Fan off ##############################################
plt.figure(dpi=200)
offset = 15
plt.plot(np.array(rs_S), np.array(S_0_fanoff)+offset, c = 'blue', label = '0 mA')
plt.plot(np.array(rs_AS)*(-1), AS_0_fanoff, c = 'blue')
plt.plot(np.array(rs_S), np.array(S_50_fanoff)+offset, c = 'lightblue', label = '50 mA')
plt.plot(np.array(rs_AS)*(-1), AS_50_fanoff, c = 'lightblue')
plt.plot(np.array(rs_S), np.array(S_150_fanoff)+offset, c = 'green', label = '150 mA')
plt.plot(np.array(rs_AS)*(-1), AS_150_fanoff, c = 'green')
plt.plot(np.array(rs_S), np.array(S_300_fanoff)+offset, c = 'k', label = '300 mA')
plt.plot(np.array(rs_AS)*(-1), AS_300_fanoff, c = 'k')
plt.plot(np.array(rs_S), np.array(S_450_fanoff)+offset, c = 'r', label = '450 mA')
plt.plot(np.array(rs_AS)*(-1), AS_450_fanoff, c = 'r')
plt.plot(np.array(rs_S), np.array(S_600_fanoff)+offset, c = 'y', label = '600 mA')
plt.plot(np.array(rs_AS)*(-1), AS_600_fanoff, c = 'y')
plt.plot(np.array(rs_S), np.array(S_750_fanoff)+offset, c = 'violet', label = '750 mA')
plt.plot(np.array(rs_AS)*(-1), AS_750_fanoff, c = 'violet')
plt.ylim(-5, 90)
plt.plot([320, 800], [0,0], 'k--')
plt.plot([320, 800], [offset,offset], 'k--')
plt.xlim(470, 530)
plt.text(531, 0, 'anti-Stokes', c = 'k', fontsize = 13)
plt.text(531, 15, 'Stokes', c = 'k', fontsize = 13)
plt.legend(frameon = False)
plt.xlabel('Absolute Value Raman shift (cm$^{-1}$)')
plt.ylabel('Intensity (Counts / mW*sec)')
plt.yticks([])
plt.tick_params(left = True, right = True, bottom = True, top = True)
plt.tick_params(axis = 'both', direction = 'in')
plt.title('Stokes and anti-Stokes Raman data - Fan off')
plt.show()

########## Plot Stokes and anti-Stokes data - Bose corrected data - Fan off  ##############################################
plt.figure(dpi=200)
offset = 10
plt.plot([320, 800], [0,0], 'k--')
plt.plot([320, 800], [offset,offset], 'k--')
plt.plot([320, 800], [2*offset,2*offset], 'k--')
plt.plot([320, 800], [3*offset,3*offset], 'k--')

plt.plot(np.array(rs_S), np.array(S_0_fanoff_bose), c = 'blue', label = '0 mA')
plt.plot(np.array(rs_AS)*(-1), AS_0_fanoff_bose, c = 'darkblue', linestyle = '--')
#plt.plot(np.array(rs_S), np.array(S_50_fanoff)+offset, c = 'lightblue', label = '50 mA')
#plt.plot(np.array(rs_AS)*(-1), np.array(AS_50_fanoff)+offset, c = 'lightblue')
plt.plot(np.array(rs_S), np.array(S_150_fanoff_bose)+1*offset, c = 'lime', label = '150 mA')
plt.plot(np.array(rs_AS)*(-1), np.array(AS_150_fanoff_bose)+1*offset, c = 'green', linestyle = '--')
#plt.plot(np.array(rs_S), np.array(S_300_fanoff)+3*offset, c = 'k', label = '300 mA')
#plt.plot(np.array(rs_AS)*(-1), np.array(AS_300_fanoff)+3*offset, c = 'k')
plt.plot(np.array(rs_S), np.array(S_450_fanoff_bose)+2*offset, c = 'tomato', label = '450 mA')
plt.plot(np.array(rs_AS)*(-1), np.array(AS_450_fanoff_bose)+2*offset, c = 'firebrick', linestyle = '--')
#plt.plot(np.array(rs_S), np.array(S_600_fanoff)+5*offset, c = 'y', label = '600 mA')
#plt.plot(np.array(rs_AS)*(-1), np.array(AS_600_fanoff)+5*offset, c = 'y')
plt.plot(np.array(rs_S), np.array(S_750_fanoff_bose)+3*offset, c = 'hotpink', label = '750 mA')
plt.plot(np.array(rs_AS)*(-1), np.array(AS_750_fanoff_bose)+3*offset, c = 'purple', linestyle = '--')
plt.ylim(-5, 70)
plt.xlim(470, 530)
plt.legend(frameon = False)
plt.xlabel('Absolute Value Raman shift (cm$^{-1}$)')
plt.ylabel('Raman susceptibility')
plt.yticks([])
plt.tick_params(left = True, right = True, bottom = True, top = True)
plt.tick_params(axis = 'both', direction = 'in')
plt.title('Stokes (solid) vs. Anti-Stokes (dashed), Bose corrected')
plt.show()

########## Plot Stokes and anti-Stokes data - Bose corrected ##############################################
plt.figure(dpi=200)
offset = 2
plt.plot(np.array(rs_S), np.array(S_0_fanoff_bose)+offset, c = 'blue', label = '0 mA')
plt.plot(np.array(rs_AS)*(-1), AS_0_fanoff_bose, c = 'blue')
#plt.plot(np.array(rs_S), np.array(S_50_fanoff_bose)+offset, c = 'lightblue', label = '50 mA')
#plt.plot(np.array(rs_AS)*(-1), AS_50_fanoff_bose, c = 'lightblue')
plt.plot(np.array(rs_S), np.array(S_150_fanoff_bose)+offset, c = 'green', label = '150 mA')
plt.plot(np.array(rs_AS)*(-1), AS_150_fanoff_bose, c = 'green')
#plt.plot(np.array(rs_S), np.array(S_300_fanoff)+offset, c = 'k', label = '300 mA')
#plt.plot(np.array(rs_AS)*(-1), AS_300_fanoff, c = 'k')
plt.plot(np.array(rs_S), np.array(S_450_fanoff_bose)+offset, c = 'red', label = '450 mA')
plt.plot(np.array(rs_AS)*(-1), AS_450_fanoff_bose, c = 'red')
#plt.plot(np.array(rs_S), np.array(S_600_fanoff)+offset, c = 'y', label = '600 mA')
#plt.plot(np.array(rs_AS)*(-1), AS_600_fanoff, c = 'y')
plt.plot(np.array(rs_S), np.array(S_750_fanoff_bose)+offset, c = 'hotpink', label = '750 mA')
plt.plot(np.array(rs_AS)*(-1), AS_750_fanoff_bose, c = 'hotpink')
plt.ylim(-1, 5)
plt.plot([320, 800], [0,0], 'k--')
plt.plot([320, 800], [offset,offset], 'k--')
plt.xlim(300, 800)
plt.text(804, 0, 'anti-Stokes', c = 'k', fontsize = 13)
plt.text(804, 2, 'Stokes', c = 'k', fontsize = 13)
plt.legend(frameon = False)
plt.xlabel('Absolute Value Raman shift (cm$^{-1}$)')
plt.ylabel('Raman susceptibility')
plt.yticks([])
plt.tick_params(left = True, right = True, bottom = True, top = True)
plt.tick_params(axis = 'both', direction = 'in')
plt.title('Stokes and anti-Stokes data, zoomed in, Bose corrected')
plt.show()


######### Phonons Widths - Fan on ######################################################
plt.figure(figsize=(3.5, 4), dpi=200)
plt.errorbar(x_axis, width_S, yerr = width_S_std, marker = 'o', color = 'red', alpha=0.7, capsize = 1.2, linewidth=1.1, label = 'Stokes')
plt.errorbar(x_axis, width_AS, yerr = width_AS_std, marker = 'o', color = 'blue', alpha=0.7, capsize = 1.2, linewidth=1.1, label = 'Anti-Stokes')
plt.ylim(0, 12)
#plt.xlim(470, 560)
plt.legend(frameon = False)
plt.xlabel('I (mA)')
plt.ylabel('$\Gamma$ (cm$^{-1}$)')
#plt.yticks([])
plt.grid(linestyle=':')
plt.tick_params(left = True, right = True, bottom = True, top = True)
plt.tick_params(axis = 'both', direction = 'in')
plt.title('$\Gamma$ - Fan on')
plt.show()

######### Phonon Frequencies - Fan on ##################################################
plt.figure(figsize=(3.5, 4), dpi=200)
plt.errorbar(x_axis, pos_S, yerr = pos_S_std, marker = 'o', color = 'red', alpha=0.7, capsize = 1.2, linewidth=1.1, label = 'Stokes')
plt.errorbar(x_axis, pos_AS, yerr = pos_AS_std, marker = 'o', color = 'blue', alpha=0.7, capsize = 1.2, linewidth=1.1, label = 'Anti-Stokes')
plt.ylim(500, 529)
#plt.xlim(470, 560)
plt.legend(frameon = False)
plt.xlabel('I (mA)')
plt.ylabel('$\omega_0$ (cm$^{-1}$)')
#plt.yticks([])
plt.grid(linestyle=':')
plt.tick_params(left = True, right = True, bottom = True, top = True)
plt.tick_params(axis = 'both', direction = 'in')
plt.title('$\omega_0$ - Fan on')
plt.show()

######### Phonon Pos as a function of Temperature #############################
plt.figure(figsize=(3.5, 4), dpi=200)
plt.errorbar(Temperatures, pos_S, yerr = pos_S_std, marker = 'o', color = 'red', ls='none', alpha=0.7, capsize = 1.2, linewidth=1.1, label = 'Stokes')
plt.errorbar(Temperatures, pos_AS, yerr = pos_AS_std, marker = 'o', color = 'blue', ls='none', alpha=0.7, capsize = 1.2, linewidth=1.1, label = 'Anti-Stokes')
plt.plot(shifts['x'], shifts['y'], '+k', label = 'PRB')# ref from PRB
plt.plot(shifts_fit['x'], shifts_fit['y'], 'k--')# fit from PRB
plt.ylim(500, 529)
plt.xlim(100, 1100)
plt.legend(frameon = False)
plt.xlabel('T (K)')
plt.ylabel('$\omega_0$ (cm$^{-1}$)')
#plt.yticks([])
plt.grid(linestyle=':')
plt.tick_params(left = True, right = True, bottom = True, top = True)
plt.tick_params(axis = 'both', direction = 'in')
plt.title('$\omega_0$ as a Function of Temp - Fan on')
plt.show()

######### Phonon Width as a function of Temperature #############################
print(width_ref_err)
plt.figure(figsize=(3.5, 4), dpi=200)
plt.errorbar(Temperatures, width_S, yerr = width_S_std, marker = 'o', ls='none', color = 'red', alpha=0.7, capsize = 1.2, linewidth=1.1, label = 'Stokes')
plt.errorbar(Temperatures, width_AS, yerr = width_AS_std, marker = 'o', ls='none', color = 'blue', alpha=0.7, capsize = 1.2, linewidth=1.1, label = 'Anti-Stokes')
plt.errorbar([20, 460, 770], width_ref, yerr = width_ref_err, marker = 'o', color = 'k', ls='none', alpha=0.7, capsize = 1.2, label = 'PRB Paper')
plt.errorbar([460, 770], width_ref_AS, yerr = width_ref_AS_err, marker = 'o', color = 'gray', ls='none', alpha=0.7, capsize = 1.2)
plt.ylim(2, 15)
plt.xlim(0, 1100)
plt.legend(frameon = False)
plt.xlabel('T (K)')
plt.ylabel('$\Gamma$ (cm$^{-1}$)')
#plt.yticks([])
plt.grid(linestyle=':')
plt.tick_params(left = True, right = True, bottom = True, top = True)
plt.tick_params(axis = 'both', direction = 'in')
plt.title('$\Gamma$ as a Function of Temp - Fan on')
plt.show()

######### Calculated Temperatures #############################################
plt.figure(figsize=(3.5, 4), dpi=200)
#plt.plot(x_axis, Temperatures, 'ok--')
plt.errorbar(x_axis, Temperatures, yerr = err, marker = 'o', color = 'k', alpha=0.7, capsize = 1.2, linewidth=1.1, label = 'Stokes')
plt.ylim(250, 1100)
plt.xlim(-50, 800)
plt.xlabel('I (mA)')
plt.ylabel('Temperature (K)')
#plt.yticks([])
plt.grid(linestyle=':')
plt.tick_params(left = True, right = True, bottom = True, top = True)
plt.tick_params(axis = 'both', direction = 'in')
plt.title('Temperatures - Fan on')
plt.show()

######### Phonons Widths ######################################################
plt.figure(figsize=(3.5, 4), dpi=200)
plt.errorbar(x_axis_fanoff, width_S_fanoff, yerr = width_S_fanoff_std, marker = 'o', color = 'red', alpha=0.7, capsize = 1.2, linewidth=1.1, label = 'Stokes')
plt.errorbar(x_axis_fanoff, width_AS_fanoff, yerr = width_AS_fanoff_std, marker = 'o', color = 'blue', alpha=0.7, capsize = 1.2, linewidth=1.1, label = 'Anti-Stokes')
plt.ylim(0, 12)
#plt.xlim(470, 560)
plt.legend(frameon = False)
plt.xlabel('I (mA)')
plt.ylabel('$\Gamma$ (cm$^{-1}$)')
#plt.yticks([])
plt.grid(linestyle=':')
plt.tick_params(left = True, right = True, bottom = True, top = True)
plt.tick_params(axis = 'both', direction = 'in')
plt.title('$\Gamma$ - Fan off')
plt.show()

######### Phonon Frequencies ##################################################
plt.figure(figsize=(3.5, 4), dpi=200)
plt.errorbar(x_axis_fanoff, pos_S_fanoff, yerr = pos_S_fanoff_std, marker = 'o', color = 'red', alpha=0.7, capsize = 1.2, linewidth=1.1, label = 'Stokes')
plt.errorbar(x_axis_fanoff, pos_AS_fanoff, yerr = pos_AS_fanoff_std, marker = 'o', color = 'blue', alpha=0.7, capsize = 1.2, linewidth=1.1, label = 'Anti-Stokes')
plt.ylim(500, 529)
#plt.xlim(470, 560)
plt.legend(frameon = False)
plt.xlabel('I (mA)')
plt.ylabel('$\omega_0$ (cm$^{-1}$)')
#plt.yticks([])
plt.grid(linestyle=':')
plt.tick_params(left = True, right = True, bottom = True, top = True)
plt.tick_params(axis = 'both', direction = 'in')
plt.title('$\omega_0$ - Fan off')
plt.show()


######### Phonon Pos as a function of Temperature #############################
plt.figure(figsize=(3.5, 4), dpi=200)
plt.errorbar(Temperatures_fanoff, pos_S_fanoff, yerr = pos_S_fanoff_std, marker = 'o', color = 'red', ls='none', alpha=0.7, capsize = 1.2, linewidth=1.1, label = 'Stokes')
plt.errorbar(Temperatures_fanoff, pos_AS_fanoff, yerr = pos_AS_fanoff_std, marker = 'o', color = 'blue', ls='none', alpha=0.7, capsize = 1.2, linewidth=1.1, label = 'Anti-Stokes')
plt.plot(shifts['x'], shifts['y'], '+k', label = 'PRB')# ref from PRB
plt.plot(shifts_fit['x'], shifts_fit['y'], 'k--')# fit from PRB
plt.ylim(500, 529)
plt.xlim(100, 1100)
plt.legend(frameon = False)
plt.xlabel('T (K)')
plt.ylabel('$\omega_0$ (cm$^{-1}$)')
#plt.yticks([])
plt.grid(linestyle=':')
plt.tick_params(left = True, right = True, bottom = True, top = True)
plt.tick_params(axis = 'both', direction = 'in')
plt.title('$\omega_0$ as a Function of Temp - Fan off')
plt.show()

######### Phonon Width as a function of Temperature #############################
print(width_ref_err)
plt.figure(figsize=(3.5, 4), dpi=200)
plt.errorbar(Temperatures_fanoff, width_S_fanoff, yerr = width_S_fanoff_std, marker = 'o', ls='none', color = 'red', alpha=0.7, capsize = 1.2, linewidth=1.1, label = 'Stokes')
plt.errorbar(Temperatures_fanoff, width_AS_fanoff, yerr = width_AS_fanoff_std, marker = 'o', ls='none', color = 'blue', alpha=0.7, capsize = 1.2, linewidth=1.1, label = 'Anti-Stokes')
plt.errorbar([20, 460, 770], width_ref, yerr = width_ref_err, marker = 'o', color = 'k', ls='none', alpha=0.7, capsize = 1.2, label = 'PRB Paper')
plt.errorbar([460, 770], width_ref_AS, yerr = width_ref_AS_err, marker = 'o', color = 'gray', ls='none', alpha=0.7, capsize = 1.2)
plt.ylim(2, 15)
plt.xlim(0, 1100)
plt.legend(frameon = False)
plt.xlabel('T (K)')
plt.ylabel('$\Gamma$ (cm$^{-1}$)')
#plt.yticks([])
plt.grid(linestyle=':')
plt.tick_params(left = True, right = True, bottom = True, top = True)
plt.tick_params(axis = 'both', direction = 'in')
plt.title('$\Gamma$ as a Function of Temp - Fan off')
plt.show()

######### Calculated Temperatures Comparison #############################################
plt.figure(figsize=(3.5, 4), dpi=200)
#plt.plot(x_axis, Temperatures, 'ok--')
plt.errorbar(x_axis_fanoff, Temperatures_fanoff, yerr = err_fanoff, marker = 'o', color = 'k', alpha=0.7, capsize = 1.2, linewidth=1.1, label = 'Fan off')
plt.errorbar(x_axis, Temperatures, yerr = err, marker = 'o', color = 'b', alpha=0.7, capsize = 1.2, linewidth=1.1, label = 'Fan on')
plt.ylim(250, 1100)
plt.xlim(-50, 800)
plt.xlabel('I (mA)')
plt.ylabel('T (K)')
#plt.yticks([])
plt.grid(linestyle=':')
plt.tick_params(left = True, right = True, bottom = True, top = True)
plt.tick_params(axis = 'both', direction = 'in')
plt.title('Temperatures')
plt.legend(frameon=False)
plt.show()

######### Linewidth_Comparison #############################################
plt.figure(figsize=(3.5, 4), dpi=200)
#plt.plot(x_axis, Temperatures, 'ok--')
plt.errorbar(Temperatures_fanoff, width_S_fanoff, yerr = width_S_fanoff_std, marker = 'o', color = 'k', alpha=0.7, capsize = 1.2, linewidth=1.1, label = 'Fan off')
plt.errorbar(Temperatures,width_S, yerr = width_S_std, marker = 'o', color = 'b', alpha=0.7, capsize = 1.2, linewidth=1.1, label = 'Fan on')
plt.ylim(2, 15)
plt.xlim(0, 1100)
plt.xlabel('T (K)')
plt.ylabel('$\Gamma$ (cm-1)')
#plt.yticks([])
plt.grid(linestyle=':')
plt.tick_params(left = True, right = True, bottom = True, top = True)
plt.tick_params(axis = 'both', direction = 'in')
plt.title('$\Gamma$ Comparison Fan on/off')
plt.legend(frameon=False)
plt.show()

print('#########################################################')
print('Temperatures Fan: ', Temperatures)
print('Temperatures without Fan: ', Temperatures_fanoff)

# %%
'''
save_data(rs_S, S_0, 'S_0mA_fan_on')
save_data(rs_S, S_150, 'S_150mA_fan_on')
save_data(rs_S, S_300, 'S_300mA_fan_on')
save_data(rs_S, S_450, 'S_450mA_fan_on')
save_data(rs_S, S_600, 'S_600mA_fan_on')
save_data(rs_S, S_750, 'S_750mA_fan_on')

save_data(rs_AS, AS_0, 'AS_0mA_fan_on')
save_data(rs_AS, AS_150, 'AS_150mA_fan_on')
save_data(rs_AS, AS_300, 'AS_300mA_fan_on')
save_data(rs_AS, AS_450, 'AS_450mA_fan_on')
save_data(rs_AS, AS_600, 'AS_600mA_fan_on')
save_data(rs_AS, AS_750, 'AS_750mA_fan_on')

save_data(rs_S_fanoff, S_0_fanoff, 'S_0mA_fan_off')
save_data(rs_S_fanoff, S_50_fanoff, 'S_50mA_fan_off')
save_data(rs_S_fanoff, S_150_fanoff, 'S_150mA_fan_off')
save_data(rs_S_fanoff, S_300_fanoff, 'S_300mA_fan_off')
save_data(rs_S_fanoff, S_450_fanoff, 'S_450mA_fan_off')
save_data(rs_S_fanoff, S_600_fanoff, 'S_600mA_fan_off')
save_data(rs_S_fanoff, S_750_fanoff, 'S_750mA_fan_off')

save_data(rs_AS_fanoff, AS_0_fanoff, 'AS_0mA_fan_off')
save_data(rs_AS_fanoff, AS_50_fanoff, 'AS_50mA_fan_off')
save_data(rs_AS_fanoff, AS_150_fanoff, 'AS_150mA_fan_off')
save_data(rs_AS_fanoff, AS_300_fanoff, 'AS_300mA_fan_off')
save_data(rs_AS_fanoff, AS_450_fanoff, 'AS_450mA_fan_off')
save_data(rs_AS_fanoff, AS_600_fanoff, 'AS_600mA_fan_off')
save_data(rs_AS_fanoff, AS_750_fanoff, 'AS_750mA_fan_off')
'''
#export results
zip(x_axis, Temperatures, np.array(err), width_S, np.array(width_S_std), width_AS, np.array(width_AS_std), pos_S, np.array(pos_S_std), pos_AS, np.array(pos_AS_std))
with open('Fan_on_analysis_results.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['Current (mA)', 'calculated Temp (K)', 'Temp err', 'Stokes Phonon width', 'Stokes width err', 'AS Phonon width', 'AS width err', 'Stokes phonon pos (w0)', 'Stokes w0 err', 'AS phonon pos (w0)', 'AS w0 err'])
    writer.writerows(zip(x_axis, Temperatures, err, width_S, width_S_std, width_AS, width_AS_std, pos_S, pos_S_std, pos_AS, pos_AS_std))
    
#export results
zip(x_axis_fanoff, Temperatures_fanoff, np.array(err_fanoff), width_S_fanoff, np.array(width_S_fanoff_std), width_AS_fanoff, np.array(width_AS_fanoff_std), pos_S_fanoff, np.array(pos_S_fanoff_std), pos_AS_fanoff, np.array(pos_AS_fanoff_std))
with open('Fan_off_analysis_results.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['Current (mA)', 'calculated Temp (K)', 'Temp err', 'Stokes Phonon width', 'Stokes width err', 'AS Phonon width', 'AS width err', 'Stokes phonon pos (w0)', 'Stokes w0 err', 'AS phonon pos (w0)', 'AS w0 err'])
    writer.writerows(zip(x_axis, Temperatures, err, width_S, width_S_std, width_AS, width_AS_std, pos_S, pos_S_std, pos_AS, pos_AS_std))
