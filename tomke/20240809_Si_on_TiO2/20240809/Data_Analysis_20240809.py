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
def bg_subtraction(path_bg, N_bg, corr, multi, data):
    bg, std, rs = median_acc(path_bg, N_bg)
    #smooth bg
    bg_sm = [bg[0],bg[1]]
    for i in range(2, len(bg)-2):
        bg_sm.append((bg[i-2]+bg[i-1]+bg[i]+bg[i+1]+bg[i+2])/5)
    bg_sm.append(bg[len(bg)-2])
    bg_sm.append(bg[len(bg)-1])
    
    data_sub = np.array(data)-(np.array(bg_sm)*multi+corr)
    
    plt.figure(dpi=200)
    plt.title('Background Correction')
    plt.plot(data, 'k')
    plt.plot(np.array(bg)*multi+corr, 'k')
    plt.plot(np.array(bg_sm)*multi+corr, 'r')
    #plt.plot(data_sub, 'g')
    plt.xlim(50, 1300)
    plt.xlabel('Raman Shift (cm-1)')
    plt.ylabel('Counts')
    plt.tick_params(left = True, right = True, bottom = True, top = True)
    plt.tick_params(axis = 'both', direction = 'in')
    plt.grid(linestyle=':')
    plt.show()
    return data_sub
    
    
def bg_corr(path_bg, bg_region, data, rs, N_bg):#Background correction
    #Take a bg spectrum (same room conditions, same integration time, laser off)
    #bg will be fitted to avoid extra noise due to subtraction
    
    # if no bg data is available, fit bg level between phonons ################
    if path_bg != False:
        
        bg_wave, std, rs = median_acc(path_bg, N_bg)
        x = np.linspace(1, len(bg_wave), len(bg_wave))
        def fitfunc(x, y0, a, A1, G1, x0_1, A2, G2, x0_2):
            return y0+a*x+A1*x*G1/((x**2-x0_1**2)**2+G1**2*x**2)+A2*x*G2/((x**2-x0_2**2)**2+G2**2*x**2)
        gmodel = Model(fitfunc)
        result = gmodel.fit(bg_wave, x=x, y0=60000, a = 0.001, A1=1000000, G1=5, x0_1=600, A2=1000000, G2=5,x0_2=620)
        #print('######## Background fitting #########')
        #print(result.fit_report())
        data_corr = np.array(data)-np.array(result.best_fit)
        
        plt.figure(dpi=200)
        plt.title('Background Correction')
        plt.plot(bg_wave, 'k')
        plt.plot(result.best_fit, 'r')
        #plt.plot(result.init_fit, 'b')
        plt.plot(data, 'k')
        plt.xlim(500, 800)
        #plt.plot(data_corr, 'r')
        plt.xlabel('Raman Shift (cm-1)')
        plt.ylabel('Counts')
        plt.tick_params(left = True, right = True, bottom = True, top = True)
        plt.tick_params(axis = 'both', direction = 'in')
        plt.grid(linestyle=':')
        plt.show()
        
    # fit bg data #######################################################
    else:
        i_start=abs(np.array(rs)-bg_region[1]).argmin()
        i_end=abs(np.array(rs)-bg_region[0]).argmin()
        bg_wave = data
        weight = []
        for i in range(len(data)):
            if i < i_start:
                weight.append(0)
            elif i > i_end:
                weight.append(0)
            else:
                weight.append(1)
        x = np.linspace(1, len(bg_wave), len(bg_wave))
        def fitfunc(x, y0, a):
            return y0+a*x
        gmodel = Model(fitfunc)
        result = gmodel.fit(bg_wave, weights = weight, x=x, y0=600, a = 0.0001)
        #print('######## Background fitting #########')
        #print(result.fit_report())
        data_corr = np.array(data)-np.array(result.best_fit)
        
        plt.figure(dpi=200)
        plt.title('Background Correction')
        plt.plot(rs, bg_wave, 'k')
        plt.plot(rs, result.best_fit, 'r')
        plt.plot(rs, data, 'k')
        #plt.plot(rs, data_corr, 'r')
        plt.xlabel('Raman Shift (cm-1)')
        plt.ylabel('Counts')
        #plt.xlim(50, 1300)
        plt.tick_params(left = True, right = True, bottom = True, top = True)
        plt.tick_params(axis = 'both', direction = 'in')
        plt.grid(linestyle=':')
        plt.show()
        
    return data_corr
# %% Normalization
def norm(data, std, time, power):#Normalization to compare different measurements
    data_norm = np.array(data)/(time*power)
    std = np.array(std)/(time*power)
    return data_norm, std
   
# %% Lorentzian Fitting with lmfit
def Lor_fitting(data, std, rs, fit_range, plot_range, pos, win):                    # plot_range = range on x-Axis shown in the figure
    if win == 'S':
        x = rs
        i_end=abs(np.array(x)-fit_range[0]).argmin()                         # fit_range = range on x-Axis used for the fit algorithm
        i_start=abs(np.array(x)-fit_range[1]).argmin()
    else:
        x = -np.array(rs)
        i_start=abs(np.array(x)-fit_range[0]).argmin()                         # fit_range = range on x-Axis used for the fit algorithm
        i_end=abs(np.array(x)-fit_range[1]).argmin()

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
    
    ##### plot fitting result #################################################
    plt.figure(dpi=200)
    plt.title('Lorentz Fit')
    if std[0] == std[1] and std[0] == 0:
        plt.plot(x, np.array(data), 'k', label='data median')
    else:
        plt.plot(x, np.array(data), 'k', label='data median')
        #plt.errorbar(rs, np.array(data), yerr = std, linestyle='-', color = 'black', alpha=0.4, capsize = 1.2, linewidth=1.1)
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
    m = np.amax(data[i_start:i_end])
    plt.ylim(-m*0.5, m*1.8)
    #plt.yticks([])
    plt.show()
    
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
    ratio = amp_AS/amp_S
    #print('ratio = ', ratio)                                                     #Amplitudes from Lorentz fit                              
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


# %% load data 

x_axis = [0, 40, 50, 60, 70, 100]# current I in mA
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
path = os.path.join(__location__, 'raw_data/Si_onTiO2_0mA_30mW_30sec_4acc_545nm_02.txt')
path_bg = os.path.join(__location__, 'raw_data/Si_onTiO2_0mA_30mW_30sec_3acc_545nm_bg_new.txt')

time = 30 #sec
power = 30 #mW

data, std, rs = median_acc(path, 4)
data_corr = bg_subtraction(path_bg, 3, 0, 1, data)
data_norm, std = norm(data_corr, std, time, power)

S_0 = data_norm
rs_S = np.array(rs)

parameters = Lor_fitting(data_norm, std, rs_S, [480,550], [450,550], 520, 'S')

width_S.append(parameters[0])
width_S_std.append(parameters[1])
pos_S.append(parameters[2])
pos_S_std.append(parameters[3])
amp_S.append(parameters[4])
amp_S_std.append(parameters[5])

#0 Volts ## Anti-Stokes ############################################################
print('######### 0 Volts - Anti-Stokes')
path = os.path.join(__location__, 'raw_data/Si_onTiO2_0mA_30mW_30sec_4acc_517nm_02.txt')
path_bg = os.path.join(__location__, 'raw_data/Si_onTiO2_0mA_30mW_30sec_3acc_517nm_bg_new.txt')

time = 30 #sec
power = 30 #mW

data, std, rs = median_acc(path, 4)
data_corr = bg_subtraction(path_bg, 3, 0, 1, data)
data_norm, std = norm(data_corr, std, time, power)

AS_0 = data_norm
rs_AS = np.array(rs)

parameters = Lor_fitting(data_norm, std, rs_AS, [480,550], [450,550], 520, 'AS')

width_AS.append(parameters[0])
width_AS_std.append(parameters[1])
pos_AS.append(parameters[2])
pos_AS_std.append(parameters[3])
amp_AS.append(parameters[4])
amp_AS_std.append(parameters[5])

#40 mA ## Stokes ############################################################
print('######### 40 mA - Stokes')
path = os.path.join(__location__, 'raw_data/Si_onTiO2_40mA_30mW_30sec_5acc_545nm.txt')
path_bg = os.path.join(__location__, 'raw_data/Si_onTiO2_40mA_30mW_30sec_5acc_545nm_bg.txt')

time = 30 #sec
power = 30 #mW

data, std, rs = median_acc(path, 5)
data_corr = bg_subtraction(path_bg, 5, 500, 1, data)
data_norm, std = norm(data_corr, std, time, power)

S_40 = data_norm
rs_S = np.array(rs)

parameters = Lor_fitting(data_norm, std, rs_S, [470,530], [450,550], 500, 'S')

width_S.append(parameters[0])
width_S_std.append(parameters[1])
pos_S.append(parameters[2])
pos_S_std.append(parameters[3])
amp_S.append(parameters[4])
amp_S_std.append(parameters[5])

#40 mA ## Anti-Stokes ############################################################
print('######### 40 mA - Anti-Stokes')
path = os.path.join(__location__, 'raw_data/Si_onTiO2_40mA_30mW_30sec_5acc_517nm.txt')
path_bg = os.path.join(__location__, 'raw_data/Si_onTiO2_40mA_30mW_30sec_5acc_517nm_bg.txt')

time = 30 #sec
power = 30 #mW

data, std, rs = median_acc(path, 5)
data_corr = bg_subtraction(path_bg, 5, 0, 1.008, data)
data_norm, std = norm(data_corr, std, time, power)

AS_40 = data_norm
rs_AS = np.array(rs)

parameters = Lor_fitting(data_norm, std, rs_AS, [470,530], [450,550], 500, 'AS')

width_AS.append(parameters[0])
width_AS_std.append(parameters[1])
pos_AS.append(parameters[2])
pos_AS_std.append(parameters[3])
amp_AS.append(parameters[4])
amp_AS_std.append(parameters[5])

#50 mA ## Stokes ############################################################
print('######### 50 mA - Stokes')
path = os.path.join(__location__, 'raw_data/Si_onTiO2_50mA_30mW_30sec_5acc_545nm.txt')
path_bg = os.path.join(__location__, 'raw_data/Si_onTiO2_50mA_30mW_30sec_5acc_545nm_bg.txt')

time = 30 #sec
power = 30 #mW

data, std, rs = median_acc(path, 5)
data_corr = bg_subtraction(path_bg, 5, 3100, 1.005, data)
data_norm, std = norm(data_corr, std, time, power)

S_50 = data_norm
rs_S = np.array(rs)

parameters = Lor_fitting(data_norm, std, rs_S, [470,530], [450,550], 500, 'S')

width_S.append(parameters[0])
width_S_std.append(parameters[1])
pos_S.append(parameters[2])
pos_S_std.append(parameters[3])
amp_S.append(parameters[4])
amp_S_std.append(parameters[5])

#50 mA ## Anti-Stokes ############################################################
print('######### 50 mA - Anti-Stokes')
path = os.path.join(__location__, 'raw_data/Si_onTiO2_50mA_30mW_30sec_5acc_517nm.txt')
path_bg = os.path.join(__location__, 'raw_data/Si_onTiO2_50mA_30mW_30sec_3acc_517nm_bg.txt')

time = 30 #sec
power = 30 #mW

data, std, rs = median_acc(path, 5)
data_corr = bg_subtraction(path_bg, 3, 0, 1.01, data)
data_norm, std = norm(data_corr, std, time, power)

AS_50 = data_norm
rs_AS = np.array(rs)

parameters = Lor_fitting(data_norm, std, rs_AS, [470,530], [450,550], 500, 'AS')

width_AS.append(parameters[0])
width_AS_std.append(parameters[1])
pos_AS.append(parameters[2])
pos_AS_std.append(parameters[3])
amp_AS.append(parameters[4])
amp_AS_std.append(parameters[5])

#60 mA ## Stokes ############################################################
print('######### 60 mA - Stokes')
path = os.path.join(__location__, 'raw_data/Si_onTiO2_60mA_30mW_30sec_5acc_545nm.txt')
path_bg = os.path.join(__location__, 'raw_data/Si_onTiO2_60mA_30mW_30sec_5acc_545nm_bg.txt')

time = 30 #sec
power = 30 #mW

data, std, rs = median_acc(path, 5)
data_corr = bg_subtraction(path_bg, 5, 4000, 1.03, data)
data_norm, std = norm(data_corr, std, time, power)

S_60 = data_norm
rs_S = np.array(rs)

parameters = Lor_fitting(data_norm, std, rs_S, [470,530], [450,550], 500, 'S')

width_S.append(parameters[0])
width_S_std.append(parameters[1])
pos_S.append(parameters[2])
pos_S_std.append(parameters[3])
amp_S.append(parameters[4])
amp_S_std.append(parameters[5])

#60 mA ## Anti-Stokes ############################################################
print('######### 60 mA - Anti-Stokes')
path = os.path.join(__location__, 'raw_data/Si_onTiO2_60mA_30mW_30sec_5acc_517nm.txt')
path_bg = os.path.join(__location__, 'raw_data/Si_onTiO2_60mA_30mW_30sec_5acc_517nm_bg.txt')

time = 30 #sec
power = 30 #mW

data, std, rs = median_acc(path, 5)
data_corr = bg_subtraction(path_bg, 5, 7000, 0.8, data)
data_norm, std = norm(data_corr, std, time, power)

AS_60 = data_norm
rs_AS = np.array(rs)

parameters = Lor_fitting(data_norm, std, rs_AS, [470,530], [450,550], 500, 'AS')

width_AS.append(parameters[0])
width_AS_std.append(parameters[1])
pos_AS.append(parameters[2])
pos_AS_std.append(parameters[3])
amp_AS.append(parameters[4])
amp_AS_std.append(parameters[5])

#70 mA ## Stokes ############################################################
print('######### 70 mA - Stokes')
path = os.path.join(__location__, 'raw_data/Si_onTiO2_70mA_30mW_30sec_5acc_545nm.txt')
path_bg = os.path.join(__location__, 'raw_data/Si_onTiO2_70mA_30mW_30sec_5acc_545nm_bg.txt')

time = 30 #sec
power = 30 #mW

data, std, rs = median_acc(path, 5)
data_corr = bg_subtraction(path_bg, 5, 0, 1, data)
data_norm, std = norm(data_corr, std, time, power)

S_70 = data_norm
rs_S = np.array(rs)

parameters = Lor_fitting(data_norm, std, rs_S, [470,530], [450,550], 500, 'S')

width_S.append(parameters[0])
width_S_std.append(parameters[1])
pos_S.append(parameters[2])
pos_S_std.append(parameters[3])
amp_S.append(parameters[4])
amp_S_std.append(parameters[5])

#70 mA ## Anti-Stokes ############################################################
print('######### 70 mA - Anti-Stokes')
path = os.path.join(__location__, 'raw_data/Si_onTiO2_70mA_30mW_30sec_5acc_517nm.txt')
path_bg = os.path.join(__location__, 'raw_data/Si_onTiO2_70mA_30mW_30sec_5acc_517nm_bg.txt')

time = 30 #sec
power = 30 #mW

data, std, rs = median_acc(path, 5)
data_corr = bg_subtraction(path_bg, 5, 0, 1.035, data)
data_norm, std = norm(data_corr, std, time, power)

AS_70 = data_norm
rs_AS = np.array(rs)

parameters = Lor_fitting(data_norm, std, rs_AS, [470,530], [450,550], 500, 'AS')

width_AS.append(parameters[0])
width_AS_std.append(parameters[1])
pos_AS.append(parameters[2])
pos_AS_std.append(parameters[3])
amp_AS.append(parameters[4])
amp_AS_std.append(parameters[5])

''' 80 mA data has large error bars ....
#80 mA ## Stokes ############################################################
print('######### 80 mA - Stokes')
path = os.path.join(__location__, 'raw_data/Si_onTiO2_80mA_30mW_30sec_5acc_545nm.txt')
path_bg = os.path.join(__location__, 'raw_data/Si_onTiO2_80mA_30mW_30sec_5acc_545nm_bg.txt')

time = 30 #sec
power = 30 #mW

data, std, rs = median_acc(path, 5)
data_corr = bg_subtraction(path_bg, 5, -50000, 2.1, data)
data_norm, std = norm(data_corr, std, time, power)

S_80 = data_norm
rs_S = np.array(rs)

parameters = Lor_fitting(data_norm, std, rs_S, [480,510], [450,550], 500, 'S')

width_S.append(parameters[0])
width_S_std.append(parameters[1])
pos_S.append(parameters[2])
pos_S_std.append(parameters[3])
amp_S.append(parameters[4])
amp_S_std.append(parameters[5])

#80 mA ## Anti-Stokes ############################################################
print('######### 80 mA - Anti-Stokes')
path = os.path.join(__location__, 'raw_data/Si_onTiO2_80mA_30mW_30sec_5acc_517nm.txt')
path_bg = os.path.join(__location__, 'raw_data/Si_onTiO2_80mA_30mW_30sec_5acc_517nm_bg.txt')

bg_region = [-540, -510]
time = 30 #sec
power = 30 #mW
spike_cutoff = 150
spike_ratio = 100

data, std, rs = median_acc(path, 5)
#data_corr = bg_corr(path_bg, bg_region, data, rs, 5)
data_corr = bg_subtraction(path_bg, 5, -290000, 7, data)
#data_wospikes = remove_spikes(spike_cutoff, spike_ratio, data_corr)
data_norm, std = norm(data_corr, std, time, power)
AS_80 = data_norm
rs_AS = np.array(rs)
parameters = Lor_fitting(data_norm, std, rs_AS, [480,510], [450,550], 500, 'AS')

width_AS.append(parameters[0])
width_AS_std.append(parameters[1])
pos_AS.append(parameters[2])
pos_AS_std.append(parameters[3])
amp_AS.append(parameters[4])
amp_AS_std.append(parameters[5])
'''
#100 mA ## Stokes ############################################################
print('######### 100 mA - Stokes')
path = os.path.join(__location__, 'raw_data/Si_onTiO2_100mA_30mW_30sec_5acc_545nm.txt')
path_bg = os.path.join(__location__, 'raw_data/Si_onTiO2_100mA_30mW_30sec_5acc_545nm_bg.txt')

time = 30 #sec
power = 30 #mW

data, std, rs = median_acc(path, 5)
data_corr = bg_subtraction(path_bg, 5, 0, 0.83, data)
data_norm, std = norm(data_corr, std, time, power)

S_100 = data_norm
rs_S = np.array(rs)

parameters = Lor_fitting(data_norm, std, rs_S, [480,530], [450,550], 500, 'S')

width_S.append(parameters[0])
width_S_std.append(parameters[1])
pos_S.append(parameters[2])
pos_S_std.append(parameters[3])
amp_S.append(parameters[4])
amp_S_std.append(parameters[5])

#100 mA ## Anti-Stokes ############################################################
print('######### 100 mA - Anti-Stokes')
path = os.path.join(__location__, 'raw_data/Si_onTiO2_100mA_30mW_30sec_5acc_517nm.txt')
path_bg = os.path.join(__location__, 'raw_data/Si_onTiO2_100mA_30mW_30sec_5acc_517nm_bg.txt')

time = 30 #sec
power = 30 #mW

data, std, rs = median_acc(path, 5)
data_corr = bg_subtraction(path_bg, 5, 6000, 1, data)
data_norm, std = norm(data_corr, std, time, power)

AS_100 = data_norm
rs_AS = np.array(rs)

parameters = Lor_fitting(data_norm, std, rs_AS, [480,530], [450,550], 500, 'AS')

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
        Temp(pos_S[5], 531.79, amp_S[5], amp_AS[5], amp_S_std[5], amp_AS_std[5])[0],
        #Temp(pos_S[6], 531.79, amp_S[6], amp_AS[6], amp_S_std[6], amp_AS_std[6])[0]
        ]

err = [Temp(pos_S[0], 531.79, amp_S[0], amp_AS[0], amp_S_std[0], amp_AS_std[0])[1],
        Temp(pos_S[1], 531.79, amp_S[1], amp_AS[1], amp_S_std[1], amp_AS_std[1])[1],
        Temp(pos_S[2], 531.79, amp_S[2], amp_AS[2],amp_S_std[2], amp_AS_std[2])[1],
        Temp(pos_S[3], 531.79, amp_S[3], amp_AS[3], amp_S_std[3], amp_AS_std[3])[1],
        Temp(pos_S[4], 531.79, amp_S[4], amp_AS[4], amp_S_std[4], amp_AS_std[4])[1],
        Temp(pos_S[5], 531.79, amp_S[5], amp_AS[5], amp_S_std[5], amp_AS_std[5])[1],
        #Temp(pos_S[6], 531.79, amp_S[6], amp_AS[6], amp_S_std[6], amp_AS_std[6])[1]
        ]

# Data from Si study
Temp_20240730=[305.1231365611692, 602.9829493821721, 717.1199570105318, 790.689012426263, 854.3993017741943, 898.453820332073, 957.8641389598992]
Pos_20240730 = [520.855445742211, 513.7168955196854, 510.58867629901306, 508.65623703648106, 507.2020548415821, 505.98198619405775, 504.81142659000255]
Width_20240730 = [3.759945715009675, 6.435249389511506, 7.85113249516355, 8.692421907666668, 9.424056888822651, 10.022830798471317, 10.59179456346091]
# %% Figures

######### Plot Stokes data ####################################################
plt.figure(dpi=200)
plt.plot(np.array(rs_S), S_0, c = 'blue', label = '0 A')
plt.plot(np.array(rs_AS), AS_0, c = 'blue')
plt.plot(np.array(rs_S), S_40, c = 'green', label = '40 mA')
plt.plot(np.array(rs_AS), AS_40, c = 'green')
plt.plot(np.array(rs_S), S_50, c = 'k', label = '50 mA')
plt.plot(np.array(rs_AS), AS_50, c = 'k')
plt.plot(np.array(rs_S), S_60, c = 'r', label = '60 mA')
plt.plot(np.array(rs_AS), AS_60, c = 'r')
plt.plot(np.array(rs_S), S_70, c = 'y', label = '70 mA')
plt.plot(np.array(rs_AS), AS_70, c = 'y')
#plt.plot(np.array(rs_S), S_80, c = 'violet', label = '80 mA')
#plt.plot(np.array(rs_AS), AS_80, c = 'violet')
plt.plot(np.array(rs_S), np.array(S_100)-10, c = 'violet', label = '100 mA')
plt.plot(np.array(rs_AS), AS_100, c = 'violet')
plt.ylim(-5, 70)
plt.plot([320, 830], [0,0], 'k--')
plt.xlim(470, 560)
plt.legend(frameon = False)
plt.xlabel('Raman shift (cm$^{-1}$)')
plt.ylabel('Intensity (Counts / mW*sec)')
plt.yticks([])
plt.tick_params(left = True, right = True, bottom = True, top = True)
plt.tick_params(axis = 'both', direction = 'in')
plt.title('Stokes Raman data')
plt.show()

########## Plot anti-Stokes data ##############################################
plt.figure(dpi=200)
plt.plot(np.array(rs_S), S_0, c = 'blue', label = '0 A')
plt.plot(np.array(rs_AS), AS_0, c = 'blue')
plt.plot(np.array(rs_S), S_40, c = 'green', label = '40 mA')
plt.plot(np.array(rs_AS), AS_40, c = 'green')
plt.plot(np.array(rs_S), S_50, c = 'k', label = '50 mA')
plt.plot(np.array(rs_AS), AS_50, c = 'k')
plt.plot(np.array(rs_S), S_60, c = 'r', label = '60 mA')
plt.plot(np.array(rs_AS), AS_60, c = 'r')
plt.plot(np.array(rs_S), S_70, c = 'y', label = '70 mA')
plt.plot(np.array(rs_AS), AS_70, c = 'y')
#plt.plot(np.array(rs_S), S_80, c = 'violet', label = '80 mA')
#plt.plot(np.array(rs_AS), AS_80, c = 'violet')
plt.plot(np.array(rs_S), S_100, c = 'violet', label = '100 mA')
plt.plot(np.array(rs_AS), AS_100, c = 'violet')
plt.ylim(-5, 70)
plt.plot([-830, -320], [0,0], 'k--')
plt.xlim(-560, -470)
plt.legend(frameon = False)
plt.xlabel('Raman shift (cm$^{-1}$)')
plt.ylabel('Intensity (Counts / mW*sec)')
plt.yticks([])
plt.tick_params(left = True, right = True, bottom = True, top = True)
plt.tick_params(axis = 'both', direction = 'in')
plt.title('Anti-Stokes Raman data')
plt.show()

######### Phonons Widths ######################################################
plt.figure(figsize=(3.5, 4), dpi=200)
plt.errorbar(x_axis, width_S, yerr = np.array(width_S_std)*10, marker = 'o', ls='none', color = 'red', alpha=0.7, capsize = 1.2, linewidth=1.1, label = 'Stokes')
plt.errorbar(x_axis, width_AS, yerr = np.array(width_AS_std)*10, marker = 'o', ls='none', color = 'blue', alpha=0.7, capsize = 1.2, linewidth=1.1, label = 'Anti-Stokes')
plt.ylim(0, 25)
#plt.xlim(470, 560)
plt.legend(frameon = False)
plt.xlabel('I (mA)')
plt.ylabel('$\Gamma$ (cm$^{-1}$)')
#plt.yticks([])
plt.grid(linestyle=':')
plt.tick_params(left = True, right = True, bottom = True, top = True)
plt.tick_params(axis = 'both', direction = 'in')
plt.title('$\Gamma$ - Si on TiO2')
plt.show()

######### Phonon Frequencies ##################################################
plt.figure(figsize=(3.5, 4), dpi=200)
plt.errorbar(x_axis, pos_S, yerr = np.array(pos_S_std)*8, marker = 'o', color = 'red', ls='none', alpha=0.7, capsize = 1.2, linewidth=1.1, label = 'Stokes')
plt.errorbar(x_axis, pos_AS, yerr = np.array(pos_AS_std)*8, marker = 'o', color = 'blue', ls='none', alpha=0.7, capsize = 1.2, linewidth=1.1, label = 'Anti-Stokes')
plt.ylim(490, 529)
#plt.xlim(470, 560)
plt.legend(frameon = False)
plt.xlabel('I (mA)')
plt.ylabel('$\omega_0$ (cm$^{-1}$)')
#plt.yticks([])
plt.grid(linestyle=':')
plt.tick_params(left = True, right = True, bottom = True, top = True)
plt.tick_params(axis = 'both', direction = 'in')
plt.title('$\omega_0$ - Si on TiO2')
plt.show()

######### Phonon Pos as a function of Temperature #############################
plt.figure(figsize=(3.5, 4), dpi=200)
plt.errorbar(Temperatures, pos_S, xerr = np.array(err)*3, yerr = np.array(pos_S_std)*8, marker = 'o', color = 'red', ls='none', alpha=0.7, capsize = 1.2, linewidth=1.1, label = 'Stokes')
plt.errorbar(Temperatures, pos_AS, xerr = np.array(err)*3, yerr = np.array(pos_AS_std)*8, marker = 'o', color = 'blue', ls='none', alpha=0.7, capsize = 1.2, linewidth=1.1, label = 'Anti-Stokes')
plt.plot(shifts['x'], shifts['y'], '+k', label = 'PRB')# ref from PRB
plt.plot(shifts_fit['x'], shifts_fit['y'], 'k--')# fit from PRB
plt.plot(Temp_20240730, Pos_20240730, 'o--', color = 'lightgreen', label = 'Si earlier results')
plt.ylim(496, 545)
plt.xlim(100, 1700)
plt.legend(frameon = False)
plt.xlabel('T (K)')
plt.ylabel('$\omega_0$ (cm$^{-1}$)')
#plt.yticks([])
plt.grid(linestyle=':')
plt.tick_params(left = True, right = True, bottom = True, top = True)
plt.tick_params(axis = 'both', direction = 'in')
plt.title('$\omega_0$ as a Function of Temp')
plt.show()

######### Phonon Width as a function of Temperature #############################
plt.figure(figsize=(3.5, 4), dpi=200)
plt.errorbar(Temperatures, width_S, xerr = np.array(err)*3, yerr = np.array(width_S_std)*10, marker = 'o', ls='none', color = 'red', alpha=0.7, capsize = 1.2, linewidth=1.1, label = 'Stokes')
plt.errorbar(Temperatures, width_AS, xerr = np.array(err)*3, yerr = np.array(width_AS_std)*10, marker = 'o', ls='none', color = 'blue', alpha=0.7, capsize = 1.2, linewidth=1.1, label = 'Anti-Stokes')
plt.errorbar([20, 460, 770], width_ref, yerr = width_ref_err, marker = 'o', color = 'k', ls='none', alpha=0.7, capsize = 1.2, label = 'PRB Paper')
plt.errorbar([460, 770], width_ref_AS, yerr = width_ref_AS_err, marker = 'o', color = 'gray', ls='none', alpha=0.7, capsize = 1.2)
plt.plot(Temp_20240730, Width_20240730, 'o--', color = 'lightgreen', label = 'Si earlier results')
plt.ylim(2, 20)
plt.xlim(0, 1700)
plt.legend(frameon = False)
plt.xlabel('T (K)')
plt.ylabel('$\Gamma$ (cm$^{-1}$)')
#plt.yticks([])
plt.grid(linestyle=':')
plt.tick_params(left = True, right = True, bottom = True, top = True)
plt.tick_params(axis = 'both', direction = 'in')
plt.title('$\Gamma$ as a Function of Temp')
plt.show()

######### Calculated Temperatures #############################################
plt.figure(figsize=(3.5, 4), dpi=200)
#plt.plot(x_axis, Temperatures, 'ok--')
plt.errorbar(x_axis, Temperatures, yerr = np.array(err)*3, marker = 'o', color = 'k', ls = 'none', alpha=0.7, capsize = 1.2, linewidth=1.1, label = 'Stokes')
plt.ylim(250, 1800)
plt.xlim(-10, 110)
plt.xlabel('I (mA)')
plt.ylabel('Temperature (K)')
#plt.yticks([])
plt.grid(linestyle=':')
plt.tick_params(left = True, right = True, bottom = True, top = True)
plt.tick_params(axis = 'both', direction = 'in')
plt.title('Temperatures')
plt.show()

print('############### Temperatures ################')
print('Temp: ', Temperatures)
print('errors: ', np.array(err)*3)

# %%
'''
save_data(rs_S, S_0, 'S_0mA')
save_data(rs_S, S_40, 'S_40mA')
save_data(rs_S, S_50, 'S_50mA')
save_data(rs_S, S_60, 'S_60mA')
save_data(rs_S, S_70, 'S_70mA')
save_data(rs_S, S_100, 'S_100mA')

save_data(rs_AS, AS_0, 'AS_0mA')
save_data(rs_AS, AS_40, 'AS_40mA')
save_data(rs_AS, AS_50, 'AS_50mA')
save_data(rs_AS, AS_60, 'AS_60mA')
save_data(rs_AS, AS_70, 'AS_70mA')
save_data(rs_AS, AS_100, 'AS_100mA')

#export results
#I multiplied the errors (resulting from the fits) to get reasonable error bars given that the uncertainty is higher due to the high background
zip(x_axis, Temperatures, np.array(err)*3, width_S, np.array(width_S_std)*10, width_AS, np.array(width_AS_std)*10, pos_S, np.array(pos_S_std)*8, pos_AS, np.array(pos_AS_std)*8)
with open('Si_on_TiO2_analysis_results.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['Current (mA)', 'calculated Temp (K)', 'Temp err', 'Stokes Phonon width', 'Stokes width err', 'AS Phonon width', 'AS width err', 'Stokes phonon pos (w0)', 'Stokes w0 err', 'AS phonon pos (w0)', 'AS w0 err'])
    writer.writerows(zip(x_axis, Temperatures, err, width_S, width_S_std, width_AS, width_AS_std, pos_S, pos_S_std, pos_AS, pos_AS_std))
'''
