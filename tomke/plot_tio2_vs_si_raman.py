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


fig, ax = plt.subplots(1,2,figsize=(4,3),
                       gridspec_kw={'height_ratios':[1],'width_ratios':[1,1],
                                    'hspace':0.1,'wspace':0.1},clear=True)
s_ax = ax[1]
as_ax = ax[0]

# --------------------------------------------------------

# plot Si data
directory = '20240730_Si_flash/20240730/corrected_Raman_data_export'

# room-T, no fan
shift = 0

filename = 'S_0mA_fan_off.txt'
e, i = np.loadtxt(os.path.join(directory,filename),unpack=True)
e = e[::2]
i = i[::2]
s_ax.plot(e,i+shift,marker='o',ms=3,lw=0,c='b',mfc='none',mew=0.5)
s_ax.plot([-1000,1000],[shift,shift],ms=0,lw=1,ls=(0,(1,1)),c=(0.25,0.25,0.25))

data = np.loadtxt('si_S_off_fits.txt')
e = data[0,:]
i = data[1,:]
s_ax.plot(e,i+shift,ms=0,lw=1,c='k',ls='-')#(0,(2,1,1,1)))

filename = 'AS_0mA_fan_off.txt'
e, i = np.loadtxt(os.path.join(directory,filename),unpack=True)
e = e[::2]
i = i[::2]
as_ax.plot(e,i+shift,marker='o',ms=3,lw=0,c='b',mfc='none',mew=0.5)
as_ax.plot([-1000,1000],[shift,shift],ms=0,lw=1,ls=(0,(1,1)),c=(0.25,0.25,0.25))

data = np.loadtxt('si_AS_off_fits.txt')
e = data[0,:]
i = data[1,:]
as_ax.plot(-e,i+shift,ms=0,lw=1,c='k',ls='-')#(0,(2,1,1,1)))

# high-T, no fan
shift = 20

filename = 'S_750mA_fan_off.txt'
e, i = np.loadtxt(os.path.join(directory,filename),unpack=True)
e = e[::2]
i = i[::2]
s_ax.plot(e,i+shift,marker='o',ms=3,lw=0,c='m',mfc='none',mew=0.5)
s_ax.plot([-1000,1000],[shift,shift],ms=0,lw=1,ls=(0,(1,1)),c=(0.25,0.25,0.25))

data = np.loadtxt('si_S_off_fits.txt')
e = data[0,:]
i = data[-1,:]
s_ax.plot(e,i+shift,ms=0,lw=1,c='k',ls='-')#(0,(2,1,1,1)))

filename = 'AS_750mA_fan_off.txt'
e, i = np.loadtxt(os.path.join(directory,filename),unpack=True)
e = e[::2]
i = i[::2]
as_ax.plot(e,i+shift,marker='o',ms=3,lw=0,c='m',mfc='none',mew=0.5)
as_ax.plot([-1000,1000],[shift,shift],ms=0,lw=1,ls=(0,(1,1)),c=(0.25,0.25,0.25))

data = np.loadtxt('si_AS_off_fits.txt')
e = data[0,:]
i = data[-1,:]
as_ax.plot(-e,i+shift,ms=0,lw=1,c='k',ls='-')#(0,(2,1,1,1)))

# --------------------------------------------------------

# plot TiO2 data
directory = '20240809/corrected Raman data export'

"""
# room-T 
shift = 5

filename = 'S_0mA.txt'
e, i = np.loadtxt(os.path.join(directory,filename),unpack=True)
s_ax.plot(e,i+shift,marker='o',ms=3,lw=1,c='m')
s_ax.plot([-1000,1000],[shift,shift],ms=0,lw=1,ls=(0,(1,1)),c=(0.25,0.25,0.25))

data = np.loadtxt('tio2_S_fits.txt')
e = data[0,:]
i = data[1,:]
s_ax.plot(e,i+shift,ms=0,lw=1,c='k',ls=(0,(2,1,1,1)))

filename = 'AS_0mA.txt'
e, i = np.loadtxt(os.path.join(directory,filename),unpack=True)
as_ax.plot(e,i+shift,marker='o',ms=3,lw=1,c='m')
as_ax.plot([-1000,1000],[shift,shift],ms=0,lw=1,ls=(0,(1,1)),c=(0.25,0.25,0.25))

data = np.loadtxt('tio2_AS_fits.txt')
e = data[0,:]
i = data[1,:]
as_ax.plot(-e,i+shift,ms=0,lw=1,c='k',ls=(0,(2,1,1,1)))
"""

# high-T
shift = 25

filename = 'S_50mA.txt'
e, i = np.loadtxt(os.path.join(directory,filename),unpack=True)
e = e[::2]
i = i[::2]
s_ax.plot(e,i+shift,marker='o',ms=3,lw=0,c='r',mfc='none',mew=0.5)
s_ax.plot([-1000,1000],[shift,shift],ms=0,lw=1,ls=(0,(1,1)),c=(0.25,0.25,0.25))

data = np.loadtxt('tio2_S_fits.txt')
e = data[0,:]
i = data[3,:]
s_ax.plot(e,i+shift,ms=0,lw=1,c='k',ls='-')#(0,(2,1,1,1)))

filename = 'AS_50mA.txt'
e, i = np.loadtxt(os.path.join(directory,filename),unpack=True)
e = e[::2]
i = i[::2]
as_ax.plot(e,i+shift,marker='o',ms=3,lw=0,c='r',mfc='none',mew=0.5)
as_ax.plot([-1000,1000],[shift,shift],ms=0,lw=1,ls=(0,(1,1)),c=(0.25,0.25,0.25))

data = np.loadtxt('tio2_AS_fits.txt')
e = data[0,:]
i = data[3,:]
as_ax.plot(-e,i+shift,ms=0,lw=1,c='k',ls='-')#(0,(2,1,1,1)))

# --------------------------------------------------------

axes = [s_ax,as_ax]

for _ax in axes:
    for axis in ['top','bottom','left','right']:
        _ax.spines[axis].set_linewidth(1.5)
    _ax.minorticks_on()
    _ax.tick_params(which='both',width=1,labelsize='large')
    _ax.tick_params(which='major',length=5)
    _ax.tick_params(which='minor',length=2)
    _ax.set_rasterized = True
    
as_ax.spines.right.set_visible(False)
as_ax.tick_params(axis='y',which='both',right=False,labelright=False)

s_ax.spines.left.set_visible(False)
s_ax.tick_params(axis='y',which='both',left=False,labelleft=False)

d = 0.1  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-d, -d), (d, d)], markersize=5,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)

as_ax.plot((1,1), (0,0), transform=as_ax.transAxes, **kwargs)
as_ax.plot((1,1), (1,1), transform=as_ax.transAxes, **kwargs)
s_ax.plot((0,0), (0,0), transform=s_ax.transAxes, **kwargs)
s_ax.plot((0,0), (1,1), transform=s_ax.transAxes, **kwargs)

as_ax.plot((1,1), (0,0), transform=as_ax.transAxes, **kwargs)
as_ax.plot((1,1), (1,1), transform=as_ax.transAxes, **kwargs)
s_ax.plot((0,0), (0,0), transform=s_ax.transAxes, **kwargs)
s_ax.plot((0,0), (1,1), transform=s_ax.transAxes, **kwargs)

as_ax.plot([1,1],[0,1], transform=as_ax.transAxes, 
                lw=1, ls=(0,(4,1,2,1)),ms=0, c='k')
s_ax.plot([0,0],[0,1], transform=s_ax.transAxes, 
                lw=1, ls=(0,(4,1,2,1)),ms=0, c='k')

xlim = [-550,-430]
as_ax.set_xlim(xlim)

xlim = [430,550]
s_ax.set_xlim(xlim)

ylim = [-5,45]
as_ax.set_ylim(ylim)
s_ax.set_ylim(ylim)

# as_ax.annotate(rf'fan on',xy=(0.1,0.9),xycoords='axes fraction',fontsize='large')

as_ax.set_ylabel('Intensity [arb. units]',fontsize='large',labelpad=10)

fig.supxlabel(r'Raman shift [cm$^{-1}$]',fontsize='large',y=-0.05)
# fig.suptitle('Flashing Si',fontsize='large',y=0.93)

as_ax.annotate('(a)',xy=(0.05,0.925),xycoords='axes fraction',fontsize='large')  
s_ax.annotate('(b)',xy=(0.05,0.925),xycoords='axes fraction',fontsize='large')  

#as_ax.annotate(r'TiO$_2$, 0 mA',xy=(0.5,0.24),xycoords='axes fraction',fontsize='medium') 
as_ax.annotate(r'TiO$_2$',xy=(0.025,0.66),
                    xycoords='axes fraction',fontsize='large',color='r') 
as_ax.annotate(r'26$\frac{\textrm{mA}}{\textrm{mm}^2}$',xy=(0.6,0.66),
                    xycoords='axes fraction',fontsize='large',color='r')


#as_ax.annotate(r'Si, 0 mA',xy=(0.55,0.025),xycoords='axes fraction',fontsize='medium') 
as_ax.annotate('ambient',xy=(0.6,0.14),xycoords='axes fraction',fontsize='large',c='b')

as_ax.annotate(r'680$\frac{\textrm{mA}}{\textrm{mm}^2}$',xy=(0.575,0.4),
                    xycoords='axes fraction',fontsize='large',color='m') 
as_ax.annotate(r'Si',xy=(0.05,0.4),
                    xycoords='axes fraction',fontsize='large',color='m') 



#s_ax.annotate(r'300 K',xy=(0.05,0.025),xycoords='axes fraction',fontsize='medium') 
s_ax.annotate(r'923 K',xy=(0.05,0.425),xycoords='axes fraction',fontsize='large',color='m') 


s_ax.annotate(r'300 K',xy=(0.05,0.14),xycoords='axes fraction',fontsize='large',color='b') 
s_ax.annotate(r'888 K',xy=(0.05,0.66),xycoords='axes fraction',fontsize='large',color='r') 


plt.savefig('raman_temps.png',dpi=300,bbox_inches='tight')

