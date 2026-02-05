import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FormatStrFormatter
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
import sys
import os
import h5py


blue = '#377eb8'
orange = '#ff7f00'
green = '#4daf4a'



# mpl.rc('text', usetex=True)
# mpl.rcParams['text.latex.preamble'] = r"\usepackage{bm}"

kB = 8.617333e-5 # eV/K
invcm_2_eV = 0.001/8.064516

lambda_0 = 532 # nm
nu_0 = 1/(lambda_0*1e-9)/100 # 1/cm
E_laser = nu_0 * invcm_2_eV


fig, ax = plt.subplots(1,2,figsize=(4.5,4.5),
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
s_ax.plot(e,i+shift,marker='o',ms=3,lw=0,c=blue,mfc='none',mew=0.5)
s_ax.plot([-1000,1000],[shift,shift],ms=0,lw=1,ls=(0,(1,1)),c=(0.25,0.25,0.25))

data = np.loadtxt('si_S_off_fits.txt')
e = data[0,:]
i = data[1,:]
s_ax.plot(e,i+shift,ms=0,lw=1,c='k',ls='-')#(0,(2,1,1,1)))

filename = 'AS_0mA_fan_off.txt'
e, i = np.loadtxt(os.path.join(directory,filename),unpack=True)
e = e[::2]
i = i[::2]
as_ax.plot(e,i+shift,marker='o',ms=3,lw=0,c=blue,mfc='none',mew=0.5)
as_ax.plot([-1000,1000],[shift,shift],ms=0,lw=1,ls=(0,(1,1)),c=(0.25,0.25,0.25))

data = np.loadtxt('si_AS_off_fits.txt')
e = data[0,:]
i = data[1,:]
as_ax.plot(-e,i+shift,ms=0,lw=1,c='k',ls='-')#(0,(2,1,1,1)))

# # high-T, no fan
shift = 15

# filename = 'S_600mA_fan_off.txt'
# e, i = np.loadtxt(os.path.join(directory,filename),unpack=True)
# e = e[::2]
# i = i[::2]
# s_ax.plot(e,i+shift,marker='o',ms=3,lw=0,c=orange,mfc='none',mew=0.5)
# s_ax.plot([-1000,1000],[shift,shift],ms=0,lw=1,ls=(0,(1,1)),c=(0.25,0.25,0.25))

# data = np.loadtxt('si_S_off_fits.txt')
# e = data[0,:]
# i = data[-2,:]
# s_ax.plot(e,i+shift,ms=0,lw=1,c='k',ls='-')#(0,(2,1,1,1)))

# filename = 'AS_600mA_fan_off.txt'
# e, i = np.loadtxt(os.path.join(directory,filename),unpack=True)
# e = e[::2]
# i = i[::2]
# as_ax.plot(e,i+shift,marker='o',ms=3,lw=0,c=orange,mfc='none',mew=0.5)
# as_ax.plot([-1000,1000],[shift,shift],ms=0,lw=1,ls=(0,(1,1)),c=(0.25,0.25,0.25))

# data = np.loadtxt('si_AS_off_fits.txt')
# e = data[0,:]
# i = data[-2,:]
# as_ax.plot(-e,i+shift,ms=0,lw=1,c='k',ls='-')#(0,(2,1,1,1)))

with h5py.File('si_800mA.hdf5','r') as db:
    raman_shift = db['raman_shift'][...]
    intensity = db['raman_intensity'][...]
    temp = db['temperature'][...]
    as_shift = db['antistokes_shift'][...]
    s_shift = db['stokes_shift'][...]
    as_int = db['antistokes_intensity'][...]
    s_int = db['stokes_intensity'][...]

print(temp)

area = 0.083 * 0.523  # cm    

scale = 15
intensity /= scale
as_int /= scale
s_int /= scale

inds = (raman_shift < 0)
as_ax.plot(raman_shift[inds],intensity[inds]+shift,marker='o',ms=3,lw=0,c=orange,mfc='none',mew=0.5)
as_ax.plot(as_shift,as_int+shift,ms=0,lw=1,c='k',ls='-')#(0,(2,1,1,1)))

inds = (raman_shift > 0)
s_ax.plot(raman_shift[inds],intensity[inds]+shift,marker='o',ms=3,lw=0,c=orange,mfc='none',mew=0.5)
s_ax.plot(s_shift,s_int+shift,ms=0,lw=1,c='k',ls='-')#(0,(2,1,1,1)))

# --------------------------------------------------------

# plot TiO2 data
directory = '20240809/corrected Raman data export'

# high-T
shift = 25

filename = 'S_40mA.txt'
e, i = np.loadtxt(os.path.join(directory,filename),unpack=True)
e = e[::2]
i = i[::2]
s_ax.plot(e,i+shift,marker='o',ms=3,lw=0,c=green,mfc='none',mew=0.5)
s_ax.plot([-1000,1000],[shift,shift],ms=0,lw=1,ls=(0,(1,1)),c=(0.25,0.25,0.25))

data = np.loadtxt('tio2_S_fits.txt')
e = data[0,:]
i = data[2,:]
s_ax.plot(e,i+shift,ms=0,lw=1,c='k',ls='-')#(0,(2,1,1,1)))

filename = 'AS_40mA.txt'
e, i = np.loadtxt(os.path.join(directory,filename),unpack=True)
e = e[::2]
i = i[::2]
as_ax.plot(e,i+shift,marker='o',ms=3,lw=0,c=green,mfc='none',mew=0.5)
as_ax.plot([-1000,1000],[shift,shift],ms=0,lw=1,ls=(0,(1,1)),c=(0.25,0.25,0.25))

data = np.loadtxt('tio2_AS_fits.txt')
e = data[0,:]
i = data[2,:]
as_ax.plot(-e,i+shift,ms=0,lw=1,c='k',ls='-')#(0,(2,1,1,1)))

# --------------------------------------------------------



# --------------------------------------------------------

axes = [s_ax,as_ax]

for _ax in axes:
    for axis in ['top','bottom','left','right']:
        _ax.spines[axis].set_linewidth(1.5)
    _ax.minorticks_on()
    _ax.tick_params(which='both',width=1,labelsize=12,direction='in')
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

# as_ax.annotate(rf'fan on',xy=(0.1,0.9),xycoords='axes fraction',fontsize=16)

as_ax.set_ylabel('Intensity [arb. units]',fontsize=16,labelpad=10)

fig.supxlabel(r'Raman shift [cm$^{-1}$]',fontsize=16,y=0.0)
# fig.supxlabel(r'Raman shift [cm$^{-1}$]',fontsize=16,y=-0.05)
# fig.suptitle('Flashing Si',fontsize=16,y=0.93)

# as_ax.annotate('(a)',xy=(0.05,0.925),xycoords='axes fraction',fontsize=16)  
# s_ax.annotate('(b)',xy=(0.05,0.925),xycoords='axes fraction',fontsize=16)  

#as_ax.annotate(r'TiO$_2$, 0 mA',xy=(0.5,0.24),xycoords='axes fraction',fontsize='medium') 
as_ax.annotate('Si on\n'+r'TiO$_2$',xy=(0.025,0.66),
                    xycoords='axes fraction',fontsize=12,color=green) 
as_ax.annotate(r'2.1$\frac{\rm{A}}{\rm{cm}^2}$',xy=(0.6,0.66),
                    xycoords='axes fraction',fontsize=12,color=green)


#as_ax.annotate(r'Si, 0 mA',xy=(0.55,0.025),xycoords='axes fraction',fontsize='medium') 
as_ax.annotate('ambient',xy=(0.55,0.14),xycoords='axes fraction',fontsize=12,c=blue)

# as_ax.annotate(r'54.5$\frac{\rm{A}}{\rm{cm}^2}$',xy=(0.575,0.4),
#                     xycoords='axes fraction',fontsize=12,color=orange) 
as_ax.annotate(f'{0.8/area:.1f}'+r'$\frac{\rm{A}}{\rm{cm}^2}$',xy=(0.575,0.44),
                    xycoords='axes fraction',fontsize=12,color=orange) 
as_ax.annotate(r'Si',xy=(0.05,0.44),
                    xycoords='axes fraction',fontsize=12,color=orange) 



#s_ax.annotate(r'300 K',xy=(0.05,0.025),xycoords='axes fraction',fontsize='medium') 
# s_ax.annotate(r'868 K',xy=(0.05,0.415),xycoords='axes fraction',fontsize=16,color=orange) 
# s_ax.annotate(r'300 K',xy=(0.05,0.14),xycoords='axes fraction',fontsize=16,color=blue) 
# s_ax.annotate(r'905 K',xy=(0.05,0.66),xycoords='axes fraction',fontsize=16,color=green) 

# s_ax.annotate(r'595 C',xy=(0.05,0.415),xycoords='axes fraction',fontsize=12,color=orange) 
s_ax.annotate(r'485 C',xy=(0.05,0.45),xycoords='axes fraction',fontsize=12,color=orange) 
s_ax.annotate(r'25 C',xy=(0.05,0.14),xycoords='axes fraction',fontsize=12,color=blue) 
s_ax.annotate(r'632 C',xy=(0.05,0.66),xycoords='axes fraction',fontsize=12,color=green) 


plt.savefig('raman_temps.png',dpi=300,bbox_inches='tight')
plt.show()
