
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
import sys
import os

kB = 8.617333e-5 # eV/K
invcm_2_eV = 0.001/8.064516

lambda_0 = 532 # nm
nu_0 = 1/(lambda_0*1e-9)/100 # 1/cm
E_laser = nu_0 * invcm_2_eV

# --------------------------------------------------------------------------------------------------

def plot_vs_currents():

    fig, ax = plt.subplots(1,figsize=(4,4),
                           gridspec_kw={'hspace':0.1,'wspace':0.075})
    
    _T, _dT, _w, _dw, _g, _dg, _I = np.loadtxt('si_on_tio2_data.txt',unpack=True)

    _I /= 1.9345
    ax.errorbar(_I,_T,yerr=_dT,ms=6,lw=0,elinewidth=1, #ls=(0,(4,2,2,2)),
                  c='m',marker='^',label='Si on TiO2',markerfacecolor=None,
                  markeredgewidth=1.5,zorder=1000)
    ax.legend(frameon=False,fontsize='large',loc='lower right',
                bbox_to_anchor=(1.0,0.0),handletextpad=0.1,ncols=1,labelspacing=0.25)
                #labelspacing=0.1,handlelength=0.5,handletextpad=0.7)
    _T_fit = _T[1:]
    _I_fit = _I[1:]
    _inds = np.argsort(_I_fit)
    _T_fit = _T_fit[_inds]
    _I_fit = _I_fit[_inds]
    coeff = np.polynomial.polynomial.polyfit(_I_fit,_T_fit,deg=1)
    ax.plot(_I_fit,coeff[0]+_I_fit*coeff[1],lw=1,ls=(0,(4,2,2,2)),c='k')

    axes = [ax]


    for _ax in axes:
        for axis in ['top','bottom','left','right']:
            _ax.spines[axis].set_linewidth(1.5)
        _ax.minorticks_on()
        _ax.tick_params(which='both',width=1,labelsize='large')
        _ax.tick_params(which='major',length=5)
        _ax.tick_params(which='minor',length=2)
        _ax.set_rasterized = True

    xlim = [-50,100]
    ax.set_xlim(xlim)

    ylim = [200,1200]
    ax.set_ylim(ylim)

    ax.set_ylabel('Temperature [K]',fontsize='large',labelpad=5)
    ax.set_xlabel(r'Current density [mA/mm$^2$]',fontsize='large')

    fig_name = f'temps_vs_currents.png'
    plt.savefig(fig_name,dpi=300,bbox_inches='tight')

    print('a')
    plt.show()


if __name__ == '__main__':

    plot_vs_currents()