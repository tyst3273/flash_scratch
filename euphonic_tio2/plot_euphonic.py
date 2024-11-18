
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os


dE = 2

filename = 'new_T_low_Ei_300K_ZB.hdf5'

with h5py.File(filename,'r') as db:

    Qpts = db['Qpts'][...]
    nQ = Qpts.shape[0]

    sqe = db['cmap_structure_factors'][...]
    nE = sqe.shape[1]

    E = db['cmap_energies'][...]

inds = np.intersect1d(np.flatnonzero(E < 30),np.flatnonzero(E > 5))

pdf_filename = filename.replace('hdf5','pdf')
print(pdf_filename)
with PdfPages(pdf_filename) as pdf:

    for ii in range(nQ):

        sqe_max = sqe[ii,inds].max()
        Q = Qpts[ii,:] 

        if Q[0] == Q[1] or Q[2] != 0:
            continue

        #if sqe_max < 1:
        #    continue

        #above_2 = np.flatnonzero(sqe[ii,inds] > 2)
        #E_above_2 = E[above_2]
        #if E_above_2.max() - E_above_2.min() < dE:
        #    continue

        print(ii)

        fig, ax = plt.subplots(figsize=(8,6))
        
        #bose = get_bose(flash[:,0])
        #ax.errorbar(flash[:,0],flash[:,1]/bose,yerr=flash[:,2]/bose,
        #        marker='o',ms=3,barsabove=True,c='b',ls='-',lw=1,label='flashing')

        ax.plot(E,sqe[ii,:],marker='o',ms=3,c='m',ls='-',lw=1,label='DFT')

        ax.set_xlabel('E [meV]',fontsize='large')
        ax.set_ylabel('intensity (arb. units)',fontsize='large')

        fig.suptitle(f'Q=({Q[0]: 5.3f},{Q[1]: 5.3f},{Q[2]: 5.3f})',fontsize='large',y=0.95)

        ax.set_ylim(-0.01,sqe_max*1.1)

        ax.set_xticks(np.arange(-30,35,5))

        #ax.set_yscale('log')
        #ax.set_ylim(1e-3,1)

        pdf.savefig(fig,dpi=50,bbox_inches='tight')

        plt.close()
        plt.clf()





