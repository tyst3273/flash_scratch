
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os



with h5py.File('all_Q_T_300K.hdf5','r') as db:

    Qpts = db['Qpts'][...]
    nQ = Qpts.shape[0]

    sqe = db['cmap_structure_factors'][...]
    nE = sqe.shape[1]

    E = db['cmap_energies'][...]


with PdfPages('all_Q.pdf') as pdf:

    for ii in range(nQ):

        print(ii)

        Q = Qpts[ii,:] 

        fig, ax = plt.subplots(figsize=(8,6))
        
        #bose = get_bose(flash[:,0])
        #ax.errorbar(flash[:,0],flash[:,1]/bose,yerr=flash[:,2]/bose,
        #        marker='o',ms=3,barsabove=True,c='b',ls='-',lw=1,label='flashing')

        ax.plot(E,sqe[ii,:],marker='o',ms=3,c='m',ls='-',lw=1,label='DFT')


        ax.set_xlabel('E [meV]',fontsize='large')
        ax.set_ylabel('intensity (arb. units)',fontsize='large')

        fig.suptitle(f'Q=({Q[0]: 5.3f},{Q[1]: 5.3f},{Q[2]: 5.3f})',fontsize='large',y=0.95)

        ax.set_ylim(-0.01,1.0)

        #ax.set_yscale('log')
        #ax.set_ylim(1e-3,1)

        pdf.savefig(fig,dpi=50,bbox_inches='tight')

        plt.close()
        plt.clf()





