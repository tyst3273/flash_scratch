import numpy as np
import matplotlib.pyplot as plt
import h5py


fig, ax = plt.subplots(1,2,figsize=(6,5),gridspec_kw={'wspace':0.1})

temps = [600]

shift = 6

for ii, T in enumerate(temps):

    f = f'T_{T:g}K.hdf5'
    with h5py.File(f,'r') as db:
        Qpts = db['Qpts'][...]
        sqe = db['cmap_structure_factors'][...]
        E = db['cmap_energies'][...]

    ax[0].plot(E,sqe[0,:]+ii*shift,marker='o',ms=2,c='r',ls='-',lw=1)
    ax[1].plot(E,sqe[1,:]+ii*shift,marker='o',ms=2,c='b',ls='-',lw=1)


axlist = [ax[0],ax[1]]
for _ax in axlist:
    for axis in ['top','bottom','left','right']:
        _ax.spines[axis].set_linewidth(1.5)

    _ax.minorticks_on()
    _ax.tick_params(which='both',width=1,labelsize='x-large')
    _ax.tick_params(which='major',length=5)
    _ax.tick_params(which='minor',length=2)

    _ax.axis([-45,45,-2,20])

ax[0].plot([-50,50],[0,0],lw=2,ls='--',c='k')
ax[1].plot([-50,50],[0,0],lw=2,ls='--',c='k')

ax[1].set_yticklabels([])
ax[0].set_ylabel('Intensity (arb. units)',fontsize='x-large')
ax[0].set_xlabel('Energy (meV)',fontsize='x-large')
ax[1].set_xlabel('Energy (meV)',fontsize='x-large')

Q = Qpts[0]
ax[0].set_title(f'Q=({Q[0]:3.2f},{Q[1]:3.2f},{Q[2]:3.2f})',fontsize='x-large',y=1)

Q = Qpts[1]
ax[1].set_title(f'Q=({Q[0]:3.2f},{Q[1]:3.2f},{Q[2]:3.2f})',fontsize='x-large',y=1)

ax[0].annotate('T=300 K',fontsize='x-large',xycoords='axes fraction',xy=(0.15,0.025))
ax[0].annotate('T=600 K',fontsize='x-large',xycoords='axes fraction',xy=(0.15,0.3))
ax[0].annotate('T=900 K',fontsize='x-large',xycoords='axes fraction',xy=(0.15,0.575))

ax[1].annotate('T=300 K',fontsize='x-large',xycoords='axes fraction',xy=(0.15,0.025))
ax[1].annotate('T=600 K',fontsize='x-large',xycoords='axes fraction',xy=(0.15,0.3))
ax[1].annotate('T=900 K',fontsize='x-large',xycoords='axes fraction',xy=(0.15,0.575))

plt.savefig('Q_calc.png',dpi=200,bbox_inches='tight')
plt.show()


