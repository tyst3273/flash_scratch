
import importlib 

import m_euphonic_sqw 
importlib.reload(m_euphonic_sqw)

import matplotlib.pyplot as plt
import numpy as np
import h5py

sqw = m_euphonic_sqw.c_euphonic_sqw(phonopy_file='phonopy_no_dipole.yaml')
#sqw = m_euphonic_sqw.c_euphonic_sqw(phonopy_file='phonopy.yaml')

h = np.arange(0.5,4.5,0.5)
l = np.arange(0.5,6.5,0.5)
Qpts = np.meshgrid(h,h,l,indexing='ij')
Qpts = np.c_[Qpts[0].flatten(),Qpts[1].flatten(),Qpts[2].flatten()]
print(Qpts.shape)

sqw.set_Qpts(Qpts)

temps = [600]

for T in temps:

    sqw.calculate_structure_factors(temperature=T,dw_grid=[20,20,20])
    sqw.get_colormap(E_min=-50,E_max=50,dE=0.05,E_width=2.5,temperature=T)
    sqw.save_structure_factors_to_hdf5(hdf5_file=f'T_{T:g}K.hdf5')
    #sqw.save_structure_factors_to_hdf5(f'split_all_{ii}.hdf5')

















