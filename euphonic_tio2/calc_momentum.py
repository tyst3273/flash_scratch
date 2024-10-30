import numpy as np

hbar = 1.054571817e-34 # J*s
m = 1.67492749804e-27 # neutron mass in kg
meV_2_J = 1.6021773e-22 # meV to J 

a = 4.584 # angstrom
c = 2.953 # angstrom

h = 2*np.pi/a
l = 2*np.pi/c

print('')
print('h =',h,'2 pi / A')
print('l =',l,'2 pi / A')

# -----------------------------------------------

print('\nE = 14 meV\n')

E_14 = 14*meV_2_J

p_14 = np.sqrt(2*m*E_14)

print('p(E=14) =',p_14,'kg*m/s')

qmax_14 = 2*p_14/hbar/1e10 # 2 pi / meters -> 2 pi / angstrom

print('2p(E=14)/hbar =',qmax_14,'2 pi / A')

hmax_14 = qmax_14 / h
lmax_14 = qmax_14 / l

print('h_max(E=14) =',hmax_14,'rlu')
print('l_max(E=14) =',lmax_14,'rlu')

# -----------------------------------------------

print('\nE = 35 meV\n')

E_35 = 35*meV_2_J

p_35 = np.sqrt(2*m*E_35)

print('p(E=35) =',p_35,'kg*m/s')

qmax_35 = 2*p_35/hbar/1e10 # 2 pi / meters -> 2 pi / angstrom

print('2p(E=35)/hbar =',qmax_35,'2 pi / A')

hmax_35 = qmax_35 / h
lmax_35 = qmax_35 / l

print('h_max(E=35) =',hmax_35,'rlu')
print('l_max(E=35) =',lmax_35,'rlu')

# ---------------------------------------------------


print('\nmax accessible h, l =')
print('h(E=14) =',np.floor(hmax_14),'rlu')
print('h(E=14) =',np.floor(lmax_14),'rlu')
print('\nh(E=35) =',np.floor(hmax_35),'rlu')
print('l(E=35) =',np.floor(lmax_35),'rlu')
