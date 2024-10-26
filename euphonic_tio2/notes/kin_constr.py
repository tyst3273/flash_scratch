

import numpy as np
import matplotlib.pyplot as plt

m = 1 # 1.674927e-27 # kg

Ei = 100

num_E = 500
E = np.linspace(-250,250,num_E)

num_angles = 500
theta = np.linspace(0,2*np.pi,num_angles)

E_m, theta_m = np.meshgrid(E,theta,indexing='ij')
Ef = Ei - E

p = np.sqrt(2*m*(Ei + Ef - 2*np.sqrt(Ei*Ef)*np.cos(theta)))

print(p)

plt.plot(theta/2/np.pi,p,lw=1,ls='-',ms=0,c='m')
plt.show()




