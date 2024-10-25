

import numpy as np
import matplotlib.pyplot as plt

m = 1 # 1.674927e-27 # kg

Ei = 100

num_E = 500
Ef = np.linspace(0,250,num_E)
#Ef = 10

num_angles = 500
theta = np.linspace(0,2*np.pi,num_angles)

Ef_m, theta_m = np.meshgrid(Ef,theta,indexing='ij')

p = np.sqrt(2*m*(Ei + Ef - 2*np.sqrt(Ei*Ef)*np.cos(theta)))

print(p)

plt.plot(theta/2/np.pi,p,lw=1,ls='-',ms=0,c='m')
plt.show()




