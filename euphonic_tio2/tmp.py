import numpy as np

h = np.array([1,1.5,2,2.5,3])

h, k = np.meshgrid(h,h,indexing='ij')

h = h.flatten()
k = k.flatten()

Q = np.c_[h,k]
print(Q)

