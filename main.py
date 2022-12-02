import numpy as np
from matplotlib import pyplot as plt
from scipy import fft

import Function as fun

mylambda=532E-6





# plt.figure()


# I,i_c=fun.Diffraction_Rect(mylambda,0.4,0.4,10,1000,400)
# I1,iic=fun.FFT_Rect(mylambda,0.4,0.4,10,800,)
# plt.subplot(131)
# plt.imshow(I,cmap="gray")
# plt.subplot(132)
# plt.imshow(I1,cmap="gray")
# plt.subplot(133)
# plt.imshow(iic,cmap="gray")
# plt.show()
#
# plt.subplot(121)
# plt.imshow(I,cmap="gray")
# plt.subplot(122)
# plt.imshow(I1,cmap="gray")
# plt.show()
a=0.01
d=0.02
plt.subplot(131)
I,n=fun.Diffraction_Grating(mylambda,a,d,int(10//d),10,1000,)
plt.imshow(I,cmap="gray")
plt.subplot(132)
I,n=fun.FFT_Grating(mylambda,a,d,int(10//d),10,1000,2000)
plt.imshow(I,cmap="gray")
plt.subplot(133)

plt.imshow(n,cmap="gray")


plt.show()