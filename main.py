import numpy as np
from matplotlib import pyplot as plt
from scipy import fft

import Function as fun

mylambda=532E-6





plt.figure()
I,i_c=fun.Diffraction_Rect(mylambda, 0.4,0.4,20)
plt.plot(i_c)

# I,i_c=fun.Diffraction_Grating(mylambda,0.06,0.5,10,10)
# plt.imshow(I,cmap="gray")
# plt.show()

size=10
size_rect=0.4
l=1000
k = 2 * np.pi / mylambda
x=np.arange(-size/2,size/2,size/1000)
xx=np.zeros(1000)
xx[np.abs(x-0)<size_rect/2]=1
y=xx
XX,YY=np.meshgrid(xx,y)
rect=XX *YY



omega = fft.fftshift(fft.fftfreq(len(x), 1 / 1000))
omegaLen = len(omega)
omegaLenF, omegaLenE = omegaLen * 3 // 8, omegaLen * 5 // 8 + 1

E=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(rect)))

E=E*np.exp(1j*k*l)*np.exp(1j*k*(XX**2+YY*2)/(2*l))/(1j*l*mylambda)
# E=E[omegaLenF:omegaLenE,omegaLenF:omegaLenE]
I = np.abs(E ** 2)
I = (I - np.min(I)) / (np.max(I) - np.min(I))

I = np.uint8(I * 255)
I=I[omegaLenF:omegaLenE,omegaLenF:omegaLenE]
# plt.figure()
# plt.imshow(I,cmap="gray")
plt.plot(I[126])
plt.show()
