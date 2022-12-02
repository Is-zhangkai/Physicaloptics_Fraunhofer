# -*- coding: utf-8 -*-
# @Time    : 2022/11/28 16:40
# @Author  : zhangkai
# @File    : test.py
# @Software: PyCharm
import matplotlib.pyplot as plt
import numpy as np
from scipy import fft

import Function

# n=0.4
# x=np.arange(0,600+n,n)
# y=np.sin(2*np.pi*x/5)
# fmax=1/n/2
# Y=np.abs(fft.fftshift(fft.fft(y)))
# fx=np.linspace(-fmax,fmax,1501)
#
# plt.plot(fx,Y)
# plt.show()
# def Rect( x_L, y_L, radius, center=(0, 0)):
#
#     x_L = abs(x_L - center[0]) <= radius
#     y_L = abs(y_L - center[1]) <= radius
#     XX, YY = np.meshgrid(x_L, y_L)
#     return XX & YY
#
# rect_a=0.04
# rect_b=0.04
# size_x=6
# size_y=6
# dp=5E-3
# z0=1000
# mylambda=532E-6
# k=2*np.pi/mylambda
# num1=k*rect_a/(2*z0)
# num2=k*rect_b/(2*z0)
# num0=k/(2*z0)
# xx=np.arange(-size_x/2,size_x/2,dp)
# yy=np.arange(-size_y/2,size_y/2,dp)
# XX,YY=np.meshgrid(xx,yy)
#
#
#
# I = (np.sinc(num1*XX)*np.sinc(num2*YY))**2
# I=(I-np.min(I))/(np.max(I)-np.min(I))
#
# plt.imshow(I,cmap="gray")
# plt.show()

# mylambda=532E-6
# k=2*np.pi/mylambda
# f=2000
# N=400
# a=0.04
# b=0.04
# size=150
#
# X=np.linspace(-size/2.,size/2.,N)
# Y=X
#
# aerfa=(k*a*X)/(2.*f)
# beita=(k*b*Y)/(2.*f)
#
# A,B=np.meshgrid(aerfa,beita)
#
# I=((np.sin(A)/A)**2)*((np.sin(B)/B)**2)
# # I=(I-np.min(I))/(np.max(I)-np.min(I))
# plt.imshow(I,cmap="gray")
# plt.show()



size_x=10
size_y=10
mylambda=532E-6
k=2*np.pi/mylambda
distance=1000
f=200
xx=np.arange(-size_x/2,size_x/2,size_x/f)
yy=np.zeros(f)

a=0.1
b=0.5

n=size_x/b
c=b

# yy[np.abs(xx)<a/2]=True
# yy[np.abs(xx)>a/2]=False
# yy[np.abs(xx)>(c-a)]=True
# yy[np.abs(xx)>(c)]=False
# yy[np.abs(xx) > (2*c - a)] = True
# yy[np.abs(xx) > (2*c)] = False
# yy[np.abs(xx) > (3*c - a)] = True
# yy[np.abs(xx) > (3*c)] = False
#
#
# XX,YY=np.meshgrid(yy,yy)

# plt.figure()
# plt.imshow(XX)
# plt.show()
# E = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(XX)))
# E = E * np.exp(1j * k * distance) * np.exp(1j * k * (XX ** 2 + YY * 2) / (2 * distance)) / (
#         1j * distance * mylambda)
#
# I = np.abs(E ** 2)
# I = (I - np.min(I)) / (np.max(I) - np.min(I))
# I = np.uint8(I * 255)
# plt.figure()
# plt.plot(I[100])
#
# I,ic= Function.Diffraction_Grating(mylambda,a,b,7,size_x,distance,200)
# # plt.imshow(I,cmap="gray")
#
# plt.plot(ic,"r")
# plt.show()




circ=Function.Rect((xx+0.5)/0.5)+Function.Rect((xx-0.5)/0.5)

XX,YY=np.meshgrid(circ,circ)
plt.subplot(121)
plt.imshow(XX,"gray")
E = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(XX)))
# E = E * np.exp(1j * k * distance) * np.exp(1j * k * (XX ** 2 + YY * 2) / (2 * distance)) / (
#         1j * distance * mylambda)

I = np.abs(E ** 2)
I = (I - np.min(I)) / (np.max(I) - np.min(I))
I = np.uint8(I * 255)
plt.subplot(122)
plt.imshow(I,"gray")
plt.show()