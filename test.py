# -*- coding: utf-8 -*-
# @Time    : 2022/11/28 16:40
# @Author  : zhangkai
# @File    : test.py
# @Software: PyCharm
import matplotlib.pyplot as plt
import numpy as np
from scipy import fft
from scipy.special import jve

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

def Diffraction_Rect(mylambda, size_hole_x, size_hole_y, size_screen, distance=1000, f=400):
    k = 2 * np.pi / mylambda
    xx = np.arange(-size_screen / 2, size_screen / 2, size_screen / f)
    yy = xx

    XX, YY = np.meshgrid(xx, yy)
    BB = (k * size_hole_x * XX) / (2. * distance)
    HH = (k * size_hole_y * YY) / (2. * distance)

    # E = size_hole_x * size_hole_y * (np.sin(BB) / BB) * (np.sin(HH) / HH)
    E = size_hole_x * size_hole_y * (np.sin(BB) / BB) * (np.sin(HH) / HH) * np.exp(
        1j * k * ((XX ** 2 + YY ** 2) / (2 * distance)))

    I = np.abs(E ** 2)

    I = (I - np.min(I)) / (np.max(I) - np.min(I))
    I = np.uint8(I * 255)
    I_center = I[round(np.size(I, 1) / 2)]
    return I, I_center


def Diffraction_Circle(mylambda, radius, size_screen, distance=1000, f=400):
    k = 2 * np.pi / mylambda
    xx = np.arange(-size_screen / 2, size_screen / 2, size_screen / f)
    yy = xx

    XX, YY = np.meshgrid(xx, yy)
    theta = np.arctan(np.sqrt(XX ** 2 + YY ** 2) / distance)
    # theta=np.sqrt(XX**2+YY**2)/distance       #近似取值
    theta[theta == 0] = None

    E = radius ** 2 * np.pi * 2 * jve(1, k * radius * theta) / (k * radius * theta)
    # E=radius**2*np.pi*2*jve(1,k*radius*theta)/(k*radius*theta)*np.exp(1j * k * radius**2 / (2 * distance))

    I = np.abs(E ** 2)
    I = (I - np.min(I)) / (np.max(I) - np.min(I))

    I = np.uint8(I * 255)
    print((np.size(I, 1)))
    I_center = I[round(np.size(I, 1) / 2)]
    return I, I_center


def Diffraction_Grating(mylambda, a, d, N, size_screen, distance=1000, f=400):
    xx = np.arange(-size_screen / 2, size_screen / 2, size_screen / f)
    yy = xx
    XX, YY = np.meshgrid(xx, yy)
    theta = np.arctan(np.sqrt(XX ** 2) / distance)
    # theta=np.sqrt(XX**2+YY**2)/distance       #近似取值
    theta[theta == 0] = None

    alpha = a * np.pi * np.sin(theta) / mylambda
    delta = 2 * d * np.pi * np.sin(theta) / mylambda

    E = np.sinc(alpha) * np.sin(N * delta / 2) / np.sin(delta / 2) * np.exp(1j * (N - 1) * delta / 2)
    I = np.abs(E ** 2)
    I = (I - np.min(I)) / (np.max(I) - np.min(I))

    I = np.uint8(I * 255)
    print((np.size(I, 1)))
    I_center = I[round(np.size(I, 1) / 2)]
    return I, I_center


def FFT_Rect(mylambda, size_hole_x, size_hole_y, size_screen, distance=1000, f=400):
    k = 2 * np.pi / mylambda

    size_screen = np.arange(-size_screen / 2, size_screen / 2, size_screen / f)
    xx = yy = np.zeros(f)
    xx[np.abs(size_screen - 0) < (size_hole_x / 2)] = 1
    yy[np.abs(size_screen - 0) < (size_hole_y / 2)] = 1
    XX, YY = np.meshgrid(xx, yy)
    rect = XX * YY
    rect=np.uint8(rect * 255)
    E = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(rect)))
    E = E * np.exp(1j * k * distance) * np.exp(1j * k * (XX ** 2 + YY * 2) / (2 * distance)) / (
            1j * distance * mylambda)

    I = np.abs(E ** 2)
    I = (I - np.min(I)) / (np.max(I) - np.min(I))
    I = np.uint8(I * 255)
    I_center = I[round(np.size(I, 1) / 2)]
    return I, I_center


def FFT_Circle(mylambda, radius, size_screen, distance=1000, f=400):
    k = 2 * np.pi / mylambda

    xx = np.arange(-size_screen / 2, size_screen / 2, size_screen / f)
    XX, YY = np.meshgrid(xx, xx)

    circ = np.zeros((f, f))
    circ[np.sqrt(XX ** 2 + YY ** 2) < radius] = 1

    E = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(circ)))
    E = E * np.exp(1j * k * distance) * np.exp(1j * k * (XX ** 2 + YY * 2) / (2 * distance)) / (
            1j * distance * mylambda)

    I = np.abs(E ** 2)
    I = (I - np.min(I)) / (np.max(I) - np.min(I))
    I = np.uint8(I * 255)
    print((np.size(I, 1)))
    I_center = I[round(np.size(I, 1) / 2)]
    return I, I_center


def FFT_Grating(mylambda, a, d, N, size_screen, distance=1000, f=400):
    k = 2 * np.pi / mylambda
    xx = np.arange(-size_screen / 2, size_screen / 2, size_screen / f)
    # E = np.zeros(f)
    # mm = np.zeros(f)
    E = np.zeros((f,f))
    mm = np.zeros(f)
    print(f)


    if N %2 == 0:
        for i in range(N // 2 ):
            print(i)
            circ = Rect((xx + d / 2 + i  * d) / a)
            # circ, YY = np.meshgrid(circ, circ)
            # E = E + np.fft.fftshift(np.fft.fft2(np.fft.fftshift(circ)))
            mm = mm + circ
            circ = Rect((xx - d / 2 - i  * d) / a)
            # circ, YY = np.meshgrid(circ, circ)
            mm = mm + circ
            # E = E + np.fft.fftshift(np.fft.fft2(np.fft.fftshift(circ)))
            # E = E * np.exp(1j * k * distance) * np.exp(1j * k * (XX ** 2 + YY * 2) / (2 * distance)) / (
            #         1j * distance * mylambda)

    else:

        circ = Rect( xx / a)
        # circ, YY = np.meshgrid(circ, circ)
        mm = mm + circ
        # E = E + np.fft.fftshift(np.fft.fft2(np.fft.fftshift(circ)))
        for i in range(1, N // 2+1):
            print(i)
            circ = Rect((xx + i * d) / a)
            # circ, YY = np.meshgrid(circ, circ)
            mm = mm + circ
            # E = E + np.fft.fftshift(np.fft.fft2(np.fft.fftshift(circ)))
            circ = Rect((xx - i * d) / a)
            # circ, YY = np.meshgrid(circ, circ)
            mm = mm + circ
            # E = E + np.fft.fftshift(np.fft.fft2(np.fft.fftshift(circ)))

    # mm=(mm-np.min(mm))/(np.max(mm)-np.min(mm))
    aasd, ass = np.meshgrid(mm, mm)

    E = np.fft.fftshift(np.fft.fft(np.fft.fftshift(aasd)))


    # E ,Ey=np.meshgrid(E,E)
    # E = E * np.exp(1j
    # * k * distance) * np.exp(1j * k * (XX ** 2 + YY * 2) / (2 * distance)) / (
    #         1j * distance * mylambda)


    I = np.abs(E ** 2)
    I = (I - np.min(I)) / (np.max(I) - np.min(I))
    I = np.uint8(I * 255)

    I_center = I[round(np.size(I, 1) / 2)]

    return I, aasd


def Rect(x):
    result = np.zeros(np.size(x))
    result[abs(x) <= 0.5] = True
    return result

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