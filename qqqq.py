# -*- coding: utf-8 -*-
# @Time    : 2022/12/2 16:52
# @Author  : zhangkai
# @File    : qqqq.py
# @Software: PyCharm
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import jve

import  Function as fun




def Diffraction_Grating(mylambda,a, d, size_source, distance=1000, N=400):
    k = 2 * np.pi / mylambda
    delt = size_source / N

    xx = np.arange(-size_source / 2, size_source / 2, delt)
    num = int(size_source // d + 1)
    grating=Grating(a,d,xx,N,num)
    I_fft,XX,YY=Fraunhofer_prop(grating,mylambda,delt,distance,N)
    I_fft[:]=I_fft[round(np.size(I_fft, 1) / 2)]

    theta = np.arctan(np.sqrt(XX ** 2) / distance)
    # theta=np.sqrt(XX**2+YY**2)/distance       #近似取值
    theta[theta == 0] = None

    alpha = a * np.pi * np.sin(theta) / mylambda
    delta = 2 * d * np.pi * np.sin(theta) / mylambda

    E = np.sinc(alpha) * np.sin(N * delta / 2) / np.sin(delta / 2) * np.exp(1j * (N - 1) * delta / 2)
    I = np.abs(E ** 2)
    I = (I - np.min(I)) / (np.max(I) - np.min(I))

    I = np.uint8(I * 255)

    return I,I_fft


def Rect(x):
    result = np.zeros(np.size(x))
    result[abs(x) <= 0.5] = 1
    return result


def Circ(xx,yy,radius):
    result = np.zeros((np.size(xx,1),np.size(yy,1)))
    result[abs(xx**2+yy**2) <= radius**2] = 1
    return result
def Grating(a,d,xx,N,num):
    grating = np.zeros(N)
    if N % 2 == 0:
        for i in range(num // 2):
            print(i)
            rect = Rect((xx + d / 2 + i * d) / a)
            grating += rect
            rect = Rect((xx - d / 2 - i * d) / a)
            grating += rect
    else:
        rect = Rect(xx / a)
        grating += rect
        for i in range(1, num // 2 + 1):
            print(i)
            rect = Rect((xx + i * d) / a)
            grating += rect
            rect = Rect((xx - i * d) / a)
            grating += rect
    grating, ass = np.meshgrid(grating, grating)
    return grating

def Fraunhofer_prop(img_In,mylambda,delta,distance,N):



    k = 2 * np.pi /mylambda

    fx = np.arange (-1 / 2, 1 / 2 ,1/N)/delta
    xx=mylambda *distance*fx
    XX,YY=np.meshgrid(xx,xx)
    E=np.fft.fftshift(np.fft.fft2(np.fft.fftshift(img_In)))*delta**2
    E=E*np.exp(1j * k * distance) * np.exp(1j * k * (XX ** 2 + YY * 2) / (2 * distance)) / (
            1j * distance * mylambda)

    I_fft = np.abs(E ** 2)
    I_fft = (I_fft - np.min(I_fft)) / (np.max(I_fft) - np.min(I_fft))
    I_fft = np.uint8(I_fft* 255)

    return I_fft,XX,YY






if __name__ == '__main__':
    mylambda = 532E-6
    distance = 2000
    size_screen = 10

    ic,i = Diffraction_Grating(mylambda, 0.05,0.1, 10, 1000, 1000)

    plt.subplot(121)
    plt.imshow(ic, cmap="gray")
    plt.subplot(122)
    plt.imshow(i, cmap="gray")
    plt.show()


