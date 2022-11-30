# -*- coding: utf-8 -*-
# @Time    : 2022/11/30 11:12
# @Author  : zhangkai
# @File    : Function.py
# @Software: PyCharm
import numpy as np


from scipy.special import jve


def Diffraction_Rect(mylambda,size_hole_x,size_hole_y,size_screen,distance=1000,l=1000):
    k = 2 * np.pi / mylambda
    xx=np.arange(-size_screen/2,size_screen/2,size_screen/l)
    yy=xx

    XX,YY=np.meshgrid(xx,yy)
    BB = (k * size_hole_x * XX) / (2. * distance)
    HH = (k * size_hole_y * YY) / (2. * distance)

    # E = size_hole_x * size_hole_y * (np.sin(BB) / BB) * (np.sin(HH) / HH)
    E = size_hole_x * size_hole_y * (np.sin(BB) / BB) * (np.sin(HH) / HH) * np.exp(1j * k * ((XX ** 2 + YY ** 2) / (2 * distance)))

    I = np.abs(E ** 2)

    I = (I - np.min(I)) / (np.max(I) - np.min(I))
    I = np.uint8(I * 255)
    I_center=I[round(np.size(I,1)/2)]
    return I,I_center


def Diffraction_Circle(mylambda, radius, size_screen, distance=1000, l=1000):
    k = 2 * np.pi / mylambda
    xx = np.arange(-size_screen / 2, size_screen / 2, size_screen/l)
    yy = xx

    XX, YY = np.meshgrid(xx, yy)
    theta=np.arctan(np.sqrt(XX**2+YY**2)/distance)
    # theta=np.sqrt(XX**2+YY**2)/distance       #近似取值
    theta[theta==0]=None


    E=radius**2*np.pi*2*jve(1,k*radius*theta)/(k*radius*theta)
    # E=radius**2*np.pi*2*jve(1,k*radius*theta)/(k*radius*theta)*np.exp(1j * k * radius**2 / (2 * distance))

    I = np.abs(E ** 2)
    I = (I - np.min(I)) / (np.max(I) - np.min(I))

    I = np.uint8(I * 255)
    print((np.size(I,1)))
    I_center = I[round(np.size(I,1)/2)]
    return I,I_center


def Diffraction_Grating(mylambda, a,d,N, size_screen, distance=1000, l=800):

    xx = np.arange(-size_screen / 2, size_screen / 2, size_screen/l)
    yy = xx
    XX, YY = np.meshgrid(xx, yy)
    theta = np.arctan(np.sqrt(XX ** 2) / distance)
    # theta=np.sqrt(XX**2+YY**2)/distance       #近似取值
    theta[theta == 0] = None

    alpha =a*np.pi*np.sin(theta)/mylambda
    delta=2*d*np.pi*np.sin(theta)/mylambda

    E = np.sinc(alpha) * np.sin(N * delta / 2) / np.sin(delta / 2) * np.exp(1j * (N - 1) * delta / 2)
    I = np.abs(E ** 2)
    I = (I - np.min(I)) / (np.max(I) - np.min(I))

    I = np.uint8(I * 255)
    print((np.size(I,1)))
    I_center = I[round(np.size(I,1)/2)]
    return I,I_center



def FFT_Rect(mylambda,size_hole_x,size_hole_y,size_screen,distance=1000,l=800):
    k = 2 * np.pi / mylambda
    xx=np.arange(-size_screen/2,size_screen/2,size_screen/l)
    yy=xx

    XX,YY=np.meshgrid(xx,yy)
    BB = (k * size_hole_x * XX) / (2. * distance)
    HH = (k * size_hole_y * YY) / (2. * distance)

    E = size_hole_x * size_hole_y * (np.sin(BB) / BB) * (np.sin(HH) / HH)
    # E = size_hole_x * size_hole_y * (np.sin(BB) / BB) * (np.sin(HH) / HH) * np.exp(1j * k * ((XX ** 2 + YY ** 2) / (2 * distance)))

    I = np.abs(E ** 2)

    I = (I - np.min(I)) / (np.max(I) - np.min(I))
    I = np.uint8(I * 255)
    I_center=I[round(np.size(I,1)/2)]
    return I,I_center


def FFT_Circle(mylambda, radius, size_screen, distance=1000, l=800):
    k = 2 * np.pi / mylambda
    xx = np.arange(-size_screen / 2, size_screen / 2, size_screen/l)
    yy = xx

    XX, YY = np.meshgrid(xx, yy)
    theta=np.arctan(np.sqrt(XX**2+YY**2)/distance)
    # theta=np.sqrt(XX**2+YY**2)/distance       #近似取值
    theta[theta==0]=None


    E=radius**2*np.pi*2*jve(1,k*radius*theta)/(k*radius*theta)
    # E=radius**2*np.pi*2*jve(1,k*radius*theta)/(k*radius*theta)*np.exp(1j * k * radius**2 / (2 * distance))

    I = np.abs(E ** 2)
    I = (I - np.min(I)) / (np.max(I) - np.min(I))

    I = np.uint8(I * 255)
    print((np.size(I,1)))
    I_center = I[round(np.size(I,1)/2)]
    return I,I_center


def FFT_Grating(mylambda, a,d,N, size_screen, distance=1000, l=800):

    xx = np.arange(-size_screen / 2, size_screen / 2, size_screen/l)
    yy = xx
    XX, YY = np.meshgrid(xx, yy)
    theta = np.arctan(np.sqrt(XX ** 2) / distance)
    # theta=np.sqrt(XX**2+YY**2)/distance       #近似取值
    theta[theta == 0] = None

    alpha =a*np.pi*np.sin(theta)/mylambda
    delta=2*d*np.pi*np.sin(theta)/mylambda

    E = np.sinc(alpha) * np.sin(N * delta / 2) / np.sin(delta / 2) * np.exp(1j * (N - 1) * delta / 2)
    I = np.abs(E ** 2)
    I = (I - np.min(I)) / (np.max(I) - np.min(I))

    I = np.uint8(I * 255)
    print((np.size(I,1)))
    I_center = I[round(np.size(I,1)/2)]
    return I,I_center