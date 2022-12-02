# -*- coding: utf-8 -*-
# @Time    : 2022/11/30 11:12
# @Author  : zhangkai
# @File    : Function.py
# @Software: PyCharm

import numpy as np

from scipy.special import jve


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

    E = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(rect)))
    # E = E * np.exp(1j * k * distance) * np.exp(1j * k * (XX ** 2 + YY * 2) / (2 * distance)) / (
    #         1j * distance * mylambda)

    I = np.abs(np.real(E) ** 2)
    I = (I - np.min(I)) / (np.max(I) - np.min(I))
    I = np.uint8(I * 255)
    I_center = I[round(np.size(I, 1) / 2)]
    return I, rect


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
    result[abs(x) < 0.5] = True
    return result
