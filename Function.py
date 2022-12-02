# -*- coding: utf-8 -*-
# @Time    : 2022/11/30 11:12
# @Author  : zhangkai
# @File    : Function.py
# @Software: PyCharm

import numpy as np
from scipy.special import jve


def Diffraction_Rect(mylambda, size_hole, size_source=10, distance=1000, N=400):
    """
    @Author:zhangkai
    :param mylambda:
    :param size_hole:
    :param size_source:
    :param distance:
    :param N:
    :return:
    """
    k = 2 * np.pi / mylambda
    delt = size_source / N

    xx = np.arange(-size_source / 2, size_source / 2, delt)
    xx = Rect(xx / size_hole)
    XX, YY = np.meshgrid(xx, xx)
    rect = XX * YY
    I_fft, XX, YY = Fraunhofer_prop(rect, mylambda, delt, distance, N)

    BB = (k * size_hole * XX) / (2. * distance)
    HH = (k * size_hole * YY) / (2. * distance)

    # E = size_hole * size_hole * (np.sin(BB) / BB) * (np.sin(HH) / HH)
    E = size_hole * size_hole * (np.sin(BB) / BB) * (np.sin(HH) / HH) * np.exp(
        1j * k * ((XX ** 2 + YY ** 2) / (2 * distance)))

    I = np.abs(E ** 2)
    I = (I - np.min(I)) / (np.max(I) - np.min(I))
    I = np.uint8(I * 255)

    return I, I_fft


def Diffraction_Circle(mylambda, radius, size_source=10, distance=1000, N=400):
    """
    @Author:zhangkai
    :param mylambda:
    :param radius:
    :param size_source:
    :param distance:
    :param N:
    :return:
    """
    k = 2 * np.pi / mylambda
    delt = size_source / N

    xx = np.arange(-size_source / 2, size_source / 2, delt)

    XX, YY = np.meshgrid(xx, xx)
    circ = Circ(XX, YY, radius)

    I_fft, XX, YY = Fraunhofer_prop(circ, mylambda, delt, distance, N)

    theta = np.arctan(np.sqrt(XX ** 2 + YY ** 2) / distance)
    # theta=np.sqrt(XX**2+YY**2)/distance       #近似取值
    theta[theta == 0] = None

    E = radius ** 2 * np.pi * 2 * jve(1, k * radius * theta) / (k * radius * theta)
    # E=radius**2*np.pi*2*jve(1,k*radius*theta)/(k*radius*theta)*np.exp(1j * k * radius**2 / (2 * distance))

    I = np.abs(E ** 2)
    I = (I - np.min(I)) / (np.max(I) - np.min(I))

    I = np.uint8(I * 255)

    return I, I_fft


def Diffraction_Grating(mylambda, a, d,num, size_source=10, distance=1000, N=1000):
    """
    @Author:zhangkai
    :param mylambda:
    :param a:
    :param d:
    :param size_source:
    :param distance:
    :param N:
    :return:
    """

    delt = size_source / N

    xx = np.arange(-size_source / 2, size_source / 2, delt)

    grating = Grating(a, d, xx, N, num)
    I_fft, XX, YY = Fraunhofer_prop(grating, mylambda, delt, distance, N)
    I_fft[:] = I_fft[round(np.size(I_fft, 1) / 2)]

    theta = np.arctan(np.sqrt(XX ** 2) / distance)
    # theta=np.sqrt(XX**2+YY**2)/distance       #近似取值
    theta[theta == 0] = None

    alpha = a * np.pi * np.sin(theta) / mylambda
    delta = 2 * d * np.pi * np.sin(theta) / mylambda

    E = np.sinc(alpha) * np.sin(N * delta / 2) / np.sin(delta / 2) * np.exp(1j * (N - 1) * delta / 2)
    I = np.abs(E ** 2)
    I = (I - np.min(I)) / (np.max(I) - np.min(I))

    I = np.uint8(I * 255)

    return I, I_fft


def Fraunhofer_prop(img_In, mylambda, delta, distance, N):
    """
    @Author:zhangkai
    :param img_In:
    :param mylambda:
    :param delta:
    :param distance:
    :param N:
    :return:
    """
    k = 2 * np.pi / mylambda
    fx = np.arange(-1 / 2, 1 / 2, 1 / N) / delta
    xx = mylambda * distance * fx
    XX, YY = np.meshgrid(xx, xx)
    E = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(img_In))) * delta ** 2

    E = E * np.exp(1j * k * distance) * np.exp(1j * k * (XX ** 2 + YY * 2) / (2 * distance)) / (
            1j * distance * mylambda)

    I_fft = np.abs(E ** 2)
    I_fft = (I_fft - np.min(I_fft)) / (np.max(I_fft) - np.min(I_fft))
    I_fft = np.uint8(I_fft * 255)

    return I_fft, XX, YY


def Rect(x):
    """
    @Author:zhangkai
    :param x:
    :return:
    """
    result = np.zeros(np.size(x))
    result[abs(x) <= 0.5] = 1
    return result


def Circ(xx, yy, radius):
    """
    @Author:zhangkai
    :param xx:
    :param yy:
    :param radius:
    :return:
    """
    result = np.zeros((np.size(xx, 1), np.size(yy, 1)))
    result[abs(xx ** 2 + yy ** 2) <= radius ** 2] = 1
    return result


def Grating(a, d, xx, N, num):
    """
    @Author:zhangkai
    :param a:
    :param d:
    :param xx:
    :param N:
    :param num:
    :return:
    """
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
