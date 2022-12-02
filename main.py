import sys

import numpy as np
from matplotlib import pyplot as plt
from scipy import fft

from PyQt5 import QtWidgets

from Package import MainWindow

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()

    window.show()
    sys.exit(app.exec_())
# plt.show()
#
# plt.subplot(121)
# plt.imshow(I,cmap="gray")
# plt.subplot(122)
# plt.imshow(I1,cmap="gray")
# plt.show()

#
# a=0.01
# d=0.02
# plt.subplot(131)
# I,n=fun.Diffraction_Grating(mylambda,a,d,int(10//d),10,1000,)
# plt.imshow(I,cmap="gray")
# plt.subplot(132)
# I,n=fun.FFT_Grating(mylambda,a,d,int(10//d),10,1000,1000)
# plt.imshow(I,cmap="gray")
# plt.subplot(133)
#
# plt.imshow(n,cmap="gray")
#

plt.show()