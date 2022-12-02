# -*- coding: utf-8 -*-
# @Time    : 2022/12/2 16:52
# @Author  : zhangkai
# @File    : main.py
# @Software: PyCharm

import sys
from PyQt5 import QtWidgets
from Package import MainWindow

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()

    window.show()
    sys.exit(app.exec_())

