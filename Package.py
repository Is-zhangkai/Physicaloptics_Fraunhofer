# -*- coding: utf-8 -*-
# @Time    : 2022/12/2 14:05
# @Author  : zhangkai
# @File    : Package.py
# @Software: PyCharm
import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QGraphicsScene

import Function as fun
from Fraunhofer import Ui_MainWindow
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class figureCanvas(FigureCanvas):
    # 通过继承FigureCanvas类，使得该类既是一个PyQt5的Qwidget，又是一个matplotlib的FigureCanvas，这是连接pyqt5与matplot lib的关键

    def __init__(self, parent=None, axis=2, width=4, height=4):
        fig = Figure(figsize=(width, height), dpi=120)
        # 创建一个Figure，注意：该Figure为matplotlib下的figure，不是matplotlib.pyplot下面的figure
        FigureCanvas.__init__(self, fig)  # 初始化父类
        self.setParent(parent)
        if axis == 2:
            self.axes = fig.add_subplot(111)  # 调用figure下面的add_subplot方法，类似于matplotlib.pyplot下面的subplot方法
        else:
            self.axes3d = fig.gca(projection='3d')


class MainWindow(QMainWindow, Ui_MainWindow):


    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

        self.btn_rect.clicked.connect(self.Click_Rect)
        self.btn_circ.clicked.connect(self.Click_Circ)
        self.btn_grating.clicked.connect(self.Click_Grating)


    def Click_Rect(self):

        print("Click_Rect")
        try:
            mylambda=int(self.lineEdit_lambda_rect.text())*1E-6
            distance=int(self.lineEdit_distance_rect.text())
            size_source=float(self.lineEdit_source_rect.text())
            size_h=float(self.lineEdit_hole_rect.text())
            N=int(self.lineEdit_N_rect.text())
            I_diffraction,I_fft=fun.Diffraction_Rect(mylambda,size_h,size_source,distance,N)
            self.show_result(I_diffraction,I_fft,self.comboBox_rect.currentText())
        except:
            print("error:Click_Rect")

    def Click_Circ(self):
        print("Click_Circ")
        try:
            mylambda = int(self.lineEdit_lambda_circ.text()) * 1E-6
            distance = int(self.lineEdit_distance_circ.text())
            size_source = float(self.lineEdit_source_rect.text())
            size_h = float(self.lineEdit_hole_circ.text())
            N = int(self.lineEdit_N_circ.text())
            I_diffraction, I_fft = fun.Diffraction_Circle(mylambda, size_h, size_source, distance, N)
            self.show_result(I_diffraction, I_fft, self.comboBox_circ.currentText())
        except:
            print("error:Click_Circ")




    def Click_Grating(self):
        print("Click_Grating")
        try:
            mylambda = int(self.lineEdit_lambda_grating.text()) * 1E-6
            distance = int(self.lineEdit_distance_grating.text())
            size_source = float(self.lineEdit_source_grating.text())
            a = float(self.lineEdit_a_grating.text())
            d=float(self.lineEdit_d_grating.text())
            N = int(self.lineEdit_N_grating.text())

            I_diffraction, I_fft = fun.Diffraction_Grating(mylambda, a,d, int(size_source // d + 1),size_source, distance, N)
            self.show_result(I_diffraction, I_fft, self.comboBox_grating.currentText())
        except:
            print("error:Click_Grating")


    def show_image(self, img, view):
        """
        :param img:
        :param view:
        :return:
        """
        flag = len(img.shape)
        if flag == 2:
            height, width = img.shape
            bytesPer = width
            imgQ = QImage(img.data, width, height, bytesPer, QImage.Format_Indexed8)
        else:
            height, width, ans = img.shape
            bytesPer = width * 3
            imgQ = QImage(img.data, width, height, bytesPer, QImage.Format_RGB888)

        imgQPixmap = QPixmap.fromImage(imgQ).scaledToWidth(view.width()-4)
        scene = QGraphicsScene()
        scene.addPixmap(imgQPixmap)
        view.setScene(scene)
        view.show()
    def show_result(self,I_diffraction,I_fft,text):
        """

        :param I_diffraction:
        :param I_fft:
        :param text:
        :return:
        """

        if text=="平面图像":
            self.show_image(I_diffraction, self.graphicsView_left)
            cv2.imwrite("1.jpg",I_fft)
            I_fft=cv2.imread("1.jpg")
            self.show_image(I_fft,self.graphicsView_right)
        else:
            y1=I_diffraction[round(np.size(I_diffraction, 0) / 2)]
            y2=I_fft[round(np.size(I_fft, 0) / 2)]
            figure1 = figureCanvas()
            figure1.axes.plot(y1 ,'b', linewidth=1, label="diffraction")
            figure1.axes.plot(y2,'r', linewidth=0.5,linestyle="--", label="fft")
            figure1.axes.legend()
            figure1.axes.grid()
            figure1.axes.set_title("Distribution of light intensity")
            graphicscene1 = QGraphicsScene()
            graphicscene1.addWidget(figure1)
            self.graphicsView_left.setScene(graphicscene1)
            self.graphicsView_left.show()

            figure2 = figureCanvas()
            figure2.axes.plot( y1-y2)
            figure2.axes.grid()
            figure2.axes.set_title("I_diffraction-I_fft")
            graphicscene2 = QGraphicsScene()
            graphicscene2.addWidget(figure2)
            self.graphicsView_right.setScene(graphicscene2)
            self.graphicsView_right.show()
