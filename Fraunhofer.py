# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Fraunhofer.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1080, 720)
        MainWindow.setMinimumSize(QtCore.QSize(1080, 720))
        MainWindow.setMaximumSize(QtCore.QSize(1080, 720))
        MainWindow.setSizeIncrement(QtCore.QSize(1080, 720))
        MainWindow.setBaseSize(QtCore.QSize(1080, 720))
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(13)
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, -1, 1081, 51))
        font = QtGui.QFont()
        font.setFamily("楷体")
        font.setPointSize(24)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(30, 50, 1020, 150))
        self.tabWidget.setMinimumSize(QtCore.QSize(1020, 150))
        self.tabWidget.setMaximumSize(QtCore.QSize(1020, 150))
        self.tabWidget.setSizeIncrement(QtCore.QSize(1020, 150))
        self.tabWidget.setBaseSize(QtCore.QSize(1020, 150))
        self.tabWidget.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.tabWidget.setAutoFillBackground(False)
        self.tabWidget.setTabPosition(QtWidgets.QTabWidget.North)
        self.tabWidget.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.tabWidget.setElideMode(QtCore.Qt.ElideMiddle)
        self.tabWidget.setObjectName("tabWidget")
        self.tab1 = QtWidgets.QWidget()
        self.tab1.setObjectName("tab1")
        self.lineEdit = QtWidgets.QLineEdit(self.tab1)
        self.lineEdit.setGeometry(QtCore.QRect(110, 20, 160, 36))
        self.lineEdit.setMinimumSize(QtCore.QSize(160, 36))
        self.lineEdit.setMaximumSize(QtCore.QSize(160, 36))
        self.lineEdit.setSizeIncrement(QtCore.QSize(160, 36))
        self.lineEdit.setBaseSize(QtCore.QSize(160, 36))
        self.lineEdit.setObjectName("lineEdit")
        self.pushButton = QtWidgets.QPushButton(self.tab1)
        self.pushButton.setGeometry(QtCore.QRect(860, 40, 140, 42))
        self.pushButton.setMinimumSize(QtCore.QSize(140, 42))
        self.pushButton.setMaximumSize(QtCore.QSize(140, 42))
        self.pushButton.setSizeIncrement(QtCore.QSize(140, 42))
        self.pushButton.setBaseSize(QtCore.QSize(120, 42))
        self.pushButton.setObjectName("pushButton")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.tab1)
        self.lineEdit_2.setGeometry(QtCore.QRect(110, 70, 160, 36))
        self.lineEdit_2.setMinimumSize(QtCore.QSize(160, 36))
        self.lineEdit_2.setMaximumSize(QtCore.QSize(160, 36))
        self.lineEdit_2.setSizeIncrement(QtCore.QSize(160, 36))
        self.lineEdit_2.setBaseSize(QtCore.QSize(160, 36))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.lineEdit_3 = QtWidgets.QLineEdit(self.tab1)
        self.lineEdit_3.setGeometry(QtCore.QRect(390, 20, 160, 36))
        self.lineEdit_3.setMinimumSize(QtCore.QSize(160, 36))
        self.lineEdit_3.setMaximumSize(QtCore.QSize(160, 36))
        self.lineEdit_3.setSizeIncrement(QtCore.QSize(160, 36))
        self.lineEdit_3.setBaseSize(QtCore.QSize(160, 36))
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.lineEdit_4 = QtWidgets.QLineEdit(self.tab1)
        self.lineEdit_4.setGeometry(QtCore.QRect(390, 70, 160, 36))
        self.lineEdit_4.setMinimumSize(QtCore.QSize(160, 36))
        self.lineEdit_4.setMaximumSize(QtCore.QSize(160, 36))
        self.lineEdit_4.setSizeIncrement(QtCore.QSize(160, 36))
        self.lineEdit_4.setBaseSize(QtCore.QSize(160, 36))
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.lineEdit_5 = QtWidgets.QLineEdit(self.tab1)
        self.lineEdit_5.setGeometry(QtCore.QRect(670, 20, 160, 36))
        self.lineEdit_5.setMinimumSize(QtCore.QSize(160, 36))
        self.lineEdit_5.setMaximumSize(QtCore.QSize(160, 36))
        self.lineEdit_5.setSizeIncrement(QtCore.QSize(160, 36))
        self.lineEdit_5.setBaseSize(QtCore.QSize(160, 36))
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.lineEdit_6 = QtWidgets.QLineEdit(self.tab1)
        self.lineEdit_6.setGeometry(QtCore.QRect(670, 70, 160, 36))
        self.lineEdit_6.setMinimumSize(QtCore.QSize(160, 36))
        self.lineEdit_6.setMaximumSize(QtCore.QSize(160, 36))
        self.lineEdit_6.setSizeIncrement(QtCore.QSize(160, 36))
        self.lineEdit_6.setBaseSize(QtCore.QSize(160, 36))
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.tabWidget.addTab(self.tab1, "")
        self.tab2 = QtWidgets.QWidget()
        self.tab2.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.tab2.setObjectName("tab2")
        self.pushButton_2 = QtWidgets.QPushButton(self.tab2)
        self.pushButton_2.setGeometry(QtCore.QRect(860, 40, 140, 42))
        self.pushButton_2.setMinimumSize(QtCore.QSize(140, 42))
        self.pushButton_2.setMaximumSize(QtCore.QSize(140, 42))
        self.pushButton_2.setSizeIncrement(QtCore.QSize(140, 42))
        self.pushButton_2.setBaseSize(QtCore.QSize(120, 42))
        self.pushButton_2.setObjectName("pushButton_2")
        self.tabWidget.addTab(self.tab2, "")
        self.tab3 = QtWidgets.QWidget()
        self.tab3.setObjectName("tab3")
        self.pushButton_3 = QtWidgets.QPushButton(self.tab3)
        self.pushButton_3.setGeometry(QtCore.QRect(860, 40, 140, 42))
        self.pushButton_3.setMinimumSize(QtCore.QSize(140, 42))
        self.pushButton_3.setMaximumSize(QtCore.QSize(140, 42))
        self.pushButton_3.setSizeIncrement(QtCore.QSize(140, 42))
        self.pushButton_3.setBaseSize(QtCore.QSize(120, 42))
        self.pushButton_3.setObjectName("pushButton_3")
        self.tabWidget.addTab(self.tab3, "")
        self.graphicsView_left = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView_left.setGeometry(QtCore.QRect(30, 200, 500, 500))
        self.graphicsView_left.setMinimumSize(QtCore.QSize(500, 500))
        self.graphicsView_left.setMaximumSize(QtCore.QSize(500, 500))
        self.graphicsView_left.setSizeIncrement(QtCore.QSize(500, 500))
        self.graphicsView_left.setBaseSize(QtCore.QSize(500, 500))
        self.graphicsView_left.setObjectName("graphicsView_left")
        self.graphicsView_right = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView_right.setGeometry(QtCore.QRect(550, 200, 500, 500))
        self.graphicsView_right.setMinimumSize(QtCore.QSize(500, 500))
        self.graphicsView_right.setMaximumSize(QtCore.QSize(500, 500))
        self.graphicsView_right.setSizeIncrement(QtCore.QSize(500, 500))
        self.graphicsView_right.setBaseSize(QtCore.QSize(500, 500))
        self.graphicsView_right.setObjectName("graphicsView_right")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "夫朗和费衍射"))
        self.pushButton.setText(_translate("MainWindow", "PushButton"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab1), _translate("MainWindow", "矩孔"))
        self.pushButton_2.setText(_translate("MainWindow", "PushButton"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab2), _translate("MainWindow", "圆孔"))
        self.pushButton_3.setText(_translate("MainWindow", "PushButton"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab3), _translate("MainWindow", "光栅"))
