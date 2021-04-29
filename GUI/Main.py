'''
@ Date : 2020-12-21
@ Author : Harry
@ Edit Date : 2020-12-22
@ brief : 시스템 이미지 1차 분류기 (OPTICAL201 ~ 203)
'''

import sys
import os
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
import shutil


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(499, 313)
        MainWindow.setFixedSize(499, 313)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("Y:/_사내업무/회사양식/마스코리아 로고(저용량).png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 10, 471, 61))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.lineEdit = QtWidgets.QLineEdit(self.horizontalLayoutWidget)
        self.lineEdit.setObjectName("lineEdit")
        self.horizontalLayout.addWidget(self.lineEdit)
        self.pushButton = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout.addWidget(self.pushButton)
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(410, 280, 75, 23))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(310, 280, 91, 23))
        self.pushButton_3.setObjectName("pushButton_3")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(10, 80, 471, 192))
        self.textBrowser.setObjectName("textBrowser")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(11, 283, 201, 20))
        self.label_2.setObjectName("label_2")
        MainWindow.setCentralWidget(self.centralwidget)
        # Directory Select Event
        self.pushButton.clicked.connect(self.pushButtonClicked)
        # Classifier Button Event
        self.pushButton_3.clicked.connect(self.classifierEvent)
        # Exit Button Event
        self.pushButton_2.clicked.connect(QCoreApplication.instance().quit)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "System Image Classifier"))
        self.label.setText(_translate("MainWindow", "Directory : "))
        self.pushButton.setText(_translate("MainWindow", "Select"))
        self.pushButton_2.setText(_translate("MainWindow", "EXIT"))
        self.pushButton_3.setText(_translate("MainWindow", "Classfication"))
        self.label_2.setText(_translate("MainWindow", "Made in Harry @MaaS-Korea Corp"))

    def pushButtonClicked(self):
        global directory, dir_list

        directory = str(QFileDialog.getExistingDirectory())
        try:
            self.lineEdit.setText(directory)
            self.textBrowser.setText('=' * 70)
            self.textBrowser.append(f'Directory : {directory}')
            self.textBrowser.append('=' * 70)
            dir_list = os.listdir(directory)
            self.textBrowser.append('Folder List')

            num = 0

            for root in dir_list:
                if not '.' in root:
                    self.textBrowser.append(f'{num} : {root}')
                    num += 1
                else:
                    continue

        except Exception as err:
            self.textBrowser.setText('Error : 디렉토리를 선택해주세요.')
            print(err)




    def classifierEvent(self):

        self.textBrowser.append('=' * 70)
        self.textBrowser.append('시스템 이미지 분류를 시작합니다.')
        self.textBrowser.append('=' * 70)
        try:
            self.create_system_folder()
        except:
            print('error!!!!')
        try:
            self.textBrowser.append('201 이미지 파일을 옮깁니다.')
            self.classifier_system_image(201)
        except:
            self.textBrowser.append('201 이미지 파일이 존재하지 않습니다.')

        try:
            self.textBrowser.append('202 이미지 파일을 옮깁니다.')
            self.classifier_system_image(202)
        except:
            self.textBrowser.append('202 이미지 파일이 존재하지 않습니다.')

        try:
            self.textBrowser.append('203 이미지 파일을 옮깁니다.')
            self.classifier_system_image(203)
        except:
            self.textBrowser.append('203 이미지 파일이 존재하지 않습니다.')

    def create_system_folder(self):
        global dir_path
        dir_path = []
        system_folder = [201, 202, 203]
        self.textBrowser.append(f'시스템 폴더를 생성합니다.')
        try:
            for root in dir_list:
                PATH = f'{directory}/{root}'
                if os.path.isdir(PATH):
                    dir_path.append(PATH)

            for root in dir_path:
                # 분류 폴더 생성
                for folder in system_folder:
                    if not os.path.isdir(f'{root}/{folder}'):
                        os.mkdir(f'{root}/{folder}')
                        # 상세 폴더 생성
                        if folder == 201:
                            os.mkdir(f'{root}/{folder}/dry_day')
                            os.mkdir(f'{root}/{folder}/dry_night')
                            os.mkdir(f'{root}/{folder}/dry_sunrise')
                            os.mkdir(f'{root}/{folder}/dry_sunset')
                            os.mkdir(f'{root}/{folder}/wet_day')
                            os.mkdir(f'{root}/{folder}/wet_night')
                            os.mkdir(f'{root}/{folder}/snowcover_day')
                            os.mkdir(f'{root}/{folder}/snowcover_night')
                            os.mkdir(f'{root}/{folder}/slush_day')
                            os.mkdir(f'{root}/{folder}/slush_night')
                            os.mkdir(f'{root}/{folder}/wet_sunrise')
                            os.mkdir(f'{root}/{folder}/wet_sunset')
                            os.mkdir(f'{root}/{folder}/ing')
                            os.mkdir(f'{root}/{folder}/etc')
                        if folder == 202:
                            os.mkdir(f'{root}/{folder}/normal_day')
                            os.mkdir(f'{root}/{folder}/normal_night')
                            os.mkdir(f'{root}/{folder}/fog_day')
                            os.mkdir(f'{root}/{folder}/fog_night')
                            os.mkdir(f'{root}/{folder}/rainfall_day')
                            os.mkdir(f'{root}/{folder}/rainfall_night')
                            os.mkdir(f'{root}/{folder}/snow_day')
                            os.mkdir(f'{root}/{folder}/snow_night')
                            os.mkdir(f'{root}/{folder}/sunrise')
                            os.mkdir(f'{root}/{folder}/sunset')
                            os.mkdir(f'{root}/{folder}/etc')
                        # if folder == 203:
                        #     os.mkdir(f'{root}/{folder}/')
        except Exception as err:
            self.textBrowser.append('시스템 폴더가 존재합니다.')
            print(err)

    def classifier_system_image(self, x):
        for root in dir_path:
            # 파일 존재 확인
            for file in os.listdir(root):
                PATH = f'{root}/{file}'
                if os.path.isfile(PATH):
                    if f'OPTICAL{x}' in file:
                        shutil.move(PATH, f'{root}/{x}')

        self.textBrowser.append(f'{x} 작업이 완료되었습니다.')

class ThreadClass(QtCore.QThread):
    def __init__(self, parent = None):
        super(ThreadClass, self).__init__(parent)
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
