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



class Ui_MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)

        self.setObjectName("MainWindow")
        self.resize(499, 313)
        self.setFixedSize(499, 313)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("Y:/_사내업무/회사양식/마스코리아 로고(저용량).png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget()
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
        self.setCentralWidget(self.centralwidget)
        # Directory Select Event
        self.pushButton.clicked.connect(self.pushButtonClicked)
        # Classifier Button Event
        self.pushButton_3.clicked.connect(self.test_sample)
        # Exit Button Event
        self.pushButton_2.clicked.connect(QCoreApplication.instance().quit)

        self.retranslateUi()
        self.threads = Thread1(self)
        self.threads.trigger.connect(self.update_text)
        # QtCore.QMetaObject.connectSlotsByName()

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "System Image Classifier"))
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

    def test_sample(self):
        system_folder = [201, 202, 203]
        self.textBrowser.append('시스템 이미지 분류를 시작합니다.')
        self.textBrowser.append('=' * 70)
        self.textBrowser.append(f'시스템 폴더를 생성합니다.')
        th = Thread1()
        th.run()
        th2 = Thread2()
        for folder in system_folder:
            try:
                self.textBrowser.append(f'{folder} 이미지 파일을 옮깁니다.')
                th2.run(folder)
            except:
                self.textBrowser.append(f'{folder} 이미지 파일이 존재하지 않습니다.')

            self.textBrowser.append(f'{folder} 작업이 완료되었습니다.')




    def update_text(self):
        # self.text_area.append('thread # %d finished'%thread_no)
        # print('thread # %d finished'%thread_no)

        self.text_area.append(err)


class Thread1(QThread):
    trigger = pyqtSignal(str)
    def __init__(self, parent = None):
        super(Thread1, self).__init__(parent)

    def run(self):
        global dir_path, err
        dir_path = []
        system_folder = [201, 202, 203]
        try:
            for root in dir_list:
                PATH = f'{directory}/{root}'
                if os.path.isdir(PATH):
                    dir_path.append(PATH)
                    for folder in system_folder:
                        if not os.path.isdir(f'{PATH}/{folder}'):
                            os.mkdir(f'{PATH}/{folder}')
                            # 상세 폴더 생성
                            if folder == 201:
                                os.mkdir(f'{PATH}/{folder}/dry_day')
                                os.mkdir(f'{PATH}/{folder}/dry_night')
                                os.mkdir(f'{PATH}/{folder}/sunrise')
                                os.mkdir(f'{PATH}/{folder}/sunset')
                                os.mkdir(f'{PATH}/{folder}/wet_day')
                                os.mkdir(f'{PATH}/{folder}/wet_night')
                                os.mkdir(f'{PATH}/{folder}/snowcover_day')
                                os.mkdir(f'{PATH}/{folder}/snowcover_night')
                                os.mkdir(f'{PATH}/{folder}/slush_day')
                                os.mkdir(f'{PATH}/{folder}/slush_night')
                                os.mkdir(f'{PATH}/{folder}/ing')
                                os.mkdir(f'{PATH}/{folder}/etc')
                            if folder == 202:
                                os.mkdir(f'{PATH}/{folder}/normal_day')
                                os.mkdir(f'{PATH}/{folder}/normal_night')
                                os.mkdir(f'{PATH}/{folder}/fog_day')
                                os.mkdir(f'{PATH}/{folder}/fog_night')
                                os.mkdir(f'{PATH}/{folder}/rainfall_day')
                                os.mkdir(f'{PATH}/{folder}/rainfall_night')
                                os.mkdir(f'{PATH}/{folder}/snow_day')
                                os.mkdir(f'{PATH}/{folder}/snow_night')
                                os.mkdir(f'{PATH}/{folder}/sunrise')
                                os.mkdir(f'{PATH}/{folder}/sunset')
                                os.mkdir(f'{PATH}/{folder}/etc')
                            # if folder == 203:
                            #     os.mkdir(f'{root}/{folder}/')
                        else:
                            self.trigger.emit('1')

        except Exception as err:
            self.trigger.emit(err)
            print(err)

        self.wait(2000)


class Thread2(QThread):
    trigger = pyqtSignal(str)
    def __init__(self, parent = None):
        super(Thread2, self).__init__(parent)

    def run(self, x):
        self.wait(2000)
        for root in dir_path:
            # 파일 존재 확인
            for file in os.listdir(root):
                PATH = f'{root}/{file}'
                if os.path.isfile(PATH):
                    if f'OPTICAL{x}' in file:
                        shutil.move(PATH, f'{root}/{x}')



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = Ui_MainWindow()
    MainWindow.show()
    sys.exit(app.exec_())
