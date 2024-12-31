import os
import sys
from os import mkdir
from time import time

import cv2
import numpy as np
import pygetwindow as gw
import tensorflow as tf
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtWidgets import QSystemTrayIcon, QMenu, QAction
from plyer import notification
import logging
import traceback

os.makedirs('./logs', exist_ok=True)
logging.basicConfig(filename='./logs/detection.log', level=logging.DEBUG)

# 配置 TensorFlow 环境
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# 全局变量
roi = None  # 保存选择的ROI区域
hide_list = ["崩坏：星穹铁道", "原神", "绝区零", "米哈游启动器", "鸣潮", "欢乐麻将"]
selecting = False
start_point = None


def select_roi_range(event, x, y, flags, param):
    """
    鼠标回调函数，用于选择 ROI 区域
    """
    global roi, selecting, start_point

    if event == cv2.EVENT_LBUTTONDOWN:  # 鼠标左键按下
        selecting = True
        start_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and selecting:  # 鼠标拖动
        frame_copy = param.copy()
        cv2.rectangle(frame_copy, start_point, (x, y), (0, 255, 0), 2)
        cv2.imshow('Video Stream', frame_copy)

    elif event == cv2.EVENT_LBUTTONUP:  # 鼠标左键释放
        selecting = False
        end_point = (x, y)
        roi = (start_point[0], start_point[1], end_point[0], end_point[1])  # 记录选择的区域
        cv2.rectangle(param, start_point, end_point, (0, 255, 0), 2)
        cv2.imshow('Video Stream', param)


def switch_to_desktop():
    """
    切换到桌面
    """
    for _window in gw.getAllWindows():
        if any(keyword in _window.title for keyword in hide_list) and _window.visible:
            print(_window.title)
            _window.minimize()
    notification.notify(title="C", message="End", timeout=1)


class FaceDetectionThread(QThread):
    """
    负责执行人脸检测的线程
    """
    update_frame_signal = pyqtSignal(np.ndarray)
    set_roi_signal = pyqtSignal(tuple)
    face_detected_signal = pyqtSignal(list)
    status_signal = pyqtSignal(bool)

    def __init__(self, model_path):
        super(FaceDetectionThread, self).__init__()
        self.status_signal.emit(False)  # 状态切换为停止
        self.model = tf.keras.models.load_model(model_path)
        self.cap = None  # 视频捕获器初始化为空
        self.detect = False  # 检测
        self.select_roi = False  # 选择范围
        self.roi = None
        self.save_dir = "detected_frames"

        # 创建保存目录
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def set_roi(self, roi):
        self.roi = roi

    def run(self):
        try:
            # 临时重定向 stdout 到 devnull  防止pythonw运行时  predict 输出报错
            sys.stdout = open(os.devnull, 'w')
            self.status_signal.emit(True)  # 状态切换为运行
            # 初始化 VideoCapture
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                raise Exception("摄像头初始化失败")
            if self.select_roi:
                while True:
                    ret, frame = self.cap.read()
                    if not ret:
                        return
                    # 设置鼠标回调函数
                    cv2.imshow('Video Stream', frame)
                    cv2.setMouseCallback('Video Stream', select_roi_range, param=frame)

                    # 任意 F 键退出初始显示
                    if cv2.waitKey(1) & 0xFF == 0 or roi:
                        cv2.destroyAllWindows()
                        break
                # 通知框选结束
                self.set_roi_signal.emit(roi)
                # 更新一帧到界面
                self.update_frame_signal.emit(frame)
            else:
                while self.detect:
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    self.update_frame_signal.emit(frame)

                    # 如果定义了 ROI，裁剪图像
                    if self.roi:
                        x1, y1, x2, y2 = self.roi
                        frame = frame[y1:y2, x1:x2]

                    # 预处理图像
                    h, w, _ = frame.shape
                    img = cv2.resize(frame, (320, 240))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = (img - 127.0) / 128.0

                    # 进行人脸检测
                    results = self.model.predict(np.expand_dims(img, axis=0))
                    face_detected = False
                    for result in results:
                        start_x = int(result[-4] * w)
                        start_y = int(result[-3] * h)
                        end_x = int(result[-2] * w)
                        end_y = int(result[-1] * h)

                        if start_x > 0 and start_y > 0 and end_x > 0 and end_y > 0:
                            face_detected = True
                            break

                    if face_detected:
                        self.face_detected_signal.emit([start_x, start_y, end_x, end_y])
                        # 连续保存10帧  YYYY-MM-DD-HH-MM-SS
                        t = os.path.join(self.save_dir, f"{time():.0f}")
                        mkdir(t)
                        for i in range(0, 10):
                            filename = os.path.join(t, f"{i}.jpg")
                            cv2.imwrite(filename, frame)

                            # 截取下一帧
                            ret, frame = self.cap.read()
                            if not ret:
                                break

                        # 停止检测
                        self.stop()
                        break
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            traceback.print_exc()  # 打印堆栈信息
            logging.error(f"Error occurred: {str(e)}")
            logging.error(traceback.format_exc())  # 记录堆栈信息
        finally:
            # 恢复标准输出
            sys.stdout = sys.__stdout__

        self.stop()

    def stop(self):
        self.status_signal.emit(False)  # 状态切换为停止
        self.detect = False
        self.select_roi = False
        if self.cap:
            self.cap.release()
        self.quit()


class MainWindow(QtWidgets.QMainWindow):
    """
    主窗口
    """

    def init_detector(self):
        self.face_thread = FaceDetectionThread("export_models/slim/")
        self.face_thread.update_frame_signal.connect(self.update_frame)
        self.face_thread.set_roi_signal.connect(self.set_roi)
        self.face_thread.face_detected_signal.connect(self.on_face_detected)
        self.face_thread.status_signal.connect(self.on_status_changed)

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("ByPass")
        self.setGeometry(0, 0, 650, 600)
        self.setFixedSize(650, 600)

        # 界面元素
        self.start_button = QtWidgets.QPushButton("Start", self)
        self.start_button.setGeometry(50, 500, 100, 40)
        self.start_button.clicked.connect(self.start_detection)

        self.stop_button = QtWidgets.QPushButton("Stop", self)
        self.stop_button.setGeometry(275, 500, 100, 40)
        self.stop_button.clicked.connect(self.stop_detection)
        self.stop_button.setEnabled(False)

        self.select_roi_button = QtWidgets.QPushButton("Select ROI", self)
        self.select_roi_button.setGeometry(500, 500, 100, 40)
        self.select_roi_button.clicked.connect(self.start_selecting_roi)

        self.video_label = QtWidgets.QLabel(self)
        self.video_label.setGeometry(50, 0, 550, 480)
        # self.video_label.setStyleSheet("background-color: white;")

        # 初始化检测线程
        self.init_detector()

        # 初始化托盘图标
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(QIcon("pic/normal.png"))
        self.tray_icon.activated.connect(self.on_tray_icon_activated)

        # 托盘菜单
        tray_menu = QMenu(self)
        show_action = QAction("显示", self)
        show_action.triggered.connect(self.show_and_activate)
        tray_menu.addAction(show_action)

        quit_action = QAction("退出", self)
        quit_action.triggered.connect(self.close)
        tray_menu.addAction(quit_action)
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()

    def show_and_activate(self):
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def changeEvent(self, event):
        if event.type() == QtCore.QEvent.WindowStateChange:
            if self.windowState() & QtCore.Qt.WindowMinimized:
                QtCore.QTimer.singleShot(0, self.hide)
        super(MainWindow, self).changeEvent(event)

    def on_tray_icon_activated(self, reason):
        """托盘图标点击事件"""
        if reason == QSystemTrayIcon.Trigger:
            self.show_and_activate()

    def set_roi(self, _roi):
        """
        接收 ROI 区域，并传递给检测线程
        """
        if _roi:
            self.face_thread.set_roi(_roi)
            print(f"ROI 设置为: {_roi}")
            notification.notify(title="C", message="结束框选", timeout=2)
        self.start_button.setEnabled(True)
        self.select_roi_button.setEnabled(True)

    def start_selecting_roi(self):
        global roi
        self.face_thread.roi = None
        roi = None
        self.face_thread.select_roi = True
        self.face_thread.start()
        self.start_button.setEnabled(False)
        self.select_roi_button.setEnabled(False)

    def start_detection(self):
        self.face_thread.detect = True
        self.face_thread.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.select_roi_button.setEnabled(False)
        self.hide()

    def stop_detection(self):
        self.face_thread.detect = False
        self.stop_button.setEnabled(False)
        self.start_button.setEnabled(True)
        self.select_roi_button.setEnabled(True)

    def update_frame(self, frame):
        """更新视频帧"""
        if frame is None or not self.isVisible():
            return
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        qimg = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        # 画上roi
        if self.face_thread.roi:
            x1, y1, x2, y2 = self.face_thread.roi
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        pixmap = QPixmap(qimg)
        self.video_label.setPixmap(pixmap)

    def on_face_detected(self, face_box):
        """人脸检测到时的回调"""
        print("Face detected:", face_box)
        self.stop_detection()
        switch_to_desktop()

    def on_status_changed(self, running):
        icon_path = "pic/running.png" if running else "pic/pause.png"
        self.tray_icon.setIcon(QIcon(icon_path))

    def closeEvent(self, event):
        """关闭窗口时释放资源"""
        self.face_thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
