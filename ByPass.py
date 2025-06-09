import logging
import os
import sys
import traceback
from os import makedirs
from time import time

import cv2
import numpy as np
import pygetwindow as gw
import tensorflow as tf
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QRect
from PyQt5.QtGui import QIcon, QImage, QPixmap, QPainter, QPen
from PyQt5.QtWidgets import QSystemTrayIcon, QMenu, QAction
from win10toast import ToastNotifier

toaster = ToastNotifier()

os.makedirs('./logs', exist_ok=True)
logging.basicConfig(filename='./logs/detection.log', level=logging.DEBUG)

# 配置 TensorFlow 环境
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# 全局变量
hide_list = ["崩坏：星穹铁道", "原神", "绝区零", "米哈游启动器", "鸣潮", "欢乐麻将"]

# 保存识别到的人脸图片的连续帧数
save_frame_count = 0


def switch_to_desktop():
    """
    切换到桌面
    """
    for _window in gw.getAllWindows():
        if any(keyword in _window.title for keyword in hide_list) and _window.visible:
            print(_window.title)
            _window.minimize()
    toaster.show_toast("C", "End", duration=1, threaded=True)


class FaceDetectionThread(QThread):
    """
    负责执行人脸检测的线程
    """
    update_frame_signal = pyqtSignal(np.ndarray)
    face_detected_signal = pyqtSignal(list)
    status_signal = pyqtSignal(bool)

    def __init__(self, model_path):
        super(FaceDetectionThread, self).__init__()
        self.status_signal.emit(False)  # 状态切换为停止
        self.model = tf.keras.models.load_model(model_path)
        self.cap = None  # 视频捕获器初始化为空
        self.detect = False  # 检测
        self.roi = None
        self.save_dir = "detected_frames"

        # 创建保存目录
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def set_roi(self, roi):
        """设置ROI区域，None表示全屏检测"""
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

            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                # 转换颜色空间从BGR到RGB（Qt使用RGB）
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.update_frame_signal.emit(frame_rgb)

                if self.detect:
                    roi_frame, y1, x1, y2, x2 = self._get_roi_frame(frame)
                    if roi_frame.size == 0:
                        continue
                    h, w, _ = roi_frame.shape
                    img = cv2.resize(roi_frame, (320, 240))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = (img - 127.0) / 128.0

                    # 进行人脸检测
                    results = self.model.predict(np.expand_dims(img, axis=0))

                    if self._has_face(results, w, h):
                        self.face_detected_signal.emit([x1, y1, x2, y2])
                        if save_frame_count > 0:
                            # 连续保存N帧
                            t = os.path.join(self.save_dir, f"{time():.0f}")
                            makedirs(t, exist_ok=True)
                            for i in range(save_frame_count):
                                filename = os.path.join(t, f"{i}.jpg")
                                cv2.imwrite(filename, roi_frame)

                                # 截取下一帧
                                ret, frame = self.cap.read()
                                if not ret:
                                    break
                                roi_frame, _, _, _, _ = self._get_roi_frame(frame)
                        # 停止检测
                        self.stop()
                        break
                # 如果不是检测模式，单纯刷新视频帧
                QtCore.QThread.msleep(30)
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            traceback.print_exc()  # 打印堆栈信息
            logging.error(f"Error occurred: {str(e)}")
            logging.error(traceback.format_exc())  # 记录堆栈信息
        finally:
            # 恢复标准输出
            sys.stdout = sys.__stdout__

        self.stop()

    def _get_roi_frame(self, frame):
        if self.roi:
            x1, y1, x2, y2 = self.roi
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            return frame[y1:y2, x1:x2], y1, x1, y2, x2
        else:
            h, w = frame.shape[:2]
            return frame, 0, 0, h, w

    def _has_face(self, results, w, h):
        for result in results:
            start_x = int(result[-4] * w)
            start_y = int(result[-3] * h)
            end_x = int(result[-2] * w)
            end_y = int(result[-1] * h)
            if start_x > 0 and start_y > 0 and end_x > 0 and end_y > 0:
                return True
        return False

    def stop(self):
        self.status_signal.emit(False)  # 状态切换为停止
        self.detect = False
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
        self.face_thread.face_detected_signal.connect(self.on_face_detected)
        self.face_thread.status_signal.connect(self.on_status_changed)

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("ByPass")
        self.setGeometry(0, 0, 650, 600)
        self.setFixedSize(650, 600)

        # ROI选择相关变量
        self.selecting_roi = False
        self.roi_start_point = None
        self.roi_end_point = None
        self.current_roi = None
        self.temp_frame = None

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

        self.video_label = ROISelectLabel(self)
        self.video_label.setGeometry(50, 0, 550, 480)
        self.video_label.roi_selected.connect(self.on_roi_selected)

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

        continue_action = QAction("继续", self)
        continue_action.triggered.connect(self.start_detection)
        tray_menu.addAction(continue_action)

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

    def on_roi_selected(self, roi):
        """
        接收 ROI 区域，并传递给检测线程
        """
        self.current_roi = roi
        if roi:
            x1, y1, x2, y2 = roi
            # 确保坐标顺序正确（左上角到右下角）
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            self.face_thread.set_roi((x1, y1, x2, y2))
            print(f"ROI 设置为: {(x1, y1, x2, y2)}")
            # 停止视频流，确保temp_frame为最后一帧
            self.face_thread.stop()
            QtWidgets.QApplication.processEvents()  # 确保线程停止
            # 保留最后一帧和框选的框的叠加图片，并显示到界面
            if self.temp_frame is not None:
                frame = self.temp_frame.copy()
                height, width, channel = frame.shape
                bytes_per_line = 3 * width
                qimg = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)
                painter = QPainter(pixmap)
                pen = QPen(Qt.green, 2, Qt.SolidLine)
                painter.setPen(pen)
                rect = QRect(x1, y1, x2 - x1, y2 - y1)
                painter.drawRect(rect)
                painter.end()
                self.video_label.setPixmap(pixmap)
                self.video_label.update()
            toaster.show_toast("C", "结束框选", duration=2, threaded=True)
        self.selecting_roi = False
        self.start_button.setEnabled(True)
        self.select_roi_button.setEnabled(True)

    def start_selecting_roi(self):
        self.selecting_roi = True
        self.video_label.start_roi_selection()
        if not self.face_thread.isRunning():
            self.face_thread.detect = False
            self.face_thread.set_roi(None)  # ROI为None，显示全屏
            self.face_thread.start()
        self.start_button.setEnabled(False)
        self.select_roi_button.setEnabled(False)

    def start_detection(self):
        if not self.current_roi:
            self.face_thread.set_roi(None)  # 没有ROI时全屏检测
        self.face_thread.detect = True
        if not self.face_thread.isRunning():
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
        self.temp_frame = frame.copy()
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        qimg = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        # 如果有ROI，实时叠加显示ROI框
        if self.current_roi:
            x1, y1, x2, y2 = self.current_roi
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            painter = QPainter(pixmap)
            pen = QPen(Qt.green, 2, Qt.SolidLine)
            painter.setPen(pen)
            rect = QRect(x1, y1, x2 - x1, y2 - y1)
            painter.drawRect(rect)
            painter.end()
        self.video_label.setPixmap(pixmap)
        self.video_label.update()  # 强制更新

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


class ROISelectLabel(QtWidgets.QLabel):
    """
    用于选择ROI的标签
    """
    roi_selected = pyqtSignal(tuple)

    def __init__(self, parent=None):
        super(ROISelectLabel, self).__init__(parent)
        self.selecting = False
        self.start_point = None
        self.end_point = None
        self.roi_selection_mode = False

    def start_roi_selection(self):
        self.roi_selection_mode = True
        self.setCursor(Qt.CrossCursor)  # 设置鼠标为十字光标

    def mousePressEvent(self, event):
        if not self.roi_selection_mode:
            return super(ROISelectLabel, self).mousePressEvent(event)

        if event.button() == Qt.LeftButton:
            self.selecting = True
            self.start_point = event.pos()
            self.end_point = self.start_point  # 初始化终点为起点
        return None

    def mouseMoveEvent(self, event):
        if not self.roi_selection_mode or not self.selecting:
            return super(ROISelectLabel, self).mouseMoveEvent(event)

        self.end_point = event.pos()
        self.update()  # 触发重绘

    def mouseReleaseEvent(self, event):
        if not self.roi_selection_mode:
            return super(ROISelectLabel, self).mouseReleaseEvent(event)

        if event.button() == Qt.LeftButton and self.selecting:
            self.selecting = False
            self.end_point = event.pos()
            self.update()  # 最后一次重绘

            # 发送ROI信号
            if self.start_point and self.end_point:
                roi = (self.start_point.x(), self.start_point.y(),
                       self.end_point.x(), self.end_point.y())
                self.roi_selection_mode = False
                self.setCursor(Qt.ArrowCursor)  # 恢复鼠标为箭头
                self.roi_selected.emit(roi)
        elif event.button() == Qt.RightButton:
            # 右键清空ROI并停止
            self.selecting = False
            self.start_point = None
            self.end_point = None
            self.roi_selection_mode = False
            self.setCursor(Qt.ArrowCursor)
            # 通知主窗口清空ROI
            if hasattr(self.parent(), 'on_roi_selected'):
                self.parent().on_roi_selected(None)
            self.update()
        return None

    def paintEvent(self, event):
        super(ROISelectLabel, self).paintEvent(event)

        # 如果正在选择ROI，绘制矩形
        if self.roi_selection_mode and self.start_point and self.end_point:
            painter = QPainter(self)
            pen = QPen(Qt.green, 2, Qt.SolidLine)
            painter.setPen(pen)

            # 计算矩形
            rect = QRect(self.start_point, self.end_point)
            painter.drawRect(rect)


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
