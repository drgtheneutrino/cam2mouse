"""
Alea-AirCursor v2.0.0 - Enhanced Edition
=========================================
Features:
- Two-hand tracking with independent gesture recognition
- 3D model interaction support (rotate, scale, translate)
- Superimposed camera overlay (picture-in-picture)
- Kalman filtering for smooth, accurate cursor movement
- Optimized performance with multi-threaded processing
"""

import sys
import os
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal, QPropertyAnimation, QEasingCurve
from PyQt5.QtWidgets import QGraphicsDropShadowEffect
from PyQt5.QtOpenGL import QGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *
from gesture import HandTracker
from tray import SystemTray
from model_viewer import ModelViewer
import platform
import subprocess

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


class FrameProcessor(QThread):
    """Separate thread for frame processing to improve performance"""
    frame_ready = pyqtSignal(np.ndarray, dict, dict)
    
    def __init__(self, hand_tracker):
        super().__init__()
        self.hand_tracker = hand_tracker
        self.running = False
        self.frame = None
        self.frame_lock = QtCore.QMutex()
        
    def set_frame(self, frame):
        self.frame_lock.lock()
        self.frame = frame.copy()
        self.frame_lock.unlock()
        
    def run(self):
        self.running = True
        while self.running:
            self.frame_lock.lock()
            if self.frame is not None:
                frame = self.frame.copy()
                self.frame = None
                self.frame_lock.unlock()
                
                processed_frame, left_hand_data, right_hand_data = self.hand_tracker.process_frame(frame)
                self.frame_ready.emit(processed_frame, left_hand_data, right_hand_data)
            else:
                self.frame_lock.unlock()
            self.msleep(1)
            
    def stop(self):
        self.running = False
        self.wait()


class CameraOverlay(QtWidgets.QWidget):
    """Floating camera overlay widget (picture-in-picture)"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        self.overlay_size = (320, 240)
        self.setFixedSize(self.overlay_size[0] + 20, self.overlay_size[1] + 40)
        
        # Main layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Container with rounded corners
        self.container = QtWidgets.QFrame()
        self.container.setStyleSheet("""
            QFrame {
                background-color: rgba(20, 20, 30, 230);
                border-radius: 12px;
                border: 2px solid rgba(100, 200, 255, 0.5);
            }
        """)
        container_layout = QtWidgets.QVBoxLayout(self.container)
        container_layout.setContentsMargins(8, 8, 8, 8)
        container_layout.setSpacing(4)
        
        # Title bar
        title_bar = QtWidgets.QHBoxLayout()
        self.title_label = QtWidgets.QLabel("📹 Camera Feed")
        self.title_label.setStyleSheet("color: #64C8FF; font-weight: bold; font-size: 11px;")
        title_bar.addWidget(self.title_label)
        title_bar.addStretch()
        
        # Close button
        self.close_btn = QtWidgets.QPushButton("×")
        self.close_btn.setFixedSize(20, 20)
        self.close_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 100, 100, 0.7);
                border: none;
                border-radius: 10px;
                color: white;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: rgba(255, 100, 100, 1);
            }
        """)
        self.close_btn.clicked.connect(self.hide)
        title_bar.addWidget(self.close_btn)
        container_layout.addLayout(title_bar)
        
        # Camera feed label
        self.feed_label = QtWidgets.QLabel()
        self.feed_label.setFixedSize(self.overlay_size[0], self.overlay_size[1])
        self.feed_label.setStyleSheet("border-radius: 8px; background-color: #000;")
        self.feed_label.setAlignment(Qt.AlignCenter)
        container_layout.addWidget(self.feed_label)
        
        layout.addWidget(self.container)
        
        # Shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QtGui.QColor(0, 100, 200, 100))
        shadow.setOffset(0, 5)
        self.container.setGraphicsEffect(shadow)
        
        # Dragging support
        self.dragging = False
        self.drag_position = None
        
    def update_frame(self, frame):
        """Update the overlay with new frame"""
        if frame is not None:
            frame_small = cv2.resize(frame, self.overlay_size)
            rgb_image = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            self.feed_label.setPixmap(QtGui.QPixmap.fromImage(qt_image))
            
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            
    def mouseMoveEvent(self, event):
        if self.dragging and event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self.drag_position)
            
    def mouseReleaseEvent(self, event):
        self.dragging = False


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AirCursor 2.0 - Enhanced Edition")
        self.setMinimumSize(900, 750)
        
        icon_path = resource_path("assets/icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QtGui.QIcon(icon_path))
        
        # Apply dark theme
        self.apply_theme()
        
        # Initialize components
        self.hand_tracker = HandTracker(
            cooldown=0.3,
            complexity=1,
            click_radius=25,
            hover_radius=35,
            right_click_radius=45,
            right_hover_radius=55,
            hold_click_radius=35,
            hold_hover_radius=45,
            scroll_radius=30,
            scroll_hover_radius=40,
            scroll_speed=80,
            enable_two_hands=True,
            enable_3d_mode=True
        )
        
        self.capture = None
        self.tracking_active = False
        self.show_main_camera = True
        self.show_overlay = True
        
        # Frame processor thread
        self.frame_processor = None
        
        # Camera overlay
        self.camera_overlay = CameraOverlay()
        
        # 3D Model viewer
        self.model_viewer = None
        self.model_viewer_enabled = False
        
        # Setup UI
        self.setup_ui()
        
        # Setup system tray
        self.tray = SystemTray(self)
        if os.path.exists(icon_path):
            self.tray.setIcon(QtGui.QIcon(icon_path))
        
        # Setup timer for camera capture
        self.capture_timer = QTimer(self)
        self.capture_timer.timeout.connect(self.capture_frame)
        
        # Camera detection
        self.available_cameras = self.detect_available_cameras()
        self.selected_camera_index = 0
        self.populate_camera_combo()
        
        # Position overlay
        self.position_overlay()
        
    def apply_theme(self):
        """Apply modern dark theme"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0D1117;
            }
            QWidget {
                font-family: 'Segoe UI', 'SF Pro Display', sans-serif;
                font-size: 13px;
                color: #E6EDF3;
            }
            QPushButton {
                background-color: #21262D;
                border: 1px solid #30363D;
                border-radius: 8px;
                padding: 10px 20px;
                color: #E6EDF3;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #30363D;
                border-color: #58A6FF;
            }
            QPushButton:pressed {
                background-color: #1F6FEB;
            }
            QPushButton:disabled {
                background-color: #161B22;
                color: #484F58;
                border-color: #21262D;
            }
            QPushButton#startButton {
                background-color: #238636;
                border-color: #2EA043;
            }
            QPushButton#startButton:hover {
                background-color: #2EA043;
            }
            QPushButton#stopButton {
                background-color: #DA3633;
                border-color: #F85149;
            }
            QPushButton#stopButton:hover {
                background-color: #F85149;
            }
            QComboBox {
                background-color: #21262D;
                border: 1px solid #30363D;
                border-radius: 8px;
                padding: 8px 15px;
                color: #E6EDF3;
            }
            QComboBox:hover {
                border-color: #58A6FF;
            }
            QComboBox::drop-down {
                border: none;
                padding-right: 10px;
            }
            QComboBox QAbstractItemView {
                background-color: #21262D;
                border: 1px solid #30363D;
                selection-background-color: #1F6FEB;
            }
            QLabel#cameraFeedLabel {
                background-color: #161B22;
                border: 2px solid #30363D;
                border-radius: 12px;
            }
            QStatusBar {
                background-color: #161B22;
                border-top: 1px solid #21262D;
                color: #8B949E;
            }
            QMenuBar {
                background-color: #161B22;
                border-bottom: 1px solid #21262D;
            }
            QMenuBar::item {
                padding: 8px 15px;
            }
            QMenuBar::item:selected {
                background-color: #21262D;
            }
            QMenu {
                background-color: #21262D;
                border: 1px solid #30363D;
            }
            QMenu::item:selected {
                background-color: #1F6FEB;
            }
            QGroupBox {
                border: 1px solid #30363D;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 5px;
            }
            QCheckBox {
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 1px solid #30363D;
                background-color: #21262D;
            }
            QCheckBox::indicator:checked {
                background-color: #1F6FEB;
                border-color: #58A6FF;
            }
        """)
        
    def setup_ui(self):
        """Setup the main UI"""
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Header
        header_layout = QtWidgets.QHBoxLayout()
        title_label = QtWidgets.QLabel("🖐️ AirCursor 2.0")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #58A6FF;")
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        # Status indicator
        self.status_indicator = QtWidgets.QLabel("● Inactive")
        self.status_indicator.setStyleSheet("color: #8B949E; font-size: 14px;")
        header_layout.addWidget(self.status_indicator)
        main_layout.addLayout(header_layout)
        
        # Control buttons
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.setSpacing(10)
        
        self.startButton = QtWidgets.QPushButton("▶ Start Tracking")
        self.startButton.setObjectName("startButton")
        self.startButton.setMinimumHeight(45)
        self.startButton.clicked.connect(self.start_tracking)
        button_layout.addWidget(self.startButton)
        
        self.stopButton = QtWidgets.QPushButton("■ Stop")
        self.stopButton.setObjectName("stopButton")
        self.stopButton.setMinimumHeight(45)
        self.stopButton.setEnabled(False)
        self.stopButton.clicked.connect(self.stop_tracking)
        button_layout.addWidget(self.stopButton)
        
        self.cameraButton = QtWidgets.QPushButton("📷 Hide Camera")
        self.cameraButton.setMinimumHeight(45)
        self.cameraButton.clicked.connect(self.toggle_camera_display)
        button_layout.addWidget(self.cameraButton)
        
        self.overlayButton = QtWidgets.QPushButton("🖼️ Toggle Overlay")
        self.overlayButton.setMinimumHeight(45)
        self.overlayButton.clicked.connect(self.toggle_overlay)
        button_layout.addWidget(self.overlayButton)
        
        self.minimizeButton = QtWidgets.QPushButton("➖ Minimize")
        self.minimizeButton.setMinimumHeight(45)
        self.minimizeButton.clicked.connect(self.minimize_to_tray)
        button_layout.addWidget(self.minimizeButton)
        
        main_layout.addLayout(button_layout)
        
        # Camera selection and options
        options_layout = QtWidgets.QHBoxLayout()
        options_layout.setSpacing(10)
        
        # Camera combo
        camera_group = QtWidgets.QWidget()
        camera_layout = QtWidgets.QHBoxLayout(camera_group)
        camera_layout.setContentsMargins(0, 0, 0, 0)
        
        camera_label = QtWidgets.QLabel("Camera:")
        camera_label.setStyleSheet("color: #8B949E;")
        camera_layout.addWidget(camera_label)
        
        self.cameraComboBox = QtWidgets.QComboBox()
        self.cameraComboBox.setMinimumWidth(300)
        camera_layout.addWidget(self.cameraComboBox)
        
        self.selectCameraButton = QtWidgets.QPushButton("Select")
        self.selectCameraButton.clicked.connect(self.select_camera)
        camera_layout.addWidget(self.selectCameraButton)
        
        options_layout.addWidget(camera_group)
        options_layout.addStretch()
        
        # 3D Mode toggle
        self.enable3DCheck = QtWidgets.QCheckBox("Enable 3D Mode")
        self.enable3DCheck.setChecked(False)
        self.enable3DCheck.stateChanged.connect(self.toggle_3d_mode)
        options_layout.addWidget(self.enable3DCheck)
        
        # Two-hand mode toggle
        self.twoHandCheck = QtWidgets.QCheckBox("Two-Hand Mode")
        self.twoHandCheck.setChecked(True)
        self.twoHandCheck.stateChanged.connect(self.toggle_two_hand_mode)
        options_layout.addWidget(self.twoHandCheck)
        
        main_layout.addLayout(options_layout)
        
        # Camera feed area
        feed_container = QtWidgets.QWidget()
        feed_layout = QtWidgets.QHBoxLayout(feed_container)
        feed_layout.setContentsMargins(0, 0, 0, 0)
        feed_layout.setSpacing(15)
        
        # Main camera feed
        self.cameraFeedLabel = QtWidgets.QLabel()
        self.cameraFeedLabel.setObjectName("cameraFeedLabel")
        self.cameraFeedLabel.setMinimumSize(640, 480)
        self.cameraFeedLabel.setAlignment(Qt.AlignCenter)
        self.cameraFeedLabel.setText("📷 Camera Ready\nClick 'Start Tracking' to begin")
        self.cameraFeedLabel.setStyleSheet("""
            QLabel {
                background-color: #161B22;
                border: 2px solid #30363D;
                border-radius: 12px;
                font-size: 16px;
                color: #8B949E;
            }
        """)
        feed_layout.addWidget(self.cameraFeedLabel, stretch=2)
        
        # Info panel
        info_panel = QtWidgets.QWidget()
        info_panel.setMaximumWidth(250)
        info_layout = QtWidgets.QVBoxLayout(info_panel)
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(10)
        
        # Gesture guide
        gesture_group = QtWidgets.QGroupBox("Gesture Guide")
        gesture_layout = QtWidgets.QVBoxLayout(gesture_group)
        
        gestures = [
            ("👆 Index Finger", "Move Cursor"),
            ("👌 Pinch (Thumb+Index)", "Left Click"),
            ("🤏 Hold Pinch", "Click & Drag"),
            ("🤙 Pinky to Wrist", "Right Click"),
            ("✌️ Two Fingers", "Scroll Up/Down"),
            ("🖐️ Open Palm", "Stop Action"),
        ]
        
        for icon_gesture, action in gestures:
            row = QtWidgets.QHBoxLayout()
            gesture_label = QtWidgets.QLabel(icon_gesture)
            gesture_label.setStyleSheet("font-size: 12px;")
            action_label = QtWidgets.QLabel(action)
            action_label.setStyleSheet("color: #8B949E; font-size: 11px;")
            row.addWidget(gesture_label)
            row.addStretch()
            row.addWidget(action_label)
            gesture_layout.addLayout(row)
        
        info_layout.addWidget(gesture_group)
        
        # Hand status
        status_group = QtWidgets.QGroupBox("Hand Status")
        status_layout = QtWidgets.QVBoxLayout(status_group)
        
        self.leftHandStatus = QtWidgets.QLabel("Left Hand: Not detected")
        self.leftHandStatus.setStyleSheet("color: #8B949E;")
        status_layout.addWidget(self.leftHandStatus)
        
        self.rightHandStatus = QtWidgets.QLabel("Right Hand: Not detected")
        self.rightHandStatus.setStyleSheet("color: #8B949E;")
        status_layout.addWidget(self.rightHandStatus)
        
        self.fpsLabel = QtWidgets.QLabel("FPS: --")
        self.fpsLabel.setStyleSheet("color: #58A6FF;")
        status_layout.addWidget(self.fpsLabel)
        
        info_layout.addWidget(status_group)
        
        # 3D Controls (initially hidden)
        self.controls3DGroup = QtWidgets.QGroupBox("3D Controls")
        controls3D_layout = QtWidgets.QVBoxLayout(self.controls3DGroup)
        
        controls_3d = [
            ("🔄 Left Hand", "Rotate Model"),
            ("📏 Two Hands Apart", "Scale Model"),
            ("✋ Right Palm", "Pan/Translate"),
        ]
        
        for icon_gesture, action in controls_3d:
            row = QtWidgets.QHBoxLayout()
            gesture_label = QtWidgets.QLabel(icon_gesture)
            gesture_label.setStyleSheet("font-size: 12px;")
            action_label = QtWidgets.QLabel(action)
            action_label.setStyleSheet("color: #8B949E; font-size: 11px;")
            row.addWidget(gesture_label)
            row.addStretch()
            row.addWidget(action_label)
            controls3D_layout.addLayout(row)
        
        self.controls3DGroup.hide()
        info_layout.addWidget(self.controls3DGroup)
        
        info_layout.addStretch()
        feed_layout.addWidget(info_panel)
        
        main_layout.addWidget(feed_container, stretch=1)
        
        # Menu bar
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu("File")
        self.actionLoad3DModel = QtWidgets.QAction("Load 3D Model...", self)
        self.actionLoad3DModel.triggered.connect(self.load_3d_model)
        file_menu.addAction(self.actionLoad3DModel)
        file_menu.addSeparator()
        exit_action = QtWidgets.QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        view_menu = menubar.addMenu("View")
        self.actionShowOverlay = QtWidgets.QAction("Show Camera Overlay", self)
        self.actionShowOverlay.setCheckable(True)
        self.actionShowOverlay.setChecked(True)
        self.actionShowOverlay.triggered.connect(self.toggle_overlay)
        view_menu.addAction(self.actionShowOverlay)
        
        help_menu = menubar.addMenu("Help")
        about_action = QtWidgets.QAction("About", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)
        
        # Status bar
        self.statusbar = QtWidgets.QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("Ready")
        
    def detect_available_cameras(self):
        """Detect available cameras with descriptive names"""
        available = []
        os_name = platform.system()
        
        if os_name == "Windows":
            try:
                from pygrabber.dshow_graph import FilterGraph
                graph = FilterGraph()
                devices = graph.get_input_devices()
                
                for index, name in enumerate(devices):
                    cap = cv2.VideoCapture(index)
                    if cap.isOpened():
                        available.append((index, name))
                        cap.release()
            except ImportError:
                for i in range(10):
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        available.append((i, f"Camera {i+1}"))
                        cap.release()
            except Exception as e:
                print(f"Error in Windows camera detection: {e}")
                for i in range(10):
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        available.append((i, f"Camera {i+1}"))
                        cap.release()
        
        elif os_name == "Linux":
            try:
                result = subprocess.run(
                    ['v4l2-ctl', '--list-devices'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                if result.returncode == 0:
                    output = result.stdout
                    devices = output.strip().split('\n\n')
                    
                    for device_block in devices:
                        if not device_block:
                            continue
                            
                        lines = device_block.split('\n')
                        camera_name = lines[0].split('(')[0].strip()
                        
                        for line in lines[1:]:
                            if '/dev/video' in line:
                                dev_path = line.strip().split()[0]
                                index = int(dev_path.split('/dev/video')[-1])
                                
                                cap = cv2.VideoCapture(index)
                                if cap.isOpened():
                                    available.append((index, camera_name))
                                    cap.release()
                else:
                    raise Exception("v4l2-ctl command failed")
            except Exception as e:
                print(f"Error in Linux camera detection: {e}")
                for i in range(10):
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        available.append((i, f"Camera {i+1}"))
                        cap.release()
        
        if not available:
            for i in range(10):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    available.append((i, f"Camera {i+1}"))
                    cap.release()
        
        return available

    def populate_camera_combo(self):
        """Populate the camera combo box"""
        self.cameraComboBox.clear()
        
        if not self.available_cameras:
            self.cameraComboBox.addItem("No cameras detected", -1)
            self.selectCameraButton.setEnabled(False)
            return
        
        for idx, name in self.available_cameras:
            self.cameraComboBox.addItem(f"{name} (Device {idx})", idx)
        
        if self.available_cameras:
            self.cameraComboBox.setCurrentIndex(0)
            self.selected_camera_index = self.available_cameras[0][0]

    def select_camera(self):
        """Handle camera selection"""
        selected_data = self.cameraComboBox.currentData()
        
        if selected_data == -1:
            QtWidgets.QMessageBox.warning(self, "No Camera", "No cameras available")
            return
        
        self.selected_camera_index = selected_data
        self.statusbar.showMessage(f"Selected: {self.cameraComboBox.currentText()}")

    def position_overlay(self):
        """Position the camera overlay in the bottom-right corner"""
        screen = QtWidgets.QApplication.primaryScreen().geometry()
        overlay_x = screen.width() - self.camera_overlay.width() - 20
        overlay_y = screen.height() - self.camera_overlay.height() - 80
        self.camera_overlay.move(overlay_x, overlay_y)

    def start_tracking(self):
        """Start hand tracking"""
        if not self.tracking_active:
            self.cameraComboBox.setEnabled(False)
            self.selectCameraButton.setEnabled(False)
            
            self.capture = cv2.VideoCapture(self.selected_camera_index)
            if not self.capture.isOpened():
                self.cameraComboBox.setEnabled(True)
                self.selectCameraButton.setEnabled(True)
                QtWidgets.QMessageBox.critical(
                    self, "Error",
                    "Could not open camera. Please check your camera connection."
                )
                return
            
            # Set camera properties for better performance
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.capture.set(cv2.CAP_PROP_FPS, 60)
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Start frame processor thread
            self.frame_processor = FrameProcessor(self.hand_tracker)
            self.frame_processor.frame_ready.connect(self.on_frame_processed)
            self.frame_processor.start()
            
            self.tracking_active = True
            self.capture_timer.start(16)  # ~60 FPS capture
            
            self.startButton.setEnabled(False)
            self.stopButton.setEnabled(True)
            self.status_indicator.setText("● Active")
            self.status_indicator.setStyleSheet("color: #3FB950; font-size: 14px;")
            self.statusbar.showMessage("Tracking active...")
            
            if self.show_overlay:
                self.camera_overlay.show()

    def stop_tracking(self):
        """Stop hand tracking"""
        if self.tracking_active:
            self.capture_timer.stop()
            self.tracking_active = False
            
            if self.frame_processor:
                self.frame_processor.stop()
                self.frame_processor = None
            
            if self.capture:
                self.capture.release()
            self.capture = None
            
            self.startButton.setEnabled(True)
            self.stopButton.setEnabled(False)
            self.status_indicator.setText("● Inactive")
            self.status_indicator.setStyleSheet("color: #8B949E; font-size: 14px;")
            self.statusbar.showMessage("Tracking stopped")
            
            self.cameraFeedLabel.setText("📷 Camera Ready\nClick 'Start Tracking' to begin")
            self.camera_overlay.hide()
            
            self.cameraComboBox.setEnabled(True)
            self.selectCameraButton.setEnabled(True)
            
            self.leftHandStatus.setText("Left Hand: Not detected")
            self.leftHandStatus.setStyleSheet("color: #8B949E;")
            self.rightHandStatus.setText("Right Hand: Not detected")
            self.rightHandStatus.setStyleSheet("color: #8B949E;")

    def capture_frame(self):
        """Capture frame from camera"""
        if not self.capture or not self.capture.isOpened():
            return
            
        ret, frame = self.capture.read()
        if ret and self.frame_processor:
            self.frame_processor.set_frame(frame)

    def on_frame_processed(self, processed_frame, left_hand_data, right_hand_data):
        """Handle processed frame from worker thread"""
        # Update main camera feed
        if self.show_main_camera:
            rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qt_image)
            self.cameraFeedLabel.setPixmap(
                pixmap.scaled(
                    self.cameraFeedLabel.width(),
                    self.cameraFeedLabel.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )
        
        # Update overlay
        if self.show_overlay and self.camera_overlay.isVisible():
            self.camera_overlay.update_frame(processed_frame)
        
        # Update hand status
        if left_hand_data.get('detected', False):
            gesture = left_hand_data.get('gesture', 'None')
            self.leftHandStatus.setText(f"Left Hand: {gesture}")
            self.leftHandStatus.setStyleSheet("color: #3FB950;")
        else:
            self.leftHandStatus.setText("Left Hand: Not detected")
            self.leftHandStatus.setStyleSheet("color: #8B949E;")
            
        if right_hand_data.get('detected', False):
            gesture = right_hand_data.get('gesture', 'None')
            self.rightHandStatus.setText(f"Right Hand: {gesture}")
            self.rightHandStatus.setStyleSheet("color: #58A6FF;")
        else:
            self.rightHandStatus.setText("Right Hand: Not detected")
            self.rightHandStatus.setStyleSheet("color: #8B949E;")
        
        # Update FPS
        fps = left_hand_data.get('fps', 0) or right_hand_data.get('fps', 0)
        if fps > 0:
            self.fpsLabel.setText(f"FPS: {fps:.1f}")
        
        # Handle 3D model interaction
        if self.model_viewer_enabled and self.model_viewer:
            self.handle_3d_interaction(left_hand_data, right_hand_data)

    def handle_3d_interaction(self, left_hand_data, right_hand_data):
        """Handle 3D model manipulation based on hand gestures"""
        if not self.model_viewer:
            return
            
        # Rotation with left hand
        if left_hand_data.get('detected') and left_hand_data.get('gesture') == 'rotating':
            delta_x = left_hand_data.get('delta_x', 0)
            delta_y = left_hand_data.get('delta_y', 0)
            self.model_viewer.rotate(delta_x * 2, delta_y * 2)
        
        # Scaling with two hands
        if left_hand_data.get('detected') and right_hand_data.get('detected'):
            if left_hand_data.get('gesture') == 'scaling' or right_hand_data.get('gesture') == 'scaling':
                scale_factor = left_hand_data.get('scale_factor', 1.0)
                self.model_viewer.scale(scale_factor)
        
        # Translation with right hand palm
        if right_hand_data.get('detected') and right_hand_data.get('gesture') == 'translating':
            trans_x = right_hand_data.get('trans_x', 0)
            trans_y = right_hand_data.get('trans_y', 0)
            self.model_viewer.translate(trans_x * 0.01, trans_y * 0.01)

    def toggle_camera_display(self):
        """Toggle main camera feed visibility"""
        self.show_main_camera = not self.show_main_camera
        if not self.show_main_camera:
            self.cameraButton.setText("📷 Show Camera")
            if self.tracking_active:
                self.cameraFeedLabel.setText("📷 Camera feed hidden")
        else:
            self.cameraButton.setText("📷 Hide Camera")
            if not self.tracking_active:
                self.cameraFeedLabel.setText("📷 Camera Ready\nClick 'Start Tracking' to begin")

    def toggle_overlay(self):
        """Toggle camera overlay visibility"""
        self.show_overlay = not self.show_overlay
        self.actionShowOverlay.setChecked(self.show_overlay)
        
        if self.tracking_active:
            if self.show_overlay:
                self.camera_overlay.show()
            else:
                self.camera_overlay.hide()

    def toggle_3d_mode(self, state):
        """Toggle 3D mode"""
        self.model_viewer_enabled = state == Qt.Checked
        self.hand_tracker.enable_3d_mode = self.model_viewer_enabled
        
        if self.model_viewer_enabled:
            self.controls3DGroup.show()
            if not self.model_viewer:
                self.model_viewer = ModelViewer()
        else:
            self.controls3DGroup.hide()
            if self.model_viewer:
                self.model_viewer.hide()

    def toggle_two_hand_mode(self, state):
        """Toggle two-hand mode"""
        self.hand_tracker.enable_two_hands = state == Qt.Checked

    def load_3d_model(self):
        """Load a 3D model file"""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load 3D Model",
            "",
            "3D Models (*.obj *.stl *.ply);;All Files (*)"
        )
        
        if file_path:
            if not self.model_viewer:
                self.model_viewer = ModelViewer()
            self.model_viewer.load_model(file_path)
            self.model_viewer.show()
            self.model_viewer_enabled = True
            self.enable3DCheck.setChecked(True)

    def minimize_to_tray(self):
        """Minimize to system tray"""
        self.hide()
        self.tray.show()
        icon_path = resource_path("assets/icon.png")
        if os.path.exists(icon_path):
            self.tray.showMessage(
                "AirCursor 2.0",
                "Application minimized to tray. Click to restore.",
                QtGui.QIcon(icon_path),
                2000
            )

    def show_about_dialog(self):
        """Show about dialog"""
        about_text = """
        <center>
            <h2>🖐️ AirCursor 2.0</h2>
            <h4>Enhanced Edition</h4>
            <hr>
            <p><b>Features:</b></p>
            <p>✨ Two-hand tracking support</p>
            <p>🎮 3D model interaction</p>
            <p>🎯 Kalman filtering for accuracy</p>
            <p>⚡ Multi-threaded processing</p>
            <p>🖼️ Picture-in-picture overlay</p>
            <hr>
            <p><i>Transform your gestures into seamless control</i></p>
            <p style="color: #8B949E; font-size: 11px;">Based on Alea-AirCursor by Alea Farrel</p>
        </center>
        """
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setWindowTitle("About AirCursor 2.0")
        msg_box.setTextFormat(QtCore.Qt.RichText)
        msg_box.setText(about_text)
        msg_box.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg_box.exec_()

    def closeEvent(self, event):
        """Handle close event"""
        self.stop_tracking()
        self.hand_tracker.release()
        self.tray.hide()
        self.camera_overlay.close()
        if self.model_viewer:
            self.model_viewer.close()
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
