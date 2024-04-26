import numpy as np
import cv2 as cv
import time
import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QApplication, QSizePolicy
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor
from PyQt5.QtCore import Qt, QTimer

HEIGHT = 874
WIDTH = 1164
PP = np.array([WIDTH / 2., HEIGHT / 2.], dtype=np.float32)

FOCAL_LENGTH = 910

MAX_YAW = 0.10
MAX_PITCH = 0.05

CAMERA_MAT = np.array([
    [FOCAL_LENGTH, 0, PP[0]],
    [0, FOCAL_LENGTH, PP[1]],
    [0, 0, 1]],
    dtype=np.float32)


def slam(p0, p1):
    E, mask = cv.findEssentialMat(
        points1=p0,
        points2=p1,
        cameraMatrix=CAMERA_MAT,
        method=cv.LMEDS,
        prob=0.99,
        mask=None,
    )

    _, _, t, _, _ = cv.recoverPose(
        E=E,
        points1=p0,
        points2=p1,
        cameraMatrix=CAMERA_MAT,
        distanceThresh=1e5,
        mask=mask
    )
    return t


def get_center_of_motion(t):
    angles = np.arccos(t)
    yaw = -(angles[2, 0] - np.pi) - 0.015
    pitch = -(angles[1, 0] - np.pi / 2) + 0.001
    return np.clip(np.array([yaw, pitch]), a_min=(-MAX_YAW, -MAX_PITCH), a_max=(MAX_YAW, MAX_PITCH))


def get_matches_lk(old_frame, frame, webcam=False):
    # Get the frame dimensions
    height, width = frame.shape[:2]

    # Define the ROI coordinates
    roi_left = 200
    roi_right = width - 150
    roi_top = 200
    roi_bottom = height - 350

    # Create a mask for the ROI
    roi_mask = np.zeros_like(old_frame[:, :, 0], dtype=np.uint8)
    roi_mask[roi_top:roi_bottom, roi_left:roi_right] = 255

    p0 = cv.goodFeaturesToTrack(
        cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY),
        maxCorners=1000,
        qualityLevel=0.001,
        minDistance=20,
        blockSize=5,
        mask=None if webcam else roi_mask
    )

    p1, status, _ = cv.calcOpticalFlowPyrLK(
        prevImg=old_frame,
        nextImg=frame,
        prevPts=p0,
        nextPts=None,
        winSize=(30, 30),
        maxLevel=10,
        criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    # Create a copy of the frame to draw optical flow vectors
    flow_frame = frame.copy()

    # Draw optical flow vectors on the frame
    for (x1, y1), (x2, y2) in zip(p0[status == 1].reshape(-1, 2), p1[status == 1].reshape(-1, 2)):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv.line(flow_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.circle(flow_frame, (x2, y2), 3, (0, 255, 0), -1)

    return p0[status == 1], p1[status == 1], flow_frame


def angles_to_com(yaw_pitch):
    return (PP + np.tan(yaw_pitch) * np.array([1., -1.]) * FOCAL_LENGTH).astype(int)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Visual Odometry")
        self.setGeometry(100, 100, 1000, 800)

        # Set background color
        palette = QPalette()
        palette.setColor(QPalette.Background, QColor(40, 40, 40))
        self.setAutoFillBackground(True)
        self.setPalette(palette)

        self.title_label = QLabel("Visual Odometry")
        self.title_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 24px;
                font-weight: bold;
                margin-bottom: 20px;
            }
        """)
        self.title_label.setAlignment(Qt.AlignCenter)

        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet("color: white; font-size: 16px;")

        self.upload_button = QPushButton("Upload File")
        self.upload_button.clicked.connect(self.upload_file)
        self.upload_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                padding: 10px 20px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)

        self.webcam_button = QPushButton("Open Webcam")
        self.webcam_button.clicked.connect(self.open_webcam)
        self.webcam_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-size: 16px;
                padding: 10px 20px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        self.close_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-size: 16px;
                padding: 10px 20px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)

        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.optical_flow_label = QLabel()
        self.optical_flow_label.setAlignment(Qt.AlignCenter)
        self.optical_flow_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        result_layout = QVBoxLayout()
        result_layout.addWidget(self.result_label)
        result_layout.setSpacing(5)

        optical_flow_layout = QVBoxLayout()
        optical_flow_layout.addWidget(self.optical_flow_label)
        optical_flow_layout.setSpacing(5)

        display_layout = QHBoxLayout()
        display_layout.setSpacing(20)
        display_layout.addLayout(result_layout)
        display_layout.addLayout(optical_flow_layout)

        file_layout = QHBoxLayout()
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.upload_button)
        file_layout.addWidget(self.webcam_button)
        file_layout.addWidget(self.close_button)
        file_layout.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        layout.addWidget(self.title_label)
        layout.addLayout(display_layout)
        layout.addLayout(file_layout)
        self.setLayout(layout)

        self.webcam_smoothed = None
        self.webcam_timer = QTimer()
        self.webcam_timer.timeout.connect(self.process_webcam_frame)

    def upload_file(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select Video File")
        if file_path:
            self.file_label.setText(file_path)
            self.process_file(file_path)

    def open_webcam(self):
        self.cap = cv.VideoCapture(0)  # Open the default webcam
        if not self.cap.isOpened():
            print("Failed to open webcam")
            return

        ret, frame = self.cap.read()
        if cv.waitKey(1) == "q":
            return
        # Resize the frame to the desired size
        self.webcam_old_frame = cv.resize(frame, (WIDTH, HEIGHT))

        if not ret:
            print("Failed to read frame from webcam")
            self.cap.release()
            self.cap = None
            return

        self.webcam_timer.start(15)  # Process frames every 30 milliseconds

    def process_webcam_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.webcam_timer.stop()
            self.cap.release()
            self.cap = None
            return

        # Resize the frame to the desired size
        frame = cv.resize(frame, (WIDTH, HEIGHT))

        try:
            # track features across frames
            p0, p1, flow_frame = get_matches_lk(self.webcam_old_frame, frame, True)

            # run SLAM on tracking result and get center of motion
            t = slam(p0, p1)
            yaw_pitch = get_center_of_motion(t)

            if self.webcam_smoothed is None:
                self.webcam_smoothed = yaw_pitch
            else:
                self.webcam_smoothed = (0.9 * self.webcam_smoothed) + (0.1 * yaw_pitch)

            # Draw predicted angles on the frame
            cv.putText(frame, f"Yaw: {yaw_pitch[0]:.4f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv.putText(frame, f"Pitch: {yaw_pitch[1]:.4f}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Draw principal point on the frame
            cv.circle(frame, PP.astype(int), radius=10, color=(0, 0, 255), thickness=-1)

            # Draw predicted center of motion on the frame
            cv.circle(frame, angles_to_com(self.webcam_smoothed), radius=10, color=(255, 255, 0), thickness=-1)

            self.webcam_old_frame = frame
        except Exception as e:
            print(e)

        # Display the frame in the result label
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame_image = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format_RGB888)
        self.result_label.setPixmap(QPixmap.fromImage(frame_image).scaled(self.result_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # Display the optical flow frame in the optical flow label
        flow_frame_rgb = cv.cvtColor(flow_frame, cv.COLOR_BGR2RGB)
        flow_frame_image = QImage(flow_frame_rgb.data, flow_frame_rgb.shape[1], flow_frame_rgb.shape[0], QImage.Format_RGB888)
        self.optical_flow_label.setPixmap(QPixmap.fromImage(flow_frame_image).scaled(self.optical_flow_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def process_file(self, file_path):
        scene = file_path.split("/")[-1].split(".")[0]

        cap = cv.VideoCapture(file_path)

        labeled = True
        try:
            yaw_pitch_labels = np.flip(np.loadtxt(f'labeled/{scene}.txt'), axis=-1)
        except FileNotFoundError:
            labeled = False

        ret, old_frame = cap.read()

        preds = []
        smoothed = None
        n_frame = 1
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                if labeled:
                    # draw label
                    frame = cv.circle(frame, angles_to_com(yaw_pitch_labels[n_frame]), radius=10, color=(255, 255, 255), thickness=-1)

                # track features across frames
                p0, p1, flow_frame = get_matches_lk(old_frame, frame)

                # run SLAM on tracking result and get center of motion
                t = slam(p0, p1)
                yaw_pitch = get_center_of_motion(t)

                if smoothed is None:
                    smoothed = yaw_pitch
                else:
                    smoothed = (0.9 * smoothed) + (0.1 * yaw_pitch)

                preds.append(smoothed)

                # Draw predicted angles on the frame
                cv.putText(frame, f"Yaw: {smoothed[0]:.4f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv.putText(frame, f"Pitch: {smoothed[1]:.4f}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if labeled:
                    # Draw ground truth angles on the frame
                    cv.putText(frame, f"GT Yaw: {yaw_pitch_labels[n_frame][0]:.4f}", (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv.putText(frame, f"GT Pitch: {yaw_pitch_labels[n_frame][1]:.4f}", (10, 120), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                    # Draw ground truth center of motion on the frame
                    cv.circle(frame, angles_to_com(yaw_pitch_labels[n_frame]), radius=10, color=(255, 255, 255), thickness=-1)

                # Draw principal point on the frame
                cv.circle(frame, PP.astype(int), radius=10, color=(0, 0, 255), thickness=-1)

                # Draw predicted center of motion on the frame
                cv.circle(frame, angles_to_com(smoothed), radius=10, color=(255, 255, 0), thickness=-1)

            except Exception:
                preds.append(np.array([np.nan, np.nan]))

            # Display the frame in the result label
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frame_image = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format_RGB888)
            self.result_label.setPixmap(QPixmap.fromImage(frame_image).scaled(self.result_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

            # Display the optical flow frame in the optical flow label
            flow_frame_rgb = cv.cvtColor(flow_frame, cv.COLOR_BGR2RGB)
            flow_frame_image = QImage(flow_frame_rgb.data, flow_frame_rgb.shape[1], flow_frame_rgb.shape[0], QImage.Format_RGB888)
            self.optical_flow_label.setPixmap(QPixmap.fromImage(flow_frame_image).scaled(self.optical_flow_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

            QtWidgets.QApplication.processEvents()

            old_frame = frame
            n_frame += 1

            time.sleep(0.015)

        np.savetxt(f"output/{scene}.txt", np.flip(np.stack(preds[:1] + preds), axis=-1))

    def closeEvent(self, event):
        if self.webcam_timer.isActive():
            self.webcam_timer.stop()
        if self.cap is not None:
            self.cap.release()
        QtWidgets.QApplication.quit()
        sys.exit(0)


if __name__ == "__main__":
    app = QApplication([])
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()
    app.exec_()
