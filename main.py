import numpy as np
import cv2 as cv
import argparse

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


def get_matches_lk(old_frame, frame):
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
        mask=roi_mask
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

    # Display the frame with optical flow vectors
    cv.imshow('Optical Flow', flow_frame)

    return p0[status == 1], p1[status == 1]


def angles_to_com(yaw_pitch):
    return (PP + np.tan(yaw_pitch) * np.array([1., -1.]) * FOCAL_LENGTH).astype(int)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='path to image file')
    args = parser.parse_args()

    scene = args.file.split("/")[1].split(".")[0]

    cap = cv.VideoCapture(args.file)

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
            frame = cv.circle(frame, angles_to_com(yaw_pitch_labels[n_frame]), radius=5, color=(255, 255, 255), thickness=-1)
            
            p0, p1 = get_matches_lk(old_frame, frame)

            t = slam(p0, p1)
            yaw_pitch = get_center_of_motion(t)

            if smoothed is None:
                smoothed = yaw_pitch
            else:
                diff = np.abs(smoothed - yaw_pitch)
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
                cv.circle(frame, angles_to_com(yaw_pitch_labels[n_frame]), radius=5, color=(255, 255, 255), thickness=-1)

            # Draw principal point on the frame
            cv.circle(frame, PP.astype(int), radius=5, color=(0, 0, 255), thickness=-1)

            # Draw predicted center of motion on the frame
            cv.circle(frame, angles_to_com(smoothed), radius=5, color=(255, 255, 0), thickness=-1)

        except Exception:
            preds.append(np.array([np.nan, np.nan]))
            # pass

        # Display the frame
        cv.imshow('frame', frame)

        k = cv.waitKey(1) & 0xff
        if k == 27:
            break

        old_frame = frame
        n_frame += 1

    np.savetxt(f"output/{scene}.txt", np.flip(np.stack(preds[:1] + preds), axis=-1))
    cv.destroyAllWindows()
