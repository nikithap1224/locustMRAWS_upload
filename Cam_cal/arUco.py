import cv2
import numpy as np

# Load calibration file
with np.load('calibration_data.npz') as data:
    mtx = data['mtx']
    dist = data['dist']

# ArUco setup
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

MARKER_SIZE = 5.0  # cm

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        for i, corner in enumerate(corners):

            # 3D marker points
            obj_points = np.array([
                [-MARKER_SIZE/2,  MARKER_SIZE/2, 0],
                [ MARKER_SIZE/2,  MARKER_SIZE/2, 0],
                [ MARKER_SIZE/2, -MARKER_SIZE/2, 0],
                [-MARKER_SIZE/2, -MARKER_SIZE/2, 0]
            ], dtype=np.float32)

            img_points = corner.reshape(4, 2)

            success, rvec, tvec = cv2.solvePnP(obj_points, img_points, mtx, dist)

            if success:
                # Draw axis
                cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, 3)

                # Distance (Z-axis)
                distance = tvec[2][0]

                cv2.putText(frame,
                            f"ID:{ids[i][0]} Dist:{distance:.2f} cm",
                            (10, 30 + i*30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0,255,0), 2)

    cv2.imshow("ArUco Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()