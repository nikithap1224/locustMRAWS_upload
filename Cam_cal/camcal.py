#!/usr/bin/env python

import cv2
import numpy as np
import os
import glob

# =========================
# SETTINGS
# =========================
CHECKERBOARD = (7, 7)   # inner corners
SQUARE_SIZE = 23        # mm (your square size)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# =========================
# PREPARE OBJECT POINTS
# =========================
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []  # 3D points
imgpoints = []  # 2D points

# =========================
# LOAD IMAGES
# =========================
images = glob.glob('./images/*')

if len(images) == 0:
    print(" No images found in ./images/")
    exit()

print(f"Found {len(images)} images")

# =========================
# PROCESS EACH IMAGE
# =========================
for fname in images:
    img = cv2.imread(fname)

    if img is None:
        print(f" Could not read {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Improve contrast (helps your images)
    gray = cv2.equalizeHist(gray)

    # Use robust detection
    ret, corners = cv2.findChessboardCornersSB(gray, CHECKERBOARD, None)

    print(f"{fname} → corners found: {ret}")

    if ret:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )

        imgpoints.append(corners2)

        # Draw corners
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

    cv2.imshow('Detection', img)
    cv2.waitKey(500)

cv2.destroyAllWindows()

# =========================
# CHECK BEFORE CALIBRATION
# =========================
print("Total valid images:", len(objpoints))

if len(objpoints) < 5:
    print(" ERROR: Not enough valid images for calibration")
    exit()

# =========================
# CALIBRATION
# =========================
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("\nCalibration successful!")
print("\nCamera matrix:\n", mtx)
print("\nDistortion coefficients:\n", dist)

# =========================
# TEST UNDISTORTION
# =========================
img = cv2.imread(images[0])
h, w = img.shape[:2]

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

cv2.imshow("Undistorted Image", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

