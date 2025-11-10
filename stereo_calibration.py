import cv2
import numpy as np
import glob

# TODO
# print checkerboard
# capture 15-30 image pairs of that checkerboard at various angles
# https://chatgpt.com/c/6891188d-a7f0-8001-96e8-8a1112460b5d

# Checkerboard configuration
CHECKERBOARD = (9, 6)  # inner corners per row/col, adjust to your printed board
square_size = 0.025  # meters (or arbitrary units, consistent) size of one square

# Prepare object points: (0,0,0), (1,0,0), ..., in the checkerboard plane
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), dtype=np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays to store object points and image points from all pairs
objpoints = []  # 3d points in real world
imgpoints_left = []
imgpoints_right = []

# Assume you have lists of paths or loaded images: left_images[], right_images[]
# They must be corresponding pairs.
left_images = sorted(glob.glob("calib/left/*.jpg"))
right_images = sorted(glob.glob("calib/right/*.jpg"))

assert len(left_images) == len(right_images), "Need equal number of left/right pairs"

# Criteria for cornerSubPix
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

for left_path, right_path in zip(left_images, right_images):
    imgL = cv2.imread(left_path)
    imgR = cv2.imread(right_path)
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    retL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD, None)
    retR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD, None)

    if retL and retR:
        cv2.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
        cv2.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)

        objpoints.append(objp)
        imgpoints_left.append(cornersL)
        imgpoints_right.append(cornersR)

# Calibrate each camera individually (optional but helps)
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpoints_left, grayL.shape[::-1], None, None)
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpoints_right, grayR.shape[::-1], None, None)

# Stereo calibration
flags = cv2.CALIB_FIX_INTRINSIC  # if intrinsics are trusted; else omit and let stereo refine them
criteria_stereo = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
retStereo, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    objpoints,
    imgpoints_left,
    imgpoints_right,
    mtxL,
    distL,
    mtxR,
    distR,
    grayL.shape[::-1],
    criteria=criteria_stereo,
    flags=flags
)

# Compute rectification transforms
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    mtxL, distL, mtxR, distR, grayL.shape[::-1], R, T, alpha=0
)

# Save these matrices (mtxL, distL, mtxR, distR, R, T, R1, R2, P1, P2, Q) for reuse
np.savez("stereo_calib.npz",
         mtxL=mtxL, distL=distL,
         mtxR=mtxR, distR=distR,
         R1=R1, R2=R2, P1=P1, P2=P2, Q=Q)

