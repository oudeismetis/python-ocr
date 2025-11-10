import argparse
import os
from datetime import datetime

import cv2
import numpy as np

def crop_image(image, x, y, w, h):
    padding = 0
    x_padded = max(0, x - padding)
    y_padded = max(0, y - padding)
    w_padded = min(image.shape[1] - x_padded, w + 2 * padding)
    h_padded = min(image.shape[0] - y_padded, h + 2 * padding)

    cropped_hull = image[y_padded:y_padded+h_padded, x_padded:x_padded+w_padded]
    return cropped_hull

def main(folder):
    """
    # After we have run stereo_calibration.py
    # https://chatgpt.com/c/6891188d-a7f0-8001-96e8-8a1112460b5d

    # Load stereo calibration / rectification
    data = np.load("stereo_calib.npz")
    mtxL = data["mtxL"]
    distL = data["distL"]
    mtxR = data["mtxR"]
    distR = data["distR"]
    R1 = data["R1"]
    R2 = data["R2"]
    P1 = data["P1"]
    P2 = data["P2"]
    Q  = data["Q"]  # optional, for depth reprojection
    """

    # Load your stereo pair (left and right)
    # left = cv2.imread(f"{folder}/left.png", cv2.IMREAD_COLOR)
    # right = cv2.imread(f"{folder}/right.png", cv2.IMREAD_COLOR)
    left = cv2.imread(f"{folder}/cropped_left.jpg", cv2.IMREAD_COLOR)
    right = cv2.imread(f"{folder}/cropped_right.jpg", cv2.IMREAD_COLOR)

    if left is None or right is None:
        raise ValueError("Failed to load one of the images. Check the paths.")
    if left.shape[:2] != right.shape[:2]:
        right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_LINEAR)

    """
    # After we have run stereo_calibration.py
    # Assume they are already same resolution as used in calibration.
    h, w = left_raw.shape[:2]

    # Precompute rectification maps (do this once if reusing)
    map1x, map1y = cv2.initUndistortRectifyMap(
        mtxL, distL, R1, P1, (w, h), cv2.CV_32FC1
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        mtxR, distR, R2, P2, (w, h), cv2.CV_32FC1
    )

    # Rectify images
    left = cv2.remap(left_raw, map1x, map1y, interpolation=cv2.INTER_LINEAR)
    right = cv2.remap(right_raw, map2x, map2y, interpolation=cv2.INTER_LINEAR)
    """
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_folder = f"stereo_debug_output_{timestamp}"
    os.makedirs(debug_folder, exist_ok=True)

    # crop_left = crop_image(left, 200, 150, 200 + 950, 750)
    # crop_right = crop_image(right, 0, 150, 950, 750)

    cv2.imwrite(f"{debug_folder}/01_cropped_left.jpg", left)
    cv2.imwrite(f"{debug_folder}/02_cropped_right.jpg", right)
    
    # return

    # Convert to grayscale for disparity computation
    gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    # Create StereoSGBM matcher (tune parameters for your setup)
    window_size = 5
    min_disp = 0
    num_disp = 16 * 8  # must be divisible by 16
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )

    # Compute disparity (scaled by 16 internally)
    disp = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0

    # Normalize for visualization if you want:
    disp_vis = cv2.normalize(disp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disp_vis = np.uint8(disp_vis)

    # Create a confidence mask: valid disparity > 0
    valid_mask = disp > min_disp

    # Warp right image to left viewpoint using disparity:
    # Build a map for remapping right into left. For each pixel in left, we know disparity d, so corresponding x in right is x - d.
    h, w = gray_left.shape
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            d = disp[y, x]
            if d > 0:
                xr = x - d
            else:
                xr = x
            map_x[y, x] = xr
            map_y[y, x] = y

    warped_right = cv2.remap(right, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # Optional: compute a per-pixel weight based on disparity confidence (e.g., higher weight where disparity is valid)
    weight_left = (valid_mask.astype(np.float32) * 0.5) + 0.5  # give baseline weight
    weight_right = (valid_mask.astype(np.float32) * 0.5) + 0.5

    # Simple fusion: normalized weighted average
    fused = (left.astype(np.float32) * weight_left[..., None] + warped_right.astype(np.float32) * weight_right[..., None])
    norm = (weight_left + weight_right)[..., None]
    fused = fused / np.clip(norm, 1e-5, None)
    fused = np.clip(fused, 0, 255).astype(np.uint8)

    # Optional: refine with a guided filter using left as guide to clean artifacts
    # fused_refined = cv2.ximgproc.guidedFilter(guide=gray_left, src=fused, radius=8, eps=1e-2)
    # fused_blur = cv2.bilateralFilter(fused, d=9, sigmaColor=75, sigmaSpace=75)

    # Save or show
    cv2.imwrite(f"{debug_folder}/03_fused.jpg", fused)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text from book image")
    parser.add_argument("folder", help="Path to the input image file")
    args = parser.parse_args()
    
    main(args.folder)
