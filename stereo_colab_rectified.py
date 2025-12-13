import cv2
import numpy as np

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

# Load your stereo pair
left_raw = cv2.imread("left.jpg", cv2.IMREAD_COLOR)
right_raw = cv2.imread("right.jpg", cv2.IMREAD_COLOR)
if left_raw is None or right_raw is None:
    raise RuntimeError("Failed to load images")

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

# Grayscale for disparity
gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

# Stereo matcher (tune these if needed)
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

# Compute disparity (rectified)
disp = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0  # disparity in pixels

# Confidence / validity mask
valid_mask = disp > min_disp

# Warp right into left using disparity: in rectified images, epipolar lines are horizontal,
# so corresponding x in right is x - disp
h, w = gray_left.shape
map_x = np.repeat(np.arange(w)[None, :], h, axis=0).astype(np.float32) - disp
map_y = np.repeat(np.arange(h)[:, None], w, axis=1).astype(np.float32)
warped_right = cv2.remap(right, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

# Simple confidence-weighted fusion
weight_left = (valid_mask.astype(np.float32) * 0.5) + 0.5
weight_right = (valid_mask.astype(np.float32) * 0.5) + 0.5
fused = (left.astype(np.float32) * weight_left[..., None] + warped_right.astype(np.float32) * weight_right[..., None])
norm = (weight_left + weight_right)[..., None]
fused = fused / np.clip(norm, 1e-5, None)
fused = np.clip(fused, 0, 255).astype(np.uint8)

# Optional: refine with a simple edge-aware filter if you don’t have ximgproc
def fast_guided_filter(I, p, r, eps):
    I = I.astype(np.float32) / 255.0
    p = p.astype(np.float32) / 255.0
    mean_I = cv2.boxFilter(I, -1, (2*r+1, 2*r+1))
    mean_p = cv2.boxFilter(p, -1, (2*r+1, 2*r+1))
    corr_I = cv2.boxFilter(I * I, -1, (2*r+1, 2*r+1))
    corr_Ip = cv2.boxFilter(I * p, -1, (2*r+1, 2*r+1))
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, -1, (2*r+1, 2*r+1))
    mean_b = cv2.boxFilter(b, -1, (2*r+1, 2*r+1))
    q = mean_a * I + mean_b
    return np.clip(q * 255, 0, 255).astype(np.uint8)

# Use left grayscale as guide to refine fused
gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
fused_refined = fast_guided_filter(gray_left, cv2.cvtColor(fused, cv2.COLOR_BGR2GRAY), r=8, eps=1e-2)
# If you want color refinement, you can apply the guided filter per-channel or upsample the gray guidance

# Save result
cv2.imwrite("fused.png", fused)
cv2.imwrite("fused_refined.png", fused_refined)

