import os
import sys
from datetime import datetime

from PIL import Image
import cv2
import numpy as np

def create_output_folder():
    """Create timestamped output folder for debugging images"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"blur_debug_output_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)
    os.makedirs(f"{folder_name}/images", exist_ok=True)
    os.makedirs(f"{folder_name}/text", exist_ok=True)
    return folder_name

def load_image(img_name):
    img_orig = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
    if img_orig is None:
        img_orig = np.array(Image.open(img_name))
        if len(img_orig.shape) == 3 and img_orig.shape[2] == 3:
            img_orig = cv2.cvtColor(img_orig, cv2.COLOR_RGB2BGR)
    return img_orig

def adaptive_canny(gray, block_size=15):
    """
    Apply Canny edge detection with locally adaptive thresholds
    based on local contrast
    """
    # Calculate local standard deviation as measure of contrast
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    mean = cv2.blur(gray.astype(np.float32), (block_size, block_size))
    mean_sq = cv2.blur((gray.astype(np.float32) ** 2), (block_size, block_size))
    std = np.sqrt(np.maximum(mean_sq - mean * mean, 0))
    
    # Normalize std to get adaptive thresholds
    if np.max(std) > np.min(std):
        std_norm = cv2.normalize(std, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        # If uniform, create a default contrast map
        std_norm = np.ones_like(gray) * 128
    
    # Use median of local contrast for threshold
    median_std = np.median(std_norm)
    lower = max(50, int(median_std * 0.5))
    upper = min(200, int(median_std * 1.5))
    
    edges = cv2.Canny(blur, lower, upper)
    return edges, std_norm

def is_page_like(contour, img_shape):
    """
    Check if contour has properties consistent with a book page
    """
    # Minimum area (pages should be substantial)
    area = cv2.contourArea(contour)
    img_area = img_shape[0] * img_shape[1]
    if area < img_area * 0.02:  # At least 1% of image
        return False
    
    # Shouldn't be the whole image
    if area > img_area * 0.8:
        return False

    # Check if we can fit a rectangle
    rect = cv2.minAreaRect(contour)
    box_area = rect[1][0] * rect[1][1]
    if box_area == 0:
        return False
    
    # Extent: ratio of contour area to bounding box area
    extent = area / box_area
    if extent < 0.5:  # Should fill most of bounding box
        return False
    
    # Aspect ratio (typical book page)
    width, height = rect[1]
    if width == 0 or height == 0:
        return False
    aspect = max(width, height) / min(width, height)
    if aspect < 0.5 or aspect > 2.5:  # Reasonable page proportions
        return False
    
    # Convexity
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return False
    convexity = area / hull_area
    if convexity < 0.75:  # Pages should be mostly convex
        return False
    
    return True

def fit_quadrilateral(contour):
    """
    Fit a 4-sided polygon to the contour
    Returns None if can't fit to quadrilateral
    """
    # Approximate contour to polygon
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # Try increasingly aggressive approximations
    for factor in [0.02, 0.03, 0.04, 0.05]:
        epsilon = factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            return approx
        elif len(approx) < 4:
            break
    
    # If we can't get exactly 4, try to use minAreaRect
    if len(approx) != 4:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        approx = np.int0(box).reshape((-1, 1, 2))
    
    return approx if len(approx) == 4 else None

def are_adjacent(quad1, quad2, img_shape):
    """
    Check if two quadrilaterals are adjacent (sharing an edge)
    as left and right pages would
    """
    # Get centers
    M1 = cv2.moments(quad1)
    M2 = cv2.moments(quad2)
    if M1['m00'] == 0 or M2['m00'] == 0:
        return False
    
    cx1 = M1['m10'] / M1['m00']
    cx2 = M2['m10'] / M2['m00']
    cy1 = M1['m01'] / M1['m00']
    cy2 = M2['m01'] / M2['m00']
    
    # Should be roughly side-by-side (not above/below)
    vertical_diff = abs(cy1 - cy2)
    horizontal_diff = abs(cx1 - cx2)
    
    if vertical_diff > img_shape[0] * 0.2:  # Too far apart vertically
        return False
    
    if horizontal_diff < img_shape[1] * 0.1:  # Too close horizontally
        return False
    
    # Check if they're touching or very close
    # Find minimum distance between any two points
    min_dist = float('inf')
    for p1 in quad1:
        for p2 in quad2:
            dist = np.linalg.norm(p1[0] - p2[0])
            min_dist = min(min_dist, dist)
    
    # Should be close (sharing spine area)
    max_gap = img_shape[1] * 0.15
    return min_dist < max_gap

def estimate_flatness(quad):
    """
    Estimate how flat a page is based on corner angles
    Returns 0-1, where 1 is perfectly flat
    """
    if quad is None or len(quad) != 4:
        return 0.0
    
    points = quad.reshape(4, 2).astype(np.float32)
    
    # Calculate angles at each corner
    angles = []
    for i in range(4):
        p1 = points[i]
        p2 = points[(i + 1) % 4]
        p0 = points[(i - 1) % 4]
        
        v1 = p0 - p1
        v2 = p2 - p1
        
        # Calculate angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        angles.append(angle)
    
    # Perfect rectangle has 90 degree (pi/2) corners
    angle_deviations = [abs(a - np.pi/2) for a in angles]
    avg_deviation = np.mean(angle_deviations)
    
    # Convert to 0-1 score (less deviation = higher flatness)
    # Max expected deviation is pi/4 (45 degrees)
    flatness = 1.0 - min(avg_deviation / (np.pi/4), 1.0)
    
    return flatness

def find_best_page_pair(candidates, img_shape):
    """
    Find the best pair of adjacent pages from candidates
    """
    if len(candidates) < 2:
        return None, None
    
    best_pair = None
    best_score = -1
    
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            if are_adjacent(candidates[i], candidates[j], img_shape):
                # Score based on size and position
                area1 = cv2.contourArea(candidates[i])
                area2 = cv2.contourArea(candidates[j])
                
                # Prefer larger, similarly-sized pages
                size_similarity = min(area1, area2) / max(area1, area2)
                avg_size = (area1 + area2) / 2
                size_score = avg_size / (img_shape[0] * img_shape[1])
                
                score = size_similarity * size_score
                
                if score > best_score:
                    best_score = score
                    # Order by x position (left page first)
                    M1 = cv2.moments(candidates[i])
                    M2 = cv2.moments(candidates[j])
                    cx1 = M1['m10'] / M1['m00']
                    cx2 = M2['m10'] / M2['m00']
                    
                    if cx1 < cx2:
                        best_pair = (candidates[i], candidates[j])
                    else:
                        best_pair = (candidates[j], candidates[i])
    
    if best_pair:
        return best_pair[0], best_pair[1]
    
    # Fallback: return two largest
    candidates_sorted = sorted(candidates, key=lambda c: cv2.contourArea(c), reverse=True)
    if len(candidates_sorted) >= 2:
        M1 = cv2.moments(candidates_sorted[0])
        M2 = cv2.moments(candidates_sorted[1])
        cx1 = M1['m10'] / M1['m00']
        cx2 = M2['m10'] / M2['m00']
        
        if cx1 < cx2:
            return candidates_sorted[0], candidates_sorted[1]
        else:
            return candidates_sorted[1], candidates_sorted[0]
    
    return None, None

def main(img_name):
    output_folder = create_output_folder()
    
    # Load image
    img = load_image(img_name)
    cv2.imwrite(os.path.join(output_folder, "01_original.png"), img)
    
    # Step 1: Downscale for speed
    scale_factor = 0.5  # Adjust based on your needs
    small = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    print(f"Downscaled to: {small.shape}")
    cv2.imwrite(os.path.join(output_folder, "02_downscaled.png"), small)
    
    # Step 2: Bilateral filter (preserves edges while reducing noise)
    blurred = cv2.bilateralFilter(small, 9, 75, 75)
    cv2.imwrite(os.path.join(output_folder, "03_bilateral_filtered.png"), blurred)
    
    # Step 3: Convert to grayscale
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(output_folder, "04_grayscale.png"), gray)
    
    # Step 4: Adaptive Canny edge detection
    edges, contrast_map = adaptive_canny(gray)
    print(f"Edge detection complete")
    cv2.imwrite(os.path.join(output_folder, "05_contrast_map.png"), contrast_map)
    cv2.imwrite(os.path.join(output_folder, "06_edges.png"), edges)

    # Other than letters not being filled in and having a border, this looks REAL good !!!!!!!!
    # thresh = cv2.bitwise_not(edges)
    # cv2.imwrite(os.path.join(output_folder, "06b_flipped.png"), thresh)


    # NEW: Step 4.5: Morphological operations to close gaps and form page regions
    # Dilate to connect nearby edges (especially page borders)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    cv2.imwrite(os.path.join(output_folder, "06b_dilated.png"), dilated)
    
    # Close to fill small holes
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    cv2.imwrite(os.path.join(output_folder, "06c_closed.png"), closed)
    
    # Optional: Erode slightly to restore edge positions
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    processed_edges = cv2.erode(closed, kernel_erode, iterations=1)
    cv2.imwrite(os.path.join(output_folder, "06d_processed_edges.png"), processed_edges)
    

    
    # Step 5: Find contours
    contours, hierarchy = cv2.findContours(processed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Found {len(contours)} contours")
    
    # Draw all contours for debugging
    all_contours_img = small.copy()
    cv2.drawContours(all_contours_img, contours, -1, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_folder, "07_all_contours.png"), all_contours_img)
    
    # Step 6: Filter contours by page-like properties
    candidates = []
    for contour in contours:
        if is_page_like(contour, small.shape):
            candidates.append(contour)
    
    print(f"Found {len(candidates)} page-like candidates")
    
    # Draw candidate contours
    candidates_img = small.copy()
    cv2.drawContours(candidates_img, candidates, -1, (0, 255, 255), 2)
    cv2.imwrite(os.path.join(output_folder, "08_candidate_contours.png"), candidates_img)
    
    # Step 7: Fit quadrilaterals
    quadrilaterals = []
    for contour in candidates:
        quad = fit_quadrilateral(contour)
        if quad is not None:
            quadrilaterals.append(quad)
    
    print(f"Fitted {len(quadrilaterals)} quadrilaterals")
    
    # Draw quadrilaterals
    quads_img = small.copy()
    for quad in quadrilaterals:
        cv2.drawContours(quads_img, [quad], 0, (255, 0, 255), 3)
    cv2.imwrite(os.path.join(output_folder, "09_quadrilaterals.png"), quads_img)
    
    # Step 8: Find best pair of adjacent pages
    left_page, right_page = find_best_page_pair(quadrilaterals, small.shape)
    
    # Step 9: Draw final result
    result_img = small.copy()
    
    if left_page is not None:
        cv2.drawContours(result_img, [left_page], 0, (0, 255, 0), 3)
        flatness_left = estimate_flatness(left_page)
        print(f"Left page flatness: {flatness_left:.2f}")
        
        # Add label
        M = cv2.moments(left_page)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.putText(result_img, f"L: {flatness_left:.2f}", (cx-30, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if right_page is not None:
        cv2.drawContours(result_img, [right_page], 0, (255, 0, 0), 3)
        flatness_right = estimate_flatness(right_page)
        print(f"Right page flatness: {flatness_right:.2f}")
        
        # Add label
        M = cv2.moments(right_page)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.putText(result_img, f"R: {flatness_right:.2f}", (cx-30, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    cv2.imwrite(os.path.join(output_folder, "10_final_result.png"), result_img)
    
    # Scale coordinates back to original resolution if needed
    if left_page is not None and scale_factor != 1.0:
        left_page_full = (left_page / scale_factor).astype(np.int32)
    else:
        left_page_full = left_page
    
    if right_page is not None and scale_factor != 1.0:
        right_page_full = (right_page / scale_factor).astype(np.int32)
    else:
        right_page_full = right_page
    
    return {
        'left_page': left_page_full,
        'right_page': right_page_full,
        'flatness_left': flatness_left if left_page is not None else 0.0,
        'flatness_right': flatness_right if right_page is not None else 0.0,
        'num_candidates': len(candidates)
    }

if __name__ == "__main__":
    img_name = sys.argv[1]
    # Example usage
    result = main(img_name)
    print(f"\nDetection Results:")
    print(f"Left page detected: {result['left_page'] is not None}")
    print(f"Right page detected: {result['right_page'] is not None}")
    print(f"Flatness scores - Left: {result['flatness_left']:.2f}, Right: {result['flatness_right']:.2f}")
