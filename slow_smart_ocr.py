import argparse
from PIL import Image
from pytesseract import image_to_string
import numpy as np
import cv2


def detect_text_regions_mser(img, debug_folder=None):
    """
    Use MSER to detect text regions and find their bounding corners.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    mser = cv2.MSER_create(
        delta=5,
        min_area=60,
        max_area=14400,
        max_variation=0.25,
        min_diversity=0.2,
        max_evolution=200,
        area_threshold=1.01,
        min_margin=0.003,
        edge_blur_size=5
    )
    """
    mser = cv2.MSER_create(
        delta=8,           # Increased from 5 - less sensitive to small variations
        min_area=120,      # Increased from 60 - filter out tiny noise regions
        max_area=7200,     # Decreased from 14400 - avoid huge regions
        max_variation=0.15, # Decreased from 0.25 - more restrictive on variation
        min_diversity=0.3,  # Increased from 0.2 - more diverse regions
        max_evolution=150,  # Decreased from 200 - less evolution steps
        area_threshold=1.02, # Slightly increased from 1.01
        min_margin=0.005,   # Increased from 0.003
        edge_blur_size=3    # Decreased from 5 - less blurring
    )
    """
    
    regions, _ = mser.detectRegions(gray)
    
    if debug_folder:
        mser_vis = img.copy()
        for region in regions:
            hull = cv2.convexHull(region.reshape(-1, 1, 2))
            cv2.polylines(mser_vis, [hull], True, (0, 255, 0), 1)
        print(f"Writing image to: {debug_folder}/02_mser_regions.jpg")
        cv2.imwrite(f"{debug_folder}/02_mser_regions.jpg", mser_vis)
    
    # Filter regions that look like text
    text_regions = []
    for region in regions:
        x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
        aspect_ratio = w / h if h > 0 else 0
        
        # Text-like characteristics
        if 0.1 < aspect_ratio < 10 and w > 20 and h > 10:
            text_regions.append(region)
    
    if text_regions:
        # Combine all text regions to find overall text area
        all_points = np.vstack(text_regions)
        text_hull = cv2.convexHull(all_points.reshape(-1, 1, 2))
        
        if debug_folder:
            # Visualize the combined hull
            hull_img = img.copy()
            cv2.polylines(hull_img, [text_hull], True, (255, 0, 255), 4)
            cv2.imwrite(f"{debug_folder}/03_mser_combined_hull.jpg", hull_img)
        
        # Try standard approximation first
        epsilon = 0.02 * cv2.arcLength(text_hull, True)
        approx = cv2.approxPolyDP(text_hull, epsilon, True)
        
        if len(approx) == 4:
            # Standard approximation worked
            final_corners = approx.reshape(4, 2)
        else:
            # Force a 4-corner rectangle that encompasses 90% of regions
            print(f"Standard approximation gave {len(approx)} corners, forcing 4-corner rectangle...")
            final_corners = force_rectangle_from_regions(text_regions, coverage_threshold=0.9)
        
        if debug_folder and final_corners is not None:
            # Debug visualization of the final result
            bbox_img = img.copy()
            cv2.polylines(bbox_img, [final_corners.astype(int)], True, (0, 255, 0), 3)
            for i, pt in enumerate(final_corners):
                cv2.circle(bbox_img, tuple(pt.astype(int)), 8, (0, 0, 255), -1)
                cv2.putText(bbox_img, str(i), tuple((pt + [12, 12]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            area = cv2.contourArea(final_corners.astype(int))
            cv2.putText(bbox_img, f"Area: {area:.0f}", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imwrite(f"{debug_folder}/04_mser_final_rectangle.jpg", bbox_img)
        return final_corners


def force_rectangle_from_regions(text_regions, coverage_threshold=0.9):
    """
    Force creation of a 4-corner rectangle that encompasses at least 
    coverage_threshold (e.g., 90%) of the text regions.
    """
    if not text_regions:
        return None
    
    # Get all region centers and bounding boxes
    region_centers = []
    region_bounds = []
    
    for region in text_regions:
        # Get center of each region
        center = np.mean(region, axis=0)
        region_centers.append(center)
        
        # Get bounding box of each region
        x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
        region_bounds.append((x, y, x+w, y+h))
    
    region_centers = np.array(region_centers)
    total_regions = len(text_regions)
    target_regions = int(total_regions * coverage_threshold)
    
    print(f"Trying to cover {target_regions} out of {total_regions} regions ({coverage_threshold*100}%)")
    
    # Method 1: Find rectangle that covers target percentage by expanding from center
    all_points = np.vstack(text_regions)
    
    # Get extreme points with some outlier removal
    x_coords = all_points[:, 0]
    y_coords = all_points[:, 1]
    
    # Remove extreme outliers (bottom/top 5% in each dimension)
    x_sorted = np.sort(x_coords)
    y_sorted = np.sort(y_coords)
    
    outlier_percent = 0.05  # Remove 5% outliers
    x_min = x_sorted[int(len(x_sorted) * outlier_percent)]
    x_max = x_sorted[int(len(x_sorted) * (1 - outlier_percent))]
    y_min = y_sorted[int(len(y_sorted) * outlier_percent)]
    y_max = y_sorted[int(len(y_sorted) * (1 - outlier_percent))]
    
    # Create rectangle corners
    forced_corners = np.array([
        [x_min, y_min],  # top-left
        [x_max, y_min],  # top-right
        [x_max, y_max],  # bottom-right
        [x_min, y_max]   # bottom-left
    ], dtype=np.float32)
    
    # Verify coverage
    covered_regions = 0
    for region in text_regions:
        region_center = np.mean(region, axis=0)
        if (x_min <= region_center[0] <= x_max and 
            y_min <= region_center[1] <= y_max):
            covered_regions += 1
    
    coverage_actual = covered_regions / total_regions
    print(f"Forced rectangle covers {covered_regions}/{total_regions} regions ({coverage_actual*100:.1f}%)")
    
    if coverage_actual >= coverage_threshold * 0.8:  # Accept if at least 80% of target
        return forced_corners
    else:
        # Fallback: use full convex hull corners approximated more aggressively
        print("Coverage too low, using aggressive hull approximation...")
        all_points = np.vstack(text_regions)
        hull = cv2.convexHull(all_points.reshape(-1, 1, 2))
        
        # Try more aggressive epsilon values
        for eps_factor in [0.05, 0.08, 0.1, 0.15]:
            epsilon = eps_factor * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)
            if len(approx) == 4:
                return approx.reshape(4, 2)
        
        # Final fallback: find 4 most extreme points
        return find_extreme_rectangle(all_points)


def find_extreme_rectangle(points):
    """
    Find 4 corners by selecting the most extreme points.
    """
    # Find extreme points
    leftmost = points[np.argmin(points[:, 0])]
    rightmost = points[np.argmax(points[:, 0])]
    topmost = points[np.argmin(points[:, 1])]
    bottommost = points[np.argmax(points[:, 1])]
    
    # Create rectangle from extremes
    x_min, x_max = leftmost[0], rightmost[0]
    y_min, y_max = topmost[1], bottommost[1]
    
    return np.array([
        [x_min, y_min],  # top-left
        [x_max, y_min],  # top-right
        [x_max, y_max],  # bottom-right
        [x_min, y_max]   # bottom-left
    ], dtype=np.float32)


def detect_book_corners(img, debug_folder=None):
    """
    Detect the corners of a book in an image using multiple detection strategies.
    Returns the four corner points of the largest rectangular contour.
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    h, w = gray.shape
    min_area = (h * w) * 0.1  # Minimum 10% of image area
    
    print(f"Image dimensions: {w}x{h}, minimum contour area: {min_area}")
    
    best_corners = None
    best_area = 0
    
    text_corners = detect_text_regions_mser(img, debug_folder)
    if text_corners is not None and is_valid_rectangle(text_corners, w, h):
        area = cv2.contourArea(text_corners)
        if area > best_area:
            best_corners = text_corners
            best_area = area
            print(f"  Found good text region with area {area}")
    
    # Fallback: if no good rectangle found, use edge-based detection
    if best_corners is None:
        print("No rectangular contour found, trying edge-based detection...")
        best_corners = detect_corners_by_edges(gray, debug_folder)
    
    # Final fallback: use full image
    if best_corners is None:
        print("Warning: Could not detect book corners. Using full image.")
        best_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    
    return best_corners


def is_valid_rectangle(corners, img_width, img_height):
    """
    Validate if the detected corners form a reasonable rectangle.
    """
    # Check if corners are within image bounds
    if np.any(corners < 0) or np.any(corners[:, 0] >= img_width) or np.any(corners[:, 1] >= img_height):
        return False
    
    # Calculate area using shoelace formula
    x = corners[:, 0]
    y = corners[:, 1]
    area = 0.5 * abs(sum(x[i]*y[(i+1)%4] - x[(i+1)%4]*y[i] for i in range(4)))
    
    # Check if area is reasonable (at least 5% of image)
    min_area = (img_width * img_height) * 0.05
    if area < min_area:
        return False
    
    # Check aspect ratio (should be reasonable for a book)
    # Calculate width and height of the rectangle
    widths = [np.linalg.norm(corners[(i+1)%4] - corners[i]) for i in range(0, 4, 2)]
    heights = [np.linalg.norm(corners[(i+1)%4] - corners[i]) for i in range(1, 4, 2)]
    
    avg_width = np.mean(widths)
    avg_height = np.mean(heights)
    aspect_ratio = max(avg_width, avg_height) / min(avg_width, avg_height)
    
    # Book aspect ratio should be reasonable (between 0.5 and 3.0)
    if aspect_ratio > 3.0:
        return False
    
    return True


def detect_corners_by_edges(gray, debug_folder=None):
    """
    Alternative corner detection using edge detection and Hough lines.
    """
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    if debug_folder:
        cv2.imwrite(f"{debug_folder}/04_edges.jpg", edges)
    
    # Detect lines using Hough transform
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=200)
    
    if lines is None:
        return None
    
    # Group lines into horizontal and vertical
    horizontal_lines = []
    vertical_lines = []
    
    for line in lines:
        rho, theta = line[0]
        angle = theta * 180 / np.pi
        
        # Horizontal lines (angle near 0 or 180)
        if abs(angle) < 20 or abs(angle - 180) < 20:
            horizontal_lines.append((rho, theta))
        # Vertical lines (angle near 90)
        elif abs(angle - 90) < 20:
            vertical_lines.append((rho, theta))
    
    # Find intersections to get corners
    corners = []
    for h_line in horizontal_lines[:3]:  # Top few horizontal lines
        for v_line in vertical_lines[:3]:  # Top few vertical lines
            intersection = line_intersection(h_line, v_line)
            if intersection is not None:
                corners.append(intersection)
    
    if len(corners) >= 4:
        # Select the 4 corners that form the largest rectangle
        corners = np.array(corners)
        # Simple selection: take corners that are most spread out
        return select_corner_points(corners)
    
    return None


def line_intersection(line1, line2):
    """
    Find intersection point of two lines in polar form (rho, theta).
    """
    rho1, theta1 = line1
    rho2, theta2 = line2
    
    # Convert to cartesian form: ax + by = c
    a1, b1 = np.cos(theta1), np.sin(theta1)
    a2, b2 = np.cos(theta2), np.sin(theta2)
    c1, c2 = rho1, rho2
    
    # Solve system of equations
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-10:  # Lines are parallel
        return None
    
    x = (c1 * b2 - c2 * b1) / det
    y = (a1 * c2 - a2 * c1) / det
    
    return np.array([x, y])


def select_corner_points(points):
    """
    Select 4 corner points from a larger set of points.
    """
    if len(points) < 4:
        return None
    
    # Find convex hull
    hull = cv2.convexHull(points.astype(np.float32))
    
    if len(hull) >= 4:
        # Approximate to 4 points
        epsilon = 0.02 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        
        if len(approx) == 4:
            return approx.reshape(4, 2)
    
    # Fallback: select 4 most extreme points
    top_left = points[np.argmin(points[:, 0] + points[:, 1])]
    top_right = points[np.argmax(points[:, 0] - points[:, 1])]
    bottom_right = points[np.argmax(points[:, 0] + points[:, 1])]
    bottom_left = points[np.argmin(points[:, 0] - points[:, 1])]
    
    return np.array([top_left, top_right, bottom_right, bottom_left])


def order_corners(corners):
    """
    Order corners in a consistent manner: top-left, top-right, bottom-right, bottom-left
    """
    # Sort by y-coordinate (top points first)
    corners = corners[np.argsort(corners[:, 1])]
    
    # Get top two and bottom two points
    top_points = corners[:2]
    bottom_points = corners[2:]
    
    # Sort top points by x-coordinate (left first)
    top_points = top_points[np.argsort(top_points[:, 0])]
    # Sort bottom points by x-coordinate (right first for bottom-right, bottom-left order)
    bottom_points = bottom_points[np.argsort(bottom_points[:, 0])[::-1]]
    
    return np.array([top_points[0], top_points[1], bottom_points[0], bottom_points[1]], dtype=np.float32)


def correct_perspective(img, corners):
    """
    Apply perspective correction to straighten the book image.
    Accounts for page warping by adding conservative margins.
    """
    # Order the corners
    ordered_corners = order_corners(corners)
    
    # Calculate the width and height of the corrected image
    width_top = np.linalg.norm(ordered_corners[1] - ordered_corners[0])
    width_bottom = np.linalg.norm(ordered_corners[2] - ordered_corners[3])
    width = int(max(width_top, width_bottom))
    
    height_left = np.linalg.norm(ordered_corners[3] - ordered_corners[0])
    height_right = np.linalg.norm(ordered_corners[2] - ordered_corners[1])
    height = int(max(height_left, height_right))
    
    # Add conservative margins to account for page warping and detection errors
    width_margin = int(width * 0.05)  # 5% margin on left/right
    height_margin_top = int(height * 0.08)  # 8% margin on top (more for warping)
    height_margin_bottom = int(height * 0.12)  # 12% margin on bottom (more for warping)
    
    # Adjust final dimensions
    final_width = width + (2 * width_margin)
    final_height = height + height_margin_top + height_margin_bottom
    
    # Define destination points for perspective correction with margins
    dst_corners = np.array([
        [width_margin, height_margin_top],  # top-left with margins
        [width + width_margin, height_margin_top],  # top-right
        [width + width_margin, height + height_margin_top],  # bottom-right
        [width_margin, height + height_margin_top]  # bottom-left
    ], dtype=np.float32)
    
    # Calculate perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(ordered_corners, dst_corners)
    
    # Apply perspective correction with the expanded canvas
    corrected = cv2.warpPerspective(img, matrix, (final_width, final_height))
    
    return corrected


def main(img_name, visualize=False):
    """
    Main function to process the book image with corner detection and OCR.
    """
    try:
        # Create output directory structure
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_folder = f"ocr_debug_output_{timestamp}"
        images_folder = f"{debug_folder}/images"
        
        import os
        os.makedirs(images_folder, exist_ok=True)
        
        # Load the image using OpenCV to avoid color issues
        img = cv2.imread(img_name)
        if img is None:
            print(f"Error: Could not load image {img_name}")
            return None
        
        print(f"Processing image: {img_name}")
        print(f"Image shape: {img.shape}")
        
        # Save original image for reference
        cv2.imwrite(f"{images_folder}/01_original.jpg", img)
        
        # Detect book corners
        print("Detecting book corners...")
        corners = detect_book_corners(img, images_folder)
        print(f"Detected corners: {corners}")
        
        # Apply perspective correction
        print("Applying perspective correction...")
        corrected_img = correct_perspective(img, corners)
        
        # Save corrected image for inspection
        cv2.imwrite(f"{images_folder}/05_corrected.jpg", corrected_img)
        print(f"Corrected image saved as {images_folder}/05_corrected.jpg")
        
        # Perform OCR on the corrected image
        print("Performing OCR...")
        corrected_pil = Image.fromarray(corrected_img)
        text = image_to_string(corrected_pil)
        
        # Save OCR result to file
        with open(f"{debug_folder}/ocr_result.txt", "w", encoding="utf-8") as f:
            f.write(text)
        return text
        
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text from book images with corner detection")
    parser.add_argument("image", help="Path to the input image file")
    parser.add_argument("--visualize", "-v", action="store_true", 
                       help="Save visualization of detected corners")
    
    args = parser.parse_args()
    
    main(args.image, args.visualize)
