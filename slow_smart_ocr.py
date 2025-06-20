import argparse
from PIL import Image
from pytesseract import image_to_string
import numpy as np
import cv2


def bbox_debug_images(img, filtered_regions, clusters, debug_folder):
    """
    Debug visualization for morphological clustering results
    
    Args:
        img: Original image
        filtered_regions: List of MSER regions after preprocessing
        clusters: List of cluster lists from morphological clustering
        debug_folder: Output folder for debug images
    """
    if not debug_folder:
        return
    
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), 
              (255, 0, 255), (0, 255, 255)]  # Green, Red, Blue, Yellow, Magenta, Cyan
    
    # 1. Show all filtered regions (before clustering)
    all_regions_vis = img.copy()
    for region in filtered_regions:
        hull = cv2.convexHull(region.reshape(-1, 1, 2))
        cv2.polylines(all_regions_vis, [hull], True, (0, 255, 0), 1)
    
    print(f"Writing image to: {debug_folder}/03_all_filtered_regions.jpg")
    cv2.imwrite(f"{debug_folder}/03_all_filtered_regions.jpg", all_regions_vis)
    
    # 2. Show morphological mask (intermediate step)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for region in filtered_regions:
        cv2.fillPoly(mask, [region], 255)
    
    # Apply morphological closing
    kernel_size = min(img.shape[:2]) // 50
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    print(f"Writing image to: {debug_folder}/04_morphological_mask.jpg")
    cv2.imwrite(f"{debug_folder}/04_morphological_mask.jpg", closed)
    
    # 3. Show clustered regions with different colors
    cluster_vis = img.copy()
    
    # Create a mapping from region to cluster
    region_to_cluster = {}
    for cluster_id, cluster_regions in enumerate(clusters):
        for region in cluster_regions:
            # Use region as key (convert to tuple for hashing)
            region_key = tuple(region.flatten())
            region_to_cluster[region_key] = cluster_id
    
    # Draw regions colored by cluster
    unclustered_count = 0
    for region in filtered_regions:
        region_key = tuple(region.flatten())
        if region_key in region_to_cluster:
            cluster_id = region_to_cluster[region_key]
            color = colors[cluster_id % len(colors)]
        else:
            color = (128, 128, 128)  # Gray for unclustered (noise)
            unclustered_count += 1
        
        hull = cv2.convexHull(region.reshape(-1, 1, 2))
        cv2.polylines(cluster_vis, [hull], True, color, 1)
    
    print(f"Writing image to: {debug_folder}/05_clustered_regions.jpg")
    cv2.imwrite(f"{debug_folder}/05_clustered_regions.jpg", cluster_vis)
    
    # 4. Show final bounding boxes for main clusters (with outlier removal)
    bbox_vis = img.copy()
    
    def get_convex_hull_bbox(cluster_regions):
        """
        Use convex hull of centroids instead of all points
        """
        centroids = []
        for region in cluster_regions:
            M = cv2.moments(region)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centroids.append([cx, cy])
        
        if len(centroids) < 3:
            all_points = np.vstack(cluster_regions)
            return cv2.boundingRect(all_points)
        
        centroids = np.array(centroids, dtype=np.int32)
        hull = cv2.convexHull(centroids)
        return cv2.boundingRect(hull)

    
    for i, cluster_regions in enumerate(clusters):
        if cluster_regions:  # Make sure cluster is not empty
            # Get tight bounding box with outlier removal
            # bbox_result = get_tight_bounding_box(cluster_regions, outlier_percentile=10)
            bbox_result = get_convex_hull_bbox(cluster_regions)
            if bbox_result is not None:
                regions_used = 999
                (x, y, w, h) = bbox_result
                
                # Draw bounding box
                color = colors[i % len(colors)]
                cv2.rectangle(bbox_vis, (x, y), (x + w, y + h), color, 3)
                
                # Add cluster label
                cv2.putText(bbox_vis, f'Page {i + 1}', (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                # Add cluster info
                cv2.putText(bbox_vis, f'{regions_used}/{len(cluster_regions)} regions', 
                           (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                print(f"Cluster {i + 1}: {regions_used}/{len(cluster_regions)} regions used, "
                      f"bbox: ({x}, {y}, {w}, {h})")
    
    print(f"Writing image to: {debug_folder}/06_final_bounding_boxes.jpg")
    cv2.imwrite(f"{debug_folder}/06_final_bounding_boxes.jpg", bbox_vis)
    
    # 5. Summary overlay showing both regions and bounding boxes
    summary_vis = img.copy()
    
    # Draw all clustered regions lightly
    for i, cluster_regions in enumerate(clusters):
        color = colors[i % len(colors)]
        for region in cluster_regions:
            hull = cv2.convexHull(region.reshape(-1, 1, 2))
            cv2.polylines(summary_vis, [hull], True, color, 1)
        
        # Draw bounding box on top
        if cluster_regions:
            all_points = np.vstack(cluster_regions)
            x, y, w, h = cv2.boundingRect(all_points)
            cv2.rectangle(summary_vis, (x, y), (x + w, y + h), color, 3)
            cv2.putText(summary_vis, f'Page {i + 1}', (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    print(f"Writing image to: {debug_folder}/07_summary_overlay.jpg")
    cv2.imwrite(f"{debug_folder}/07_summary_overlay.jpg", summary_vis)
    
    # Print summary stats
    print(f"\n=== Clustering Summary ===")
    print(f"Total filtered regions: {len(filtered_regions)}")
    print(f"Number of clusters found: {len(clusters)}")
    print(f"Unclustered regions: {unclustered_count}")
    print(f"Kernel size used: {kernel_size}")
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i + 1}: {len(cluster)} regions")
    print("=========================\n")

def bbox_debug_images_kmeans(img, filtered_regions, centers, centers_final, labels, K, debug_folder=None):
    if debug_folder:
        # Visualize clustered regions with different colors
        cluster_vis = img.copy()
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]  # Green, Red, Blue, Yellow
        labels_flat = labels.flatten()
		# Draw each region colored by its cluster
        for i, region in enumerate(filtered_regions):
            if i < len(labels_flat):  # Make sure we have a label for this region
                color = colors[labels_flat[i] % len(colors)]
                hull = cv2.convexHull(region.reshape(-1, 1, 2))
                cv2.polylines(cluster_vis, [hull], True, color, 1)

        print(f"Writing image to: {debug_folder}/03_clustered_regions.jpg")
        cv2.imwrite(f"{debug_folder}/03_clustered_regions.jpg", cluster_vis)

        # Visualize final bounding boxes
        bbox_vis = img.copy()

        # Find bounding boxes for each cluster
        for cluster_id in range(K):
            # Get all regions belonging to this cluster
            cluster_regions = [filtered_regions[i] for i in range(len(filtered_regions)) 
                if i < len(labels_flat) and labels_flat[i] == cluster_id]
    
            if cluster_regions:
                # Combine all points from this cluster
                all_points = np.vstack(cluster_regions)
                x, y, w, h = cv2.boundingRect(all_points)

                # Draw bounding box
                color = colors[cluster_id % len(colors)]
                cv2.rectangle(bbox_vis, (x, y), (x + w, y + h), color, 3)

                # Add cluster label
                cv2.putText(bbox_vis, f'Page {cluster_id + 1}', (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                print(f"Cluster {cluster_id}: {len(cluster_regions)} regions, "
                      f"bbox: ({x}, {y}, {w}, {h})")

        print(f"Writing image to: {debug_folder}/04_final_bounding_boxes.jpg")
        cv2.imwrite(f"{debug_folder}/04_final_bounding_boxes.jpg", bbox_vis)

        # Optional: Show cluster centers
        centers_vis = img.copy()
        for i, center in enumerate(centers):
            color = colors[labels_flat[i] % len(colors)] if i < len(labels_flat) else (128, 128, 128)
            cv2.circle(centers_vis, tuple(map(int, center)), 3, color, -1)

        # Draw final cluster centers from k-means
        for i, final_center in enumerate(centers_final):
            cv2.circle(centers_vis, tuple(map(int, final_center)), 8, colors[i], 3)
            cv2.putText(centers_vis, f'C{i}', tuple(map(int, final_center + 15)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)

        print(f"Writing image to: {debug_folder}/05_cluster_centers.jpg")
        cv2.imwrite(f"{debug_folder}/05_cluster_centers.jpg", centers_vis)


def cluster_regions_by_spatial_separation(regions, img_shape):
    """
    Cluster regions by finding natural spatial separation (like book spine)
    """
    # Get centroids
    centroids = []
    for region in regions:
        M = cv2.moments(region)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            centroids.append((cx, cy))
    
    if len(centroids) < 20:
        return [regions]  # Not enough regions to cluster
    
    centroids = np.array(centroids)
    
    # Find the vertical line that best separates the regions (book spine)
    x_coords = centroids[:, 0]
    
    # Try different vertical split lines and find the one with minimum overlap
    img_width = img_shape[1]
    best_split = img_width // 2
    best_balance = float('inf')
    
    # Test splits from 1/3 to 2/3 of image width
    for split_x in range(img_width // 3, 2 * img_width // 3, 10):
        left_count = sum(x < split_x for x in x_coords)
        right_count = sum(x >= split_x for x in x_coords)
        
        # We want roughly balanced clusters
        if left_count > 10 and right_count > 10:
            balance = abs(left_count - right_count)
            if balance < best_balance:
                best_balance = balance
                best_split = split_x
    
    print(f"Best vertical split at x={best_split}")
    
    # Split regions based on centroid x-coordinate
    left_cluster = []
    right_cluster = []
    
    for i, region in enumerate(regions):
        if i < len(centroids):
            cx = centroids[i][0]
            if cx < best_split:
                left_cluster.append(region)
            else:
                right_cluster.append(region)
    
    clusters = []
    if len(left_cluster) > 10:
        clusters.append(left_cluster)
    if len(right_cluster) > 10:
        clusters.append(right_cluster)
    
    print(f"Spatial clustering: {len(left_cluster)} left, {len(right_cluster)} right")
    return clusters

def cluster_regions_morphological(regions, img_shape):
    # Create binary mask from all regions
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    for region in regions:
        cv2.fillPoly(mask, [region], 255)
    
    # Morphological closing to connect nearby regions
    # Adjust kernel size based on expected text spacing
    kernel_size = min(img_shape[:2]) // 50  # Try 100?
    # kernel_size = max(kernel_size, 10)  # But not too small
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find connected components
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Group original regions by which connected component they belong to
    clusters = []
    for contour in contours:
        cluster_regions = []
        for region in regions:
            # Check if region centroid is inside this connected component
            M = cv2.moments(region)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                if cv2.pointPolygonTest(contour, (cx, cy), False) >= 0:
                    cluster_regions.append(region)
        
        if len(cluster_regions) > 10:  # Filter small clusters (noise)
            clusters.append(cluster_regions)
    
    # Sort clusters by size and return the 2 largest
    clusters.sort(key=len, reverse=True)
    return clusters[:2] if len(clusters) >= 2 else clusters

def filter_regions(regions):
    areas = [cv2.contourArea(region) for region in regions]
    q25, q75 = np.percentile(areas, [25, 75])
    iqr = q75 - q25
    min_area = max(1, q25 - 1.5 * iqr)
    max_area = q75 + 1.5 * iqr

    return [region for region in regions if min_area <= cv2.contourArea(region) <= max_area]

def get_region_centers(regions):
    """
    Get the center point of all regions
    Makes it easier to do bounding box math
    """
    centroids = [cv2.moments(region) for region in regions]
    centers = [(int(M['m10']/M['m00']), int(M['m01']/M['m00'])) for M in centroids]
    return np.float32(centers)

def detect_text_regions_mser(img, debug_folder=None):
    """
    Use MSER to detect text regions and find their bounding corners.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    mser = cv2.MSER_create(
        delta=5,
        min_area=10,
        max_area=300,
        max_variation=0.5,
        min_diversity=0.5,
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
    print(f"Found {len(regions)} regions using MSER")
    filtered_regions = filter_regions(regions)
    print(f"Filtered down to {len(filtered_regions)} regions")
    # centers = get_region_centers(filtered_regions)
    # K = 3  # Number of clusters (2 pages)
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    # attempts = 10
    # flags = cv2.KMEANS_RANDOM_CENTERS
    # More robust version
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    # attempts = 20
    # flags = cv2.KMEANS_PP_CENTERS  # Often gives better clustering
    # compactness, labels, centers_final = cv2.kmeans(centers, K, None, criteria, attempts, flags)

    clusters = cluster_regions_morphological(filtered_regions, img.shape)
    # clusters = cluster_regions_by_spatial_separation(filtered_regions, img.shape)
    
    if debug_folder:
        mser_vis = img.copy()
        for region in regions:
            hull = cv2.convexHull(region.reshape(-1, 1, 2))
            cv2.polylines(mser_vis, [hull], True, (0, 255, 0), 1)
        print(f"Writing image to: {debug_folder}/02_mser_regions.jpg")
        cv2.imwrite(f"{debug_folder}/02_mser_regions.jpg", mser_vis)
    # bbox_debug_images(img, filtered_regions, centers, centers_final, labels, K, debug_folder)
    bbox_debug_images(img, filtered_regions, clusters, debug_folder)
    
    # Filter regions that look like text
    text_regions = []
    for region in regions:
        x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
        aspect_ratio = w / h if h > 0 else 0
        
        # Text-like characteristics
        if 0.1 < aspect_ratio < 10 and w > 20 and h > 10:
            text_regions.append(region)
    
    print(f"Found {len(text_regions)} text regions using MSER")
    if text_regions:
        # Combine all text regions to find overall text area
        all_points = np.vstack(text_regions)
        text_hull = cv2.convexHull(all_points.reshape(-1, 1, 2))
        
        if debug_folder:
            # Visualize the combined hull
            hull_img = img.copy()
            cv2.polylines(hull_img, [text_hull], True, (255, 0, 255), 4)
            cv2.imwrite(f"{debug_folder}/08_mser_combined_hull.jpg", hull_img)
        
        # Try standard approximation first
        epsilon = 0.02 * cv2.arcLength(text_hull, True)
        approx = cv2.approxPolyDP(text_hull, epsilon, True)
        
        print(f"Standard approximation gave {len(approx)} corners, forcing 4-corner rectangle...")
        coverage_actual = 0
        coverage_threshold = 0.97
        # Cut of 9% of regions and test our threshold. Cut off less until we achieve our target
        for step in [0.09, 0.07, 0.05, 0.03, 0.02, 0.01, 0.005]:
            print(f"Taking step {step} percent")
            coverage_actual, final_corners = force_rectangle_from_regions(text_regions, step)
            if coverage_actual >= coverage_threshold:
                print(f"Found good rectangle with coverage {coverage_actual}")
                continue
        
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
            cv2.imwrite(f"{debug_folder}/09_mser_final_rectangle.jpg", bbox_img)
        return final_corners


def force_rectangle_from_regions(text_regions, outlier_percent):
    """
    Force creation of a 4-corner rectangle that encompasses a a region
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
    
    # Method 1: Find rectangle that covers target percentage by expanding from center
    all_points = np.vstack(text_regions)
    
    # Get extreme points with some outlier removal
    x_coords = all_points[:, 0]
    y_coords = all_points[:, 1]
    
    # Remove extreme outliers (bottom/top 5% in each dimension)
    x_sorted = np.sort(x_coords)
    y_sorted = np.sort(y_coords)
    
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
    return coverage_actual, forced_corners


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
        cv2.imwrite(f"{images_folder}/10_corrected.jpg", corrected_img)
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
